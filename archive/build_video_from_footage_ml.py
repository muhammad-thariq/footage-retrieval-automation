# build_video_from_footage_ml.py
# Semantic selection (SentenceTransformer) + single-pass robust ffmpeg graph for Shorts/TikTok (9:16 default).
# - 1080x1920 portrait by default, --fit cover|contain|blurpad
# - Exact cuts with trim/atrim, concat via filter (timestamps reset)
# - CFR 30 fps, yuv420p, AAC 48k
# - *_m files muted; optional mix of source audio under TTS
# - No repeats in a single video; across runs prefer unseen then least-used within similarity band
# - Final duration EXACTLY equals WAV via in-graph trim/pad
# - Klasifikasi kata dari folder/filename (meow, purr, ...) → pilih kandidat dengan counter paling sedikit
# - NEW DEBUG: progress per label, pool info, and before->after counters for each selected footage.

import argparse, json, math, random, re, subprocess, tempfile, wave, contextlib, os
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Union
from collections import defaultdict
from statistics import median

import numpy as np

# ---------------- SentenceTransformer ----------------
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise SystemExit("Install deps:\n  pip install -U sentence-transformers numpy\n" + str(e))

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
MIN_CLIP = 4.0
MAX_CLIP = 8.0

DEBUG_POOL_SHOW = 20  # limit preview candidates shown in debug

# --------------- Durations / IO helpers --------------
def ffprobe(fmt_args: list) -> str:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", *fmt_args],
        stderr=subprocess.STDOUT, text=True
    )
    return out.strip()

def run_ffprobe_duration(path: Path) -> float:
    return float(ffprobe(["-show_entries", "format=duration", "-of", "default=nk=1:nw=1", str(path)]))

def audio_duration(path: Path) -> float:
    if path.suffix.lower()==".wav":
        with contextlib.closing(wave.open(str(path),"rb")) as w:
            return w.getnframes()/float(w.getframerate())
    return run_ffprobe_duration(path)

def has_audio_stream(path: Path) -> bool:
    try:
        out = ffprobe(["-show_streams", "-select_streams", "a", "-of", "csv=p=0", str(path)])
        return len(out.strip()) > 0
    except subprocess.CalledProcessError:
        return False

def list_video_files(folder: Path) -> List[Path]:
    # RECURSIVE: dukung struktur folder seperti footage/meow/, footage/purr/, dst
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS])

def parse_labels(txt_path: Path) -> List[str]:
    raw = txt_path.read_text(encoding="utf-8", errors="ignore")
    tokens = re.findall(r"\[([^\]]+)\]", raw)
    if tokens:
        return [t.strip() for t in tokens if t.strip()]
    return [ln.strip() for ln in raw.splitlines() if ln.strip()]

# --------------- Text / embeddings -------------------
def norm_for_text(s: str) -> str:
    s = s.replace("_"," ").replace("-"," ")
    s = re.sub(r"\s+"," ", s).strip()
    return s

def filename_family(stem: str) -> str:
    return re.sub(r"_[0-9]+$","", stem.lower())

def load_counters(path: Path) -> Dict[str,int]:
    if path.exists():
        try: return json.loads(path.read_text(encoding="utf-8"))
        except Exception: pass
    return {}

def save_counters(path: Path, counters: Dict[str,int]):
    path.write_text(json.dumps(counters, indent=2), encoding="utf-8")

def load_or_make_file_embeds(model: SentenceTransformer, files: List[Path], cache_path: Path, model_id: str, debug=False):
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
    updated=False

    def needs_embed(p: Path):
        rec = cache.get(p.name)
        if not rec: return True
        if rec.get("model") != model_id: return True
        try:
            mtime = p.stat().st_mtime
        except FileNotFoundError:
            return True
        return rec.get("mtime") != mtime

    to_embed, idxs = [], []
    for i, f in enumerate(files):
        if needs_embed(f):
            to_embed.append(norm_for_text(f.stem))
            idxs.append(i)

    if to_embed:
        if debug: print(f"[debug] embedding {len(to_embed)} names with '{model_id}'...")
        embs = model.encode(to_embed, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        for j, i in enumerate(idxs):
            f = files[i]
            cache[f.name] = {
                "model": model_id,
                "mtime": f.stat().st_mtime,
                "text": norm_for_text(f.stem),
                "vec": embs[j].tolist(),
            }
        updated=True

    mat=[]
    for f in files:
        rec = cache.get(f.name)
        if not rec:
            v = model.encode([norm_for_text(f.stem)], normalize_embeddings=True)[0]
            cache[f.name] = {
                "model": model_id, "mtime": f.stat().st_mtime,
                "text": norm_for_text(f.stem), "vec": v.tolist()
            }
            updated=True
            mat.append(v)
        else:
            mat.append(np.asarray(rec["vec"], dtype=np.float32))
    mat = np.vstack(mat).astype(np.float32)

    if updated:
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    return mat

def embed_labels(model: SentenceTransformer, labels: List[str]) -> np.ndarray:
    texts = [norm_for_text(lbl.strip("[]")) for lbl in labels]
    vecs = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)

# ---- Klasifikasi kata dari folder & filename ----
_word_re = re.compile(r"[a-z]+")

def _tokenize_words(s: str) -> List[str]:
    return _word_re.findall(s.lower())

def build_class_index(files: List[Path], root: Path) -> Tuple[Dict[str, List[int]], List[Set[str]]]:
    class_index: Dict[str, List[int]] = defaultdict(list)
    per_file_keywords: List[Set[str]] = []
    for i, f in enumerate(files):
        kws: Set[str] = set()
        # dari filename
        kws.update(_tokenize_words(f.stem))
        # dari folder (relative terhadap root)
        try:
            rel = f.relative_to(root)
            for part in rel.parent.parts:
                kws.update(_tokenize_words(part))
        except Exception:
            for part in f.parent.parts:
                kws.update(_tokenize_words(part))
        kws = {k for k in kws if k.isalpha()}
        per_file_keywords.append(kws)
        for k in kws:
            class_index[k].append(i)
    return class_index, per_file_keywords

def label_to_keywords(lbl: str, known: Set[str]) -> List[str]:
    text = norm_for_text(lbl.lower())
    return [k for k in known if re.search(rf"(?:^|\b){re.escape(k)}(?:\b|$)", text)]

def choose_least_used(
    pool: List[int],
    files: List[Path],
    counters: Dict[str,int],
    rng: random.Random,
    family_diversity: bool,
    used_idx: Set[int]
) -> int:
    if family_diversity:
        fam_used = set(filename_family(files[i].stem) for i in used_idx)
        fresh = [i for i in pool if filename_family(files[i].stem) not in fam_used]
        if fresh:
            pool = fresh
    min_count = min(counters.get(files[i].name, 0) for i in pool)
    least = [i for i in pool if counters.get(files[i].name, 0) == min_count]
    return rng.choice(least)

# ---- Fallback selection (embedding) with debug bundle ----
def select_index_for_label(
    label_vec: np.ndarray,
    file_mat: np.ndarray,
    files: List[Path],
    rng: random.Random,
    epsilon: float,
    allow_reuse: bool,
    used_idx: set,
    counters: Dict[str,int],
    family_diversity: bool,
    return_debug: bool = False
) -> Union[int, Tuple[int, dict]]:
    sims = (file_mat @ label_vec).astype(np.float32)
    top = float(np.max(sims))

    def pool_within(band: float) -> List[int]:
        idxs = np.where((top - sims) <= band)[0].tolist()
        if not allow_reuse:
            idxs = [i for i in idxs if i not in used_idx]
        return idxs

    widen_steps = [epsilon, epsilon + 0.03, epsilon + 0.06, epsilon + 0.10]
    pool = []; band_used: Optional[float] = None
    for band in widen_steps:
        pool = pool_within(band)
        if pool:
            band_used = band
            break

    if not pool:
        pool = [i for i in range(len(files)) if (allow_reuse or i not in used_idx)]
        if not pool:
            pool = [int(np.argmax(sims))]  # truly no choice

    zero_used = [i for i in pool if counters.get(files[i].name, 0) == 0]
    if zero_used:
        pool = zero_used

    if family_diversity:
        fam_used = set(filename_family(files[i].stem) for i in used_idx)
        fresh_family = [i for i in pool if filename_family(files[i].stem) not in fam_used]
        if fresh_family:
            pool = fresh_family

    min_count = min(counters.get(files[i].name, 0) for i in pool)
    least_used = [i for i in pool if counters.get(files[i].name, 0) == min_count]
    chosen = rng.choice(least_used)

    if not return_debug:
        return chosen
    # build pool preview
    preview_items = [(files[j].name, counters.get(files[j].name, 0)) for j in pool]
    preview_items.sort(key=lambda x: (x[1], x[0]))
    dbg = {
        "mode": "embed",
        "top_similarity": top,
        "band_used": band_used,
        "pool_size": len(pool),
        "min_count": min_count,
        "pool_preview": preview_items[:DEBUG_POOL_SHOW],
    }
    return chosen, dbg

# --------------- Timing allocation ---------------
def clamp_labels_to_fit(labels: List[str], total_seconds: float) -> List[str]:
    min_n = math.ceil(total_seconds / MAX_CLIP)
    max_n = math.floor(total_seconds / MIN_CLIP)
    min_n = max(min_n, 1); max_n = max(max_n, 1)
    target_n = min(max(len(labels), min_n), max_n)
    if target_n == len(labels): return labels
    if target_n < len(labels):  return labels[:target_n]
    out=list(labels); i=0
    while len(out) < target_n:
        out.append(labels[i % len(labels)]); i+=1
    return out

def allocate_durations(n: int, total: float) -> List[float]:
    base = max(MIN_CLIP, min(MAX_CLIP, total/n))
    durs = [base]*n; diff = total - sum(durs); i=0; step=0.05; guard=0
    while abs(diff)>1e-3 and guard<200000:
        if diff>0:
            add=min(step, diff, MAX_CLIP-durs[i])
            if add>0: durs[i]+=add; diff-=add
        else:
            sub=min(step, -diff, durs[i]-MIN_CLIP)
            if sub>0: durs[i]-=sub; diff+=sub
        i=(i+1)%n; guard+=1
        if guard%n==0 and all(abs((MAX_CLIP-d) if diff>0 else (d-MIN_CLIP))<1e-6 for d in durs): break
    return [round(x,2) for x in durs]

# --------------- Build & run ffmpeg (single pass) ---------------
def build_and_run_ffmpeg(
    clips: List[Path],
    starts: List[float],
    durs: List[float],
    mute_flags: List[bool],
    has_audio_flags: List[bool],
    audio_wav: Path,
    out_video: Path,
    out_size: str,
    mix_source: bool,
    source_db: float,
    fit_mode: str,
    bg_color: str,
    bg_blur: str,
    debug: bool
):
    m = re.match(r"^(\d+)x(\d+)$", out_size.strip().lower())
    if not m: raise ValueError(f"Invalid --size '{out_size}', expected like 1080x1920")
    W, H = int(m.group(1)), int(m.group(2))

    args = ["ffmpeg", "-y"]
    for p in clips: args += ["-i", str(p)]
    args += ["-i", str(audio_wav)]

    a_dur = audio_duration(audio_wav)
    vol_lin = 10.0**(source_db/20.0)

    fl = []
    v_labels = []
    a_labels = []
    for i, (start, dur, mute, has_a) in enumerate(zip(starts, durs, mute_flags, has_audio_flags)):
        if fit_mode == "contain":
            v_chain = (
                f"[{i}:v]"
                f"trim=start={start:.3f}:end={(start+dur):.3f},"
                f"setpts=PTS-STARTPTS,"
                f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
                f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color={bg_color},"
                f"fps=30,format=yuv420p,setsar=1[v{i}]"
            )
        elif fit_mode == "blurpad":
            v_chain = (
                f"[{i}:v]"
                f"trim=start={start:.3f}:end={(start+dur):.3f},setpts=PTS-STARTPTS,split=2[va{i}][vb{i}];"
                f"[va{i}]scale={W}:{H}:force_original_aspect_ratio=increase,boxblur={bg_blur},scale={W}:{H}[bg{i}];"
                f"[vb{i}]scale={W}:{H}:force_original_aspect_ratio=decrease[fg{i}];"
                f"[bg{i}][fg{i}]overlay=(W-w)/2:(H-h)/2,format=yuv420p,fps=30,setsar=1[v{i}]"
            )
        else:  # cover
            v_chain = (
                f"[{i}:v]"
                f"trim=start={start:.3f}:end={(start+dur):.3f},"
                f"setpts=PTS-STARTPTS,"
                f"scale={W}:{H}:force_original_aspect_ratio=increase,"
                f"crop={W}:{H},"
                f"fps=30,format=yuv420p,setsar=1[v{i}]"
            )
        fl.append(v_chain)
        v_labels.append(f"[v{i}]")

        if has_a:
            a = (
                f"[{i}:a]"
                f"atrim=start={start:.3f}:end={(start+dur):.3f},"
                f"asetpts=PTS-STARTPTS,"
                f"aformat=sample_fmts=fltp:channel_layouts=stereo,"
                f"aresample=48000"
            )
            if mute:
                a += ",volume=0"
            a += f"[a{i}]"
            fl.append(a)
        else:
            a = f"aevalsrc=0:c=stereo:s=48000:d={dur:.3f}[a{i}]"
            fl.append(a)
        a_labels.append(f"[a{i}]")

    concat_in = "".join([v_labels[i] + a_labels[i] for i in range(len(v_labels))])
    fl.append(f"{concat_in}concat=n={len(v_labels)}:v=1:a=1[vcat][acat]")
    fl.append(f"[{len(clips)}:a]atrim=0:{a_dur:.3f},asetpts=PTS-STARTPTS[atts]")

    if mix_source:
        fl.append(f"[acat]volume={vol_lin}[a0]")
        fl.append(f"[a0][atts]amix=inputs=2:normalize=0:dropout_transition=0,atrim=0:{a_dur:.3f},asetpts=PTS-STARTPTS[aout]")
    else:
        fl.append(f"[atts]anull[aout]")

    fl.append(f"[vcat]setpts=PTS-STARTPTS,trim=duration={a_dur:.3f},fps=30,format=yuv420p,setsar=1[vout]")

    filter_complex = ";".join(fl)

    args += [
        "-filter_complex", filter_complex,
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(out_video)
    ]

    subprocess.check_call(args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

# ---------------------------- main ----------------------------
def main():
    p=argparse.ArgumentParser(description="Build video matching WAV using semantic file selection and robust concat.")
    p.add_argument("--footage_dir", type=str, default="footage")
    p.add_argument("--input_txt",   type=str, default="footage_example.txt")
    p.add_argument("--audio_wav",   type=str, default="heart_all.wav")
    p.add_argument("--out_video",   type=str, default="heart_all_visual.mp4")

    # selection/matching
    p.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--exclude_file", action="append", default=[], help="Filename(s) to exclude for this run; can repeat.")
    p.add_argument("--family_diversity", choices=["on","off"], default="off", help="Avoid same filename family in one video.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow_reuse", action="store_true")
    p.add_argument("--counters_file", type=str, default=".selection_counters.json")
    p.add_argument("--embed_cache", type=str, default=".file_embeddings.json")
    p.add_argument("--debug", action="store_true")

    # rendering
    p.add_argument("--size", type=str, default="1080x1920", help="Output resolution WxH (default 9:16 portrait)")
    p.add_argument("--fit", choices=["contain","cover","blurpad"], default="cover",
                   help="Frame fit: contain=pad, cover=crop to fill, blurpad=blurred background")
    p.add_argument("--bg_color", type=str, default="black", help="Pad color for --fit contain")
    p.add_argument("--bg_blur", type=str, default="24:1", help="boxblur radius:power for --fit blurpad")

    # audio mix
    p.add_argument("--no_mix_source", action="store_true")
    p.add_argument("--mix_source_db", type=float, default=-15.0)

    args=p.parse_args()

    footage_dir=Path(args.footage_dir); input_txt=Path(args.input_txt); audio_wav=Path(args.audio_wav)
    out_video=Path(args.out_video); counters_path=footage_dir/args.counters_file; embed_cache=footage_dir/args.embed_cache

    if not footage_dir.exists(): raise FileNotFoundError(f"Footage folder not found: {footage_dir}")
    if not input_txt.exists():   raise FileNotFoundError(f"Labels file not found: {input_txt}")
    if not audio_wav.exists():   raise FileNotFoundError(f"Audio file not found: {audio_wav}")

    total_audio=audio_duration(audio_wav); print(f"[+] Audio duration: {total_audio:.2f}s")

    labels=parse_labels(input_txt)
    if not labels: raise ValueError("No labels found in input.")
    print(f"[+] Labels loaded: {len(labels)}")

    files=list_video_files(footage_dir)
    if not files: raise ValueError("No video files in footage_dir.")

    # per-run exclusions (temporary)
    exclude_set = { (name.lower()) for name in (args.exclude_file or []) }
    if exclude_set:
        files = [f for f in files if f.name.lower() not in exclude_set]
        if not files:
            raise ValueError("After applying --exclude_file, no video files remain.")
        if args.debug:
            print("[debug] excluded files:", sorted(exclude_set))

    # counters
    counters=load_counters(counters_path)
    for f in files: counters.setdefault(f.name, 0)

    if args.debug:
        cnts = [counters[f.name] for f in files]
        unseen = sum(1 for c in cnts if c == 0)
        print(f"[debug] counters summary: files={len(files)} unseen={unseen} min={min(cnts) if cnts else 0} "
              f"median={int(median(cnts)) if cnts else 0} max={max(cnts) if cnts else 0}")
        print(f"[debug] counters file: {counters_path}")

    # class index
    class_index, per_file_keywords = build_class_index(files, footage_dir)
    known_keywords: Set[str] = set(class_index.keys())
    if args.debug:
        kk = ", ".join(sorted(list(known_keywords))[:40])
        more = "" if len(known_keywords) <= 40 else f" ... (+{len(known_keywords)-40})"
        print(f"[debug] scraped keywords: {kk}{more}")

    # model + embeddings (fallback)
    model = SentenceTransformer(args.model)
    file_mat = load_or_make_file_embeds(model, files, embed_cache, model_id=args.model, debug=args.debug)
    label_vecs = embed_labels(model, labels)

    # timing plan
    labels_use = clamp_labels_to_fit(labels, total_audio)
    if len(labels_use)!=len(labels):
        print(f"[!] Adjusted labels to {len(labels_use)} to fit audio with {MIN_CLIP:.0f}–{MAX_CLIP:.0f}s clips.")
    n=len(labels_use); per_clip=allocate_durations(n, total_audio)
    print(f"[+] Clips: {n} | per-clip ~ {sum(per_clip)/n:.2f}s (min={min(per_clip):.2f}, max={max(per_clip):.2f})")

    # choose files
    rng=random.Random(args.seed)
    used_idx:set[int]=set(); chosen=[]; starts=[]; mutes=[]; has_a=[]
    counters_before = dict(counters)

    def _progress_bar(i: int, n: int) -> str:
        length = 24
        filled = int(length * i / n)
        return "#" * filled + "-" * (length - filled)

    for i, lbl in enumerate(labels_use, start=1):
        lbl_vec = label_vecs[(i-1) % len(labels)]
        if args.debug:
            print(f"\n[progress] [{i:02d}/{n}] |{_progress_bar(i, n)}| label: {lbl}")

        matched_keys = label_to_keywords(lbl, known_keywords)
        idx = None

        if matched_keys:
            # union kandidat dari keyword yang match
            pool = sorted(set(j for k in matched_keys for j in class_index.get(k, [])))
            if not args.allow_reuse:
                pool = [j for j in pool if j not in used_idx]

            if args.debug:
                print(f"[debug] class match: {', '.join(matched_keys)}  | pool={len(pool)}")
                for k in matched_keys:
                    cand = class_index.get(k, [])
                    cand = [c for c in cand if (args.allow_reuse or c not in used_idx)]
                    if cand:
                        kmin = min(counters.get(files[j].name, 0) for j in cand)
                        kmax = max(counters.get(files[j].name, 0) for j in cand)
                        print(f"        - '{k}': candidates={len(cand)} min_count={kmin} max_count={kmax}")
                # preview pool by counter asc
                preview = [(files[j].name, counters.get(files[j].name, 0)) for j in pool]
                preview.sort(key=lambda x: (x[1], x[0]))
                if preview:
                    show = ", ".join([f"{n}:{c}" for n,c in preview[:DEBUG_POOL_SHOW]])
                    print(f"        pool preview: {show}{' ...' if len(preview)>DEBUG_POOL_SHOW else ''}")

            if pool:
                idx = choose_least_used(
                    pool=pool,
                    files=files,
                    counters=counters,
                    rng=rng,
                    family_diversity=(args.family_diversity=="on"),
                    used_idx=used_idx
                )

                name = files[idx].name
                before = counters.get(name, 0)
                if args.debug:
                    print(f"[pick] by class  → {name}  (before={before})")

        # ---- Fallback: embedding similarity ----
        if idx is None:
            sel = select_index_for_label(
                lbl_vec, file_mat, files, rng, args.epsilon, args.allow_reuse, used_idx,
                counters, family_diversity=(args.family_diversity=="on"), return_debug=args.debug
            )
            if args.debug:
                idx, dbg = sel
                name = files[idx].name
                before = counters.get(name, 0)
                print(f"[debug] fallback embed: band_used={dbg.get('band_used')} top_sim={dbg.get('top_similarity'):.3f} "
                      f"pool={dbg.get('pool_size')} min_count={dbg.get('min_count')}")
                prev = dbg.get('pool_preview', [])
                if prev:
                    show = ", ".join([f"{n}:{c}" for n,c in prev])
                    print(f"        pool preview: {show}{' ...' if dbg.get('pool_size',0)>len(prev) else ''}")
                print(f"[pick] by embed  → {name}  (before={before})")
            else:
                idx = sel  # type: ignore[assignment]

        if not args.allow_reuse:
            used_idx.add(idx)  # EXCLUDE sementara dalam satu video

        f=files[idx]
        before = counters.get(f.name, 0)
        counters[f.name]=before+1
        after = counters[f.name]
        chosen.append(f)

        # compute start time randomly (fit inside source)
        dur = per_clip[i-1]
        vdur = run_ffprobe_duration(f)
        window = max(0.0, vdur - dur - 0.25)
        start = rng.uniform(0, window) if window>0 else 0.0
        starts.append(round(start,3))

        mutes.append(f.stem.endswith("_m"))
        has_a.append(has_audio_stream(f))

        if args.debug:
            print(f"[apply] {f.name}: counter {before} -> {after} (+1)")
            print(f"        start={start:.2f}s  dur={dur:.2f}s  mute={mutes[-1]}  has_audio={has_a[-1]}")

    # persist counters
    save_counters(counters_path, counters)

    if args.debug:
        print("\n[debug] counter increments this run:")
        changes = []
        for name, newv in counters.items():
            oldv = counters_before.get(name, 0)
            if newv != oldv:
                changes.append((name, newv - oldv, newv))
        changes.sort(key=lambda x: (-x[1], x[0]))
        for nm, d, nv in changes:
            print(f"  +{d}  {nm}  (now {nv})")

        print("\n[debug] selected order:")
        for idx, f in enumerate(chosen, 1):
            print(f"  [{idx:02d}] {f.name}")

    # build & run single ffmpeg
    build_and_run_ffmpeg(
        clips=chosen,
        starts=starts,
        durs=per_clip,
        mute_flags=mutes,
        has_audio_flags=has_a,
        audio_wav=audio_wav,
        out_video=out_video,
        out_size=args.size,
        mix_source=(not args.no_mix_source),
        source_db=args.mix_source_db,
        fit_mode=args.fit,
        bg_color=args.bg_color,
        bg_blur=args.bg_blur,
        debug=args.debug
    )

    # verify
    try:
        v_dur = float(ffprobe(["-show_entries","format=duration","-of","default=nk=1:nw=1", str(out_video)]))
        a_dur = float(ffprobe(["-show_entries","format=duration","-of","default=nk=1:nw=1", str(audio_wav)]))
        print(f"[OK] Wrote {out_video.resolve()}  |  durations -> video: {v_dur:.3f}s, audio: {a_dur:.3f}s")
    except Exception:
        print(f"[OK] Wrote {out_video.resolve()}")
if __name__ == "__main__":
    import sys, os
    from pathlib import Path

    # biar print langsung keluar di terminal
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    # --- JALANKAN PIPELINE UTAMA DULU ---
    try:
        main()
    except SystemExit:
        # argparse bisa nge-raise SystemExit, biarkan bubble up
        raise
    except Exception as e:
        print(f"[fatal] main() error: {e}", flush=True)
        raise

    # --- BARU SETELAH ITU: GUARD POST-PROCESS DURATION FIX ---
    def _arg_val(flag: str, default: str):
        try:
            i = sys.argv.index(flag)
            return sys.argv[i + 1]
        except Exception:
            return default

    try:
        out_video = Path(_arg_val("--out_video", "heart_all_visual.mp4"))
        audio_wav = Path(_arg_val("--audio_wav", "heart_all.wav"))
        if out_video.exists() and audio_wav.exists():
            target = audio_duration(audio_wav)  # exact length from WAV
            tmp_fix = out_video.with_name(out_video.stem + "_durfix.mp4")

            filter_complex = (
                f"[0:v]trim=duration={target:.6f},setpts=PTS-STARTPTS,fps=30,format=yuv420p,setsar=1[v];"
                f"[0:a]atrim=0:{target:.6f},asetpts=PTS-STARTPTS[a]"
            )

            cmd = [
                "ffmpeg", "-y",
                "-i", str(out_video),
                "-filter_complex", filter_complex,
                "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                str(tmp_fix),
            ]

            try:
                subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                os.replace(tmp_fix, out_video)
                print(f"[post] Duration fixed → {target:.3f}s (matches input audio).", flush=True)
            except subprocess.CalledProcessError:
                if tmp_fix.exists():
                    try: os.remove(tmp_fix)
                    except: pass
                print("[post] Duration guard failed; keeping previous output.", flush=True)
        else:
            print("[post] Duration guard skipped: outputs missing.", flush=True)
    except Exception as e:
        print(f"[post] Duration guard skipped: {e}", flush=True)
