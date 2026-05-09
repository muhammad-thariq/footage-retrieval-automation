# build_video_from_footage_ml.py
# Input-driven (one sentence per line) video builder + keyword classification + sentence-boundary durations + timeline print.
# - Classify each sentence to scraped keywords (from folders & filenames)
# - Show Top-5 ranked keywords per sentence (semantic + lexical) with pool stats (enable --debug)
# - Compute per-sentence durations using first-word boundaries from SRT:
#     * s1: [0.0, t(first word of s2))
#     * si: [t(first word of si), t(first word of s(i+1)))
#     * slast: [t(first word of slast), audio_dur]
# - Select clips from top keyword pools with least-used preference & optional family diversity
# - Special case: first & last lines share *one* footage with a contiguous window (first part -> last line, second part -> first line)
# - Single-pass robust ffmpeg concat; final trim to match WAV length
# - At the end, print a timeline: "from xx to xx" for each footage on the final video

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

DEBUG_POOL_SHOW = 20  # limit preview candidates shown in debug
PAIR_MARGIN = 0.25    # safety margin when carving contiguous window from a single clip

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
    # RECURSIVE: support structure like footage/meow/, footage/purr/, etc.
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS])

# --------------- Text / embeddings -------------------
_word_re = re.compile(r"[a-z]+")
_tag_re  = re.compile(r"<[^>]+>")

def norm_for_text(s: str) -> str:
    s = s.replace("_"," ").replace("-"," ")
    s = re.sub(r"\s+"," ", s).strip()
    return s

def _tokenize_words(s: str) -> List[str]:
    return _word_re.findall(s.lower())

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
        if debug: print(f"[debug] embedding {len(to_embed)} filenames with '{model_id}'...")
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

def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    vecs = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)

# ---- Class index (keywords from folder & filename) ----
def build_class_index(files: List[Path], root: Path) -> Tuple[Dict[str, List[int]], List[Set[str]]]:
    class_index: Dict[str, List[int]] = defaultdict(list)
    per_file_keywords: List[Set[str]] = []
    for i, f in enumerate(files):
        kws: Set[str] = set()
        # from filename
        kws.update(_tokenize_words(f.stem))
        # from folder (relative to root)
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

# --------------- Input sentences ---------------------
def read_input_sentences(txt_path: Path) -> List[str]:
    # strictly one sentence per line; ignore blank lines
    raw = txt_path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.strip() for ln in raw.splitlines()]
    return [ln for ln in lines if ln]

# --------------- SRT parsing & sentence-boundary durations ---------------
def _parse_time(ts: str) -> float:
    # "HH:MM:SS,mmm"
    hh, mm, rest = ts.split(":")
    ss, ms = rest.split(",")
    return int(hh)*3600 + int(mm)*60 + int(ss) + int(ms)/1000.0

def parse_srt_entries(srt_path: Path) -> List[Tuple[float, float, str, List[str]]]:
    """
    Returns list of (start, end, raw_text, tokens) per cue.
    Strips HTML-like tags before tokenizing.
    """
    txt = srt_path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n\s*\n", txt.strip())
    entries = []
    for b in blocks:
        lines = [x.strip("\ufeff") for x in b.splitlines()]
        if len(lines) < 2:
            continue
        # find time line
        time_line = None
        for li in lines:
            if "-->" in li:
                time_line = li
                break
        if not time_line:
            continue
        a, b = [x.strip() for x in time_line.split("-->")]
        start = _parse_time(a)
        end   = _parse_time(b)
        # take text after the time line
        after = False; texts=[]
        for li in lines:
            if after:
                texts.append(li)
            if li is time_line or ("-->" in li):
                after = True
                texts = []
        raw = "\n".join(texts).strip()
        raw = _tag_re.sub("", raw)  # strip <font ...> etc.
        toks = _tokenize_words(raw)
        entries.append((start, end, raw, toks))
    return entries

def build_word_timeline_from_srt(srt_path: Path) -> Tuple[List[str], List[float]]:
    """
    Create a flat word timeline across the SRT:
      words[k]  -> lowercased token k
      times[k]  -> estimated start time of that token
    If a cue has multiple tokens but no per-word times, distribute uniformly over the cue's duration.
    """
    entries = parse_srt_entries(srt_path)
    words: List[str] = []
    times: List[float] = []
    for (st, en, _raw, toks) in entries:
        if not toks:
            continue
        dur = max(0.0, en - st)
        n = len(toks)
        if n == 1 or dur <= 1e-6:
            words.append(toks[0]); times.append(st)
        else:
            # linear spacing across cue; first word at cue start
            for i, w in enumerate(toks):
                t = st + (i / n) * dur
                words.append(w); times.append(t)
    return words, times

def _first_ngram(tokens: List[str], max_n: int = 3) -> List[List[str]]:
    """Return list of candidate n-grams from the start of tokens, longest-first."""
    n = min(max_n, len(tokens))
    return [tokens[:k] for k in range(n, 0, -1)]

def _find_ngram_forward(hay: List[str], needle: List[str], start_idx: int) -> Optional[int]:
    """Find needle sequence in hay starting at or after start_idx; return first index or None."""
    if not needle: return None
    m = len(needle)
    for i in range(start_idx, len(hay) - m + 1):
        if hay[i:i+m] == needle:
            return i
    return None

def durations_from_line_boundaries(
    sentences: List[str],
    srt_path: Path,
    total_audio: float,
    debug: bool=False
) -> List[float]:
    """
    Compute durations by sentence boundaries:
      boundary[0] = 0.0
      boundary[i] = time of FIRST WORD of sentence i (i>=1) found in SRT timeline
      last boundary = total_audio
      dur[i] = boundary[i+1] - boundary[i]
    Robustness:
      - Uses trigram→bigram→unigram of each sentence's opening tokens to locate first word reliably.
      - Searches forward from previous match to ensure monotonic boundaries.
      - If a boundary isn't found, it fills by splitting the remaining span evenly.
    """
    words, times = build_word_timeline_from_srt(srt_path)
    boundaries: List[Optional[float]] = [0.0]
    cursor = 0  # search start index into word timeline

    # Pre-tokenize sentences
    sent_tokens = [_tokenize_words(s) for s in sentences]

    for si in range(1, len(sentences)):  # for sentence 2..N, find its first-word time
        toks = sent_tokens[si]
        candidates = _first_ngram(toks, max_n=3)  # [[w1,w2,w3], [w1,w2], [w1]]
        match_idx = None
        for gram in candidates:
            idx = _find_ngram_forward(words, gram, cursor)
            if idx is not None:
                match_idx = idx
                break
        if match_idx is None:
            boundaries.append(None)
        else:
            t = times[match_idx]
            t = max(t, float(boundaries[-1] if boundaries[-1] is not None else 0.0) + 1e-3)
            boundaries.append(t)
            cursor = match_idx

    # Close with total_audio
    boundaries.append(total_audio)

    # Fix any None boundaries by even split of remaining span
    i = 1
    while i < len(boundaries)-1:
        if boundaries[i] is None:
            j = i
            while j < len(boundaries)-1 and boundaries[j] is None:
                j += 1
            left = float(boundaries[i-1])
            right = float(boundaries[j])
            span = max(0.0, right - left)
            slots = j - i + 1
            step = span / slots if slots > 0 else 0.0
            for k in range(i, j):
                boundaries[k] = left + step * (k - (i-1))
            i = j
        else:
            i += 1

    # Now compute durations
    durs: List[float] = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        a = float(a); b = float(b)
        d = max(0.0, b - a)
        durs.append(round(max(0.10, d), 3))

    if debug:
        print("[debug] Sentence boundaries:")
        for idx, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:]), 1):
            print(f"  s{idx:02d}: start={float(a):8.3f}  end={float(b):8.3f}  dur={max(0.0, float(b)-float(a)):7.3f}s")
        print(f"[debug] sum(durations)={sum(durs):.3f}s  | audio={total_audio:.3f}s")
    return durs

# --------------- Keyword ranking for each sentence ---------------
def rank_keywords_for_sentence(
    sentence: str,
    sent_vec: np.ndarray,
    keywords: List[str],
    kw_vecs: np.ndarray,
    class_index: Dict[str, List[int]],
    files: List[Path],
    counters: Dict[str,int],
    topk: int = 5
):
    """
    Score = semantic_similarity + 0.75 * lexical_match
    semantic_similarity = cosine between sent_vec and kw_vec (both normalized)
    lexical_match = 1.0 if keyword appears as whole word in sentence, else 0.0
    Returns top-k list of dicts with debug info.
    """
    sims = (kw_vecs @ sent_vec).astype(np.float32)  # [-1,1]
    sims = np.clip(sims, -1.0, 1.0)
    s_lower = sentence.lower()
    out = []
    for i, kw in enumerate(keywords):
        lex = 1.0 if re.search(rf"(?:^|\b){re.escape(kw)}(?:\b|$)", s_lower) else 0.0
        score = float(sims[i]) + 0.75*lex
        pool = class_index.get(kw, [])
        min_count = min((counters.get(files[j].name, 0) for j in pool), default=0)
        out.append({
            "keyword": kw,
            "score": score,
            "sim": float(sims[i]),
            "lex": lex,
            "candidates": len(pool),
            "min_count": int(min_count),
        })
    out.sort(key=lambda d: (-d["score"], -d["sim"], d["keyword"]))
    return out[:topk]

# --------------- Selection helpers ---------------
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

# --------------- Build & run ffmpeg (single pass) ---------------
def build_and_run_ffmpeg(
    clips: List[Path],
    starts: List[float],
    durs: List[float],
    mute_flags: List[bool],
    has_audio_flags: List[bool],
    sfx_flags: List[Optional[str]],
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

    extra_inputs = [s for s in sfx_flags if s is not None]

    args = ["ffmpeg", "-y"]
    for p in clips: args += ["-i", str(p)]
    args += ["-i", str(audio_wav)]
    for sfx in extra_inputs:
        args += ["-stream_loop", "-1", "-i", str(sfx)]

    a_dur = audio_duration(audio_wav)
    vol_lin = 10.0**(source_db/20.0)

    fl = []
    v_labels = []
    a_labels = []
    sfx_counter = 0
    for i, (start, dur, mute, has_a, sfx) in enumerate(zip(starts, durs, mute_flags, has_audio_flags, sfx_flags)):
        v_chain = None
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
        fl.append(v_chain); v_labels.append(f"[v{i}]")

        if sfx is not None:
            sfx_idx = len(clips) + 1 + sfx_counter
            sfx_counter += 1
            a = (
                f"[{sfx_idx}:a]"
                f"atrim=start=0:end={dur:.3f},"
                f"asetpts=PTS-STARTPTS,"
                f"aformat=sample_fmts=fltp:channel_layouts=stereo,"
                f"aresample=48000[a{i}]"
            )
            fl.append(a)
        elif has_a:
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

# --------------- Helpers: formatting for timeline print ---------------
def _fmt_ts(t: float) -> str:
    t = max(0.0, float(t))
    ms = int(round((t - int(t)) * 1000))
    total = int(t)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

# ---------------------------- main ----------------------------
def main():
    p=argparse.ArgumentParser(description="Build video matching WAV using keyword classification + sentence-boundary durations.")
    p.add_argument("--footage_dir", type=str, default="footage")
    p.add_argument("--input_txt",   type=str, default="input.txt", help="One sentence per line.")
    p.add_argument("--subs_srt",    type=str, default="heart_all.srt", help="Subtitle SRT aligned to the audio.")
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
    srt_path=Path(args.subs_srt); out_video=Path(args.out_video)
    counters_path=footage_dir/args.counters_file; embed_cache=footage_dir/args.embed_cache

    if not footage_dir.exists(): raise FileNotFoundError(f"Footage folder not found: {footage_dir}")
    if not input_txt.exists():   raise FileNotFoundError(f"Input file not found: {input_txt}")
    if not srt_path.exists():    raise FileNotFoundError(f"SRT file not found: {srt_path}")
    if not audio_wav.exists():   raise FileNotFoundError(f"Audio file not found: {audio_wav}")

    total_audio=audio_duration(audio_wav); print(f"[+] Audio duration: {total_audio:.2f}s")

    sentences = read_input_sentences(input_txt)
    if not sentences: raise ValueError("No sentences found in input.txt (one per line).")
    print(f"[+] Sentences loaded: {len(sentences)}")

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

    # class index (scraped keywords)
    class_index, per_file_keywords = build_class_index(files, footage_dir)
    known_keywords: List[str] = sorted(set(class_index.keys()))

    if args.debug:
        kk = ", ".join(known_keywords[:40])
        more = "" if len(known_keywords) <= 40 else f" ... (+{len(known_keywords)-40})"
        print(f"[debug] scraped keywords: {kk}{more}")

    # model + embeddings
    model = SentenceTransformer(args.model)
    file_mat = load_or_make_file_embeds(model, files, embed_cache, model_id=args.model, debug=args.debug)
    sent_vecs = embed_texts(model, [norm_for_text(s) for s in sentences])
    kw_vecs   = embed_texts(model, known_keywords)

    # durations per sentence from FIRST-WORD BOUNDARIES in SRT
    per_clip = durations_from_line_boundaries(sentences, srt_path, total_audio, debug=args.debug)
    n=len(per_clip)
    if args.debug:
        avg = sum(per_clip)/n
        print(f"[+] Per-sentence durations (boundary-based): {n} | avg={avg:.2f}s (min={min(per_clip):.2f}, max={max(per_clip):.2f})")

    rng=random.Random(args.seed)

    # Prepare containers (we will pre-fill first/last when paired)
    chosen: List[Optional[Path]] = [None]*n
    starts: List[Optional[float]] = [None]*n
    mutes:  List[Optional[bool]]  = [None]*n
    has_a:  List[Optional[bool]]  = [None]*n
    sfxs:   List[Optional[str]]   = [None]*n

    used_idx:set[int]=set()
    counters_before = dict(counters)

    def _progress_bar(i: int, n: int) -> str:
        length = 24
        filled = int(length * i / n)
        return "#" * filled + "-" * (length - filled)

    # ---------- SPECIAL PAIR: first & last share one footage ----------
    pair_done = False
    if n >= 2:
        d_first = float(per_clip[0])
        d_last  = float(per_clip[-1])
        pair_len = d_first + d_last

        # Rank keywords using FIRST sentence only
        top5_first = rank_keywords_for_sentence(
            sentence=sentences[0],
            sent_vec=sent_vecs[0],
            keywords=known_keywords,
            kw_vecs=kw_vecs,
            class_index=class_index,
            files=files,
            counters=counters
        )

        # Build pool from best keywords (progressively widen until non-empty)
        pool = []
        widen_used = 0
        for widen in [1, 2, 3, 5]:  # try top-1, top-2, top-3, then top-5 union
            cand_kw = [it["keyword"] for it in top5_first[:widen]]
            pool = sorted(set(j for k in cand_kw for j in class_index.get(k, [])))
            if pool:
                widen_used = widen
                break

        # If empty, fallback to embedding sim against filenames (ignoring used_idx for now)
        if not pool:
            sel = select_index_for_label(
                sent_vecs[0], file_mat, files, rng, args.epsilon, True, set(),  # True allow_reuse here
                counters, family_diversity=(args.family_diversity=="on"), return_debug=args.debug
            )
            if args.debug:
                idx_pair, dbg = sel  # type: ignore
                if isinstance(dbg, dict):
                    print(f"[debug][pair] fallback embed: band_used={dbg.get('band_used')} top_sim={dbg.get('top_similarity'):.3f} "f"pool={dbg.get('pool_size')} min_count={dbg.get('min_count')}")
                    prev = dbg.get('pool_preview', [])
                    if prev:
                        show = ", ".join([f"{n}:{c}" for n,c in prev])
                        print(f"pool preview: {show}{' ...' if dbg.get('pool_size',0)>len(prev) else ''}")
            else:
                idx_pair = sel  # type: ignore
        else:
            # Choose least-used from pool (ignore used_idx for pairing step)
            idx_pair = choose_least_used(
                pool=pool,
                files=files,
                counters=counters,
                rng=rng,
                family_diversity=(args.family_diversity=="on"),
                used_idx=set()
            )

            if args.debug:
                preview = [(files[j].name, counters.get(files[j].name, 0)) for j in pool]
                preview.sort(key=lambda x: (x[1], x[0]))
                show = ", ".join([f"{n}:{c}" for n,c in preview[:DEBUG_POOL_SHOW]])
                print(f"[pair] from top-{widen_used} keywords → {len(pool)} candidates")
                print(f"       pool preview: {show}{' ...' if len(preview)>DEBUG_POOL_SHOW else ''}")

        fpair = files[idx_pair]
        vdur_pair = run_ffprobe_duration(fpair)

        if vdur_pair >= pair_len + PAIR_MARGIN:
            # one contiguous window: [start, start+pair_len)
            window_max = max(0.0, vdur_pair - pair_len - PAIR_MARGIN)
            start_pair = rng.uniform(0, window_max) if window_max>0 else 0.0

            # mapping: first part -> LAST line, second part -> FIRST line
            start_last  = start_pair
            start_first = start_pair + d_last

            # Fill slots (no-repeat for the rest)
            chosen[0] = fpair
            starts[0] = round(start_first, 3)
            mutes[0]  = fpair.stem.endswith("_m")
            has_a[0]  = has_audio_stream(fpair)
            sfxs[0]   = "sfx/purr.wav" if fpair.stem.endswith("-pr") else ("sfx/meow.wav" if fpair.stem.endswith("-mw") else None)

            chosen[-1] = fpair
            starts[-1] = round(start_last, 3)
            mutes[-1]  = fpair.stem.endswith("_m")
            has_a[-1]  = has_audio_stream(fpair)
            sfxs[-1]   = "sfx/purr.wav" if fpair.stem.endswith("-pr") else ("sfx/meow.wav" if fpair.stem.endswith("-mw") else None)

            # counters: used twice
            counters[fpair.name] = counters.get(fpair.name, 0) + 2

            used_idx.add(idx_pair)  # prevent reuse for other lines

            pair_done = True
            if args.debug:
                print(f"[pair] using ONE clip for first & last: {fpair.name}")
                print(f"       window_len={pair_len:.3f}s  start_pair={start_pair:.3f}s  "
                      f"(last:{d_last:.3f}s @ {start_last:.3f}s | first:{d_first:.3f}s @ {start_first:.3f}s)")
        else:
            if args.debug:
                need = pair_len + PAIR_MARGIN
                print(f"[pair] fallback: '{fpair.name}' too short for contiguous window "
                      f"(need ≥ {need:.2f}s, has {vdur_pair:.2f}s). Using separate picks.")

    # ---------- NORMAL SELECTION for remaining slots ----------
    for i, sentence in enumerate(sentences):
        if chosen[i] is not None:
            # already assigned by pair step
            continue

        if args.debug:
            print(f"\n[progress] [{i+1:02d}/{n}] |{_progress_bar(i+1, n)}| sentence: {sentence}")
            print(f"          dur_from_boundaries={per_clip[i]:.2f}s")

        sent_vec = sent_vecs[i]

        # Rank keywords (Top-5 debug)
        top5 = rank_keywords_for_sentence(
            sentence=sentence,
            sent_vec=sent_vec,
            keywords=known_keywords,
            kw_vecs=kw_vecs,
            class_index=class_index,
            files=files,
            counters=counters
        )
        if args.debug:
            print(" [rank] Top-5 keywords:")
            for rnk, it in enumerate(top5, 1):
                print(f"    {rnk}) {it['keyword']:<16} score={it['score']:.3f}  sim={it['sim']:.3f}  "
                      f"lex={int(it['lex'])}  candidates={it['candidates']}  min_count={it['min_count']}")

        # Build pool from best keywords (progressively widen until non-empty), respecting no-repeats
        idx = None
        pool = []
        for widen in [1, 2, 3, 5]:
            cand_kw = [it["keyword"] for it in top5[:widen]]
            pool = sorted(set(j for k in cand_kw for j in class_index.get(k, [])))
            if not args.allow_reuse:
                pool = [j for j in pool if j not in used_idx]
            if pool:
                break

        if args.debug:
            print(f" [pool] from top-{min(len(top5), widen)} keywords → {len(pool)} candidates")
            if pool:
                preview = [(files[j].name, counters.get(files[j].name, 0)) for j in pool]
                preview.sort(key=lambda x: (x[1], x[0]))
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
                print(f"[pick] by keyword → {name}  (before={before})")

        # Fallback: embedding similarity against filenames
        if idx is None:
            sel = select_index_for_label(
                sent_vec, file_mat, files, rng, args.epsilon, args.allow_reuse, used_idx,
                counters, family_diversity=(args.family_diversity=="on"), return_debug=args.debug
            )
            if args.debug:
                idx, dbg = sel  # type: ignore
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
                idx = sel  # type: ignore

        if not args.allow_reuse:
            used_idx.add(idx)  # exclude within one video

        f=files[idx]
        before = counters.get(f.name, 0)
        counters[f.name]=before+1
        after = counters[f.name]
        chosen[i]=f

        # compute start time randomly (fit inside source), clamp dur to source length
        vdur = run_ffprobe_duration(f)
        eff_dur = min(max(0.10, per_clip[i]), max(0.10, vdur - 0.01))  # safety floors
        window = max(0.0, vdur - eff_dur - 0.25)
        start = rng.uniform(0, window) if window>0 else 0.0
        starts[i]=round(start,3)
        per_clip[i] = round(eff_dur, 3)

        mutes[i]=(f.stem.endswith("_m"))
        has_a[i]=has_audio_stream(f)
        sfxs[i]="sfx/purr.wav" if f.stem.endswith("-pr") else ("sfx/meow.wav" if f.stem.endswith("-mw") else None)

        if args.debug:
            print(f"[apply] {f.name}: counter {before} -> {after} (+1)")
            print(f"        start={start:.2f}s  dur={eff_dur:.2f}s  mute={mutes[i]}  has_audio={has_a[i]}")

    # Finalize lists (type: ignore safety)
    final_chosen: List[Path] = [c for c in chosen]  # type: ignore
    final_starts: List[float] = [float(s) for s in starts]  # type: ignore
    final_mutes:  List[bool]  = [bool(m) for m in mutes]    # type: ignore
    final_has_a:  List[bool]  = [bool(a) for a in has_a]    # type: ignore
    final_sfxs:   List[Optional[str]] = [s for s in sfxs]

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
        for idx, f in enumerate(final_chosen, 1):
            print(f"  [{idx:02d}] {f.name}")

    # build & run single ffmpeg
    build_and_run_ffmpeg(
        clips=final_chosen,
        starts=final_starts,
        durs=per_clip,
        mute_flags=final_mutes,
        has_audio_flags=final_has_a,
        sfx_flags=final_sfxs,
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

    # --- FINAL: show each footage duration from xx to xx on the output timeline ---
    try:
        print("\n[timeline] Footage ranges (on final video):")
        cum = 0.0
        for i, (f, d, s0) in enumerate(zip(final_chosen, per_clip, final_starts), 1):
            start_t = cum
            end_t = cum + d
            # snap last segment to audio length to avoid tiny rounding gaps
            if i == len(per_clip):
                end_t = total_audio
            print(f"  [{i:02d}] {f.name:<30} {_fmt_ts(start_t)} → {_fmt_ts(end_t)}  "
                  f"(dur={d:.3f}s, src_start={s0:.3f}s)")
            cum = end_t
        print(f"[timeline] Total: {_fmt_ts(0.0)} → {_fmt_ts(total_audio)}  (audio={total_audio:.3f}s)")
    except Exception as _e:
        print(f"[timeline] Skipped printing ranges: {_e}")

if __name__ == "__main__":
    import sys, os
    from pathlib import Path

    # print immediately in terminal
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    # --- MAIN PIPELINE ---
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"[fatal] main() error: {e}", flush=True)
        raise

    # --- POST: duration guard (exactly match WAV) ---
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
            target = audio_duration(audio_wav)
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
