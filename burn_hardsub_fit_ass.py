# burn_hardsub_fit_ass.py
# Burn SRT into hardsubs that ALWAYS fit inside the frame (e.g., 1080x1920).
# - Converts SRT -> ASS with explicit PlayResX/PlayResY
# - Preserves <font color="#RRGGBB"> or "#RGB" to ASS inline colours
# - Ignores ASS tags while measuring/wrapping so colours don't shrink the font
# - Lets you flip inline colour order (BGR vs RGB) and 6/8-digit form to fix env-specific swaps
# - Windows-safe escaping for ffmpeg subtitles filter.

import argparse
import subprocess
import tempfile
import re
import html
from pathlib import Path

# ---------- ffprobe ----------

def ffprobe(args:list) -> str:
    out = subprocess.check_output(["ffprobe","-v","error",*args], stderr=subprocess.STDOUT, text=True)
    return out.strip()

def probe_resolution(video: Path):
    try:
        line = ffprobe(["-select_streams","v:0","-show_entries","stream=width,height","-of","csv=s=x:p=0", str(video)])
        w,h = line.split("x")
        return int(w), int(h)
    except Exception:
        return None, None

# ---------- SRT parsing ----------

def read_srt_utf8(srt_in: Path) -> str:
    data = srt_in.read_bytes()
    for enc in ("utf-8-sig","utf-8","cp1252","latin-1"):
        try:
            return data.decode(enc).replace("\r\n","\n").replace("\r","\n")
        except Exception:
            continue
    return data.decode("utf-8", errors="replace").replace("\r\n","\n").replace("\r","\n")

SRT_BLOCK = re.compile(r"\n?\s*\d+\s*\n(\d\d:\d\d:\d\d,\d\d\d)\s*-->\s*(\d\d:\d\d:\d\d,\d\d\d)\s*\n([\s\S]*?)(?=\n\s*\n|\Z)")
TS = re.compile(r"^(\d\d):(\d\d):(\d\d),(\d\d\d)$")

def srt_time_to_ass(ts: str) -> str:
    m = TS.match(ts)
    h,mn,s,ms = map(int, m.groups())
    cs = int(round(ms/10.0))
    return f"{h}:{mn:02d}:{s:02d}.{cs:02d}"

# ---------- HTML -> ASS sanitization ----------

# Support #RGB and #RRGGBB
FONT_RE = re.compile(
    r"<font[^>]*color=\s*[\"']?#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})[\"']?[^>]*>(.*?)</font>",
    re.I | re.S
)
BR_RE   = re.compile(r"<br\s*/?>", re.I)
TAG_RE  = re.compile(r"<[^>]+>")

# ---------- ASS tag awareness (fix for font shrink) ----------

ASS_TAG = re.compile(r"{\\[^}]*}")  # matches {\1c...}, {\r}, {\b1}, etc.

def visible_text(s: str) -> str:
    """Strip ASS override tags so we measure only what's actually drawn."""
    return ASS_TAG.sub("", s)

def visible_len(s: str) -> int:
    return len(visible_text(s))

# ---------- fit & wrap ----------

def measure_fontsize_to_fit(W:int, H:int, texts:list[str], safe_ratio:float, base_scale:float, char_aspect:float) -> int:
    """Pick a font size so the longest unbreakable word fits inside W*safe_ratio.
    Approximate average glyph width = fontsize * char_aspect.
    """
    safe_w = int(W * safe_ratio)
    longest_word = 1
    for t in texts:
        for w in re.findall(r"\S+", t):
            longest_word = max(longest_word, len(w))
    fs_base = max(16, int(round(H * base_scale)))
    fs_word_limit = max(12, int(safe_w / max(1, int(longest_word * char_aspect))))
    return max(16, min(fs_base, fs_word_limit))

def wrap_text_to_width(text: str, max_chars:int) -> str:
    # Preserve manual line breaks; measure using visible length (ignore ASS tags)
    text = text.replace("\r", "\n")
    parts = text.split("\n")
    out_lines = []
    for part in parts:
        words = part.strip().split()
        if not words:
            continue
        cur = words[0]
        for w in words[1:]:
            if visible_len(cur) + 1 + visible_len(w) <= max_chars:
                cur += " " + w
            else:
                out_lines.append(cur)
                cur = w
        out_lines.append(cur)
    return "\\N".join(out_lines)

# ---------- ASS build ----------

ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
ScaledBorderAndShadow: yes
WrapStyle: 2
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{fs},{primary},{primary},{outline},&H64000000,0,0,0,0,100,100,0,0,1,{outline_px},{shadow},{align},{margin_l},{margin_r},{margin_v},1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

COLOUR_WHITE = "&H00FFFFFF"   # &HAABBGGRR (AA=00)
COLOUR_BLACK = "&H00000000"
# ---------- colour helpers ----------

def rgb_to_ass(hexcode: str, order: str = "bgr", with_alpha: bool = True) -> str:
    """
    Convert '#RRGGBB' (or 'RGB') to ASS override colour.
    order='bgr' -> &H00BBGGRR (standard ASS format with alpha)
    order='rgb' -> &H00RRGGBB (fallback if env swaps channels)
    with_alpha=True -> return &H00BBGGRR (ABGR with AA=00 for full opacity)
    """
    hexcode = hexcode.strip().lstrip("#")
    if len(hexcode) == 3:  # expand #rgb -> #rrggbb
        hexcode = "".join(ch*2 for ch in hexcode)
    r = int(hexcode[0:2], 16); g = int(hexcode[2:4], 16); b = int(hexcode[4:6], 16)

    if order == "bgr":      # spec / libass default for inline
        X1, X2, X3 = b, g, r  # BB GG RR
    else:                   # some environments appear swapped
        X1, X2, X3 = r, g, b  # RR GG BB

    if with_alpha:
        return f"&H00{X1:02X}{X2:02X}{X3:02X}"  # &H00BBGGRR (AA=00 for full opacity)
    return f"&H{X1:02X}{X2:02X}{X3:02X}"        # &HBBGGRR (or &HRRGGBB if order=rgb)

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Burn SRT into MP4 with guaranteed-fit hardsubs (HTML-safe)")
    ap.add_argument("--video_in", type=str, default="output_9x16_letterbox.mp4")
    ap.add_argument("--srt_in",   type=str, default="heart_all.srt")
    ap.add_argument("--video_out", type=str, default="heart_all_visual_hardsub.mp4")

    # appearance & fit
    ap.add_argument("--font", type=str, default="Arial")
    ap.add_argument("--base_scale", type=float, default=0.068, help="Base font as H*scale before fitting")
    ap.add_argument("--safe_ratio", type=float, default=0.72, help="Safe width = W*ratio (side margins)")
    ap.add_argument("--char_aspect", type=float, default=0.58, help="Avg glyph width = fontsize*char_aspect")
    ap.add_argument("--align", type=int, default=2, help="ASS alignment (2=bottom-center)")
    ap.add_argument("--margin_v_ratio", type=float, default=0.10)
    ap.add_argument("--margin_lr_ratio", type=float, default=0.14)
    ap.add_argument("--outline_px", type=int, default=3)
    ap.add_argument("--shadow", type=int, default=0)
    ap.add_argument("--primary", type=str, default=COLOUR_WHITE)
    ap.add_argument("--outline", type=str, default=COLOUR_BLACK)
    ap.add_argument("--keep_font_color", action="store_true", help="Convert <font color> to ASS and keep colours; otherwise strip all HTML tags")

    # inline colour behaviour (to fix green→blue on some builds)
    ap.add_argument("--ass_color_order", choices=["bgr","rgb"], default="bgr",
                    help="Inline ASS override colour order. 'bgr' is spec; use 'rgb' if colours look swapped in your env.")
    ap.add_argument("--ass_override_len", choices=["6","8"], default="8",
                    help="Use 6-digit (&HBBGGRR) or 8-digit (&H00BBGGRR) inline override. Some builds prefer 8.")

    # encode
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--preset", type=str, default="veryfast")
    ap.add_argument("--fps", type=int, default=30)

    args = ap.parse_args()
    # --- Set video_out from first line of input.txt (if present) ---
    # Keep everything else unchanged.
    input_title_file = Path("input.txt")
    if input_title_file.exists():
        try:
            first_line = input_title_file.read_text(encoding="utf-8", errors="ignore").splitlines()[0].strip()
        except Exception:
            first_line = ""
        # Remove characters illegal in Windows/macOS filenames and control chars
        safe_title = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "", first_line).strip().rstrip(".")
        if safe_title:
            # Keep the same directory as the original --video_out argument, only change the filename
            _out_dir = Path(args.video_out).resolve().parent
            args.video_out = str((_out_dir / f"{safe_title}.mp4").resolve())

    video_in = Path(args.video_in).resolve()
    srt_in   = Path(args.srt_in).resolve()
    video_out= Path(args.video_out).resolve()

    if not video_in.exists():
        raise SystemExit(f"Video not found: {video_in}")
    if not srt_in.exists():
        raise SystemExit(f"SRT not found: {srt_in}")

    W,H = probe_resolution(video_in)
    if not W or not H:
        W,H = 1080,1920

    # colorizer closure for html_to_ass
    def colorize_hex(hx: str) -> str:
        return rgb_to_ass(
            hx,
            order=args.ass_color_order,
            with_alpha=(args.ass_override_len == "8")
        )

    def html_to_ass(text: str, keep_color: bool) -> str:
        t = html.unescape(text)
        if keep_color:
            def _repl(m):
                col = colorize_hex(m.group(1))
                inner = html_to_ass(m.group(2), keep_color=True)
                # Use \1c for primary color (the actual text color)
                # Don't use & at the end, just the color code
                return "{\\1c" + col + "}" + inner + "{\\r}"
            t = re.sub(FONT_RE, _repl, t)
        # treat <br> as newline for wrapping
        t = re.sub(BR_RE, " \n ", t)
        # drop any remaining tags (i, b, u, span ...)
        t = re.sub(TAG_RE, "", t)
        return t

    srt_txt = read_srt_utf8(srt_in)
    blocks = SRT_BLOCK.findall(srt_txt)
    if not blocks:
        raise SystemExit("No subtitles found in SRT")

    # HTML sanitize first to avoid literal <font ...> output
    cleaned_texts = [html_to_ass(b[2], keep_color=args.keep_font_color) for b in blocks]

    # ignore ASS tags when computing font size so coloured words don't shrink the font
    texts_for_fit = [visible_text(t).replace("\n"," ") for t in cleaned_texts]
    fs = measure_fontsize_to_fit(W, H, texts_for_fit, args.safe_ratio, args.base_scale, args.char_aspect)

    safe_w = int(W * args.safe_ratio)
    max_chars = max(12, int(safe_w / max(1, int(fs * args.char_aspect))))

    margin_l = int(round(W * args.margin_lr_ratio))
    margin_r = margin_l
    margin_v = int(round(H * args.margin_v_ratio))

    header = ASS_HEADER.format(
        W=W, H=H,
        font=args.font,
        fs=fs,
        primary=args.primary,
        outline=args.outline,
        outline_px=args.outline_px,
        shadow=args.shadow,
        align=args.align,
        margin_l=margin_l,
        margin_r=margin_r,
        margin_v=margin_v,
    )

    ass_lines = [header]
    for (start,end,_txt), clean in zip(blocks, cleaned_texts):
        # wrap after cleaning; respect manual \n from <br>; width math ignores ASS tags
        wrapped = wrap_text_to_width(clean, max_chars)
        ass_lines.append(f"Dialogue: 0,{srt_time_to_ass(start)},{srt_time_to_ass(end)},Default,,0,0,0,,{wrapped}")

    tmp_ass_dir = Path(tempfile.mkdtemp(prefix="assfit_"))
    tmp_ass = tmp_ass_dir / (srt_in.stem + "_fit.ass")
    tmp_ass.write_text("\n".join(ass_lines), encoding="utf-8")

    # Debug: print the first few lines of the ASS file
    print("[DEBUG] First few dialogue lines from ASS:")
    for line in ass_lines[:15]:
        if line.startswith("Dialogue:"):
            print(line[:200])

    # Windows-safe escaping for subtitles path
    sub_path = tmp_ass.as_posix().replace(':', r'\:').replace("'", r"\'")
    vf = f"subtitles='{sub_path}'"

    cmd = [
        "ffmpeg","-y",
        "-i", str(video_in),
        "-vf", vf,
        "-r", str(args.fps),
        "-c:v","libx264","-preset",args.preset,"-crf",str(args.crf),"-pix_fmt","yuv420p",
        "-c:a","copy",
        "-movflags","+faststart",
        str(video_out)
    ]

    try:
        subprocess.check_call(cmd)
        print(f"[OK] Wrote {video_out} (font={fs}px, max_chars_per_line={max_chars})")
        print(f"[INFO] Inline colour order='{args.ass_color_order}', override_len='{args.ass_override_len}'")
        print(f"[INFO] Temp ASS file saved at: {tmp_ass}")
    finally:
        # Comment out cleanup so you can inspect the ASS file
        # try:
        #     import shutil as _sh
        #     _sh.rmtree(tmp_ass_dir, ignore_errors=True)
        # except Exception:
        #     pass
        pass


if __name__ == "__main__":
    main()

# flags