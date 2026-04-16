# kokoro_from_txt_single.py (preserve punctuation & spacing)
# - Reads input.txt verbatim (no normalization, no stripping)
# - Splits into safe chunks to avoid truncation
# - Synthesizes with Kokoro (af_heart), concatenates to ONE wav

from kokoro import KPipeline
import soundfile as sf
import torch
import numpy as np
from pathlib import Path
import re

INPUT_FILE = Path("input.txt")
OUTPUT_WAV = Path("heart_all.wav")
LANG = "a"           # American English
VOICE = "af_heart"   # heart voice
SPEED = 1
SR = 24000
SILENCE_MS = 15     # gap between synthesized chunks
MAX_LEN = 240        # ~chars per chunk; lower if you still see truncation

# Split after ., !, ?, or … when followed by whitespace.
# Punctuation is kept; we don't touch hyphens, commas, etc.
_SENT_SPLIT = re.compile(r"(?<=[.!?\u2026])\s+(?=[^\s])")

def split_into_sentences(text: str):
    parts = _SENT_SPLIT.split(text)
    return [p for p in parts if p] or [text]

def regroup_by_length(sentences, max_len=240):
    chunks, cur = [], ""
    for s in sentences:
        if not cur:
            cur = s
        elif len(cur) + 1 + len(s) <= max_len:
            cur = cur + " " + s  # only add a single space between sentences
        else:
            chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return chunks

def read_verbatim_lines(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"'{path}' not found. Put your lines in input.txt (UTF-8).")
    # Read exactly as-is (no strip). Keep lines that aren't purely whitespace.
    lines = path.read_text(encoding="utf-8").splitlines(keepends=False)
    return [ln for ln in lines if ln.strip() != ""]

def main():
    raw_lines = read_verbatim_lines(INPUT_FILE)
    if not raw_lines:
        raise ValueError("input.txt has no non-empty lines.")

    # Join with newline to preserve your intentional line breaks visually;
    # this does NOT change punctuation or inner spacing.
    text = "\n".join(raw_lines)

    sentences = split_into_sentences(text)
    chunks = regroup_by_length(sentences, max_len=MAX_LEN)

    print(f"Total sentences: {len(sentences)} | Synth chunks: {len(chunks)}")
    print("PyTorch CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))

    pipe = KPipeline(lang_code=LANG)
    silence = np.zeros(int(round(SR * (SILENCE_MS / 1000.0))), dtype=np.float32)
    combined = np.zeros(0, dtype=np.float32)

    for i, chunk in enumerate(chunks, start=1):
        print(f"[{i}/{len(chunks)}] {chunk[:80]}{'...' if len(chunk)>80 else ''}")
        gen = pipe(chunk, voice=VOICE, speed=SPEED)

        parts = []
        for _, _, audio in gen:
            parts.append(np.asarray(audio, dtype=np.float32).flatten())
        if not parts:
            print("  Warning: no audio returned for this chunk; skipping.")
            continue

        chunk_audio = np.concatenate(parts)
        combined = chunk_audio if combined.size == 0 else np.concatenate([combined, silence, chunk_audio])

    if combined.size == 0:
        raise RuntimeError("No audio was generated.")

    # Soft peak guard (no dynamics change, just avoids clipping)
    peak = float(np.max(np.abs(combined)))
    if peak > 0.99:
        combined = combined / peak * 0.99

    sf.write(str(OUTPUT_WAV), combined, SR)
    print(f"Done. Wrote '{OUTPUT_WAV}' ({len(combined)/SR:.2f}s).")

if __name__ == "__main__":
    main()
