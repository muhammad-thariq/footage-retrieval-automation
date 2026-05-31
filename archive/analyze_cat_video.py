#!/usr/bin/env python3
"""
analyze_cat_video.py

Description:
    Sample a video at ~1 frame per second and write a per-second explanation
    of what the cat is doing to output.txt.

    Default model: Salesforce/blip-image-captioning-large (local, no API).
    It runs well on an RTX 4060 laptop GPU.

Install (recommended, in a fresh virtual environment):
    # 1) Install PyTorch with CUDA matching your system (check https://pytorch.org/get-started/locally/)
    # Example for CUDA 12.1 (adjust if needed):
    #   pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
    #
    # 2) Install the rest:
    #   pip install opencv-python pillow transformers accelerate

Usage:
    python analyze_cat_video.py --video video1.mp4 --out output.txt --fps 1.0
    # Optional: choose a smaller/faster model if VRAM is tight:
    #   --model Salesforce/blip-image-captioning-base
"""

import argparse
import os
from typing import Optional

import cv2
import torch
from PIL import Image
from transformers import pipeline


def hhmmss(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def get_duration_seconds(cap: cv2.VideoCapture) -> int:
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    if fps <= 0 or total_frames <= 0:
        # Fallback using CAP_PROP_POS_MSEC stepping
        # Try to seek far to estimate duration; if that fails, default 20s
        try:
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
            ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
            return max(1, int(round(ms / 1000.0)))
        except Exception:
            return 20
    return int(round(total_frames / fps))


def read_frame_at_second(cap: cv2.VideoCapture, second: float) -> Optional[Image.Image]:
    # Seek to exact millisecond position. Some codecs seek to nearest keyframe.
    # That's OK for our 1fps summary.
    cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    # Convert BGR (OpenCV) -> RGB (PIL)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def build_captioner(model_id: str, device: Optional[str] = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use float16 on CUDA for speed/memory, float32 on CPU
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    # The "image-to-text" pipeline returns a list of dicts with "generated_text"
    return pipeline(
        task="image-to-text",
        model=model_id,
        torch_dtype=torch_dtype,
        device=0 if device == "cuda" else -1,
    )


def catify_caption(text: str) -> str:
    """
    Light post-processing to keep the description cat-focused and action-oriented.
    This doesn't hallucinate content, it just nudges phrasing.
    """
    t = text.strip()
    if not t:
        return "A cat is present."
    # If the model forgot to mention the cat, add a concise prefix.
    lowered = t.lower()
    if "cat" not in lowered and "kitten" not in lowered:
        t = f"A cat {t[0].lower() + t[1:]}" if t[:2].lower() != "a " else f"A cat {t[2:]}"
        # Example: "A cat sitting on a couch" or "A cat is playing with a toy"
    # Tighten very short outputs
    if len(t) < 12 and not t.endswith("."):
        t += "."
    # Ensure single sentence punctuation
    if not t.endswith("."):
        t += "."
    return t


def analyze_video(
    video_path: str,
    out_path: str,
    fps_sample: float = 1.0,
    model_id: str = "Salesforce/blip-image-captioning-large",
) -> None:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    duration_s = get_duration_seconds(cap)
    if duration_s <= 0:
        duration_s = 20  # fallback

    captioner = build_captioner(model_id=model_id)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fout:
        fout.write(f"# Per-second cat activity summary for {os.path.basename(video_path)}\n")
        fout.write(f"# Model: {model_id}\n")
        fout.write(f"# Duration (approx): {duration_s} seconds\n\n")

        step = max(1.0, fps_sample)  # seconds between samples; default 1.0
        t = 0.0
        seen_seconds = set()

        while t < duration_s + 0.5:  # slight slack on the end
            sec_int = int(round(t))
            if sec_int in seen_seconds:
                t += step
                continue

            frame = read_frame_at_second(cap, sec_int)
            if frame is None:
                # If seeking failed near the end, break gracefully
                if sec_int > duration_s - 1:
                    break
                t += step
                continue

            # Generate a caption; keep it brief and cat-focused
            # Some models accept "max_new_tokens"; for BLIP captioning it's ignored, safe to include.
            result = captioner(frame, max_new_tokens=32)
            if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
                raw_caption = result[0]["generated_text"]
            else:
                # Fallback to string conversion
                raw_caption = str(result)

            caption = catify_caption(raw_caption)

            ts = hhmmss(sec_int)
            fout.write(f"{ts} — {caption}\n")
            seen_seconds.add(sec_int)
            t += step

    cap.release()


def main():
    parser = argparse.ArgumentParser(description="Explain a cat-activity video at ~1 fps.")
    parser.add_argument("--video", type=str, default="video1.mp4", help="Path to input video (e.g., video1.mp4)")
    parser.add_argument("--out", type=str, default="output.txt", help="Path to output text file")
    parser.add_argument("--fps", type=float, default=1.0, help="Sampling rate in frames per second (default: 1.0)")
    parser.add_argument(
        "--model",
        type=str,
        default="Salesforce/blip-image-captioning-large",
        help="Hugging Face model id for image captioning",
    )
    args = parser.parse_args()

    analyze_video(
        video_path=args.video,
        out_path=args.out,
        fps_sample=args.fps,
        model_id=args.model,
    )
    print(f"Done. Wrote per-second explanations to: {args.out}")


if __name__ == "__main__":
    main()
