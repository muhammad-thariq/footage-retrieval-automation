#!/usr/bin/env python3
import argparse
import math
from typing import Tuple

from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    concatenate_videoclips,
    CompositeVideoClip,
    CompositeAudioClip,
    ColorClip,
)
import moviepy.video.fx.all as vfx
import moviepy.audio.fx.all as afx


def parse_ratio(s: str) -> float:
    s = s.strip()
    if ":" in s:
        a, b = s.split(":")
        return float(a) / float(b)
    if "/" in s:
        a, b = s.split("/")
        return float(a) / float(b)
    return float(s)


def make_even(x: int) -> int:
    return x if x % 2 == 0 else x - 1 if x > 1 else 2


def center_crop_to_ratio(clip: VideoFileClip, target_ratio: float):
    w, h = clip.size
    cur_ratio = w / h
    if math.isclose(cur_ratio, target_ratio, rel_tol=1e-3):
        return clip

    if cur_ratio > target_ratio:
        new_w = int(h * target_ratio)
        x1 = (w - new_w) // 2
        x2 = x1 + new_w
        return clip.crop(x1=x1, x2=x2)
    else:
        new_h = int(w / target_ratio)
        y1 = (h - new_h) // 2
        y2 = y1 + new_h
        return clip.crop(y1=y1, y2=y2)


def letterbox_to_ratio(clip: VideoFileClip, target_ratio: float, bg_color=(0, 0, 0)):
    w, h = clip.size
    cur_ratio = w / h
    if math.isclose(cur_ratio, target_ratio, rel_tol=1e-3):
        return clip

    if cur_ratio > target_ratio:
        new_h = int(round(w / target_ratio))
        new_h = make_even(new_h)
        canvas = ColorClip(size=(w, new_h), color=bg_color, duration=clip.duration)
        y = (new_h - h) // 2
        comp = CompositeVideoClip([canvas, clip.set_position(("center", y))])
        comp = comp.set_audio(clip.audio)
        comp.duration = clip.duration
        comp.fps = clip.fps
        return comp
    else:
        new_w = int(round(h * target_ratio))
        new_w = make_even(new_w)
        canvas = ColorClip(size=(new_w, h), color=bg_color, duration=clip.duration)
        x = (new_w - w) // 2
        comp = CompositeVideoClip([canvas, clip.set_position((x, "center"))])
        comp = comp.set_audio(clip.audio)
        comp.duration = clip.duration
        comp.fps = clip.fps
        return comp


def resize_to_size(clip: VideoFileClip, size: Tuple[int, int]):
    w, h = size
    w, h = make_even(int(w)), make_even(int(h))
    return clip.resize((w, h))


def roll_last_seconds_first(clip: VideoFileClip, last_seconds: float):
    dur = clip.duration or 0
    if last_seconds <= 0 or last_seconds >= dur:
        return clip
    tail = clip.subclip(dur - last_seconds, dur)
    head = clip.subclip(0, dur - last_seconds)
    return concatenate_videoclips([tail, head], method="compose")


def force_video_to_duration(clip: VideoFileClip, target_duration: float) -> VideoFileClip:
    """If clip shorter -> loop to target_duration; if longer -> trim."""
    cur = clip.duration or 0
    if abs(cur - target_duration) < 1e-3:
        return clip
    if cur < target_duration:
        # loop
        looped = clip.fx(afx.audio_loop, duration=target_duration) if clip.audio else clip
        if not clip.audio:
            # for pure video we need video-looping:
            looped = clip.loop(duration=target_duration)
        else:
            # for video+audio we must loop both
            looped = clip.loop(duration=target_duration)
        return looped.set_duration(target_duration)
    else:
        # longer -> cut
        return clip.subclip(0, target_duration)


def main():
    parser = argparse.ArgumentParser(
        description="Convert video to 9:16, move last seconds to start, then mix heart audio + lowered original."
    )
    parser.add_argument("--input", "-i", type=str, default="video1.mp4", help="Input video path")
    parser.add_argument("--output", "-o", type=str, default="output_9x16.mp4", help="Output video path")
    parser.add_argument("--audio", "-a", type=str, default="heart_all.wav", help="Heartbeat / bgm audio path")
    parser.add_argument("--ratio", type=str, default="9:16", help="Target aspect ratio, e.g., '9:16'")
    parser.add_argument("--size", type=str, default="1080x1920", help="Final pixel size WxH, e.g., 1080x1920")
    parser.add_argument("--last", type=float, default=3.0, help="Seconds to move from end to start (default: 3)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fill", action="store_true", help="Fill the 9:16 frame by center-cropping (default)")
    mode.add_argument("--letterbox", action="store_true", help="Keep full image with black bars (no cropping)")
    parser.add_argument("--preset", type=str, default="medium", help="ffmpeg/libx264 preset")
    parser.add_argument("--crf", type=int, default=18, help="ffmpeg CRF quality (lower=better)")
    args = parser.parse_args()

    target_ratio = parse_ratio(args.ratio)

    if "x" in args.size.lower():
        w_str, h_str = args.size.lower().split("x")
        out_w, out_h = int(w_str), int(h_str)
    else:
        raise SystemExit("--size must be like 1080x1920 (WxH).")

    # --- load audio first (this defines target duration) ---
    heart_audio = AudioFileClip(args.audio)
    target_dur = heart_audio.duration

    with VideoFileClip(args.input) as clip:
        # ensure video duration == heart_all.wav duration (your first requirement)
        clip = force_video_to_duration(clip, target_dur)

        # normalize orientation (force reading)
        clip = clip.fx(vfx.rotate, 0)

        # aspect
        if args.letterbox:
            processed = letterbox_to_ratio(clip, target_ratio)
        else:
            processed = center_crop_to_ratio(clip, target_ratio)

        # resize
        processed = resize_to_size(processed, (out_w, out_h))

        # rearrange time
        rolled = roll_last_seconds_first(processed, args.last)

        # we want final duration to match audio (in case roll changed anything odd)
        rolled = rolled.set_duration(target_dur)

        # --- AUDIO MIXING ---
        # 1) original video audio, lowered to 25% (i.e. 75% lower)
        if rolled.audio is not None:
            orig_low = rolled.audio.volumex(0.25).set_duration(target_dur)
        else:
            orig_low = None

        # 2) heart_all.wav must also match duration (loop or cut)
        if heart_audio.duration < target_dur - 1e-3:
            heart_ready = heart_audio.fx(afx.audio_loop, duration=target_dur).set_duration(target_dur)
        else:
            heart_ready = heart_audio.subclip(0, target_dur)

        if orig_low is not None:
            final_audio = CompositeAudioClip([heart_ready, orig_low]).set_duration(target_dur)
        else:
            final_audio = heart_ready.set_duration(target_dur)

        rolled = rolled.set_audio(final_audio)

        ffmpeg_params = ["-movflags", "+faststart", "-crf", str(args.crf)]
        rolled.write_videofile(
            args.output,
            codec="libx264",
            audio_codec="aac",
            fps=clip.fps,
            preset=args.preset,
            ffmpeg_params=ffmpeg_params,
            threads=0,
            verbose=False,
            logger=None,
            audio_bitrate="192k",
        )

    heart_audio.close()


if __name__ == "__main__":
    main()
