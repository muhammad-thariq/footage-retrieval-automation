"""
PCC interface backend.

Run:
    .\.venv\Scripts\Activate.ps1
    pip install fastapi uvicorn
    python server.py

Open http://127.0.0.1:8000
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).parent
PY = sys.executable  # use the same interpreter that launched uvicorn (respects venv)

app = FastAPI(title="PCC")

# job_id -> {queue, status, out_video, task}
jobs: dict[str, dict[str, Any]] = {}


# ---------- models ----------

class RunConfig(BaseModel):
    script: str = ""
    out_video: str = "heart_all_visual.mp4"
    audio_wav: str = "heart_all.wav"
    subs_srt: str = "heart_all.srt"
    footage_dir: str = "footage"
    size: str = "1080x1920"
    fit: str = "cover"
    bg_color: str = "black"
    bg_blur: str = "24:1"
    seed: int = 42
    epsilon: float = 0.05
    family_diversity: bool = False
    allow_reuse: bool = False
    debug: bool = True
    mix_source: bool = True
    mix_source_db: float = -15.0
    exclude_files: list[str] = []
    # stage toggles (match the user's PowerShell pipeline)
    run_tts: bool = True
    run_srt: bool = True
    run_color_fix: bool = True
    run_footage: bool = True
    run_hardsub: bool = True
    # hardsub options
    hardsub_keep_font_color: bool = True
    hardsub_color_order: str = "rgb"
    hardsub_margin_v_ratio: float = 0.24
    hardsub_base_scale: float = 0.056


# ---------- routes ----------

@app.get("/")
async def index():
    return FileResponse(ROOT / "interface.html")


@app.get("/api/script")
async def read_script():
    p = ROOT / "input.txt"
    text = p.read_text(encoding="utf-8") if p.exists() else ""
    return {"script": text}


@app.get("/api/files")
async def list_files():
    """Tell the UI which intermediates already exist so user can skip stages."""
    def info(name: str) -> dict[str, Any]:
        p = ROOT / name
        return {"exists": p.exists(), "size": p.stat().st_size if p.exists() else 0}
    return {
        "input.txt": info("input.txt"),
        "heart_all.wav": info("heart_all.wav"),
        "heart_all.srt": info("heart_all.srt"),
        "heart_all_visual.mp4": info("heart_all_visual.mp4"),
    }


@app.post("/api/run")
async def start_run(cfg: RunConfig):
    job_id = uuid.uuid4().hex[:8]
    queue: asyncio.Queue = asyncio.Queue()
    jobs[job_id] = {"queue": queue, "status": "running", "out_video": cfg.out_video, "task": None}
    task = asyncio.create_task(run_pipeline(job_id, cfg))
    jobs[job_id]["task"] = task
    return {"job_id": job_id}


@app.post("/api/cancel/{job_id}")
async def cancel(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "no such job"}, status_code=404)
    task = job.get("task")
    if task and not task.done():
        task.cancel()
    return {"ok": True}


@app.get("/api/stream/{job_id}")
async def stream(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "no such job"}, status_code=404)

    async def gen():
        q: asyncio.Queue = job["queue"]
        while True:
            msg = await q.get()
            if msg is None:
                yield "event: end\ndata: {}\n\n"
                break
            yield f"data: {json.dumps(msg)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/video/{filename}")
async def video(filename: str):
    path = (ROOT / filename).resolve()
    if not str(path).startswith(str(ROOT.resolve())) or not path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, media_type="video/mp4")


# ---------- pipeline ----------

STAGES = [
    ("tts", "TTS (kokoro_heart.py)"),
    ("srt", "Subtitles (stable-ts)"),
    ("color", "SRT color fix"),
    ("footage", "Build video (footage.py)"),
    ("hardsub", "Burn hardsub (burn_hardsub_fit_ass.py)"),
]


async def run_pipeline(job_id: str, cfg: RunConfig):
    q: asyncio.Queue = jobs[job_id]["queue"]

    async def emit(kind: str, data: Any):
        await q.put({"kind": kind, "data": data})

    final_video: str = cfg.out_video  # may be overridden by the hardsub stage
    try:
        # Persist script to input.txt before anything else
        if cfg.script.strip():
            (ROOT / "input.txt").write_text(cfg.script, encoding="utf-8")
            await emit("log", {"line": "[setup] wrote input.txt", "level": "info"})

        await emit("stages", {"stages": [{"id": s, "label": l, "status": "pending"} for s, l in STAGES]})

        # 1. TTS
        if cfg.run_tts:
            await emit("stage", {"id": "tts", "status": "running"})
            await run_cmd([PY, "kokoro_heart.py"], emit)
            await emit("stage", {"id": "tts", "status": "done"})
        else:
            await emit("stage", {"id": "tts", "status": "skipped"})

        # 2. stable-ts SRT
        if cfg.run_srt:
            await emit("stage", {"id": "srt", "status": "running"})
            # stable-ts prompts to overwrite if the output already exists,
            # which would hang our subprocess (no interactive stdin). Remove it first.
            srt_path = ROOT / cfg.subs_srt
            if srt_path.exists():
                srt_path.unlink()
                await emit("log", {"line": f"[srt] removed existing {cfg.subs_srt} to avoid overwrite prompt", "level": "info"})
            await run_cmd([
                "stable-ts", cfg.audio_wav,
                "--output", cfg.subs_srt,
                "--output_format", "srt",
                "--device", "cuda",
                "--language", "en",
                "--word_timestamps", "True",
                "--max_chars", "42",
                "--max_words", "4",
                "--condition_on_previous_text", "False",
                "--vad", "True",
            ], emit)
            await emit("stage", {"id": "srt", "status": "done"})
        else:
            await emit("stage", {"id": "srt", "status": "skipped"})

        # 3. SRT color fix (Python equivalent of the PowerShell replace)
        if cfg.run_color_fix:
            await emit("stage", {"id": "color", "status": "running"})
            srt = ROOT / cfg.subs_srt
            if srt.exists():
                text = srt.read_text(encoding="utf-8")
                before = text.count("#00ff00")
                text = text.replace("#00ff00", "#ff00ffff")
                srt.write_text(text, encoding="utf-8")
                await emit("log", {"line": f"[color] replaced {before} occurrence(s) of #00ff00 → #ff00ffff", "level": "ok"})
                await emit("stage", {"id": "color", "status": "done"})
            else:
                await emit("log", {"line": f"[color] {cfg.subs_srt} not found, skipping", "level": "warn"})
                await emit("stage", {"id": "color", "status": "skipped"})
        else:
            await emit("stage", {"id": "color", "status": "skipped"})

        # 4. footage.py
        if cfg.run_footage:
            await emit("stage", {"id": "footage", "status": "running"})
            args = [
                PY, "footage.py",
                "--footage_dir", cfg.footage_dir,
                "--input_txt", "input.txt",
                "--subs_srt", cfg.subs_srt,
                "--audio_wav", cfg.audio_wav,
                "--out_video", cfg.out_video,
                "--size", cfg.size,
                "--fit", cfg.fit,
                "--bg_blur", cfg.bg_blur,
                "--bg_color", cfg.bg_color,
                "--seed", str(cfg.seed),
                "--epsilon", str(cfg.epsilon),
                "--mix_source_db", str(cfg.mix_source_db),
                "--family_diversity", "on" if cfg.family_diversity else "off",
            ]
            if cfg.allow_reuse:
                args.append("--allow_reuse")
            if cfg.debug:
                args.append("--debug")
            if not cfg.mix_source:
                args.append("--no_mix_source")
            for f in cfg.exclude_files:
                if f.strip():
                    args += ["--exclude_file", f.strip()]
            await run_cmd(args, emit)
            await emit("stage", {"id": "footage", "status": "done"})
        else:
            await emit("stage", {"id": "footage", "status": "skipped"})

        # 5. burn_hardsub_fit_ass.py
        if cfg.run_hardsub:
            await emit("stage", {"id": "hardsub", "status": "running"})
            args = [
                PY, "burn_hardsub_fit_ass.py",
                "--ass_color_order", cfg.hardsub_color_order,
                "--margin_v_ratio", str(cfg.hardsub_margin_v_ratio),
                "--base_scale", str(cfg.hardsub_base_scale),
            ]
            if cfg.hardsub_keep_font_color:
                args.append("--keep_font_color")
            # Capture the hardsub script's "[OK] Wrote <path> (...)" line so the
            # UI previews/downloads the hardsubbed file, not the bare footage cut.
            hardsub_re = re.compile(r"^\[OK\] Wrote (.+?) \(font=")
            captured = await run_cmd(args, emit, capture_re=hardsub_re)
            if captured:
                hardsub_path = Path(captured.strip())
                if hardsub_path.exists():
                    final_video = hardsub_path.name
                else:
                    await emit("log", {"line": f"[hardsub] captured path missing on disk: {hardsub_path}", "level": "warn"})
            else:
                await emit("log", {"line": "[hardsub] could not parse output path; falling back to footage cut", "level": "warn"})
            await emit("stage", {"id": "hardsub", "status": "done"})
        else:
            await emit("stage", {"id": "hardsub", "status": "skipped"})

        await emit("done", {"out_video": final_video})

    except asyncio.CancelledError:
        await emit("log", {"line": "[run] cancelled by user", "level": "warn"})
        await emit("error", {"message": "cancelled"})
    except Exception as e:
        await emit("log", {"line": f"[run] error: {e}", "level": "err"})
        await emit("error", {"message": str(e)})
    finally:
        jobs[job_id]["status"] = "done"
        await q.put(None)


async def run_cmd(cmd: list[str], emit, capture_re: re.Pattern | None = None) -> str | None:
    """Run a subprocess and stream stdout as log events.

    If `capture_re` is given, returns the last regex match's group(1) (or the
    full match if there is no group), so callers can extract values the child
    only reports in its own output (e.g. the hardsub output filename).
    """
    await emit("log", {"line": "$ " + " ".join(cmd), "level": "cmd"})
    # Force UTF-8 on child stdio so Windows cp1252 doesn't choke on chars like →.
    # FORCE_COLOR / CLICOLOR_FORCE persuade rich/click/colorama tools to keep
    # emitting ANSI even though stdout is a pipe; the front-end parses them.
    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "FORCE_COLOR": "1",
        "CLICOLOR_FORCE": "1",
        "PY_COLORS": "1",
    }
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,  # no interactive prompts — fail fast on EOF
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(ROOT),
        env=env,
    )
    assert proc.stdout is not None
    captured: str | None = None
    try:
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if line:
                if capture_re is not None:
                    m = capture_re.search(line)
                    if m:
                        captured = m.group(1) if m.groups() else m.group(0)
                await emit("log", {"line": line, "level": "out"})
    except asyncio.CancelledError:
        proc.kill()
        raise
    rc = await proc.wait()
    if rc != 0:
        raise RuntimeError(f"command exited {rc}: {' '.join(cmd)}")
    return captured


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
