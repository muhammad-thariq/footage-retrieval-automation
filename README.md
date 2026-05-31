<div align="center">
    <h1>Footage Retrieval Automation</h1>
    <p>Automated vertical short-form video assembly that matches each script sentence to stock footage via semantic embeddings, then stitches clips, voiceover, and styled subtitles into a finished video.</p>

<img width="1980" height="1778" alt="127 0 0 1_8000_ (2)" src="https://github.com/user-attachments/assets/0ab153e8-5799-4112-b1cd-8be6b96f30ba" />

</div>

<div align="center">
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-gray?style=flat&logo=python&logoColor=white&labelColor=3776AB" alt="Python"></a>
    <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-0.110+-gray?style=flat&logo=fastapi&logoColor=white&labelColor=009688" alt="FastAPI"></a>
    <a href="https://uvicorn.dev/"><img src="https://img.shields.io/badge/Uvicorn-ASGI-gray?style=flat&logo=gunicorn&logoColor=white&labelColor=499848" alt="Uvicorn"></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.5+-gray?style=flat&logo=pytorch&logoColor=white&labelColor=EE4C2C" alt="PyTorch"></a>
    <a href="https://www.sbert.net/"><img src="https://img.shields.io/badge/Sentence--Transformers-5.1+-gray?style=flat&logo=huggingface&logoColor=FFD21E&labelColor=454545" alt="Sentence-Transformers"></a>
    <a href="https://github.com/openai/whisper"><img src="https://img.shields.io/badge/OpenAI%20Whisper-stable--ts-gray?style=flat&logo=openai&logoColor=white&labelColor=412991" alt="OpenAI Whisper"></a>
    <a href="https://huggingface.co/hexgrad/Kokoro-82M"><img src="https://img.shields.io/badge/Kokoro--82M-TTS-gray?style=flat&logo=huggingface&logoColor=FFD21E&labelColor=FF9D00" alt="Kokoro-82M"></a>
    <a href="https://ffmpeg.org/"><img src="https://img.shields.io/badge/FFmpeg-6.0+-gray?style=flat&logo=ffmpeg&logoColor=white&labelColor=007808" alt="FFmpeg"></a>
</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Concepts](#key-concepts)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Scenarios](#scenarios)
- [Setup](#setup)
- [Configuration](#configuration)
- [Execution](#execution)
- [Troubleshooting](#troubleshooting)
- [Model Licenses & Acknowledgments](#model-licenses--acknowledgments)
---

## Overview

<!-- Add a screenshot of the web dashboard here -->

**Footage Retrieval Automation** is a local, web-based tool that assembles a vertical short-form video from three inputs: a script (one sentence per line), a TTS voiceover WAV, and an SRT subtitle file. Each script sentence is matched to stock footage by scoring footage keywords (folder names + filenames) with **semantic embeddings** and lexical overlap, the best clips are stitched together with **FFmpeg** so the final length exactly matches the audio, and the subtitles are burned in as styled hardsubs.

It ships with a **FastAPI** backend and a single-page web UI that streams live pipeline logs over **Server-Sent Events (SSE)**, so you can run and re-run the pipeline without memorizing CLI flags.

### Core Features
*   **🌐 Web Interface**: Single-page dashboard (`interface.html`) with a script editor, per-stage toggles, a live streaming console, and an in-browser video preview.
*   **🧠 Semantic Footage Matching**: Ranks footage per sentence with `semantic_sim + 0.75 * lexical_match` using cached sentence-transformer embeddings.
*   **⏱️ SRT-Driven Timing**: Per-sentence durations come from first-word boundaries in the SRT timeline, not equal splits, so clips track the voiceover.
*   **🔊 Neural TTS**: Generates the voiceover with **Kokoro-82M** (`af_heart` voice) directly from `temp/input.txt`.
*   **✨ Precise Subtitling**: Uses **OpenAI Whisper** via `stable-ts` for word-level timestamps, then burns them as fitted **ASS** hardsubs.
*   **🎬 Vertical Adaptation**: Scales/crops or letterboxes footage into 1080×1920 (9:16) with `cover`, `contain`, or `blurpad` fit modes.
*   **🎵 Audio Mixing**: Mixes clip-native audio with the voiceover; loops `sfx/bell.mp3` under the whole video as a quiet bed track when present.
*   **♻️ Least-Used Selection**: Persistent counters spread footage usage across runs, with optional family-diversity and clip-exclusion controls.
*   **🪝 Stage Toggles**: Skip any of TTS / subtitles / color-fix / footage / hardsub to re-use existing intermediates.

---

## Key Concepts

| Concept | Description |
| :--- | :--- |
| **Keyword Indexing** | Subfolder names and filenames under `footage/` are scraped recursively into keywords that label each clip. |
| **Semantic Retrieval** | Sentence vectors and per-file embeddings (**Sentence-Transformers**) are compared, combined with lexical matching, to rank candidate clips per sentence. |
| **Boundary-Based Timing** | Each sentence's duration is the span between its first word and the next sentence's first word in the SRT word timeline. |
| **First/Last Pair Rule** | The first and last sentences share **one** clip: a contiguous window is carved so the second half plays under the opener and the first half under the closer. |
| **Neural TTS** | **Kokoro-82M** synthesizes a single concatenated voiceover WAV from the verbatim script lines. |
| **Hardsub Fitting** | SRT is converted to **ASS** with explicit play resolution so subtitles always fit and stay readable on a vertical frame. |

---

## Architecture

The project runs as a **FastAPI + Uvicorn** application that orchestrates a sequential pipeline as an async background job, streaming logs and per-stage status to the browser over **Server-Sent Events**. The backend shells out to the pipeline scripts with the project root as the working directory so relative paths (`footage/`, `sfx/`, `temp/`, `video-saved/`) resolve.

### Pipeline Stages

| No | Stage | Script / Tool | Output |
| :-- | :--- | :--- | :--- |
| 1 | TTS Synthesis — voiceover from the script | `kokoro_heart.py` | `temp/heart_all.wav` |
| 2 | Subtitle Alignment — word-level timestamps | `stable-ts` (Whisper) | `temp/heart_all.srt` |
| 3 | SRT Color Fix — normalize subtitle color tags | `server.py` (in-process) | `temp/heart_all.srt` |
| 4 | Footage Build — match, time, and concat clips | `footage.py` | `temp/heart_all_visual.mp4` |
| 5 | Hardsub Burn — fit + burn ASS subtitles | `burn_hardsub_fit_ass.py` | `video-saved/heart_all_visual_output.mp4` |

> The script itself is supplied via the UI (or `temp/input.txt`). A post-processing duration guard re-encodes the footage cut so it matches the WAV length exactly.

### API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serve the single-page web UI |
| `GET` | `/api/script` | Return the current `temp/input.txt` contents |
| `GET` | `/api/files` | Report which intermediates exist (so stages can be skipped) |
| `POST` | `/api/run` | Persist the script and start the pipeline job |
| `POST` | `/api/cancel/{job_id}` | Cancel a running job |
| `GET` | `/api/stream/{job_id}` | SSE stream of live logs and stage status |
| `GET` | `/api/video/{filename}` | Serve a finished (or intermediate) video |

### SSE Stream Events

Each SSE message is a JSON object with a `kind` and a `data` payload.

| `kind` | Direction | Description |
| :--- | :--- | :--- |
| `log` | Server → Client | A pipeline log line with a level (`cmd`, `out`, `info`, `ok`, `warn`, `err`) |
| `stages` | Server → Client | The full ordered stage list with initial status |
| `stage` | Server → Client | A single stage's status change (`running` / `done` / `skipped`) |
| `done` | Server → Client | Pipeline finished, with the final output filename |
| `error` | Server → Client | Pipeline error (or cancellation) with a message |

### Project Structure

```
footage-retrieval-automation/
├── server/
│   ├── server.py               # FastAPI backend + pipeline orchestration (SSE)
│   ├── kokoro_heart.py         # Kokoro-82M TTS synthesis (Stage 1)
│   ├── footage.py              # Semantic footage match + FFmpeg assembly (Stage 4)
│   └── burn_hardsub_fit_ass.py # SRT → ASS hardsub burn (Stage 5)
├── interface/
│   └── interface.html          # Single-page web UI (HTTP/SSE only)
├── footage/                    # Stock clips — keyword folders + files
│   ├── .selection_counters.json  # Per-clip use counts (least-used preference)
│   └── .file_embeddings.json     # Cached per-file embeddings
├── sfx/                        # meow.wav, purr.wav, bell.mp3 (bed track)
├── temp/                       # Intermediates: input.txt, output.txt,
│                               #   heart_all.wav, heart_all.srt, heart_all_visual.mp4
├── video-saved/                # Finished (hardsubbed, titled) videos
├── archive/                    # Retired legacy scripts + stray mp4s
├── script-prompt.md            # Claude system prompt for writing scripts
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## Tech Stack

| Category | Technologies |
| :--- | :--- |
| **Backend** | Python, FastAPI, Uvicorn, Pydantic, Server-Sent Events |
| **Frontend** | HTML5, Vanilla JS, Tailwind (CDN), `EventSource` (SSE) |
| **AI / Retrieval** | Sentence-Transformers, PyTorch, Transformers |
| **AI / Speech** | Kokoro-82M (TTS), OpenAI Whisper via `stable-ts` (alignment) |
| **Video / Audio** | FFmpeg, FFprobe, NumPy |

---

## Scenarios

| Scenario | Objective |
| :--- | :--- |
| **Full Web UI Run** | Paste a script, run all five stages, and preview/download the finished vertical short. |
| **Re-run a Single Stage** | Toggle off completed stages to re-use existing `temp/` intermediates (e.g. rebuild footage without re-running TTS). |
| **Footage CLI Build** | Run `footage.py` directly with custom fit, diversity, reuse, and exclusion flags. |
| **Standalone Hardsub** | Use `burn_hardsub_fit_ass.py` to burn fitted ASS subtitles onto any existing vertical video. |

---

## Setup

### Prerequisites
*   **Python 3.10+**
*   **FFmpeg & FFprobe** installed and on your system `PATH` (verify with `ffmpeg -version`).
*   **NVIDIA GPU (recommended)** with CUDA 12.1. The pinned `torch==2.5.1+cu121` build and `stable-ts --device cuda` expect CUDA; CPU-only works but is significantly slower.
*   **Stock footage** under `footage/`, organized into keyword subfolders (see [Footage conventions](#configuration)).

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/muhammad-thariq/footage-retrieval-automation.git
    cd footage-retrieval-automation
    ```

2.  Create and activate a virtual environment (recommended):
    ```powershell
    python -m venv .venv

    # Windows (PowerShell)
    .\.venv\Scripts\Activate.ps1

    # Linux / macOS
    source .venv/bin/activate
    ```

3.  Install dependencies (PyTorch with CUDA 12.1 is recommended):
    ```bash
    # Install the CUDA build of PyTorch first
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    # Then the rest of the requirements
    pip install -r requirements.txt
    ```

> First run downloads the Kokoro, Whisper, and sentence-transformer model weights; they are cached afterward.

---

## Configuration

The pipeline is driven by `RunConfig` in `server/server.py` (sent from the UI) and by CLI flags on the individual scripts.

| Setting | Where | Notes |
| :--- | :--- | :--- |
| **Output size / fit** | `RunConfig` / `footage.py` (`--size`, `--fit`) | Default `1080x1920`; fit is `cover` (crop), `contain` (black letterbox), or `blurpad` (blurred background). |
| **Selection behavior** | `footage.py` (`--family_diversity`, `--allow_reuse`, `--exclude_file`, `--seed`, `--epsilon`) | Control diversity, reuse, exclusions, and randomness of clip selection. |
| **Audio mix** | `RunConfig` (`mix_source`, `mix_source_db`) | Toggle clip-native audio and set its mix level against the voiceover. |
| **TTS voice / speed** | `kokoro_heart.py` (`VOICE`, `SPEED`, `LANG`) | Defaults: `af_heart`, `1.25×`, American English. |
| **Subtitle styling** | `burn_hardsub_fit_ass.py` (`--margin_v_ratio`, `--base_scale`, `--ass_color_order`, `--keep_font_color`) | Tune subtitle placement, size, and color handling. |
| **Server host/port** | `server/server.py` | Defaults to `127.0.0.1:8000`. |

**Footage naming conventions** (read by `footage.py`):

| Suffix / file | Effect |
| :--- | :--- |
| `_m` (stem) | Mute the clip's own audio (e.g. `run_1_m.mp4`). |
| `-mw` | Play `sfx/meow.wav` instead of the clip audio. |
| `-pr` | Play `sfx/purr.wav` instead of the clip audio. |
| `sfx/bell.mp3` | If present, looped under the entire video at 10% as a bed track (remove/rename to disable). |

**Persistent state** lives under `footage/`: `.selection_counters.json` (use counts; delete to reset) and `.file_embeddings.json` (embedding cache, regenerated when clips change).

---

## Execution

### Web UI (recommended)

```powershell
.\.venv\Scripts\Activate.ps1
pip install fastapi uvicorn      # first time only
python server/server.py          # serves http://127.0.0.1:8000
```

Then open **http://127.0.0.1:8000** in your browser.

**Workflow:**
1.  Paste or edit the **script** (one sentence per line) in the editor.
2.  Toggle which **stages** to run (TTS, subtitles, color-fix, footage, hardsub).
3.  Adjust options (size, fit, audio mix, hardsub styling) as needed.
4.  Click **Build** — logs and per-stage status stream live into the console.
5.  **Preview** the finished video in-browser when the job completes; finished videos land in `video-saved/`.

### CLI (footage builder)

Run the video assembly step directly (defaults read `footage/`, `temp/input.txt`, `temp/heart_all.srt`, `temp/heart_all.wav` → `temp/heart_all_visual.mp4`):

```powershell
# Basic run
python server/footage.py

# Common options
python server/footage.py --debug                      # verbose selection info
python server/footage.py --fit blurpad                # blurred letterbox background
python server/footage.py --fit contain                # black letterbox
python server/footage.py --family_diversity on        # avoid clips from the same family
python server/footage.py --allow_reuse                # allow reusing a clip across sentences
python server/footage.py --exclude_file fight_1_m-mw.mp4   # skip a specific clip
python server/footage.py --out_video temp/my.mp4      # custom output path
```

### CLI (full manual pipeline)

```powershell
# Stage 1 — TTS voiceover (reads temp/input.txt)
python server/kokoro_heart.py

# Stage 2 — word-level subtitles
stable-ts temp/heart_all.wav --output temp/heart_all.srt --output_format srt `
  --device cuda --language en --word_timestamps True --max_chars 42 --max_words 4

# Stage 4 — build the footage cut
python server/footage.py

# Stage 5 — burn fitted ASS hardsubs
python server/burn_hardsub_fit_ass.py --video_in temp/heart_all_visual.mp4 `
  --srt_in temp/heart_all.srt --video_out video-saved/heart_all_visual_output.mp4 `
  --keep_font_color --ass_color_order rgb --margin_v_ratio 0.24 --base_scale 0.056
```

---

## Troubleshooting

| Symptom | Likely cause & fix |
| :--- | :--- |
| `ffmpeg`/`ffprobe` not found | FFmpeg isn't on your `PATH`. Install it and verify with `ffmpeg -version`. |
| `stable-ts` step hangs | An existing `temp/heart_all.srt` triggers an overwrite prompt. The server deletes it first; if running manually, remove it before re-running. |
| `CUDA out of memory` / no CUDA | Close other GPU processes, or run `stable-ts` with `--device cpu` and install the CPU PyTorch build (slower). |
| `torch` installs a CPU-only build | Reinstall using the CUDA index URL in [Installation](#installation) **before** other requirements. |
| First run is very slow | Kokoro / Whisper / embedding weights download on first use and are cached afterward. |
| Wrong or repeated clips | Delete `footage/.selection_counters.json` to reset usage, or add `--exclude_file` / `--family_diversity on`. |
| Subtitles misaligned or off-screen | Tune `--margin_v_ratio` and `--base_scale` in the hardsub step for your footage. |
| Stale embeddings after adding clips | `footage/.file_embeddings.json` regenerates automatically on filename/mtime change; delete it to force a rebuild. |

---

## Model Licenses & Acknowledgments

This project builds on excellent open models — please review and comply with each upstream license before commercial use:

- **[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)** — neural TTS.
- **[OpenAI Whisper](https://github.com/openai/whisper)** (via [`stable-ts`](https://github.com/jianfch/stable-ts)) — word-level alignment.
- **[Sentence-Transformers](https://www.sbert.net/)** — sentence and footage embeddings.
- **[FFmpeg](https://ffmpeg.org/)** — video/audio assembly and encoding.

---

<div align="center">

*[Back to Top](#footage-retrieval-automation)*

</div>
