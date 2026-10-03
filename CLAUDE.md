# CLAUDE.md — project rules (read automatically every session)

Project: **fifa-highlight-detector** — finds exciting moments in a football match video
from audio spikes + commentary keywords + visual activity, and cuts highlight clips.
Full design and module interfaces: **PLAN.md** (read it before writing code).

## How we work
- The project is built in phases. Only do the phase the user asks for.
- Do not rewrite or refactor files from earlier phases unless the current phase truly needs it.
  If you must change one, keep the PLAN.md interface and say what you changed.
- Follow the function names, signatures and column names in PLAN.md section 7 exactly.
- At the end of a phase: update the status table and Notes in PLAN.md, then give a short summary
  (files created, how to test, anything I must do).
- Do NOT run any git commands that change anything (`git add`, `git commit`, `git push`,
  `git reset`, `git clean`, ...). I commit myself after testing. Read-only `git status` / `git diff` is fine.
  Then stop.

## Environment
- Windows, Intel CPU, Intel Iris GPU, **no CUDA**. Everything must run well on CPU.
- Use a virtual environment in `.venv`. Create it in Phase 1 with the Windows launcher and Python 3.12:
  `py -3.12 -m venv .venv` (plain `python` on this PC may point to the Microsoft Store stub or to 3.14,
  which many audio/ML packages do not support yet).
  Always run Python and pip through it: `.venv\Scripts\python.exe` / `.venv\Scripts\python.exe -m pip`.
- Project path: `C:\projects\fifa-highlight-detector`. ffmpeg, ffprobe and git are installed and on PATH.
- Use `pathlib` for paths and `subprocess.run([...])` with argument lists (no shell strings),
  so Windows paths with spaces work.
- ffmpeg is called through subprocess. No moviepy.
- No paid APIs, no network calls at runtime (except Whisper's first model download).

## Code style
- Python 3.10+, type hints, short docstrings, `logging` instead of print inside `src/`.
- Graceful errors: missing ffmpeg, no audio track, corrupt/unreadable video → clear message.
- Long videos: stream frames, never load the whole video or audio into memory more than needed;
  cache intermediate results as described in PLAN.md section 5.
- Keep it simple. No extra frameworks, no classes where a function is enough.

## Testing
- Test videos live in `samples/` (git-ignored, never commit them):
  - `samples/test_match.mp4` — first 30 min of a real full match (unedited), H.264 + AAC, 720p, 30 fps.
    Contains 2 goals. Use this for Phases 1–4.
  - `samples/full_match_h264.mp4` — the full match (~1h40m), H.264. Use from Phase 5 onwards for the final test.
  - `samples/full_match.mp4` — original AV1 download. Do not use it (OpenCV may not decode AV1).
  - `samples/truth_test.csv` — true goal times for the test clip (`time,event`), used by evaluate.py.
- Unit tests use synthetic numpy signals only (no real video needed). Run with `pytest -q`.
