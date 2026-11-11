# fifa-highlight-detector

Finds the most exciting moments in a football match video (goals, big chances and saves, cards and fouls) and cuts
them into highlight clips and one highlight reel. It listens to the crowd, reads the commentary and watches the
on-screen action. Everything runs **locally on a CPU**, with no paid APIs and no GPU.

## What it does

Given a local match video (`.mp4` / `.mkv`, 10–90+ min) it produces, in `outputs/<video_name>/`:

- `clips/01_06-01_Goal.mp4`, … one clip per highlight (`<rank>_<mm-ss>_<event type>.mp4`)
- `highlight_reel.mp4`: all clips stitched together in match order
- `highlights.csv` / `highlights.json`: rank, time, score, event type, a reason and a commentary snippet for each moment
- `thumbs/*.jpg` and `signals.csv` (the per-second signals and score, for inspection)

You can use it from the command line (`cli.py`) or from a Streamlit web app (`app.py`) with a score timeline and one
card per highlight.

## Demo

> _Screenshot placeholder: add a screenshot of the app to `docs/screenshot.png`._
>
> `![App screenshot](docs/screenshot.png)`

## How it works

```
video ──► audio (ffmpeg, 16 kHz mono) ──► crowd loudness features ──► audio_z ─┐
      ├─► commentary (faster-whisper)  ──► keyword hits ────────────► keyword ─┼─► fused score per second
      └─► frames (OpenCV, 2 fps)       ──► motion, cuts, replays ───► video ───┘        │
                                                                                         ▼
               clips + reel + json/csv ◄── classifier (Goal / Chance / Card / Exciting) ◄── peaks (top N)
```

**1. Audio: crowd noise (`audio_z`).** Per second: loudness (RMS), spectral flux (sudden changes) and energy in the
300–3000 Hz crowd band. Each is turned into a *robust rolling z-score* and the three are averaged.

**2. Commentary keywords (`keyword_score`).** Whisper transcribes the commentary with word timestamps. Weighted phrases
from `config.yaml` ("goal", "what a save", "it's 1-0", "red card", …) score points. A hit gets a boost when the word is
stretched ("goooal"), repeated, or shouted (Whisper ends the sentence with "!"). Hits are spread over ±3 s
because commentary lags the action.

**3. Video activity (`video_z`, `replay`).** Frames are sampled at 2 fps and shrunk to 320 px. Motion = mean frame
difference, a scene cut = a sharp change of the color histogram. Several cuts within 10 s mark a **replay**. Replays
usually come *after* a big moment, so a replay up to 30 s ahead gives credit to the current second.

**Robust rolling z-score.** Every match and broadcast has a different volume and camera style, so raw values can't
be compared between videos. For each second we compare the value with its neighbourhood (a centered 60 s window):

```
z = (x − rolling_median) / (1.4826 · rolling_MAD + ε)
```

The median and the MAD (median absolute deviation) ignore short spikes, so a goal roar stands out clearly
against the normal noise around it instead of raising the baseline itself.

**Fused score.** Each signal is first scaled to 0..1, then combined:

```
audio_n       = clip(audio_z, 0, 4) / 4
video_n       = 0.7 · replay_ahead + 0.3 · clip(video_z, 0, 6) / 6
keyword_gated = keyword_score · (0.3 + 0.7 · max audio_n within ±10 s)     # recap talk over a calm crowd keeps 30 %
score_raw     = w_audio · audio_n + w_keywords · keyword_gated + w_video · video_n
score         = 5 s centered moving average of score_raw
```

Default weights are `audio 0.5, keywords 0.35, video 0.15`, normalized to sum to 1.

**Peaks → highlights.** `scipy.signal.find_peaks` picks the top-N peaks, at least 30 s apart. Each clip runs from
10 s before to 8 s after its peak; overlapping clips are merged.

**Classifier (rules).** It looks at the commentary and keywords from 5 s before to 20 s after the peak and at a replay within 60 s:
- **Goal**: a live confirmation phrase ("scores!", "it's 2-1", "gives Liverpool the lead", "pulls one back", "levels the
  score", "converts the penalty", "the goal stands", …) that is not undone by a later cancel phrase ("disallowed",
  "ruled out", "saved", "just wide", "VAR is checking", …), plus a crowd reaction (a low audio floor).
  Plain "goal", "lead" or "level" never confirm. Both phrase lists are data in `config.yaml` (`classifier` section),
  with simple patterns: `<team>` = any 1–3 words, `<score>` = a scoreline ("1-0", "two-one"), a trailing `!` = must be exclaimed.
- **Big Chance / Save**: "save", "chance", "post", "crossbar", …
- **Card / Foul**: "card", "penalty", "foul", "VAR", …
- otherwise **Exciting Moment**

The reason text explains the choice, for example `crowd spike (audio z=2.5) + confirmed by 'it's 1-0' + replay detected`.

## Project structure

```
fifa-highlight-detector/
├── app.py                  # Streamlit web app
├── cli.py                  # command line entry
├── evaluate.py             # precision / recall against a truth CSV
├── config.yaml             # all parameters (weights, keywords, windows)
├── requirements.txt
├── .streamlit/config.toml  # app theme
├── src/
│   ├── utils.py            # config, ffmpeg check, paths, cache rule, robust z-score
│   ├── audio_extractor.py  # ffmpeg → 16 kHz mono wav
│   ├── audio_features.py   # rms, flux, crowd band → audio_z
│   ├── commentary.py       # Whisper transcription + keyword scoring
│   ├── video_features.py   # motion, scene cuts, replays → video_z
│   ├── scorer.py           # fused score + peak finding
│   ├── classifier.py       # event type + reason
│   ├── clipper.py          # clip windows, ffmpeg cutting, thumbnails, reel
│   ├── pipeline.py         # runs everything, with caching
│   └── ui_style.py         # app colors and CSS
├── tests/                  # pytest (synthetic signals, no video needed)
├── notebooks/exploration.ipynb
├── scripts/                # per-phase check scripts
├── samples/                # your videos + truth CSVs (git-ignored)
└── outputs/<video_name>/   # results + cache (git-ignored)
```

## Setup

You need Python 3.12 and ffmpeg.

**Windows**
```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
winget install ffmpeg            # then close and reopen the terminal
```
If numba/librosa or faster-whisper fail with "DLL load failed", install the Microsoft Visual C++ Redistributable:
`winget install Microsoft.VCRedist.2015+.x64`.

**macOS / Linux**
```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
brew install ffmpeg              # macOS
sudo apt install ffmpeg          # Debian/Ubuntu
```
(On macOS/Linux use `.venv/bin/python` wherever this README says `.venv\Scripts\python.exe`.)

The first run of each Whisper model (`tiny`, `base`, `small`) downloads it once. After that, nothing goes to the network.

## Run the CLI

```powershell
# default settings from config.yaml (Whisper base, top 10)
.venv\Scripts\python.exe cli.py --video samples/test_match.mp4

# faster smoke test: smaller model, 5 highlights, longer build-up
.venv\Scripts\python.exe cli.py --video samples/test_match.mp4 --model tiny --top 5 --pre 15

# ignore the cache and recompute everything
.venv\Scripts\python.exe cli.py --video samples/test_match.mp4 --no-cache
```
Options: `--top N`, `--model tiny|base|small`, `--pre S`, `--post S`, `--config FILE`, `--no-cache`.
Intermediate results (audio, transcript, video features) are cached in `outputs/<video_name>/cache/` and reused
while they are newer than the video.

## Run the app

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```
Pick a video from `samples/` or upload one (saved to `samples/uploads/`), then click **Analyze Match**. **Re-score only**
reuses the cached analysis and only recomputes the scores, labels and clips. Use it after changing the weights or clip
settings under *Advanced settings*. The app never writes to `config.yaml`.

## Evaluation

Write down the true event times of a match in a CSV:

```csv
time,event
06:00,goal
11:38,save
1:02:15,card
1500,goal
```

- `time`: `mm:ss`, `h:mm:ss` or plain seconds. Minutes may go past 59 (`95:03`).
- `event` (optional): free text. Rows with `goal` are used for the goal-only line. Without this column, all rows count as goals.
- No comment lines inside the CSV. `samples/truth_example.csv` is this example.

```powershell
.venv\Scripts\python.exe evaluate.py --pred outputs/test_match/highlights.csv --truth samples/truth_test.csv
.venv\Scripts\python.exe evaluate.py --pred outputs/test_match/highlights.csv --truth samples/truth_test.csv --tolerance 10
```

A prediction counts as correct if its peak is within ±tolerance seconds (default 20) of a true event. Each true event and
each prediction can be matched only once, and the closest pairs are matched first. The report shows precision, recall,
F1, the mean time offset, a goal-only line (true goals that were found **and** labelled `Goal`), the matched pairs, the
missed events and the false positives.

Precision is only meaningful if the truth file lists *every* kind of event you want as a highlight. If it lists only
goals, near-misses and penalty incidents count as false positives.

## Expected speed on CPU

Measured on the development laptop (Intel CPU, Intel Iris Xe, no CUDA) with the 30-minute test video (720p, 30 fps):

| Step | Time |
|------|------|
| Full first run (`--no-cache`, Whisper **tiny**, top 5) | **~251 s** (audio 20 s, transcript 138 s, video 89 s, scoring 0.1 s, clips 4 s) |
| Cached run (CLI) | **~4 s** |
| Cached *Analyze Match* in the app / *Re-score only* | ~15 s / ~6.5 s |

- Video features decode at about 21× real time, so about 5 min for a full 90+ min match.
- Whisper **base** (the default) is slower than tiny. Its full-run time was not measured separately.
- Expect a full match to take roughly 3–4× as long as the 30-minute numbers above.

## Design decisions

- **CPU only, free, local.** It uses faster-whisper on CPU, OpenCV with optional iGPU decoding, and ffmpeg through
  subprocess. There are no paid APIs and no cloud calls.
- **Few parameters, on purpose.** Signals are relative (rolling robust z-scores) instead of absolute thresholds, and there
  are only three weights plus a few caps and windows. With so little labelled data, more knobs would just overfit one match.
- **Tuned on one match, checked on others.** Parameters were set on one 30-minute test video. A second match was kept
  aside as a held-out check. This README reports no held-out accuracy numbers. That second match showed a missed goal
  ("<player> gives Liverpool the lead"), and the Goal phrase list was then broadened from general football vocabulary
  (without reading that match's transcript). It is still unseen data, but it is **no longer a clean held-out test**:
  the gap it revealed guided this change. A fresh, untouched match is needed for an honest accuracy number.
- **Explainable.** Every highlight carries a reason built from the signals that triggered it, and the rules are simple
  enough to read in `src/classifier.py`.
- **Cache everything expensive.** Transcription and video decoding run once per video, so changing weights or the clip
  length takes seconds.

## Results on the tuning match (not a held-out score)

This is `evaluate.py` on the 30-minute test video, whose truth file lists its two goals, with default settings:

- Both goals were found and labelled `Goal`, at ranks 4 and 6, each 1 s after the true time (recall 1.0).
- Precision is 0.2 because the truth file lists only goals. The other 8 highlights are penalty incidents, near-misses
  and crowd moments, which are not labelled in the truth file.
- These parameters were tuned on this same match, so this shows the method works on it. It does not show how well
  it generalizes.

## Limitations

- **English commentary only** (`whisper.language: en`), and the keyword list is in English. The Goal confirm/cancel
  phrase lists are generic English football vocabulary; other phrasings or commentary styles can still be missed.
- **Goal phrases can be fooled.** Recap talk that uses a live-sounding phrase ("he scores from the spot every week")
  can still confirm a Goal if the crowd is loud enough. A cancel phrase after a real goal ("was he offside? no", "he
  misses very few of those") can undo it. A VAR "the goal stands" a minute after a goal can label the same goal twice.
- **Stadium noise.** Chants, horns or a loud home crowd can create audio spikes without anything happening on the pitch.
- **Recap commentary.** Talk about an earlier penalty or goal can create false "Card / Foul" labels. On the test
  match, several top peaks are labelled "Card / Foul" because of penalty talk. Keyword gating by crowd noise reduces this but does not remove it.
- **Replay detection is a heuristic** (several scene cuts within 10 s). Fast camera work can look like a replay, and a
  replay can be missed.
- **Videos without an audio track** cannot be analysed. The app and CLI stop with a clear message.
- **Long videos take longer**, roughly in proportion to their length. For a 90-minute match, raise top N to about 15–20 (default 10).
- Some codecs (for example AV1) may not decode in OpenCV. Re-encode to H.264 if the video stage fails.

## Future improvements

1. **Trained crowd-cheer audio classifier.** A small model that tells cheering from chanting and general noise would be
   a cleaner signal than loudness z-scores.
2. **Scoreboard OCR to confirm goals.** Read the score overlay and confirm a goal when the score changes. That would be
   independent of the commentary.
3. **Vertical 9:16 reel export** for social media, with a crop that follows the ball or the action.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest -q
```
The unit tests use synthetic signals and small temporary CSVs. The app smoke test uses saved outputs of
`samples/test_match.mp4` if they exist and is skipped otherwise.
