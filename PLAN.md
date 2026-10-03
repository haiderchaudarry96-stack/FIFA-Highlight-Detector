# PLAN.md — fifa-highlight-detector

Single source of truth for the design. Every phase reads this file first.
Do not change the interfaces below without updating this file in the same phase.

## 1. Goal
Given a local football match video (mp4/mkv, 10–90 min), automatically find the most
exciting moments (goals, big chances/saves, cards) and produce:
- individual highlight clips (mp4)
- one stitched highlight reel (mp4, chronological)
- `highlights.json` and `highlights.csv` (timestamp, score, event type, reason)

Everything runs locally and free. No paid APIs.

## 2. Target machine
- Windows laptop, Intel CPU with Intel Iris (Xe) integrated GPU. **No NVIDIA / CUDA.**
- So all heavy work must be CPU-friendly. Dev test video: `samples/test_match.mp4` (30 min, H.264, 2 goals);
  final test: `samples/full_match_h264.mp4` (~1h40m). See CLAUDE.md "Testing".
- `ffmpeg` must be on PATH (Windows: `winget install ffmpeg`).

## 3. Tech stack (fixed)
Python 3.10+, OpenCV (`opencv-python`), librosa, numpy, pandas, scipy, PyYAML,
Streamlit, plotly, pytest.
Transcription: `faster-whisper` (default, int8 on CPU) with `openai-whisper` as fallback.
ffmpeg via `subprocess` (no moviepy).

## 4. Folder structure
```
fifa-highlight-detector/
├── CLAUDE.md
├── PLAN.md
├── config.yaml
├── requirements.txt
├── README.md
├── app.py                 # Streamlit UI            (Phase 6)
├── cli.py                 # command line entry      (Phase 5)
├── evaluate.py            # precision/recall        (Phase 7)
├── src/
│   ├── __init__.py
│   ├── utils.py           # config, ffmpeg check, paths, logging, time fmt (Phase 1)
│   ├── audio_extractor.py # (Phase 1)
│   ├── audio_features.py  # (Phase 1)
│   ├── commentary.py      # (Phase 2)
│   ├── video_features.py  # (Phase 3)
│   ├── scorer.py          # (Phase 4)
│   ├── classifier.py      # (Phase 4)
│   ├── clipper.py         # (Phase 5)
│   └── pipeline.py        # (Phase 5)
├── tests/
│   ├── test_scorer.py     # (Phase 4)
│   └── test_clipper.py    # (Phase 5, window merging only)
├── notebooks/
│   └── exploration.ipynb  # (Phase 7)
├── samples/               # user puts test videos here (git-ignored)
└── outputs/<video_stem>/  # all results + cache (git-ignored)
```

## 5. Output folder layout
```
outputs/<video_stem>/
├── cache/
│   ├── audio.wav              # mono 16 kHz
│   ├── audio_features.csv
│   ├── transcript_<model>.json
│   ├── keywords.csv
│   └── video_features.csv
├── clips/01_03-12_Goal.mp4 ...  # <rank>_<mm-ss>_<EventType>.mp4
├── thumbs/01.jpg ...
├── signals.csv                # per-second fused signals + score
├── highlight_reel.mp4
├── highlights.json
└── highlights.csv
```
Cache rule: if a cache file exists and is newer than the source video, load it instead of recomputing.
A `--no-cache` / UI checkbox forces recompute.

## 6. config.yaml (shape)
```yaml
whisper:
  backend: auto          # auto | faster | openai   (auto = faster-whisper if installed)
  model: base            # tiny | base | small
  language: en
audio:
  sample_rate: 16000
  baseline_window_s: 60  # rolling median window for normalization
  crowd_band_hz: [300, 3000]
video:
  sample_fps: 2
  resize_width: 320
  cut_threshold: 0.6     # histogram correlation below this = scene cut
  replay_window_s: 10
  replay_min_cuts: 3
  baseline_window_s: 60  # motion_z rolling window
keywords:                # phrase: weight  (lowercase; full live-call list in config.yaml)
  goal: 1.0
  scores: 0.9
  penalty: 0.8
  red card: 0.8
  yellow card: 0.5
  save: 0.7
  what a save: 0.9
  chance: 0.4
  unbelievable: 0.6
  incredible: 0.6
  off the post: 0.6
  crossbar: 0.6
keyword_options:
  elongation_boost: 0.5  # "goooal"
  exclamation_boost: 0.5 # "Costa scores!" (sentence ends with "!")
  repeat_boost: 0.5      # "goal goal goal" within 3 s
  spread_s: 3            # spread a hit over ±N seconds (commentary lag)
scoring:
  weights: {audio: 0.5, keywords: 0.35, video: 0.15}
  smooth_window_s: 5
  min_peak_distance_s: 30
  top_n: 10
  min_prominence: 0.02   # find_peaks prominence on the 0..1 score
  audio_cap: 4.0         # audio_n = clip(audio_z, 0, cap) / cap
  video_cap: 6.0
  replay_lookahead_s: 30 # replay within [t, t+N] -> replay_ahead = 1 at t
  video_mix: {replay: 0.7, video_z: 0.3}
  gate_window_s: 10      # keyword gated by max audio_n within ±N s
  gate_floor: 0.3        # keyword share kept with no crowd spike
clips:
  pre_s: 10
  post_s: 8
classifier:
  keyword_window_s: [-5, 20]   # look for keywords from peak-5s to peak+20s
  replay_window_s: [0, 60]     # look for a replay from peak to peak+60s
  goal_audio_z_min: 1.0        # low sanity floor: max audio z in the keyword window for "Goal"
  confirm_keywords: ["scores!", "scored!", "N-N", takes the lead, ...]  # "!" = must be exclaimed, N-N = scoreline
  # + chance_keywords, card_keywords (see config.yaml)
```

## 7. Module interfaces (contract between phases)
All per-second tables are pandas DataFrames indexed by integer column `second` (0..N-1),
same length N = floor(video duration in seconds). Missing seconds are filled with 0.

**utils.py**
- `load_config(path="config.yaml") -> dict`
- `check_ffmpeg() -> None` — raises `FFmpegNotFoundError` with install hints for Windows/Mac/Linux.
- `get_video_duration(video_path) -> float` (ffprobe)
- `get_output_dirs(video_path, root="outputs") -> dict[str, Path]` keys: root, cache, clips, thumbs
- `is_cache_valid(cache_file, source_file) -> bool`
- `fmt_time(seconds) -> "mm:ss"`
- `get_logger(name)`
- `robust_z(x: Series, window_s) -> Series` — shared rolling robust z-score (audio + video).
- Custom exceptions: `FFmpegNotFoundError`, `NoAudioTrackError`, `VideoReadError`.

**audio_extractor.py**
- `extract_audio(video_path, out_wav, sample_rate=16000) -> Path` — raises `NoAudioTrackError`.

**audio_features.py**
- `compute_audio_features(wav_path, cfg) -> DataFrame`
  columns: `second, rms, flux, crowd, rms_z, flux_z, crowd_z, audio_z`
  Normalization: robust z = (x − rolling_median) / (1.4826 · rolling_MAD + eps), window = baseline_window_s.
  `audio_z` = mean of the three z-scores, clipped to [-3, 6].

**commentary.py**
- `transcribe(wav_path, model_size, backend, cache_path, language="en") -> list[dict]`
  each word: `{"word": str, "start": float, "end": float, "prob": float}`
- `score_keywords(words, keywords_cfg, options_cfg, n_seconds) -> DataFrame`
  columns: `second, keyword_score, keyword_hits` (hits = "goal(1.0); save(0.7)"), score scaled to 0..1.
- `get_snippet(words, start_s, end_s) -> str`

**video_features.py**
- `compute_video_features(video_path, cfg, progress_cb=None) -> DataFrame`
  columns: `second, motion, cuts, replay, motion_z, video_z`
  Streams frames with `cap.grab()` (no full load), samples at sample_fps, grayscale + resized.

**scorer.py**
- `combine_signals(audio_df, kw_df, video_df, weights, smooth_window_s, scoring_cfg=None) -> DataFrame`
  = outer-joined per-second table (all input columns, missing → 0 / "") + `audio_n, keyword_n, replay_ahead,
  video_n, audio_peak_local, keyword_gated, score_raw, score` (all 0..1). `scoring_cfg` = config `scoring` section.
- `find_highlights(score, top_n, min_distance_s, min_prominence=0.02) -> list[int]` (peak seconds, sorted by
  score desc, ties → earlier) uses `scipy.signal.find_peaks(distance=min_distance_s, prominence=...)`;
  first/last second can be peaks; flat/empty → [].
- `signals.csv` (Phase 5) = the combine_signals DataFrame.

**classifier.py**
- `classify(peak_s, signals_df, cfg) -> tuple[str, str]` → (event_type, reason); `cfg` = the full config
  (uses `classifier`, `keywords`, `keyword_options`), `signals_df` = combine_signals output.
  event types: `Goal`, `Big Chance / Save`, `Card / Foul`, `Exciting Moment`.

**clipper.py**
- `build_windows(peaks, pre_s, post_s, duration) -> list[tuple[float, float, list[int]]]`
  clamps to [0, duration], merges overlapping windows, keeps which peaks each window holds.
- `cut_clip(video_path, start, end, out_path)` — try stream copy; if output is broken/empty, re-encode (libx264, aac).
- `save_thumbnail(video_path, t, out_jpg)`
- `build_reel(clip_paths, out_path)` — ffmpeg concat demuxer, chronological order.

**pipeline.py**
- `run_pipeline(video_path, cfg, progress_cb=None, use_cache=True) -> PipelineResult`
  `progress_cb(stage: str, fraction: float)`; stages: audio, transcript, video, scoring, clips.
- `PipelineResult` dataclass: `highlights: list[dict], signals: DataFrame, reel_path, json_path, csv_path, out_dir`.

**Highlight record (json/csv row)**
`rank, peak_s, timestamp, start_s, end_s, score, event_type, reason, transcript_snippet, clip_path, thumbnail_path`

## 8. Phase status
| Phase | Scope | Status |
|------|-------|--------|
| 1 | Setup + utils + audio extraction + audio features | ✅ |
| 2 | Commentary transcription + keyword scoring | ✅ |
| 3 | Video features (motion, cuts, replays) | ✅ |
| 4 | Scorer + classifier + unit tests | ✅ |
| 5 | Clipper + pipeline + CLI (end-to-end) | ⬜ |
| 6 | Streamlit app | ⬜ |
| 7 | evaluate.py + notebook + README | ⬜ |

Each phase, when done, changes its row to ✅ and adds 1–3 lines under "Notes" below
(decisions, gotchas, anything the next phase must know).

## 9. Notes
**Phase 1**
- Windows needs the Microsoft Visual C++ 2015-2022 x64 Redistributable (`winget install Microsoft.VCRedist.2015+.x64`);
  without it numba (librosa) and ctranslate2 (faster-whisper) fail with "DLL load failed". Installed on the dev PC.
- `audio_features` streams the WAV via `librosa.stream` in 60 s blocks, STFT n_fft=1024, hop=400 (40 frames/s, `center=False`)
  so frames align to whole seconds. Per-second `rms`/`flux`/`crowd` are raw; z-scores use a centered rolling median/MAD.
- No cache logic inside `compute_audio_features` / `extract_audio` (per the interfaces); callers use `is_cache_valid`
  (see `scripts/check_audio.py`; Phase 5 pipeline should do the same). `utils.get_output_dirs` also creates the dirs.
- `extract_audio` raises `NoAudioTrackError` via ffprobe, writes through a `.part.wav` temp file so a failed run never leaves a bad cache.
- On `samples/test_match.mp4` the top audio_z seconds cluster at ~11:38 and ~21:40 (likely the two goals); 30 min processes in ~8 s.

**Phase 2**
- `transcribe` has an extra optional `language="en"` arg (pass `cfg["whisper"]["language"]`). It owns its cache: reused if newer than
  the wav and same model; `cache_path=None` disables caching. The wav goes to faster-whisper as a numpy array (soundfile) because
  faster-whisper 1.2.1's PyAV decoder breaks on PyAV 19 (`metadata_errors` kwarg). tiny: 30 min audio in ~60 s on this CPU.
- Keyword hit value = weight × (1+elongation_boost if stretched) × (1+repeat_boost if same phrase again within 3 s); phrases match
  longest-first ("what a save" is not also "save"); symmetric triangular kernel over ±spread_s; series divided by its max.
  `keyword_hits` labels sit on the hit's own second only (not the spread), e.g. `goal(1.50); penalty(1.20)`.
- On test_match (tiny) keyword peaks are near but *after* the goals (06:00 → 06:32, 25:34 → 25:47; 12:36 is the penalty recap),
  and many hits are talk about goals rather than live calls. Phase 4: keywords up to ~30 s after a peak may belong to it.
- Follow-up: "Costa scores!" (25:34) was matched but ranked 13th: one recap event's spread plateau filled 6 of the top-10 seconds,
  and every calm "goal" mention (1.0) beat the live call (0.9). Fix: `exclamation_boost` (hit × 1.5 when its sentence ends in "!",
  Whisper adds "!" to shouted calls), digits no longer stretch ("1" ≠ "11"), check script lists distinct moments ≥ 2·spread_s+1 apart.
  Scorelines like "1-0" match Whisper's split tokens ("1" "-0"). config.yaml: live-call phrases added; yellow card 0.5→0.8,
  off the post 0.6→0.8, red card 0.8→0.9; whisper.model = base.
- base on test_match: 05:59 "It's 1-0" and 25:34 "Costa scores!" are now in the keyword top 10 (#6, #3), but recap talk still
  ranks high (24:46, 12:36). Keywords alone can't separate live calls from recaps — Phase 4 should rely on audio_z for that.

**Phase 3**
- `robust_z` moved from `audio_features.py` to `utils.py` (audio output unchanged). New config key `video.baseline_window_s`
  (motion_z window, falls back to `audio.baseline_window_s`). N = floor(ffprobe duration), same as the other tables.
- Per sampled frame (2 fps): resize to 320 px wide (INTER_LINEAR, 15× faster than INTER_AREA), gray for motion, H-S histogram
  (30×32 bins) for cuts. A cut pair counts in `cuts` but is left out of `motion` (shot change ≠ motion); seconds with only cut
  pairs get interpolated motion. `replay` = centered rolling sum of cuts over replay_window_s ≥ replay_min_cuts (flag on the
  center second). `video_z = 0.6·motion_z + 0.4·3·clip(window_cuts/replay_min_cuts, 0, 1)`, clipped [-3, 6].
- Opens with `CAP_PROP_HW_ACCELERATION=ANY` (Iris iGPU decode, OpenCV falls back to software). Decoding is the bottleneck:
  test_match 30 min in ~85 s (≈21× realtime; ≈12× without HW decode + INTER_AREA). Full match ≈ 5 min. RSS flat (~195 MB).
- On test_match the replay ranges cluster right after both goals (05:57–06:51, 25:42–26:20) — useful as a post-goal signal.
  But motion_z is heavy-tailed (p99 ≈ 10, max ≈ 20; probably fast pans/zooms, not checked visually), so 28 s hit the 6.0 clip and the top-10 by video_z
  are all tied at 6.0 (11:57, 12:05, 13:31, 14:46 — not the goals). Phase 4: keep video weight low; it can't rank alone.

**Phase 4**
- Scoring design (all relative, all knobs in `scoring`): each signal is scaled to 0..1 first. `audio_n = clip(audio_z,0,audio_cap)/audio_cap`
  (cap 4 so the saturated near-miss roars don't dwarf everything); `video_n = 0.7·replay_ahead + 0.3·clip(video_z,0,6)/6` where
  `replay_ahead[t] = any replay in [t, t+replay_lookahead_s]` (replays follow the event, so the credit lands on the event, not the replay);
  `keyword_gated = keyword_score·(gate_floor + (1−gate_floor)·max audio_n within ±gate_window_s)` so recap talk over a calm crowd keeps
  only 30 %. Weights are normalized to sum 1, then a centered moving average. New config keys: `scoring.min_prominence, audio_cap,
  video_cap, replay_lookahead_s, video_mix, gate_window_s, gate_floor`; `classifier.replay_window_s, goal_audio_z_min,
  confirm_keywords, chance_keywords, card_keywords` (replaces `goal_audio_z`); `classifier.keyword_window_s` default → [-5, 20].
- Classifier (revised): "Goal" needs a `confirm_keywords` hit in the keyword window (live calls: "scores!"/"scored!" only when
  exclamation-boosted, any scoreline "N-N", "takes the lead", "equaliser") AND max audio_z ≥ goal_audio_z_min (1.0, a low sanity floor,
  not the discriminator). Plain "goal"/"penalty" never make a Goal (recaps and chance talk use them); they still help ranking via the
  keyword score. A replay only appears in the reason (replays also follow near-misses and penalty incidents).
  Hit labels only hold the final value, so with equal boosts (0.5) an exclamation can't be told from elongation/repeat — any boost that
  *could* include the exclamation counts. confirm entries only fire for phrases that are also in `keywords` ("it's a goal" is listed in
  confirm_keywords but not yet in `keywords`, so it is inert until Phase 2's keyword list + keywords.csv are updated).
- test_match (`scripts/check_scoring.py`): both goals in the top 10 at +1 s (06:01 #4, 25:35 #6), and they are the only two "Goal"
  labels: 06:01 confirmed by "it's 1-0" (z=2.5), 25:35 by "scores!" (z=1.9). The 11:38 near-miss and 12:37 penalty recap are now
  "Card / Foul" (penalty talk). Ranking unchanged: near-misses 11:38/21:45 still rank #1/#7 and the 12:37 recap #3 — the scorer, not
  the classifier, would have to fix that.
