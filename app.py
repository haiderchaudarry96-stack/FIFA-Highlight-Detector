"""Streamlit app: pick a match video, click Analyze Match, get the highlight reel, a score timeline and cards.

Run: .venv\\Scripts\\python.exe -m streamlit run app.py
All sidebar values go into an in-memory copy of config.yaml; the file itself is never written.
"""
from __future__ import annotations

import copy
import html
import json
import re
import textwrap
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.pipeline import PipelineResult, run_pipeline
from src.ui_style import CSS, DARK_TEXT, EVENT_COLORS, PRIMARY, badge_html, event_color, legend_html
from src.utils import (PROJECT_ROOT, FFmpegNotFoundError, NoAudioTrackError, VideoReadError, fmt_time,
                       get_logger, get_output_dirs, is_cache_valid, load_config)

log = get_logger("app")

SAMPLES_DIR = PROJECT_ROOT / "samples"
UPLOAD_DIR = SAMPLES_DIR / "uploads"
VIDEO_EXTS = {".mp4", ".mkv"}
HIDDEN_SAMPLES = {"test_match2.mp4", "full_match.mp4"}  # held-out test clip / AV1 original
DEFAULT_SAMPLE = "test_match.mp4"
WHISPER_MODELS = ["tiny", "base", "small"]
# pipeline stage -> (share of the progress bar, plain-language status line)
STAGES = {
    "audio": (0.06, "Listening to the crowd (extracting the sound)"),
    "transcript": (0.46, "Transcribing the commentary"),
    "video": (0.34, "Watching the action (motion, scene cuts, replays)"),
    "scoring": (0.02, "Scoring every second of the match"),
    "clips": (0.12, "Cutting the clips and building the highlight reel"),
}
# card filter label -> event type (None = all)
FILTERS = {"All": None, "Goals": "Goal", "Big Chance & Save": "Big Chance / Save",
           "Card & Foul": "Card / Foul", "Exciting": "Exciting Moment"}
SORTS = ["By rank", "By match time"]
# event type -> (singular, plural) for the counts next to the Highlights title
COUNT_NAMES = {"Goal": ("Goal", "Goals"), "Big Chance / Save": ("Big Chance", "Big Chances"),
               "Card / Foul": ("Card / Foul", "Cards / Fouls"), "Exciting Moment": ("Exciting", "Exciting")}


# ---------------------------------------------------------------- helpers
def section(title: str, sub: str = "") -> None:
    extra = f'<span class="hl-section-sub">{html.escape(sub)}</span>' if sub else ""
    st.markdown(f'<div class="hl-section">{html.escape(title)}{extra}</div>', unsafe_allow_html=True)


def friendly_reason(reason: str) -> str:
    """Plain-language version of the classifier's reason, for display only.

    "crowd spike (audio z=6.0) + 'penalty' + 'goal' x2 + replay detected"
    -> "Huge crowd roar · commentator said 'penalty', 'goal' (2×) · replay shown"
    """
    out: list[str] = []
    said: list[str] = []
    for part in (p.strip() for p in str(reason).split(" + ")):
        if m := re.fullmatch(r"(crowd spike|calm crowd) \(audio z=(-?[\d.]+)\)", part):
            z = float(m.group(2))
            if m.group(1) == "calm crowd":
                out.append("Quiet crowd")
            else:
                out.append("Huge crowd roar" if z >= 4 else "Big crowd roar" if z >= 2.5 else "Crowd roar")
        elif part.startswith("confirmed by "):
            calls = [c.strip("'") for c in part[len("confirmed by "):].split("', '")]
            out.append("commentator called it: " + ", ".join(f"“{c}”" for c in calls))
        elif part == "replay detected":
            continue  # added last so it always ends the line
        elif m := re.fullmatch(r"'(.+)'(?: x(\d+))?", part):
            said.append(f"'{m.group(1)}'" + (f" ({m.group(2)}×)" if m.group(2) else ""))
        elif part:
            out.append(part)
    if said:
        called = any(p.startswith("commentator called it") for p in out)
        out.append(("also said " if called else "commentator said ") + ", ".join(said))
    if "replay detected" in reason:
        out.append("replay shown")
    text = " · ".join(out)
    return text[:1].upper() + text[1:] if text else "Exciting moment"


def count_summary(highlights: list[dict]) -> str:
    """e.g. '2 Goals · 5 Cards / Fouls · 1 Big Chance' (event types in legend order, zero counts left out)."""
    parts = []
    for etype in EVENT_COLORS:
        n = sum(h["event_type"] == etype for h in highlights)
        if n:
            one, many = COUNT_NAMES.get(etype, (etype, etype))
            parts.append(f"{n} {one if n == 1 else many}")
    return " · ".join(parts)


def side_title(title: str) -> None:
    st.markdown(f'<div class="hl-side-title">{html.escape(title)}</div>', unsafe_allow_html=True)


def list_samples() -> list[Path]:
    """Videos directly in samples/ (not in samples/uploads), minus the hidden ones."""
    if not SAMPLES_DIR.is_dir():
        return []
    return sorted(p for p in SAMPLES_DIR.iterdir()
                  if p.is_file() and p.suffix.lower() in VIDEO_EXTS and p.name not in HIDDEN_SAMPLES)


def save_upload(uploaded) -> Path:
    """Save an uploaded video to samples/uploads/ (skipped if the same file is already there)."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / Path(uploaded.name).name
    if dest.is_file() and dest.stat().st_size == uploaded.size:
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    uploaded.seek(0)
    with open(part, "wb") as f:
        while chunk := uploaded.read(16 * 1024 * 1024):
            f.write(chunk)
    part.replace(dest)
    return dest


def read_bytes(path: str | Path):
    """Lazy file reader for download buttons (the file is only read when the user clicks)."""
    return lambda: Path(path).read_bytes()


def friendly_error(e: Exception) -> str:
    """Plain-language message for a failed run (the traceback goes to the terminal log only)."""
    if isinstance(e, FFmpegNotFoundError):
        return ("**ffmpeg is not installed** (or not on PATH). The app needs it to read the video. "
                "Install it with `winget install ffmpeg`, close and reopen the terminal, then start the app again.")
    if isinstance(e, NoAudioTrackError):
        return ("**This video has no sound track.** The detector listens to the crowd and the commentary, "
                "so it needs a video with audio. Please pick another file.")
    if isinstance(e, VideoReadError):
        return ("**We couldn't read this video.** The file may be corrupt, not fully downloaded, or in a format "
                "that can't be decoded. An H.264 .mp4 works best.")
    if isinstance(e, ImportError):
        return (f"**A required package is missing** ({getattr(e, 'name', None) or e}). "
                "Run `.venv\\Scripts\\python.exe -m pip install -r requirements.txt` and restart the app.")
    return ("**Something went wrong while analysing the match.** The details are in the terminal window that "
            "runs the app. Try again with *Use cache* switched off, or try another video.")


def missing_cache(video: Path, cfg: dict) -> list[str]:
    """Cached features that Re-score needs but that are missing or older than the video."""
    cache = get_output_dirs(video)["cache"]
    wav = cache / "audio.wav"
    model = cfg["whisper"]["model"]
    checks = {
        "the sound analysis": is_cache_valid(wav, video) and is_cache_valid(cache / "audio_features.csv", video),
        f"the commentary transcript ({model} model)": is_cache_valid(cache / f"transcript_{model}.json", wav),
        "the video analysis": is_cache_valid(cache / "video_features.csv", video),
    }
    return [name for name, ok in checks.items() if not ok]


def load_saved_result(video: Path) -> PipelineResult | None:
    """Results of the last pipeline run on this video, read back from outputs/<stem>/ (None if absent/stale)."""
    root = get_output_dirs(video)["root"]
    json_path, csv_path, signals_path = root / "highlights.json", root / "highlights.csv", root / "signals.csv"
    if not all(is_cache_valid(p, video) for p in (json_path, csv_path, signals_path)):
        return None
    try:
        highlights = json.loads(json_path.read_text(encoding="utf-8"))
        signals = pd.read_csv(signals_path)
    except (OSError, ValueError):
        return None
    reel = root / "highlight_reel.mp4"
    return PipelineResult(highlights=highlights, signals=signals, reel_path=reel if reel.is_file() else None,
                          json_path=json_path, csv_path=csv_path, out_dir=root)


def make_progress_cb(bar):
    """progress_cb(stage, fraction) -> one overall progress bar with a plain-language status line."""
    order = list(STAGES)
    starts, acc = {}, 0.0
    for name in order:
        starts[name] = acc
        acc += STAGES[name][0]
    last = {"key": None}

    def cb(stage: str, fraction: float) -> None:
        share, text = STAGES.get(stage, (0.0, stage))
        key = (stage, int(fraction * 100))
        if key == last["key"]:  # the video stage reports very often; skip repeats
            return
        last["key"] = key
        total = min(1.0, starts.get(stage, 0.0) + share * fraction)
        step = order.index(stage) + 1 if stage in order else 0
        bar.progress(total, text=f"Step {step} of {len(order)} · {text}… {fraction:.0%}")

    return cb


def run_analysis(video: Path, cfg: dict, use_cache: bool, label: str) -> None:
    """Run the pipeline with a progress bar and keep the result in session_state."""
    bar = st.progress(0.0, text=f"{label}: getting ready…")
    try:
        result = run_pipeline(video, cfg, progress_cb=make_progress_cb(bar), use_cache=use_cache)
    except Exception as e:  # shown as a friendly message, never as a traceback
        log.exception("Analysis of %s failed", video)
        bar.empty()
        st.error(friendly_error(e), icon=":material/error:")
        return
    bar.empty()
    st.session_state["result"] = result
    st.session_state["result_video"] = str(video)
    n = len(result.highlights)
    st.toast(f"Done! Found {n} highlight{'s' if n != 1 else ''} in {video.name}.", icon=":material/sports_soccer:")


# ---------------------------------------------------------------- sidebar
def sidebar(base_cfg: dict) -> tuple[Path | None, dict, bool, bool, bool]:
    """All controls. Returns (video, cfg copy with the chosen values, use_cache, analyze, rescore)."""
    cfg = copy.deepcopy(base_cfg)
    sc = cfg["scoring"]
    clips = cfg.setdefault("clips", {})
    video: Path | None = None
    with st.sidebar:
        st.markdown("#### :material/sports_soccer: Match setup")

        side_title("Video")
        source = st.radio("Where is the match?", ["Sample video", "Upload a video"], index=0, key="video_source",
                          horizontal=True, help="Pick a video from the samples/ folder, or upload your own.")
        if source == "Sample video":
            samples = list_samples()
            if samples:
                names = [p.name for p in samples]
                name = st.selectbox("Match video", names, key="sample_video",
                                    index=names.index(DEFAULT_SAMPLE) if DEFAULT_SAMPLE in names else 0,
                                    help="Match videos (.mp4 / .mkv) found in the samples/ folder.")
                video = SAMPLES_DIR / name
            else:
                st.info("No videos in samples/ yet. Put an .mp4 or .mkv there, or upload one.")
        else:
            up = st.file_uploader("Upload a match video", type=["mp4", "mkv"], key="upload",
                                  help="Up to 4 GB. It is saved to samples/uploads/ so later runs can reuse "
                                       "the cached analysis.")
            if up is not None:
                try:
                    video = save_upload(up)
                except OSError as e:
                    log.exception("Saving upload failed")
                    st.error(f"Could not save the uploaded file ({e.strerror or e}). Is the disk full?")

        side_title("Commentary model")
        default_model = cfg["whisper"].get("model", "base")
        model = st.selectbox("Whisper model", WHISPER_MODELS, key="whisper_model",
                             index=WHISPER_MODELS.index(default_model) if default_model in WHISPER_MODELS else 1,
                             help="Turns the commentary into text. tiny = fastest, small = most accurate but "
                                  "slowest. Each model is downloaded once on first use.")

        side_title("Run")
        use_cache = st.checkbox("Use cache", value=True, key="use_cache",
                                help="Reuse the sound, commentary and video analysis from earlier runs of this "
                                     "video. Switch off to analyse everything again from scratch.")
        analyze = st.button("Analyze Match", type="primary", key="analyze", icon=":material/sports_soccer:",
                            width="stretch", help="Find the highlights. A first run on a 30 min match takes a "
                                                  "few minutes; cached runs take seconds.")
        rescore = st.button("Re-score only", key="rescore", icon=":material/tune:", width="stretch",
                            help="Fast: reuses the cached analysis and only recomputes the scores, labels and "
                                 "clips with the new weights and clip settings. Needs one Analyze Match first.")

        with st.expander("Advanced settings", icon=":material/settings:"):
            side_title("Signal weights")
            w = sc["weights"]
            wa = st.slider("Crowd noise (audio)", 0.0, 1.0, float(w.get("audio", 0.5)), 0.05, key="w_audio",
                           help="How much crowd roars and loud moments count. Weights are normalized to sum "
                                "to 1, so only their ratio matters.")
            wk = st.slider("Commentary keywords", 0.0, 1.0, float(w.get("keywords", 0.35)), 0.05,
                           key="w_keywords", help="How much words like 'goal', 'what a save' or '1-0' count. "
                                                  "Normalized together with the other two weights to sum to 1.")
            wv = st.slider("On-screen action (video)", 0.0, 1.0, float(w.get("video", 0.15)), 0.05, key="w_video",
                           help="How much fast motion, scene cuts and replays count. Normalized together with "
                                "the other two weights to sum to 1.")
            total = wa + wk + wv
            if total > 0:
                st.caption(f"Mix used: crowd {wa / total:.0%} · keywords {wk / total:.0%} · video {wv / total:.0%}")
            else:
                st.caption(":red[Set at least one weight above 0.]")

            side_title("Clips")
            top_n = st.slider("Number of highlights", 1, 30, int(sc.get("top_n", 10)), key="top_n",
                              help="How many of the best moments to keep (fewer may be found in a quiet match).")
            c1, c2 = st.columns(2)
            pre = c1.number_input("Before (s)", 0.0, 120.0, float(clips.get("pre_s", 10)), 1.0, key="pre_s",
                                  format="%.0f", help="Seconds of build-up kept before each moment.")
            post = c2.number_input("After (s)", 0.0, 120.0, float(clips.get("post_s", 8)), 1.0, key="post_s",
                                   format="%.0f", help="Seconds kept after each moment (celebrations, reactions).")
            st.caption("Change these, then click **Re-score only** to update the results in seconds.")

    cfg["whisper"]["model"] = model
    sc["weights"] = {"audio": wa, "keywords": wk, "video": wv}
    sc["top_n"] = int(top_n)
    clips["pre_s"], clips["post_s"] = float(pre), float(post)
    return video, cfg, use_cache, analyze, rescore


# ---------------------------------------------------------------- main page
def header() -> None:
    st.markdown(
        '<div class="hl-header"><div class="hl-logo">⚽</div><div>'
        '<div class="hl-title">Match Highlight Detector</div>'
        '<p class="hl-tagline">Crowd roar + commentary + on-screen action → the best moments of the match, '
        'cut and ready to watch.</p></div></div>',
        unsafe_allow_html=True)


def empty_state(video: Path | None) -> None:
    what = (f"<b>{html.escape(video.name)}</b> hasn't been analysed yet." if video
            else "Pick a sample match or upload a video in the sidebar.")
    st.markdown(
        f'<div class="hl-empty"><div style="font-size:46px">⚽</div><b>Ready for kick-off</b>'
        f'<p>{what}<br>Then click <b>Analyze Match</b> to get the highlight reel, a score timeline '
        f'and a card for every big moment.</p></div>',
        unsafe_allow_html=True)


def tick_step(n_seconds: int) -> int:
    """Seconds between x-axis ticks so there are at most ~12 of them."""
    for step in (60, 120, 300, 600, 900, 1200, 1800):
        if n_seconds / step <= 12:
            return step
    return 3600


def timeline_figure(signals: pd.DataFrame, highlights: list[dict]) -> go.Figure:
    """Smoothed score over the match + one marker per highlight, colored by event type.

    Plotly's own legend is hidden: the badge row above the chart is the only legend.
    """
    sec = signals["second"].to_numpy()
    score = signals["score"].to_numpy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sec, y=score, mode="lines", name="Excitement score",
        line=dict(color=PRIMARY, width=2), fill="tozeroy", fillcolor="rgba(22, 163, 74, 0.08)",
        customdata=[fmt_time(s) for s in sec],
        hovertemplate="%{customdata}<br>score %{y:.2f}<extra></extra>"))
    types = list(EVENT_COLORS) + sorted({h["event_type"] for h in highlights} - set(EVENT_COLORS))
    for etype in types:
        pts = [h for h in highlights if h["event_type"] == etype]
        if not pts:
            continue
        fig.add_trace(go.Scatter(
            x=[h["peak_s"] for h in pts], y=[h["score"] for h in pts], mode="markers+text", name=etype,
            text=[f"#{h['rank']}" for h in pts], textposition="top center",
            textfont=dict(color=DARK_TEXT, size=12),
            marker=dict(color=event_color(etype), size=15, line=dict(color="white", width=2)),
            customdata=[[h["timestamp"], etype, h["score"],
                         "<br>".join(textwrap.wrap(html.escape(friendly_reason(h["reason"])), 60)),
                         "<br>".join(textwrap.wrap(html.escape(h["reason"]), 70))] for h in pts],
            hovertemplate="<b>%{text} · %{customdata[0]} · %{customdata[1]}</b><br>score %{customdata[2]:.2f}"
                          "<br>%{customdata[3]}<br><span style='font-size:11px;color:#7B8794'>"
                          "%{customdata[4]}</span><extra></extra>"))
    n = int(len(sec))
    step = tick_step(n)
    ticks = list(range(0, n + 1, step))
    top = max([float(score.max()) if n else 0.0] + [h["score"] for h in highlights] + [0.1])
    fig.update_layout(
        template="plotly_white", paper_bgcolor="white", plot_bgcolor="white", height=370,
        margin=dict(l=10, r=10, t=16, b=10), hovermode="closest", showlegend=False,
        font=dict(color=DARK_TEXT, size=13),
        hoverlabel=dict(bgcolor="white", bordercolor="#D7EBD3", font=dict(color=DARK_TEXT)))
    fig.update_xaxes(tickvals=ticks, ticktext=[fmt_time(t) for t in ticks], range=[0, max(n, 1)],
                     title="Match time (mm:ss)", gridcolor="#EEF3EE", zeroline=False)
    fig.update_yaxes(range=[0, top * 1.2], title="Excitement score", gridcolor="#EEF3EE", zeroline=False)
    return fig


def render_card(h: dict) -> None:
    """One highlight, always in this order: rank · mm:ss · badge · score, reason, snippet, clip, download."""
    rank, etype = h["rank"], h["event_type"]
    with st.container(key=f"card_{rank}"):
        st.markdown(
            f'<div class="hl-card-head"><span class="hl-rank">#{rank}</span>'
            f'<span class="hl-time">{html.escape(h["timestamp"])}</span>{badge_html(etype)}'
            f'<span class="hl-score">score {float(h["score"]):.2f}</span></div>'
            # the technical reason stays available as the hover tooltip
            f'<div class="hl-reason" title="{html.escape(h["reason"], quote=True)}">'
            f'{html.escape(friendly_reason(h["reason"]))}</div>',
            unsafe_allow_html=True)
        snippet = str(h.get("transcript_snippet") or "").strip()
        snippet_html = f"“{html.escape(snippet)}”" if snippet else "No commentary captured here."
        st.markdown(f'<div class="hl-snippet" title="{html.escape(snippet, quote=True)}">{snippet_html}</div>',
                    unsafe_allow_html=True)
        clip = h.get("clip_path")
        if clip and Path(clip).is_file():
            st.video(clip)
            st.download_button("Download clip", data=read_bytes(clip), file_name=Path(clip).name,
                               mime="video/mp4", key=f"dl_clip_{rank}", on_click="ignore",
                               icon=":material/download:", width="stretch")
        else:
            st.caption("Clip file not found. Run the analysis again to recreate it.")


def render_results(res: PipelineResult, video: Path, saved: bool) -> None:
    hl = res.highlights
    if saved:
        st.info(f"Showing the saved results of the last analysis of **{video.name}**. Click **Analyze Match** "
                "to run it again with the settings in the sidebar.", icon=":material/history:")

    # hero: reel + downloads
    has_reel = bool(res.reel_path and Path(res.reel_path).is_file())
    with st.container(key="hero"):
        left, right = st.columns([2.3, 1], gap="large")
        with left:
            if has_reel:
                st.video(str(res.reel_path))
            else:
                st.warning("No highlight reel yet: no exciting moments were found in this video.")
        with right:
            st.markdown(f"#### :material/movie: Highlight reel\n{video.name}")
            st.caption("Every highlight in match order, stitched into one video.")
            if has_reel:
                st.download_button("Reel (.mp4)", data=read_bytes(res.reel_path),
                                   file_name=f"{video.stem}_highlights.mp4", mime="video/mp4", key="dl_reel",
                                   on_click="ignore", type="primary", icon=":material/download:", width="stretch",
                                   help="Download the stitched highlight reel.")
            c_csv, c_json = st.columns(2)
            c_csv.download_button("CSV", data=read_bytes(res.csv_path), file_name="highlights.csv",
                                  mime="text/csv", key="dl_csv", on_click="ignore", icon=":material/table:",
                                  width="stretch", help="highlights.csv: one row per highlight.")
            c_json.download_button("JSON", data=read_bytes(res.json_path), file_name="highlights.json",
                                   mime="application/json", key="dl_json", on_click="ignore",
                                   icon=":material/data_object:", width="stretch",
                                   help="highlights.json: the same records as JSON.")
            st.caption(f"Saved in `{res.out_dir}`")

    # summary metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Match length", fmt_time(len(res.signals)), border=True, icon=":material/timer:",
              help="Length of the analysed video (mm:ss).")
    c2.metric("Highlights", len(hl), border=True, icon=":material/star:")
    c3.metric("Goals found", sum(h["event_type"] == "Goal" for h in hl), border=True,
              icon=":material/sports_soccer:", help="Moments labelled 'Goal' (a live call like 'scores!' or "
                                                    "a new scoreline plus a crowd roar).")

    section("Excitement timeline")
    st.markdown(legend_html(), unsafe_allow_html=True)
    st.plotly_chart(timeline_figure(res.signals, hl), theme=None, key="timeline",
                    config={"displayModeBar": False, "displaylogo": False})
    st.caption("Green line = excitement score each second (smoothed). Dots = highlights, numbered by rank. "
               "Hover a dot to see why it was picked.")

    section("Highlights", count_summary(hl))
    f_col, s_col = st.columns([3, 1.2], vertical_alignment="bottom")
    with f_col:
        choice = st.segmented_control("Show", list(FILTERS), default="All", key="card_filter",
                                      label_visibility="collapsed", help="Show only one type of moment.")
    with s_col:
        order = st.segmented_control("Sort", SORTS, default=SORTS[0], key="card_sort",
                                     label_visibility="collapsed", help="Order the cards by rank or by match time.")
    wanted = FILTERS.get(choice or "All")
    shown = [h for h in hl if wanted is None or h["event_type"] == wanted]
    if order == "By match time":
        shown = sorted(shown, key=lambda h: h["peak_s"])
    if not hl:
        st.info("No highlights found. Try more highlights or different weights under Advanced settings, "
                "then Re-score.")
    elif not shown:
        st.info(f"No {choice} moments among these highlights.")
    for i in range(0, len(shown), 2):
        for col, h in zip(st.columns(2, gap="large"), shown[i:i + 2]):
            with col:
                render_card(h)


def main() -> None:
    st.set_page_config(page_title="Match Highlight Detector", page_icon="⚽", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    header()
    try:
        base_cfg = load_config()
    except Exception:
        log.exception("Loading config.yaml failed")
        st.error("**config.yaml could not be loaded.** Make sure it is in the project folder and is valid YAML.")
        st.stop()
    video, cfg, use_cache, analyze, rescore = sidebar(base_cfg)

    if analyze or rescore:
        if video is None:
            st.error("**No video selected.** Pick a sample match or upload a video in the sidebar, then click "
                     "Analyze Match.", icon=":material/videocam_off:")
        elif sum(cfg["scoring"]["weights"].values()) <= 0:
            st.error("**All signal weights are 0.** Move at least one weight slider above 0.")
        elif not video.is_file():
            st.error(f"**{video.name} was not found.** It may have been moved or deleted.")
        elif rescore and (missing := missing_cache(video, cfg)):
            st.warning(f"**Re-score needs a finished analysis first.** Missing: {', '.join(missing)}. Click "
                       "**Analyze Match** once; after that, re-scoring takes only seconds.",
                       icon=":material/hourglass_empty:")
        else:
            run_analysis(video, cfg, use_cache=use_cache if analyze else True,
                         label="Analyzing the match" if analyze else "Re-scoring")

    result, saved = None, False
    if video is not None:
        if st.session_state.get("result_video") == str(video):
            result = st.session_state.get("result")
        else:
            result = load_saved_result(video)
            saved = result is not None
    if result is None:
        empty_state(video)
    else:
        render_results(result, video, saved)


main()
