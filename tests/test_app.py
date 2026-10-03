"""Headless smoke tests for the Streamlit app (streamlit.testing AppTest).

The results test reads the saved outputs of an earlier run on samples/test_match.mp4 (no pipeline run),
so it is fast. It is skipped when that video or its outputs are missing.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")
VIDEO = ROOT / "samples" / "test_match.mp4"
OUT = ROOT / "outputs" / "test_match"


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_app_loads_with_sidebar_widgets():
    at = _app()
    assert at.radio(key="video_source").options == ["Sample video", "Upload a video"]
    assert at.radio(key="video_source").value == "Sample video"  # default on first load
    assert at.selectbox(key="whisper_model").options == ["tiny", "base", "small"]
    for key in ("w_audio", "w_keywords", "w_video", "top_n"):
        assert at.slider(key=key) is not None
    assert at.number_input(key="pre_s").value >= 0
    assert at.number_input(key="post_s").value >= 0
    assert at.checkbox(key="use_cache").value is True
    assert at.button(key="analyze").label == "Analyze Match"
    assert at.button(key="rescore").label == "Re-score only"
    if VIDEO.is_file():
        names = at.selectbox(key="sample_video").options
        assert "test_match2.mp4" not in names and "full_match.mp4" not in names


def test_nothing_selected_shows_friendly_error():
    at = _app()
    at.radio(key="video_source").set_value("Upload a video").run()
    at.button(key="analyze").click().run()
    assert not at.exception
    assert any("No video selected" in e.value for e in at.error)


@pytest.mark.skipif(not (VIDEO.is_file() and (OUT / "highlights.json").is_file() and (OUT / "signals.csv").is_file()),
                    reason="needs samples/test_match.mp4 and a finished run in outputs/test_match/")
def test_saved_results_render():
    at = _app()
    assert at.selectbox(key="sample_video").value == "test_match.mp4"
    labels = [m.label for m in at.metric]
    assert labels == ["Match length", "Highlights", "Goals found"]
    n_highlights = int(at.metric[1].value)
    assert n_highlights > 0
    # one download button per clip + reel/csv/json
    assert len(at.get("download_button")) >= n_highlights + 2
    assert at.get("plotly_chart")
    assert not at.error

    # filter chips + sort control above the cards
    filt, sort = at.segmented_control(key="card_filter"), at.segmented_control(key="card_sort")
    assert filt.options == ["All", "Goals", "Big Chance & Save", "Card & Foul", "Exciting"]
    assert filt.value == "All" and sort.value == "By rank"
    n_goals = int(at.metric[2].value)
    filt.set_value("Goals").run()
    assert not at.exception
    assert len(at.get("download_button")) == 3 + n_goals  # reel/csv/json + one per Goal card
    at.segmented_control(key="card_sort").set_value("By match time").run()
    assert not at.exception


def test_friendly_reason():
    import ast
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "friendly_reason")
    ns: dict = {}
    exec(compile(ast.Module([ast.parse("import re").body[0], fn], []), "app.py", "exec"), ns)
    friendly = ns["friendly_reason"]
    assert (friendly("crowd spike (audio z=6.0) + 'penalty' + replay detected")
            == "Huge crowd roar · commentator said 'penalty' · replay shown")
    assert (friendly("crowd spike (audio z=2.5) + confirmed by 'it's 1-0' + 'goal' x2")
            == "Big crowd roar · commentator called it: “it's 1-0” · also said 'goal' (2×)")
    assert friendly("calm crowd (audio z=0.3)") == "Quiet crowd"
