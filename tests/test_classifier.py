"""Goal confirm / cancel phrases on synthetic commentary (config.yaml phrase lists, no real video)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.classifier import classify  # noqa: E402
from src.utils import load_config  # noqa: E402

CFG = load_config(ROOT / "config.yaml")
PEAK = 100


def signals(audio_z=2.5, n=300, hits=()):
    """Per-second table with a crowd spike of height audio_z around PEAK; hits = [(second, label)]."""
    df = pd.DataFrame({"second": np.arange(n), "audio_z": 0.0, "keyword_hits": "", "replay": 0})
    df.loc[PEAK - 2:PEAK + 2, "audio_z"] = audio_z
    for s, label in hits:
        df.loc[s, "keyword_hits"] = label
    return df


def words(*lines, start=PEAK):
    """Whisper-like words: each line starts 4 s after the previous one, 0.3 s per word."""
    out = []
    for k, line in enumerate(lines):
        for j, w in enumerate(line.split()):
            t = start + 4 * k + 0.3 * j
            out.append({"word": w, "start": t, "end": t + 0.3, "prob": 0.9})
    return out


def label_of(*lines, audio_z=2.5):
    return classify(PEAK, signals(audio_z), CFG, words(*lines))


GOALS = {
    # the score changes
    "gives team the lead": ["Salah gives Liverpool the lead."],
    "gives them the lead": ["And that gives them the lead."],
    "puts team ahead": ["That puts Real Madrid ahead."],
    "puts them in front": ["Header, and it puts them in front."],
    "makes it 2-1": ["That makes it 2-1."],
    "it's now 2-1": ["It's now 2-1 in the second half."],
    "word scoreline one-nil": ["It's one-nil to the visitors."],
    "word scoreline two-one": ["It's two-one now."],
    "word scoreline one-one": ["And it is one-one."],
    "exclaimed bare scoreline": ["Three-one!"],
    "takes the lead": ["Arsenal takes the lead."],
    "doubles the lead": ["Kane doubles the lead."],
    "doubles their lead": ["City double... City doubles their lead."],
    "extends the lead": ["He extends the lead for Bayern."],
    "pulls one back": ["United pulls one back."],
    "pulls a goal back": ["Everton pulls a goal back late on."],
    "gets one back": ["Leeds gets one back."],
    "halves the deficit": ["That halves the deficit."],
    # equaliser
    "equaliser exclaimed": ["The equaliser!"],
    "equalizer exclaimed": ["There's the equalizer!"],
    "levels the score": ["Mbappe levels the score."],
    "levels it": ["He levels it with a header."],
    "all square": ["And we are all square."],
    "back on level terms": ["Chelsea are back on level terms."],
    "make it level": ["They make it level!"],
    "gets the equaliser": ["Rice gets the equaliser."],
    # live calls
    "scores exclaimed": ["Costa scores!"],
    "scored exclaimed": ["He's scored!"],
    "he scores": ["Through on goal and he scores."],
    "finds the net": ["Haaland finds the net."],
    "into the back of the net": ["Smashed into the back of the net."],
    "ripples the net": ["That ripples the net."],
    "bulges the net": ["The shot bulges the net."],
    "puts it away": ["Cut back and he puts it away."],
    "slots it home": ["Saka slots it home."],
    "fires home": ["Son fires home from close range."],
    "heads home": ["Van Dijk heads home from the corner."],
    "tucks it away": ["Calmly tucks it away."],
    "buries it": ["Left foot and he buries it."],
    "GOAL shouted": ["GOAL!"],
    # penalties, own goals
    "own goal": ["Oh, it's an own goal."],
    "penalty converted": ["Penalty converted, cool as you like."],
    "scores from the spot": ["Palmer scores from the spot."],
    "converts the penalty": ["Kane converts the penalty."],
}


@pytest.mark.parametrize("lines", GOALS.values(), ids=GOALS.keys())
def test_confirm_phrase_families_are_goal(lines):
    label, reason = label_of(*lines)
    assert label == "Goal", reason
    assert "confirmed by '" in reason


NOT_GOALS = {
    "bare goal recap": ["He scored an FA Cup goal for West Ham last season."],
    "goal of the game": ["That was the goal of the game, wasn't it."],
    "lost the lead": ["They lost the lead before half time."],
    "lead didn't last": ["Their lead didn't last long."],
    "level of play": ["The level of play has dropped."],
    "level alone": ["Level pegging in possession."],
    "unexclaimed scored": ["Salah scored twice against them last year."],
    "looking for an equaliser": ["They are looking for an equaliser."],
    "one-two pass": ["Lovely one-two on the edge."],
    "old scoreline recap": ["Last time they lost four-nil to Germany."],
    "old result all square": ["Two wins, the other ending all square."],
    "formation 4-4-2": ["They line up in a 4-4-2 today."],
    "disallowed for offside": ["Goal disallowed for offside."],
    "scores but ruled out": ["He scores!", "But it's been ruled out."],
    "no goal": ["No goal!"],
    "saved by the keeper": ["Saved by the keeper!"],
    "just wide": ["Just wide!"],
    "hits the bar": ["Hits the bar!"],
    "off the post": ["Off the post!"],
    "misses": ["Oh, he misses!"],
    "chalked off": ["Scores! But it is chalked off."],
}


@pytest.mark.parametrize("lines", NOT_GOALS.values(), ids=NOT_GOALS.keys())
def test_recaps_generic_words_and_cancels_are_not_goal(lines):
    label, reason = label_of(*lines)
    assert label != "Goal", reason


def test_var_check_then_goal_stands_is_goal():
    label, reason = label_of("Scores!", "VAR is checking for offside.", "And the goal stands.")
    assert label == "Goal"
    assert "confirmed by 'goal stands'" in reason


def test_save_then_rebound_goal_is_goal():
    # a cancel only undoes confirms before it
    label, _ = label_of("Saved by the keeper!", "And he scores from the rebound!")
    assert label == "Goal"


def test_cancel_exception_keeps_goal():
    label, _ = label_of("Costa scores!", "The keeper should have saved that.")
    assert label == "Goal"


def test_cancelled_call_is_named_in_reason():
    _, reason = label_of("He scores!", "No, it's disallowed.")
    assert "goal call undone by 'disallowed'" in reason
    assert "confirmed" not in reason


def test_low_audio_with_confirm_phrase_is_not_goal():
    label, reason = label_of("Salah gives Liverpool the lead!", audio_z=0.3)
    assert label != "Goal"
    assert "calm crowd" in reason


def test_reason_names_the_phrase():
    _, reason = label_of("Salah gives Liverpool the lead.")
    assert "confirmed by 'gives liverpool the lead'" in reason
    _, reason = label_of("It's now 1-0!")
    assert "confirmed by 'it's now 1-0'" in reason   # the longest match names the call


def test_words_outside_keyword_window_are_ignored():
    # keyword_window_s = [-5, 20]: a call 30 s after the peak belongs to another moment
    df = signals()
    assert classify(PEAK, df, CFG, words("Costa scores!", start=PEAK + 30))[0] != "Goal"
    assert classify(PEAK, df, CFG, words("Costa scores!", start=PEAK - 4))[0] == "Goal"


def test_without_words_hit_labels_still_work_and_cancel():
    # fallback used by the check scripts: confirm/cancel searched in the keyword hit labels
    df = signals(hits=[(101, "scores(1.35)")])
    assert classify(PEAK, df, CFG)[0] == "Goal"
    df = signals(hits=[(101, "scores(1.35)"), (106, "saved by(0.80)")])
    assert classify(PEAK, df, CFG)[0] != "Goal"


def test_unknown_placeholder_is_a_clear_error():
    cfg = {**CFG, "classifier": {**CFG["classifier"], "confirm_keywords": ["gives <club> the lead"]}}
    with pytest.raises(ValueError, match="placeholder"):
        classify(PEAK, signals(), cfg, words("Salah gives Liverpool the lead."))
