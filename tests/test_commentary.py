"""Pure-Python tests for keyword scoring (fake word lists, no Whisper)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.commentary import get_snippet, score_keywords  # noqa: E402

KEYWORDS = {"goal": 1.0, "save": 0.7, "what a save": 0.9, "red card": 0.8}
OPTIONS = {"elongation_boost": 0.5, "repeat_boost": 0.5, "spread_s": 3}


def W(word, start):
    return {"word": word, "start": float(start), "end": float(start) + 0.4, "prob": 0.9}


def test_shape_and_range():
    df = score_keywords([W("Goal!", 10), W("save", 30)], KEYWORDS, OPTIONS, 60)
    assert list(df.columns) == ["second", "keyword_score", "keyword_hits"]
    assert len(df) == 60 and df["second"].tolist() == list(range(60))
    assert df["keyword_score"].between(0, 1).all()
    assert df["keyword_score"].max() == 1.0


def test_no_words_gives_zeros():
    df = score_keywords([], KEYWORDS, OPTIONS, 20)
    assert len(df) == 20
    assert (df["keyword_score"] == 0).all()
    assert (df["keyword_hits"] == "").all()


def test_elongated_goal_beats_plain_goal():
    words = [W("Gooooal!", 10), W("goal", 40)]
    df = score_keywords(words, KEYWORDS, OPTIONS, 60)
    assert df.loc[10, "keyword_score"] == 1.0
    assert np.isclose(df.loc[40, "keyword_score"], 1.0 / 1.5)
    assert df.loc[10, "keyword_hits"] == "goal(1.50)"
    assert df.loc[40, "keyword_hits"] == "goal(1.00)"


def test_elongation_without_vowel_change():
    df = score_keywords([W("GOOOOOL", 5)], KEYWORDS, OPTIONS, 10)
    assert df.loc[5, "keyword_hits"] == "goal(1.50)"


def test_repeated_goal_gets_boost():
    words = [W("goal", 10), W("goal,", 11), W("goal!", 12), W("goal", 50)]
    df = score_keywords(words, KEYWORDS, OPTIONS, 60)
    assert df.loc[10, "keyword_hits"] == "goal(1.50)"
    assert df.loc[50, "keyword_hits"] == "goal(1.00)"  # alone, no boost
    assert df["keyword_score"].idxmax() == 11
    assert df.loc[11, "keyword_score"] > 3 * df.loc[50, "keyword_score"]


def test_repeat_window_is_three_seconds():
    df = score_keywords([W("goal", 10), W("goal", 14)], KEYWORDS, OPTIONS, 30)
    assert df.loc[10, "keyword_hits"] == "goal(1.00)"
    assert df.loc[14, "keyword_hits"] == "goal(1.00)"


def test_multi_word_red_card():
    words = [W("That's", 19.5), W("a", 19.8), W("Red", 20.1), W("card.", 20.4)]
    df = score_keywords(words, KEYWORDS, OPTIONS, 40)
    assert df.loc[20, "keyword_hits"] == "red card(0.80)"
    assert df.loc[20, "keyword_score"] == 1.0


def test_split_words_do_not_match_phrase():
    words = [W("red", 5), W("shirt", 6), W("card", 7)]
    df = score_keywords(words, KEYWORDS, OPTIONS, 20)
    assert (df["keyword_hits"] == "").all()


def test_longest_phrase_wins():
    words = [W("What", 8), W("a", 8.3), W("save!", 8.6)]
    df = score_keywords(words, KEYWORDS, OPTIONS, 20)
    assert df.loc[8, "keyword_hits"] == "what a save(0.90)"  # not also "save"


def test_triangular_spread():
    df = score_keywords([W("goal", 10)], KEYWORDS, OPTIONS, 30)
    s = df["keyword_score"]
    assert s[10] == 1.0
    assert s[7] > 0 and s[13] > 0 and s[6] == 0 and s[14] == 0
    assert s[9] > s[8] > s[7] and np.isclose(s[9], s[11])


def test_hits_outside_range_ignored():
    df = score_keywords([W("goal", 100)], KEYWORDS, OPTIONS, 10)
    assert (df["keyword_score"] == 0).all()


def test_get_snippet():
    words = [W("What", 8), W("a", 8.3), W("save!", 8.6), W("Corner", 15)]
    assert get_snippet(words, 7, 10) == "What a save!"
    assert get_snippet(words, 20, 30) == ""


# --- live-call follow-up ("Costa scores!" at 25:34 in test_match, base model) ---

LIVE_KEYWORDS = {"goal": 1.0, "scores": 0.9, "it's 1-0": 1.0, "1-0": 0.9}
LIVE_OPTIONS = {**OPTIONS, "exclamation_boost": 0.5}

# exact word list from the base transcript around 25:34, plus the calm recap a few seconds later
COSTA_WORDS = [
    W("It's", 1530.68), W("Diego", 1531.04), W("Costa.", 1531.26), W("Can", 1532.22), W("he", 1532.46),
    W("get", 1532.56), W("the", 1532.68), W("shot", 1532.8), W("away?", 1533.0),
    W("Costa", 1533.76), W("scores!", 1534.04),
    W("Portugal", 1541.99), W("are", 1542.41), W("extremely", 1543.63), W("upset", 1544.87),
    W("about", 1545.49), W("the", 1545.97), W("manner", 1546.19), W("of", 1546.49),
    W("that", 1546.67), W("goal.", 1546.83),
]


def test_costa_scores_live_call_beats_recap_goal():
    df = score_keywords(COSTA_WORDS, LIVE_KEYWORDS, LIVE_OPTIONS, 1800)
    assert df.loc[1534, "keyword_hits"] == "scores(1.35)"  # "!" stripped for matching, but boosts
    assert df.loc[1546, "keyword_hits"] == "goal(1.00)"
    assert df["keyword_score"].idxmax() == 1534
    assert df.loc[1534, "keyword_score"] == 1.0


def test_exclamation_only_from_own_sentence():
    # "away!" ends the previous sentence; "scores." ends its own -> no boost
    words = [W("away!", 10), W("Costa", 11), W("scores.", 11.5)]
    df = score_keywords(words, LIVE_KEYWORDS, LIVE_OPTIONS, 20)
    assert df.loc[11, "keyword_hits"] == "scores(0.90)"
    # "!" a couple of words after the keyword still counts
    words = [W("Goal", 5), W("for", 5.3), W("Portugal!", 5.6)]
    df = score_keywords(words, LIVE_KEYWORDS, LIVE_OPTIONS, 20)
    assert df.loc[5, "keyword_hits"] == "goal(1.50)"


def test_scoreline_split_by_whisper():
    # base transcript at 06:00: "It's" "1" "-0" "Portugal"
    words = [W("It's", 359.46), W("1", 360.18), W("-0", 360.38), W("Portugal", 360.52)]
    df = score_keywords(words, LIVE_KEYWORDS, LIVE_OPTIONS, 400)
    assert df.loc[359, "keyword_hits"] == "it's 1-0(1.00)"


def test_digits_do_not_stretch():
    words = [W("11", 5), W("0", 5.3), W("10", 8)]
    df = score_keywords(words, LIVE_KEYWORDS, LIVE_OPTIONS, 20)
    assert (df["keyword_hits"] == "").all()
