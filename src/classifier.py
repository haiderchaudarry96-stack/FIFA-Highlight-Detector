"""Rule-based event type + human-readable reason for one highlight peak."""
from __future__ import annotations

import bisect
import functools
import itertools
import re
from collections import Counter
from typing import NamedTuple

import pandas as pd

from .commentary import _normalize
from .utils import get_logger

log = get_logger(__name__)

# Defaults for the `classifier` config keys (config.yaml overrides them with the full lists).
# chance/card keywords are matched as whole words inside a keyword hit ("red card" matches "card" and "red").
# confirm/cancel entries are phrase patterns, matched case-insensitively on whole words of the commentary
# (normalized like the keyword matcher, so "it's" = "it s"): `<team>` = any 1-3 words, `<score>` = a scoreline
# ("1-0", "two-one", "one-nil"), a trailing "!" = the call must be exclaimed ("scores!"). "N-N" = `<score>`.
CLASSIFIER_DEFAULTS = {
    "keyword_window_s": [-5, 20],
    "replay_window_s": [0, 60],
    "goal_audio_z_min": 1.0,
    "confirm_keywords": ["scores!", "scored!", "<score>!", "it's <score>", "makes it <score>", "takes the lead",
                         "gives <team> the lead",
                         "equaliser!", "equalizer!", "it's a goal", "goal stands"],
    "cancel_keywords": ["disallowed", "ruled out", "chalked off", "offside", "no goal", "not a goal", "saved",
                        "denied", "just wide", "just over", "off the post", "hits the bar", "hit the post",
                        "misses", "wasted", "var is checking"],
    "cancel_exceptions": ["should have saved", "could have saved", "never misses", "not offside"],
    "chance_keywords": ["save", "saved", "chance", "post", "crossbar", "underside"],
    "card_keywords": ["card", "penalty", "foul", "yellow", "red", "var", "spot"],
}

_LABEL = re.compile(r"^(.*)\(([-0-9.]+)\)$")
_NUMBERS = {*"0123456789", "nil", "zero", "nothing", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine"}
_SCORE = "#score"   # stand-in token for a scoreline (_normalize never produces "#")
_PLACEHOLDERS = {"<team>": r"[a-z0-9]+(?: [a-z0-9]+){0,2}", "<score>": re.escape(_SCORE)}
_EXCL_LOOKAHEAD = 2  # the "!" may sit on the call's last word or up to 2 words later ("scores for Liverpool!")


class _Tok(NamedTuple):
    text: str              # normalized token, _SCORE, or "|" (separator between keyword hits)
    mark: str              # sentence-end mark of the word ending here ("!", ".", "?" or "")
    src: tuple[int, int]   # first/last source text (word or hit) this token came from


class _Match(NamedTuple):
    start: int             # first/last token position in the stream
    end: int
    shown: str             # text for the reason, e.g. "gives liverpool the lead"


def _ccfg(cfg: dict, key: str):
    return (cfg.get("classifier") or {}).get(key, CLASSIFIER_DEFAULTS[key])


def _parse_hits(labels) -> list[tuple[str, float]]:
    """"goal(1.50); save(0.70)" cells -> [("goal", 1.5), ("save", 0.7)]."""
    hits = []
    for cell in labels:
        for part in str(cell or "").split(";"):
            m = _LABEL.match(part.strip())
            if m:
                hits.append((m.group(1).strip(), float(m.group(2))))
    return hits


def _in_group(phrase: str, terms: list[str]) -> bool:
    """True if any term (one or more words) appears as whole words in phrase."""
    toks = _normalize(phrase)
    for term in terms:
        t = _normalize(str(term))
        if t and any(toks[i:i + len(t)] == t for i in range(len(toks) - len(t) + 1)):
            return True
    return False


def _maybe_exclaimed(phrase: str, value: float, cfg: dict) -> bool:
    """True if the hit value can be explained by a boost set that includes exclamation_boost.

    Labels only store the final value (weight x boosts), so when the boosts are equal (all 0.5 by default)
    an exclamation can't be told apart from an elongated or repeated call. All three mark a live,
    shouted call rather than calm recap talk, so the ambiguity is harmless here.
    """
    weight = (cfg.get("keywords") or {}).get(phrase)
    opts = cfg.get("keyword_options") or {}
    excl = float(opts.get("exclamation_boost", 0.0))
    if not weight or excl <= 0:
        return False
    ratio = value / float(weight)
    others = [1 + float(opts.get(k, 0.0)) for k in ("elongation_boost", "repeat_boost")]
    for r in range(len(others) + 1):
        for combo in itertools.combinations(others, r):
            product = 1 + excl
            for f in combo:
                product *= f
            if abs(ratio - product) <= 0.01 * product + 0.01:   # labels are rounded to 2 decimals
                return True
    return False


# ---------------------------------------------------------------- confirm / cancel phrases

def _with_scorelines(toks: list[_Tok]) -> list[_Tok]:
    """Replace each run of exactly two number tokens ("1 0", "two one", "one nil") by one _SCORE token.

    Longer runs are formations or times ("4 4 2"), and "one two" is the passing move, so they stay words.
    """
    out: list[_Tok] = []
    i = 0
    while i < len(toks):
        j = i
        while j < len(toks) and toks[j].text in _NUMBERS:
            j += 1
        if j - i == 2 and (toks[i].text, toks[i + 1].text) != ("one", "two"):
            out.append(_Tok(_SCORE, toks[i + 1].mark, (toks[i].src[0], toks[i + 1].src[1])))
        else:
            out.extend(toks[i:max(j, i + 1)])
        i = max(j, i + 1)
    return out


def _word_stream(words: list[dict], start_s: float, end_s: float) -> tuple[list[_Tok], list[str]]:
    """Tokens of the words starting inside [start_s, end_s] (the transcript snippet's window)."""
    texts = [str(w.get("word", "")) for w in words if start_s <= float(w.get("start", 0.0)) <= end_s]
    toks = []
    for i, text in enumerate(texts):
        norm = _normalize(text)
        marks = [c for c in re.search(r"[^A-Za-z0-9]*$", text).group() if c in ".!?"]
        toks += [_Tok(t, marks[-1] if marks and j == len(norm) - 1 else "", (i, i)) for j, t in enumerate(norm)]
    return _with_scorelines(toks), texts


def _hit_stream(hits: list[tuple[str, float]], cfg: dict) -> tuple[list[_Tok], list[str]]:
    """Fallback without words: the keyword hit phrases, "!" when the hit was (maybe) exclaimed."""
    toks = []
    for i, (phrase, value) in enumerate(hits):
        norm = _normalize(phrase)
        mark = "!" if _maybe_exclaimed(phrase, value, cfg) else "."
        toks += [_Tok(t, mark if j == len(norm) - 1 else "", (i, i)) for j, t in enumerate(norm)]
        toks.append(_Tok("|", ".", (i, i)))   # phrases never match across two hits
    return _with_scorelines(toks), [p for p, _ in hits]


@functools.lru_cache(maxsize=None)
def _compile(entry: str) -> tuple[re.Pattern, bool]:
    """Config entry -> (regex over the space-joined token stream, needs an exclamation)."""
    entry = entry.strip().lower()
    needs_excl = entry.endswith("!")
    body = entry.rstrip("!").strip()
    if body == "n-n":
        body = "<score>"
    parts = []
    for piece in re.split(r"(<[^>]*>)", body):
        if piece.startswith("<"):
            if piece not in _PLACEHOLDERS:
                raise ValueError(f"classifier: unknown placeholder {piece} in {entry!r} (use <team> or <score>)")
            parts.append(_PLACEHOLDERS[piece])
        else:
            parts += [re.escape(t) for t in _normalize(piece)]
    if not parts:
        raise ValueError(f"classifier: empty phrase {entry!r}")
    return re.compile(r"(?<![^ ])" + " ".join(parts) + r"(?![^ ])"), needs_excl


def _find(stream: list[_Tok], texts: list[str], entries) -> list[_Match]:
    """All matches of the entries in the stream ("!" entries need a "!" on or just after the last word)."""
    joined = " ".join(t.text for t in stream)
    starts = list(itertools.accumulate((len(t.text) + 1 for t in stream[:-1]), initial=0))
    found = []
    for entry in entries:
        pattern, needs_excl = _compile(str(entry))
        for m in pattern.finditer(joined):
            a = bisect.bisect_right(starts, m.start()) - 1
            b = bisect.bisect_right(starts, m.end() - 1) - 1
            if needs_excl and next((t.mark for t in stream[b:b + 1 + _EXCL_LOOKAHEAD] if t.mark), "") != "!":
                continue
            src = range(stream[a].src[0], stream[b].src[1] + 1)
            shown = " ".join(texts[i] for i in src).lower().replace(" -", "-")   # Whisper splits "1" "-0"
            shown = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", shown)
            found.append(_Match(a, b, shown + ("!" if needs_excl else "")))
    return found


def _overlaps(x: _Match, y: _Match) -> bool:
    return x.start <= y.end and y.start <= x.end


def _goal_calls(stream: list[_Tok], texts: list[str], cfg: dict) -> tuple[list[str], list[str]]:
    """(confirm phrases that stand, cancel phrases that undid a confirm).

    A cancel ("disallowed", "saved", "var is checking") voids every confirm before it, so only a confirm
    after the last cancel counts ("VAR is checking ... the goal stands"). A confirm overlapping a cancel
    ("no goal!") never counts, and a cancel inside an exception ("should have saved") is no cancel.
    """
    exceptions = _find(stream, texts, _ccfg(cfg, "cancel_exceptions"))
    cancels = [c for c in _find(stream, texts, _ccfg(cfg, "cancel_keywords"))
               if not any(_overlaps(c, e) for e in exceptions)]
    confirms = [c for c in _find(stream, texts, _ccfg(cfg, "confirm_keywords"))
                if not any(_overlaps(c, x) for x in cancels)]
    last_cancel = max((c.end for c in cancels), default=-1)
    live = [c for c in confirms if c.start > last_cancel]
    # name each call once: drop a match inside a longer one ("scores!" inside "and he scores")
    live = [c for c in live if not any(o.start <= c.start and c.end <= o.end and o.end - o.start > c.end - c.start
                                       for o in live)]
    first = min((c.start for c in confirms), default=None)
    undone = [] if live or first is None else [c.shown for c in cancels if c.end > first]
    return sorted({c.shown for c in live}), sorted(set(undone))


def classify(peak_s: int, signals_df: pd.DataFrame, cfg: dict,
             words: list[dict] | None = None) -> tuple[str, str]:
    """(event_type, reason) for the highlight at peak_s.

    signals_df: per-second table from scorer.combine_signals (needs second, audio_z, keyword_hits, replay).
    cfg: the full config (uses `classifier`, `keywords`, `keyword_options`).
    words: the transcript (commentary.transcribe output). confirm/cancel phrases are searched in its words
           inside the keyword window; without it, only in the keyword hit labels.

    Rules, first match wins:
      Goal              a confirm_keywords phrase (live call: "Costa scores!", "it's 1-0", "gives Liverpool
                        the lead") in the keyword window, not undone by a later cancel_keywords phrase
                        ("ruled out", "saved"), AND max audio_z there >= goal_audio_z_min (a low sanity floor).
                        A plain "goal" or "penalty" never makes a Goal: recaps and chance talk use them too.
                        A replay is only mentioned in the reason (replays also follow near-misses).
      Big Chance / Save save/chance/post/crossbar words in the keyword window
      Card / Foul       card/penalty/foul/var words in the keyword window
      Exciting Moment   everything else
    """
    kw_start, kw_end = (int(x) for x in _ccfg(cfg, "keyword_window_s"))
    rp_start, rp_end = (int(x) for x in _ccfg(cfg, "replay_window_s"))
    audio_min = float(_ccfg(cfg, "goal_audio_z_min"))

    sec = signals_df["second"]
    kw_win = signals_df[(sec >= peak_s + kw_start) & (sec <= peak_s + kw_end)]
    rp_win = signals_df[(sec >= peak_s + rp_start) & (sec <= peak_s + rp_end)]

    audio_max = float(kw_win["audio_z"].max()) if "audio_z" in kw_win and len(kw_win) else 0.0
    replay = bool("replay" in rp_win and (rp_win["replay"] > 0).any())
    hits = _parse_hits(kw_win["keyword_hits"]) if "keyword_hits" in kw_win else []

    if words is not None:
        stream, texts = _word_stream(words, peak_s + kw_start, peak_s + kw_end)
    else:
        stream, texts = _hit_stream(hits, cfg)
    confirms, undone = _goal_calls(stream, texts, cfg) if stream else ([], [])
    chance_hits = [p for p, _ in hits if _in_group(p, list(_ccfg(cfg, "chance_keywords")))]
    card_hits = [p for p, _ in hits if _in_group(p, list(_ccfg(cfg, "card_keywords")))]

    # reason, e.g. "crowd spike (audio z=1.9) + confirmed by 'scores!' + 'goal' x2 + replay detected"
    parts = [f"{'crowd spike' if audio_max >= audio_min else 'calm crowd'} (audio z={audio_max:.1f})"]
    if confirms:
        parts.append("confirmed by " + ", ".join(f"'{c}'" for c in confirms))
    if undone:
        parts.append("goal call undone by " + ", ".join(f"'{c}'" for c in undone))
    counts = Counter(p for p, _ in hits if not any(_in_group(c, [p]) for c in confirms))
    parts += [f"'{p}' x{n}" if n > 1 else f"'{p}'" for p, n in counts.most_common(4)]
    if replay:
        parts.append("replay detected")
    reason = " + ".join(parts)

    if confirms and audio_max >= audio_min:
        return "Goal", reason
    if chance_hits:
        return "Big Chance / Save", reason
    if card_hits:
        return "Card / Foul", reason
    return "Exciting Moment", reason
