"""Rule-based event type + human-readable reason for one highlight peak."""
from __future__ import annotations

import itertools
import re
from collections import Counter

import pandas as pd

from .commentary import _normalize
from .utils import get_logger

log = get_logger(__name__)

# Defaults for the `classifier` config keys (config.yaml overrides them).
# Keyword entries are matched as whole words inside the hit phrase ("red card" matches "card" and "red").
# confirm_keywords: a trailing "!" means the hit must be exclamation-boosted ("scores!" = "Costa scores!"),
# "N-N" stands for any scoreline ("1-0", "it's 2-1").
CLASSIFIER_DEFAULTS = {
    "keyword_window_s": [-5, 20],
    "replay_window_s": [0, 60],
    "goal_audio_z_min": 1.0,
    "confirm_keywords": ["scores!", "scored!", "N-N", "takes the lead", "equaliser", "equalizer",
                         "it's a goal"],
    "chance_keywords": ["save", "saved", "chance", "post", "crossbar", "underside"],
    "card_keywords": ["card", "penalty", "foul", "yellow", "red", "var", "spot"],
}

_LABEL = re.compile(r"^(.*)\(([-0-9.]+)\)$")
_SCORELINE = re.compile(r"\d+\s*-\s*\d+")   # "1-0", "it's 2-1"


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


def _confirm_entry(phrase: str, value: float, entries: list[str], cfg: dict) -> str | None:
    """The confirm_keywords entry this hit satisfies (e.g. "scores!"), or None."""
    for entry in entries:
        entry = str(entry).strip()
        needs_excl = entry.endswith("!")
        term = entry.rstrip("!").strip()
        if term.upper() == "N-N":
            matched = bool(_SCORELINE.search(phrase))
        else:
            matched = _in_group(phrase, [term])
        if matched and (not needs_excl or _maybe_exclaimed(phrase, value, cfg)):
            return f"{phrase}!" if needs_excl else phrase
    return None


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


def classify(peak_s: int, signals_df: pd.DataFrame, cfg: dict) -> tuple[str, str]:
    """(event_type, reason) for the highlight at peak_s.

    signals_df: per-second table from scorer.combine_signals (needs second, audio_z, keyword_hits, replay).
    cfg: the full config (uses `classifier`, `keywords`, `keyword_options`).

    Rules, first match wins:
      Goal              a confirm_keywords hit (live call: "Costa scores!", "it's 1-0", "takes the lead")
                        in the keyword window AND max audio_z there >= goal_audio_z_min (a low sanity floor).
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

    confirm_entries = list(_ccfg(cfg, "confirm_keywords"))
    confirms = sorted({c for p, v in hits if (c := _confirm_entry(p, v, confirm_entries, cfg))})
    chance_hits = [p for p, _ in hits if _in_group(p, list(_ccfg(cfg, "chance_keywords")))]
    card_hits = [p for p, _ in hits if _in_group(p, list(_ccfg(cfg, "card_keywords")))]

    # reason, e.g. "crowd spike (audio z=1.9) + confirmed by 'scores!' + 'goal' x2 + replay detected"
    parts = [f"{'crowd spike' if audio_max >= audio_min else 'calm crowd'} (audio z={audio_max:.1f})"]
    if confirms:
        parts.append("confirmed by " + ", ".join(f"'{c}'" for c in confirms))
    confirmed_phrases = {c.rstrip("!") for c in confirms}
    counts = Counter(p for p, _ in hits if p not in confirmed_phrases)
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
