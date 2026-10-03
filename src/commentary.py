"""Commentary: Whisper transcription (word timestamps) + per-second keyword scoring."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import fmt_time, get_logger, is_cache_valid

log = get_logger(__name__)

_VOWELS = set("aeiouy")


# ---------------------------------------------------------------- transcription

def _load_wav_16k(wav_path: Path) -> np.ndarray:
    """Read the wav as mono float32 at 16 kHz (what Whisper expects).

    Passing an array avoids faster-whisper's PyAV decoder, which breaks with PyAV >= 15
    (`av.open(..., metadata_errors=...)` no longer exists). Whisper decodes the whole
    file into memory anyway, so this costs nothing extra (~230 MB per hour of audio).
    """
    import soundfile as sf

    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    return np.ascontiguousarray(audio, dtype=np.float32)


def _transcribe_faster(wav_path: Path, model_size: str, language: str | None) -> list[dict]:
    """faster-whisper on CPU (int8). Logs one line per decoded segment."""
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        _load_wav_16k(wav_path), language=language, word_timestamps=True, vad_filter=True,
    )
    total = info.duration or 0.0
    words: list[dict] = []
    for seg in segments:  # lazy generator: decoding happens while iterating
        for w in seg.words or []:
            words.append({"word": w.word.strip(), "start": float(w.start), "end": float(w.end),
                          "prob": float(w.probability)})
        pct = 100.0 * seg.end / total if total else 0.0
        log.info("[%s / %s %3.0f%%] %s", fmt_time(seg.end), fmt_time(total), pct, seg.text.strip())
    return words


def _transcribe_openai(wav_path: Path, model_size: str, language: str | None) -> list[dict]:
    """openai-whisper fallback (PyTorch, CPU). Progress via whisper's own tqdm bar."""
    import whisper

    model = whisper.load_model(model_size, device="cpu")
    result = model.transcribe(str(wav_path), language=language, word_timestamps=True,
                              fp16=False, verbose=False)
    words: list[dict] = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            words.append({"word": w["word"].strip(), "start": float(w["start"]), "end": float(w["end"]),
                          "prob": float(w.get("probability", 0.0))})
    return words


def _load_cached_transcript(cache_path: Path, wav_path: Path, model_size: str) -> list[dict] | None:
    """Return cached words if the cache is newer than the wav, parses, and matches the model."""
    if not is_cache_valid(cache_path, wav_path):
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        words = data["words"]
        if data.get("model") != model_size or not isinstance(words, list):
            return None
        if words and not {"word", "start", "end", "prob"} <= set(words[0]):
            return None
        return words
    except (OSError, ValueError, KeyError, TypeError):
        log.warning("Ignoring unreadable transcript cache: %s", cache_path)
        return None


def transcribe(wav_path: str | Path, model_size: str = "base", backend: str = "auto",
               cache_path: str | Path | None = None, language: str | None = "en") -> list[dict]:
    """Transcribe commentary to words: [{"word", "start", "end", "prob"}, ...].

    backend: "auto" (faster-whisper, else openai-whisper), "faster" or "openai".
    If cache_path is given, a valid cache (newer than the wav, same model) is reused,
    otherwise the result is written there.
    """
    wav_path = Path(wav_path)
    if not wav_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")
    cache_path = Path(cache_path) if cache_path else None

    if cache_path:
        cached = _load_cached_transcript(cache_path, wav_path, model_size)
        if cached is not None:
            log.info("Using cached transcript: %s (%d words)", cache_path, len(cached))
            return cached

    if backend not in ("auto", "faster", "openai"):
        raise ValueError(f"Unknown whisper backend: {backend!r} (use auto | faster | openai)")
    used = backend
    if backend in ("auto", "faster"):
        try:
            import faster_whisper  # noqa: F401
            used = "faster"
        except ImportError:
            if backend == "faster":
                raise ImportError("faster-whisper is not installed: pip install faster-whisper")
            log.warning("faster-whisper not installed, falling back to openai-whisper")
            used = "openai"
    if used == "openai":
        try:
            import whisper  # noqa: F401
        except ImportError:
            raise ImportError("No Whisper backend installed: pip install faster-whisper "
                              "(or openai-whisper)")

    log.info("Transcribing %s with %s-whisper '%s' (CPU) ...", wav_path.name, used, model_size)
    fn = _transcribe_faster if used == "faster" else _transcribe_openai
    words = fn(wav_path, model_size, language)
    log.info("Transcription done: %d words", len(words))

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".part.json")
        tmp.write_text(json.dumps({"model": model_size, "backend": used, "words": words}),
                       encoding="utf-8")
        tmp.replace(cache_path)
        log.info("Saved %s", cache_path)
    return words


# ---------------------------------------------------------------- keyword scoring

def _normalize(text: str) -> list[str]:
    """Lowercase and split on anything that is not a letter/digit ("Goal!" -> ["goal"])."""
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def _stretch_pattern(token: str) -> re.Pattern:
    """Regex that also matches elongated forms: "goal" -> g+[oa]{2,}l+ ("goooal", "gooool").

    Digits never stretch ("1" must not match "11").
    """
    parts = []
    for run in re.findall(r"[aeiouy]+|[^aeiouy]", token):
        if run.isdigit():
            parts.append(run)
        elif run[0] in _VOWELS:
            parts.append(f"[{''.join(sorted(set(run)))}]{{{len(run)},}}")
        else:
            parts.append(re.escape(run) + "+")
    return re.compile("^" + "".join(parts) + "$")


def score_keywords(words: list[dict], keywords_cfg: dict, options_cfg: dict,
                   n_seconds: int) -> pd.DataFrame:
    """Per-second keyword score (0..1) and hit labels from transcribed words.

    - Phrases (single or multi-word) are matched greedily, longest first, so
      "what a save" counts once (not also as "save").
    - A hit's value is weight * (1 + elongation_boost if stretched, e.g. "goooal")
      * (1 + exclamation_boost if its sentence ends with "!", e.g. "Costa scores!")
      * (1 + repeat_boost if the same phrase is hit again within 3 s).
      Whisper puts "!" on shouted live calls, so this separates them from calm recap talk.
    - Each hit is spread over ±spread_s seconds with a triangular kernel, then the
      whole series is divided by its max.
    """
    n_seconds = max(0, int(n_seconds))
    elong_boost = float(options_cfg.get("elongation_boost", 0.0))
    excl_boost = float(options_cfg.get("exclamation_boost", 0.0))
    repeat_boost = float(options_cfg.get("repeat_boost", 0.0))
    spread = max(0, int(options_cfg.get("spread_s", 0)))
    repeat_window = 3.0

    # phrase -> (tokens, patterns, weight), longest phrases first
    phrases = []
    for phrase, weight in (keywords_cfg or {}).items():
        toks = _normalize(str(phrase))
        if toks:
            phrases.append((str(phrase).lower(), toks, [_stretch_pattern(t) for t in toks], float(weight)))
    phrases.sort(key=lambda p: -len(p[1]))

    # flatten words into tokens (a Whisper word like "off-the-post" gives several tokens)
    # Each token keeps the sentence-end mark (".", "!", "?") of its word, on the word's last token.
    tokens: list[tuple[str, float, str]] = []
    for w in words:
        text = str(w.get("word", ""))
        toks = _normalize(text)
        tail = re.search(r"[^A-Za-z0-9]*$", text).group()
        marks = [c for c in tail if c in ".!?"]
        for j, t in enumerate(toks):
            mark = marks[-1] if marks and j == len(toks) - 1 else ""
            tokens.append((t, float(w.get("start", 0.0)), mark))

    def exclaimed(end: int) -> bool:
        """True if the sentence holding tokens[..end) ends with "!" (looks a few tokens ahead)."""
        for _, _, mark in tokens[end - 1:end + 3]:
            if mark:
                return mark == "!"
        return False

    # 1) find raw hits: (time, phrase, weight, elongated, exclaimed)
    hits: list[tuple[float, str, float, bool, bool]] = []
    i = 0
    while i < len(tokens):
        for phrase, toks, pats, weight in phrases:
            k = len(toks)
            window = tokens[i:i + k]
            if len(window) == k and all(p.match(tok) for p, (tok, _, _) in zip(pats, window)):
                elongated = any(len(tok) > len(kt) for kt, (tok, _, _) in zip(toks, window))
                hits.append((window[0][1], phrase, weight, elongated, exclaimed(i + k)))
                i += k
                break
        else:
            i += 1

    # 2) boosts + spread
    raw = np.zeros(n_seconds, dtype=float)
    labels: list[list[str]] = [[] for _ in range(n_seconds)]
    offsets = np.arange(-spread, spread + 1)
    kernel = 1.0 - np.abs(offsets) / (spread + 1)
    for idx, (t, phrase, weight, elongated, excl) in enumerate(hits):
        value = weight * (1.0 + elong_boost if elongated else 1.0)
        if excl:
            value *= 1.0 + excl_boost
        repeated = any(j != idx and p == phrase and abs(tj - t) <= repeat_window
                       for j, (tj, p, *_) in enumerate(hits))
        if repeated:
            value *= 1.0 + repeat_boost
        sec = int(t)
        if not 0 <= sec < n_seconds:
            continue
        labels[sec].append(f"{phrase}({value:.2f})")
        for off, kw in zip(offsets, kernel):
            if 0 <= sec + off < n_seconds:
                raw[sec + off] += value * kw

    peak = raw.max() if n_seconds else 0.0
    score = raw / peak if peak > 0 else raw
    return pd.DataFrame({
        "second": np.arange(n_seconds, dtype=int),
        "keyword_score": score,
        "keyword_hits": ["; ".join(lab) for lab in labels],
    })


def get_snippet(words: list[dict], start_s: float, end_s: float) -> str:
    """Join the words that start inside [start_s, end_s] into one line of text."""
    text = " ".join(str(w["word"]).strip() for w in words if start_s <= float(w["start"]) <= end_s)
    return re.sub(r"\s+", " ", text).strip()
