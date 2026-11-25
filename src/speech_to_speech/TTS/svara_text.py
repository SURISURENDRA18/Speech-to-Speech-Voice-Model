"""Text front end for Svara TTS: language choice, speaker names, and Indic text normalization."""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import threading
import unicodedata
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)

# Speaker names come from Kenpath's svara-tts-v1 voice list. The keys are the codes used by
# indic-text-normalization, which also covers exactly these 19 languages.
SVARA_LANGUAGES = {
    "as": "Assamese",
    "bho": "Bhojpuri",
    "bn": "Bengali",
    "brx": "Bodo",
    "doi": "Dogri",
    "en": "English",
    "gu": "Gujarati",
    "hi": "Hindi",
    "hne": "Chhattisgarhi",
    "kn": "Kannada",
    "mag": "Magahi",
    "mai": "Maithili",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "pa": "Punjabi",
    "sa": "Sanskrit",
    "ta": "Tamil",
    "te": "Telugu",
}

# Languages written in each script; the first entry is the default for that script.
_SCRIPT_LANGUAGES: dict[str, tuple[str, ...]] = {
    "devanagari": ("hi", "mr", "ne", "sa", "bho", "mag", "mai", "hne", "brx", "doi"),
    "bengali": ("bn", "as"),
    "gurmukhi": ("pa",),
    "gujarati": ("gu",),
    "tamil": ("ta",),
    "telugu": ("te",),
    "kannada": ("kn",),
    "malayalam": ("ml",),
    "latin": ("en",),
}

_SCRIPT_RANGES = (
    (0x0900, 0x097F, "devanagari"),
    (0x0980, 0x09FF, "bengali"),
    (0x0A00, 0x0A7F, "gurmukhi"),
    (0x0A80, 0x0AFF, "gujarati"),
    (0x0B80, 0x0BFF, "tamil"),
    (0x0C00, 0x0C7F, "telugu"),
    (0x0C80, 0x0CFF, "kannada"),
    (0x0D00, 0x0D7F, "malayalam"),
)

# Angle-bracketed text would reach the model as control tokens or style tags.
_ANGLE_TAG = re.compile(r"<[^<>]*>")
_MARKDOWN = re.compile(r"[*_#`~|]+")
_SPACES = re.compile(r"\s+")

# Commas in 1,250 and 1,25,000. The English grammar reads "₹1,250" as "₹1, two hundred and fifty".
_DIGIT_GROUP_COMMA = re.compile(r"(?<=\d),(?=\d{2,3}(?:,\d{2,3})*(?!\d))")

# The English grammar does not know the rupee sign.
_RUPEES = re.compile(r"(?:₹|\bRs\.?)\s*(\d+(?:\.\d+)?)")

# Clause ends. Full stops are left to the grammar unless they end the text, because "Dr." and "10.5"
# need them.
_CLAUSE_END = re.compile(r"(\s*(?:[,;!?।॥…]+|\.+(?=\s*$))\s*)")

# Written after a grammar is fully compiled, so that an interrupted compile is redone.
_COMPILED_MARKER = ".compiled"
_COMPILE_TIMEOUT_S = 900

# The danda marks (U+0964, U+0965) sit in the Devanagari block but are shared by most Indic scripts.
_SHARED_MARKS = {0x0964, 0x0965}


def _script_of(char: str) -> str | None:
    point = ord(char)
    if point in _SHARED_MARKS:
        return None
    for start, end, script in _SCRIPT_RANGES:
        if start <= point <= end:
            return script
    if char.isascii() and char.isalpha():
        return "latin"
    return None


def detect_language(text: str, hint: str | None = None, default: str = "en") -> str:
    """Choose the Svara language from the script of ``text``.

    ``hint`` is the language the user spoke. It separates languages that share a script (Hindi and
    Marathi, for example) and decides for text without letters, but never overrides the script.
    """
    hint = (hint or "").removesuffix("-auto").lower() or None
    if hint not in SVARA_LANGUAGES:
        hint = None

    counts = Counter(script for script in map(_script_of, text) if script)
    if not counts:
        return hint or default

    # Indic replies often carry English names and acronyms, so any real share of Indic letters wins.
    indic = Counter({script: n for script, n in counts.items() if script != "latin"})
    script = "latin"
    if indic and sum(indic.values()) >= counts["latin"] * 0.5:
        script = indic.most_common(1)[0][0]

    languages = _SCRIPT_LANGUAGES[script]
    if hint in languages:
        return hint
    if script == "bengali" and any(char in "ৰৱ" for char in text):
        return "as"  # These two letters exist only in Assamese.
    if default in languages:
        return default
    return languages[0]


def speaker_name(language: str, gender: str) -> str:
    """Return a Svara speaker such as ``Hindi (Female)``."""
    return f"{SVARA_LANGUAGES[language]} ({gender.capitalize()})"


def gender_from_voice(voice: str | None, default: str) -> str:
    """Read the gender from a session voice. Other backends' voice names fall back to the default."""
    name = (voice or "").lower()
    if "female" in name:
        return "female"
    if "male" in name:
        return "male"
    return default


def clean_text(text: str) -> str:
    """Remove markup that should not be spoken or could be read as a control token."""
    text = _ANGLE_TAG.sub(" ", text).replace("<", " ").replace(">", " ")
    return _SPACES.sub(" ", _MARKDOWN.sub(" ", text)).strip()


def has_speech(text: str) -> bool:
    return any(char.isalnum() for char in text)


def split_long_text(text: str, max_chars: int) -> list[str]:
    """Split text that is too long for one generation, at punctuation if possible, then at spaces."""
    chunks: list[str] = []
    rest = text.strip()
    while len(rest) > max_chars:
        window = rest[: max_chars + 1]
        cut = max(window.rfind(mark) for mark in ("। ", ". ", "? ", "! ", "; ", ", ")) + 1
        if cut < max_chars // 3:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = max_chars
        chunks.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        chunks.append(rest)
    return chunks


class IndicNormalizer:
    """Expand numbers, dates, currency and similar into words with indic-text-normalization.

    Compiling a language's grammar takes about a minute and holds Python's global lock, which would
    freeze audio. Grammars are therefore compiled in a separate process and cached on disk, and text
    passes through unchanged until its grammar is ready. Languages in ``preload`` are ready before the
    constructor returns. A failure never blocks speech: the input is returned.
    """

    def __init__(self, cache_dir: str, preload: list[str] | None = None) -> None:
        # Imported eagerly so that a missing dependency is reported at startup.
        from indic_text_normalization import Normalizer

        self._factory: Any = Normalizer
        self._cache_dir = Path(cache_dir).expanduser()
        self._lock = threading.Lock()
        self._normalizers: dict[str, Any] = {}
        self._pending: set[str] = set()
        for language in preload or []:
            self._load(language)

    def _load(self, language: str) -> None:
        start = perf_counter()
        # The .far files are not reliably named per language, so each language needs its own folder.
        cache = self._cache_dir / language
        built = None
        try:
            if not (cache / _COMPILED_MARKER).exists():
                logger.info("Compiling the text normalization grammar for %r", language)
                subprocess.run(
                    [sys.executable, "-m", __name__, str(self._cache_dir), language],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=_COMPILE_TIMEOUT_S,
                )
            built = self._factory(input_case="cased", lang=language, cache_dir=str(cache))
        except Exception as e:
            logger.error("Text normalization for %r is unavailable: %s: %s", language, type(e).__name__, e)
        with self._lock:
            self._normalizers[language] = built
            self._pending.discard(language)
        if built is not None:
            logger.info("Text normalization for %r ready in %.1fs", language, perf_counter() - start)

    def _get(self, language: str) -> Any:
        with self._lock:
            if language in self._normalizers:
                return self._normalizers[language]
            if language in self._pending:
                return None
            self._pending.add(language)
        threading.Thread(target=self._load, args=(language,), daemon=True, name=f"normalizer-{language}").start()
        return None

    def normalize(self, text: str, language: str) -> str:
        normalizer = self._get(language)
        if normalizer is None:
            return text
        # The Bengali grammar reads its own digits one at a time, but reads 0-9 as numbers.
        text = "".join(str(unicodedata.digit(c)) if c.isdecimal() else c for c in text)
        prepared = _DIGIT_GROUP_COMMA.sub("", text)
        if language == "en":
            prepared = _RUPEES.sub(r"\1 rupees", prepared)
        # Some grammars read punctuation aloud (Gujarati says the word for "comma") or put spaces
        # around it, so they only ever see the words between punctuation marks.
        parts = _CLAUSE_END.split(prepared)
        parts[::2] = [self._normalize_clause(normalizer, clause, language) for clause in parts[::2]]
        return _SPACES.sub(" ", "".join(parts)).strip() or text

    @staticmethod
    def _normalize_clause(normalizer: Any, clause: str, language: str) -> str:
        if not has_speech(clause):
            return clause
        try:
            return _SPACES.sub(" ", normalizer.normalize(clause)).strip() or clause
        except Exception as e:
            logger.warning("Text normalization failed for %r: %s: %s", language, type(e).__name__, e)
            return clause


def compile_grammar(cache_dir: str, language: str) -> None:
    """Compile and cache one language's grammar."""
    from indic_text_normalization import Normalizer

    cache = Path(cache_dir).expanduser() / language
    cache.mkdir(parents=True, exist_ok=True)
    Normalizer(input_case="cased", lang=language, cache_dir=str(cache))
    (cache / _COMPILED_MARKER).touch()


if __name__ == "__main__":
    # python -m speech_to_speech.TTS.svara_text CACHE_DIR LANGUAGE [LANGUAGE ...]
    for code in sys.argv[2:]:
        compile_grammar(sys.argv[1], code)
