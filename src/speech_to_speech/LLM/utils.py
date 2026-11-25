import base64
import io
import re
import unicodedata
from typing import Optional

import requests  # type: ignore[import-untyped]
from nltk import sent_tokenize
from PIL import Image

SMART_PUNCT_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
)

# Sentence terminators that NLTK's English Punkt model does not know: Devanagari danda and
# double danda (shared by most Indic scripts), Urdu/Arabic full stop and question mark,
# Ol Chiki (Santali) and Meetei Mayek (Manipuri).
EXTRA_SENTENCE_TERMINATORS = "\u0964\u0965\u06d4\u061f\u1c7e\u1c7f\uabeb"

# Arabic-script comma and semicolon, used in Urdu.
_EXTRA_PUNCTUATION = "\u060c\u061b"

# Characters that are not speakable by themselves. Python's ``\w`` excludes combining marks, so
# matches still go through ``_keep_candidate`` before they are dropped.
SPEECHABLE_PATTERN = re.compile(
    r"[^\w\s.,!?;:'\"\-()\/\\@#%&*+=$€£¥₹₽¢\[\]{}<>~`^|…—–，。！？；：、\n\r\t"
    + EXTRA_SENTENCE_TERMINATORS
    + _EXTRA_PUNCTUATION
    + "]",
    flags=re.UNICODE,
)

# Zero-width (non-)joiner: they select letter forms in Indic scripts and Urdu.
_JOINERS = "\u200c\u200d"

_EXTRA_TERMINATOR_PATTERN = re.compile(rf"[{EXTRA_SENTENCE_TERMINATORS}]+[\"'”’)\]]*\s*")


def _is_spoken_mark(char: str) -> bool:
    """Whether ``char`` is a combining mark that carries sound, such as an Indic vowel sign."""
    if "\ufe00" <= char <= "\ufe0f" or "\U000e0100" <= char <= "\U000e01ef":
        return False  # Variation selectors only pick a glyph or emoji style.
    return unicodedata.category(char) in ("Mn", "Mc")


def _keep_candidate(match: re.Match[str]) -> str:
    char = match.group()
    if char in _JOINERS:
        # Keep joiners inside words; drop the ones that glue emoji sequences together.
        text, start = match.string, match.start()
        neighbour = text[start - 1] if start > 0 else text[start + 1 : start + 2]
        return char if neighbour and (neighbour.isalnum() or _is_spoken_mark(neighbour)) else ""
    return char if _is_spoken_mark(char) else ""


def remove_unspeechable(text: str) -> str:
    """Keep only speechable characters: letters, vowel signs, digits, punctuation, whitespace.
    support unicode characters (english, arabic, chinese, japanese, korean, indic scripts, etc.)
    """
    text = text.translate(SMART_PUNCT_TRANSLATION)
    return SPEECHABLE_PATTERN.sub(_keep_candidate, text)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, including at terminators NLTK does not handle.

    The last item may be an unfinished sentence, so it keeps its trailing whitespace: NLTK strips
    it, which would glue the next streamed word onto the previous one.
    """
    sentences: list[str] = []
    start = 0
    for match in _EXTRA_TERMINATOR_PATTERN.finditer(text):
        sentences.extend(sent_tokenize(text[start : match.end()]))
        start = match.end()
    sentences.extend(sent_tokenize(text[start:]))
    if sentences:
        sentences[-1] += text[len(text.rstrip()) :]
    return sentences


# Maps an STT language code to the language name used in the "Please reply ... in {name}"
# prompt. Every language any bundled STT backend can report needs an entry here, otherwise
# `--enable_lang_prompt` silently emits no instruction for it. The names are lowercase
# because they are interpolated mid-sentence.
#
# `tests/test_llm_utils.py` asserts this covers the SUPPORTED_LANGUAGES of every bundled STT
# handler, so adding a language to a handler without adding it here fails CI.
WHISPER_LANGUAGE_TO_LLM_LANGUAGE = {
    "en": "english",
    "fr": "french",
    "es": "spanish",
    "zh": "chinese",
    "ja": "japanese",
    "ko": "korean",
    "hi": "hindi",
    "de": "german",
    "pt": "portuguese",
    "pl": "polish",
    "it": "italian",
    "nl": "dutch",
    # The remaining languages Parakeet TDT v3 (the default STT) detects and reports.
    "ru": "russian",
    "uk": "ukrainian",
    "cs": "czech",
    "sk": "slovak",
    "hu": "hungarian",
    "ro": "romanian",
    "bg": "bulgarian",
    "hr": "croatian",
    "sl": "slovenian",
    "sr": "serbian",
    "da": "danish",
    "no": "norwegian",
    "sv": "swedish",
    "fi": "finnish",
    "et": "estonian",
    "lv": "latvian",
    "lt": "lithuanian",
}


def resolve_auto_language(language_code: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Strip the ``-auto`` suffix and resolve the human-readable language name.

    Returns ``(clean_code, language_name)``.  ``language_name`` is non-None
    when the code (with or without ``-auto``) maps to a known language.
    """
    if not language_code:
        return language_code, None
    if language_code.endswith("-auto"):
        language_code = language_code[:-5]
    if language_code not in WHISPER_LANGUAGE_TO_LLM_LANGUAGE:
        return language_code, None
    return language_code, WHISPER_LANGUAGE_TO_LLM_LANGUAGE.get(language_code)


def image_url_to_pil(image_url: str) -> Image.Image:
    """Convert an image URL or base64 data URI to a PIL Image.

    Accepts:
    - 'data:image/...;base64,<b64>' data URIs
    - 'https://...`` or ``http://...' URLs (fetched with a 10s timeout)
    """
    if image_url.startswith("data:"):
        _, b64_data = image_url.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(b64_data)))
    resp = requests.get(image_url, timeout=10)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content))
