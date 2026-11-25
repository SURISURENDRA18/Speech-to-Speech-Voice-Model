import importlib
import re

import pytest

from speech_to_speech.LLM.utils import (
    WHISPER_LANGUAGE_TO_LLM_LANGUAGE,
    remove_unspeechable,
    resolve_auto_language,
    split_sentences,
)


def test_remove_unspeechable_normalizes_smart_apostrophes() -> None:
    assert remove_unspeechable("I’ll reply if here’s the plan.") == "I'll reply if here's the plan."


def test_remove_unspeechable_keeps_text_and_drops_emoji() -> None:
    assert remove_unspeechable("Hello 👋 lobster 🦞") == "Hello  lobster "


def test_remove_unspeechable_keeps_chinese_punctuation() -> None:
    text = "你好，今天怎么样？很好！停顿；说明：一、二。"
    assert remove_unspeechable(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "नमस्ते, आप कैसे हैं? मैं ठीक हूँ।",  # Hindi
        "तुम्ही कसे आहात? मी ठीक आहे.",  # Marathi
        "আপনি কেমন আছেন? আমি ভালো আছি।",  # Bengali
        "আপুনি কেনে আছে?",  # Assamese
        "તમે કેમ છો? હું મજામાં છું.",  # Gujarati
        "ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?",  # Punjabi
        "ଆପଣ କେମିତି ଅଛନ୍ତି?",  # Odia
        "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",  # Tamil
        "నమస్కారం, మీరు ఎలా ఉన్నారు?",  # Telugu
        "ನೀವು ಹೇಗಿದ್ದೀರಿ?",  # Kannada
        "നിങ്ങൾക്ക് സുഖമാണോ?",  # Malayalam
        "آپ کیسے ہیں؟ میں ٹھیک ہوں۔ شکریہ، پھر ملیں گے؛",  # Urdu
        "\u0915\u094d\u200d\u0937 \u0928\u094d\u200d ",  # Word-internal and word-final joiners
        "cafe\u0301",  # Decomposed Latin accent
    ],
)
def test_remove_unspeechable_keeps_combining_marks_and_indic_punctuation(text) -> None:
    assert remove_unspeechable(text) == text


def test_remove_unspeechable_is_stable_across_stream_deltas() -> None:
    """Streamed deltas can start with a vowel sign that belongs to the previous delta."""
    text = "नमस्ते, आप कैसे हैं? வணக்கம்!"
    assert "".join(remove_unspeechable(char) for char in text) == text


# Telugu ZWNJ, Bengali ZWJ, then emoji whose joiners must still be dropped. The joiners are
# escaped because they are invisible in source.
_JOINER_STREAM = "\u0c38\u0c3e\u0c2b\u0c4d\u200c\u0c1f\u0c4d \u09b0\u200d\u09cd\u09af\u09be\u09ae\U0001f468\u200d\U0001f469 ok\u2764\ufe0f\u200d\U0001f525"
_JOINER_STREAM_SPOKEN = "\u0c38\u0c3e\u0c2b\u0c4d\u200c\u0c1f\u0c4d \u09b0\u200d\u09cd\u09af\u09be\u09ae ok"


def test_remove_unspeechable_keeps_joiners_across_stream_deltas() -> None:
    """Tokenizers emit a joiner as its own delta, so its neighbour is in the previous one."""
    previous = ["", *_JOINER_STREAM]
    spoken = "".join(remove_unspeechable(char, prev) for prev, char in zip(previous, _JOINER_STREAM))
    assert spoken == _JOINER_STREAM_SPOKEN
    assert remove_unspeechable(_JOINER_STREAM) == _JOINER_STREAM_SPOKEN


@pytest.mark.parametrize("emoji", ["❤️", "1️⃣", "👨‍👩‍👧", "👍🏽"])
def test_remove_unspeechable_leaves_no_invisible_emoji_parts(emoji) -> None:
    assert remove_unspeechable(f"हाँ {emoji} ठीक") in ("हाँ  ठीक", "हाँ 1 ठीक")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("मैं ठीक हूँ। आप कैसे हैं? चलो।", ["मैं ठीक हूँ।", "आप कैसे हैं?", "चलो।"]),
        ("আমি ভালো আছি। আপনি কেমন আছেন?", ["আমি ভালো আছি।", "আপনি কেমন আছেন?"]),
        ("میں ٹھیک ہوں۔ آپ کیسے ہیں؟ شکریہ", ["میں ٹھیک ہوں۔", "آپ کیسے ہیں؟", "شکریہ"]),
        ("நான் நலம். நீங்கள் எப்படி?", ["நான் நலம்.", "நீங்கள் எப்படி?"]),
        ('उसने कहा "मैं ठीक हूँ।" फिर गया।', ['उसने कहा "मैं ठीक हूँ।"', "फिर गया।"]),
        ("श्लोक॥ अगला", ["श्लोक॥", "अगला"]),
        ("मैं ठीक हूँ।", ["मैं ठीक हूँ।"]),
        ("मैं ठीक हूँ। ", ["मैं ठीक हूँ। "]),
        ("Hello there. How are you?", ["Hello there.", "How are you?"]),
        ("", []),
    ],
)
def test_split_sentences(text, expected) -> None:
    assert split_sentences(text) == expected


@pytest.mark.parametrize("delta_pattern", [r"\S+\s*", r"\s*\S+", r"(?s:.)"])
@pytest.mark.parametrize(
    "text",
    ["Hello there. How are you? Good.", "मैं ठीक हूँ। आप कैसे हैं? चलो।", "வணக்கம். நீங்கள் எப்படி? நன்றி."],
)
def test_split_sentences_keeps_word_spacing_across_stream_deltas(text, delta_pattern) -> None:
    """Deltas may carry the space before or after a word; neither may glue words together."""
    complete: list[str] = []
    pending = ""
    for delta in re.findall(delta_pattern, text):
        sentences = split_sentences(pending + delta)
        complete += sentences[:-1]
        pending = sentences[-1]
    assert " ".join([*complete, pending]) == text
    assert len(complete) == 2


# --- language name coverage ---------------------------------------------------------------
#
# A language code with no entry in WHISPER_LANGUAGE_TO_LLM_LANGUAGE resolves to a `None`
# language name, and both LLM backends gate the prompt on it:
#
#     if lang_name and self.enable_lang_prompt:
#         active_chat.add_item(make_user_message(f"Please reply to my message in {lang_name}."))
#
# so `--enable_lang_prompt` silently emits nothing for that language. Parakeet TDT is the
# default STT and reports 25 languages, of which only 8 overlapped the original 12-entry map.

# Modules that declare a SUPPORTED_LANGUAGES list of codes they can report.
_STT_HANDLER_MODULES = [
    "speech_to_speech.STT.parakeet_tdt_handler",
    "speech_to_speech.STT.whisper_stt_handler",
    "speech_to_speech.STT.mlx_audio_whisper_handler",
    "speech_to_speech.STT.lightning_whisper_mlx_handler",
]

# These have no optional top-level dependency, so a skip here means something is wrong
# rather than merely uninstalled.
_ALWAYS_IMPORTABLE = {
    "speech_to_speech.STT.parakeet_tdt_handler",
    "speech_to_speech.STT.whisper_stt_handler",
    "speech_to_speech.STT.mlx_audio_whisper_handler",
}


def _supported_languages(module_name):
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    return list(module.SUPPORTED_LANGUAGES)


@pytest.mark.parametrize("module_name", _STT_HANDLER_MODULES)
def test_every_stt_language_has_an_llm_language_name(module_name):
    """Any language a bundled STT backend can report must be nameable for the prompt."""
    languages = _supported_languages(module_name)
    if languages is None:
        if module_name in _ALWAYS_IMPORTABLE:
            pytest.fail(f"{module_name} should be importable without optional extras")
        pytest.skip(f"{module_name} requires an optional dependency")

    missing = sorted(code for code in languages if code not in WHISPER_LANGUAGE_TO_LLM_LANGUAGE)
    assert missing == [], (
        f"{module_name} can report {missing}, which have no entry in "
        f"WHISPER_LANGUAGE_TO_LLM_LANGUAGE, so --enable_lang_prompt would emit no "
        f"instruction for them"
    )


def test_parakeet_default_stt_is_fully_covered():
    """Explicit guard for the default backend, independent of the parametrized sweep."""
    parakeet = importlib.import_module("speech_to_speech.STT.parakeet_tdt_handler")

    assert len(parakeet.SUPPORTED_LANGUAGES) == 25
    assert set(parakeet.SUPPORTED_LANGUAGES) <= set(WHISPER_LANGUAGE_TO_LLM_LANGUAGE)


def test_language_names_are_lowercase_and_non_empty():
    """The name is interpolated mid-sentence, so it must read as lowercase prose."""
    for code, name in WHISPER_LANGUAGE_TO_LLM_LANGUAGE.items():
        assert name and name == name.lower(), f"{code} -> {name!r}"
        assert name.isalpha(), f"{code} -> {name!r}"


# --- resolve_auto_language ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("sv", ("sv", "swedish")),
        ("sv-auto", ("sv", "swedish")),
        ("ru-auto", ("ru", "russian")),
        ("no-auto", ("no", "norwegian")),
        ("lt", ("lt", "lithuanian")),
        ("en-auto", ("en", "english")),
    ],
)
def test_resolve_auto_language_names_parakeet_languages(code, expected):
    assert resolve_auto_language(code) == expected


@pytest.mark.parametrize("code", [None, ""])
def test_resolve_auto_language_passes_through_empty_codes(code):
    assert resolve_auto_language(code) == (code, None)


def test_resolve_auto_language_returns_no_name_for_unknown_code():
    """Unknown codes still round-trip the code, they just cannot be named."""
    assert resolve_auto_language("xx-auto") == ("xx", None)
