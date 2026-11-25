import sys
from queue import Queue
from threading import Event

import httpx
import numpy as np
import pytest

from speech_to_speech.backend_registry import TTS_BACKENDS
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput
from speech_to_speech.s2s_pipeline import parse_arguments
from speech_to_speech.TTS import svara_tts_handler as svara
from speech_to_speech.TTS.svara_text import (
    IndicNormalizer,
    clean_text,
    detect_language,
    gender_from_voice,
    has_speech,
    speaker_name,
    split_long_text,
)

# --- text ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "hint", "expected"),
    [
        ("नमस्ते", None, "hi"),
        ("नमस्कार", "mr-auto", "mr"),
        ("नमस्ते", "ta", "hi"),  # The script wins over the language the user spoke.
        ("வணக்கம்", None, "ta"),
        ("నమస్కారం", None, "te"),
        ("ನಮಸ್ಕಾರ", None, "kn"),
        ("നമസ്കാരം", None, "ml"),
        ("નમસ્તે", None, "gu"),
        ("ਸਤ ਸ੍ਰੀ ਅਕਾਲ", None, "pa"),
        ("নমস্কার", None, "bn"),
        ("মোৰ নাম", None, "as"),
        ("Hello there", "hi-auto", "en"),
        ("मुझे ISRO के बारे में बताओ", None, "hi"),
        ("25", "ta-auto", "ta"),
        ("25", None, "hi"),
        ("ਠੀਕ ਹੈ।", None, "pa"),  # The danda is in the Devanagari block but is shared.
    ],
)
def test_detect_language(text, hint, expected):
    assert detect_language(text, hint=hint, default="hi") == expected


def test_detect_language_ignores_unsupported_hint():
    assert detect_language("...", hint="fr", default="hi") == "hi"


def test_speaker_and_gender():
    assert speaker_name("hi", "female") == "Hindi (Female)"
    assert gender_from_voice("Female", "male") == "female"
    assert gender_from_voice("Male", "female") == "male"
    assert gender_from_voice("Aiden", "female") == "female"
    assert gender_from_voice(None, "male") == "male"


def test_clean_text_removes_control_tokens_and_markup():
    assert clean_text("**Hi** <custom_token_2><|audio|> there") == "Hi there"
    assert "<" not in clean_text("a < b")
    assert not has_speech("… !")


def test_split_long_text_prefers_sentence_ends_and_keeps_all_words():
    text = "पहला वाक्य यहाँ है। दूसरा वाक्य यहाँ है। तीसरा वाक्य यहाँ है।"
    chunks = split_long_text(text, 30)
    assert all(len(chunk) <= 31 for chunk in chunks)
    assert " ".join(chunks) == text
    assert split_long_text("x" * 25, 10) == ["x" * 10, "x" * 10, "x" * 5]


class FakeNormalizer:
    """Reads commas aloud and fails on one word, like real grammars sometimes do."""

    def normalize(self, text):
        if "boom" in text:
            raise RuntimeError("no path")
        return text.replace("25", "पच्चीस").replace(",", " COMMA ")


def _normalizer(**languages):
    normalizer = object.__new__(IndicNormalizer)
    normalizer._lock = __import__("threading").Lock()
    normalizer._normalizers = languages
    normalizer._pending = set()
    return normalizer


def test_normalizer_never_sees_clause_punctuation():
    normalizer = _normalizer(hi=FakeNormalizer())
    assert normalizer.normalize("मैं 25, आप 25?", "hi") == "मैं पच्चीस, आप पच्चीस?"


def test_normalizer_joins_digit_groups_but_not_lists():
    class Echo:
        def normalize(self, text):
            return text

    normalizer = _normalizer(hi=Echo(), en=Echo())
    assert normalizer.normalize("₹1,25,000 और 1, 2", "hi") == "₹125000 और 1, 2"
    assert normalizer.normalize("It is ₹1,250.", "en") == "It is 1250 rupees."


def test_normalizer_failure_keeps_the_original_words():
    normalizer = _normalizer(hi=FakeNormalizer())
    assert normalizer.normalize("boom, 25", "hi") == "boom, पच्चीस"


def test_unavailable_normalizer_returns_the_text_unchanged():
    assert _normalizer(ta=None).normalize("25 ரூபாய்", "ta") == "25 ரூபாய்"


# --- audio --------------------------------------------------------------------------------


def _tokens(frames):
    return [
        svara.FIRST_AUDIO_TOKEN + position * svara.CODES_PER_LEVEL + code
        for frame in frames
        for position, code in enumerate(frame)
    ]


def test_frame_assembler_groups_codes_and_drops_misplaced_tokens():
    assembler = svara.FrameAssembler()
    tokens = _tokens([[1, 2, 3, 4, 5, 6, 7]])
    tokens.insert(3, svara.FIRST_AUDIO_TOKEN)  # Belongs to position 0, not 3.
    tokens.insert(0, 42)  # Not an audio token.
    assert [assembler.feed(token) for token in tokens].count(True) == 1
    assert assembler.frames == [[1, 2, 3, 4, 5, 6, 7]]


@pytest.mark.parametrize("sizes", [[2048] * 6, [4096, 2048, 100, 2048], [10], [7, 5, 3000]])
def test_stream_resampler_matches_one_shot_resampling(sizes):
    from scipy.signal import resample_poly

    total = sum(sizes)
    audio = np.sin(2 * np.pi * 440 * np.arange(total) / 24000).astype(np.float32)
    resampler, pieces, start = svara.StreamResampler(), [], 0
    for size in sizes:
        pieces.append(resampler.feed(audio[start : start + size]))
        start += size
    pieces.append(resampler.flush())
    result = np.concatenate(pieces)
    expected = resample_poly(np.pad(audio, (0, -total % 3)), up=2, down=3)
    assert len(result) == len(expected)
    np.testing.assert_allclose(result, expected, atol=2e-3)


@pytest.mark.parametrize("level", [0.02, 0.1, 0.6])
def test_leveler_brings_quiet_and_loud_sentences_to_the_same_level(level):
    audio = (level * np.sin(2 * np.pi * 220 * np.arange(2048 * 20) / 24000)).astype(np.float32)
    leveler = svara.LoudnessLeveler()
    pieces = [leveler.feed(chunk) for chunk in np.split(audio, 20)] + [leveler.flush()]
    result = np.concatenate(pieces)
    assert len(result) == len(audio)
    assert np.abs(result).max() <= 1.0
    assert 0.05 < np.sqrt(np.mean(result**2)) < 0.2


def test_leveler_emits_sentences_shorter_than_the_look_ahead():
    leveler = svara.LoudnessLeveler()
    assert len(leveler.feed(np.full(500, 0.1, dtype=np.float32))) == 0
    assert len(leveler.flush()) == 500


class FakeCodec:
    def decode(self, levels):
        # Each frame becomes samples holding its first code, so that order and loss are visible.
        return levels[0].reshape(-1, 1).repeat(1, svara.SAMPLES_PER_FRAME).reshape(1, 1, -1).float()


def _handler(handle_request=None, **attributes):
    handler = object.__new__(svara.SvaraTTSHandler)
    handler.stop_event, handler.should_listen = Event(), Event()
    handler.queue_in, handler.queue_out = Queue(), Queue()
    handler.gender, handler.default_language = "female", "hi"
    handler.sampling, handler.max_chars = {"temperature": 0.75}, 200
    handler.prebuffer_samples, handler.blocksize = 4000, 512
    handler.cancel_scope = handler.speculative_turns = handler.normalizer = None
    handler.device, handler.codec = "cpu", FakeCodec()
    if handle_request:
        handler.client = httpx.Client(base_url="http://svara", transport=httpx.MockTransport(handle_request))
    for name, value in attributes.items():
        setattr(handler, name, value)
    return handler


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 9])
def test_decode_stream_emits_every_frame_once_in_order(count):
    frames = [[n + 1] * 7 for n in range(count)]
    audio = np.concatenate(list(_handler()._decode_stream(iter(_tokens(frames)))))
    assert len(audio) == count * svara.SAMPLES_PER_FRAME
    assert audio[:: svara.SAMPLES_PER_FRAME].tolist() == [n + 1 for n in range(count)]


def _server(requests, frames=8):
    import json

    def handle(request):
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [11, 12]})
        tokens = _tokens([[5] * 7] * frames)
        lines = [f"data: {json.dumps({'tokens': [token], 'stop': False})}\n\n" for token in tokens]
        lines.append(f"data: {json.dumps({'tokens': [svara.END_OF_SPEECH], 'stop': False})}\n\n")
        lines.append(f"data: {json.dumps({'tokens': _tokens([[9] * 7]), 'stop': True})}\n\n")
        return httpx.Response(200, text="".join(lines))

    return handle


def test_process_builds_the_prompt_and_stops_at_end_of_speech():
    requests = []
    blocks = list(_handler(_server(requests)).process(TTSInput(text="नमस्ते <happy>", language_code="hi-auto")))
    tokenize, completion = requests
    assert tokenize[1] == {"content": "Hindi (Female): नमस्ते", "add_special": False, "parse_special": False}
    assert completion[1]["prompt"] == [128259, 128000, 11, 12, 128009, 128260, 128261, 128257]
    assert completion[1]["return_tokens"] is True
    assert {len(block) for block in blocks} == {512}
    assert all(block.dtype == np.int16 for block in blocks)
    samples = 8 * svara.SAMPLES_PER_FRAME * 2 // 3
    assert samples <= sum(map(len, blocks)) < samples + 1024


def test_process_skips_text_without_speech():
    requests = []
    assert list(_handler(_server(requests)).process(TTSInput(text="…"))) == []
    assert requests == []


def test_process_stops_when_interrupted():
    requests = []
    scope = CancelScope()
    blocks = []
    for block in _handler(_server(requests, frames=60), cancel_scope=scope).process(TTSInput(text="नमस्ते")):
        blocks.append(block)
        if len(blocks) == 1:
            scope.cancel()
    full = 60 * svara.SAMPLES_PER_FRAME * 2 // 3 // 512
    assert len(blocks) < full // 2


def test_server_errors_do_not_kill_the_handler():
    handler = _handler(lambda request: httpx.Response(500))
    assert [b for b in handler.process(TTSInput(text="नमस्ते")) if np.any(b)] == []


def test_end_of_response_is_forwarded():
    assert list(_handler().process(EndOfResponse())) == [AUDIO_RESPONSE_DONE]


# --- registration -------------------------------------------------------------------------


def test_svara_is_registered_and_lazy():
    sys.modules.pop("snac", None)
    assert TTS_BACKENDS["svara"].required_extra == "svara"
    assert "snac" not in sys.modules
    config = parse_arguments(["--tts", "svara", "--svara_tts_gender", "male"]).tts_backend.config
    assert config["gender"] == "male"
    assert config["default_language"] == "hi"
