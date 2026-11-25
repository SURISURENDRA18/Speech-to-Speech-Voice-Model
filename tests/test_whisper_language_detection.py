from queue import Queue
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from speech_to_speech.pipeline.messages import VADAudio
from speech_to_speech.STT.whisper_stt_handler import SUPPORTED_LANGUAGES, WhisperSTTHandler

LANG_IDS = {"<|en|>": 1, "<|hi|>": 2, "<|ko|>": 3, "<|ta|>": 4}


class FakeModel:
    def __init__(self, scores, lang_to_id=LANG_IDS):
        self.generation_config = SimpleNamespace(lang_to_id=lang_to_id, decoder_start_token_id=0)
        self.scores = scores
        self.generate_calls = []

    def __call__(self, input_features, decoder_input_ids):
        return SimpleNamespace(logits=torch.tensor([[self.scores]]))

    def generate(self, input_features, **kwargs):
        self.generate_calls.append(kwargs)
        return torch.tensor([[7, 8]])


def _handler(model, language="auto", allowed=None):
    handler = object.__new__(WhisperSTTHandler)
    handler.stop_event, handler.queue_in, handler.queue_out = Event(), Queue(), Queue()
    handler.model = model
    handler.start_language, handler.last_language = language, None
    handler.gen_kwargs = {} if language == "auto" else {"language": language}
    handler.allowed_languages = allowed or SUPPORTED_LANGUAGES
    handler.prepare_model_inputs = lambda audio: torch.zeros(1, 80, 10)
    handler.processor = SimpleNamespace(batch_decode=lambda *args, **kwargs: ["text"])
    return handler


def _transcribe(handler):
    return list(handler.process(VADAudio(audio=np.zeros(160, dtype=np.float32))))[0]


def test_auto_uses_the_detected_language_and_transcribes_once():
    model = FakeModel([0.0, 0.1, 0.9, 0.2, 0.3])
    assert _transcribe(_handler(model)).language_code == "hi-auto"
    assert [call["language"] for call in model.generate_calls] == ["hi"]


def test_detection_is_limited_to_the_allowed_languages():
    # Korean scores highest, but is not allowed.
    model = FakeModel([0.0, 0.1, 0.5, 0.9, 0.3])
    assert _transcribe(_handler(model, allowed=["hi", "en", "ta"])).language_code == "hi-auto"


def test_fixed_language_skips_detection():
    model = FakeModel([0.0, 0.1, 0.5, 0.9, 0.3])
    assert _transcribe(_handler(model, language="ta")).language_code == "ta"
    assert model.generate_calls == [{"language": "ta"}]


@pytest.mark.parametrize("lang_to_id", [None, {}])
def test_english_only_models_have_no_language(lang_to_id):
    assert _transcribe(_handler(FakeModel([0.0], lang_to_id))).language_code is None
