from __future__ import annotations

import logging
from copy import copy
from typing import Any, Iterator, Optional

import numpy as np
import torch
from rich.console import Console
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler

logger = logging.getLogger(__name__)
console = Console()

SUPPORTED_LANGUAGES = [
    "en",
    "fr",
    "es",
    "zh",
    "ja",
    "ko",
    "hi",
    "de",
    "pt",
    "pl",
    "it",
    "nl",
    "as",
    "bn",
    "gu",
    "kn",
    "ml",
    "mr",
    "ne",
    "pa",
    "sa",
    "ta",
    "te",
]


class WhisperSTTHandler(BaseSTTHandler):
    """
    Handles the Speech To Text generation using a Whisper model.
    """

    def setup(
        self,
        model_name: str = "distil-whisper/distil-large-v3",
        device: str = "cuda",
        torch_dtype: str = "float16",
        compile_mode: Optional[str] = None,
        language: Optional[str] = None,
        allowed_languages: Optional[str] = None,
        gen_kwargs: dict[str, Any] = {},
    ) -> None:
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        self.compile_mode = compile_mode
        self.gen_kwargs = gen_kwargs
        self.start_language = language
        self.last_language = language if language != "auto" else None
        if self.last_language is not None:
            self.gen_kwargs["language"] = self.last_language

        self.allowed_languages = SUPPORTED_LANGUAGES
        if allowed_languages:
            self.allowed_languages = [code.strip() for code in allowed_languages.split(",") if code.strip()]
            unknown = [code for code in self.allowed_languages if code not in SUPPORTED_LANGUAGES]
            if unknown or not self.allowed_languages:
                raise ValueError(f"Unsupported Whisper languages {unknown}; choose from {SUPPORTED_LANGUAGES}.")

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
        ).to(device)

        # compile
        if self.compile_mode:
            self.model.generation_config.cache_implementation = "static"
            self.model.forward = torch.compile(self.model.forward, mode=self.compile_mode, fullgraph=True)
        self.warmup()

    def prepare_model_inputs(self, audio: np.ndarray) -> torch.Tensor:
        input_features = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        input_features = input_features.to(self.device, dtype=self.torch_dtype)

        return input_features

    def warmup(self) -> None:
        logger.info(f"Warming up {self.__class__.__name__}")

        # 2 warmup steps for no compile or compile mode with CUDA graphs capture
        n_steps = 1 if self.compile_mode == "default" else 2
        dummy_input = torch.randn(
            (1, self.model.config.num_mel_bins, 3000),
            dtype=self.torch_dtype,
            device=self.device,
        )
        if self.compile_mode not in (None, "default"):
            # generating more tokens than previously will trigger CUDA graphs capture
            # one should warmup with a number of generated tokens above max tokens targeted for subsequent generation
            # hence, having min_new_tokens < max_new_tokens in the future doesn't make sense
            warmup_gen_kwargs = {
                "min_new_tokens": self.gen_kwargs["max_new_tokens"],  # Yes, assign max_new_tokens to min_new_tokens
                "max_new_tokens": self.gen_kwargs["max_new_tokens"],
                **self.gen_kwargs,
            }
        else:
            warmup_gen_kwargs = self.gen_kwargs

        if self.device == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()

        for _ in range(n_steps):
            _ = self.model.generate(dummy_input, **warmup_gen_kwargs)

        if self.device == "cuda":
            end_event.record()
            torch.cuda.synchronize()

            logger.info(
                f"{self.__class__.__name__}:  warmed up! time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s"
            )

    def _detect_language(self, input_features: torch.Tensor) -> Optional[str]:
        """Return the most likely of the allowed languages, or None for English-only models.

        Recent transformers versions no longer return the language token from ``generate``, so it is
        read from the model directly. Limiting the choice also stops short or accented speech from
        being taken for an unrelated language.
        """
        lang_to_id = getattr(self.model.generation_config, "lang_to_id", None)
        if not lang_to_id:
            return None
        ids = {code: lang_to_id[f"<|{code}|>"] for code in self.allowed_languages if f"<|{code}|>" in lang_to_id}
        if not ids:
            return None
        start = torch.tensor([[self.model.generation_config.decoder_start_token_id]], device=input_features.device)
        with torch.no_grad():
            logits = self.model(input_features=input_features, decoder_input_ids=start).logits[0, -1]
        return max(ids, key=lambda code: float(logits[ids[code]]))

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        logger.debug("infering whisper...")

        input_features = self.prepare_model_inputs(vad_audio.audio)
        gen_kwargs = copy(self.gen_kwargs)
        language_code = self.start_language if self.start_language != "auto" else None
        if language_code is None:
            language_code = self._detect_language(input_features)
            if language_code is not None:
                gen_kwargs["language"] = language_code
                self.last_language = language_code
        pred_ids = self.model.generate(input_features, **gen_kwargs)
        pred_text = self.processor.batch_decode(pred_ids, skip_special_tokens=True, decode_with_timestamps=False)[0]

        logger.debug("finished whisper inference")
        console.print(f"[yellow]USER: {pred_text}")
        logger.debug(f"Language Code Whisper: {language_code}")

        if self.start_language == "auto" and language_code is not None:
            language_code += "-auto"

        yield Transcription(
            text=pred_text,
            language_code=language_code,
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
            speech_stopped_at_s=vad_audio.created_at_s,
        )
