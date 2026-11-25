"""
Svara TTS handler

Runs Kenpath's svara-tts-v1, a 3B speech model for 18 Indian languages and Indian English.
A llama.cpp server generates audio tokens and the SNAC codec turns them into sound. Token offsets and
sampling defaults follow https://github.com/Kenpath/svara-tts-inference (Apache License 2.0). The
prompt layout follows Kenpath's demo Space instead: the inference repository adds an <|audio|> token
with which v1 speaks its own speaker label and sometimes runs on.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from threading import Event
from typing import Any

import httpx
import numpy as np
from rich.console import Console

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.TTS.svara_text import (
    SVARA_LANGUAGES,
    IndicNormalizer,
    clean_text,
    detect_language,
    gender_from_voice,
    has_speech,
    speaker_name,
    split_long_text,
)

logger = logging.getLogger(__name__)
console = Console()

SVARA_SAMPLE_RATE = 24000
PIPELINE_SAMPLE_RATE = 16000

BOS = 128000
END_OF_TURN = 128009
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
START_OF_HUMAN = 128259
END_OF_HUMAN = 128260
START_OF_AI = 128261
FIRST_AUDIO_TOKEN = 128266
CODES_PER_LEVEL = 4096

TOKENS_PER_FRAME = 7
SAMPLES_PER_FRAME = 2048
# The codec needs neighbouring frames, so each frame is decoded with one before and two after it.
WINDOW_FRAMES = 4
FRAMES_BEFORE = 1

SERVER_WAIT_S = 180.0


class FrameAssembler:
    """Group audio tokens into 7-code frames, dropping tokens that break the frame layout."""

    def __init__(self) -> None:
        self.frames: list[list[int]] = []
        self._current: list[int] = []

    def feed(self, token: int) -> bool:
        """Add a token; return True when it completes a frame."""
        code = token - FIRST_AUDIO_TOKEN - len(self._current) * CODES_PER_LEVEL
        if not 0 <= code < CODES_PER_LEVEL:
            return False
        self._current.append(code)
        if len(self._current) < TOKENS_PER_FRAME:
            return False
        self.frames.append(self._current)
        self._current = []
        return True


class StreamResampler:
    """Resample 24 kHz audio to 16 kHz chunk by chunk without clicks at chunk edges."""

    # Samples of context on each side of a chunk. A multiple of 3 keeps the 2/3 ratio exact.
    MARGIN = 96

    def __init__(self) -> None:
        self._left = np.zeros(self.MARGIN, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)

    def _emit(self, body: np.ndarray, right: np.ndarray) -> np.ndarray:
        from scipy.signal import resample_poly

        padded = resample_poly(np.concatenate([self._left, body, right]), up=2, down=3)
        start = self.MARGIN * 2 // 3
        self._left = np.concatenate([self._left, body])[-self.MARGIN :]
        return np.asarray(padded[start : start + len(body) * 2 // 3], dtype=np.float32)

    def feed(self, audio: np.ndarray) -> np.ndarray:
        self._pending = np.concatenate([self._pending, audio])
        size = (len(self._pending) - self.MARGIN) // 3 * 3
        if size <= 0:
            return np.zeros(0, dtype=np.float32)
        body, self._pending = self._pending[:size], self._pending[size:]
        return self._emit(body, self._pending[: self.MARGIN])

    def flush(self) -> np.ndarray:
        body = self._pending
        self._pending = np.zeros(0, dtype=np.float32)
        if not len(body):
            return np.zeros(0, dtype=np.float32)
        body = np.pad(body, (0, -len(body) % 3))
        return self._emit(body, np.zeros(self.MARGIN, dtype=np.float32))


class LoudnessLeveler:
    """Bring each sentence to a steady level, because Svara's loudness changes from one to the next.

    The gain follows the recent level of the voice. It rises gradually, so that breaths and quiet
    onsets are not blown up, and falls at once, so that the first loud syllable is not distorted.
    """

    TARGET_RMS = 0.12
    MIN_GAIN, MAX_GAIN = 0.4, 10.0
    FIRST_MAX_GAIN = 4.0
    MAX_RISE = 1.3
    PEAK = 0.95
    VOICED_LEVEL = 0.005
    MEMORY = 0.85
    LOOKAHEAD = 2 * SAMPLES_PER_FRAME

    def __init__(self) -> None:
        self._energy = 0.0
        self._count = 0.0
        self._gain: float | None = None
        self._held: list[np.ndarray] = []

    def _measure(self, audio: np.ndarray) -> None:
        voiced = audio[np.abs(audio) > self.VOICED_LEVEL]
        # Silence says nothing about the level of the voice, so it does not age the estimate either.
        if len(voiced) < len(audio) // 10:
            return
        memory = self.MEMORY ** (len(audio) / SAMPLES_PER_FRAME)
        self._energy = self._energy * memory + float(np.sum(voiced**2))
        self._count = self._count * memory + len(voiced)

    def _apply(self, audio: np.ndarray) -> np.ndarray:
        wanted = self._gain or 1.0
        if self._count:
            rms = (self._energy / self._count) ** 0.5
            wanted = float(np.clip(self.TARGET_RMS / rms, self.MIN_GAIN, self.MAX_GAIN))
        peak = float(np.max(np.abs(audio), initial=0.0))
        if peak * wanted > self.PEAK:
            wanted = self.PEAK / peak
        previous = self._gain if self._gain is not None else min(wanted, self.FIRST_MAX_GAIN)
        self._gain = min(wanted, previous * self.MAX_RISE)
        # Fall over 2 ms, before a loud onset is heard; rise over the whole piece.
        ramp = min(len(audio), 48) if self._gain < previous else len(audio)
        gain = np.full(len(audio), self._gain, dtype=np.float32)
        gain[:ramp] = np.linspace(previous, self._gain, ramp, dtype=np.float32)
        return np.asarray(np.clip(audio * gain, -1.0, 1.0), dtype=np.float32)

    def feed(self, audio: np.ndarray) -> np.ndarray:
        self._measure(audio)
        if self._gain is not None:
            return self._apply(audio)
        self._held.append(audio)
        if sum(map(len, self._held)) < self.LOOKAHEAD:
            return np.zeros(0, dtype=np.float32)
        return self.flush()

    def flush(self) -> np.ndarray:
        if not self._held:
            return np.zeros(0, dtype=np.float32)
        held, self._held = np.concatenate(self._held), []
        return self._apply(held)


def to_pcm16(audio: np.ndarray) -> np.ndarray:
    return np.asarray(np.clip(audio, -1.0, 1.0) * 32767, dtype=np.int16)


class SvaraTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """Speak in 18 Indian languages and Indian English, following the language of each sentence."""

    def setup(
        self,
        should_listen: Event,
        server_url: str = "http://127.0.0.1:8081",
        codec_model: str = "hubertsiuzdak/snac_24khz",
        device: str = "auto",
        gender: str = "female",
        default_language: str = "hi",
        normalize: bool = True,
        normalizer_languages: str = "hi,en",
        normalizer_cache_dir: str = "~/.cache/speech-to-speech/indic-text-normalization",
        temperature: float = 0.75,
        top_p: float = 0.9,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        max_new_tokens: int = 2048,
        max_chars: int = 200,
        prebuffer_ms: int = 250,
        blocksize: int = 512,
        gen_kwargs: dict[str, Any] | None = None,  # For compatibility with the pipeline, not used.
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
    ) -> None:
        if default_language not in SVARA_LANGUAGES:
            raise ValueError(f"Svara does not support {default_language!r}; choose one of {sorted(SVARA_LANGUAGES)}.")
        preload = [code.strip() for code in normalizer_languages.split(",") if code.strip()]
        unknown = [code for code in preload if code not in SVARA_LANGUAGES]
        if unknown:
            raise ValueError(f"Unknown normalizer languages {unknown}; choose from {sorted(SVARA_LANGUAGES)}.")

        self.should_listen = should_listen
        self.gender = gender
        self.default_language = default_language
        self.sampling = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": 0.0,
            "repeat_penalty": repetition_penalty,
            # Penalize over the whole generation, like Kenpath's vLLM setup.
            "repeat_last_n": 4096,
            "n_predict": max_new_tokens,
        }
        self.max_chars = max_chars
        self.prebuffer_samples = PIPELINE_SAMPLE_RATE * prebuffer_ms // 1000
        self.blocksize = blocksize
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns

        self.normalizer: IndicNormalizer | None = None
        if normalize:
            self.normalizer = IndicNormalizer(cache_dir=normalizer_cache_dir, preload=preload)

        self._load_codec(codec_model, device)
        self.client = httpx.Client(base_url=server_url, timeout=httpx.Timeout(30.0, connect=5.0))
        self._wait_for_server(server_url)
        self.warmup()

    def _load_codec(self, codec_model: str, device: str) -> None:
        import torch
        from snac import SNAC

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        logger.info("Loading SNAC codec %s on %s", codec_model, device)
        self.codec = SNAC.from_pretrained(codec_model).eval().to(device)

    def _wait_for_server(self, server_url: str) -> None:
        deadline = time.monotonic() + SERVER_WAIT_S
        while True:
            try:
                if self.client.get("/health").status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline or self.stop_event.is_set():
                raise RuntimeError(f"The Svara llama.cpp server at {server_url} is not ready.")
            time.sleep(1)

    def warmup(self) -> None:
        for _ in self._synthesize("नमस्ते।", speaker_name("hi", self.gender), lambda: False):
            pass
        logger.info("%s warmed up", self.__class__.__name__)

    @property
    def min_time_to_debug(self) -> float:
        return 0.1

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        speculative_turns = self.speculative_turns
        if isinstance(tts_input, EndOfResponse):
            if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
                tts_input.turn_id,
                tts_input.turn_revision,
            ):
                if tts_input.response_key is None:
                    return
                tts_input.cleanup_only = True
            yield AUDIO_RESPONSE_DONE
            return

        if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
            tts_input.turn_id,
            tts_input.turn_revision,
        ):
            logger.debug("Dropping stale TTS input for turn=%s rev=%s", tts_input.turn_id, tts_input.turn_revision)
            return
        if speculative_turns:
            speculative_turns.commit(tts_input.turn_id, tts_input.turn_revision)

        scope = self.cancel_scope
        generation = scope.generation if scope else None

        def cancelled() -> bool:
            return self.stop_event.is_set() or (
                scope is not None and generation is not None and scope.is_stale(generation)
            )

        console.print(f"[green]ASSISTANT: {tts_input.text}")
        text = clean_text(tts_input.text)
        if not has_speech(text):
            return
        language = detect_language(text, hint=tts_input.language_code, default=self.default_language)
        speaker = speaker_name(language, gender_from_voice(self._session_voice(tts_input), self.gender))
        spoken = self.normalizer.normalize(text, language) if self.normalizer else text
        if spoken != text:
            logger.debug("Normalized (%s): %r -> %r", language, text, spoken)

        for chunk in split_long_text(spoken, self.max_chars):
            yield from self._blocks(self._synthesize(chunk, speaker, cancelled))
            if cancelled():
                logger.info("TTS generation cancelled (interruption)")
                return

    @staticmethod
    def _session_voice(tts_input: Any) -> str | None:
        response = tts_input.response
        if response and response.audio and response.audio.output and response.audio.output.voice:
            return str(response.audio.output.voice)
        if tts_input.runtime_config:
            audio = tts_input.runtime_config.session.audio
            output = audio.output if audio is not None else None
            if output is not None and output.voice:
                return str(output.voice)
        return None

    def _blocks(self, pieces: Iterator[np.ndarray]) -> Iterator[np.ndarray]:
        """Regroup audio into fixed-size blocks, holding back the first moments as a buffer."""
        buffered = np.zeros(0, dtype=np.int16)
        started = False
        for piece in pieces:
            buffered = np.concatenate([buffered, piece])
            if not started and len(buffered) < self.prebuffer_samples:
                continue
            started = True
            while len(buffered) >= self.blocksize:
                yield buffered[: self.blocksize]
                buffered = buffered[self.blocksize :]
        for start in range(0, len(buffered), self.blocksize):
            block = buffered[start : start + self.blocksize]
            yield np.pad(block, (0, self.blocksize - len(block)))

    def _synthesize(self, text: str, speaker: str, cancelled: Any) -> Iterator[np.ndarray]:
        """Yield 16 kHz int16 audio for ``text`` while it is being generated."""
        start = time.perf_counter()
        first_audio: float | None = None
        leveler = LoudnessLeveler()
        resampler = StreamResampler()
        samples = 0

        for audio in self._decode_stream(self._generate(text, speaker, cancelled)):
            if cancelled():
                return
            if first_audio is None:
                first_audio = time.perf_counter() - start
            samples += len(audio)
            yield to_pcm16(resampler.feed(leveler.feed(audio)))
        yield to_pcm16(resampler.feed(leveler.flush()))
        yield to_pcm16(resampler.flush())

        if not samples:
            logger.warning("Svara produced no audio for %r", text)
            return
        elapsed = time.perf_counter() - start
        seconds = samples / SVARA_SAMPLE_RATE
        logger.info(
            "Svara %s: first audio %.0f ms, %.2fs of audio in %.2fs (%.2fx real time)",
            speaker,
            (first_audio or 0.0) * 1000,
            seconds,
            elapsed,
            seconds / elapsed if elapsed else 0.0,
        )

    def _generate(self, text: str, speaker: str, cancelled: Any) -> Iterator[int]:
        """Stream audio tokens from the llama.cpp server."""
        try:
            tokenized = self.client.post(
                "/tokenize", json={"content": f"{speaker}: {text}", "add_special": False, "parse_special": False}
            )
            tokenized.raise_for_status()
            prompt = [
                START_OF_HUMAN,
                BOS,
                *tokenized.json()["tokens"],
                END_OF_TURN,
                END_OF_HUMAN,
                START_OF_AI,
                START_OF_SPEECH,
            ]
            body = {"prompt": prompt, "stream": True, "return_tokens": True, "cache_prompt": False, **self.sampling}
            # Leaving this block closes the connection, which also stops generation on the server.
            with self.client.stream("POST", "/completion", json=body) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if cancelled():
                        return
                    if not line.startswith("data:"):
                        continue
                    data = json.loads(line[5:])
                    for token in data.get("tokens") or []:
                        if token == END_OF_SPEECH:
                            return
                        yield int(token)
                    if data.get("stop"):
                        return
        except httpx.HTTPError as e:
            logger.error("Svara llama.cpp server request failed: %s: %s", type(e).__name__, e)

    def _decode_stream(self, tokens: Iterator[int]) -> Iterator[np.ndarray]:
        """Decode frames as they arrive. Each frame is decoded once, with context on both sides."""
        assembler = FrameAssembler()
        frames = assembler.frames
        emitted = 0
        for token in tokens:
            if not assembler.feed(token) or len(frames) < WINDOW_FRAMES:
                continue
            # Window = one frame before, the frame being emitted, and two after it. The first window
            # also covers the opening frame, which would otherwise be lost.
            window = self._decode(frames[-WINDOW_FRAMES:])
            begin = 0 if emitted == 0 else FRAMES_BEFORE * SAMPLES_PER_FRAME
            yield window[begin : (FRAMES_BEFORE + 1) * SAMPLES_PER_FRAME]
            emitted = len(frames) - WINDOW_FRAMES + FRAMES_BEFORE + 1

        if emitted < len(frames):
            context = min(emitted, FRAMES_BEFORE)
            tail = self._decode(frames[emitted - context :])
            yield tail[context * SAMPLES_PER_FRAME :]

    def _decode(self, frames: list[list[int]]) -> np.ndarray:
        import torch

        codes = torch.tensor(frames, dtype=torch.int32, device=self.device)
        levels = [
            codes[:, 0].reshape(1, -1),
            codes[:, [1, 4]].reshape(1, -1),
            codes[:, [2, 3, 5, 6]].reshape(1, -1),
        ]
        with torch.inference_mode():
            audio = self.codec.decode(levels)
        return np.asarray(audio.float().cpu().numpy().reshape(-1), dtype=np.float32)

    def on_session_end(self) -> None:
        logger.debug("Svara TTS session state reset")

    def cleanup(self) -> None:
        self.client.close()
