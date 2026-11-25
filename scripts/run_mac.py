"""Mac launcher for speech-to-speech: patches at runtime only, no project files changed."""
import sys
import mlx.core as mx
from speech_to_speech.utils import mlx_lock

# 1. Never enter the GPU section without the lock: wait instead of timing out.
_orig = mlx_lock.MLXLockContext.__init__
def _init(self, handler_name="Unknown", timeout=None):
    if timeout is not None and timeout >= 1.0:
        timeout = None
    _orig(self, handler_name, timeout)
mlx_lock.MLXLockContext.__init__ = _init

# 2. Keep MLX from hoarding freed GPU buffers on an 8GB machine.
mx.set_cache_limit(256 * 1024 * 1024)

# 3. The browser picker sends Qwen voice names; Kokoro only knows its own.
from speech_to_speech.TTS.kokoro_handler import KokoroTTSHandler
_proc = KokoroTTSHandler.process
_VOICES = {"aiden": "bm_fable", "ryan": "bm_george", "dylan": "bm_daniel", "eric": "bm_lewis",
           "uncle_fu": "bm_george", "serena": "bf_emma", "vivian": "bf_isabella",
           "sohee": "bf_alice", "ono_anna": "bf_lily"}
def _map(cfg_audio):
    out = cfg_audio.output if cfg_audio is not None else None
    if out is not None and out.voice:
        v = str(out.voice).lower()
        if v in _VOICES:
            out.voice = _VOICES[v]
def process(self, tts_input):
    rc = getattr(tts_input, "runtime_config", None)
    if rc is not None:
        _map(rc.session.audio)
    resp = getattr(tts_input, "response", None)
    if resp is not None and getattr(resp, "audio", None):
        _map(resp.audio)
    text = getattr(tts_input, "text", None)
    if isinstance(text, str):
        tts_input.text = text.replace("*", "").replace("#", "").replace("`", "")
    yield from _proc(self, tts_input)
KokoroTTSHandler.process = process

# 4. Reuse the already-processed part of the conversation instead of re-reading
#    the whole prompt every turn, and sample lightly so replies do not repeat.
from mlx_lm import stream_generate as _stream
from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from speech_to_speech.LLM import language_model as _lm

_sampler = make_sampler(temp=0.6, top_p=0.9)
_procs = make_logits_processors(repetition_penalty=1.05, repetition_context_size=64)
_state = {"model": None, "cache": None, "fed": []}


def _cached_stream(model, tokenizer, prompt, max_tokens=256, **kw):
    tokens = tokenizer.encode(prompt, add_special_tokens=False) if isinstance(prompt, str) else list(prompt)
    cache, fed, common = _state["cache"], _state["fed"], 0
    if _state["model"] is model and cache is not None and can_trim_prompt_cache(cache):
        limit = min(len(fed), len(tokens) - 1)
        while common < limit and fed[common] == tokens[common]:
            common += 1
        trim_prompt_cache(cache, cache[0].offset - common)
        if cache[0].offset != common:
            cache = None
    else:
        cache = None
    if cache is None:
        cache, common = make_prompt_cache(model), 0
    _state.update(model=None, cache=None, fed=[])  # only valid again if we finish cleanly
    out = []
    gen = _stream(model, tokenizer, tokens[common:], max_tokens=max_tokens, prompt_cache=cache,
                  sampler=_sampler, logits_processors=_procs, **kw)
    ok = False
    try:
        for r in gen:
            out.append(int(r.token))
            yield r
        ok = True
    except GeneratorExit:
        ok = True
        raise
    finally:
        gen.close()
        if ok:
            seen = tokens + out
            extra = cache[0].offset - len(seen)
            if extra > 0:
                trim_prompt_cache(cache, extra)
            _state.update(model=model, cache=cache, fed=seen[: cache[0].offset])


_lm.mlx_stream_generate = _cached_stream

# 5. Stop Whisper getting stuck repeating one word on noise (it blocked for 50s).
from speech_to_speech.STT.mlx_audio_whisper_handler import MLXAudioWhisperSTTHandler

_setup = MLXAudioWhisperSTTHandler.setup


def _looping(text):
    words = [w.strip(".,!?").lower() for w in text.split()]
    return len(words) >= 8 and len(set(words)) / len(words) < 0.3


def setup(self, *args, **kwargs):
    _setup(self, *args, **kwargs)
    model = self.model

    class _Guarded:
        def __getattr__(self, name):
            return getattr(model, name)

        def generate(self, audio, **kw):
            kw.setdefault("temperature", (0.0, 0.4))
            kw.setdefault("condition_on_previous_text", False)
            kw.setdefault("sample_len", 96)
            result = model.generate(audio, **kw)
            if _looping(getattr(result, "text", "") or ""):
                result.text = ""
            return result

    self.model = _Guarded()


MLXAudioWhisperSTTHandler.setup = setup

from speech_to_speech.cli import main
sys.exit(main())
