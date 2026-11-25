from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SvaraTTSHandlerArguments:
    svara_tts_server_url: str = field(
        default="http://127.0.0.1:8081",
        metadata={
            "help": "URL of the llama.cpp server that runs the Svara GGUF model. Start it with "
            "'--override-kv tokenizer.ggml.eos_token_id=int:128258' so that it stops at the end of speech. "
            "Default is 'http://127.0.0.1:8081'."
        },
    )
    svara_tts_codec_model: str = field(
        default="hubertsiuzdak/snac_24khz",
        metadata={"help": "SNAC codec that turns audio tokens into sound. Default is 'hubertsiuzdak/snac_24khz'."},
    )
    svara_tts_device: str = field(
        default="auto",
        metadata={"help": "Device for the SNAC codec. Options: 'auto', 'cuda', 'mps', 'cpu'. Default is 'auto'."},
    )
    svara_tts_gender: Literal["female", "male"] = field(
        default="female",
        metadata={"help": "Voice used when the client does not ask for a female or male voice. Default is 'female'."},
    )
    svara_tts_default_language: str = field(
        default="hi",
        metadata={
            "help": "Language for text that its script does not identify. The spoken language otherwise follows "
            "the script of each sentence. Default is 'hi'."
        },
    )
    svara_tts_normalize: bool = field(
        default=True,
        metadata={
            "help": "Expand numbers, dates, currency and similar into words with Kenpath's "
            "indic-text-normalization before synthesis. Default is true."
        },
    )
    svara_tts_normalizer_languages: str = field(
        default="hi,en",
        metadata={
            "help": "Comma-separated languages whose normalization grammars are loaded at startup. Others load in "
            "the background on first use. Default is 'hi,en'."
        },
    )
    svara_tts_normalizer_cache_dir: str = field(
        default="~/.cache/speech-to-speech/indic-text-normalization",
        metadata={"help": "Folder for compiled normalization grammars, which are slow to build."},
    )
    svara_tts_temperature: float = field(
        default=0.75,
        metadata={"help": "Sampling temperature. Default is 0.75, as in Kenpath's inference code."},
    )
    svara_tts_top_p: float = field(
        default=0.9,
        metadata={"help": "Nucleus sampling threshold. Default is 0.9, as in Kenpath's inference code."},
    )
    svara_tts_top_k: int = field(
        default=40,
        metadata={"help": "Top-k sampling. Default is 40, as in Kenpath's inference code."},
    )
    svara_tts_repetition_penalty: float = field(
        default=1.1,
        metadata={"help": "Repetition penalty. Default is 1.1, as in Kenpath's inference code."},
    )
    svara_tts_max_new_tokens: int = field(
        default=2048,
        metadata={
            "help": "Audio token limit per chunk of text. One second of audio is about 82 tokens. Default is 2048."
        },
    )
    svara_tts_max_chars: int = field(
        default=200,
        metadata={"help": "Longest text sent in one generation. Longer sentences are split. Default is 200."},
    )
    svara_tts_prebuffer_ms: int = field(
        default=250,
        metadata={
            "help": "Audio to collect before playback of each chunk starts. Higher values hide slow generation at "
            "the cost of latency. Default is 250."
        },
    )
    svara_tts_blocksize: int = field(
        default=512,
        metadata={"help": "Audio chunk size in samples for streaming output. Default is 512."},
    )
