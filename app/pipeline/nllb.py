"""English -> Amharic translation with **NLLB-200** (Meta AI).

This module is the NLLB backend of :mod:`app.pipeline.translation`. It translates
each transcribed line into Amharic locally, with no API key and no network call
after the weights are cached::

    [TranscriptSegment, ...] -> translate_lines() -> [AdaptedDialogue, ...]

What NLLB is, and what it is not
--------------------------------
NLLB-200 is a *translation* model: 200 languages, sentence in, sentence out, and
Amharic (``amh_Ethi``) is one of them with real parallel training data behind it -
which is why it is a defensible choice for a low-resource language, and why it can
beat a much larger general LLM whose Amharic is incidental.

It is **not** a dialogue adaptor. It has no notion of scene, character, register,
or performance: it cannot be told to keep a joke, to sound colloquial, or to shorten
a line to fit a window. So the lines it produces are translations rather than
adaptations, and the performance metadata on each
:class:`~app.pipeline.translation.AdaptedDialogue` is a neutral default rather than
something the model decided. That is stated here rather than hidden, because it is
the real cost of this backend compared with the instruction-following one.

Model size
----------
``facebook/nllb-200-distilled-1.3B`` by default. The family's distilled sizes are
600M, 1.3B and 3.3B: 600M is markedly weaker on low-resource pairs, 3.3B is roughly
2.5x the memory and compute for a smaller gain, so 1.3B is the balance. The id is
configurable through ``TRANSLATION_MODEL``.

Determinism
-----------
Generation is greedy by default, so the same input gives the same output and a run
can be compared against a baseline. Beam search is available through
``TRANSLATION_NUM_BEAMS`` when quality matters more than reproducibility.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from app.config import Settings

#: Source and target language codes in NLLB's own ``<lang>_<script>`` form. Amharic
#: is written in the Ethiopic (Fidel) script, which is what ``amh_Ethi`` means - so
#: the model's *output* is already the script the Amharic voice needs, and no
#: transliteration is involved on this side of the pipeline.
SOURCE_LANGUAGE = "eng_Latn"
TARGET_LANGUAGE = "amh_Ethi"

#: Chunk length, in tokens, for a single translation. NLLB-200 is trained on
#: sentence-length input and degrades on paragraphs, so a long line is split rather
#: than truncated - dropping the second half of a line would be worse than a slightly
#: awkward seam.
MAX_SOURCE_TOKENS = 400


class NllbError(RuntimeError):
    """Base class for every error raised by this module."""


class ModelInitializationError(NllbError):
    """The NLLB model or its tokenizer could not be loaded."""


class TranslationFailure(NllbError):
    """NLLB ran but produced nothing usable for a line."""


def _require_transformers() -> tuple[Any, Any, Any]:
    """Return ``(torch, AutoTokenizer, AutoModelForSeq2SeqLM)`` or explain what is missing."""

    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - only without the runtime
        raise ModelInitializationError(
            "NLLB needs torch and transformers; install the runtime dependencies "
            f"before running translation ({type(exc).__name__}: {exc})"
        ) from exc

    return torch, AutoTokenizer, AutoModelForSeq2SeqLM


def _split_long_text(text: str, *, max_tokens: int = MAX_SOURCE_TOKENS) -> list[str]:
    """Split ``text`` on sentence boundaries so no chunk exceeds ``max_tokens`` words.

    Words are used as the unit rather than the tokenizer's own count: it keeps this
    a pure function with no model dependency, and a word budget is a conservative
    proxy for a token budget because a word is at least one token.
    """

    words = text.split()
    if len(words) <= max_tokens:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    for word in words:
        current.append(word)
        if len(current) >= max_tokens:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks


@dataclass(frozen=True, slots=True)
class NllbTranslation:
    """One translated line, with what the engine was asked for and produced."""

    text: str
    source_text: str
    chunks: int

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise TranslationFailure(
                f"NLLB produced nothing for {self.source_text!r}"
            )
        object.__setattr__(self, "text", self.text.strip())


class NllbTranslator:
    """A lazily loaded NLLB-200 translator, cached per process.

    Nothing is loaded at import time and nothing is downloaded until the first
    :meth:`translate` call, so importing this module costs nothing and the test suite
    never touches the network.
    """

    def __init__(
        self,
        *,
        model: str,
        device: str = "cuda",
        cache_dir: str | None = None,
        num_beams: int = 1,
        max_new_tokens: int = 512,
        length_penalty: float = 1.0,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ModelInitializationError("TRANSLATION_MODEL must name an NLLB checkpoint")
        if isinstance(num_beams, bool) or not isinstance(num_beams, int) or num_beams < 1:
            raise ModelInitializationError(
                f"TRANSLATION_NUM_BEAMS must be a positive integer, got {num_beams!r}"
            )
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens < 1
        ):
            raise ModelInitializationError(
                f"the output limit must be a positive integer, got {max_new_tokens!r}"
            )
        if not math.isfinite(length_penalty) or length_penalty <= 0:
            raise ModelInitializationError(
                f"the length penalty must be a positive number, got {length_penalty!r}"
            )

        self._model = model.strip()
        self._device = device.strip() or "cuda"
        self._cache_dir = cache_dir
        self._num_beams = num_beams
        self._max_new_tokens = max_new_tokens
        self._length_penalty = float(length_penalty)
        self._tokenizer: Any | None = None
        self._network: Any | None = None
        self._torch: Any | None = None
        self._target_token_id: int | None = None

    @property
    def model(self) -> str:
        """The checkpoint this translator was constructed for."""

        return self._model

    @property
    def is_loaded(self) -> bool:
        """``True`` once the weights are in memory."""

        return self._network is not None

    def _load(self) -> tuple[Any, Any, Any]:
        """Load the tokenizer and model once, and return them with torch."""

        if (
            self._tokenizer is not None
            and self._network is not None
            and self._torch is not None
            and self._target_token_id is not None
        ):
            return self._tokenizer, self._network, self._torch

        torch, auto_tokenizer, auto_model = _require_transformers()
        self._torch = torch

        try:
            self._tokenizer = auto_tokenizer.from_pretrained(
                self._model, cache_dir=self._cache_dir
            )
        except Exception as exc:
            raise ModelInitializationError(
                f"could not load the NLLB tokenizer for {self._model!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        try:
            self._network = auto_model.from_pretrained(
                self._model, cache_dir=self._cache_dir
            )
        except Exception as exc:
            raise ModelInitializationError(
                f"could not load the NLLB model {self._model!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        try:
            self._network.to(torch.device(self._device))
        except Exception as exc:
            raise ModelInitializationError(
                f"could not move {self._model!r} to device {self._device!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        self._network.eval()

        # NLLB selects the target language through a forced first token rather than a
        # prompt, which is the whole reason its output language is reliable.
        self._target_token_id = self._tokenizer.convert_tokens_to_ids(TARGET_LANGUAGE)
        if self._target_token_id is None or self._target_token_id < 0:
            raise ModelInitializationError(
                f"{self._model!r} does not know the target language {TARGET_LANGUAGE!r}"
            )

        return self._tokenizer, self._network, self._torch

    def translate(
        self, text: str, *, length_penalty: float | None = None
    ) -> NllbTranslation:
        """Translate one source line into Amharic.

        ``length_penalty`` overrides the decoding preference for one call. It is how a
        line that will not fit the time it has is asked for a more economical rendering:
        values below 1.0 bias the search towards shorter output, values above towards
        longer. ``None`` uses the configured default.
        """

        if not isinstance(text, str) or not text.strip():
            raise TranslationFailure("there is nothing to translate")

        penalty = self._length_penalty if length_penalty is None else length_penalty
        chunks = _split_long_text(text.strip())
        produced = [
            self._translate_chunk(chunk, length_penalty=penalty) for chunk in chunks
        ]
        joined = " ".join(piece for piece in produced if piece)
        return NllbTranslation(text=joined, source_text=text, chunks=len(chunks))

    def _translate_chunk(self, chunk: str, *, length_penalty: float) -> str:
        """Translate one chunk that is known to be short enough for the model."""

        tokenizer, network, torch = self._load()
        assert self._target_token_id is not None

        try:
            tokenizer.src_lang = SOURCE_LANGUAGE
            encoded = tokenizer(
                chunk, return_tensors="pt", truncation=True, max_length=MAX_SOURCE_TOKENS
            )
        except Exception as exc:
            raise TranslationFailure(
                f"NLLB could not tokenize a line: {type(exc).__name__}: {exc}"
            ) from exc

        device = next(network.parameters()).device
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}

        try:
            with torch.no_grad():
                generated = network.generate(
                    **encoded,
                    forced_bos_token_id=self._target_token_id,
                    max_new_tokens=self._max_new_tokens,
                    num_beams=self._num_beams,
                    # Biases the search towards shorter or longer renderings. At the
                    # default 1.0 this is the model's own preference; a line that cannot
                    # fit its time is re-asked lower, which is the only lever NLLB offers
                    # over the *length* of a translation - it cannot be instructed.
                    length_penalty=length_penalty,
                    # Greedy by default, so the same line always translates the same
                    # way and two runs can be compared.
                    do_sample=False,
                )
        except Exception as exc:
            raise TranslationFailure(
                f"NLLB failed to generate: {type(exc).__name__}: {exc}"
            ) from exc

        try:
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        except Exception as exc:
            raise TranslationFailure(
                f"NLLB produced output that could not be decoded: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if not decoded:
            raise TranslationFailure("NLLB returned no text for a line")
        return decoded[0].strip()


_TRANSLATORS: dict[tuple[str, str, str | None, int, int, float], NllbTranslator] = {}


def load_translator(*, settings: Settings) -> NllbTranslator:
    """Return the process-wide NLLB translator for these settings.

    Loading is lazy - the weights arrive on the first translation - but the
    translator itself is cached, so a stage run and every line in it share one model
    instead of reloading it per line.
    """

    key = (
        settings.translation_model,
        settings.device,
        str(settings.model_cache_dir),
        settings.translation_num_beams,
        settings.translation_max_new_tokens,
        settings.translation_length_penalty,
    )
    translator = _TRANSLATORS.get(key)
    if translator is None:
        translator = NllbTranslator(
            model=settings.translation_model,
            device=settings.device,
            cache_dir=str(settings.model_cache_dir),
            num_beams=settings.translation_num_beams,
            max_new_tokens=settings.translation_max_new_tokens,
            length_penalty=settings.translation_length_penalty,
        )
        _TRANSLATORS[key] = translator
    return translator


def reset_translator_cache() -> None:
    """Drop every cached translator, releasing the loaded weights."""

    _TRANSLATORS.clear()


def translate_batch(
    texts: Sequence[str], *, translator: NllbTranslator
) -> list[NllbTranslation]:
    """Translate several lines at once, in order."""

    return [translator.translate(text) for text in texts]


__all__ = [
    "MAX_SOURCE_TOKENS",
    "SOURCE_LANGUAGE",
    "TARGET_LANGUAGE",
    "ModelInitializationError",
    "NllbError",
    "NllbTranslation",
    "NllbTranslator",
    "TranslationFailure",
    "load_translator",
    "reset_translator_cache",
    "translate_batch",
]
