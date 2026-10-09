"""Tests for :mod:`app.pipeline.nllb`.

No model is downloaded and no network call is made: NLLB needs torch and
transformers, so a fake tokenizer and a fake network stand in for both, and the
translator's own logic - chunking, target-language forcing, error wrapping,
caching - is what is exercised.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.config import DEFAULT_TRANSLATION_MODEL, Settings
from app.pipeline import nllb
from app.pipeline.nllb import (
    MAX_SOURCE_TOKENS,
    SOURCE_LANGUAGE,
    TARGET_LANGUAGE,
    ModelInitializationError,
    NllbTranslator,
    TranslationFailure,
    load_translator,
    reset_translator_cache,
)


class FakeTensor:
    """Stand-in for a torch tensor: enough to be moved and iterated."""

    def __init__(self, items):
        self.items = list(items)
        self.device = "cpu"

    def to(self, device):
        self.device = device
        return self

    def __iter__(self):
        return iter(self.items)


class FakeEncoding(dict):
    def __init__(self, values):
        super().__init__(values)


class FakeTokenizer:
    """Records how it was called and returns a fixed encoding."""

    instances: list["FakeTokenizer"] = []

    def __init__(self, name, cache_dir=None):
        self.name = name
        self.cache_dir = cache_dir
        self.src_lang = None
        self.calls: list[dict] = []
        self.decoded: list[str] = ["የተተረጎመ መስመር"]
        FakeTokenizer.instances.append(self)

    @classmethod
    def from_pretrained(cls, name, cache_dir=None):
        return cls(name, cache_dir=cache_dir)

    def __call__(self, text, **kwargs):
        self.calls.append({"text": text, **kwargs})
        return FakeEncoding({"input_ids": FakeTensor([1, 2, 3])})

    def convert_tokens_to_ids(self, token):
        return 250_001 if token == TARGET_LANGUAGE else 0

    def batch_decode(self, generated, skip_special_tokens=True):
        return list(self.decoded)


class FakeNetwork:
    """Records generate() calls and returns a fixed tensor."""

    instances: list["FakeNetwork"] = []
    fail_on_generate: Exception | None = None

    def __init__(self, name, cache_dir=None):
        self.name = name
        self.cache_dir = cache_dir
        self.device = "cpu"
        self.config = SimpleNamespace()
        self.generates: list[dict] = []
        self.moved_to = None
        FakeNetwork.instances.append(self)

    @classmethod
    def from_pretrained(cls, name, cache_dir=None):
        return cls(name, cache_dir=cache_dir)

    def to(self, device):
        self.moved_to = device
        self.device = str(device)
        return self

    def eval(self):
        return self

    def parameters(self):
        return iter([SimpleNamespace(device=self.device)])

    def generate(self, **kwargs):
        self.generates.append(kwargs)
        if FakeNetwork.fail_on_generate is not None:
            raise FakeNetwork.fail_on_generate
        return FakeTensor([[7, 8, 9]])


class FakeTorch:
    """The parts of torch this module uses."""

    device_calls: list[str] = []
    seeded_with: list[int] = []

    @staticmethod
    def device(name):
        FakeTorch.device_calls.append(name)
        return f"torch.device({name})"

    @staticmethod
    def no_grad():
        class _Ctx:
            def __enter__(self):
                return None

            def __exit__(self, *exc):
                return False

        return _Ctx()

    @staticmethod
    def manual_seed(value):
        FakeTorch.seeded_with.append(value)


def _patch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install the fakes and reset their recorded state."""

    FakeTokenizer.instances = []
    FakeNetwork.instances = []
    FakeNetwork.fail_on_generate = None
    FakeTorch.device_calls = []
    FakeTorch.seeded_with = []

    monkeypatch.setattr(
        nllb,
        "_require_transformers",
        lambda: (FakeTorch, FakeTokenizer, FakeNetwork),
    )
    reset_translator_cache()


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "input_dir": "input",
        "work_dir": "work",
        "output_dir": "output",
        "model_cache_dir": "models",
        "device": "cuda",
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# laziness
# ---------------------------------------------------------------------------


def test_constructing_a_translator_loads_nothing(monkeypatch) -> None:
    _patch(monkeypatch)

    translator = NllbTranslator(model=DEFAULT_TRANSLATION_MODEL)

    assert translator.is_loaded is False
    assert FakeTokenizer.instances == []
    assert FakeNetwork.instances == []


def test_importing_the_module_downloads_nothing(monkeypatch) -> None:
    """Lazy at import: the module builds no translator until one is asked for."""

    _patch(monkeypatch)

    # Reloading would rebind the module's classes and invalidate every other test in
    # this file, so laziness is asserted against a cleared cache instead.
    reset_translator_cache()

    assert nllb._TRANSLATORS == {}
    assert FakeTokenizer.instances == []
    assert FakeNetwork.instances == []


def test_resolving_a_translator_does_not_load_weights(monkeypatch) -> None:
    """Even picking the engine is cheap; only translating pulls the weights."""

    _patch(monkeypatch)
    translator = load_translator(settings=_settings())

    assert translator.is_loaded is False
    assert FakeNetwork.instances == []


# ---------------------------------------------------------------------------
# translation
# ---------------------------------------------------------------------------


def test_a_line_is_translated(monkeypatch) -> None:
    _patch(monkeypatch)
    translator = NllbTranslator(model="facebook/nllb-200-distilled-1.3B")

    result = translator.translate("I'm here")

    assert result.text == "የተተረጎመ መስመር"
    assert result.source_text == "I'm here"
    assert result.chunks == 1
    assert translator.is_loaded is True


def test_the_source_language_is_set_on_the_tokenizer(monkeypatch) -> None:
    _patch(monkeypatch)
    translator = NllbTranslator(model="m")

    translator.translate("hello")

    assert FakeTokenizer.instances[0].src_lang == SOURCE_LANGUAGE


def test_the_target_language_is_forced_rather_than_prompted(monkeypatch) -> None:
    """NLLB selects its output language with a token id, which is why it is reliable."""

    _patch(monkeypatch)
    translator = NllbTranslator(model="m")

    translator.translate("hello")

    (call,) = FakeNetwork.instances[0].generates
    assert call["forced_bos_token_id"] == 250_001


def test_generation_is_greedy_by_default(monkeypatch) -> None:
    """Deterministic by default, so two runs can be compared."""

    _patch(monkeypatch)
    translator = NllbTranslator(model="m")

    translator.translate("hello")

    (call,) = FakeNetwork.instances[0].generates
    assert call["do_sample"] is False
    assert call["num_beams"] == 1


def test_beam_search_is_available(monkeypatch) -> None:
    _patch(monkeypatch)
    translator = NllbTranslator(model="m", num_beams=4)

    translator.translate("hello")

    assert FakeNetwork.instances[0].generates[0]["num_beams"] == 4


def test_the_model_is_moved_to_the_configured_device(monkeypatch) -> None:
    _patch(monkeypatch)
    translator = NllbTranslator(model="m", device="cuda:1")

    translator.translate("hello")

    assert FakeNetwork.instances[0].moved_to == "torch.device(cuda:1)"


def test_the_model_is_loaded_once_for_many_lines(monkeypatch) -> None:
    """Loading per line would be catastrophic for a film."""

    _patch(monkeypatch)
    translator = NllbTranslator(model="m")

    for _ in range(5):
        translator.translate("a line")

    assert len(FakeNetwork.instances) == 1
    assert len(FakeTokenizer.instances) == 1


def test_the_cache_directory_is_honoured(monkeypatch) -> None:
    _patch(monkeypatch)
    translator = NllbTranslator(model="m", cache_dir="/tmp/cache")

    translator.translate("hello")

    assert FakeTokenizer.instances[0].cache_dir == "/tmp/cache"
    assert FakeNetwork.instances[0].cache_dir == "/tmp/cache"


# ---------------------------------------------------------------------------
# long input
# ---------------------------------------------------------------------------


def test_a_very_long_line_is_chunked_rather_than_truncated(monkeypatch) -> None:
    """Dropping half a line would be worse than an awkward seam."""

    _patch(monkeypatch)
    translator = NllbTranslator(model="m")
    long_text = " ".join(f"word{index}" for index in range(MAX_SOURCE_TOKENS + 50))

    result = translator.translate(long_text)

    assert result.chunks == 2
    assert len(FakeTokenizer.instances[0].calls) == 2
    # The join of the two chunks is what comes back, not just the first.
    assert result.text.count("የተተረጎመ መስመር") == 2


def test_a_line_at_the_limit_is_not_chunked(monkeypatch) -> None:
    _patch(monkeypatch)
    translator = NllbTranslator(model="m")
    text = " ".join("w" for _ in range(MAX_SOURCE_TOKENS))

    assert translator.translate(text).chunks == 1


def test_an_empty_line_is_rejected(monkeypatch) -> None:
    _patch(monkeypatch)
    translator = NllbTranslator(model="m")

    with pytest.raises(TranslationFailure, match="nothing to translate"):
        translator.translate("   ")
    with pytest.raises(TranslationFailure, match="nothing to translate"):
        translator.translate(None)  # type: ignore[arg-type]


def test_empty_model_output_is_rejected(monkeypatch) -> None:
    _patch(monkeypatch)
    translator = NllbTranslator(model="m")
    translator.translate("prime the load")
    FakeTokenizer.instances[0].decoded = [""]

    with pytest.raises(TranslationFailure, match="produced nothing"):
        translator.translate("hello")


# ---------------------------------------------------------------------------
# failures
# ---------------------------------------------------------------------------


def test_a_tokenizer_failure_names_the_model(monkeypatch) -> None:
    _patch(monkeypatch)

    def _boom(name, cache_dir=None):
        raise OSError("no such repo")

    monkeypatch.setattr(FakeTokenizer, "from_pretrained", staticmethod(_boom))
    translator = NllbTranslator(model="bad/model")

    with pytest.raises(ModelInitializationError, match="bad/model"):
        translator.translate("hello")


def test_a_model_failure_names_the_model(monkeypatch) -> None:
    _patch(monkeypatch)

    def _boom(name, cache_dir=None):
        raise OSError("disk full")

    monkeypatch.setattr(FakeNetwork, "from_pretrained", staticmethod(_boom))
    translator = NllbTranslator(model="some/model")

    with pytest.raises(ModelInitializationError, match="some/model"):
        translator.translate("hello")


def test_a_generation_failure_is_wrapped(monkeypatch) -> None:
    _patch(monkeypatch)
    FakeNetwork.fail_on_generate = RuntimeError("cuda out of memory")
    translator = NllbTranslator(model="m")

    with pytest.raises(TranslationFailure, match="failed to generate"):
        translator.translate("hello")


def test_a_missing_runtime_is_reported_clearly(monkeypatch) -> None:
    _patch(monkeypatch)

    def _missing():
        raise ModelInitializationError(
            "NLLB needs torch and transformers; install the runtime dependencies"
        )

    monkeypatch.setattr(nllb, "_require_transformers", _missing)
    translator = NllbTranslator(model="m")

    with pytest.raises(ModelInitializationError, match="needs torch and transformers"):
        translator.translate("hello")


def test_an_unusable_model_name_is_rejected() -> None:
    with pytest.raises(ModelInitializationError, match="TRANSLATION_MODEL"):
        NllbTranslator(model="   ")
    with pytest.raises(ModelInitializationError, match="NUM_BEAMS"):
        NllbTranslator(model="m", num_beams=0)
    with pytest.raises(ModelInitializationError, match="positive integer"):
        NllbTranslator(model="m", max_new_tokens=0)


# ---------------------------------------------------------------------------
# the process-wide cache
# ---------------------------------------------------------------------------


def test_the_translator_is_cached_per_settings(monkeypatch) -> None:
    _patch(monkeypatch)
    settings = _settings(translation_model="facebook/nllb-200-distilled-1.3B")

    first = load_translator(settings=settings)

    assert load_translator(settings=settings) is first
    # A different model is a different translator.
    other = load_translator(settings=_settings(translation_model="other/model"))
    assert other is not first


def test_the_cache_can_be_reset(monkeypatch) -> None:
    _patch(monkeypatch)
    settings = _settings()

    first = load_translator(settings=settings)
    reset_translator_cache()
    second = load_translator(settings=settings)

    assert first is not second


def test_the_length_penalty_reaches_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """The only lever NLLB offers over how long a translation comes out."""

    _patch(monkeypatch)
    translator = NllbTranslator(model="facebook/nllb-200-distilled-1.3B")

    translator.translate("hello", length_penalty=0.6)

    assert FakeNetwork.instances[0].generates[-1]["length_penalty"] == 0.6


def test_the_configured_penalty_is_used_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch)
    translator = NllbTranslator(
        model="facebook/nllb-200-distilled-1.3B", length_penalty=0.75
    )

    translator.translate("hello")

    assert FakeNetwork.instances[0].generates[-1]["length_penalty"] == 0.75


def test_the_model_default_is_one(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch)

    NllbTranslator(model="facebook/nllb-200-distilled-1.3B").translate("hello")

    assert FakeNetwork.instances[0].generates[-1]["length_penalty"] == 1.0


@pytest.mark.parametrize("penalty", [0.0, -1.0, float("nan"), float("inf")])
def test_an_unusable_length_penalty_is_rejected(penalty: float) -> None:
    with pytest.raises(ModelInitializationError, match="length penalty"):
        NllbTranslator(model="facebook/nllb-200-distilled-1.3B", length_penalty=penalty)
