"""Tests for the native-Amharic voice bank and casting.

Casting exists for one audible reason: a reference cut from an English film is
the actor speaking English, and the cloning engine carries that accent into the
Amharic line. Swapping in a human Amharic reference removes the mismatch and
gives up the actor's identity, so what these tests pin down is the mechanics that
decision depends on - which rows are allowed to become a voice, which character
gets which voice, and what the profile records afterwards - rather than the audio,
which only a listen can judge.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
import soundfile as sf

from app.config import Settings
from app.pipeline import voice_bank as vb
from app.pipeline.voice_bank import (
    BANK_MANIFEST_FILENAME,
    BANK_REFERENCE_FILENAME,
    BankBuildError,
    BankFilters,
    BankVoice,
    CastingError,
    InvalidBankError,
    VoiceBank,
    VoiceRequest,
    apply_voice_bank,
    assign_bank_voices,
    build_voice_bank,
    build_voice_bank_from_samples,
    load_voice_bank,
)
from app.pipeline.voice_profiles import VoiceProfile, save_voice_profiles

RATE = 24_000

#: A sentence with 27 Fidel syllables and no Latin script or numerals, so a five
#: second reading of it lands inside the syllable-rate band the filters enforce.
TEXT = "ዛሬ ጠዋት አዲስ አበባ ውስጥ በጣም ቀዝቃዛ አየር ነበር"


def _settings(root, **overrides) -> Settings:
    values: dict[str, object] = {
        "input_dir": root,
        "work_dir": root / "work",
        "output_dir": root / "out",
        "model_cache_dir": root / "models",
        "voice_profile_dir": root / "voices",
        "voice_bank_dir": root / "bank",
        "voice_reference_min_duration": 3.0,
        "voice_reference_target_duration": 10.0,
        "voice_reference_max_duration": 15.0,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _row(**overrides) -> dict[str, object]:
    row: dict[str, object] = {
        "speaker_id": "spk_a",
        "clip_id": "clip-1",
        "sentence": TEXT,
        "gender": "male",
        "duration_s": 5.6,
        "speech_s": 5.0,
        "speech_start_s": 0.3,
        "speech_end_s": 5.3,
        "lufs": -20.0,
        "up_votes": 3,
        "down_votes": 0,
        "audio": [{"src": "https://example.invalid/clip.wav"}],
    }
    row.update(overrides)
    return row


def _wav_bytes(seconds: float = 6.0, *, rate: int = 16_000, frequency: float = 150.0) -> bytes:
    """Return a periodic, speech-like recording as WAV bytes."""

    time = np.arange(int(round(seconds * rate))) / rate
    samples = 0.3 * np.sin(2 * np.pi * frequency * time)
    samples += 0.01 * np.random.default_rng(0).standard_normal(samples.size)
    buffer = io.BytesIO()
    sf.write(buffer, samples.astype(np.float32), rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


class _FakeSource:
    """A row source with no network, so the builder can be tested offline."""

    dataset = "fake/amharic-speech"
    split = "train"
    revision = "0123456789abcdef"
    license = "cc-by-4.0"

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def count(self) -> int:
        return len(self._rows)

    def rows(self, offset: int, length: int):
        return self._rows[offset : offset + length]

    def download(self, url: str) -> bytes:
        return _wav_bytes()


def _reference(path, *, seconds: float = 5.0, rate: int = RATE) -> None:
    """Write a placeholder reference so a profile can point at a real file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.zeros(int(round(seconds * rate)), dtype=np.float32)
    sf.write(str(path), samples, rate, subtype="PCM_16")


def _profile(root, speaker: str, **overrides) -> VoiceProfile:
    reference = root / "film" / speaker / "reference.wav"
    _reference(reference)
    values: dict[str, object] = {
        "speaker_id": speaker,
        "reference_audio": reference,
        "reference_start": 12.0,
        "reference_end": 17.0,
        "reference_text": "የፊልሙ የመጀመሪያ ድምፅ",
        "quality_score": 0.82,
        "selection_reason": "film window",
        "estimated_vocal_presentation": "masculine",
        "estimated_pitch_hz": 120.0,
    }
    values.update(overrides)
    return VoiceProfile(**values)  # type: ignore[arg-type]


def _bank_voice(voice_id: str, *, gender: str, pitch: float, root, duration=5.0) -> BankVoice:
    audio = root / "clips" / f"{voice_id}.wav"
    if not audio.is_file():
        _reference(audio, seconds=duration)
    return BankVoice(
        voice_id=voice_id,
        audio=audio,
        text=TEXT,
        duration=duration,
        gender=gender,
        presentation={"male": "masculine", "female": "feminine"}.get(gender, "unknown"),
        pitch_hz=pitch,
    )


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def test_a_clean_row_is_accepted() -> None:
    assert BankFilters().rejection(_row()) is None


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"gender": ""}, "gender"),
        ({"up_votes": 0}, "votes"),
        ({"down_votes": 2}, "bad recording"),
        ({"lufs": -50.0}, "loudness"),
        ({"speech_s": 2.0}, "speech length"),
        ({"speech_s": 12.0, "duration_s": 12.9, "speech_end_s": 12.3}, "speech length"),
        ({"speaker_id": ""}, "speaker"),
        ({"sentence": ""}, "transcript"),
        ({"sentence": TEXT + " Bole"}, "Latin"),
        ({"sentence": TEXT + " 19"}, "numerals"),
        ({"sentence": "አዎ"}, "transcript length"),
    ],
)
def test_a_defective_row_is_rejected_with_a_reason(overrides, fragment) -> None:
    reason = BankFilters().rejection(_row(**overrides))
    assert reason is not None
    assert fragment in reason


def test_a_row_whose_speech_runs_past_the_recording_is_rejected() -> None:
    reason = BankFilters().rejection(_row(speech_start_s=1.0, speech_s=5.0, duration_s=5.5))
    assert reason is not None
    assert "past the recording" in reason


def test_a_transcript_that_cannot_fit_its_audio_is_rejected() -> None:
    rules = BankFilters()
    assert rules.rejection(_row(speech_s=9.0)) is not None
    assert rules.rejection(_row(speech_s=4.0)) is None


def test_filters_reject_impossible_bounds() -> None:
    with pytest.raises(InvalidBankError):
        BankFilters(min_speech_seconds=9.0, max_speech_seconds=4.0)
    with pytest.raises(InvalidBankError):
        BankFilters(min_lufs=-14.0, max_lufs=-34.0)
    with pytest.raises(InvalidBankError):
        BankFilters(min_clip_seconds=0.5)


# ---------------------------------------------------------------------------
# Voices and manifests
# ---------------------------------------------------------------------------


def test_a_bank_voice_refuses_a_cloning_length_outside_the_engine_limits() -> None:
    with pytest.raises(InvalidBankError):
        BankVoice(voice_id="v", audio=__file__, text=TEXT, duration=40.0, gender="male", presentation="masculine")
    with pytest.raises(InvalidBankError):
        BankVoice(voice_id="v", audio=__file__, text=TEXT, duration=5.0, gender="other", presentation="unknown")


def test_a_bank_refuses_duplicate_voice_ids(tmp_path) -> None:
    voice = _bank_voice("am-m-one", gender="male", pitch=120.0, root=tmp_path)
    with pytest.raises(InvalidBankError):
        VoiceBank(directory=tmp_path, voices=(voice, voice))


def test_no_manifest_means_no_bank(tmp_path) -> None:
    assert load_voice_bank(directory=tmp_path / "absent") is None
    (tmp_path / "empty").mkdir()
    assert load_voice_bank(directory=tmp_path / "empty") is None


def test_a_manifest_naming_a_missing_clip_is_an_error(tmp_path) -> None:
    voice = _bank_voice("am-m-one", gender="male", pitch=120.0, root=tmp_path)
    bank = VoiceBank(directory=tmp_path, voices=(voice,))
    (tmp_path / BANK_MANIFEST_FILENAME).write_text(
        json.dumps(bank.to_dict()), encoding="utf-8"
    )
    voice.audio.unlink()

    with pytest.raises(vb.MissingBankError):
        load_voice_bank(directory=tmp_path)


def test_a_manifest_from_a_future_version_is_refused(tmp_path) -> None:
    (tmp_path / BANK_MANIFEST_FILENAME).write_text(
        json.dumps({"version": 99, "voices": []}), encoding="utf-8"
    )
    with pytest.raises(InvalidBankError):
        load_voice_bank(directory=tmp_path)


# ---------------------------------------------------------------------------
# Casting
# ---------------------------------------------------------------------------


def test_casting_matches_gender_then_closest_pitch(tmp_path) -> None:
    bank = VoiceBank(
        directory=tmp_path,
        voices=(
            _bank_voice("am-m-low", gender="male", pitch=105.0, root=tmp_path),
            _bank_voice("am-m-high", gender="male", pitch=165.0, root=tmp_path),
            _bank_voice("am-f-mid", gender="female", pitch=210.0, root=tmp_path),
        ),
    )
    casting = assign_bank_voices(
        [
            VoiceRequest("SPEAKER_00", "masculine", 118.0),
            VoiceRequest("SPEAKER_01", "masculine", 158.0),
            VoiceRequest("SPEAKER_02", "feminine", 200.0),
        ],
        bank,
    )

    assert [casting[speaker].voice_id for speaker in ("SPEAKER_00", "SPEAKER_01", "SPEAKER_02")] == [
        "am-m-low",
        "am-m-high",
        "am-f-mid",
    ]


def test_casting_is_deterministic(tmp_path) -> None:
    bank = VoiceBank(
        directory=tmp_path,
        voices=(
            _bank_voice("am-m-b", gender="male", pitch=130.0, root=tmp_path),
            _bank_voice("am-m-a", gender="male", pitch=130.0, root=tmp_path),
        ),
    )
    requests = [VoiceRequest("SPEAKER_00", "masculine", 130.0), VoiceRequest("SPEAKER_01", "masculine", 130.0)]

    first = assign_bank_voices(requests, bank)
    second = assign_bank_voices(requests, bank)

    assert {key: voice.voice_id for key, voice in first.items()} == {
        key: voice.voice_id for key, voice in second.items()
    }
    # Equal pitch distance falls back to the voice id, not to dictionary order.
    assert first["SPEAKER_00"].voice_id == "am-m-a"


def test_no_voice_is_reused_while_an_unused_one_remains(tmp_path) -> None:
    bank = VoiceBank(
        directory=tmp_path,
        voices=(
            _bank_voice("am-m-one", gender="male", pitch=120.0, root=tmp_path),
            _bank_voice("am-f-one", gender="female", pitch=210.0, root=tmp_path),
        ),
    )
    casting = assign_bank_voices(
        [
            VoiceRequest("SPEAKER_00", "masculine", 120.0),
            VoiceRequest("SPEAKER_01", "masculine", 120.0),
        ],
        bank,
    )
    # The second masculine character takes the unused female voice rather than
    # sharing the first character's voice: two characters must not sound alike.
    assert casting["SPEAKER_00"].voice_id == "am-m-one"
    assert casting["SPEAKER_01"].voice_id == "am-f-one"


def test_a_bank_smaller_than_the_cast_reuses_voices(tmp_path) -> None:
    bank = VoiceBank(
        directory=tmp_path,
        voices=(_bank_voice("am-m-one", gender="male", pitch=120.0, root=tmp_path),),
    )
    casting = assign_bank_voices(
        [VoiceRequest("SPEAKER_00", "masculine", 120.0), VoiceRequest("SPEAKER_01", "masculine", 130.0)],
        bank,
    )
    assert {voice.voice_id for voice in casting.values()} == {"am-m-one"}


def test_an_unknown_presentation_still_gets_a_mixed_cast(tmp_path) -> None:
    bank = VoiceBank(
        directory=tmp_path,
        voices=(
            _bank_voice("am-f-a", gender="female", pitch=210.0, root=tmp_path),
            _bank_voice("am-f-b", gender="female", pitch=230.0, root=tmp_path),
            _bank_voice("am-m-a", gender="male", pitch=110.0, root=tmp_path),
            _bank_voice("am-m-b", gender="male", pitch=140.0, root=tmp_path),
        ),
    )
    casting = assign_bank_voices(
        [VoiceRequest("SPEAKER_00"), VoiceRequest("SPEAKER_01")], bank
    )

    # With no presentation to match on, ranking by voice id alone would give both
    # characters the female voices, because "am-f-..." sorts first. The cast has to
    # stay mixed instead.
    genders = sorted({casting["SPEAKER_00"].gender, casting["SPEAKER_01"].gender})
    assert genders == ["female", "male"]


def test_a_mapping_pins_a_voice_and_is_validated(tmp_path) -> None:
    bank = VoiceBank(
        directory=tmp_path,
        voices=(
            _bank_voice("am-m-one", gender="male", pitch=120.0, root=tmp_path),
            _bank_voice("am-f-one", gender="female", pitch=210.0, root=tmp_path),
        ),
    )
    requests = [VoiceRequest("SPEAKER_00", "masculine", 120.0)]

    pinned = assign_bank_voices(requests, bank, mapping="SPEAKER_00=am-f-one")
    assert pinned["SPEAKER_00"].voice_id == "am-f-one"

    with pytest.raises(CastingError):
        assign_bank_voices(requests, bank, mapping="SPEAKER_09=am-f-one")
    with pytest.raises(CastingError):
        assign_bank_voices(requests, bank, mapping="SPEAKER_00")
    with pytest.raises(InvalidBankError):
        assign_bank_voices(requests, bank, mapping="SPEAKER_00=am-missing")


def test_a_mapping_cannot_give_one_voice_to_two_characters(tmp_path) -> None:
    bank = VoiceBank(
        directory=tmp_path,
        voices=(
            _bank_voice("am-m-one", gender="male", pitch=120.0, root=tmp_path),
            _bank_voice("am-f-one", gender="female", pitch=210.0, root=tmp_path),
        ),
    )
    with pytest.raises(InvalidBankError):
        assign_bank_voices(
            [VoiceRequest("SPEAKER_00"), VoiceRequest("SPEAKER_01")],
            bank,
            mapping="SPEAKER_00=am-m-one,SPEAKER_01=am-m-one",
        )


# ---------------------------------------------------------------------------
# Applying a bank to profiles
# ---------------------------------------------------------------------------


def test_applying_a_bank_replaces_the_reference_and_records_the_casting(tmp_path) -> None:
    settings = _settings(tmp_path)
    bank = VoiceBank(
        directory=tmp_path / "bank",
        voices=(_bank_voice("am-m-one", gender="male", pitch=118.0, root=tmp_path),),
        dataset="fake/amharic-speech",
        revision="abcdef0123456789",
        license="cc-by-4.0",
    )
    profiles = {"SPEAKER_00": _profile(tmp_path, "SPEAKER_00")}

    replaced = apply_voice_bank(profiles, settings=settings, bank=bank)
    profile = replaced["SPEAKER_00"]

    assert profile.reference_audio.name == BANK_REFERENCE_FILENAME
    assert profile.reference_text == TEXT
    assert profile.reference_start == 0.0
    assert profile.reference_end == pytest.approx(bank.voices[0].duration, abs=1e-3)
    assert profile.reference_source == "voice_bank"
    assert profile.reference_voice_id == "am-m-one"
    assert profile.clone_prompt_path is None
    assert profile.quality_score == 0.82
    assert profile.estimated_vocal_presentation == "masculine"
    assert profile.estimated_pitch_hz == 118.0
    assert "voice bank am-m-one" in profile.selection_reason

    info = sf.info(str(profile.reference_audio))
    assert info.samplerate == 24_000
    assert info.channels == 1


def test_applying_without_a_bank_changes_nothing(tmp_path) -> None:
    settings = _settings(tmp_path)
    profiles = {"SPEAKER_00": _profile(tmp_path, "SPEAKER_00")}

    replaced = apply_voice_bank(profiles, settings=settings)

    assert replaced == profiles
    assert replaced["SPEAKER_00"].reference_source == "film"


def test_apply_casts_only_the_named_characters(tmp_path) -> None:
    """A fallback is per character: the actor keeps their voice when theirs is usable."""

    settings = _settings(tmp_path)
    bank = VoiceBank(
        directory=tmp_path / "bank",
        voices=(_bank_voice("am-m-one", gender="male", pitch=118.0, root=tmp_path),),
    )
    profiles = {
        "SPEAKER_00": _profile(tmp_path, "SPEAKER_00"),
        "SPEAKER_01": _profile(tmp_path, "SPEAKER_01"),
    }

    replaced = apply_voice_bank(
        profiles, settings=settings, bank=bank, speaker_ids=["SPEAKER_01"]
    )

    assert replaced["SPEAKER_00"] == profiles["SPEAKER_00"]
    assert replaced["SPEAKER_00"].reference_source == "film"
    assert replaced["SPEAKER_01"].reference_source == "voice_bank"
    assert replaced["SPEAKER_01"].reference_voice_id == "am-m-one"
    assert set(replaced) == {"SPEAKER_00", "SPEAKER_01"}


def test_a_casted_character_with_no_film_profile_is_still_returned(tmp_path) -> None:
    """The character the bank exists for may have no measurable film window at all."""

    settings = _settings(tmp_path)
    bank = VoiceBank(
        directory=tmp_path / "bank",
        voices=(_bank_voice("am-f-one", gender="female", pitch=210.0, root=tmp_path),),
    )

    replaced = apply_voice_bank(
        {}, settings=settings, bank=bank, speaker_ids=["SPEAKER_07"]
    )

    profile = replaced["SPEAKER_07"]
    assert profile.reference_source == "voice_bank"
    assert profile.reference_voice_id == "am-f-one"
    assert profile.quality_score == 0.0
    assert profile.source_vocal_presentation == "unknown"
    assert profile.source_pitch_hz is None
    assert profile.reference_audio.name == BANK_REFERENCE_FILENAME


def test_measured_requests_cast_a_character_without_a_profile(tmp_path) -> None:
    """A measurable actor keeps their register even when no reference could be built."""

    settings = _settings(tmp_path)
    bank = VoiceBank(
        directory=tmp_path / "bank",
        voices=(
            _bank_voice("am-m-one", gender="male", pitch=118.0, root=tmp_path),
            _bank_voice("am-f-one", gender="female", pitch=212.0, root=tmp_path),
        ),
    )

    replaced = apply_voice_bank(
        {},
        settings=settings,
        bank=bank,
        requests=[VoiceRequest("SPEAKER_02", "feminine", 214.0)],
    )

    profile = replaced["SPEAKER_02"]
    assert profile.reference_voice_id == "am-f-one"
    assert profile.source_vocal_presentation == "feminine"
    assert profile.source_pitch_hz == 214.0


def test_a_cast_selection_cannot_be_given_twice(tmp_path) -> None:
    settings = _settings(tmp_path)
    bank = VoiceBank(
        directory=tmp_path / "bank",
        voices=(_bank_voice("am-m-one", gender="male", pitch=118.0, root=tmp_path),),
    )

    with pytest.raises(CastingError):
        apply_voice_bank(
            {},
            settings=settings,
            bank=bank,
            speaker_ids=["SPEAKER_02"],
            requests=[VoiceRequest("SPEAKER_02")],
        )


def test_casting_keeps_the_actors_measured_register_for_audit(tmp_path) -> None:
    settings = _settings(tmp_path)
    bank = VoiceBank(
        directory=tmp_path / "bank",
        voices=(_bank_voice("am-f-one", gender="female", pitch=210.0, root=tmp_path),),
    )
    profiles = {
        "SPEAKER_00": _profile(
            tmp_path,
            "SPEAKER_00",
            estimated_vocal_presentation="masculine",
            estimated_pitch_hz=118.0,
            source_vocal_presentation="masculine",
            source_pitch_hz=116.0,
        )
    }

    replaced = apply_voice_bank(profiles, settings=settings, bank=bank)

    profile = replaced["SPEAKER_00"]
    assert profile.estimated_vocal_presentation == "feminine"
    assert profile.estimated_pitch_hz == 210.0
    # What the film contained, not what the character now sounds like.
    assert profile.source_vocal_presentation == "masculine"
    assert profile.source_pitch_hz == 116.0


def test_an_older_profile_without_source_register_keeps_its_estimate(tmp_path) -> None:
    """A profile written before ``source_*`` existed still has a usable audit value."""

    settings = _settings(tmp_path)
    bank = VoiceBank(
        directory=tmp_path / "bank",
        voices=(_bank_voice("am-f-one", gender="female", pitch=210.0, root=tmp_path),),
    )
    profiles = {
        "SPEAKER_00": _profile(
            tmp_path,
            "SPEAKER_00",
            estimated_vocal_presentation="masculine",
            estimated_pitch_hz=118.0,
        )
    }

    replaced = apply_voice_bank(profiles, settings=settings, bank=bank)

    assert replaced["SPEAKER_00"].source_vocal_presentation == "masculine"
    assert replaced["SPEAKER_00"].source_pitch_hz == 118.0


def test_casting_refuses_an_unnamed_speaker(tmp_path) -> None:
    settings = _settings(tmp_path)
    bank = VoiceBank(
        directory=tmp_path / "bank",
        voices=(_bank_voice("am-m-one", gender="male", pitch=118.0, root=tmp_path),),
    )

    with pytest.raises(CastingError):
        apply_voice_bank({}, settings=settings, bank=bank, speaker_ids=["   "])


def test_an_operator_pin_is_validated_against_the_whole_film(tmp_path) -> None:
    """A typo must be reported even when that character kept their film reference."""

    settings = _settings(tmp_path, voice_bank_voice_map="SPEAKER_09=am-m-one")
    bank = VoiceBank(
        directory=tmp_path / "bank",
        voices=(_bank_voice("am-m-one", gender="male", pitch=118.0, root=tmp_path),),
    )
    profiles = {
        "SPEAKER_00": _profile(tmp_path, "SPEAKER_00"),
        "SPEAKER_01": _profile(tmp_path, "SPEAKER_01"),
    }

    with pytest.raises(CastingError, match="SPEAKER_09"):
        apply_voice_bank(
            profiles, settings=settings, bank=bank, speaker_ids=["SPEAKER_01"]
        )


def test_the_operator_mapping_is_honoured_when_applying(tmp_path) -> None:
    settings = _settings(
        tmp_path, voice_bank_voice_map="SPEAKER_00=am-f-one"
    )
    bank = VoiceBank(
        directory=tmp_path / "bank",
        voices=(
            _bank_voice("am-m-one", gender="male", pitch=118.0, root=tmp_path),
            _bank_voice("am-f-one", gender="female", pitch=210.0, root=tmp_path),
        ),
    )
    profiles = {"SPEAKER_00": _profile(tmp_path, "SPEAKER_00")}

    replaced = apply_voice_bank(profiles, settings=settings, bank=bank)

    assert replaced["SPEAKER_00"].reference_voice_id == "am-f-one"


def test_a_reused_voice_is_reported_in_the_profile(tmp_path) -> None:
    settings = _settings(tmp_path)
    bank = VoiceBank(
        directory=tmp_path / "bank",
        voices=(_bank_voice("am-m-one", gender="male", pitch=118.0, root=tmp_path),),
    )
    profiles = {
        "SPEAKER_00": _profile(tmp_path, "SPEAKER_00"),
        "SPEAKER_01": _profile(tmp_path, "SPEAKER_01"),
    }

    replaced = apply_voice_bank(profiles, settings=settings, bank=bank)

    assert "reused" in replaced["SPEAKER_00"].selection_reason
    assert "reused" in replaced["SPEAKER_01"].selection_reason


def test_casting_provenance_survives_a_save_and_load(tmp_path) -> None:
    settings = _settings(tmp_path)
    bank = VoiceBank(
        directory=tmp_path / "bank",
        voices=(_bank_voice("am-m-one", gender="male", pitch=118.0, root=tmp_path),),
    )
    profiles = apply_voice_bank(
        {"SPEAKER_00": _profile(tmp_path, "SPEAKER_00")}, settings=settings, bank=bank
    )
    save_voice_profiles(profiles, settings=settings)

    payload = json.loads(
        (tmp_path / "voices" / "SPEAKER_00" / "profile.json").read_text(encoding="utf-8")
    )
    assert payload["reference_source"] == "voice_bank"
    assert payload["reference_voice_id"] == "am-m-one"

    rebuilt = VoiceProfile.from_dict(payload)
    assert rebuilt.reference_source == "voice_bank"
    assert rebuilt.reference_voice_id == "am-m-one"


def test_a_profile_cannot_claim_a_bank_reference_without_a_voice(tmp_path) -> None:
    with pytest.raises(Exception):
        _profile(tmp_path, "SPEAKER_00", reference_source="voice_bank")


# ---------------------------------------------------------------------------
# Building a bank
# ---------------------------------------------------------------------------


def test_building_writes_a_reloadable_bank(tmp_path) -> None:
    settings = _settings(tmp_path)
    rows = [
        _row(speaker_id="spk_a", clip_id="a1", gender="male"),
        _row(speaker_id="spk_a", clip_id="a2", gender="male", speech_s=5.2),
        _row(speaker_id="spk_b", clip_id="b1", gender="female"),
        _row(speaker_id="spk_c", clip_id="c1", gender="female", up_votes=0),
    ]
    lines: list[str] = []

    bank = build_voice_bank(
        destination=tmp_path / "bank",
        settings=settings,
        voices=4,
        source=_FakeSource(rows),
        balance_genders=False,
        report=lines.append,
    )

    assert len(bank) == 2
    assert {voice.gender for voice in bank} == {"male", "female"}
    assert any("rejected" in line for line in lines)

    manifest = tmp_path / "bank" / BANK_MANIFEST_FILENAME
    assert manifest.is_file()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["dataset"] == "fake/amharic-speech"
    assert payload["revision"] == "0123456789abcdef"
    assert payload["license"] == "cc-by-4.0"
    assert payload["filters"]["target_speech_seconds"] == 5.0
    assert (tmp_path / "bank" / vb.BANK_ATTRIBUTION_FILENAME).is_file()

    reloaded = load_voice_bank(directory=tmp_path / "bank")
    assert reloaded is not None
    assert {voice.voice_id for voice in reloaded} == {voice.voice_id for voice in bank}

    for voice in reloaded:
        info = sf.info(str(voice.audio))
        assert info.samplerate == 24_000
        assert info.channels == 1
        assert voice.text == TEXT
        assert 4.0 <= voice.duration <= 10.0


def test_building_keeps_the_best_clip_per_speaker(tmp_path) -> None:
    settings = _settings(tmp_path)
    rows = [
        _row(speaker_id="spk_a", clip_id="far", speech_s=8.8),
        _row(speaker_id="spk_a", clip_id="near", speech_s=5.0),
    ]
    bank = build_voice_bank(
        destination=tmp_path / "bank",
        settings=settings,
        voices=1,
        source=_FakeSource(rows),
        balance_genders=False,
    )
    assert bank.voices[0].source_clip_id == "near"


def test_a_bounded_build_still_finds_both_genders(tmp_path) -> None:
    """A page limit must not silently produce a one-sided bank.

    This corpus stores each speaker's clips together, so a short scan can hold a single
    gender - and that bank casts every character, women included, in a man's voice. The
    limit makes a build quick; it is not allowed to make it unbalanced.
    """

    settings = _settings(tmp_path)
    rows = [_row(speaker_id="spk_m1", clip_id="m1", gender="male")]
    # One page of page-size rows, all male, then a female speaker beyond the limit.
    rows += [
        _row(speaker_id=f"spk_m{index}", clip_id=f"m{index}", gender="male")
        for index in range(2, 101)
    ]
    rows.append(_row(speaker_id="spk_f1", clip_id="f1", gender="female"))

    bank = build_voice_bank(
        destination=tmp_path / "bank",
        settings=settings,
        voices=2,
        source=_FakeSource(rows),
        max_pages=1,
    )

    assert {voice.gender for voice in bank} == {"male", "female"}


def test_a_corpus_with_one_gender_says_so(tmp_path) -> None:
    settings = _settings(tmp_path)
    lines: list[str] = []
    rows = [_row(speaker_id=f"spk{index}", clip_id=f"c{index}", gender="male") for index in range(3)]

    bank = build_voice_bank(
        destination=tmp_path / "bank",
        settings=settings,
        voices=2,
        source=_FakeSource(rows),
        report=lines.append,
    )

    assert {voice.gender for voice in bank} == {"male"}
    assert any("other register" in line for line in lines)


def test_building_fails_when_nothing_qualifies(tmp_path) -> None:
    settings = _settings(tmp_path)
    with pytest.raises(BankBuildError):
        build_voice_bank(
            destination=tmp_path / "bank",
            settings=settings,
            source=_FakeSource([_row(gender="")]),
        )


def test_building_refuses_a_useless_voice_count(tmp_path) -> None:
    with pytest.raises(BankBuildError):
        build_voice_bank(
            destination=tmp_path / "bank",
            settings=_settings(tmp_path),
            voices=0,
            source=_FakeSource([_row()]),
        )


# ---------------------------------------------------------------------------
# An operator's own recordings
# ---------------------------------------------------------------------------

FIDEL_WORD = "\u123b\u122d"  # ሻር


def _sample(
    directory: Path,
    stem: str,
    *,
    seconds: float = 6.0,
    text: str | None = None,
    suffix: str = ".wav",
    frequency: float = 150.0,
    rate: int = 48_000,
) -> Path:
    """Write a compliant recording and its transcript, returning the audio path."""

    directory.mkdir(parents=True, exist_ok=True)
    time = np.arange(int(seconds * rate)) / rate
    path = directory / f"{stem}{suffix}"
    sf.write(
        str(path),
        (0.3 * np.sin(2 * np.pi * frequency * time)).astype(np.float32),
        rate,
    )
    if text is None:
        # ~5 syllables per second, inside the accepted band.
        text = " ".join([FIDEL_WORD] * max(8, int(seconds * 5 / 2))) + "\u1362"
    (directory / f"{stem}.txt").write_text(text, encoding="utf-8")
    return path


def test_a_local_sample_reads_as_a_voice_with_derived_metadata(tmp_path) -> None:
    samples = tmp_path / "samples"
    _sample(samples, "male-01", seconds=6.0, frequency=120.0)
    _sample(samples, "female-07", seconds=5.0, frequency=210.0, suffix=".mp3")

    bank = build_voice_bank_from_samples(
        samples, destination=tmp_path / "bank", settings=_settings(tmp_path)
    )

    voices = {voice.voice_id: voice for voice in bank}
    assert set(voices) == {"am-m-01", "am-f-07"}
    assert voices["am-m-01"].gender == "male"
    assert voices["am-m-01"].presentation == "masculine"
    assert voices["am-f-07"].gender == "female"
    assert voices["am-f-07"].presentation == "feminine"
    # Duration, pitch and the clip itself are measured, never taken from the operator.
    assert voices["am-m-01"].duration == pytest.approx(6.0, abs=0.02)
    assert voices["am-m-01"].pitch_hz == pytest.approx(120.0, abs=8.0)
    assert voices["am-f-07"].pitch_hz == pytest.approx(210.0, abs=8.0)

    for voice in bank:
        info = sf.info(str(voice.audio))
        assert info.samplerate == 24_000
        assert info.channels == 1
        assert " ".join(voice.text.split())[:0] == ""

    assert bank.dataset == "local/native-speakers"
    assert (tmp_path / "bank" / BANK_MANIFEST_FILENAME).is_file()
    assert (tmp_path / "bank" / vb.BANK_ATTRIBUTION_FILENAME).is_file()
    reloaded = load_voice_bank(directory=tmp_path / "bank")
    assert reloaded is not None and len(reloaded) == 2


def test_a_local_sample_that_is_too_long_is_refused_with_advice(tmp_path) -> None:
    samples = tmp_path / "samples"
    _sample(samples, "male-01", seconds=25.0)

    with pytest.raises(BankBuildError) as info:
        build_voice_bank_from_samples(
            samples, destination=tmp_path / "bank", settings=_settings(tmp_path)
        )

    message = str(info.value)
    assert "male-01.wav" in message
    assert "25.00s" in message
    assert "make the transcript match the trimmed audio" in message
    assert "4-10s" in message
    assert not (tmp_path / "bank" / BANK_MANIFEST_FILENAME).exists()


def test_every_problem_is_reported_at_once(tmp_path) -> None:
    """One round trip per operator, not one per file."""

    samples = tmp_path / "samples"
    _sample(samples, "male-01", seconds=20.0)
    # Two problems in one file: too long *and* an empty transcript.
    audio = _sample(samples, "female-01", seconds=30.0)
    (samples / "female-01.txt").write_text("   ", encoding="utf-8")
    _sample(samples, "voice-01", seconds=6.0)  # no gender in the name

    with pytest.raises(BankBuildError) as info:
        build_voice_bank_from_samples(
            samples, destination=tmp_path / "bank", settings=_settings(tmp_path)
        )

    message = str(info.value)
    assert "3 of 3 recording(s)" in message
    assert "4 problem(s)" in message
    assert "female-01.txt is empty" in message
    assert "female-01.wav: is 30.00s" in message
    assert "voice-01.wav: name does not start with male/female" in message
    assert audio.is_file()


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("\u123b\u122d Bole", "Latin script"),
        ("\u123b\u122d 19", "numerals"),
    ],
)
def test_a_transcript_that_is_not_fidel_is_refused(tmp_path, text, fragment) -> None:
    samples = tmp_path / "samples"
    _sample(samples, "male-01", text=text)

    with pytest.raises(BankBuildError) as info:
        build_voice_bank_from_samples(
            samples, destination=tmp_path / "bank", settings=_settings(tmp_path)
        )

    assert fragment in str(info.value)


def test_a_transcript_that_cannot_fit_its_audio_is_refused(tmp_path) -> None:
    samples = tmp_path / "samples"
    _sample(samples, "male-01", seconds=6.0, text=" ".join([FIDEL_WORD] * 120))

    with pytest.raises(BankBuildError) as info:
        build_voice_bank_from_samples(
            samples, destination=tmp_path / "bank", settings=_settings(tmp_path)
        )

    assert "no speaker could produce" in str(info.value)


def test_a_recording_without_a_transcript_is_refused(tmp_path) -> None:
    samples = tmp_path / "samples"
    audio = _sample(samples, "male-01")
    (samples / "male-01.txt").unlink()

    with pytest.raises(BankBuildError) as info:
        build_voice_bank_from_samples(
            samples, destination=tmp_path / "bank", settings=_settings(tmp_path)
        )

    assert "no male-01.txt beside it" in str(info.value)
    assert audio.is_file()


def test_a_hot_recording_is_reported_but_still_banked(tmp_path) -> None:
    """Above full scale is a warning, not a rejection: only a listen can judge it."""

    samples = tmp_path / "samples"
    path = _sample(samples, "male-01")
    # A float WAV, because PCM_16 clamps at full scale: the overshoot that makes a
    # recording "hot" only survives in a float container (or across an MP3 decode).
    samples_out, rate = sf.read(str(path), dtype="float32")
    sf.write(str(path), (samples_out * 4.0).astype(np.float32), rate, subtype="FLOAT")

    lines: list[str] = []
    bank = build_voice_bank_from_samples(
        samples,
        destination=tmp_path / "bank",
        settings=_settings(tmp_path),
        report=lines.append,
    )

    assert len(bank) == 1
    assert any("peaks at" in line for line in lines)
    # Attenuated to the reference peak on the way in, so the bank clip itself is safe.
    peak = float(np.abs(sf.read(str(bank.voices[0].audio), dtype="float32")[0]).max())
    assert peak <= 1.0


def test_an_empty_sample_directory_is_refused(tmp_path) -> None:
    empty = tmp_path / "samples"
    empty.mkdir()
    with pytest.raises(BankBuildError, match="no"):
        build_voice_bank_from_samples(
            empty, destination=tmp_path / "bank", settings=_settings(tmp_path)
        )


def test_a_rebuild_prunes_clips_it_no_longer_uses(tmp_path) -> None:
    """A bank directory outlives the build that made it."""

    samples = tmp_path / "samples"
    _sample(samples, "male-01", seconds=6.0)
    _sample(samples, "female-01", seconds=6.0)
    bank_dir = tmp_path / "bank"
    build_voice_bank_from_samples(
        samples, destination=bank_dir, settings=_settings(tmp_path)
    )
    assert len(list((bank_dir / "clips").glob("*.wav"))) == 2

    (samples / "female-01.wav").unlink()
    (samples / "female-01.txt").unlink()
    lines: list[str] = []
    bank = build_voice_bank_from_samples(
        samples,
        destination=bank_dir,
        settings=_settings(tmp_path),
        report=lines.append,
    )

    assert [voice.voice_id for voice in bank] == ["am-m-01"]
    assert [path.stem for path in (bank_dir / "clips").glob("*.wav")] == ["am-m-01"]
    assert any("removed 1 clip" in line for line in lines)


def test_a_mildly_unusual_delivery_rate_is_warned_about_not_refused(tmp_path) -> None:
    """A fast speaker is not a mismatched transcript.

    Measured delivery across one operator's eighteen recordings ran 6.2-9.3
    syllables/second, so a bound tight enough to reject 9.3 would reject good files on a
    coin flip. Only a rate no speaker could produce means the text describes other audio.
    """

    samples = tmp_path / "samples"
    # 32 syllables over 3.2s = 10 syllables/second: brisk, but a real delivery.
    _sample(
        samples,
        "female-07",
        seconds=3.2,
        text=" ".join([FIDEL_WORD] * 16) + "\u1362",
    )
    lines: list[str] = []

    bank = build_voice_bank_from_samples(
        samples,
        destination=tmp_path / "bank",
        settings=_settings(tmp_path),
        report=lines.append,
        filters=BankFilters(min_clip_seconds=3.0),
    )

    assert len(bank) == 1
    assert any("outside the usual band" in line for line in lines)


def test_an_impossible_delivery_rate_is_refused(tmp_path) -> None:
    samples = tmp_path / "samples"
    # A whole paragraph claimed against six seconds: ~25 syllables/second.
    _sample(samples, "male-01", seconds=6.0, text=" ".join([FIDEL_WORD] * 150))

    with pytest.raises(BankBuildError) as info:
        build_voice_bank_from_samples(
            samples, destination=tmp_path / "bank", settings=_settings(tmp_path)
        )

    assert "no speaker could produce" in str(info.value)


@pytest.mark.parametrize(
    ("stem", "gender"),
    [
        ("male-01", "male"),
        ("Female_3", "female"),
        ("woman07", "female"),
        ("MAN-2", "male"),
        ("voice-01", None),
        ("-01", None),
    ],
)
def test_a_file_name_declares_a_gender_or_it_does_not(stem, gender) -> None:
    assert vb.sample_gender(stem) == gender

