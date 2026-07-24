"""Fechamento da normalização PCM assinada e não assinada."""

from __future__ import annotations

from dataclasses import replace
from math import log10, sqrt

import numpy as np
import pytest
import soundfile as sf

from belllab import (
    ExcitationCharacterizationSettings,
    ExcitationCondition,
    Signal,
    characterize_excitation_signal,
)
from belllab.io import load_wav


def _analyze(
    values,
    *,
    sample_rate: int = 100,
    unit: str = "digital",
    settings: ExcitationCharacterizationSettings | None = None,
):
    samples = tuple(values)
    signal = Signal(
        samples=(samples,),
        sample_rate=sample_rate,
        time=tuple(index / sample_rate for index in range(len(samples))),
        duration=len(samples) / sample_rate,
        channels=1,
        unit=unit,
    )
    cfg = settings or ExcitationCharacterizationSettings(
        analysis_window_start_s=0.0,
        analysis_window_end_s=max(1, len(samples) - 1) / sample_rate,
        background_window_start_s=-0.2,
        background_window_end_s=-0.1,
    )
    return characterize_excitation_signal(
        signal,
        "pcm",
        ExcitationCondition("mf", 0),
        0.0,
        cfg,
    )


def test_int16_minimum_zero_and_maximum_use_single_linear_scale() -> None:
    result = _analyze(np.array(
        [-32768, -16384, 0, 16384, 32767], dtype=np.int16
    ))
    assert result.negative_peak == -1.0
    assert result.positive_peak == pytest.approx(32767 / 32768)
    assert result.peak_absolute_amplitude == 1.0
    assert result.dc_offset == pytest.approx(-1 / (5 * 32768))
    assert "pcm_original_dtype=int16" in result.diagnostics
    assert "pcm_signed=true" in result.diagnostics
    assert "pcm_zero_point=0" in result.diagnostics
    assert "pcm_full_scale=32768" in result.diagnostics
    assert "pcm_parameters=inferred_from_dtype" in result.diagnostics


def test_int8_uses_same_twos_complement_policy() -> None:
    result = _analyze(np.array([-128, -64, 0, 64, 127], dtype=np.int8))
    assert result.negative_peak == -1.0
    assert result.positive_peak == pytest.approx(127 / 128)
    assert "pcm_full_scale=128" in result.diagnostics


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (0, -1.0),
        (64, -0.5),
        (127, -1 / 128),
        (128, 0.0),
        (129, 1 / 128),
        (192, 0.5),
        (255, 127 / 128),
    ],
    ids=[
        "minimum", "below_midpoint", "immediately_below_midpoint", "midpoint",
        "immediately_above_midpoint", "above_midpoint", "maximum",
    ],
)
def test_uint8_codes_are_centered_before_scaling(code, expected) -> None:
    result = _analyze(np.array([code], dtype=np.uint8))
    assert result.peak_signed_amplitude == pytest.approx(expected)
    assert result.dc_offset == pytest.approx(expected)
    assert "pcm_original_dtype=uint8" in result.diagnostics
    assert "pcm_signed=false" in result.diagnostics
    assert "pcm_zero_point=128" in result.diagnostics
    assert "pcm_full_scale=128" in result.diagnostics


def test_uint8_midpoint_is_silent_without_artificial_offset_or_dbfs() -> None:
    result = _analyze(np.full(8, 128, dtype=np.uint8))
    assert result.dc_offset == 0.0
    assert result.rms_amplitude == 0.0
    assert result.signal_energy == 0.0
    assert result.equivalent_level_dbfs is None
    assert "silent_window" in result.diagnostics


def test_uint8_symmetric_signal_has_zero_mean_known_rms_and_energy() -> None:
    result = _analyze(np.array([64, 192, 64, 192], dtype=np.uint8))
    assert result.negative_peak == -0.5
    assert result.positive_peak == 0.5
    assert result.dc_offset == 0.0
    assert result.rms_amplitude == 0.5
    assert result.mean_square_amplitude == 0.25
    assert result.signal_energy == pytest.approx(0.01)
    assert result.equivalent_level_dbfs == pytest.approx(20 * log10(0.5))


def test_uint8_subtraction_cannot_wrap_around() -> None:
    result = _analyze(np.array([0, 127, 128, 129], dtype=np.uint8))
    assert result.negative_peak == -1.0
    assert result.positive_peak == pytest.approx(1 / 128)
    assert "pcm_converted_to_float_before_centering" in result.diagnostics


@pytest.mark.parametrize(
    ("values", "threshold", "clipped"),
    [
        (np.array([0], dtype=np.uint8), 0.999, True),
        (np.array([255], dtype=np.uint8), 0.99, True),
        (np.array([128], dtype=np.uint8), 0.1, False),
        (np.array([192], dtype=np.uint8), 0.5, True),
        (np.array([191], dtype=np.uint8), 0.5, False),
    ],
    ids=[
        "negative_extreme", "positive_extreme", "midpoint",
        "exact_normalized_threshold", "below_normalized_threshold",
    ],
)
def test_uint8_clipping_uses_normalized_amplitude(values, threshold, clipped) -> None:
    cfg = ExcitationCharacterizationSettings(
        analysis_window_start_s=0.0,
        analysis_window_end_s=0.01,
        background_window_start_s=-0.2,
        background_window_end_s=-0.1,
        clipping_threshold=threshold,
        near_clipping_threshold=min(0.05, threshold),
    )
    assert _analyze(values, settings=cfg).clipping_detected is clipped


def test_uint8_positive_extreme_can_be_near_clipping_without_clipping() -> None:
    cfg = ExcitationCharacterizationSettings(
        analysis_window_start_s=0.0,
        analysis_window_end_s=0.01,
        background_window_start_s=-0.2,
        background_window_end_s=-0.1,
        clipping_threshold=0.999,
        near_clipping_threshold=0.98,
    )
    result = _analyze(np.array([255], dtype=np.uint8), settings=cfg)
    assert not result.clipping_detected
    assert result.near_clipping_detected


def test_float_int16_and_uint8_equivalent_codes_have_equal_metrics() -> None:
    floating = _analyze(
        np.array([-0.5, 0.0, 0.5], dtype=np.float64), unit="normalized"
    )
    signed = _analyze(np.array([-16384, 0, 16384], dtype=np.int16))
    unsigned = _analyze(np.array([64, 128, 192], dtype=np.uint8))
    for result in (signed, unsigned):
        assert result.dc_offset == pytest.approx(floating.dc_offset)
        assert result.rms_amplitude == pytest.approx(floating.rms_amplitude)
        assert result.signal_energy == pytest.approx(floating.signal_energy)
        assert result.equivalent_level_dbfs == pytest.approx(
            floating.equivalent_level_dbfs
        )
    assert floating.rms_amplitude == pytest.approx(sqrt(1 / 6))


def test_uint8_full_range_documents_inevitable_endpoint_asymmetry() -> None:
    result = _analyze(np.array([0, 255], dtype=np.uint8))
    assert result.negative_peak == -1.0
    assert result.positive_peak == pytest.approx(127 / 128)
    assert result.peak_asymmetry == pytest.approx(-1 / 128)


def test_explicit_unsigned_zero_point_and_scale_are_auditable() -> None:
    cfg = ExcitationCharacterizationSettings(
        analysis_window_start_s=0.0,
        analysis_window_end_s=0.02,
        background_window_start_s=-0.2,
        background_window_end_s=-0.1,
        pcm_zero_point=128,
        pcm_full_scale=128,
    )
    result = _analyze(np.array([0, 128, 255], dtype=np.uint8), settings=cfg)
    assert "pcm_parameters=configured" in result.diagnostics
    assert result.dc_offset == pytest.approx((-1 + 0 + 127 / 128) / 3)


@pytest.mark.parametrize(
    ("values", "changes", "message"),
    [
        (np.array([0], dtype=np.uint8), {"pcm_zero_point": -1}, "dtype limits"),
        (np.array([0], dtype=np.uint8), {"pcm_zero_point": 256}, "dtype limits"),
        (np.array([0], dtype=np.uint8), {"pcm_full_scale": 127}, "cover"),
        (np.array([0], dtype=np.int16), {"pcm_zero_point": 1}, "zero_point=0"),
        (np.array([0.0]), {"pcm_full_scale": 1}, "floating-point"),
        (np.array([0.0]), {"pcm_zero_point": 0}, "floating-point"),
    ],
    ids=[
        "zero_below_dtype", "zero_above_dtype", "scale_too_small",
        "signed_nonzero_point", "float_with_scale", "float_with_zero_point",
    ],
)
def test_incoherent_pcm_parameters_are_rejected(values, changes, message) -> None:
    cfg = replace(ExcitationCharacterizationSettings(
        analysis_window_start_s=0.0,
        analysis_window_end_s=0.01,
        background_window_start_s=-0.2,
        background_window_end_s=-0.1,
    ), **changes)
    with pytest.raises(ValueError, match=message):
        _analyze(values, settings=cfg)


@pytest.mark.parametrize(
    "values",
    [
        np.array([True, False], dtype=bool),
        np.array([1, "2"], dtype=object),
        np.array([1 + 2j], dtype=np.complex128),
    ],
    ids=["boolean", "object", "complex"],
)
def test_invalid_sample_dtypes_are_rejected(values) -> None:
    with pytest.raises(ValueError, match="dtype"):
        _analyze(values)


@pytest.mark.parametrize(
    "changes",
    [
        {"pcm_full_scale": -1.0},
        {"pcm_full_scale": 0.0},
        {"pcm_full_scale": float("nan")},
        {"pcm_full_scale": float("inf")},
        {"pcm_zero_point": float("nan")},
        {"pcm_zero_point": float("-inf")},
    ],
    ids=[
        "negative_scale", "zero_scale", "nan_scale", "infinite_scale",
        "nan_zero_point", "infinite_zero_point",
    ],
)
def test_nonfinite_or_nonpositive_pcm_settings_are_rejected(changes) -> None:
    with pytest.raises(ValueError):
        ExcitationCharacterizationSettings(**changes)


def test_pcm_normalization_is_deterministic() -> None:
    values = np.array([0, 64, 128, 192, 255], dtype=np.uint8)
    assert _analyze(values) == _analyze(values)


def test_pcm_u8_wav_loader_delivers_centered_signal_without_offset(tmp_path) -> None:
    path = tmp_path / "unsigned.wav"
    sf.write(
        path,
        np.array([-0.5, 0.0, 0.5], dtype=np.float64),
        100,
        subtype="PCM_U8",
    )
    signal, _ = load_wav(path)
    result = characterize_excitation_signal(
        signal,
        "loaded-u8",
        ExcitationCondition("mf", 0),
        0.0,
        ExcitationCharacterizationSettings(
            analysis_window_start_s=0.0,
            analysis_window_end_s=0.02,
            background_window_start_s=-0.2,
            background_window_end_s=-0.1,
        ),
    )
    assert result.dc_offset == pytest.approx(0.0, abs=1 / 256)
    assert result.rms_amplitude == pytest.approx(sqrt(1 / 6), abs=1 / 256)
