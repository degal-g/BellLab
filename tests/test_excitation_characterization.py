"""Caracterização física relativa e instrumental da condição de excitação."""

from __future__ import annotations

from dataclasses import replace
from math import log10, sqrt

import numpy as np
import pytest

from belllab import (
    DynamicOrderConsistencyResult,
    ExcitationCharacterizationSettings,
    ExcitationCondition,
    Signal,
    Recording,
    characterize_excitation,
    characterize_excitation_signal,
    evaluate_dynamic_order_consistency,
)


def _signal(
    excitation,
    *,
    background=(),
    sample_rate=100,
    unit="normalized",
    channels=None,
) -> tuple[Signal, float, ExcitationCharacterizationSettings]:
    values = tuple(background) + tuple(excitation)
    samples = channels or (values,)
    impact_index = len(background)
    signal = Signal(
        samples=tuple(tuple(channel) for channel in samples),
        sample_rate=sample_rate,
        time=tuple(index / sample_rate for index in range(len(values))),
        duration=len(values) / sample_rate,
        channels=len(samples),
        unit=unit,
    )
    background_start = -len(background) / sample_rate
    background_end = -1 / sample_rate
    settings = ExcitationCharacterizationSettings(
        analysis_window_start_s=0.0,
        analysis_window_end_s=max(1, len(excitation) - 1) / sample_rate,
        background_window_start_s=background_start if background else -0.2,
        background_window_end_s=background_end if background else -0.1,
    )
    return signal, impact_index / sample_rate, settings


def _condition(label="mf", *, session="S1", **changes):
    return ExcitationCondition(label, 0, session_id=session, **changes)


def _characterize(excitation, *, background=(), condition=None, **settings_changes):
    signal, impact, settings = _signal(excitation, background=background)
    settings = replace(settings, **settings_changes)
    return characterize_excitation_signal(
        signal, "recording-1", condition or _condition(), impact, settings
    )


def test_constant_signal_peak_rms_energy_level_and_crest() -> None:
    result = _characterize((0.5,) * 10)
    assert result.peak_absolute_amplitude == 0.5
    assert result.peak_signed_amplitude == 0.5
    assert result.positive_peak == 0.5
    assert result.negative_peak == 0.0
    assert result.rms_amplitude == 0.5
    assert result.mean_square_amplitude == 0.25
    assert result.signal_energy == pytest.approx(0.025)
    assert result.equivalent_level_dbfs == pytest.approx(20 * log10(0.5))
    assert result.crest_factor == 1.0
    assert result.crest_factor_db == 0.0


def test_known_sine_metrics() -> None:
    values = np.sin(2 * np.pi * np.arange(100) / 10)
    result = _characterize(tuple(values))
    assert result.peak_absolute_amplitude == pytest.approx(0.9510565163)
    assert result.rms_amplitude == pytest.approx(1 / sqrt(2))
    assert result.signal_energy == pytest.approx(0.5)
    assert result.crest_factor == pytest.approx(0.9510565163 * sqrt(2))


@pytest.mark.parametrize(
    ("values", "expected_rms", "expected_energy", "expected_crest"),
    [
        ((1.0,) + (0.0,) * 9, 1 / sqrt(10), 0.01, sqrt(10)),
        ((1.0, 1.0) + (0.0,) * 8, sqrt(0.2), 0.02, sqrt(5)),
    ],
    ids=["unit_impulse", "two_sample_impulse"],
)
def test_impulse_metrics(values, expected_rms, expected_energy, expected_crest) -> None:
    result = _characterize(values)
    assert result.rms_amplitude == pytest.approx(expected_rms)
    assert result.signal_energy == pytest.approx(expected_energy)
    assert result.crest_factor == pytest.approx(expected_crest)


def test_silent_window_uses_none_for_logarithmic_metrics() -> None:
    result = _characterize((0.0,) * 10)
    assert result.rms_amplitude == 0.0
    assert result.signal_energy == 0.0
    assert result.equivalent_level_dbfs is None
    assert result.crest_factor is None
    assert result.impulse_duration_s is None
    assert "silent_window" in result.diagnostics


def test_asymmetric_signed_peaks_and_definition() -> None:
    result = _characterize((0.8, -0.4, 0.1, 0.0))
    assert result.peak_absolute_amplitude == 0.8
    assert result.peak_signed_amplitude == 0.8
    assert result.positive_peak == 0.8
    assert result.negative_peak == -0.4
    assert result.peak_asymmetry == pytest.approx(0.5)


def test_negative_dominant_peak_preserves_its_sign() -> None:
    result = _characterize((0.2, -0.9, 0.3))
    assert result.peak_absolute_amplitude == 0.9
    assert result.peak_signed_amplitude == -0.9


def test_energy_depends_on_duration_while_rms_does_not() -> None:
    short = _characterize((0.5,) * 10)
    long = _characterize((0.5,) * 20)
    assert short.rms_amplitude == long.rms_amplitude == 0.5
    assert long.signal_energy == pytest.approx(2 * short.signal_energy)


def test_dc_policy_preserves_offset_and_changes_power_metrics() -> None:
    values = (0.25, 0.75) * 5
    raw = _characterize(values)
    centered = _characterize(values, remove_dc_for_power=True)
    assert raw.dc_offset == centered.dc_offset == 0.5
    assert raw.rms_amplitude == pytest.approx(sqrt(0.3125))
    assert centered.rms_amplitude == pytest.approx(0.25)
    assert raw.signal_energy == pytest.approx(0.03125)
    assert centered.signal_energy == pytest.approx(0.00625)
    assert "dc_removed_for_rms_and_energy" in centered.diagnostics


def test_impulse_energy_percentile_times_are_controlled() -> None:
    result = _characterize(
        (0.0, 1.0, 1.0, 1.0, 1.0, 0.0),
        impulse_energy_start_fraction=0.25,
        impulse_energy_end_fraction=0.75,
    )
    assert result.impulse_start_time_s == pytest.approx(result.impact_time_s + 0.01)
    assert result.impulse_end_time_s == pytest.approx(result.impact_time_s + 0.03)
    assert result.impulse_duration_s == pytest.approx(0.02)


def test_impulse_duration_is_invariant_to_amplitude_scale() -> None:
    first = _characterize((0.0, 1.0, 2.0, 1.0, 0.0))
    second = _characterize((0.0, 3.0, 6.0, 3.0, 0.0))
    assert first.impulse_start_time_s == second.impulse_start_time_s
    assert first.impulse_end_time_s == second.impulse_end_time_s
    assert first.impulse_duration_s == second.impulse_duration_s


def test_attack_and_time_to_peak_are_explicit() -> None:
    result = _characterize((0.0, 0.2, 0.5, 1.0, 0.4))
    assert result.attack_start_time_s == pytest.approx(result.impact_time_s + 0.01)
    assert result.peak_time_s == pytest.approx(result.impact_time_s + 0.03)
    assert result.attack_duration_s == pytest.approx(0.02)
    assert result.time_to_peak_s == pytest.approx(0.03)


@pytest.mark.parametrize(
    ("value", "clipped", "near"),
    [
        (0.998, False, True),
        (0.999, True, True),
        (1.0, True, True),
        (-0.999, True, True),
        (0.94, False, False),
    ],
    ids=["below_clip", "exact_clip", "above_clip", "negative_clip", "below_near"],
)
def test_clipping_thresholds_are_inclusive(value, clipped, near) -> None:
    result = _characterize((value, 0.0))
    assert result.clipping_detected is clipped
    assert result.near_clipping_detected is near


def test_consecutive_clipping_and_fractions() -> None:
    result = _characterize((0.0, 1.0, 1.0, 1.0, 0.0))
    assert result.clipped_sample_count == 3
    assert result.clipped_sample_fraction == pytest.approx(0.6)
    assert result.longest_clipped_run == 3


def test_integer_pcm_requires_explicit_full_scale_and_normalizes() -> None:
    signal, impact, settings = _signal((0, 16384, -32768), unit="pcm_s16")
    with pytest.raises(ValueError, match="pcm_full_scale"):
        characterize_excitation_signal(signal, "pcm", _condition(), impact, settings)
    result = characterize_excitation_signal(
        signal, "pcm", _condition(), impact,
        replace(settings, pcm_full_scale=32768),
    )
    assert result.amplitude_unit == "normalized_from_pcm"
    assert result.peak_absolute_amplitude == 1.0
    assert result.clipping_detected
    assert "pcm_full_scale=32768" in result.diagnostics
    assert "pcm_parameters=configured" in result.diagnostics


def test_physical_unit_does_not_manufacture_dbfs_or_digital_clipping() -> None:
    signal, impact, settings = _signal((2.0, -2.0), unit="Pa")
    result = characterize_excitation_signal(
        signal, "calibrated", _condition(), impact, settings
    )
    assert result.amplitude_unit == "Pa"
    assert result.equivalent_level_dbfs is None
    assert not result.clipping_detected
    assert "dbfs_unavailable_without_digital_reference" in result.diagnostics


def test_known_background_snr_two_and_ten_times() -> None:
    twice = _characterize((0.2,) * 10, background=(0.1,) * 10)
    ten = _characterize((1.0,) * 10, background=(0.1,) * 10)
    assert twice.background_rms == pytest.approx(0.1)
    assert twice.signal_to_background_ratio == pytest.approx(2.0)
    assert twice.signal_to_background_db == pytest.approx(20 * log10(2))
    assert ten.signal_to_background_ratio == pytest.approx(10.0)
    assert ten.signal_to_background_db == pytest.approx(20.0)


def test_zero_background_avoids_division_by_zero() -> None:
    result = _characterize((0.5,) * 5, background=(0.0,) * 5)
    assert result.background_rms == 0.0
    assert result.signal_to_background_ratio is None
    assert result.signal_to_background_db is None
    assert "zero_background_rms" in result.diagnostics


def test_empty_background_is_nonfatal_and_auditable() -> None:
    result = _characterize((0.5,) * 5)
    assert result.valid
    assert result.background_rms is None
    assert "insufficient_background_samples" in result.diagnostics


def test_nonfinite_excitation_and_background_are_discarded() -> None:
    result = _characterize(
        (0.5, float("nan"), float("inf"), 0.5),
        background=(0.1, float("-inf"), 0.1),
    )
    assert result.sample_count == 4
    assert result.finite_sample_count == 2
    assert result.discarded_sample_count == 2
    assert result.background_finite_sample_count == 2
    assert result.rms_amplitude == 0.5
    assert "nonfinite_excitation_samples_discarded" in result.diagnostics
    assert "nonfinite_background_samples_discarded" in result.diagnostics


def test_background_clipping_is_reported_separately() -> None:
    result = _characterize((0.5,) * 3, background=(0.0, 1.0, 0.0))
    assert result.background_clipping_detected
    assert not result.clipping_detected


def test_multichannel_requires_and_records_explicit_channel() -> None:
    signal, impact, settings = _signal(
        (0.0, 0.0),
        channels=((0.1, 0.1), (0.8, 0.8)),
    )
    first = characterize_excitation_signal(
        signal, "multi", _condition(), impact, settings
    )
    second = characterize_excitation_signal(
        signal, "multi", _condition(), impact, replace(settings, channel_index=1)
    )
    assert first.channel_index == 0 and first.rms_amplitude == 0.1
    assert second.channel_index == 1 and second.rms_amplitude == 0.8
    with pytest.raises(ValueError, match="outside"):
        characterize_excitation_signal(
            signal, "multi", _condition(), impact,
            replace(settings, channel_index=2),
        )
    with pytest.raises(ValueError, match="condition channel"):
        characterize_excitation_signal(
            signal, "multi", _condition(channel=1), impact, settings
        )


def test_complete_acquisition_metadata_is_preserved() -> None:
    condition = _condition(
        microphone_id="mic-1", interface_id="if-1", channel=0,
        acquisition_gain=12.0, microphone_distance_m=0.5,
        microphone_orientation="on-axis", impact_location="rim",
        exciter_type="hammer", exciter_mass_kg=0.025,
        operator_label="operator", notes="controlled",
    )
    result = _characterize((0.5,) * 5, condition=condition)
    assert result.microphone_id == "mic-1"
    assert result.interface_id == "if-1"
    assert result.acquisition_gain == 12.0
    assert result.microphone_distance_m == 0.5
    assert "cross_recording_amplitude_comparability_unverified" not in result.diagnostics


@pytest.mark.parametrize(
    "changes",
    [
        {"microphone_distance_m": -1.0},
        {"exciter_mass_kg": -0.1},
        {"acquisition_gain": float("nan")},
        {"microphone_id": "  "},
        {"interface_id": ""},
        {"channel": -1},
    ],
    ids=[
        "negative_distance", "negative_mass", "nan_gain", "blank_microphone",
        "blank_interface", "negative_channel",
    ],
)
def test_invalid_acquisition_metadata_is_rejected(changes) -> None:
    with pytest.raises(ValueError):
        _condition(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"analysis_window_start_s": 0.2, "analysis_window_end_s": 0.1},
        {"background_window_start_s": -0.1, "background_window_end_s": -0.2},
        {"background_window_end_s": 0.0},
        {"clipping_threshold": 0.9, "near_clipping_threshold": 0.95},
        {"impulse_energy_start_fraction": 0.9, "impulse_energy_end_fraction": 0.1},
        {"impulse_energy_start_fraction": -0.1},
        {"attack_threshold_fraction": 0.0},
        {"minimum_sample_count": -1},
        {"pcm_full_scale": 0.0},
        {"clipping_threshold": float("nan")},
        {"analysis_window_start_s": float("-inf")},
    ],
    ids=[
        "analysis_inverted", "background_inverted", "overlapping_windows",
        "clipping_order", "percentiles_inverted", "negative_percentile",
        "zero_attack", "negative_count", "zero_pcm_scale", "nan_threshold",
        "infinite_window",
    ],
)
def test_invalid_settings_are_rejected(changes) -> None:
    with pytest.raises(ValueError):
        ExcitationCharacterizationSettings(**changes)


def _dynamic(label: str, value: float, *, session="S1", **metadata):
    condition = _condition(label, session=session, **metadata)
    return _characterize((value,) * 10, condition=condition)


def test_perfect_pp_to_ff_rms_order() -> None:
    values = {"pp": 0.05, "p": 0.09, "mf": 0.16, "f": 0.28, "ff": 0.50}
    result = evaluate_dynamic_order_consistency(
        tuple(_dynamic(label, value) for label, value in reversed(tuple(values.items())))
    )
    assert result.ordered_labels == ("pp", "p", "mf", "f", "ff")
    assert result.observed_values == pytest.approx(tuple(values.values()))
    assert result.inversion_count == 0
    assert result.tie_count == 0
    assert result.consistent


@pytest.mark.parametrize(
    ("values", "inversions"),
    [
        ((0.05, 0.20, 0.16, 0.28, 0.50), 1),
        ((0.20, 0.10, 0.08, 0.28, 0.25), 3),
    ],
    ids=["one_inversion", "multiple_inversions"],
)
def test_dynamic_inversions_are_auditable_not_renamed(values, inversions) -> None:
    result = evaluate_dynamic_order_consistency(tuple(
        _dynamic(label, value) for label, value in zip(
            ("pp", "p", "mf", "f", "ff"), values
        )
    ))
    assert result.inversion_count == inversions
    assert not result.consistent
    assert result.ordered_labels == ("pp", "p", "mf", "f", "ff")


@pytest.mark.parametrize(
    ("upper", "tolerance", "ties"),
    [(0.1, 0.0, 1), (0.1005, 0.001, 1), (0.102, 0.001, 0)],
    ids=["exact_tie", "within_tolerance", "outside_tolerance"],
)
def test_dynamic_ties_use_configured_tolerance(upper, tolerance, ties) -> None:
    result = evaluate_dynamic_order_consistency((
        _dynamic("p", 0.1), _dynamic("mf", upper)
    ), tolerance=tolerance)
    assert result.tie_count == ties
    assert result.inversion_count == 0


def test_missing_and_single_dynamic_conditions_are_diagnostic() -> None:
    partial = evaluate_dynamic_order_consistency((
        _dynamic("pp", 0.1), _dynamic("f", 0.3)
    ))
    single = evaluate_dynamic_order_consistency((_dynamic("mf", 0.2),))
    assert "dynamic_conditions_missing" in partial.diagnostics
    assert "dynamic_conditions_missing" in single.diagnostics
    assert "insufficient_dynamic_conditions" in single.diagnostics
    assert not single.consistent


def test_dynamic_input_order_is_deterministic() -> None:
    items = tuple(_dynamic(label, value) for label, value in (
        ("pp", 0.1), ("mf", 0.3), ("ff", 0.5)
    ))
    assert evaluate_dynamic_order_consistency(items) == (
        evaluate_dynamic_order_consistency(tuple(reversed(items)))
    )


def test_dynamic_consistency_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="one explicit session"):
        evaluate_dynamic_order_consistency((
            _dynamic("pp", 0.1, session="A"),
            _dynamic("p", 0.2, session="B"),
        ))
    with pytest.raises(ValueError, match="duplicated"):
        evaluate_dynamic_order_consistency((
            _dynamic("pp", 0.1), _dynamic("pp", 0.2)
        ))
    with pytest.raises(ValueError, match="musical"):
        evaluate_dynamic_order_consistency((_dynamic("unspecified", 0.1),))
    with pytest.raises(ValueError, match="supported"):
        evaluate_dynamic_order_consistency((_dynamic("pp", 0.1),), metric="invalid")


def test_different_microphones_and_gains_mark_comparability_unverified() -> None:
    first = _dynamic(
        "pp", 0.1, microphone_id="mic-a", interface_id="if",
        acquisition_gain=10.0, microphone_distance_m=1.0,
    )
    second = _dynamic(
        "p", 0.2, microphone_id="mic-b", interface_id="if",
        acquisition_gain=20.0, microphone_distance_m=1.0,
    )
    result = evaluate_dynamic_order_consistency((first, second))
    assert "cross_recording_amplitude_comparability_unverified" in result.diagnostics


def test_identical_complete_metadata_supports_direct_relative_comparison() -> None:
    metadata = dict(
        microphone_id="mic", interface_id="if",
        acquisition_gain=10.0, microphone_distance_m=1.0,
    )
    result = evaluate_dynamic_order_consistency((
        _dynamic("pp", 0.1, **metadata), _dynamic("p", 0.2, **metadata)
    ))
    assert "cross_recording_amplitude_comparability_unverified" not in result.diagnostics


def test_clipping_prevents_direct_cross_recording_comparability() -> None:
    metadata = dict(
        microphone_id="mic", interface_id="if",
        acquisition_gain=10.0, microphone_distance_m=1.0,
    )
    clipped = _dynamic("pp", 1.0, **metadata)
    clean = _dynamic("p", 0.8, **metadata)
    result = evaluate_dynamic_order_consistency((clipped, clean))
    assert "cross_recording_amplitude_comparability_unverified" in result.diagnostics
    assert (
        "cross_recording_amplitude_comparability_compromised_by_clipping"
        in clipped.diagnostics
    )


def test_characterization_and_dynamic_order_are_reproducible() -> None:
    first = _dynamic("pp", 0.1)
    second = _dynamic("p", 0.2)
    assert first == _dynamic("pp", 0.1)
    result = evaluate_dynamic_order_consistency((first, second))
    assert result == evaluate_dynamic_order_consistency((second, first))


def test_dynamic_result_contract_is_public_and_immutable() -> None:
    result = evaluate_dynamic_order_consistency((_dynamic("mf", 0.2),))
    assert isinstance(result, DynamicOrderConsistencyResult)
    with pytest.raises(Exception):
        result.consistent = False


def test_insufficient_excitation_samples_is_a_structured_failure() -> None:
    signal, impact, settings = _signal((float("nan"), float("inf")))
    result = characterize_excitation_signal(
        signal, "invalid-window", _condition(), impact, settings
    )
    assert not result.valid
    assert result.failure_reason == "insufficient_excitation_samples"
    assert result.peak_absolute_amplitude is None
    assert result.rms_amplitude is None
    assert result.signal_energy is None


def test_peak_before_impact_is_preserved_with_alignment_diagnostic() -> None:
    signal = Signal(
        samples=((1.0, 0.2, 0.1),),
        sample_rate=100,
        time=(0.0, 0.01, 0.02),
        duration=0.03,
        channels=1,
        unit="normalized",
    )
    settings = ExcitationCharacterizationSettings(
        analysis_window_start_s=-0.02,
        analysis_window_end_s=0.01,
        background_window_start_s=-0.1,
        background_window_end_s=-0.03,
    )
    result = characterize_excitation_signal(
        signal, "early-peak", _condition(), 0.01, settings
    )
    assert result.time_to_peak_s == pytest.approx(-0.01)
    assert "peak_precedes_impact_time" in result.diagnostics


def test_recording_adapter_does_not_read_files_and_uses_stable_identity(tmp_path) -> None:
    signal, impact, settings = _signal((0.5,) * 3)
    recording = Recording(tmp_path / "not-read.wav", "bell", signal)
    result = characterize_excitation(
        recording, _condition(), impact, settings, recording_id="take-7"
    )
    assert result.recording_id == "take-7"
    assert result.rms_amplitude == 0.5


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mean_square_amplitude", 999.0, "mean_square"),
        ("clipped_sample_count", 1, "clipping flag"),
        ("clipped_sample_fraction", 2.0, "fractions"),
        ("signal_to_background_ratio", -1.0, "ratio"),
    ],
    ids=["mean_square_mismatch", "clipping_mismatch", "invalid_fraction", "negative_ratio"],
)
def test_characterization_invariants_reject_manual_inconsistency(
    field, value, message
) -> None:
    valid = _characterize((0.5,) * 3)
    with pytest.raises(ValueError, match=message):
        replace(valid, **{field: value})
