"""Validação quantitativa completa da caracterização de trajetórias espectrais."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from belllab import (
    SpectralTrack,
    TrackFrequencyFit,
    characterize_spectral_track,
)
from belllab.tracking import _fit_frequency


def _track(
    frequencies: tuple[float, ...],
    *,
    refined: tuple[float | None, ...] | None = None,
    amplitudes: tuple[float, ...] | None = None,
    frames: tuple[int, ...] | None = None,
    times: tuple[float, ...] | None = None,
    unit: str = "linear_amplitude",
    analysis_final_frame: int | None = None,
) -> SpectralTrack:
    """Constrói uma trajetória auditável sem executar matching."""
    count = len(frequencies)
    frames = frames or tuple(range(count))
    times = times or tuple(float(index) for index in range(count))
    refined = refined if refined is not None else frequencies
    amplitudes = amplitudes or tuple(1.0 for _ in range(count))
    finite_frequencies = np.asarray(
        [value for value in frequencies if np.isfinite(value)], dtype=float
    )
    finite_amplitudes = np.asarray(
        [value for value in amplitudes if np.isfinite(value)], dtype=float
    )
    representative_frequency = (
        float(np.mean(finite_frequencies)) if finite_frequencies.size else 0.0
    )
    representative_amplitude = (
        float(np.max(finite_amplitudes)) if finite_amplitudes.size else 0.0
    )
    gaps = tuple(
        current - previous - 1
        for previous, current in zip(frames, frames[1:])
    )
    return SpectralTrack(
        track_id=3,
        frame_indices=frames,
        times_s=times,
        bin_frequencies_hz=frequencies,
        refined_frequencies_hz=refined,
        amplitudes=amplitudes,
        amplitude_unit=unit,
        prominences=tuple(None for _ in frames),
        widths_hz=tuple(None for _ in frames),
        local_snr_db=tuple(None for _ in frames),
        peak_references=tuple((frame, 0) for frame in frames),
        first_frame=frames[0],
        last_frame=frames[-1],
        duration_s=times[-1] - times[0],
        observation_count=count,
        gap_count=sum(gap > 0 for gap in gaps),
        total_missing_frames=sum(gaps),
        largest_gap_frames=max(gaps, default=0),
        mean_frequency_hz=representative_frequency,
        median_frequency_hz=representative_frequency,
        frequency_std_hz=0.0,
        initial_frequency_hz=representative_frequency,
        final_frequency_hz=representative_frequency,
        frequency_drift_hz=0.0,
        mean_drift_hz_per_s=0.0 if count > 1 else None,
        max_amplitude=representative_amplitude,
        initial_amplitude=representative_amplitude,
        final_amplitude=representative_amplitude,
        median_amplitude=representative_amplitude,
        analysis_final_frame=analysis_final_frame,
    )


@pytest.mark.parametrize(
    ("frequencies", "expected_slope", "expected_intercept", "expected_drift"),
    [
        ((100.0, 100.0, 100.0), 0.0, 100.0, 0.0),
        ((100.0, 102.0, 104.0), 2.0, 100.0, 4.0),
        ((104.0, 102.0, 100.0), -2.0, 104.0, -4.0),
    ],
    ids=["constant", "linear_increasing", "linear_decreasing"],
)
def test_frequency_descriptive_metrics_and_linear_fit(
    frequencies, expected_slope, expected_intercept, expected_drift
) -> None:
    result = characterize_spectral_track(_track(frequencies))
    fit = result.frequency_fit
    assert result.frequency_initial_hz == pytest.approx(frequencies[0])
    assert result.frequency_final_hz == pytest.approx(frequencies[-1])
    assert result.frequency_mean_hz == pytest.approx(np.mean(frequencies))
    assert result.frequency_median_hz == pytest.approx(np.median(frequencies))
    assert result.frequency_min_hz == pytest.approx(min(frequencies))
    assert result.frequency_max_hz == pytest.approx(max(frequencies))
    assert result.frequency_std_hz == pytest.approx(np.std(frequencies))
    assert result.frequency_total_drift_hz == pytest.approx(expected_drift)
    assert result.frequency_peak_to_peak_hz == pytest.approx(
        max(frequencies) - min(frequencies)
    )
    assert result.relative_frequency_stability == pytest.approx(
        np.std(frequencies) / np.median(frequencies)
    )
    assert fit.success and fit.method == "linear_frequency_drift"
    assert fit.slope_hz_per_s == pytest.approx(expected_slope, abs=1e-12)
    assert fit.intercept_hz == pytest.approx(expected_intercept)
    assert fit.rmse_hz == pytest.approx(0.0, abs=1e-12)
    if expected_drift == 0:
        assert fit.r_squared is None
        assert "constant_frequency" in result.diagnostics
    else:
        assert fit.r_squared == pytest.approx(1.0)


def test_deterministic_frequency_perturbation_has_known_regression() -> None:
    result = characterize_spectral_track(_track((100.0, 102.0, 101.0)))
    assert result.frequency_mean_hz == pytest.approx(101.0)
    assert result.frequency_median_hz == pytest.approx(101.0)
    assert result.frequency_std_hz == pytest.approx(np.sqrt(2 / 3))
    assert result.frequency_total_drift_hz == pytest.approx(1.0)
    assert result.frequency_peak_to_peak_hz == pytest.approx(2.0)
    assert result.frequency_fit.slope_hz_per_s == pytest.approx(0.5)
    assert result.frequency_fit.intercept_hz == pytest.approx(100.5)
    assert result.frequency_fit.rmse_hz == pytest.approx(np.sqrt(1 / 2))
    assert result.frequency_fit.r_squared == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("refined", "expected_source"),
    [
        ((100.2, 101.2, 102.2), "interpolated"),
        ((None, None, None), "bin"),
        ((100.2, None, 102.2), "mixed"),
    ],
    ids=["interpolated_only", "bin_only", "mixed"],
)
def test_canonical_frequency_source_is_auditable(refined, expected_source) -> None:
    result = characterize_spectral_track(
        _track((100.0, 101.0, 102.0), refined=refined)
    )
    expected = tuple(
        raw if value is None else value
        for raw, value in zip((100.0, 101.0, 102.0), refined, strict=True)
    )
    assert result.frequency_source == expected_source
    assert result.frequency_initial_hz == pytest.approx(expected[0])
    assert result.frequency_final_hz == pytest.approx(expected[-1])
    assert ("mixed_frequency_source" in result.diagnostics) is (
        expected_source == "mixed"
    )
    assert result.frequency_series == "refined_or_bin"


def test_nonfinite_frequency_values_are_discarded_without_nan_sentinels() -> None:
    result = characterize_spectral_track(
        _track((100.0, np.nan, 102.0), refined=(None, None, None))
    )
    assert (
        result.frequency_available_point_count,
        result.frequency_finite_point_count,
        result.frequency_discarded_point_count,
    ) == (3, 2, 1)
    assert result.frequency_mean_hz == pytest.approx(101.0)
    assert result.frequency_fit.slope_hz_per_s == pytest.approx(1.0)
    assert "nonfinite_frequency_values_discarded" in result.diagnostics
    assert all(
        value is None or np.isfinite(value)
        for value in (
            result.frequency_mean_hz,
            result.frequency_std_hz,
            result.frequency_fit.rmse_hz,
        )
    )


def test_all_nonfinite_frequency_values_produce_absent_metrics() -> None:
    result = characterize_spectral_track(
        _track((np.nan, np.inf), refined=(None, None))
    )
    assert result.frequency_finite_point_count == 0
    assert result.frequency_discarded_point_count == 2
    assert result.frequency_mean_hz is None
    assert result.frequency_std_hz is None
    assert result.relative_frequency_stability is None
    assert not result.frequency_fit.success
    assert result.frequency_fit.failure_reason == "insufficient_frequency_points"


def test_single_frequency_point_has_structured_failed_fit() -> None:
    result = characterize_spectral_track(_track((80.0,)))
    fit = result.frequency_fit
    assert not fit.success
    assert fit.failure_reason == "insufficient_frequency_points"
    assert fit.slope_hz_per_s is fit.intercept_hz is fit.rmse_hz is None
    assert "insufficient_frequency_points" in result.diagnostics


def test_equal_times_fail_frequency_fit_without_nan() -> None:
    fit, diagnostics = _fit_frequency(
        np.asarray((1.0, 1.0)),
        np.asarray((100.0, 101.0)),
        available_point_count=2,
    )
    assert not fit.success
    assert fit.failure_reason == "non_distinct_frequency_times"
    assert "non_distinct_frequency_times" in diagnostics


def test_zero_frequency_median_has_no_relative_stability() -> None:
    result = characterize_spectral_track(
        _track((0.0, 0.0), refined=(None, None))
    )
    assert result.frequency_median_hz == 0.0
    assert result.relative_frequency_stability is None
    assert "zero_frequency_median" in result.diagnostics


@pytest.mark.parametrize(
    ("amplitudes", "slope", "fractions"),
    [
        ((3.0, 2.0, 1.0), -1.0, (0.0, 1.0, 0.0)),
        ((1.0, 2.0, 3.0), 1.0, (1.0, 0.0, 0.0)),
        ((2.0, 2.0, 2.0), 0.0, (0.0, 0.0, 1.0)),
        ((1.0, 2.0, 1.0, 2.0), 0.2, (2 / 3, 1 / 3, 0.0)),
    ],
    ids=["decreasing", "increasing", "constant", "oscillating"],
)
def test_amplitude_metrics_and_variation_fractions(
    amplitudes, slope, fractions
) -> None:
    result = characterize_spectral_track(
        _track(tuple(100.0 for _ in amplitudes), amplitudes=amplitudes)
    )
    assert result.amplitude_initial == pytest.approx(amplitudes[0])
    assert result.amplitude_final == pytest.approx(amplitudes[-1])
    assert result.amplitude_mean == pytest.approx(np.mean(amplitudes))
    assert result.amplitude_median == pytest.approx(np.median(amplitudes))
    assert result.amplitude_min == pytest.approx(min(amplitudes))
    assert result.amplitude_max == pytest.approx(max(amplitudes))
    assert result.amplitude_std == pytest.approx(np.std(amplitudes))
    assert result.amplitude_peak_to_peak == pytest.approx(
        max(amplitudes) - min(amplitudes)
    )
    assert result.amplitude_slope_per_s == pytest.approx(slope, abs=1e-12)
    actual = (
        result.amplitude_increase_fraction,
        result.amplitude_decrease_fraction,
        result.amplitude_constant_fraction,
    )
    assert actual == pytest.approx(fractions)
    assert sum(actual) == pytest.approx(1.0)


def test_dbfs_amplitude_metrics_remain_in_level_domain() -> None:
    result = characterize_spectral_track(
        _track(
            (100.0, 100.0, 100.0),
            amplitudes=(-3.0, -6.0, -9.0),
            unit="dbfs_amplitude",
        )
    )
    assert result.amplitude_unit == "dbfs_amplitude"
    assert result.amplitude_mean == pytest.approx(-6.0)
    assert result.amplitude_slope_per_s == pytest.approx(-3.0)
    assert result.amplitude_decrease_fraction == 1.0
    assert result.amplitude_fit.fit_domain == "dbfs"


def test_nonfinite_amplitudes_are_filtered_from_all_descriptive_metrics() -> None:
    result = characterize_spectral_track(
        _track(
            (100.0,) * 5,
            amplitudes=(1.0, np.nan, np.inf, 0.5, 0.25),
        )
    )
    assert (
        result.amplitude_available_point_count,
        result.amplitude_finite_point_count,
        result.amplitude_discarded_point_count,
    ) == (5, 3, 2)
    assert result.amplitude_mean == pytest.approx(7 / 12)
    assert result.amplitude_median == pytest.approx(0.5)
    assert result.amplitude_decrease_fraction == 1.0
    assert result.amplitude_fit.finite_point_count == 3
    assert "nonfinite_amplitude_values_discarded" in result.diagnostics


def test_all_nonfinite_amplitudes_produce_absent_metrics() -> None:
    result = characterize_spectral_track(
        _track((100.0, 101.0), amplitudes=(np.nan, np.inf))
    )
    assert result.amplitude_finite_point_count == 0
    assert result.amplitude_discarded_point_count == 2
    assert result.amplitude_mean is None
    assert result.amplitude_std is None
    assert result.amplitude_increase_fraction is None
    assert not result.amplitude_fit.success
    assert result.amplitude_fit.failure_reason == "insufficient_points"


def test_single_amplitude_point_has_no_slope_or_variation_fractions() -> None:
    result = characterize_spectral_track(_track((100.0,), amplitudes=(0.5,)))
    assert result.amplitude_initial == result.amplitude_final == 0.5
    assert result.amplitude_slope_per_s is None
    assert result.amplitude_increase_fraction is None
    assert result.amplitude_decrease_fraction is None
    assert result.amplitude_constant_fraction is None
    assert "insufficient_amplitude_points" in result.diagnostics


@pytest.mark.parametrize(
    ("frames", "analysis_final", "expected"),
    [
        ((0, 1, 2, 3), 3, (4, 1.0, 0, 0, 0, True)),
        ((0, 2, 3), 3, (4, 0.75, 1, 1, 1, True)),
        ((0, 2, 5, 6), 8, (7, 4 / 7, 2, 3, 2, False)),
        ((0, 3, 7), 7, (8, 3 / 8, 2, 5, 3, True)),
        ((0, 7), 9, (8, 0.25, 1, 6, 6, False)),
    ],
    ids=["complete", "one_gap", "multiple_gaps", "consecutive_gaps", "endpoints_only"],
)
def test_temporal_coverage_and_gap_metrics(frames, analysis_final, expected) -> None:
    times = tuple(frame * 0.25 for frame in frames)
    result = characterize_spectral_track(
        _track(
            tuple(100.0 for _ in frames),
            frames=frames,
            times=times,
            analysis_final_frame=analysis_final,
        )
    )
    span, coverage, gaps, missing, largest, reaches = expected
    assert (result.first_frame, result.last_frame) == (frames[0], frames[-1])
    assert (result.start_time_s, result.end_time_s) == pytest.approx(
        (times[0], times[-1])
    )
    assert result.observed_duration_s == pytest.approx(times[-1] - times[0])
    assert result.observation_count == len(frames)
    assert result.frame_span_count == span
    assert result.coverage_fraction == pytest.approx(coverage)
    assert (result.gap_count, result.total_missing_frames) == (gaps, missing)
    assert result.largest_gap_frames == largest
    assert result.reached_analysis_final_frame is reaches


def _valid_characterization():
    return characterize_spectral_track(
        _track(
            (100.0, 101.0, 102.0),
            amplitudes=(3.0, 2.0, 1.0),
            analysis_final_frame=2,
        )
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"frequency_min_hz": 103.0}, "frequency bounds"),
        ({"frequency_mean_hz": 200.0}, "frequency mean"),
        ({"frequency_peak_to_peak_hz": 3.0}, "peak-to-peak"),
        ({"frequency_total_drift_hz": 3.0}, "drift"),
        ({"relative_frequency_stability": -0.1}, "stability"),
        ({"frequency_source": "unknown"}, "frequency_source"),
        ({"amplitude_min": 4.0}, "amplitude bounds"),
        ({"amplitude_mean": 9.0}, "amplitude mean"),
        ({"amplitude_increase_fraction": 1.1}, "fractions"),
        ({"amplitude_constant_fraction": 0.2}, "sum to one"),
        ({"coverage_fraction": -0.1}, "coverage"),
        ({"coverage_fraction": 1.1}, "coverage"),
        ({"observed_duration_s": -1.0}, "duration"),
        ({"largest_gap_frames": 1}, "gap"),
        ({"frequency_mean_hz": float("nan")}, "finite"),
    ],
    ids=[
        "frequency_bounds", "frequency_mean", "frequency_peak_to_peak",
        "frequency_drift", "negative_stability", "unknown_source",
        "amplitude_bounds", "amplitude_mean", "fraction_out_of_range",
        "fraction_sum", "negative_coverage", "coverage_above_one",
        "negative_duration", "incoherent_gaps", "nonfinite_metric",
    ],
)
def test_characterization_rejects_invalid_invariants(changes, message) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_valid_characterization(), **changes)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"rmse_hz": -1.0}, "rmse"),
        ({"method": "other"}, "required"),
        ({"used_point_count": 1}, "at least two"),
        ({"start_time_s": 3.0}, "inverted"),
        ({"diagnostics": ("repeat", "repeat")}, "duplicates"),
        ({"slope_hz_per_s": float("nan")}, "finite"),
    ],
    ids=["negative_rmse", "method", "too_few_used", "inverted_time", "diagnostics", "nonfinite"],
)
def test_track_frequency_fit_rejects_invalid_success_contract(changes, message) -> None:
    fit = _valid_characterization().frequency_fit
    with pytest.raises(ValueError, match=message):
        replace(fit, **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"failure_reason": None},
        {"slope_hz_per_s": 1.0},
        {"rmse_hz": 0.0},
    ],
    ids=["missing_reason", "regression_slope", "regression_rmse"],
)
def test_failed_frequency_fit_cannot_carry_success_state(changes) -> None:
    failed = characterize_spectral_track(_track((100.0,))).frequency_fit
    with pytest.raises(ValueError):
        replace(failed, **changes)


def test_characterization_rejects_incompatible_fit_counts_and_units() -> None:
    valid = _valid_characterization()
    incompatible_frequency = TrackFrequencyFit(
        True, "linear_frequency_drift", 1.0, 100.0, 1.0, 0.0,
        2, 2, 2, 0, 0.0, 1.0,
    )
    with pytest.raises(ValueError, match="frequency_fit counts"):
        replace(valid, frequency_fit=incompatible_frequency)
    dbfs = characterize_spectral_track(
        _track((100.0, 101.0, 102.0), amplitudes=(-1.0, -2.0, -3.0), unit="dbfs_amplitude")
    )
    with pytest.raises(ValueError, match="amplitude_fit unit"):
        replace(valid, amplitude_fit=dbfs.amplitude_fit)


def test_complete_characterization_is_reproducible() -> None:
    track = _track(
        (100.0, 101.2, 101.8, 103.1),
        refined=(100.1, None, 101.9, 103.0),
        amplitudes=(1.0, 0.8, 0.81, 0.6),
        frames=(0, 1, 3, 4),
        analysis_final_frame=4,
    )
    first = characterize_spectral_track(track)
    second = characterize_spectral_track(track)
    assert first == second
    assert first.frequency_fit == second.frequency_fit
    assert first.amplitude_fit == second.amplitude_fit
    assert first.diagnostics == second.diagnostics
