"""Quantitative tests for time-resolved spectral characterization."""

from __future__ import annotations

import numpy as np
import pytest

from belllab import (
    GlobalSpectralCharacterizationSettings,
    Signal,
    SpectralBand,
    TimeResolvedSpectralCharacterizationSettings,
    characterize_signal_spectrum,
    characterize_time_resolved_spectrum,
    evaluate_time_resolved_spectral_comparability,
)
from belllab.synthetic import damped_exponential, sine_sum


def _signal(samples: np.ndarray, sample_rate: int) -> Signal:
    return Signal(
        samples=(tuple(float(value) for value in samples),),
        sample_rate=sample_rate,
        time=tuple(index / sample_rate for index in range(samples.size)),
        duration=samples.size / sample_rate,
        channels=1,
        unit="normalized",
    )


def _sine_segment(
    frequency_hz: float,
    *,
    sample_rate: int,
    sample_count: int,
    amplitude: float = 1.0,
) -> np.ndarray:
    time = np.arange(sample_count) / sample_rate
    return amplitude * np.sin(2.0 * np.pi * frequency_hz * time)


def _piecewise_sines(
    frequencies: tuple[float, ...],
    *,
    sample_rate: int = 2048,
    sample_count: int = 512,
) -> Signal:
    samples = np.concatenate([
        _sine_segment(frequency, sample_rate=sample_rate, sample_count=sample_count)
        for frequency in frequencies
    ])
    return _signal(samples, sample_rate)


def _settings(**kwargs: object) -> TimeResolvedSpectralCharacterizationSettings:
    defaults = {
        "frame_duration_s": 0.125,
        "hop_duration_s": 0.125,
        "fft_size": 256,
        "window_name": "rectangular",
        "detrend_policy": "none",
        "frequency_min_hz": 16.0,
        "frequency_max_hz": 900.0,
        "peak_min_prominence": 1e-3,
        "early_window_s": (0.0, 0.25),
        "middle_window_s": (0.25, 0.55),
        "late_window_s": (0.75, 1.05),
    }
    defaults.update(kwargs)
    return TimeResolvedSpectralCharacterizationSettings(**defaults)


def _trend(result, metric: str):
    return next(item for item in result.temporal_trends if item.metric_name == metric)


def test_single_frame_matches_global_characterization() -> None:
    signal = sine_sum(
        (64.0, 128.0),
        amplitudes=(0.8, 0.4),
        duration_s=0.25,
        sample_rate=1024,
    )
    time_settings = TimeResolvedSpectralCharacterizationSettings(
        analysis_window_end_s=0.25,
        frame_duration_s=0.25,
        hop_duration_s=0.25,
        fft_size=256,
        window_name="rectangular",
        detrend_policy="none",
        frequency_min_hz=16,
        frequency_max_hz=400,
        peak_min_prominence=1e-3,
        early_window_s=(0.0, 0.25),
        middle_window_s=None,
        late_window_s=None,
    )
    global_settings = GlobalSpectralCharacterizationSettings(
        start_time_s=0.0,
        end_time_s=0.25,
        fft_size=256,
        window_name="rectangular",
        detrend_policy="none",
        frequency_min_hz=16,
        frequency_max_hz=400,
        peak_min_prominence=1e-3,
    )

    time_resolved = characterize_time_resolved_spectrum(signal, 0.0, time_settings)
    global_result = characterize_signal_spectrum(signal, global_settings)
    frame = time_resolved.frames[0]

    assert time_resolved.frame_count == 1
    assert frame.spectral_energy == pytest.approx(global_result.total_spectral_energy)
    assert frame.spectral_centroid_hz == pytest.approx(global_result.spectral_centroid_hz)
    assert frame.spectral_spread_hz == pytest.approx(global_result.spectral_spread_hz)
    assert frame.spectral_rolloff_50_hz == pytest.approx(global_result.spectral_rolloff_50_hz)
    assert frame.spectral_rolloff_85_hz == pytest.approx(global_result.spectral_rolloff_85_hz)
    assert frame.spectral_rolloff_95_hz == pytest.approx(global_result.spectral_rolloff_95_hz)
    assert frame.spectral_flatness == pytest.approx(global_result.spectral_flatness)
    assert frame.spectral_entropy == pytest.approx(global_result.spectral_entropy)
    assert frame.spectral_crest_factor == pytest.approx(global_result.spectral_crest_factor)
    assert frame.significant_peak_count == global_result.significant_peak_count
    assert frame.tonal_energy_fraction == pytest.approx(global_result.tonal_energy_fraction)
    assert frame.occupied_bandwidth_hz == pytest.approx(global_result.occupied_bandwidth_hz)


def test_damped_sine_has_decreasing_energy_and_stable_dominant_frequency() -> None:
    signal = damped_exponential(128, 3, duration_s=1.0, sample_rate=2048)
    result = characterize_time_resolved_spectrum(signal, 0.0, _settings())

    energies = [frame.spectral_energy for frame in result.frames]
    dominant = [frame.dominant_frequency_hz for frame in result.frames]

    assert all(left > right for left, right in zip(energies, energies[1:]))
    assert dominant == [128.0] * len(dominant)
    assert max(frame.spectral_centroid_hz for frame in result.frames) - min(
        frame.spectral_centroid_hz for frame in result.frames
    ) < 1e-9
    assert max(frame.spectral_flatness for frame in result.frames) < 0.001
    assert _trend(result, "energy_db").slope_per_s < 0
    assert "time_resolved_peaks_are_not_modal_modes" in result.diagnostics


def test_decreasing_white_noise_has_higher_flatness_and_entropy_than_sine() -> None:
    sample_rate = 2048
    time = np.arange(sample_rate) / sample_rate
    rng = np.random.default_rng(123)
    noise = _signal(0.2 * np.exp(-3 * time) * rng.normal(size=time.size), sample_rate)
    sine = damped_exponential(128, 3, duration_s=1.0, sample_rate=sample_rate)
    settings = _settings()

    noise_result = characterize_time_resolved_spectrum(noise, 0.0, settings)
    sine_result = characterize_time_resolved_spectrum(sine, 0.0, settings)

    assert noise_result.summary.initial_flatness > sine_result.summary.initial_flatness
    assert noise_result.summary.final_flatness > sine_result.summary.final_flatness
    assert noise_result.summary.initial_entropy > sine_result.summary.initial_entropy
    assert noise_result.summary.final_entropy > sine_result.summary.final_entropy
    assert _trend(noise_result, "energy_db").slope_per_s < 0
    assert characterize_time_resolved_spectrum(noise, 0.0, settings) == noise_result


def test_broadband_attack_followed_by_tonal_tail_is_described_without_regime_label() -> None:
    sample_rate = 2048
    time = np.arange(sample_rate) / sample_rate
    rng = np.random.default_rng(12)
    attack_mask = time < 0.22
    attack = np.zeros_like(time)
    attack[attack_mask] = 0.6 * rng.normal(size=int(np.count_nonzero(attack_mask)))
    tail = np.where(
        time >= 0.15,
        0.7 * np.exp(-2 * (time - 0.15)) * np.sin(2 * np.pi * 128 * time),
        0.0,
    )
    settings = _settings(
        hop_duration_s=0.0625,
        early_window_s=(0.0, 0.20),
        middle_window_s=(0.20, 0.50),
        late_window_s=(0.50, 0.95),
        change_point_thresholds=(
            ("spectral_entropy", 0.15),
            ("spectral_flatness", 0.10),
            ("tonal_energy_fraction", 0.20),
        ),
        change_point_window_frames=1,
        change_point_minimum_persistence_frames=1,
    )

    result = characterize_time_resolved_spectrum(_signal(attack + tail, sample_rate), 0.0, settings)
    early, _, late = result.summary.regions

    assert early.median_flatness > late.median_flatness
    assert early.median_entropy > late.median_entropy
    assert late.median_tonal_energy_fraction > early.median_tonal_energy_fraction
    assert late.median_residual_energy_fraction < early.median_residual_energy_fraction
    assert any(0.18 <= point.time_s <= 0.35 for point in result.change_points)
    assert "evolucao_metricas_espectrais_nao_e_transicao_fisica_comprovada" in result.diagnostics


def test_tonal_attack_and_noisy_tail_can_change_in_the_opposite_direction() -> None:
    sample_rate = 2048
    time = np.arange(sample_rate) / sample_rate
    rng = np.random.default_rng(90)
    tonal_attack = np.where(time < 0.45, np.sin(2 * np.pi * 128 * time), 0.0)
    noisy_tail = np.where(time >= 0.45, 0.25 * rng.normal(size=time.size), 0.0)
    settings = _settings(
        hop_duration_s=0.0625,
        early_window_s=(0.0, 0.25),
        middle_window_s=(0.25, 0.55),
        late_window_s=(0.65, 0.95),
    )

    result = characterize_time_resolved_spectrum(_signal(tonal_attack + noisy_tail, sample_rate), 0.0, settings)

    assert result.summary.flatness_change > 0
    assert result.summary.entropy_change > 0
    assert result.summary.tonal_fraction_change < 0
    assert result.summary.residual_fraction_change > 0


def test_high_frequency_component_decays_faster_and_centroid_descends() -> None:
    sample_rate = 2048
    time = np.arange(sample_rate) / sample_rate
    samples = (
        0.6 * np.exp(-1 * time) * np.sin(2 * np.pi * 128 * time)
        + 0.8 * np.exp(-8 * time) * np.sin(2 * np.pi * 640 * time)
    )
    settings = _settings(
        bands=(
            SpectralBand("low", 64, 256),
            SpectralBand("high", 512, 768),
        )
    )

    result = characterize_time_resolved_spectrum(_signal(samples, sample_rate), 0.0, settings)
    low, high = result.summary.band_summaries

    assert result.summary.final_centroid_hz < result.summary.initial_centroid_hz
    assert result.frames[-1].spectral_rolloff_95_hz < result.frames[0].spectral_rolloff_95_hz
    assert _trend(result, "spectral_centroid_hz").slope_per_s < 0
    assert high.final_energy_fraction < high.initial_energy_fraction
    assert high.energy_fraction_trend.slope_per_s < 0
    assert low.final_energy_fraction > low.initial_energy_fraction


def test_peak_density_tracks_successive_component_counts() -> None:
    sample_rate = 2048
    count = 512
    rng = np.random.default_rng(4)
    time = np.arange(count) / sample_rate
    many = sum(np.sin(2 * np.pi * frequency * time) / 6 for frequency in (96, 160, 224, 288, 352, 416))
    samples = np.concatenate([
        _sine_segment(128, sample_rate=sample_rate, sample_count=count),
        sum(_sine_segment(frequency, sample_rate=sample_rate, sample_count=count) / 3 for frequency in (128, 256, 384)),
        many,
        0.2 * rng.normal(size=count),
    ])
    settings = TimeResolvedSpectralCharacterizationSettings(
        frame_duration_s=count / sample_rate,
        hop_duration_s=count / sample_rate,
        fft_size=count,
        window_name="rectangular",
        detrend_policy="none",
        frequency_min_hz=32,
        frequency_max_hz=800,
        peak_min_prominence=1e-3,
        early_window_s=(0.0, 0.30),
        middle_window_s=(0.30, 0.70),
        late_window_s=(0.70, 1.10),
    )

    result = characterize_time_resolved_spectrum(_signal(samples, sample_rate), 0.0, settings)

    assert [frame.significant_peak_count for frame in result.frames[:3]] == [1, 3, 6]
    assert result.frames[0].peak_density_per_hz < result.frames[1].peak_density_per_hz
    assert result.frames[1].peak_density_per_hz < result.frames[2].peak_density_per_hz
    assert all(frame.peak_density_per_octave is not None for frame in result.frames)


def test_peak_density_per_octave_is_unavailable_when_range_starts_at_zero() -> None:
    result = characterize_time_resolved_spectrum(
        _piecewise_sines((128.0, 256.0)),
        0.0,
        _settings(frequency_min_hz=0.0),
    )

    assert all(frame.peak_density_per_octave is None for frame in result.frames)


def test_invalid_frames_are_preserved_with_structured_reasons() -> None:
    sample_rate = 1000
    samples = np.concatenate([
        np.zeros(100),
        np.full(100, 1e-5),
        _sine_segment(100, sample_rate=sample_rate, sample_count=100),
        np.full(100, np.nan),
    ])
    settings = TimeResolvedSpectralCharacterizationSettings(
        frame_duration_s=0.1,
        hop_duration_s=0.1,
        fft_size=128,
        window_name="rectangular",
        detrend_policy="none",
        frequency_min_hz=0,
        frequency_max_hz=500,
        minimum_frame_energy=1e-6,
        peak_min_prominence=1e-3,
        early_window_s=(0.0, 0.2),
        middle_window_s=(0.2, 0.3),
        late_window_s=(0.3, 0.4),
    )

    result = characterize_time_resolved_spectrum(_signal(samples, sample_rate), 0.0, settings)

    assert [frame.valid for frame in result.frames] == [False, False, True, False]
    assert [frame.failure_reason for frame in result.frames] == [
        "zero_total_spectral_energy",
        "frame_energy_below_threshold",
        None,
        "nonfinite_frame_samples",
    ]
    assert result.valid_frame_count == 1
    assert result.discarded_frame_count == 3


def test_all_silent_frames_make_result_invalid_without_removing_frames() -> None:
    signal = _signal(np.zeros(512), 1024)
    settings = TimeResolvedSpectralCharacterizationSettings(
        frame_duration_s=0.125,
        hop_duration_s=0.125,
        fft_size=128,
        window_name="rectangular",
        detrend_policy="none",
    )

    result = characterize_time_resolved_spectrum(signal, 0.0, settings)

    assert not result.valid
    assert result.failure_reason == "all_frames_invalid"
    assert result.frame_count == 4
    assert all(not frame.valid for frame in result.frames)


def test_window_outside_signal_reports_no_complete_frames() -> None:
    signal = _signal(np.zeros(256), 1024)
    result = characterize_time_resolved_spectrum(
        signal,
        2.0,
        TimeResolvedSpectralCharacterizationSettings(
            analysis_window_start_s=0.0,
            analysis_window_end_s=0.5,
            frame_duration_s=0.1,
            hop_duration_s=0.1,
        ),
    )

    assert not result.valid
    assert result.failure_reason == "no_complete_frames_in_analysis_window"
    assert result.frames == ()
    assert "analysis_window_outside_signal" in result.diagnostics


def test_partial_final_frame_is_discarded_by_default_and_padding_is_explicit() -> None:
    signal = _signal(np.ones(350), 1000)
    settings = TimeResolvedSpectralCharacterizationSettings(
        frame_duration_s=0.1,
        hop_duration_s=0.1,
        fft_size=128,
        window_name="rectangular",
        detrend_policy="none",
        pad_end=False,
    )
    padded_settings = TimeResolvedSpectralCharacterizationSettings(
        frame_duration_s=0.1,
        hop_duration_s=0.1,
        fft_size=128,
        window_name="rectangular",
        detrend_policy="none",
        pad_end=True,
    )

    discarded = characterize_time_resolved_spectrum(signal, 0.0, settings)
    padded = characterize_time_resolved_spectrum(signal, 0.0, padded_settings)

    assert discarded.frame_count == 3
    assert any(item.startswith("final_incomplete_frame_discarded_samples=") for item in discarded.diagnostics)
    assert padded.frame_count == 4
    assert "temporal_padding_applied" in padded.diagnostics


def test_hop_can_exceed_frame_duration_and_is_diagnostic() -> None:
    signal = _piecewise_sines((128.0, 128.0, 128.0, 128.0))
    result = characterize_time_resolved_spectrum(
        signal,
        0.0,
        _settings(frame_duration_s=0.125, hop_duration_s=0.25),
    )

    assert result.frame_count == 4
    assert "hop_duration_exceeds_frame_duration" in result.diagnostics


def test_temporal_regions_use_valid_frame_medians() -> None:
    signal = _piecewise_sines((128.0, 128.0, 256.0, 256.0))
    settings = TimeResolvedSpectralCharacterizationSettings(
        frame_duration_s=0.25,
        hop_duration_s=0.25,
        fft_size=512,
        window_name="rectangular",
        detrend_policy="none",
        frequency_min_hz=16,
        frequency_max_hz=900,
        peak_min_prominence=1e-3,
        early_window_s=(0.0, 0.30),
        middle_window_s=None,
        late_window_s=(0.70, 1.10),
    )

    result = characterize_time_resolved_spectrum(signal, 0.0, settings)
    early, late = result.summary.regions

    assert early.valid_frame_count == 1
    assert late.valid_frame_count == 1
    assert early.median_centroid_hz == pytest.approx(128.0)
    assert late.median_centroid_hz == pytest.approx(256.0)
    assert result.summary.regions == tuple(result.summary.regions)


def test_empty_and_invalid_regions_are_auditable() -> None:
    signal = _signal(np.zeros(512), 1024)
    settings = TimeResolvedSpectralCharacterizationSettings(
        frame_duration_s=0.125,
        hop_duration_s=0.125,
        fft_size=128,
        early_window_s=(0.0, 0.125),
        middle_window_s=(2.0, 2.5),
        late_window_s=None,
    )

    result = characterize_time_resolved_spectrum(signal, 0.0, settings)
    early, middle = result.summary.regions

    assert not early.valid
    assert "region_contains_no_valid_frames" in early.diagnostics
    assert not middle.valid
    assert "region_contains_no_frames" in middle.diagnostics


def test_centroid_trend_is_exact_for_controlled_piecewise_sines() -> None:
    signal = _piecewise_sines((128.0, 256.0, 384.0), sample_count=512)
    settings = TimeResolvedSpectralCharacterizationSettings(
        frame_duration_s=0.25,
        hop_duration_s=0.25,
        fft_size=512,
        window_name="rectangular",
        detrend_policy="none",
        frequency_min_hz=16,
        frequency_max_hz=900,
        peak_min_prominence=1e-3,
        early_window_s=(0.0, 0.30),
        middle_window_s=(0.30, 0.60),
        late_window_s=(0.60, 0.90),
        minimum_regression_frame_count=3,
    )

    result = characterize_time_resolved_spectrum(signal, 0.0, settings)
    fit = _trend(result, "spectral_centroid_hz")

    assert fit.success
    assert fit.slope_per_s == pytest.approx(512.0)
    assert fit.rmse == pytest.approx(0.0, abs=1e-10)
    assert fit.r_squared == pytest.approx(1.0)


def test_trends_fail_structurally_with_too_few_frames() -> None:
    signal = _piecewise_sines((128.0,), sample_count=512)
    result = characterize_time_resolved_spectrum(
        signal,
        0.0,
        TimeResolvedSpectralCharacterizationSettings(
            frame_duration_s=0.25,
            hop_duration_s=0.25,
            fft_size=512,
            window_name="rectangular",
            detrend_policy="none",
            minimum_regression_frame_count=3,
        ),
    )

    fit = _trend(result, "spectral_centroid_hz")
    assert not fit.success
    assert fit.failure_reason == "insufficient_points"
    assert fit.slope_per_s is None


def test_change_point_policy_is_inclusive_and_operational() -> None:
    signal = _piecewise_sines((128.0, 128.0, 384.0, 384.0), sample_count=512)
    settings = TimeResolvedSpectralCharacterizationSettings(
        frame_duration_s=0.25,
        hop_duration_s=0.25,
        fft_size=512,
        window_name="rectangular",
        detrend_policy="none",
        frequency_min_hz=16,
        frequency_max_hz=900,
        peak_min_prominence=1e-3,
        change_point_thresholds=(("spectral_centroid_hz", 256.0),),
        change_point_window_frames=1,
        change_point_minimum_persistence_frames=1,
    )

    result = characterize_time_resolved_spectrum(signal, 0.0, settings)

    assert len(result.change_points) == 1
    point = result.change_points[0]
    assert point.metric_name == "spectral_centroid_hz"
    assert point.difference == pytest.approx(256.0)
    assert point.direction == "increase"
    assert "operational_change_point_not_physical_regime_transition" in point.diagnostics


def test_isolated_outlier_is_not_persistent_change_point() -> None:
    signal = _piecewise_sines((128.0, 128.0, 384.0, 128.0, 128.0), sample_count=512)
    settings = TimeResolvedSpectralCharacterizationSettings(
        frame_duration_s=0.25,
        hop_duration_s=0.25,
        fft_size=512,
        window_name="rectangular",
        detrend_policy="none",
        frequency_min_hz=16,
        frequency_max_hz=900,
        peak_min_prominence=1e-3,
        change_point_thresholds=(("spectral_centroid_hz", 100.0),),
        change_point_window_frames=2,
        change_point_minimum_persistence_frames=2,
    )

    result = characterize_time_resolved_spectrum(signal, 0.0, settings)

    assert result.change_points == ()


def test_band_summaries_report_persistence_and_threshold_crossing() -> None:
    sample_rate = 2048
    time = np.arange(sample_rate) / sample_rate
    samples = (
        0.6 * np.exp(-1 * time) * np.sin(2 * np.pi * 128 * time)
        + 0.8 * np.exp(-8 * time) * np.sin(2 * np.pi * 640 * time)
    )
    settings = _settings(
        bands=(
            SpectralBand("low", 64, 256),
            SpectralBand("high", 512, 768),
        ),
        band_presence_energy_fraction_threshold=0.05,
    )

    result = characterize_time_resolved_spectrum(_signal(samples, sample_rate), 0.0, settings)
    low, high = result.summary.band_summaries

    assert low.coverage_fraction == pytest.approx(1.0)
    assert low.time_until_below_threshold_s is None
    assert high.coverage_fraction < 1.0
    assert high.time_until_below_threshold_s is not None
    assert high.final_energy_fraction < high.initial_energy_fraction


def test_band_energy_is_conservative_per_frame_when_bands_cover_the_range() -> None:
    signal = _piecewise_sines((128.0, 384.0), sample_count=512)
    settings = TimeResolvedSpectralCharacterizationSettings(
        frame_duration_s=0.25,
        hop_duration_s=0.25,
        fft_size=512,
        window_name="rectangular",
        detrend_policy="none",
        frequency_min_hz=0,
        frequency_max_hz=1024,
        peak_min_prominence=1e-3,
        bands=(SpectralBand("all", 0, 1025),),
    )

    result = characterize_time_resolved_spectrum(signal, 0.0, settings)

    for frame in result.frames:
        assert frame.band_energy_metrics[0].energy == pytest.approx(frame.spectral_energy)
        assert frame.band_energy_metrics[0].energy_fraction == pytest.approx(1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frame_duration_s": 0},
        {"hop_duration_s": 0},
        {"fft_size": 0},
        {"frequency_min_hz": -1},
        {"frequency_min_hz": 100, "frequency_max_hz": 50},
        {"rolloff_fractions": (0.5, 0.5)},
        {"minimum_bin_count": 0},
        {"minimum_frame_energy": -1},
        {"smoothing_method": "median", "smoothing_window_frames": 2},
        {"smoothing_method": "unknown"},
        {"minimum_regression_frame_count": -1},
        {"change_point_window_frames": 0},
        {"change_point_thresholds": (("a", -1.0),)},
        {"early_window_s": (0.3, 0.1)},
        {"early_window_s": (0.0, 0.4), "middle_window_s": (0.3, 0.5)},
        {"window_name": "blackman"},
        {"detrend_policy": "linear"},
    ],
)
def test_invalid_settings_fail_early(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        TimeResolvedSpectralCharacterizationSettings(**kwargs)


def test_fft_size_must_cover_frame_sample_count() -> None:
    signal = _piecewise_sines((128.0,), sample_rate=2048, sample_count=512)
    with pytest.raises(ValueError, match="fft_size"):
        characterize_time_resolved_spectrum(
            signal,
            0.0,
            TimeResolvedSpectralCharacterizationSettings(
                frame_duration_s=0.25,
                hop_duration_s=0.25,
                fft_size=256,
            ),
        )


def test_temporal_comparability_reports_each_material_difference() -> None:
    signal = _piecewise_sines((128.0, 128.0))
    first = characterize_time_resolved_spectrum(signal, 0.0, _settings())
    second = characterize_time_resolved_spectrum(
        signal,
        0.1,
        _settings(
            frame_duration_s=0.25,
            hop_duration_s=0.25,
            fft_size=512,
            window_name="hann",
            detrend_policy="mean",
            frequency_min_hz=32,
            peak_min_prominence=0.1,
            smoothing_method="median",
        ),
    )

    result = evaluate_time_resolved_spectral_comparability(first, second)

    assert not result.comparable
    assert {
        "incompatible_frame_duration",
        "incompatible_hop_duration",
        "incompatible_fft_size",
        "incompatible_window",
        "incompatible_detrending",
        "incompatible_frequency_range",
        "incompatible_impact_time",
        "incompatible_peak_criteria",
        "incompatible_smoothing",
    } <= set(result.incompatibilities)


def test_identical_results_and_comparability_are_deterministic() -> None:
    signal = _piecewise_sines((128.0, 256.0, 128.0))
    settings = _settings(
        smoothing_method="median",
        change_point_thresholds=(("spectral_centroid_hz", 64.0),),
        change_point_window_frames=1,
        change_point_minimum_persistence_frames=1,
    )

    first = characterize_time_resolved_spectrum(signal, 0.0, settings, recording_id="fixed")
    second = characterize_time_resolved_spectrum(signal, 0.0, settings, recording_id="fixed")

    assert first == second
    assert evaluate_time_resolved_spectral_comparability(first, second).comparable
