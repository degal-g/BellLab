"""Testes quantitativos da caracterização espectral global."""

from __future__ import annotations

import numpy as np
import pytest

from belllab import (
    GlobalSpectralCharacterizationSettings,
    Signal,
    SpectralBand,
    Spectrum,
    characterize_global_spectrum,
    characterize_signal_spectrum,
    evaluate_spectral_characterization_comparability,
)
from belllab.synthetic import ideal_impulse, mix_signal_and_noise, pure_sine, sine_sum, white_noise


def controlled(power: tuple[float, ...], *, start_hz: float = 0.0) -> Spectrum:
    frequencies = tuple(start_hz + index for index in range(len(power)))
    return Spectrum(
        frequencies_hz=frequencies,
        magnitudes=power,
        magnitude_unit="linear power",
        window_name="rectangular",
        fft_size=max(2, 2 * (len(power) - 1)),
        sample_rate_hz=max(2, 2 * (len(power) - 1)),
        original_size=max(2, 2 * (len(power) - 1)),
        bin_spacing_hz=1.0,
        normalization="test_power",
        interval_start_s=0.0,
        interval_end_s=1.0,
        remove_mean=False,
    )


def settings(**kwargs: object) -> GlobalSpectralCharacterizationSettings:
    return GlobalSpectralCharacterizationSettings(
        spectral_input_domain="linear_power",
        minimum_bin_count=3,
        **kwargs,
    )


def test_uniform_power_exact_distribution_metrics() -> None:
    result = characterize_global_spectrum(controlled((1, 1, 1, 1, 1)), settings())

    assert result.spectral_centroid_hz == pytest.approx(2.0)
    assert result.spectral_variance_hz2 == pytest.approx(2.0)
    assert result.spectral_spread_hz == pytest.approx(np.sqrt(2))
    assert result.spectral_skewness == pytest.approx(0.0)
    assert result.spectral_kurtosis == pytest.approx(1.7)
    assert result.spectral_flatness == pytest.approx(1.0)
    assert result.spectral_entropy == pytest.approx(1.0)
    assert result.spectral_crest_factor == pytest.approx(1.0)


def test_single_bin_energy_and_undefined_shape_metrics() -> None:
    result = characterize_global_spectrum(controlled((0, 0, 4, 0, 0)), settings())

    assert result.spectral_centroid_hz == 2
    assert result.spectral_spread_hz == 0
    assert result.spectral_skewness is None
    assert result.spectral_kurtosis is None
    assert result.spectral_flatness is None
    assert result.spectral_entropy == 0
    assert result.spectral_crest_factor == 5


@pytest.mark.parametrize(
    ("power", "expected_centroid"),
    [
        ((0, 1, 0, 1, 0), 2.0),
        ((1, 2, 3, 2, 1), 2.0),
        ((1, 2, 3, 4, 5), 8.0 / 3.0),
        ((1, 1, 8, 1, 1), 2.0),
    ],
)
def test_controlled_centroids(power: tuple[float, ...], expected_centroid: float) -> None:
    result = characterize_global_spectrum(controlled(power), settings())
    assert result.spectral_centroid_hz == pytest.approx(expected_centroid)


def test_rolloff_uses_first_inclusive_bin_without_interpolation() -> None:
    result = characterize_global_spectrum(controlled((1, 1, 1, 1)), settings())

    assert result.rolloff(0.50) == 1
    assert result.rolloff(0.85) == 3
    assert result.rolloff(0.90) == 3
    assert result.rolloff(0.95) == 3


def test_rolloff_exact_threshold_is_inclusive() -> None:
    result = characterize_global_spectrum(controlled((1, 1, 2)), settings())
    assert result.rolloff(0.50) == 1


def test_zero_spectrum_has_explicit_failure_without_sentinels() -> None:
    result = characterize_global_spectrum(controlled((0, 0, 0, 0)), settings())

    assert not result.valid
    assert result.failure_reason == "zero_total_spectral_energy"
    assert result.spectral_centroid_hz is None
    assert result.spectral_flatness is None
    assert result.spectral_entropy is None
    assert result.tonal_energy_fraction is None


def test_nonfinite_bins_are_discarded_and_counted() -> None:
    result = characterize_global_spectrum(
        controlled((1, float("inf"), 1, float("-inf"), 2)), settings()
    )
    assert result.finite_bin_count == 3
    assert result.discarded_bin_count == 2
    assert all(np.isfinite(value) for value in (result.total_spectral_energy,))


def test_negative_linear_power_is_rejected() -> None:
    with pytest.raises(ValueError, match="negative"):
        characterize_global_spectrum(controlled((1, -1, 1)), settings())


@pytest.mark.parametrize("scale", [0.25, 3.0, 100.0])
def test_flatness_and_entropy_are_scale_invariant(scale: float) -> None:
    base = np.asarray((1.0, 2.0, 4.0, 8.0))
    first = characterize_global_spectrum(controlled(tuple(base)), settings())
    scaled = characterize_global_spectrum(controlled(tuple(scale * base)), settings())
    assert scaled.spectral_flatness == pytest.approx(first.spectral_flatness)
    assert scaled.spectral_entropy == pytest.approx(first.spectral_entropy)


def test_flatness_ignores_zeros_without_hidden_epsilon() -> None:
    result = characterize_global_spectrum(controlled((0, 1, 1, 0)), settings())
    assert result.spectral_flatness == pytest.approx(1.0)
    assert result.zero_bin_count == 2
    assert "flatness:positive_bins_only_no_epsilon" in result.diagnostics


def test_peak_density_spacing_and_tonal_fraction() -> None:
    result = characterize_global_spectrum(
        controlled((0, 1, 0, 2, 0, 1, 0), start_hz=1),
        settings(peak_min_prominence=0.5),
    )
    assert result.significant_peak_count == 3
    assert result.minimum_peak_spacing_hz == pytest.approx(2)
    assert result.maximum_peak_spacing_hz == pytest.approx(2)
    assert result.median_peak_spacing_hz == pytest.approx(2)
    assert result.peak_spacing_standard_deviation_hz == pytest.approx(0)
    assert result.peak_density_per_hz == pytest.approx(0.5)
    assert result.peak_density_per_octave is not None
    assert 0 <= (result.tonal_energy_fraction or 0) <= 1
    assert (result.tonal_energy_fraction or 0) + (result.residual_energy_fraction or 0) == pytest.approx(1)


@pytest.mark.parametrize("power", [(0, 0, 1, 0, 0), (0, 1, 0, 0, 0)])
def test_fewer_than_two_peaks_has_no_spacing_statistics(power: tuple[int, ...]) -> None:
    result = characterize_global_spectrum(controlled(power), settings())
    assert result.mean_peak_spacing_hz is None
    assert result.median_peak_spacing_hz is None


def test_peak_width_boundaries_and_resolution_diagnostic() -> None:
    result = characterize_global_spectrum(
        controlled((0, 1, 4, 1, 0)), settings(peak_min_prominence=1)
    )
    peak = result.peak_metrics[0]
    assert peak.left_frequency_hz <= peak.representative_frequency_hz <= peak.right_frequency_hz
    assert peak.width_bins > 0
    assert peak.width_hz == pytest.approx(peak.width_bins)
    assert peak.resolution_limited == (peak.width_hz <= result.frequency_resolution_hz)


def test_tonal_intervals_do_not_double_count_energy() -> None:
    result = characterize_global_spectrum(
        controlled((0, 1, 3, 2, 3, 1, 0)),
        settings(peak_min_prominence=0.5, tonal_neighborhood_width_factor=4),
    )
    assert result.tonal_energy <= result.total_spectral_energy
    assert result.residual_energy >= 0


def test_no_peaks_means_all_energy_is_residual() -> None:
    result = characterize_global_spectrum(controlled((1, 1, 1, 1)), settings())
    assert result.significant_peak_count == 0
    assert result.tonal_energy_fraction == 0
    assert result.residual_energy_fraction == 1


def test_adjacent_band_energy_is_conservative_without_boundary_double_count() -> None:
    cfg = settings(bands=(
        SpectralBand("low", 0, 2),
        SpectralBand("high", 2, 5),
    ))
    result = characterize_global_spectrum(controlled((1, 2, 3, 4, 5)), cfg)
    low, high = result.band_energy_metrics
    assert low.energy == 3
    assert high.energy == 12
    assert low.bin_count == 2
    assert high.bin_count == 3
    assert low.energy + high.energy == result.total_spectral_energy
    assert low.energy_fraction + high.energy_fraction == pytest.approx(1)


@pytest.mark.parametrize(
    "bands",
    [
        (SpectralBand("a", 0, 2), SpectralBand("b", 1, 3)),
        (SpectralBand("b", 2, 3), SpectralBand("a", 0, 2)),
    ],
)
def test_overlapping_or_unordered_bands_are_rejected(bands: tuple[SpectralBand, ...]) -> None:
    with pytest.raises(ValueError):
        settings(bands=bands)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frequency_min_hz": -1},
        {"frequency_min_hz": 2, "frequency_max_hz": 1},
        {"fft_size": 0},
        {"rolloff_fractions": (0.5, 0.5)},
        {"rolloff_fractions": (0.0, 0.5)},
        {"minimum_bin_count": 0},
        {"tonal_neighborhood_width_factor": -1},
        {"power_reference": float("nan")},
        {"window_name": "blackman"},
        {"detrend_policy": "linear"},
    ],
)
def test_invalid_settings_fail_early(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        GlobalSpectralCharacterizationSettings(**kwargs)


def test_dbfs_is_explicitly_recovered_to_linear_power() -> None:
    spectrum = controlled((1, 1, 1))
    db = Spectrum(
        **{
            field: getattr(spectrum, field)
            for field in spectrum.__dataclass_fields__
            if field not in {"magnitudes", "magnitude_unit"}
        },
        magnitudes=(0.0, -6.020599913279624, -float("inf")),
        magnitude_unit="dBFS amplitude (ref=1.0)",
    )
    result = characterize_global_spectrum(db, GlobalSpectralCharacterizationSettings())
    assert result.original_spectral_domain == "dbfs_amplitude"
    assert result.total_spectral_energy == pytest.approx(1.25)


def test_sine_noise_and_mixture_have_expected_descriptive_order() -> None:
    tone = pure_sine(128, duration_s=1, sample_rate=2048, amplitude=0.8)
    noise = white_noise(duration_s=1, sample_rate=2048, rms_amplitude=0.1, seed=17)
    mixture = mix_signal_and_noise(tone, noise)
    cfg = GlobalSpectralCharacterizationSettings(peak_min_prominence=1e-4)
    tone_result = characterize_signal_spectrum(tone, cfg, recording_id="tone")
    noise_result = characterize_signal_spectrum(noise, cfg, recording_id="noise")
    mixed_result = characterize_signal_spectrum(mixture, cfg, recording_id="mixed")
    assert tone_result.spectral_flatness < mixed_result.spectral_flatness < noise_result.spectral_flatness
    assert tone_result.spectral_entropy < mixed_result.spectral_entropy < noise_result.spectral_entropy


def test_multiple_sines_produce_multiple_significant_peaks() -> None:
    signal = sine_sum(
        (100, 300, 600), amplitudes=(1, 0.5, 0.25),
        duration_s=1, sample_rate=2048,
    )
    result = characterize_signal_spectrum(
        signal,
        GlobalSpectralCharacterizationSettings(peak_min_prominence=1e-3),
    )
    assert [round(peak.representative_frequency_hz) for peak in result.peak_metrics] == [100, 300, 600]


def test_two_separated_sines_produce_two_peaks() -> None:
    result = characterize_signal_spectrum(
        sine_sum((100, 400), duration_s=1, sample_rate=2048),
        GlobalSpectralCharacterizationSettings(peak_min_prominence=1e-3),
    )
    assert [round(peak.representative_frequency_hz) for peak in result.peak_metrics] == [100, 400]


def test_impulse_has_broader_distribution_than_sine() -> None:
    impulse = characterize_signal_spectrum(
        ideal_impulse(duration_s=1, sample_rate=1024, sample_index=128)
    )
    sine = characterize_signal_spectrum(
        pure_sine(128, duration_s=1, sample_rate=1024)
    )
    assert impulse.spectral_entropy > sine.spectral_entropy
    assert impulse.occupied_bandwidth_hz > sine.occupied_bandwidth_hz


@pytest.mark.parametrize(("offset", "clip"), [(0.5, False), (0.0, True)])
def test_offset_and_clipped_signals_remain_descriptive(offset: float, clip: bool) -> None:
    original = pure_sine(64, duration_s=1, sample_rate=1024, amplitude=0.8)
    samples = np.asarray(original.samples[0]) + offset
    if clip:
        samples = np.clip(2 * samples, -1, 1)
    signal = Signal(
        samples=(tuple(float(value) for value in samples),),
        sample_rate=original.sample_rate,
        time=original.time,
        duration=original.duration,
        channels=1,
        unit="normalized",
    )
    result = characterize_signal_spectrum(signal)
    assert result.valid
    assert result.spectral_centroid_hz is not None
    assert "metricas_espectrais_globais_nao_sao_diagnostico_fisico" in result.diagnostics


def test_silent_time_signal_returns_explicit_invalid_result() -> None:
    result = characterize_signal_spectrum(
        pure_sine(64, duration_s=1, sample_rate=1024, amplitude=0)
    )
    assert not result.valid
    assert result.failure_reason == "zero_total_spectral_energy"


def test_zero_padding_changes_bin_spacing_but_not_physical_resolution() -> None:
    signal = pure_sine(64, duration_s=1, sample_rate=1024)
    raw = characterize_signal_spectrum(signal)
    padded = characterize_signal_spectrum(
        signal, GlobalSpectralCharacterizationSettings(fft_size=4096)
    )
    assert padded.bin_spacing_hz < raw.bin_spacing_hz
    assert padded.frequency_resolution_hz == raw.frequency_resolution_hz == 1


def test_comparability_reports_each_material_configuration_difference() -> None:
    signal = pure_sine(64, duration_s=1, sample_rate=1024)
    first = characterize_signal_spectrum(signal)
    second = characterize_signal_spectrum(
        signal,
        GlobalSpectralCharacterizationSettings(
            fft_size=2048,
            window_name="rectangular",
            detrend_policy="none",
            peak_min_prominence=0.1,
        ),
    )
    result = evaluate_spectral_characterization_comparability(first, second)
    assert not result.comparable
    assert {
        "incompatible_fft_size", "incompatible_window", "incompatible_detrending",
        "incompatible_peak_criteria",
    } <= set(result.incompatibilities)


def test_identical_characterizations_are_comparable_and_deterministic() -> None:
    signal = white_noise(duration_s=1, sample_rate=1024, rms_amplitude=0.1, seed=7)
    first = characterize_signal_spectrum(signal, recording_id="fixed")
    second = characterize_signal_spectrum(signal, recording_id="fixed")
    assert first == second
    assert evaluate_spectral_characterization_comparability(first, second).comparable
