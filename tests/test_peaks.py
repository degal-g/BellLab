"""Testes científicos para a detecção de observações de picos espectrais."""

from __future__ import annotations

import numpy as np
import pytest

from belllab import (
    PeakDetectionResults,
    PeakDetectionSettings,
    Signal,
    SpectrumAnalysisSettings,
)
from belllab.spectrum import analyze_spectrum, detect_spectral_peaks
from belllab.synthetic import (
    DampedSineComponent,
    damped_exponential_sum,
    ideal_impulse,
    mix_signal_and_noise,
    pink_noise,
    pure_sine,
    sine_sum,
    white_noise,
    with_dc_offset,
)


def peaks_for(signal: Signal, **kwargs: object) -> PeakDetectionResults:
    """Calcula espectro e picos com configuração de detecção local."""
    spectrum = analyze_spectrum(signal).spectrum
    assert spectrum is not None
    return detect_spectral_peaks(spectrum, PeakDetectionSettings(**kwargs))


def test_centered_sine_has_one_peak_with_bin_frequency() -> None:
    """Senoide em bin deve produzir pico matemático na frequência do bin."""
    signal = pure_sine(64.0, duration_s=1.0, sample_rate=1_024, amplitude=0.8)
    result = peaks_for(signal)

    assert result.accepted_count == 1
    peak = result.peaks[0]
    assert peak.bin_frequency_hz == pytest.approx(64.0)
    assert peak.bin_amplitude == pytest.approx(0.8, rel=1e-5)
    assert peak.prominence is not None
    assert peak.width_hz is not None


def test_sub_bin_interpolation_improves_between_bin_frequency() -> None:
    """Parábola logarítmica deve melhorar a estimativa operacional sub-bin."""
    signal = pure_sine(64.35, duration_s=1.0, sample_rate=1_024)
    interpolated = peaks_for(signal, interpolate=True)
    raw = peaks_for(signal, interpolate=False)
    refined_peak = max(interpolated.peaks, key=lambda item: item.bin_amplitude)
    raw_peak = max(raw.peaks, key=lambda item: item.bin_amplitude)
    refined = refined_peak.refined_frequency_hz

    assert refined is not None
    assert abs(refined - 64.35) < abs(raw_peak.bin_frequency_hz - 64.35)


def test_separated_and_different_amplitude_sines_are_ordered_by_amplitude() -> None:
    """Picos bem separados devem manter amplitude e ordenação configuráveis."""
    signal = sine_sum(
        (50.0, 150.0),
        duration_s=1.0,
        sample_rate=1_024,
        amplitudes=(0.8, 0.2),
    )
    result = peaks_for(signal, sort_by="amplitude")

    assert [peak.bin_frequency_hz for peak in result.peaks[:2]] == [50.0, 150.0]


def test_distance_filter_merges_close_sines_operationally() -> None:
    """Distância mínima deve limitar picos próximos sem alegação modal."""
    signal = sine_sum(
        (100.0, 102.0),
        duration_s=1.0,
        sample_rate=1_024,
        amplitudes=(1.0, 0.7),
    )
    result = peaks_for(signal, distance_bins=4, sort_by="frequency")

    assert result.accepted_count == 1


@pytest.mark.parametrize("noise_factory", [white_noise, pink_noise])
def test_noise_floor_and_snr_are_reported_for_noisy_sine(noise_factory: object) -> None:
    """Piso local e SNR devem ser operacionais e finitos na presença de ruído."""
    tone = pure_sine(80.0, duration_s=1.0, sample_rate=1_024, amplitude=0.8)
    noise = noise_factory(  # type: ignore[operator]
        duration_s=1.0,
        sample_rate=1_024,
        rms_amplitude=0.02,
        seed=4,
    )
    result = peaks_for(mix_signal_and_noise(tone, noise), min_prominence=0.05)
    peak = max(result.peaks, key=lambda item: item.bin_amplitude)

    assert peak.local_noise_floor is not None
    assert peak.local_snr_db is not None and peak.local_snr_db > 10.0


def test_dc_silence_impulse_and_edges_do_not_create_invalid_peaks() -> None:
    """Casos de borda devem evitar NaN e picos falsos em silêncio."""
    silence = pure_sine(10.0, duration_s=1.0, sample_rate=128, amplitude=0.0)
    dc = with_dc_offset(silence, offset=0.8)
    impulse = ideal_impulse(duration_s=1.0, sample_rate=128, sample_index=0)

    assert peaks_for(silence).accepted_count == 0
    assert peaks_for(dc, min_prominence=0.01).accepted_count == 0
    assert all(np.isfinite(peak.bin_amplitude) for peak in peaks_for(impulse).peaks)


@pytest.mark.parametrize("frequency", [1.0, 63.0])
def test_peaks_near_dc_and_nyquist_remain_bounded(frequency: float) -> None:
    """Picos próximos às bordas devem ficar dentro dos bins vizinhos válidos."""
    result = peaks_for(pure_sine(frequency, duration_s=1.0, sample_rate=128))
    strongest = max(result.peaks, key=lambda item: item.bin_amplitude)

    assert strongest.bin_frequency_hz == pytest.approx(frequency)
    if strongest.refined_frequency_hz is not None:
        assert abs(strongest.refined_frequency_hz - strongest.bin_frequency_hz) < 1.0


def test_frequency_range_max_peaks_dbfs_and_zero_padding() -> None:
    """Faixa, limite, escala dB e grade interpolada devem permanecer coerentes."""
    signal = sine_sum(
        (30.0, 90.0, 150.0),
        duration_s=1.0,
        sample_rate=512,
        amplitudes=(0.3, 0.8, 0.4),
    )
    spectrum = analyze_spectrum(signal).spectrum
    limited = detect_spectral_peaks(
        spectrum,
        PeakDetectionSettings(min_frequency_hz=50.0, max_frequency_hz=120.0),
    )
    db_spectrum = analyze_spectrum(
        signal,
        settings=SpectrumAnalysisSettings(scale="dbfs"),
    ).spectrum
    padded = analyze_spectrum(
        signal,
        settings=SpectrumAnalysisSettings(n_fft=2_048),
    ).spectrum

    assert limited.accepted_count == 1
    assert limited.peaks[0].bin_frequency_hz == pytest.approx(90.0)
    assert padded.bin_spacing_hz == pytest.approx(0.25)
    assert detect_spectral_peaks(db_spectrum).accepted_count == 3
    assert db_spectrum.frequency_resolution_hz == db_spectrum.bin_spacing_hz
    capped = detect_spectral_peaks(spectrum, PeakDetectionSettings(max_peaks=2))
    assert capped.accepted_count == 2


def test_damped_signal_and_invalid_settings() -> None:
    """Sinal amortecido é detectável e configurações inválidas falham cedo."""
    signal = damped_exponential_sum(
        (DampedSineComponent(70.0, 3.0),), duration_s=1.0, sample_rate=1_024
    )
    result = peaks_for(signal)

    strongest = max(result.peaks, key=lambda item: item.bin_amplitude)
    assert strongest.bin_frequency_hz == pytest.approx(70.0, abs=2.0)
    with pytest.raises(ValueError, match="max_peaks"):
        PeakDetectionSettings(max_peaks=0)
    with pytest.raises(ValueError, match="distance_bins"):
        PeakDetectionSettings(distance_bins=0)
