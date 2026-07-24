"""Testes quantitativos da FFT unilateral estacionária do BellLab."""

from __future__ import annotations

from math import isinf

import numpy as np
import pytest

from belllab import Signal, SpectrumAnalysisSettings
from belllab.spectrum import analyze_spectrum
from belllab.synthetic import (
    DampedSineComponent,
    damped_exponential_sum,
    ideal_impulse,
    pure_sine,
    sine_sum,
    with_dc_offset,
)


def magnitude_array(result: object) -> np.ndarray:
    """Extrai magnitudes do resultado para asserções numéricas locais."""
    return np.asarray(result.spectrum.magnitudes)  # type: ignore[union-attr]


def frequency_array(result: object) -> np.ndarray:
    """Extrai frequências do resultado para asserções numéricas locais."""
    return np.asarray(result.spectrum.frequencies_hz)  # type: ignore[union-attr]


def test_centered_sine_recovers_peak_amplitude_and_frequency() -> None:
    """Uma senoide em bin deve recuperar amplitude de pico com janela Hann."""
    signal = pure_sine(64.0, duration_s=1.0, sample_rate=1_024, amplitude=0.7)
    result = analyze_spectrum(signal)
    magnitudes = magnitude_array(result)
    frequencies = frequency_array(result)
    peak_index = int(np.argmax(magnitudes))

    assert frequencies[peak_index] == pytest.approx(64.0)
    assert magnitudes[peak_index] == pytest.approx(0.7, rel=1e-5, abs=1e-12)
    assert result.spectrum.bin_spacing_hz == pytest.approx(1.0)
    assert result.spectrum.frequency_resolution_hz == result.spectrum.bin_spacing_hz


def test_sine_between_bins_leaks_to_neighboring_bins() -> None:
    """Uma senoide fora de bin deve concentrar-se nos bins vizinhos esperados."""
    signal = pure_sine(64.5, duration_s=1.0, sample_rate=1_024, amplitude=1.0)
    result = analyze_spectrum(signal)
    peak_frequency = frequency_array(result)[int(np.argmax(magnitude_array(result)))]

    assert peak_frequency in {64.0, 65.0}
    assert magnitude_array(result)[int(np.argmax(magnitude_array(result)))] < 1.0


def test_two_sines_recover_distinct_amplitudes() -> None:
    """A normalização deve distinguir duas senoides alinhadas aos bins."""
    signal = sine_sum(
        (50.0, 150.0),
        duration_s=1.0,
        sample_rate=1_024,
        amplitudes=(0.8, 0.25),
    )
    result = analyze_spectrum(signal)
    magnitudes = magnitude_array(result)

    assert magnitudes[50] == pytest.approx(0.8, rel=1e-5)
    assert magnitudes[150] == pytest.approx(0.25, rel=1e-5)


def test_mean_removal_controls_the_dc_bin() -> None:
    """Remoção de média deve eliminar o componente DC de um sinal constante."""
    signal = with_dc_offset(
        pure_sine(10.0, duration_s=1.0, sample_rate=1_024, amplitude=0.0),
        offset=0.5,
    )
    with_dc = analyze_spectrum(
        signal,
        SpectrumAnalysisSettings(window_name="rectangular", remove_mean=False),
    )
    without_dc = analyze_spectrum(
        signal,
        SpectrumAnalysisSettings(window_name="rectangular", remove_mean=True),
    )

    assert magnitude_array(with_dc)[0] == pytest.approx(0.5)
    assert magnitude_array(without_dc)[0] == pytest.approx(0.0, abs=1e-14)


def test_null_signal_dbfs_uses_negative_infinity_without_nan() -> None:
    """Silêncio em dBFS deve ser explícito e não produzir NaN."""
    signal = pure_sine(10.0, duration_s=1.0, sample_rate=1_024, amplitude=0.0)
    result = analyze_spectrum(signal, SpectrumAnalysisSettings(scale="dbfs"))
    magnitudes = magnitude_array(result)

    assert isinf(magnitudes[0]) and magnitudes[0] < 0
    assert not np.any(np.isnan(magnitudes))


def test_ideal_impulse_has_flat_one_sided_amplitude_shape() -> None:
    """O impulso deve respeitar os tratamentos especiais de DC e Nyquist."""
    signal = ideal_impulse(duration_s=1.0, sample_rate=64, sample_index=0)
    result = analyze_spectrum(
        signal,
        SpectrumAnalysisSettings(window_name="rectangular", remove_mean=False),
    )
    magnitudes = magnitude_array(result)

    assert len(magnitudes) == 33
    assert magnitudes[0] == pytest.approx(1.0 / 64.0)
    assert magnitudes[-1] == pytest.approx(1.0 / 64.0)
    assert magnitudes[1] == pytest.approx(2.0 / 64.0)


def test_damped_sine_produces_finite_peak_near_its_frequency() -> None:
    """Uma senoide amortecida deve fornecer espectro finito e pico localizável."""
    signal = damped_exponential_sum(
        (DampedSineComponent(80.0, 4.0, 1.0),),
        duration_s=1.0,
        sample_rate=1_024,
    )
    result = analyze_spectrum(signal)
    peak_frequency = frequency_array(result)[int(np.argmax(magnitude_array(result)))]

    assert peak_frequency == pytest.approx(80.0, abs=2.0)
    assert np.all(np.isfinite(magnitude_array(result)))


def test_multichannel_default_select_avoids_implicit_phase_cancellation() -> None:
    """A política padrão deve analisar um canal, não a média de canais opostos."""
    sample_rate = 1_024
    time = np.arange(sample_rate) / sample_rate
    first = 0.6 * np.sin(2 * np.pi * 64 * time)
    signal = Signal(
        samples=(tuple(first), tuple(-first)),
        sample_rate=sample_rate,
        time=tuple(time),
        duration=1.0,
        channels=2,
        unit="normalized",
    )
    selected = analyze_spectrum(signal)
    averaged = analyze_spectrum(
        signal,
        SpectrumAnalysisSettings(channel_policy="mean"),
    )

    assert magnitude_array(selected)[64] == pytest.approx(0.6, rel=1e-5)
    assert magnitude_array(averaged)[64] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("window_name", ["rectangular", "hann"])
def test_window_normalization_recovers_centered_sine(window_name: str) -> None:
    """Ganho coerente deve recuperar amplitude para todas as janelas suportadas."""
    signal = pure_sine(32.0, duration_s=1.0, sample_rate=256, amplitude=0.4)
    result = analyze_spectrum(signal, SpectrumAnalysisSettings(window_name=window_name))

    assert magnitude_array(result)[32] == pytest.approx(0.4, rel=1e-5)


def test_zero_padding_changes_bin_spacing_not_centered_amplitude() -> None:
    """Zero padding deve densificar o eixo sem alterar amplitude normalizada."""
    signal = pure_sine(16.0, duration_s=1.0, sample_rate=128, amplitude=0.5)
    result = analyze_spectrum(signal, SpectrumAnalysisSettings(n_fft=512))

    assert len(magnitude_array(result)) == 257
    assert result.spectrum.bin_spacing_hz == pytest.approx(0.25)
    assert magnitude_array(result)[64] == pytest.approx(0.5, rel=1e-5)
    assert result.diagnostics


@pytest.mark.parametrize(
    "settings",
    [
        SpectrumAnalysisSettings(channel_index=1),
        SpectrumAnalysisSettings(n_fft=4),
        SpectrumAnalysisSettings(start_time_s=1.0),
    ],
)
def test_invalid_spectrum_inputs_raise_clear_errors(
    settings: SpectrumAnalysisSettings,
) -> None:
    """Configurações incompatíveis com o sinal devem falhar explicitamente."""
    signal = pure_sine(8.0, duration_s=1.0, sample_rate=64)

    with pytest.raises(ValueError):
        analyze_spectrum(signal, settings)
