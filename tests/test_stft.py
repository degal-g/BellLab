"""Testes quantitativos da primeira STFT do BellLab."""
import numpy as np
import pytest

from belllab import (
    AnalysisSettings,
    PeakDetectionSettings,
    STFTSettings,
    SpectrumAnalysisSettings,
)
from belllab.spectrum import analyze_spectrum, analyze_stft
from belllab.synthetic import damped_exponential, ideal_impulse, linear_chirp, pure_sine, sine_sum
from belllab.types import Signal


def _multichannel(*channels: np.ndarray, sample_rate: int) -> Signal:
    """Constrói sinal multicanal imutável para testes de política de canais."""
    sample_count = channels[0].size
    return Signal(
        samples=tuple(tuple(float(value) for value in channel) for channel in channels),
        sample_rate=sample_rate,
        time=tuple(index / sample_rate for index in range(sample_count)),
        duration=sample_count / sample_rate,
        channels=len(channels),
        unit="normalized",
    )


def test_centered_sine_stft_amplitude_and_axes() -> None:
    signal = pure_sine(64.0, duration_s=1.0, sample_rate=1024, amplitude=0.5)
    result = analyze_stft(signal, STFTSettings(window_length=256, hop_length=128))
    tf = result.time_frequency
    values = np.asarray(tf.values)
    index = int(np.argmax(values[:, 0]))
    assert tf.frequencies_hz[index] == pytest.approx(64.0)
    assert values[index, 0] == pytest.approx(0.5, rel=1e-5)
    assert tf.frame_spacing_s == pytest.approx(0.125)
    assert tf.times_s[0] == pytest.approx(0.125)


def test_chirp_rises_and_padding_is_explicit() -> None:
    chirp = linear_chirp(20, 100, duration_s=1.0, sample_rate=512)
    result = analyze_stft(chirp, STFTSettings(window_length=128, hop_length=100, pad_end=True))
    values = np.asarray(result.time_frequency.values)
    peaks = np.argmax(values, axis=0)
    frequencies = np.asarray(result.time_frequency.frequencies_hz)[peaks]
    assert frequencies[-1] > frequencies[0]
    assert "padding_end_applied" in result.diagnostics
    assert any(item.startswith("padded_samples=") for item in result.diagnostics)


def test_silence_dbfs_and_multichannel_policy() -> None:
    silence = pure_sine(10, duration_s=1, sample_rate=256, amplitude=0)
    result = analyze_stft(silence, STFTSettings(window_length=64, hop_length=32, scale="dbfs"))
    assert not np.any(np.isnan(np.asarray(result.time_frequency.values)))
    assert np.isneginf(np.asarray(result.time_frequency.values)).all()
    assert "dbfs_contains_negative_infinity" in result.diagnostics


def test_invalid_stft_time_settings_fail() -> None:
    """Tempos inválidos são rejeitados pela configuração antes da execução."""
    for kwargs in (
        {"start_time_s": -0.1},
        {"end_time_s": -0.1},
        {"start_time_s": 0.2, "end_time_s": 0.2},
        {"start_time_s": 0.3, "end_time_s": 0.2},
    ):
        with pytest.raises(ValueError):
            STFTSettings(**kwargs)


def test_stft_interval_frequency_and_analysis_settings_contracts() -> None:
    """Intervalo efetivo, corte espectral e configuração agregada são preservados."""
    signal = pure_sine(32, duration_s=1, sample_rate=256)
    cfg = STFTSettings(
        window_length=64,
        hop_length=32,
        start_time_s=0.25,
        end_time_s=0.75,
        frequency_min_hz=16,
        frequency_max_hz=64,
    )
    result = analyze_stft(signal, AnalysisSettings(stft=cfg))
    tf = result.time_frequency

    assert result.settings is cfg
    assert tf.interval_start_s == pytest.approx(0.25)
    assert tf.interval_end_s == pytest.approx(0.75)
    assert min(tf.frequencies_hz) >= 16
    assert max(tf.frequencies_hz) <= 64
    with pytest.raises(ValueError, match="duration"):
        analyze_stft(signal, STFTSettings(window_length=64, hop_length=32, start_time_s=1.0))
    with pytest.raises(ValueError, match="Nyquist"):
        analyze_stft(signal, STFTSettings(window_length=64, hop_length=32, frequency_max_hz=200))


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (STFTSettings, {"channel_policy": "implicit_mean"}),
        (STFTSettings, {"window_name": "blackman"}),
        (STFTSettings, {"scale": "power"}),
        (STFTSettings, {"pad_end": 1}),
        (STFTSettings, {"remove_mean": "frame_mean"}),
        (SpectrumAnalysisSettings, {"channel_policy": "implicit_mean"}),
        (SpectrumAnalysisSettings, {"window_name": "blackman"}),
        (SpectrumAnalysisSettings, {"scale": "power"}),
        (SpectrumAnalysisSettings, {"normalization": "unitary"}),
    ],
)
def test_invalid_closed_spectrum_options_fail_at_runtime(factory, kwargs) -> None:
    """Literal annotations are enforced even when callers bypass type checking."""
    with pytest.raises(ValueError):
        factory(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"interpolation_method": "quadratic"},
        {"noise_method": "global_mean"},
        {"sort_by": "quality"},
    ],
)
def test_invalid_peak_options_fail_at_runtime(kwargs) -> None:
    """Peak settings reject unsupported closed options before analysis starts."""
    with pytest.raises(ValueError):
        PeakDetectionSettings(**kwargs)


def test_frame_mean_removes_dc_without_erasing_centered_sine() -> None:
    """Per-frame detrending reduces DC while retaining the oscillatory component."""
    signal = pure_sine(64.0, duration_s=1.0, sample_rate=1024, amplitude=0.5)
    samples = np.asarray(signal.samples[0]) + 0.75
    offset_signal = _multichannel(samples, sample_rate=1024)
    common = dict(window_length=256, hop_length=256, window_name="hann")
    raw = analyze_stft(offset_signal, STFTSettings(remove_mean=False, **common))
    detrended = analyze_stft(offset_signal, STFTSettings(remove_mean=True, **common))
    raw_values = np.asarray(raw.time_frequency.values)
    detrended_values = np.asarray(detrended.time_frequency.values)
    frequency_index = list(raw.time_frequency.frequencies_hz).index(64.0)

    assert raw_values[0, 0] == pytest.approx(0.75, rel=1e-5)
    assert detrended_values[0, 0] < 1e-5
    assert detrended_values[frequency_index, 0] == pytest.approx(0.5, rel=1e-5)
    assert detrended.time_frequency.parameters["detrend_method"] == "frame_mean"
    assert raw.time_frequency.parameters["detrend_method"] == "none"
    assert "frame_mean_removed" in detrended.diagnostics


def test_stft_matches_equivalent_stationary_fft_bin_by_bin() -> None:
    """A single STFT frame shares the stationary FFT normalization exactly."""
    signal = sine_sum(
        (32.0, 96.0),
        amplitudes=(0.6, 0.2),
        duration_s=0.25,
        sample_rate=1024,
    )
    spectrum = analyze_spectrum(
        signal,
        SpectrumAnalysisSettings(
            window_name="hann", n_fft=512, remove_mean=False,
        ),
    ).spectrum
    stft = analyze_stft(
        signal,
        STFTSettings(
            window_name="hann", window_length=256, n_fft=512,
            hop_length=256, remove_mean=False,
        ),
    ).time_frequency
    assert stft.frequencies_hz == spectrum.frequencies_hz
    assert stft.bin_spacing_hz == spectrum.bin_spacing_hz
    np.testing.assert_allclose(np.asarray(stft.values)[:, 0], spectrum.magnitudes,
                               rtol=1e-12, atol=1e-12)


def test_final_frame_and_short_signal_policies_are_auditable() -> None:
    """Discard and zero-padding policies report only operations that occurred."""
    signal = pure_sine(16, duration_s=0.35, sample_rate=100)
    discarded = analyze_stft(signal, STFTSettings(window_length=16, hop_length=10))
    assert len(discarded.time_frequency.times_s) == 2
    assert "final_incomplete_segment_discarded:9" in discarded.diagnostics
    assert discarded.time_frequency.parameters["discarded_samples"] == 9

    short = pure_sine(16, duration_s=0.08, sample_rate=100)
    with pytest.raises(ValueError, match="shorter"):
        analyze_stft(short, STFTSettings(window_length=16, hop_length=8))
    padded = analyze_stft(short, STFTSettings(window_length=16, hop_length=8, pad_end=True))
    assert len(padded.time_frequency.times_s) == 1
    assert padded.time_frequency.parameters["padded_samples"] == 8
    assert "segment_shorter_than_window" in padded.diagnostics


def test_frequency_crop_zero_padding_and_channel_policy_diagnostics() -> None:
    """Frequency and explicit channel operations remain visible in diagnostics."""
    left = np.asarray(pure_sine(32, duration_s=1, sample_rate=256, amplitude=0.5).samples[0])
    multichannel = _multichannel(left, -left, sample_rate=256)
    selected = analyze_stft(
        multichannel,
        STFTSettings(
            channel_policy="select", channel_index=1, window_length=64,
            hop_length=32, n_fft=128, frequency_min_hz=16, frequency_max_hz=64,
        ),
    )
    assert "channel_selected:1" in selected.diagnostics
    assert "frequency_range_cropped" in selected.diagnostics
    assert "spectral_zero_padding_applied" in selected.diagnostics
    peak = np.max(np.asarray(selected.time_frequency.values)[:, 0])
    assert peak == pytest.approx(0.5, rel=1e-4)

    mean = analyze_stft(
        multichannel,
        STFTSettings(channel_policy="mean", window_length=64, hop_length=32),
    )
    assert np.allclose(np.asarray(mean.time_frequency.values), 0.0)
    assert "channels_meaned_explicitly" in mean.diagnostics


def test_stft_scientific_signal_cases_are_finite_and_reproducible() -> None:
    """Damped sinusoids and impulses have stable, finite linear STFT results."""
    damped = damped_exponential(48, 4, duration_s=1, sample_rate=512)
    first = analyze_stft(damped, STFTSettings(window_length=128, hop_length=64))
    second = analyze_stft(damped, STFTSettings(window_length=128, hop_length=64))
    values = np.asarray(first.time_frequency.values)
    peak_bins = np.argmax(values, axis=0)
    peak_frequencies = np.asarray(first.time_frequency.frequencies_hz)[peak_bins]
    assert np.allclose(peak_frequencies, 48.0)
    assert values[peak_bins[0], 0] > values[peak_bins[-1], -1]
    np.testing.assert_array_equal(values, np.asarray(second.time_frequency.values))

    impulse = ideal_impulse(duration_s=1, sample_rate=512, sample_index=100)
    impulse_stft = analyze_stft(impulse, STFTSettings(window_length=128, hop_length=64))
    assert np.isfinite(np.asarray(impulse_stft.time_frequency.values)).all()
    impulse_values = np.asarray(impulse_stft.time_frequency.values)
    assert np.max(impulse_values[:, 0]) > 0.0
    assert np.max(impulse_values[:, 1]) > 0.0
    assert np.max(impulse_values[:, 2]) == 0.0
