"""Testes sintéticos para as análises independentes no domínio temporal."""

from __future__ import annotations

from math import isinf

import numpy as np
import pytest

from belllab import AnalysisSettings, Signal, TemporalAnalysisSettings
from belllab.temporal import (
    analyze_temporal,
    calculate_temporal_metrics,
    detect_impact,
    envelope_cumulative_energy,
    envelope_hilbert,
    envelope_moving_peak,
    envelope_moving_rms,
    estimate_noise,
)


def make_signal(samples: np.ndarray, sample_rate: int = 1_000) -> Signal:
    """Cria um Signal mono sintético a partir de uma série de amostras."""
    samples = np.asarray(samples, dtype=np.float64)
    return Signal(
        samples=(tuple(samples),),
        sample_rate=sample_rate,
        time=tuple(np.arange(samples.size) / sample_rate),
        duration=samples.size / sample_rate,
        channels=1,
        unit="normalized",
    )


def test_detect_impact_finds_synthetic_impulse() -> None:
    """O detector deve encontrar o início de um impulso artificial isolado."""
    samples = np.zeros(1_000)
    samples[200] = 1.0
    signal = make_signal(samples)

    report = detect_impact(signal)

    assert report.impact_sample == 200
    assert report.peak_sample == 200
    assert report.impact_time_s == pytest.approx(0.2)
    assert report.transient_duration_s >= 0.001
    assert 0.0 <= report.confidence <= 1.0


@pytest.mark.parametrize("channel_count", [1, 2])
def test_detect_impact_uses_the_temporal_axis_for_baseline(
    channel_count: int,
) -> None:
    """A janela basal deve depender de amostras, e não da quantidade de canais."""
    samples = np.zeros(1_000)
    baseline = np.resize(np.array([0.09, 0.11]), 100)
    baseline[0] = 0.15
    samples[:100] = baseline
    samples[100] = 0.20
    samples[101] = 0.50
    channels = tuple(tuple(samples) for _ in range(channel_count))
    signal = Signal(
        samples=channels,
        sample_rate=1_000,
        time=tuple(np.arange(samples.size) / 1_000),
        duration=1.0,
        channels=channel_count,
        unit="normalized",
    )

    report = detect_impact(signal)

    assert report.impact_sample == 100
    assert report.parameters["baseline_sample_count"] == 100


def test_analyze_temporal_returns_all_canonical_results() -> None:
    """A API agregada deve devolver impacto, ruído, métricas e envelope."""
    settings = AnalysisSettings(
        temporal=TemporalAnalysisSettings(
            noise_window_duration_s=0.010,
            envelope_method="moving_rms",
            envelope_window_size_samples=4,
        )
    )
    samples = np.r_[np.zeros(20), 1.0, np.zeros(79)]
    result = analyze_temporal(make_signal(samples), settings)

    assert result.impact is not None
    assert result.noise is not None
    assert result.metrics is not None
    assert result.envelope is not None
    assert result.envelope.method == "moving-rms"
    assert result.settings == settings.temporal


def test_estimate_noise_selects_quiet_tail_of_synthetic_signal() -> None:
    """A estimativa deve selecionar a região menos energética disponível."""
    samples = np.concatenate((np.full(500, 0.5), np.full(500, 0.01)))
    report = estimate_noise(make_signal(samples), window_duration_s=0.1)

    assert report.rms_noise == pytest.approx(0.01)
    assert report.window_start_s >= 0.5
    assert report.noise_floor_dbfs == pytest.approx(-40.0)


def test_temporal_metrics_for_constant_synthetic_signal() -> None:
    """Métricas de um sinal constante possuem valores analiticamente conhecidos."""
    metrics = calculate_temporal_metrics(make_signal(np.full(100, 0.5), 100))

    assert metrics.peak == pytest.approx(0.5)
    assert metrics.rms == pytest.approx(0.5)
    assert metrics.crest_factor == pytest.approx(1.0)
    assert metrics.dynamic_range_db == pytest.approx(0.0)
    assert metrics.total_energy == pytest.approx(0.25)


def test_temporal_metrics_reports_silence_without_failure() -> None:
    """Silêncio deve ter RMS nulo e faixa dinâmica indefinida explicitamente."""
    metrics = calculate_temporal_metrics(make_signal(np.zeros(16)))

    assert metrics.crest_factor is None
    assert isinf(metrics.dynamic_range_db)
    assert metrics.dynamic_range_db < 0


def test_hilbert_envelope_of_sine_is_approximately_constant() -> None:
    """O envelope de Hilbert de seno artificial deve recuperar sua amplitude."""
    sample_rate = 10_000
    time = np.arange(sample_rate) / sample_rate
    signal = make_signal(0.7 * np.sin(2 * np.pi * 100 * time), sample_rate)

    envelope = envelope_hilbert(signal)

    assert envelope.method == "hilbert"
    assert np.median(envelope.amplitudes) == pytest.approx(0.7, rel=0.01)


def test_moving_envelopes_and_cumulative_energy() -> None:
    """Os três envelopes temporais restantes devem ter comportamento esperado."""
    signal = make_signal(np.array([0.0, 1.0, -1.0, 0.0]), 4)

    rms = envelope_moving_rms(signal, window_size_samples=1)
    peak = envelope_moving_peak(signal, window_size_samples=3)
    energy = envelope_cumulative_energy(signal)

    assert rms.amplitudes == pytest.approx((0.0, 1.0, 1.0, 0.0))
    assert peak.amplitudes[1] == pytest.approx(1.0)
    assert peak.parameters["window_size_samples"] == 3
    assert energy.amplitudes[-1] == pytest.approx(0.5)
    assert energy.unit == "normalized^2 s"


@pytest.mark.parametrize("window_size", [0, -1])
def test_moving_envelopes_reject_invalid_window_sizes(window_size: int) -> None:
    """Janelas não positivas devem gerar erros explícitos."""
    signal = make_signal(np.ones(8))

    with pytest.raises(ValueError, match="window_size_samples"):
        envelope_moving_rms(signal, window_size_samples=window_size)
    with pytest.raises(ValueError, match="window_size_samples"):
        envelope_moving_peak(signal, window_size_samples=window_size)
