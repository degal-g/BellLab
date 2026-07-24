"""Análises científicas independentes no domínio temporal.

O módulo extrai propriedades temporais de um :class:`belllab.types.Signal` sem
executar análise espectral, gerar figuras ou escrever arquivos. As funções
públicas são independentes: cada uma recebe somente o sinal e os parâmetros de
que necessita e retorna um relatório ou estrutura científica tipada.
"""

from __future__ import annotations

from math import inf, log10
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy.signal import hilbert

from belllab.config import AnalysisSettings, TemporalAnalysisSettings
from belllab.results import TemporalResults
from belllab.types import Envelope, ImpactReport, NoiseReport, Signal, TemporalMetrics


def detect_impact(signal: Signal) -> ImpactReport:
    """Detecta o início e a extensão inicial do transiente de impacto.

    A detecção usa a amplitude máxima entre canais e um limiar adaptativo: o
    maior valor entre cinco por cento do pico e uma estimativa robusta do nível
    basal. O fim do transiente é o primeiro trecho posterior ao pico que se
    mantém abaixo de vinte por cento do pico por pelo menos 2 ms. Esses limiares
    são heurísticos, declarados no campo ``method``, e não substituem inspeção
    especializada de gravações com múltiplos impactos ou saturação extensa.

    Args:
        signal: Sinal carregado cuja emissão percutida será examinada.

    Returns:
        Relatório tipado com índices, instantes e confiança da detecção.

    Raises:
        ValueError: Se o sinal não possuir amostras ou canais consistentes.

    References:
        Z. Antoni, "Blind separation of vibration components," *Mechanical
        Systems and Signal Processing*, 2005. A referência contextualiza o uso
        de medidas robustas para caracterizar transientes em sinais vibratórios.
    """
    samples = _normalized_matrix(signal)
    amplitude = np.max(np.abs(samples), axis=0)
    peak_sample = int(np.argmax(amplitude))
    peak = float(amplitude[peak_sample])
    baseline_size = max(1, min(peak_sample, amplitude.size // 10))
    baseline = amplitude[:baseline_size]
    baseline_median = float(np.median(baseline))
    baseline_mad = float(np.median(np.abs(baseline - baseline_median)))
    threshold = max(0.05 * peak, baseline_median + 8.0 * baseline_mad)

    candidates = np.flatnonzero(amplitude[: peak_sample + 1] >= threshold)
    impact_sample = int(candidates[0]) if candidates.size else peak_sample
    transient_end = _find_transient_end(
        amplitude,
        peak_sample,
        threshold=max(0.20 * peak, threshold),
        hold_samples=max(1, round(0.002 * signal.sample_rate)),
    )
    contrast = peak / max(threshold, np.finfo(np.float64).eps)
    confidence = float(min(1.0, max(0.0, log10(max(contrast, 1.0)) / 2.0)))
    impact_time = impact_sample / signal.sample_rate
    transient_end_s = transient_end / signal.sample_rate

    return ImpactReport(
        impact_time_s=impact_time,
        impact_sample=impact_sample,
        peak_sample=peak_sample,
        peak_time_s=peak_sample / signal.sample_rate,
        transient_end_s=transient_end_s,
        transient_duration_s=transient_end_s - impact_time,
        confidence=confidence,
        method="adaptive-peak-threshold-v1",
        parameters=MappingProxyType(
            {
                "peak_threshold_fraction": 0.05,
                "baseline_mad_multiplier": 8.0,
                "transient_threshold_fraction": 0.20,
                "transient_hold_s": 0.002,
                "baseline_sample_count": baseline_size,
            }
        ),
    )


def estimate_noise(signal: Signal, *, window_duration_s: float = 0.050) -> NoiseReport:
    """Estima o ruído de fundo pela janela temporal contígua mais silenciosa.

    O sinal é combinado por média entre canais e avaliado em janelas sobrepostas
    de duração configurável. A janela de menor RMS é tratada como a melhor
    observação disponível do ruído de fundo. A medida é expressa em dBFS e não
    pressupõe calibração para pressão sonora.

    Args:
        signal: Sinal carregado a ser avaliado.
        window_duration_s: Duração nominal de cada janela, em segundos. Para
            sinais menores, a janela é limitada à duração disponível.

    Returns:
        Relatório tipado da janela de ruído selecionada.

    Raises:
        ValueError: Se ``window_duration_s`` não for positivo ou o sinal for
            inválido.

    References:
        IEC 61672-1:2013, *Electroacoustics—Sound level meters—Part 1:
        Specifications*. A referência é pertinente ao uso de RMS para descrição
        de níveis de sinal, sem implicar conformidade metrológica.
    """
    if window_duration_s <= 0:
        raise ValueError("window_duration_s must be positive.")

    waveform = _mono_waveform(signal)
    requested_size = round(window_duration_s * signal.sample_rate)
    window_size = min(waveform.size, max(1, requested_size))
    starts = list(range(0, waveform.size - window_size + 1, max(1, window_size // 2)))
    final_start = waveform.size - window_size
    if starts[-1] != final_start:
        starts.append(final_start)

    rms_values = np.array([
        float(np.sqrt(np.mean(np.square(waveform[start : start + window_size]))))
        for start in starts
    ])
    selected = int(np.argmin(rms_values))
    start = starts[selected]
    rms_noise = float(rms_values[selected])
    confidence = 1.0 if len(starts) >= 3 else 0.5

    return NoiseReport(
        noise_floor_dbfs=_to_dbfs(rms_noise),
        rms_noise=rms_noise,
        window_start_s=start / signal.sample_rate,
        window_end_s=(start + window_size) / signal.sample_rate,
        confidence=confidence,
        method="quietest-rms-window-v1",
        parameters=MappingProxyType({"window_duration_s": window_duration_s}),
    )


def calculate_temporal_metrics(signal: Signal) -> TemporalMetrics:
    """Calcula pico, RMS, fator de crista, faixa dinâmica e energia total.

    As métricas são obtidas do sinal combinado pela média dos canais. A faixa
    dinâmica usa percentis 95 e 5 da amplitude absoluta não nula para reduzir a
    influência de amostras isoladas. A energia é uma medida discreta de
    amplitude quadrática integrada no tempo; sua unidade depende de
    ``Signal.unit`` e não deve ser interpretada como energia física sem
    calibração apropriada.

    Args:
        signal: Sinal carregado a caracterizar.

    Returns:
        Métricas temporais globais e independentes de análise espectral.

    References:
        A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*,
        3rd ed., Pearson, 2010, capítulos sobre medidas de energia e potência.
    """
    waveform = _mono_waveform(signal)
    absolute = np.abs(waveform)
    peak = float(np.max(absolute))
    rms = float(np.sqrt(np.mean(np.square(waveform))))
    crest_factor = peak / rms if rms else None
    nonzero = absolute[absolute > 0]
    if nonzero.size == 0:
        dynamic_range_db = -inf
    else:
        lower, upper = np.percentile(nonzero, [5.0, 95.0])
        dynamic_range_db = 20.0 * log10(float(upper / lower)) if lower else inf

    return TemporalMetrics(
        peak=peak,
        rms=rms,
        crest_factor=crest_factor,
        dynamic_range_db=dynamic_range_db,
        total_energy=float(np.sum(np.square(waveform)) / signal.sample_rate),
        parameters=MappingProxyType({"channel_combination": "mean"}),
    )


def envelope_hilbert(signal: Signal) -> Envelope:
    """Calcula o envelope de amplitude usando o sinal analítico de Hilbert.

    A implementação delega a transformação de Hilbert à rotina validada do
    SciPy; este módulo não implementa FFT ou qualquer análise espectral. Para
    sinais multicanais, as amostras são combinadas por média antes do cálculo.

    Args:
        signal: Sinal carregado cuja envoltória será estimada.

    Returns:
        Envelope de mesmo comprimento do sinal, na unidade de ``signal``.

    References:
        L. Cohen, *Time-Frequency Analysis*, Prentice Hall, 1995, seção sobre
        sinal analítico e transformada de Hilbert.
    """
    waveform = _mono_waveform(signal)
    amplitudes = np.abs(hilbert(waveform))
    return _make_envelope(signal, amplitudes, "hilbert", MappingProxyType({}))


def envelope_moving_rms(signal: Signal, *, window_size_samples: int) -> Envelope:
    """Calcula um envelope RMS em janela móvel centrada.

    Args:
        signal: Sinal carregado cuja envoltória será estimada.
        window_size_samples: Número positivo de amostras em cada janela.

    Returns:
        Envelope RMS de mesmo comprimento do sinal.

    Raises:
        ValueError: Se ``window_size_samples`` não for positivo.

    References:
        IEC 61672-1:2013, para o uso de medidas RMS na descrição de níveis.
    """
    if window_size_samples <= 0:
        raise ValueError("window_size_samples must be positive.")
    waveform = _mono_waveform(signal)
    kernel = np.full(window_size_samples, 1.0 / window_size_samples)
    amplitudes = np.sqrt(np.convolve(np.square(waveform), kernel, mode="same"))
    return _make_envelope(
        signal,
        amplitudes,
        "moving-rms",
        MappingProxyType({"window_size_samples": window_size_samples}),
    )


def envelope_moving_peak(signal: Signal, *, window_size_samples: int) -> Envelope:
    """Calcula um envelope de pico em janela móvel centrada.

    Args:
        signal: Sinal carregado cuja envoltória será estimada.
        window_size_samples: Número positivo de amostras em cada janela.

    Returns:
        Envelope de pico de mesmo comprimento do sinal.

    Raises:
        ValueError: Se ``window_size_samples`` não for positivo.
    """
    if window_size_samples <= 0:
        raise ValueError("window_size_samples must be positive.")
    waveform = _mono_waveform(signal)
    half_left = window_size_samples // 2
    half_right = window_size_samples - half_left
    amplitudes = np.array(
        [
            np.max(np.abs(waveform[max(0, index - half_left) : index + half_right]))
            for index in range(waveform.size)
        ]
    )
    return _make_envelope(
        signal,
        amplitudes,
        "moving-peak",
        MappingProxyType({"window_size_samples": window_size_samples}),
    )


def envelope_cumulative_energy(signal: Signal) -> Envelope:
    """Calcula a energia acumulada do sinal ao longo do tempo.

    O resultado acumula amplitude quadrática multiplicada pelo intervalo entre
    amostras. A unidade é expressa como ``<unidade-do-sinal>^2 s`` e requer
    calibração externa para ter interpretação de energia física.

    Args:
        signal: Sinal carregado cuja energia acumulada será estimada.

    Returns:
        Envelope monotônico de energia acumulada, de mesmo comprimento do sinal.
    """
    waveform = _mono_waveform(signal)
    amplitudes = np.cumsum(np.square(waveform)) / signal.sample_rate
    return _make_envelope(
        signal,
        amplitudes,
        "cumulative-energy",
        MappingProxyType({"sample_interval_s": 1.0 / signal.sample_rate}),
        unit=f"{signal.unit}^2 s",
    )


def analyze_temporal(
    signal: Signal,
    settings: AnalysisSettings | None = None,
) -> TemporalResults:
    """Produz o subconjunto temporal compatível com ``TemporalResults``.

    Esta função usa as configurações temporais fornecidas e retorna todos os
    contratos científicos calculados pela execução. A relação sinal-ruído é a
    diferença, em dBFS, entre o RMS global e o RMS da janela selecionada.
    """
    effective_settings = (settings or AnalysisSettings()).temporal
    noise = estimate_noise(
        signal,
        window_duration_s=effective_settings.noise_window_duration_s,
    )
    metrics = calculate_temporal_metrics(signal)
    if metrics.rms == 0.0:
        signal_to_noise_ratio_db = 0.0
    elif noise.rms_noise == 0.0:
        signal_to_noise_ratio_db = inf
    else:
        signal_to_noise_ratio_db = _to_dbfs(metrics.rms) - noise.noise_floor_dbfs
    noise = NoiseReport(
        noise_floor_dbfs=noise.noise_floor_dbfs,
        rms_noise=noise.rms_noise,
        window_start_s=noise.window_start_s,
        window_end_s=noise.window_end_s,
        confidence=noise.confidence,
        method=noise.method,
        parameters=noise.parameters,
        signal_to_noise_ratio_db=signal_to_noise_ratio_db,
    )
    return TemporalResults(
        impact=detect_impact(signal),
        noise=noise,
        metrics=metrics,
        envelope=_configured_envelope(signal, effective_settings),
        settings=effective_settings,
    )


def _configured_envelope(
    signal: Signal,
    settings: TemporalAnalysisSettings,
) -> Envelope:
    """Seleciona o envelope configurado sem introduzir novos métodos."""
    if settings.envelope_method == "hilbert":
        return envelope_hilbert(signal)
    if settings.envelope_method == "cumulative_energy":
        return envelope_cumulative_energy(signal)
    if settings.envelope_method == "moving_rms":
        return envelope_moving_rms(
            signal,
            window_size_samples=settings.envelope_window_size_samples or 1,
        )
    return envelope_moving_peak(
        signal,
        window_size_samples=settings.envelope_window_size_samples or 1,
    )


def _normalized_matrix(signal: Signal) -> np.ndarray:
    """Converte canais do Signal em matriz temporal normalizada, internamente."""
    if signal.sample_rate <= 0:
        raise ValueError("signal.sample_rate must be positive.")
    samples = np.asarray(signal.samples)
    invalid_shape = (
        samples.ndim != 2
        or samples.shape[0] != signal.channels
        or samples.shape[1] == 0
    )
    if invalid_shape:
        raise ValueError("signal samples must be a non-empty channel-by-sample matrix.")

    if np.issubdtype(samples.dtype, np.integer):
        limits = np.iinfo(samples.dtype)
        full_scale = float(max(abs(limits.min), limits.max))
        return samples.astype(np.float64) / full_scale
    return samples.astype(np.float64, copy=False)


def _mono_waveform(signal: Signal) -> np.ndarray:
    """Combina canais normalizados por média para métricas globais temporais."""
    return np.mean(_normalized_matrix(signal), axis=0)


def _find_transient_end(
    amplitude: np.ndarray,
    peak_sample: int,
    *,
    threshold: float,
    hold_samples: int,
) -> int:
    """Localiza o primeiro trecho sustentadamente abaixo do limiar transiente."""
    for start in range(peak_sample + 1, amplitude.size):
        stop = min(amplitude.size, start + hold_samples)
        if stop - start == hold_samples and np.all(amplitude[start:stop] < threshold):
            return start
    return amplitude.size - 1


def _make_envelope(
    signal: Signal,
    amplitudes: np.ndarray,
    method: str,
    parameters: Mapping[str, object],
    *,
    unit: str | None = None,
) -> Envelope:
    """Constrói um Envelope imutável sem executar análise adicional."""
    return Envelope(
        times_s=tuple(index / signal.sample_rate for index in range(amplitudes.size)),
        amplitudes=tuple(float(value) for value in amplitudes),
        method=method,
        unit=signal.unit if unit is None else unit,
        parameters=parameters,
    )


def _to_dbfs(value: float) -> float:
    """Converte uma amplitude normalizada para dBFS, preservando o silêncio."""
    return 20.0 * log10(value) if value else -inf
