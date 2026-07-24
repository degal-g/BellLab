"""Geração determinística de sinais artificiais para validação científica.

Este módulo fornece estímulos mono em memória para testes de algoritmos do
BellLab. Não executa análise, transformadas de Fourier, visualização ou escrita
em disco. Todas as funções retornam :class:`belllab.types.Signal`.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Sequence

import numpy as np

from belllab.types import Signal


@dataclass(frozen=True, slots=True)
class DampedSineComponent:
    """Define uma componente senoidal com decaimento exponencial.

    Args:
        frequency_hz: Frequência da componente, em hertz.
        decay_rate_per_s: Taxa de decaimento exponencial, por segundo.
        amplitude: Amplitude inicial da componente.
        phase_rad: Fase inicial, em radianos.
    """

    frequency_hz: float
    decay_rate_per_s: float
    amplitude: float = 1.0
    phase_rad: float = 0.0


def pure_sine(
    frequency_hz: float,
    *,
    duration_s: float,
    sample_rate: int,
    amplitude: float = 1.0,
    phase_rad: float = 0.0,
) -> Signal:
    """Gera uma senoide pura de duração e taxa de amostragem declaradas.

    Args:
        frequency_hz: Frequência da senoide, em hertz.
        duration_s: Duração do sinal, em segundos.
        sample_rate: Taxa de amostragem, em hertz.
        amplitude: Amplitude de pico da senoide.
        phase_rad: Fase inicial, em radianos.

    Returns:
        Um sinal mono normalizado contendo a senoide.
    """
    time = _time_axis(duration_s, sample_rate)
    samples = amplitude * np.sin((2.0 * pi * frequency_hz * time) + phase_rad)
    return _signal_from_samples(samples, sample_rate)


def linear_chirp(
    start_frequency_hz: float, end_frequency_hz: float, *, duration_s: float,
    sample_rate: int, amplitude: float = 1.0, phase_rad: float = 0.0,
) -> Signal:
    """Gera chirp linear determinístico de frequência inicial a final."""
    time = _time_axis(duration_s, sample_rate)
    rate = (end_frequency_hz - start_frequency_hz) / duration_s
    phase = 2.0 * pi * (start_frequency_hz * time + 0.5 * rate * time**2) + phase_rad
    return _signal_from_samples(amplitude * np.sin(phase), sample_rate)


def sine_sum(
    frequencies_hz: Sequence[float],
    *,
    duration_s: float,
    sample_rate: int,
    amplitudes: Sequence[float] | None = None,
    phases_rad: Sequence[float] | None = None,
) -> Signal:
    """Gera a soma de um número arbitrário de senoides.

    Args:
        frequencies_hz: Frequências das componentes, em hertz.
        duration_s: Duração do sinal, em segundos.
        sample_rate: Taxa de amostragem, em hertz.
        amplitudes: Amplitudes de pico das componentes. O padrão é um para cada
            frequência.
        phases_rad: Fases iniciais, em radianos. O padrão é zero para todas as
            componentes.

    Returns:
        Um sinal mono que contém a soma das componentes fornecidas.

    Raises:
        ValueError: Se as sequências de componentes possuírem comprimentos
            incompatíveis ou nenhuma frequência for fornecida.
    """
    if not frequencies_hz:
        raise ValueError("frequencies_hz must contain at least one component.")
    component_count = len(frequencies_hz)
    amplitudes = _component_values(amplitudes, component_count, 1.0, "amplitudes")
    phases_rad = _component_values(phases_rad, component_count, 0.0, "phases_rad")
    time = _time_axis(duration_s, sample_rate)
    samples = np.zeros(time.size, dtype=np.float64)
    for frequency, component_amplitude, phase in zip(
        frequencies_hz,
        amplitudes,
        phases_rad,
        strict=True,
    ):
        samples += component_amplitude * np.sin((2.0 * pi * frequency * time) + phase)
    return _signal_from_samples(samples, sample_rate)


def damped_exponential(
    frequency_hz: float,
    decay_rate_per_s: float,
    *,
    duration_s: float,
    sample_rate: int,
    amplitude: float = 1.0,
    phase_rad: float = 0.0,
) -> Signal:
    """Gera uma senoide com envelope exponencialmente amortecido.

    Args:
        frequency_hz: Frequência da senoide, em hertz.
        decay_rate_per_s: Taxa de decaimento exponencial, por segundo.
        duration_s: Duração do sinal, em segundos.
        sample_rate: Taxa de amostragem, em hertz.
        amplitude: Amplitude inicial da componente.
        phase_rad: Fase inicial, em radianos.

    Returns:
        Um sinal mono de senoide amortecida.
    """
    component = DampedSineComponent(
        frequency_hz=frequency_hz,
        decay_rate_per_s=decay_rate_per_s,
        amplitude=amplitude,
        phase_rad=phase_rad,
    )
    return damped_exponential_sum(
        (component,),
        duration_s=duration_s,
        sample_rate=sample_rate,
    )


def damped_exponential_sum(
    components: Sequence[DampedSineComponent],
    *,
    duration_s: float,
    sample_rate: int,
) -> Signal:
    """Gera uma soma de componentes exponencialmente amortecidas.

    Este modelo simplificado é útil como sinal sintético de idiofone percutido:
    cada componente representa uma oscilação amortecida independente, sem
    afirmar que o modelo descreve integralmente um instrumento físico.

    Args:
        components: Componentes amortecidas a combinar.
        duration_s: Duração do sinal, em segundos.
        sample_rate: Taxa de amostragem, em hertz.

    Returns:
        Um sinal mono que contém a soma das componentes.

    Raises:
        ValueError: Se nenhuma componente for fornecida.
    """
    if not components:
        raise ValueError("components must contain at least one component.")
    time = _time_axis(duration_s, sample_rate)
    samples = np.zeros(time.size, dtype=np.float64)
    for component in components:
        envelope = np.exp(-component.decay_rate_per_s * time)
        carrier = np.sin(
            (2.0 * pi * component.frequency_hz * time) + component.phase_rad
        )
        samples += component.amplitude * envelope * carrier
    return _signal_from_samples(samples, sample_rate)


def ideal_impulse(
    *,
    duration_s: float,
    sample_rate: int,
    sample_index: int = 0,
    amplitude: float = 1.0,
) -> Signal:
    """Gera um impulso ideal de uma amostra em posição controlada.

    Args:
        duration_s: Duração do sinal, em segundos.
        sample_rate: Taxa de amostragem, em hertz.
        sample_index: Índice da amostra que receberá o impulso.
        amplitude: Amplitude do impulso.

    Returns:
        Um sinal mono nulo exceto pelo impulso especificado.

    Raises:
        ValueError: Se ``sample_index`` não pertencer ao sinal gerado.
    """
    time = _time_axis(duration_s, sample_rate)
    if not 0 <= sample_index < time.size:
        raise ValueError("sample_index must refer to an existing sample.")
    samples = np.zeros(time.size, dtype=np.float64)
    samples[sample_index] = amplitude
    return _signal_from_samples(samples, sample_rate)


def white_noise(
    *,
    duration_s: float,
    sample_rate: int,
    rms_amplitude: float = 1.0,
    seed: int | None = None,
) -> Signal:
    """Gera ruído branco de média zero e RMS controlado.

    Args:
        duration_s: Duração do sinal, em segundos.
        sample_rate: Taxa de amostragem, em hertz.
        rms_amplitude: RMS alvo do ruído gerado.
        seed: Semente opcional do gerador pseudoaleatório local.

    Returns:
        Um sinal mono de ruído branco. A mesma ``seed`` e os mesmos parâmetros
        produzem exatamente as mesmas amostras.
    """
    time = _time_axis(duration_s, sample_rate)
    samples = np.random.default_rng(seed).standard_normal(time.size)
    return _signal_from_samples(_set_rms(samples, rms_amplitude), sample_rate)


def pink_noise(
    *,
    duration_s: float,
    sample_rate: int,
    rms_amplitude: float = 1.0,
    seed: int | None = None,
) -> Signal:
    """Gera ruído rosa aproximado por filtragem recursiva no tempo.

    O gerador usa a aproximação de filtro de Paul Kellet, aplicada a uma fonte
    branca determinística. Não há uso de FFT. A saída é reescalada para o RMS
    solicitado, facilitando a construção de casos sintéticos comparáveis.

    Args:
        duration_s: Duração do sinal, em segundos.
        sample_rate: Taxa de amostragem, em hertz.
        rms_amplitude: RMS alvo do ruído gerado.
        seed: Semente opcional do gerador pseudoaleatório local.

    Returns:
        Um sinal mono de ruído rosa aproximado e reproduzível por ``seed``.

    References:
        P. Kellet, "Refined Noise Formulas," 2002. Implementação de filtro
        recursivo amplamente usada para síntese aproximada de ruído rosa.
    """
    time = _time_axis(duration_s, sample_rate)
    source = np.random.default_rng(seed).standard_normal(time.size)
    samples = _kellet_pink_filter(source)
    return _signal_from_samples(_set_rms(samples, rms_amplitude), sample_rate)


def controlled_clipping(signal: Signal, *, threshold: float) -> Signal:
    """Cria uma cópia de sinal com clipping simétrico em limiar controlado.

    Args:
        signal: Sinal mono de origem.
        threshold: Limite positivo de amplitude após clipping.

    Returns:
        Um novo sinal com amostras limitadas ao intervalo simétrico definido.

    Raises:
        ValueError: Se o limiar não for positivo ou o sinal não for mono.
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive.")
    samples = _mono_samples(signal)
    clipped = np.clip(samples, -threshold, threshold)
    return _signal_from_samples(clipped, signal.sample_rate)


def with_dc_offset(signal: Signal, *, offset: float) -> Signal:
    """Cria uma cópia de sinal mono com offset DC adicionado.

    Args:
        signal: Sinal mono de origem.
        offset: Valor constante a acrescentar a cada amostra.

    Returns:
        Um novo sinal com o offset solicitado e a mesma base temporal.
    """
    return _signal_from_samples(_mono_samples(signal) + offset, signal.sample_rate)


def mix_signal_and_noise(
    signal: Signal,
    noise: Signal,
    *,
    signal_gain: float = 1.0,
    noise_gain: float = 1.0,
) -> Signal:
    """Mistura dois sinais mono de mesma duração e taxa de amostragem.

    Args:
        signal: Componente determinística ou sinal de interesse.
        noise: Componente de ruído a adicionar.
        signal_gain: Ganho linear aplicado ao sinal de interesse.
        noise_gain: Ganho linear aplicado ao ruído.

    Returns:
        Um novo sinal mono que contém a soma ponderada das entradas.

    Raises:
        ValueError: Se taxa de amostragem ou número de amostras diferirem.
    """
    signal_samples = _mono_samples(signal)
    noise_samples = _mono_samples(noise)
    if signal.sample_rate != noise.sample_rate:
        raise ValueError("signal and noise must have the same sample_rate.")
    if signal_samples.size != noise_samples.size:
        raise ValueError("signal and noise must have the same number of samples.")
    samples = (signal_gain * signal_samples) + (noise_gain * noise_samples)
    return _signal_from_samples(samples, signal.sample_rate)


def _time_axis(duration_s: float, sample_rate: int) -> np.ndarray:
    """Cria um eixo temporal discreto e valida os parâmetros comuns."""
    if duration_s <= 0:
        raise ValueError("duration_s must be positive.")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    sample_count = round(duration_s * sample_rate)
    if sample_count <= 0:
        raise ValueError("duration_s and sample_rate must produce at least one sample.")
    return np.arange(sample_count, dtype=np.float64) / sample_rate


def _component_values(
    values: Sequence[float] | None,
    component_count: int,
    default: float,
    name: str,
) -> Sequence[float]:
    """Valida ou completa valores opcionais associados a componentes."""
    if values is None:
        return (default,) * component_count
    if len(values) != component_count:
        raise ValueError(f"{name} must match the number of frequencies.")
    return values


def _set_rms(samples: np.ndarray, rms_amplitude: float) -> np.ndarray:
    """Reescala uma sequência não nula para o RMS solicitado."""
    if rms_amplitude < 0:
        raise ValueError("rms_amplitude must not be negative.")
    current_rms = float(np.sqrt(np.mean(np.square(samples))))
    if current_rms == 0.0:
        return np.zeros_like(samples)
    return samples * (rms_amplitude / current_rms)


def _kellet_pink_filter(source: np.ndarray) -> np.ndarray:
    """Aplica a aproximação recursiva de ruído rosa de Kellet no tempo."""
    output = np.empty_like(source)
    b0 = b1 = b2 = b3 = b4 = b5 = b6 = 0.0
    for index, value in enumerate(source):
        b0 = (0.99886 * b0) + (value * 0.0555179)
        b1 = (0.99332 * b1) + (value * 0.0750759)
        b2 = (0.96900 * b2) + (value * 0.1538520)
        b3 = (0.86650 * b3) + (value * 0.3104856)
        b4 = (0.55000 * b4) + (value * 0.5329522)
        b5 = (-0.7616 * b5) - (value * 0.0168980)
        output[index] = b0 + b1 + b2 + b3 + b4 + b5 + b6 + (value * 0.5362)
        b6 = value * 0.115926
    return output


def _mono_samples(signal: Signal) -> np.ndarray:
    """Extrai amostras mono normalizadas de Signal para geração auxiliar."""
    if signal.channels != 1 or len(signal.samples) != 1:
        raise ValueError("synthetic signal operations require a mono Signal.")
    return np.asarray(signal.samples[0], dtype=np.float64)


def _signal_from_samples(samples: np.ndarray, sample_rate: int) -> Signal:
    """Cria um Signal mono imutável a partir de amostras já geradas."""
    values = np.asarray(samples, dtype=np.float64)
    return Signal(
        samples=(tuple(values),),
        sample_rate=sample_rate,
        time=tuple(np.arange(values.size, dtype=np.float64) / sample_rate),
        duration=values.size / sample_rate,
        channels=1,
        unit="normalized",
    )
