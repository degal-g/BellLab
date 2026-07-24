"""Testes unitários dos geradores sintéticos do BellLab."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from belllab import Signal
from belllab.synthetic import (
    DampedSineComponent,
    controlled_clipping,
    damped_exponential,
    damped_exponential_sum,
    ideal_impulse,
    mix_signal_and_noise,
    pink_noise,
    pure_sine,
    sine_sum,
    white_noise,
    with_dc_offset,
)


NoiseGenerator = Callable[..., Signal]


def samples_of(signal: Signal) -> np.ndarray:
    """Converte as amostras mono de um Signal sintético para ndarray."""
    return np.asarray(signal.samples[0])


def test_pure_sine_has_requested_timebase_and_amplitude() -> None:
    """Uma senoide deve respeitar duração, taxa, contagem e amplitude."""
    signal = pure_sine(25.0, duration_s=1.0, sample_rate=100, amplitude=0.5)

    assert signal.duration == pytest.approx(1.0)
    assert signal.sample_rate == 100
    assert len(signal.samples[0]) == 100
    assert np.max(np.abs(samples_of(signal))) == pytest.approx(0.5)


def test_sine_sum_combines_arbitrary_components() -> None:
    """A soma deve respeitar amplitudes e fases declaradas das componentes."""
    signal = sine_sum(
        (1.0, 1.0),
        duration_s=1.0,
        sample_rate=10,
        amplitudes=(0.25, 0.75),
        phases_rad=(0.0, np.pi / 2),
    )

    assert samples_of(signal)[0] == pytest.approx(0.75)
    assert len(signal.time) == 10


def test_damped_generators_respect_initial_amplitude_and_duration() -> None:
    """Componentes amortecidas devem produzir sinais com a duração declarada."""
    single = damped_exponential(
        1.0,
        2.0,
        duration_s=1.0,
        sample_rate=100,
        amplitude=0.5,
        phase_rad=np.pi / 2,
    )
    combined = damped_exponential_sum(
        (DampedSineComponent(1.0, 2.0, 0.5, np.pi / 2),),
        duration_s=1.0,
        sample_rate=100,
    )

    assert samples_of(single)[0] == pytest.approx(0.5)
    assert samples_of(single)[-1] < samples_of(single)[0]
    assert samples_of(combined) == pytest.approx(samples_of(single))


def test_ideal_impulse_has_one_controlled_nonzero_sample() -> None:
    """Um impulso ideal deve ocupar somente a amostra solicitada."""
    signal = ideal_impulse(
        duration_s=0.1,
        sample_rate=100,
        sample_index=4,
        amplitude=-0.25,
    )

    assert len(signal.samples[0]) == 10
    assert np.count_nonzero(samples_of(signal)) == 1
    assert samples_of(signal)[4] == pytest.approx(-0.25)


@pytest.mark.parametrize("generator", [white_noise, pink_noise])
def test_noise_generators_are_seed_reproducible(
    generator: NoiseGenerator,
) -> None:
    """Uma mesma seed deve reproduzir exatamente cada ruído sintético."""
    first = generator(duration_s=0.25, sample_rate=1_000, seed=123)
    second = generator(duration_s=0.25, sample_rate=1_000, seed=123)
    different = generator(
        duration_s=0.25,
        sample_rate=1_000,
        seed=124,
    )

    assert first.duration == pytest.approx(0.25)
    assert first.sample_rate == 1_000
    assert len(first.samples[0]) == 250
    assert samples_of(first) == pytest.approx(samples_of(second))
    assert not np.array_equal(samples_of(first), samples_of(different))


def test_white_noise_has_requested_rms_amplitude() -> None:
    """O ruído branco deve ser normalizado para o RMS solicitado."""
    signal = white_noise(
        duration_s=0.1,
        sample_rate=1_000,
        rms_amplitude=0.2,
        seed=7,
    )

    assert np.sqrt(np.mean(np.square(samples_of(signal)))) == pytest.approx(0.2)


def test_clipping_offset_and_mixing_preserve_signal_dimensions() -> None:
    """Operações auxiliares devem manter tempo, taxa e tamanho do sinal."""
    base = pure_sine(5.0, duration_s=1.0, sample_rate=100, amplitude=1.0)
    clipped = controlled_clipping(base, threshold=0.4)
    offset = with_dc_offset(clipped, offset=0.1)
    noise = white_noise(duration_s=1.0, sample_rate=100, rms_amplitude=0.0, seed=1)
    mixed = mix_signal_and_noise(offset, noise)

    assert np.max(np.abs(samples_of(clipped))) <= 0.4
    assert np.mean(samples_of(offset) - samples_of(clipped)) == pytest.approx(0.1)
    assert mixed.duration == base.duration
    assert mixed.sample_rate == base.sample_rate
    assert len(mixed.samples[0]) == len(base.samples[0])
