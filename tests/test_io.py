"""Testes unitários para carregamento WAV sem análise espectral."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from belllab.io import WavFormatError, WavReadError, load_wav


def test_load_wav_reads_mono_pcm_and_metrics(tmp_path: Path) -> None:
    """O leitor deve preservar PCM mono e calcular métricas básicas."""
    path = tmp_path / "mono.wav"
    samples = np.array([0, 16_384, -32_768, 0], dtype=np.int16)
    sf.write(path, samples, 8_000, subtype="PCM_16")

    signal, metrics = load_wav(path)

    assert signal.channels == 1
    assert signal.samples[0] == tuple(samples)
    assert signal.sample_rate == 8_000
    assert signal.time == (0.0, 0.000125, 0.00025, 0.000375)
    assert metrics.sample_count == 4
    assert metrics.duration_s == pytest.approx(0.0005)
    assert metrics.peak == pytest.approx(1.0)
    assert metrics.clipping_detected is True
    assert metrics.max_level_dbfs == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("subtype", "dtype"),
    [("FLOAT", np.float32), ("DOUBLE", np.float64)],
)
def test_load_wav_preserves_float_precision(
    tmp_path: Path,
    subtype: str,
    dtype: type[np.floating],
) -> None:
    """O leitor deve manter o subtipo float32 ou float64 do WAV."""
    path = tmp_path / f"{subtype}.wav"
    samples = np.array([[0.25, -0.5], [0.125, 0.75]], dtype=dtype)
    sf.write(path, samples, 44_100, subtype=subtype)

    signal, metrics = load_wav(path)

    assert signal.channels == 2
    assert signal.samples == tuple(tuple(channel) for channel in samples.T)
    assert isinstance(signal.samples[0][0], dtype)
    assert metrics.peak == pytest.approx(0.75)
    assert metrics.clipping_detected is False


def test_load_wav_rejects_missing_file(tmp_path: Path) -> None:
    """Um caminho inexistente deve produzir uma exceção clara."""
    with pytest.raises(FileNotFoundError, match="not found"):
        load_wav(tmp_path / "missing.wav")


def test_load_wav_rejects_non_wav_file(tmp_path: Path) -> None:
    """Um arquivo de áudio existente que não é WAV deve ser rejeitado."""
    path = tmp_path / "audio.flac"
    sf.write(path, np.zeros(8, dtype=np.float32), 8_000, format="FLAC")

    with pytest.raises(WavFormatError, match="Expected a WAV"):
        load_wav(path)


def test_load_wav_rejects_corrupted_file(tmp_path: Path) -> None:
    """Um WAV inválido deve produzir uma exceção de leitura clara."""
    path = tmp_path / "corrupted.wav"
    path.write_bytes(b"not a valid wave file")

    with pytest.raises(WavReadError, match="Could not read WAV"):
        load_wav(path)
