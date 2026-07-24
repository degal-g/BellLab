"""Leitura de arquivos WAV para o modelo de dados científico do BellLab.

Este módulo limita-se a validar e carregar áudio WAV em memória, preservando a
representação inteira ou de ponto flutuante apropriada ao subtipo do arquivo.
Ele também calcula métricas descritivas básicas da gravação. Não executa
transformadas, análises tempo-frequência ou identificação modal.
"""

from __future__ import annotations

from math import inf, log10
from pathlib import Path

import numpy as np
import soundfile as sf

from belllab.types import RecordingMetrics, Signal


class WavFormatError(ValueError):
    """Indica que um arquivo existe, mas não contém áudio WAV compatível."""


class WavReadError(OSError):
    """Indica que um arquivo WAV não pôde ser lido integralmente."""


_PCM_FORMATS: dict[str, tuple[str, float, float]] = {
    "PCM_U8": ("int16", 2.0**15, 2.0**8),
    "PCM_16": ("int16", 2.0**15, 1.0),
    "PCM_24": ("int32", 2.0**31, 2.0**8),
    "PCM_32": ("int32", 2.0**31, 1.0),
}
_FLOAT_DTYPES: dict[str, str] = {"FLOAT": "float32", "DOUBLE": "float64"}


def load_wav(path: str | Path) -> tuple[Signal, RecordingMetrics]:
    """Carrega um arquivo WAV e produz o sinal e suas métricas descritivas.

    A leitura aceita áudio mono ou multicanal, incluindo estéreo, nos subtipos
    PCM inteiro, float32 e float64 suportados por ``soundfile``. As amostras
    inteiras permanecem inteiras no ``Signal``; amostras de ponto flutuante
    preservam ``float32`` ou ``float64``. Métricas de amplitude são calculadas
    em escala digital normalizada de fundo de escala para permitir valores em
    dBFS sem alterar as amostras armazenadas.

    Args:
        path: Caminho do arquivo WAV a carregar.

    Returns:
        Uma tupla ``(signal, metrics)``. ``signal`` contém as amostras por
        canal, o eixo temporal e propriedades do áudio. ``metrics`` contém
        duração, dimensões, pico, RMS, fator de crista, nível máximo em dBFS,
        indicação de clipping e offset DC.

    Raises:
        FileNotFoundError: Se ``path`` não existir ou não for um arquivo.
        WavFormatError: Se o arquivo não for WAV ou usar um subtipo não
            suportado.
        WavReadError: Se o WAV estiver corrompido ou não puder ser decodificado.
    """
    wav_path = Path(path)
    if not wav_path.is_file():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    try:
        info = sf.info(wav_path)
    except (OSError, RuntimeError) as error:
        raise WavReadError(f"Could not read WAV file: {wav_path}") from error

    if info.format != "WAV":
        raise WavFormatError(
            f"Expected a WAV file, but detected {info.format!r}: {wav_path}"
        )

    dtype, full_scale, least_significant_bit = _select_dtype_and_scale(info.subtype)
    try:
        samples = sf.read(wav_path, dtype=dtype, always_2d=True)[0]
    except (OSError, RuntimeError) as error:
        raise WavReadError(f"Could not decode WAV audio data: {wav_path}") from error

    return _build_signal_and_metrics(
        samples,
        info.samplerate,
        full_scale,
        least_significant_bit,
    )


def _select_dtype_and_scale(subtype: str) -> tuple[str, float, float]:
    """Seleciona o tipo de leitura e a referência de fundo de escala.

    Esta função é interna à leitura WAV. Ela não transforma ou processa o
    conteúdo acústico; somente preserva o tipo numérico adequado ao subtipo.
    """
    if subtype in _PCM_FORMATS:
        return _PCM_FORMATS[subtype]
    if subtype in _FLOAT_DTYPES:
        return _FLOAT_DTYPES[subtype], 1.0, 0.0
    raise WavFormatError(f"Unsupported WAV subtype: {subtype}")


def _build_signal_and_metrics(
    samples: np.ndarray,
    sample_rate: int,
    full_scale: float,
    least_significant_bit: float,
) -> tuple[Signal, RecordingMetrics]:
    """Cria os objetos públicos a partir de amostras WAV já carregadas."""
    frame_count, channel_count = samples.shape
    duration = frame_count / sample_rate
    time = tuple((np.arange(frame_count, dtype=np.float64) / sample_rate).tolist())
    normalized = samples.astype(np.float64, copy=False) / full_scale
    peak = float(np.max(np.abs(normalized))) if normalized.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(normalized)))) if normalized.size else 0.0
    crest_factor = peak / rms if rms else None
    max_level_dbfs = 20.0 * log10(peak) if peak else -inf
    clipping_threshold = 1.0 - (least_significant_bit / full_scale)
    clipping_detected = bool(np.any(np.abs(normalized) >= clipping_threshold))
    dc_offset = float(np.mean(normalized)) if normalized.size else 0.0

    signal = Signal(
        samples=tuple(tuple(channel) for channel in samples.T),
        sample_rate=sample_rate,
        time=time,
        duration=duration,
        channels=channel_count,
        unit="digital",
    )
    metrics = RecordingMetrics(
        duration_s=duration,
        sample_rate_hz=sample_rate,
        channel_count=channel_count,
        sample_count=frame_count,
        peak=peak,
        rms=rms,
        crest_factor=crest_factor,
        max_level_dbfs=max_level_dbfs,
        clipping_detected=clipping_detected,
        dc_offset=dc_offset,
    )
    return signal, metrics
