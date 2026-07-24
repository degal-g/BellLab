"""Análises espectrais científicas atualmente implementadas no BellLab.

O módulo fornece FFT estacionária unilateral, detecção e caracterização de
picos espectrais, e STFT unilateral. Rastreamento de picos, trajetórias
espectrais e interpretação modal física permanecem fora de escopo.
"""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths

from belllab.config import (
    AnalysisSettings,
    PeakDetectionSettings,
    STFTSettings,
    SpectrumAnalysisSettings,
)
from belllab.results import PeakDetectionResults, STFTResults, SpectrumResults
from belllab.types import Signal, SpectralPeak, Spectrum, TimeFrequencySpectrum


def analyze_spectrum(
    signal: Signal,
    settings: AnalysisSettings | SpectrumAnalysisSettings | None = None,
) -> SpectrumResults:
    """Calcula uma FFT real unilateral de amplitude normalizada.

    O sinal é organizado como ``(canais, amostras)``. Por padrão, somente o
    canal de índice zero é analisado; a média entre canais ocorre apenas quando
    ``channel_policy="mean"`` é escolhida explicitamente. O intervalo temporal
    é semiaberto, ``[start_time_s, end_time_s)``.

    A janela é aplicada após a remoção opcional de média. A magnitude linear é
    normalizada pelo número de amostras originais e pelo ganho coerente da
    janela. Os bins internos do espectro unilateral são duplicados; DC e
    Nyquist (quando ``n_fft`` é par) não são. Assim, uma senoide real centrada
    em um bin recupera sua amplitude de pico, sujeita a erro numérico.

    ``dbfs`` representa amplitude, e não potência: ``0 dBFS`` corresponde a
    amplitude linear normalizada de um. Bins nulos recebem ``-inf``; NaN nunca
    é produzido pela conversão de escala.

    Args:
        signal: Sinal de entrada mono ou multicanal.
        settings: Configuração espectral isolada ou configuração agregada. O
            padrão usa ``AnalysisSettings().spectrum``.

    Returns:
        ``SpectrumResults`` com um ``Spectrum`` e a configuração efetivamente
        usada.

    Raises:
        ValueError: Se o trecho, o canal ou ``n_fft`` forem incompatíveis com o
            sinal.
    """
    effective_settings = _resolve_settings(settings)
    samples = _normalized_matrix(signal)
    waveform, channel_description = _select_channel(samples, effective_settings)
    start_index, end_index = _select_interval(signal, effective_settings)
    segment = waveform[start_index:end_index]
    n_samples = segment.size
    n_fft = effective_settings.n_fft or n_samples
    if n_fft < n_samples:
        raise ValueError("n_fft must be at least the selected segment length.")

    prepared = segment.copy()
    if effective_settings.remove_mean:
        prepared -= np.mean(prepared)
    window = _window(effective_settings.window_name, n_samples)
    coherent_gain = float(np.mean(window))
    if coherent_gain == 0.0:
        raise ValueError("window coherent gain must not be zero.")

    coefficients = np.fft.rfft(prepared * window, n=n_fft)
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / signal.sample_rate)
    magnitudes = _one_sided_amplitude(coefficients, n_samples, coherent_gain, n_fft)
    magnitude_values, magnitude_unit = _scale_magnitudes(
        magnitudes,
        effective_settings.scale,
    )
    start_time_s = start_index / signal.sample_rate
    end_time_s = end_index / signal.sample_rate
    spectrum = Spectrum(
        frequencies_hz=tuple(float(value) for value in frequencies),
        magnitudes=tuple(float(value) for value in magnitude_values),
        magnitude_unit=magnitude_unit,
        window_name=effective_settings.window_name,
        fft_size=n_fft,
        overlap=None,
        timestamp=start_time_s,
        sample_rate_hz=signal.sample_rate,
        original_size=n_samples,
        bin_spacing_hz=signal.sample_rate / n_fft,
        channel_policy=effective_settings.channel_policy,
        channel_index=(
            effective_settings.channel_index
            if effective_settings.channel_policy == "select"
            else None
        ),
        normalization=effective_settings.normalization,
        interval_start_s=start_time_s,
        interval_end_s=end_time_s,
        remove_mean=effective_settings.remove_mean,
        parameters=MappingProxyType(
            {
                "coherent_gain": coherent_gain,
                "channel_description": channel_description,
                "dbfs_reference_amplitude": 1.0,
                "one_sided_interior_bins_doubled": True,
            }
        ),
    )
    diagnostics = ()
    if n_fft > n_samples:
        diagnostics = ("zero padding applied; bin spacing is interpolated",)
    return SpectrumResults(
        spectrum=spectrum,
        settings=effective_settings,
        diagnostics=diagnostics,
    )


def _resolve_settings(
    settings: AnalysisSettings | SpectrumAnalysisSettings | None,
) -> SpectrumAnalysisSettings:
    """Obtém a configuração espectral efetiva sem ignorar a entrada pública."""
    if settings is None:
        return AnalysisSettings().spectrum
    if isinstance(settings, AnalysisSettings):
        return settings.spectrum
    return settings


def _normalized_matrix(signal: Signal) -> np.ndarray:
    """Converte canais para matriz float64 em escala digital normalizada."""
    samples = np.asarray(signal.samples)
    if samples.ndim != 2 or samples.shape[0] != signal.channels:
        raise ValueError("signal samples must be a channel-by-sample matrix.")
    if samples.shape[1] == 0:
        raise ValueError("spectrum analysis requires at least one sample.")
    if np.issubdtype(samples.dtype, np.integer):
        limits = np.iinfo(samples.dtype)
        full_scale = float(max(abs(limits.min), limits.max))
        return samples.astype(np.float64) / full_scale
    return samples.astype(np.float64, copy=False)


def _select_channel(
    samples: np.ndarray,
    settings: SpectrumAnalysisSettings,
) -> tuple[np.ndarray, str]:
    """Aplica a política explícita de canais sem média implícita."""
    if settings.channel_policy == "select":
        if settings.channel_index >= samples.shape[0]:
            raise ValueError("channel_index is outside the available channel range.")
        description = f"channel:{settings.channel_index}"
        return samples[settings.channel_index].copy(), description
    return np.mean(samples, axis=0), "mean"


def _select_interval(
    signal: Signal,
    settings: SpectrumAnalysisSettings,
) -> tuple[int, int]:
    """Converte o intervalo semiaberto configurado em índices de amostra."""
    start_time_s = (
        settings.start_time_s if settings.start_time_s is not None else 0.0
    )
    end_time_s = (
        settings.end_time_s if settings.end_time_s is not None else signal.duration
    )
    if start_time_s >= signal.duration:
        raise ValueError("start_time_s must be within the signal duration.")
    if end_time_s > signal.duration + 1e-12:
        raise ValueError("end_time_s must not exceed the signal duration.")
    start_index = int(np.ceil(start_time_s * signal.sample_rate))
    end_index = int(np.ceil(end_time_s * signal.sample_rate))
    if end_index <= start_index:
        raise ValueError("selected interval must contain at least one sample.")
    return start_index, end_index


def _window(name: str, size: int) -> np.ndarray:
    """Constrói uma janela suportada para a FFT estacionária."""
    if name == "rectangular":
        return np.ones(size, dtype=np.float64)
    if name == "hann":
        return np.hanning(size)
    raise ValueError(f"unsupported window_name: {name}")


def _one_sided_amplitude(
    coefficients: np.ndarray,
    original_size: int,
    coherent_gain: float,
    n_fft: int,
    frequency_axis: int = 0,
) -> np.ndarray:
    """Normaliza uma RFFT unilateral para amplitude de pico de senoides reais.

    ``frequency_axis`` identifica o eixo de bins da RFFT. Isso permite que a
    FFT estacionária e a STFT usem precisamente a mesma convenção para DC,
    Nyquist e duplicação dos bins internos.
    """
    magnitudes = np.abs(coefficients) / (original_size * coherent_gain)
    if magnitudes.shape[frequency_axis] > 1:
        index = [slice(None)] * magnitudes.ndim
        if n_fft % 2 == 0:
            index[frequency_axis] = slice(1, -1)
        else:
            index[frequency_axis] = slice(1, None)
        magnitudes[tuple(index)] *= 2.0
    return magnitudes


def _scale_magnitudes(
    magnitudes: np.ndarray,
    scale: str,
) -> tuple[np.ndarray, str]:
    """Aplica a escala de amplitude solicitada sem gerar NaN para zeros."""
    if scale == "linear_amplitude":
        return magnitudes, "normalized amplitude (peak)"
    values = np.full(magnitudes.shape, -np.inf, dtype=np.float64)
    positive = magnitudes > 0.0
    values[positive] = 20.0 * np.log10(magnitudes[positive])
    return values, "dBFS amplitude (ref=1.0)"


def detect_spectral_peaks(
    spectrum: Spectrum,
    settings: PeakDetectionSettings | None = None,
) -> PeakDetectionResults:
    """Detecta e caracteriza picos matemáticos em um Spectrum já calculado.

    Nenhuma FFT é recalculada. Os limiares de altura e proeminência usam a
    escala do espectro: lineares para amplitude linear e dB para dBFS. A SNR
    local é sempre expressa em dB, como razão de amplitudes lineares ou como
    diferença de níveis dB. O piso é uma mediana local operacional, não uma
    medição física de ruído.
    """
    effective = settings or PeakDetectionSettings()
    frequencies = np.asarray(spectrum.frequencies_hz, dtype=np.float64)
    display_values = np.asarray(spectrum.magnitudes, dtype=np.float64)
    linear_values = _linear_magnitudes(display_values, spectrum.magnitude_unit)
    mask = np.ones(frequencies.size, dtype=bool)
    if effective.min_frequency_hz is not None:
        mask &= frequencies >= effective.min_frequency_hz
    if effective.max_frequency_hz is not None:
        mask &= frequencies <= effective.max_frequency_hz
    indices = np.flatnonzero(mask)
    if indices.size < 3:
        raise ValueError("analyzed frequency range must contain at least three bins.")
    values = display_values[indices]
    if not np.any(np.isfinite(values)):
        return PeakDetectionResults(
            spectrum=spectrum,
            peaks=(),
            settings=effective,
            method="scipy-find-peaks-v1",
            warnings=("spectrum contains no finite values in the analyzed range",),
            analyzed_frequency_range_hz=_analyzed_range(frequencies, indices),
        )
    candidate_values = np.where(np.isfinite(values), values, -np.inf)
    peaks, _ = find_peaks(
        candidate_values,
        height=effective.min_amplitude,
        prominence=(effective.min_prominence, None),
        distance=effective.distance_bins,
        width=(effective.min_width_bins, effective.max_width_bins),
    )
    candidate_count = int(peaks.size)
    if not candidate_count:
        return PeakDetectionResults(
            spectrum=spectrum,
            peaks=(),
            settings=effective,
            method="scipy-find-peaks-v1",
            candidate_count=candidate_count,
            accepted_count=0,
            analyzed_frequency_range_hz=_analyzed_range(frequencies, indices),
        )
    prominences = peak_prominences(candidate_values, peaks)[0]
    widths = peak_widths(candidate_values, peaks, rel_height=0.5)[0]
    bin_spacing = spectrum.bin_spacing_hz or _infer_bin_spacing(frequencies)
    records = [
        _build_peak(
            int(indices[peak]),
            spectrum,
            display_values,
            linear_values,
            float(prominence),
            float(width),
            bin_spacing,
            effective,
            indices,
        )
        for peak, prominence, width in zip(peaks, prominences, widths, strict=True)
    ]
    records = _sort_peaks(records, effective.sort_by)
    if effective.max_peaks is not None:
        records = records[: effective.max_peaks]
    return PeakDetectionResults(
        spectrum=spectrum,
        peaks=tuple(records),
        settings=effective,
        method="scipy-find-peaks-v1",
        candidate_count=candidate_count,
        accepted_count=len(records),
        analyzed_frequency_range_hz=_analyzed_range(frequencies, indices),
    )


def _linear_magnitudes(values: np.ndarray, unit: str) -> np.ndarray:
    """Converte amplitude dBFS em amplitude linear para interpolação e SNR."""
    if "dbfs" not in unit.lower():
        return values.copy()
    linear = np.zeros(values.shape, dtype=np.float64)
    finite = np.isfinite(values)
    linear[finite] = np.power(10.0, values[finite] / 20.0)
    return linear


def _build_peak(
    index: int,
    spectrum: Spectrum,
    display: np.ndarray,
    linear: np.ndarray,
    prominence: float,
    width_bins: float,
    bin_spacing: float,
    settings: PeakDetectionSettings,
    analyzed_indices: np.ndarray,
) -> SpectralPeak:
    """Constrói um contrato de pico sem expor estruturas internas do SciPy."""
    refined_frequency, refined_amplitude, interpolation, diagnostics = (
        _interpolate_peak(
            index,
            spectrum,
            linear,
            settings,
            analyzed_indices,
        )
    )
    floor = _local_noise_floor(
        display,
        index,
        settings.noise_window_bins,
        analyzed_indices,
    )
    snr = _local_snr_db(display[index], floor, spectrum.magnitude_unit)
    if floor is None:
        diagnostics = (*diagnostics, "local_noise_floor_unavailable")
    return SpectralPeak(
        bin_index=index,
        bin_frequency_hz=spectrum.frequencies_hz[index],
        bin_amplitude=float(display[index]),
        amplitude_unit=spectrum.magnitude_unit,
        refined_frequency_hz=refined_frequency,
        refined_amplitude=refined_amplitude,
        prominence=prominence,
        width=width_bins,
        width_bins=width_bins,
        width_hz=width_bins * bin_spacing,
        local_noise_floor=floor,
        local_snr_db=snr,
        interpolation_method=interpolation,
        width_method="half-prominence",
        diagnostics=diagnostics,
    )


def _interpolate_peak(
    index: int,
    spectrum: Spectrum,
    linear: np.ndarray,
    settings: PeakDetectionSettings,
    analyzed_indices: np.ndarray,
) -> tuple[float | None, float | None, str | None, tuple[str, ...]]:
    """Aplica interpolação parabólica logarítmica de três pontos quando válida."""
    if not settings.interpolate:
        return None, None, None, ()
    if index == 0 or index == linear.size - 1 or index not in analyzed_indices:
        return None, None, None, ("interpolation_unavailable_at_edge",)
    triplet = linear[index - 1 : index + 2]
    if np.any(triplet <= 0) or not np.all(np.isfinite(triplet)):
        return None, None, None, ("interpolation_unavailable_nonpositive_neighbor",)
    left, center, right = np.log(triplet)
    denominator = left - (2.0 * center) + right
    if denominator == 0.0:
        return None, None, None, ("interpolation_unavailable_flat_curvature",)
    offset = 0.5 * (left - right) / denominator
    if not -1.0 < offset < 1.0:
        return None, None, None, ("interpolation_unavailable_outside_neighbors",)
    frequencies = np.asarray(spectrum.frequencies_hz)
    spacing = spectrum.bin_spacing_hz or _infer_bin_spacing(frequencies)
    amplitude = float(np.exp(center - 0.25 * (left - right) * offset))
    if "dbfs" in spectrum.magnitude_unit.lower():
        amplitude = 20.0 * np.log10(amplitude)
    return (
        float(spectrum.frequencies_hz[index] + (offset * spacing)),
        amplitude,
        "parabolic_log_magnitude",
        (),
    )


def _local_noise_floor(
    values: np.ndarray,
    index: int,
    window_bins: int,
    analyzed_indices: np.ndarray,
) -> float | None:
    """Estima a mediana local excluindo o pico e seus vizinhos imediatos."""
    lower = max(int(analyzed_indices[0]), index - window_bins // 2)
    upper = min(int(analyzed_indices[-1]) + 1, index + window_bins // 2 + 1)
    before = values[lower : max(lower, index - 1)]
    after = values[index + 2 : upper]
    neighborhood = np.r_[before, after]
    finite = neighborhood[np.isfinite(neighborhood)]
    return float(np.median(finite)) if finite.size else None


def _local_snr_db(peak: float, floor: float | None, unit: str) -> float | None:
    """Calcula SNR local operacional em dB sem misturar escalas."""
    if floor is None:
        return None
    if "dbfs" in unit.lower():
        return peak - floor
    if floor <= 0 or peak <= 0:
        return None
    return 20.0 * np.log10(peak / floor)


def _infer_bin_spacing(frequencies: np.ndarray) -> float:
    """Obtém espaçamento uniforme quando metadados legados não o contêm."""
    if frequencies.size < 2:
        raise ValueError("at least two frequency bins are required.")
    return float(frequencies[1] - frequencies[0])


def _analyzed_range(
    frequencies: np.ndarray,
    indices: np.ndarray,
) -> tuple[float, float]:
    """Retorna os extremos inclusivos da faixa de bins analisados."""
    return float(frequencies[indices[0]]), float(frequencies[indices[-1]])


def _sort_peaks(peaks: list[SpectralPeak], key: str) -> list[SpectralPeak]:
    """Ordena contratos sem alterar o critério de aceitação do detector."""
    if key == "frequency":
        return sorted(peaks, key=lambda peak: peak.bin_frequency_hz)
    if key == "amplitude":
        return sorted(peaks, key=lambda peak: peak.bin_amplitude, reverse=True)
    return sorted(peaks, key=lambda peak: peak.prominence or -np.inf, reverse=True)


def analyze_stft(
    signal: Signal,
    settings: AnalysisSettings | STFTSettings | None = None,
) -> STFTResults:
    """Calcula uma STFT unilateral com ``values[frequency, time]``.

    Os tempos públicos são os centros das janelas. ``remove_mean=True``
    significa ``frame_mean``: a média é subtraída independentemente de cada
    quadro antes da janela e da FFT, e não do trecho inteiro. O trecho temporal
    é semiaberto, ``[start_time_s, end_time_s)``; segundos são convertidos por
    ``ceil(t * sample_rate)`` e os índices finais são exclusivos.

    Sem ``pad_end``, quadros finais incompletos são descartados. Com padding,
    somente as amostras necessárias ao último quadro completo são preenchidas
    com zero. O centro desse quadro pode ficar além dos dados reais, mas nunca
    representa uma extensão da duração registrada do sinal.
    """
    if settings is None:
        cfg = STFTSettings()
    elif isinstance(settings, AnalysisSettings):
        cfg = settings.stft
    elif isinstance(settings, STFTSettings):
        cfg = settings
    else:
        raise TypeError("settings must be STFTSettings, AnalysisSettings, or None.")
    matrix = _normalized_matrix(signal)
    diagnostics: list[str] = []
    if cfg.channel_policy == "select":
        if cfg.channel_index >= matrix.shape[0]:
            raise ValueError("channel_index is outside the available channel range.")
        waveform = matrix[cfg.channel_index]
        channel_index: int | None = cfg.channel_index
        diagnostics.append(f"channel_selected:{cfg.channel_index}")
    else:
        waveform = np.mean(matrix, axis=0)
        channel_index = None
        diagnostics.append("channels_meaned_explicitly")
    start, end = _select_interval(signal, cfg)
    segment = waveform[start:end]
    if not segment.size:
        raise ValueError("selected interval must contain samples.")
    padded = 0
    discarded = 0
    if segment.size < cfg.window_length:
        if not cfg.pad_end:
            raise ValueError("signal segment is shorter than window_length.")
        padded = cfg.window_length - segment.size
        segment = np.pad(segment, (0, padded))
        diagnostics.append("segment_shorter_than_window")
    remainder = (segment.size - cfg.window_length) % cfg.hop_length
    if remainder and cfg.pad_end:
        padded += cfg.hop_length - remainder
        segment = np.pad(segment, (0, cfg.hop_length - remainder))
    elif remainder:
        discarded = remainder
        diagnostics.append(f"final_incomplete_segment_discarded:{discarded}")
    frame_count = 1 + (segment.size - cfg.window_length) // cfg.hop_length
    frames = np.stack([segment[i * cfg.hop_length:i * cfg.hop_length + cfg.window_length] for i in range(frame_count)])
    if cfg.remove_mean:
        frames = frames - np.mean(frames, axis=1, keepdims=True)
        diagnostics.append("frame_mean_removed")
    window = _window(cfg.window_name, cfg.window_length)
    gain = float(np.mean(window))
    n_fft = cfg.n_fft or cfg.window_length
    coefficients = np.fft.rfft(frames * window, n=n_fft, axis=1)
    values = _one_sided_amplitude(
        coefficients,
        cfg.window_length,
        gain,
        n_fft,
        frequency_axis=1,
    )
    display, unit = _scale_magnitudes(values, cfg.scale)
    if cfg.scale == "dbfs" and np.any(np.isneginf(display)):
        diagnostics.append("dbfs_contains_negative_infinity")
    frequencies = np.fft.rfftfreq(n_fft, 1 / signal.sample_rate)
    nyquist = signal.sample_rate / 2
    if cfg.frequency_min_hz is not None and cfg.frequency_min_hz > nyquist:
        raise ValueError("frequency_min_hz must not exceed Nyquist.")
    if cfg.frequency_max_hz is not None and cfg.frequency_max_hz > nyquist:
        raise ValueError("frequency_max_hz must not exceed Nyquist.")
    mask = np.ones(frequencies.size, dtype=bool)
    if cfg.frequency_min_hz is not None: mask &= frequencies >= cfg.frequency_min_hz
    if cfg.frequency_max_hz is not None: mask &= frequencies <= cfg.frequency_max_hz
    if not np.any(mask):
        raise ValueError("frequency range does not contain an FFT bin.")
    if cfg.frequency_min_hz is not None or cfg.frequency_max_hz is not None:
        diagnostics.append("frequency_range_cropped")
    if n_fft > cfg.window_length:
        diagnostics.append("spectral_zero_padding_applied")
    if padded:
        diagnostics.extend(("padding_end_applied", f"padded_samples={padded}"))
    centers = (start + np.arange(frame_count) * cfg.hop_length + cfg.window_length / 2) / signal.sample_rate
    tf = TimeFrequencySpectrum(
        times_s=tuple(float(x) for x in centers), frequencies_hz=tuple(float(x) for x in frequencies[mask]),
        values=tuple(tuple(float(x) for x in row) for row in display[:, mask].T), magnitude_unit=unit,
        sample_rate_hz=signal.sample_rate, window_length=cfg.window_length, fft_size=n_fft,
        hop_length=cfg.hop_length, bin_spacing_hz=signal.sample_rate / n_fft,
        frame_spacing_s=cfg.hop_length / signal.sample_rate, window_name=cfg.window_name,
        coherent_gain=gain, channel_policy=cfg.channel_policy, channel_index=channel_index,
        interval_start_s=start / signal.sample_rate, interval_end_s=end / signal.sample_rate,
        padding_policy="zero_pad_end" if cfg.pad_end else "discard_incomplete",
        parameters=MappingProxyType(
            {
                "padded_samples": padded,
                "discarded_samples": discarded,
                "detrend_method": "frame_mean" if cfg.remove_mean else "none",
            }
        ),
    )
    return STFTResults(tf, cfg, diagnostics=tuple(diagnostics))
