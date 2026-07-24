"""Estruturas de dados fundamentais para análises acústicas do BellLab.

As dataclasses deste módulo representam medições e séries científicas. Elas
substituem dicionários ad hoc nas interfaces de análise para tornar contratos,
unidades e resultados explícitos e verificáveis por type checkers.

Nenhuma classe executa processamento de sinais nesta etapa.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from math import isclose, isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, slots=True)
class RecordingMetrics:
    """Métricas descritivas de uma gravação de áudio.

    Args:
        duration_s: Duração total da gravação, em segundos.
        sample_rate_hz: Taxa de amostragem, em hertz.
        channel_count: Número de canais de áudio.
        sample_count: Número total de amostras por canal.
        peak: Pico absoluto, normalizado para a escala digital de fundo de
            escala.
        rms: Amplitude RMS, normalizada para a escala digital de fundo de
            escala.
        crest_factor: Razão entre ``peak`` e ``rms``; ``None`` para sinal nulo.
        max_level_dbfs: Nível máximo em dBFS; ``-inf`` para sinal nulo.
        clipping_detected: Indica a presença de amostras no limite da escala.
        dc_offset: Média das amostras, normalizada para fundo de escala.
        peak_dbfs: Nível do pico em dBFS, quando disponível.
        rms_dbfs: Nível RMS em dBFS, quando disponível.
        crest_factor_db: Fator de crista em decibéis, quando disponível.
        clipping_fraction: Fração de amostras identificadas como clipping,
            quando disponível.
        clipping_sample_count: Quantidade de amostras identificadas como
            clipping, quando disponível.

    Cada campo poderá ser preenchido pelo futuro carregador WAV. Esta classe
    apenas preserva os valores que forem obtidos.
    """

    duration_s: float
    sample_rate_hz: int
    channel_count: int
    sample_count: int
    peak: float | None = None
    rms: float | None = None
    crest_factor: float | None = None
    max_level_dbfs: float | None = None
    clipping_detected: bool | None = None
    dc_offset: float | None = None
    peak_dbfs: float | None = None
    rms_dbfs: float | None = None
    crest_factor_db: float | None = None
    clipping_fraction: float | None = None
    clipping_sample_count: int | None = None


@dataclass(frozen=True, slots=True)
class Signal:
    """Representa um sinal de áudio carregado e disponível em memória.

    Args:
        samples: Amostras por canal, organizadas como uma sequência imutável de
            canais; cada canal contém suas amostras em ordem temporal.
        sample_rate: Taxa de amostragem do sinal, em hertz.
        time: Instantes correspondentes às amostras, em segundos.
        duration: Duração total do sinal, em segundos.
        channels: Número de canais presentes em ``samples``.
        unit: Unidade ou escala das amplitudes, como ``"normalized"`` ou
            ``"Pa"``.
        path: Caminho do arquivo de origem, se o sinal foi carregado de arquivo.
        filename: Nome do arquivo de origem, se disponível.
        sha256: Identificador SHA-256 do conteúdo de origem, se registrado.
        loaded_at: Instante em que o sinal foi carregado, se registrado.

    Esta classe apenas preserva dados carregados. A leitura do WAV e qualquer
    validação ou transformação das amostras serão responsabilidades futuras.
    """

    samples: tuple[tuple[int | float, ...], ...]
    sample_rate: int
    time: tuple[float, ...]
    duration: float
    channels: int
    unit: str
    path: Path | None = None
    filename: str | None = None
    sha256: str | None = None
    loaded_at: datetime | None = None

    def __post_init__(self) -> None:
        """Valida coerência dimensional sem alterar as amostras fornecidas."""
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if self.channels <= 0:
            raise ValueError("channels must be positive.")
        if len(self.samples) != self.channels:
            raise ValueError("channels must match the number of sample channels.")
        lengths = {len(channel) for channel in self.samples}
        if len(lengths) > 1:
            raise ValueError("all sample channels must have the same length.")
        sample_count = next(iter(lengths), 0)
        if len(self.time) != sample_count:
            raise ValueError("time must have one entry per sample.")
        if self.duration < 0:
            raise ValueError("duration must not be negative.")
        expected_duration = sample_count / self.sample_rate
        if abs(self.duration - expected_duration) > 1e-12:
            raise ValueError("duration must equal sample_count / sample_rate.")
        if any(not isfinite(value) for value in self.time):
            raise ValueError("time values must be finite.")
        if any(later < earlier for earlier, later in zip(self.time, self.time[1:])):
            raise ValueError("time values must be ordered.")


@dataclass(frozen=True, slots=True)
class NoiseMetrics:
    """Contrato legado para métricas estimadas de ruído.

    Args:
        floor_db: Nível estimado do piso de ruído, em decibéis.
        signal_to_noise_ratio_db: Relação sinal-ruído, em decibéis.
        window_start_s: Início da janela usada para a estimativa, em segundos.
        window_end_s: Fim da janela usada para a estimativa, em segundos.

    ``NoiseReport`` é a representação canônica usada por ``TemporalResults``.
    Esta classe permanece disponível para compatibilidade com importações e
    consumidores anteriores.
    """

    floor_db: float
    signal_to_noise_ratio_db: float
    window_start_s: float
    window_end_s: float


@dataclass(frozen=True, slots=True)
class ImpactReport:
    """Resultado canônico da detecção temporal de impacto.

    ``confidence`` é uma pontuação heurística, não uma probabilidade calibrada.
    ``parameters`` registra os limiares efetivamente empregados pelo método.
    """

    impact_time_s: float
    impact_sample: int
    peak_sample: int
    peak_time_s: float
    transient_end_s: float
    transient_duration_s: float
    confidence: float
    method: str
    parameters: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )


@dataclass(frozen=True, slots=True)
class NoiseReport:
    """Resultado canônico de uma estimativa temporal de ruído.

    Args:
        noise_floor_dbfs: RMS da janela de ruído em dBFS.
        rms_noise: RMS normalizado da janela de ruído.
        window_start_s: Início da janela selecionada, em segundos.
        window_end_s: Fim da janela selecionada, em segundos.
        confidence: Pontuação heurística de disponibilidade da estimativa.
        method: Identificador e versão do método empregado.
        parameters: Parâmetros efetivos do método.
        signal_to_noise_ratio_db: SNR global opcional derivada da análise.
    """

    noise_floor_dbfs: float
    rms_noise: float
    window_start_s: float
    window_end_s: float
    confidence: float
    method: str
    parameters: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )
    signal_to_noise_ratio_db: float | None = None


@dataclass(frozen=True, slots=True)
class TemporalMetrics:
    """Métricas globais produzidas exclusivamente no domínio temporal."""

    peak: float
    rms: float
    crest_factor: float | None
    dynamic_range_db: float
    total_energy: float
    method: str = "temporal-global-v1"
    parameters: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )


@dataclass(frozen=True, slots=True)
class Envelope:
    """Série temporal que representa o envelope de amplitude de um sinal.

    Args:
        times_s: Instantes de cada ponto do envelope, em segundos.
        amplitudes: Amplitudes correspondentes, na unidade definida pela análise.
        method: Nome do método que produziu o envelope.
        unit: Unidade ou escala das amplitudes do envelope.
        parameters: Parâmetros declarados pelo método de construção. A interface
            de somente leitura evita o uso de dicionários ad hoc em resultados.

    As sequências são imutáveis para que um resultado armazenado possa ser
    compartilhado e relatado sem alterações acidentais.
    """

    times_s: tuple[float, ...]
    amplitudes: tuple[float, ...]
    method: str = "unspecified"
    unit: str = "unspecified"
    parameters: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        """Valida séries temporais e protege parâmetros contra mutação externa."""
        if len(self.times_s) != len(self.amplitudes):
            raise ValueError("times_s and amplitudes must have the same length.")
        if any(not isfinite(value) for value in self.times_s):
            raise ValueError("times_s values must be finite.")
        if any(not isfinite(value) for value in self.amplitudes):
            raise ValueError("amplitudes values must be finite.")
        if any(
            later < earlier for earlier, later in zip(self.times_s, self.times_s[1:])
        ):
            raise ValueError("times_s values must be ordered.")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


@dataclass(frozen=True, slots=True)
class DecayFit:
    """Parâmetros de um futuro ajuste de decaimento da amplitude.

    Args:
        model_name: Nome do modelo matemático adotado pelo ajuste.
        decay_rate_per_s: Taxa de decaimento estimada, por segundo.
        intercept: Intercepto do modelo na escala de amplitude adotada.
        coefficient_of_determination: Qualidade do ajuste, representada por R².
        fit_start_s: Início do intervalo temporal ajustado, em segundos.
        fit_end_s: Fim do intervalo temporal ajustado, em segundos.
    """

    model_name: str
    decay_rate_per_s: float
    intercept: float
    coefficient_of_determination: float
    fit_start_s: float
    fit_end_s: float


@dataclass(frozen=True, slots=True)
class Spectrum:
    """Série de valores espectrais associada a uma gravação.

    Args:
        frequencies_hz: Frequências de cada bin espectral, em hertz.
        magnitudes: Magnitudes correspondentes na escala indicada.
        magnitude_unit: Unidade ou escala das magnitudes, como ``"dB"``.
        window_name: Nome da janela temporal associada ao espectro, se houver.
        fft_size: Tamanho da FFT associada ao espectro, se houver.
        overlap: Sobreposição entre janelas sucessivas, se aplicável.
        timestamp: Instante temporal, em segundos, associado ao espectro; usado
            quando ele representa uma fatia de uma análise temporal.
        sample_rate_hz: Taxa de amostragem do trecho analisado, em hertz.
        original_size: Número de amostras antes de zero padding.
        bin_spacing_hz: Espaçamento entre bins da FFT, em hertz. Não é uma
            resolução espectral efetiva; zero padding apenas reduz esta grade.
        channel_policy: Política aplicada aos canais de entrada.
        channel_index: Canal selecionado, quando aplicável.
        normalization: Convenção usada para normalizar a magnitude.
        interval_start_s: Início do trecho analisado, em segundos.
        interval_end_s: Fim do trecho analisado, em segundos.
        remove_mean: Indica se a média do trecho foi removida.
        parameters: Parâmetros efetivos e referências da magnitude.

    Os valores podem ser calculados pela FFT estacionária do
    :mod:`belllab.spectrum` ou fornecidos por uma implementação compatível.
    """

    frequencies_hz: tuple[float, ...]
    magnitudes: tuple[float, ...]
    magnitude_unit: str
    window_name: str | None = None
    fft_size: int | None = None
    overlap: float | None = None
    timestamp: float | None = None
    sample_rate_hz: int | None = None
    original_size: int | None = None
    bin_spacing_hz: float | None = None
    channel_policy: str | None = None
    channel_index: int | None = None
    normalization: str | None = None
    interval_start_s: float | None = None
    interval_end_s: float | None = None
    remove_mean: bool | None = None
    parameters: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        """Valida a malha espectral sem modificar magnitudes calculadas."""
        if len(self.frequencies_hz) != len(self.magnitudes):
            raise ValueError("frequencies_hz and magnitudes must have the same length.")
        if any(not isfinite(value) or value < 0 for value in self.frequencies_hz):
            raise ValueError("frequencies_hz must be finite and non-negative.")
        if any(
            later < earlier
            for earlier, later in zip(self.frequencies_hz, self.frequencies_hz[1:])
        ):
            raise ValueError("frequencies_hz must be ordered.")
        if any(value != value for value in self.magnitudes):
            raise ValueError("magnitudes must not contain NaN.")
        if self.overlap is not None and not 0 <= self.overlap < 1:
            raise ValueError("overlap must be a fraction in the interval [0, 1).")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))

    @property
    def frequency_resolution_hz(self) -> float | None:
        """Alias legado para ``bin_spacing_hz``.

        O valor é o espaçamento da grade FFT, não uma medida universal da
        resolução espectral efetiva.
        """
        return self.bin_spacing_hz


@dataclass(frozen=True, slots=True)
class SpectralPeak:
    """Observação matemática de um pico em um espectro estacionário.

    Não representa automaticamente um modo físico. Campos opcionais são
    ``None`` quando o método correspondente não pôde ser aplicado. Frequências
    refinadas e larguras são estimativas operacionais, não incertezas formais.
    """

    bin_index: int
    bin_frequency_hz: float
    bin_amplitude: float
    amplitude_unit: str
    refined_frequency_hz: float | None = None
    refined_amplitude: float | None = None
    prominence: float | None = None
    width: float | None = None
    width_bins: float | None = None
    width_hz: float | None = None
    local_noise_floor: float | None = None
    local_snr_db: float | None = None
    interpolation_method: str | None = None
    width_method: str | None = None
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TimeFrequencySpectrum:
    """STFT unilateral com ``values[frequency_index, time_index]``."""

    times_s: tuple[float, ...]
    frequencies_hz: tuple[float, ...]
    values: tuple[tuple[float, ...], ...]
    magnitude_unit: str
    sample_rate_hz: int
    window_length: int
    fft_size: int
    hop_length: int
    bin_spacing_hz: float
    frame_spacing_s: float
    window_name: str
    coherent_gain: float
    channel_policy: str
    channel_index: int | None
    interval_start_s: float
    interval_end_s: float
    padding_policy: str
    parameters: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True, slots=True)
class FramePeaks:
    """Observações de picos pertencentes a um único quadro STFT.

    Esta classe descreve observações matemáticas independentes. A ausência de
    picos, inclusive em silêncio, é um resultado válido e não um erro.
    """

    frame_index: int
    time_s: float
    peaks: tuple[SpectralPeak, ...]
    candidate_count: int
    accepted_count: int
    frame_maximum: float | None = None
    spectral_floor: float | None = None
    diagnostics: tuple[str, ...] = ()
    parameters: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        """Garante contagens coerentes e uma referência temporal válida."""
        if self.frame_index < 0:
            raise ValueError("frame_index must not be negative.")
        if not isfinite(self.time_s):
            raise ValueError("time_s must be finite.")
        if self.candidate_count < 0 or self.accepted_count < 0:
            raise ValueError("peak counts must not be negative.")
        if self.accepted_count > self.candidate_count:
            raise ValueError("accepted_count must not exceed candidate_count.")
        if self.accepted_count != len(self.peaks):
            raise ValueError("accepted_count must equal the number of peaks.")
        if len({peak.bin_index for peak in self.peaks}) != len(self.peaks):
            raise ValueError("a frame must not contain duplicated peak bins.")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


@dataclass(frozen=True, slots=True)
class SpectralTrack:
    """Trajetória matemática de picos associados ao longo de quadros STFT.

    Não é um modo físico. Frequências, amplitudes e métricas resumidas são
    descritivas e dependem da STFT, dos limiares de picos e da associação.
    """

    track_id: int
    frame_indices: tuple[int, ...]
    times_s: tuple[float, ...]
    bin_frequencies_hz: tuple[float, ...]
    refined_frequencies_hz: tuple[float | None, ...]
    amplitudes: tuple[float, ...]
    amplitude_unit: str
    prominences: tuple[float | None, ...]
    widths_hz: tuple[float | None, ...]
    local_snr_db: tuple[float | None, ...]
    peak_references: tuple[tuple[int, int], ...]
    first_frame: int
    last_frame: int
    duration_s: float
    observation_count: int
    gap_count: int
    total_missing_frames: int
    largest_gap_frames: int
    mean_frequency_hz: float
    median_frequency_hz: float
    frequency_std_hz: float
    initial_frequency_hz: float
    final_frequency_hz: float
    frequency_drift_hz: float
    mean_drift_hz_per_s: float | None
    max_amplitude: float
    initial_amplitude: float
    final_amplitude: float
    median_amplitude: float
    mean_association_cost: float | None = None
    max_association_cost: float | None = None
    diagnostics: tuple[str, ...] = ()
    analysis_final_frame: int | None = None

    def __post_init__(self) -> None:
        """Valida invariantes de uma sequência observada, sem interpolação."""
        sequences = (
            self.frame_indices,
            self.times_s,
            self.bin_frequencies_hz,
            self.refined_frequencies_hz,
            self.amplitudes,
            self.prominences,
            self.widths_hz,
            self.local_snr_db,
            self.peak_references,
        )
        if self.track_id < 0:
            raise ValueError("track_id must not be negative.")
        if not self.frame_indices:
            raise ValueError("a spectral track must contain at least one observation.")
        if any(len(values) != len(self.frame_indices) for values in sequences):
            raise ValueError("all track observation vectors must have equal length.")
        if any(later <= earlier for earlier, later in zip(self.frame_indices, self.frame_indices[1:])):
            raise ValueError("track frame_indices must be strictly increasing.")
        if any(later <= earlier for earlier, later in zip(self.times_s, self.times_s[1:])):
            raise ValueError("track times_s must be strictly increasing.")
        if any(isfinite(value) and value < 0 for value in self.bin_frequencies_hz):
            raise ValueError("finite track bin frequencies must be non-negative.")
        if self.amplitude_unit not in {"linear_amplitude", "dbfs_amplitude"}:
            raise ValueError("track amplitude_unit must be linear_amplitude or dbfs_amplitude.")
        if self.first_frame != self.frame_indices[0] or self.last_frame != self.frame_indices[-1]:
            raise ValueError("track frame bounds must match its observations.")
        if self.observation_count != len(self.frame_indices):
            raise ValueError("observation_count must match the number of observations.")
        gap_lengths = tuple(
            current - previous - 1
            for previous, current in zip(self.frame_indices, self.frame_indices[1:])
        )
        if self.gap_count != sum(length > 0 for length in gap_lengths):
            raise ValueError("gap_count must equal the number of gap intervals.")
        if self.total_missing_frames != sum(gap_lengths):
            raise ValueError("total_missing_frames must equal absent internal frames.")
        expected_largest = max(gap_lengths, default=0)
        if self.largest_gap_frames != expected_largest:
            raise ValueError("largest_gap_frames is inconsistent with frame_indices.")
        expected_duration = self.times_s[-1] - self.times_s[0]
        if abs(self.duration_s - expected_duration) > 1e-12:
            raise ValueError("duration_s must equal the observed time span.")
        if self.analysis_final_frame is not None and self.analysis_final_frame < self.last_frame:
            raise ValueError("analysis_final_frame must not precede the track.")


@dataclass(frozen=True, slots=True)
class TrackFrequencyFit:
    """Regressão linear descritiva da frequência rastreada; não é ajuste modal."""

    success: bool
    method: str | None
    slope_hz_per_s: float | None
    intercept_hz: float | None
    r_squared: float | None
    rmse_hz: float | None
    available_point_count: int
    finite_point_count: int
    used_point_count: int
    discarded_point_count: int
    start_time_s: float | None
    end_time_s: float | None
    failure_reason: str | None = None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Valida sucesso, falha, contagens e campos numéricos auditáveis."""
        counts = (
            self.available_point_count,
            self.finite_point_count,
            self.used_point_count,
            self.discarded_point_count,
        )
        if min(counts) < 0:
            raise ValueError("TrackFrequencyFit counts must not be negative.")
        if self.used_point_count > self.finite_point_count or self.finite_point_count > self.available_point_count:
            raise ValueError("TrackFrequencyFit point counts are inconsistent.")
        if self.discarded_point_count != self.available_point_count - self.finite_point_count:
            raise ValueError("TrackFrequencyFit discarded count is inconsistent.")
        if not isinstance(self.diagnostics, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.diagnostics
        ):
            raise ValueError("TrackFrequencyFit diagnostics must be nonempty strings.")
        if len(set(self.diagnostics)) != len(self.diagnostics):
            raise ValueError("TrackFrequencyFit diagnostics must not contain duplicates.")
        values = (
            self.slope_hz_per_s,
            self.intercept_hz,
            self.r_squared,
            self.rmse_hz,
            self.start_time_s,
            self.end_time_s,
        )
        if any(value is not None and not isfinite(value) for value in values):
            raise ValueError("TrackFrequencyFit optional values must be finite.")
        if self.rmse_hz is not None and self.rmse_hz < 0:
            raise ValueError("TrackFrequencyFit rmse_hz must not be negative.")
        if self.start_time_s is not None and self.end_time_s is not None and self.start_time_s > self.end_time_s:
            raise ValueError("TrackFrequencyFit time interval is inverted.")
        regression = (
            self.method,
            self.slope_hz_per_s,
            self.intercept_hz,
            self.rmse_hz,
            self.start_time_s,
            self.end_time_s,
        )
        if self.success:
            if self.failure_reason is not None:
                raise ValueError("successful TrackFrequencyFit must not have failure_reason.")
            if self.method != "linear_frequency_drift" or any(value is None for value in regression):
                raise ValueError("successful TrackFrequencyFit lacks required regression fields.")
            if self.used_point_count < 2:
                raise ValueError("successful TrackFrequencyFit requires at least two points.")
            if self.start_time_s == self.end_time_s:
                raise ValueError("successful TrackFrequencyFit requires distinct times.")
        else:
            if not self.failure_reason:
                raise ValueError("failed TrackFrequencyFit requires failure_reason.")
            if any(value is not None for value in (*regression, self.r_squared)):
                raise ValueError("failed TrackFrequencyFit must not contain regression fields.")


@dataclass(frozen=True, slots=True)
class SpectralTrackCharacterization:
    """Caracterização matemática operacional de uma trajetória espectral.

    Não estabelece a existência de um modo físico. ``decay_tau_s`` descreve
    somente um ajuste de amplitudes rastreadas quando seus pressupostos valem.
    """

    track_id: int
    frequency_source: str
    frequency_initial_hz: float | None
    frequency_final_hz: float | None
    frequency_mean_hz: float | None
    frequency_median_hz: float | None
    frequency_min_hz: float | None
    frequency_max_hz: float | None
    frequency_std_hz: float | None
    frequency_total_drift_hz: float | None
    frequency_peak_to_peak_hz: float | None
    relative_frequency_stability: float | None
    frequency_available_point_count: int
    frequency_finite_point_count: int
    frequency_discarded_point_count: int
    frequency_fit: TrackFrequencyFit
    amplitude_initial: float | None
    amplitude_final: float | None
    amplitude_mean: float | None
    amplitude_median: float | None
    amplitude_min: float | None
    amplitude_max: float | None
    amplitude_std: float | None
    amplitude_peak_to_peak: float | None
    amplitude_slope_per_s: float | None
    amplitude_increase_fraction: float | None
    amplitude_decrease_fraction: float | None
    amplitude_constant_fraction: float | None
    amplitude_available_point_count: int
    amplitude_finite_point_count: int
    amplitude_discarded_point_count: int
    amplitude_unit: str
    amplitude_fit: TrackAmplitudeFit
    first_frame: int
    last_frame: int
    start_time_s: float
    end_time_s: float
    observed_duration_s: float
    observation_count: int
    frame_span_count: int
    coverage_fraction: float
    gap_count: int
    total_missing_frames: int
    largest_gap_frames: int
    reached_analysis_final_frame: bool | None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Rejeita métricas operacionais contraditórias ou não físicas."""
        if self.track_id < 0:
            raise ValueError("track_id must not be negative.")
        if self.frequency_source not in {"interpolated", "bin", "mixed"}:
            raise ValueError("frequency_source must be interpolated, bin, or mixed.")
        if self.amplitude_unit not in {"linear_amplitude", "dbfs_amplitude"}:
            raise ValueError("characterization amplitude_unit is not recognized.")
        if not isinstance(self.diagnostics, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.diagnostics
        ) or len(set(self.diagnostics)) != len(self.diagnostics):
            raise ValueError("characterization diagnostics must be unique nonempty strings.")
        frequency_counts = (
            self.frequency_available_point_count,
            self.frequency_finite_point_count,
            self.frequency_discarded_point_count,
        )
        amplitude_counts = (
            self.amplitude_available_point_count,
            self.amplitude_finite_point_count,
            self.amplitude_discarded_point_count,
        )
        if min(*frequency_counts, *amplitude_counts) < 0:
            raise ValueError("characterization point counts must not be negative.")
        if self.frequency_finite_point_count + self.frequency_discarded_point_count != self.frequency_available_point_count:
            raise ValueError("frequency point counts are inconsistent.")
        if self.amplitude_finite_point_count + self.amplitude_discarded_point_count != self.amplitude_available_point_count:
            raise ValueError("amplitude point counts are inconsistent.")
        frequency_values = (
            self.frequency_initial_hz, self.frequency_final_hz,
            self.frequency_mean_hz, self.frequency_median_hz,
            self.frequency_min_hz, self.frequency_max_hz,
            self.frequency_std_hz, self.frequency_total_drift_hz,
            self.frequency_peak_to_peak_hz, self.relative_frequency_stability,
        )
        amplitude_values = (
            self.amplitude_initial, self.amplitude_final, self.amplitude_mean,
            self.amplitude_median, self.amplitude_min, self.amplitude_max,
            self.amplitude_std, self.amplitude_peak_to_peak,
            self.amplitude_slope_per_s, self.amplitude_increase_fraction,
            self.amplitude_decrease_fraction, self.amplitude_constant_fraction,
        )
        temporal_values = (
            self.start_time_s, self.end_time_s, self.observed_duration_s,
            self.coverage_fraction,
        )
        if any(value is not None and not isfinite(value) for value in (*frequency_values, *amplitude_values, *temporal_values)):
            raise ValueError("characterization numeric values must be finite when present.")
        if self.frequency_finite_point_count == 0:
            if any(value is not None for value in frequency_values):
                raise ValueError("frequency metrics require finite frequency points.")
        else:
            required_frequency = frequency_values[:9]
            if any(value is None for value in required_frequency):
                raise ValueError("finite frequency data require descriptive metrics.")
            if self.frequency_min_hz < 0 or self.frequency_max_hz < self.frequency_min_hz:
                raise ValueError("frequency bounds are inconsistent.")
            if not self.frequency_min_hz <= self.frequency_mean_hz <= self.frequency_max_hz:
                raise ValueError("frequency mean must lie within bounds.")
            if not self.frequency_min_hz <= self.frequency_median_hz <= self.frequency_max_hz:
                raise ValueError("frequency median must lie within bounds.")
            if self.frequency_std_hz < 0 or self.frequency_peak_to_peak_hz < 0:
                raise ValueError("frequency dispersion must not be negative.")
            if not isclose(self.frequency_peak_to_peak_hz, self.frequency_max_hz - self.frequency_min_hz, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("frequency peak-to-peak is inconsistent.")
            if not isclose(self.frequency_total_drift_hz, self.frequency_final_hz - self.frequency_initial_hz, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("frequency drift is inconsistent.")
        if self.relative_frequency_stability is not None and self.relative_frequency_stability < 0:
            raise ValueError("relative_frequency_stability must not be negative.")
        if self.frequency_fit.available_point_count != self.frequency_available_point_count or self.frequency_fit.finite_point_count != self.frequency_finite_point_count or self.frequency_fit.discarded_point_count != self.frequency_discarded_point_count:
            raise ValueError("frequency_fit counts must match characterization.")
        if self.amplitude_finite_point_count == 0:
            if any(value is not None for value in amplitude_values):
                raise ValueError("amplitude metrics require finite amplitude points.")
        else:
            required_amplitude = amplitude_values[:8]
            if any(value is None for value in required_amplitude):
                raise ValueError("finite amplitude data require descriptive metrics.")
            if self.amplitude_max < self.amplitude_min:
                raise ValueError("amplitude bounds are inconsistent.")
            if not self.amplitude_min <= self.amplitude_mean <= self.amplitude_max:
                raise ValueError("amplitude mean must lie within bounds.")
            if not self.amplitude_min <= self.amplitude_median <= self.amplitude_max:
                raise ValueError("amplitude median must lie within bounds.")
            if self.amplitude_std < 0 or self.amplitude_peak_to_peak < 0:
                raise ValueError("amplitude dispersion must not be negative.")
            if not isclose(self.amplitude_peak_to_peak, self.amplitude_max - self.amplitude_min, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("amplitude peak-to-peak is inconsistent.")
        fractions = (
            self.amplitude_increase_fraction,
            self.amplitude_decrease_fraction,
            self.amplitude_constant_fraction,
        )
        if any(value is not None and not 0 <= value <= 1 for value in fractions):
            raise ValueError("amplitude variation fractions must be in [0, 1].")
        if any(value is None for value in fractions) != all(value is None for value in fractions):
            raise ValueError("amplitude variation fractions must be all present or all absent.")
        if all(value is not None for value in fractions) and not isclose(sum(fractions), 1.0, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("amplitude variation fractions must sum to one.")
        if self.amplitude_fit.amplitude_unit != self.amplitude_unit:
            raise ValueError("amplitude_fit unit must match characterization.")
        if self.amplitude_fit.available_point_count != self.amplitude_available_point_count or self.amplitude_fit.finite_point_count != self.amplitude_finite_point_count:
            raise ValueError("amplitude_fit counts must match characterization.")
        if self.first_frame < 0 or self.last_frame < self.first_frame:
            raise ValueError("characterization frame interval is invalid.")
        if self.end_time_s < self.start_time_s or self.observed_duration_s < 0:
            raise ValueError("characterization time interval or duration is invalid.")
        if not isclose(self.observed_duration_s, self.end_time_s - self.start_time_s, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("observed_duration_s must equal the observed time span.")
        if self.observation_count < 1 or self.frame_span_count != self.last_frame - self.first_frame + 1:
            raise ValueError("characterization observation or frame-span count is invalid.")
        if not 0 <= self.coverage_fraction <= 1:
            raise ValueError("coverage_fraction must be in [0, 1].")
        if not isclose(self.coverage_fraction, self.observation_count / self.frame_span_count, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("coverage_fraction is inconsistent with frame counts.")
        if min(self.gap_count, self.total_missing_frames, self.largest_gap_frames) < 0:
            raise ValueError("gap metrics must not be negative.")
        if self.total_missing_frames != self.frame_span_count - self.observation_count:
            raise ValueError("total_missing_frames is inconsistent with coverage.")
        if self.largest_gap_frames > self.total_missing_frames or self.gap_count > self.total_missing_frames:
            raise ValueError("gap metrics are inconsistent.")
        if self.reached_analysis_final_frame is not None and not isinstance(self.reached_analysis_final_frame, bool):
            raise ValueError("reached_analysis_final_frame must be boolean or None.")

    @property
    def frequency_series(self) -> str:
        """Alias legado da política canônica interpolada-com-fallback."""
        return "refined_or_bin"

    @property
    def frequency_slope_hz_per_s(self) -> float | None:
        """Alias legado para ``frequency_fit.slope_hz_per_s``."""
        return self.frequency_fit.slope_hz_per_s

    @property
    def frequency_fit_rmse_hz(self) -> float | None:
        """Alias legado para ``frequency_fit.rmse_hz``."""
        return self.frequency_fit.rmse_hz

    @property
    def observed_frame_fraction(self) -> float:
        """Alias legado para ``coverage_fraction``."""
        return self.coverage_fraction

    @property
    def decay_method(self) -> str | None:
        """Alias legado para ``amplitude_fit.method``."""
        aliases = {
            "log_linear_amplitude_decay": "log_linear_amplitude",
            "linear_dbfs_decay": "linear_dbfs_amplitude",
        }
        return aliases.get(self.amplitude_fit.method, self.amplitude_fit.method)

    @property
    def decay_tau_s(self) -> float | None:
        """Alias legado para ``amplitude_fit.tau_s``."""
        return self.amplitude_fit.tau_s

    @property
    def decay_slope(self) -> float | None:
        """Alias legado para ``amplitude_fit.slope``."""
        return self.amplitude_fit.slope

    @property
    def decay_r_squared(self) -> float | None:
        """Alias legado para ``amplitude_fit.r_squared``."""
        return self.amplitude_fit.r_squared

    @property
    def decay_points_used(self) -> int:
        """Alias legado para ``amplitude_fit.used_point_count``."""
        return self.amplitude_fit.used_point_count

    @property
    def decay_points_discarded(self) -> int:
        """Alias legado para ``amplitude_fit.discarded_point_count``."""
        return self.amplitude_fit.discarded_point_count


@dataclass(frozen=True, slots=True)
class TrackAssignmentDiagnostic:
    """Diagnóstico operacional público de uma associação aceita."""

    frame_index: int
    track_id: int
    peak_index: int
    selected_cost: float
    row_assignment_margin: float | None
    column_assignment_margin: float | None
    assignment_margin: float | None
    ambiguous: bool
    near_threshold: bool
    frequency_distance: float
    frequency_distance_unit: str
    amplitude_distance: float | None
    frequency_cost_component: float
    amplitude_cost_component: float
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Valida diagnóstico auditável sem usar NaN ou infinitos públicos."""
        if self.frame_index < 0 or self.track_id < 0 or self.peak_index < 0:
            raise ValueError("assignment diagnostic indices must not be negative.")
        values = (self.selected_cost, self.frequency_distance,
                  self.frequency_cost_component, self.amplitude_cost_component)
        if any(not isfinite(value) for value in values):
            raise ValueError("assignment diagnostic costs must be finite.")
        if any(value < 0 for value in values):
            raise ValueError("assignment diagnostic costs must not be negative.")
        if self.frequency_distance_unit not in {"hz", "relative", "cents"}:
            raise ValueError("assignment diagnostic frequency_distance_unit is unknown.")
        for margin in (self.row_assignment_margin, self.column_assignment_margin, self.assignment_margin):
            if margin is not None and (not isfinite(margin) or margin < 0):
                raise ValueError("assignment margins must be finite and non-negative.")
        available = [value for value in (self.row_assignment_margin, self.column_assignment_margin) if value is not None]
        expected = min(available) if available else None
        if self.assignment_margin != expected:
            raise ValueError("assignment_margin must equal the minimum available margin.")
        if not isclose(self.selected_cost, self.frequency_cost_component + self.amplitude_cost_component, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("selected_cost must equal the sum of cost components.")


@dataclass(frozen=True, slots=True)
class TrackAmplitudeFit:
    """Ajuste operacional de amplitude rastreada; não é amortecimento modal."""

    success: bool
    decay_detected: bool
    method: str | None
    amplitude_unit: str
    fit_domain: str | None
    slope: float | None
    intercept: float | None
    tau_s: float | None
    r_squared: float | None
    rmse: float | None
    rmse_unit: str | None
    available_point_count: int
    finite_point_count: int
    used_point_count: int
    discarded_point_count: int
    start_time_s: float | None
    end_time_s: float | None
    failure_reason: str | None = None
    diagnostics: tuple[str, ...] = ()
    nonfinite_discarded_point_count: int = 0
    nonpositive_discarded_point_count: int = 0
    slope_unit: str | None = None

    def __post_init__(self) -> None:
        """Validate successful fits fully; failed fits keep only audit metadata."""
        if self.amplitude_unit not in {"linear_amplitude", "dbfs_amplitude"}:
            raise ValueError("TrackAmplitudeFit amplitude_unit is not recognized.")
        if min(self.available_point_count, self.finite_point_count, self.used_point_count, self.discarded_point_count, self.nonfinite_discarded_point_count, self.nonpositive_discarded_point_count) < 0:
            raise ValueError("TrackAmplitudeFit counts must not be negative.")
        if self.used_point_count > self.finite_point_count or self.finite_point_count > self.available_point_count:
            raise ValueError("TrackAmplitudeFit point counts are inconsistent.")
        if self.nonfinite_discarded_point_count != self.available_point_count - self.finite_point_count:
            raise ValueError("TrackAmplitudeFit discarded count is inconsistent.")
        if self.discarded_point_count != self.nonfinite_discarded_point_count + self.nonpositive_discarded_point_count:
            raise ValueError("TrackAmplitudeFit total discarded count is inconsistent.")
        if self.used_point_count + self.nonpositive_discarded_point_count != self.finite_point_count:
            raise ValueError("TrackAmplitudeFit positive-point count is inconsistent.")
        if not isinstance(self.diagnostics, tuple) or any(
            not isinstance(item, str) or not item.strip()
            for item in self.diagnostics
        ):
            raise ValueError("TrackAmplitudeFit diagnostics must be nonempty strings.")
        if len(set(self.diagnostics)) != len(self.diagnostics):
            raise ValueError("TrackAmplitudeFit diagnostics must not contain duplicates.")
        if self.success and self.failure_reason is not None:
            raise ValueError("successful TrackAmplitudeFit must not have failure_reason.")
        expected = {
            "linear_amplitude": ("log_linear_amplitude_decay", "natural_log_amplitude", "1/s", "ln(amplitude)"),
            "dbfs_amplitude": ("linear_dbfs_decay", "dbfs", "dB/s", "dB"),
        }
        if self.decay_detected != (self.tau_s is not None):
            raise ValueError("decay_detected must agree with tau_s.")
        if self.tau_s is not None and (not isfinite(self.tau_s) or self.tau_s <= 0):
            raise ValueError("TrackAmplitudeFit tau_s must be finite and positive.")
        if self.decay_detected and (not self.success or self.slope is None or self.slope >= 0):
            raise ValueError("detected decay requires successful negative slope.")
        if self.success:
            method, domain, slope_unit, rmse_unit = expected[self.amplitude_unit]
            if (self.method, self.fit_domain, self.slope_unit, self.rmse_unit) != (method, domain, slope_unit, rmse_unit):
                raise ValueError("TrackAmplitudeFit method, domain, or units are incompatible.")
        if self.success and (self.method is None or self.fit_domain is None or self.slope is None or self.intercept is None or self.rmse is None or self.start_time_s is None or self.end_time_s is None or self.used_point_count < 2):
            raise ValueError("successful TrackAmplitudeFit lacks required regression fields.")
        if not self.success and not self.failure_reason:
            raise ValueError("failed TrackAmplitudeFit requires failure_reason.")
        if not self.success and any(value is not None for value in (
            self.method, self.fit_domain, self.slope, self.intercept, self.rmse,
            self.r_squared, self.rmse_unit, self.start_time_s, self.end_time_s, self.tau_s,
            self.slope_unit,
        )):
            raise ValueError("failed TrackAmplitudeFit must not contain regression fields.")
        for value in (self.slope, self.intercept, self.r_squared, self.rmse, self.start_time_s, self.end_time_s):
            if value is not None and not isfinite(value):
                raise ValueError("TrackAmplitudeFit optional values must be finite.")
        if self.rmse is not None and self.rmse < 0:
            raise ValueError("TrackAmplitudeFit rmse must not be negative.")
        if self.start_time_s is not None and self.end_time_s is not None and self.start_time_s > self.end_time_s:
            raise ValueError("TrackAmplitudeFit time interval is inverted.")


@dataclass(frozen=True, slots=True)
class PreImpactEvidence:
    """Evidência operacional de alteração espectral causada pelo impacto."""

    source_track_id: int
    impact_time_s: float
    amplitude_unit: str
    preimpact_available_point_count: int
    preimpact_finite_point_count: int
    postimpact_available_point_count: int
    postimpact_finite_point_count: int
    preimpact_coverage_fraction: float
    preimpact_detected: bool
    preimpact_level: float | None
    preimpact_median_level: float | None
    preimpact_variability: float | None
    preimpact_slope_per_s: float | None
    postimpact_initial_level: float | None
    postimpact_level: float | None
    postimpact_median_level: float | None
    postimpact_variability: float | None
    postimpact_slope_per_s: float | None
    postimpact_decay_detected: bool
    impact_level_change: float | None
    impact_level_change_db: float | None
    post_to_pre_ratio: float | None
    impact_excited: bool
    background_contaminated: bool
    preimpact_decay_detected: bool
    classification: str
    success: bool
    failure_reason: str | None = None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Valida contagens, métricas, classificação e estados lógicos."""
        classifications = {
            "not_detected_preimpact",
            "impact_emergent",
            "impact_amplified",
            "persistent_background_tone",
            "preexisting_decay",
            "reexcited_preexisting_component",
            "insufficient_preimpact_data",
            "insufficient_postimpact_data",
            "indeterminate",
        }
        if self.source_track_id < 0:
            raise ValueError("PreImpactEvidence source_track_id must not be negative.")
        if not isfinite(self.impact_time_s):
            raise ValueError("PreImpactEvidence impact_time_s must be finite.")
        if self.amplitude_unit not in {"linear_amplitude", "dbfs_amplitude"}:
            raise ValueError("PreImpactEvidence amplitude_unit is not recognized.")
        counts = (
            self.preimpact_available_point_count,
            self.preimpact_finite_point_count,
            self.postimpact_available_point_count,
            self.postimpact_finite_point_count,
        )
        if min(counts) < 0:
            raise ValueError("PreImpactEvidence counts must not be negative.")
        if (
            self.preimpact_finite_point_count > self.preimpact_available_point_count
            or self.postimpact_finite_point_count > self.postimpact_available_point_count
        ):
            raise ValueError("PreImpactEvidence finite counts exceed available counts.")
        if not isfinite(self.preimpact_coverage_fraction) or not 0 <= self.preimpact_coverage_fraction <= 1:
            raise ValueError("preimpact_coverage_fraction must be finite and in [0, 1].")
        numeric = (
            self.preimpact_level,
            self.preimpact_median_level,
            self.preimpact_variability,
            self.preimpact_slope_per_s,
            self.postimpact_initial_level,
            self.postimpact_level,
            self.postimpact_median_level,
            self.postimpact_variability,
            self.postimpact_slope_per_s,
            self.impact_level_change,
            self.impact_level_change_db,
            self.post_to_pre_ratio,
        )
        if any(value is not None and not isfinite(value) for value in numeric):
            raise ValueError("PreImpactEvidence optional metrics must be finite.")
        if self.preimpact_variability is not None and self.preimpact_variability < 0:
            raise ValueError("preimpact_variability must not be negative.")
        if self.postimpact_variability is not None and self.postimpact_variability < 0:
            raise ValueError("postimpact_variability must not be negative.")
        if self.post_to_pre_ratio is not None and self.post_to_pre_ratio <= 0:
            raise ValueError("post_to_pre_ratio must be positive.")
        if self.classification not in classifications:
            raise ValueError("PreImpactEvidence classification is unknown.")
        if self.success and self.failure_reason is not None:
            raise ValueError("successful PreImpactEvidence must not have failure_reason.")
        if not self.success and not self.failure_reason:
            raise ValueError("failed PreImpactEvidence requires failure_reason.")
        insufficient = {
            "insufficient_preimpact_data",
            "insufficient_postimpact_data",
        }
        if self.success and self.classification in insufficient:
            raise ValueError("successful PreImpactEvidence cannot be classified as insufficient.")
        if not self.success and (
            self.classification not in insufficient
            or self.failure_reason != self.classification
        ):
            raise ValueError("failed PreImpactEvidence requires a matching insufficiency classification.")
        if self.preimpact_detected and self.preimpact_median_level is None:
            raise ValueError("preimpact_detected requires a finite median level.")
        if self.background_contaminated and not self.preimpact_detected:
            raise ValueError("background contamination requires preimpact detection.")
        if self.preimpact_decay_detected and (
            self.preimpact_slope_per_s is None or self.preimpact_slope_per_s >= 0
        ):
            raise ValueError("preimpact decay requires a negative slope.")
        if self.postimpact_decay_detected and (
            self.postimpact_slope_per_s is None or self.postimpact_slope_per_s >= 0
        ):
            raise ValueError("postimpact decay requires a negative slope.")
        if self.impact_excited:
            if not self.success:
                raise ValueError("failed PreImpactEvidence cannot be impact-excited.")
            if self.classification not in {
                "impact_emergent",
                "impact_amplified",
                "reexcited_preexisting_component",
            }:
                raise ValueError("impact_excited requires an excitation classification.")
            if self.preimpact_detected and (
                self.impact_level_change_db is None
                or self.impact_level_change_db <= 0
            ):
                raise ValueError("excited preexisting line requires positive level change.")
        if not isinstance(self.diagnostics, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.diagnostics
        ) or len(self.diagnostics) != len(set(self.diagnostics)):
            raise ValueError("PreImpactEvidence diagnostics must be unique nonempty strings.")


@dataclass(frozen=True, slots=True)
class CandidateCriterionResult:
    """Resultado auditável de um critério operacional de candidatura modal."""

    criterion: str
    observed: float | int | bool | str | None
    operator: str
    threshold: float | int | bool | str | None
    enabled: bool
    applicable: bool
    passed: bool | None
    reason: str

    def __post_init__(self) -> None:
        """Valida estado habilitado, aplicabilidade e valores documentais."""
        if not self.criterion.strip() or not self.operator.strip() or not self.reason.strip():
            raise ValueError("criterion name, operator, and reason must be nonempty.")
        for value in (self.observed, self.threshold):
            if isinstance(value, float) and not isfinite(value):
                raise ValueError("criterion numeric values must be finite.")
        if not self.enabled and (self.applicable or self.passed is not None):
            raise ValueError("disabled criterion must be non-applicable with passed=None.")
        if self.enabled and self.applicable and self.passed is None:
            raise ValueError("applicable enabled criterion requires a boolean result.")
        if self.enabled and not self.applicable and self.passed is not None:
            raise ValueError("non-applicable criterion must have passed=None.")


@dataclass(frozen=True, slots=True)
class ModalCandidate:
    """Trajetória promovida por critérios operacionais; não é um modo físico."""

    candidate_id: int
    source_track_id: int
    characterization: SpectralTrackCharacterization
    representative_frequency_hz: float | None
    accepted_assignment_count: int
    ambiguous_assignment_count: int
    near_threshold_assignment_count: int
    ambiguous_assignment_fraction: float | None
    near_threshold_assignment_fraction: float | None
    minimum_assignment_margin: float | None
    accepted: bool
    criteria_results: tuple[CandidateCriterionResult, ...]
    acceptance_reasons: tuple[str, ...]
    rejection_reasons: tuple[str, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Impõe coerência entre origem, critérios, razões e resumos públicos."""
        if self.candidate_id < 0 or self.source_track_id < 0:
            raise ValueError("candidate and source track IDs must not be negative.")
        if self.source_track_id != self.characterization.track_id:
            raise ValueError("source_track_id must match characterization.track_id.")
        if self.representative_frequency_hz is not None and (
            not isfinite(self.representative_frequency_hz)
            or self.representative_frequency_hz <= 0
        ):
            raise ValueError("representative_frequency_hz must be finite and positive.")
        if self.accepted and self.representative_frequency_hz is None:
            raise ValueError("accepted candidate requires representative frequency.")
        counts = (
            self.accepted_assignment_count,
            self.ambiguous_assignment_count,
            self.near_threshold_assignment_count,
        )
        if min(counts) < 0:
            raise ValueError("candidate assignment counts must not be negative.")
        if (
            self.ambiguous_assignment_count > self.accepted_assignment_count
            or self.near_threshold_assignment_count > self.accepted_assignment_count
        ):
            raise ValueError("candidate assignment counts exceed accepted assignments.")
        expected_ambiguous = (
            self.ambiguous_assignment_count / self.accepted_assignment_count
            if self.accepted_assignment_count else None
        )
        expected_near = (
            self.near_threshold_assignment_count / self.accepted_assignment_count
            if self.accepted_assignment_count else None
        )
        for actual, expected in (
            (self.ambiguous_assignment_fraction, expected_ambiguous),
            (self.near_threshold_assignment_fraction, expected_near),
        ):
            if actual is not None and (not isfinite(actual) or not 0 <= actual <= 1):
                raise ValueError("candidate assignment fractions must be finite and in [0, 1].")
            if actual != expected:
                raise ValueError("candidate assignment fraction is inconsistent with counts.")
        if self.minimum_assignment_margin is not None and (
            not isfinite(self.minimum_assignment_margin)
            or self.minimum_assignment_margin < 0
        ):
            raise ValueError("minimum_assignment_margin must be finite and non-negative.")
        names = tuple(item.criterion for item in self.criteria_results)
        if len(names) != len(set(names)):
            raise ValueError("candidate criteria must not contain duplicate names.")
        for collection in (
            self.acceptance_reasons,
            self.rejection_reasons,
            self.diagnostics,
        ):
            if not isinstance(collection, tuple) or any(
                not isinstance(item, str) or not item.strip() for item in collection
            ) or len(collection) != len(set(collection)):
                raise ValueError("candidate textual collections must contain unique nonempty strings.")
        failed = tuple(
            item for item in self.criteria_results
            if item.enabled and item.applicable and item.passed is False
        )
        passed = tuple(
            item for item in self.criteria_results
            if item.enabled and item.applicable and item.passed is True
        )
        expected_acceptance_reasons = tuple(item.reason for item in passed)
        expected_rejection_reasons = tuple(item.reason for item in failed)
        if self.acceptance_reasons != expected_acceptance_reasons:
            raise ValueError(
                "acceptance_reasons must match passed criteria in criterion order."
            )
        if self.rejection_reasons != expected_rejection_reasons:
            raise ValueError(
                "rejection_reasons must match failed criteria in criterion order."
            )
        if self.accepted:
            if failed or self.rejection_reasons:
                raise ValueError("accepted candidate cannot have failed criteria or rejection reasons.")
            if not self.acceptance_reasons:
                raise ValueError("accepted candidate requires acceptance reasons.")
            if any(
                item == "candidate_rejected"
                or item.startswith(("rejection:", "structural_rejection:"))
                for item in self.diagnostics
            ):
                raise ValueError(
                    "accepted candidate cannot contain rejection diagnostics."
                )
        else:
            if not failed:
                raise ValueError(
                    "rejected candidate requires at least one failed enabled applicable criterion."
                )
            if not self.rejection_reasons:
                raise ValueError("rejected candidate requires rejection reasons.")

    @property
    def frequency_source(self) -> str:
        return self.characterization.frequency_source

    @property
    def amplitude_unit(self) -> str:
        return self.characterization.amplitude_unit

    @property
    def observation_count(self) -> int:
        return self.characterization.observation_count

    @property
    def coverage_fraction(self) -> float:
        return self.characterization.coverage_fraction

    @property
    def duration_s(self) -> float:
        return self.characterization.observed_duration_s

    @property
    def frequency_stability(self) -> float | None:
        return self.characterization.relative_frequency_stability

    @property
    def frequency_drift_hz(self) -> float | None:
        return self.characterization.frequency_total_drift_hz

    @property
    def frequency_fit_rmse_hz(self) -> float | None:
        return self.characterization.frequency_fit.rmse_hz

    @property
    def amplitude_decay_detected(self) -> bool:
        return self.characterization.amplitude_fit.decay_detected

    @property
    def amplitude_tau_s(self) -> float | None:
        return self.characterization.amplitude_fit.tau_s

    @property
    def amplitude_fit_r_squared(self) -> float | None:
        return self.characterization.amplitude_fit.r_squared


@dataclass(frozen=True, slots=True)
class ModalMode:
    """Representa um modo acústico identificado em uma gravação de idiofone.

    Args:
        name: Rótulo descritivo ou classificação do modo.
        frequency_hz: Frequência central do modo, em hertz.
        amplitude: Amplitude estimada na escala de análise.
        damping_ratio: Razão de amortecimento estimada, se disponível.
        quality_factor: Fator de qualidade estimado, se disponível.

    Campos opcionais permanecem ``None`` quando o método futuro não puder ou
    não dever estimá-los.
    """

    name: str
    frequency_hz: float
    amplitude: float | None = None
    damping_ratio: float | None = None
    quality_factor: float | None = None
