"""Configurações centralizadas e imutáveis do BellLab.

As opções de análise serão acrescentadas aqui conforme os módulos científicos
forem implementados, mantendo configurações explícitas e reprodutíveis.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Literal


_CHANNEL_POLICIES = frozenset({"select", "mean"})
_WINDOW_NAMES = frozenset({"rectangular", "hann"})
_SPECTRUM_SCALES = frozenset({"linear_amplitude", "dbfs"})
_SPECTRUM_NORMALIZATIONS = frozenset({"coherent_gain_amplitude"})
_PEAK_INTERPOLATION_METHODS = frozenset({"parabolic_log_magnitude"})
_PEAK_NOISE_METHODS = frozenset({"local_median"})
_PEAK_SORT_ORDERS = frozenset({"frequency", "amplitude", "prominence"})
_FRAME_SILENCE_POLICIES = frozenset({"skip", "process"})
_TRACKING_DISTANCE_UNITS = frozenset({"hz", "relative", "cents"})
_TRACKING_METHODS = frozenset({"hungarian"})


@dataclass(frozen=True, slots=True)
class BellLabConfig:
    """Agrupa caminhos-padrão usados pelo projeto.

    Args:
        data_directory: Diretório previsto para gravações e dados de entrada.
        output_directory: Diretório previsto para relatórios e resultados.
    """

    data_directory: Path = Path("data")
    output_directory: Path = Path("outputs")


@dataclass(frozen=True, slots=True)
class TemporalAnalysisSettings:
    """Configura os métodos temporais atualmente implementados.

    Args:
        noise_window_duration_s: Duração da janela candidata de ruído, em
            segundos. O padrão de 50 ms preserva o comportamento anterior.
        envelope_method: Método de envelope usado por ``analyze_temporal``.
        envelope_window_size_samples: Janela usada somente por envelopes móveis.
            Deve ser fornecida quando ``envelope_method`` for ``"moving_rms"``
            ou ``"moving_peak"``.
    """

    noise_window_duration_s: float = 0.050
    envelope_method: Literal[
        "hilbert", "moving_rms", "moving_peak", "cumulative_energy"
    ] = "hilbert"
    envelope_window_size_samples: int | None = None

    def __post_init__(self) -> None:
        """Valida somente os parâmetros efetivamente usados no módulo temporal."""
        if self.noise_window_duration_s <= 0:
            raise ValueError("noise_window_duration_s must be positive.")
        needs_window = self.envelope_method in {"moving_rms", "moving_peak"}
        if needs_window and (
            self.envelope_window_size_samples is None
            or self.envelope_window_size_samples <= 0
        ):
            raise ValueError(
                "envelope_window_size_samples must be positive for moving envelopes."
            )


@dataclass(frozen=True, slots=True)
class SpectrumAnalysisSettings:
    """Configura uma FFT real unilateral estacionária.

    Args:
        channel_policy: ``"select"`` escolhe um canal por índice; ``"mean"``
            combina canais por média apenas quando solicitado explicitamente.
        channel_index: Índice do canal quando ``channel_policy`` é ``"select"``.
        start_time_s: Início opcional, em segundos, do intervalo semiaberto.
        end_time_s: Fim opcional, em segundos, do intervalo semiaberto.
        remove_mean: Remove a média do trecho antes da janela e FFT.
        window_name: Janela temporal aplicada ao trecho.
        n_fft: Tamanho da FFT. ``None`` usa o tamanho do trecho; valores maiores
            aplicam zero padding e valores menores não são aceitos.
        scale: Escala de saída: amplitude linear ou amplitude em dBFS.
        normalization: Convenção de normalização atualmente suportada.
    """

    channel_policy: Literal["select", "mean"] = "select"
    channel_index: int = 0
    start_time_s: float | None = None
    end_time_s: float | None = None
    remove_mean: bool = True
    window_name: Literal["rectangular", "hann"] = "hann"
    n_fft: int | None = None
    scale: Literal["linear_amplitude", "dbfs"] = "linear_amplitude"
    normalization: Literal["coherent_gain_amplitude"] = "coherent_gain_amplitude"

    def __post_init__(self) -> None:
        """Valida o contrato da FFT sem depender de um Signal específico."""
        if self.channel_policy not in _CHANNEL_POLICIES:
            raise ValueError("channel_policy must be 'select' or 'mean'.")
        if self.window_name not in _WINDOW_NAMES:
            raise ValueError("window_name must be 'rectangular' or 'hann'.")
        if self.scale not in _SPECTRUM_SCALES:
            raise ValueError("scale must be 'linear_amplitude' or 'dbfs'.")
        if self.normalization not in _SPECTRUM_NORMALIZATIONS:
            raise ValueError(
                "normalization must be 'coherent_gain_amplitude'."
            )
        if self.channel_index < 0:
            raise ValueError("channel_index must not be negative.")
        if self.start_time_s is not None and self.start_time_s < 0:
            raise ValueError("start_time_s must not be negative.")
        if self.end_time_s is not None and self.end_time_s < 0:
            raise ValueError("end_time_s must not be negative.")
        if (
            self.start_time_s is not None
            and self.end_time_s is not None
            and self.end_time_s <= self.start_time_s
        ):
            raise ValueError("end_time_s must be greater than start_time_s.")
        if self.n_fft is not None and self.n_fft <= 0:
            raise ValueError("n_fft must be positive when provided.")


@dataclass(frozen=True, slots=True)
class PeakDetectionSettings:
    """Configura a detecção de observações de picos em um Spectrum.

    Limiares de amplitude, proeminência e piso local usam a mesma escala do
    espectro analisado: amplitude linear ou dBFS. ``distance_bins`` e larguras
    são expressas em bins; nenhuma dessas grandezas é incerteza metrológica.
    """

    min_frequency_hz: float | None = None
    max_frequency_hz: float | None = None
    min_amplitude: float | None = None
    min_prominence: float | None = None
    distance_bins: int | None = None
    min_width_bins: float | None = None
    max_width_bins: float | None = None
    max_peaks: int | None = None
    interpolate: bool = True
    interpolation_method: Literal["parabolic_log_magnitude"] = "parabolic_log_magnitude"
    noise_method: Literal["local_median"] = "local_median"
    noise_window_bins: int = 21
    sort_by: Literal["frequency", "amplitude", "prominence"] = "frequency"

    def __post_init__(self) -> None:
        """Valida apenas parâmetros usados pela implementação atual."""
        if self.interpolation_method not in _PEAK_INTERPOLATION_METHODS:
            raise ValueError(
                "interpolation_method must be 'parabolic_log_magnitude'."
            )
        if self.noise_method not in _PEAK_NOISE_METHODS:
            raise ValueError("noise_method must be 'local_median'.")
        if self.sort_by not in _PEAK_SORT_ORDERS:
            raise ValueError(
                "sort_by must be 'frequency', 'amplitude', or 'prominence'."
            )
        for name, value in (
            ("min_frequency_hz", self.min_frequency_hz),
            ("max_frequency_hz", self.max_frequency_hz),
            ("min_amplitude", self.min_amplitude),
            ("min_prominence", self.min_prominence),
            ("min_width_bins", self.min_width_bins),
            ("max_width_bins", self.max_width_bins),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must not be negative.")
        if (
            self.min_frequency_hz is not None
            and self.max_frequency_hz is not None
            and self.max_frequency_hz < self.min_frequency_hz
        ):
            raise ValueError("max_frequency_hz must not be below min_frequency_hz.")
        if self.distance_bins is not None and self.distance_bins <= 0:
            raise ValueError("distance_bins must be positive when provided.")
        if (
            self.min_width_bins is not None
            and self.max_width_bins is not None
            and self.min_width_bins > self.max_width_bins
        ):
            raise ValueError("min_width_bins must not exceed max_width_bins.")
        if self.max_peaks is not None and self.max_peaks <= 0:
            raise ValueError("max_peaks must be positive when provided.")
        if self.noise_window_bins < 3:
            raise ValueError("noise_window_bins must be at least 3.")


@dataclass(frozen=True, slots=True)
class STFTSettings:
    """Configura STFT unilateral; tempos representam centros de janela."""

    channel_policy: Literal["select", "mean"] = "select"
    channel_index: int = 0
    start_time_s: float | None = None
    end_time_s: float | None = None
    remove_mean: bool = True
    window_name: Literal["rectangular", "hann"] = "hann"
    window_length: int = 1024
    n_fft: int | None = None
    hop_length: int = 256
    scale: Literal["linear_amplitude", "dbfs"] = "linear_amplitude"
    pad_end: bool = False
    frequency_min_hz: float | None = None
    frequency_max_hz: float | None = None

    def __post_init__(self) -> None:
        if self.channel_policy not in _CHANNEL_POLICIES:
            raise ValueError("channel_policy must be 'select' or 'mean'.")
        if self.window_name not in _WINDOW_NAMES:
            raise ValueError("window_name must be 'rectangular' or 'hann'.")
        if self.scale not in _SPECTRUM_SCALES:
            raise ValueError("scale must be 'linear_amplitude' or 'dbfs'.")
        if not isinstance(self.remove_mean, bool):
            raise ValueError("remove_mean must be a boolean.")
        if not isinstance(self.pad_end, bool):
            raise ValueError("pad_end must be a boolean.")
        if self.start_time_s is not None and self.start_time_s < 0:
            raise ValueError("start_time_s must not be negative.")
        if self.end_time_s is not None and self.end_time_s < 0:
            raise ValueError("end_time_s must not be negative.")
        if (
            self.start_time_s is not None
            and self.end_time_s is not None
            and self.end_time_s <= self.start_time_s
        ):
            raise ValueError("end_time_s must be greater than start_time_s.")
        if self.window_length <= 0 or self.hop_length <= 0:
            raise ValueError("window_length and hop_length must be positive.")
        if self.hop_length > self.window_length:
            raise ValueError("hop_length must not exceed window_length.")
        if self.n_fft is not None and self.n_fft < self.window_length:
            raise ValueError("n_fft must be at least window_length.")
        if self.channel_index < 0:
            raise ValueError("channel_index must not be negative.")
        if self.frequency_min_hz is not None and self.frequency_min_hz < 0:
            raise ValueError("frequency_min_hz must not be negative.")
        if self.frequency_max_hz is not None and self.frequency_max_hz < 0:
            raise ValueError("frequency_max_hz must not be negative.")
        if (
            self.frequency_min_hz is not None
            and self.frequency_max_hz is not None
            and self.frequency_max_hz < self.frequency_min_hz
        ):
            raise ValueError("frequency_max_hz must not be below frequency_min_hz.")


@dataclass(frozen=True, slots=True)
class FramePeakDetectionSettings:
    """Configura detecção independente de picos em quadros de uma STFT.

    ``peak_settings`` permanece a única fonte de limites espectrais,
    proeminência, largura, interpolação e piso local. Quadros silenciosos podem
    ser ignorados explicitamente, pois a ausência de picos não é um erro.
    """

    peak_settings: PeakDetectionSettings = PeakDetectionSettings()
    start_frame: int | None = None
    end_frame: int | None = None
    max_peaks_per_frame: int | None = None
    min_frame_amplitude: float | None = None
    silence_policy: Literal["skip", "process"] = "skip"
    store_frame_diagnostics: bool = True

    def __post_init__(self) -> None:
        """Valida seleção de quadros e políticas usadas pela implementação."""
        if self.start_frame is not None and self.start_frame < 0:
            raise ValueError("start_frame must not be negative.")
        if self.end_frame is not None and self.end_frame < 0:
            raise ValueError("end_frame must not be negative.")
        if (
            self.start_frame is not None
            and self.end_frame is not None
            and self.end_frame <= self.start_frame
        ):
            raise ValueError("end_frame must be greater than start_frame.")
        if self.max_peaks_per_frame is not None and self.max_peaks_per_frame <= 0:
            raise ValueError("max_peaks_per_frame must be positive when provided.")
        if (
            self.min_frame_amplitude is not None
            and not isfinite(self.min_frame_amplitude)
        ):
            raise ValueError("min_frame_amplitude must be finite when provided.")
        if self.silence_policy not in _FRAME_SILENCE_POLICIES:
            raise ValueError("silence_policy must be 'skip' or 'process'.")
        if not isinstance(self.store_frame_diagnostics, bool):
            raise ValueError("store_frame_diagnostics must be a boolean.")


@dataclass(frozen=True, slots=True)
class SpectralTrackingSettings:
    """Configura associação determinística de picos, não interpretação modal.

    A distância frequencial é normalizada pela tolerância em ``hz``,
    ``relative`` ou ``cents``. O método atual resolve associações um-para-um
    pelo algoritmo Húngaro. ``amplitude_weight`` é opcional e usa diferença em
    dB para espectros dBFS ou diferença relativa para amplitude linear.
    """

    frequency_tolerance: float = 0.05
    frequency_distance_unit: Literal["hz", "relative", "cents"] = "relative"
    max_gap_frames: int = 1
    min_track_length: int = 2
    frequency_weight: float = 1.0
    amplitude_weight: float = 0.0
    ambiguity_margin: float = 0.05
    maximum_association_cost: float = 2.0
    near_threshold_ratio: float = 0.9
    use_refined_frequency: bool = True
    association_method: Literal["hungarian"] = "hungarian"

    def __post_init__(self) -> None:
        """Valida parâmetros efetivamente usados pelo tracking inicial."""
        if self.frequency_tolerance <= 0:
            raise ValueError("frequency_tolerance must be positive.")
        if self.frequency_distance_unit not in _TRACKING_DISTANCE_UNITS:
            raise ValueError(
                "frequency_distance_unit must be 'hz', 'relative', or 'cents'."
            )
        if self.max_gap_frames < 0:
            raise ValueError("max_gap_frames must not be negative.")
        if self.min_track_length <= 0:
            raise ValueError("min_track_length must be positive.")
        if self.frequency_weight < 0 or self.amplitude_weight < 0:
            raise ValueError("tracking weights must not be negative.")
        if self.frequency_weight == 0 and self.amplitude_weight == 0:
            raise ValueError("at least one tracking weight must be positive.")
        if self.ambiguity_margin < 0:
            raise ValueError("ambiguity_margin must not be negative.")
        if not isfinite(self.maximum_association_cost) or self.maximum_association_cost <= 0:
            raise ValueError("maximum_association_cost must be finite and positive.")
        if not isfinite(self.near_threshold_ratio) or not 0 <= self.near_threshold_ratio <= 1:
            raise ValueError("near_threshold_ratio must be finite and in [0, 1].")
        if not isinstance(self.use_refined_frequency, bool):
            raise ValueError("use_refined_frequency must be a boolean.")
        if self.association_method not in _TRACKING_METHODS:
            raise ValueError("association_method must be 'hungarian'.")


@dataclass(frozen=True, slots=True)
class ModalCandidateSettings:
    """Critérios operacionais opcionais para promover trajetórias a candidatas.

    ``None`` desabilita um limite numérico. Os requisitos booleanos são
    habilitados somente quando ``True``. Os padrões exigem apenas duas
    observações e uma frequência representativa válida.
    """

    minimum_observation_count: int | None = 2
    minimum_coverage_fraction: float | None = None
    minimum_duration_s: float | None = None
    maximum_relative_frequency_stability: float | None = None
    maximum_absolute_frequency_drift_hz: float | None = None
    maximum_frequency_fit_rmse_hz: float | None = None
    require_successful_frequency_fit: bool = False
    require_amplitude_decay: bool = False
    minimum_amplitude_fit_r_squared: float | None = None
    minimum_decay_tau_s: float | None = None
    maximum_decay_tau_s: float | None = None
    maximum_ambiguous_assignment_fraction: float | None = None
    maximum_near_threshold_assignment_fraction: float | None = None
    minimum_assignment_margin: float | None = None
    allow_mixed_frequency_source: bool = True
    require_reaches_final_frame: bool = False
    require_impact_excitation: bool = False
    reject_persistent_background_tone: bool = False
    minimum_impact_level_increase_db: float | None = None

    def __post_init__(self) -> None:
        """Rejeita limites contraditórios, não finitos ou fora de domínio."""
        if self.minimum_observation_count is not None and self.minimum_observation_count < 0:
            raise ValueError("minimum_observation_count must not be negative.")
        fractions = {
            "minimum_coverage_fraction": self.minimum_coverage_fraction,
            "minimum_amplitude_fit_r_squared": self.minimum_amplitude_fit_r_squared,
            "maximum_ambiguous_assignment_fraction": self.maximum_ambiguous_assignment_fraction,
            "maximum_near_threshold_assignment_fraction": self.maximum_near_threshold_assignment_fraction,
        }
        for name, value in fractions.items():
            if value is not None and (not isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{name} must be finite and in [0, 1].")
        nonnegative = {
            "minimum_duration_s": self.minimum_duration_s,
            "maximum_relative_frequency_stability": self.maximum_relative_frequency_stability,
            "maximum_absolute_frequency_drift_hz": self.maximum_absolute_frequency_drift_hz,
            "maximum_frequency_fit_rmse_hz": self.maximum_frequency_fit_rmse_hz,
            "minimum_assignment_margin": self.minimum_assignment_margin,
            "minimum_impact_level_increase_db": self.minimum_impact_level_increase_db,
        }
        for name, value in nonnegative.items():
            if value is not None and (not isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative.")
        for name, value in {
            "minimum_decay_tau_s": self.minimum_decay_tau_s,
            "maximum_decay_tau_s": self.maximum_decay_tau_s,
        }.items():
            if value is not None and (not isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive.")
        if (
            self.minimum_decay_tau_s is not None
            and self.maximum_decay_tau_s is not None
            and self.minimum_decay_tau_s > self.maximum_decay_tau_s
        ):
            raise ValueError("minimum_decay_tau_s must not exceed maximum_decay_tau_s.")
        for name in (
            "require_successful_frequency_fit",
            "require_amplitude_decay",
            "allow_mixed_frequency_source",
            "require_reaches_final_frame",
            "require_impact_excitation",
            "reject_persistent_background_tone",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")


@dataclass(frozen=True, slots=True)
class PreImpactAnalysisSettings:
    """Janelas relativas ao impacto e limiares operacionais de excitação."""

    preimpact_window_start_s: float = -1.0
    preimpact_window_end_s: float = -0.1
    postimpact_window_start_s: float = 0.02
    postimpact_window_end_s: float = 0.30
    minimum_preimpact_point_count: int = 2
    minimum_postimpact_point_count: int = 2
    minimum_preimpact_coverage_fraction: float = 0.0
    minimum_preimpact_level: float | None = None
    minimum_impact_level_increase_db: float = 6.0
    unchanged_level_tolerance_db: float = 1.0
    preimpact_decay_slope_tolerance: float = 1e-12
    postimpact_decay_slope_tolerance: float = 1e-12
    require_postimpact_decay: bool = False
    absent_preimpact_is_excited: bool = True

    def __post_init__(self) -> None:
        """Valida janelas, contagens, frações e limiares finitos."""
        windows = (
            self.preimpact_window_start_s,
            self.preimpact_window_end_s,
            self.postimpact_window_start_s,
            self.postimpact_window_end_s,
        )
        if any(not isfinite(value) for value in windows):
            raise ValueError("pre/post-impact window bounds must be finite.")
        if self.preimpact_window_start_s >= self.preimpact_window_end_s:
            raise ValueError("preimpact window start must precede its end.")
        if self.preimpact_window_end_s >= 0:
            raise ValueError("preimpact window must end before impact.")
        if self.postimpact_window_start_s < 0:
            raise ValueError("postimpact window must not start before impact.")
        if self.postimpact_window_start_s >= self.postimpact_window_end_s:
            raise ValueError("postimpact window start must precede its end.")
        if min(self.minimum_preimpact_point_count, self.minimum_postimpact_point_count) < 0:
            raise ValueError("pre/post-impact minimum point counts must not be negative.")
        if not isfinite(self.minimum_preimpact_coverage_fraction) or not 0 <= self.minimum_preimpact_coverage_fraction <= 1:
            raise ValueError("minimum_preimpact_coverage_fraction must be finite and in [0, 1].")
        if self.minimum_preimpact_level is not None and not isfinite(self.minimum_preimpact_level):
            raise ValueError("minimum_preimpact_level must be finite when provided.")
        for name in (
            "minimum_impact_level_increase_db",
            "unchanged_level_tolerance_db",
            "preimpact_decay_slope_tolerance",
            "postimpact_decay_slope_tolerance",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        for name in ("require_postimpact_decay", "absent_preimpact_is_excited"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")


@dataclass(frozen=True, slots=True)
class AnalysisSettings:
    """Agrupa as configurações dos módulos científicos já implementados.

    Args:
        temporal: Parâmetros da análise temporal.
        spectrum: Parâmetros da análise espectral estacionária.
    """

    temporal: TemporalAnalysisSettings = TemporalAnalysisSettings()
    spectrum: SpectrumAnalysisSettings = SpectrumAnalysisSettings()
    stft: STFTSettings = STFTSettings()
    frame_peaks: FramePeakDetectionSettings = FramePeakDetectionSettings()
    tracking: SpectralTrackingSettings = SpectralTrackingSettings()
