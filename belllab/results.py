"""Agrupamentos tipados de resultados de análise do BellLab.

Cada dataclass agrega estruturas de :mod:`belllab.types` por domínio de análise.
Módulos futuros devem devolver estas classes, em vez de dicionários, para
oferecer uma interface estável a comparações, gráficos e relatórios.
"""

from __future__ import annotations

from dataclasses import dataclass

from belllab.config import (
    AnalysisSettings,
    FramePeakDetectionSettings,
    PeakDetectionSettings,
    SpectralTrackingSettings,
    SpectrumAnalysisSettings,
    STFTSettings,
    TemporalAnalysisSettings,
)
from belllab.types import (
    DecayFit,
    Envelope,
    FramePeaks,
    ImpactReport,
    ModalMode,
    NoiseReport,
    Spectrum,
    TimeFrequencySpectrum,
    SpectralPeak,
    SpectralTrack,
    TrackAssignmentDiagnostic,
    TemporalMetrics,
)
from belllab.types import Signal


@dataclass(frozen=True, slots=True)
class TemporalResults:
    """Reúne os resultados canônicos de uma execução temporal completa.

    Args:
        impact: Detecção de impacto, se calculada.
        noise: Estimativa canônica de ruído, se calculada.
        metrics: Métricas globais temporais, se calculadas.
        envelope: Envelope de amplitude calculado, se solicitado.
        decay_fit: Ajuste de decaimento calculado, se aplicável.
        settings: Parâmetros temporais efetivamente usados.

    ``None`` significa que uma grandeza não foi calculada ou não estava
    disponível para a execução representada por esta instância.
    """

    impact: ImpactReport | None = None
    noise: NoiseReport | None = None
    metrics: TemporalMetrics | None = None
    envelope: Envelope | None = None
    decay_fit: DecayFit | None = None
    settings: TemporalAnalysisSettings | None = None


@dataclass(frozen=True, slots=True)
class SpectrumResults:
    """Reúne o resultado de uma FFT estacionária unilateral.

    Args:
        spectrum: Série espectral calculada para o trecho selecionado.
        settings: Parâmetros espectrais efetivamente usados.
        diagnostics: Diagnósticos textuais não fatais da execução.
    """

    spectrum: Spectrum | None = None
    settings: SpectrumAnalysisSettings | None = None
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PeakDetectionResults:
    """Resultado ordenado de detecção de picos em um Spectrum existente."""

    spectrum: Spectrum
    peaks: tuple[SpectralPeak, ...]
    settings: PeakDetectionSettings
    method: str
    diagnostics: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    candidate_count: int = 0
    accepted_count: int = 0
    analyzed_frequency_range_hz: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class STFTResults:
    """Resultado de uma STFT sem tracking ou interpretação modal."""

    time_frequency: TimeFrequencySpectrum
    settings: STFTSettings
    diagnostics: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TimeFrequencyPeakResults:
    """Picos independentes detectados nos quadros de uma STFT.

    Picos por quadro ainda não constituem trajetórias, nem modos físicos.
    """

    time_frequency: TimeFrequencySpectrum
    frames: tuple[FramePeaks, ...]
    settings: FramePeakDetectionSettings
    processed_frame_count: int
    total_peak_count: int
    frames_without_peaks: int
    diagnostics: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Valida coerência pública entre quadros, STFT e contagens."""
        if self.processed_frame_count != len(self.frames):
            raise ValueError("processed_frame_count must match frames.")
        indices = tuple(frame.frame_index for frame in self.frames)
        times = tuple(frame.time_s for frame in self.frames)
        if any(later <= earlier for earlier, later in zip(indices, indices[1:])):
            raise ValueError("frame indices must be strictly increasing and unique.")
        if any(later <= earlier for earlier, later in zip(times, times[1:])):
            raise ValueError("frame times must be strictly increasing.")
        if any(index >= len(self.time_frequency.times_s) for index in indices):
            raise ValueError("frame index is outside the STFT.")
        if any(
            abs(frame.time_s - self.time_frequency.times_s[frame.frame_index]) > 1e-12
            for frame in self.frames
        ):
            raise ValueError("frame time must match the referenced STFT frame.")
        if self.total_peak_count != sum(frame.accepted_count for frame in self.frames):
            raise ValueError("total_peak_count must equal accepted frame peaks.")
        if self.frames_without_peaks != sum(not frame.peaks for frame in self.frames):
            raise ValueError("frames_without_peaks is inconsistent with frames.")
        units = {
            _canonical_amplitude_unit(peak.amplitude_unit)
            for frame in self.frames
            for peak in frame.peaks
        }
        if len(units) > 1 or (units and None in units):
            raise ValueError("all tracking peaks must share a known amplitude unit.")


@dataclass(frozen=True, slots=True)
class SpectralTrackingResults:
    """Trajetórias associadas de picos, sem interpretação modal física."""

    frame_peaks: TimeFrequencyPeakResults
    tracks: tuple[SpectralTrack, ...]
    rejected_tracks: tuple[SpectralTrack, ...]
    settings: SpectralTrackingSettings
    track_count: int
    tracks_reaching_final_frame: int
    ambiguous_assignment_count: int = 0
    near_threshold_assignment_count: int = 0
    assignment_margin_min: float | None = None
    assignment_diagnostics: tuple[TrackAssignmentDiagnostic, ...] = ()
    diagnostics: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def active_track_count(self) -> int:
        """Alias legado para trajetórias abertas no último quadro offline."""
        return self.tracks_reaching_final_frame


def _canonical_amplitude_unit(unit: str) -> str | None:
    """Normaliza apenas unidades de amplitude aceitas pelo tracking."""
    normalized = unit.strip().lower()
    if "dbfs" in normalized and "amplitude" in normalized:
        return "dbfs_amplitude"
    if normalized in {"normalized amplitude (peak)", "linear_amplitude"}:
        return "linear_amplitude"
    return None


@dataclass(frozen=True, slots=True)
class ModalResults:
    """Reúne os modos acústicos identificados em uma futura análise modal.

    Args:
        modes: Modos identificados, ordenados conforme o critério da análise.
    """

    modes: tuple[ModalMode, ...] = ()


@dataclass(frozen=True, slots=True)
class ProcessingContext:
    """Agrupa as entradas e resultados associados a um ciclo de análise.

    Args:
        signal: Sinal carregado que serve de entrada para as análises.
        settings: Configurações que devem tornar o processamento reprodutível.
        temporal_results: Resultado temporal disponível, se já produzido.
        spectrum_results: Resultado espectral disponível, se já produzido.
        modal_results: Resultado modal disponível, se já produzido.

    A classe descreve uma execução analítica imutável; ela é a proprietária dos
    resultados durante a execução. ``Recording`` pode posteriormente receber
    uma cópia-associação desses resultados como snapshot de domínio, mas os dois
    objetos não devem ser atualizados de forma concorrente.
    """

    signal: Signal
    settings: AnalysisSettings
    temporal_results: TemporalResults | None = None
    spectrum_results: SpectrumResults | None = None
    modal_results: ModalResults | None = None
