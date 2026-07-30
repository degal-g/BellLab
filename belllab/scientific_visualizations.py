"""Reproducible scientific visualizations for existing BellLab results.

This module renders figures from already computed BellLab objects. It does not
open audio files, rerun FFT/STFT, rebuild tracks, characterize candidates,
re-estimate modal parameters, recompute Q, or infer physical conclusions from
visual patterns.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from belllab.dynamic_comparison import DYNAMIC_LABEL_ORDER
from belllab.experiment_pipeline import (
    ExperimentAnalysisResult,
    ExperimentAnalysisStatus,
    ExperimentConditionAnalysisResult,
    ExperimentCrossConditionAnalysisResult,
    ExperimentRecordingAnalysisResult,
)
from belllab.results import (
    PeakDetectionResults,
    SpectralTrackingResults,
    SpectrumResults,
    STFTResults,
    TemporalResults,
)
from belllab.results_export import (
    ExportOverwritePolicy,
    ResultsExportSettings,
    export_artifact_checksum,
    normalize_experiment_for_export,
)
from belllab.synthetic_validation import (
    SyntheticMonteCarloValidation,
    SyntheticScenarioValidationResult,
    SyntheticValidationCampaignResult,
)
from belllab.types import Envelope, Signal, Spectrum, SpectralPeak, SpectralTrack, TimeFrequencySpectrum


class ScientificVisualizationStatus(str, Enum):
    """Mutually exclusive status for a scientific visualization."""

    CREATED = "created"
    CREATED_WITH_RESERVATIONS = "created_with_reservations"
    SKIPPED = "skipped"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    FAILED = "failed"
    INVALID_INPUT = "invalid_input"


class ScientificVisualizationReason(str, Enum):
    """Typed reasons for figure support, reservations, insufficiency and failure."""

    FIGURE_CREATED = "figure_created"
    REQUESTED_LAYERS_RENDERED = "requested_layers_rendered"
    UNCERTAINTY_RENDERED = "uncertainty_rendered"
    PROVENANCE_ATTACHED = "provenance_attached"
    PARTIAL_SOURCE_RESULT = "partial_source_result"
    SOURCE_REQUIRES_REVIEW = "source_requires_review"
    MISSING_OPTIONAL_LAYER = "missing_optional_layer"
    INTERPOLATION_RENDERED = "interpolation_rendered"
    RESOLUTION_LIMITED = "resolution_limited"
    LOG_SCALE_CLIPPED = "log_scale_clipped"
    TOO_MANY_LABELS = "too_many_labels"
    OVERLAPPING_ANNOTATIONS_REDUCED = "overlapping_annotations_reduced"
    NONUNIFORM_TIME_AXIS = "nonuniform_time_axis"
    NONUNIFORM_FREQUENCY_AXIS = "nonuniform_frequency_axis"
    POSSIBLE_SPLIT_CONTEXT = "possible_split_context"
    POSSIBLE_MERGE_CONTEXT = "possible_merge_context"
    POSSIBLE_BEATING_CONTEXT = "possible_beating_context"
    MISSING_TIME_SERIES = "missing_time_series"
    MISSING_SPECTRUM = "missing_spectrum"
    MISSING_STFT = "missing_stft"
    MISSING_TRACKS = "missing_tracks"
    MISSING_CANDIDATES = "missing_candidates"
    MISSING_HYPOTHESES = "missing_hypotheses"
    MISSING_PARAMETERS = "missing_parameters"
    MISSING_Q_ESTIMATES = "missing_q_estimates"
    MISSING_ENERGY_EXCHANGE = "missing_energy_exchange"
    INSUFFICIENT_POINTS = "insufficient_points"
    NO_VALID_VALUES = "no_valid_values"
    RENDERING_ERROR = "rendering_error"
    UNSUPPORTED_PLOT_TYPE = "unsupported_plot_type"
    FILESYSTEM_ERROR = "filesystem_error"
    INVALID_AXIS_LIMITS = "invalid_axis_limits"
    INVALID_NUMERIC_VALUES = "invalid_numeric_values"
    BACKEND_ERROR = "backend_error"


class ScientificColorPolicy(str, Enum):
    """Deterministic color policy for scientific plots."""

    DYNAMIC_CONDITION = "dynamic_condition"
    STATUS = "status"
    STABLE_HASH = "stable_hash"
    MONOCHROME = "monochrome"
    CUSTOM = "custom"


class ScientificFigureType(str, Enum):
    """Canonical figure type identifiers used by collections and filenames."""

    WAVEFORM = "waveform"
    TEMPORAL_ENVELOPE = "temporal_envelope"
    DECAY_ESTIMATE = "decay_estimate"
    GLOBAL_SPECTRUM = "global_spectrum"
    SPECTRAL_PEAKS = "spectral_peaks"
    SPECTROGRAM = "spectrogram"
    FREQUENCY_TRACKS = "frequency_tracks"
    MODAL_CANDIDATES = "modal_candidates"
    WITHIN_CONDITION_ASSOCIATIONS = "within_condition_associations"
    CROSS_CONDITION_ASSOCIATIONS = "cross_condition_associations"
    CANDIDATE_CHAINS = "candidate_chains"
    MODAL_HYPOTHESES = "modal_hypotheses"
    MODAL_FREQUENCY_TRAJECTORIES = "modal_frequency_trajectories"
    MODAL_PARAMETERS = "modal_parameters"
    MODAL_Q_FACTORS = "modal_q_factors"
    MODAL_BANDWIDTH = "modal_bandwidth"
    DYNAMIC_CONDITION_COMPARISON = "dynamic_condition_comparison"
    MODAL_ENERGY_EXCHANGE_EVIDENCE = "modal_energy_exchange_evidence"
    MODAL_ENERGY_EXCHANGE_CORRELATION = "modal_energy_exchange_correlation"
    SYNTHETIC_VALIDATION_RESULT = "synthetic_validation_result"
    SYNTHETIC_VALIDATION_CAMPAIGN = "synthetic_validation_campaign"
    EXPERIMENT_SUMMARY = "experiment_summary"


@dataclass(frozen=True, slots=True)
class ScientificStatusStyle:
    """Visual style for a result status that does not rely on color alone."""

    marker: str = "o"
    line_style: str = "-"
    alpha: float = 1.0
    facecolor: str = "white"
    edgecolor: str = "black"
    hatch: str | None = None

    def __post_init__(self) -> None:
        if not self.marker:
            raise ValueError("marker must not be empty.")
        if not self.line_style:
            raise ValueError("line_style must not be empty.")
        if not math.isfinite(self.alpha) or not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be finite and in [0, 1].")
        if not isinstance(self.facecolor, str) or not self.facecolor.strip():
            raise ValueError("facecolor must be a nonempty string.")
        if not isinstance(self.edgecolor, str) or not self.edgecolor.strip():
            raise ValueError("edgecolor must be a nonempty string.")
        if self.hatch is not None:
            if not isinstance(self.hatch, str) or not self.hatch.strip():
                raise ValueError("hatch must be a nonempty string.")


DEFAULT_CONDITION_COLORS: Mapping[str, str] = MappingProxyType({
    "pp": "#0072B2",
    "p": "#009E73",
    "mf": "#E69F00",
    "f": "#D55E00",
    "ff": "#CC79A7",
})

DEFAULT_CONDITION_MARKERS: Mapping[str, str] = MappingProxyType({
    "pp": "o",
    "p": "s",
    "mf": "^",
    "f": "D",
    "ff": "P",
})

DEFAULT_STATUS_STYLES: Mapping[str, ScientificStatusStyle] = MappingProxyType({
    "valid": ScientificStatusStyle(marker="o", line_style="-", alpha=1.0, facecolor="white", edgecolor="#0072B2"),
    "accepted": ScientificStatusStyle(marker="o", line_style="-", alpha=1.0, facecolor="white", edgecolor="#0072B2"),
    "supported": ScientificStatusStyle(marker="o", line_style="-", alpha=1.0, facecolor="white", edgecolor="#0072B2"),
    "completed": ScientificStatusStyle(marker="o", line_style="-", alpha=1.0, facecolor="white", edgecolor="#0072B2"),
    "valid_with_reservations": ScientificStatusStyle(marker="s", line_style="--", alpha=0.9, facecolor="white", edgecolor="#E69F00", hatch="//"),
    "accepted_with_reservations": ScientificStatusStyle(marker="s", line_style="--", alpha=0.9, facecolor="white", edgecolor="#E69F00", hatch="//"),
    "supported_with_reservations": ScientificStatusStyle(marker="s", line_style="--", alpha=0.9, facecolor="white", edgecolor="#E69F00", hatch="//"),
    "partial": ScientificStatusStyle(marker="^", line_style="-.", alpha=0.85, facecolor="white", edgecolor="#56B4E9"),
    "inconclusive": ScientificStatusStyle(marker="v", line_style=":", alpha=0.8, facecolor="white", edgecolor="#999999"),
    "rejected": ScientificStatusStyle(marker="x", line_style=":", alpha=0.75, facecolor="white", edgecolor="#D55E00"),
    "not_supported": ScientificStatusStyle(marker="x", line_style=":", alpha=0.75, facecolor="white", edgecolor="#D55E00"),
    "insufficient_evidence": ScientificStatusStyle(marker=".", line_style=":", alpha=0.6, facecolor="white", edgecolor="#666666"),
    "invalid_input": ScientificStatusStyle(marker="X", line_style="--", alpha=0.7, facecolor="white", edgecolor="#CC0000", hatch="xx"),
    "failed": ScientificStatusStyle(marker="X", line_style="--", alpha=0.7, facecolor="white", edgecolor="#CC0000", hatch="xx"),
})

DEFAULT_FIGURE_TYPES: tuple[ScientificFigureType, ...] = (
    ScientificFigureType.WAVEFORM,
    ScientificFigureType.TEMPORAL_ENVELOPE,
    ScientificFigureType.GLOBAL_SPECTRUM,
    ScientificFigureType.SPECTRAL_PEAKS,
    ScientificFigureType.SPECTROGRAM,
    ScientificFigureType.FREQUENCY_TRACKS,
    ScientificFigureType.MODAL_CANDIDATES,
    ScientificFigureType.WITHIN_CONDITION_ASSOCIATIONS,
    ScientificFigureType.CROSS_CONDITION_ASSOCIATIONS,
    ScientificFigureType.CANDIDATE_CHAINS,
    ScientificFigureType.MODAL_HYPOTHESES,
    ScientificFigureType.MODAL_FREQUENCY_TRAJECTORIES,
    ScientificFigureType.MODAL_PARAMETERS,
    ScientificFigureType.MODAL_Q_FACTORS,
    ScientificFigureType.MODAL_BANDWIDTH,
    ScientificFigureType.DYNAMIC_CONDITION_COMPARISON,
    ScientificFigureType.MODAL_ENERGY_EXCHANGE_EVIDENCE,
    ScientificFigureType.MODAL_ENERGY_EXCHANGE_CORRELATION,
    ScientificFigureType.EXPERIMENT_SUMMARY,
)


@dataclass(frozen=True, slots=True)
class ScientificVisualizationSettings:
    """Explicit, deterministic settings for BellLab scientific figures."""

    output_directory: str | Path = Path("belllab-figures")
    file_prefix: str = "belllab_figure"
    formats: tuple[str, ...] = ("png", "svg")
    dpi: int = 120
    transparent_background: bool = False
    bbox_inches: str | None = "tight"
    pad_inches: float = 0.05
    overwrite_policy: ExportOverwritePolicy = ExportOverwritePolicy.ERROR
    atomic_write: bool = True
    close_after_save: bool = True

    figure_width_in: float = 7.0
    figure_height_in: float = 4.2
    aspect_policy: str = "standard"
    panel_spacing: float = 0.35
    constrained_layout: bool = True
    tight_layout: bool = False

    font_family: str = "DejaVu Sans"
    base_font_size: float = 9.0
    title_font_size: float = 11.0
    label_font_size: float = 9.0
    tick_font_size: float = 8.0
    legend_font_size: float = 8.0
    annotation_font_size: float = 7.0
    mathtext_fontset: str = "dejavusans"

    line_width: float = 1.4
    marker_size: float = 4.5
    errorbar_capsize: float = 2.5
    grid: bool = True
    grid_alpha: float = 0.25
    minor_ticks: bool = True
    show_legend: bool = True
    legend_location: str = "best"

    time_unit: str = "s"
    frequency_unit: str = "Hz"
    amplitude_scale: str = "linear"
    frequency_scale: str = "linear"
    spectrum_db_floor: float = -120.0
    spectrogram_db_floor: float = -120.0
    log_frequency_min_hz: float = 1.0

    show_uncertainty: bool = True
    show_rejected_results: bool = True
    show_inconclusive_results: bool = True
    show_reservations: bool = True
    show_diagnostics: bool = False
    show_ids: bool = False
    show_provenance_footer: bool = True
    show_thresholds: bool = True
    show_fit_curves: bool = True
    show_raw_points: bool = True
    show_interpolated_points: bool = True
    show_confidence_intervals: bool = True

    maximum_waveform_points: int = 3000
    maximum_spectrum_points: int = 4000
    maximum_track_points: int = 1500
    decimation_method: str = "minmax"
    preserve_extrema: bool = True

    color_policy: ScientificColorPolicy = ScientificColorPolicy.DYNAMIC_CONDITION
    condition_color_mapping: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType(dict(DEFAULT_CONDITION_COLORS))
    )
    status_style_mapping: Mapping[str, ScientificStatusStyle] = field(
        default_factory=lambda: MappingProxyType(dict(DEFAULT_STATUS_STYLES))
    )
    marker_policy: str = "dynamic_condition"
    line_style_policy: str = "status"
    grayscale_compatible: bool = True
    colorblind_safe: bool = True

    maximum_annotation_count: int = 16
    annotation_priority_policy: str = "status_then_amplitude"
    minimum_label_separation: float = 0.02
    label_collision_policy: str = "reduce"
    figure_types: tuple[ScientificFigureType | str, ...] = DEFAULT_FIGURE_TYPES

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_directory", Path(self.output_directory))
        _text(self.file_prefix, "file_prefix")
        formats = tuple(str(item).lower() for item in self.formats)
        if not formats:
            raise ValueError("formats must not be empty.")
        unsupported = tuple(item for item in formats if item not in {"png", "svg", "pdf"})
        if unsupported:
            raise ValueError(f"unsupported figure format: {unsupported[0]}")
        object.__setattr__(self, "formats", tuple(dict.fromkeys(formats)))
        if self.dpi <= 0:
            raise ValueError("dpi must be positive.")
        if self.figure_width_in <= 0 or self.figure_height_in <= 0:
            raise ValueError("figure dimensions must be positive.")
        if self.pad_inches < 0 or self.panel_spacing < 0:
            raise ValueError("padding and panel spacing must not be negative.")
        for name in (
            "transparent_background",
            "atomic_write",
            "close_after_save",
            "constrained_layout",
            "tight_layout",
            "grid",
            "minor_ticks",
            "show_legend",
            "show_uncertainty",
            "show_rejected_results",
            "show_inconclusive_results",
            "show_reservations",
            "show_diagnostics",
            "show_ids",
            "show_provenance_footer",
            "show_thresholds",
            "show_fit_curves",
            "show_raw_points",
            "show_interpolated_points",
            "show_confidence_intervals",
            "preserve_extrema",
            "grayscale_compatible",
            "colorblind_safe",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")
        object.__setattr__(
            self,
            "overwrite_policy",
            _coerce_enum(self.overwrite_policy, ExportOverwritePolicy),
        )
        object.__setattr__(
            self,
            "color_policy",
            _coerce_enum(self.color_policy, ScientificColorPolicy),
        )
        if self.aspect_policy not in {"standard", "wide", "square"}:
            raise ValueError("aspect_policy is not recognized.")
        for name in (
            "base_font_size",
            "title_font_size",
            "label_font_size",
            "tick_font_size",
            "legend_font_size",
            "annotation_font_size",
            "line_width",
            "marker_size",
            "errorbar_capsize",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not 0 <= self.grid_alpha <= 1:
            raise ValueError("grid_alpha must be in [0, 1].")
        if self.amplitude_scale not in {"linear", "db"}:
            raise ValueError("amplitude_scale must be linear or db.")
        if self.frequency_scale not in {"linear", "log"}:
            raise ValueError("frequency_scale must be linear or log.")
        if not math.isfinite(self.spectrum_db_floor) or not math.isfinite(self.spectrogram_db_floor):
            raise ValueError("dB floors must be finite.")
        if not math.isfinite(self.log_frequency_min_hz) or self.log_frequency_min_hz <= 0:
            raise ValueError("log_frequency_min_hz must be finite and positive.")
        for name in ("maximum_waveform_points", "maximum_spectrum_points", "maximum_track_points"):
            if getattr(self, name) <= 1:
                raise ValueError(f"{name} must be greater than one.")
        if self.decimation_method not in {"minmax", "stride", "none"}:
            raise ValueError("decimation_method is not recognized.")
        if self.marker_policy not in {"dynamic_condition", "status", "stable_hash", "fixed"}:
            raise ValueError("marker_policy is not recognized.")
        if self.line_style_policy not in {"status", "stable_hash", "fixed"}:
            raise ValueError("line_style_policy is not recognized.")
        if self.maximum_annotation_count < 0:
            raise ValueError("maximum_annotation_count must not be negative.")
        if self.annotation_priority_policy not in {"status_then_amplitude", "frequency", "id"}:
            raise ValueError("annotation_priority_policy is not recognized.")
        if not math.isfinite(self.minimum_label_separation) or self.minimum_label_separation < 0:
            raise ValueError("minimum_label_separation must be finite and non-negative.")
        if self.label_collision_policy not in {"reduce", "none"}:
            raise ValueError("label_collision_policy is not recognized.")
        object.__setattr__(
            self,
            "condition_color_mapping",
            MappingProxyType({str(key): str(value) for key, value in self.condition_color_mapping.items()}),
        )
        object.__setattr__(
            self,
            "status_style_mapping",
            MappingProxyType({
                str(key): value if isinstance(value, ScientificStatusStyle) else ScientificStatusStyle(**dict(value))
                for key, value in self.status_style_mapping.items()
            }),
        )
        figure_types = tuple(_coerce_enum(item, ScientificFigureType) for item in self.figure_types)
        if len(figure_types) != len(set(figure_types)):
            raise ValueError("figure_types must not contain duplicates.")
        object.__setattr__(self, "figure_types", figure_types)


@dataclass(frozen=True, slots=True)
class ScientificFigureProvenance:
    """Deterministic provenance attached to each figure."""

    figure_id: str
    figure_type: str
    analysis_id: str | None
    experiment_id: str | None
    recording_ids: tuple[str, ...]
    condition_labels: tuple[str, ...]
    source_result_ids: tuple[str, ...]
    settings_fingerprint: str
    belllab_version: str | None
    source_statuses: tuple[str, ...]
    interpolations_used: tuple[str, ...]
    decimations_used: tuple[str, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.figure_id, "figure_id")
        _text(self.figure_type, "figure_type")
        _text(self.settings_fingerprint, "settings_fingerprint")
        for name in (
            "recording_ids",
            "condition_labels",
            "source_result_ids",
            "source_statuses",
            "interpolations_used",
            "decimations_used",
            "diagnostics",
        ):
            object.__setattr__(self, name, _unique_texts(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class ScientificFigureArtifact:
    """A saved visual artifact with content checksum."""

    artifact_id: str
    figure_id: str
    format: str
    path: str | None
    relative_path: str | None
    checksum: str | None
    size_bytes: int | None
    width_px: int | None
    height_px: int | None
    dpi: int
    status: ScientificVisualizationStatus
    reasons: tuple[ScientificVisualizationReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.artifact_id, "artifact_id")
        _text(self.figure_id, "figure_id")
        if self.format not in {"png", "svg", "pdf"}:
            raise ValueError("artifact format is not supported.")
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ValueError("size_bytes must not be negative.")
        if self.width_px is not None and self.width_px <= 0:
            raise ValueError("width_px must be positive when present.")
        if self.height_px is not None and self.height_px <= 0:
            raise ValueError("height_px must be positive when present.")
        if self.dpi <= 0:
            raise ValueError("dpi must be positive.")
        object.__setattr__(self, "status", _coerce_enum(self.status, ScientificVisualizationStatus))
        object.__setattr__(self, "reasons", _reason_tuple(self.reasons))
        object.__setattr__(self, "diagnostics", _unique_texts(self.diagnostics))


@dataclass(frozen=True, slots=True)
class ScientificFigureResult:
    """Result of creating one scientific figure."""

    figure_id: str
    figure_type: ScientificFigureType
    status: ScientificVisualizationStatus
    figure: object | None
    axes: tuple[object, ...]
    artifacts: tuple[ScientificFigureArtifact, ...]
    source_ids: tuple[str, ...]
    provenance: ScientificFigureProvenance
    supporting_reasons: tuple[ScientificVisualizationReason, ...]
    reservation_reasons: tuple[ScientificVisualizationReason, ...]
    skipped_reasons: tuple[ScientificVisualizationReason, ...]
    insufficient_evidence_reasons: tuple[ScientificVisualizationReason, ...]
    failure_reasons: tuple[ScientificVisualizationReason, ...]
    valid: bool
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.figure_id, "figure_id")
        object.__setattr__(self, "figure_type", _coerce_enum(self.figure_type, ScientificFigureType))
        object.__setattr__(self, "status", _coerce_enum(self.status, ScientificVisualizationStatus))
        for name in ("axes", "artifacts", "source_ids"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in (
            "supporting_reasons",
            "reservation_reasons",
            "skipped_reasons",
            "insufficient_evidence_reasons",
            "failure_reasons",
        ):
            object.__setattr__(self, name, _reason_tuple(getattr(self, name)))
        expected_valid = self.status in {
            ScientificVisualizationStatus.CREATED,
            ScientificVisualizationStatus.CREATED_WITH_RESERVATIONS,
        }
        if self.valid != expected_valid:
            raise ValueError("valid must mirror created statuses.")
        object.__setattr__(self, "diagnostics", _unique_texts(self.diagnostics))


@dataclass(frozen=True, slots=True)
class ScientificFigureCollection:
    """A deterministic collection of requested experiment figures."""

    collection_id: str
    analysis_id: str | None
    experiment_id: str | None
    figures: tuple[ScientificFigureResult, ...]
    artifacts: tuple[ScientificFigureArtifact, ...]
    requested_figure_types: tuple[ScientificFigureType, ...]
    completed_figure_types: tuple[ScientificFigureType, ...]
    skipped_figure_types: tuple[ScientificFigureType, ...]
    failed_figure_types: tuple[ScientificFigureType, ...]
    status: ScientificVisualizationStatus
    settings: ScientificVisualizationSettings
    provenance: Mapping[str, object]
    valid: bool
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.collection_id, "collection_id")
        for name in ("figures", "artifacts"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ("requested_figure_types", "completed_figure_types", "skipped_figure_types", "failed_figure_types"):
            object.__setattr__(self, name, tuple(_coerce_enum(item, ScientificFigureType) for item in getattr(self, name)))
        object.__setattr__(self, "status", _coerce_enum(self.status, ScientificVisualizationStatus))
        expected_valid = self.status in {
            ScientificVisualizationStatus.CREATED,
            ScientificVisualizationStatus.CREATED_WITH_RESERVATIONS,
        }
        if self.valid != expected_valid:
            raise ValueError("collection valid must mirror created statuses.")
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))
        object.__setattr__(self, "diagnostics", _unique_texts(self.diagnostics))


def plot_waveform(
    source: Signal | ExperimentRecordingAnalysisResult | ExperimentAnalysisResult,
    settings: ScientificVisualizationSettings | None = None,
    *,
    save: bool = False,
) -> ScientificFigureResult:
    """Plot an already loaded time-domain waveform without reopening audio."""

    cfg = settings or ScientificVisualizationSettings()
    signal, source_id, context = _extract_signal(source)
    if signal is None:
        return _insufficient_result(ScientificFigureType.WAVEFORM, cfg, source, ScientificVisualizationReason.MISSING_TIME_SERIES)
    try:
        times = _finite_array(signal.time, "time")
        if times.size < 2:
            return _insufficient_result(ScientificFigureType.WAVEFORM, cfg, source, ScientificVisualizationReason.INSUFFICIENT_POINTS)
        samples = np.asarray(signal.samples, dtype=float)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        if samples.shape[1] != times.size:
            return _invalid_result(ScientificFigureType.WAVEFORM, cfg, source, ScientificVisualizationReason.INVALID_NUMERIC_VALUES)
        fig, ax = _new_figure(cfg)
        diagnostics: list[str] = ["waveform_uses_loaded_signal_only", "no_downmix_or_normalization_performed"]
        for channel_index, channel in enumerate(samples):
            if not np.all(np.isfinite(channel)):
                return _invalid_result(ScientificFigureType.WAVEFORM, cfg, source, ScientificVisualizationReason.INVALID_NUMERIC_VALUES)
            x, y, decimation = _decimate_xy(times, channel, cfg.maximum_waveform_points, cfg)
            diagnostics.extend(decimation)
            ax.plot(x, y, linewidth=cfg.line_width, color=_color_for(str(channel_index), cfg), label=f"channel {channel_index}")
        ax.set_title(f"Waveform - {source_id}", fontsize=cfg.title_font_size)
        ax.set_xlabel(f"Time ({cfg.time_unit})")
        ax.set_ylabel(f"Amplitude ({signal.unit})")
        _apply_axes_style(ax, cfg)
        clipped_sample_count = int(np.count_nonzero(np.isclose(np.abs(samples), 1.0, rtol=0.0, atol=1e-12)))
        minimum_clipped_samples = max(2, int(math.ceil(samples.size * 0.005)))
        if cfg.show_thresholds and clipped_sample_count >= minimum_clipped_samples:
            ax.axhline(1.0, color="#D55E00", linestyle=":", linewidth=cfg.line_width, label="clipping threshold")
            ax.axhline(-1.0, color="#D55E00", linestyle=":", linewidth=cfg.line_width)
            diagnostics.append("clipping_threshold_rendered_from_loaded_samples")
        return _created_result(ScientificFigureType.WAVEFORM, fig, (ax,), cfg, source, context, tuple(diagnostics), save=save)
    except Exception as exc:
        return _failed_result(ScientificFigureType.WAVEFORM, cfg, source, exc)


def plot_temporal_envelope(
    source: Envelope | TemporalResults | ExperimentRecordingAnalysisResult | ExperimentAnalysisResult,
    settings: ScientificVisualizationSettings | None = None,
    *,
    save: bool = False,
) -> ScientificFigureResult:
    """Plot an existing temporal envelope without recalculating it."""

    cfg = settings or ScientificVisualizationSettings()
    envelope, source_id, context = _extract_envelope(source)
    if envelope is None:
        return _insufficient_result(ScientificFigureType.TEMPORAL_ENVELOPE, cfg, source, ScientificVisualizationReason.MISSING_TIME_SERIES)
    try:
        times = _finite_array(envelope.times_s, "time")
        amplitudes = _finite_array(envelope.amplitudes, "amplitude")
        if times.size < 2 or amplitudes.size < 2:
            return _insufficient_result(ScientificFigureType.TEMPORAL_ENVELOPE, cfg, source, ScientificVisualizationReason.INSUFFICIENT_POINTS)
        y, ylabel, diagnostics = _amplitude_for_display(amplitudes, envelope.unit, cfg, floor=cfg.spectrum_db_floor)
        x, y_decimated, decimation = _decimate_xy(times, y, cfg.maximum_waveform_points, cfg)
        fig, ax = _new_figure(cfg)
        ax.plot(x, y_decimated, color="#0072B2", linewidth=cfg.line_width, label=envelope.method)
        ax.set_title(f"Temporal envelope - {source_id}", fontsize=cfg.title_font_size)
        ax.set_xlabel(f"Time ({cfg.time_unit})")
        ax.set_ylabel(ylabel)
        if cfg.show_thresholds and amplitudes.size and np.nanmax(amplitudes) > 0:
            ax.axhline(float(np.nanmax(amplitudes) / math.e), color="#E69F00", linestyle=":", linewidth=cfg.line_width, label="1/e amplitude level")
        _apply_axes_style(ax, cfg)
        return _created_result(
            ScientificFigureType.TEMPORAL_ENVELOPE,
            fig,
            (ax,),
            cfg,
            source,
            context,
            tuple(("envelope_source_not_recalculated", *diagnostics, *decimation)),
            save=save,
        )
    except Exception as exc:
        return _failed_result(ScientificFigureType.TEMPORAL_ENVELOPE, cfg, source, exc)


def plot_decay_estimate(
    source: TemporalResults | ExperimentRecordingAnalysisResult | Any,
    settings: ScientificVisualizationSettings | None = None,
    *,
    save: bool = False,
) -> ScientificFigureResult:
    """Plot already available decay information without fitting a new model."""

    cfg = settings or ScientificVisualizationSettings()
    envelope, source_id, context = _extract_envelope(source)
    decay_fit = _extract_decay_fit(source)
    parameter = source if hasattr(source, "decay_estimate") else None
    if envelope is None and parameter is None:
        return _insufficient_result(ScientificFigureType.DECAY_ESTIMATE, cfg, source, ScientificVisualizationReason.MISSING_PARAMETERS)
    try:
        fig, ax = _new_figure(cfg)
        diagnostics = ["decay_plot_uses_existing_fit_or_modal_parameter_only"]
        if envelope is not None:
            times = _finite_array(envelope.times_s, "time")
            amplitudes = _finite_array(envelope.amplitudes, "amplitude")
            y, ylabel, amp_diag = _amplitude_for_display(amplitudes, envelope.unit, cfg, floor=cfg.spectrum_db_floor)
            x, y_plot, decimation = _decimate_xy(times, y, cfg.maximum_waveform_points, cfg)
            ax.plot(x, y_plot, ".", markersize=cfg.marker_size, color="#0072B2", label="envelope points")
            diagnostics.extend(amp_diag)
            diagnostics.extend(decimation)
            if decay_fit is not None and cfg.show_fit_curves and decay_fit.fit_start_s is not None and decay_fit.fit_end_s is not None:
                fit_times = np.linspace(decay_fit.fit_start_s, decay_fit.fit_end_s, 80)
                if decay_fit.model_name and "decay" in decay_fit.model_name and decay_fit.decay_rate_per_s is not None:
                    fit = math.exp(decay_fit.intercept) * np.exp(-decay_fit.decay_rate_per_s * fit_times)
                    fit_y, _, _ = _amplitude_for_display(fit, envelope.unit, cfg, floor=cfg.spectrum_db_floor)
                    ax.plot(fit_times, fit_y, "--", color="#D55E00", linewidth=cfg.line_width, label="existing decay fit")
                    diagnostics.append("existing_decay_fit_rendered")
            ax.set_ylabel(ylabel)
        if parameter is not None and getattr(parameter, "decay_estimate", None) is not None:
            decay = parameter.decay_estimate
            tau = getattr(decay, "representative_tau_s", None)
            if tau is not None:
                ax.axvline(float(tau), color="#009E73", linestyle=":", linewidth=cfg.line_width, label="representative tau")
                diagnostics.append("modal_tau_rendered")
        ax.set_title(f"Decay estimate - {source_id}", fontsize=cfg.title_font_size)
        ax.set_xlabel(f"Time ({cfg.time_unit})")
        _apply_axes_style(ax, cfg)
        return _created_result(ScientificFigureType.DECAY_ESTIMATE, fig, (ax,), cfg, source, context, tuple(diagnostics), save=save)
    except Exception as exc:
        return _failed_result(ScientificFigureType.DECAY_ESTIMATE, cfg, source, exc)


def plot_global_spectrum(
    source: Spectrum | SpectrumResults | ExperimentRecordingAnalysisResult | ExperimentAnalysisResult,
    settings: ScientificVisualizationSettings | None = None,
    *,
    save: bool = False,
) -> ScientificFigureResult:
    """Plot an existing global spectrum without recalculating FFT."""

    cfg = settings or ScientificVisualizationSettings()
    spectrum, source_id, context = _extract_spectrum(source)
    if spectrum is None:
        return _insufficient_result(ScientificFigureType.GLOBAL_SPECTRUM, cfg, source, ScientificVisualizationReason.MISSING_SPECTRUM)
    try:
        fig, ax = _new_figure(cfg)
        _plot_spectrum_series(ax, spectrum, cfg, label="spectrum")
        ax.set_title(f"Global spectrum - {source_id}", fontsize=cfg.title_font_size)
        _apply_axes_style(ax, cfg)
        diagnostics = ["global_spectrum_from_existing_spectrum_result", "no_fft_recalculated"]
        if _nonuniform(spectrum.frequencies_hz):
            diagnostics.append(ScientificVisualizationReason.NONUNIFORM_FREQUENCY_AXIS.value)
        return _created_result(ScientificFigureType.GLOBAL_SPECTRUM, fig, (ax,), cfg, source, context, tuple(diagnostics), save=save)
    except Exception as exc:
        return _failed_result(ScientificFigureType.GLOBAL_SPECTRUM, cfg, source, exc)


def plot_spectral_peaks(
    source: PeakDetectionResults | ExperimentRecordingAnalysisResult | ExperimentAnalysisResult,
    settings: ScientificVisualizationSettings | None = None,
    *,
    save: bool = False,
) -> ScientificFigureResult:
    """Plot detected spectral peaks on an existing global spectrum."""

    cfg = settings or ScientificVisualizationSettings()
    peak_result, spectrum, source_id, context = _extract_peak_result(source)
    if spectrum is None:
        return _insufficient_result(ScientificFigureType.SPECTRAL_PEAKS, cfg, source, ScientificVisualizationReason.MISSING_SPECTRUM)
    try:
        fig, ax = _new_figure(cfg)
        _plot_spectrum_series(ax, spectrum, cfg, label="spectrum")
        diagnostics = ["spectral_peaks_from_existing_peak_result", "no_peak_detection_recomputed"]
        if peak_result is None or not peak_result.peaks:
            diagnostics.append(ScientificVisualizationReason.MISSING_OPTIONAL_LAYER.value)
        else:
            _plot_peaks(ax, spectrum, peak_result.peaks, cfg)
            diagnostics.append("detected_peaks_rendered")
        ax.set_title(f"Spectral peaks - {source_id}", fontsize=cfg.title_font_size)
        _apply_axes_style(ax, cfg)
        reservations = (ScientificVisualizationReason.MISSING_OPTIONAL_LAYER,) if peak_result is None or not peak_result.peaks else ()
        return _created_result(ScientificFigureType.SPECTRAL_PEAKS, fig, (ax,), cfg, source, context, tuple(diagnostics), reservations=reservations, save=save)
    except Exception as exc:
        return _failed_result(ScientificFigureType.SPECTRAL_PEAKS, cfg, source, exc)


def plot_spectrogram(
    source: TimeFrequencySpectrum | STFTResults | ExperimentRecordingAnalysisResult | ExperimentAnalysisResult,
    settings: ScientificVisualizationSettings | None = None,
    *,
    save: bool = False,
) -> ScientificFigureResult:
    """Plot an existing STFT spectrogram without recalculating STFT."""

    cfg = settings or ScientificVisualizationSettings()
    tf, source_id, context = _extract_time_frequency(source)
    if tf is None:
        return _insufficient_result(ScientificFigureType.SPECTROGRAM, cfg, source, ScientificVisualizationReason.MISSING_STFT)
    try:
        times = _finite_array(tf.times_s, "time")
        freqs = _finite_array(tf.frequencies_hz, "frequency")
        values = np.asarray(tf.values, dtype=float)
        if values.shape != (freqs.size, times.size) or values.size == 0:
            return _invalid_result(ScientificFigureType.SPECTROGRAM, cfg, source, ScientificVisualizationReason.INVALID_NUMERIC_VALUES)
        if not np.all(np.isfinite(values)):
            return _invalid_result(ScientificFigureType.SPECTROGRAM, cfg, source, ScientificVisualizationReason.INVALID_NUMERIC_VALUES)
        if cfg.frequency_scale == "log":
            keep = freqs >= cfg.log_frequency_min_hz
            if not np.any(keep):
                return _insufficient_result(ScientificFigureType.SPECTROGRAM, cfg, source, ScientificVisualizationReason.NO_VALID_VALUES)
            values = values[keep, :]
            freqs = freqs[keep]
        display, unit, diagnostics = _matrix_for_display(values, tf.magnitude_unit, cfg)
        fig, ax = _new_figure(cfg)
        extent = (float(times[0]), float(times[-1]), float(freqs[0]), float(freqs[-1]))
        image = ax.imshow(display, origin="lower", aspect="auto", extent=extent, interpolation="nearest")
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label(unit)
        ax.set_xlabel(f"Time ({cfg.time_unit})")
        ax.set_ylabel(f"Frequency ({cfg.frequency_unit})")
        ax.set_title(f"Spectrogram - {source_id}", fontsize=cfg.title_font_size)
        if cfg.frequency_scale == "log":
            ax.set_yscale("log")
            diagnostics.append(ScientificVisualizationReason.LOG_SCALE_CLIPPED.value)
        _overlay_tracks_on_axis(ax, _extract_tracking(source)[0], cfg)
        _apply_axes_style(ax, cfg)
        return _created_result(ScientificFigureType.SPECTROGRAM, fig, (ax,), cfg, source, context, tuple(("spectrogram_from_existing_stft", "no_stft_recalculated", *diagnostics)), save=save)
    except Exception as exc:
        return _failed_result(ScientificFigureType.SPECTROGRAM, cfg, source, exc)


def plot_frequency_tracks(
    source: SpectralTrackingResults | ExperimentRecordingAnalysisResult | ExperimentAnalysisResult,
    settings: ScientificVisualizationSettings | None = None,
    *,
    save: bool = False,
) -> ScientificFigureResult:
    """Plot existing spectral tracks while preserving gaps."""

    cfg = settings or ScientificVisualizationSettings()
    tracking, source_id, context = _extract_tracking(source)
    if tracking is None or not tracking.tracks:
        return _insufficient_result(ScientificFigureType.FREQUENCY_TRACKS, cfg, source, ScientificVisualizationReason.MISSING_TRACKS)
    try:
        fig, ax = _new_figure(cfg)
        diagnostics = ["frequency_tracks_from_existing_tracking_result", "track_gaps_not_connected"]
        for track in sorted(tracking.tracks, key=lambda item: item.track_id):
            _plot_track(ax, track, cfg)
        ax.set_title(f"Frequency tracks - {source_id}", fontsize=cfg.title_font_size)
        ax.set_xlabel(f"Time ({cfg.time_unit})")
        ax.set_ylabel(f"Frequency ({cfg.frequency_unit})")
        _apply_axes_style(ax, cfg)
        if tracking.ambiguous_assignment_count:
            diagnostics.append("ambiguous_assignments_present")
        return _created_result(ScientificFigureType.FREQUENCY_TRACKS, fig, (ax,), cfg, source, context, tuple(diagnostics), save=save)
    except Exception as exc:
        return _failed_result(ScientificFigureType.FREQUENCY_TRACKS, cfg, source, exc)


def plot_modal_candidates(
    source: ExperimentAnalysisResult | ExperimentConditionAnalysisResult,
    settings: ScientificVisualizationSettings | None = None,
    *,
    save: bool = False,
) -> ScientificFigureResult:
    """Plot operational modal candidates by dynamic condition."""

    return _plot_rows_by_condition(
        source,
        settings,
        ScientificFigureType.MODAL_CANDIDATES,
        "candidates",
        "representative_frequency_hz",
        "Candidate representative frequency",
        ScientificVisualizationReason.MISSING_CANDIDATES,
        save=save,
    )


def plot_within_condition_associations(
    source: ExperimentAnalysisResult | ExperimentConditionAnalysisResult | Any,
    settings: ScientificVisualizationSettings | None = None,
    *,
    save: bool = False,
) -> ScientificFigureResult:
    """Plot within-condition association clusters or unmatched candidates."""

    cfg = settings or ScientificVisualizationSettings()
    rows, context = _rows_from_source(source, "within_condition_associations")
    if not rows:
        return _insufficient_result(ScientificFigureType.WITHIN_CONDITION_ASSOCIATIONS, cfg, source, ScientificVisualizationReason.MISSING_CANDIDATES)
    try:
        fig, ax = _new_figure(cfg)
        y_values = tuple(_float(row.get("representative_frequency_hz")) for row in rows)
        x_values = tuple(range(len(rows)))
        colors = tuple(_color_for(str(row.get("dynamic_label") or row.get("recording_id") or index), cfg) for index, row in enumerate(rows))
        ax.scatter(x_values, y_values, c=colors, s=cfg.marker_size ** 2, marker="o")
        _annotate_rows(ax, x_values, y_values, rows, ("cluster_id", "candidate_id"), cfg)
        ax.set_title("Within-condition candidate associations", fontsize=cfg.title_font_size)
        ax.set_xlabel("Association row")
        ax.set_ylabel(f"Frequency ({cfg.frequency_unit})")
        _apply_axes_style(ax, cfg)
        return _created_result(ScientificFigureType.WITHIN_CONDITION_ASSOCIATIONS, fig, (ax,), cfg, source, context, ("within_condition_associations_rendered",), save=save)
    except Exception as exc:
        return _failed_result(ScientificFigureType.WITHIN_CONDITION_ASSOCIATIONS, cfg, source, exc)


def plot_cross_condition_associations(
    source: ExperimentAnalysisResult | ExperimentCrossConditionAnalysisResult | Any,
    settings: ScientificVisualizationSettings | None = None,
    *,
    save: bool = False,
) -> ScientificFigureResult:
    """Plot accepted/rejected adjacent cross-condition candidate matches."""

    cfg = settings or ScientificVisualizationSettings()
    rows, context = _rows_from_source(source, "cross_condition_matches")
    if not rows:
        return _insufficient_result(ScientificFigureType.CROSS_CONDITION_ASSOCIATIONS, cfg, source, ScientificVisualizationReason.MISSING_CANDIDATES)
    try:
        fig, ax = _new_figure(cfg)
        diagnostics = ["cross_condition_associations_are_adjacent_only", "no_non_adjacent_edges_drawn"]
        labels = _canonical_labels_from_rows(rows, ("lower_dynamic_label", "higher_dynamic_label"))
        positions = {label: index for index, label in enumerate(labels)}
        for row in rows:
            left_label = str(row.get("lower_dynamic_label"))
            right_label = str(row.get("higher_dynamic_label"))
            if abs(_condition_index(right_label) - _condition_index(left_label)) != 1:
                diagnostics.append("non_adjacent_match_row_not_connected")
                continue
            y0 = _float(row.get("frequency_change_hz")) or 0.0
            cost = _float(row.get("association_cost"))
            alpha = 0.9 if bool(row.get("accepted")) else 0.35
            ax.plot(
                (positions[left_label], positions[right_label]),
                (0.0, y0),
                color=_color_for(left_label, cfg),
                linestyle="-" if bool(row.get("accepted")) else ":",
                linewidth=cfg.line_width,
                alpha=alpha,
            )
            if cfg.show_ids and cost is not None:
                ax.text(positions[right_label], y0, f"{cost:.2g}", fontsize=cfg.annotation_font_size)
        ax.set_xticks(tuple(positions.values()), tuple(positions))
        ax.set_title("Adjacent cross-condition associations", fontsize=cfg.title_font_size)
        ax.set_ylabel(f"Frequency change ({cfg.frequency_unit})")
        _apply_axes_style(ax, cfg)
        return _created_result(ScientificFigureType.CROSS_CONDITION_ASSOCIATIONS, fig, (ax,), cfg, source, context, tuple(diagnostics), save=save)
    except Exception as exc:
        return _failed_result(ScientificFigureType.CROSS_CONDITION_ASSOCIATIONS, cfg, source, exc)


def plot_candidate_chains(
    source: ExperimentAnalysisResult | Any,
    settings: ScientificVisualizationSettings | None = None,
    *,
    save: bool = False,
) -> ScientificFigureResult:
    """Plot candidate chains over ordinal dynamic condition positions."""

    cfg = settings or ScientificVisualizationSettings()
    rows, context = _rows_from_source(source, "candidate_chain_nodes")
    if not rows:
        return _insufficient_result(ScientificFigureType.CANDIDATE_CHAINS, cfg, source, ScientificVisualizationReason.MISSING_CANDIDATES)
    try:
        fig, ax = _new_figure(cfg)
        diagnostics = ["candidate_chains_are_operational_not_physical_modes", "gaps_are_not_connected"]
        by_chain: dict[str, list[Mapping[str, object]]] = {}
        for row in rows:
            by_chain.setdefault(str(row.get("chain_id")), []).append(row)
        for chain_id, chain_rows in sorted(by_chain.items()):
            ordered = sorted(chain_rows, key=lambda row: _condition_index(str(row.get("dynamic_label"))))
            x = [_condition_index(str(row.get("dynamic_label"))) for row in ordered]
            y = [_float(row.get("representative_frequency_hz")) for row in ordered]
            color = _color_for(chain_id, cfg)
            _plot_without_non_adjacent_gaps(ax, x, y, color, cfg, label=chain_id if cfg.show_ids else None)
        ax.set_xticks(tuple(range(len(DYNAMIC_LABEL_ORDER))), DYNAMIC_LABEL_ORDER)
        ax.set_title("Candidate chains (operational)", fontsize=cfg.title_font_size)
        ax.set_xlabel("Dynamic condition (ordinal graphic position)")
        ax.set_ylabel(f"Frequency ({cfg.frequency_unit})")
        _apply_axes_style(ax, cfg)
        return _created_result(ScientificFigureType.CANDIDATE_CHAINS, fig, (ax,), cfg, source, context, tuple(diagnostics), save=save)
    except Exception as exc:
        return _failed_result(ScientificFigureType.CANDIDATE_CHAINS, cfg, source, exc)


def plot_modal_hypotheses(
    source: ExperimentAnalysisResult | Any,
    settings: ScientificVisualizationSettings | None = None,
    *,
    save: bool = False,
) -> ScientificFigureResult:
    """Plot modal hypotheses explicitly as operational hypotheses."""

    cfg = settings or ScientificVisualizationSettings()
    rows, context = _rows_from_source(source, "modal_hypotheses")
    if not rows:
        return _insufficient_result(ScientificFigureType.MODAL_HYPOTHESES, cfg, source, ScientificVisualizationReason.MISSING_HYPOTHESES)
    try:
        fig, ax = _new_figure(cfg)
        x = np.arange(len(rows))
        scores = np.asarray([_float(row.get("score")) if _float(row.get("score")) is not None else 0.0 for row in rows], dtype=float)
        for index, row in enumerate(rows):
            status = str(row.get("status") or "")
            if not _visible_status(status, cfg):
                continue
            style = _style_for(status, cfg)
            ax.scatter(index, scores[index], marker=style.marker, s=cfg.marker_size ** 2, facecolors=style.facecolor, edgecolors=style.edgecolor, alpha=style.alpha)
        ax.set_title("Modal hypotheses (operational, not physical modes)", fontsize=cfg.title_font_size)
        ax.set_xlabel("Hypothesis")
        ax.set_ylabel("Normalized score")
        _annotate_rows(ax, x, scores, rows, ("hypothesis_id",), cfg)
        _apply_axes_style(ax, cfg)
        reservations = tuple(_reservation_reasons_from_rows(rows))
        return _created_result(ScientificFigureType.MODAL_HYPOTHESES, fig, (ax,), cfg, source, context, ("modal_hypotheses_rendered_without_physical_promotion",), reservations=reservations, save=save)
    except Exception as exc:
        return _failed_result(ScientificFigureType.MODAL_HYPOTHESES, cfg, source, exc)


def plot_modal_frequency_trajectories(
    source: ExperimentAnalysisResult | Any,
    settings: ScientificVisualizationSettings | None = None,
    *,
    save: bool = False,
) -> ScientificFigureResult:
    """Plot modal frequency summaries versus ordinal condition positions."""

    cfg = settings or ScientificVisualizationSettings()
    rows, context = _rows_from_source(source, "modal_parameters")
    if not rows:
        return _insufficient_result(ScientificFigureType.MODAL_FREQUENCY_TRAJECTORIES, cfg, source, ScientificVisualizationReason.MISSING_PARAMETERS)
    try:
        fig, ax = _new_figure(cfg)
        x = np.arange(len(rows))
        y = [_float(row.get("representative_frequency_hz")) for row in rows]
        err = [_float(row.get("frequency_uncertainty_hz")) for row in rows]
        for index, row in enumerate(rows):
            status = str(row.get("status") or "")
            style = _style_for(status, cfg)
            if cfg.show_uncertainty and err[index] is not None:
                ax.errorbar(index, y[index], yerr=err[index], fmt=style.marker, color=style.edgecolor, capsize=cfg.errorbar_capsize)
            else:
                ax.scatter(index, y[index], marker=style.marker, s=cfg.marker_size ** 2, facecolors=style.facecolor, edgecolors=style.edgecolor)
        ax.set_title("Modal frequency trajectories (descriptive)", fontsize=cfg.title_font_size)
        ax.set_xlabel("Parameter estimate")
        ax.set_ylabel(f"Representative frequency ({cfg.frequency_unit})")
        _annotate_rows(ax, x, y, rows, ("hypothesis_id",), cfg)
        _apply_axes_style(ax, cfg)
        return _created_result(ScientificFigureType.MODAL_FREQUENCY_TRAJECTORIES, fig, (ax,), cfg, source, context, ("frequency_trajectory_not_hardening_or_softening_proof",), save=save)
    except Exception as exc:
        return _failed_result(ScientificFigureType.MODAL_FREQUENCY_TRAJECTORIES, cfg, source, exc)


def plot_modal_parameters(
    source: ExperimentAnalysisResult | Any,
    settings: ScientificVisualizationSettings | None = None,
    *,
    save: bool = False,
) -> ScientificFigureResult:
    """Plot representative frequency, tau and decay rate from existing estimates."""

    cfg = settings or ScientificVisualizationSettings()
    rows, context = _rows_from_source(source, "modal_parameters")
    if not rows:
        return _insufficient_result(ScientificFigureType.MODAL_PARAMETERS, cfg, source, ScientificVisualizationReason.MISSING_PARAMETERS)
    try:
        fig, axes = _new_subplots(cfg, 3, 1)
        x = np.arange(len(rows))
        panels = (
            (axes[0], "representative_frequency_hz", f"Frequency ({cfg.frequency_unit})"),
            (axes[1], "representative_tau_s", "Tau (s)"),
            (axes[2], "amplitude_decay_rate_per_s", "Amplitude decay rate (1/s)"),
        )
        for ax, key, label in panels:
            values = [_float(row.get(key)) for row in rows]
            ax.plot(x, values, marker="o", linestyle="", color="#0072B2")
            ax.set_ylabel(label)
            _apply_axes_style(ax, cfg)
        axes[0].set_title("Modal parameters (operational estimates)", fontsize=cfg.title_font_size)
        axes[-1].set_xlabel("Parameter estimate")
        return _created_result(ScientificFigureType.MODAL_PARAMETERS, fig, tuple(axes), cfg, source, context, ("modal_parameters_rendered_without_combining_absences",), save=save)
    except Exception as exc:
        return _failed_result(ScientificFigureType.MODAL_PARAMETERS, cfg, source, exc)


def plot_modal_q_factors(
    source: ExperimentAnalysisResult | Any,
    settings: ScientificVisualizationSettings | None = None,
    *,
    save: bool = False,
) -> ScientificFigureResult:
    """Plot Q estimates by decay, bandwidth and representative policy."""

    cfg = settings or ScientificVisualizationSettings()
    rows, context = _rows_from_source(source, "modal_q_factors")
    if not rows:
        return _insufficient_result(ScientificFigureType.MODAL_Q_FACTORS, cfg, source, ScientificVisualizationReason.MISSING_Q_ESTIMATES)
    try:
        fig, ax = _new_figure(cfg)
        x = np.arange(len(rows))
        series = (
            ("q_decay", "Q decay", "o"),
            ("q_bandwidth", "Q bandwidth", "s"),
            ("representative_q", "Q representative", "^"),
        )
        for key, label, marker in series:
            y = [_float(row.get(key)) for row in rows]
            ax.plot(x, y, marker=marker, linestyle="", markersize=cfg.marker_size, label=label)
        ax.set_title("Q and bandwidth estimates (operational)", fontsize=cfg.title_font_size)
        ax.set_xlabel("Q estimate")
        ax.set_ylabel("Q")
        _apply_axes_style(ax, cfg)
        reservations = (ScientificVisualizationReason.RESOLUTION_LIMITED,) if any(row.get("resolution_ratio") is not None and (_float(row.get("resolution_ratio")) or 0) <= 2 for row in rows) else ()
        return _created_result(ScientificFigureType.MODAL_Q_FACTORS, fig, (ax,), cfg, source, context, ("q_methods_rendered_without_hiding_disagreement",), reservations=reservations, save=save)
    except Exception as exc:
        return _failed_result(ScientificFigureType.MODAL_Q_FACTORS, cfg, source, exc)


def plot_modal_bandwidth(
    source: ExperimentAnalysisResult | Any,
    settings: ScientificVisualizationSettings | None = None,
    *,
    save: bool = False,
) -> ScientificFigureResult:
    """Plot existing bandwidth estimates as center and crossing markers."""

    cfg = settings or ScientificVisualizationSettings()
    rows, context = _bandwidth_rows(source)
    if not rows:
        return _insufficient_result(ScientificFigureType.MODAL_BANDWIDTH, cfg, source, ScientificVisualizationReason.MISSING_Q_ESTIMATES)
    try:
        fig, ax = _new_figure(cfg)
        diagnostics = ["bandwidth_plot_uses_existing_bandwidth_estimates"]
        for index, row in enumerate(rows):
            center = _float(row.get("center_frequency_hz") or row.get("representative_frequency_hz"))
            lower = _float(row.get("lower_frequency_hz"))
            upper = _float(row.get("upper_frequency_hz"))
            bandwidth = _float(row.get("bandwidth_hz"))
            if lower is not None and upper is not None:
                ax.hlines(index, lower, upper, color="#0072B2", linewidth=2 * cfg.line_width, label="bandwidth" if index == 0 else None)
            if center is not None:
                ax.plot(center, index, marker="o", color="#D55E00", label="center" if index == 0 else None)
            elif bandwidth is not None:
                ax.plot(bandwidth, index, marker="s", color="#009E73", label="bandwidth value" if index == 0 else None)
        ax.set_title("Modal bandwidth estimates", fontsize=cfg.title_font_size)
        ax.set_xlabel(f"Frequency or bandwidth ({cfg.frequency_unit})")
        ax.set_ylabel("Estimate")
        _apply_axes_style(ax, cfg)
        return _created_result(ScientificFigureType.MODAL_BANDWIDTH, fig, (ax,), cfg, source, context, tuple(diagnostics), save=save)
    except Exception as exc:
        return _failed_result(ScientificFigureType.MODAL_BANDWIDTH, cfg, source, exc)


def plot_dynamic_condition_comparison(
    source: ExperimentAnalysisResult,
    settings: ScientificVisualizationSettings | None = None,
    *,
    save: bool = False,
) -> ScientificFigureResult:
    """Plot condition-level counts and frequencies on ordinal graphic positions."""

    cfg = settings or ScientificVisualizationSettings()
    rows, context = _rows_from_source(source, "conditions")
    candidate_rows, _ = _rows_from_source(source, "candidates")
    if not rows and not candidate_rows:
        return _insufficient_result(ScientificFigureType.DYNAMIC_CONDITION_COMPARISON, cfg, source, ScientificVisualizationReason.MISSING_CANDIDATES)
    try:
        fig, ax = _new_figure(cfg)
        diagnostics = ["dynamic_labels_plotted_as_ordinal_graphic_positions", "no_nonlinearity_inferred"]
        if candidate_rows:
            for row in candidate_rows:
                label = str(row.get("dynamic_label"))
                y = _float(row.get("representative_frequency_hz"))
                if y is not None:
                    ax.scatter(_condition_index(label), y, color=_color_for(label, cfg), marker=_marker_for(label, cfg), s=cfg.marker_size ** 2)
            ax.set_ylabel(f"Candidate frequency ({cfg.frequency_unit})")
        else:
            labels = [str(row.get("dynamic_label")) for row in rows]
            counts = [_float(row.get("candidate_reference_count")) or 0.0 for row in rows]
            ax.bar([_condition_index(label) for label in labels], counts, color=[_color_for(label, cfg) for label in labels])
            ax.set_ylabel("Candidate references")
        ax.set_xticks(tuple(range(len(DYNAMIC_LABEL_ORDER))), DYNAMIC_LABEL_ORDER)
        ax.set_xlabel("Dynamic condition (ordinal graphic position)")
        ax.set_title("Dynamic condition comparison", fontsize=cfg.title_font_size)
        _apply_axes_style(ax, cfg)
        return _created_result(ScientificFigureType.DYNAMIC_CONDITION_COMPARISON, fig, (ax,), cfg, source, context, tuple(diagnostics), save=save)
    except Exception as exc:
        return _failed_result(ScientificFigureType.DYNAMIC_CONDITION_COMPARISON, cfg, source, exc)


def plot_modal_energy_exchange_evidence(
    source: ExperimentAnalysisResult | Any,
    settings: ScientificVisualizationSettings | None = None,
    *,
    save: bool = False,
) -> ScientificFigureResult:
    """Plot operational evidence compatible with possible energy redistribution."""

    cfg = settings or ScientificVisualizationSettings()
    evidence = _first_energy_evidence(source)
    context = _source_context(source, ScientificFigureType.MODAL_ENERGY_EXCHANGE_EVIDENCE, cfg)
    if evidence is None:
        return _insufficient_result(ScientificFigureType.MODAL_ENERGY_EXCHANGE_EVIDENCE, cfg, source, ScientificVisualizationReason.MISSING_ENERGY_EXCHANGE)
    try:
        fig, axes = _new_subplots(cfg, 2, 1)
        alignment = evidence.alignment
        times = np.asarray(alignment.common_times_s, dtype=float)
        axes[0].plot(times, alignment.aligned_amplitudes_a, color="#0072B2", linewidth=cfg.line_width, label=evidence.source_a_id)
        axes[0].plot(times, alignment.aligned_amplitudes_b, color="#D55E00", linewidth=cfg.line_width, label=evidence.source_b_id)
        axes[0].set_ylabel("Envelope amplitude")
        axes[0].set_title("Evidencia operacional de possivel redistribuicao", fontsize=cfg.title_font_size)
        pair_energy = evidence.pair_energy_evidence
        if pair_energy.pair_energy:
            axes[1].plot(times[:len(pair_energy.pair_energy)], pair_energy.pair_energy, color="#009E73", linewidth=cfg.line_width, label="pair energy proxy")
        axes[1].set_ylabel("Operational energy proxy")
        axes[1].set_xlabel(f"Time ({cfg.time_unit})")
        for ax in axes:
            _apply_axes_style(ax, cfg)
        reservations: list[ScientificVisualizationReason] = []
        if evidence.beating_evidence.possible_beating:
            reservations.append(ScientificVisualizationReason.POSSIBLE_BEATING_CONTEXT)
        return _created_result(ScientificFigureType.MODAL_ENERGY_EXCHANGE_EVIDENCE, fig, tuple(axes), cfg, source, context, ("energy_evidence_rendered_without_causality",), reservations=tuple(reservations), save=save)
    except Exception as exc:
        return _failed_result(ScientificFigureType.MODAL_ENERGY_EXCHANGE_EVIDENCE, cfg, source, exc)


def plot_modal_energy_exchange_correlation(
    source: ExperimentAnalysisResult | Any,
    settings: ScientificVisualizationSettings | None = None,
    *,
    save: bool = False,
) -> ScientificFigureResult:
    """Plot correlation versus lag without interpreting lag as causality."""

    cfg = settings or ScientificVisualizationSettings()
    evidence = _first_energy_evidence(source)
    context = _source_context(source, ScientificFigureType.MODAL_ENERGY_EXCHANGE_CORRELATION, cfg)
    if evidence is None:
        return _insufficient_result(ScientificFigureType.MODAL_ENERGY_EXCHANGE_CORRELATION, cfg, source, ScientificVisualizationReason.MISSING_ENERGY_EXCHANGE)
    try:
        corr = evidence.correlation_evidence
        if not corr.lag_values_s:
            return _insufficient_result(ScientificFigureType.MODAL_ENERGY_EXCHANGE_CORRELATION, cfg, source, ScientificVisualizationReason.INSUFFICIENT_POINTS)
        fig, ax = _new_figure(cfg)
        lags = np.asarray(corr.lag_values_s, dtype=float)
        values = np.asarray([np.nan if value is None else value for value in corr.lagged_correlations], dtype=float)
        ax.plot(lags, values, color="#0072B2", linewidth=cfg.line_width, marker="o", markersize=cfg.marker_size)
        if corr.best_negative_lag_s is not None:
            ax.axvline(corr.best_negative_lag_s, color="#D55E00", linestyle=":", linewidth=cfg.line_width, label="best negative lag")
        ax.axhline(0.0, color="#666666", linewidth=0.8)
        ax.set_title("Envelope correlation versus lag (operational)", fontsize=cfg.title_font_size)
        ax.set_xlabel("Lag (s); positive means A precedes B, not causal direction")
        ax.set_ylabel("Correlation")
        _apply_axes_style(ax, cfg)
        return _created_result(ScientificFigureType.MODAL_ENERGY_EXCHANGE_CORRELATION, fig, (ax,), cfg, source, context, ("lag_rendered_without_causal_interpretation",), save=save)
    except Exception as exc:
        return _failed_result(ScientificFigureType.MODAL_ENERGY_EXCHANGE_CORRELATION, cfg, source, exc)


def plot_synthetic_validation_result(
    source: SyntheticScenarioValidationResult,
    settings: ScientificVisualizationSettings | None = None,
    *,
    save: bool = False,
) -> ScientificFigureResult:
    """Plot true-versus-estimated synthetic validation metrics."""

    cfg = settings or ScientificVisualizationSettings()
    if not isinstance(source, SyntheticScenarioValidationResult):
        return _invalid_result(ScientificFigureType.SYNTHETIC_VALIDATION_RESULT, cfg, source, ScientificVisualizationReason.INVALID_NUMERIC_VALUES)
    try:
        fig, axes = _new_subplots(cfg, 2, 2)
        context = _source_context(source, ScientificFigureType.SYNTHETIC_VALIDATION_RESULT, cfg)
        _scatter_true_estimated(axes[0], source.frequency_validations, "true_frequency_hz", "estimated_frequency_hz", "Frequency (Hz)")
        _scatter_true_estimated(axes[1], source.decay_validations, "true_tau_s", "estimated_tau_s", "Tau (s)")
        _scatter_true_estimated(axes[2], source.q_validations, "true_q", "representative_q", "Q")
        _scatter_true_estimated(axes[3], source.bandwidth_validations, "true_bandwidth_hz", "estimated_bandwidth_hz", "Bandwidth (Hz)")
        axes[0].set_title(f"Synthetic validation: {source.scenario.scenario_id}", fontsize=cfg.title_font_size)
        for ax in axes:
            _apply_axes_style(ax, cfg)
        return _created_result(ScientificFigureType.SYNTHETIC_VALIDATION_RESULT, fig, tuple(axes), cfg, source, context, ("synthetic_truth_not_used_by_estimators", "synthetic_success_not_real_data_proof"), save=save)
    except Exception as exc:
        return _failed_result(ScientificFigureType.SYNTHETIC_VALIDATION_RESULT, cfg, source, exc)


def plot_synthetic_validation_campaign(
    source: SyntheticValidationCampaignResult | SyntheticMonteCarloValidation,
    settings: ScientificVisualizationSettings | None = None,
    *,
    save: bool = False,
) -> ScientificFigureResult:
    """Plot synthetic campaign or Monte Carlo outcomes without physical generalization."""

    cfg = settings or ScientificVisualizationSettings()
    if not isinstance(source, (SyntheticValidationCampaignResult, SyntheticMonteCarloValidation)):
        return _invalid_result(ScientificFigureType.SYNTHETIC_VALIDATION_CAMPAIGN, cfg, source, ScientificVisualizationReason.INVALID_NUMERIC_VALUES)
    try:
        fig, axes = _new_subplots(cfg, 1, 2)
        context = _source_context(source, ScientificFigureType.SYNTHETIC_VALIDATION_CAMPAIGN, cfg)
        if isinstance(source, SyntheticValidationCampaignResult):
            labels = ("passed", "reservations", "failed", "inconclusive", "insufficient", "pipeline_error")
            counts = (source.passed_count, source.passed_with_reservations_count, source.failed_count, source.inconclusive_count, source.insufficient_evidence_count, source.pipeline_error_count)
            axes[0].bar(labels, counts, color="#0072B2")
            axes[0].tick_params(axis="x", rotation=45)
            scenario_ids = [item.scenario.scenario_id for item in source.scenario_results]
            status_values = [_status_numeric(item.status.value) for item in source.scenario_results]
            axes[1].plot(range(len(scenario_ids)), status_values, marker="o", linestyle="", color="#D55E00")
            axes[1].set_xlabel("Scenario")
            axes[1].set_ylabel("Status code")
        else:
            axes[0].bar(("pass", "reservation", "failure"), (source.pass_count, source.reservation_count, source.failure_count), color="#0072B2")
            axes[1].plot(source.seeds, [_status_numeric(item.status.value) for item in source.trial_results], marker="o", linestyle="", color="#D55E00")
            axes[1].set_xlabel("Seed")
            axes[1].set_ylabel("Status code")
        axes[0].set_title("Synthetic validation campaign", fontsize=cfg.title_font_size)
        for ax in axes:
            _apply_axes_style(ax, cfg)
        return _created_result(ScientificFigureType.SYNTHETIC_VALIDATION_CAMPAIGN, fig, tuple(axes), cfg, source, context, ("synthetic_campaign_not_universal_real_data_validation",), save=save)
    except Exception as exc:
        return _failed_result(ScientificFigureType.SYNTHETIC_VALIDATION_CAMPAIGN, cfg, source, exc)


def create_experiment_visualizations(
    result: ExperimentAnalysisResult,
    settings: ScientificVisualizationSettings | None = None,
) -> ScientificFigureCollection:
    """Create and save requested experiment visualizations from existing results."""

    cfg = settings or ScientificVisualizationSettings()
    if not isinstance(result, ExperimentAnalysisResult):
        figure = _invalid_result(ScientificFigureType.EXPERIMENT_SUMMARY, cfg, result, ScientificVisualizationReason.INVALID_NUMERIC_VALUES)
        return _collection(result, cfg, (figure,))
    figures: list[ScientificFigureResult] = []
    requested = tuple(cfg.figure_types)
    for figure_type in requested:
        if figure_type is ScientificFigureType.WAVEFORM:
            figures.extend(plot_waveform(item, cfg, save=True) for item in _recordings_with_signal(result))
            if not any(item.figure_type is figure_type for item in figures):
                figures.append(_insufficient_result(figure_type, cfg, result, ScientificVisualizationReason.MISSING_TIME_SERIES))
        elif figure_type is ScientificFigureType.TEMPORAL_ENVELOPE:
            figures.extend(plot_temporal_envelope(item, cfg, save=True) for item in _recordings_with_envelope(result))
            if not any(item.figure_type is figure_type for item in figures):
                figures.append(_insufficient_result(figure_type, cfg, result, ScientificVisualizationReason.MISSING_TIME_SERIES))
        elif figure_type is ScientificFigureType.DECAY_ESTIMATE:
            figures.extend(plot_decay_estimate(item, cfg, save=True) for item in _recordings_with_envelope(result))
            if not any(item.figure_type is figure_type for item in figures):
                figures.append(_insufficient_result(figure_type, cfg, result, ScientificVisualizationReason.MISSING_PARAMETERS))
        elif figure_type is ScientificFigureType.GLOBAL_SPECTRUM:
            figures.extend(plot_global_spectrum(item, cfg, save=True) for item in _recordings_with_spectrum(result))
            if not any(item.figure_type is figure_type for item in figures):
                figures.append(_insufficient_result(figure_type, cfg, result, ScientificVisualizationReason.MISSING_SPECTRUM))
        elif figure_type is ScientificFigureType.SPECTRAL_PEAKS:
            figures.extend(plot_spectral_peaks(item, cfg, save=True) for item in _recordings_with_spectrum(result))
            if not any(item.figure_type is figure_type for item in figures):
                figures.append(_insufficient_result(figure_type, cfg, result, ScientificVisualizationReason.MISSING_SPECTRUM))
        elif figure_type is ScientificFigureType.SPECTROGRAM:
            figures.extend(plot_spectrogram(item, cfg, save=True) for item in _recordings_with_stft(result))
            if not any(item.figure_type is figure_type for item in figures):
                figures.append(_insufficient_result(figure_type, cfg, result, ScientificVisualizationReason.MISSING_STFT))
        elif figure_type is ScientificFigureType.FREQUENCY_TRACKS:
            figures.extend(plot_frequency_tracks(item, cfg, save=True) for item in _recordings_with_tracks(result))
            if not any(item.figure_type is figure_type for item in figures):
                figures.append(_insufficient_result(figure_type, cfg, result, ScientificVisualizationReason.MISSING_TRACKS))
        elif figure_type is ScientificFigureType.MODAL_CANDIDATES:
            figures.append(plot_modal_candidates(result, cfg, save=True))
        elif figure_type is ScientificFigureType.WITHIN_CONDITION_ASSOCIATIONS:
            figures.append(plot_within_condition_associations(result, cfg, save=True))
        elif figure_type is ScientificFigureType.CROSS_CONDITION_ASSOCIATIONS:
            figures.append(plot_cross_condition_associations(result, cfg, save=True))
        elif figure_type is ScientificFigureType.CANDIDATE_CHAINS:
            figures.append(plot_candidate_chains(result, cfg, save=True))
        elif figure_type is ScientificFigureType.MODAL_HYPOTHESES:
            figures.append(plot_modal_hypotheses(result, cfg, save=True))
        elif figure_type is ScientificFigureType.MODAL_FREQUENCY_TRAJECTORIES:
            figures.append(plot_modal_frequency_trajectories(result, cfg, save=True))
        elif figure_type is ScientificFigureType.MODAL_PARAMETERS:
            figures.append(plot_modal_parameters(result, cfg, save=True))
        elif figure_type is ScientificFigureType.MODAL_Q_FACTORS:
            figures.append(plot_modal_q_factors(result, cfg, save=True))
        elif figure_type is ScientificFigureType.MODAL_BANDWIDTH:
            figures.append(plot_modal_bandwidth(result, cfg, save=True))
        elif figure_type is ScientificFigureType.DYNAMIC_CONDITION_COMPARISON:
            figures.append(plot_dynamic_condition_comparison(result, cfg, save=True))
        elif figure_type is ScientificFigureType.MODAL_ENERGY_EXCHANGE_EVIDENCE:
            figures.append(plot_modal_energy_exchange_evidence(result, cfg, save=True))
        elif figure_type is ScientificFigureType.MODAL_ENERGY_EXCHANGE_CORRELATION:
            figures.append(plot_modal_energy_exchange_correlation(result, cfg, save=True))
        elif figure_type is ScientificFigureType.EXPERIMENT_SUMMARY:
            figures.append(_plot_experiment_summary(result, cfg, save=True))
        else:
            figures.append(_skipped_result(figure_type, cfg, result, ScientificVisualizationReason.UNSUPPORTED_PLOT_TYPE))
    return _collection(result, cfg, tuple(figures))


def save_scientific_figure(
    figure_result_or_figure: ScientificFigureResult | object,
    settings: ScientificVisualizationSettings | None = None,
    *,
    figure_id: str | None = None,
    figure_type: ScientificFigureType | str | None = None,
) -> tuple[ScientificFigureArtifact, ...]:
    """Save a figure using deterministic names, atomic writes and checksums."""

    cfg = settings or ScientificVisualizationSettings()
    if isinstance(figure_result_or_figure, ScientificFigureResult):
        figure = figure_result_or_figure.figure
        resolved_id = figure_result_or_figure.figure_id
        resolved_type = figure_result_or_figure.figure_type
    else:
        figure = figure_result_or_figure
        resolved_id = figure_id
        resolved_type = _coerce_enum(figure_type, ScientificFigureType) if figure_type is not None else None
    if figure is None or resolved_id is None or resolved_type is None:
        return (
            _artifact_failure(
                resolved_id or "missing-figure",
                str(resolved_type.value if isinstance(resolved_type, ScientificFigureType) else "unknown"),
                "png",
                cfg,
                ScientificVisualizationReason.INVALID_NUMERIC_VALUES,
                "missing figure, figure_id or figure_type",
            ),
        )
    artifacts = []
    for fmt in cfg.formats:
        artifacts.append(_save_one_format(figure, resolved_id, resolved_type, fmt, cfg))
    return tuple(artifacts)


def summarize_scientific_visualizations(
    result: ScientificFigureResult | ScientificFigureCollection,
) -> dict[str, object]:
    """Return a stable summary for visualization results."""

    if isinstance(result, ScientificFigureCollection):
        return {
            "collection_id": result.collection_id,
            "analysis_id": result.analysis_id,
            "experiment_id": result.experiment_id,
            "status": result.status.value,
            "figure_count": len(result.figures),
            "artifact_count": len(result.artifacts),
            "completed_figure_types": tuple(item.value for item in result.completed_figure_types),
            "skipped_figure_types": tuple(item.value for item in result.skipped_figure_types),
            "failed_figure_types": tuple(item.value for item in result.failed_figure_types),
            "valid": result.valid,
            "diagnostics": result.diagnostics,
        }
    if isinstance(result, ScientificFigureResult):
        return {
            "figure_id": result.figure_id,
            "figure_type": result.figure_type.value,
            "status": result.status.value,
            "artifact_count": len(result.artifacts),
            "source_ids": result.source_ids,
            "valid": result.valid,
            "diagnostics": result.diagnostics,
        }
    raise ValueError("result must be ScientificFigureResult or ScientificFigureCollection.")


def scientific_visualization_settings_fingerprint(
    settings: ScientificVisualizationSettings | None = None,
) -> str:
    """Return a deterministic settings fingerprint independent of output path."""

    cfg = settings or ScientificVisualizationSettings()
    payload = {
        field.name: _canonicalize(getattr(cfg, field.name))
        for field in fields(cfg)
        if field.name != "output_directory"
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def scientific_figure_checksum(path: str | Path) -> str:
    """Return a SHA-256 checksum for a rendered figure artifact."""

    return export_artifact_checksum(path)


def _plot_experiment_summary(
    result: ExperimentAnalysisResult,
    cfg: ScientificVisualizationSettings,
    *,
    save: bool,
) -> ScientificFigureResult:
    rows, context = _rows_from_source(result, "experiment_summary")
    summary = rows[0] if rows else {}
    fig, ax = _new_figure(cfg)
    labels = ("recordings", "conditions", "candidates", "chains", "hypotheses", "parameters", "q", "energy")
    values = (
        _float(summary.get("recording_count")) or 0.0,
        _float(summary.get("condition_count")) or 0.0,
        _float(summary.get("candidate_count")) or 0.0,
        _float(summary.get("chain_count")) or 0.0,
        _float(summary.get("hypothesis_count")) or 0.0,
        _float(summary.get("parameter_estimate_count")) or 0.0,
        _float(summary.get("q_estimate_count")) or 0.0,
        _float(summary.get("energy_pair_count")) or 0.0,
    )
    ax.bar(labels, values, color="#0072B2")
    ax.tick_params(axis="x", rotation=35)
    ax.set_title("Experiment visual summary", fontsize=cfg.title_font_size)
    ax.set_ylabel("Count")
    _apply_axes_style(ax, cfg)
    return _created_result(ScientificFigureType.EXPERIMENT_SUMMARY, fig, (ax,), cfg, result, context, ("summary_visual_preserves_operational_counts",), save=save)


def _plot_rows_by_condition(
    source: object,
    settings: ScientificVisualizationSettings | None,
    figure_type: ScientificFigureType,
    table_name: str,
    value_key: str,
    title: str,
    missing_reason: ScientificVisualizationReason,
    *,
    save: bool,
) -> ScientificFigureResult:
    cfg = settings or ScientificVisualizationSettings()
    rows, context = _rows_from_source(source, table_name)
    rows = tuple(row for row in rows if _visible_status(str(row.get("status") or "valid"), cfg))
    if not rows:
        return _insufficient_result(figure_type, cfg, source, missing_reason)
    try:
        fig, ax = _new_figure(cfg)
        diagnostics = [f"{table_name}_rendered_from_normalized_existing_results"]
        for row in rows:
            label = str(row.get("dynamic_label") or "")
            value = _float(row.get(value_key))
            if value is None:
                continue
            status = str(row.get("status") or ("accepted" if row.get("accepted") else "rejected"))
            style = _style_for(status, cfg)
            ax.scatter(
                _condition_index(label),
                value,
                marker=_marker_for(label, cfg) if cfg.marker_policy == "dynamic_condition" else style.marker,
                s=cfg.marker_size ** 2,
                facecolors=style.facecolor,
                edgecolors=_color_for(label, cfg),
                alpha=style.alpha,
            )
        ax.set_xticks(tuple(range(len(DYNAMIC_LABEL_ORDER))), DYNAMIC_LABEL_ORDER)
        ax.set_title(title, fontsize=cfg.title_font_size)
        ax.set_xlabel("Dynamic condition (ordinal graphic position)")
        ax.set_ylabel(f"Frequency ({cfg.frequency_unit})")
        _annotate_rows(ax, range(len(rows)), [_float(row.get(value_key)) for row in rows], rows, ("candidate_id",), cfg)
        _apply_axes_style(ax, cfg)
        return _created_result(figure_type, fig, (ax,), cfg, source, context, tuple(diagnostics), save=save)
    except Exception as exc:
        return _failed_result(figure_type, cfg, source, exc)


def _plot_spectrum_series(ax: object, spectrum: Spectrum, cfg: ScientificVisualizationSettings, *, label: str) -> None:
    freqs = _finite_array(spectrum.frequencies_hz, "frequency")
    mags = _finite_array(spectrum.magnitudes, "magnitude")
    if cfg.frequency_scale == "log":
        keep = freqs >= cfg.log_frequency_min_hz
        freqs = freqs[keep]
        mags = mags[keep]
    y, ylabel, _ = _amplitude_for_display(mags, spectrum.magnitude_unit, cfg, floor=cfg.spectrum_db_floor)
    x, y, _ = _decimate_xy(freqs, y, cfg.maximum_spectrum_points, cfg)
    ax.plot(x, y, color="#0072B2", linewidth=cfg.line_width, label=label)
    ax.set_xlabel(f"Frequency ({cfg.frequency_unit})")
    ax.set_ylabel(ylabel)
    if cfg.frequency_scale == "log":
        ax.set_xscale("log")


def _plot_peaks(ax: object, spectrum: Spectrum, peaks: Sequence[SpectralPeak], cfg: ScientificVisualizationSettings) -> None:
    peak_freqs = []
    peak_values = []
    for peak in peaks:
        freq = peak.refined_frequency_hz or peak.bin_frequency_hz
        value = peak.refined_amplitude if peak.refined_amplitude is not None else peak.bin_amplitude
        peak_freqs.append(freq)
        peak_values.append(value)
    display, _, _ = _amplitude_for_display(np.asarray(peak_values, dtype=float), spectrum.magnitude_unit, cfg, floor=cfg.spectrum_db_floor)
    ax.scatter(peak_freqs, display, marker="o", s=cfg.marker_size ** 2, facecolors="white", edgecolors="#D55E00", label="detected peaks")
    for peak, y in zip(peaks[:cfg.maximum_annotation_count], display, strict=False):
        freq = peak.refined_frequency_hz or peak.bin_frequency_hz
        if cfg.show_ids:
            ax.text(freq, y, str(peak.bin_index), fontsize=cfg.annotation_font_size)
        if peak.width_hz is not None and peak.width_hz > 0:
            ax.hlines(y, freq - peak.width_hz / 2.0, freq + peak.width_hz / 2.0, colors="#E69F00", linestyles=":", linewidth=cfg.line_width)


def _plot_track(ax: object, track: SpectralTrack, cfg: ScientificVisualizationSettings) -> None:
    freqs = np.asarray([
        value if value is not None else fallback
        for value, fallback in zip(track.refined_frequencies_hz, track.bin_frequencies_hz, strict=True)
    ], dtype=float)
    times = np.asarray(track.times_s, dtype=float)
    frames = tuple(track.frame_indices)
    color = _color_for(str(track.track_id), cfg)
    start = 0
    for index in range(1, len(frames) + 1):
        if index == len(frames) or frames[index] != frames[index - 1] + 1:
            x, y, _ = _decimate_xy(times[start:index], freqs[start:index], cfg.maximum_track_points, cfg)
            ax.plot(x, y, marker="o", markersize=cfg.marker_size, linewidth=cfg.line_width, color=color, label=f"track {track.track_id}" if start == 0 and cfg.show_ids else None)
            start = index


def _overlay_tracks_on_axis(ax: object, tracking: SpectralTrackingResults | None, cfg: ScientificVisualizationSettings) -> None:
    if tracking is None:
        return
    for track in sorted(tracking.tracks, key=lambda item: item.track_id):
        _plot_track(ax, track, cfg)


def _plot_without_non_adjacent_gaps(
    ax: object,
    x_values: Sequence[int],
    y_values: Sequence[float | None],
    color: str,
    cfg: ScientificVisualizationSettings,
    *,
    label: str | None,
) -> None:
    segment_x: list[int] = []
    segment_y: list[float] = []
    previous_x: int | None = None
    first = True
    for x, y in zip(x_values, y_values, strict=True):
        if y is None:
            if segment_x:
                ax.plot(segment_x, segment_y, marker="o", linestyle="-", color=color, linewidth=cfg.line_width, label=label if first else None)
                first = False
            segment_x, segment_y, previous_x = [], [], None
            continue
        if previous_x is not None and x - previous_x != 1:
            ax.plot(segment_x, segment_y, marker="o", linestyle="-", color=color, linewidth=cfg.line_width, label=label if first else None)
            first = False
            segment_x, segment_y = [], []
        segment_x.append(x)
        segment_y.append(y)
        previous_x = x
    if segment_x:
        ax.plot(segment_x, segment_y, marker="o", linestyle="-", color=color, linewidth=cfg.line_width, label=label if first else None)


def _scatter_true_estimated(ax: object, rows: Sequence[object], true_name: str, estimate_name: str, label: str) -> None:
    true_values = []
    estimated_values = []
    for row in rows:
        true = _float(getattr(row, true_name, None))
        estimate = _float(getattr(row, estimate_name, None))
        if true is not None and estimate is not None:
            true_values.append(true)
            estimated_values.append(estimate)
    if true_values:
        ax.scatter(true_values, estimated_values, marker="o", facecolors="white", edgecolors="#0072B2")
        lo = min(min(true_values), min(estimated_values))
        hi = max(max(true_values), max(estimated_values))
        ax.plot((lo, hi), (lo, hi), linestyle=":", color="#666666", linewidth=1.0)
    ax.set_xlabel(f"True {label}")
    ax.set_ylabel(f"Estimated {label}")


def _created_result(
    figure_type: ScientificFigureType,
    fig: object,
    axes: tuple[object, ...],
    cfg: ScientificVisualizationSettings,
    source: object,
    context: Mapping[str, object],
    diagnostics: tuple[str, ...],
    *,
    reservations: tuple[ScientificVisualizationReason, ...] = (),
    save: bool,
) -> ScientificFigureResult:
    context = _effective_context(context, figure_type, cfg)
    figure_id = _figure_id(figure_type, context, diagnostics)
    provenance = _provenance(figure_id, figure_type, context, cfg, diagnostics)
    _attach_footer(fig, provenance, cfg)
    artifacts = save_scientific_figure(fig, cfg, figure_id=figure_id, figure_type=figure_type) if save else ()
    failed = tuple(item for item in artifacts if item.status in {ScientificVisualizationStatus.FAILED, ScientificVisualizationStatus.INVALID_INPUT})
    if failed:
        status = ScientificVisualizationStatus.FAILED
    else:
        status = ScientificVisualizationStatus.CREATED_WITH_RESERVATIONS if reservations or _source_requires_review(source) else ScientificVisualizationStatus.CREATED
    reasons = (ScientificVisualizationReason.FIGURE_CREATED, ScientificVisualizationReason.REQUESTED_LAYERS_RENDERED, ScientificVisualizationReason.PROVENANCE_ATTACHED)
    final_fig = fig
    final_axes = axes
    if save and cfg.close_after_save and hasattr(fig, "clear"):
        fig.clear()
        final_fig = None
        final_axes = ()
    return ScientificFigureResult(
        figure_id=figure_id,
        figure_type=figure_type,
        status=status,
        figure=final_fig,
        axes=final_axes,
        artifacts=artifacts,
        source_ids=tuple(context.get("source_result_ids", ())),
        provenance=provenance,
        supporting_reasons=reasons if not failed else (),
        reservation_reasons=tuple(reservations) + ((ScientificVisualizationReason.SOURCE_REQUIRES_REVIEW,) if _source_requires_review(source) else ()),
        skipped_reasons=(),
        insufficient_evidence_reasons=(),
        failure_reasons=(ScientificVisualizationReason.FILESYSTEM_ERROR,) if failed else (),
        valid=status in {ScientificVisualizationStatus.CREATED, ScientificVisualizationStatus.CREATED_WITH_RESERVATIONS},
        diagnostics=tuple((*diagnostics, "figure_visualization_does_not_add_physical_evidence")),
    )


def _insufficient_result(
    figure_type: ScientificFigureType,
    cfg: ScientificVisualizationSettings,
    source: object,
    reason: ScientificVisualizationReason,
) -> ScientificFigureResult:
    context = _source_context(source, figure_type, cfg)
    figure_id = _figure_id(figure_type, context, (reason.value,))
    provenance = _provenance(figure_id, figure_type, context, cfg, (reason.value,))
    return ScientificFigureResult(
        figure_id=figure_id,
        figure_type=figure_type,
        status=ScientificVisualizationStatus.INSUFFICIENT_EVIDENCE,
        figure=None,
        axes=(),
        artifacts=(),
        source_ids=tuple(context.get("source_result_ids", ())),
        provenance=provenance,
        supporting_reasons=(),
        reservation_reasons=(),
        skipped_reasons=(),
        insufficient_evidence_reasons=(reason,),
        failure_reasons=(),
        valid=False,
        diagnostics=(reason.value, "missing_values_preserved_not_zero"),
    )


def _invalid_result(
    figure_type: ScientificFigureType,
    cfg: ScientificVisualizationSettings,
    source: object,
    reason: ScientificVisualizationReason,
) -> ScientificFigureResult:
    context = _source_context(source, figure_type, cfg)
    figure_id = _figure_id(figure_type, context, (reason.value,))
    provenance = _provenance(figure_id, figure_type, context, cfg, (reason.value,))
    return ScientificFigureResult(
        figure_id=figure_id,
        figure_type=figure_type,
        status=ScientificVisualizationStatus.INVALID_INPUT,
        figure=None,
        axes=(),
        artifacts=(),
        source_ids=tuple(context.get("source_result_ids", ())),
        provenance=provenance,
        supporting_reasons=(),
        reservation_reasons=(),
        skipped_reasons=(),
        insufficient_evidence_reasons=(),
        failure_reasons=(reason,),
        valid=False,
        diagnostics=(reason.value,),
    )


def _failed_result(
    figure_type: ScientificFigureType,
    cfg: ScientificVisualizationSettings,
    source: object,
    exc: Exception,
) -> ScientificFigureResult:
    context = _source_context(source, figure_type, cfg)
    figure_id = _figure_id(figure_type, context, (exc.__class__.__name__, str(exc)))
    provenance = _provenance(figure_id, figure_type, context, cfg, (str(exc),))
    return ScientificFigureResult(
        figure_id=figure_id,
        figure_type=figure_type,
        status=ScientificVisualizationStatus.FAILED,
        figure=None,
        axes=(),
        artifacts=(),
        source_ids=tuple(context.get("source_result_ids", ())),
        provenance=provenance,
        supporting_reasons=(),
        reservation_reasons=(),
        skipped_reasons=(),
        insufficient_evidence_reasons=(),
        failure_reasons=(ScientificVisualizationReason.RENDERING_ERROR,),
        valid=False,
        diagnostics=(f"{exc.__class__.__name__}: {exc}",),
    )


def _skipped_result(
    figure_type: ScientificFigureType,
    cfg: ScientificVisualizationSettings,
    source: object,
    reason: ScientificVisualizationReason,
) -> ScientificFigureResult:
    context = _source_context(source, figure_type, cfg)
    figure_id = _figure_id(figure_type, context, (reason.value,))
    provenance = _provenance(figure_id, figure_type, context, cfg, (reason.value,))
    return ScientificFigureResult(
        figure_id=figure_id,
        figure_type=figure_type,
        status=ScientificVisualizationStatus.SKIPPED,
        figure=None,
        axes=(),
        artifacts=(),
        source_ids=tuple(context.get("source_result_ids", ())),
        provenance=provenance,
        supporting_reasons=(),
        reservation_reasons=(),
        skipped_reasons=(reason,),
        insufficient_evidence_reasons=(),
        failure_reasons=(),
        valid=False,
        diagnostics=(reason.value,),
    )


def _collection(
    source: object,
    cfg: ScientificVisualizationSettings,
    figures: tuple[ScientificFigureResult, ...],
) -> ScientificFigureCollection:
    artifacts = tuple(artifact for figure in figures for artifact in figure.artifacts)
    requested = tuple(cfg.figure_types)
    completed = tuple(dict.fromkeys(figure.figure_type for figure in figures if figure.status in {ScientificVisualizationStatus.CREATED, ScientificVisualizationStatus.CREATED_WITH_RESERVATIONS}))
    skipped = tuple(dict.fromkeys(figure.figure_type for figure in figures if figure.status in {ScientificVisualizationStatus.SKIPPED, ScientificVisualizationStatus.INSUFFICIENT_EVIDENCE}))
    failed = tuple(dict.fromkeys(figure.figure_type for figure in figures if figure.status in {ScientificVisualizationStatus.FAILED, ScientificVisualizationStatus.INVALID_INPUT}))
    if failed:
        status = ScientificVisualizationStatus.FAILED
    elif completed and skipped:
        status = ScientificVisualizationStatus.CREATED_WITH_RESERVATIONS
    elif completed:
        status = ScientificVisualizationStatus.CREATED_WITH_RESERVATIONS if _source_requires_review(source) else ScientificVisualizationStatus.CREATED
    elif skipped:
        status = ScientificVisualizationStatus.INSUFFICIENT_EVIDENCE
    else:
        status = ScientificVisualizationStatus.SKIPPED
    context = _source_context(source, ScientificFigureType.EXPERIMENT_SUMMARY, cfg)
    collection_id = _stable_id("figure-collection", context, tuple((fig.figure_type.value, fig.figure_id, fig.status.value) for fig in figures), tuple((artifact.relative_path, artifact.checksum) for artifact in artifacts))
    return ScientificFigureCollection(
        collection_id=collection_id,
        analysis_id=context.get("analysis_id"),
        experiment_id=context.get("experiment_id"),
        figures=figures,
        artifacts=artifacts,
        requested_figure_types=requested,
        completed_figure_types=completed,
        skipped_figure_types=skipped,
        failed_figure_types=failed,
        status=status,
        settings=cfg,
        provenance=MappingProxyType({
            "settings_fingerprint": scientific_visualization_settings_fingerprint(cfg),
            "belllab_version": context.get("belllab_version"),
            "source_statuses": context.get("source_statuses", ()),
            "artifact_checksums": tuple((item.relative_path, item.checksum) for item in artifacts),
        }),
        valid=status in {ScientificVisualizationStatus.CREATED, ScientificVisualizationStatus.CREATED_WITH_RESERVATIONS},
        diagnostics=(
            "visualization_collection_does_not_recalculate_scientific_analysis",
            "figures_are_operational_representations_only",
        ),
    )


def _extract_signal(source: object) -> tuple[Signal | None, str, Mapping[str, object]]:
    if isinstance(source, Signal):
        return source, source.filename or source.sha256 or "signal", _source_context(source, ScientificFigureType.WAVEFORM, ScientificVisualizationSettings())
    recording = _first_recording(source, require_signal=True)
    if recording is not None and recording.loaded_recording is not None:
        return recording.loaded_recording.signal, recording.recording_definition.recording_id or "recording", _source_context(recording, ScientificFigureType.WAVEFORM, ScientificVisualizationSettings())
    return None, "missing-signal", _source_context(source, ScientificFigureType.WAVEFORM, ScientificVisualizationSettings())


def _extract_envelope(source: object) -> tuple[Envelope | None, str, Mapping[str, object]]:
    if isinstance(source, Envelope):
        return source, "envelope", _source_context(source, ScientificFigureType.TEMPORAL_ENVELOPE, ScientificVisualizationSettings())
    if isinstance(source, TemporalResults):
        return source.envelope, "temporal-result", _source_context(source, ScientificFigureType.TEMPORAL_ENVELOPE, ScientificVisualizationSettings())
    recording = _first_recording(source, require_signal=False)
    if recording is not None and recording.temporal_result is not None:
        return recording.temporal_result.envelope, recording.recording_definition.recording_id or "recording", _source_context(recording, ScientificFigureType.TEMPORAL_ENVELOPE, ScientificVisualizationSettings())
    return None, "missing-envelope", _source_context(source, ScientificFigureType.TEMPORAL_ENVELOPE, ScientificVisualizationSettings())


def _extract_decay_fit(source: object) -> object | None:
    if isinstance(source, TemporalResults):
        return source.decay_fit
    if isinstance(source, ExperimentRecordingAnalysisResult) and source.temporal_result is not None:
        return source.temporal_result.decay_fit
    return None


def _extract_spectrum(source: object) -> tuple[Spectrum | None, str, Mapping[str, object]]:
    if isinstance(source, Spectrum):
        return source, "spectrum", _source_context(source, ScientificFigureType.GLOBAL_SPECTRUM, ScientificVisualizationSettings())
    if isinstance(source, SpectrumResults):
        return source.spectrum, "spectrum-result", _source_context(source, ScientificFigureType.GLOBAL_SPECTRUM, ScientificVisualizationSettings())
    recording = _first_recording(source, require_signal=False)
    if recording is not None and recording.spectral_result is not None:
        return recording.spectral_result.spectrum, recording.recording_definition.recording_id or "recording", _source_context(recording, ScientificFigureType.GLOBAL_SPECTRUM, ScientificVisualizationSettings())
    return None, "missing-spectrum", _source_context(source, ScientificFigureType.GLOBAL_SPECTRUM, ScientificVisualizationSettings())


def _extract_peak_result(source: object) -> tuple[PeakDetectionResults | None, Spectrum | None, str, Mapping[str, object]]:
    if isinstance(source, PeakDetectionResults):
        return source, source.spectrum, "peaks", _source_context(source, ScientificFigureType.SPECTRAL_PEAKS, ScientificVisualizationSettings())
    recording = _first_recording(source, require_signal=False)
    if recording is not None:
        peak = recording.peak_result
        spectrum = peak.spectrum if peak is not None else recording.spectral_result.spectrum if recording.spectral_result is not None else None
        return peak, spectrum, recording.recording_definition.recording_id or "recording", _source_context(recording, ScientificFigureType.SPECTRAL_PEAKS, ScientificVisualizationSettings())
    return None, None, "missing-peaks", _source_context(source, ScientificFigureType.SPECTRAL_PEAKS, ScientificVisualizationSettings())


def _extract_time_frequency(source: object) -> tuple[TimeFrequencySpectrum | None, str, Mapping[str, object]]:
    if isinstance(source, TimeFrequencySpectrum):
        return source, "time-frequency", _source_context(source, ScientificFigureType.SPECTROGRAM, ScientificVisualizationSettings())
    if isinstance(source, STFTResults):
        return source.time_frequency, "stft", _source_context(source, ScientificFigureType.SPECTROGRAM, ScientificVisualizationSettings())
    recording = _first_recording(source, require_signal=False)
    if recording is not None and recording.stft_result is not None:
        return recording.stft_result.time_frequency, recording.recording_definition.recording_id or "recording", _source_context(recording, ScientificFigureType.SPECTROGRAM, ScientificVisualizationSettings())
    return None, "missing-stft", _source_context(source, ScientificFigureType.SPECTROGRAM, ScientificVisualizationSettings())


def _extract_tracking(source: object) -> tuple[SpectralTrackingResults | None, str, Mapping[str, object]]:
    if isinstance(source, SpectralTrackingResults):
        return source, "tracking", _source_context(source, ScientificFigureType.FREQUENCY_TRACKS, ScientificVisualizationSettings())
    recording = _first_recording(source, require_signal=False)
    if recording is not None:
        return recording.tracking_result, recording.recording_definition.recording_id or "recording", _source_context(recording, ScientificFigureType.FREQUENCY_TRACKS, ScientificVisualizationSettings())
    return None, "missing-tracking", _source_context(source, ScientificFigureType.FREQUENCY_TRACKS, ScientificVisualizationSettings())


def _first_recording(source: object, *, require_signal: bool) -> ExperimentRecordingAnalysisResult | None:
    if isinstance(source, ExperimentRecordingAnalysisResult):
        return source
    if isinstance(source, ExperimentAnalysisResult):
        for item in sorted(source.recording_results, key=lambda row: row.recording_definition.recording_id or ""):
            if require_signal:
                if item.loaded_recording is not None and item.loaded_recording.signal is not None:
                    return item
            else:
                return item
    return None


def _recordings_with_signal(result: ExperimentAnalysisResult) -> tuple[ExperimentRecordingAnalysisResult, ...]:
    return tuple(item for item in sorted(result.recording_results, key=lambda row: row.recording_definition.recording_id or "") if item.loaded_recording is not None and item.loaded_recording.signal is not None)


def _recordings_with_envelope(result: ExperimentAnalysisResult) -> tuple[ExperimentRecordingAnalysisResult, ...]:
    return tuple(item for item in sorted(result.recording_results, key=lambda row: row.recording_definition.recording_id or "") if item.temporal_result is not None and item.temporal_result.envelope is not None)


def _recordings_with_spectrum(result: ExperimentAnalysisResult) -> tuple[ExperimentRecordingAnalysisResult, ...]:
    return tuple(item for item in sorted(result.recording_results, key=lambda row: row.recording_definition.recording_id or "") if item.spectral_result is not None and item.spectral_result.spectrum is not None)


def _recordings_with_stft(result: ExperimentAnalysisResult) -> tuple[ExperimentRecordingAnalysisResult, ...]:
    return tuple(item for item in sorted(result.recording_results, key=lambda row: row.recording_definition.recording_id or "") if item.stft_result is not None)


def _recordings_with_tracks(result: ExperimentAnalysisResult) -> tuple[ExperimentRecordingAnalysisResult, ...]:
    return tuple(item for item in sorted(result.recording_results, key=lambda row: row.recording_definition.recording_id or "") if item.tracking_result is not None and item.tracking_result.tracks)


def _rows_from_source(source: object, table_name: str) -> tuple[tuple[Mapping[str, object], ...], Mapping[str, object]]:
    if isinstance(source, ExperimentAnalysisResult):
        normalized = normalize_experiment_for_export(source, ResultsExportSettings())
        return tuple(normalized.tables.get(table_name, ())), _source_context(source, ScientificFigureType.EXPERIMENT_SUMMARY, ScientificVisualizationSettings())
    if isinstance(source, ExperimentConditionAnalysisResult):
        fake_rows: tuple[Mapping[str, object], ...] = tuple(
            {
                "dynamic_label": source.dynamic_label,
                "recording_id": ref.recording_id,
                "candidate_id": ref.candidate_id,
                "source_track_id": ref.source_track_id,
                "representative_frequency_hz": ref.representative_frequency_hz,
                "accepted": True,
            }
            for ref in source.candidate_references
        )
        return fake_rows, _source_context(source, ScientificFigureType.EXPERIMENT_SUMMARY, ScientificVisualizationSettings())
    return (), _source_context(source, ScientificFigureType.EXPERIMENT_SUMMARY, ScientificVisualizationSettings())


def _bandwidth_rows(source: object) -> tuple[tuple[Mapping[str, object], ...], Mapping[str, object]]:
    if hasattr(source, "bandwidth_hz") and hasattr(source, "center_frequency_hz"):
        return (
            (
                MappingProxyType({
                    "estimate_id": getattr(source, "estimate_id", "bandwidth"),
                    "center_frequency_hz": getattr(source, "center_frequency_hz", None),
                    "lower_frequency_hz": getattr(source, "lower_frequency_hz", None),
                    "upper_frequency_hz": getattr(source, "upper_frequency_hz", None),
                    "bandwidth_hz": getattr(source, "bandwidth_hz", None),
                    "frequency_resolution_hz": getattr(source, "frequency_resolution_hz", None),
                    "status": "valid" if getattr(source, "valid", False) else "insufficient_evidence",
                }),
            ),
            _source_context(source, ScientificFigureType.MODAL_BANDWIDTH, ScientificVisualizationSettings()),
        )
    if hasattr(source, "bandwidth_estimate"):
        bandwidth = getattr(source, "bandwidth_estimate")
        if bandwidth is not None:
            return _bandwidth_rows(bandwidth)
    rows, context = _rows_from_source(source, "modal_q_factors")
    filtered = tuple(row for row in rows if row.get("bandwidth_hz") is not None)
    return filtered, context


def _first_energy_evidence(source: object) -> object | None:
    if hasattr(source, "alignment") and hasattr(source, "correlation_evidence"):
        return source
    if hasattr(source, "pair_evidences"):
        evidences = getattr(source, "pair_evidences")
        return evidences[0] if evidences else None
    if isinstance(source, ExperimentAnalysisResult):
        for result in source.energy_exchange_results:
            if result.pair_evidences:
                return result.pair_evidences[0]
    return None


def _source_context(source: object, figure_type: ScientificFigureType, cfg: ScientificVisualizationSettings) -> Mapping[str, object]:
    analysis_id = getattr(source, "analysis_id", None)
    experiment_id = None
    belllab_version = None
    recording_ids: tuple[str, ...] = ()
    condition_labels: tuple[str, ...] = ()
    statuses: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    if isinstance(source, ExperimentAnalysisResult):
        experiment_id = source.experiment.experiment_id
        belllab_version = source.provenance.belllab_version
        recording_ids = tuple(item.recording_definition.recording_id or "" for item in source.recording_results)
        condition_labels = tuple(item.dynamic_label for item in source.condition_results)
        statuses = (source.status.value,)
        source_ids = (source.analysis_id,)
    elif isinstance(source, ExperimentRecordingAnalysisResult):
        recording_ids = (source.recording_definition.recording_id or "",)
        condition_labels = (source.recording_definition.dynamic_label,)
        statuses = (str(source.valid),)
        source_ids = (source.recording_definition.recording_id or "",)
    elif isinstance(source, ExperimentConditionAnalysisResult):
        recording_ids = tuple(item.recording_definition.recording_id or "" for item in source.recording_results)
        condition_labels = (source.dynamic_label,)
        statuses = (str(source.valid),)
        source_ids = (source.dynamic_label,)
    elif isinstance(source, SyntheticScenarioValidationResult):
        source_ids = (source.scenario.scenario_id,)
        statuses = (source.status.value,)
    elif isinstance(source, SyntheticValidationCampaignResult):
        source_ids = (source.campaign_id,)
        statuses = ("valid" if source.valid else "invalid",)
    elif isinstance(source, SyntheticMonteCarloValidation):
        source_ids = (source.base_scenario_id,)
        statuses = ("valid" if source.valid else "invalid",)
    else:
        object_id = getattr(source, "estimate_id", None) or getattr(source, "hypothesis_id", None) or getattr(source, "evidence_id", None) or getattr(source, "source_id", None)
        source_ids = (str(object_id or type(source).__name__),)
        status = getattr(source, "status", None)
        statuses = (status.value if isinstance(status, Enum) else str(status),) if status is not None else ()
    return MappingProxyType({
        "analysis_id": analysis_id,
        "experiment_id": experiment_id,
        "belllab_version": belllab_version,
        "recording_ids": _unique_texts(recording_ids),
        "condition_labels": _unique_texts(condition_labels),
        "source_result_ids": _unique_texts(source_ids),
        "source_statuses": _unique_texts(statuses),
        "figure_type": figure_type.value,
        "settings_fingerprint": scientific_visualization_settings_fingerprint(cfg),
    })


def _provenance(
    figure_id: str,
    figure_type: ScientificFigureType,
    context: Mapping[str, object],
    cfg: ScientificVisualizationSettings,
    diagnostics: Sequence[str],
) -> ScientificFigureProvenance:
    decimations = tuple(item for item in diagnostics if "decimation" in str(item) or "decimated" in str(item))
    interpolations = tuple(item for item in diagnostics if "interpolation" in str(item) or "interpolated" in str(item))
    return ScientificFigureProvenance(
        figure_id=figure_id,
        figure_type=figure_type.value,
        analysis_id=context.get("analysis_id"),
        experiment_id=context.get("experiment_id"),
        recording_ids=tuple(context.get("recording_ids", ())),
        condition_labels=tuple(context.get("condition_labels", ())),
        source_result_ids=tuple(context.get("source_result_ids", ())),
        settings_fingerprint=scientific_visualization_settings_fingerprint(cfg),
        belllab_version=context.get("belllab_version"),
        source_statuses=tuple(context.get("source_statuses", ())),
        interpolations_used=_unique_texts(interpolations),
        decimations_used=_unique_texts(decimations),
        diagnostics=tuple(diagnostics),
    )


def _effective_context(
    context: Mapping[str, object],
    figure_type: ScientificFigureType,
    cfg: ScientificVisualizationSettings,
) -> Mapping[str, object]:
    payload = dict(context)
    payload["figure_type"] = figure_type.value
    payload["settings_fingerprint"] = scientific_visualization_settings_fingerprint(cfg)
    return MappingProxyType(payload)


def _figure_id(figure_type: ScientificFigureType, context: Mapping[str, object], diagnostics: Sequence[str]) -> str:
    return _stable_id("figure", figure_type.value, context, tuple(diagnostics))


def _new_figure(cfg: ScientificVisualizationSettings) -> tuple[object, object]:
    _, Figure, FigureCanvasAgg = _matplotlib()
    with _rc_context(cfg):
        fig = Figure(
            figsize=(_width(cfg), cfg.figure_height_in),
            dpi=cfg.dpi,
            constrained_layout=cfg.constrained_layout,
        )
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        return fig, ax


def _new_subplots(cfg: ScientificVisualizationSettings, rows: int, columns: int) -> tuple[object, tuple[object, ...]]:
    _, Figure, FigureCanvasAgg = _matplotlib()
    with _rc_context(cfg):
        fig = Figure(
            figsize=(_width(cfg), max(cfg.figure_height_in, rows * 2.2)),
            dpi=cfg.dpi,
            constrained_layout=cfg.constrained_layout,
        )
        FigureCanvasAgg(fig)
        axes = fig.subplots(rows, columns, squeeze=False).ravel()
        return fig, tuple(axes)


def _matplotlib() -> tuple[object, object, object]:
    cache = Path(os.environ.get("MPLCONFIGDIR", "/tmp/belllab-matplotlib"))
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    return matplotlib, Figure, FigureCanvasAgg


def _rc_context(cfg: ScientificVisualizationSettings) -> object:
    matplotlib, _, _ = _matplotlib()
    return matplotlib.rc_context({
        "font.family": cfg.font_family,
        "font.size": cfg.base_font_size,
        "axes.titlesize": cfg.title_font_size,
        "axes.labelsize": cfg.label_font_size,
        "xtick.labelsize": cfg.tick_font_size,
        "ytick.labelsize": cfg.tick_font_size,
        "legend.fontsize": cfg.legend_font_size,
        "mathtext.fontset": cfg.mathtext_fontset,
        "figure.dpi": cfg.dpi,
        "savefig.dpi": cfg.dpi,
    })


def _width(cfg: ScientificVisualizationSettings) -> float:
    if cfg.aspect_policy == "wide":
        return cfg.figure_width_in * 1.25
    if cfg.aspect_policy == "square":
        return cfg.figure_height_in
    return cfg.figure_width_in


def _apply_axes_style(ax: object, cfg: ScientificVisualizationSettings) -> None:
    if cfg.grid:
        ax.grid(True, alpha=cfg.grid_alpha)
    if cfg.minor_ticks and hasattr(ax, "minorticks_on"):
        ax.minorticks_on()
    handles, labels = ax.get_legend_handles_labels()
    if cfg.show_legend and handles:
        ax.legend(loc=cfg.legend_location)


def _attach_footer(fig: object, provenance: ScientificFigureProvenance, cfg: ScientificVisualizationSettings) -> None:
    if not cfg.show_provenance_footer:
        return
    text = f"{provenance.figure_id} | {provenance.settings_fingerprint[:12]} | operational visualization only"
    fig.text(0.01, 0.005, text, fontsize=max(5.0, cfg.annotation_font_size - 1), ha="left", va="bottom", alpha=0.7)


def _save_one_format(
    figure: object,
    figure_id: str,
    figure_type: ScientificFigureType,
    fmt: str,
    cfg: ScientificVisualizationSettings,
) -> ScientificFigureArtifact:
    path = _figure_path(figure_id, figure_type, fmt, cfg)
    try:
        target = _resolve_write_path(path, cfg)
        if target is None:
            return ScientificFigureArtifact(
                artifact_id=_stable_id("figure-artifact", figure_id, fmt, "skipped"),
                figure_id=figure_id,
                format=fmt,
                path=str(path),
                relative_path=_relative_path(path, cfg),
                checksum=scientific_figure_checksum(path) if path.is_file() else None,
                size_bytes=path.stat().st_size if path.is_file() else None,
                width_px=_width_px(figure, cfg),
                height_px=_height_px(figure, cfg),
                dpi=cfg.dpi,
                status=ScientificVisualizationStatus.SKIPPED,
                reasons=(ScientificVisualizationReason.FILESYSTEM_ERROR,),
                diagnostics=("existing_file_preserved_by_skip_policy",),
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        save_path = target.with_name(f".{target.name}.tmp") if cfg.atomic_write else target
        metadata = _save_metadata(fmt)
        figure.savefig(
            save_path,
            format=fmt,
            dpi=cfg.dpi,
            transparent=cfg.transparent_background,
            bbox_inches=cfg.bbox_inches,
            pad_inches=cfg.pad_inches,
            metadata=metadata,
        )
        if cfg.atomic_write:
            os.replace(save_path, target)
        checksum = scientific_figure_checksum(target)
        return ScientificFigureArtifact(
            artifact_id=_stable_id("figure-artifact", figure_id, fmt, _relative_path(target, cfg), checksum),
            figure_id=figure_id,
            format=fmt,
            path=str(target),
            relative_path=_relative_path(target, cfg),
            checksum=checksum,
            size_bytes=target.stat().st_size,
            width_px=_width_px(figure, cfg),
            height_px=_height_px(figure, cfg),
            dpi=cfg.dpi,
            status=ScientificVisualizationStatus.CREATED,
            reasons=(ScientificVisualizationReason.FIGURE_CREATED,),
            diagnostics=("figure_saved_with_content_checksum",),
        )
    except Exception as exc:
        tmp = path.with_name(f".{path.name}.tmp")
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        return _artifact_failure(figure_id, figure_type.value, fmt, cfg, ScientificVisualizationReason.FILESYSTEM_ERROR, f"{exc.__class__.__name__}: {exc}")


def _save_metadata(fmt: str) -> Mapping[str, object] | None:
    if fmt == "png":
        return {"Software": "BellLab"}
    if fmt == "svg":
        return {"Creator": "BellLab", "Date": None}
    if fmt == "pdf":
        return {"Creator": "BellLab", "CreationDate": None, "ModDate": None}
    return None


def _artifact_failure(
    figure_id: str,
    figure_type: str,
    fmt: str,
    cfg: ScientificVisualizationSettings,
    reason: ScientificVisualizationReason,
    diagnostic: str,
) -> ScientificFigureArtifact:
    return ScientificFigureArtifact(
        artifact_id=_stable_id("figure-artifact", figure_id, fmt, "failed", diagnostic),
        figure_id=figure_id,
        format=fmt,
        path=None,
        relative_path=None,
        checksum=None,
        size_bytes=None,
        width_px=None,
        height_px=None,
        dpi=cfg.dpi,
        status=ScientificVisualizationStatus.FAILED,
        reasons=(reason,),
        diagnostics=(figure_type, diagnostic),
    )


def _figure_path(
    figure_id: str,
    figure_type: ScientificFigureType,
    fmt: str,
    cfg: ScientificVisualizationSettings,
) -> Path:
    suffix = _sanitize(figure_id.replace("figure-", ""))
    return Path(cfg.output_directory) / f"{_sanitize(cfg.file_prefix)}_{figure_type.value}_{suffix}.{fmt}"


def _resolve_write_path(path: Path, cfg: ScientificVisualizationSettings) -> Path | None:
    if not path.exists():
        return path
    if cfg.overwrite_policy is ExportOverwritePolicy.ERROR:
        raise FileExistsError(f"figure artifact already exists: {path}")
    if cfg.overwrite_policy is ExportOverwritePolicy.SKIP:
        return None
    if cfg.overwrite_policy is ExportOverwritePolicy.REPLACE:
        return path
    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}_v{index:03d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"could not allocate versioned figure filename: {path}")


def _relative_path(path: Path, cfg: ScientificVisualizationSettings) -> str:
    try:
        return str(path.relative_to(Path(cfg.output_directory)))
    except ValueError:
        return path.name


def _width_px(figure: object, cfg: ScientificVisualizationSettings) -> int:
    return int(round(float(figure.get_figwidth()) * cfg.dpi))


def _height_px(figure: object, cfg: ScientificVisualizationSettings) -> int:
    return int(round(float(figure.get_figheight()) * cfg.dpi))


def _finite_array(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} values must be a finite 1D series.")
    return array


def _amplitude_for_display(
    values: np.ndarray,
    unit: str,
    cfg: ScientificVisualizationSettings,
    *,
    floor: float,
) -> tuple[np.ndarray, str, tuple[str, ...]]:
    values = np.asarray(values, dtype=float)
    diagnostics: list[str] = []
    if cfg.amplitude_scale == "db" and "db" not in unit.lower():
        positive = values > 0
        if not np.any(positive):
            raise ValueError("log amplitude requested without positive values.")
        safe = np.where(positive, values, 10 ** (floor / 20.0))
        display = np.maximum(20.0 * np.log10(safe), floor)
        if not np.all(positive):
            diagnostics.append(ScientificVisualizationReason.LOG_SCALE_CLIPPED.value)
        return display, "Amplitude (dB, presentation)", tuple(diagnostics)
    if cfg.amplitude_scale == "linear" and "db" in unit.lower():
        diagnostics.append("db_values_rendered_without_linear_reconstruction")
    return values, f"Amplitude ({unit})", tuple(diagnostics)


def _matrix_for_display(
    values: np.ndarray,
    unit: str,
    cfg: ScientificVisualizationSettings,
) -> tuple[np.ndarray, str, list[str]]:
    diagnostics: list[str] = []
    if cfg.amplitude_scale == "db" and "db" not in unit.lower():
        positive = values > 0
        if not np.any(positive):
            raise ValueError("dB spectrogram requested without positive values.")
        safe = np.where(positive, values, 10 ** (cfg.spectrogram_db_floor / 20.0))
        display = np.maximum(20.0 * np.log10(safe), cfg.spectrogram_db_floor)
        if not np.all(positive):
            diagnostics.append(ScientificVisualizationReason.LOG_SCALE_CLIPPED.value)
        return display, "Magnitude (dB, presentation)", diagnostics
    return values, f"Magnitude ({unit})", diagnostics


def _decimate_xy(
    x_values: Sequence[float],
    y_values: Sequence[float],
    maximum_points: int,
    cfg: ScientificVisualizationSettings,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if x.size != y.size:
        raise ValueError("x and y values must align.")
    if x.size <= maximum_points or cfg.decimation_method == "none":
        return x.copy(), y.copy(), ()
    if cfg.decimation_method == "stride" or not cfg.preserve_extrema:
        indices = np.linspace(0, x.size - 1, maximum_points, dtype=int)
        indices = np.unique(indices)
    else:
        block_count = max(1, (maximum_points - 2) // 2)
        block_edges = np.linspace(1, x.size - 2, block_count + 1, dtype=int)
        selected = {0, x.size - 1}
        for start, end in zip(block_edges[:-1], block_edges[1:], strict=False):
            if end <= start:
                continue
            block = y[start:end]
            if block.size:
                selected.add(start + int(np.nanargmin(block)))
                selected.add(start + int(np.nanargmax(block)))
        indices = np.asarray(sorted(selected), dtype=int)
    if indices.size > maximum_points:
        indices = indices[np.linspace(0, indices.size - 1, maximum_points, dtype=int)]
    return x[indices].copy(), y[indices].copy(), (f"visual_decimation:{x.size}->{indices.size}",)


def _nonuniform(values: Sequence[float], tolerance: float = 1e-9) -> bool:
    if len(values) < 3:
        return False
    diffs = np.diff(np.asarray(values, dtype=float))
    return bool(np.max(np.abs(diffs - diffs[0])) > tolerance)


def _annotate_rows(
    ax: object,
    x_values: Iterable[object],
    y_values: Iterable[object],
    rows: Sequence[Mapping[str, object]],
    keys: tuple[str, ...],
    cfg: ScientificVisualizationSettings,
) -> None:
    if not cfg.show_ids or cfg.maximum_annotation_count == 0:
        return
    count = 0
    for x, y, row in zip(tuple(x_values), tuple(y_values), rows, strict=False):
        if count >= cfg.maximum_annotation_count:
            break
        y_float = _float(y)
        if y_float is None:
            continue
        label = next((str(row.get(key)) for key in keys if row.get(key) is not None), None)
        if label:
            ax.text(float(x), y_float, label, fontsize=cfg.annotation_font_size)
            count += 1


def _canonical_labels_from_rows(rows: Sequence[Mapping[str, object]], keys: tuple[str, ...]) -> tuple[str, ...]:
    labels = tuple(str(row.get(key)) for row in rows for key in keys if row.get(key) is not None)
    canonical = tuple(label for label in DYNAMIC_LABEL_ORDER if label in labels)
    extras = tuple(sorted(label for label in set(labels) if label not in DYNAMIC_LABEL_ORDER))
    return canonical + extras


def _condition_index(label: str) -> int:
    try:
        return DYNAMIC_LABEL_ORDER.index(label)
    except ValueError:
        return len(DYNAMIC_LABEL_ORDER)


def _color_for(key: str, cfg: ScientificVisualizationSettings) -> str:
    if cfg.color_policy is ScientificColorPolicy.MONOCHROME:
        return "#222222"
    if cfg.color_policy is ScientificColorPolicy.DYNAMIC_CONDITION and key in cfg.condition_color_mapping:
        return cfg.condition_color_mapping[key]
    if cfg.color_policy is ScientificColorPolicy.STATUS and key in cfg.status_style_mapping:
        return cfg.status_style_mapping[key].edgecolor
    if cfg.color_policy is ScientificColorPolicy.CUSTOM and key in cfg.condition_color_mapping:
        return cfg.condition_color_mapping[key]
    palette = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")
    digest = hashlib.sha256(key.encode("utf-8")).digest()[0]
    return palette[digest % len(palette)]


def _marker_for(key: str, cfg: ScientificVisualizationSettings) -> str:
    if key in DEFAULT_CONDITION_MARKERS:
        return DEFAULT_CONDITION_MARKERS[key]
    markers = ("o", "s", "^", "D", "P", "v", "x")
    return markers[hashlib.sha256(key.encode("utf-8")).digest()[1] % len(markers)]


def _style_for(status: str, cfg: ScientificVisualizationSettings) -> ScientificStatusStyle:
    return cfg.status_style_mapping.get(status, ScientificStatusStyle(edgecolor=_color_for(status, cfg)))


def _visible_status(status: str, cfg: ScientificVisualizationSettings) -> bool:
    if status in {"rejected", "not_supported"}:
        return cfg.show_rejected_results
    if status in {"inconclusive", "insufficient_evidence"}:
        return cfg.show_inconclusive_results
    return True


def _reservation_reasons_from_rows(rows: Sequence[Mapping[str, object]]) -> Iterable[ScientificVisualizationReason]:
    for row in rows:
        status = str(row.get("status") or "")
        if "reservation" in status or row.get("reservation_reasons"):
            yield ScientificVisualizationReason.SOURCE_REQUIRES_REVIEW


def _source_requires_review(source: object) -> bool:
    if isinstance(source, ExperimentAnalysisResult):
        return source.requires_review
    if isinstance(source, ExperimentRecordingAnalysisResult):
        return not source.valid
    if isinstance(source, ExperimentConditionAnalysisResult):
        return not source.valid
    requires = getattr(source, "requires_review", None)
    if isinstance(requires, bool):
        return requires
    status = getattr(source, "status", None)
    if isinstance(status, Enum):
        return status.value not in {"valid", "accepted", "supported", "completed", "passed"}
    return False


def _float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _status_numeric(status: str) -> float:
    order = {
        "passed": 1.0,
        "completed": 1.0,
        "valid": 1.0,
        "supported": 1.0,
        "passed_with_reservations": 0.8,
        "completed_with_reservations": 0.8,
        "valid_with_reservations": 0.8,
        "partial": 0.6,
        "inconclusive": 0.4,
        "insufficient_evidence": 0.2,
        "failed": 0.0,
        "invalid_input": 0.0,
        "pipeline_error": 0.0,
    }
    return order.get(status, 0.5)


def _sanitize(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return cleaned.strip("_") or "figure"


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")


def _unique_texts(values: Iterable[object]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values if str(value).strip()))


def _reason_tuple(values: Iterable[ScientificVisualizationReason]) -> tuple[ScientificVisualizationReason, ...]:
    return tuple(dict.fromkeys(_coerce_enum(value, ScientificVisualizationReason) for value in values))


def _coerce_enum(value: object, enum_type: type[Enum]) -> Any:
    if isinstance(value, enum_type):
        return value
    return enum_type(value)


def _stable_id(prefix: str, *parts: object) -> str:
    encoded = json.dumps(_canonicalize(parts), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return f"{prefix}-{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:16]}"


def _canonicalize(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _canonicalize(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(item) for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))}
    if isinstance(value, np.ndarray):
        return _canonicalize(value.tolist())
    if isinstance(value, (tuple, list)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if value == math.inf:
            return "Infinity"
        if value == -math.inf:
            return "-Infinity"
    return value
