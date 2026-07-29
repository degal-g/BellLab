"""High-level orchestration for real BellLab acoustic experiments.

This module coordinates the existing BellLab scientific layers over explicit
WAV recordings and metadata.  It does not add new modal identification logic,
does not calibrate thresholds from the analyzed data, and does not promote a
successful pipeline execution to a physical conclusion.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from belllab.config import (
    AnalysisSettings,
    FramePeakDetectionSettings,
    ModalCandidateSettings,
    PeakDetectionSettings,
    PreImpactAnalysisSettings,
    STFTSettings,
    SpectralTrackingSettings,
    SpectrumAnalysisSettings,
)
from belllab.cross_condition import (
    CrossConditionCandidateAssociationResult,
    CrossConditionCandidateAssociationSettings,
    associate_candidates_across_adjacent_conditions,
)
from belllab.candidate_chains import (
    CrossConditionCandidateChainResult,
    build_cross_condition_candidate_chains,
)
from belllab.dynamic_comparison import (
    DYNAMIC_LABEL_ORDER,
    DynamicConditionComparisonResult,
    DynamicConditionComparisonSettings,
    DynamicConditionRecordingAnalysis,
    compare_dynamic_conditions,
)
from belllab.excitation import (
    ExcitationCharacterization,
    ExcitationCharacterizationSettings,
    characterize_excitation_signal,
)
from belllab.global_spectrum import (
    GlobalSpectralCharacterization,
    GlobalSpectralCharacterizationSettings,
    characterize_signal_spectrum,
)
from belllab.io import load_wav
from belllab.modal_candidates import select_modal_candidates
from belllab.modal_energy_exchange import (
    ModalEnergyExchangeResult,
    ModalEnergyExchangeSettings,
    evaluate_modal_energy_exchange,
)
from belllab.modal_hypotheses import (
    ModalHypothesisResult,
    ModalHypothesisSettings,
    build_modal_hypotheses,
)
from belllab.modal_parameters import (
    ModalParameterEstimationResult,
    ModalParameterEstimationSettings,
    estimate_modal_parameters,
)
from belllab.modal_q_factors import (
    ModalQFactorEstimationResult,
    ModalQFactorEstimationSettings,
    estimate_modal_q_factors,
)
from belllab.preimpact import analyze_preimpact_evidence
from belllab.results import (
    PeakDetectionResults,
    SpectralTrackingResults,
    SpectrumResults,
    STFTResults,
    TemporalResults,
    TimeFrequencyPeakResults,
)
from belllab.spectrum import analyze_spectrum, analyze_stft, detect_spectral_peaks
from belllab.temporal import analyze_temporal
from belllab.time_resolved_spectrum import (
    TimeResolvedSpectralCharacterization,
    TimeResolvedSpectralCharacterizationSettings,
    characterize_time_resolved_spectrum,
)
from belllab.tracking import (
    characterize_spectral_track,
    detect_stft_peaks,
    track_spectral_peaks,
)
from belllab.types import (
    ModalCandidate,
    PreImpactEvidence,
    RecordingMetrics,
    Signal,
    SpectralTrackCharacterization,
)
from belllab.within_condition import (
    CandidateReference,
    ExcitationCondition,
    RecordingCandidateSet,
    WithinConditionAssociationResult,
    WithinConditionAssociationSettings,
    associate_candidates_within_condition,
)


class ExperimentDefinitionError(ValueError):
    """Invalid experiment metadata or configuration."""


class ExperimentInputError(ValueError):
    """Invalid recording path, channel, offset, or loaded signal."""


class ExperimentPipelineDependencyError(ValueError):
    """A requested pipeline stage lacks an explicit dependency."""


class ExperimentStageExecutionError(RuntimeError):
    """A configured stage failed while fail-fast execution was requested."""


class ExperimentPrecomputedResultError(ValueError):
    """A precomputed stage result is incompatible with this run."""


class ExperimentReplicatePolicy(str, Enum):
    """Explicit policy for multiple takes of the same dynamic condition."""

    ANALYZE_ALL_SEPARATELY = "analyze_all_separately"
    EXPLICIT_REFERENCE = "explicit_reference"
    SELECT_BY_QUALITY_AFTER_ANALYSIS = "select_by_quality_after_analysis"
    COMBINE_SUMMARIES_ONLY = "combine_summaries_only"
    REJECT_MULTIPLE_REPLICATES = "reject_multiple_replicates"


class ExperimentPipelineStage(str, Enum):
    """Stages in the deterministic real-experiment pipeline graph."""

    LOAD = "load"
    VALIDATE_INPUT = "validate_input"
    TEMPORAL = "temporal"
    GLOBAL_SPECTRUM = "global_spectrum"
    STFT = "stft"
    TRACKING = "tracking"
    PREIMPACT = "preimpact"
    EXCITATION = "excitation"
    MODAL_CANDIDATES = "modal_candidates"
    WITHIN_CONDITION = "within_condition"
    DYNAMIC_CONDITION_COMPARISON = "dynamic_condition_comparison"
    CROSS_CONDITION = "cross_condition"
    CANDIDATE_CHAINS = "candidate_chains"
    MODAL_HYPOTHESES = "modal_hypotheses"
    MODAL_PARAMETERS = "modal_parameters"
    MODAL_Q = "modal_q"
    MODAL_ENERGY_EXCHANGE = "modal_energy_exchange"
    SUMMARY = "summary"


class ExperimentPipelineStageStatus(str, Enum):
    """Mutually exclusive terminal status for each pipeline stage."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_RESERVATIONS = "completed_with_reservations"
    SKIPPED = "skipped"
    BLOCKED = "blocked"
    FAILED = "failed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    INVALID_INPUT = "invalid_input"


class ExperimentAnalysisStatus(str, Enum):
    """Mutually exclusive status for a full experiment analysis."""

    COMPLETED = "completed"
    COMPLETED_WITH_RESERVATIONS = "completed_with_reservations"
    PARTIAL = "partial"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    FAILED = "failed"
    INVALID_INPUT = "invalid_input"


EXPERIMENT_PIPELINE_STAGE_ORDER: tuple[ExperimentPipelineStage, ...] = (
    ExperimentPipelineStage.LOAD,
    ExperimentPipelineStage.VALIDATE_INPUT,
    ExperimentPipelineStage.TEMPORAL,
    ExperimentPipelineStage.GLOBAL_SPECTRUM,
    ExperimentPipelineStage.STFT,
    ExperimentPipelineStage.TRACKING,
    ExperimentPipelineStage.PREIMPACT,
    ExperimentPipelineStage.EXCITATION,
    ExperimentPipelineStage.MODAL_CANDIDATES,
    ExperimentPipelineStage.WITHIN_CONDITION,
    ExperimentPipelineStage.DYNAMIC_CONDITION_COMPARISON,
    ExperimentPipelineStage.CROSS_CONDITION,
    ExperimentPipelineStage.CANDIDATE_CHAINS,
    ExperimentPipelineStage.MODAL_HYPOTHESES,
    ExperimentPipelineStage.MODAL_PARAMETERS,
    ExperimentPipelineStage.MODAL_Q,
    ExperimentPipelineStage.MODAL_ENERGY_EXCHANGE,
    ExperimentPipelineStage.SUMMARY,
)

EXPERIMENT_PIPELINE_STAGE_DEPENDENCIES: Mapping[
    ExperimentPipelineStage, tuple[ExperimentPipelineStage, ...]
] = MappingProxyType({
    ExperimentPipelineStage.LOAD: (),
    ExperimentPipelineStage.VALIDATE_INPUT: (),
    ExperimentPipelineStage.TEMPORAL: (ExperimentPipelineStage.LOAD,),
    ExperimentPipelineStage.GLOBAL_SPECTRUM: (ExperimentPipelineStage.LOAD,),
    ExperimentPipelineStage.STFT: (ExperimentPipelineStage.LOAD,),
    ExperimentPipelineStage.TRACKING: (ExperimentPipelineStage.STFT,),
    ExperimentPipelineStage.PREIMPACT: (
        ExperimentPipelineStage.TRACKING,
        ExperimentPipelineStage.TEMPORAL,
    ),
    ExperimentPipelineStage.EXCITATION: (
        ExperimentPipelineStage.LOAD,
        ExperimentPipelineStage.TEMPORAL,
    ),
    ExperimentPipelineStage.MODAL_CANDIDATES: (ExperimentPipelineStage.TRACKING,),
    ExperimentPipelineStage.WITHIN_CONDITION: (
        ExperimentPipelineStage.MODAL_CANDIDATES,
    ),
    ExperimentPipelineStage.DYNAMIC_CONDITION_COMPARISON: (
        ExperimentPipelineStage.GLOBAL_SPECTRUM,
    ),
    ExperimentPipelineStage.CROSS_CONDITION: (
        ExperimentPipelineStage.WITHIN_CONDITION,
    ),
    ExperimentPipelineStage.CANDIDATE_CHAINS: (
        ExperimentPipelineStage.CROSS_CONDITION,
    ),
    ExperimentPipelineStage.MODAL_HYPOTHESES: (
        ExperimentPipelineStage.CANDIDATE_CHAINS,
    ),
    ExperimentPipelineStage.MODAL_PARAMETERS: (
        ExperimentPipelineStage.MODAL_HYPOTHESES,
    ),
    ExperimentPipelineStage.MODAL_Q: (ExperimentPipelineStage.MODAL_PARAMETERS,),
    ExperimentPipelineStage.MODAL_ENERGY_EXCHANGE: (
        ExperimentPipelineStage.TRACKING,
    ),
    ExperimentPipelineStage.SUMMARY: (),
})


@dataclass(frozen=True, slots=True)
class ExperimentRecordingDefinition:
    """Explicit description of one real recording supplied by the user."""

    file_path: str | Path
    dynamic_label: str
    recording_id: str | None = None
    take_index: int = 0
    replicate_group: str | None = None
    specimen_id: str | None = None
    impact_id: str | None = None
    microphone_position: str | None = None
    microphone_distance_m: float | None = None
    microphone_axis: str | None = None
    gain_setting: str | None = None
    channel: int | None = None
    start_offset_s: float | None = None
    end_offset_s: float | None = None
    polarity: int = 1
    enabled: bool = True
    notes: str | None = None
    metadata: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        path = Path(self.file_path)
        if not str(path):
            raise ExperimentDefinitionError("recording file_path must not be empty.")
        object.__setattr__(self, "file_path", path)
        _validate_dynamic_label(self.dynamic_label)
        if self.take_index < 0:
            raise ExperimentDefinitionError("take_index must not be negative.")
        if self.microphone_distance_m is not None and (
            not isfinite(self.microphone_distance_m) or self.microphone_distance_m <= 0.0
        ):
            raise ExperimentDefinitionError(
                "microphone_distance_m must be finite and positive when provided."
            )
        if self.channel is not None and self.channel < 0:
            raise ExperimentDefinitionError("channel must not be negative.")
        _finite_nonnegative_optional(self.start_offset_s, "start_offset_s")
        _finite_nonnegative_optional(self.end_offset_s, "end_offset_s")
        if (
            self.start_offset_s is not None
            and self.end_offset_s is not None
            and self.end_offset_s <= self.start_offset_s
        ):
            raise ExperimentDefinitionError(
                "end_offset_s must be greater than start_offset_s."
            )
        if self.polarity not in {-1, 1}:
            raise ExperimentDefinitionError("polarity must be either 1 or -1.")
        if not isinstance(self.enabled, bool):
            raise ExperimentDefinitionError("enabled must be a boolean.")
        for name in (
            "recording_id",
            "replicate_group",
            "specimen_id",
            "impact_id",
            "microphone_position",
            "microphone_axis",
            "gain_setting",
            "notes",
        ):
            _validate_optional_text(getattr(self, name), name)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        _validate_text_tuple(self.diagnostics, "recording diagnostics")
        if self.recording_id is None:
            object.__setattr__(
                self,
                "recording_id",
                _stable_id(
                    "recording",
                    str(path),
                    self.dynamic_label,
                    self.take_index,
                    self.channel,
                    self.start_offset_s,
                    self.end_offset_s,
                ),
            )


@dataclass(frozen=True, slots=True)
class ExperimentDefinition:
    """Top-level explicit real-experiment description."""

    name: str
    recordings: tuple[ExperimentRecordingDefinition, ...]
    experiment_id: str | None = None
    description: str | None = None
    specimen_id: str | None = None
    instrument_type: str | None = None
    location: str | None = None
    operator: str | None = None
    acquisition_date: str | None = None
    dynamic_labels: tuple[str, ...] = DYNAMIC_LABEL_ORDER
    reference_recording_id: str | None = None
    microphone: str | None = None
    audio_interface: str | None = None
    acquisition_notes: str | None = None
    environment_notes: str | None = None
    expected_sample_rate_hz: int | None = None
    expected_channel_count: int | None = None
    settings: "ExperimentPipelineSettings | None" = None
    metadata: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_text(self.name, "name")
        if not isinstance(self.recordings, tuple):
            object.__setattr__(self, "recordings", tuple(self.recordings))
        if not self.recordings:
            raise ExperimentDefinitionError("experiment requires at least one recording.")
        for recording in self.recordings:
            if not isinstance(recording, ExperimentRecordingDefinition):
                raise ExperimentDefinitionError(
                    "recordings must contain ExperimentRecordingDefinition objects."
                )
        ids = tuple(recording.recording_id for recording in self.recordings)
        if len(ids) != len(set(ids)):
            raise ExperimentDefinitionError("recording IDs must be unique.")
        labels = tuple(self.dynamic_labels)
        if not labels:
            raise ExperimentDefinitionError("dynamic_labels must not be empty.")
        for label in labels:
            _validate_dynamic_label(label)
        if len(labels) != len(set(labels)):
            raise ExperimentDefinitionError("dynamic_labels must be unique.")
        object.__setattr__(self, "dynamic_labels", _canonical_dynamic_labels(labels))
        if self.expected_sample_rate_hz is not None and self.expected_sample_rate_hz <= 0:
            raise ExperimentDefinitionError("expected_sample_rate_hz must be positive.")
        if self.expected_channel_count is not None and self.expected_channel_count <= 0:
            raise ExperimentDefinitionError("expected_channel_count must be positive.")
        for name in (
            "experiment_id",
            "description",
            "specimen_id",
            "instrument_type",
            "location",
            "operator",
            "acquisition_date",
            "reference_recording_id",
            "microphone",
            "audio_interface",
            "acquisition_notes",
            "environment_notes",
        ):
            _validate_optional_text(getattr(self, name), name)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        _validate_text_tuple(self.diagnostics, "experiment diagnostics")
        if self.experiment_id is None:
            object.__setattr__(
                self,
                "experiment_id",
                _stable_id(
                    "experiment",
                    self.name,
                    self.specimen_id,
                    self.dynamic_labels,
                    tuple(
                        (
                            recording.recording_id,
                            str(recording.file_path),
                            recording.dynamic_label,
                            recording.take_index,
                        )
                        for recording in _ordered_recordings(self.recordings)
                    ),
                    self.metadata,
                ),
            )


@dataclass(frozen=True, slots=True)
class ExperimentPipelineSettings:
    """Explicit, deterministic configuration for real-experiment orchestration."""

    run_loading: bool = True
    run_input_validation: bool = True
    run_temporal_analysis: bool = True
    run_global_spectrum: bool = True
    run_stft: bool = True
    run_tracking: bool = True
    run_preimpact_analysis: bool = True
    run_excitation_characterization: bool = True
    run_modal_candidate_characterization: bool = True
    run_within_condition_association: bool = True
    run_dynamic_condition_comparison: bool = True
    run_cross_condition_association: bool = True
    run_candidate_chains: bool = True
    run_modal_hypotheses: bool = True
    run_modal_parameter_estimation: bool = True
    run_modal_q_estimation: bool = True
    run_modal_energy_exchange: bool = True

    continue_after_stage_failure: bool = True
    allow_partial_results: bool = True
    fail_fast_on_invalid_input: bool = False
    reuse_precomputed_results: bool = False
    validate_precomputed_results: bool = True
    preserve_intermediate_results: bool = True
    maximum_worker_count: int = 1
    parallel_execution_policy: str = "sequential"

    require_uniform_sample_rate: bool = True
    require_uniform_channel_count: bool = True
    require_all_dynamic_conditions: bool = False
    allow_missing_dynamic_conditions: bool = True
    allow_duplicate_dynamic_labels: bool = True
    minimum_recording_count: int = 1
    minimum_condition_count: int = 1

    apply_recording_offsets: bool = True
    minimum_analysis_duration_s: float | None = None
    maximum_analysis_duration_s: float | None = None
    reject_empty_trimmed_signal: bool = True

    replicate_policy: ExperimentReplicatePolicy = (
        ExperimentReplicatePolicy.ANALYZE_ALL_SEPARATELY
    )
    minimum_replicates_per_condition: int = 1
    maximum_replicates_per_condition: int | None = None
    replicate_quality_metric: str = "quality_score"
    replicate_selection_policy: str = "highest_quality"

    stage_error_policy: str = "record_and_continue"
    missing_stage_dependency_policy: str = "block"
    unsupported_input_policy: str = "record"

    analysis_settings: AnalysisSettings = AnalysisSettings()
    peak_detection_settings: PeakDetectionSettings = PeakDetectionSettings()
    global_spectrum_settings: GlobalSpectralCharacterizationSettings = (
        GlobalSpectralCharacterizationSettings()
    )
    time_resolved_spectrum_settings: TimeResolvedSpectralCharacterizationSettings = (
        TimeResolvedSpectralCharacterizationSettings()
    )
    modal_candidate_settings: ModalCandidateSettings = ModalCandidateSettings()
    preimpact_settings: PreImpactAnalysisSettings = PreImpactAnalysisSettings()
    excitation_settings: ExcitationCharacterizationSettings = (
        ExcitationCharacterizationSettings()
    )
    dynamic_condition_settings: DynamicConditionComparisonSettings = (
        DynamicConditionComparisonSettings(pair_comparison_policy="adjacent_only")
    )
    within_condition_settings: WithinConditionAssociationSettings = (
        WithinConditionAssociationSettings()
    )
    cross_condition_settings: CrossConditionCandidateAssociationSettings = (
        CrossConditionCandidateAssociationSettings()
    )
    modal_hypothesis_settings: ModalHypothesisSettings = ModalHypothesisSettings()
    modal_parameter_settings: ModalParameterEstimationSettings = (
        ModalParameterEstimationSettings()
    )
    modal_q_settings: ModalQFactorEstimationSettings = ModalQFactorEstimationSettings()
    modal_energy_exchange_settings: ModalEnergyExchangeSettings = (
        ModalEnergyExchangeSettings()
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "replicate_policy",
            _coerce_enum(self.replicate_policy, ExperimentReplicatePolicy),
        )
        for name in (
            "run_loading",
            "run_input_validation",
            "run_temporal_analysis",
            "run_global_spectrum",
            "run_stft",
            "run_tracking",
            "run_preimpact_analysis",
            "run_excitation_characterization",
            "run_modal_candidate_characterization",
            "run_within_condition_association",
            "run_dynamic_condition_comparison",
            "run_cross_condition_association",
            "run_candidate_chains",
            "run_modal_hypotheses",
            "run_modal_parameter_estimation",
            "run_modal_q_estimation",
            "run_modal_energy_exchange",
            "continue_after_stage_failure",
            "allow_partial_results",
            "fail_fast_on_invalid_input",
            "reuse_precomputed_results",
            "validate_precomputed_results",
            "preserve_intermediate_results",
            "require_uniform_sample_rate",
            "require_uniform_channel_count",
            "require_all_dynamic_conditions",
            "allow_missing_dynamic_conditions",
            "allow_duplicate_dynamic_labels",
            "apply_recording_offsets",
            "reject_empty_trimmed_signal",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ExperimentDefinitionError(f"{name} must be a boolean.")
        if self.maximum_worker_count <= 0:
            raise ExperimentDefinitionError("maximum_worker_count must be positive.")
        if self.parallel_execution_policy not in {"sequential", "recordings"}:
            raise ExperimentDefinitionError(
                "parallel_execution_policy must be 'sequential' or 'recordings'."
            )
        if self.stage_error_policy not in {"record_and_continue", "raise"}:
            raise ExperimentDefinitionError(
                "stage_error_policy must be 'record_and_continue' or 'raise'."
            )
        if self.missing_stage_dependency_policy not in {"block", "raise"}:
            raise ExperimentDefinitionError(
                "missing_stage_dependency_policy must be 'block' or 'raise'."
            )
        if self.unsupported_input_policy not in {"record", "raise"}:
            raise ExperimentDefinitionError(
                "unsupported_input_policy must be 'record' or 'raise'."
            )
        if self.replicate_selection_policy not in {"highest_quality", "explicit"}:
            raise ExperimentDefinitionError(
                "replicate_selection_policy must be 'highest_quality' or 'explicit'."
            )
        _validate_text(self.replicate_quality_metric, "replicate_quality_metric")
        if self.require_all_dynamic_conditions and self.allow_missing_dynamic_conditions:
            raise ExperimentDefinitionError(
                "require_all_dynamic_conditions conflicts with allow_missing_dynamic_conditions."
            )
        for name in (
            "minimum_recording_count",
            "minimum_condition_count",
            "minimum_replicates_per_condition",
        ):
            if getattr(self, name) <= 0:
                raise ExperimentDefinitionError(f"{name} must be positive.")
        if self.maximum_replicates_per_condition is not None and (
            self.maximum_replicates_per_condition <= 0
            or self.maximum_replicates_per_condition
            < self.minimum_replicates_per_condition
        ):
            raise ExperimentDefinitionError(
                "maximum_replicates_per_condition must be positive and not below the minimum."
            )
        _finite_positive_optional(
            self.minimum_analysis_duration_s,
            "minimum_analysis_duration_s",
        )
        _finite_positive_optional(
            self.maximum_analysis_duration_s,
            "maximum_analysis_duration_s",
        )
        if (
            self.minimum_analysis_duration_s is not None
            and self.maximum_analysis_duration_s is not None
            and self.maximum_analysis_duration_s < self.minimum_analysis_duration_s
        ):
            raise ExperimentDefinitionError(
                "maximum_analysis_duration_s must not be below minimum_analysis_duration_s."
            )
        self._validate_stage_dependencies()

    def _validate_stage_dependencies(self) -> None:
        enabled = _enabled_stages(self)
        if not self.run_loading and any(
            stage in enabled
            for stage in (
                ExperimentPipelineStage.TEMPORAL,
                ExperimentPipelineStage.GLOBAL_SPECTRUM,
                ExperimentPipelineStage.STFT,
                ExperimentPipelineStage.EXCITATION,
            )
        ) and not self.reuse_precomputed_results:
            raise ExperimentPipelineDependencyError(
                "loaded recordings are required unless compatible precomputed results are reused."
            )
        for stage in EXPERIMENT_PIPELINE_STAGE_ORDER:
            if stage not in enabled:
                continue
            for dependency in EXPERIMENT_PIPELINE_STAGE_DEPENDENCIES[stage]:
                if dependency not in enabled:
                    if self.missing_stage_dependency_policy == "raise":
                        raise ExperimentPipelineDependencyError(
                            f"{stage.value} requires {dependency.value}."
                        )
                    raise ExperimentPipelineDependencyError(
                        f"{stage.value} requires {dependency.value}; disable the dependent stage explicitly."
                    )


@dataclass(frozen=True, slots=True)
class LoadedExperimentRecording:
    """Loaded and explicitly channel-selected recording data."""

    recording_id: str
    file_path: Path
    file_fingerprint: str | None
    original_signal: Signal | None
    signal: Signal | None
    metrics: RecordingMetrics | None
    selected_channel: int | None
    original_duration_s: float | None
    analyzed_duration_s: float | None
    offsets_applied: bool
    valid: bool
    failure_reason: str | None = None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_text(self.recording_id, "recording_id")
        object.__setattr__(self, "file_path", Path(self.file_path))
        if self.selected_channel is not None and self.selected_channel < 0:
            raise ExperimentInputError("selected_channel must not be negative.")
        for name in ("original_duration_s", "analyzed_duration_s"):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or value < 0.0):
                raise ExperimentInputError(f"{name} must be finite and non-negative.")
        if self.valid and self.failure_reason is not None:
            raise ExperimentInputError("valid loaded recording must not have failure_reason.")
        if not self.valid and self.failure_reason is None:
            raise ExperimentInputError("invalid loaded recording requires failure_reason.")
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        _validate_text_tuple(self.diagnostics, "loaded recording diagnostics")


@dataclass(frozen=True, slots=True)
class ExperimentInputValidation:
    """Structured validation of experiment inputs and loaded recordings."""

    recording_count: int
    enabled_recording_count: int
    dynamic_labels_present: tuple[str, ...]
    dynamic_labels_missing: tuple[str, ...]
    duplicate_recording_ids: tuple[str, ...]
    missing_files: tuple[str, ...]
    sample_rates_hz: tuple[int, ...]
    channel_counts: tuple[int, ...]
    durations_s: tuple[float, ...]
    uniform_sample_rate: bool | None
    uniform_channel_count: bool | None
    sufficient_duration: bool | None
    valid: bool
    reasons: tuple[str, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.recording_count < 0 or self.enabled_recording_count < 0:
            raise ExperimentInputError("recording counts must not be negative.")
        if self.enabled_recording_count > self.recording_count:
            raise ExperimentInputError("enabled_recording_count cannot exceed recording_count.")
        object.__setattr__(self, "dynamic_labels_present", tuple(self.dynamic_labels_present))
        object.__setattr__(self, "dynamic_labels_missing", tuple(self.dynamic_labels_missing))
        object.__setattr__(self, "duplicate_recording_ids", tuple(self.duplicate_recording_ids))
        object.__setattr__(self, "missing_files", tuple(self.missing_files))
        object.__setattr__(self, "sample_rates_hz", tuple(self.sample_rates_hz))
        object.__setattr__(self, "channel_counts", tuple(self.channel_counts))
        object.__setattr__(self, "durations_s", tuple(self.durations_s))
        object.__setattr__(self, "reasons", tuple(dict.fromkeys(self.reasons)))
        object.__setattr__(self, "diagnostics", tuple(dict.fromkeys(self.diagnostics)))
        _validate_text_tuple(self.dynamic_labels_present, "dynamic_labels_present")
        _validate_text_tuple(self.dynamic_labels_missing, "dynamic_labels_missing")
        _validate_text_tuple(self.duplicate_recording_ids, "duplicate_recording_ids")
        _validate_text_tuple(self.missing_files, "missing_files")
        _validate_text_tuple(self.reasons, "input validation reasons")
        _validate_text_tuple(self.diagnostics, "input validation diagnostics")


@dataclass(frozen=True, slots=True)
class ExperimentPipelineStageResult:
    """Auditable terminal result for one stage execution."""

    stage: ExperimentPipelineStage
    status: ExperimentPipelineStageStatus
    started: bool
    completed: bool
    input_ids: tuple[str, ...] = ()
    output_ids: tuple[str, ...] = ()
    result: object | None = None
    dependency_stages: tuple[ExperimentPipelineStage, ...] = ()
    supporting_reasons: tuple[str, ...] = ()
    reservation_reasons: tuple[str, ...] = ()
    skipped_reasons: tuple[str, ...] = ()
    blocked_reasons: tuple[str, ...] = ()
    failure_reasons: tuple[str, ...] = ()
    insufficient_evidence_reasons: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", _coerce_enum(self.stage, ExperimentPipelineStage))
        object.__setattr__(
            self,
            "status",
            _coerce_enum(self.status, ExperimentPipelineStageStatus),
        )
        object.__setattr__(self, "input_ids", tuple(self.input_ids))
        object.__setattr__(self, "output_ids", tuple(self.output_ids))
        object.__setattr__(
            self,
            "dependency_stages",
            tuple(_coerce_enum(item, ExperimentPipelineStage) for item in self.dependency_stages),
        )
        for name in (
            "supporting_reasons",
            "reservation_reasons",
            "skipped_reasons",
            "blocked_reasons",
            "failure_reasons",
            "insufficient_evidence_reasons",
            "diagnostics",
        ):
            object.__setattr__(self, name, tuple(dict.fromkeys(getattr(self, name))))
            _validate_text_tuple(getattr(self, name), name)
        if not isinstance(self.started, bool) or not isinstance(self.completed, bool):
            raise ExperimentInputError("started and completed must be booleans.")
        if self.status in {
            ExperimentPipelineStageStatus.COMPLETED,
            ExperimentPipelineStageStatus.COMPLETED_WITH_RESERVATIONS,
            ExperimentPipelineStageStatus.FAILED,
            ExperimentPipelineStageStatus.INSUFFICIENT_EVIDENCE,
            ExperimentPipelineStageStatus.INVALID_INPUT,
        } and not self.completed:
            raise ExperimentInputError("terminal executed stage statuses require completed=True.")
        if self.status in {
            ExperimentPipelineStageStatus.SKIPPED,
            ExperimentPipelineStageStatus.BLOCKED,
        } and self.started:
            raise ExperimentInputError("skipped or blocked stage must not be marked started.")


@dataclass(frozen=True, slots=True)
class ExperimentRecordingAnalysisResult:
    """All available results for one enabled recording."""

    recording_definition: ExperimentRecordingDefinition
    loaded_recording: LoadedExperimentRecording | None = None
    input_validation: ExperimentInputValidation | None = None
    temporal_result: TemporalResults | None = None
    spectral_result: SpectrumResults | None = None
    peak_result: PeakDetectionResults | None = None
    global_spectral_characterization: GlobalSpectralCharacterization | None = None
    stft_result: STFTResults | None = None
    time_frequency_peak_result: TimeFrequencyPeakResults | None = None
    tracking_result: SpectralTrackingResults | None = None
    track_characterizations: tuple[SpectralTrackCharacterization, ...] = ()
    preimpact_result: tuple[PreImpactEvidence, ...] = ()
    excitation_result: ExcitationCharacterization | None = None
    time_resolved_spectral_characterization: TimeResolvedSpectralCharacterization | None = None
    modal_candidate_result: tuple[ModalCandidate, ...] = ()
    recording_candidate_set: RecordingCandidateSet | None = None
    stage_results: tuple[ExperimentPipelineStageResult, ...] = ()
    valid: bool = False
    failure_reason: str | None = None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.recording_definition, ExperimentRecordingDefinition):
            raise ExperimentInputError("recording_definition has invalid type.")
        for name in (
            "track_characterizations",
            "preimpact_result",
            "modal_candidate_result",
            "stage_results",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "diagnostics", tuple(dict.fromkeys(self.diagnostics)))
        _validate_text_tuple(self.diagnostics, "recording analysis diagnostics")
        if self.valid and self.failure_reason is not None:
            raise ExperimentInputError("valid recording result must not have failure_reason.")


@dataclass(frozen=True, slots=True)
class ExperimentReplicateQuality:
    """Auditable score used only to select a reference take when configured."""

    recording_id: str
    dynamic_label: str
    clipping_fraction: float | None
    dynamic_range_db: float | None
    signal_to_noise_proxy: float | None
    tracking_coverage: float | None
    candidate_count: int
    accepted_candidate_count: int
    analysis_duration_s: float | None
    quality_components: Mapping[str, float | None]
    quality_score: float | None
    rank: int | None = None
    selected: bool = False
    reasons: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_text(self.recording_id, "recording_id")
        _validate_dynamic_label(self.dynamic_label)
        _fraction_optional(self.clipping_fraction, "clipping_fraction")
        _finite_nonnegative_optional(self.dynamic_range_db, "dynamic_range_db")
        _finite_optional(self.signal_to_noise_proxy, "signal_to_noise_proxy")
        _fraction_optional(self.tracking_coverage, "tracking_coverage")
        if self.candidate_count < 0 or self.accepted_candidate_count < 0:
            raise ExperimentInputError("candidate counts must not be negative.")
        if self.accepted_candidate_count > self.candidate_count:
            raise ExperimentInputError("accepted_candidate_count cannot exceed candidate_count.")
        _finite_nonnegative_optional(self.analysis_duration_s, "analysis_duration_s")
        object.__setattr__(self, "quality_components", MappingProxyType(dict(self.quality_components)))
        _fraction_optional(self.quality_score, "quality_score")
        if self.rank is not None and self.rank < 0:
            raise ExperimentInputError("rank must not be negative.")
        if not isinstance(self.selected, bool):
            raise ExperimentInputError("selected must be a boolean.")
        object.__setattr__(self, "reasons", tuple(dict.fromkeys(self.reasons)))
        object.__setattr__(self, "diagnostics", tuple(dict.fromkeys(self.diagnostics)))
        _validate_text_tuple(self.reasons, "replicate quality reasons")
        _validate_text_tuple(self.diagnostics, "replicate quality diagnostics")


@dataclass(frozen=True, slots=True)
class ExperimentConditionAnalysisResult:
    """Results grouped by one nominal dynamic condition."""

    dynamic_label: str
    recording_results: tuple[ExperimentRecordingAnalysisResult, ...]
    selected_reference_recording_id: str | None
    replicate_summary: tuple[ExperimentReplicateQuality, ...]
    within_condition_result: WithinConditionAssociationResult | None
    candidate_references: tuple[CandidateReference, ...]
    recording_candidate_sets: tuple[RecordingCandidateSet, ...]
    valid_recording_count: int
    invalid_recording_count: int
    valid: bool
    failure_reason: str | None = None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_dynamic_label(self.dynamic_label)
        for name in (
            "recording_results",
            "replicate_summary",
            "candidate_references",
            "recording_candidate_sets",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.valid_recording_count < 0 or self.invalid_recording_count < 0:
            raise ExperimentInputError("condition recording counts must not be negative.")
        if self.valid and self.failure_reason is not None:
            raise ExperimentInputError("valid condition result must not have failure_reason.")
        object.__setattr__(self, "diagnostics", tuple(dict.fromkeys(self.diagnostics)))
        _validate_text_tuple(self.diagnostics, "condition diagnostics")


@dataclass(frozen=True, slots=True)
class ExperimentCrossConditionAnalysisResult:
    """Results produced between nominally adjacent dynamic conditions only."""

    dynamic_labels: tuple[str, ...]
    adjacent_pair_results: tuple[CrossConditionCandidateAssociationResult, ...]
    comparison_results: DynamicConditionComparisonResult | None = None
    candidate_chain_result: CrossConditionCandidateChainResult | None = None
    candidate_chain_results: tuple[CrossConditionCandidateChainResult, ...] = ()
    modal_hypothesis_result: ModalHypothesisResult | None = None
    modal_hypothesis_results: tuple[ModalHypothesisResult, ...] = ()
    modal_parameter_result: ModalParameterEstimationResult | None = None
    modal_parameter_results: tuple[ModalParameterEstimationResult, ...] = ()
    modal_q_result: ModalQFactorEstimationResult | None = None
    modal_q_results: tuple[ModalQFactorEstimationResult, ...] = ()
    valid: bool = False
    failure_reason: str | None = None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "dynamic_labels", tuple(self.dynamic_labels))
        for label in self.dynamic_labels:
            _validate_dynamic_label(label)
        for name in (
            "adjacent_pair_results",
            "candidate_chain_results",
            "modal_hypothesis_results",
            "modal_parameter_results",
            "modal_q_results",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.valid and self.failure_reason is not None:
            raise ExperimentInputError("valid cross-condition result must not have failure_reason.")
        object.__setattr__(self, "diagnostics", tuple(dict.fromkeys(self.diagnostics)))
        _validate_text_tuple(self.diagnostics, "cross-condition diagnostics")


@dataclass(frozen=True, slots=True)
class ExperimentProvenance:
    """Deterministic audit trail for a real experiment analysis."""

    experiment_id: str
    recording_ids: tuple[str, ...]
    file_paths: tuple[str, ...]
    file_fingerprints: Mapping[str, str | None]
    dynamic_labels: tuple[str, ...]
    selected_replicates: Mapping[str, str | None]
    settings_fingerprint: str
    belllab_version: str
    pipeline_stage_order: tuple[str, ...]
    completed_stages: tuple[str, ...]
    skipped_stages: tuple[str, ...]
    failed_stages: tuple[str, ...]
    input_metadata: Mapping[str, object]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_text(self.experiment_id, "experiment_id")
        for name in (
            "recording_ids",
            "file_paths",
            "dynamic_labels",
            "pipeline_stage_order",
            "completed_stages",
            "skipped_stages",
            "failed_stages",
            "diagnostics",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
            _validate_text_tuple(getattr(self, name), name)
        object.__setattr__(self, "file_fingerprints", MappingProxyType(dict(self.file_fingerprints)))
        object.__setattr__(self, "selected_replicates", MappingProxyType(dict(self.selected_replicates)))
        object.__setattr__(self, "input_metadata", MappingProxyType(dict(self.input_metadata)))


@dataclass(frozen=True, slots=True)
class ExperimentAnalysisResult:
    """Top-level result of a real-experiment BellLab pipeline run."""

    analysis_id: str
    experiment: ExperimentDefinition
    input_validation: ExperimentInputValidation | None
    recording_results: tuple[ExperimentRecordingAnalysisResult, ...]
    condition_results: tuple[ExperimentConditionAnalysisResult, ...]
    cross_condition_result: ExperimentCrossConditionAnalysisResult | None
    energy_exchange_results: tuple[ModalEnergyExchangeResult, ...]
    stage_results: tuple[ExperimentPipelineStageResult, ...]
    provenance: ExperimentProvenance
    completed_stages: tuple[str, ...]
    skipped_stages: tuple[str, ...]
    blocked_stages: tuple[str, ...]
    failed_stages: tuple[str, ...]
    status: ExperimentAnalysisStatus
    valid: bool
    requires_review: bool
    failure_reason: str | None = None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_text(self.analysis_id, "analysis_id")
        object.__setattr__(self, "status", _coerce_enum(self.status, ExperimentAnalysisStatus))
        for name in (
            "recording_results",
            "condition_results",
            "energy_exchange_results",
            "stage_results",
            "completed_stages",
            "skipped_stages",
            "blocked_stages",
            "failed_stages",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "diagnostics", tuple(dict.fromkeys(self.diagnostics)))
        _validate_text_tuple(self.diagnostics, "analysis diagnostics")
        if self.valid != (self.status is ExperimentAnalysisStatus.COMPLETED):
            raise ExperimentInputError("analysis valid flag must reflect completed status only.")
        if self.requires_review != (self.status is not ExperimentAnalysisStatus.COMPLETED):
            raise ExperimentInputError("requires_review must reflect non-completed status.")


def experiment_file_fingerprint(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a deterministic SHA-256 fingerprint of a supplied file."""

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"file not found for fingerprint: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def experiment_settings_fingerprint(
    settings: ExperimentPipelineSettings | None = None,
) -> str:
    """Return a deterministic fingerprint for effective pipeline settings."""

    cfg = settings or ExperimentPipelineSettings()
    payload = json.dumps(_canonicalize(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_experiment_definition(
    experiment: ExperimentDefinition | object,
    settings: ExperimentPipelineSettings | None = None,
    *,
    loaded_recordings: Iterable[LoadedExperimentRecording] | None = None,
) -> ExperimentInputValidation:
    """Validate experiment metadata, paths, labels, and loaded compatibility."""

    cfg = settings or (
        experiment.settings if isinstance(experiment, ExperimentDefinition) and experiment.settings is not None else None
    ) or ExperimentPipelineSettings(
        run_temporal_analysis=False,
        run_global_spectrum=False,
        run_stft=False,
        run_tracking=False,
        run_preimpact_analysis=False,
        run_excitation_characterization=False,
        run_modal_candidate_characterization=False,
        run_within_condition_association=False,
        run_dynamic_condition_comparison=False,
        run_cross_condition_association=False,
        run_candidate_chains=False,
        run_modal_hypotheses=False,
        run_modal_parameter_estimation=False,
        run_modal_q_estimation=False,
        run_modal_energy_exchange=False,
    )
    if not isinstance(experiment, ExperimentDefinition):
        return ExperimentInputValidation(
            recording_count=0,
            enabled_recording_count=0,
            dynamic_labels_present=(),
            dynamic_labels_missing=(),
            duplicate_recording_ids=(),
            missing_files=(),
            sample_rates_hz=(),
            channel_counts=(),
            durations_s=(),
            uniform_sample_rate=None,
            uniform_channel_count=None,
            sufficient_duration=None,
            valid=False,
            reasons=("invalid_experiment_definition",),
            diagnostics=("input_is_not_experiment_definition",),
        )

    enabled = tuple(recording for recording in experiment.recordings if recording.enabled)
    ids = tuple(recording.recording_id or "" for recording in enabled)
    duplicate_ids = tuple(sorted({item for item in ids if ids.count(item) > 1}))
    missing_files = tuple(
        sorted(
            str(recording.file_path)
            for recording in enabled
            if not Path(recording.file_path).is_file()
        )
    )
    labels_present = _canonical_dynamic_labels(
        tuple(recording.dynamic_label for recording in enabled)
    )
    labels_missing = tuple(
        label for label in experiment.dynamic_labels if label not in labels_present
    )
    loaded = tuple(loaded_recordings or ())
    sample_rates = tuple(
        item.metrics.sample_rate_hz
        for item in loaded
        if item.metrics is not None and item.valid
    )
    channel_counts = tuple(
        item.metrics.channel_count
        for item in loaded
        if item.metrics is not None and item.valid
    )
    durations = tuple(
        item.analyzed_duration_s
        for item in loaded
        if item.analyzed_duration_s is not None and item.valid
    )
    uniform_sample_rate = len(set(sample_rates)) <= 1 if sample_rates else None
    uniform_channel_count = len(set(channel_counts)) <= 1 if channel_counts else None
    sufficient_duration = (
        all(
            duration >= cfg.minimum_analysis_duration_s
            for duration in durations
        )
        if durations and cfg.minimum_analysis_duration_s is not None
        else (True if durations else None)
    )

    reasons: list[str] = []
    diagnostics: list[str] = [
        "input_validation_preserves_missing_metadata",
        "dynamic_labels_use_canonical_order",
        "no_audio_file_modified",
    ]
    if duplicate_ids:
        reasons.append("duplicate_recording_ids")
    if missing_files:
        reasons.append("missing_files")
    if len(enabled) < cfg.minimum_recording_count:
        reasons.append("insufficient_recording_count")
    if len(labels_present) < cfg.minimum_condition_count:
        reasons.append("insufficient_condition_count")
    if cfg.require_all_dynamic_conditions and labels_missing:
        reasons.append("missing_required_dynamic_conditions")
    if not cfg.allow_duplicate_dynamic_labels and len(labels_present) != len(enabled):
        reasons.append("duplicate_dynamic_labels_not_allowed")
    if experiment.expected_sample_rate_hz is not None and sample_rates:
        if any(rate != experiment.expected_sample_rate_hz for rate in sample_rates):
            reasons.append("sample_rate_mismatch")
    if experiment.expected_channel_count is not None and channel_counts:
        if any(count != experiment.expected_channel_count for count in channel_counts):
            reasons.append("channel_count_mismatch")
    if cfg.require_uniform_sample_rate and uniform_sample_rate is False:
        reasons.append("nonuniform_sample_rate")
    if cfg.require_uniform_channel_count and uniform_channel_count is False:
        reasons.append("nonuniform_channel_count")
    if sufficient_duration is False:
        reasons.append("insufficient_analysis_duration")
    invalid_loaded = tuple(item for item in loaded if not item.valid)
    if invalid_loaded:
        reasons.append("invalid_loaded_recordings")
        diagnostics.extend(
            f"invalid_loaded_recording:{item.recording_id}:{item.failure_reason}"
            for item in invalid_loaded
        )
    return ExperimentInputValidation(
        recording_count=len(experiment.recordings),
        enabled_recording_count=len(enabled),
        dynamic_labels_present=labels_present,
        dynamic_labels_missing=labels_missing,
        duplicate_recording_ids=duplicate_ids,
        missing_files=missing_files,
        sample_rates_hz=sample_rates,
        channel_counts=channel_counts,
        durations_s=durations,
        uniform_sample_rate=uniform_sample_rate,
        uniform_channel_count=uniform_channel_count,
        sufficient_duration=sufficient_duration,
        valid=not reasons,
        reasons=tuple(reasons),
        diagnostics=tuple(diagnostics),
    )


def load_experiment_recordings(
    experiment: ExperimentDefinition,
    settings: ExperimentPipelineSettings | None = None,
) -> tuple[ExperimentRecordingAnalysisResult, ...]:
    """Load enabled WAV recordings through the canonical BellLab WAV loader."""

    cfg = settings or experiment.settings or ExperimentPipelineSettings()
    results: list[ExperimentRecordingAnalysisResult] = []
    for definition in _ordered_recordings(experiment.recordings):
        if not definition.enabled:
            continue
        stage_results: list[ExperimentPipelineStageResult] = []
        try:
            file_fingerprint = experiment_file_fingerprint(definition.file_path)
            original_signal, metrics = load_wav(definition.file_path)
            loaded = _select_and_trim_recording(
                definition,
                original_signal,
                metrics,
                file_fingerprint,
                cfg,
            )
            load_status = (
                ExperimentPipelineStageStatus.COMPLETED
                if loaded.valid else ExperimentPipelineStageStatus.INVALID_INPUT
            )
            stage = _stage_result(
                ExperimentPipelineStage.LOAD,
                load_status,
                input_ids=(definition.recording_id or "",),
                output_ids=(definition.recording_id or "",) if loaded.valid else (),
                result=loaded,
                supporting=("wav_loaded_with_canonical_loader",) if loaded.valid else (),
                failures=() if loaded.valid else (loaded.failure_reason or "invalid_loaded_recording",),
                diagnostics=loaded.diagnostics,
            )
            valid = loaded.valid
            failure = loaded.failure_reason
        except Exception as exc:
            loaded = LoadedExperimentRecording(
                recording_id=definition.recording_id or "unknown-recording",
                file_path=Path(definition.file_path),
                file_fingerprint=None,
                original_signal=None,
                signal=None,
                metrics=None,
                selected_channel=definition.channel,
                original_duration_s=None,
                analyzed_duration_s=None,
                offsets_applied=False,
                valid=False,
                failure_reason=f"{exc.__class__.__name__}: {exc}",
                diagnostics=("wav_load_failed", "recording_result_preserved"),
            )
            stage = _stage_result(
                ExperimentPipelineStage.LOAD,
                ExperimentPipelineStageStatus.FAILED,
                input_ids=(definition.recording_id or "",),
                result=loaded,
                failures=(loaded.failure_reason or "wav_load_failed",),
                diagnostics=loaded.diagnostics,
            )
            valid = False
            failure = loaded.failure_reason
            if cfg.fail_fast_on_invalid_input or cfg.stage_error_policy == "raise":
                raise ExperimentStageExecutionError(loaded.failure_reason) from exc
        stage_results.append(stage)
        results.append(ExperimentRecordingAnalysisResult(
            recording_definition=definition,
            loaded_recording=loaded,
            stage_results=tuple(stage_results),
            valid=valid,
            failure_reason=failure,
            diagnostics=loaded.diagnostics,
        ))
    return tuple(results)


def analyze_experiment_recording(
    recording_definition: ExperimentRecordingDefinition,
    loaded_recording: LoadedExperimentRecording | None = None,
    settings: ExperimentPipelineSettings | None = None,
    *,
    precomputed_result: ExperimentRecordingAnalysisResult | None = None,
) -> ExperimentRecordingAnalysisResult:
    """Analyze one already loaded recording through configured public stages."""

    cfg = settings or ExperimentPipelineSettings()
    if precomputed_result is not None and cfg.reuse_precomputed_results:
        validate_precomputed_experiment_stage(
            ExperimentPipelineStage.SUMMARY,
            precomputed_result,
            expected_recording_id=recording_definition.recording_id,
            expected_file_fingerprint=(
                loaded_recording.file_fingerprint
                if loaded_recording is not None else None
            ),
        )
        return precomputed_result
    loaded = loaded_recording
    if loaded is None:
        loaded_results = load_experiment_recordings(
            ExperimentDefinition(
                name="single-recording-analysis",
                recordings=(recording_definition,),
                dynamic_labels=(recording_definition.dynamic_label,),
            ),
            cfg,
        )
        return loaded_results[0] if loaded_results else _invalid_recording_result(
            recording_definition,
            "recording_not_loaded",
        )
    if not loaded.valid or loaded.signal is None:
        return ExperimentRecordingAnalysisResult(
            recording_definition=recording_definition,
            loaded_recording=loaded,
            stage_results=(
                _stage_result(
                    ExperimentPipelineStage.LOAD,
                    ExperimentPipelineStageStatus.FAILED,
                    input_ids=(recording_definition.recording_id or "",),
                    result=loaded,
                    failures=(loaded.failure_reason or "invalid_loaded_recording",),
                    diagnostics=loaded.diagnostics,
                ),
            ),
            valid=False,
            failure_reason=loaded.failure_reason or "invalid_loaded_recording",
            diagnostics=loaded.diagnostics,
        )

    stage_results: list[ExperimentPipelineStageResult] = [
        _stage_result(
            ExperimentPipelineStage.LOAD,
            ExperimentPipelineStageStatus.COMPLETED,
            input_ids=(recording_definition.recording_id or "",),
            output_ids=(recording_definition.recording_id or "",),
            result=loaded,
            supporting=("loaded_recording_reused_in_memory",),
            diagnostics=loaded.diagnostics,
        )
    ]
    diagnostics: list[str] = [
        "recording_analysis_uses_public_belllab_stages",
        "no_audio_file_modified",
        "no_resampling_or_downmixing_performed",
    ]
    temporal = _execute_recording_stage(
        ExperimentPipelineStage.TEMPORAL,
        cfg.run_temporal_analysis,
        cfg,
        stage_results,
        lambda: analyze_temporal(loaded.signal, _analysis_settings_for_loaded(cfg, loaded)),
        input_ids=(recording_definition.recording_id or "",),
    )
    spectrum = _execute_recording_stage(
        ExperimentPipelineStage.GLOBAL_SPECTRUM,
        cfg.run_global_spectrum,
        cfg,
        stage_results,
        lambda: analyze_spectrum(loaded.signal, _analysis_settings_for_loaded(cfg, loaded)),
        input_ids=(recording_definition.recording_id or "",),
    )
    peaks = None
    global_characterization = None
    if spectrum is not None:
        peaks = _run_substage(
            "global_peak_detection",
            cfg,
            diagnostics,
            lambda: detect_spectral_peaks(spectrum.spectrum, cfg.peak_detection_settings),
        )
        global_characterization = _run_substage(
            "global_spectral_characterization",
            cfg,
            diagnostics,
            lambda: characterize_signal_spectrum(
                loaded.signal,
                cfg.global_spectrum_settings,
                recording_id=recording_definition.recording_id or "recording",
            ),
        )
    stft = _execute_recording_stage(
        ExperimentPipelineStage.STFT,
        cfg.run_stft,
        cfg,
        stage_results,
        lambda: analyze_stft(loaded.signal, _analysis_settings_for_loaded(cfg, loaded)),
        input_ids=(recording_definition.recording_id or "",),
    )
    frame_peaks = None
    tracking = None
    characterizations: tuple[SpectralTrackCharacterization, ...] = ()
    if stft is not None and cfg.run_tracking:
        frame_peaks = _run_substage(
            "time_frequency_peak_detection",
            cfg,
            diagnostics,
            lambda: detect_stft_peaks(
                stft.time_frequency,
                _analysis_settings_for_loaded(cfg, loaded),
            ),
        )
        if frame_peaks is not None:
            tracking = _run_substage(
                "spectral_tracking",
                cfg,
                diagnostics,
                lambda: track_spectral_peaks(
                    frame_peaks,
                    _analysis_settings_for_loaded(cfg, loaded),
                ),
            )
    elif cfg.run_tracking:
        diagnostics.append("tracking_blocked_by_missing_stft")
    if tracking is not None:
        stage_results.append(_stage_result(
            ExperimentPipelineStage.TRACKING,
            ExperimentPipelineStageStatus.COMPLETED,
            input_ids=(recording_definition.recording_id or "",),
            output_ids=(recording_definition.recording_id or "",),
            result=tracking,
            supporting=("tracking_completed_from_existing_stft_peaks",),
        ))
        characterizations = tuple(
            characterize_spectral_track(track)
            for track in sorted(tracking.tracks, key=lambda item: item.track_id)
        )
    elif cfg.run_tracking:
        stage_results.append(_stage_result(
            ExperimentPipelineStage.TRACKING,
            ExperimentPipelineStageStatus.BLOCKED,
            input_ids=(recording_definition.recording_id or "",),
            blocked=("tracking_requires_stft_and_frame_peaks",),
        ))

    preimpact: tuple[PreImpactEvidence, ...] = ()
    if cfg.run_preimpact_analysis and tracking is not None and temporal is not None:
        preimpact = tuple(
            _run_substage(
                f"preimpact_track_{track.track_id}",
                cfg,
                diagnostics,
                lambda track=track, characterization=characterization: analyze_preimpact_evidence(
                    track,
                    characterization,
                    temporal.impact.impact_time_s,
                    cfg.preimpact_settings,
                ),
            )
            for track, characterization in zip(
                sorted(tracking.tracks, key=lambda item: item.track_id),
                characterizations,
                strict=True,
            )
        )
        preimpact = tuple(item for item in preimpact if item is not None)
        stage_results.append(_stage_result(
            ExperimentPipelineStage.PREIMPACT,
            ExperimentPipelineStageStatus.COMPLETED if preimpact else ExperimentPipelineStageStatus.INSUFFICIENT_EVIDENCE,
            input_ids=(recording_definition.recording_id or "",),
            output_ids=tuple(str(item.source_track_id) for item in preimpact),
            result=preimpact,
            supporting=("preimpact_evidence_from_existing_tracks",) if preimpact else (),
            insufficient=() if preimpact else ("no_track_preimpact_evidence",),
        ))
    elif cfg.run_preimpact_analysis:
        stage_results.append(_stage_result(
            ExperimentPipelineStage.PREIMPACT,
            ExperimentPipelineStageStatus.BLOCKED,
            input_ids=(recording_definition.recording_id or "",),
            blocked=("preimpact_requires_temporal_and_tracking_results",),
        ))

    excitation = None
    if cfg.run_excitation_characterization and temporal is not None:
        condition = _excitation_condition_for_recording(recording_definition, loaded)
        excitation_settings = _excitation_settings_for_loaded(cfg, loaded)
        excitation = _execute_recording_stage(
            ExperimentPipelineStage.EXCITATION,
            True,
            cfg,
            stage_results,
            lambda: characterize_excitation_signal(
                loaded.signal,
                recording_definition.recording_id or "recording",
                condition,
                temporal.impact.impact_time_s,
                excitation_settings,
            ),
            input_ids=(recording_definition.recording_id or "",),
        )
    elif cfg.run_excitation_characterization:
        stage_results.append(_stage_result(
            ExperimentPipelineStage.EXCITATION,
            ExperimentPipelineStageStatus.BLOCKED,
            input_ids=(recording_definition.recording_id or "",),
            blocked=("excitation_requires_temporal_result",),
        ))

    time_resolved = None
    if stft is not None and temporal is not None:
        time_resolved = _run_substage(
            "time_resolved_spectral_characterization",
            cfg,
            diagnostics,
            lambda: characterize_time_resolved_spectrum(
                loaded.signal,
                temporal.impact.impact_time_s,
                cfg.time_resolved_spectrum_settings,
                recording_id=recording_definition.recording_id or "recording",
            ),
        )

    candidates: tuple[ModalCandidate, ...] = ()
    candidate_set = None
    if cfg.run_modal_candidate_characterization and tracking is not None:
        evidence_by_track = {item.source_track_id: item for item in preimpact}
        candidates = _execute_recording_stage(
            ExperimentPipelineStage.MODAL_CANDIDATES,
            True,
            cfg,
            stage_results,
            lambda: select_modal_candidates(
                characterizations,
                tracking,
                cfg.modal_candidate_settings,
                evidence_by_track,
            ),
            input_ids=(recording_definition.recording_id or "",),
        ) or ()
        condition = _excitation_condition_for_recording(recording_definition, loaded)
        candidate_set = RecordingCandidateSet(
            recording_definition.recording_id or "recording",
            condition,
            tuple(candidates),
            preimpact,
        )
    elif cfg.run_modal_candidate_characterization:
        stage_results.append(_stage_result(
            ExperimentPipelineStage.MODAL_CANDIDATES,
            ExperimentPipelineStageStatus.BLOCKED,
            input_ids=(recording_definition.recording_id or "",),
            blocked=("modal_candidates_require_tracking",),
        ))

    stage_failures = tuple(
        reason
        for stage in stage_results
        for reason in stage.failure_reasons + stage.blocked_reasons
        if stage.status in {
            ExperimentPipelineStageStatus.FAILED,
            ExperimentPipelineStageStatus.BLOCKED,
            ExperimentPipelineStageStatus.INVALID_INPUT,
        }
    )
    valid = loaded.valid and not stage_failures
    failure = None if valid else (stage_failures[0] if stage_failures else loaded.failure_reason)
    return ExperimentRecordingAnalysisResult(
        recording_definition=recording_definition,
        loaded_recording=loaded,
        input_validation=None,
        temporal_result=temporal,
        spectral_result=spectrum,
        peak_result=peaks,
        global_spectral_characterization=global_characterization,
        stft_result=stft,
        time_frequency_peak_result=frame_peaks,
        tracking_result=tracking,
        track_characterizations=characterizations,
        preimpact_result=preimpact,
        excitation_result=excitation,
        time_resolved_spectral_characterization=time_resolved,
        modal_candidate_result=tuple(candidates),
        recording_candidate_set=candidate_set,
        stage_results=tuple(stage_results),
        valid=valid,
        failure_reason=failure,
        diagnostics=tuple(diagnostics),
    )


def analyze_experiment_condition(
    dynamic_label: str,
    recording_results: Iterable[ExperimentRecordingAnalysisResult],
    settings: ExperimentPipelineSettings | None = None,
    *,
    reference_recording_id: str | None = None,
) -> ExperimentConditionAnalysisResult:
    """Group analyzed recordings of one condition without waveform averaging."""

    cfg = settings or ExperimentPipelineSettings()
    _validate_dynamic_label(dynamic_label)
    ordered = tuple(sorted(recording_results, key=lambda item: (
        item.recording_definition.take_index,
        item.recording_definition.recording_id or "",
    )))
    if any(item.recording_definition.dynamic_label != dynamic_label for item in ordered):
        raise ExperimentInputError("condition analysis cannot mix dynamic labels.")
    qualities = tuple(
        _replicate_quality(item, cfg)
        for item in ordered
    )
    selected_id, ranked = select_experiment_reference_replicate(
        qualities,
        cfg,
        reference_recording_id=reference_recording_id,
    )
    candidate_sets = tuple(
        item.recording_candidate_set
        for item in ordered
        if item.recording_candidate_set is not None
    )
    candidate_sets = tuple(item for item in candidate_sets if item is not None)
    within = None
    diagnostics = [
        "replicates_analyzed_separately",
        "no_waveform_average_performed",
    ]
    if cfg.replicate_policy is ExperimentReplicatePolicy.REJECT_MULTIPLE_REPLICATES and len(ordered) > 1:
        diagnostics.append("multiple_replicates_rejected_by_policy")
    if cfg.run_within_condition_association:
        if candidate_sets:
            try:
                within = associate_candidates_within_condition(
                    candidate_sets,
                    cfg.within_condition_settings,
                )
            except Exception as exc:
                diagnostics.append(f"within_condition_failed:{exc.__class__.__name__}:{exc}")
        else:
            diagnostics.append("within_condition_requires_candidate_sets")
    else:
        diagnostics.append("within_condition_stage_skipped")
    if within is not None:
        refs = within.candidate_references
    else:
        refs = tuple(
            _candidate_reference_from_set(candidate_set, candidate, None)
            for candidate_set in candidate_sets
            for candidate in candidate_set.candidates
            if candidate.representative_frequency_hz is not None
            and candidate.representative_frequency_hz > 0.0
        )
    valid_count = sum(item.valid for item in ordered)
    invalid_count = len(ordered) - valid_count
    enough_replicates = len(ordered) >= cfg.minimum_replicates_per_condition
    not_too_many = (
        cfg.maximum_replicates_per_condition is None
        or len(ordered) <= cfg.maximum_replicates_per_condition
    )
    valid = (
        bool(ordered)
        and valid_count > 0
        and enough_replicates
        and not_too_many
        and not (
            cfg.replicate_policy is ExperimentReplicatePolicy.REJECT_MULTIPLE_REPLICATES
            and len(ordered) > 1
        )
    )
    failure = None
    if not ordered:
        failure = "missing_condition_recordings"
    elif valid_count == 0:
        failure = "no_valid_recordings_for_condition"
    elif not enough_replicates:
        failure = "insufficient_replicates_for_condition"
    elif not not_too_many:
        failure = "too_many_replicates_for_condition"
    elif cfg.replicate_policy is ExperimentReplicatePolicy.REJECT_MULTIPLE_REPLICATES and len(ordered) > 1:
        failure = "multiple_replicates_rejected_by_policy"
    return ExperimentConditionAnalysisResult(
        dynamic_label=dynamic_label,
        recording_results=ordered,
        selected_reference_recording_id=selected_id,
        replicate_summary=ranked,
        within_condition_result=within,
        candidate_references=tuple(sorted(refs, key=_candidate_reference_key)),
        recording_candidate_sets=candidate_sets,
        valid_recording_count=valid_count,
        invalid_recording_count=invalid_count,
        valid=valid,
        failure_reason=failure,
        diagnostics=tuple(diagnostics),
    )


def select_experiment_reference_replicate(
    replicate_qualities: Iterable[ExperimentReplicateQuality],
    settings: ExperimentPipelineSettings | None = None,
    *,
    reference_recording_id: str | None = None,
) -> tuple[str | None, tuple[ExperimentReplicateQuality, ...]]:
    """Select a reference replicate only under an explicit auditable policy."""

    cfg = settings or ExperimentPipelineSettings()
    qualities = tuple(sorted(
        replicate_qualities,
        key=lambda item: (
            _dynamic_sort_key(item.dynamic_label),
            -(item.quality_score if item.quality_score is not None else -1.0),
            item.recording_id,
        ),
    ))
    if not qualities:
        return None, ()
    selected_id = None
    reasons = ("reference_replicate_not_selected",)
    if cfg.replicate_policy is ExperimentReplicatePolicy.EXPLICIT_REFERENCE:
        selected_id = reference_recording_id
        reasons = ("explicit_reference_recording_selected",)
        if selected_id not in {item.recording_id for item in qualities}:
            selected_id = None
            reasons = ("explicit_reference_recording_missing",)
    elif cfg.replicate_policy in {
        ExperimentReplicatePolicy.SELECT_BY_QUALITY_AFTER_ANALYSIS,
        ExperimentReplicatePolicy.ANALYZE_ALL_SEPARATELY,
        ExperimentReplicatePolicy.COMBINE_SUMMARIES_ONLY,
    }:
        selected_id = qualities[0].recording_id
        reasons = ("highest_quality_reference_selected",)
    ranked = tuple(
        replace(
            item,
            rank=index,
            selected=item.recording_id == selected_id,
            reasons=(
                reasons if item.recording_id == selected_id
                else ("replicate_preserved_not_selected",)
            ),
        )
        for index, item in enumerate(qualities)
    )
    return selected_id, tuple(sorted(ranked, key=lambda item: item.recording_id))


def analyze_experiment_cross_conditions(
    condition_results: Iterable[ExperimentConditionAnalysisResult],
    settings: ExperimentPipelineSettings | None = None,
) -> ExperimentCrossConditionAnalysisResult:
    """Associate only nominally adjacent present dynamic conditions."""

    cfg = settings or ExperimentPipelineSettings()
    ordered = tuple(sorted(condition_results, key=lambda item: _dynamic_sort_key(item.dynamic_label)))
    labels = tuple(item.dynamic_label for item in ordered)
    diagnostics: list[str] = [
        "cross_condition_pipeline_uses_adjacent_condition_association_only",
        "no_non_adjacent_association_created",
        "missing_dynamic_conditions_remain_gaps",
    ]
    dynamic_comparison = None
    analyses = tuple(
        _dynamic_recording_analysis(item)
        for condition in ordered
        for item in condition.recording_results
        if item.excitation_result is not None
        or item.global_spectral_characterization is not None
        or item.time_resolved_spectral_characterization is not None
    )
    if cfg.run_dynamic_condition_comparison and len({item.condition.dynamic_label for item in analyses}) >= 2:
        try:
            dynamic_comparison = compare_dynamic_conditions(analyses, cfg.dynamic_condition_settings)
        except Exception as exc:
            diagnostics.append(f"dynamic_condition_comparison_failed:{exc.__class__.__name__}:{exc}")
    elif cfg.run_dynamic_condition_comparison:
        diagnostics.append("dynamic_condition_comparison_requires_two_conditions")

    by_label = {item.dynamic_label: item for item in ordered}
    adjacent_results: list[CrossConditionCandidateAssociationResult] = []
    run_groups: list[list[CrossConditionCandidateAssociationResult]] = []
    current_group: list[CrossConditionCandidateAssociationResult] = []
    for lower, higher in zip(DYNAMIC_LABEL_ORDER, DYNAMIC_LABEL_ORDER[1:]):
        lower_result = by_label.get(lower)
        higher_result = by_label.get(higher)
        if lower_result is None or higher_result is None:
            if current_group:
                run_groups.append(current_group)
                current_group = []
            if lower_result is not None or higher_result is not None:
                diagnostics.append(f"adjacent_condition_gap:{lower}->{higher}")
            continue
        if not cfg.run_cross_condition_association:
            diagnostics.append("cross_condition_stage_skipped")
            continue
        try:
            pair_result = associate_candidates_across_adjacent_conditions(
                lower_result.candidate_references,
                higher_result.candidate_references,
                cfg.cross_condition_settings,
            )
            adjacent_results.append(pair_result)
            current_group.append(pair_result)
        except Exception as exc:
            diagnostics.append(f"cross_condition_failed:{lower}->{higher}:{exc.__class__.__name__}:{exc}")
            if current_group:
                run_groups.append(current_group)
                current_group = []
    if current_group:
        run_groups.append(current_group)

    chain_results: list[CrossConditionCandidateChainResult] = []
    hypothesis_results: list[ModalHypothesisResult] = []
    parameter_results: list[ModalParameterEstimationResult] = []
    q_results: list[ModalQFactorEstimationResult] = []
    for group in run_groups:
        if not cfg.run_candidate_chains:
            continue
        try:
            chain_result = build_cross_condition_candidate_chains(tuple(group))
            chain_results.append(chain_result)
        except Exception as exc:
            diagnostics.append(f"candidate_chains_failed:{exc.__class__.__name__}:{exc}")
            continue
        if cfg.run_modal_hypotheses:
            try:
                hypothesis_result = build_modal_hypotheses(chain_result, settings=cfg.modal_hypothesis_settings)
                hypothesis_results.append(hypothesis_result)
            except Exception as exc:
                diagnostics.append(f"modal_hypotheses_failed:{exc.__class__.__name__}:{exc}")
                continue
        if cfg.run_modal_parameter_estimation and hypothesis_results:
            try:
                parameter_result = estimate_modal_parameters(
                    hypothesis_results[-1],
                    cfg.modal_parameter_settings,
                )
                parameter_results.append(parameter_result)
            except Exception as exc:
                diagnostics.append(f"modal_parameters_failed:{exc.__class__.__name__}:{exc}")
                continue
        if cfg.run_modal_q_estimation and parameter_results:
            try:
                q_result = estimate_modal_q_factors(
                    parameter_results[-1],
                    cfg.modal_q_settings,
                    bandwidth_sources=_bandwidth_sources_for_parameters(
                        parameter_results[-1],
                        by_label,
                    ),
                )
                q_results.append(q_result)
            except Exception as exc:
                diagnostics.append(f"modal_q_failed:{exc.__class__.__name__}:{exc}")

    valid = bool(adjacent_results or dynamic_comparison or chain_results)
    failure = None if valid else "insufficient_cross_condition_results"
    return ExperimentCrossConditionAnalysisResult(
        dynamic_labels=labels,
        adjacent_pair_results=tuple(adjacent_results),
        comparison_results=dynamic_comparison,
        candidate_chain_result=chain_results[0] if chain_results else None,
        candidate_chain_results=tuple(chain_results),
        modal_hypothesis_result=hypothesis_results[0] if hypothesis_results else None,
        modal_hypothesis_results=tuple(hypothesis_results),
        modal_parameter_result=parameter_results[0] if parameter_results else None,
        modal_parameter_results=tuple(parameter_results),
        modal_q_result=q_results[0] if q_results else None,
        modal_q_results=tuple(q_results),
        valid=valid,
        failure_reason=failure,
        diagnostics=tuple(diagnostics),
    )


def analyze_experiment(
    experiment: ExperimentDefinition,
    settings: ExperimentPipelineSettings | None = None,
    *,
    precomputed_results: Mapping[str, object] | None = None,
) -> ExperimentAnalysisResult:
    """Run the configured real-experiment pipeline and preserve all outcomes."""

    cfg = settings or experiment.settings or ExperimentPipelineSettings()
    initial_validation = validate_experiment_definition(experiment, cfg)
    if not initial_validation.valid and cfg.fail_fast_on_invalid_input:
        raise ExperimentInputError(",".join(initial_validation.reasons))

    loaded_results = load_experiment_recordings(experiment, cfg) if cfg.run_loading else ()
    loaded = tuple(item.loaded_recording for item in loaded_results if item.loaded_recording is not None)
    input_validation = validate_experiment_definition(
        experiment,
        cfg,
        loaded_recordings=loaded,
    ) if cfg.run_input_validation else initial_validation
    if not input_validation.valid and cfg.fail_fast_on_invalid_input:
        raise ExperimentInputError(",".join(input_validation.reasons))

    precomputed_by_recording = _precomputed_recording_results(precomputed_results)
    recording_results: list[ExperimentRecordingAnalysisResult] = []
    loaded_by_id = {
        item.recording_definition.recording_id: item.loaded_recording
        for item in loaded_results
    }
    for definition in _ordered_recordings(experiment.recordings):
        if not definition.enabled:
            continue
        reused = precomputed_by_recording.get(definition.recording_id or "")
        try:
            recording_results.append(analyze_experiment_recording(
                definition,
                loaded_by_id.get(definition.recording_id),
                cfg,
                precomputed_result=reused,
            ))
        except Exception as exc:
            if cfg.stage_error_policy == "raise":
                raise
            recording_results.append(_invalid_recording_result(
                definition,
                f"{exc.__class__.__name__}: {exc}",
            ))

    condition_results = tuple(
        analyze_experiment_condition(
            label,
            tuple(
                item for item in recording_results
                if item.recording_definition.dynamic_label == label
            ),
            cfg,
            reference_recording_id=experiment.reference_recording_id,
        )
        for label in DYNAMIC_LABEL_ORDER
        if any(item.recording_definition.dynamic_label == label for item in recording_results)
    )
    cross_condition = None
    if any(
        (
            cfg.run_dynamic_condition_comparison,
            cfg.run_cross_condition_association,
            cfg.run_candidate_chains,
            cfg.run_modal_hypotheses,
            cfg.run_modal_parameter_estimation,
            cfg.run_modal_q_estimation,
        )
    ):
        cross_condition = analyze_experiment_cross_conditions(condition_results, cfg)

    energy_results: list[ModalEnergyExchangeResult] = []
    if cfg.run_modal_energy_exchange:
        for recording in recording_results:
            if recording.tracking_result is None or len(recording.tracking_result.tracks) < 2:
                continue
            try:
                energy_results.append(evaluate_modal_energy_exchange(
                    recording.tracking_result.tracks,
                    cfg.modal_energy_exchange_settings,
                    dynamic_label=recording.recording_definition.dynamic_label,
                ))
            except Exception:
                if cfg.stage_error_policy == "raise":
                    raise

    stage_results = _experiment_stage_results(
        cfg,
        input_validation,
        recording_results,
        condition_results,
        cross_condition,
        tuple(energy_results),
    )
    completed = tuple(
        item.stage.value
        for item in stage_results
        if item.status in {
            ExperimentPipelineStageStatus.COMPLETED,
            ExperimentPipelineStageStatus.COMPLETED_WITH_RESERVATIONS,
        }
    )
    skipped = tuple(item.stage.value for item in stage_results if item.status is ExperimentPipelineStageStatus.SKIPPED)
    blocked = tuple(item.stage.value for item in stage_results if item.status is ExperimentPipelineStageStatus.BLOCKED)
    failed = tuple(
        item.stage.value
        for item in stage_results
        if item.status in {
            ExperimentPipelineStageStatus.FAILED,
            ExperimentPipelineStageStatus.INVALID_INPUT,
        }
    )
    file_fingerprints = {
        item.recording_definition.recording_id or "": (
            item.loaded_recording.file_fingerprint
            if item.loaded_recording is not None else None
        )
        for item in recording_results
    }
    selected_replicates = {
        item.dynamic_label: item.selected_reference_recording_id
        for item in condition_results
    }
    provenance = ExperimentProvenance(
        experiment_id=experiment.experiment_id or "",
        recording_ids=tuple(sorted(file_fingerprints)),
        file_paths=tuple(
            str(item.recording_definition.file_path)
            for item in sorted(recording_results, key=lambda result: result.recording_definition.recording_id or "")
        ),
        file_fingerprints=file_fingerprints,
        dynamic_labels=tuple(item.dynamic_label for item in condition_results),
        selected_replicates=selected_replicates,
        settings_fingerprint=experiment_settings_fingerprint(cfg),
        belllab_version=_belllab_version(),
        pipeline_stage_order=tuple(stage.value for stage in EXPERIMENT_PIPELINE_STAGE_ORDER),
        completed_stages=completed,
        skipped_stages=skipped,
        failed_stages=failed,
        input_metadata=experiment.metadata,
        diagnostics=(
            "experiment_provenance_is_deterministic",
            "file_fingerprints_use_content_hashes",
            "timestamps_not_used_in_identity",
        ),
    )
    analysis_id = _stable_id(
        "analysis",
        experiment.experiment_id,
        file_fingerprints,
        provenance.settings_fingerprint,
        provenance.belllab_version,
        tuple(stage.value for stage in _enabled_stages(cfg)),
    )
    status = _analysis_status(
        input_validation,
        recording_results,
        condition_results,
        cross_condition,
        tuple(energy_results),
        stage_results,
        cfg,
    )
    diagnostics = (
        "pipeline_success_is_not_physical_validity",
        "results_are_operational_and_conditioned_on_configuration",
        "comparison_between_conditions_is_not_nonlinearity_proof",
        "hypothesis_modal_not_physical_mode_proof",
        "possible_energy_redistribution_not_physical_transfer_proof",
        "no_split_or_merge_resolution",
        "no_non_adjacent_association_created",
        "no_gap_closure_performed",
    )
    return ExperimentAnalysisResult(
        analysis_id=analysis_id,
        experiment=experiment,
        input_validation=input_validation,
        recording_results=tuple(recording_results),
        condition_results=condition_results,
        cross_condition_result=cross_condition,
        energy_exchange_results=tuple(energy_results),
        stage_results=stage_results,
        provenance=provenance,
        completed_stages=completed,
        skipped_stages=skipped,
        blocked_stages=blocked,
        failed_stages=failed,
        status=status,
        valid=status is ExperimentAnalysisStatus.COMPLETED,
        requires_review=status is not ExperimentAnalysisStatus.COMPLETED,
        failure_reason=None if status is ExperimentAnalysisStatus.COMPLETED else status.value,
        diagnostics=diagnostics,
    )


def resume_experiment_analysis(
    previous_result: ExperimentAnalysisResult,
    *,
    experiment: ExperimentDefinition | None = None,
    settings: ExperimentPipelineSettings | None = None,
) -> ExperimentAnalysisResult:
    """Conservatively rerun while offering previous recording results for reuse."""

    cfg = settings or previous_result.experiment.settings or ExperimentPipelineSettings(
        reuse_precomputed_results=True,
    )
    if not cfg.reuse_precomputed_results:
        cfg = replace(cfg, reuse_precomputed_results=True)
    return analyze_experiment(
        experiment or previous_result.experiment,
        cfg,
        precomputed_results={"recording_results": previous_result.recording_results},
    )


def validate_precomputed_experiment_stage(
    stage: ExperimentPipelineStage,
    result: object,
    *,
    expected_recording_id: str | None = None,
    expected_file_fingerprint: str | None = None,
    expected_settings_fingerprint: str | None = None,
    expected_belllab_version: str | None = None,
) -> ExperimentPipelineStageResult:
    """Validate compatibility of a precomputed object before reuse."""

    stage = _coerce_enum(stage, ExperimentPipelineStage)
    diagnostics: list[str] = ["precomputed_result_validated_before_reuse"]
    if isinstance(result, ExperimentRecordingAnalysisResult):
        recording_id = result.recording_definition.recording_id
        if expected_recording_id is not None and recording_id != expected_recording_id:
            raise ExperimentPrecomputedResultError("precomputed recording ID mismatch.")
        if expected_file_fingerprint is not None:
            actual = (
                result.loaded_recording.file_fingerprint
                if result.loaded_recording is not None else None
            )
            if actual != expected_file_fingerprint:
                raise ExperimentPrecomputedResultError("precomputed file fingerprint mismatch.")
        return _stage_result(
            stage,
            ExperimentPipelineStageStatus.COMPLETED_WITH_RESERVATIONS,
            input_ids=(recording_id or "",),
            output_ids=(recording_id or "",),
            result=result,
            reservations=("precomputed_result_reused",),
            diagnostics=tuple(diagnostics),
        )
    if isinstance(result, ExperimentAnalysisResult):
        if expected_settings_fingerprint is not None and (
            result.provenance.settings_fingerprint != expected_settings_fingerprint
        ):
            raise ExperimentPrecomputedResultError("precomputed settings fingerprint mismatch.")
        if expected_belllab_version is not None and (
            result.provenance.belllab_version != expected_belllab_version
        ):
            raise ExperimentPrecomputedResultError("precomputed BellLab version mismatch.")
        return _stage_result(
            stage,
            ExperimentPipelineStageStatus.COMPLETED_WITH_RESERVATIONS,
            input_ids=(result.analysis_id,),
            output_ids=(result.analysis_id,),
            result=result,
            reservations=("precomputed_analysis_result_reused",),
            diagnostics=tuple(diagnostics),
        )
    if isinstance(result, ExperimentPipelineStageResult):
        if result.stage is not stage:
            raise ExperimentPrecomputedResultError("precomputed stage mismatch.")
        if result.status not in {
            ExperimentPipelineStageStatus.COMPLETED,
            ExperimentPipelineStageStatus.COMPLETED_WITH_RESERVATIONS,
        }:
            raise ExperimentPrecomputedResultError("precomputed stage is not complete.")
        return result
    raise ExperimentPrecomputedResultError("unsupported precomputed result object.")


def summarize_experiment_analysis(result: ExperimentAnalysisResult) -> dict[str, object]:
    """Return a compact deterministic structured summary for applications."""

    if not isinstance(result, ExperimentAnalysisResult):
        raise ExperimentInputError("result must be an ExperimentAnalysisResult.")
    return {
        "analysis_id": result.analysis_id,
        "experiment_id": result.experiment.experiment_id,
        "name": result.experiment.name,
        "status": result.status.value,
        "valid": result.valid,
        "requires_review": result.requires_review,
        "recording_count": len(result.recording_results),
        "condition_count": len(result.condition_results),
        "dynamic_labels": tuple(item.dynamic_label for item in result.condition_results),
        "completed_stages": result.completed_stages,
        "skipped_stages": result.skipped_stages,
        "blocked_stages": result.blocked_stages,
        "failed_stages": result.failed_stages,
        "candidate_count": sum(len(item.modal_candidate_result) for item in result.recording_results),
        "accepted_candidate_count": sum(
            sum(candidate.accepted for candidate in item.modal_candidate_result)
            for item in result.recording_results
        ),
        "chain_count": sum(
            chain_result.chain_count
            for chain_result in (
                result.cross_condition_result.candidate_chain_results
                if result.cross_condition_result is not None else ()
            )
        ),
        "hypothesis_count": sum(
            hypothesis_result.hypothesis_count
            for hypothesis_result in (
                result.cross_condition_result.modal_hypothesis_results
                if result.cross_condition_result is not None else ()
            )
        ),
        "parameter_estimate_count": sum(
            parameter_result.estimate_count
            for parameter_result in (
                result.cross_condition_result.modal_parameter_results
                if result.cross_condition_result is not None else ()
            )
        ),
        "q_estimate_count": sum(
            q_result.estimate_count
            for q_result in (
                result.cross_condition_result.modal_q_results
                if result.cross_condition_result is not None else ()
            )
        ),
        "energy_pair_count": sum(item.pair_count for item in result.energy_exchange_results),
        "failure_reason": result.failure_reason,
        "provenance": {
            "settings_fingerprint": result.provenance.settings_fingerprint,
            "belllab_version": result.provenance.belllab_version,
            "file_fingerprints": dict(result.provenance.file_fingerprints),
        },
    }


def _select_and_trim_recording(
    definition: ExperimentRecordingDefinition,
    original: Signal,
    metrics: RecordingMetrics,
    file_fingerprint: str,
    cfg: ExperimentPipelineSettings,
) -> LoadedExperimentRecording:
    diagnostics = [
        "wav_loaded_by_belllab_io_load_wav",
        "original_signal_preserved",
        "no_audio_file_modified",
    ]
    channel = definition.channel if definition.channel is not None else 0
    if definition.channel is None and original.channels > 1:
        diagnostics.append("channel_defaulted_to_zero_no_downmix")
    if channel >= original.channels:
        return LoadedExperimentRecording(
            recording_id=definition.recording_id or "recording",
            file_path=Path(definition.file_path),
            file_fingerprint=file_fingerprint,
            original_signal=original,
            signal=None,
            metrics=metrics,
            selected_channel=channel,
            original_duration_s=original.duration,
            analyzed_duration_s=None,
            offsets_applied=False,
            valid=False,
            failure_reason="channel_index_outside_signal",
            diagnostics=tuple(diagnostics + ["invalid_channel"]),
        )
    selected = tuple(original.samples[channel])
    if definition.polarity == -1:
        selected = tuple(-value for value in selected)
    start_s = definition.start_offset_s or 0.0
    end_s = definition.end_offset_s if definition.end_offset_s is not None else original.duration
    offsets_applied = False
    if cfg.apply_recording_offsets:
        if cfg.maximum_analysis_duration_s is not None:
            end_s = min(end_s, start_s + cfg.maximum_analysis_duration_s)
        start_index = int(round(start_s * original.sample_rate))
        end_index = int(round(end_s * original.sample_rate))
        if start_index < 0 or end_index > len(selected) or end_index <= start_index:
            return LoadedExperimentRecording(
                recording_id=definition.recording_id or "recording",
                file_path=Path(definition.file_path),
                file_fingerprint=file_fingerprint,
                original_signal=original,
                signal=None,
                metrics=metrics,
                selected_channel=channel,
                original_duration_s=original.duration,
                analyzed_duration_s=None,
                offsets_applied=False,
                valid=False,
                failure_reason="invalid_recording_offsets",
                diagnostics=tuple(diagnostics + ["invalid_recording_offsets"]),
            )
        if start_index or end_index != len(selected):
            offsets_applied = True
            diagnostics.append("recording_offsets_applied")
        selected = selected[start_index:end_index]
    elif definition.start_offset_s is not None or definition.end_offset_s is not None:
        diagnostics.append("recording_offsets_configured_but_not_applied")
    if not selected and cfg.reject_empty_trimmed_signal:
        return LoadedExperimentRecording(
            recording_id=definition.recording_id or "recording",
            file_path=Path(definition.file_path),
            file_fingerprint=file_fingerprint,
            original_signal=original,
            signal=None,
            metrics=metrics,
            selected_channel=channel,
            original_duration_s=original.duration,
            analyzed_duration_s=0.0,
            offsets_applied=offsets_applied,
            valid=False,
            failure_reason="empty_analysis_signal",
            diagnostics=tuple(diagnostics + ["empty_analysis_signal"]),
        )
    duration = len(selected) / original.sample_rate
    if cfg.minimum_analysis_duration_s is not None and duration < cfg.minimum_analysis_duration_s:
        diagnostics.append("analysis_duration_below_minimum")
        valid = False
        failure = "analysis_duration_below_minimum"
    else:
        valid = True
        failure = None
    signal = Signal(
        samples=(selected,),
        sample_rate=original.sample_rate,
        time=tuple(float(index / original.sample_rate) for index in range(len(selected))),
        duration=duration,
        channels=1,
        unit=original.unit,
        path=Path(definition.file_path),
        filename=Path(definition.file_path).name,
        sha256=file_fingerprint,
        loaded_at=None,
    )
    diagnostics.append(f"selected_channel:{channel}")
    return LoadedExperimentRecording(
        recording_id=definition.recording_id or "recording",
        file_path=Path(definition.file_path),
        file_fingerprint=file_fingerprint,
        original_signal=original,
        signal=signal,
        metrics=metrics,
        selected_channel=channel,
        original_duration_s=original.duration,
        analyzed_duration_s=duration,
        offsets_applied=offsets_applied,
        valid=valid,
        failure_reason=failure,
        diagnostics=tuple(diagnostics),
    )


def _execute_recording_stage(
    stage: ExperimentPipelineStage,
    enabled: bool,
    cfg: ExperimentPipelineSettings,
    stage_results: list[ExperimentPipelineStageResult],
    fn: Any,
    *,
    input_ids: tuple[str, ...] = (),
) -> Any:
    if not enabled:
        stage_results.append(_stage_result(
            stage,
            ExperimentPipelineStageStatus.SKIPPED,
            input_ids=input_ids,
            skipped=(f"{stage.value}_disabled",),
        ))
        return None
    try:
        result = fn()
    except Exception as exc:
        failure = f"{exc.__class__.__name__}: {exc}"
        stage_results.append(_stage_result(
            stage,
            ExperimentPipelineStageStatus.FAILED,
            input_ids=input_ids,
            failures=(failure,),
        ))
        if cfg.stage_error_policy == "raise" or not cfg.continue_after_stage_failure:
            raise ExperimentStageExecutionError(failure) from exc
        return None
    status = (
        ExperimentPipelineStageStatus.COMPLETED
        if result is not None else ExperimentPipelineStageStatus.INSUFFICIENT_EVIDENCE
    )
    stage_results.append(_stage_result(
        stage,
        status,
        input_ids=input_ids,
        output_ids=input_ids,
        result=result,
        supporting=(f"{stage.value}_completed",) if result is not None else (),
        insufficient=() if result is not None else (f"{stage.value}_returned_none",),
    ))
    return result


def _run_substage(
    name: str,
    cfg: ExperimentPipelineSettings,
    diagnostics: list[str],
    fn: Any,
) -> Any:
    try:
        return fn()
    except Exception as exc:
        diagnostics.append(f"{name}_failed:{exc.__class__.__name__}:{exc}")
        if cfg.stage_error_policy == "raise" or not cfg.continue_after_stage_failure:
            raise
        return None


def _experiment_stage_results(
    cfg: ExperimentPipelineSettings,
    input_validation: ExperimentInputValidation | None,
    recording_results: tuple[ExperimentRecordingAnalysisResult, ...] | list[ExperimentRecordingAnalysisResult],
    condition_results: tuple[ExperimentConditionAnalysisResult, ...],
    cross_condition: ExperimentCrossConditionAnalysisResult | None,
    energy_results: tuple[ModalEnergyExchangeResult, ...],
) -> tuple[ExperimentPipelineStageResult, ...]:
    enabled = _enabled_stages(cfg)
    results: list[ExperimentPipelineStageResult] = []
    for stage in EXPERIMENT_PIPELINE_STAGE_ORDER:
        if stage not in enabled and stage is not ExperimentPipelineStage.SUMMARY:
            results.append(_stage_result(
                stage,
                ExperimentPipelineStageStatus.SKIPPED,
                skipped=(f"{stage.value}_disabled",),
            ))
            continue
        if stage is ExperimentPipelineStage.LOAD:
            loaded_count = sum(item.loaded_recording is not None and item.loaded_recording.valid for item in recording_results)
            failed_count = sum(item.loaded_recording is not None and not item.loaded_recording.valid for item in recording_results)
            status = (
                ExperimentPipelineStageStatus.COMPLETED
                if loaded_count and not failed_count
                else ExperimentPipelineStageStatus.COMPLETED_WITH_RESERVATIONS
                if loaded_count and failed_count
                else ExperimentPipelineStageStatus.FAILED
            )
            results.append(_stage_result(
                stage,
                status,
                output_ids=tuple(
                    item.recording_definition.recording_id or ""
                    for item in recording_results
                    if item.loaded_recording is not None and item.loaded_recording.valid
                ),
                failures=() if status is not ExperimentPipelineStageStatus.FAILED else ("no_recordings_loaded",),
                reservations=("some_recordings_failed_loading",) if failed_count and loaded_count else (),
            ))
        elif stage is ExperimentPipelineStage.VALIDATE_INPUT:
            valid = input_validation.valid if input_validation is not None else False
            results.append(_stage_result(
                stage,
                ExperimentPipelineStageStatus.COMPLETED if valid else ExperimentPipelineStageStatus.INVALID_INPUT,
                result=input_validation,
                failures=() if valid else tuple(input_validation.reasons if input_validation is not None else ("missing_input_validation",)),
                diagnostics=input_validation.diagnostics if input_validation is not None else (),
            ))
        elif stage in {
            ExperimentPipelineStage.TEMPORAL,
            ExperimentPipelineStage.GLOBAL_SPECTRUM,
            ExperimentPipelineStage.STFT,
            ExperimentPipelineStage.TRACKING,
            ExperimentPipelineStage.PREIMPACT,
            ExperimentPipelineStage.EXCITATION,
            ExperimentPipelineStage.MODAL_CANDIDATES,
        }:
            stage_name = stage.value
            matching = tuple(
                item
                for recording in recording_results
                for item in recording.stage_results
                if item.stage is stage
            )
            completed = sum(
                item.status in {
                    ExperimentPipelineStageStatus.COMPLETED,
                    ExperimentPipelineStageStatus.COMPLETED_WITH_RESERVATIONS,
                }
                for item in matching
            )
            failed = sum(
                item.status in {
                    ExperimentPipelineStageStatus.FAILED,
                    ExperimentPipelineStageStatus.BLOCKED,
                    ExperimentPipelineStageStatus.INVALID_INPUT,
                }
                for item in matching
            )
            if completed and not failed:
                status = ExperimentPipelineStageStatus.COMPLETED
            elif completed and failed:
                status = ExperimentPipelineStageStatus.COMPLETED_WITH_RESERVATIONS
            elif failed:
                status = ExperimentPipelineStageStatus.BLOCKED if any(item.status is ExperimentPipelineStageStatus.BLOCKED for item in matching) else ExperimentPipelineStageStatus.FAILED
            else:
                status = ExperimentPipelineStageStatus.INSUFFICIENT_EVIDENCE
            results.append(_stage_result(
                stage,
                status,
                output_ids=tuple(
                    recording.recording_definition.recording_id or ""
                    for recording in recording_results
                    if any(item.stage is stage and item.status is ExperimentPipelineStageStatus.COMPLETED for item in recording.stage_results)
                ),
                failures=tuple(reason for item in matching for reason in item.failure_reasons),
                blocked=tuple(reason for item in matching for reason in item.blocked_reasons),
                reservations=(f"{stage_name}_partial_across_recordings",) if completed and failed else (),
            ))
        elif stage is ExperimentPipelineStage.WITHIN_CONDITION:
            completed = sum(item.within_condition_result is not None for item in condition_results)
            status = ExperimentPipelineStageStatus.COMPLETED if completed else ExperimentPipelineStageStatus.INSUFFICIENT_EVIDENCE
            results.append(_stage_result(stage, status, output_ids=tuple(item.dynamic_label for item in condition_results if item.within_condition_result is not None)))
        elif stage is ExperimentPipelineStage.DYNAMIC_CONDITION_COMPARISON:
            status = (
                ExperimentPipelineStageStatus.COMPLETED
                if cross_condition is not None and cross_condition.comparison_results is not None
                else ExperimentPipelineStageStatus.INSUFFICIENT_EVIDENCE
            )
            results.append(_stage_result(stage, status, result=cross_condition.comparison_results if cross_condition is not None else None))
        elif stage is ExperimentPipelineStage.CROSS_CONDITION:
            status = (
                ExperimentPipelineStageStatus.COMPLETED
                if cross_condition is not None and cross_condition.adjacent_pair_results
                else ExperimentPipelineStageStatus.INSUFFICIENT_EVIDENCE
            )
            results.append(_stage_result(stage, status, result=cross_condition))
        elif stage is ExperimentPipelineStage.CANDIDATE_CHAINS:
            status = (
                ExperimentPipelineStageStatus.COMPLETED
                if cross_condition is not None and cross_condition.candidate_chain_results
                else ExperimentPipelineStageStatus.INSUFFICIENT_EVIDENCE
            )
            results.append(_stage_result(stage, status, result=cross_condition.candidate_chain_results if cross_condition is not None else None))
        elif stage is ExperimentPipelineStage.MODAL_HYPOTHESES:
            status = (
                ExperimentPipelineStageStatus.COMPLETED
                if cross_condition is not None and cross_condition.modal_hypothesis_results
                else ExperimentPipelineStageStatus.INSUFFICIENT_EVIDENCE
            )
            results.append(_stage_result(stage, status, result=cross_condition.modal_hypothesis_results if cross_condition is not None else None))
        elif stage is ExperimentPipelineStage.MODAL_PARAMETERS:
            status = (
                ExperimentPipelineStageStatus.COMPLETED
                if cross_condition is not None and cross_condition.modal_parameter_results
                else ExperimentPipelineStageStatus.INSUFFICIENT_EVIDENCE
            )
            results.append(_stage_result(stage, status, result=cross_condition.modal_parameter_results if cross_condition is not None else None))
        elif stage is ExperimentPipelineStage.MODAL_Q:
            status = (
                ExperimentPipelineStageStatus.COMPLETED
                if cross_condition is not None and cross_condition.modal_q_results
                else ExperimentPipelineStageStatus.INSUFFICIENT_EVIDENCE
            )
            results.append(_stage_result(stage, status, result=cross_condition.modal_q_results if cross_condition is not None else None))
        elif stage is ExperimentPipelineStage.MODAL_ENERGY_EXCHANGE:
            status = (
                ExperimentPipelineStageStatus.COMPLETED
                if energy_results else ExperimentPipelineStageStatus.INSUFFICIENT_EVIDENCE
            )
            results.append(_stage_result(stage, status, result=energy_results))
        elif stage is ExperimentPipelineStage.SUMMARY:
            results.append(_stage_result(
                stage,
                ExperimentPipelineStageStatus.COMPLETED,
                supporting=("summary_built_from_structured_results",),
            ))
    return tuple(results)


def _stage_result(
    stage: ExperimentPipelineStage,
    status: ExperimentPipelineStageStatus,
    *,
    input_ids: tuple[str, ...] = (),
    output_ids: tuple[str, ...] = (),
    result: object | None = None,
    dependency_stages: tuple[ExperimentPipelineStage, ...] | None = None,
    supporting: tuple[str, ...] = (),
    reservations: tuple[str, ...] = (),
    skipped: tuple[str, ...] = (),
    blocked: tuple[str, ...] = (),
    failures: tuple[str, ...] = (),
    insufficient: tuple[str, ...] = (),
    diagnostics: tuple[str, ...] = (),
) -> ExperimentPipelineStageResult:
    return ExperimentPipelineStageResult(
        stage=stage,
        status=status,
        started=status not in {
            ExperimentPipelineStageStatus.SKIPPED,
            ExperimentPipelineStageStatus.BLOCKED,
        },
        completed=status not in {
            ExperimentPipelineStageStatus.SKIPPED,
            ExperimentPipelineStageStatus.BLOCKED,
            ExperimentPipelineStageStatus.PENDING,
            ExperimentPipelineStageStatus.RUNNING,
        },
        input_ids=tuple(sorted(input_ids)),
        output_ids=tuple(sorted(output_ids)),
        result=result,
        dependency_stages=dependency_stages
        if dependency_stages is not None
        else EXPERIMENT_PIPELINE_STAGE_DEPENDENCIES[stage],
        supporting_reasons=supporting,
        reservation_reasons=reservations,
        skipped_reasons=skipped,
        blocked_reasons=blocked,
        failure_reasons=failures,
        insufficient_evidence_reasons=insufficient,
        diagnostics=diagnostics,
    )


def _analysis_settings_for_loaded(
    cfg: ExperimentPipelineSettings,
    loaded: LoadedExperimentRecording,
) -> AnalysisSettings:
    return AnalysisSettings(
        temporal=cfg.analysis_settings.temporal,
        spectrum=replace(
            cfg.analysis_settings.spectrum,
            channel_policy="select",
            channel_index=0,
        ),
        stft=replace(
            cfg.analysis_settings.stft,
            channel_policy="select",
            channel_index=0,
        ),
        frame_peaks=cfg.analysis_settings.frame_peaks,
        tracking=cfg.analysis_settings.tracking,
    )


def _excitation_settings_for_loaded(
    cfg: ExperimentPipelineSettings,
    loaded: LoadedExperimentRecording,
) -> ExcitationCharacterizationSettings:
    return replace(cfg.excitation_settings, channel_index=0)


def _excitation_condition_for_recording(
    recording: ExperimentRecordingDefinition,
    loaded: LoadedExperimentRecording,
) -> ExcitationCondition:
    return ExcitationCondition(
        dynamic_label=recording.dynamic_label,
        repeat_index=recording.take_index,
        amplitude_unit=loaded.signal.unit if loaded.signal is not None else None,
        session_id=recording.replicate_group,
        microphone_id=recording.microphone_position,
        interface_id=recording.gain_setting,
        channel=0 if loaded.selected_channel is not None else None,
        microphone_distance_m=recording.microphone_distance_m,
        microphone_orientation=recording.microphone_axis,
        operator_label=None,
        notes=recording.notes,
        diagnostics=("experiment_recording_condition",),
    )


def _candidate_reference_from_set(
    candidate_set: RecordingCandidateSet,
    candidate: ModalCandidate,
    evidence: PreImpactEvidence | None,
) -> CandidateReference:
    return CandidateReference(
        candidate_set.recording_id,
        candidate.candidate_id,
        candidate.source_track_id,
        candidate.representative_frequency_hz or 0.0,
        candidate_set.condition.dynamic_label,
        candidate.accepted,
        evidence.impact_excited if evidence is not None else None,
        evidence.classification if evidence is not None else None,
        candidate.characterization.relative_frequency_stability,
        candidate.characterization.amplitude_fit.tau_s,
        candidate.characterization.amplitude_fit.r_squared,
        candidate.characterization.frequency_total_drift_hz,
        candidate.characterization.frequency_fit.rmse_hz,
        candidate.characterization.coverage_fraction,
        candidate.ambiguous_assignment_fraction,
        candidate.near_threshold_assignment_fraction,
        candidate.minimum_assignment_margin,
        evidence.classification if evidence is not None else None,
        None,
        ("experiment_pipeline_candidate_reference",),
    )


def _replicate_quality(
    result: ExperimentRecordingAnalysisResult,
    cfg: ExperimentPipelineSettings,
) -> ExperimentReplicateQuality:
    loaded = result.loaded_recording
    metrics = loaded.metrics if loaded is not None else None
    clipping = (
        metrics.clipping_fraction
        if metrics is not None and metrics.clipping_fraction is not None
        else (1.0 if metrics is not None and metrics.clipping_detected else 0.0 if metrics is not None else None)
    )
    dynamic_range = None
    if metrics is not None and metrics.max_level_dbfs is not None:
        dynamic_range = max(0.0, -float(metrics.max_level_dbfs)) if isfinite(metrics.max_level_dbfs) else None
    snr = result.temporal_result.noise.signal_to_noise_ratio_db if result.temporal_result is not None else None
    coverages = tuple(
        candidate.characterization.coverage_fraction
        for candidate in result.modal_candidate_result
        if candidate.characterization.coverage_fraction is not None
    )
    tracking_coverage = float(sum(coverages) / len(coverages)) if coverages else None
    accepted = sum(candidate.accepted for candidate in result.modal_candidate_result)
    duration = loaded.analyzed_duration_s if loaded is not None else None
    components = {
        "not_clipped": None if clipping is None else max(0.0, 1.0 - clipping),
        "duration": _duration_quality(duration, cfg.minimum_analysis_duration_s),
        "snr": _snr_quality(snr),
        "tracking_coverage": tracking_coverage,
        "accepted_candidate_fraction": (
            accepted / len(result.modal_candidate_result)
            if result.modal_candidate_result else None
        ),
    }
    finite_components = tuple(value for value in components.values() if value is not None)
    score = float(sum(finite_components) / len(finite_components)) if finite_components else None
    reasons = ("replicate_quality_is_operational",)
    if clipping and clipping > 0:
        reasons += ("clipping_present",)
    if duration is not None and cfg.minimum_analysis_duration_s is not None and duration < cfg.minimum_analysis_duration_s:
        reasons += ("short_duration",)
    return ExperimentReplicateQuality(
        recording_id=result.recording_definition.recording_id or "",
        dynamic_label=result.recording_definition.dynamic_label,
        clipping_fraction=clipping,
        dynamic_range_db=dynamic_range,
        signal_to_noise_proxy=snr,
        tracking_coverage=tracking_coverage,
        candidate_count=len(result.modal_candidate_result),
        accepted_candidate_count=accepted,
        analysis_duration_s=duration,
        quality_components=components,
        quality_score=score,
        reasons=reasons,
        diagnostics=("no_reference_selected_by_volume_alone",),
    )


def _dynamic_recording_analysis(
    result: ExperimentRecordingAnalysisResult,
) -> DynamicConditionRecordingAnalysis:
    loaded = result.loaded_recording
    condition = _excitation_condition_for_recording(
        result.recording_definition,
        loaded if loaded is not None else LoadedExperimentRecording(
            result.recording_definition.recording_id or "",
            Path(result.recording_definition.file_path),
            None,
            None,
            None,
            None,
            result.recording_definition.channel,
            None,
            None,
            False,
            False,
            "not_loaded",
        ),
    )
    return DynamicConditionRecordingAnalysis(
        recording_id=result.recording_definition.recording_id or "",
        condition=condition,
        excitation=result.excitation_result,
        global_spectrum=result.global_spectral_characterization,
        time_resolved=result.time_resolved_spectral_characterization,
        diagnostics=("experiment_pipeline_dynamic_recording_analysis",),
    )


def _bandwidth_sources_for_parameters(
    parameters: ModalParameterEstimationResult,
    condition_results: Mapping[str, ExperimentConditionAnalysisResult],
) -> Mapping[str, object]:
    sources: dict[str, object] = {}
    spectra_by_label: dict[str, object] = {}
    for label, condition in condition_results.items():
        selected = condition.selected_reference_recording_id
        chosen = next(
            (
                item
                for item in condition.recording_results
                if item.recording_definition.recording_id == selected
            ),
            condition.recording_results[0] if condition.recording_results else None,
        )
        if chosen is not None and chosen.spectral_result is not None:
            spectra_by_label[label] = chosen.spectral_result.spectrum
    for estimate in parameters.estimates:
        labels = estimate.provenance.condition_labels
        source = next((spectra_by_label[label] for label in labels if label in spectra_by_label), None)
        if source is not None:
            sources[estimate.estimate_id] = source
            sources[estimate.hypothesis_id] = source
    return MappingProxyType(sources)


def _analysis_status(
    input_validation: ExperimentInputValidation,
    recording_results: tuple[ExperimentRecordingAnalysisResult, ...] | list[ExperimentRecordingAnalysisResult],
    condition_results: tuple[ExperimentConditionAnalysisResult, ...],
    cross_condition: ExperimentCrossConditionAnalysisResult | None,
    energy_results: tuple[ModalEnergyExchangeResult, ...],
    stage_results: tuple[ExperimentPipelineStageResult, ...],
    cfg: ExperimentPipelineSettings,
) -> ExperimentAnalysisStatus:
    if not input_validation.valid:
        return ExperimentAnalysisStatus.INVALID_INPUT
    valid_recordings = tuple(item for item in recording_results if item.loaded_recording is not None and item.loaded_recording.valid)
    if not valid_recordings:
        return ExperimentAnalysisStatus.INSUFFICIENT_EVIDENCE
    if any(item.status is ExperimentPipelineStageStatus.FAILED for item in stage_results):
        return ExperimentAnalysisStatus.FAILED
    if any(item.status is ExperimentPipelineStageStatus.BLOCKED for item in stage_results):
        return ExperimentAnalysisStatus.PARTIAL
    if not cfg.allow_partial_results and any(
        item.status is ExperimentPipelineStageStatus.INSUFFICIENT_EVIDENCE
        for item in stage_results
        if item.stage in _enabled_stages(cfg)
    ):
        return ExperimentAnalysisStatus.INSUFFICIENT_EVIDENCE
    if any(item.status is ExperimentPipelineStageStatus.INSUFFICIENT_EVIDENCE for item in stage_results if item.stage in _enabled_stages(cfg)):
        return ExperimentAnalysisStatus.PARTIAL
    if any(not item.valid for item in condition_results):
        return ExperimentAnalysisStatus.COMPLETED_WITH_RESERVATIONS
    if cross_condition is not None and not cross_condition.valid and (
        cfg.run_cross_condition_association or cfg.run_candidate_chains
    ):
        return ExperimentAnalysisStatus.COMPLETED_WITH_RESERVATIONS
    if any(not item.valid for item in energy_results):
        return ExperimentAnalysisStatus.COMPLETED_WITH_RESERVATIONS
    if any(item.status is ExperimentPipelineStageStatus.COMPLETED_WITH_RESERVATIONS for item in stage_results):
        return ExperimentAnalysisStatus.COMPLETED_WITH_RESERVATIONS
    return ExperimentAnalysisStatus.COMPLETED


def _precomputed_recording_results(
    precomputed_results: Mapping[str, object] | None,
) -> Mapping[str, ExperimentRecordingAnalysisResult]:
    if not precomputed_results:
        return MappingProxyType({})
    value = precomputed_results.get("recording_results")
    if value is None:
        return MappingProxyType({})
    results = tuple(value) if isinstance(value, Iterable) else ()
    mapping = {
        item.recording_definition.recording_id or "": item
        for item in results
        if isinstance(item, ExperimentRecordingAnalysisResult)
    }
    return MappingProxyType(mapping)


def _enabled_stages(settings: ExperimentPipelineSettings) -> frozenset[ExperimentPipelineStage]:
    enabled: set[ExperimentPipelineStage] = {ExperimentPipelineStage.SUMMARY}
    flags = {
        ExperimentPipelineStage.LOAD: settings.run_loading,
        ExperimentPipelineStage.VALIDATE_INPUT: settings.run_input_validation,
        ExperimentPipelineStage.TEMPORAL: settings.run_temporal_analysis,
        ExperimentPipelineStage.GLOBAL_SPECTRUM: settings.run_global_spectrum,
        ExperimentPipelineStage.STFT: settings.run_stft,
        ExperimentPipelineStage.TRACKING: settings.run_tracking,
        ExperimentPipelineStage.PREIMPACT: settings.run_preimpact_analysis,
        ExperimentPipelineStage.EXCITATION: settings.run_excitation_characterization,
        ExperimentPipelineStage.MODAL_CANDIDATES: settings.run_modal_candidate_characterization,
        ExperimentPipelineStage.WITHIN_CONDITION: settings.run_within_condition_association,
        ExperimentPipelineStage.DYNAMIC_CONDITION_COMPARISON: settings.run_dynamic_condition_comparison,
        ExperimentPipelineStage.CROSS_CONDITION: settings.run_cross_condition_association,
        ExperimentPipelineStage.CANDIDATE_CHAINS: settings.run_candidate_chains,
        ExperimentPipelineStage.MODAL_HYPOTHESES: settings.run_modal_hypotheses,
        ExperimentPipelineStage.MODAL_PARAMETERS: settings.run_modal_parameter_estimation,
        ExperimentPipelineStage.MODAL_Q: settings.run_modal_q_estimation,
        ExperimentPipelineStage.MODAL_ENERGY_EXCHANGE: settings.run_modal_energy_exchange,
    }
    enabled.update(stage for stage, flag in flags.items() if flag)
    return frozenset(enabled)


def _invalid_recording_result(
    definition: ExperimentRecordingDefinition,
    failure_reason: str,
) -> ExperimentRecordingAnalysisResult:
    return ExperimentRecordingAnalysisResult(
        recording_definition=definition,
        loaded_recording=None,
        stage_results=(
            _stage_result(
                ExperimentPipelineStage.LOAD,
                ExperimentPipelineStageStatus.FAILED,
                input_ids=(definition.recording_id or "",),
                failures=(failure_reason,),
            ),
        ),
        valid=False,
        failure_reason=failure_reason,
        diagnostics=("recording_analysis_failed_structurally",),
    )


def _ordered_recordings(
    recordings: Iterable[ExperimentRecordingDefinition],
) -> tuple[ExperimentRecordingDefinition, ...]:
    return tuple(sorted(
        tuple(recordings),
        key=lambda item: (
            _dynamic_sort_key(item.dynamic_label),
            item.take_index,
            item.recording_id or "",
            str(item.file_path),
        ),
    ))


def _candidate_reference_key(ref: CandidateReference) -> tuple[object, ...]:
    return (
        _dynamic_sort_key(ref.dynamic_label),
        ref.recording_id,
        ref.candidate_id,
        ref.source_track_id,
        ref.representative_frequency_hz,
    )


def _dynamic_sort_key(label: str) -> tuple[int, str]:
    try:
        return (DYNAMIC_LABEL_ORDER.index(label), label)
    except ValueError:
        return (len(DYNAMIC_LABEL_ORDER), label)


def _canonical_dynamic_labels(labels: Sequence[str]) -> tuple[str, ...]:
    unique = tuple(dict.fromkeys(labels))
    return tuple(sorted(unique, key=_dynamic_sort_key))


def _duration_quality(duration: float | None, minimum: float | None) -> float | None:
    if duration is None:
        return None
    if minimum is None:
        return 1.0 if duration > 0.0 else 0.0
    return max(0.0, min(1.0, duration / minimum))


def _snr_quality(snr_db: float | None) -> float | None:
    if snr_db is None or not isfinite(snr_db):
        return None
    return max(0.0, min(1.0, snr_db / 60.0))


def _stable_id(prefix: str, *parts: object) -> str:
    payload = json.dumps(_canonicalize(parts), sort_keys=True, separators=(",", ":"))
    return f"{prefix}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _belllab_version() -> str:
    module = sys.modules.get("belllab")
    return str(getattr(module, "__version__", "0.1.0"))


def _canonicalize(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonicalize(getattr(value, field.name))
            for field in fields(value)
            if field.name not in {"result", "original_signal", "signal"}
        }
    if isinstance(value, (tuple, list)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, float):
        if value == float("inf"):
            return "Infinity"
        if value == float("-inf"):
            return "-Infinity"
        return value
    return value


def _coerce_enum(value: object, enum_type: type[Enum]) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)  # type: ignore[call-arg]
    except ValueError as exc:
        raise ExperimentDefinitionError(
            f"{enum_type.__name__} value is not recognized."
        ) from exc


def _validate_dynamic_label(label: str) -> None:
    if label not in DYNAMIC_LABEL_ORDER:
        raise ExperimentDefinitionError(
            "dynamic_label must be one of pp, p, mf, f, ff."
        )


def _validate_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentDefinitionError(f"{name} must be a non-empty string.")


def _validate_optional_text(value: str | None, name: str) -> None:
    if value is not None:
        _validate_text(value, name)


def _validate_text_tuple(values: Sequence[str], name: str) -> None:
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ExperimentDefinitionError(f"{name} must contain only non-empty strings.")


def _finite_optional(value: float | None, name: str) -> None:
    if value is not None and not isfinite(value):
        raise ExperimentDefinitionError(f"{name} must be finite when provided.")


def _finite_nonnegative_optional(value: float | None, name: str) -> None:
    if value is not None and (not isfinite(value) or value < 0.0):
        raise ExperimentDefinitionError(f"{name} must be finite and non-negative.")


def _finite_positive_optional(value: float | None, name: str) -> None:
    if value is not None and (not isfinite(value) or value <= 0.0):
        raise ExperimentDefinitionError(f"{name} must be finite and positive.")


def _fraction_optional(value: float | None, name: str) -> None:
    if value is not None and (not isfinite(value) or not 0.0 <= value <= 1.0):
        raise ExperimentDefinitionError(f"{name} must be finite and in [0, 1].")
