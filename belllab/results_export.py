"""Deterministic export of already computed BellLab scientific results.

This module serializes and writes existing experiment-analysis results.  It
does not reopen audio files, rerun the real-experiment pipeline, recalculate
spectra, reinterpret modal hypotheses, or convert operational evidence into
physical conclusions.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from belllab.experiment_pipeline import (
    ExperimentAnalysisResult,
    ExperimentAnalysisStatus,
    ExperimentConditionAnalysisResult,
    ExperimentCrossConditionAnalysisResult,
    ExperimentRecordingAnalysisResult,
    summarize_experiment_analysis,
)


class ResultsExportStatus(str, Enum):
    """Mutually exclusive status for a reproducible results export."""

    COMPLETED = "completed"
    COMPLETED_WITH_RESERVATIONS = "completed_with_reservations"
    PARTIAL = "partial"
    FAILED = "failed"
    INVALID_INPUT = "invalid_input"


class ResultsExportReason(str, Enum):
    """Typed reasons for export support, reservations, skips and failures."""

    ALL_REQUESTED_ARTIFACTS_WRITTEN = "all_requested_artifacts_written"
    OPTIONAL_ARTIFACT_SKIPPED = "optional_artifact_skipped"
    MISSING_OPTIONAL_RESULT = "missing_optional_result"
    MISSING_REQUIRED_RESULT = "missing_required_result"
    UNSUPPORTED_RESULT_TYPE = "unsupported_result_type"
    SERIALIZATION_FAILURE = "serialization_failure"
    FILESYSTEM_ERROR = "filesystem_error"
    EXISTING_FILE_CONFLICT = "existing_file_conflict"
    NONFINITE_VALUE_DETECTED = "nonfinite_value_detected"
    INVALID_OUTPUT_DIRECTORY = "invalid_output_directory"
    INVALID_EXPORT_CONFIGURATION = "invalid_export_configuration"
    PARTIAL_EXPERIMENT_RESULT = "partial_experiment_result"
    SOURCE_RESULT_REQUIRES_REVIEW = "source_result_requires_review"
    SOURCE_RESULT_INVALID = "source_result_invalid"
    CHECKSUM_FAILURE = "checksum_failure"


class ExportOverwritePolicy(str, Enum):
    """Explicit overwrite policy for generated artifacts."""

    ERROR = "error"
    SKIP = "skip"
    REPLACE = "replace"
    VERSIONED_FILENAME = "versioned_filename"


class ExportMissingValuePolicy(str, Enum):
    """Presentation policy for values that are genuinely absent."""

    NULL = "null"
    EMPTY = "empty"
    NA = "na"
    DASH = "dash"


class ExportNonfiniteValuePolicy(str, Enum):
    """Policy for NaN and infinities in exportable structures."""

    ERROR = "error"
    NULL_WITH_DIAGNOSTIC = "null_with_diagnostic"
    STRING_WITH_DIAGNOSTIC = "string_with_diagnostic"


class BellLabExportSchemaVersion(str, Enum):
    """Version of the normalized BellLab export schema."""

    V1_0 = "1.0"


@dataclass(frozen=True, slots=True)
class ExportNumericFormatting:
    """Presentation-only numeric formatting policy."""

    default_decimal_places: int = 6
    uncertainty_significant_digits: int = 2
    match_value_precision_to_uncertainty: bool = False
    use_scientific_notation: bool = False
    decimal_separator: str = "."
    thousands_separator: str = ""

    def __post_init__(self) -> None:
        if self.default_decimal_places < 0:
            raise ValueError("default_decimal_places must not be negative.")
        if self.uncertainty_significant_digits <= 0:
            raise ValueError("uncertainty_significant_digits must be positive.")
        if self.decimal_separator not in {".", ","}:
            raise ValueError("decimal_separator must be '.' or ','.")
        if self.thousands_separator not in {"", ",", ".", " "}:
            raise ValueError("thousands_separator is not recognized.")


@dataclass(frozen=True, slots=True)
class ResultsExportSettings:
    """Explicit deterministic configuration for reproducible export."""

    export_json: bool = True
    export_csv: bool = True
    export_latex: bool = True
    export_markdown: bool = True
    export_manifest: bool = True
    export_summary: bool = True
    export_intermediate_tables: bool = True

    include_recordings: bool = True
    include_conditions: bool = True
    include_candidates: bool = True
    include_associations: bool = True
    include_chains: bool = True
    include_hypotheses: bool = True
    include_modal_parameters: bool = True
    include_q_factors: bool = True
    include_energy_exchange: bool = True
    include_stage_results: bool = True
    include_diagnostics: bool = True
    include_settings: bool = True
    include_provenance: bool = True
    include_invalid_results: bool = True
    include_rejected_results: bool = True
    include_inconclusive_results: bool = True

    float_precision: int = 10
    scientific_notation_threshold: float = 1e6
    preserve_full_precision_in_json: bool = True
    missing_value_representation: ExportMissingValuePolicy = ExportMissingValuePolicy.NULL
    nonfinite_value_policy: ExportNonfiniteValuePolicy = ExportNonfiniteValuePolicy.ERROR

    output_directory: str | Path = Path("belllab-export")
    file_prefix: str = "belllab_results"
    overwrite_policy: ExportOverwritePolicy = ExportOverwritePolicy.ERROR
    create_output_directory: bool = True
    atomic_write: bool = True
    write_checksums: bool = True
    checksum_algorithm: str = "sha256"

    table_layout: str = "normalized"
    column_selection: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    column_order: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    sort_policy: str = "deterministic"
    latex_booktabs: bool = True
    latex_escape_text: bool = True
    markdown_alignment: str = "left"

    include_analysis_id_in_filename: bool = True
    include_experiment_id_in_filename: bool = False
    include_settings_fingerprint_in_manifest: bool = True
    include_belllab_version: bool = True
    json_indent: int = 2
    numeric_formatting: ExportNumericFormatting = ExportNumericFormatting()

    def __post_init__(self) -> None:
        for name in (
            "export_json",
            "export_csv",
            "export_latex",
            "export_markdown",
            "export_manifest",
            "export_summary",
            "export_intermediate_tables",
            "include_recordings",
            "include_conditions",
            "include_candidates",
            "include_associations",
            "include_chains",
            "include_hypotheses",
            "include_modal_parameters",
            "include_q_factors",
            "include_energy_exchange",
            "include_stage_results",
            "include_diagnostics",
            "include_settings",
            "include_provenance",
            "include_invalid_results",
            "include_rejected_results",
            "include_inconclusive_results",
            "preserve_full_precision_in_json",
            "create_output_directory",
            "atomic_write",
            "write_checksums",
            "latex_booktabs",
            "latex_escape_text",
            "include_analysis_id_in_filename",
            "include_experiment_id_in_filename",
            "include_settings_fingerprint_in_manifest",
            "include_belllab_version",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")
        if not any(
            (
                self.export_json,
                self.export_csv,
                self.export_latex,
                self.export_markdown,
                self.export_manifest,
                self.export_summary,
            )
        ):
            raise ValueError("at least one export artifact must be requested.")
        if self.float_precision <= 0:
            raise ValueError("float_precision must be positive.")
        if not math.isfinite(self.scientific_notation_threshold) or self.scientific_notation_threshold <= 0:
            raise ValueError("scientific_notation_threshold must be finite and positive.")
        object.__setattr__(
            self,
            "missing_value_representation",
            _coerce_enum(self.missing_value_representation, ExportMissingValuePolicy),
        )
        object.__setattr__(
            self,
            "nonfinite_value_policy",
            _coerce_enum(self.nonfinite_value_policy, ExportNonfiniteValuePolicy),
        )
        object.__setattr__(
            self,
            "overwrite_policy",
            _coerce_enum(self.overwrite_policy, ExportOverwritePolicy),
        )
        output = Path(self.output_directory)
        if not str(output):
            raise ValueError("output_directory must not be empty.")
        object.__setattr__(self, "output_directory", output)
        if not self.file_prefix.strip():
            raise ValueError("file_prefix must not be empty.")
        if self.checksum_algorithm != "sha256":
            raise ValueError("only sha256 checksums are supported.")
        if self.table_layout != "normalized":
            raise ValueError("table_layout must be 'normalized'.")
        if self.sort_policy != "deterministic":
            raise ValueError("sort_policy must be 'deterministic'.")
        if self.markdown_alignment not in {"left", "center", "right"}:
            raise ValueError("markdown_alignment is not recognized.")
        if self.json_indent < 0:
            raise ValueError("json_indent must not be negative.")
        object.__setattr__(
            self,
            "column_selection",
            MappingProxyType({str(key): tuple(value) for key, value in self.column_selection.items()}),
        )
        object.__setattr__(
            self,
            "column_order",
            MappingProxyType({str(key): tuple(value) for key, value in self.column_order.items()}),
        )


@dataclass(frozen=True, slots=True)
class NormalizedExperimentExport:
    """Serializable normalized model used by all export formats."""

    schema_version: BellLabExportSchemaVersion
    belllab_version: str
    analysis_id: str
    experiment_id: str | None
    experiment: Mapping[str, object]
    summary: Mapping[str, object]
    recordings: tuple[Mapping[str, object], ...]
    conditions: tuple[Mapping[str, object], ...]
    cross_condition: Mapping[str, object] | None
    chains: tuple[Mapping[str, object], ...]
    modal_hypotheses: tuple[Mapping[str, object], ...]
    modal_parameters: tuple[Mapping[str, object], ...]
    modal_q_factors: tuple[Mapping[str, object], ...]
    energy_exchange: tuple[Mapping[str, object], ...]
    stage_results: tuple[Mapping[str, object], ...]
    provenance: Mapping[str, object]
    settings: Mapping[str, object]
    diagnostics: tuple[str, ...]
    tables: Mapping[str, tuple[Mapping[str, object], ...]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            _coerce_enum(self.schema_version, BellLabExportSchemaVersion),
        )
        for name in ("belllab_version", "analysis_id"):
            _nonempty_text(getattr(self, name), name)
        for name in ("experiment", "summary", "provenance", "settings"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        for name in (
            "recordings",
            "conditions",
            "chains",
            "modal_hypotheses",
            "modal_parameters",
            "modal_q_factors",
            "energy_exchange",
            "stage_results",
        ):
            object.__setattr__(
                self,
                name,
                tuple(MappingProxyType(dict(row)) for row in getattr(self, name)),
            )
        if self.cross_condition is not None:
            object.__setattr__(self, "cross_condition", MappingProxyType(dict(self.cross_condition)))
        object.__setattr__(self, "diagnostics", tuple(dict.fromkeys(self.diagnostics)))
        object.__setattr__(
            self,
            "tables",
            MappingProxyType({
                str(name): tuple(MappingProxyType(dict(row)) for row in rows)
                for name, rows in self.tables.items()
            }),
        )


@dataclass(frozen=True, slots=True)
class ExportedArtifact:
    """Result of one generated export artifact."""

    artifact_id: str
    artifact_type: str
    format: str
    path: str | None
    relative_path: str | None
    checksum: str | None
    size_bytes: int | None
    row_count: int | None
    status: ResultsExportStatus
    reasons: tuple[ResultsExportReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty_text(self.artifact_id, "artifact_id")
        _nonempty_text(self.artifact_type, "artifact_type")
        _nonempty_text(self.format, "format")
        object.__setattr__(self, "status", _coerce_enum(self.status, ResultsExportStatus))
        object.__setattr__(self, "reasons", _reason_tuple(self.reasons))
        object.__setattr__(self, "diagnostics", tuple(dict.fromkeys(self.diagnostics)))
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ValueError("size_bytes must not be negative.")
        if self.row_count is not None and self.row_count < 0:
            raise ValueError("row_count must not be negative.")


@dataclass(frozen=True, slots=True)
class ExperimentExportManifest:
    """Provenance manifest for one export run."""

    manifest_schema_version: str
    analysis_id: str
    experiment_id: str | None
    belllab_version: str
    export_schema_version: str
    settings_fingerprint: str
    source_file_fingerprints: Mapping[str, str | None]
    generated_artifacts: tuple[ExportedArtifact, ...]
    artifact_checksums: Mapping[str, str | None]
    artifact_sizes_bytes: Mapping[str, int | None]
    artifact_row_counts: Mapping[str, int | None]
    completed_exports: tuple[str, ...]
    skipped_exports: tuple[str, ...]
    failed_exports: tuple[str, ...]
    source_status: str
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("manifest_schema_version", "analysis_id", "belllab_version", "export_schema_version", "settings_fingerprint", "source_status"):
            _nonempty_text(getattr(self, name), name)
        object.__setattr__(
            self,
            "source_file_fingerprints",
            MappingProxyType(dict(self.source_file_fingerprints)),
        )
        object.__setattr__(self, "artifact_checksums", MappingProxyType(dict(self.artifact_checksums)))
        object.__setattr__(self, "artifact_sizes_bytes", MappingProxyType(dict(self.artifact_sizes_bytes)))
        object.__setattr__(self, "artifact_row_counts", MappingProxyType(dict(self.artifact_row_counts)))
        for name in ("completed_exports", "skipped_exports", "failed_exports", "diagnostics"):
            object.__setattr__(self, name, tuple(dict.fromkeys(getattr(self, name))))


@dataclass(frozen=True, slots=True)
class ExperimentExportResult:
    """Top-level result of reproducible export."""

    export_id: str
    analysis_id: str
    experiment_id: str | None
    output_directory: str
    status: ResultsExportStatus
    artifacts: tuple[ExportedArtifact, ...]
    manifest: ExperimentExportManifest | None
    completed_artifacts: tuple[ExportedArtifact, ...]
    skipped_artifacts: tuple[ExportedArtifact, ...]
    failed_artifacts: tuple[ExportedArtifact, ...]
    artifact_count: int
    completed_count: int
    skipped_count: int
    failed_count: int
    settings: ResultsExportSettings
    valid: bool
    failure_reason: str | None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty_text(self.export_id, "export_id")
        _nonempty_text(self.analysis_id, "analysis_id")
        object.__setattr__(self, "status", _coerce_enum(self.status, ResultsExportStatus))
        for name in ("artifacts", "completed_artifacts", "skipped_artifacts", "failed_artifacts"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.artifact_count != len(self.artifacts):
            raise ValueError("artifact_count must match artifacts.")
        if self.completed_count != len(self.completed_artifacts):
            raise ValueError("completed_count must match completed_artifacts.")
        if self.skipped_count != len(self.skipped_artifacts):
            raise ValueError("skipped_count must match skipped_artifacts.")
        if self.failed_count != len(self.failed_artifacts):
            raise ValueError("failed_count must match failed_artifacts.")
        expected_valid = self.status in {
            ResultsExportStatus.COMPLETED,
            ResultsExportStatus.COMPLETED_WITH_RESERVATIONS,
        }
        if self.valid != expected_valid:
            raise ValueError("valid flag must mirror successful export statuses.")
        object.__setattr__(self, "diagnostics", tuple(dict.fromkeys(self.diagnostics)))


@dataclass(frozen=True, slots=True)
class ExperimentExportValidation:
    """Structured validation of exported artifacts."""

    expected_artifacts: tuple[str, ...]
    existing_artifacts: tuple[str, ...]
    missing_artifacts: tuple[str, ...]
    checksum_matches: Mapping[str, bool]
    schema_valid: bool
    row_counts_valid: bool
    foreign_keys_valid: bool
    json_roundtrip_valid: bool
    manifest_consistent: bool
    valid: bool
    reasons: tuple[ResultsExportReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("expected_artifacts", "existing_artifacts", "missing_artifacts", "diagnostics"):
            object.__setattr__(self, name, tuple(dict.fromkeys(getattr(self, name))))
        object.__setattr__(self, "checksum_matches", MappingProxyType(dict(self.checksum_matches)))
        object.__setattr__(self, "reasons", _reason_tuple(self.reasons))


EXPORT_SCHEMA_DESCRIPTION: Mapping[str, object] = MappingProxyType({
    "schema_version": BellLabExportSchemaVersion.V1_0.value,
    "required_fields": (
        "schema_version",
        "belllab_version",
        "analysis_id",
        "experiment_id",
        "summary",
        "tables",
        "provenance",
    ),
    "optional_fields": (
        "diagnostics",
        "settings",
        "energy_exchange",
        "modal_q_factors",
    ),
    "minimum_compatible_schema": "1.0",
    "missing_value_meaning": "null means unavailable, not zero.",
})


CSV_TABLE_ORDER: tuple[str, ...] = (
    "experiment_summary",
    "recordings",
    "conditions",
    "candidates",
    "within_condition_associations",
    "cross_condition_matches",
    "candidate_chains",
    "candidate_chain_nodes",
    "modal_hypotheses",
    "modal_parameters",
    "modal_q_factors",
    "energy_exchange_pairs",
    "pipeline_stages",
    "diagnostics",
)

CSV_COLUMN_ORDER: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "experiment_summary": (
        "analysis_id",
        "experiment_id",
        "name",
        "status",
        "valid",
        "requires_review",
        "recording_count",
        "condition_count",
        "candidate_count",
        "chain_count",
        "hypothesis_count",
        "parameter_estimate_count",
        "q_estimate_count",
        "energy_pair_count",
        "failure_reason",
        "belllab_version",
        "settings_fingerprint",
    ),
    "recordings": (
        "recording_id",
        "experiment_id",
        "dynamic_label",
        "take_index",
        "file_path",
        "file_fingerprint",
        "sample_rate_hz",
        "channel_count",
        "selected_channel",
        "original_duration_s",
        "analyzed_duration_s",
        "offsets_applied",
        "candidate_count",
        "accepted_candidate_count",
        "valid",
        "failure_reason",
        "diagnostics",
    ),
    "conditions": (
        "dynamic_label",
        "selected_reference_recording_id",
        "recording_count",
        "valid_recording_count",
        "invalid_recording_count",
        "candidate_reference_count",
        "within_condition_cluster_count",
        "valid",
        "failure_reason",
        "diagnostics",
    ),
    "candidates": (
        "recording_id",
        "dynamic_label",
        "candidate_id",
        "source_track_id",
        "representative_frequency_hz",
        "frequency_stability",
        "frequency_drift_hz",
        "frequency_fit_rmse_hz",
        "amplitude_tau_s",
        "amplitude_fit_r_squared",
        "coverage_fraction",
        "accepted",
        "impact_excited",
        "preimpact_classification",
        "ambiguous_assignment_fraction",
        "near_threshold_assignment_fraction",
        "minimum_assignment_margin",
        "acceptance_reasons",
        "rejection_reasons",
        "diagnostics",
    ),
    "within_condition_associations": (
        "dynamic_label",
        "association_kind",
        "cluster_id",
        "recording_id",
        "candidate_id",
        "source_track_id",
        "representative_frequency_hz",
        "member_count",
        "recording_count",
        "accepted",
        "ambiguous",
        "reason",
        "minimum_cost_observed",
        "diagnostics",
    ),
    "cross_condition_matches": (
        "match_id",
        "lower_dynamic_label",
        "higher_dynamic_label",
        "lower_recording_id",
        "lower_candidate_id",
        "higher_recording_id",
        "higher_candidate_id",
        "frequency_change_hz",
        "frequency_change_relative",
        "frequency_change_classification",
        "association_cost",
        "ambiguous",
        "near_threshold",
        "accepted",
        "diagnostics",
    ),
    "candidate_chains": (
        "chain_id",
        "start_dynamic_label",
        "end_dynamic_label",
        "condition_count",
        "match_count",
        "complete_across_requested_sequence",
        "partial_chain",
        "isolated_candidate",
        "initial_frequency_hz",
        "final_frequency_hz",
        "total_frequency_change_hz",
        "maximum_association_cost",
        "contains_ambiguous_match",
        "contains_near_threshold_match",
        "contains_possible_split_context",
        "contains_possible_merge_context",
        "diagnostics",
    ),
    "candidate_chain_nodes": (
        "chain_id",
        "node_index",
        "dynamic_label",
        "recording_id",
        "candidate_id",
        "source_track_id",
        "representative_frequency_hz",
        "incoming_match_id",
        "outgoing_match_id",
        "incoming_association_cost",
        "outgoing_association_cost",
        "diagnostics",
    ),
    "modal_hypotheses": (
        "hypothesis_id",
        "source_chain_id",
        "status",
        "accepted",
        "requires_review",
        "score",
        "condition_coverage_fraction",
        "maximum_step_change_hz",
        "mean_match_cost",
        "mean_tracking_coverage_fraction",
        "tau_available_count",
        "supporting_reasons",
        "reservation_reasons",
        "rejection_reasons",
        "missing_evidence_reasons",
        "diagnostics",
    ),
    "modal_parameters": (
        "estimate_id",
        "hypothesis_id",
        "status",
        "representative_frequency_hz",
        "frequency_uncertainty_hz",
        "frequency_lower_bound_hz",
        "frequency_upper_bound_hz",
        "frequency_range_hz",
        "relative_frequency_range",
        "trajectory_total_signed_change_hz",
        "trajectory_slope_hz_per_condition_step",
        "trajectory_rmse_hz",
        "representative_tau_s",
        "tau_uncertainty_log",
        "tau_lower_bound_s",
        "tau_upper_bound_s",
        "amplitude_decay_rate_per_s",
        "time_to_minus_20_db_s",
        "time_to_minus_40_db_s",
        "time_to_minus_60_db_s",
        "supporting_reasons",
        "reservation_reasons",
        "insufficient_evidence_reasons",
        "invalid_reasons",
        "provenance_candidate_ids",
        "provenance_match_ids",
        "diagnostics",
    ),
    "modal_q_factors": (
        "estimate_id",
        "modal_parameter_estimate_id",
        "hypothesis_id",
        "status",
        "q_decay",
        "q_bandwidth",
        "bandwidth_hz",
        "bandwidth_definition",
        "frequency_resolution_hz",
        "resolution_ratio",
        "isolated_peak",
        "method_relative_symmetric_difference",
        "representative_q",
        "representative_q_method",
        "representative_q_uncertainty",
        "supporting_reasons",
        "reservation_reasons",
        "inconclusive_reasons",
        "insufficient_evidence_reasons",
        "invalid_reasons",
        "diagnostics",
    ),
    "energy_exchange_pairs": (
        "evidence_id",
        "dynamic_label",
        "source_a_id",
        "source_b_id",
        "status",
        "normalized_score",
        "opposed_trends",
        "zero_lag_correlation",
        "best_negative_lag_s",
        "best_negative_correlation",
        "best_negative_p_value",
        "delayed_growth_a",
        "delayed_growth_b",
        "recovery_a",
        "recovery_b",
        "pair_energy_relative_range",
        "approximately_conserved_pair_energy",
        "alternating_dominance",
        "possible_beating",
        "supporting_reasons",
        "reservation_reasons",
        "inconclusive_reasons",
        "not_supported_reasons",
        "diagnostics",
    ),
    "pipeline_stages": (
        "stage",
        "status",
        "started",
        "completed",
        "input_ids",
        "output_ids",
        "dependency_stages",
        "supporting_reasons",
        "reservation_reasons",
        "skipped_reasons",
        "blocked_reasons",
        "failure_reasons",
        "insufficient_evidence_reasons",
        "diagnostics",
    ),
    "diagnostics": (
        "scope",
        "source_id",
        "diagnostic_index",
        "diagnostic",
    ),
})


def normalize_experiment_for_export(
    result: ExperimentAnalysisResult,
    settings: ResultsExportSettings | None = None,
) -> NormalizedExperimentExport:
    """Build the normalized export model from an existing analysis result."""

    cfg = settings or ResultsExportSettings()
    if not isinstance(result, ExperimentAnalysisResult):
        raise TypeError("result must be an ExperimentAnalysisResult.")
    diagnostics = [
        "export_success_is_not_scientific_validity",
        "export_layer_does_not_recalculate_analysis",
        "missing_values_are_preserved_not_zero",
        "modal_hypothesis_not_physical_mode_proof",
        "possible_energy_redistribution_not_physical_transfer_proof",
    ]
    if result.requires_review:
        diagnostics.append("source_result_requires_review")
    tables = _build_export_tables(result, cfg)
    summary = summarize_experiment_analysis(result)
    experiment_dict = _experiment_row(result)
    provenance = _to_serializable(result.provenance, cfg)
    settings_payload = {
        "export_settings_fingerprint": export_settings_fingerprint(cfg),
        "export_schema": dict(EXPORT_SCHEMA_DESCRIPTION),
        "experiment_settings_fingerprint": result.provenance.settings_fingerprint,
    }
    if cfg.include_settings:
        settings_payload["export_settings"] = _export_settings_payload(cfg)
    normalized = NormalizedExperimentExport(
        schema_version=BellLabExportSchemaVersion.V1_0,
        belllab_version=result.provenance.belllab_version,
        analysis_id=result.analysis_id,
        experiment_id=result.experiment.experiment_id,
        experiment=experiment_dict,
        summary=_to_serializable(summary, cfg),
        recordings=tables.get("recordings", ()),
        conditions=tables.get("conditions", ()),
        cross_condition=_cross_condition_row(result.cross_condition_result),
        chains=tables.get("candidate_chains", ()),
        modal_hypotheses=tables.get("modal_hypotheses", ()),
        modal_parameters=tables.get("modal_parameters", ()),
        modal_q_factors=tables.get("modal_q_factors", ()),
        energy_exchange=tables.get("energy_exchange_pairs", ()),
        stage_results=tables.get("pipeline_stages", ()),
        provenance=provenance if cfg.include_provenance else {},
        settings=settings_payload,
        diagnostics=tuple(diagnostics + list(result.diagnostics if cfg.include_diagnostics else ())),
        tables=tables,
    )
    _assert_no_nonfinite_tokens(normalized, cfg)
    return normalized


def serialize_experiment_result(
    result: ExperimentAnalysisResult | NormalizedExperimentExport,
    settings: ResultsExportSettings | None = None,
) -> dict[str, object]:
    """Return a JSON-ready structure without writing any file."""

    cfg = settings or ResultsExportSettings()
    normalized = (
        result
        if isinstance(result, NormalizedExperimentExport)
        else normalize_experiment_for_export(result, cfg)
    )
    payload = _to_serializable(normalized, cfg)
    if not isinstance(payload, dict):
        raise TypeError("normalized export must serialize to a dictionary.")
    payload["schema_description"] = dict(EXPORT_SCHEMA_DESCRIPTION)
    return payload


def export_experiment_json(
    result: ExperimentAnalysisResult | NormalizedExperimentExport,
    settings: ResultsExportSettings | None = None,
) -> ExportedArtifact:
    """Write the normalized experiment export as deterministic JSON."""

    cfg = settings or ResultsExportSettings()
    normalized = (
        result
        if isinstance(result, NormalizedExperimentExport)
        else normalize_experiment_for_export(result, cfg)
    )
    payload = serialize_experiment_result(normalized, cfg)
    content = json.dumps(
        payload,
        ensure_ascii=False,
        indent=cfg.json_indent,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    path = _artifact_path(cfg, normalized, "experiment_export", "json")
    return _write_artifact(
        normalized.analysis_id,
        "experiment_export",
        "json",
        path,
        content,
        cfg,
        row_count=1,
    )


def export_experiment_csv_tables(
    result: ExperimentAnalysisResult | NormalizedExperimentExport,
    settings: ResultsExportSettings | None = None,
) -> tuple[ExportedArtifact, ...]:
    """Write separate normalized CSV tables for the experiment result."""

    cfg = settings or ResultsExportSettings()
    normalized = (
        result
        if isinstance(result, NormalizedExperimentExport)
        else normalize_experiment_for_export(result, cfg)
    )
    artifacts: list[ExportedArtifact] = []
    tables = _selected_csv_tables(normalized.tables, cfg)
    for table_name in CSV_TABLE_ORDER:
        if table_name not in tables:
            continue
        rows = tables[table_name]
        if not rows and not _write_empty_table(table_name):
            artifacts.append(_skipped_artifact(
                normalized.analysis_id,
                table_name,
                "csv",
                ResultsExportReason.MISSING_OPTIONAL_RESULT,
            ))
            continue
        content = _csv_content(table_name, rows, cfg)
        path = _artifact_path(cfg, normalized, table_name, "csv")
        artifacts.append(_write_artifact(
            normalized.analysis_id,
            table_name,
            "csv",
            path,
            content,
            cfg,
            row_count=len(rows),
        ))
    return tuple(artifacts)


def export_experiment_latex_tables(
    result: ExperimentAnalysisResult | NormalizedExperimentExport,
    settings: ResultsExportSettings | None = None,
) -> tuple[ExportedArtifact, ...]:
    """Write LaTeX table fragments, not a PDF or full document."""

    cfg = settings or ResultsExportSettings()
    normalized = (
        result
        if isinstance(result, NormalizedExperimentExport)
        else normalize_experiment_for_export(result, cfg)
    )
    selected = {
        "experiment_summary": normalized.tables.get("experiment_summary", ()),
        "recordings": normalized.tables.get("recordings", ()),
        "modal_hypotheses": normalized.tables.get("modal_hypotheses", ()),
        "modal_parameters": normalized.tables.get("modal_parameters", ()),
        "modal_q_factors": normalized.tables.get("modal_q_factors", ()),
        "energy_exchange_pairs": normalized.tables.get("energy_exchange_pairs", ()),
        "failures_and_reservations": _failure_rows(normalized),
    }
    artifacts: list[ExportedArtifact] = []
    for table_name, rows in selected.items():
        content = _latex_table_content(table_name, rows, cfg)
        path = _artifact_path(cfg, normalized, table_name, "tex")
        artifacts.append(_write_artifact(
            normalized.analysis_id,
            table_name,
            "latex",
            path,
            content,
            cfg,
            row_count=len(rows),
        ))
    return tuple(artifacts)


def export_experiment_markdown_summary(
    result: ExperimentAnalysisResult | NormalizedExperimentExport,
    settings: ResultsExportSettings | None = None,
) -> ExportedArtifact:
    """Write a concise Markdown summary without physical interpretation."""

    cfg = settings or ResultsExportSettings()
    normalized = (
        result
        if isinstance(result, NormalizedExperimentExport)
        else normalize_experiment_for_export(result, cfg)
    )
    content = _markdown_content(normalized, cfg)
    path = _artifact_path(cfg, normalized, "experiment_summary", "md")
    return _write_artifact(
        normalized.analysis_id,
        "experiment_summary",
        "markdown",
        path,
        content,
        cfg,
        row_count=None,
    )


def build_experiment_export_manifest(
    normalized: NormalizedExperimentExport,
    artifacts: Iterable[ExportedArtifact],
    settings: ResultsExportSettings | None = None,
) -> ExperimentExportManifest:
    """Build a deterministic manifest from already attempted artifacts."""

    cfg = settings or ResultsExportSettings()
    artifact_tuple = tuple(sorted(artifacts, key=lambda item: (item.relative_path or "", item.artifact_type)))
    manifest_artifacts = tuple(_manifest_artifact_view(item) for item in artifact_tuple)
    completed = tuple(
        item.relative_path or item.artifact_type
        for item in artifact_tuple
        if item.status in {ResultsExportStatus.COMPLETED, ResultsExportStatus.COMPLETED_WITH_RESERVATIONS}
    )
    skipped = tuple(
        item.artifact_type
        for item in artifact_tuple
        if item.status is ResultsExportStatus.PARTIAL
        and ResultsExportReason.OPTIONAL_ARTIFACT_SKIPPED in item.reasons
    )
    failed = tuple(
        item.artifact_type
        for item in artifact_tuple
        if item.status in {ResultsExportStatus.FAILED, ResultsExportStatus.INVALID_INPUT}
    )
    provenance = normalized.provenance
    source_fingerprints = provenance.get("file_fingerprints", {}) if isinstance(provenance, Mapping) else {}
    return ExperimentExportManifest(
        manifest_schema_version="1.0",
        analysis_id=normalized.analysis_id,
        experiment_id=normalized.experiment_id,
        belllab_version=normalized.belllab_version,
        export_schema_version=normalized.schema_version.value,
        settings_fingerprint=export_settings_fingerprint(cfg),
        source_file_fingerprints=source_fingerprints if isinstance(source_fingerprints, Mapping) else {},
        generated_artifacts=manifest_artifacts,
        artifact_checksums={
            item.relative_path or item.artifact_type: (
                None if item.artifact_type == "manifest" else item.checksum
            )
            for item in artifact_tuple
        },
        artifact_sizes_bytes={item.relative_path or item.artifact_type: item.size_bytes for item in artifact_tuple},
        artifact_row_counts={item.relative_path or item.artifact_type: item.row_count for item in artifact_tuple},
        completed_exports=completed,
        skipped_exports=skipped,
        failed_exports=failed,
        source_status=str(normalized.summary.get("status", "")),
        diagnostics=(
            "manifest_written_after_artifacts",
            "checksums_are_content_hashes",
            "exported_files_do_not_recalculate_analysis",
        ),
    )


def export_experiment_results(
    result: ExperimentAnalysisResult,
    settings: ResultsExportSettings | None = None,
) -> ExperimentExportResult:
    """Export requested artifacts and write the manifest last."""

    cfg = settings or ResultsExportSettings()
    if not isinstance(result, ExperimentAnalysisResult):
        return _invalid_export_result(
            "invalid-analysis",
            None,
            cfg,
            ResultsExportReason.UNSUPPORTED_RESULT_TYPE,
        )
    try:
        normalized = normalize_experiment_for_export(result, cfg)
    except Exception as exc:
        return _invalid_export_result(
            getattr(result, "analysis_id", "invalid-analysis"),
            getattr(getattr(result, "experiment", None), "experiment_id", None),
            cfg,
            ResultsExportReason.SERIALIZATION_FAILURE,
            diagnostics=(f"{exc.__class__.__name__}: {exc}",),
        )

    artifacts: list[ExportedArtifact] = []
    if cfg.export_json:
        artifacts.append(_attempt(lambda: export_experiment_json(normalized, cfg), normalized.analysis_id, "experiment_export", "json"))
    else:
        artifacts.append(_skipped_artifact(normalized.analysis_id, "experiment_export", "json"))
    if cfg.export_summary:
        artifacts.append(_attempt(lambda: _export_summary_json(normalized, cfg), normalized.analysis_id, "summary", "json"))
    else:
        artifacts.append(_skipped_artifact(normalized.analysis_id, "summary", "json"))
    if cfg.export_csv:
        for artifact in _attempt_many(
            lambda: export_experiment_csv_tables(normalized, cfg),
            normalized.analysis_id,
            "csv_tables",
            "csv",
        ):
            artifacts.append(artifact)
    else:
        artifacts.append(_skipped_artifact(normalized.analysis_id, "csv_tables", "csv"))
    if cfg.export_latex:
        for artifact in _attempt_many(
            lambda: export_experiment_latex_tables(normalized, cfg),
            normalized.analysis_id,
            "latex_tables",
            "latex",
        ):
            artifacts.append(artifact)
    else:
        artifacts.append(_skipped_artifact(normalized.analysis_id, "latex_tables", "latex"))
    if cfg.export_markdown:
        artifacts.append(_attempt(lambda: export_experiment_markdown_summary(normalized, cfg), normalized.analysis_id, "experiment_summary", "markdown"))
    else:
        artifacts.append(_skipped_artifact(normalized.analysis_id, "experiment_summary", "markdown"))

    manifest: ExperimentExportManifest | None = None
    if cfg.export_manifest:
        manifest_path = _artifact_path(cfg, normalized, "manifest", "json")
        manifest_placeholder = ExportedArtifact(
            artifact_id=_stable_id("artifact", normalized.analysis_id, "manifest", "json", "self_checksum_omitted"),
            artifact_type="manifest",
            format="json",
            path=str(manifest_path),
            relative_path=_relative_to_output(manifest_path, cfg),
            checksum=None,
            size_bytes=None,
            row_count=len(artifacts) + 1,
            status=ResultsExportStatus.COMPLETED,
            reasons=(ResultsExportReason.ALL_REQUESTED_ARTIFACTS_WRITTEN,),
            diagnostics=("manifest_self_checksum_omitted_to_avoid_recursive_identity",),
        )
        manifest = build_experiment_export_manifest(normalized, tuple(artifacts) + (manifest_placeholder,), cfg)
        manifest_artifact = _attempt(
            lambda: _write_manifest_artifact(normalized, manifest, cfg),
            normalized.analysis_id,
            "manifest",
            "json",
        )
        artifacts.append(manifest_artifact)
    else:
        artifacts.append(_skipped_artifact(normalized.analysis_id, "manifest", "json"))

    completed = tuple(
        item for item in artifacts
        if item.status in {ResultsExportStatus.COMPLETED, ResultsExportStatus.COMPLETED_WITH_RESERVATIONS}
    )
    skipped = tuple(
        item for item in artifacts
        if item.status is ResultsExportStatus.PARTIAL
        and ResultsExportReason.OPTIONAL_ARTIFACT_SKIPPED in item.reasons
    )
    failed = tuple(
        item for item in artifacts
        if item.status in {ResultsExportStatus.FAILED, ResultsExportStatus.INVALID_INPUT}
    )
    status = _export_status(result, completed, failed)
    reasons = _export_diagnostics(result, completed, failed)
    export_id = _stable_id(
        "export",
        normalized.analysis_id,
        normalized.experiment_id,
        export_settings_fingerprint(cfg),
        tuple((item.artifact_type, item.format, item.checksum) for item in artifacts),
    )
    return ExperimentExportResult(
        export_id=export_id,
        analysis_id=normalized.analysis_id,
        experiment_id=normalized.experiment_id,
        output_directory=str(Path(cfg.output_directory)),
        status=status,
        artifacts=tuple(artifacts),
        manifest=manifest,
        completed_artifacts=completed,
        skipped_artifacts=skipped,
        failed_artifacts=failed,
        artifact_count=len(artifacts),
        completed_count=len(completed),
        skipped_count=len(skipped),
        failed_count=len(failed),
        settings=cfg,
        valid=status in {ResultsExportStatus.COMPLETED, ResultsExportStatus.COMPLETED_WITH_RESERVATIONS},
        failure_reason=None if not failed else failed[0].diagnostics[0] if failed[0].diagnostics else failed[0].artifact_type,
        diagnostics=reasons,
    )


def summarize_experiment_export(result: ExperimentExportResult) -> dict[str, object]:
    """Return a small deterministic summary of an export run."""

    if not isinstance(result, ExperimentExportResult):
        raise TypeError("result must be an ExperimentExportResult.")
    return {
        "export_id": result.export_id,
        "analysis_id": result.analysis_id,
        "experiment_id": result.experiment_id,
        "status": result.status.value,
        "valid": result.valid,
        "artifact_count": result.artifact_count,
        "completed_count": result.completed_count,
        "skipped_count": result.skipped_count,
        "failed_count": result.failed_count,
        "artifacts": tuple(
            {
                "artifact_type": artifact.artifact_type,
                "format": artifact.format,
                "relative_path": artifact.relative_path,
                "status": artifact.status.value,
                "checksum": artifact.checksum,
                "row_count": artifact.row_count,
            }
            for artifact in result.artifacts
        ),
        "failure_reason": result.failure_reason,
    }


def validate_experiment_export(result: ExperimentExportResult) -> ExperimentExportValidation:
    """Validate artifact presence, checksums, JSON roundtrip and table metadata."""

    if not isinstance(result, ExperimentExportResult):
        raise TypeError("result must be an ExperimentExportResult.")
    expected = tuple(
        artifact.relative_path or artifact.artifact_type
        for artifact in result.artifacts
        if ResultsExportReason.OPTIONAL_ARTIFACT_SKIPPED not in artifact.reasons
    )
    existing: list[str] = []
    missing: list[str] = []
    checksum_matches: dict[str, bool] = {}
    for artifact in result.artifacts:
        if artifact.relative_path is None:
            continue
        path = Path(result.output_directory) / artifact.relative_path
        key = artifact.relative_path
        if path.is_file():
            existing.append(key)
            if artifact.checksum is not None:
                checksum_matches[key] = export_artifact_checksum(path) == artifact.checksum
        else:
            missing.append(key)
            checksum_matches[key] = False
    schema_valid = result.manifest is None or result.manifest.export_schema_version == BellLabExportSchemaVersion.V1_0.value
    row_counts_valid = all(
        artifact.row_count is None or artifact.row_count >= 0
        for artifact in result.artifacts
    )
    json_roundtrip = _json_roundtrip_valid(result)
    manifest_consistent = _manifest_consistent(result)
    foreign_keys_valid = _csv_foreign_keys_valid(result)
    valid = (
        not missing
        and all(checksum_matches.values() or (True,))
        and schema_valid
        and row_counts_valid
        and json_roundtrip
        and manifest_consistent
        and foreign_keys_valid
    )
    reasons = (
        (ResultsExportReason.ALL_REQUESTED_ARTIFACTS_WRITTEN,)
        if valid else (ResultsExportReason.CHECKSUM_FAILURE,)
    )
    return ExperimentExportValidation(
        expected_artifacts=expected,
        existing_artifacts=tuple(existing),
        missing_artifacts=tuple(missing),
        checksum_matches=checksum_matches,
        schema_valid=schema_valid,
        row_counts_valid=row_counts_valid,
        foreign_keys_valid=foreign_keys_valid,
        json_roundtrip_valid=json_roundtrip,
        manifest_consistent=manifest_consistent,
        valid=valid,
        reasons=reasons,
        diagnostics=(
            "json_roundtrip_reconstructs_structure_not_dataclasses",
            "csv_foreign_keys_checked_for_exported_ids",
        ),
    )


def export_settings_fingerprint(settings: ResultsExportSettings | None = None) -> str:
    """Fingerprint export settings without binding identity to output path."""

    cfg = settings or ResultsExportSettings()
    payload = {
        field.name: _to_serializable(getattr(cfg, field.name), cfg)
        for field in fields(cfg)
        if field.name != "output_directory"
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def export_artifact_checksum(path: str | Path, algorithm: str = "sha256") -> str:
    """Return a content checksum for a written artifact."""

    if algorithm != "sha256":
        raise ValueError("only sha256 checksums are supported.")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_export_tables(
    result: ExperimentAnalysisResult,
    cfg: ResultsExportSettings,
) -> Mapping[str, tuple[Mapping[str, object], ...]]:
    tables: dict[str, tuple[Mapping[str, object], ...]] = {
        "experiment_summary": (_experiment_summary_row(result),),
        "recordings": tuple(_recording_row(result, item) for item in _recordings(result)) if cfg.include_recordings else (),
        "conditions": tuple(_condition_row(item) for item in _conditions(result)) if cfg.include_conditions else (),
        "candidates": tuple(_candidate_rows(result)) if cfg.include_candidates else (),
        "within_condition_associations": tuple(_within_condition_rows(result)) if cfg.include_associations else (),
        "cross_condition_matches": tuple(_cross_condition_match_rows(result)) if cfg.include_associations else (),
        "candidate_chains": tuple(_chain_rows(result)) if cfg.include_chains else (),
        "candidate_chain_nodes": tuple(_chain_node_rows(result)) if cfg.include_chains else (),
        "modal_hypotheses": tuple(_hypothesis_rows(result, cfg)) if cfg.include_hypotheses else (),
        "modal_parameters": tuple(_parameter_rows(result, cfg)) if cfg.include_modal_parameters else (),
        "modal_q_factors": tuple(_q_rows(result, cfg)) if cfg.include_q_factors else (),
        "energy_exchange_pairs": tuple(_energy_rows(result, cfg)) if cfg.include_energy_exchange else (),
        "pipeline_stages": tuple(_stage_rows(result)) if cfg.include_stage_results else (),
        "diagnostics": tuple(_diagnostic_rows(result)) if cfg.include_diagnostics else (),
    }
    return MappingProxyType({
        key: tuple(MappingProxyType(dict(row)) for row in rows)
        for key, rows in tables.items()
    })


def _experiment_summary_row(result: ExperimentAnalysisResult) -> Mapping[str, object]:
    summary = summarize_experiment_analysis(result)
    return {
        **summary,
        "belllab_version": result.provenance.belllab_version,
        "settings_fingerprint": result.provenance.settings_fingerprint,
    }


def _experiment_row(result: ExperimentAnalysisResult) -> Mapping[str, object]:
    experiment = result.experiment
    return {
        "experiment_id": experiment.experiment_id,
        "name": experiment.name,
        "description": experiment.description,
        "specimen_id": experiment.specimen_id,
        "instrument_type": experiment.instrument_type,
        "location": experiment.location,
        "operator": experiment.operator,
        "acquisition_date": experiment.acquisition_date,
        "dynamic_labels": experiment.dynamic_labels,
        "reference_recording_id": experiment.reference_recording_id,
        "microphone": experiment.microphone,
        "audio_interface": experiment.audio_interface,
        "metadata": dict(experiment.metadata),
        "diagnostics": experiment.diagnostics,
    }


def _recording_row(
    result: ExperimentAnalysisResult,
    item: ExperimentRecordingAnalysisResult,
) -> Mapping[str, object]:
    definition = item.recording_definition
    loaded = item.loaded_recording
    metrics = loaded.metrics if loaded is not None else None
    return {
        "recording_id": definition.recording_id,
        "experiment_id": result.experiment.experiment_id,
        "dynamic_label": definition.dynamic_label,
        "take_index": definition.take_index,
        "file_path": str(definition.file_path),
        "file_fingerprint": loaded.file_fingerprint if loaded is not None else None,
        "sample_rate_hz": metrics.sample_rate_hz if metrics is not None else None,
        "channel_count": metrics.channel_count if metrics is not None else None,
        "selected_channel": loaded.selected_channel if loaded is not None else definition.channel,
        "original_duration_s": loaded.original_duration_s if loaded is not None else None,
        "analyzed_duration_s": loaded.analyzed_duration_s if loaded is not None else None,
        "offsets_applied": loaded.offsets_applied if loaded is not None else None,
        "candidate_count": len(item.modal_candidate_result),
        "accepted_candidate_count": sum(candidate.accepted for candidate in item.modal_candidate_result),
        "valid": item.valid,
        "failure_reason": item.failure_reason,
        "diagnostics": item.diagnostics,
    }


def _condition_row(item: ExperimentConditionAnalysisResult) -> Mapping[str, object]:
    within = item.within_condition_result
    return {
        "dynamic_label": item.dynamic_label,
        "selected_reference_recording_id": item.selected_reference_recording_id,
        "recording_count": len(item.recording_results),
        "valid_recording_count": item.valid_recording_count,
        "invalid_recording_count": item.invalid_recording_count,
        "candidate_reference_count": len(item.candidate_references),
        "within_condition_cluster_count": len(within.clusters) if within is not None else 0,
        "valid": item.valid,
        "failure_reason": item.failure_reason,
        "diagnostics": item.diagnostics,
    }


def _candidate_rows(result: ExperimentAnalysisResult) -> Iterable[Mapping[str, object]]:
    preimpact_by_recording = {
        item.recording_definition.recording_id: {
            evidence.source_track_id: evidence
            for evidence in item.preimpact_result
        }
        for item in result.recording_results
    }
    for recording in _recordings(result):
        recording_id = recording.recording_definition.recording_id
        evidence_by_track = preimpact_by_recording.get(recording_id, {})
        for candidate in sorted(recording.modal_candidate_result, key=lambda value: value.candidate_id):
            evidence = evidence_by_track.get(candidate.source_track_id)
            yield {
                "recording_id": recording_id,
                "dynamic_label": recording.recording_definition.dynamic_label,
                "candidate_id": candidate.candidate_id,
                "source_track_id": candidate.source_track_id,
                "representative_frequency_hz": candidate.representative_frequency_hz,
                "frequency_stability": candidate.frequency_stability,
                "frequency_drift_hz": candidate.frequency_drift_hz,
                "frequency_fit_rmse_hz": candidate.frequency_fit_rmse_hz,
                "amplitude_tau_s": candidate.amplitude_tau_s,
                "amplitude_fit_r_squared": candidate.amplitude_fit_r_squared,
                "coverage_fraction": candidate.coverage_fraction,
                "accepted": candidate.accepted,
                "impact_excited": evidence.impact_excited if evidence is not None else None,
                "preimpact_classification": evidence.classification if evidence is not None else None,
                "ambiguous_assignment_fraction": candidate.ambiguous_assignment_fraction,
                "near_threshold_assignment_fraction": candidate.near_threshold_assignment_fraction,
                "minimum_assignment_margin": candidate.minimum_assignment_margin,
                "acceptance_reasons": candidate.acceptance_reasons,
                "rejection_reasons": candidate.rejection_reasons,
                "diagnostics": candidate.diagnostics,
            }


def _within_condition_rows(result: ExperimentAnalysisResult) -> Iterable[Mapping[str, object]]:
    for condition in _conditions(result):
        within = condition.within_condition_result
        if within is None:
            continue
        for cluster in within.clusters:
            for ref in cluster.member_candidate_refs:
                yield {
                    "dynamic_label": within.dynamic_label,
                    "association_kind": "cluster_member",
                    "cluster_id": cluster.cluster_id,
                    "recording_id": ref.recording_id,
                    "candidate_id": ref.candidate_id,
                    "source_track_id": ref.source_track_id,
                    "representative_frequency_hz": ref.representative_frequency_hz,
                    "member_count": cluster.member_count,
                    "recording_count": cluster.recording_count,
                    "accepted": cluster.accepted,
                    "ambiguous": cluster.ambiguous,
                    "reason": _join_values(cluster.rejection_reasons),
                    "minimum_cost_observed": None,
                    "diagnostics": cluster.diagnostics,
                }
        for unmatched in within.unmatched_candidates:
            ref = unmatched.reference
            yield {
                "dynamic_label": within.dynamic_label,
                "association_kind": "unmatched",
                "cluster_id": None,
                "recording_id": ref.recording_id,
                "candidate_id": ref.candidate_id,
                "source_track_id": ref.source_track_id,
                "representative_frequency_hz": ref.representative_frequency_hz,
                "member_count": None,
                "recording_count": None,
                "accepted": False,
                "ambiguous": None,
                "reason": unmatched.reason,
                "minimum_cost_observed": unmatched.minimum_cost_observed,
                "diagnostics": unmatched.diagnostics,
            }


def _cross_condition_match_rows(result: ExperimentAnalysisResult) -> Iterable[Mapping[str, object]]:
    cross = result.cross_condition_result
    if cross is None:
        return
    for pair in cross.adjacent_pair_results:
        for match in pair.matches:
            yield {
                "match_id": match.match_id,
                "lower_dynamic_label": pair.lower_dynamic_label,
                "higher_dynamic_label": pair.higher_dynamic_label,
                "lower_recording_id": match.lower_candidate_ref.recording_id,
                "lower_candidate_id": match.lower_candidate_ref.candidate_id,
                "higher_recording_id": match.higher_candidate_ref.recording_id,
                "higher_candidate_id": match.higher_candidate_ref.candidate_id,
                "frequency_change_hz": match.frequency_change_hz,
                "frequency_change_relative": match.frequency_change_relative,
                "frequency_change_classification": match.frequency_change_classification,
                "association_cost": match.association_diagnostic.total_cost,
                "ambiguous": match.ambiguous,
                "near_threshold": match.near_threshold,
                "accepted": match.accepted,
                "diagnostics": match.diagnostics,
            }


def _chain_rows(result: ExperimentAnalysisResult) -> Iterable[Mapping[str, object]]:
    for chain_result in _chain_results(result):
        for chain in chain_result.chains:
            yield {
                "chain_id": chain.chain_id,
                "start_dynamic_label": chain.start_dynamic_label,
                "end_dynamic_label": chain.end_dynamic_label,
                "condition_count": chain.condition_count,
                "match_count": chain.match_count,
                "complete_across_requested_sequence": chain.complete_across_requested_sequence,
                "partial_chain": chain.partial_chain,
                "isolated_candidate": chain.isolated_candidate,
                "initial_frequency_hz": chain.initial_frequency_hz,
                "final_frequency_hz": chain.final_frequency_hz,
                "total_frequency_change_hz": chain.total_frequency_change_hz,
                "maximum_association_cost": chain.maximum_association_cost,
                "contains_ambiguous_match": chain.contains_ambiguous_match,
                "contains_near_threshold_match": chain.contains_near_threshold_match,
                "contains_possible_split_context": chain.contains_possible_split_context,
                "contains_possible_merge_context": chain.contains_possible_merge_context,
                "diagnostics": chain.diagnostics,
            }


def _chain_node_rows(result: ExperimentAnalysisResult) -> Iterable[Mapping[str, object]]:
    for chain_result in _chain_results(result):
        for chain in chain_result.chains:
            for index, node in enumerate(chain.nodes):
                ref = node.candidate_ref
                yield {
                    "chain_id": chain.chain_id,
                    "node_index": index,
                    "dynamic_label": node.dynamic_label,
                    "recording_id": ref.recording_id,
                    "candidate_id": ref.candidate_id,
                    "source_track_id": ref.source_track_id,
                    "representative_frequency_hz": ref.representative_frequency_hz,
                    "incoming_match_id": node.incoming_match_id,
                    "outgoing_match_id": node.outgoing_match_id,
                    "incoming_association_cost": node.incoming_association_cost,
                    "outgoing_association_cost": node.outgoing_association_cost,
                    "diagnostics": node.diagnostics,
                }


def _hypothesis_rows(
    result: ExperimentAnalysisResult,
    cfg: ResultsExportSettings,
) -> Iterable[Mapping[str, object]]:
    for hypothesis_result in _hypothesis_results(result):
        for hypothesis in hypothesis_result.hypotheses:
            if not _include_status(hypothesis.status.value, cfg):
                continue
            yield {
                "hypothesis_id": hypothesis.hypothesis_id,
                "source_chain_id": hypothesis.source_chain_id,
                "status": hypothesis.status.value,
                "accepted": hypothesis.accepted,
                "requires_review": hypothesis.requires_review,
                "score": hypothesis.score.normalized_score,
                "condition_coverage_fraction": hypothesis.coverage_evidence.condition_coverage_fraction,
                "maximum_step_change_hz": hypothesis.frequency_evidence.maximum_step_change_hz,
                "mean_match_cost": hypothesis.association_evidence.mean_match_cost,
                "mean_tracking_coverage_fraction": hypothesis.tracking_evidence.mean_coverage_fraction,
                "tau_available_count": hypothesis.decay_evidence.available_tau_count,
                "supporting_reasons": hypothesis.supporting_reasons,
                "reservation_reasons": hypothesis.reservation_reasons,
                "rejection_reasons": hypothesis.rejection_reasons,
                "missing_evidence_reasons": hypothesis.missing_evidence_reasons,
                "diagnostics": hypothesis.diagnostics,
            }


def _parameter_rows(
    result: ExperimentAnalysisResult,
    cfg: ResultsExportSettings,
) -> Iterable[Mapping[str, object]]:
    for parameter_result in _parameter_results(result):
        for estimate in parameter_result.estimates:
            if not _include_status(estimate.status.value, cfg):
                continue
            freq = estimate.frequency_estimate
            fun = estimate.frequency_uncertainty
            traj = estimate.frequency_trajectory
            decay = estimate.decay_estimate
            dun = estimate.decay_uncertainty
            rate = estimate.decay_rate_estimate
            yield {
                "estimate_id": estimate.estimate_id,
                "hypothesis_id": estimate.hypothesis_id,
                "status": estimate.status.value,
                "representative_frequency_hz": freq.representative_frequency_hz,
                "frequency_uncertainty_hz": fun.standard_uncertainty_hz,
                "frequency_lower_bound_hz": fun.lower_bound_hz,
                "frequency_upper_bound_hz": fun.upper_bound_hz,
                "frequency_range_hz": freq.frequency_range_hz,
                "relative_frequency_range": freq.relative_frequency_range,
                "trajectory_total_signed_change_hz": traj.total_signed_change_hz,
                "trajectory_slope_hz_per_condition_step": traj.linear_slope_hz_per_condition_step,
                "trajectory_rmse_hz": traj.linear_fit_rmse_hz,
                "representative_tau_s": decay.representative_tau_s,
                "tau_uncertainty_log": dun.standard_uncertainty_log_tau,
                "tau_lower_bound_s": dun.lower_bound_tau_s,
                "tau_upper_bound_s": dun.upper_bound_tau_s,
                "amplitude_decay_rate_per_s": rate.amplitude_decay_rate_per_s,
                "time_to_minus_20_db_s": rate.time_to_minus_20_db_s,
                "time_to_minus_40_db_s": rate.time_to_minus_40_db_s,
                "time_to_minus_60_db_s": rate.time_to_minus_60_db_s,
                "supporting_reasons": estimate.supporting_reasons,
                "reservation_reasons": estimate.reservation_reasons,
                "insufficient_evidence_reasons": estimate.insufficient_evidence_reasons,
                "invalid_reasons": estimate.invalid_reasons,
                "provenance_candidate_ids": estimate.provenance.candidate_ids,
                "provenance_match_ids": estimate.provenance.match_ids,
                "diagnostics": estimate.diagnostics,
            }


def _q_rows(
    result: ExperimentAnalysisResult,
    cfg: ResultsExportSettings,
) -> Iterable[Mapping[str, object]]:
    for q_result in _q_results(result):
        for estimate in q_result.estimates:
            if not _include_status(estimate.status.value, cfg):
                continue
            decay = estimate.decay_q_estimate
            bandwidth = estimate.bandwidth_estimate
            bandwidth_q = estimate.bandwidth_q_estimate
            comparison = estimate.method_comparison
            yield {
                "estimate_id": estimate.estimate_id,
                "modal_parameter_estimate_id": estimate.modal_parameter_estimate_id,
                "hypothesis_id": estimate.hypothesis_id,
                "status": estimate.status.value,
                "q_decay": decay.q_decay if decay is not None else None,
                "q_bandwidth": bandwidth_q.q_bandwidth if bandwidth_q is not None else None,
                "bandwidth_hz": bandwidth.bandwidth_hz if bandwidth is not None else None,
                "bandwidth_definition": bandwidth.bandwidth_definition.value if bandwidth is not None else None,
                "frequency_resolution_hz": bandwidth.frequency_resolution_hz if bandwidth is not None else None,
                "resolution_ratio": bandwidth.resolution_ratio if bandwidth is not None else None,
                "isolated_peak": bandwidth.isolated_peak if bandwidth is not None else None,
                "method_relative_symmetric_difference": comparison.relative_symmetric_difference if comparison is not None else None,
                "representative_q": estimate.representative_q,
                "representative_q_method": estimate.representative_q_method,
                "representative_q_uncertainty": estimate.representative_q_uncertainty,
                "supporting_reasons": estimate.supporting_reasons,
                "reservation_reasons": estimate.reservation_reasons,
                "inconclusive_reasons": estimate.inconclusive_reasons,
                "insufficient_evidence_reasons": estimate.insufficient_evidence_reasons,
                "invalid_reasons": estimate.invalid_reasons,
                "diagnostics": estimate.diagnostics,
            }


def _energy_rows(
    result: ExperimentAnalysisResult,
    cfg: ResultsExportSettings,
) -> Iterable[Mapping[str, object]]:
    for energy_result in result.energy_exchange_results:
        for evidence in energy_result.pair_evidences:
            if not _include_status(evidence.status.value, cfg):
                continue
            delayed_a, delayed_b = evidence.delayed_growth_evidence
            recovery_a, recovery_b = evidence.recovery_evidence
            yield {
                "evidence_id": evidence.evidence_id,
                "dynamic_label": evidence.dynamic_label,
                "source_a_id": evidence.source_a_id,
                "source_b_id": evidence.source_b_id,
                "status": evidence.status.value,
                "normalized_score": evidence.score.normalized_score,
                "opposed_trends": evidence.trend_evidence.opposed_trends,
                "zero_lag_correlation": evidence.correlation_evidence.zero_lag_correlation,
                "best_negative_lag_s": evidence.correlation_evidence.best_negative_lag_s,
                "best_negative_correlation": evidence.correlation_evidence.best_negative_correlation,
                "best_negative_p_value": evidence.correlation_evidence.best_negative_p_value,
                "delayed_growth_a": delayed_a.supported,
                "delayed_growth_b": delayed_b.supported,
                "recovery_a": recovery_a.supported,
                "recovery_b": recovery_b.supported,
                "pair_energy_relative_range": evidence.pair_energy_evidence.pair_energy_relative_range,
                "approximately_conserved_pair_energy": evidence.pair_energy_evidence.approximately_conserved,
                "alternating_dominance": evidence.alternating_dominance_evidence.alternating_dominance,
                "possible_beating": evidence.beating_evidence.possible_beating,
                "supporting_reasons": evidence.supporting_reasons,
                "reservation_reasons": evidence.reservation_reasons,
                "inconclusive_reasons": evidence.inconclusive_reasons,
                "not_supported_reasons": evidence.not_supported_reasons,
                "diagnostics": evidence.diagnostics,
            }


def _stage_rows(result: ExperimentAnalysisResult) -> Iterable[Mapping[str, object]]:
    for stage in sorted(result.stage_results, key=lambda item: item.stage.value):
        yield {
            "stage": stage.stage.value,
            "status": stage.status.value,
            "started": stage.started,
            "completed": stage.completed,
            "input_ids": stage.input_ids,
            "output_ids": stage.output_ids,
            "dependency_stages": tuple(item.value for item in stage.dependency_stages),
            "supporting_reasons": stage.supporting_reasons,
            "reservation_reasons": stage.reservation_reasons,
            "skipped_reasons": stage.skipped_reasons,
            "blocked_reasons": stage.blocked_reasons,
            "failure_reasons": stage.failure_reasons,
            "insufficient_evidence_reasons": stage.insufficient_evidence_reasons,
            "diagnostics": stage.diagnostics,
        }


def _diagnostic_rows(result: ExperimentAnalysisResult) -> Iterable[Mapping[str, object]]:
    sources: list[tuple[str, str, Sequence[str]]] = [
        ("analysis", result.analysis_id, result.diagnostics),
        ("input_validation", result.analysis_id, result.input_validation.diagnostics if result.input_validation is not None else ()),
        ("provenance", result.analysis_id, result.provenance.diagnostics),
    ]
    sources.extend(
        ("recording", item.recording_definition.recording_id or "", item.diagnostics)
        for item in result.recording_results
    )
    sources.extend(("condition", item.dynamic_label, item.diagnostics) for item in result.condition_results)
    if result.cross_condition_result is not None:
        sources.append(("cross_condition", "-".join(result.cross_condition_result.dynamic_labels), result.cross_condition_result.diagnostics))
    for scope, source_id, diagnostics in sources:
        for index, diagnostic in enumerate(tuple(dict.fromkeys(diagnostics))):
            yield {
                "scope": scope,
                "source_id": source_id,
                "diagnostic_index": index,
                "diagnostic": diagnostic,
            }


def _cross_condition_row(
    cross: ExperimentCrossConditionAnalysisResult | None,
) -> Mapping[str, object] | None:
    if cross is None:
        return None
    return {
        "dynamic_labels": cross.dynamic_labels,
        "adjacent_pair_count": len(cross.adjacent_pair_results),
        "candidate_chain_result_count": len(cross.candidate_chain_results),
        "modal_hypothesis_result_count": len(cross.modal_hypothesis_results),
        "modal_parameter_result_count": len(cross.modal_parameter_results),
        "modal_q_result_count": len(cross.modal_q_results),
        "valid": cross.valid,
        "failure_reason": cross.failure_reason,
        "diagnostics": cross.diagnostics,
    }


def _recordings(result: ExperimentAnalysisResult) -> tuple[ExperimentRecordingAnalysisResult, ...]:
    return tuple(sorted(result.recording_results, key=lambda item: item.recording_definition.recording_id or ""))


def _conditions(result: ExperimentAnalysisResult) -> tuple[ExperimentConditionAnalysisResult, ...]:
    order = {"pp": 0, "p": 1, "mf": 2, "f": 3, "ff": 4}
    return tuple(sorted(result.condition_results, key=lambda item: (order.get(item.dynamic_label, 99), item.dynamic_label)))


def _chain_results(result: ExperimentAnalysisResult) -> tuple[Any, ...]:
    cross = result.cross_condition_result
    return tuple(cross.candidate_chain_results if cross is not None else ())


def _hypothesis_results(result: ExperimentAnalysisResult) -> tuple[Any, ...]:
    cross = result.cross_condition_result
    return tuple(cross.modal_hypothesis_results if cross is not None else ())


def _parameter_results(result: ExperimentAnalysisResult) -> tuple[Any, ...]:
    cross = result.cross_condition_result
    return tuple(cross.modal_parameter_results if cross is not None else ())


def _q_results(result: ExperimentAnalysisResult) -> tuple[Any, ...]:
    cross = result.cross_condition_result
    return tuple(cross.modal_q_results if cross is not None else ())


def _include_status(status: str, cfg: ResultsExportSettings) -> bool:
    if status in {"invalid_input", "failed"} and not cfg.include_invalid_results:
        return False
    if status in {"rejected", "not_supported"} and not cfg.include_rejected_results:
        return False
    if status in {"inconclusive", "insufficient_evidence", "partial"} and not cfg.include_inconclusive_results:
        return False
    return True


def _selected_csv_tables(
    tables: Mapping[str, tuple[Mapping[str, object], ...]],
    cfg: ResultsExportSettings,
) -> Mapping[str, tuple[Mapping[str, object], ...]]:
    if cfg.export_intermediate_tables:
        return tables
    keep = {"experiment_summary", "recordings", "conditions", "pipeline_stages", "diagnostics"}
    return MappingProxyType({name: rows for name, rows in tables.items() if name in keep})


def _csv_content(
    table_name: str,
    rows: Sequence[Mapping[str, object]],
    cfg: ResultsExportSettings,
) -> str:
    import io

    columns = _columns_for_table(table_name, rows, cfg)
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: _format_cell(row.get(column), cfg, "csv") for column in columns})
    return handle.getvalue()


def _latex_table_content(
    table_name: str,
    rows: Sequence[Mapping[str, object]],
    cfg: ResultsExportSettings,
) -> str:
    columns = _columns_for_table(table_name, rows, cfg)
    if len(columns) > 8:
        columns = columns[:8]
    if not columns:
        columns = ("status",)
        rows = ({"status": None},)
    align = "l" * len(columns)
    lines = [
        f"\\begin{{tabular}}{{{align}}}",
    ]
    if cfg.latex_booktabs:
        lines.append("\\toprule")
    lines.append(" & ".join(_latex_escape(_header(column), cfg) for column in columns) + r" \\")
    lines.append("\\midrule" if cfg.latex_booktabs else "\\hline")
    for row in rows:
        lines.append(" & ".join(_latex_escape(_format_cell(row.get(column), cfg, "latex"), cfg) for column in columns) + r" \\")
    if cfg.latex_booktabs:
        lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"% label: tab:belllab-{_sanitize_file_part(table_name)}")
    return "\n".join(lines) + "\n"


def _markdown_content(
    normalized: NormalizedExperimentExport,
    cfg: ResultsExportSettings,
) -> str:
    tables = normalized.tables
    summary = normalized.summary
    lines = [
        f"# BellLab Export: {summary.get('name')}",
        "",
        "Exportacao bem-sucedida nao e resultado cientificamente valido por definicao.",
        "",
        "## Identificacao do experimento",
        "",
        _markdown_kv({
            "analysis_id": normalized.analysis_id,
            "experiment_id": normalized.experiment_id,
            "status": summary.get("status"),
            "belllab_version": normalized.belllab_version,
        }),
        "",
        "## Gravacoes",
        "",
        _markdown_table(tables.get("recordings", ()), cfg, limit=8),
        "",
        "## Condicoes",
        "",
        _markdown_table(tables.get("conditions", ()), cfg, limit=8),
        "",
        "## Estagios",
        "",
        _markdown_table(tables.get("pipeline_stages", ()), cfg, limit=12),
        "",
        "## Principais contagens",
        "",
        _markdown_kv({
            "candidates": summary.get("candidate_count"),
            "chains": summary.get("chain_count"),
            "hypotheses": summary.get("hypothesis_count"),
            "parameters": summary.get("parameter_estimate_count"),
            "q_estimates": summary.get("q_estimate_count"),
            "operational_energy_pairs": summary.get("energy_pair_count"),
        }),
        "",
        "## Hipoteses",
        "",
        _markdown_table(tables.get("modal_hypotheses", ()), cfg, limit=8),
        "",
        "## Parametros",
        "",
        _markdown_table(tables.get("modal_parameters", ()), cfg, limit=8),
        "",
        "## Q",
        "",
        _markdown_table(tables.get("modal_q_factors", ()), cfg, limit=8),
        "",
        "## Evidencia operacional de possivel redistribuicao",
        "",
        _markdown_table(tables.get("energy_exchange_pairs", ()), cfg, limit=8),
        "",
        "## Falhas",
        "",
        _markdown_table(_failure_rows(normalized), cfg, limit=12),
        "",
        "## Ressalvas",
        "",
        "Valores ausentes permanecem ausentes; hipoteses modais nao sao modos fisicos comprovados; evidencia operacional de possivel redistribuicao nao comprova transferencia fisica.",
        "",
        "## Proveniencia",
        "",
        _markdown_kv({
            "settings_fingerprint": normalized.provenance.get("settings_fingerprint"),
            "export_settings_fingerprint": normalized.settings.get("export_settings_fingerprint"),
        }),
        "",
        "## Limitacoes",
        "",
        "- Esta exportacao nao recalcula analise cientifica.",
        "- Esta exportacao nao gera PDF, HTML, dashboard, figuras finais ou narrativa fisica automatica.",
        "- Arquivo reproduzivel nao significa experimento fisicamente reproduzido.",
        "",
    ]
    return "\n".join(lines)


def _failure_rows(normalized: NormalizedExperimentExport) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = []
    for table_name in ("pipeline_stages", "modal_hypotheses", "modal_parameters", "modal_q_factors", "energy_exchange_pairs"):
        for row in normalized.tables.get(table_name, ()):
            status = str(row.get("status") or "")
            if status in {"failed", "invalid_input", "rejected", "inconclusive", "insufficient_evidence", "partial", "not_supported"}:
                rows.append({
                    "source_table": table_name,
                    "source_id": row.get("estimate_id") or row.get("hypothesis_id") or row.get("evidence_id") or row.get("stage"),
                    "status": status,
                    "failure_or_reservation": row.get("failure_reasons") or row.get("reservation_reasons") or row.get("invalid_reasons") or row.get("rejection_reasons") or row.get("not_supported_reasons"),
                })
    return tuple(rows)


def _export_summary_json(
    normalized: NormalizedExperimentExport,
    cfg: ResultsExportSettings,
) -> ExportedArtifact:
    payload = {
        "schema_version": normalized.schema_version.value,
        "analysis_id": normalized.analysis_id,
        "experiment_id": normalized.experiment_id,
        "summary": normalized.summary,
        "provenance": normalized.provenance,
    }
    content = json.dumps(_to_serializable(payload, cfg), ensure_ascii=False, indent=cfg.json_indent, sort_keys=True, allow_nan=False) + "\n"
    path = _artifact_path(cfg, normalized, "summary", "json")
    return _write_artifact(normalized.analysis_id, "summary", "json", path, content, cfg, row_count=1)


def _write_manifest_artifact(
    normalized: NormalizedExperimentExport,
    manifest: ExperimentExportManifest,
    cfg: ResultsExportSettings,
) -> ExportedArtifact:
    content = json.dumps(_to_serializable(manifest, cfg), ensure_ascii=False, indent=cfg.json_indent, sort_keys=True, allow_nan=False) + "\n"
    path = _artifact_path(cfg, normalized, "manifest", "json")
    return _write_artifact(normalized.analysis_id, "manifest", "json", path, content, cfg, row_count=len(manifest.generated_artifacts))


def _write_artifact(
    analysis_id: str,
    artifact_type: str,
    artifact_format: str,
    path: Path,
    content: str,
    cfg: ResultsExportSettings,
    *,
    row_count: int | None,
) -> ExportedArtifact:
    try:
        actual_path, skipped = _write_text(path, content, cfg)
        if skipped:
            checksum = export_artifact_checksum(actual_path) if actual_path.is_file() and cfg.write_checksums else None
            size = actual_path.stat().st_size if actual_path.is_file() else None
            relative_path = _relative_to_output(actual_path, cfg)
            return ExportedArtifact(
                artifact_id=_stable_id("artifact", analysis_id, artifact_type, artifact_format, relative_path, checksum),
                artifact_type=artifact_type,
                format=artifact_format,
                path=str(actual_path),
                relative_path=relative_path,
                checksum=checksum,
                size_bytes=size,
                row_count=row_count,
                status=ResultsExportStatus.PARTIAL,
                reasons=(ResultsExportReason.OPTIONAL_ARTIFACT_SKIPPED, ResultsExportReason.EXISTING_FILE_CONFLICT),
                diagnostics=("existing_artifact_preserved_by_skip_policy",),
            )
        checksum = export_artifact_checksum(actual_path) if cfg.write_checksums else None
        return ExportedArtifact(
            artifact_id=_stable_id("artifact", analysis_id, artifact_type, artifact_format, checksum),
            artifact_type=artifact_type,
            format=artifact_format,
            path=str(actual_path),
            relative_path=_relative_to_output(actual_path, cfg),
            checksum=checksum,
            size_bytes=actual_path.stat().st_size,
            row_count=row_count,
            status=ResultsExportStatus.COMPLETED,
            reasons=(ResultsExportReason.ALL_REQUESTED_ARTIFACTS_WRITTEN,),
            diagnostics=("artifact_written_without_recalculating_analysis",),
        )
    except FileExistsError as exc:
        return _failed_artifact(analysis_id, artifact_type, artifact_format, ResultsExportReason.EXISTING_FILE_CONFLICT, str(exc))
    except Exception as exc:
        return _failed_artifact(analysis_id, artifact_type, artifact_format, ResultsExportReason.FILESYSTEM_ERROR, f"{exc.__class__.__name__}: {exc}")


def _write_text(path: Path, content: str, cfg: ResultsExportSettings) -> tuple[Path, bool]:
    directory = path.parent
    if not directory.exists():
        if cfg.create_output_directory:
            directory.mkdir(parents=True, exist_ok=True)
        else:
            raise FileNotFoundError(f"output directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {directory}")
    target = _resolve_overwrite_path(path, cfg)
    if target is None:
        return path, True
    if cfg.atomic_write:
        tmp = target.with_name(f".{target.name}.tmp")
        try:
            with tmp.open("w", encoding="utf-8", newline="") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            tmp.replace(target)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise
    else:
        with target.open("w", encoding="utf-8", newline="") as handle:
            handle.write(content)
    return target, False


def _resolve_overwrite_path(path: Path, cfg: ResultsExportSettings) -> Path | None:
    if not path.exists():
        return path
    if cfg.overwrite_policy is ExportOverwritePolicy.ERROR:
        raise FileExistsError(f"artifact already exists: {path}")
    if cfg.overwrite_policy is ExportOverwritePolicy.SKIP:
        return None
    if cfg.overwrite_policy is ExportOverwritePolicy.REPLACE:
        return path
    if cfg.overwrite_policy is ExportOverwritePolicy.VERSIONED_FILENAME:
        for index in range(1, 10_000):
            candidate = path.with_name(f"{path.stem}_v{index:03d}{path.suffix}")
            if not candidate.exists():
                return candidate
    raise FileExistsError(f"could not create versioned filename for: {path}")


def _artifact_path(
    cfg: ResultsExportSettings,
    normalized: NormalizedExperimentExport,
    artifact_name: str,
    extension: str,
) -> Path:
    parts = [_sanitize_file_part(cfg.file_prefix)]
    if cfg.include_experiment_id_in_filename and normalized.experiment_id:
        parts.append(_sanitize_file_part(normalized.experiment_id))
    if cfg.include_analysis_id_in_filename:
        parts.append(_sanitize_file_part(normalized.analysis_id))
    parts.append(_sanitize_file_part(artifact_name))
    return Path(cfg.output_directory) / f"{'_'.join(parts)}.{extension}"


def _manifest_artifact_view(artifact: ExportedArtifact) -> ExportedArtifact:
    """Return an artifact record suitable for a path-portable manifest."""

    return ExportedArtifact(
        artifact_id=artifact.artifact_id,
        artifact_type=artifact.artifact_type,
        format=artifact.format,
        path=artifact.relative_path,
        relative_path=artifact.relative_path,
        checksum=artifact.checksum,
        size_bytes=artifact.size_bytes,
        row_count=artifact.row_count,
        status=artifact.status,
        reasons=artifact.reasons,
        diagnostics=artifact.diagnostics,
    )


def _relative_to_output(path: Path, cfg: ResultsExportSettings) -> str:
    try:
        return str(path.relative_to(Path(cfg.output_directory)))
    except ValueError:
        return path.name


def _attempt(fn: Any, analysis_id: str, artifact_type: str, artifact_format: str) -> ExportedArtifact:
    try:
        return fn()
    except Exception as exc:
        return _failed_artifact(
            analysis_id,
            artifact_type,
            artifact_format,
            ResultsExportReason.FILESYSTEM_ERROR,
            f"{exc.__class__.__name__}: {exc}",
        )


def _attempt_many(
    fn: Any,
    analysis_id: str,
    artifact_type: str,
    artifact_format: str,
) -> tuple[ExportedArtifact, ...]:
    try:
        return tuple(fn())
    except Exception as exc:
        return (
            _failed_artifact(
                analysis_id,
                artifact_type,
                artifact_format,
                ResultsExportReason.FILESYSTEM_ERROR,
                f"{exc.__class__.__name__}: {exc}",
            ),
        )


def _skipped_artifact(
    analysis_id: str,
    artifact_type: str,
    artifact_format: str,
    reason: ResultsExportReason = ResultsExportReason.OPTIONAL_ARTIFACT_SKIPPED,
) -> ExportedArtifact:
    return ExportedArtifact(
        artifact_id=_stable_id("artifact", analysis_id, artifact_type, artifact_format, "skipped"),
        artifact_type=artifact_type,
        format=artifact_format,
        path=None,
        relative_path=None,
        checksum=None,
        size_bytes=None,
        row_count=None,
        status=ResultsExportStatus.PARTIAL,
        reasons=(reason, ResultsExportReason.OPTIONAL_ARTIFACT_SKIPPED),
        diagnostics=("artifact_not_requested_or_content_unavailable",),
    )


def _failed_artifact(
    analysis_id: str,
    artifact_type: str,
    artifact_format: str,
    reason: ResultsExportReason,
    diagnostic: str,
) -> ExportedArtifact:
    return ExportedArtifact(
        artifact_id=_stable_id("artifact", analysis_id, artifact_type, artifact_format, "failed", diagnostic),
        artifact_type=artifact_type,
        format=artifact_format,
        path=None,
        relative_path=None,
        checksum=None,
        size_bytes=None,
        row_count=None,
        status=ResultsExportStatus.FAILED,
        reasons=(reason,),
        diagnostics=(diagnostic,),
    )


def _invalid_export_result(
    analysis_id: str,
    experiment_id: str | None,
    cfg: ResultsExportSettings,
    reason: ResultsExportReason,
    diagnostics: tuple[str, ...] = (),
) -> ExperimentExportResult:
    failed = _failed_artifact(analysis_id, "export", "none", reason, diagnostics[0] if diagnostics else reason.value)
    export_id = _stable_id("export", analysis_id, experiment_id, reason.value)
    return ExperimentExportResult(
        export_id=export_id,
        analysis_id=analysis_id,
        experiment_id=experiment_id,
        output_directory=str(Path(cfg.output_directory)),
        status=ResultsExportStatus.INVALID_INPUT,
        artifacts=(failed,),
        manifest=None,
        completed_artifacts=(),
        skipped_artifacts=(),
        failed_artifacts=(failed,),
        artifact_count=1,
        completed_count=0,
        skipped_count=0,
        failed_count=1,
        settings=cfg,
        valid=False,
        failure_reason=reason.value,
        diagnostics=diagnostics,
    )


def _export_status(
    result: ExperimentAnalysisResult,
    completed: tuple[ExportedArtifact, ...],
    failed: tuple[ExportedArtifact, ...],
) -> ResultsExportStatus:
    if failed and completed:
        return ResultsExportStatus.PARTIAL
    if failed:
        return ResultsExportStatus.FAILED
    if result.status is ExperimentAnalysisStatus.INVALID_INPUT:
        return ResultsExportStatus.COMPLETED_WITH_RESERVATIONS
    if result.requires_review:
        return ResultsExportStatus.COMPLETED_WITH_RESERVATIONS
    return ResultsExportStatus.COMPLETED


def _export_diagnostics(
    result: ExperimentAnalysisResult,
    completed: tuple[ExportedArtifact, ...],
    failed: tuple[ExportedArtifact, ...],
) -> tuple[str, ...]:
    diagnostics = [
        "export_did_not_recalculate_scientific_analysis",
        "source_audio_files_not_modified",
        "missing_values_not_replaced_by_zero",
        "rejected_and_inconclusive_results_preserved_when_configured",
        "export_is_not_physical_validity_proof",
    ]
    if result.requires_review:
        diagnostics.append(ResultsExportReason.SOURCE_RESULT_REQUIRES_REVIEW.value)
    if failed:
        diagnostics.append(ResultsExportReason.FILESYSTEM_ERROR.value)
    if completed and not failed:
        diagnostics.append(ResultsExportReason.ALL_REQUESTED_ARTIFACTS_WRITTEN.value)
    return tuple(dict.fromkeys(diagnostics))


def _to_serializable(
    value: object,
    cfg: ResultsExportSettings,
    diagnostics: list[str] | None = None,
    path: str = "$",
) -> object:
    diag = diagnostics if diagnostics is not None else []
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _to_serializable(getattr(value, field.name), cfg, diag, f"{path}.{field.name}")
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _to_serializable(val, cfg, diag, f"{path}.{key}")
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_to_serializable(item, cfg, diag, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value if cfg.preserve_full_precision_in_json else round(value, cfg.float_precision)
        diag.append(f"nonfinite_value:{path}")
        if cfg.nonfinite_value_policy is ExportNonfiniteValuePolicy.ERROR:
            raise ValueError(f"nonfinite value at {path}")
        if cfg.nonfinite_value_policy is ExportNonfiniteValuePolicy.NULL_WITH_DIAGNOSTIC:
            return None
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, bytes):
        return value.hex()
    return str(value)


def _export_settings_payload(cfg: ResultsExportSettings) -> Mapping[str, object]:
    payload = {
        field.name: _to_serializable(getattr(cfg, field.name), cfg)
        for field in fields(cfg)
        if field.name != "output_directory"
    }
    payload["output_directory"] = "omitted_from_normalized_export_identity"
    return MappingProxyType(payload)


def _assert_no_nonfinite_tokens(
    normalized: NormalizedExperimentExport,
    cfg: ResultsExportSettings,
) -> None:
    if cfg.nonfinite_value_policy is not ExportNonfiniteValuePolicy.ERROR:
        _to_serializable(normalized, cfg)
        return
    _to_serializable(normalized, cfg)


def _format_cell(value: object, cfg: ResultsExportSettings, format_name: str) -> str:
    if value is None:
        return _missing_value(cfg, format_name)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (tuple, list)):
        return ";".join(_format_cell(item, cfg, format_name) for item in value)
    if isinstance(value, Mapping):
        return json.dumps(_to_serializable(value, cfg), ensure_ascii=False, sort_keys=True, allow_nan=False)
    if isinstance(value, float):
        if not math.isfinite(value):
            if cfg.nonfinite_value_policy is ExportNonfiniteValuePolicy.ERROR:
                raise ValueError("nonfinite value detected during formatting")
            if cfg.nonfinite_value_policy is ExportNonfiniteValuePolicy.NULL_WITH_DIAGNOSTIC:
                return _missing_value(cfg, format_name)
            if math.isnan(value):
                return "NaN"
            return "Infinity" if value > 0 else "-Infinity"
        if cfg.numeric_formatting.use_scientific_notation or abs(value) >= cfg.scientific_notation_threshold:
            text = f"{value:.{cfg.float_precision}e}"
        else:
            text = f"{value:.{cfg.float_precision}f}".rstrip("0").rstrip(".")
        if cfg.numeric_formatting.decimal_separator == ",":
            text = text.replace(".", ",")
        return text
    return str(value)


def _missing_value(cfg: ResultsExportSettings, format_name: str) -> str:
    policy = cfg.missing_value_representation
    if policy is ExportMissingValuePolicy.NULL:
        return "null" if format_name != "csv" else ""
    if policy is ExportMissingValuePolicy.EMPTY:
        return ""
    if policy is ExportMissingValuePolicy.NA:
        return "NA"
    return "-"


def _columns_for_table(
    table_name: str,
    rows: Sequence[Mapping[str, object]],
    cfg: ResultsExportSettings,
) -> tuple[str, ...]:
    if table_name in cfg.column_selection:
        return cfg.column_selection[table_name]
    if table_name in cfg.column_order:
        return cfg.column_order[table_name]
    preferred = CSV_COLUMN_ORDER.get(table_name, ())
    present = tuple(sorted({str(key) for row in rows for key in row}))
    return tuple(column for column in preferred if column in present) + tuple(
        column for column in present if column not in preferred
    )


def _write_empty_table(table_name: str) -> bool:
    return table_name in {"experiment_summary", "recordings", "conditions", "pipeline_stages", "diagnostics"}


def _markdown_table(
    rows: Sequence[Mapping[str, object]],
    cfg: ResultsExportSettings,
    *,
    limit: int,
) -> str:
    if not rows:
        return "_Sem linhas exportaveis._"
    rows = tuple(rows[:limit])
    columns = _columns_for_table("markdown", rows, cfg)[:6]
    header = "| " + " | ".join(_header(column) for column in columns) + " |"
    divider = "| " + " | ".join(":---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_markdown_escape(_format_cell(row.get(column), cfg, "markdown")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider] + body)


def _markdown_kv(values: Mapping[str, object]) -> str:
    return "\n".join(f"- `{key}`: `{value}`" for key, value in values.items())


def _header(column: str) -> str:
    units = {
        "representative_frequency_hz": "representative frequency (Hz)",
        "frequency_uncertainty_hz": "frequency uncertainty (Hz)",
        "representative_tau_s": "representative tau (s)",
        "bandwidth_hz": "bandwidth (Hz)",
        "q_decay": "Q decay",
        "q_bandwidth": "Q bandwidth",
    }
    return units.get(column, column.replace("_", " "))


def _latex_escape(value: str, cfg: ResultsExportSettings) -> str:
    if not cfg.latex_escape_text:
        return value
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
    }
    return "".join(replacements.get(char, char) for char in value)


def _markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _join_values(values: object) -> str | None:
    if values is None:
        return None
    if isinstance(values, (tuple, list)):
        return ";".join(str(item.value if isinstance(item, Enum) else item) for item in values)
    return str(values)


def _sanitize_file_part(value: str) -> str:
    text = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return text.strip("_") or "artifact"


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
        return {str(key): _canonicalize(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
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


def _coerce_enum(value: object, enum_type: type[Enum]) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)  # type: ignore[call-arg]
    except ValueError as exc:
        raise ValueError(f"{enum_type.__name__} value is not recognized.") from exc


def _reason_tuple(values: Iterable[ResultsExportReason]) -> tuple[ResultsExportReason, ...]:
    return tuple(dict.fromkeys(
        value if isinstance(value, ResultsExportReason) else ResultsExportReason(value)
        for value in values
    ))


def _nonempty_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")


def _json_roundtrip_valid(result: ExperimentExportResult) -> bool:
    for artifact in result.artifacts:
        if artifact.format != "json" or artifact.relative_path is None or artifact.artifact_type == "manifest":
            continue
        path = Path(result.output_directory) / artifact.relative_path
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return False
        if "analysis_id" in payload and payload["analysis_id"] != result.analysis_id:
            return False
    return True


def _manifest_consistent(result: ExperimentExportResult) -> bool:
    if result.manifest is None:
        return True
    manifest_paths = set(result.manifest.artifact_checksums)
    artifact_paths = {
        artifact.relative_path or artifact.artifact_type
        for artifact in result.artifacts
    }
    return manifest_paths.issubset(artifact_paths)


def _csv_foreign_keys_valid(result: ExperimentExportResult) -> bool:
    csv_paths = {
        artifact.artifact_type: Path(result.output_directory) / artifact.relative_path
        for artifact in result.artifacts
        if artifact.format == "csv" and artifact.relative_path is not None
    }
    recordings_path = csv_paths.get("recordings")
    candidates_path = csv_paths.get("candidates")
    if recordings_path is None or candidates_path is None or not recordings_path.exists() or not candidates_path.exists():
        return True
    with recordings_path.open("r", encoding="utf-8", newline="") as handle:
        recording_ids = {row["recording_id"] for row in csv.DictReader(handle)}
    with candidates_path.open("r", encoding="utf-8", newline="") as handle:
        return all(row["recording_id"] in recording_ids for row in csv.DictReader(handle) if row.get("recording_id"))
