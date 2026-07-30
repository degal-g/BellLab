"""Reproducible scientific report generation for existing BellLab results.

This module organizes already computed BellLab analysis, export tables and
figures into Markdown, LaTeX and optional PDF artifacts.  It does not reopen
audio files, rerun FFT/STFT, rebuild tracks, regenerate figures, recalculate
tables, reinterpret modal hypotheses, or create physical conclusions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from belllab.experiment_pipeline import (
    ExperimentAnalysisResult,
    ExperimentAnalysisStatus,
    summarize_experiment_analysis,
)
from belllab.results_export import (
    ExportMissingValuePolicy,
    ExportNumericFormatting,
    ExportOverwritePolicy,
    ExperimentExportResult,
    NormalizedExperimentExport,
    ResultsExportSettings,
    export_artifact_checksum,
    normalize_experiment_for_export,
)
from belllab.scientific_visualizations import (
    ScientificFigureArtifact,
    ScientificFigureCollection,
    ScientificVisualizationStatus,
)


class ScientificReportStatus(str, Enum):
    """Mutually exclusive status for scientific report generation."""

    CREATED = "created"
    CREATED_WITH_RESERVATIONS = "created_with_reservations"
    PARTIAL = "partial"
    COMPILATION_FAILED = "compilation_failed"
    FAILED = "failed"
    INVALID_INPUT = "invalid_input"


class ScientificReportReason(str, Enum):
    """Typed reasons for report support, reservations and failures."""

    ALL_REQUESTED_SECTIONS_CREATED = "all_requested_sections_created"
    ALL_REQUESTED_ARTIFACTS_INCLUDED = "all_requested_artifacts_included"
    CROSS_REFERENCES_VALID = "cross_references_valid"
    MANIFEST_CREATED = "manifest_created"
    PDF_COMPILED = "pdf_compiled"
    SOURCE_ANALYSIS_PARTIAL = "source_analysis_partial"
    SOURCE_ANALYSIS_REQUIRES_REVIEW = "source_analysis_requires_review"
    MISSING_OPTIONAL_SECTION = "missing_optional_section"
    MISSING_OPTIONAL_TABLE = "missing_optional_table"
    MISSING_OPTIONAL_FIGURE = "missing_optional_figure"
    SOURCE_RESULT_INCONCLUSIVE = "source_result_inconclusive"
    SOURCE_RESULT_INSUFFICIENT = "source_result_insufficient"
    SOURCE_RESULT_INVALID = "source_result_invalid"
    LATEX_COMPILER_UNAVAILABLE = "latex_compiler_unavailable"
    LATEX_COMPILATION_WARNING = "latex_compilation_warning"
    UNRESOLVED_CROSS_REFERENCE = "unresolved_cross_reference"
    LONG_TABLE_SPLIT = "long_table_split"
    FIGURE_DOWNSCALED = "figure_downscaled"
    EXCESSIVE_APPENDIX_SIZE = "excessive_appendix_size"
    MISSING_REQUIRED_SOURCE = "missing_required_source"
    TEMPLATE_RENDERING_FAILURE = "template_rendering_failure"
    FILESYSTEM_ERROR = "filesystem_error"
    LATEX_COMPILATION_FAILURE = "latex_compilation_failure"
    INVALID_CROSS_REFERENCE = "invalid_cross_reference"
    ARTIFACT_CHECKSUM_MISMATCH = "artifact_checksum_mismatch"
    UNSUPPORTED_REPORT_FORMAT = "unsupported_report_format"
    INVALID_CONFIGURATION = "invalid_configuration"
    EXISTING_FILE_CONFLICT = "existing_file_conflict"


class ScientificReportSchemaVersion(str, Enum):
    """Version of the normalized BellLab scientific report schema."""

    V1_0 = "1.0"


class ScientificReportContentBlockType(str, Enum):
    """Typed content block identifiers in the normalized report document."""

    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    FIGURE = "figure"
    EQUATION = "equation"
    SCIENTIFIC_NOTICE = "scientific_notice"
    LIMITATION = "limitation"
    PROVENANCE = "provenance"
    DIAGNOSTIC = "diagnostic"
    CROSS_REFERENCE = "cross_reference"
    PAGE_BREAK = "page_break"


REPORT_SCHEMA_DESCRIPTION: Mapping[str, object] = MappingProxyType({
    "schema_version": ScientificReportSchemaVersion.V1_0.value,
    "required_fields": (
        "report_id",
        "analysis_id",
        "experiment_id",
        "sections",
        "figures",
        "tables",
        "provenance",
        "settings_fingerprint",
    ),
    "optional_fields": (
        "appendices",
        "references",
        "diagnostics",
        "compilation",
    ),
    "minimum_compatible_schema": "1.0",
    "missing_value_meaning": "missing values mean unavailable, not zero.",
})


DEFAULT_SECTION_ORDER: tuple[str, ...] = (
    "title_page",
    "abstract",
    "experiment_description",
    "acquisition_metadata",
    "methodology",
    "recording_quality",
    "temporal_results",
    "spectral_results",
    "tracking_results",
    "candidate_results",
    "associations",
    "candidate_chains",
    "modal_hypotheses",
    "modal_parameters",
    "q_factors",
    "energy_exchange",
    "synthetic_validation",
    "factual_synthesis",
    "limitations",
    "provenance_reproducibility",
    "appendices",
)


FORBIDDEN_AUTOMATIC_CLAIMS: tuple[str, ...] = (
    "confirmed physical mode",
    "confirmed energy transfer",
    "proven nonlinear behavior",
    "causality demonstrated",
    "physical split",
    "physical merge",
    "modo fisico confirmado",
    "transferencia de energia confirmada",
    "comportamento nao linear comprovado",
    "causalidade demonstrada",
    "split fisico",
    "merge fisico",
)


@dataclass(frozen=True, slots=True)
class ScientificReportSettings:
    """Explicit deterministic settings for a reproducible scientific report."""

    generate_markdown: bool = True
    generate_latex: bool = True
    compile_pdf: bool = False
    generate_manifest: bool = True
    generate_bibliography_file: bool = False
    generate_makefile: bool = False
    generate_latexmkrc: bool = False

    include_title_page: bool = True
    include_abstract: bool = True
    include_executive_summary: bool = True
    include_experiment_description: bool = True
    include_acquisition_metadata: bool = True
    include_methodology: bool = True
    include_recording_quality: bool = True
    include_temporal_results: bool = True
    include_spectral_results: bool = True
    include_tracking_results: bool = True
    include_candidate_results: bool = True
    include_associations: bool = True
    include_candidate_chains: bool = True
    include_modal_hypotheses: bool = True
    include_modal_parameters: bool = True
    include_q_factors: bool = True
    include_energy_exchange: bool = True
    include_synthetic_validation: bool = True
    include_limitations: bool = True
    include_provenance: bool = True
    include_diagnostics: bool = True
    include_appendices: bool = True

    include_rejected_results: bool = True
    include_inconclusive_results: bool = True
    include_invalid_results: bool = True
    include_full_diagnostics: bool = False
    include_settings_dump: bool = True
    include_file_fingerprints: bool = True
    include_checksums: bool = True
    include_software_environment: bool = True

    include_figures: bool = True
    figure_format_preference: str = "png"
    maximum_figures_per_section: int | None = 8
    figure_width_fraction: float = 0.85
    figure_placement_policy: str = "here"
    allow_figure_downscaling: bool = True
    include_figure_provenance: bool = True

    include_tables: bool = True
    table_format_preference: str = "markdown"
    maximum_rows_inline: int = 12
    large_table_policy: str = "appendix"
    include_table_notes: bool = True
    include_uncertainties: bool = True
    include_status_columns: bool = True
    include_reason_columns: bool = True

    language: str = "pt-BR"
    title: str | None = None
    subtitle: str | None = None
    authors: tuple[str, ...] = ()
    affiliations: tuple[str, ...] = ()
    corresponding_author: str | None = None
    report_date: str | None = None
    abstract_text: str | None = None
    keywords: tuple[str, ...] = ()
    acknowledgments: str | None = None
    funding_statement: str | None = None
    conflict_of_interest_statement: str | None = None
    user_context_text: str | None = None
    user_acquisition_text: str | None = None
    user_discussion_text: str | None = None
    user_conclusions_text: str | None = None

    document_class: str = "article"
    document_class_options: tuple[str, ...] = ("11pt", "a4paper")
    latex_engine: str = "auto"
    bibliography_backend: str = "none"
    latex_compilation_runs: int = 1
    shell_escape: bool = False
    halt_on_error: bool = True
    interaction_mode: str = "nonstopmode"
    keep_auxiliary_files: bool = True
    pdf_compilation_timeout_s: float = 60.0

    output_directory: str | Path = Path("belllab-report")
    file_prefix: str = "belllab_report"
    overwrite_policy: ExportOverwritePolicy = ExportOverwritePolicy.ERROR
    atomic_write: bool = True
    copy_artifacts: bool = False
    use_relative_paths: bool = True
    validate_checksums_before_render: bool = True

    numeric_formatting: ExportNumericFormatting = ExportNumericFormatting()
    missing_value_representation: ExportMissingValuePolicy = ExportMissingValuePolicy.DASH
    uncertainty_format: str = "plus_minus"
    decimal_separator: str = "."

    def __post_init__(self) -> None:
        for name in (
            "generate_markdown",
            "generate_latex",
            "compile_pdf",
            "generate_manifest",
            "generate_bibliography_file",
            "generate_makefile",
            "generate_latexmkrc",
            "include_title_page",
            "include_abstract",
            "include_executive_summary",
            "include_experiment_description",
            "include_acquisition_metadata",
            "include_methodology",
            "include_recording_quality",
            "include_temporal_results",
            "include_spectral_results",
            "include_tracking_results",
            "include_candidate_results",
            "include_associations",
            "include_candidate_chains",
            "include_modal_hypotheses",
            "include_modal_parameters",
            "include_q_factors",
            "include_energy_exchange",
            "include_synthetic_validation",
            "include_limitations",
            "include_provenance",
            "include_diagnostics",
            "include_appendices",
            "include_rejected_results",
            "include_inconclusive_results",
            "include_invalid_results",
            "include_full_diagnostics",
            "include_settings_dump",
            "include_file_fingerprints",
            "include_checksums",
            "include_software_environment",
            "include_figures",
            "allow_figure_downscaling",
            "include_figure_provenance",
            "include_tables",
            "include_table_notes",
            "include_uncertainties",
            "include_status_columns",
            "include_reason_columns",
            "shell_escape",
            "halt_on_error",
            "keep_auxiliary_files",
            "atomic_write",
            "copy_artifacts",
            "use_relative_paths",
            "validate_checksums_before_render",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")
        if not any((
            self.generate_markdown,
            self.generate_latex,
            self.compile_pdf,
            self.generate_manifest,
            self.generate_bibliography_file,
            self.generate_makefile,
            self.generate_latexmkrc,
        )):
            raise ValueError("at least one report artifact must be requested.")
        if self.figure_format_preference not in {"png", "svg", "pdf"}:
            raise ValueError("figure_format_preference is not supported.")
        if self.maximum_figures_per_section is not None and self.maximum_figures_per_section <= 0:
            raise ValueError("maximum_figures_per_section must be positive or None.")
        if not math.isfinite(self.figure_width_fraction) or not 0.0 < self.figure_width_fraction <= 1.0:
            raise ValueError("figure_width_fraction must be in (0, 1].")
        if self.figure_placement_policy not in {"here", "float"}:
            raise ValueError("figure_placement_policy is not recognized.")
        if self.table_format_preference not in {"markdown", "latex"}:
            raise ValueError("table_format_preference is not supported.")
        if self.maximum_rows_inline <= 0:
            raise ValueError("maximum_rows_inline must be positive.")
        if self.large_table_policy not in {"inline", "appendix", "skip_with_note"}:
            raise ValueError("large_table_policy is not recognized.")
        if self.language not in {"pt-BR", "en"}:
            raise ValueError("language must be 'pt-BR' or 'en'.")
        if self.latex_engine not in {"auto", "latexmk", "tectonic", "pdflatex", "lualatex", "xelatex"}:
            raise ValueError("latex_engine is not supported.")
        if self.bibliography_backend not in {"none", "bibtex", "biber"}:
            raise ValueError("bibliography_backend is not supported.")
        if self.latex_compilation_runs <= 0:
            raise ValueError("latex_compilation_runs must be positive.")
        if self.interaction_mode not in {"nonstopmode", "batchmode", "scrollmode", "errorstopmode"}:
            raise ValueError("interaction_mode is not supported.")
        if not math.isfinite(self.pdf_compilation_timeout_s) or self.pdf_compilation_timeout_s <= 0:
            raise ValueError("pdf_compilation_timeout_s must be finite and positive.")
        output = Path(self.output_directory)
        if not str(output):
            raise ValueError("output_directory must not be empty.")
        object.__setattr__(self, "output_directory", output)
        _text(self.file_prefix, "file_prefix")
        object.__setattr__(self, "overwrite_policy", _coerce_enum(self.overwrite_policy, ExportOverwritePolicy))
        object.__setattr__(
            self,
            "missing_value_representation",
            _coerce_enum(self.missing_value_representation, ExportMissingValuePolicy),
        )
        if self.uncertainty_format not in {"plus_minus", "interval", "separate"}:
            raise ValueError("uncertainty_format is not recognized.")
        if self.decimal_separator not in {".", ","}:
            raise ValueError("decimal_separator must be '.' or ','.")
        for name in (
            "title",
            "subtitle",
            "corresponding_author",
            "report_date",
            "abstract_text",
            "acknowledgments",
            "funding_statement",
            "conflict_of_interest_statement",
            "user_context_text",
            "user_acquisition_text",
            "user_discussion_text",
            "user_conclusions_text",
        ):
            _optional_text(getattr(self, name), name)
        for name in ("authors", "affiliations", "keywords", "document_class_options"):
            object.__setattr__(self, name, _text_tuple(getattr(self, name), name))
        _text(self.document_class, "document_class")


@dataclass(frozen=True, slots=True)
class ScientificReportContentBlock:
    """Typed content block in a normalized scientific report section."""

    block_id: str
    block_type: ScientificReportContentBlockType
    content: object
    source_ids: tuple[str, ...] = ()
    status: ScientificReportStatus = ScientificReportStatus.CREATED
    style: str | None = None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.block_id, "block_id")
        object.__setattr__(self, "block_type", _coerce_enum(self.block_type, ScientificReportContentBlockType))
        object.__setattr__(self, "status", _coerce_enum(self.status, ScientificReportStatus))
        object.__setattr__(self, "source_ids", _unique_texts(self.source_ids))
        object.__setattr__(self, "diagnostics", _unique_texts(self.diagnostics))
        _optional_text(self.style, "style")


@dataclass(frozen=True, slots=True)
class ScientificReportTable:
    """Normalized table included in the report document."""

    table_id: str
    title: str
    columns: tuple[str, ...]
    rows: tuple[Mapping[str, object], ...]
    source_ids: tuple[str, ...] = ()
    status: ScientificReportStatus = ScientificReportStatus.CREATED
    notes: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.table_id, "table_id")
        _text(self.title, "title")
        object.__setattr__(self, "columns", _text_tuple(self.columns, "columns", allow_empty=False))
        object.__setattr__(
            self,
            "rows",
            tuple(MappingProxyType(dict(row)) for row in self.rows),
        )
        object.__setattr__(self, "source_ids", _unique_texts(self.source_ids))
        object.__setattr__(self, "status", _coerce_enum(self.status, ScientificReportStatus))
        object.__setattr__(self, "notes", _unique_texts(self.notes))
        object.__setattr__(self, "diagnostics", _unique_texts(self.diagnostics))


@dataclass(frozen=True, slots=True)
class ScientificReportFigureReference:
    """Reference to an already generated figure artifact."""

    figure_id: str
    figure_type: str
    title: str
    caption: str
    relative_path: str | None
    source_ids: tuple[str, ...]
    checksum: str | None
    status: ScientificReportStatus
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.figure_id, "figure_id")
        _text(self.figure_type, "figure_type")
        _text(self.title, "title")
        _text(self.caption, "caption")
        _optional_text(self.relative_path, "relative_path")
        object.__setattr__(self, "source_ids", _unique_texts(self.source_ids))
        object.__setattr__(self, "status", _coerce_enum(self.status, ScientificReportStatus))
        object.__setattr__(self, "diagnostics", _unique_texts(self.diagnostics))


@dataclass(frozen=True, slots=True)
class ScientificReportSection:
    """A section in the normalized report document."""

    section_id: str
    title: str
    level: int
    order: int
    content_blocks: tuple[ScientificReportContentBlock, ...] = ()
    table_ids: tuple[str, ...] = ()
    figure_ids: tuple[str, ...] = ()
    subsections: tuple["ScientificReportSection", ...] = ()
    source_ids: tuple[str, ...] = ()
    status: ScientificReportStatus = ScientificReportStatus.CREATED
    reasons: tuple[ScientificReportReason, ...] = ()
    provenance: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.section_id, "section_id")
        _text(self.title, "title")
        if self.level <= 0 or self.order < 0:
            raise ValueError("section level must be positive and order non-negative.")
        object.__setattr__(self, "content_blocks", tuple(self.content_blocks))
        object.__setattr__(self, "table_ids", _unique_texts(self.table_ids))
        object.__setattr__(self, "figure_ids", _unique_texts(self.figure_ids))
        object.__setattr__(self, "subsections", tuple(self.subsections))
        object.__setattr__(self, "source_ids", _unique_texts(self.source_ids))
        object.__setattr__(self, "status", _coerce_enum(self.status, ScientificReportStatus))
        object.__setattr__(self, "reasons", _reason_tuple(self.reasons))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))
        object.__setattr__(self, "diagnostics", _unique_texts(self.diagnostics))


@dataclass(frozen=True, slots=True)
class ScientificReportDocument:
    """Format-independent normalized scientific report document."""

    report_id: str
    analysis_id: str
    experiment_id: str | None
    title: str | None
    subtitle: str | None
    authors: tuple[str, ...]
    affiliations: tuple[str, ...]
    language: str
    sections: tuple[ScientificReportSection, ...]
    figures: tuple[ScientificReportFigureReference, ...]
    tables: tuple[ScientificReportTable, ...]
    appendices: tuple[ScientificReportSection, ...]
    references: Mapping[str, str]
    provenance: Mapping[str, object]
    limitations: tuple[str, ...]
    diagnostics: tuple[str, ...]
    settings_fingerprint: str
    valid: bool

    def __post_init__(self) -> None:
        _text(self.report_id, "report_id")
        _text(self.analysis_id, "analysis_id")
        _optional_text(self.experiment_id, "experiment_id")
        _optional_text(self.title, "title")
        _optional_text(self.subtitle, "subtitle")
        object.__setattr__(self, "authors", _text_tuple(self.authors, "authors", allow_empty=True))
        object.__setattr__(self, "affiliations", _text_tuple(self.affiliations, "affiliations", allow_empty=True))
        if self.language not in {"pt-BR", "en"}:
            raise ValueError("language is not recognized.")
        for name in ("sections", "figures", "tables", "appendices"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "references", MappingProxyType(dict(self.references)))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))
        object.__setattr__(self, "limitations", _unique_texts(self.limitations))
        object.__setattr__(self, "diagnostics", _unique_texts(self.diagnostics))
        _text(self.settings_fingerprint, "settings_fingerprint")


@dataclass(frozen=True, slots=True)
class ScientificReportCompilationResult:
    """Structured result of optional LaTeX to PDF compilation."""

    requested: bool
    compiler: str | None
    command: tuple[str, ...]
    return_code: int | None
    pdf_path: str | None
    log_path: str | None
    auxiliary_files: tuple[str, ...]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    status: ScientificReportStatus
    valid: bool
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _optional_text(self.compiler, "compiler")
        object.__setattr__(self, "command", _text_tuple(self.command, "command", allow_empty=True))
        if self.return_code is not None and self.return_code < 0:
            raise ValueError("return_code must not be negative.")
        _optional_text(self.pdf_path, "pdf_path")
        _optional_text(self.log_path, "log_path")
        for name in ("auxiliary_files", "warnings", "errors", "diagnostics"):
            object.__setattr__(self, name, _unique_texts(getattr(self, name)))
        object.__setattr__(self, "status", _coerce_enum(self.status, ScientificReportStatus))


@dataclass(frozen=True, slots=True)
class ScientificReportArtifact:
    """Generated report artifact with a content checksum."""

    artifact_id: str
    artifact_type: str
    format: str
    path: str | None
    relative_path: str | None
    checksum: str | None
    size_bytes: int | None
    status: ScientificReportStatus
    reasons: tuple[ScientificReportReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.artifact_id, "artifact_id")
        _text(self.artifact_type, "artifact_type")
        _text(self.format, "format")
        _optional_text(self.path, "path")
        _optional_text(self.relative_path, "relative_path")
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ValueError("size_bytes must not be negative.")
        object.__setattr__(self, "status", _coerce_enum(self.status, ScientificReportStatus))
        object.__setattr__(self, "reasons", _reason_tuple(self.reasons))
        object.__setattr__(self, "diagnostics", _unique_texts(self.diagnostics))


@dataclass(frozen=True, slots=True)
class ScientificReportManifest:
    """Manifest written last for a reproducible scientific report."""

    report_schema_version: str
    report_id: str
    analysis_id: str
    experiment_id: str | None
    belllab_version: str
    report_settings_fingerprint: str
    source_analysis_fingerprint: str
    source_export_id: str | None
    source_figure_collection_id: str | None
    sections: tuple[Mapping[str, object], ...]
    tables: tuple[Mapping[str, object], ...]
    figures: tuple[Mapping[str, object], ...]
    appendices: tuple[Mapping[str, object], ...]
    artifacts: tuple[ScientificReportArtifact, ...]
    checksums: Mapping[str, str | None]
    compilation: Mapping[str, object] | None
    source_statuses: tuple[str, ...]
    limitations: tuple[str, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("report_schema_version", "report_id", "analysis_id", "belllab_version", "report_settings_fingerprint", "source_analysis_fingerprint"):
            _text(getattr(self, name), name)
        _optional_text(self.experiment_id, "experiment_id")
        _optional_text(self.source_export_id, "source_export_id")
        _optional_text(self.source_figure_collection_id, "source_figure_collection_id")
        for name in ("sections", "tables", "figures", "appendices"):
            object.__setattr__(self, name, tuple(MappingProxyType(dict(item)) for item in getattr(self, name)))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "checksums", MappingProxyType(dict(self.checksums)))
        if self.compilation is not None:
            object.__setattr__(self, "compilation", MappingProxyType(dict(self.compilation)))
        for name in ("source_statuses", "limitations", "diagnostics"):
            object.__setattr__(self, name, _unique_texts(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class ScientificReportValidation:
    """Validation summary for a generated scientific report."""

    expected_artifacts: tuple[str, ...]
    existing_artifacts: tuple[str, ...]
    missing_artifacts: tuple[str, ...]
    checksum_matches: Mapping[str, bool]
    cross_references_valid: bool
    manifest_consistent: bool
    conservative_narrative_valid: bool
    valid: bool
    reasons: tuple[ScientificReportReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("expected_artifacts", "existing_artifacts", "missing_artifacts", "diagnostics"):
            object.__setattr__(self, name, _unique_texts(getattr(self, name)))
        object.__setattr__(self, "checksum_matches", MappingProxyType(dict(self.checksum_matches)))
        object.__setattr__(self, "reasons", _reason_tuple(self.reasons))


@dataclass(frozen=True, slots=True)
class ScientificReportResult:
    """Top-level result for reproducible scientific report generation."""

    report_id: str
    analysis_id: str
    experiment_id: str | None
    document: ScientificReportDocument | None
    status: ScientificReportStatus
    artifacts: tuple[ScientificReportArtifact, ...]
    manifest: ScientificReportManifest | None
    compilation_result: ScientificReportCompilationResult | None
    completed_sections: tuple[ScientificReportSection, ...]
    partial_sections: tuple[ScientificReportSection, ...]
    skipped_sections: tuple[str, ...]
    failed_sections: tuple[ScientificReportSection, ...]
    completed_artifacts: tuple[ScientificReportArtifact, ...]
    failed_artifacts: tuple[ScientificReportArtifact, ...]
    valid: bool
    requires_review: bool
    failure_reason: str | None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.report_id, "report_id")
        _text(self.analysis_id, "analysis_id")
        _optional_text(self.experiment_id, "experiment_id")
        object.__setattr__(self, "status", _coerce_enum(self.status, ScientificReportStatus))
        for name in (
            "artifacts",
            "completed_sections",
            "partial_sections",
            "failed_sections",
            "completed_artifacts",
            "failed_artifacts",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "skipped_sections", _unique_texts(self.skipped_sections))
        expected_valid = self.status in {
            ScientificReportStatus.CREATED,
            ScientificReportStatus.CREATED_WITH_RESERVATIONS,
        }
        if self.valid != expected_valid:
            raise ValueError("valid must mirror created report statuses.")
        _optional_text(self.failure_reason, "failure_reason")
        object.__setattr__(self, "diagnostics", _unique_texts(self.diagnostics))


def build_scientific_report_document(
    analysis: ExperimentAnalysisResult,
    export_result: ExperimentExportResult | None = None,
    figure_collection: ScientificFigureCollection | None = None,
    settings: ScientificReportSettings | None = None,
) -> ScientificReportDocument:
    """Build the normalized report document without rendering or recalculation."""

    cfg = settings or ScientificReportSettings()
    if not isinstance(analysis, ExperimentAnalysisResult):
        raise TypeError("analysis must be an ExperimentAnalysisResult.")
    _validate_source_compatibility(analysis, export_result, figure_collection, cfg)
    normalized = normalize_experiment_for_export(analysis, _normalization_settings(cfg))
    tables = _build_report_tables(normalized, cfg)
    figures = _build_report_figures(figure_collection, cfg)
    limitations = _derive_limitations(analysis, normalized, export_result, figure_collection, cfg)
    sections = build_scientific_report_sections(analysis, normalized, tables, figures, settings=cfg)
    appendices = _build_appendices(normalized, tables, cfg)
    diagnostics = (
        "report_document_organizes_existing_results_only",
        "no_scientific_analysis_recalculated",
        "no_figures_regenerated",
        "no_physical_conclusions_generated",
    )
    if export_result is None:
        diagnostics += ("export_result_not_supplied_tables_built_from_normalized_analysis",)
    if figure_collection is None:
        diagnostics += ("figure_collection_not_supplied_figures_marked_missing",)
    provenance = _document_provenance(analysis, export_result, figure_collection, cfg, normalized)
    report_id = _stable_id(
        "scientific-report",
        analysis.analysis_id,
        analysis.experiment.experiment_id,
        scientific_report_settings_fingerprint(cfg),
        getattr(export_result, "export_id", None),
        getattr(figure_collection, "collection_id", None),
        tuple(section.section_id for section in sections),
        tuple(table.table_id for table in tables),
        tuple((figure.figure_id, figure.checksum) for figure in figures),
    )
    references = _build_references(sections, tables, figures, appendices)
    return ScientificReportDocument(
        report_id=report_id,
        analysis_id=analysis.analysis_id,
        experiment_id=analysis.experiment.experiment_id,
        title=cfg.title,
        subtitle=cfg.subtitle,
        authors=cfg.authors,
        affiliations=cfg.affiliations,
        language=cfg.language,
        sections=sections,
        figures=figures,
        tables=tables,
        appendices=appendices,
        references=references,
        provenance=provenance,
        limitations=limitations,
        diagnostics=diagnostics,
        settings_fingerprint=scientific_report_settings_fingerprint(cfg),
        valid=True,
    )


def build_scientific_report_sections(
    analysis: ExperimentAnalysisResult,
    normalized: NormalizedExperimentExport,
    tables: Sequence[ScientificReportTable],
    figures: Sequence[ScientificReportFigureReference],
    settings: ScientificReportSettings | None = None,
) -> tuple[ScientificReportSection, ...]:
    """Build deterministic report sections from normalized existing results."""

    cfg = settings or ScientificReportSettings()
    summary = normalized.summary
    table_by_id = {table.table_id: table for table in tables}
    figures_by_type: dict[str, list[str]] = {}
    for figure in figures:
        figures_by_type.setdefault(figure.figure_type, []).append(figure.figure_id)

    sections: list[ScientificReportSection] = []
    add = sections.append
    order = 0
    if cfg.include_title_page:
        add(_section(
            "title_page",
            "Capa" if cfg.language == "pt-BR" else "Title Page",
            order,
            (
                _paragraph("title-page-main", _report_title(analysis, cfg), analysis.analysis_id),
                _paragraph("title-page-ids", f"experiment_id: {analysis.experiment.experiment_id}; analysis_id: {analysis.analysis_id}", analysis.analysis_id),
                _paragraph("title-page-date", f"report_date: {_missing_or_text(cfg.report_date, cfg)}", analysis.analysis_id),
            ),
            status=ScientificReportStatus.CREATED,
        ))
        order += 1
    if cfg.include_abstract:
        abstract = cfg.abstract_text or _automatic_abstract(summary, normalized, cfg)
        add(_section(
            "abstract",
            "Resumo" if cfg.language == "pt-BR" else "Abstract",
            order,
            (
                _notice("abstract-caution", _scientific_caution_text(cfg), analysis.analysis_id),
                _paragraph("abstract-text", abstract, analysis.analysis_id),
            ),
        ))
        order += 1
    if cfg.include_experiment_description:
        blocks = [
            _paragraph(
                "experiment-description-facts",
                f"Foram analisadas {summary.get('recording_count')} gravacoes em {summary.get('condition_count')} condicoes dinamicas: {_join(summary.get('dynamic_labels', ()), cfg)}.",
                analysis.analysis_id,
            )
        ]
        if cfg.user_context_text is not None:
            blocks.append(_paragraph("experiment-description-user-text", f"user_provided_text: {cfg.user_context_text}", analysis.analysis_id))
        add(_section("experiment_description", "Identificacao do experimento", order, tuple(blocks), table_ids=("table-experiment-summary",)))
        order += 1
    if cfg.include_acquisition_metadata:
        blocks = [_paragraph("acquisition-metadata-note", "Metadados ausentes permanecem ausentes; nenhum operador, local, financiamento ou conflito e inventado.", analysis.analysis_id)]
        if cfg.user_acquisition_text is not None:
            blocks.append(_paragraph("acquisition-user-text", f"user_provided_text: {cfg.user_acquisition_text}", analysis.analysis_id))
        add(_section("acquisition_metadata", "Metadados de aquisicao", order, tuple(blocks), table_ids=("table-recordings", "table-conditions")))
        order += 1
    if cfg.include_methodology:
        add(_section(
            "methodology",
            "Metodologia computacional",
            order,
            (
                _paragraph("methodology-pipeline", "O relatorio organiza resultados existentes do BellLab e nao recalcula analise cientifica.", analysis.analysis_id),
                _equation("methodology-decay-equation", "A(t) = A_0 \\exp(-t/\\tau)", analysis.analysis_id),
                _equation("methodology-q-decay-equation", "Q_{decay} = \\pi f \\tau", analysis.analysis_id),
                _equation("methodology-q-bandwidth-equation", "Q_{bandwidth} = f_0 / \\Delta f", analysis.analysis_id),
                _notice("methodology-caution", "As equacoes sao convencoes operacionais condicionadas aos metodos executados; elas nao provam identidade modal fisica.", analysis.analysis_id),
            ),
            table_ids=("table-pipeline-stages",),
        ))
        order += 1
    if cfg.include_recording_quality:
        add(_section(
            "recording_quality",
            "Qualidade das gravacoes",
            order,
            (_paragraph("recording-quality-facts", "Duracao, sample rate, canais, clipping, offsets, falhas e selecao de repeticao sao preservados nas tabelas de gravacoes e condicoes.", analysis.analysis_id),),
            table_ids=("table-recordings", "table-conditions"),
            figure_ids=_limit_figures(figures_by_type.get("waveform", ()), cfg),
        ))
        order += 1
    if cfg.include_temporal_results:
        add(_section(
            "temporal_results",
            "Analise temporal",
            order,
            (_paragraph("temporal-results-facts", "Waveform, envelope, decaimento, tau e tempos em dB sao relatados somente quando ja existem nos resultados de origem.", analysis.analysis_id),),
            table_ids=("table-modal-parameters",),
            figure_ids=_limit_figures(tuple(figures_by_type.get("temporal_envelope", ())) + tuple(figures_by_type.get("decay_estimate", ())), cfg),
        ))
        order += 1
    if cfg.include_spectral_results:
        add(_section(
            "spectral_results",
            "Analise espectral",
            order,
            (_paragraph("spectral-results-facts", "Espectros, picos, espectrogramas, largura e resolucao sao apresentados como resultados operacionais ja calculados.", analysis.analysis_id),),
            table_ids=("table-modal-q-factors",),
            figure_ids=_limit_figures(tuple(figures_by_type.get("global_spectrum", ())) + tuple(figures_by_type.get("spectral_peaks", ())) + tuple(figures_by_type.get("spectrogram", ())), cfg),
        ))
        order += 1
    if cfg.include_tracking_results:
        add(_section(
            "tracking_results",
            "Tracking e tracks",
            order,
            (_paragraph("tracking-results-facts", "Tracks, gaps, fragmentacao e possiveis swaps sao descritores operacionais; gaps nao sao ligados no texto.", analysis.analysis_id),),
            table_ids=("table-candidates",),
            figure_ids=_limit_figures(figures_by_type.get("frequency_tracks", ()), cfg),
        ))
        order += 1
    if cfg.include_candidate_results:
        add(_section(
            "candidate_results",
            "Candidatos modais",
            order,
            (_notice("candidate-caution", "Candidatos modais nao sao modos fisicos comprovados.", analysis.analysis_id),),
            table_ids=("table-candidates",),
            figure_ids=_limit_figures(figures_by_type.get("modal_candidates", ()), cfg),
        ))
        order += 1
    if cfg.include_associations:
        add(_section(
            "associations",
            "Associacoes",
            order,
            (_notice("association-caution", "Associacao entre condicoes adjacentes nao prova identidade fisica e nao cria associacao nao adjacente.", analysis.analysis_id),),
            table_ids=("table-within-condition-associations", "table-cross-condition-matches"),
            figure_ids=_limit_figures(tuple(figures_by_type.get("within_condition_associations", ())) + tuple(figures_by_type.get("cross_condition_associations", ())), cfg),
        ))
        order += 1
    if cfg.include_candidate_chains:
        add(_section(
            "candidate_chains",
            "Cadeias de candidatos",
            order,
            (_paragraph("chains-facts", "Cadeias preservam condicoes cobertas, gaps, custos, ambiguidades e contexto operacional de split/merge sem resolver identidade fisica.", analysis.analysis_id),),
            table_ids=("table-candidate-chains", "table-candidate-chain-nodes"),
            figure_ids=_limit_figures(figures_by_type.get("candidate_chains", ()), cfg),
        ))
        order += 1
    if cfg.include_modal_hypotheses:
        add(_section(
            "modal_hypotheses",
            "Hipoteses modais",
            order,
            (_notice("hypothesis-caution", "Uma hipotese modal aceita permanece hipotese operacional, nao modo fisico comprovado.", analysis.analysis_id),),
            table_ids=("table-modal-hypotheses",),
            figure_ids=_limit_figures(figures_by_type.get("modal_hypotheses", ()), cfg),
        ))
        order += 1
    if cfg.include_modal_parameters:
        add(_section(
            "modal_parameters",
            "Parametros modais operacionais",
            order,
            (_notice("parameters-caution", "Frequencia representativa, trajetoria, drift, tau e taxa de decaimento sao estimativas operacionais; slope positivo ou negativo nao e prova de nao linearidade.", analysis.analysis_id),),
            table_ids=("table-modal-parameters",),
            figure_ids=_limit_figures(tuple(figures_by_type.get("modal_frequency_trajectories", ())) + tuple(figures_by_type.get("modal_parameters", ())), cfg),
        ))
        order += 1
    if cfg.include_q_factors:
        add(_section(
            "q_factors",
            "Fator Q e largura de banda",
            order,
            (_paragraph("q-facts", "Q por decaimento, Q por largura de banda, Q representativo, bandwidth, resolucao e discordancias entre metodos sao preservados quando disponiveis.", analysis.analysis_id),),
            table_ids=("table-modal-q-factors",),
            figure_ids=_limit_figures(tuple(figures_by_type.get("modal_q_factors", ())) + tuple(figures_by_type.get("modal_bandwidth", ())), cfg),
        ))
        order += 1
    if cfg.include_energy_exchange:
        add(_section(
            "energy_exchange",
            "Evidencia operacional de possivel redistribuicao entre componentes",
            order,
            (_notice("energy-caution", "Anticorrelacao temporal, lag ou proxy de energia estavel nao comprovam transferencia fisica de energia nem causalidade.", analysis.analysis_id),),
            table_ids=("table-energy-exchange-pairs",),
            figure_ids=_limit_figures(tuple(figures_by_type.get("modal_energy_exchange_evidence", ())) + tuple(figures_by_type.get("modal_energy_exchange_correlation", ())), cfg),
        ))
        order += 1
    if cfg.include_synthetic_validation:
        add(_section(
            "synthetic_validation",
            "Validacao sintetica",
            order,
            (_notice("synthetic-caution", "O desempenho em sinais sinteticos nao garante desempenho equivalente em gravacoes reais.", analysis.analysis_id),),
            figure_ids=_limit_figures(tuple(figures_by_type.get("synthetic_validation_result", ())) + tuple(figures_by_type.get("synthetic_validation_campaign", ())), cfg),
            reasons=(ScientificReportReason.MISSING_OPTIONAL_TABLE,),
            status=ScientificReportStatus.CREATED_WITH_RESERVATIONS,
        ))
        order += 1
    if cfg.include_executive_summary:
        add(_section(
            "factual_synthesis",
            "Sintese factual dos resultados",
            order,
            tuple(_factual_synthesis_blocks(summary, normalized, analysis.analysis_id)),
        ))
        order += 1
    if cfg.include_limitations:
        add(_section(
            "limitations",
            "Limitacoes",
            order,
            tuple(_limitation("limitation", item, analysis.analysis_id) for item in _derive_limitations(analysis, normalized, None, None, cfg)),
        ))
        order += 1
    if cfg.include_provenance:
        add(_section(
            "provenance_reproducibility",
            "Proveniencia e reprodutibilidade",
            order,
            (
                _paragraph("provenance-ids", f"BellLab version: {normalized.belllab_version}; settings fingerprint: {normalized.provenance.get('settings_fingerprint')}", analysis.analysis_id),
                _paragraph("provenance-policy", "Caminhos absolutos nao sao necessarios no conteudo do relatorio; fingerprints e checksums sao usados para auditoria.", analysis.analysis_id),
            ),
            table_ids=("table-pipeline-stages", "table-diagnostics"),
        ))
        order += 1
    return tuple(_mark_section_availability(section, table_by_id, figures, cfg) for section in sections)


def render_scientific_report_markdown(
    document: ScientificReportDocument,
    settings: ScientificReportSettings | None = None,
) -> str:
    """Render a normalized report document to deterministic Markdown."""

    cfg = settings or ScientificReportSettings()
    _validate_document(document)
    table_map = {table.table_id: table for table in document.tables}
    figure_map = {figure.figure_id: figure for figure in document.figures}
    lines: list[str] = []
    title = _safe_md(document.title or _report_fallback_title(document))
    lines.append(f"# {title}")
    if document.subtitle:
        lines.append("")
        lines.append(_safe_md(document.subtitle))
    lines.append("")
    lines.append(_scientific_caution_text(cfg))
    lines.append("")
    for section in document.sections:
        _render_markdown_section(section, table_map, figure_map, lines, cfg)
    if document.appendices:
        lines.append("")
        lines.append("# Apendices" if cfg.language == "pt-BR" else "# Appendices")
        for section in document.appendices:
            _render_markdown_section(section, table_map, figure_map, lines, cfg)
    lines.append("")
    lines.append("<!-- belllab_report_schema_version: 1.0 -->")
    content = "\n".join(lines).rstrip() + "\n"
    _assert_conservative_narrative(content)
    return content


def render_scientific_report_latex(
    document: ScientificReportDocument,
    settings: ScientificReportSettings | None = None,
) -> str:
    """Render a normalized report document to deterministic LaTeX."""

    cfg = settings or ScientificReportSettings()
    _validate_document(document)
    table_map = {table.table_id: table for table in document.tables}
    figure_map = {figure.figure_id: figure for figure in document.figures}
    lines = [
        f"\\documentclass[{','.join(cfg.document_class_options)}]{{{_latex_escape(cfg.document_class)}}}",
        "\\usepackage[utf8]{inputenc}",
        "\\usepackage[T1]{fontenc}",
        "\\usepackage{booktabs}",
        "\\usepackage{longtable}",
        "\\usepackage{graphicx}",
        "\\usepackage{amsmath}",
        "\\usepackage{hyperref}",
        "\\usepackage{geometry}",
        "\\geometry{margin=2.5cm}",
        f"\\title{{{_latex_escape(document.title or _report_fallback_title(document))}}}",
        f"\\author{{{_latex_escape(', '.join(document.authors)) if document.authors else ''}}}",
        f"\\date{{{_latex_escape(_missing_or_text(cfg.report_date, cfg))}}}",
        "\\begin{document}",
        "\\maketitle",
        _latex_paragraph(_scientific_caution_text(cfg)),
    ]
    for section in document.sections:
        _render_latex_section(section, table_map, figure_map, lines, cfg)
    if document.appendices:
        lines.append("\\appendix")
        for section in document.appendices:
            _render_latex_section(section, table_map, figure_map, lines, cfg)
    lines.append("\\end{document}")
    content = "\n".join(lines) + "\n"
    _assert_conservative_narrative(content)
    return content


def compile_scientific_report_pdf(
    latex_path: str | Path | None,
    settings: ScientificReportSettings | None = None,
) -> ScientificReportCompilationResult:
    """Optionally compile a generated LaTeX report to PDF."""

    cfg = settings or ScientificReportSettings()
    if not cfg.compile_pdf:
        return ScientificReportCompilationResult(
            requested=False,
            compiler=None,
            command=(),
            return_code=None,
            pdf_path=None,
            log_path=None,
            auxiliary_files=(),
            warnings=(),
            errors=(),
            status=ScientificReportStatus.CREATED,
            valid=True,
            diagnostics=("pdf_compilation_not_requested",),
        )
    if latex_path is None:
        return ScientificReportCompilationResult(
            requested=True,
            compiler=None,
            command=(),
            return_code=None,
            pdf_path=None,
            log_path=None,
            auxiliary_files=(),
            warnings=(),
            errors=("latex_source_missing",),
            status=ScientificReportStatus.COMPILATION_FAILED,
            valid=False,
            diagnostics=(ScientificReportReason.MISSING_REQUIRED_SOURCE.value,),
        )
    path = Path(latex_path)
    if not path.is_file():
        return ScientificReportCompilationResult(
            requested=True,
            compiler=None,
            command=(),
            return_code=None,
            pdf_path=None,
            log_path=None,
            auxiliary_files=(),
            warnings=(),
            errors=(f"latex source not found: {path}",),
            status=ScientificReportStatus.COMPILATION_FAILED,
            valid=False,
            diagnostics=(ScientificReportReason.MISSING_REQUIRED_SOURCE.value,),
        )
    compiler = _select_latex_compiler(cfg)
    if compiler is None:
        return ScientificReportCompilationResult(
            requested=True,
            compiler=None,
            command=(),
            return_code=None,
            pdf_path=None,
            log_path=None,
            auxiliary_files=(),
            warnings=(ScientificReportReason.LATEX_COMPILER_UNAVAILABLE.value,),
            errors=(),
            status=ScientificReportStatus.PARTIAL,
            valid=False,
            diagnostics=("latex_compiler_unavailable_pdf_optional_sources_preserved",),
        )
    command = _latex_command(compiler, path.name, cfg)
    stdout_log = path.with_suffix(".compile.log")
    return_code = 0
    output_text = ""
    try:
        runs = 1 if compiler in {"latexmk", "tectonic"} else cfg.latex_compilation_runs
        for _ in range(runs):
            completed = subprocess.run(
                command,
                cwd=path.parent,
                text=True,
                capture_output=True,
                timeout=cfg.pdf_compilation_timeout_s,
                check=False,
            )
            return_code = completed.returncode
            output_text += completed.stdout + "\n" + completed.stderr + "\n"
            if return_code != 0:
                break
        stdout_log.write_text(output_text, encoding="utf-8")
    except Exception as exc:
        stdout_log.write_text(f"{exc.__class__.__name__}: {exc}", encoding="utf-8")
        return ScientificReportCompilationResult(
            requested=True,
            compiler=compiler,
            command=tuple(command),
            return_code=None,
            pdf_path=None,
            log_path=str(stdout_log),
            auxiliary_files=(),
            warnings=(),
            errors=(f"{exc.__class__.__name__}: {exc}",),
            status=ScientificReportStatus.COMPILATION_FAILED,
            valid=False,
            diagnostics=(ScientificReportReason.LATEX_COMPILATION_FAILURE.value,),
        )
    pdf = path.with_suffix(".pdf")
    aux_files = tuple(sorted(str(item) for item in path.parent.glob(f"{path.stem}.*") if item.suffix not in {".tex", ".pdf"}))
    if return_code == 0 and pdf.is_file():
        return ScientificReportCompilationResult(
            requested=True,
            compiler=compiler,
            command=tuple(command),
            return_code=return_code,
            pdf_path=str(pdf),
            log_path=str(stdout_log),
            auxiliary_files=aux_files,
            warnings=_extract_latex_warnings(output_text),
            errors=(),
            status=ScientificReportStatus.CREATED,
            valid=True,
            diagnostics=(ScientificReportReason.PDF_COMPILED.value,),
        )
    return ScientificReportCompilationResult(
        requested=True,
        compiler=compiler,
        command=tuple(command),
        return_code=return_code,
        pdf_path=str(pdf) if pdf.exists() else None,
        log_path=str(stdout_log),
        auxiliary_files=aux_files,
        warnings=_extract_latex_warnings(output_text),
        errors=_extract_latex_errors(output_text) or ("latex compilation failed",),
        status=ScientificReportStatus.COMPILATION_FAILED,
        valid=False,
        diagnostics=(ScientificReportReason.LATEX_COMPILATION_FAILURE.value,),
    )


def build_scientific_report_manifest(
    document: ScientificReportDocument,
    artifacts: Sequence[ScientificReportArtifact],
    settings: ScientificReportSettings | None = None,
    *,
    export_result: ExperimentExportResult | None = None,
    figure_collection: ScientificFigureCollection | None = None,
    compilation_result: ScientificReportCompilationResult | None = None,
) -> ScientificReportManifest:
    """Build the report manifest. It should be written after other artifacts."""

    cfg = settings or ScientificReportSettings()
    checksums = {
        artifact.relative_path or artifact.artifact_id: artifact.checksum
        for artifact in artifacts
    }
    return ScientificReportManifest(
        report_schema_version=ScientificReportSchemaVersion.V1_0.value,
        report_id=document.report_id,
        analysis_id=document.analysis_id,
        experiment_id=document.experiment_id,
        belllab_version=str(document.provenance.get("belllab_version", _belllab_version())),
        report_settings_fingerprint=scientific_report_settings_fingerprint(cfg),
        source_analysis_fingerprint=str(document.provenance.get("source_analysis_fingerprint")),
        source_export_id=getattr(export_result, "export_id", None),
        source_figure_collection_id=getattr(figure_collection, "collection_id", None),
        sections=tuple(_section_manifest_row(section) for section in document.sections),
        tables=tuple(_table_manifest_row(table) for table in document.tables),
        figures=tuple(_figure_manifest_row(figure) for figure in document.figures),
        appendices=tuple(_section_manifest_row(section) for section in document.appendices),
        artifacts=tuple(artifacts),
        checksums=checksums,
        compilation=_to_jsonable(compilation_result, cfg) if compilation_result is not None else None,
        source_statuses=tuple(document.provenance.get("source_statuses", ())),
        limitations=document.limitations,
        diagnostics=document.diagnostics + ("manifest_created_last",),
    )


def create_scientific_report(
    analysis: ExperimentAnalysisResult,
    export_result: ExperimentExportResult | None = None,
    figure_collection: ScientificFigureCollection | None = None,
    settings: ScientificReportSettings | None = None,
) -> ScientificReportResult:
    """Create requested Markdown, LaTeX, manifest and optional PDF artifacts."""

    cfg = settings or ScientificReportSettings()
    try:
        document = build_scientific_report_document(analysis, export_result, figure_collection, cfg)
    except Exception as exc:
        return _invalid_report_result(
            getattr(analysis, "analysis_id", "invalid-analysis"),
            getattr(getattr(analysis, "experiment", None), "experiment_id", None),
            cfg,
            f"{exc.__class__.__name__}: {exc}",
        )

    artifacts: list[ScientificReportArtifact] = []
    latex_artifact: ScientificReportArtifact | None = None
    if cfg.generate_markdown:
        artifacts.append(_write_report_artifact(document, "report", "markdown", "md", render_scientific_report_markdown(document, cfg), cfg))
    if cfg.generate_latex:
        latex_artifact = _write_report_artifact(document, "report", "latex", "tex", render_scientific_report_latex(document, cfg), cfg)
        artifacts.append(latex_artifact)
    if cfg.generate_bibliography_file:
        artifacts.append(_write_report_artifact(document, "bibliography", "bibtex", "bib", _bibliography_content(document, cfg), cfg))
    if cfg.generate_makefile:
        artifacts.append(_write_report_artifact(document, "makefile", "makefile", "Makefile", _makefile_content(document, cfg), cfg, literal_extension=True))
    if cfg.generate_latexmkrc:
        artifacts.append(_write_report_artifact(document, "latexmkrc", "latexmkrc", "latexmkrc", _latexmkrc_content(cfg), cfg, literal_extension=True))

    compilation = None
    if cfg.compile_pdf:
        latex_path = latex_artifact.path if latex_artifact and latex_artifact.status is ScientificReportStatus.CREATED else None
        compilation = compile_scientific_report_pdf(latex_path, cfg)
        if compilation.pdf_path and Path(compilation.pdf_path).is_file():
            artifacts.append(_artifact_from_existing_file(document.report_id, "report", "pdf", Path(compilation.pdf_path), cfg, (ScientificReportReason.PDF_COMPILED,)))
        if compilation.log_path and Path(compilation.log_path).is_file():
            artifacts.append(_artifact_from_existing_file(document.report_id, "compilation_log", "log", Path(compilation.log_path), cfg, (ScientificReportReason.LATEX_COMPILATION_WARNING,) if compilation.warnings else (ScientificReportReason.PDF_COMPILED,)))
    else:
        compilation = compile_scientific_report_pdf(None, cfg)

    manifest = None
    if cfg.generate_manifest:
        manifest_placeholder = _manifest_placeholder(document, cfg)
        manifest = build_scientific_report_manifest(document, tuple(artifacts) + (manifest_placeholder,), cfg, export_result=export_result, figure_collection=figure_collection, compilation_result=compilation)
        manifest_content = json.dumps(_to_jsonable(manifest, cfg), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        manifest_artifact = _write_report_artifact(document, "manifest", "json", "json", manifest_content + "\n", cfg, explicit_name="manifest")
        artifacts.append(manifest_artifact)

    completed_artifacts = tuple(item for item in artifacts if item.status is ScientificReportStatus.CREATED)
    failed_artifacts = tuple(item for item in artifacts if item.status in {ScientificReportStatus.FAILED, ScientificReportStatus.INVALID_INPUT, ScientificReportStatus.COMPILATION_FAILED})
    completed_sections = tuple(section for section in document.sections if section.status is ScientificReportStatus.CREATED)
    partial_sections = tuple(section for section in document.sections if section.status in {ScientificReportStatus.CREATED_WITH_RESERVATIONS, ScientificReportStatus.PARTIAL})
    failed_sections = tuple(section for section in document.sections if section.status in {ScientificReportStatus.FAILED, ScientificReportStatus.INVALID_INPUT})
    present_section_ids = {section.section_id for section in document.sections}
    if document.appendices:
        present_section_ids.add("appendices")
        present_section_ids.update(section.section_id for section in document.appendices)
    skipped_sections = tuple(section_id for section_id in DEFAULT_SECTION_ORDER if section_id not in present_section_ids)
    status = _report_status(analysis, artifacts, compilation, partial_sections, failed_sections)
    diagnostics = document.diagnostics + _result_diagnostics(status, artifacts, compilation)
    return ScientificReportResult(
        report_id=document.report_id,
        analysis_id=document.analysis_id,
        experiment_id=document.experiment_id,
        document=document,
        status=status,
        artifacts=tuple(artifacts),
        manifest=manifest,
        compilation_result=compilation,
        completed_sections=completed_sections,
        partial_sections=partial_sections,
        skipped_sections=skipped_sections,
        failed_sections=failed_sections,
        completed_artifacts=completed_artifacts,
        failed_artifacts=failed_artifacts,
        valid=status in {ScientificReportStatus.CREATED, ScientificReportStatus.CREATED_WITH_RESERVATIONS},
        requires_review=status is not ScientificReportStatus.CREATED or analysis.requires_review,
        failure_reason=None if status in {ScientificReportStatus.CREATED, ScientificReportStatus.CREATED_WITH_RESERVATIONS} else status.value,
        diagnostics=diagnostics,
    )


def validate_scientific_report(result: ScientificReportResult) -> ScientificReportValidation:
    """Validate artifact checksums, cross references and conservative narrative."""

    if not isinstance(result, ScientificReportResult):
        raise TypeError("result must be a ScientificReportResult.")
    expected = tuple(artifact.relative_path or artifact.artifact_id for artifact in result.artifacts)
    existing: list[str] = []
    missing: list[str] = []
    checksum_matches: dict[str, bool] = {}
    for artifact in result.artifacts:
        key = artifact.relative_path or artifact.artifact_id
        if artifact.path and Path(artifact.path).is_file():
            existing.append(key)
            if artifact.checksum:
                checksum_matches[key] = scientific_report_artifact_checksum(artifact.path) == artifact.checksum
        elif artifact.status is ScientificReportStatus.CREATED:
            missing.append(key)
            checksum_matches[key] = False
    cross_refs = _document_cross_references_valid(result.document) if result.document else False
    manifest_consistent = result.manifest is None or set(result.manifest.checksums).issubset(set(expected))
    conservative = True
    diagnostics: list[str] = []
    for artifact in result.artifacts:
        if artifact.path and artifact.format in {"markdown", "latex", "tex"} and Path(artifact.path).is_file():
            text = Path(artifact.path).read_text(encoding="utf-8")
            try:
                _assert_conservative_narrative(text)
            except ValueError as exc:
                conservative = False
                diagnostics.append(str(exc))
    checksum_valid = all(checksum_matches.values()) if checksum_matches else True
    valid = (
        result.valid
        and not missing
        and checksum_valid
        and cross_refs
        and manifest_consistent
        and conservative
    )
    reasons = (
        (ScientificReportReason.CROSS_REFERENCES_VALID,) if cross_refs else (ScientificReportReason.INVALID_CROSS_REFERENCE,)
    )
    return ScientificReportValidation(
        expected_artifacts=expected,
        existing_artifacts=tuple(existing),
        missing_artifacts=tuple(missing),
        checksum_matches=checksum_matches,
        cross_references_valid=cross_refs,
        manifest_consistent=manifest_consistent,
        conservative_narrative_valid=conservative,
        valid=valid,
        reasons=reasons if valid or checksum_valid else reasons + (ScientificReportReason.ARTIFACT_CHECKSUM_MISMATCH,),
        diagnostics=tuple(diagnostics),
    )


def summarize_scientific_report(result: ScientificReportResult | ScientificReportDocument) -> dict[str, object]:
    """Return a small deterministic report summary."""

    if isinstance(result, ScientificReportDocument):
        return {
            "report_id": result.report_id,
            "analysis_id": result.analysis_id,
            "experiment_id": result.experiment_id,
            "section_count": len(result.sections),
            "table_count": len(result.tables),
            "figure_count": len(result.figures),
            "valid": result.valid,
            "diagnostics": result.diagnostics,
        }
    if isinstance(result, ScientificReportResult):
        return {
            "report_id": result.report_id,
            "analysis_id": result.analysis_id,
            "experiment_id": result.experiment_id,
            "status": result.status.value,
            "artifact_count": len(result.artifacts),
            "completed_artifact_count": len(result.completed_artifacts),
            "failed_artifact_count": len(result.failed_artifacts),
            "completed_section_count": len(result.completed_sections),
            "partial_section_count": len(result.partial_sections),
            "skipped_sections": result.skipped_sections,
            "valid": result.valid,
            "requires_review": result.requires_review,
            "diagnostics": result.diagnostics,
        }
    raise TypeError("result must be a ScientificReportResult or ScientificReportDocument.")


def scientific_report_settings_fingerprint(settings: ScientificReportSettings | None = None) -> str:
    """Return a deterministic report-settings fingerprint independent of output path."""

    cfg = settings or ScientificReportSettings()
    payload = {
        field.name: _canonicalize(getattr(cfg, field.name))
        for field in fields(cfg)
        if field.name != "output_directory"
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def scientific_report_artifact_checksum(path: str | Path) -> str:
    """Return a SHA-256 checksum for a report artifact."""

    return export_artifact_checksum(path)


def _normalization_settings(cfg: ScientificReportSettings) -> ResultsExportSettings:
    return ResultsExportSettings(
        output_directory=cfg.output_directory,
        export_json=False,
        export_csv=True,
        export_latex=True,
        export_markdown=True,
        export_manifest=False,
        export_summary=True,
        overwrite_policy=cfg.overwrite_policy,
        atomic_write=cfg.atomic_write,
        missing_value_representation=cfg.missing_value_representation,
        numeric_formatting=cfg.numeric_formatting,
    )


def _build_report_tables(
    normalized: NormalizedExperimentExport,
    cfg: ScientificReportSettings,
) -> tuple[ScientificReportTable, ...]:
    if not cfg.include_tables:
        return ()
    tables: list[ScientificReportTable] = []
    for name, rows in normalized.tables.items():
        if not rows and name not in {"experiment_summary", "pipeline_stages", "diagnostics"}:
            continue
        columns = _table_columns(rows)
        if not columns:
            columns = ("status",)
        table_id = f"table-{name.replace('_', '-')}"
        notes: list[str] = []
        status = ScientificReportStatus.CREATED
        if len(rows) > cfg.maximum_rows_inline and cfg.large_table_policy in {"appendix", "skip_with_note"}:
            notes.append(ScientificReportReason.LONG_TABLE_SPLIT.value)
            status = ScientificReportStatus.CREATED_WITH_RESERVATIONS
        tables.append(ScientificReportTable(
            table_id=table_id,
            title=_table_title(name),
            columns=columns,
            rows=tuple(rows),
            source_ids=(normalized.analysis_id,),
            status=status,
            notes=tuple(notes),
            diagnostics=("table_reused_from_normalized_export_model",),
        ))
    return tuple(sorted(tables, key=lambda table: table.table_id))


def _build_report_figures(
    collection: ScientificFigureCollection | None,
    cfg: ScientificReportSettings,
) -> tuple[ScientificReportFigureReference, ...]:
    if not cfg.include_figures or collection is None:
        return ()
    figures: list[ScientificReportFigureReference] = []
    for figure in sorted(collection.figures, key=lambda item: (item.figure_type.value, item.figure_id)):
        artifact = _preferred_figure_artifact(figure.artifacts, cfg)
        if artifact is None:
            figures.append(ScientificReportFigureReference(
                figure_id=figure.figure_id,
                figure_type=figure.figure_type.value,
                title=_figure_title(figure.figure_type.value),
                caption=_figure_caption(figure.figure_type.value, figure.status.value),
                relative_path=None,
                source_ids=figure.source_ids,
                checksum=None,
                status=ScientificReportStatus.PARTIAL,
                diagnostics=(ScientificReportReason.MISSING_OPTIONAL_FIGURE.value,),
            ))
            continue
        diagnostics = list(figure.diagnostics) + list(artifact.diagnostics)
        status = ScientificReportStatus.CREATED
        if figure.status is not ScientificVisualizationStatus.CREATED:
            status = ScientificReportStatus.CREATED_WITH_RESERVATIONS
        if cfg.validate_checksums_before_render and artifact.path and artifact.checksum:
            actual = scientific_report_artifact_checksum(artifact.path)
            if actual != artifact.checksum:
                status = ScientificReportStatus.FAILED
                diagnostics.append(ScientificReportReason.ARTIFACT_CHECKSUM_MISMATCH.value)
        figures.append(ScientificReportFigureReference(
            figure_id=figure.figure_id,
            figure_type=figure.figure_type.value,
            title=_figure_title(figure.figure_type.value),
            caption=_figure_caption(figure.figure_type.value, figure.status.value),
            relative_path=_relative_path(Path(artifact.path), cfg) if artifact.path else artifact.relative_path,
            source_ids=figure.source_ids,
            checksum=artifact.checksum,
            status=status,
            diagnostics=tuple(diagnostics),
        ))
    return tuple(figures)


def _preferred_figure_artifact(
    artifacts: Sequence[ScientificFigureArtifact],
    cfg: ScientificReportSettings,
) -> ScientificFigureArtifact | None:
    created = tuple(item for item in artifacts if item.status is ScientificVisualizationStatus.CREATED and item.path)
    if not created:
        return None
    for artifact in created:
        if artifact.format == cfg.figure_format_preference:
            return artifact
    return sorted(created, key=lambda item: (item.format, item.relative_path or ""))[0]


def _build_appendices(
    normalized: NormalizedExperimentExport,
    tables: Sequence[ScientificReportTable],
    cfg: ScientificReportSettings,
) -> tuple[ScientificReportSection, ...]:
    if not cfg.include_appendices:
        return ()
    blocks: list[ScientificReportContentBlock] = []
    table_ids: list[str] = []
    if cfg.include_settings_dump:
        blocks.append(_provenance_block("appendix-settings", normalized.settings, normalized.analysis_id))
    if cfg.include_full_diagnostics:
        table_ids.append("table-diagnostics")
    for table in tables:
        if len(table.rows) > cfg.maximum_rows_inline:
            table_ids.append(table.table_id)
    if not blocks and not table_ids:
        blocks.append(_paragraph("appendix-note", "Nenhum apendice extenso foi necessario; conteudos completos permanecem no modelo normalizado.", normalized.analysis_id))
    return (
        _section(
            "appendix_artifacts_and_diagnostics",
            "Apendices e inventario",
            0,
            tuple(blocks),
            table_ids=tuple(table_ids),
            status=ScientificReportStatus.CREATED_WITH_RESERVATIONS if table_ids else ScientificReportStatus.CREATED,
            reasons=(ScientificReportReason.LONG_TABLE_SPLIT,) if table_ids else (),
            level=1,
        ),
    )


def _validate_source_compatibility(
    analysis: ExperimentAnalysisResult,
    export_result: ExperimentExportResult | None,
    collection: ScientificFigureCollection | None,
    cfg: ScientificReportSettings,
) -> None:
    if export_result is not None:
        if export_result.analysis_id != analysis.analysis_id:
            raise ValueError("export_result analysis_id does not match analysis.")
        if export_result.experiment_id != analysis.experiment.experiment_id:
            raise ValueError("export_result experiment_id does not match analysis.")
        if cfg.validate_checksums_before_render:
            for artifact in export_result.artifacts:
                if artifact.path and artifact.checksum and Path(artifact.path).is_file():
                    if scientific_report_artifact_checksum(artifact.path) != artifact.checksum:
                        raise ValueError(f"export artifact checksum mismatch: {artifact.relative_path or artifact.path}")
    if collection is not None:
        if collection.analysis_id not in {None, analysis.analysis_id}:
            raise ValueError("figure_collection analysis_id does not match analysis.")
        if collection.experiment_id not in {None, analysis.experiment.experiment_id}:
            raise ValueError("figure_collection experiment_id does not match analysis.")
        if cfg.validate_checksums_before_render:
            for artifact in collection.artifacts:
                if artifact.path and artifact.checksum and Path(artifact.path).is_file():
                    if scientific_report_artifact_checksum(artifact.path) != artifact.checksum:
                        raise ValueError(f"figure artifact checksum mismatch: {artifact.relative_path or artifact.path}")


def _document_provenance(
    analysis: ExperimentAnalysisResult,
    export_result: ExperimentExportResult | None,
    collection: ScientificFigureCollection | None,
    cfg: ScientificReportSettings,
    normalized: NormalizedExperimentExport,
) -> Mapping[str, object]:
    source_statuses = [analysis.status.value]
    if export_result is not None:
        source_statuses.append(export_result.status.value)
    if collection is not None:
        source_statuses.append(collection.status.value)
    source_fingerprint = _stable_id(
        "analysis-source",
        analysis.analysis_id,
        analysis.status.value,
        analysis.provenance.settings_fingerprint,
        analysis.provenance.file_fingerprints,
        summarize_experiment_analysis(analysis),
    )
    return MappingProxyType({
        "analysis_id": analysis.analysis_id,
        "experiment_id": analysis.experiment.experiment_id,
        "belllab_version": normalized.belllab_version,
        "source_analysis_fingerprint": source_fingerprint,
        "source_export_id": getattr(export_result, "export_id", None),
        "source_figure_collection_id": getattr(collection, "collection_id", None),
        "report_settings_fingerprint": scientific_report_settings_fingerprint(cfg),
        "analysis_settings_fingerprint": analysis.provenance.settings_fingerprint,
        "file_fingerprints": analysis.provenance.file_fingerprints,
        "source_statuses": tuple(source_statuses),
        "schema_version": ScientificReportSchemaVersion.V1_0.value,
    })


def _derive_limitations(
    analysis: ExperimentAnalysisResult,
    normalized: NormalizedExperimentExport,
    export_result: ExperimentExportResult | None,
    collection: ScientificFigureCollection | None,
    cfg: ScientificReportSettings,
) -> tuple[str, ...]:
    limitations = [
        "Relatorio compilado nao e conclusao cientifica comprovada.",
        "Hipotese modal nao e modo fisico comprovado.",
        "Associacao entre condicoes nao prova identidade fisica.",
        "Trajetoria de frequencia nao prova nao linearidade.",
        "Anticorrelacao temporal nao comprova transferencia fisica de energia.",
        "Validacao sintetica nao e validacao experimental universal.",
    ]
    if analysis.status is not ExperimentAnalysisStatus.COMPLETED:
        limitations.append(f"Source analysis status is {analysis.status.value}.")
    if analysis.requires_review:
        limitations.append("Source analysis requires review.")
    if export_result is None:
        limitations.append("Structured export result was not supplied; report tables were built from the normalized analysis model.")
    elif not export_result.valid:
        limitations.append(f"Export result status is {export_result.status.value}.")
    if collection is None:
        limitations.append("Figure collection was not supplied; figure sections report missing optional figures.")
    elif not collection.valid:
        limitations.append(f"Figure collection status is {collection.status.value}.")
    for row in normalized.tables.get("pipeline_stages", ()):
        status = str(row.get("status"))
        if status in {"failed", "blocked", "invalid_input", "insufficient_evidence", "skipped"}:
            limitations.append(f"Pipeline stage {row.get('stage')} has status {status}.")
    for row in normalized.tables.get("modal_q_factors", ()):
        status = str(row.get("status"))
        if status in {"inconclusive", "insufficient_evidence", "invalid_input", "partial"}:
            limitations.append(f"Q estimate {row.get('estimate_id')} has status {status}.")
    return _unique_texts(limitations)


def _section(
    section_id: str,
    title: str,
    order: int,
    blocks: tuple[ScientificReportContentBlock, ...],
    *,
    table_ids: tuple[str, ...] = (),
    figure_ids: tuple[str, ...] = (),
    status: ScientificReportStatus = ScientificReportStatus.CREATED,
    reasons: tuple[ScientificReportReason, ...] = (),
    source_ids: tuple[str, ...] = (),
    level: int = 1,
) -> ScientificReportSection:
    return ScientificReportSection(
        section_id=section_id,
        title=title,
        level=level,
        order=order,
        content_blocks=blocks,
        table_ids=table_ids,
        figure_ids=figure_ids,
        source_ids=source_ids,
        status=status,
        reasons=reasons,
        provenance=MappingProxyType({"section_schema": ScientificReportSchemaVersion.V1_0.value}),
        diagnostics=("section_generated_from_structured_sources",),
    )


def _mark_section_availability(
    section: ScientificReportSection,
    table_by_id: Mapping[str, ScientificReportTable],
    figures: Sequence[ScientificReportFigureReference],
    cfg: ScientificReportSettings,
) -> ScientificReportSection:
    available_figures = {figure.figure_id: figure for figure in figures if figure.status in {ScientificReportStatus.CREATED, ScientificReportStatus.CREATED_WITH_RESERVATIONS}}
    missing_tables = tuple(table_id for table_id in section.table_ids if table_id not in table_by_id)
    missing_figures = tuple(figure_id for figure_id in section.figure_ids if figure_id not in available_figures)
    reasons = list(section.reasons)
    diagnostics = list(section.diagnostics)
    status = section.status
    if missing_tables:
        reasons.append(ScientificReportReason.MISSING_OPTIONAL_TABLE)
        diagnostics.append(f"missing_tables:{','.join(missing_tables)}")
        status = ScientificReportStatus.CREATED_WITH_RESERVATIONS
    if missing_figures:
        reasons.append(ScientificReportReason.MISSING_OPTIONAL_FIGURE)
        diagnostics.append(f"missing_figures:{','.join(missing_figures)}")
        status = ScientificReportStatus.CREATED_WITH_RESERVATIONS
    if len(section.figure_ids) and cfg.maximum_figures_per_section is not None and len(section.figure_ids) > cfg.maximum_figures_per_section:
        reasons.append(ScientificReportReason.MISSING_OPTIONAL_FIGURE)
        diagnostics.append("figure_count_limited_by_maximum_figures_per_section")
        status = ScientificReportStatus.CREATED_WITH_RESERVATIONS
    return ScientificReportSection(
        section_id=section.section_id,
        title=section.title,
        level=section.level,
        order=section.order,
        content_blocks=section.content_blocks,
        table_ids=section.table_ids,
        figure_ids=section.figure_ids,
        subsections=section.subsections,
        source_ids=section.source_ids,
        status=status,
        reasons=tuple(reasons),
        provenance=section.provenance,
        diagnostics=tuple(diagnostics),
    )


def _paragraph(block_id: str, text: str, source_id: str) -> ScientificReportContentBlock:
    return ScientificReportContentBlock(block_id=block_id, block_type=ScientificReportContentBlockType.PARAGRAPH, content=text, source_ids=(source_id,))


def _notice(block_id: str, text: str, source_id: str) -> ScientificReportContentBlock:
    return ScientificReportContentBlock(block_id=block_id, block_type=ScientificReportContentBlockType.SCIENTIFIC_NOTICE, content=text, source_ids=(source_id,), style="scientific_caution")


def _limitation(block_id_prefix: str, text: str, source_id: str) -> ScientificReportContentBlock:
    return ScientificReportContentBlock(block_id=_stable_id(block_id_prefix, text), block_type=ScientificReportContentBlockType.LIMITATION, content=text, source_ids=(source_id,))


def _equation(block_id: str, text: str, source_id: str) -> ScientificReportContentBlock:
    return ScientificReportContentBlock(block_id=block_id, block_type=ScientificReportContentBlockType.EQUATION, content=text, source_ids=(source_id,))


def _provenance_block(block_id: str, payload: Mapping[str, object], source_id: str) -> ScientificReportContentBlock:
    return ScientificReportContentBlock(block_id=block_id, block_type=ScientificReportContentBlockType.PROVENANCE, content=MappingProxyType(dict(payload)), source_ids=(source_id,))


def _render_markdown_section(
    section: ScientificReportSection,
    table_map: Mapping[str, ScientificReportTable],
    figure_map: Mapping[str, ScientificReportFigureReference],
    lines: list[str],
    cfg: ScientificReportSettings,
) -> None:
    lines.append("")
    lines.append(f"{'#' * (section.level + 1)} {_safe_md(section.title)}")
    lines.append(f"<!-- section_id: {section.section_id} -->")
    for block in section.content_blocks:
        _render_markdown_block(block, lines, cfg)
    for table_id in section.table_ids:
        table = table_map.get(table_id)
        if table is not None:
            _render_markdown_table(table, lines, cfg)
    for figure_id in section.figure_ids:
        figure = figure_map.get(figure_id)
        if figure is not None:
            _render_markdown_figure(figure, lines, cfg)


def _render_markdown_block(block: ScientificReportContentBlock, lines: list[str], cfg: ScientificReportSettings) -> None:
    lines.append("")
    if block.block_type is ScientificReportContentBlockType.PARAGRAPH:
        lines.append(_safe_md(str(block.content)))
    elif block.block_type is ScientificReportContentBlockType.SCIENTIFIC_NOTICE:
        lines.append(f"> **Aviso cientifico:** {_safe_md(str(block.content))}")
    elif block.block_type is ScientificReportContentBlockType.LIMITATION:
        lines.append(f"- {_safe_md(str(block.content))}")
    elif block.block_type is ScientificReportContentBlockType.EQUATION:
        lines.append("$$")
        lines.append(str(block.content))
        lines.append("$$")
    elif block.block_type is ScientificReportContentBlockType.PROVENANCE:
        lines.append("```json")
        lines.append(json.dumps(_to_jsonable(block.content, cfg), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
        lines.append("```")
    elif block.block_type is ScientificReportContentBlockType.LIST and isinstance(block.content, Sequence):
        for item in block.content:
            lines.append(f"- {_safe_md(str(item))}")
    elif block.block_type is ScientificReportContentBlockType.PAGE_BREAK:
        lines.append("\\newpage")
    else:
        lines.append(_safe_md(str(block.content)))


def _render_markdown_table(table: ScientificReportTable, lines: list[str], cfg: ScientificReportSettings) -> None:
    lines.append("")
    lines.append(f"Table `{table.table_id}`: {_safe_md(table.title)}")
    rows = table.rows if cfg.large_table_policy == "inline" else table.rows[: cfg.maximum_rows_inline]
    columns = table.columns
    lines.append("| " + " | ".join(_safe_md(column) for column in columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_safe_md(_format_value(row.get(column), cfg)) for column in columns) + " |")
    if len(rows) < len(table.rows):
        lines.append("")
        lines.append(f"_Tabela longa: {len(rows)} de {len(table.rows)} linhas exibidas inline; conteudo completo permanece no modelo do relatorio._")
    for note in table.notes:
        lines.append(f"_Note: {_safe_md(note)}_")


def _render_markdown_figure(figure: ScientificReportFigureReference, lines: list[str], cfg: ScientificReportSettings) -> None:
    lines.append("")
    if figure.relative_path:
        lines.append(f"![{_safe_md(figure.title)}]({_safe_link(figure.relative_path)})")
    else:
        lines.append(f"Figura `{figure.figure_id}` indisponivel.")
    lines.append(f"_Figure `{figure.figure_id}`: {_safe_md(figure.caption)}_")
    if cfg.include_figure_provenance:
        lines.append(f"_Source IDs: {_safe_md('; '.join(figure.source_ids))}; status: {figure.status.value}_")


def _render_latex_section(
    section: ScientificReportSection,
    table_map: Mapping[str, ScientificReportTable],
    figure_map: Mapping[str, ScientificReportFigureReference],
    lines: list[str],
    cfg: ScientificReportSettings,
) -> None:
    command = "section" if section.level == 1 else "subsection"
    lines.append(f"\\{command}{{{_latex_escape(section.title)}}}\\label{{sec:{_latex_label(section.section_id)}}}")
    for block in section.content_blocks:
        _render_latex_block(block, lines, cfg)
    for table_id in section.table_ids:
        table = table_map.get(table_id)
        if table is not None:
            _render_latex_table(table, lines, cfg)
    for figure_id in section.figure_ids:
        figure = figure_map.get(figure_id)
        if figure is not None:
            _render_latex_figure(figure, lines, cfg)


def _render_latex_block(block: ScientificReportContentBlock, lines: list[str], cfg: ScientificReportSettings) -> None:
    if block.block_type is ScientificReportContentBlockType.PARAGRAPH:
        lines.append(_latex_paragraph(str(block.content)))
    elif block.block_type is ScientificReportContentBlockType.SCIENTIFIC_NOTICE:
        lines.append(_latex_paragraph(f"Aviso cientifico: {block.content}"))
    elif block.block_type is ScientificReportContentBlockType.LIMITATION:
        lines.append(f"\\begin{{itemize}}\\item {_latex_escape(str(block.content))}\\end{{itemize}}")
    elif block.block_type is ScientificReportContentBlockType.EQUATION:
        lines.append("\\begin{equation}")
        lines.append(str(block.content))
        lines.append("\\end{equation}")
    elif block.block_type is ScientificReportContentBlockType.PROVENANCE:
        lines.append("\\begin{verbatim}")
        lines.append(json.dumps(_to_jsonable(block.content, cfg), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
        lines.append("\\end{verbatim}")
    elif block.block_type is ScientificReportContentBlockType.PAGE_BREAK:
        lines.append("\\clearpage")
    else:
        lines.append(_latex_paragraph(str(block.content)))


def _render_latex_table(table: ScientificReportTable, lines: list[str], cfg: ScientificReportSettings) -> None:
    rows = table.rows if cfg.large_table_policy == "inline" else table.rows[: cfg.maximum_rows_inline]
    environment = "longtable" if len(table.rows) > cfg.maximum_rows_inline else "tabular"
    spec = "l" * len(table.columns)
    if environment == "longtable":
        lines.append(f"\\begin{{longtable}}{{{spec}}}")
        lines.append(f"\\caption{{{_latex_escape(table.title)}}}\\label{{tab:{_latex_label(table.table_id)}}}\\\\")
    else:
        lines.append(f"\\begin{{table}}[htbp]\\centering\\caption{{{_latex_escape(table.title)}}}\\label{{tab:{_latex_label(table.table_id)}}}")
        lines.append(f"\\begin{{tabular}}{{{spec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(_latex_escape(column.replace("_", " ")) for column in table.columns) + r" \\")
    lines.append("\\midrule")
    for row in rows:
        lines.append(" & ".join(_latex_escape(_format_value(row.get(column), cfg)) for column in table.columns) + r" \\")
    lines.append("\\bottomrule")
    if environment == "longtable":
        lines.append("\\end{longtable}")
        if len(rows) < len(table.rows):
            lines.append(_latex_paragraph(f"Tabela longa: {len(rows)} de {len(table.rows)} linhas exibidas inline; conteudo completo permanece no modelo do relatorio."))
        return
    lines.append("\\end{tabular}")
    if len(rows) < len(table.rows):
        lines.append(_latex_paragraph(f"Tabela longa: {len(rows)} de {len(table.rows)} linhas exibidas inline; conteudo completo permanece no modelo do relatorio."))
    lines.append("\\end{table}")


def _render_latex_figure(figure: ScientificReportFigureReference, lines: list[str], cfg: ScientificReportSettings) -> None:
    lines.append("\\begin{figure}[htbp]")
    lines.append("\\centering")
    if figure.relative_path:
        lines.append(f"\\includegraphics[width={cfg.figure_width_fraction:.3f}\\linewidth]{{\\detokenize{{{figure.relative_path}}}}}")
    else:
        lines.append(_latex_escape(f"Figure unavailable: {figure.figure_id}"))
    lines.append(f"\\caption{{{_latex_escape(figure.caption)}}}\\label{{fig:{_latex_label(figure.figure_id)}}}")
    lines.append("\\end{figure}")


def _write_report_artifact(
    document: ScientificReportDocument,
    artifact_type: str,
    artifact_format: str,
    extension: str,
    content: str,
    cfg: ScientificReportSettings,
    *,
    explicit_name: str | None = None,
    literal_extension: bool = False,
) -> ScientificReportArtifact:
    name = explicit_name or artifact_type
    path = _report_artifact_path(document, cfg, name, extension, literal_extension=literal_extension)
    try:
        actual_path, skipped = _write_text(path, content, cfg)
        if skipped:
            checksum = scientific_report_artifact_checksum(actual_path) if actual_path.is_file() else None
            return ScientificReportArtifact(
                artifact_id=_stable_id("report-artifact", document.report_id, artifact_type, artifact_format, "skipped", checksum),
                artifact_type=artifact_type,
                format=artifact_format,
                path=str(actual_path),
                relative_path=_relative_path(actual_path, cfg),
                checksum=checksum,
                size_bytes=actual_path.stat().st_size if actual_path.is_file() else None,
                status=ScientificReportStatus.PARTIAL,
                reasons=(ScientificReportReason.EXISTING_FILE_CONFLICT,),
                diagnostics=("existing_report_artifact_preserved_by_skip_policy",),
            )
        checksum = scientific_report_artifact_checksum(actual_path)
        return ScientificReportArtifact(
            artifact_id=_stable_id("report-artifact", document.report_id, artifact_type, artifact_format, checksum),
            artifact_type=artifact_type,
            format=artifact_format,
            path=str(actual_path),
            relative_path=_relative_path(actual_path, cfg),
            checksum=checksum,
            size_bytes=actual_path.stat().st_size,
            status=ScientificReportStatus.CREATED,
            reasons=(ScientificReportReason.ALL_REQUESTED_ARTIFACTS_INCLUDED,),
            diagnostics=("report_artifact_written_without_recalculating_sources",),
        )
    except FileExistsError as exc:
        return _failed_artifact(document.report_id, artifact_type, artifact_format, ScientificReportReason.EXISTING_FILE_CONFLICT, str(exc))
    except Exception as exc:
        return _failed_artifact(document.report_id, artifact_type, artifact_format, ScientificReportReason.FILESYSTEM_ERROR, f"{exc.__class__.__name__}: {exc}")


def _write_text(path: Path, content: str, cfg: ScientificReportSettings) -> tuple[Path, bool]:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
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
            os.replace(tmp, target)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise
    else:
        target.write_text(content, encoding="utf-8")
    return target, False


def _resolve_overwrite_path(path: Path, cfg: ScientificReportSettings) -> Path | None:
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


def _report_artifact_path(
    document: ScientificReportDocument,
    cfg: ScientificReportSettings,
    artifact_name: str,
    extension: str,
    *,
    literal_extension: bool = False,
) -> Path:
    base = f"{_sanitize(cfg.file_prefix)}_{_sanitize(document.report_id)}_{_sanitize(artifact_name)}"
    if literal_extension and extension == "Makefile":
        return Path(cfg.output_directory) / "Makefile"
    if literal_extension and extension == "latexmkrc":
        return Path(cfg.output_directory) / ".latexmkrc"
    return Path(cfg.output_directory) / f"{base}.{extension}"


def _artifact_from_existing_file(
    report_id: str,
    artifact_type: str,
    artifact_format: str,
    path: Path,
    cfg: ScientificReportSettings,
    reasons: tuple[ScientificReportReason, ...],
) -> ScientificReportArtifact:
    checksum = scientific_report_artifact_checksum(path)
    return ScientificReportArtifact(
        artifact_id=_stable_id("report-artifact", report_id, artifact_type, artifact_format, checksum),
        artifact_type=artifact_type,
        format=artifact_format,
        path=str(path),
        relative_path=_relative_path(path, cfg),
        checksum=checksum,
        size_bytes=path.stat().st_size,
        status=ScientificReportStatus.CREATED,
        reasons=reasons,
        diagnostics=("existing_compilation_artifact_registered",),
    )


def _manifest_placeholder(
    document: ScientificReportDocument,
    cfg: ScientificReportSettings,
) -> ScientificReportArtifact:
    path = _report_artifact_path(document, cfg, "manifest", "json")
    return ScientificReportArtifact(
        artifact_id=_stable_id("report-artifact", document.report_id, "manifest", "json", "self_checksum_omitted"),
        artifact_type="manifest",
        format="json",
        path=str(path),
        relative_path=_relative_path(path, cfg),
        checksum=None,
        size_bytes=None,
        status=ScientificReportStatus.CREATED,
        reasons=(ScientificReportReason.MANIFEST_CREATED,),
        diagnostics=("manifest_self_checksum_omitted_to_avoid_recursive_identity",),
    )


def _failed_artifact(
    report_id: str,
    artifact_type: str,
    artifact_format: str,
    reason: ScientificReportReason,
    diagnostic: str,
) -> ScientificReportArtifact:
    return ScientificReportArtifact(
        artifact_id=_stable_id("report-artifact", report_id, artifact_type, artifact_format, "failed", diagnostic),
        artifact_type=artifact_type,
        format=artifact_format,
        path=None,
        relative_path=None,
        checksum=None,
        size_bytes=None,
        status=ScientificReportStatus.FAILED,
        reasons=(reason,),
        diagnostics=(diagnostic,),
    )


def _invalid_report_result(
    analysis_id: str,
    experiment_id: str | None,
    cfg: ScientificReportSettings,
    reason: str,
) -> ScientificReportResult:
    report_id = _stable_id("scientific-report", analysis_id, experiment_id, scientific_report_settings_fingerprint(cfg), "invalid", reason)
    return ScientificReportResult(
        report_id=report_id,
        analysis_id=analysis_id,
        experiment_id=experiment_id,
        document=None,
        status=ScientificReportStatus.INVALID_INPUT,
        artifacts=(),
        manifest=None,
        compilation_result=None,
        completed_sections=(),
        partial_sections=(),
        skipped_sections=DEFAULT_SECTION_ORDER,
        failed_sections=(),
        completed_artifacts=(),
        failed_artifacts=(),
        valid=False,
        requires_review=True,
        failure_reason=reason,
        diagnostics=(ScientificReportReason.MISSING_REQUIRED_SOURCE.value, reason),
    )


def _report_status(
    analysis: ExperimentAnalysisResult,
    artifacts: Sequence[ScientificReportArtifact],
    compilation: ScientificReportCompilationResult | None,
    partial_sections: Sequence[ScientificReportSection],
    failed_sections: Sequence[ScientificReportSection],
) -> ScientificReportStatus:
    if failed_sections:
        return ScientificReportStatus.FAILED
    if any(artifact.status in {ScientificReportStatus.FAILED, ScientificReportStatus.INVALID_INPUT} for artifact in artifacts):
        return ScientificReportStatus.FAILED
    if compilation and compilation.status is ScientificReportStatus.COMPILATION_FAILED:
        return ScientificReportStatus.COMPILATION_FAILED
    if not any(artifact.status is ScientificReportStatus.CREATED for artifact in artifacts):
        return ScientificReportStatus.PARTIAL
    if partial_sections or analysis.status is not ExperimentAnalysisStatus.COMPLETED or analysis.requires_review:
        return ScientificReportStatus.CREATED_WITH_RESERVATIONS
    if any(artifact.status is ScientificReportStatus.PARTIAL for artifact in artifacts):
        return ScientificReportStatus.CREATED_WITH_RESERVATIONS
    if compilation and compilation.requested and not compilation.valid:
        return ScientificReportStatus.CREATED_WITH_RESERVATIONS
    return ScientificReportStatus.CREATED


def _result_diagnostics(
    status: ScientificReportStatus,
    artifacts: Sequence[ScientificReportArtifact],
    compilation: ScientificReportCompilationResult | None,
) -> tuple[str, ...]:
    diagnostics = ["report_treated_as_reproducible_organization_not_physical_proof"]
    if status is not ScientificReportStatus.CREATED:
        diagnostics.append(f"report_status:{status.value}")
    if any(artifact.status is ScientificReportStatus.PARTIAL for artifact in artifacts):
        diagnostics.append("partial_artifact_present")
    if compilation and compilation.requested and not compilation.valid:
        diagnostics.append("pdf_compilation_not_successful_or_unavailable")
    return tuple(diagnostics)


def _select_latex_compiler(cfg: ScientificReportSettings) -> str | None:
    candidates = ("latexmk", "tectonic", "pdflatex", "lualatex") if cfg.latex_engine == "auto" else (cfg.latex_engine,)
    for candidate in candidates:
        if shutil.which(candidate):
            return candidate
    return None


def _latex_command(compiler: str, filename: str, cfg: ScientificReportSettings) -> list[str]:
    if compiler == "latexmk":
        command = ["latexmk", "-pdf", f"-interaction={cfg.interaction_mode}"]
        if cfg.halt_on_error:
            command.append("-halt-on-error")
        if cfg.shell_escape:
            command.append("-shell-escape")
        command.append(filename)
        return command
    if compiler == "tectonic":
        command = ["tectonic"]
        if not cfg.shell_escape:
            command.append("--untrusted")
        command.append(filename)
        return command
    command = [compiler, f"-interaction={cfg.interaction_mode}"]
    if cfg.halt_on_error:
        command.append("-halt-on-error")
    if cfg.shell_escape:
        command.append("-shell-escape")
    command.append(filename)
    return command


def _extract_latex_warnings(text: str) -> tuple[str, ...]:
    return tuple(line.strip()[:240] for line in text.splitlines() if "Warning" in line)


def _extract_latex_errors(text: str) -> tuple[str, ...]:
    return tuple(line.strip()[:240] for line in text.splitlines() if line.startswith("!") or "Error" in line)


def _bibliography_content(document: ScientificReportDocument, cfg: ScientificReportSettings) -> str:
    return "% BellLab report bibliography placeholder. No bibliography entries were invented.\n"


def _makefile_content(document: ScientificReportDocument, cfg: ScientificReportSettings) -> str:
    tex = _report_artifact_path(document, cfg, "report", "tex").name
    return f"all:\n\tlatexmk -pdf -interaction={cfg.interaction_mode} {tex}\n"


def _latexmkrc_content(cfg: ScientificReportSettings) -> str:
    return "$pdf_mode = 1;\n$interaction = 'nonstopmode';\n"


def _automatic_abstract(
    summary: Mapping[str, object],
    normalized: NormalizedExperimentExport,
    cfg: ScientificReportSettings,
) -> str:
    return (
        f"Foram analisadas {summary.get('recording_count')} gravacoes e "
        f"{summary.get('condition_count')} condicoes dinamicas. "
        f"O pipeline registrou {summary.get('candidate_count')} candidatos, "
        f"{summary.get('chain_count')} cadeias, {summary.get('hypothesis_count')} hipoteses, "
        f"{summary.get('parameter_estimate_count')} estimativas de parametros, "
        f"{summary.get('q_estimate_count')} estimativas de Q e "
        f"{summary.get('energy_pair_count')} pares de evidencia operacional de possivel redistribuicao. "
        "Este resumo e factual e nao declara causalidade, identidade modal fisica ou validade experimental universal."
    )


def _factual_synthesis_blocks(
    summary: Mapping[str, object],
    normalized: NormalizedExperimentExport,
    analysis_id: str,
) -> tuple[ScientificReportContentBlock, ...]:
    blocks = [
        _paragraph(
            "factual-synthesis-counts",
            f"Contagens principais: candidatos={summary.get('candidate_count')}, chains={summary.get('chain_count')}, hipoteses={summary.get('hypothesis_count')}, parametros={summary.get('parameter_estimate_count')}, Q={summary.get('q_estimate_count')}.",
            analysis_id,
        )
    ]
    parameter_rows = normalized.tables.get("modal_parameters", ())
    frequencies = tuple(row.get("representative_frequency_hz") for row in parameter_rows if row.get("representative_frequency_hz") is not None)
    if frequencies:
        blocks.append(_paragraph("factual-synthesis-frequency", f"Frequencias representativas disponiveis: {_join(frequencies[:8], ScientificReportSettings())}.", analysis_id))
    else:
        blocks.append(_paragraph("factual-synthesis-frequency-missing", "Nenhuma frequencia representativa disponivel nas tabelas normalizadas.", analysis_id))
    return tuple(blocks)


def _scientific_caution_text(cfg: ScientificReportSettings) -> str:
    return (
        "Relatorio compilado nao e conclusao cientifica comprovada; figura incluida nao e evidencia adicional; "
        "hipotese modal nao e modo fisico comprovado; associacao entre condicoes nao prova identidade fisica; "
        "trajetoria de frequencia nao prova nao linearidade; anticorrelacao temporal nao comprova transferencia fisica de energia."
    )


def _report_title(analysis: ExperimentAnalysisResult, cfg: ScientificReportSettings) -> str:
    return cfg.title or f"BellLab scientific report: {analysis.experiment.name}"


def _report_fallback_title(document: ScientificReportDocument) -> str:
    return f"BellLab scientific report {document.experiment_id or document.analysis_id}"


def _table_columns(rows: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    columns: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(str(key))
    return tuple(columns)


def _table_title(name: str) -> str:
    titles = {
        "experiment_summary": "Resumo do experimento",
        "recordings": "Gravacoes",
        "conditions": "Condicoes dinamicas",
        "candidates": "Candidatos modais",
        "within_condition_associations": "Associacoes dentro da condicao",
        "cross_condition_matches": "Associacoes entre condicoes adjacentes",
        "candidate_chains": "Cadeias de candidatos",
        "candidate_chain_nodes": "Nos das cadeias",
        "modal_hypotheses": "Hipoteses modais",
        "modal_parameters": "Parametros modais operacionais",
        "modal_q_factors": "Q e largura de banda",
        "energy_exchange_pairs": "Evidencia operacional de possivel redistribuicao",
        "pipeline_stages": "Estagios do pipeline",
        "diagnostics": "Diagnosticos",
    }
    return titles.get(name, name.replace("_", " "))


def _figure_title(figure_type: str) -> str:
    return figure_type.replace("_", " ").title()


def _figure_caption(figure_type: str, status: str) -> str:
    if figure_type == "modal_energy_exchange_evidence":
        return f"Evidencia operacional de possivel redistribuicao, status {status}; nao e transferencia fisica comprovada."
    if figure_type == "modal_hypotheses":
        return f"Hipoteses modais, status {status}; nao sao modos fisicos comprovados."
    if figure_type == "modal_frequency_trajectories":
        return f"Trajetorias de frequencia, status {status}; nao sao prova de nao linearidade."
    return f"Figura operacional {figure_type}, status {status}; representa resultados de origem sem adicionar evidencia fisica."


def _limit_figures(figure_ids: Sequence[str], cfg: ScientificReportSettings) -> tuple[str, ...]:
    values = tuple(figure_ids)
    if cfg.maximum_figures_per_section is None:
        return values
    return values[: cfg.maximum_figures_per_section]


def _build_references(
    sections: Sequence[ScientificReportSection],
    tables: Sequence[ScientificReportTable],
    figures: Sequence[ScientificReportFigureReference],
    appendices: Sequence[ScientificReportSection],
) -> Mapping[str, str]:
    refs: dict[str, str] = {}
    for section in tuple(sections) + tuple(appendices):
        refs[f"sec:{section.section_id}"] = section.title
    for table in tables:
        refs[f"tab:{table.table_id}"] = table.title
    for figure in figures:
        refs[f"fig:{figure.figure_id}"] = figure.title
    return MappingProxyType(refs)


def _document_cross_references_valid(document: ScientificReportDocument | None) -> bool:
    if document is None:
        return False
    refs = set(document.references)
    expected = {f"sec:{section.section_id}" for section in tuple(document.sections) + tuple(document.appendices)}
    expected.update(f"tab:{table.table_id}" for table in document.tables)
    expected.update(f"fig:{figure.figure_id}" for figure in document.figures)
    return refs == expected and len(refs) == len(document.references)


def _validate_document(document: ScientificReportDocument) -> None:
    if not isinstance(document, ScientificReportDocument):
        raise TypeError("document must be ScientificReportDocument.")
    ids = [section.section_id for section in tuple(document.sections) + tuple(document.appendices)]
    ids += [table.table_id for table in document.tables]
    ids += [figure.figure_id for figure in document.figures]
    if len(ids) != len(set(ids)):
        raise ValueError("report document contains duplicate IDs.")
    if not _document_cross_references_valid(document):
        raise ValueError("report document has invalid cross references.")


def _section_manifest_row(section: ScientificReportSection) -> Mapping[str, object]:
    return MappingProxyType({
        "section_id": section.section_id,
        "title": section.title,
        "status": section.status.value,
        "table_ids": section.table_ids,
        "figure_ids": section.figure_ids,
        "reasons": tuple(reason.value for reason in section.reasons),
    })


def _table_manifest_row(table: ScientificReportTable) -> Mapping[str, object]:
    return MappingProxyType({
        "table_id": table.table_id,
        "title": table.title,
        "row_count": len(table.rows),
        "status": table.status.value,
        "notes": table.notes,
    })


def _figure_manifest_row(figure: ScientificReportFigureReference) -> Mapping[str, object]:
    return MappingProxyType({
        "figure_id": figure.figure_id,
        "figure_type": figure.figure_type,
        "relative_path": figure.relative_path,
        "checksum": figure.checksum,
        "status": figure.status.value,
    })


def _to_jsonable(value: object, cfg: ScientificReportSettings) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _to_jsonable(getattr(value, field.name), cfg) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(val, cfg) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [_to_jsonable(item, cfg) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    return value


def _format_value(value: object, cfg: ScientificReportSettings) -> str:
    if value is None:
        if cfg.missing_value_representation is ExportMissingValuePolicy.EMPTY:
            return ""
        if cfg.missing_value_representation is ExportMissingValuePolicy.NA:
            return "NA"
        if cfg.missing_value_representation is ExportMissingValuePolicy.DASH:
            return "-"
        return "null"
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nonfinite"
        return f"{value:.{cfg.numeric_formatting.default_decimal_places}g}"
    if isinstance(value, (tuple, list)):
        return "; ".join(_format_value(item, cfg) for item in value)
    return str(value)


def _missing_or_text(value: str | None, cfg: ScientificReportSettings) -> str:
    return _format_value(value, cfg)


def _join(values: object, cfg: ScientificReportSettings) -> str:
    if values is None:
        return _format_value(None, cfg)
    if isinstance(values, (tuple, list)):
        return ", ".join(_format_value(item, cfg) for item in values)
    return _format_value(values, cfg)


def _safe_md(text: str) -> str:
    return text.replace("|", "\\|").replace("<", "&lt;").replace(">", "&gt;").replace("\n", " ")


def _safe_link(text: str) -> str:
    return text.replace(" ", "%20").replace("(", "%28").replace(")", "%29")


def _latex_paragraph(text: str) -> str:
    return _latex_escape(text) + "\n"


def _latex_escape(value: str) -> str:
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


def _latex_label(value: str) -> str:
    return _sanitize(value).replace("_", "-")


def _relative_path(path: Path, cfg: ScientificReportSettings) -> str:
    if not cfg.use_relative_paths:
        return str(path)
    return os.path.relpath(path, Path(cfg.output_directory))


def _sanitize(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value))
    return cleaned.strip("_") or "artifact"


def _assert_conservative_narrative(text: str) -> None:
    lower = text.lower()
    for phrase in FORBIDDEN_AUTOMATIC_CLAIMS:
        if phrase.lower() in lower:
            raise ValueError(f"forbidden automatic claim present: {phrase}")


def _canonicalize(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _canonicalize(getattr(value, field.name)) for field in fields(value) if field.name not in {"figure", "axes"}}
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


def _stable_id(prefix: str, *parts: object) -> str:
    encoded = json.dumps(_canonicalize(parts), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return f"{prefix}-{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:16]}"


def _coerce_enum(value: object, enum_type: type[Enum]) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)  # type: ignore[call-arg]
    except ValueError as exc:
        raise ValueError(f"{enum_type.__name__} value is not recognized.") from exc


def _reason_tuple(values: Iterable[ScientificReportReason]) -> tuple[ScientificReportReason, ...]:
    return tuple(dict.fromkeys(value if isinstance(value, ScientificReportReason) else ScientificReportReason(value) for value in values))


def _unique_texts(values: Iterable[object]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(item) for item in values))


def _text_tuple(values: Iterable[object], name: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    result = tuple(str(item) for item in values)
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty.")
    for item in result:
        _text(item, name)
    return result


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")


def _optional_text(value: str | None, name: str) -> None:
    if value is not None:
        _text(value, name)


def _belllab_version() -> str:
    module = sys.modules.get("belllab")
    return str(getattr(module, "__version__", "0.1.0"))
