"""Public command-line interface for BellLab.

The CLI is intentionally a thin adapter over public BellLab APIs. It validates
inputs, maps command-line options to existing settings objects, preserves
structured status and provenance, and writes deterministic result bundles when
requested. It does not implement scientific estimators or reinterpret results.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping, Sequence
import copyreg
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum, IntEnum
import hashlib
import json
import logging
import os
from pathlib import Path
import pickle
import platform
import sys
from types import MappingProxyType
from typing import Any

try:  # Python 3.11+
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    tomllib = None  # type: ignore[assignment]

from belllab.experiment_pipeline import (
    EXPERIMENT_PIPELINE_STAGE_ORDER,
    ExperimentAnalysisResult,
    ExperimentAnalysisStatus,
    ExperimentDefinition,
    ExperimentDefinitionError,
    ExperimentInputError,
    ExperimentPipelineDependencyError,
    ExperimentPipelineSettings,
    ExperimentPipelineStage,
    ExperimentPrecomputedResultError,
    ExperimentRecordingDefinition,
    analyze_experiment,
    experiment_settings_fingerprint,
    resume_experiment_analysis,
    summarize_experiment_analysis,
    validate_experiment_definition,
)
from belllab.results_export import (
    BellLabExportSchemaVersion,
    ExportMissingValuePolicy,
    ExportNonfiniteValuePolicy,
    ExportOverwritePolicy,
    ExperimentExportResult,
    ResultsExportSettings,
    ResultsExportStatus,
    export_experiment_results,
    export_settings_fingerprint,
    summarize_experiment_export,
    validate_experiment_export,
)
from belllab.scientific_report import (
    ScientificReportResult,
    ScientificReportSchemaVersion,
    ScientificReportSettings,
    ScientificReportStatus,
    create_scientific_report,
    scientific_report_settings_fingerprint,
    summarize_scientific_report,
    validate_scientific_report,
)
from belllab.scientific_visualizations import (
    ScientificColorPolicy,
    ScientificFigureCollection,
    ScientificFigureType,
    ScientificVisualizationSettings,
    ScientificVisualizationStatus,
    create_experiment_visualizations,
    scientific_visualization_settings_fingerprint,
    summarize_scientific_visualizations,
)
from belllab.synthetic_validation import (
    SyntheticClippingMode,
    SyntheticNoiseModel,
    SyntheticValidationSettings,
    SyntheticValidationStatus,
    generate_synthetic_validation_scenario,
    run_synthetic_monte_carlo_validation,
    run_synthetic_validation_campaign,
    summarize_synthetic_validation,
    synthetic_validation_settings_fingerprint,
    validate_synthetic_scenario,
)


def _mapping_proxy_from_dict(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(value))


def _reduce_mapping_proxy(value: Mapping[str, object]) -> tuple[object, tuple[dict[str, object]]]:
    return (_mapping_proxy_from_dict, (dict(value),))


_MAPPING_PROXY_TYPE = type(MappingProxyType({}))
copyreg.pickle(_MAPPING_PROXY_TYPE, _reduce_mapping_proxy)


class BellLabCLICommand(str, Enum):
    """Public BellLab command identifiers."""

    ANALYZE = "analyze"
    EXPORT = "export"
    VISUALIZE = "visualize"
    REPORT = "report"
    VALIDATE_SYNTHETIC = "validate_synthetic"
    INSPECT = "inspect"
    VERSION = "version"


class BellLabCLIOutputFormat(str, Enum):
    """Supported CLI stdout formats."""

    TEXT = "text"
    JSON = "json"
    QUIET = "quiet"


class BellLabCLIExitCode(IntEnum):
    """Documented BellLab CLI exit codes."""

    COMPLETED = 0
    COMPLETED_WITH_RESERVATIONS = 1
    USAGE_OR_CONFIGURATION_INVALID = 2
    INPUT_INVALID = 3
    INSUFFICIENT_EVIDENCE = 4
    PARTIAL_EXECUTION = 5
    STAGE_FAILURE = 6
    INTERNAL_ERROR = 7
    ARTIFACT_VALIDATION_FAILED = 8
    REPORT_COMPILATION_FAILED = 9


class BellLabCLIError(RuntimeError):
    """Base class for expected CLI errors."""


class BellLabCLIConfigurationError(BellLabCLIError):
    """Raised for invalid CLI configuration or arguments."""


class BellLabCLIExecutionError(BellLabCLIError):
    """Raised for expected command execution failures."""


@dataclass(frozen=True, slots=True)
class BellLabCLISettings:
    """Resolved process-level CLI settings."""

    output_format: BellLabCLIOutputFormat = BellLabCLIOutputFormat.TEXT
    quiet: bool = False
    verbose: bool = False
    debug: bool = False
    log_file: str | Path | None = None
    log_level: str = "WARNING"
    redact_paths: bool = False
    dry_run: bool = False
    print_effective_config: bool = False
    configuration_fingerprint: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "output_format",
            _coerce_enum(self.output_format, BellLabCLIOutputFormat),
        )
        for name in ("quiet", "verbose", "debug", "redact_paths", "dry_run", "print_effective_config"):
            if not isinstance(getattr(self, name), bool):
                raise BellLabCLIConfigurationError(f"{name} must be a boolean.")
        if self.quiet:
            object.__setattr__(self, "output_format", BellLabCLIOutputFormat.QUIET)
        if self.log_file is not None:
            object.__setattr__(self, "log_file", Path(self.log_file))
        if self.log_level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise BellLabCLIConfigurationError("log_level is not recognized.")
        object.__setattr__(self, "log_level", self.log_level.upper())


@dataclass(frozen=True, slots=True)
class BellLabCLIResult:
    """Serializable result returned by :func:`run_cli`."""

    command: BellLabCLICommand | str
    exit_code: BellLabCLIExitCode | int
    status: str
    message: str
    analysis_id: str | None = None
    experiment_id: str | None = None
    artifact_paths: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()
    payload: Mapping[str, object] = MappingProxyType({})
    valid: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.command, BellLabCLICommand):
            command: BellLabCLICommand | str = self.command
        elif str(self.command) in _COMMAND_VALUES:
            command = BellLabCLICommand(str(self.command))
        else:
            command = str(self.command)
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "exit_code", _coerce_enum(self.exit_code, BellLabCLIExitCode))
        object.__setattr__(self, "artifact_paths", tuple(str(item) for item in self.artifact_paths))
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))
        object.__setattr__(self, "errors", tuple(str(item) for item in self.errors))
        object.__setattr__(self, "diagnostics", tuple(str(item) for item in self.diagnostics))
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))


_COMMAND_VALUES = {item.value for item in BellLabCLICommand}
CLI_BUNDLE_SCHEMA_VERSION = "1.0"
CLI_CONFIGURATION_SCHEMA_VERSION = "1.0"
DEFAULT_DYNAMIC_LABELS = ("pp", "p", "mf", "f", "ff")
_RESULT_BUNDLE_TYPES = {
    "analysis": ExperimentAnalysisResult,
    "export": ExperimentExportResult,
    "figures": ScientificFigureCollection,
    "report": ScientificReportResult,
}
_PIPELINE_FLAG_BY_STAGE = {
    ExperimentPipelineStage.LOAD: "run_loading",
    ExperimentPipelineStage.VALIDATE_INPUT: "run_input_validation",
    ExperimentPipelineStage.TEMPORAL: "run_temporal_analysis",
    ExperimentPipelineStage.GLOBAL_SPECTRUM: "run_global_spectrum",
    ExperimentPipelineStage.STFT: "run_stft",
    ExperimentPipelineStage.TRACKING: "run_tracking",
    ExperimentPipelineStage.PREIMPACT: "run_preimpact_analysis",
    ExperimentPipelineStage.EXCITATION: "run_excitation_characterization",
    ExperimentPipelineStage.MODAL_CANDIDATES: "run_modal_candidate_characterization",
    ExperimentPipelineStage.WITHIN_CONDITION: "run_within_condition_association",
    ExperimentPipelineStage.DYNAMIC_CONDITION_COMPARISON: "run_dynamic_condition_comparison",
    ExperimentPipelineStage.CROSS_CONDITION: "run_cross_condition_association",
    ExperimentPipelineStage.CANDIDATE_CHAINS: "run_candidate_chains",
    ExperimentPipelineStage.MODAL_HYPOTHESES: "run_modal_hypotheses",
    ExperimentPipelineStage.MODAL_PARAMETERS: "run_modal_parameter_estimation",
    ExperimentPipelineStage.MODAL_Q: "run_modal_q_estimation",
    ExperimentPipelineStage.MODAL_ENERGY_EXCHANGE: "run_modal_energy_exchange",
}
_STAGE_BY_VALUE = {stage.value: stage for stage in EXPERIMENT_PIPELINE_STAGE_ORDER}
_KNOWN_CONFIG_KEYS = {
    "cli",
    "experiment",
    "recordings",
    "pipeline",
    "export",
    "visualization",
    "report",
    "synthetic",
}
_KNOWN_CLI_KEYS = {
    "output_format",
    "quiet",
    "verbose",
    "debug",
    "log_file",
    "log_level",
    "redact_paths",
}


def build_cli_parser() -> argparse.ArgumentParser:
    """Build the public BellLab argument parser."""

    parser = argparse.ArgumentParser(
        prog="belllab",
        description=(
            "BellLab public CLI. Commands orchestrate existing BellLab APIs and "
            "do not add physical inference."
        ),
    )
    parser.add_argument("--version", action="store_true", help="Show BellLab version information and exit.")
    _add_common_options(parser)
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    analyze = subparsers.add_parser(
        "analyze",
        help="Analyze real experiment recordings through the public pipeline.",
        description="Run analyze_experiment on an explicit experiment definition.",
        epilog="Example: belllab analyze --config experiment.toml --save-result result.json",
    )
    _add_common_options(analyze)
    analyze.add_argument("--config", type=Path, help="JSON or TOML experiment configuration.")
    analyze.add_argument("--generate-config", type=Path, help="Write an example TOML configuration and exit.")
    analyze.add_argument("--output-dir", type=Path, help="Directory for saved analysis bundles.")
    analyze.add_argument("--until-stage", choices=tuple(_STAGE_BY_VALUE), help="Run only stages up to and including this stage.")
    analyze.add_argument("--skip-stage", action="append", default=None, choices=tuple(_STAGE_BY_VALUE), help="Disable one pipeline stage.")
    analyze.add_argument("--only-stage", action="append", default=None, choices=tuple(_STAGE_BY_VALUE), help="Run one stage plus required predecessors.")
    analyze.add_argument("--continue-after-failure", action="store_true", help="Record expected stage failures and continue.")
    analyze.add_argument("--fail-fast", action="store_true", help="Raise on invalid input instead of preserving partial results.")
    analyze.add_argument("--include-partial", action="store_true", help="Allow partial analysis results.")
    analyze.add_argument("--recording", action="append", default=None, metavar="LABEL=PATH", help="Quick recording definition; also supports LABEL:TAKE=PATH.")
    analyze.add_argument("--condition", action="append", default=None, help="Restrict dynamic labels to include in the definition.")
    analyze.add_argument("--channel", type=int, help="Explicit channel index for quick recordings.")
    analyze.add_argument("--start-offset", type=float, help="Start offset in seconds for quick recordings.")
    analyze.add_argument("--end-offset", type=float, help="End offset in seconds for quick recordings.")
    analyze.add_argument("--workers", type=int, help="Maximum worker count; execution remains deterministic.")
    analyze.add_argument("--save-result", nargs="?", const="", type=Path, help="Write a BellLab analysis bundle JSON.")
    analyze.add_argument("--result-format", choices=("belllab-json", "json"), default="belllab-json", help="Saved result bundle format.")
    analyze.add_argument("--resume-from", type=Path, help="Resume from a compatible BellLab analysis bundle.")
    analyze.add_argument("--print-effective-config", action="store_true", help="Print merged configuration and do not analyze.")
    analyze.set_defaults(handler=_command_analyze)

    export = subparsers.add_parser(
        "export",
        help="Export already computed analysis results.",
        description="Run export_experiment_results on an analysis bundle.",
        epilog="Example: belllab export --analysis result.json --json --csv --output-dir results",
    )
    _add_common_options(export)
    export.add_argument("--analysis", type=Path, required=True, help="BellLab analysis bundle JSON.")
    export.add_argument("--output-dir", type=Path, required=True, help="Output directory for export artifacts.")
    export.add_argument("--json", action="store_true", help="Write JSON artifacts.")
    export.add_argument("--csv", action="store_true", help="Write CSV tables.")
    export.add_argument("--latex", action="store_true", help="Write LaTeX tables.")
    export.add_argument("--markdown", action="store_true", help="Write Markdown summary.")
    export.add_argument("--manifest", action="store_true", help="Write export manifest.")
    export.add_argument("--overwrite", choices=tuple(item.value for item in ExportOverwritePolicy), default=ExportOverwritePolicy.ERROR.value)
    export.add_argument("--missing-value", choices=tuple(item.value for item in ExportMissingValuePolicy), default=ExportMissingValuePolicy.NULL.value)
    export.add_argument("--nonfinite-policy", choices=tuple(item.value for item in ExportNonfiniteValuePolicy), default=ExportNonfiniteValuePolicy.ERROR.value)
    export.add_argument("--precision", type=int, default=10)
    export.add_argument("--validate", action="store_true", help="Validate written export artifacts.")
    export.add_argument("--save-result", type=Path, help="Write an export result bundle.")
    export.set_defaults(handler=_command_export)

    visualize = subparsers.add_parser(
        "visualize",
        help="Create figures from an existing analysis result.",
        description="Run create_experiment_visualizations without recalculating scientific data.",
        epilog="Example: belllab visualize --analysis result.json --all --format png --format svg",
    )
    _add_common_options(visualize)
    visualize.add_argument("--analysis", type=Path, required=True)
    visualize.add_argument("--output-dir", type=Path, required=True)
    visualize.add_argument("--figure", action="append", default=None, choices=tuple(item.value for item in ScientificFigureType))
    visualize.add_argument("--all", action="store_true", help="Request all default figure types.")
    visualize.add_argument("--format", action="append", default=None, choices=("png", "svg", "pdf"))
    visualize.add_argument("--dpi", type=int, default=120)
    visualize.add_argument("--overwrite", choices=tuple(item.value for item in ExportOverwritePolicy), default=ExportOverwritePolicy.ERROR.value)
    visualize.add_argument("--show-rejected", action="store_true")
    visualize.add_argument("--show-inconclusive", action="store_true")
    visualize.add_argument("--show-ids", action="store_true")
    visualize.add_argument("--monochrome", action="store_true")
    visualize.add_argument("--save-result", type=Path, help="Write a figure collection bundle.")
    visualize.set_defaults(handler=_command_visualize)

    report = subparsers.add_parser(
        "report",
        help="Create a reproducible scientific report from existing artifacts.",
        description="Run create_scientific_report with optional export and figure bundles.",
        epilog="Example: belllab report --analysis result.json --markdown --latex --output-dir report",
    )
    _add_common_options(report)
    report.add_argument("--analysis", type=Path, required=True)
    report.add_argument("--export-result", type=Path)
    report.add_argument("--figure-collection", type=Path)
    report.add_argument("--output-dir", type=Path, required=True)
    report.add_argument("--markdown", action="store_true")
    report.add_argument("--latex", action="store_true")
    report.add_argument("--pdf", action="store_true")
    report.add_argument("--title")
    report.add_argument("--subtitle")
    report.add_argument("--author", action="append", default=None)
    report.add_argument("--affiliation", action="append", default=None)
    report.add_argument("--language", choices=("pt-BR", "en"), default="pt-BR")
    report.add_argument("--compiler", choices=("auto", "latexmk", "tectonic", "pdflatex", "lualatex", "xelatex"), default="auto")
    report.add_argument("--overwrite", choices=tuple(item.value for item in ExportOverwritePolicy), default=ExportOverwritePolicy.ERROR.value)
    report.add_argument("--include-invalid", action="store_true")
    report.add_argument("--include-inconclusive", action="store_true")
    report.add_argument("--save-result", type=Path, help="Write a report result bundle.")
    report.set_defaults(handler=_command_report)

    synthetic = subparsers.add_parser(
        "validate-synthetic",
        help="Run controlled synthetic validation scenarios.",
        description="Validate BellLab behavior on known synthetic truth without changing estimators.",
        epilog="Example: belllab validate-synthetic --scenario single_ideal --trials 3 --seed 123",
    )
    _add_common_options(synthetic)
    synthetic.add_argument("--scenario", action="append", default=None)
    synthetic.add_argument("--all-scenarios", action="store_true")
    synthetic.add_argument("--trials", type=int, default=1)
    synthetic.add_argument("--seed", type=int, default=0)
    synthetic.add_argument("--snr-db", type=float)
    synthetic.add_argument("--clipping", choices=tuple(item.value for item in SyntheticClippingMode), default=SyntheticClippingMode.NONE.value)
    synthetic.add_argument("--sample-rate", type=int, default=8000)
    synthetic.add_argument("--duration", type=float, default=8.0)
    synthetic.add_argument("--output", type=Path)
    synthetic.set_defaults(handler=_command_validate_synthetic)

    inspect = subparsers.add_parser(
        "inspect",
        help="Inspect a BellLab result, manifest or configuration JSON/TOML.",
        description="Read and summarize a BellLab artifact without modifying it.",
        epilog="Example: belllab inspect result.json --validate --show-ids",
    )
    _add_common_options(inspect)
    inspect.add_argument("path", type=Path)
    inspect.add_argument("--summary", action="store_true")
    inspect.add_argument("--validate", action="store_true")
    inspect.add_argument("--show-ids", action="store_true")
    inspect.add_argument("--show-diagnostics", action="store_true")
    inspect.set_defaults(handler=_command_inspect)

    version = subparsers.add_parser(
        "version",
        help="Show BellLab and schema versions.",
        description="Show canonical BellLab, export schema and report schema versions.",
    )
    _add_common_options(version)
    version.set_defaults(handler=_command_version)
    return parser


def run_cli(argv: Sequence[str]) -> BellLabCLIResult:
    """Run the BellLab CLI in-process and return a structured result."""

    original = tuple(argv)
    parser = build_cli_parser()
    try:
        namespace = parser.parse_args(tuple(original))
    except SystemExit as exc:
        code = int(exc.code or 0)
        status = "help" if code == 0 else "usage_error"
        return BellLabCLIResult(
            command=getattr(exc, "command", "unknown"),
            exit_code=BellLabCLIExitCode.COMPLETED if code == 0 else BellLabCLIExitCode.USAGE_OR_CONFIGURATION_INVALID,
            status=status,
            message="Argument parsing completed." if code == 0 else "Invalid command-line usage.",
            valid=code == 0,
        )
    if getattr(namespace, "version", False) and namespace.command is None:
        namespace.command = "version"
        namespace.handler = _command_version
    if namespace.command is None:
        return BellLabCLIResult(
            command="unknown",
            exit_code=BellLabCLIExitCode.USAGE_OR_CONFIGURATION_INVALID,
            status="usage_error",
            message="No command supplied. Use --help to list commands.",
            valid=False,
            errors=("missing_command",),
        )
    _configure_logging(namespace)
    try:
        result = namespace.handler(namespace)
        return result
    except BellLabCLIConfigurationError as exc:
        return BellLabCLIResult(
            command=_namespace_command(namespace),
            exit_code=BellLabCLIExitCode.USAGE_OR_CONFIGURATION_INVALID,
            status="invalid_configuration",
            message=str(exc),
            valid=False,
            errors=(str(exc),),
        )
    except (ExperimentDefinitionError, ExperimentPipelineDependencyError, ExperimentPrecomputedResultError) as exc:
        return BellLabCLIResult(
            command=_namespace_command(namespace),
            exit_code=BellLabCLIExitCode.USAGE_OR_CONFIGURATION_INVALID,
            status="invalid_configuration",
            message=str(exc),
            valid=False,
            errors=(str(exc),),
        )
    except ExperimentInputError as exc:
        return BellLabCLIResult(
            command=_namespace_command(namespace),
            exit_code=BellLabCLIExitCode.INPUT_INVALID,
            status="invalid_input",
            message=str(exc),
            valid=False,
            errors=(str(exc),),
        )
    except ValueError as exc:
        return BellLabCLIResult(
            command=_namespace_command(namespace),
            exit_code=BellLabCLIExitCode.USAGE_OR_CONFIGURATION_INVALID,
            status="invalid_configuration",
            message=str(exc),
            valid=False,
            errors=(str(exc),),
        )
    except BellLabCLIExecutionError as exc:
        return BellLabCLIResult(
            command=_namespace_command(namespace),
            exit_code=BellLabCLIExitCode.STAGE_FAILURE,
            status="execution_failed",
            message=str(exc),
            valid=False,
            errors=(str(exc),),
        )
    except Exception as exc:  # unexpected errors are reclassified explicitly
        logging.getLogger(__name__).debug("unexpected CLI error", exc_info=True)
        return BellLabCLIResult(
            command=_namespace_command(namespace),
            exit_code=BellLabCLIExitCode.INTERNAL_ERROR,
            status="internal_error",
            message=f"{exc.__class__.__name__}: {exc}",
            valid=False,
            errors=(f"{exc.__class__.__name__}: {exc}",),
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point used by ``belllab`` and ``python -m belllab``."""

    result = run_cli(tuple(sys.argv[1:] if argv is None else argv))
    if result.status == "help":
        return int(result.exit_code)
    output_format = _result_output_format(result)
    if output_format is BellLabCLIOutputFormat.JSON:
        print(format_cli_json_output(result))
    elif output_format is BellLabCLIOutputFormat.TEXT:
        text = format_cli_text_output(result)
        if text:
            print(text)
    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    for error in result.errors:
        if result.exit_code != BellLabCLIExitCode.COMPLETED:
            print(f"error: {error}", file=sys.stderr)
    return int(result.exit_code)


def load_cli_configuration(path: str | Path) -> Mapping[str, object]:
    """Load a JSON or TOML CLI configuration without mutating it."""

    cfg_path = Path(path).expanduser()
    if not cfg_path.is_file():
        raise BellLabCLIConfigurationError(f"configuration file not found: {cfg_path}")
    try:
        if cfg_path.suffix.lower() == ".json":
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
        elif cfg_path.suffix.lower() == ".toml":
            raw = cfg_path.read_text(encoding="utf-8")
            data = tomllib.loads(raw) if tomllib is not None else _parse_toml_subset(raw)
        else:
            raise BellLabCLIConfigurationError("configuration must be JSON or TOML.")
    except json.JSONDecodeError as exc:
        raise BellLabCLIConfigurationError(f"invalid JSON configuration: {exc}") from exc
    except Exception as exc:
        if isinstance(exc, BellLabCLIConfigurationError):
            raise
        raise BellLabCLIConfigurationError(f"invalid configuration: {exc}") from exc
    if not isinstance(data, Mapping):
        raise BellLabCLIConfigurationError("configuration root must be an object.")
    validate_cli_configuration(data)
    return MappingProxyType(_deepcopy_jsonable(dict(data)))


def validate_cli_configuration(configuration: Mapping[str, object]) -> bool:
    """Validate supported CLI configuration keys."""

    unknown = sorted(set(configuration) - _KNOWN_CONFIG_KEYS)
    if unknown:
        raise BellLabCLIConfigurationError(f"unknown configuration key: {unknown[0]}")
    if "cli" in configuration:
        _validate_mapping_keys(configuration["cli"], _KNOWN_CLI_KEYS, "cli")
    if "experiment" in configuration:
        _validate_mapping_keys(configuration["experiment"], _EXPERIMENT_KEYS, "experiment")
    if "recordings" in configuration:
        recordings = configuration["recordings"]
        if not isinstance(recordings, Sequence) or isinstance(recordings, (str, bytes)):
            raise BellLabCLIConfigurationError("recordings must be a list.")
        for recording in recordings:
            _validate_mapping_keys(recording, _RECORDING_KEYS, "recordings")
    if "pipeline" in configuration:
        _validate_mapping_keys(configuration["pipeline"], _PIPELINE_KEYS, "pipeline")
    if "export" in configuration:
        _validate_mapping_keys(configuration["export"], _EXPORT_KEYS, "export")
    if "visualization" in configuration:
        _validate_mapping_keys(configuration["visualization"], _VISUALIZATION_KEYS, "visualization")
    if "report" in configuration:
        _validate_mapping_keys(configuration["report"], _REPORT_KEYS, "report")
    if "synthetic" in configuration:
        _validate_mapping_keys(configuration["synthetic"], _SYNTHETIC_KEYS, "synthetic")
    return True


def merge_cli_configuration(
    defaults: Mapping[str, object] | None,
    file_configuration: Mapping[str, object] | None,
    command_line_configuration: Mapping[str, object] | None,
) -> Mapping[str, object]:
    """Merge CLI configuration using defaults < file < CLI precedence."""

    merged = _deep_merge(defaults or {}, file_configuration or {})
    merged = _deep_merge(merged, command_line_configuration or {})
    validate_cli_configuration(merged)
    return MappingProxyType(_deepcopy_jsonable(merged))


def serialize_cli_configuration(configuration: Mapping[str, object]) -> Mapping[str, object]:
    """Return a deterministic serializable configuration payload."""

    validate_cli_configuration(configuration)
    return MappingProxyType(
        {
            "schema_version": CLI_CONFIGURATION_SCHEMA_VERSION,
            "configuration": _canonicalize(configuration),
            "fingerprint": cli_settings_fingerprint(configuration),
        }
    )


def write_cli_configuration(path: str | Path, configuration: Mapping[str, object] | None = None) -> Path:
    """Write a deterministic example CLI configuration."""

    output = Path(path).expanduser()
    payload = configuration or _example_configuration()
    validate_cli_configuration(payload)
    if output.suffix.lower() == ".json":
        output.write_text(json.dumps(_canonicalize(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    elif output.suffix.lower() == ".toml":
        output.write_text(_toml_example_text(payload), encoding="utf-8")
    else:
        raise BellLabCLIConfigurationError("generated configuration must be .json or .toml.")
    return output


def cli_settings_fingerprint(configuration: Mapping[str, object] | BellLabCLISettings | None = None) -> str:
    """Return a deterministic fingerprint for CLI settings/configuration."""

    payload = _canonicalize(configuration or BellLabCLISettings())
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def map_result_status_to_exit_code(status: object) -> BellLabCLIExitCode:
    """Map BellLab scientific status values to documented CLI exit codes."""

    value = status.value if isinstance(status, Enum) else str(status)
    if value in {"completed", "created", "passed"}:
        return BellLabCLIExitCode.COMPLETED
    if value in {"completed_with_reservations", "created_with_reservations", "passed_with_reservations"}:
        return BellLabCLIExitCode.COMPLETED_WITH_RESERVATIONS
    if value in {"partial"}:
        return BellLabCLIExitCode.PARTIAL_EXECUTION
    if value in {"insufficient_evidence", "inconclusive", "not_supported"}:
        return BellLabCLIExitCode.INSUFFICIENT_EVIDENCE
    if value in {"invalid_input", "invalid_scenario"}:
        return BellLabCLIExitCode.INPUT_INVALID
    if value in {"compilation_failed"}:
        return BellLabCLIExitCode.REPORT_COMPILATION_FAILED
    if value in {"failed", "pipeline_error"}:
        return BellLabCLIExitCode.STAGE_FAILURE
    return BellLabCLIExitCode.INTERNAL_ERROR


def format_cli_text_output(result: BellLabCLIResult) -> str:
    """Format a stable human-readable CLI result."""

    if _result_output_format(result) is BellLabCLIOutputFormat.QUIET:
        return ""
    if result.command is BellLabCLICommand.VERSION:
        lines = [result.message]
        for key in ("export_schema_version", "report_schema_version", "cli_schema_version", "python", "platform"):
            if key in result.payload:
                label = "CLI Schema Version" if key == "cli_schema_version" else key.replace("_", " ").title()
                lines.append(f"{label}: {result.payload[key]}")
        return "\n".join(lines)
    lines = [
        f"BellLab {result.command.value if isinstance(result.command, BellLabCLICommand) else result.command}: {result.status}",
        result.message,
    ]
    if result.experiment_id:
        lines.append(f"Experiment ID: {result.experiment_id}")
    if result.analysis_id:
        lines.append(f"Analysis ID: {result.analysis_id}")
    if result.artifact_paths:
        lines.append("Artifacts:")
        lines.extend(f"- {_display_path(path, result.payload)}" for path in result.artifact_paths)
    if result.warnings:
        lines.append(f"Warnings: {len(result.warnings)}")
    if result.errors:
        lines.append(f"Errors: {len(result.errors)}")
    if result.diagnostics:
        lines.append(f"Diagnostics: {len(result.diagnostics)}")
    return "\n".join(lines)


def format_cli_json_output(result: BellLabCLIResult) -> str:
    """Format a CLI result as JSON-only stdout content."""

    return json.dumps(_to_jsonable(result), sort_keys=True, ensure_ascii=False, allow_nan=False)


def parse_cli_recording_spec(
    spec: str,
    *,
    channel: int | None = None,
    start_offset_s: float | None = None,
    end_offset_s: float | None = None,
    base_directory: str | Path | None = None,
) -> ExperimentRecordingDefinition:
    """Parse ``LABEL=PATH`` or ``LABEL:TAKE=PATH`` into a recording definition."""

    if "=" not in spec:
        raise BellLabCLIConfigurationError("recording spec must use LABEL=PATH.")
    left, raw_path = spec.split("=", 1)
    if not left or not raw_path:
        raise BellLabCLIConfigurationError("recording spec label and path must be non-empty.")
    if ":" in left:
        label, take_text = left.split(":", 1)
        try:
            take = int(take_text)
        except ValueError as exc:
            raise BellLabCLIConfigurationError("recording take must be an integer.") from exc
    else:
        label = left
        take = 0
    path = Path(raw_path).expanduser()
    if base_directory is not None and not path.is_absolute():
        path = Path(base_directory) / path
    recording_id = f"{label}_take_{take}"
    return ExperimentRecordingDefinition(
        file_path=path,
        dynamic_label=label,
        recording_id=recording_id,
        take_index=take,
        channel=channel,
        start_offset_s=start_offset_s,
        end_offset_s=end_offset_s,
    )


def load_serialized_analysis_result(path: str | Path) -> ExperimentAnalysisResult:
    """Load a BellLab analysis bundle written by the CLI."""

    return _load_cli_bundle(path, "analysis")


def load_serialized_export_result(path: str | Path) -> ExperimentExportResult:
    """Load a BellLab export bundle written by the CLI."""

    return _load_cli_bundle(path, "export")


def load_serialized_figure_collection(path: str | Path) -> ScientificFigureCollection:
    """Load a BellLab figure collection bundle written by the CLI."""

    return _load_cli_bundle(path, "figures")


def load_serialized_report_result(path: str | Path) -> ScientificReportResult:
    """Load a BellLab report bundle written by the CLI."""

    return _load_cli_bundle(path, "report")


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-format", choices=tuple(item.value for item in BellLabCLIOutputFormat), default=None, help="CLI output format: text, json or quiet.")
    parser.add_argument("--quiet", action="store_true", help="Suppress normal stdout while preserving exit codes.")
    parser.add_argument("--verbose", action="store_true", help="Enable informational logs on stderr.")
    parser.add_argument("--debug", action="store_true", help="Enable debug diagnostics for unexpected errors.")
    parser.add_argument("--log-file", type=Path, help="Optional log file.")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"), default=None)
    parser.add_argument("--redact-paths", action="store_true", help="Redact absolute paths from CLI output.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and show planned work without writing files.")


def _command_analyze(args: argparse.Namespace) -> BellLabCLIResult:
    if args.generate_config:
        if args.dry_run:
            return _cli_result(
                BellLabCLICommand.ANALYZE,
                BellLabCLIExitCode.COMPLETED,
                "dry_run",
                f"Would write example configuration to {args.generate_config}.",
                args,
                payload={"planned_artifacts": (str(args.generate_config),)},
                diagnostics=("dry_run_no_files_written",),
            )
        path = write_cli_configuration(args.generate_config)
        return _cli_result(
            BellLabCLICommand.ANALYZE,
            BellLabCLIExitCode.COMPLETED,
            "configuration_created",
            "Example configuration written.",
            args,
            artifact_paths=(str(path),),
        )

    file_cfg, base_dir = _config_from_args(args)
    cli_cfg = _analyze_cli_overrides(args)
    effective = merge_cli_configuration({}, file_cfg, cli_cfg)
    cli_settings = _cli_settings_from_args(args, effective)
    if args.print_effective_config:
        payload = serialize_cli_configuration(effective)
        return _cli_result(
            BellLabCLICommand.ANALYZE,
            BellLabCLIExitCode.COMPLETED,
            "effective_configuration",
            "Effective configuration resolved; analysis not executed.",
            args,
            payload=payload,
            diagnostics=("print_effective_config_skips_analysis",),
        )
    experiment = _experiment_from_configuration(effective, base_dir=base_dir)
    settings = _pipeline_settings_from_configuration(effective)
    if args.dry_run:
        validation = validate_experiment_definition(experiment, settings)
        planned = _planned_analysis_artifacts(args, experiment)
        return _cli_result(
            BellLabCLICommand.ANALYZE,
            BellLabCLIExitCode.COMPLETED if validation.valid else BellLabCLIExitCode.INPUT_INVALID,
            "dry_run" if validation.valid else "invalid_input",
            "Dry run completed; no analysis executed and no files written.",
            args,
            experiment_id=experiment.experiment_id,
            artifact_paths=planned,
            payload={
                "validation": _to_jsonable(validation),
                "settings_fingerprint": experiment_settings_fingerprint(settings),
                "effective_configuration": _canonicalize(effective),
            },
            warnings=tuple(validation.reasons),
            diagnostics=("dry_run_no_heavy_analysis", "dry_run_no_files_written"),
            valid=validation.valid,
        )

    previous = load_serialized_analysis_result(args.resume_from) if args.resume_from else None
    if previous is not None:
        result = resume_experiment_analysis(previous, experiment=experiment, settings=replace(settings, reuse_precomputed_results=True))
    else:
        result = analyze_experiment(experiment, settings)
    artifact_paths: tuple[str, ...] = ()
    if args.save_result is not None:
        destination = _analysis_save_path(args.save_result, result, args.output_dir)
        artifact_paths = (_write_cli_bundle(destination, "analysis", result, summary=summarize_experiment_analysis(result)),)
    summary = summarize_experiment_analysis(result)
    exit_code = map_result_status_to_exit_code(result.status)
    return _cli_result(
        BellLabCLICommand.ANALYZE,
        exit_code,
        result.status.value,
        _scientific_status_message("analysis", result.status.value),
        args,
        analysis_id=result.analysis_id,
        experiment_id=result.experiment.experiment_id,
        artifact_paths=artifact_paths,
        warnings=_analysis_warnings(result),
        diagnostics=result.diagnostics,
        payload={
            "summary": summary,
            "settings_fingerprint": experiment_settings_fingerprint(settings),
            "configuration_fingerprint": cli_settings.configuration_fingerprint,
        },
        valid=result.valid,
    )


def _command_export(args: argparse.Namespace) -> BellLabCLIResult:
    analysis = load_serialized_analysis_result(args.analysis)
    if args.dry_run:
        planned = _planned_export_artifacts(analysis, args)
        return _cli_result(
            BellLabCLICommand.EXPORT,
            BellLabCLIExitCode.COMPLETED,
            "dry_run",
            "Dry run completed; export artifacts not written.",
            args,
            analysis_id=analysis.analysis_id,
            experiment_id=analysis.experiment.experiment_id,
            artifact_paths=planned,
            diagnostics=("dry_run_no_export_written", "analysis_not_recalculated"),
            payload={"planned_artifacts": planned},
        )
    settings = _export_settings_from_args(args)
    result = export_experiment_results(analysis, settings)
    validation_payload: Mapping[str, object] | None = None
    validation_exit = None
    warnings = tuple(result.diagnostics)
    if args.validate:
        validation = validate_experiment_export(result)
        validation_payload = _to_jsonable(validation)
        if not validation.valid:
            validation_exit = BellLabCLIExitCode.ARTIFACT_VALIDATION_FAILED
            warnings += tuple(validation.diagnostics)
    artifact_paths = tuple(artifact.path for artifact in result.artifacts if artifact.path)
    if args.save_result:
        artifact_paths += (_write_cli_bundle(args.save_result, "export", result, summary=summarize_experiment_export(result)),)
    exit_code = validation_exit or map_result_status_to_exit_code(result.status)
    return _cli_result(
        BellLabCLICommand.EXPORT,
        exit_code,
        result.status.value,
        _scientific_status_message("export", result.status.value),
        args,
        analysis_id=result.analysis_id,
        experiment_id=result.experiment_id,
        artifact_paths=artifact_paths,
        warnings=warnings,
        diagnostics=result.diagnostics + ("analysis_not_recalculated",),
        payload={
            "summary": summarize_experiment_export(result),
            "validation": validation_payload,
            "settings_fingerprint": export_settings_fingerprint(settings),
        },
        valid=result.valid and validation_exit is None,
    )


def _command_visualize(args: argparse.Namespace) -> BellLabCLIResult:
    analysis = load_serialized_analysis_result(args.analysis)
    if args.dry_run:
        planned = _planned_visualization_artifacts(analysis, args)
        return _cli_result(
            BellLabCLICommand.VISUALIZE,
            BellLabCLIExitCode.COMPLETED,
            "dry_run",
            "Dry run completed; figures not rendered or written.",
            args,
            analysis_id=analysis.analysis_id,
            experiment_id=analysis.experiment.experiment_id,
            artifact_paths=planned,
            diagnostics=("dry_run_no_figures_written", "no_fft_stft_tracking_recalculated"),
            payload={"planned_artifacts": planned},
        )
    settings = _visualization_settings_from_args(args)
    result = create_experiment_visualizations(analysis, settings)
    artifact_paths = tuple(artifact.path for artifact in result.artifacts if artifact.path)
    if args.save_result:
        artifact_paths += (_write_cli_bundle(args.save_result, "figures", result, summary=summarize_scientific_visualizations(result)),)
    exit_code = map_result_status_to_exit_code(result.status)
    return _cli_result(
        BellLabCLICommand.VISUALIZE,
        exit_code,
        result.status.value,
        _scientific_status_message("visualization", result.status.value),
        args,
        analysis_id=result.analysis_id,
        experiment_id=result.experiment_id,
        artifact_paths=artifact_paths,
        warnings=result.diagnostics if result.status is not ScientificVisualizationStatus.CREATED else (),
        diagnostics=result.diagnostics + ("no_fft_stft_tracking_recalculated",),
        payload={
            "summary": summarize_scientific_visualizations(result),
            "settings_fingerprint": scientific_visualization_settings_fingerprint(settings),
        },
        valid=result.valid,
    )


def _command_report(args: argparse.Namespace) -> BellLabCLIResult:
    analysis = load_serialized_analysis_result(args.analysis)
    export = load_serialized_export_result(args.export_result) if args.export_result else None
    figures = load_serialized_figure_collection(args.figure_collection) if args.figure_collection else None
    if args.dry_run:
        planned = _planned_report_artifacts(analysis, args)
        return _cli_result(
            BellLabCLICommand.REPORT,
            BellLabCLIExitCode.COMPLETED,
            "dry_run",
            "Dry run completed; report artifacts not written.",
            args,
            analysis_id=analysis.analysis_id,
            experiment_id=analysis.experiment.experiment_id,
            artifact_paths=planned,
            diagnostics=("dry_run_no_report_written", "no_scientific_analysis_recalculated", "no_figures_regenerated"),
            payload={"planned_artifacts": planned},
        )
    settings = _report_settings_from_args(args)
    result = create_scientific_report(analysis, export, figures, settings)
    validation = validate_scientific_report(result)
    artifact_paths = tuple(artifact.path for artifact in result.artifacts if artifact.path)
    if args.save_result:
        artifact_paths += (_write_cli_bundle(args.save_result, "report", result, summary=summarize_scientific_report(result)),)
    exit_code = map_result_status_to_exit_code(result.status)
    if result.status is ScientificReportStatus.COMPILATION_FAILED:
        exit_code = BellLabCLIExitCode.REPORT_COMPILATION_FAILED
    if not validation.valid and result.valid:
        exit_code = BellLabCLIExitCode.ARTIFACT_VALIDATION_FAILED
    return _cli_result(
        BellLabCLICommand.REPORT,
        exit_code,
        result.status.value,
        _scientific_status_message("report", result.status.value),
        args,
        analysis_id=result.analysis_id,
        experiment_id=result.experiment_id,
        artifact_paths=artifact_paths,
        warnings=result.diagnostics + validation.diagnostics if result.requires_review or not validation.valid else (),
        diagnostics=result.diagnostics + ("no_scientific_analysis_recalculated", "no_figures_regenerated"),
        payload={
            "summary": summarize_scientific_report(result),
            "validation": _to_jsonable(validation),
            "settings_fingerprint": scientific_report_settings_fingerprint(settings),
        },
        valid=result.valid and validation.valid,
    )


def _command_validate_synthetic(args: argparse.Namespace) -> BellLabCLIResult:
    settings = SyntheticValidationSettings(
        sample_rate_hz=args.sample_rate,
        duration_s=args.duration,
        random_seed=args.seed,
        trial_count=max(1, args.trials),
        signal_to_noise_ratio_db=args.snr_db,
        noise_model=SyntheticNoiseModel.WHITE if args.snr_db is not None else SyntheticNoiseModel.NONE,
        clipping_mode=args.clipping,
        clipping_threshold=0.7 if args.clipping != SyntheticClippingMode.NONE.value else None,
    )
    names = tuple(args.scenario or ("single_ideal",))
    if args.all_scenarios:
        names = ()
    if args.dry_run:
        return _cli_result(
            BellLabCLICommand.VALIDATE_SYNTHETIC,
            BellLabCLIExitCode.COMPLETED,
            "dry_run",
            "Dry run completed; synthetic pipeline not executed.",
            args,
            diagnostics=("dry_run_no_synthetic_pipeline_run",),
            payload={"scenarios": names or ("all",), "settings_fingerprint": synthetic_validation_settings_fingerprint(settings)},
        )
    if args.all_scenarios:
        result = run_synthetic_validation_campaign(settings=settings)
    elif args.trials > 1 and len(names) == 1:
        scenario = generate_synthetic_validation_scenario(names[0], settings)
        result = run_synthetic_monte_carlo_validation(scenario, settings)
    elif len(names) == 1:
        scenario = generate_synthetic_validation_scenario(names[0], settings)
        result = validate_synthetic_scenario(scenario, settings)
    else:
        scenarios = tuple(generate_synthetic_validation_scenario(name, settings) for name in names)
        result = run_synthetic_validation_campaign(scenarios, settings)
    summary = summarize_synthetic_validation(result)
    artifact_paths: tuple[str, ...] = ()
    if args.output:
        path = args.output
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_to_jsonable(summary), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        artifact_paths = (str(path),)
    status = getattr(result, "status", None) or ("passed" if getattr(result, "valid", False) else "failed")
    exit_code = map_result_status_to_exit_code(status)
    return _cli_result(
        BellLabCLICommand.VALIDATE_SYNTHETIC,
        exit_code,
        status.value if isinstance(status, Enum) else str(status),
        _scientific_status_message("synthetic validation", status.value if isinstance(status, Enum) else str(status)),
        args,
        artifact_paths=artifact_paths,
        warnings=tuple(getattr(result, "diagnostics", ())) if getattr(result, "requires_review", False) else (),
        diagnostics=tuple(getattr(result, "diagnostics", ())) + ("truth_not_used_to_calibrate_thresholds",),
        payload={"summary": summary, "settings_fingerprint": synthetic_validation_settings_fingerprint(settings)},
        valid=bool(getattr(result, "valid", False)),
    )


def _command_inspect(args: argparse.Namespace) -> BellLabCLIResult:
    info = _inspect_path(args.path)
    warnings: tuple[str, ...] = ()
    exit_code = BellLabCLIExitCode.COMPLETED
    if args.validate:
        validation = _validate_inspected(info, args.path)
        info = {**info, "validation": validation}
        if validation.get("valid") is False:
            exit_code = BellLabCLIExitCode.ARTIFACT_VALIDATION_FAILED
            warnings = tuple(validation.get("diagnostics", ()))
    return _cli_result(
        BellLabCLICommand.INSPECT,
        exit_code,
        str(info.get("status", "inspected")),
        "Artifact inspected without modification.",
        args,
        analysis_id=info.get("analysis_id") if isinstance(info.get("analysis_id"), str) else None,
        experiment_id=info.get("experiment_id") if isinstance(info.get("experiment_id"), str) else None,
        warnings=warnings,
        diagnostics=("inspect_does_not_modify_files",),
        payload=info,
        valid=exit_code is BellLabCLIExitCode.COMPLETED,
    )


def _command_version(args: argparse.Namespace) -> BellLabCLIResult:
    payload = {
        "belllab_version": _belllab_version(),
        "export_schema_version": BellLabExportSchemaVersion.V1_0.value,
        "report_schema_version": ScientificReportSchemaVersion.V1_0.value,
        "cli_schema_version": CLI_BUNDLE_SCHEMA_VERSION,
    }
    if getattr(args, "verbose", False):
        payload.update({"python": sys.version.split()[0], "platform": platform.platform()})
    return _cli_result(
        BellLabCLICommand.VERSION,
        BellLabCLIExitCode.COMPLETED,
        "version",
        f"BellLab {_belllab_version()}",
        args,
        payload=payload,
    )


def _config_from_args(args: argparse.Namespace) -> tuple[Mapping[str, object], Path | None]:
    if not getattr(args, "config", None):
        return MappingProxyType({}), None
    path = Path(args.config).expanduser()
    return load_cli_configuration(path), path.parent


def _analyze_cli_overrides(args: argparse.Namespace) -> Mapping[str, object]:
    overrides: dict[str, object] = {}
    if args.recording:
        overrides["recordings"] = tuple(
            _recording_to_config(parse_cli_recording_spec(
                item,
                channel=args.channel,
                start_offset_s=args.start_offset,
                end_offset_s=args.end_offset,
            ))
            for item in args.recording
        )
        overrides["experiment"] = {"name": "BellLab CLI experiment"}
    if args.condition:
        overrides.setdefault("experiment", {})
        overrides["experiment"]["dynamic_labels"] = tuple(args.condition)  # type: ignore[index]
    pipeline: dict[str, object] = {}
    if args.workers is not None:
        pipeline["maximum_worker_count"] = args.workers
        pipeline["parallel_execution_policy"] = "recordings" if args.workers > 1 else "sequential"
    if args.continue_after_failure:
        pipeline["continue_after_stage_failure"] = True
        pipeline["stage_error_policy"] = "record_and_continue"
    if args.fail_fast:
        pipeline["fail_fast_on_invalid_input"] = True
        pipeline["stage_error_policy"] = "raise"
    if args.include_partial:
        pipeline["allow_partial_results"] = True
    if args.until_stage:
        pipeline.update(_stage_flags_until(args.until_stage))
    if args.only_stage:
        pipeline.update(_stage_flags_only(tuple(args.only_stage)))
    if args.skip_stage:
        for stage_name in tuple(args.skip_stage):
            stage = _STAGE_BY_VALUE[stage_name]
            if stage in _PIPELINE_FLAG_BY_STAGE:
                pipeline[_PIPELINE_FLAG_BY_STAGE[stage]] = False
    if pipeline:
        overrides["pipeline"] = pipeline
    return MappingProxyType(overrides)


def _experiment_from_configuration(configuration: Mapping[str, object], *, base_dir: Path | None) -> ExperimentDefinition:
    experiment_cfg = dict(configuration.get("experiment", {}))
    recordings_cfg = configuration.get("recordings", ())
    if not recordings_cfg:
        raise BellLabCLIConfigurationError("experiment requires recordings from --recording or config file.")
    recordings = tuple(_recording_from_config(item, base_dir=base_dir) for item in recordings_cfg)  # type: ignore[arg-type]
    return ExperimentDefinition(
        name=str(experiment_cfg.get("name", "BellLab CLI experiment")),
        recordings=recordings,
        experiment_id=_optional_str(experiment_cfg.get("experiment_id")),
        description=_optional_str(experiment_cfg.get("description")),
        specimen_id=_optional_str(experiment_cfg.get("specimen_id")),
        instrument_type=_optional_str(experiment_cfg.get("instrument_type")),
        location=_optional_str(experiment_cfg.get("location")),
        operator=_optional_str(experiment_cfg.get("operator")),
        acquisition_date=_optional_str(experiment_cfg.get("acquisition_date")),
        dynamic_labels=tuple(experiment_cfg.get("dynamic_labels", DEFAULT_DYNAMIC_LABELS)),
        reference_recording_id=_optional_str(experiment_cfg.get("reference_recording_id")),
        microphone=_optional_str(experiment_cfg.get("microphone")),
        audio_interface=_optional_str(experiment_cfg.get("audio_interface")),
        acquisition_notes=_optional_str(experiment_cfg.get("acquisition_notes")),
        environment_notes=_optional_str(experiment_cfg.get("environment_notes")),
        expected_sample_rate_hz=experiment_cfg.get("expected_sample_rate_hz"),
        expected_channel_count=experiment_cfg.get("expected_channel_count"),
        metadata=experiment_cfg.get("metadata", {}),
    )


def _recording_from_config(value: object, *, base_dir: Path | None) -> ExperimentRecordingDefinition:
    if not isinstance(value, Mapping):
        raise BellLabCLIConfigurationError("recording entry must be an object.")
    path = Path(str(value["file_path"])).expanduser()
    if base_dir is not None and not path.is_absolute():
        path = base_dir / path
    return ExperimentRecordingDefinition(
        file_path=path,
        dynamic_label=str(value["dynamic_label"]),
        recording_id=_optional_str(value.get("recording_id")),
        take_index=int(value.get("take_index", 0)),
        replicate_group=_optional_str(value.get("replicate_group")),
        specimen_id=_optional_str(value.get("specimen_id")),
        impact_id=_optional_str(value.get("impact_id")),
        microphone_position=_optional_str(value.get("microphone_position")),
        microphone_distance_m=value.get("microphone_distance_m"),
        microphone_axis=_optional_str(value.get("microphone_axis")),
        gain_setting=_optional_str(value.get("gain_setting")),
        channel=value.get("channel"),
        start_offset_s=value.get("start_offset_s"),
        end_offset_s=value.get("end_offset_s"),
        polarity=int(value.get("polarity", 1)),
        enabled=bool(value.get("enabled", True)),
        notes=_optional_str(value.get("notes")),
        metadata=value.get("metadata", {}),
    )


def _pipeline_settings_from_configuration(configuration: Mapping[str, object]) -> ExperimentPipelineSettings:
    values = dict(configuration.get("pipeline", {}))
    allowed = {field.name for field in fields(ExperimentPipelineSettings)}
    ctor = {key: value for key, value in values.items() if key in allowed}
    return ExperimentPipelineSettings(**ctor)


def _export_settings_from_args(args: argparse.Namespace) -> ResultsExportSettings:
    requested = any((args.json, args.csv, args.latex, args.markdown, args.manifest))
    return ResultsExportSettings(
        output_directory=args.output_dir,
        export_json=args.json if requested else True,
        export_csv=args.csv if requested else False,
        export_latex=args.latex if requested else False,
        export_markdown=args.markdown if requested else False,
        export_manifest=args.manifest if requested else True,
        export_summary=True,
        overwrite_policy=args.overwrite,
        missing_value_representation=args.missing_value,
        nonfinite_value_policy=args.nonfinite_policy,
        float_precision=args.precision,
    )


def _visualization_settings_from_args(args: argparse.Namespace) -> ScientificVisualizationSettings:
    figure_types = None
    if not args.all and args.figure:
        figure_types = tuple(ScientificFigureType(item) for item in args.figure)
    formats = tuple(args.format) if args.format else ("png", "svg")
    return ScientificVisualizationSettings(
        output_directory=args.output_dir,
        formats=formats,
        dpi=args.dpi,
        overwrite_policy=args.overwrite,
        show_rejected_results=args.show_rejected,
        show_inconclusive_results=args.show_inconclusive,
        show_ids=args.show_ids,
        color_policy=ScientificColorPolicy.MONOCHROME if args.monochrome else ScientificColorPolicy.DYNAMIC_CONDITION,
        figure_types=figure_types or ScientificVisualizationSettings().figure_types,
    )


def _report_settings_from_args(args: argparse.Namespace) -> ScientificReportSettings:
    requested = any((args.markdown, args.latex, args.pdf))
    return ScientificReportSettings(
        output_directory=args.output_dir,
        generate_markdown=args.markdown if requested else True,
        generate_latex=args.latex or args.pdf if requested else True,
        compile_pdf=args.pdf,
        title=args.title,
        subtitle=args.subtitle,
        authors=tuple(args.author or ()),
        affiliations=tuple(args.affiliation or ()),
        language=args.language,
        latex_engine=args.compiler,
        overwrite_policy=args.overwrite,
        include_invalid_results=args.include_invalid,
        include_inconclusive_results=args.include_inconclusive,
    )


def _cli_settings_from_args(args: argparse.Namespace, effective: Mapping[str, object] | None = None) -> BellLabCLISettings:
    cli_cfg = dict((effective or {}).get("cli", {})) if isinstance((effective or {}).get("cli", {}), Mapping) else {}
    output_format = args.output_format or cli_cfg.get("output_format") or BellLabCLIOutputFormat.TEXT.value
    quiet = bool(args.quiet or cli_cfg.get("quiet", False))
    return BellLabCLISettings(
        output_format=BellLabCLIOutputFormat.QUIET if quiet else output_format,
        quiet=quiet,
        verbose=bool(args.verbose or cli_cfg.get("verbose", False)),
        debug=bool(args.debug or cli_cfg.get("debug", False)),
        log_file=args.log_file or cli_cfg.get("log_file"),
        log_level=args.log_level or cli_cfg.get("log_level", "WARNING"),
        redact_paths=bool(args.redact_paths or cli_cfg.get("redact_paths", False)),
        dry_run=bool(args.dry_run),
        print_effective_config=bool(getattr(args, "print_effective_config", False)),
        configuration_fingerprint=cli_settings_fingerprint(effective or {}),
    )


def _cli_result(
    command: BellLabCLICommand,
    exit_code: BellLabCLIExitCode,
    status: str,
    message: str,
    args: argparse.Namespace,
    *,
    analysis_id: str | None = None,
    experiment_id: str | None = None,
    artifact_paths: Sequence[str] = (),
    warnings: Sequence[str] = (),
    errors: Sequence[str] = (),
    diagnostics: Sequence[str] = (),
    payload: Mapping[str, object] | None = None,
    valid: bool | None = None,
) -> BellLabCLIResult:
    settings = _cli_settings_from_args(args, {})
    payload_dict = dict(payload or {})
    payload_dict["output_format"] = settings.output_format.value
    payload_dict["redact_paths"] = settings.redact_paths
    payload_dict["scientific_caution"] = (
        "command_success_is_not_scientific_proof",
        "hypothesis_modal_not_physical_mode_proof",
        "operational_evidence_not_causality_or_confirmed_transfer",
    )
    paths = tuple(_redact_path(path) for path in artifact_paths) if settings.redact_paths else tuple(str(path) for path in artifact_paths)
    return BellLabCLIResult(
        command=command,
        exit_code=exit_code,
        status=status,
        message=message,
        analysis_id=analysis_id,
        experiment_id=experiment_id,
        artifact_paths=paths,
        warnings=tuple(str(item) for item in warnings),
        errors=tuple(str(item) for item in errors),
        diagnostics=tuple(str(item) for item in diagnostics),
        payload=payload_dict,
        valid=(exit_code in {BellLabCLIExitCode.COMPLETED, BellLabCLIExitCode.COMPLETED_WITH_RESERVATIONS} if valid is None else valid),
    )


def _write_cli_bundle(
    path: str | Path,
    result_type: str,
    obj: object,
    *,
    summary: Mapping[str, object] | None = None,
) -> str:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload_bytes = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    payload_checksum = hashlib.sha256(payload_bytes).hexdigest()
    payload = {
        "schema_version": CLI_BUNDLE_SCHEMA_VERSION,
        "type": result_type,
        "belllab_version": _belllab_version(),
        "payload_encoding": "pickle-base64",
        "payload_security_note": "trusted BellLab local object payload; do not load untrusted files",
        "payload_checksum_sha256": payload_checksum,
        "payload": base64.b64encode(payload_bytes).decode("ascii"),
        "summary": _to_jsonable(summary or {}),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return str(output)


def _load_cli_bundle(path: str | Path, expected_type: str) -> Any:
    bundle_path = Path(path).expanduser()
    if not bundle_path.is_file():
        raise BellLabCLIConfigurationError(f"result bundle not found: {bundle_path}")
    try:
        data = json.loads(bundle_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BellLabCLIConfigurationError(f"invalid result bundle JSON: {exc}") from exc
    if not isinstance(data, Mapping):
        raise BellLabCLIConfigurationError("result bundle root must be an object.")
    if data.get("schema_version") != CLI_BUNDLE_SCHEMA_VERSION:
        raise BellLabCLIConfigurationError("unsupported CLI result bundle schema.")
    if data.get("type") != expected_type:
        raise BellLabCLIConfigurationError(f"expected {expected_type} bundle, got {data.get('type')!r}.")
    if data.get("payload_encoding") != "pickle-base64":
        raise BellLabCLIConfigurationError("only trusted BellLab pickle-base64 bundles are supported for full reconstruction.")
    payload = data.get("payload")
    if not isinstance(payload, str):
        raise BellLabCLIConfigurationError("result bundle payload is missing.")
    raw = base64.b64decode(payload.encode("ascii"))
    checksum = hashlib.sha256(raw).hexdigest()
    if checksum != data.get("payload_checksum_sha256"):
        raise BellLabCLIConfigurationError("result bundle payload checksum mismatch.")
    obj = pickle.loads(raw)
    expected_class = _RESULT_BUNDLE_TYPES[expected_type]
    if not isinstance(obj, expected_class):
        raise BellLabCLIConfigurationError("result bundle payload type is incompatible.")
    return obj


def _inspect_path(path: Path) -> dict[str, object]:
    source = path.expanduser()
    if not source.is_file():
        raise BellLabCLIConfigurationError(f"file not found: {source}")
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
        kind = "json"
    except json.JSONDecodeError:
        if source.suffix.lower() == ".toml":
            data = load_cli_configuration(source)
            kind = "configuration"
        else:
            raise BellLabCLIConfigurationError("inspect supports JSON bundles/manifests and TOML configuration.")
    info: dict[str, object] = {
        "path": str(source),
        "kind": kind,
        "size_bytes": source.stat().st_size,
        "checksum_sha256": _file_checksum(source),
    }
    if isinstance(data, Mapping):
        if data.get("schema_version") == CLI_BUNDLE_SCHEMA_VERSION and data.get("type") in _RESULT_BUNDLE_TYPES:
            info.update({"kind": f"cli_{data['type']}_bundle", "schema_version": data.get("schema_version")})
            summary = data.get("summary", {})
            if isinstance(summary, Mapping):
                info.update({key: value for key, value in summary.items() if key in {"status", "analysis_id", "experiment_id", "export_id", "collection_id", "report_id", "artifact_count"}})
        elif "manifest_schema_version" in data:
            info.update({"kind": "export_manifest", "schema_version": data.get("manifest_schema_version")})
        elif "report_schema_version" in data:
            info.update({"kind": "report_manifest", "schema_version": data.get("report_schema_version")})
        elif "export_schema_version" in data:
            info.update({"kind": "normalized_export", "schema_version": data.get("export_schema_version")})
        elif "configuration" in data and "fingerprint" in data:
            info.update({"kind": "cli_configuration", "schema_version": data.get("schema_version"), "fingerprint": data.get("fingerprint")})
        else:
            info.update({"top_level_keys": tuple(sorted(str(key) for key in data.keys()))})
    return info


def _validate_inspected(info: Mapping[str, object], path: Path) -> dict[str, object]:
    actual = _file_checksum(path)
    return {"valid": True, "checksum_sha256": actual, "diagnostics": ("content_checksum_calculated",)}


def _stage_flags_until(stage_name: str) -> dict[str, object]:
    target = _STAGE_BY_VALUE[stage_name]
    enabled = True
    flags: dict[str, object] = {}
    for stage in EXPERIMENT_PIPELINE_STAGE_ORDER:
        if stage is ExperimentPipelineStage.SUMMARY:
            continue
        flags[_PIPELINE_FLAG_BY_STAGE[stage]] = enabled
        if stage is target:
            enabled = False
    return flags


def _stage_flags_only(stage_names: Sequence[str]) -> dict[str, object]:
    selected = {_STAGE_BY_VALUE[name] for name in stage_names}
    required: set[ExperimentPipelineStage] = set(selected)
    changed = True
    from belllab.experiment_pipeline import EXPERIMENT_PIPELINE_STAGE_DEPENDENCIES
    while changed:
        changed = False
        for stage in tuple(required):
            for dependency in EXPERIMENT_PIPELINE_STAGE_DEPENDENCIES[stage]:
                if dependency not in required:
                    required.add(dependency)
                    changed = True
    return {
        _PIPELINE_FLAG_BY_STAGE[stage]: stage in required
        for stage in EXPERIMENT_PIPELINE_STAGE_ORDER
        if stage in _PIPELINE_FLAG_BY_STAGE
    }


def _analysis_save_path(raw: Path, result: ExperimentAnalysisResult, output_dir: Path | None) -> Path:
    if str(raw):
        return raw
    base = output_dir or Path(".")
    return base / f"{result.analysis_id}.json"


def _planned_analysis_artifacts(args: argparse.Namespace, experiment: ExperimentDefinition) -> tuple[str, ...]:
    if args.save_result is None:
        return ()
    raw = args.save_result
    if str(raw):
        return (str(raw),)
    base = args.output_dir or Path(".")
    return (str(base / f"analysis-for-{experiment.experiment_id}.json"),)


def _planned_export_artifacts(analysis: ExperimentAnalysisResult, args: argparse.Namespace) -> tuple[str, ...]:
    requested = []
    if args.json or not any((args.json, args.csv, args.latex, args.markdown, args.manifest)):
        requested.append("experiment_export.json")
    if args.csv:
        requested.append("csv_tables/*.csv")
    if args.latex:
        requested.append("latex_tables/*.tex")
    if args.markdown:
        requested.append("experiment_summary.md")
    if args.manifest or not requested:
        requested.append("manifest.json")
    return tuple(str(args.output_dir / item) for item in requested)


def _planned_visualization_artifacts(analysis: ExperimentAnalysisResult, args: argparse.Namespace) -> tuple[str, ...]:
    formats = tuple(args.format) if args.format else ("png", "svg")
    figures = tuple(args.figure) if args.figure and not args.all else ("default_figures",)
    return tuple(str(args.output_dir / f"{figure}.{fmt}") for figure in figures for fmt in formats)


def _planned_report_artifacts(analysis: ExperimentAnalysisResult, args: argparse.Namespace) -> tuple[str, ...]:
    requested = []
    if args.markdown or not any((args.markdown, args.latex, args.pdf)):
        requested.append("report.md")
    if args.latex or args.pdf or not any((args.markdown, args.latex, args.pdf)):
        requested.append("report.tex")
    if args.pdf:
        requested.append("report.pdf")
    requested.append("manifest.json")
    return tuple(str(args.output_dir / item) for item in requested)


def _analysis_warnings(result: ExperimentAnalysisResult) -> tuple[str, ...]:
    values = list(result.failed_stages) + list(result.blocked_stages)
    if result.requires_review:
        values.append(result.status.value)
    return tuple(values)


def _scientific_status_message(noun: str, status: str) -> str:
    return (
        f"BellLab {noun} finished with status {status}. "
        "A successful command is not a physical proof; review statuses, reasons and provenance."
    )


def _result_output_format(result: BellLabCLIResult) -> BellLabCLIOutputFormat:
    payload_format = result.payload.get("output_format")
    if payload_format in {item.value for item in BellLabCLIOutputFormat}:
        return BellLabCLIOutputFormat(payload_format)
    return BellLabCLIOutputFormat.TEXT


def _namespace_command(namespace: argparse.Namespace) -> str:
    return str(getattr(namespace, "command", "unknown") or "unknown").replace("-", "_")


def _configure_logging(args: argparse.Namespace) -> None:
    logger = logging.getLogger()
    for handler in tuple(logger.handlers):
        if getattr(handler, "_belllab_cli_handler", False):
            logger.removeHandler(handler)
    level_name = args.log_level or ("DEBUG" if args.debug else "INFO" if args.verbose else "WARNING")
    handlers: list[logging.Handler] = []
    stream = logging.StreamHandler(sys.stderr)
    stream._belllab_cli_handler = True  # type: ignore[attr-defined]
    handlers.append(stream)
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file, encoding="utf-8")
        file_handler._belllab_cli_handler = True  # type: ignore[attr-defined]
        handlers.append(file_handler)
    logger.setLevel(getattr(logging, level_name))
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        logger.addHandler(handler)


def _recording_to_config(recording: ExperimentRecordingDefinition) -> Mapping[str, object]:
    return {
        "recording_id": recording.recording_id,
        "dynamic_label": recording.dynamic_label,
        "file_path": str(recording.file_path),
        "take_index": recording.take_index,
        "channel": recording.channel,
        "start_offset_s": recording.start_offset_s,
        "end_offset_s": recording.end_offset_s,
    }


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _display_path(path: str, payload: Mapping[str, object]) -> str:
    return path


def _redact_path(path: str) -> str:
    value = str(path)
    return Path(value).name if Path(value).is_absolute() else value


def _file_checksum(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _belllab_version() -> str:
    module = sys.modules.get("belllab")
    return str(getattr(module, "__version__", "0+unknown"))


def _to_jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _to_jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list, set, frozenset)):
        return tuple(_to_jsonable(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _canonicalize(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonicalize(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return tuple(_canonicalize(item) for item in value)
    return value


def _deepcopy_jsonable(value: object) -> object:
    return json.loads(json.dumps(_to_jsonable(value), sort_keys=True, ensure_ascii=False, allow_nan=False))


def _deep_merge(base: Mapping[str, object], override: Mapping[str, object]) -> dict[str, object]:
    result = dict(_deepcopy_jsonable(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)  # type: ignore[arg-type]
        else:
            result[key] = _deepcopy_jsonable(value)
    return result


def _coerce_enum(value: object, enum_type: type[Enum]) -> Any:
    if isinstance(value, enum_type):
        return value
    if issubclass(enum_type, IntEnum):
        return enum_type(int(value))  # type: ignore[call-arg]
    return enum_type(str(value))


def _validate_mapping_keys(value: object, allowed: set[str], name: str) -> None:
    if not isinstance(value, Mapping):
        raise BellLabCLIConfigurationError(f"{name} must be an object.")
    unknown = sorted(set(str(key) for key in value) - allowed)
    if unknown:
        raise BellLabCLIConfigurationError(f"unknown {name} key: {unknown[0]}")


_EXPERIMENT_KEYS = {
    "experiment_id",
    "name",
    "description",
    "specimen_id",
    "instrument_type",
    "location",
    "operator",
    "acquisition_date",
    "dynamic_labels",
    "reference_recording_id",
    "microphone",
    "audio_interface",
    "acquisition_notes",
    "environment_notes",
    "expected_sample_rate_hz",
    "expected_channel_count",
    "metadata",
}
_RECORDING_KEYS = {
    "recording_id",
    "file_path",
    "dynamic_label",
    "take_index",
    "replicate_group",
    "specimen_id",
    "impact_id",
    "microphone_position",
    "microphone_distance_m",
    "microphone_axis",
    "gain_setting",
    "channel",
    "start_offset_s",
    "end_offset_s",
    "polarity",
    "enabled",
    "notes",
    "metadata",
}
_PIPELINE_KEYS = {field.name for field in fields(ExperimentPipelineSettings)}
_EXPORT_KEYS = {field.name for field in fields(ResultsExportSettings)}
_VISUALIZATION_KEYS = {field.name for field in fields(ScientificVisualizationSettings)}
_REPORT_KEYS = {field.name for field in fields(ScientificReportSettings)}
_SYNTHETIC_KEYS = {field.name for field in fields(SyntheticValidationSettings)} | {"scenarios"}


def _parse_toml_subset(text: str) -> Mapping[str, object]:
    """Parse the small TOML subset used by BellLab examples on Python 3.10."""

    root: dict[str, object] = {}
    current: dict[str, object] | None = root
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[[") and line.endswith("]]"):
            parent, name = _toml_parent_table(root, line[2:-2].strip())
            array = parent.setdefault(name, [])
            if not isinstance(array, list):
                raise BellLabCLIConfigurationError(f"TOML array {name} conflicts with scalar section.")
            item: dict[str, object] = {}
            array.append(item)
            current = item
            continue
        if line.startswith("[") and line.endswith("]"):
            parent, name = _toml_parent_table(root, line[1:-1].strip())
            section = parent.setdefault(name, {})
            if not isinstance(section, dict):
                raise BellLabCLIConfigurationError(f"TOML section {name} conflicts with array.")
            current = section
            continue
        if "=" not in line or current is None:
            raise BellLabCLIConfigurationError(f"unsupported TOML line: {raw_line}")
        key, value = line.split("=", 1)
        current[key.strip()] = _parse_toml_value(value.strip())
    return root


def _toml_parent_table(root: dict[str, object], dotted_name: str) -> tuple[dict[str, object], str]:
    """Return the parent mapping and final key for a supported TOML table."""

    parts = dotted_name.split(".")
    if not all(parts):
        raise BellLabCLIConfigurationError(f"invalid TOML table name: {dotted_name}")
    current = root
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise BellLabCLIConfigurationError(f"TOML table {dotted_name} conflicts with scalar value.")
        current = child
    return current, parts[-1]


def _parse_toml_value(value: str) -> object:
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value in {"true", "false"}:
        return value == "true"
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_toml_value(item.strip()) for item in inner.split(",")]
    if value in {"null", "None"}:
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _toml_example_text(configuration: Mapping[str, object]) -> str:
    experiment = configuration["experiment"]  # type: ignore[index]
    recordings = configuration["recordings"]  # type: ignore[index]
    pipeline = configuration.get("pipeline", {})
    lines = [
        "# BellLab example experiment configuration.",
        "# Replace placeholder paths with explicit WAV files before running analyze.",
        "[experiment]",
    ]
    for key, value in experiment.items():  # type: ignore[union-attr]
        lines.append(f"{key} = {_toml_value(value)}")
    lines.append("")
    for recording in recordings:  # type: ignore[union-attr]
        lines.append("[[recordings]]")
        for key, value in recording.items():
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    if pipeline:
        lines.append("[pipeline]")
        for key, value in pipeline.items():  # type: ignore[union-attr]
            lines.append(f"{key} = {_toml_value(value)}")
    return "\n".join(lines).rstrip() + "\n"


def _toml_value(value: object) -> str:
    if value is None:
        return '""'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return json.dumps(str(value), ensure_ascii=False)


def _example_configuration() -> Mapping[str, object]:
    return {
        "experiment": {
            "experiment_id": "example-bell-001",
            "name": "BellLab CLI example experiment",
            "instrument_type": "bell",
            "dynamic_labels": list(DEFAULT_DYNAMIC_LABELS),
        },
        "recordings": [
            {"recording_id": f"example-{label}-01", "dynamic_label": label, "file_path": f"audio/{label}.wav", "channel": 0, "take_index": 0}
            for label in DEFAULT_DYNAMIC_LABELS
        ],
        "pipeline": {
            "run_stft": True,
            "run_tracking": True,
            "run_modal_energy_exchange": True,
            "continue_after_stage_failure": True,
            "allow_partial_results": True,
        },
    }
