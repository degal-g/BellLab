"""Tests for the public BellLab command-line interface."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

import belllab.cli as cli_module
from belllab import (
    BellLabCLICommand,
    BellLabCLIConfigurationError,
    BellLabCLIExitCode,
    BellLabCLIOutputFormat,
    BellLabCLIResult,
    BellLabCLISettings,
    ExperimentAnalysisStatus,
    ResultsExportStatus,
    ScientificReportStatus,
    ScientificVisualizationStatus,
    SyntheticValidationStatus,
    __version__,
    build_cli_parser,
    cli_settings_fingerprint,
    format_cli_json_output,
    format_cli_text_output,
    load_cli_configuration,
    load_serialized_analysis_result,
    load_serialized_export_result,
    load_serialized_figure_collection,
    map_result_status_to_exit_code,
    merge_cli_configuration,
    parse_cli_recording_spec,
    run_cli,
    serialize_cli_configuration,
    validate_cli_configuration,
    write_cli_configuration,
)


def _write_wav(path: Path, frequency_hz: float = 300.0, *, channels: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 4096
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    signal = 0.25 * np.exp(-time / 2.0) * np.sin(2 * np.pi * frequency_hz * time)
    if channels == 2:
        signal = np.column_stack((signal, 0.5 * signal))
    sf.write(path, signal.astype(np.float32), sample_rate, subtype="FLOAT")
    return path


@pytest.fixture(scope="module")
def cli_artifacts(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("belllab-cli")
    wav = _write_wav(root / "audio" / "pp.wav")
    analysis = root / "analysis.json"
    analyzed = run_cli([
        "analyze",
        "--recording",
        f"pp={wav}",
        "--until-stage",
        "global_spectrum",
        "--save-result",
        str(analysis),
        "--output-format",
        "json",
    ])
    assert analyzed.exit_code is BellLabCLIExitCode.COMPLETED
    export_bundle = root / "export-bundle.json"
    exported = run_cli([
        "export",
        "--analysis",
        str(analysis),
        "--output-dir",
        str(root / "export"),
        "--json",
        "--manifest",
        "--overwrite",
        "replace",
        "--save-result",
        str(export_bundle),
        "--output-format",
        "json",
    ])
    assert exported.exit_code is BellLabCLIExitCode.COMPLETED
    figure_bundle = root / "figures.json"
    visualized = run_cli([
        "visualize",
        "--analysis",
        str(analysis),
        "--output-dir",
        str(root / "figures"),
        "--figure",
        "global_spectrum",
        "--format",
        "png",
        "--overwrite",
        "replace",
        "--save-result",
        str(figure_bundle),
        "--output-format",
        "json",
    ])
    assert visualized.exit_code is BellLabCLIExitCode.COMPLETED
    return {
        "root": root,
        "wav": wav,
        "analysis": analysis,
        "export": export_bundle,
        "figures": figure_bundle,
    }


def test_public_cli_contracts_are_importable() -> None:
    assert BellLabCLICommand.ANALYZE.value == "analyze"
    assert BellLabCLIOutputFormat.JSON.value == "json"
    assert BellLabCLIExitCode.REPORT_COMPILATION_FAILED.value == 9
    assert isinstance(BellLabCLISettings(), BellLabCLISettings)
    assert callable(build_cli_parser)
    assert callable(run_cli)
    assert callable(format_cli_text_output)
    assert callable(format_cli_json_output)


def test_parser_exposes_required_subcommands() -> None:
    help_text = build_cli_parser().format_help()
    for command in ("analyze", "export", "visualize", "report", "validate-synthetic", "inspect", "version"):
        assert command in help_text


def test_no_argument_usage_error() -> None:
    result = run_cli(())
    assert result.exit_code is BellLabCLIExitCode.USAGE_OR_CONFIGURATION_INVALID
    assert result.valid is False


def test_unknown_command_is_usage_error() -> None:
    result = run_cli(("unknown",))
    assert result.exit_code is BellLabCLIExitCode.USAGE_OR_CONFIGURATION_INVALID
    assert result.status == "usage_error"


def test_version_command_text_and_json() -> None:
    assert __version__ == "0.13.0"
    text_result = run_cli(("version",))
    assert text_result.exit_code is BellLabCLIExitCode.COMPLETED
    text_output = format_cli_text_output(text_result)
    assert "BellLab 0.13.0" in text_output
    assert "CLI Schema Version: 1.0" in text_output
    json_result = run_cli(("version", "--output-format", "json"))
    decoded = json.loads(format_cli_json_output(json_result))
    assert decoded["payload"]["belllab_version"] == "0.13.0"
    assert decoded["payload"]["export_schema_version"] == "1.0"
    assert decoded["payload"]["report_schema_version"] == "1.0"


def test_quiet_result_formats_empty_text() -> None:
    result = run_cli(("version", "--quiet"))
    assert result.exit_code is BellLabCLIExitCode.COMPLETED
    assert format_cli_text_output(result) == ""


def test_parse_recording_spec_label_take_paths_spaces_and_accents(tmp_path: Path) -> None:
    path = tmp_path / "áudio com espaço.wav"
    recording = parse_cli_recording_spec(f"mf:3={path}", channel=1, start_offset_s=0.1, end_offset_s=0.9)
    assert recording.dynamic_label == "mf"
    assert recording.take_index == 3
    assert recording.channel == 1
    assert recording.file_path == path


def test_parse_recording_spec_rejects_invalid_values() -> None:
    with pytest.raises(BellLabCLIConfigurationError):
        parse_cli_recording_spec("pp")
    with pytest.raises(BellLabCLIConfigurationError):
        parse_cli_recording_spec("pp:x=file.wav")


def test_load_cli_configuration_json_and_toml(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "audio" / "pp.wav")
    payload = {
        "experiment": {"name": "config experiment", "dynamic_labels": ["pp"]},
        "recordings": [{"dynamic_label": "pp", "file_path": str(wav), "recording_id": "pp"}],
        "pipeline": {"run_stft": False, "run_tracking": False},
    }
    json_path = tmp_path / "experiment.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    toml_path = tmp_path / "experiment.toml"
    toml_path.write_text(
        """
[experiment]
name = "config experiment"
dynamic_labels = ["pp"]

[[recordings]]
dynamic_label = "pp"
file_path = "audio/pp.wav"
recording_id = "pp"

[pipeline]
run_stft = false
run_tracking = false
""".strip()
        + "\n",
        encoding="utf-8",
    )
    assert load_cli_configuration(json_path)["experiment"]["name"] == "config experiment"
    assert load_cli_configuration(toml_path)["recordings"][0]["file_path"] == "audio/pp.wav"


def test_toml_compatibility_parser_without_tomllib(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The documented TOML subset remains usable on Python 3.10."""

    monkeypatch.setattr(cli_module, "tomllib", None)
    example = Path(__file__).resolve().parents[1] / "examples" / "experiment.example.toml"
    loaded_example = load_cli_configuration(example)
    assert loaded_example["experiment"]["dynamic_labels"] == ["pp", "p", "mf", "f", "ff"]
    assert len(loaded_example["recordings"]) == 5
    assert loaded_example["pipeline"]["run_stft"] is True

    compatible = tmp_path / "compatible.toml"
    compatible.write_text(
        """
[experiment]
name = "Sino áureo"
dynamic_labels = ["pp", "mf"]

[experiment.metadata]
material = "bronze"

[[recordings]]
recording_id = "pp-01"
dynamic_label = "pp"
file_path = "audio/pp.wav"
channel = 0
enabled = true

[[recordings]]
recording_id = "mf-01"
dynamic_label = "mf"
file_path = "audio/mf.wav"
microphone_distance_m = 0.125
enabled = false

[pipeline]
run_stft = false
""".strip()
        + "\n",
        encoding="utf-8",
    )
    loaded = load_cli_configuration(compatible)
    assert loaded["experiment"]["metadata"]["material"] == "bronze"
    assert loaded["recordings"][1]["file_path"] == "audio/mf.wav"
    assert loaded["recordings"][1]["microphone_distance_m"] == pytest.approx(0.125)
    assert loaded["recordings"][1]["enabled"] is False

    invalid = tmp_path / "invalid.toml"
    invalid.write_text("[experiment\nname = \"broken\"\n", encoding="utf-8")
    with pytest.raises(BellLabCLIConfigurationError, match="unsupported TOML line"):
        load_cli_configuration(invalid)

    unknown = tmp_path / "unknown.toml"
    unknown.write_text("[unknown]\nvalue = 1\n", encoding="utf-8")
    with pytest.raises(BellLabCLIConfigurationError, match="unknown configuration key"):
        load_cli_configuration(unknown)


def test_configuration_unknown_key_is_rejected() -> None:
    with pytest.raises(BellLabCLIConfigurationError):
        validate_cli_configuration({"unknown": {}})
    with pytest.raises(BellLabCLIConfigurationError):
        validate_cli_configuration({"pipeline": {"not_a_setting": True}})


def test_configuration_precedence_and_fingerprint_are_deterministic() -> None:
    merged = merge_cli_configuration(
        {"cli": {"output_format": "text"}, "pipeline": {"maximum_worker_count": 1}},
        {"cli": {"output_format": "json"}},
        {"pipeline": {"maximum_worker_count": 2}},
    )
    assert merged["cli"]["output_format"] == "json"
    assert merged["pipeline"]["maximum_worker_count"] == 2
    assert cli_settings_fingerprint(merged) == cli_settings_fingerprint(dict(reversed(tuple(merged.items()))))
    serialized = serialize_cli_configuration(merged)
    assert serialized["fingerprint"] == cli_settings_fingerprint(merged)


def test_write_cli_configuration_examples_are_loadable(tmp_path: Path) -> None:
    toml_path = write_cli_configuration(tmp_path / "experiment.example.toml")
    json_path = write_cli_configuration(tmp_path / "experiment.example.json")
    assert load_cli_configuration(toml_path)["experiment"]["instrument_type"] == "bell"
    assert load_cli_configuration(json_path)["recordings"][0]["dynamic_label"] == "pp"


def test_analyze_dry_run_reports_missing_file_without_writing(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    result = run_cli([
        "analyze",
        "--recording",
        f"pp={tmp_path / 'missing.wav'}",
        "--dry-run",
        "--output-dir",
        str(output_dir),
        "--save-result",
        "--output-format",
        "json",
    ])
    assert result.exit_code is BellLabCLIExitCode.INPUT_INVALID
    assert result.status == "invalid_input"
    assert "missing_files" in result.warnings
    assert not output_dir.exists()


def test_analyze_print_effective_config_skips_analysis(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "pp.wav")
    result = run_cli([
        "analyze",
        "--recording",
        f"pp={wav}",
        "--until-stage",
        "temporal",
        "--print-effective-config",
        "--output-format",
        "json",
    ])
    assert result.status == "effective_configuration"
    assert "effective_configuration" not in result.payload
    assert "configuration" in result.payload


def test_analyze_saves_and_loads_result_bundle(cli_artifacts: dict[str, Path]) -> None:
    analysis = load_serialized_analysis_result(cli_artifacts["analysis"])
    assert analysis.status is ExperimentAnalysisStatus.COMPLETED
    assert analysis.analysis_id
    assert analysis.experiment.recordings[0].file_path == cli_artifacts["wav"]


def test_analyze_resume_from_bundle(cli_artifacts: dict[str, Path], tmp_path: Path) -> None:
    resumed = tmp_path / "resumed.json"
    result = run_cli([
        "analyze",
        "--recording",
        f"pp={cli_artifacts['wav']}",
        "--until-stage",
        "global_spectrum",
        "--resume-from",
        str(cli_artifacts["analysis"]),
        "--save-result",
        str(resumed),
    ])
    assert result.exit_code is BellLabCLIExitCode.COMPLETED
    assert resumed.exists()


def test_analyze_rejects_invalid_channel() -> None:
    result = run_cli(("analyze", "--recording", "pp=file.wav", "--channel", "-1"))
    assert result.exit_code is BellLabCLIExitCode.USAGE_OR_CONFIGURATION_INVALID
    assert "channel" in result.message


def test_analyze_invalid_stage_dependency_is_configuration_error(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "pp.wav")
    result = run_cli(("analyze", "--recording", f"pp={wav}", "--skip-stage", "stft"))
    assert result.exit_code is BellLabCLIExitCode.USAGE_OR_CONFIGURATION_INVALID
    assert "requires" in result.message


def test_analyze_only_stage_includes_dependencies(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "pp.wav")
    result = run_cli(("analyze", "--recording", f"pp={wav}", "--only-stage", "global_spectrum", "--dry-run"))
    assert result.exit_code is BellLabCLIExitCode.COMPLETED
    assert result.payload["settings_fingerprint"]


def test_export_command_writes_requested_artifacts(cli_artifacts: dict[str, Path], tmp_path: Path) -> None:
    bundle = tmp_path / "export.json"
    result = run_cli([
        "export",
        "--analysis",
        str(cli_artifacts["analysis"]),
        "--output-dir",
        str(tmp_path / "export"),
        "--json",
        "--csv",
        "--manifest",
        "--overwrite",
        "replace",
        "--validate",
        "--save-result",
        str(bundle),
    ])
    assert result.exit_code is BellLabCLIExitCode.COMPLETED
    assert bundle.exists()
    export_result = load_serialized_export_result(bundle)
    assert export_result.status is ResultsExportStatus.COMPLETED
    assert any(path.endswith(".csv") for path in result.artifact_paths)


def test_export_dry_run_writes_nothing(cli_artifacts: dict[str, Path], tmp_path: Path) -> None:
    output = tmp_path / "export"
    result = run_cli(("export", "--analysis", str(cli_artifacts["analysis"]), "--output-dir", str(output), "--json", "--dry-run"))
    assert result.status == "dry_run"
    assert not output.exists()


def test_export_rejects_incompatible_schema(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": "0", "type": "analysis"}), encoding="utf-8")
    result = run_cli(("export", "--analysis", str(bad), "--output-dir", str(tmp_path / "export")))
    assert result.exit_code is BellLabCLIExitCode.USAGE_OR_CONFIGURATION_INVALID
    assert "schema" in result.message


def test_export_overwrite_error_preserves_existing_files(cli_artifacts: dict[str, Path], tmp_path: Path) -> None:
    output = tmp_path / "export"
    first = run_cli(("export", "--analysis", str(cli_artifacts["analysis"]), "--output-dir", str(output), "--json", "--overwrite", "replace"))
    second = run_cli(("export", "--analysis", str(cli_artifacts["analysis"]), "--output-dir", str(output), "--json", "--overwrite", "error"))
    assert first.exit_code is BellLabCLIExitCode.COMPLETED
    assert second.exit_code in {BellLabCLIExitCode.STAGE_FAILURE, BellLabCLIExitCode.PARTIAL_EXECUTION}


def test_visualize_command_writes_headless_figure_bundle(cli_artifacts: dict[str, Path], tmp_path: Path) -> None:
    bundle = tmp_path / "figures.json"
    result = run_cli([
        "visualize",
        "--analysis",
        str(cli_artifacts["analysis"]),
        "--output-dir",
        str(tmp_path / "figures"),
        "--figure",
        "global_spectrum",
        "--format",
        "png",
        "--overwrite",
        "replace",
        "--save-result",
        str(bundle),
    ])
    assert result.exit_code is BellLabCLIExitCode.COMPLETED
    assert bundle.exists()
    figures = load_serialized_figure_collection(bundle)
    assert figures.status is ScientificVisualizationStatus.CREATED


def test_visualize_dry_run_writes_nothing(cli_artifacts: dict[str, Path], tmp_path: Path) -> None:
    output = tmp_path / "figures"
    result = run_cli(("visualize", "--analysis", str(cli_artifacts["analysis"]), "--output-dir", str(output), "--figure", "global_spectrum", "--dry-run"))
    assert result.status == "dry_run"
    assert not output.exists()


def test_report_command_writes_markdown_latex_bundle(cli_artifacts: dict[str, Path], tmp_path: Path) -> None:
    bundle = tmp_path / "report.json"
    result = run_cli([
        "report",
        "--analysis",
        str(cli_artifacts["analysis"]),
        "--export-result",
        str(cli_artifacts["export"]),
        "--figure-collection",
        str(cli_artifacts["figures"]),
        "--output-dir",
        str(tmp_path / "report"),
        "--markdown",
        "--latex",
        "--overwrite",
        "replace",
        "--title",
        "CLI report",
        "--author",
        "BellLab test",
        "--save-result",
        str(bundle),
    ])
    assert result.exit_code in {BellLabCLIExitCode.COMPLETED, BellLabCLIExitCode.COMPLETED_WITH_RESERVATIONS}
    assert bundle.exists()
    assert any(path.endswith(".md") for path in result.artifact_paths)


def test_report_dry_run_writes_nothing(cli_artifacts: dict[str, Path], tmp_path: Path) -> None:
    output = tmp_path / "report"
    result = run_cli(("report", "--analysis", str(cli_artifacts["analysis"]), "--output-dir", str(output), "--markdown", "--dry-run"))
    assert result.status == "dry_run"
    assert not output.exists()


def test_report_rejects_incompatible_figure_bundle(cli_artifacts: dict[str, Path], tmp_path: Path) -> None:
    bad = tmp_path / "bad-figures.json"
    bad.write_text(json.dumps({"schema_version": "1.0", "type": "analysis", "payload": ""}), encoding="utf-8")
    result = run_cli(("report", "--analysis", str(cli_artifacts["analysis"]), "--figure-collection", str(bad), "--output-dir", str(tmp_path / "report")))
    assert result.exit_code is BellLabCLIExitCode.USAGE_OR_CONFIGURATION_INVALID


def test_validate_synthetic_single_scenario_and_monte_carlo() -> None:
    single = run_cli(("validate-synthetic", "--scenario", "single_ideal", "--seed", "123", "--duration", "1.0", "--sample-rate", "2048"))
    monte = run_cli(("validate-synthetic", "--scenario", "single_ideal", "--trials", "2", "--seed", "123", "--duration", "1.0", "--sample-rate", "2048"))
    assert single.exit_code in {BellLabCLIExitCode.COMPLETED, BellLabCLIExitCode.COMPLETED_WITH_RESERVATIONS, BellLabCLIExitCode.STAGE_FAILURE}
    assert monte.payload["summary"]
    assert "truth_not_used_to_calibrate_thresholds" in monte.diagnostics


def test_validate_synthetic_all_scenarios_dry_run() -> None:
    result = run_cli(("validate-synthetic", "--all-scenarios", "--dry-run", "--output-format", "json"))
    assert result.exit_code is BellLabCLIExitCode.COMPLETED
    assert result.payload["scenarios"] == ("all",)


def test_inspect_cli_bundles_and_config(cli_artifacts: dict[str, Path], tmp_path: Path) -> None:
    inspected = run_cli(("inspect", str(cli_artifacts["analysis"]), "--validate", "--show-ids"))
    assert inspected.exit_code is BellLabCLIExitCode.COMPLETED
    assert inspected.payload["kind"] == "cli_analysis_bundle"
    config = write_cli_configuration(tmp_path / "experiment.example.toml")
    inspected_config = run_cli(("inspect", str(config), "--summary"))
    assert inspected_config.payload["kind"] == "configuration"


def test_inspect_invalid_file_is_input_error(tmp_path: Path) -> None:
    result = run_cli(("inspect", str(tmp_path / "missing.json")))
    assert result.exit_code is BellLabCLIExitCode.USAGE_OR_CONFIGURATION_INVALID


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (ExperimentAnalysisStatus.COMPLETED, BellLabCLIExitCode.COMPLETED),
        (ExperimentAnalysisStatus.COMPLETED_WITH_RESERVATIONS, BellLabCLIExitCode.COMPLETED_WITH_RESERVATIONS),
        (ExperimentAnalysisStatus.PARTIAL, BellLabCLIExitCode.PARTIAL_EXECUTION),
        (ExperimentAnalysisStatus.INSUFFICIENT_EVIDENCE, BellLabCLIExitCode.INSUFFICIENT_EVIDENCE),
        (ExperimentAnalysisStatus.INVALID_INPUT, BellLabCLIExitCode.INPUT_INVALID),
        (ExperimentAnalysisStatus.FAILED, BellLabCLIExitCode.STAGE_FAILURE),
        (ScientificReportStatus.COMPILATION_FAILED, BellLabCLIExitCode.REPORT_COMPILATION_FAILED),
        (SyntheticValidationStatus.PIPELINE_ERROR, BellLabCLIExitCode.STAGE_FAILURE),
    ],
)
def test_exit_code_mapping(status: object, code: BellLabCLIExitCode) -> None:
    assert map_result_status_to_exit_code(status) is code


def test_json_output_contains_only_json_payload() -> None:
    result = run_cli(("version", "--output-format", "json"))
    text = format_cli_json_output(result)
    assert json.loads(text)["command"] == "version"
    assert "NaN" not in text
    assert "Infinity" not in text


def test_text_output_separates_artifacts_and_warnings() -> None:
    result = BellLabCLIResult(
        command=BellLabCLICommand.EXPORT,
        exit_code=BellLabCLIExitCode.COMPLETED_WITH_RESERVATIONS,
        status="completed_with_reservations",
        message="done",
        artifact_paths=("artifact.json",),
        warnings=("source_result_requires_review",),
    )
    text = format_cli_text_output(result)
    assert "Artifacts:" in text
    assert "Warnings: 1" in text


def test_redact_paths_removes_absolute_paths_from_result(tmp_path: Path) -> None:
    result = run_cli(("analyze", "--recording", f"pp={tmp_path / 'missing.wav'}", "--dry-run", "--redact-paths", "--save-result", "--output-dir", str(tmp_path / "out")))
    assert all(not path.startswith("/") for path in result.artifact_paths)
    assert result.payload["redact_paths"] is True


def test_deterministic_dry_run_order_independent(tmp_path: Path) -> None:
    a = _write_wav(tmp_path / "a.wav", 300.0)
    b = _write_wav(tmp_path / "b.wav", 301.0)
    one = run_cli(("analyze", "--recording", f"pp={a}", "--recording", f"p={b}", "--until-stage", "temporal", "--dry-run", "--output-format", "json"))
    two = run_cli(("analyze", "--recording", f"p={b}", "--recording", f"pp={a}", "--until-stage", "temporal", "--dry-run", "--output-format", "json"))
    assert one.payload["settings_fingerprint"] == two.payload["settings_fingerprint"]
    assert set(one.payload["validation"]["dynamic_labels_present"]) == set(two.payload["validation"]["dynamic_labels_present"])


def test_local_perturbation_changes_only_related_path(tmp_path: Path) -> None:
    a = _write_wav(tmp_path / "a.wav", 300.0)
    b = _write_wav(tmp_path / "b.wav", 300.0)
    one = run_cli(("analyze", "--recording", f"pp={a}", "--dry-run"))
    two = run_cli(("analyze", "--recording", f"pp={b}", "--dry-run"))
    assert one.experiment_id != two.experiment_id


def test_argv_is_not_modified() -> None:
    argv = ["version", "--output-format", "json"]
    original = list(argv)
    run_cli(argv)
    assert argv == original


def test_logging_handlers_do_not_accumulate() -> None:
    logger = logging.getLogger()
    before = len([handler for handler in logger.handlers if getattr(handler, "_belllab_cli_handler", False)])
    run_cli(("version", "--verbose"))
    run_cli(("version", "--verbose"))
    after = len([handler for handler in logger.handlers if getattr(handler, "_belllab_cli_handler", False)])
    assert after <= max(1, before + 1)


def test_subprocess_help_version_and_json_stdout() -> None:
    help_run = subprocess.run([sys.executable, "-m", "belllab", "--help"], cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, check=False)
    assert help_run.returncode == 0
    assert "validate-synthetic" in help_run.stdout
    assert "Traceback" not in help_run.stderr

    version_run = subprocess.run([sys.executable, "-m", "belllab", "version", "--output-format", "json"], cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, check=False)
    assert version_run.returncode == 0
    assert json.loads(version_run.stdout)["command"] == "version"
    assert version_run.stderr == ""


def test_subprocess_analyze_dry_run_and_expected_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.wav"
    completed = subprocess.run(
        [sys.executable, "-m", "belllab", "analyze", "--recording", f"pp={missing}", "--dry-run", "--output-format", "json"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == BellLabCLIExitCode.INPUT_INVALID
    assert json.loads(completed.stdout)["status"] == "invalid_input"
    assert "Traceback" not in completed.stderr

    bad = subprocess.run([sys.executable, "-m", "belllab", "not-a-command"], cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, check=False)
    assert bad.returncode == BellLabCLIExitCode.USAGE_OR_CONFIGURATION_INVALID
    assert "Traceback" not in bad.stderr


def test_config_relative_path_resolution(tmp_path: Path) -> None:
    _write_wav(tmp_path / "audio" / "pp.wav")
    config = tmp_path / "experiment.toml"
    config.write_text(
        """
[experiment]
name = "relative config"
dynamic_labels = ["pp"]

[[recordings]]
dynamic_label = "pp"
file_path = "audio/pp.wav"
recording_id = "pp"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    result = run_cli(("analyze", "--config", str(config), "--until-stage", "temporal", "--dry-run"))
    assert result.exit_code is BellLabCLIExitCode.COMPLETED
    assert result.payload["validation"]["missing_files"] == ()


def test_print_effective_config_has_cli_precedence(tmp_path: Path) -> None:
    _write_wav(tmp_path / "audio" / "pp.wav")
    config = tmp_path / "experiment.json"
    config.write_text(json.dumps({
        "experiment": {"name": "from file", "dynamic_labels": ["pp"]},
        "recordings": [{"dynamic_label": "pp", "file_path": "audio/pp.wav", "recording_id": "pp"}],
        "pipeline": {"maximum_worker_count": 1},
    }), encoding="utf-8")
    result = run_cli(("analyze", "--config", str(config), "--workers", "2", "--print-effective-config", "--output-format", "json"))
    assert result.payload["configuration"]["pipeline"]["maximum_worker_count"] == 2
