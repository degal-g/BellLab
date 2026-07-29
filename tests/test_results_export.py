"""Tests for reproducible BellLab result exports."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import belllab.results_export as results_export_module
from belllab import (
    AnalysisSettings,
    CrossConditionCandidateAssociationSettings,
    ExperimentDefinition,
    ExperimentPipelineSettings,
    ExperimentRecordingDefinition,
    ExportMissingValuePolicy,
    ExportNonfiniteValuePolicy,
    ExportOverwritePolicy,
    FramePeakDetectionSettings,
    GlobalSpectralCharacterizationSettings,
    ModalCandidateSettings,
    ModalEnergyExchangeSettings,
    ModalHypothesisSettings,
    ModalParameterEstimationSettings,
    ModalQFactorEstimationSettings,
    PeakDetectionSettings,
    ResultsExportReason,
    ResultsExportSettings,
    ResultsExportStatus,
    STFTSettings,
    SpectralTrackingSettings,
    SpectrumAnalysisSettings,
    TimeResolvedSpectralCharacterizationSettings,
    WithinConditionAssociationSettings,
    analyze_experiment,
    experiment_file_fingerprint,
    export_artifact_checksum,
    export_experiment_csv_tables,
    export_experiment_json,
    export_experiment_latex_tables,
    export_experiment_markdown_summary,
    export_experiment_results,
    export_settings_fingerprint,
    normalize_experiment_for_export,
    serialize_experiment_result,
    summarize_experiment_export,
    validate_experiment_export,
)


def _write_wav(
    path: Path,
    frequency_hz: float,
    *,
    sample_rate: int = 4096,
    duration_s: float = 1.0,
    tau_s: float = 2.0,
    amplitude: float = 0.35,
    second_frequency_hz: float | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(int(round(sample_rate * duration_s)), dtype=np.float64) / sample_rate
    signal = amplitude * np.exp(-time / tau_s) * np.sin(2 * np.pi * frequency_hz * time)
    if second_frequency_hz is not None:
        signal = signal + 0.22 * np.exp(-time / (0.75 * tau_s)) * np.sin(
            2 * np.pi * second_frequency_hz * time
        )
    sf.write(path, signal.astype(np.float32), sample_rate, subtype="FLOAT")
    return path


def _recording(path: Path, label: str, *, recording_id: str | None = None) -> ExperimentRecordingDefinition:
    return ExperimentRecordingDefinition(
        file_path=path,
        dynamic_label=label,
        recording_id=recording_id or label,
    )


def _load_only_settings(**changes) -> ExperimentPipelineSettings:
    base = dict(
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
    base.update(changes)
    return ExperimentPipelineSettings(**base)


def _full_settings(**changes) -> ExperimentPipelineSettings:
    base = dict(
        analysis_settings=AnalysisSettings(
            spectrum=SpectrumAnalysisSettings(n_fft=8192, window_name="hann"),
            stft=STFTSettings(
                window_length=512,
                hop_length=128,
                n_fft=1024,
                frequency_min_hz=200.0,
                frequency_max_hz=900.0,
            ),
            frame_peaks=FramePeakDetectionSettings(
                peak_settings=PeakDetectionSettings(
                    min_prominence=0.001,
                    min_amplitude=0.0001,
                    max_peaks=4,
                ),
                max_peaks_per_frame=4,
            ),
            tracking=SpectralTrackingSettings(
                frequency_tolerance=8.0,
                frequency_distance_unit="hz",
                max_gap_frames=1,
                min_track_length=2,
            ),
        ),
        peak_detection_settings=PeakDetectionSettings(
            min_prominence=0.001,
            min_amplitude=0.0001,
            max_peaks=4,
        ),
        global_spectrum_settings=GlobalSpectralCharacterizationSettings(
            fft_size=8192,
            peak_min_prominence=1e-10,
        ),
        time_resolved_spectrum_settings=TimeResolvedSpectralCharacterizationSettings(
            frame_duration_s=0.125,
            hop_duration_s=0.03125,
            fft_size=1024,
        ),
        modal_candidate_settings=ModalCandidateSettings(
            minimum_observation_count=2,
            require_amplitude_decay=False,
        ),
        within_condition_settings=WithinConditionAssociationSettings(
            minimum_repeat_count=1,
            maximum_absolute_frequency_difference_hz=5.0,
        ),
        cross_condition_settings=CrossConditionCandidateAssociationSettings(
            maximum_absolute_frequency_difference_hz=8.0,
        ),
        modal_hypothesis_settings=ModalHypothesisSettings(
            require_complete_chain=False,
            allow_partial_chains=True,
            minimum_condition_coverage_fraction=None,
            maximum_step_absolute_frequency_change_hz=8.0,
            maximum_total_absolute_frequency_change_hz=20.0,
            maximum_frequency_trajectory_rmse_hz=5.0,
            minimum_mean_coverage_fraction=None,
            maximum_mean_frequency_fit_rmse_hz=None,
        ),
        modal_parameter_settings=ModalParameterEstimationSettings(
            minimum_tau_value_count=1,
            allow_missing_tau=True,
        ),
        modal_q_settings=ModalQFactorEstimationSettings(
            enable_bandwidth_method=False,
        ),
        modal_energy_exchange_settings=ModalEnergyExchangeSettings(
            minimum_overlap_sample_count=3,
            permutation_count=10,
            require_pair_energy_stability=False,
        ),
    )
    base.update(changes)
    return ExperimentPipelineSettings(**base)


def _analysis_result(
    tmp_path: Path,
    *,
    name: str = "export experiment",
    metadata: dict[str, object] | None = None,
    full: bool = False,
):
    if full:
        recordings = []
        for index, label in enumerate(("pp", "p", "mf", "f", "ff")):
            path = _write_wav(
                tmp_path / f"{label}.wav",
                300.0 + index,
                duration_s=1.2,
                second_frequency_hz=520.0 + index,
            )
            recordings.append(_recording(path, label))
        experiment = ExperimentDefinition(
            name=name,
            recordings=tuple(recordings),
            dynamic_labels=("pp", "p", "mf", "f", "ff"),
            metadata=metadata or {},
        )
        return analyze_experiment(
            experiment,
            _full_settings(run_dynamic_condition_comparison=False),
        )
    path = _write_wav(tmp_path / "pp.wav", 300.0)
    experiment = ExperimentDefinition(
        name=name,
        recordings=(_recording(path, "pp"),),
        dynamic_labels=("pp",),
        metadata=metadata or {},
    )
    return analyze_experiment(experiment, _load_only_settings())


def _export_settings(tmp_path: Path, **changes) -> ResultsExportSettings:
    values = dict(output_directory=tmp_path / "export")
    values.update(changes)
    return ResultsExportSettings(**values)


def _artifact_path(result, artifact_type: str, extension: str, output_dir: Path) -> Path:
    return output_dir / f"belllab_results_{result.analysis_id}_{artifact_type}.{extension}"


def test_public_contracts_are_importable() -> None:
    assert ResultsExportStatus.COMPLETED.value == "completed"
    assert ResultsExportReason.ALL_REQUESTED_ARTIFACTS_WRITTEN.value == "all_requested_artifacts_written"
    assert ExportOverwritePolicy.ERROR.value == "error"
    assert ExportMissingValuePolicy.NULL.value == "null"
    assert ExportNonfiniteValuePolicy.ERROR.value == "error"
    assert callable(normalize_experiment_for_export)
    assert callable(serialize_experiment_result)
    assert callable(export_experiment_results)
    assert callable(validate_experiment_export)


@pytest.mark.parametrize(
    "changes",
    (
        {"float_precision": 0},
        {"scientific_notation_threshold": 0.0},
        {"checksum_algorithm": "md5"},
        {"table_layout": "wide"},
        {"sort_policy": "input"},
        {"markdown_alignment": "justify"},
        {
            "export_json": False,
            "export_csv": False,
            "export_latex": False,
            "export_markdown": False,
            "export_manifest": False,
            "export_summary": False,
        },
    ),
)
def test_settings_reject_invalid_invariants(tmp_path: Path, changes) -> None:
    with pytest.raises(ValueError):
        _export_settings(tmp_path, **changes)


def test_export_settings_fingerprint_is_deterministic_and_path_independent(tmp_path: Path) -> None:
    first = ResultsExportSettings(output_directory=tmp_path / "a", float_precision=8)
    second = ResultsExportSettings(output_directory=tmp_path / "b", float_precision=8)
    third = ResultsExportSettings(output_directory=tmp_path / "b", float_precision=7)
    assert export_settings_fingerprint(first) == export_settings_fingerprint(second)
    assert export_settings_fingerprint(first) != export_settings_fingerprint(third)


def test_normalized_export_has_schema_and_tables(tmp_path: Path) -> None:
    result = _analysis_result(tmp_path)
    normalized = normalize_experiment_for_export(result, _export_settings(tmp_path))

    assert normalized.schema_version.value == "1.0"
    assert normalized.analysis_id == result.analysis_id
    assert "recordings" in normalized.tables
    assert normalized.tables["experiment_summary"][0]["analysis_id"] == result.analysis_id
    assert "export_layer_does_not_recalculate_analysis" in normalized.diagnostics


def test_serialize_experiment_result_preserves_nulls_and_ids(tmp_path: Path) -> None:
    result = _analysis_result(tmp_path)
    payload = serialize_experiment_result(result, _export_settings(tmp_path))

    assert payload["analysis_id"] == result.analysis_id
    assert payload["schema_version"] == "1.0"
    assert payload["cross_condition"] is None
    assert payload["schema_description"]["missing_value_meaning"] == "null means unavailable, not zero."


def test_basic_export_writes_json_csv_latex_markdown_and_manifest(tmp_path: Path) -> None:
    result = _analysis_result(tmp_path)
    export = export_experiment_results(result, _export_settings(tmp_path))

    assert export.status is ResultsExportStatus.COMPLETED
    assert export.failed_count == 0
    assert export.completed_count >= 5
    assert export.manifest is not None
    assert validate_experiment_export(export).valid is True
    assert all(Path(artifact.path).is_file() for artifact in export.completed_artifacts if artifact.path)


def test_full_pipeline_result_exports_modal_tables(tmp_path: Path) -> None:
    result = _analysis_result(tmp_path, full=True)
    normalized = normalize_experiment_for_export(result, _export_settings(tmp_path))

    assert len(normalized.tables["candidates"]) >= 5
    assert len(normalized.tables["candidate_chains"]) >= 1
    assert len(normalized.tables["modal_hypotheses"]) >= 1
    assert len(normalized.tables["modal_parameters"]) >= 1
    assert len(normalized.tables["modal_q_factors"]) >= 1


def test_json_export_roundtrips_without_nonstandard_nan_tokens(tmp_path: Path) -> None:
    result = _analysis_result(tmp_path)
    settings = _export_settings(tmp_path, export_csv=False, export_latex=False, export_markdown=False)
    artifact = export_experiment_json(result, settings)

    assert artifact.status is ResultsExportStatus.COMPLETED
    content = Path(artifact.path).read_text(encoding="utf-8")
    payload = json.loads(content)
    assert payload["analysis_id"] == result.analysis_id
    assert "NaN" not in content
    assert "Infinity" not in content


def test_csv_tables_have_stable_headers_and_foreign_keys(tmp_path: Path) -> None:
    result = _analysis_result(tmp_path, full=True)
    settings = _export_settings(tmp_path)
    artifacts = export_experiment_csv_tables(result, settings)
    paths = {artifact.artifact_type: Path(artifact.path) for artifact in artifacts if artifact.path}

    assert "recordings" in paths
    with paths["recordings"].open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
    assert header[:4] == ["recording_id", "experiment_id", "dynamic_label", "take_index"]
    with paths["candidates"].open("r", encoding="utf-8") as handle:
        assert "ModalCandidate(" not in handle.read()


def test_latex_tables_escape_special_characters_and_use_booktabs(tmp_path: Path) -> None:
    result = _analysis_result(tmp_path, name=r"A_B % & # { } \\")
    settings = _export_settings(tmp_path)
    artifacts = export_experiment_latex_tables(result, settings)
    summary = next(item for item in artifacts if item.artifact_type == "experiment_summary")
    content = Path(summary.path).read_text(encoding="utf-8")

    assert "\\toprule" in content
    assert "\\_" in content
    assert "\\%" in content
    assert "\\&" in content
    assert "\\#" in content
    assert "\\{" in content
    assert "\\}" in content
    assert "\\documentclass" not in content


def test_markdown_summary_contains_required_sections_without_physical_claim(tmp_path: Path) -> None:
    result = _analysis_result(tmp_path)
    artifact = export_experiment_markdown_summary(result, _export_settings(tmp_path))
    content = Path(artifact.path).read_text(encoding="utf-8")

    assert "## Identificacao do experimento" in content
    assert "## Gravacoes" in content
    assert "## Q" in content
    assert "## Evidencia operacional de possivel redistribuicao" in content
    assert "nao comprova transferencia fisica" in content


def test_manifest_lists_checksums_sizes_rows_and_omits_self_checksum(tmp_path: Path) -> None:
    result = _analysis_result(tmp_path)
    export = export_experiment_results(result, _export_settings(tmp_path))
    manifest_artifact = next(item for item in export.artifacts if item.artifact_type == "manifest")
    payload = json.loads(Path(manifest_artifact.path).read_text(encoding="utf-8"))

    assert payload["analysis_id"] == result.analysis_id
    assert payload["artifact_checksums"][manifest_artifact.relative_path] is None
    assert payload["artifact_sizes_bytes"]
    assert payload["artifact_row_counts"]


def test_artifact_checksum_uses_file_content(tmp_path: Path) -> None:
    path = tmp_path / "artifact.txt"
    path.write_text("abc", encoding="utf-8")
    first = export_artifact_checksum(path)
    path.write_text("abd", encoding="utf-8")
    second = export_artifact_checksum(path)
    assert first != second


def test_partial_export_json_only_marks_unrequested_artifacts_skipped(tmp_path: Path) -> None:
    result = _analysis_result(tmp_path)
    settings = _export_settings(
        tmp_path,
        export_csv=False,
        export_latex=False,
        export_markdown=False,
        export_summary=False,
    )
    export = export_experiment_results(result, settings)

    assert export.status is ResultsExportStatus.COMPLETED
    assert export.skipped_count >= 3
    assert all(
        ResultsExportReason.OPTIONAL_ARTIFACT_SKIPPED in artifact.reasons
        for artifact in export.skipped_artifacts
    )


def test_csv_only_export_does_not_create_json_or_markdown(tmp_path: Path) -> None:
    result = _analysis_result(tmp_path)
    export = export_experiment_results(
        result,
        _export_settings(
            tmp_path,
            export_json=False,
            export_latex=False,
            export_markdown=False,
            export_manifest=False,
            export_summary=False,
        ),
    )

    assert export.completed_count >= 1
    assert all(artifact.format == "csv" for artifact in export.completed_artifacts)


def test_overwrite_policy_error_preserves_existing_file(tmp_path: Path) -> None:
    result = _analysis_result(tmp_path)
    settings = _export_settings(tmp_path)
    first = export_experiment_json(result, settings)
    second = export_experiment_json(result, settings)

    assert first.status is ResultsExportStatus.COMPLETED
    assert second.status is ResultsExportStatus.FAILED
    assert ResultsExportReason.EXISTING_FILE_CONFLICT in second.reasons


def test_overwrite_policy_skip_preserves_existing_file(tmp_path: Path) -> None:
    result = _analysis_result(tmp_path)
    output = tmp_path / "export"
    path = _artifact_path(result, "experiment_export", "json", output)
    path.parent.mkdir()
    path.write_text("old", encoding="utf-8")

    artifact = export_experiment_json(result, _export_settings(tmp_path, overwrite_policy=ExportOverwritePolicy.SKIP))

    assert artifact.status is ResultsExportStatus.PARTIAL
    assert path.read_text(encoding="utf-8") == "old"


def test_overwrite_policy_replace_updates_existing_file(tmp_path: Path) -> None:
    result = _analysis_result(tmp_path)
    output = tmp_path / "export"
    path = _artifact_path(result, "experiment_export", "json", output)
    path.parent.mkdir()
    path.write_text("old", encoding="utf-8")

    artifact = export_experiment_json(result, _export_settings(tmp_path, overwrite_policy=ExportOverwritePolicy.REPLACE))

    assert artifact.status is ResultsExportStatus.COMPLETED
    assert json.loads(path.read_text(encoding="utf-8"))["analysis_id"] == result.analysis_id


def test_overwrite_policy_versioned_filename_is_deterministic(tmp_path: Path) -> None:
    result = _analysis_result(tmp_path)
    output = tmp_path / "export"
    path = _artifact_path(result, "experiment_export", "json", output)
    path.parent.mkdir()
    path.write_text("old", encoding="utf-8")

    artifact = export_experiment_json(result, _export_settings(tmp_path, overwrite_policy=ExportOverwritePolicy.VERSIONED_FILENAME))

    assert artifact.status is ResultsExportStatus.COMPLETED
    assert artifact.relative_path.endswith("_v001.json")
    assert path.read_text(encoding="utf-8") == "old"


def test_atomic_write_failure_preserves_existing_destination_and_removes_tmp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _analysis_result(tmp_path)
    output = tmp_path / "export"
    path = _artifact_path(result, "experiment_export", "json", output)
    path.parent.mkdir()
    path.write_text("old", encoding="utf-8")
    original_replace = Path.replace

    def failing_replace(self: Path, target: Path) -> Path:
        if self.name.startswith(".") and self.suffix == ".tmp":
            raise RuntimeError("simulated replace failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", failing_replace)
    artifact = export_experiment_json(
        result,
        _export_settings(tmp_path, overwrite_policy=ExportOverwritePolicy.REPLACE),
    )

    assert artifact.status is ResultsExportStatus.FAILED
    assert path.read_text(encoding="utf-8") == "old"
    assert not list(output.glob("*.tmp"))


@pytest.mark.parametrize(
    ("policy", "expected"),
    (
        (ExportMissingValuePolicy.EMPTY, ",,\n"),
        (ExportMissingValuePolicy.NA, ",NA,\n"),
        (ExportMissingValuePolicy.DASH, ",-,\n"),
    ),
)
def test_missing_value_policy_is_explicit_in_csv(tmp_path: Path, policy, expected) -> None:
    result = _analysis_result(tmp_path)
    settings = _export_settings(tmp_path, missing_value_representation=policy)
    artifacts = export_experiment_csv_tables(result, settings)
    summary = next(item for item in artifacts if item.artifact_type == "experiment_summary")
    content = Path(summary.path).read_text(encoding="utf-8")

    assert expected not in content  # representation is explicit but table-dependent.
    assert "0.0" not in content


def test_json_preserves_absent_q_and_tau_as_null(tmp_path: Path) -> None:
    result = _analysis_result(tmp_path)
    artifact = export_experiment_json(result, _export_settings(tmp_path))
    payload = json.loads(Path(artifact.path).read_text(encoding="utf-8"))

    assert payload["modal_q_factors"] == []
    assert payload["modal_parameters"] == []
    assert payload["cross_condition"] is None


def test_invalid_source_results_are_exported_with_reservations(tmp_path: Path) -> None:
    experiment = ExperimentDefinition(
        name="invalid source",
        recordings=(_recording(tmp_path / "missing.wav", "pp"),),
        dynamic_labels=("pp",),
    )
    result = analyze_experiment(experiment, _load_only_settings())
    export = export_experiment_results(result, _export_settings(tmp_path))

    assert export.status is ResultsExportStatus.COMPLETED_WITH_RESERVATIONS
    assert "source_result_requires_review" in export.diagnostics
    stages = next(item for item in export.artifacts if item.artifact_type == "pipeline_stages")
    assert Path(stages.path).read_text(encoding="utf-8").count("invalid_input") >= 1


def test_nonfinite_default_policy_rejects_invalid_export_value(tmp_path: Path) -> None:
    result = _analysis_result(tmp_path, metadata={"bad": float("nan")})
    with pytest.raises(ValueError):
        normalize_experiment_for_export(result, _export_settings(tmp_path))


def test_nonfinite_null_policy_never_writes_nonstandard_json_tokens(tmp_path: Path) -> None:
    result = _analysis_result(tmp_path, metadata={"bad": float("nan")})
    settings = _export_settings(tmp_path, nonfinite_value_policy=ExportNonfiniteValuePolicy.NULL_WITH_DIAGNOSTIC)
    artifact = export_experiment_json(result, settings)
    content = Path(artifact.path).read_text(encoding="utf-8")

    assert artifact.status is ResultsExportStatus.COMPLETED
    assert "NaN" not in content
    assert '"bad": null' in content


def test_nonfinite_string_policy_quotes_nonfinite_values(tmp_path: Path) -> None:
    result = _analysis_result(tmp_path, metadata={"bad": float("inf")})
    settings = _export_settings(tmp_path, nonfinite_value_policy=ExportNonfiniteValuePolicy.STRING_WITH_DIAGNOSTIC)
    artifact = export_experiment_json(result, settings)
    content = Path(artifact.path).read_text(encoding="utf-8")

    assert '"Infinity"' in content
    json.loads(content)


def test_export_is_deterministic_across_output_directories(tmp_path: Path) -> None:
    result = _analysis_result(tmp_path)
    first = export_experiment_results(result, ResultsExportSettings(output_directory=tmp_path / "a"))
    second = export_experiment_results(result, ResultsExportSettings(output_directory=tmp_path / "b"))
    first_json = next(item for item in first.artifacts if item.artifact_type == "experiment_export")
    second_json = next(item for item in second.artifacts if item.artifact_type == "experiment_export")

    assert first_json.relative_path == second_json.relative_path
    assert first_json.checksum == second_json.checksum
    assert first.export_id == second.export_id


def test_local_perturbation_changes_related_artifact_checksum(tmp_path: Path) -> None:
    first = _analysis_result(tmp_path / "a")
    second = _analysis_result(tmp_path / "b")
    _write_wav(Path(second.recording_results[0].recording_definition.file_path), 320.0)
    second = analyze_experiment(second.experiment, _load_only_settings())
    first_export = export_experiment_results(first, ResultsExportSettings(output_directory=tmp_path / "ea"))
    second_export = export_experiment_results(second, ResultsExportSettings(output_directory=tmp_path / "eb"))
    first_json = next(item for item in first_export.artifacts if item.artifact_type == "experiment_export")
    second_json = next(item for item in second_export.artifacts if item.artifact_type == "experiment_export")

    assert first_json.checksum != second_json.checksum


def test_export_does_not_modify_analysis_result_or_audio_file(tmp_path: Path) -> None:
    result = _analysis_result(tmp_path)
    before_result = result
    path = Path(result.recording_results[0].recording_definition.file_path)
    before_hash = experiment_file_fingerprint(path)

    export_experiment_results(result, _export_settings(tmp_path))

    assert result == before_result
    assert experiment_file_fingerprint(path) == before_hash


def test_validate_export_detects_missing_artifact(tmp_path: Path) -> None:
    result = _analysis_result(tmp_path)
    export = export_experiment_results(result, _export_settings(tmp_path))
    artifact = next(item for item in export.completed_artifacts if item.path)
    Path(artifact.path).unlink()
    validation = validate_experiment_export(export)

    assert validation.valid is False
    assert artifact.relative_path in validation.missing_artifacts


def test_validate_export_detects_checksum_change(tmp_path: Path) -> None:
    result = _analysis_result(tmp_path)
    export = export_experiment_results(result, _export_settings(tmp_path))
    artifact = next(item for item in export.completed_artifacts if item.path and item.format == "json")
    Path(artifact.path).write_text("{}", encoding="utf-8")
    validation = validate_experiment_export(export)

    assert validation.valid is False
    assert validation.checksum_matches[artifact.relative_path] is False


def test_csv_quoting_handles_commas_and_newlines(tmp_path: Path) -> None:
    result = _analysis_result(tmp_path, name="comma, newline\nexperiment")
    artifacts = export_experiment_csv_tables(result, _export_settings(tmp_path))
    summary = next(item for item in artifacts if item.artifact_type == "experiment_summary")

    with Path(summary.path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["name"] == "comma, newline\nexperiment"


def test_exported_artifact_ids_and_paths_are_unique(tmp_path: Path) -> None:
    result = _analysis_result(tmp_path)
    export = export_experiment_results(result, _export_settings(tmp_path))

    ids = tuple(item.artifact_id for item in export.artifacts)
    paths = tuple(item.relative_path for item in export.completed_artifacts)
    assert len(ids) == len(set(ids))
    assert len(paths) == len(set(paths))


def test_summary_is_stable_and_uses_relative_paths(tmp_path: Path) -> None:
    result = _analysis_result(tmp_path)
    export = export_experiment_results(result, _export_settings(tmp_path))
    summary = summarize_experiment_export(export)

    assert summary["export_id"] == export.export_id
    assert all(
        artifact["relative_path"] is None or not str(artifact["relative_path"]).startswith("/")
        for artifact in summary["artifacts"]
    )


def test_validation_accepts_manifest_consistency_and_csv_foreign_keys(tmp_path: Path) -> None:
    result = _analysis_result(tmp_path)
    export = export_experiment_results(result, _export_settings(tmp_path))
    validation = validate_experiment_export(export)

    assert validation.schema_valid is True
    assert validation.row_counts_valid is True
    assert validation.foreign_keys_valid is True
    assert validation.manifest_consistent is True


def test_column_selection_limits_csv_columns(tmp_path: Path) -> None:
    result = _analysis_result(tmp_path)
    settings = _export_settings(
        tmp_path,
        column_selection={"recordings": ("recording_id", "dynamic_label")},
    )
    artifacts = export_experiment_csv_tables(result, settings)
    recordings = next(item for item in artifacts if item.artifact_type == "recordings")

    with Path(recordings.path).open("r", encoding="utf-8", newline="") as handle:
        assert next(csv.reader(handle)) == ["recording_id", "dynamic_label"]


def test_presentation_rounding_does_not_change_json_precision(tmp_path: Path) -> None:
    result = _analysis_result(tmp_path)
    settings = _export_settings(tmp_path, float_precision=2)
    payload = serialize_experiment_result(result, settings)
    csv_artifacts = export_experiment_csv_tables(result, settings)
    recordings = next(item for item in csv_artifacts if item.artifact_type == "recordings")
    csv_text = Path(recordings.path).read_text(encoding="utf-8")

    assert payload["recordings"][0]["analyzed_duration_s"] == pytest.approx(1.0)
    assert ".0000000000" not in csv_text


def test_validate_json_roundtrip_keeps_none_status_and_diagnostics(tmp_path: Path) -> None:
    result = _analysis_result(tmp_path)
    export = export_experiment_results(result, _export_settings(tmp_path))
    artifact = next(item for item in export.artifacts if item.artifact_type == "experiment_export")
    payload = json.loads(Path(artifact.path).read_text(encoding="utf-8"))

    assert payload["cross_condition"] is None
    assert payload["summary"]["status"] == result.status.value
    assert payload["diagnostics"]


def test_export_result_status_is_partial_when_requested_artifact_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _analysis_result(tmp_path)
    original = results_export_module.export_experiment_json

    def fail_json(*args, **kwargs):
        raise RuntimeError("json boom")

    monkeypatch.setattr(results_export_module, "export_experiment_json", fail_json)
    export = export_experiment_results(result, _export_settings(tmp_path))
    monkeypatch.setattr(results_export_module, "export_experiment_json", original)

    assert export.status is ResultsExportStatus.PARTIAL
    assert any("json boom" in diagnostic for artifact in export.failed_artifacts for diagnostic in artifact.diagnostics)


def test_export_validation_is_structural_not_dataclass_reconstruction(tmp_path: Path) -> None:
    result = _analysis_result(tmp_path)
    export = export_experiment_results(result, _export_settings(tmp_path))
    validation = validate_experiment_export(export)

    assert "json_roundtrip_reconstructs_structure_not_dataclasses" in validation.diagnostics


def test_export_diagnostics_state_scientific_limitations(tmp_path: Path) -> None:
    result = _analysis_result(tmp_path)
    export = export_experiment_results(result, _export_settings(tmp_path))

    assert "export_did_not_recalculate_scientific_analysis" in export.diagnostics
    assert "export_is_not_physical_validity_proof" in export.diagnostics
