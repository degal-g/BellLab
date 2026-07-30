"""Tests for reproducible BellLab scientific report generation."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import belllab.scientific_report as report_module
from belllab import (
    AnalysisSettings,
    CrossConditionCandidateAssociationSettings,
    ExperimentDefinition,
    ExperimentPipelineSettings,
    ExperimentRecordingDefinition,
    ExportOverwritePolicy,
    FramePeakDetectionSettings,
    GlobalSpectralCharacterizationSettings,
    ModalCandidateSettings,
    ModalEnergyExchangeSettings,
    ModalHypothesisSettings,
    ModalParameterEstimationSettings,
    ModalQFactorEstimationSettings,
    PeakDetectionSettings,
    ResultsExportSettings,
    STFTSettings,
    ScientificFigureType,
    ScientificReportContentBlock,
    ScientificReportContentBlockType,
    ScientificReportReason,
    ScientificReportSchemaVersion,
    ScientificReportSection,
    ScientificReportSettings,
    ScientificReportStatus,
    ScientificVisualizationSettings,
    SpectralTrackingSettings,
    SpectrumAnalysisSettings,
    TimeResolvedSpectralCharacterizationSettings,
    WithinConditionAssociationSettings,
    analyze_experiment,
    build_scientific_report_document,
    build_scientific_report_manifest,
    build_scientific_report_sections,
    compile_scientific_report_pdf,
    create_experiment_visualizations,
    create_scientific_report,
    export_experiment_results,
    render_scientific_report_latex,
    render_scientific_report_markdown,
    scientific_report_artifact_checksum,
    scientific_report_settings_fingerprint,
    summarize_scientific_report,
    validate_scientific_report,
)


def _write_wav(
    path: Path,
    frequency_hz: float,
    *,
    sample_rate: int = 4096,
    duration_s: float = 1.0,
    tau_s: float = 2.0,
    second_frequency_hz: float | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(int(round(sample_rate * duration_s)), dtype=np.float64) / sample_rate
    signal = 0.30 * np.exp(-time / tau_s) * np.sin(2 * np.pi * frequency_hz * time)
    if second_frequency_hz is not None:
        signal += 0.18 * np.exp(-time / (0.75 * tau_s)) * np.sin(2 * np.pi * second_frequency_hz * time)
    sf.write(path, signal.astype(np.float32), sample_rate, subtype="FLOAT")
    return path


def _recording(path: Path, label: str) -> ExperimentRecordingDefinition:
    return ExperimentRecordingDefinition(
        file_path=path,
        dynamic_label=label,
        recording_id=label,
        channel=0,
    )


def _pipeline_settings(**changes) -> ExperimentPipelineSettings:
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
        run_dynamic_condition_comparison=False,
    )
    base.update(changes)
    return ExperimentPipelineSettings(**base)


@pytest.fixture(scope="module")
def analysis_result(tmp_path_factory):
    root = tmp_path_factory.mktemp("report-analysis")
    recordings = []
    for index, label in enumerate(("pp", "p", "mf", "f", "ff")):
        path = _write_wav(
            root / f"{label}.wav",
            300.0 + index,
            second_frequency_hz=520.0 + index,
        )
        recordings.append(_recording(path, label))
    experiment = ExperimentDefinition(
        name="scientific report test experiment",
        specimen_id="test-bell",
        dynamic_labels=("pp", "p", "mf", "f", "ff"),
        recordings=tuple(recordings),
    )
    return analyze_experiment(experiment, _pipeline_settings())


@pytest.fixture(scope="module")
def export_result(analysis_result, tmp_path_factory):
    return export_experiment_results(
        analysis_result,
        ResultsExportSettings(
            output_directory=tmp_path_factory.mktemp("report-export"),
            overwrite_policy=ExportOverwritePolicy.REPLACE,
        ),
    )


@pytest.fixture(scope="module")
def figure_collection(analysis_result, tmp_path_factory):
    return create_experiment_visualizations(
        analysis_result,
        ScientificVisualizationSettings(
            output_directory=tmp_path_factory.mktemp("report-figures"),
            overwrite_policy=ExportOverwritePolicy.REPLACE,
            close_after_save=True,
            figure_types=(
                ScientificFigureType.WAVEFORM,
                ScientificFigureType.GLOBAL_SPECTRUM,
                ScientificFigureType.MODAL_HYPOTHESES,
                ScientificFigureType.MODAL_PARAMETERS,
                ScientificFigureType.MODAL_Q_FACTORS,
                ScientificFigureType.MODAL_ENERGY_EXCHANGE_EVIDENCE,
                ScientificFigureType.EXPERIMENT_SUMMARY,
            ),
        ),
    )


def _settings(tmp_path: Path, **changes) -> ScientificReportSettings:
    values = dict(
        output_directory=tmp_path / "report",
        overwrite_policy=ExportOverwritePolicy.REPLACE,
        title="BellLab scientific report test",
        authors=("A. Researcher",),
        report_date="2026-07-30",
    )
    values.update(changes)
    return ScientificReportSettings(**values)


def test_public_contracts_are_importable() -> None:
    assert ScientificReportStatus.CREATED.value == "created"
    assert ScientificReportReason.MANIFEST_CREATED.value == "manifest_created"
    assert ScientificReportSchemaVersion.V1_0.value == "1.0"
    assert ScientificReportContentBlockType.SCIENTIFIC_NOTICE.value == "scientific_notice"
    assert callable(build_scientific_report_document)
    assert callable(create_scientific_report)
    assert callable(validate_scientific_report)


@pytest.mark.parametrize(
    "changes",
    (
        {"generate_markdown": False, "generate_latex": False, "compile_pdf": False, "generate_manifest": False},
        {"language": "fr"},
        {"figure_format_preference": "jpg"},
        {"maximum_figures_per_section": 0},
        {"figure_width_fraction": 0.0},
        {"table_format_preference": "html"},
        {"maximum_rows_inline": 0},
        {"large_table_policy": "hide"},
        {"latex_engine": "network-latex"},
        {"latex_compilation_runs": 0},
        {"pdf_compilation_timeout_s": 0.0},
        {"decimal_separator": ";"},
    ),
)
def test_settings_reject_invalid_configuration(tmp_path: Path, changes) -> None:
    with pytest.raises(ValueError):
        _settings(tmp_path, **changes)


def test_report_settings_fingerprint_is_path_independent(tmp_path: Path) -> None:
    first = ScientificReportSettings(output_directory=tmp_path / "a", title="A")
    second = ScientificReportSettings(output_directory=tmp_path / "b", title="A")
    third = ScientificReportSettings(output_directory=tmp_path / "b", title="B")

    assert scientific_report_settings_fingerprint(first) == scientific_report_settings_fingerprint(second)
    assert scientific_report_settings_fingerprint(first) != scientific_report_settings_fingerprint(third)


def test_build_document_preserves_ids_tables_figures_and_provenance(
    analysis_result,
    export_result,
    figure_collection,
    tmp_path: Path,
) -> None:
    document = build_scientific_report_document(
        analysis_result,
        export_result,
        figure_collection,
        _settings(tmp_path),
    )

    assert document.analysis_id == analysis_result.analysis_id
    assert document.experiment_id == analysis_result.experiment.experiment_id
    assert len(document.sections) >= 18
    assert any(table.table_id == "table-modal-hypotheses" for table in document.tables)
    assert any(figure.figure_type == "modal_hypotheses" for figure in document.figures)
    assert document.provenance["source_export_id"] == export_result.export_id
    assert document.provenance["source_figure_collection_id"] == figure_collection.collection_id
    assert "no_scientific_analysis_recalculated" in document.diagnostics


def test_build_sections_can_be_called_directly(analysis_result, tmp_path: Path) -> None:
    settings = _settings(tmp_path, include_figures=False)
    document = build_scientific_report_document(analysis_result, settings=settings)
    sections = build_scientific_report_sections(
        analysis_result,
        report_module.normalize_experiment_for_export(analysis_result, report_module._normalization_settings(settings)),
        document.tables,
        (),
        settings,
    )

    assert tuple(section.section_id for section in sections) == tuple(section.section_id for section in document.sections)


def test_markdown_contains_sections_tables_figures_equations_and_cautions(
    analysis_result,
    export_result,
    figure_collection,
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    document = build_scientific_report_document(analysis_result, export_result, figure_collection, settings)
    markdown = render_scientific_report_markdown(document, settings)

    assert "# BellLab scientific report test" in markdown
    assert "Q_{decay} = \\pi f \\tau" in markdown
    assert "Table `table-modal-parameters`" in markdown
    assert "![Modal Hypotheses]" in markdown
    assert "Aviso cientifico" in markdown
    assert "modo fisico confirmado" not in markdown.lower()
    assert "transferencia de energia confirmada" not in markdown.lower()


def test_latex_contains_required_packages_labels_and_escaped_user_text(
    analysis_result,
    export_result,
    figure_collection,
    tmp_path: Path,
) -> None:
    settings = _settings(
        tmp_path,
        title="Report_with_%_&_#",
        user_context_text="Context_with_%_&_#_{value}\\path",
    )
    document = build_scientific_report_document(analysis_result, export_result, figure_collection, settings)
    latex = render_scientific_report_latex(document, settings)

    assert "\\usepackage{booktabs}" in latex
    assert "\\usepackage{longtable}" in latex
    assert "\\usepackage{graphicx}" in latex
    assert "\\label{sec:modal-hypotheses}" in latex
    assert r"Report\_with\_\%\_\&\_\#" in latex
    assert "shell-escape" not in latex


def test_create_report_writes_markdown_latex_and_manifest_with_checksums(
    analysis_result,
    export_result,
    figure_collection,
    tmp_path: Path,
) -> None:
    result = create_scientific_report(
        analysis_result,
        export_result,
        figure_collection,
        _settings(tmp_path),
    )
    validation = validate_scientific_report(result)
    summary = summarize_scientific_report(result)

    assert result.status in {ScientificReportStatus.CREATED, ScientificReportStatus.CREATED_WITH_RESERVATIONS}
    assert result.valid is True
    assert validation.valid is True
    assert summary["artifact_count"] == 3
    assert {artifact.format for artifact in result.artifacts} == {"markdown", "latex", "json"}
    for artifact in result.artifacts:
        assert artifact.path is not None
        assert Path(artifact.path).is_file()
        assert artifact.checksum == scientific_report_artifact_checksum(artifact.path)


def test_manifest_lists_sections_tables_figures_and_artifacts(
    analysis_result,
    export_result,
    figure_collection,
    tmp_path: Path,
) -> None:
    result = create_scientific_report(
        analysis_result,
        export_result,
        figure_collection,
        _settings(tmp_path),
    )

    assert result.manifest is not None
    assert result.manifest.report_schema_version == "1.0"
    assert result.manifest.sections
    assert result.manifest.tables
    assert result.manifest.figures
    assert result.manifest.checksums
    assert result.manifest.source_export_id == export_result.export_id
    assert result.manifest.source_figure_collection_id == figure_collection.collection_id


def test_manifest_builder_can_be_called_directly(analysis_result, export_result, figure_collection, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    document = build_scientific_report_document(analysis_result, export_result, figure_collection, settings)
    manifest = build_scientific_report_manifest(document, (), settings, export_result=export_result, figure_collection=figure_collection)

    assert manifest.report_id == document.report_id
    assert manifest.sections
    assert manifest.limitations


def test_report_without_export_or_figures_is_partial_but_renderable(analysis_result, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    result = create_scientific_report(analysis_result, settings=settings)

    assert result.status is ScientificReportStatus.CREATED_WITH_RESERVATIONS
    assert result.valid is True
    assert "export_result_not_supplied_tables_built_from_normalized_analysis" in result.document.diagnostics
    assert any("Figure collection was not supplied" in item for item in result.document.limitations)


def test_partial_report_when_energy_section_is_disabled(analysis_result, export_result, figure_collection, tmp_path: Path) -> None:
    settings = _settings(tmp_path, include_energy_exchange=False, include_synthetic_validation=False)
    document = build_scientific_report_document(analysis_result, export_result, figure_collection, settings)

    section_ids = {section.section_id for section in document.sections}
    assert "energy_exchange" not in section_ids
    assert "synthetic_validation" not in section_ids


def test_invalid_export_or_figure_ids_are_rejected(analysis_result, export_result, figure_collection, tmp_path: Path) -> None:
    bad_export = replace(export_result, analysis_id="other-analysis")
    bad_figures = replace(figure_collection, analysis_id="other-analysis")

    with pytest.raises(ValueError):
        build_scientific_report_document(analysis_result, bad_export, figure_collection, _settings(tmp_path))
    with pytest.raises(ValueError):
        build_scientific_report_document(analysis_result, export_result, bad_figures, _settings(tmp_path))


def test_checksum_mismatch_is_detected_before_render(
    analysis_result,
    export_result,
    figure_collection,
    tmp_path: Path,
) -> None:
    artifact = next(item for item in figure_collection.artifacts if item.path)
    tampered_path = tmp_path / "tampered.png"
    tampered_path.write_text("tampered figure", encoding="utf-8")
    bad_artifact = replace(artifact, path=str(tampered_path), relative_path="tampered.png")
    bad_collection = replace(
        figure_collection,
        artifacts=(bad_artifact,) + tuple(item for item in figure_collection.artifacts if item is not artifact),
    )

    result = create_scientific_report(
        analysis_result,
        export_result,
        bad_collection,
        _settings(tmp_path),
    )

    assert result.status is ScientificReportStatus.INVALID_INPUT
    assert "checksum mismatch" in result.failure_reason


def test_missing_metadata_is_not_invented(analysis_result, tmp_path: Path) -> None:
    settings = ScientificReportSettings(
        output_directory=tmp_path / "report",
        overwrite_policy=ExportOverwritePolicy.REPLACE,
        title=None,
        authors=(),
        affiliations=(),
        report_date=None,
    )
    document = build_scientific_report_document(analysis_result, settings=settings)
    markdown = render_scientific_report_markdown(document, settings)

    assert document.authors == ()
    assert document.affiliations == ()
    assert "report_date: -" in markdown
    assert "Unknown" not in markdown
    assert "Anonymous" not in markdown


def test_user_text_is_labelled_and_not_expanded(analysis_result, tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        user_context_text="Historic note supplied by user.",
        user_acquisition_text="Acquisition details supplied by user.",
    )
    markdown = render_scientific_report_markdown(
        build_scientific_report_document(analysis_result, settings=settings),
        settings,
    )

    assert "user_provided_text: Historic note supplied by user." in markdown
    assert "user_provided_text: Acquisition details supplied by user." in markdown
    assert "strongly nonlinear" not in markdown


def test_conservative_narrative_validator_rejects_forbidden_claim(tmp_path: Path) -> None:
    section = ScientificReportSection(
        section_id="bad",
        title="Bad",
        level=1,
        order=0,
        content_blocks=(
            ScientificReportContentBlock(
                block_id="bad-block",
                block_type=ScientificReportContentBlockType.PARAGRAPH,
                content="confirmed energy transfer",
            ),
        ),
    )
    document = report_module.ScientificReportDocument(
        report_id="report-bad",
        analysis_id="analysis-bad",
        experiment_id="experiment-bad",
        title="Bad",
        subtitle=None,
        authors=(),
        affiliations=(),
        language="en",
        sections=(section,),
        figures=(),
        tables=(),
        appendices=(),
        references={"sec:bad": "Bad"},
        provenance={},
        limitations=(),
        diagnostics=(),
        settings_fingerprint="fingerprint",
        valid=True,
    )

    with pytest.raises(ValueError):
        render_scientific_report_markdown(document, _settings(tmp_path, language="en"))


def test_cross_reference_validation_rejects_duplicate_or_missing_labels(analysis_result, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    document = build_scientific_report_document(analysis_result, settings=settings)
    bad = replace(document, references={})

    with pytest.raises(ValueError):
        render_scientific_report_markdown(bad, settings)


def test_compile_pdf_not_requested_is_structured(tmp_path: Path) -> None:
    result = compile_scientific_report_pdf(None, _settings(tmp_path, compile_pdf=False))

    assert result.requested is False
    assert result.valid is True
    assert result.status is ScientificReportStatus.CREATED


def test_compile_pdf_unavailable_is_reservation_not_total_failure(tmp_path: Path, monkeypatch) -> None:
    latex = tmp_path / "report.tex"
    latex.write_text("\\documentclass{article}\\begin{document}x\\end{document}\n", encoding="utf-8")
    monkeypatch.setattr(report_module.shutil, "which", lambda _name: None)

    result = compile_scientific_report_pdf(latex, _settings(tmp_path, compile_pdf=True))

    assert result.requested is True
    assert result.valid is False
    assert result.status is ScientificReportStatus.PARTIAL
    assert "latex_compiler_unavailable" in result.warnings


def test_create_report_with_pdf_unavailable_keeps_sources(
    analysis_result,
    export_result,
    figure_collection,
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(report_module.shutil, "which", lambda _name: None)

    result = create_scientific_report(
        analysis_result,
        export_result,
        figure_collection,
        _settings(tmp_path, compile_pdf=True),
    )

    assert result.status is ScientificReportStatus.CREATED_WITH_RESERVATIONS
    assert result.compilation_result is not None
    assert result.compilation_result.status is ScientificReportStatus.PARTIAL
    assert any(artifact.format == "latex" for artifact in result.artifacts)


def test_extra_source_artifacts_can_be_generated(analysis_result, export_result, figure_collection, tmp_path: Path) -> None:
    result = create_scientific_report(
        analysis_result,
        export_result,
        figure_collection,
        _settings(
            tmp_path,
            generate_bibliography_file=True,
            generate_makefile=True,
            generate_latexmkrc=True,
        ),
    )

    formats = {artifact.format for artifact in result.artifacts}
    assert {"bibtex", "makefile", "latexmkrc"}.issubset(formats)


def test_overwrite_error_skip_replace_and_versioned(analysis_result, export_result, figure_collection, tmp_path: Path) -> None:
    replace_settings = _settings(tmp_path / "replace", overwrite_policy=ExportOverwritePolicy.REPLACE)
    error_settings = _settings(tmp_path / "error", overwrite_policy=ExportOverwritePolicy.ERROR)
    skip_settings = _settings(tmp_path / "skip", overwrite_policy=ExportOverwritePolicy.SKIP)
    versioned_settings = _settings(tmp_path / "versioned", overwrite_policy=ExportOverwritePolicy.VERSIONED_FILENAME)

    first = create_scientific_report(analysis_result, export_result, figure_collection, replace_settings)
    error_first = create_scientific_report(analysis_result, export_result, figure_collection, error_settings)
    error_second = create_scientific_report(analysis_result, export_result, figure_collection, error_settings)
    skip_first = create_scientific_report(analysis_result, export_result, figure_collection, skip_settings)
    skip_second = create_scientific_report(analysis_result, export_result, figure_collection, skip_settings)
    versioned_first = create_scientific_report(analysis_result, export_result, figure_collection, versioned_settings)
    versioned_second = create_scientific_report(analysis_result, export_result, figure_collection, versioned_settings)

    assert first.valid is True
    assert error_first.valid is True
    assert error_second.status is ScientificReportStatus.FAILED
    assert any(ScientificReportReason.EXISTING_FILE_CONFLICT in artifact.reasons for artifact in error_second.failed_artifacts)
    assert skip_first.valid is True
    assert skip_second.status is ScientificReportStatus.PARTIAL
    assert any(artifact.status is ScientificReportStatus.PARTIAL for artifact in skip_second.artifacts)
    assert versioned_first.valid is True
    assert any("_v001" in (artifact.relative_path or "") for artifact in versioned_second.artifacts)


def test_atomic_write_failure_preserves_existing_file(
    analysis_result,
    export_result,
    figure_collection,
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path, overwrite_policy=ExportOverwritePolicy.REPLACE)
    first = create_scientific_report(analysis_result, export_result, figure_collection, settings)
    markdown = next(item for item in first.artifacts if item.format == "markdown")
    before = Path(markdown.path).read_bytes()

    def failing_replace(_src, _dst):
        raise OSError("simulated atomic failure")

    monkeypatch.setattr(report_module.os, "replace", failing_replace)
    failed = create_scientific_report(analysis_result, export_result, figure_collection, settings)

    assert failed.status is ScientificReportStatus.FAILED
    assert Path(markdown.path).read_bytes() == before
    assert not Path(markdown.path).with_name(f".{Path(markdown.path).name}.tmp").exists()


def test_determinism_across_output_directories(analysis_result, export_result, figure_collection, tmp_path: Path) -> None:
    first_settings = _settings(tmp_path / "a")
    second_settings = _settings(tmp_path / "b")
    first = create_scientific_report(analysis_result, export_result, figure_collection, first_settings)
    second = create_scientific_report(analysis_result, export_result, figure_collection, second_settings)

    assert first.report_id == second.report_id
    first_md = next(item for item in first.artifacts if item.format == "markdown")
    second_md = next(item for item in second.artifacts if item.format == "markdown")
    assert scientific_report_artifact_checksum(first_md.path) == scientific_report_artifact_checksum(second_md.path)


def test_local_perturbation_changes_report_content(tmp_path: Path) -> None:
    first_path = _write_wav(tmp_path / "first" / "pp.wav", 300.0)
    second_path = _write_wav(tmp_path / "second" / "pp.wav", 330.0)
    common = dict(
        run_stft=False,
        run_tracking=False,
        run_preimpact_analysis=False,
        run_excitation_characterization=False,
        run_modal_candidate_characterization=False,
        run_within_condition_association=False,
        run_cross_condition_association=False,
        run_candidate_chains=False,
        run_modal_hypotheses=False,
        run_modal_parameter_estimation=False,
        run_modal_q_estimation=False,
        run_modal_energy_exchange=False,
        run_dynamic_condition_comparison=False,
    )
    first = analyze_experiment(
        ExperimentDefinition(name="first", recordings=(_recording(first_path, "pp"),), dynamic_labels=("pp",)),
        _pipeline_settings(**common),
    )
    second = analyze_experiment(
        ExperimentDefinition(name="second", recordings=(_recording(second_path, "pp"),), dynamic_labels=("pp",)),
        _pipeline_settings(**common),
    )
    first_report = create_scientific_report(first, settings=_settings(tmp_path / "r1"))
    second_report = create_scientific_report(second, settings=_settings(tmp_path / "r2"))

    assert first_report.report_id != second_report.report_id
    first_md = next(item for item in first_report.artifacts if item.format == "markdown")
    second_md = next(item for item in second_report.artifacts if item.format == "markdown")
    assert scientific_report_artifact_checksum(first_md.path) != scientific_report_artifact_checksum(second_md.path)


def test_report_generation_does_not_modify_inputs(analysis_result, export_result, figure_collection, tmp_path: Path) -> None:
    before_analysis = analysis_result
    before_export = export_result
    before_figures = figure_collection

    create_scientific_report(
        analysis_result,
        export_result,
        figure_collection,
        _settings(tmp_path),
    )

    assert analysis_result == before_analysis
    assert export_result == before_export
    assert figure_collection == before_figures


def test_none_values_are_rendered_as_missing_not_zero(analysis_result, tmp_path: Path) -> None:
    settings = _settings(tmp_path, missing_value_representation=report_module.ExportMissingValuePolicy.DASH)
    markdown = render_scientific_report_markdown(
        build_scientific_report_document(analysis_result, settings=settings),
        settings,
    )

    assert "| - |" in markdown or " - " in markdown
    assert "None" not in markdown


def test_summary_reports_stable_shape(analysis_result, export_result, figure_collection, tmp_path: Path) -> None:
    result = create_scientific_report(
        analysis_result,
        export_result,
        figure_collection,
        _settings(tmp_path),
    )
    summary = summarize_scientific_report(result)

    assert summary["report_id"] == result.report_id
    assert summary["analysis_id"] == analysis_result.analysis_id
    assert summary["artifact_count"] == len(result.artifacts)
    assert summary["valid"] is True
