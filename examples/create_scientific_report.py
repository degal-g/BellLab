#!/usr/bin/env python3
"""Create a reproducible BellLab scientific report from existing results."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from enum import Enum
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from belllab import (
    AnalysisSettings,
    CrossConditionCandidateAssociationSettings,
    ExperimentDefinition,
    ExperimentPipelineSettings,
    ExperimentRecordingDefinition,
    ExportOverwritePolicy,
    FramePeakDetectionSettings,
    ModalCandidateSettings,
    ModalEnergyExchangeSettings,
    ModalHypothesisSettings,
    ModalParameterEstimationSettings,
    ModalQFactorEstimationSettings,
    PeakDetectionSettings,
    ResultsExportSettings,
    STFTSettings,
    ScientificFigureType,
    ScientificReportSettings,
    ScientificReportStatus,
    ScientificVisualizationSettings,
    SpectralTrackingSettings,
    SpectrumAnalysisSettings,
    TimeResolvedSpectralCharacterizationSettings,
    WithinConditionAssociationSettings,
    analyze_experiment,
    create_experiment_visualizations,
    create_scientific_report,
    export_experiment_results,
    scientific_report_artifact_checksum,
    summarize_scientific_report,
    validate_scientific_report,
)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="belllab-report-example-") as temp_dir:
        root = Path(temp_dir)
        experiment = ExperimentDefinition(
            name="BellLab reproducible scientific report example",
            specimen_id="example-bell",
            dynamic_labels=("pp", "p", "mf", "f", "ff"),
            recordings=_temporary_recordings(root / "audio"),
            acquisition_notes="Temporary synthetic WAVs used only to create an example report.",
        )
        analysis = analyze_experiment(experiment, _example_pipeline_settings())
        export = export_experiment_results(
            analysis,
            ResultsExportSettings(
                output_directory=root / "export",
                overwrite_policy=ExportOverwritePolicy.REPLACE,
            ),
        )
        figures = create_experiment_visualizations(
            analysis,
            ScientificVisualizationSettings(
                output_directory=root / "figures",
                formats=("png", "svg"),
                overwrite_policy=ExportOverwritePolicy.REPLACE,
                close_after_save=True,
                figure_types=(
                    ScientificFigureType.WAVEFORM,
                    ScientificFigureType.GLOBAL_SPECTRUM,
                    ScientificFigureType.SPECTROGRAM,
                    ScientificFigureType.FREQUENCY_TRACKS,
                    ScientificFigureType.MODAL_CANDIDATES,
                    ScientificFigureType.CANDIDATE_CHAINS,
                    ScientificFigureType.MODAL_HYPOTHESES,
                    ScientificFigureType.MODAL_PARAMETERS,
                    ScientificFigureType.MODAL_Q_FACTORS,
                    ScientificFigureType.MODAL_ENERGY_EXCHANGE_EVIDENCE,
                    ScientificFigureType.EXPERIMENT_SUMMARY,
                ),
            ),
        )
        compile_pdf = _latex_tool_available()
        report = create_scientific_report(
            analysis,
            export,
            figures,
            ScientificReportSettings(
                output_directory=root / "report",
                overwrite_policy=ExportOverwritePolicy.REPLACE,
                title="BellLab reproducible scientific report example",
                authors=("BellLab example",),
                report_date="2026-07-30",
                compile_pdf=compile_pdf,
                generate_makefile=True,
                generate_latexmkrc=True,
            ),
        )
        validation = validate_scientific_report(report)
        checksums_valid = all(
            artifact.path
            and artifact.checksum == scientific_report_artifact_checksum(artifact.path)
            for artifact in report.artifacts
            if artifact.status is ScientificReportStatus.CREATED
        )

        print(json.dumps(_json_ready(summarize_scientific_report(report)), indent=2, sort_keys=True))
        print("Artifacts:")
        for artifact in report.artifacts:
            if artifact.relative_path:
                print(f"- {artifact.relative_path}: {artifact.status.value}")
        print(f"PDF requested: {compile_pdf}")
        print(f"Checksums valid: {checksums_valid}")
        print(f"Report validation valid: {validation.valid}")
        return 0 if report.valid and validation.valid and checksums_valid else 1


def _temporary_recordings(directory: Path) -> tuple[ExperimentRecordingDefinition, ...]:
    directory.mkdir(parents=True, exist_ok=True)
    sample_rate = 4096
    duration_s = 1.2
    time = np.arange(int(sample_rate * duration_s), dtype=np.float64) / sample_rate
    recordings = []
    for index, label in enumerate(("pp", "p", "mf", "f", "ff")):
        primary = 300.0 + index
        secondary = 520.0 + index
        signal = (
            0.28 * np.exp(-time / 2.0) * np.sin(2 * np.pi * primary * time)
            + 0.18 * np.exp(-time / 1.4) * np.sin(2 * np.pi * secondary * time)
        )
        path = directory / f"{label}.wav"
        sf.write(path, signal.astype(np.float32), sample_rate, subtype="FLOAT")
        recordings.append(
            ExperimentRecordingDefinition(
                file_path=path,
                dynamic_label=label,
                recording_id=f"{label}_take_0",
                take_index=0,
                channel=0,
            )
        )
    return tuple(recordings)


def _example_pipeline_settings() -> ExperimentPipelineSettings:
    analysis = AnalysisSettings(
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
            min_track_length=2,
        ),
    )
    return ExperimentPipelineSettings(
        analysis_settings=analysis,
        peak_detection_settings=analysis.frame_peaks.peak_settings,
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


def _latex_tool_available() -> bool:
    return any(shutil.which(name) for name in ("latexmk", "tectonic", "pdflatex", "lualatex"))


def _json_ready(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
