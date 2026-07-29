#!/usr/bin/env python3
"""Export a small BellLab experiment result to reproducible artifacts."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from belllab import (
    ExperimentDefinition,
    ExperimentPipelineSettings,
    ExperimentRecordingDefinition,
    ExportOverwritePolicy,
    ResultsExportSettings,
    analyze_experiment,
    export_experiment_results,
    summarize_experiment_export,
    validate_experiment_export,
)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="belllab-export-example-") as temp_dir:
        root = Path(temp_dir)
        recordings = _temporary_recordings(root / "audio")
        experiment = ExperimentDefinition(
            name="BellLab reproducible export example",
            specimen_id="example-bell",
            dynamic_labels=("pp", "p", "mf", "f", "ff"),
            recordings=recordings,
            acquisition_notes="Temporary synthetic WAVs used only to produce an example result.",
        )
        analysis = analyze_experiment(experiment, _example_pipeline_settings())
        export = export_experiment_results(
            analysis,
            ResultsExportSettings(
                output_directory=root / "export",
                overwrite_policy=ExportOverwritePolicy.REPLACE,
            ),
        )
        validation = validate_experiment_export(export)
        summary = summarize_experiment_export(export)

        print(json.dumps(summary, indent=2, sort_keys=True))
        print("Artifacts:")
        for artifact in export.artifacts:
            if artifact.relative_path is not None:
                print(f"- {artifact.relative_path}: {artifact.status.value}")
        print(f"Checksums valid: {validation.valid}")
        if not validation.valid:
            print(json.dumps(validation.diagnostics, indent=2, sort_keys=True))
        return 0 if export.valid and validation.valid else 1


def _temporary_recordings(directory: Path) -> tuple[ExperimentRecordingDefinition, ...]:
    directory.mkdir(parents=True, exist_ok=True)
    sample_rate = 4096
    duration_s = 1.0
    time = np.arange(int(sample_rate * duration_s), dtype=np.float64) / sample_rate
    recordings = []
    for index, label in enumerate(("pp", "p", "mf", "f", "ff")):
        frequency = 300.0 + index
        signal = 0.35 * np.exp(-time / 2.0) * np.sin(2 * np.pi * frequency * time)
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
    return ExperimentPipelineSettings(
        run_temporal_analysis=True,
        run_global_spectrum=True,
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


if __name__ == "__main__":
    raise SystemExit(main())
