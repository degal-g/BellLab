#!/usr/bin/env python3
"""Minimal executable example for the real-experiment BellLab pipeline.

Run without arguments to analyze temporary synthetic WAVs for
``pp, p, mf, f, ff``.  Pass real files with repeated ``--recording LABEL=PATH``
arguments, for example:

    python examples/analyze_real_experiment.py --recording pp=pp.wav --recording p=p.wav
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Mapping
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
    ExperimentAnalysisStatus,
    ExperimentDefinition,
    ExperimentPipelineSettings,
    ExperimentRecordingDefinition,
    FramePeakDetectionSettings,
    ModalCandidateSettings,
    ModalEnergyExchangeSettings,
    ModalHypothesisSettings,
    ModalParameterEstimationSettings,
    ModalQFactorEstimationSettings,
    PeakDetectionSettings,
    STFTSettings,
    SpectralTrackingSettings,
    SpectrumAnalysisSettings,
    TimeResolvedSpectralCharacterizationSettings,
    WithinConditionAssociationSettings,
    analyze_experiment,
    summarize_experiment_analysis,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recording",
        action="append",
        default=(),
        metavar="LABEL=PATH",
        help="Recording definition such as pp=/path/to/pp.wav.",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="belllab-example-") as temp_dir:
        recordings = (
            _parse_recordings(args.recording)
            if args.recording
            else _temporary_recordings(Path(temp_dir))
        )
        experiment = ExperimentDefinition(
            name="BellLab real experiment pipeline example",
            specimen_id="example-bell",
            dynamic_labels=("pp", "p", "mf", "f", "ff"),
            recordings=recordings,
            acquisition_notes=(
                "Temporary synthetic WAVs are used when --recording is omitted."
                if not args.recording else None
            ),
        )
        result = analyze_experiment(experiment, _example_settings())
        print(json.dumps(_json_ready(summarize_experiment_analysis(result)), indent=2, sort_keys=True))
        if result.failure_reason is not None:
            print(f"failure_reason: {result.failure_reason}")
        return 0 if result.status in {
            ExperimentAnalysisStatus.COMPLETED,
            ExperimentAnalysisStatus.COMPLETED_WITH_RESERVATIONS,
            ExperimentAnalysisStatus.PARTIAL,
        } else 1


def _parse_recordings(values: tuple[str, ...] | list[str]) -> tuple[ExperimentRecordingDefinition, ...]:
    recordings = []
    for index, item in enumerate(values):
        if "=" not in item:
            raise SystemExit(f"Invalid --recording value {item!r}; expected LABEL=PATH.")
        label, path = item.split("=", 1)
        recordings.append(ExperimentRecordingDefinition(
            file_path=Path(path),
            dynamic_label=label,
            recording_id=f"{label}_take_{index}",
            take_index=index,
            channel=0,
        ))
    return tuple(recordings)


def _temporary_recordings(directory: Path) -> tuple[ExperimentRecordingDefinition, ...]:
    sample_rate = 4096
    duration_s = 1.2
    time = np.arange(int(sample_rate * duration_s), dtype=np.float64) / sample_rate
    recordings = []
    for index, label in enumerate(("pp", "p", "mf", "f", "ff")):
        frequency = 300.0 + index
        secondary = 520.0 + index
        signal = (
            0.28 * np.exp(-time / 2.0) * np.sin(2 * np.pi * frequency * time)
            + 0.18 * np.exp(-time / 1.4) * np.sin(2 * np.pi * secondary * time)
        )
        path = directory / f"{label}.wav"
        sf.write(path, signal.astype(np.float32), sample_rate, subtype="FLOAT")
        recordings.append(ExperimentRecordingDefinition(
            file_path=path,
            dynamic_label=label,
            recording_id=f"{label}_take_0",
            take_index=0,
            channel=0,
        ))
    return tuple(recordings)


def _example_settings() -> ExperimentPipelineSettings:
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


def _json_ready(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
