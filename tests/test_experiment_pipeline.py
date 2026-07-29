"""Tests for the real-experiment BellLab orchestration layer."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from belllab import (
    AnalysisSettings,
    CrossConditionCandidateAssociationSettings,
    ExperimentAnalysisStatus,
    ExperimentDefinition,
    ExperimentDefinitionError,
    ExperimentInputError,
    ExperimentPipelineDependencyError,
    ExperimentPipelineSettings,
    ExperimentPipelineStage,
    ExperimentPipelineStageResult,
    ExperimentPipelineStageStatus,
    ExperimentPrecomputedResultError,
    ExperimentRecordingDefinition,
    ExperimentReplicatePolicy,
    FramePeakDetectionSettings,
    GlobalSpectralCharacterizationSettings,
    ModalCandidateSettings,
    ModalEnergyExchangeSettings,
    ModalHypothesisSettings,
    ModalQFactorEstimationSettings,
    ModalParameterEstimationSettings,
    PeakDetectionSettings,
    STFTSettings,
    SpectralTrackingSettings,
    SpectrumAnalysisSettings,
    TimeResolvedSpectralCharacterizationSettings,
    WithinConditionAssociationSettings,
    analyze_experiment,
    analyze_experiment_condition,
    analyze_experiment_cross_conditions,
    analyze_experiment_recording,
    experiment_file_fingerprint,
    experiment_settings_fingerprint,
    load_experiment_recordings,
    resume_experiment_analysis,
    select_experiment_reference_replicate,
    summarize_experiment_analysis,
    validate_experiment_definition,
    validate_precomputed_experiment_stage,
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
    channels: int = 1,
    clipping: bool = False,
) -> Path:
    time = np.arange(int(round(sample_rate * duration_s)), dtype=np.float64) / sample_rate
    signal = amplitude * np.exp(-time / tau_s) * np.sin(2 * np.pi * frequency_hz * time)
    if second_frequency_hz is not None:
        signal = signal + 0.22 * np.exp(-time / (0.75 * tau_s)) * np.sin(
            2 * np.pi * second_frequency_hz * time
        )
    if clipping:
        signal = np.clip(4.0 * signal, -1.0, 1.0)
    if channels == 2:
        samples = np.column_stack((signal, 0.5 * signal))
    else:
        samples = signal
    sf.write(path, samples.astype(np.float32), sample_rate, subtype="FLOAT")
    return path


def _recording(path: Path, label: str, *, recording_id: str | None = None, **changes) -> ExperimentRecordingDefinition:
    return ExperimentRecordingDefinition(
        file_path=path,
        dynamic_label=label,
        recording_id=recording_id or label,
        **changes,
    )


def _experiment(recordings: tuple[ExperimentRecordingDefinition, ...], *, labels=None, name="experiment"):
    return ExperimentDefinition(
        name=name,
        recordings=recordings,
        dynamic_labels=tuple(labels or tuple(recording.dynamic_label for recording in recordings)),
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


def test_public_contracts_are_importable_from_package_root() -> None:
    assert ExperimentPipelineStage.LOAD.value == "load"
    assert ExperimentPipelineStageStatus.COMPLETED.value == "completed"
    assert ExperimentAnalysisStatus.COMPLETED.value == "completed"
    assert ExperimentReplicatePolicy.ANALYZE_ALL_SEPARATELY.value == "analyze_all_separately"
    assert callable(analyze_experiment)
    assert callable(validate_experiment_definition)
    assert callable(load_experiment_recordings)
    assert callable(analyze_experiment_recording)
    assert callable(analyze_experiment_condition)
    assert callable(analyze_experiment_cross_conditions)
    assert callable(select_experiment_reference_replicate)
    assert callable(summarize_experiment_analysis)
    assert callable(experiment_file_fingerprint)
    assert callable(experiment_settings_fingerprint)
    assert callable(validate_precomputed_experiment_stage)
    assert callable(resume_experiment_analysis)


def test_definition_ids_are_deterministic_and_metadata_order_independent(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "pp.wav", 300.0)
    first = ExperimentDefinition(
        name="deterministic",
        recordings=(_recording(path, "pp"),),
        dynamic_labels=("pp",),
        metadata={"b": 2, "a": 1},
    )
    second = ExperimentDefinition(
        name="deterministic",
        recordings=(_recording(path, "pp"),),
        dynamic_labels=("pp",),
        metadata={"a": 1, "b": 2},
    )
    assert first.experiment_id == second.experiment_id


@pytest.mark.parametrize(
    "changes",
    (
        {"dynamic_label": "mp"},
        {"take_index": -1},
        {"microphone_distance_m": 0.0},
        {"channel": -1},
        {"start_offset_s": -0.1},
        {"start_offset_s": 0.2, "end_offset_s": 0.1},
        {"polarity": 0},
    ),
)
def test_recording_definition_rejects_invalid_invariants(tmp_path: Path, changes) -> None:
    path = tmp_path / "x.wav"
    local = dict(changes)
    label = local.pop("dynamic_label", "pp")
    with pytest.raises((ExperimentDefinitionError, ExperimentInputError, ValueError)):
        ExperimentRecordingDefinition(path, label, **local)


@pytest.mark.parametrize(
    "changes",
    (
        {"run_stft": False, "run_tracking": True},
        {"run_cross_condition_association": False, "run_candidate_chains": True},
        {"run_modal_parameter_estimation": False, "run_modal_q_estimation": True},
        {"run_tracking": False, "run_modal_energy_exchange": True},
        {"maximum_worker_count": 0},
        {"require_all_dynamic_conditions": True, "allow_missing_dynamic_conditions": True},
    ),
)
def test_settings_reject_invalid_dependencies_and_limits(changes) -> None:
    with pytest.raises((ExperimentDefinitionError, ExperimentPipelineDependencyError)):
        _full_settings(**changes)


def test_settings_fingerprint_is_deterministic() -> None:
    first = _load_only_settings(maximum_worker_count=1)
    second = _load_only_settings(maximum_worker_count=1)
    third = _load_only_settings(maximum_worker_count=2)
    assert experiment_settings_fingerprint(first) == experiment_settings_fingerprint(second)
    assert experiment_settings_fingerprint(first) != experiment_settings_fingerprint(third)


def test_file_fingerprint_is_content_based_and_changes_with_wav_content(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "tone.wav", 300.0)
    first = experiment_file_fingerprint(path)
    _write_wav(path, 310.0)
    second = experiment_file_fingerprint(path)
    assert first != second


def test_load_selects_explicit_stereo_channel_without_downmix(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "stereo.wav", 300.0, channels=2)
    experiment = _experiment((_recording(path, "pp", channel=1),), labels=("pp",))
    result = load_experiment_recordings(experiment, _load_only_settings())[0]

    assert result.loaded_recording is not None
    assert result.loaded_recording.original_signal is not None
    assert result.loaded_recording.original_signal.channels == 2
    assert result.loaded_recording.signal is not None
    assert result.loaded_recording.signal.channels == 1
    assert result.loaded_recording.selected_channel == 1
    assert "selected_channel:1" in result.loaded_recording.diagnostics


def test_load_invalid_channel_is_structured_failure(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "mono.wav", 300.0)
    experiment = _experiment((_recording(path, "pp", channel=2),), labels=("pp",))
    result = load_experiment_recordings(experiment, _load_only_settings())[0]

    assert result.valid is False
    assert result.loaded_recording is not None
    assert result.loaded_recording.failure_reason == "channel_index_outside_signal"
    assert result.stage_results[0].status is ExperimentPipelineStageStatus.INVALID_INPUT


@pytest.mark.parametrize(
    ("start", "end", "expected_duration"),
    ((0.0, 0.5, 0.5), (0.25, 0.75, 0.5), (0.5, None, 0.5)),
)
def test_offsets_trim_analysis_signal_and_preserve_original_duration(
    tmp_path: Path,
    start: float,
    end: float | None,
    expected_duration: float,
) -> None:
    path = _write_wav(tmp_path / "offset.wav", 300.0, duration_s=1.0)
    experiment = _experiment((
        _recording(path, "pp", start_offset_s=start, end_offset_s=end),
    ), labels=("pp",))
    result = load_experiment_recordings(experiment, _load_only_settings())[0]

    assert result.loaded_recording is not None
    assert result.loaded_recording.original_duration_s == pytest.approx(1.0)
    assert result.loaded_recording.analyzed_duration_s == pytest.approx(expected_duration)
    assert result.loaded_recording.signal is not None
    assert result.loaded_recording.signal.duration == pytest.approx(expected_duration)


def test_offset_outside_file_is_explicit_failure(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "short.wav", 300.0, duration_s=0.25)
    experiment = _experiment((
        _recording(path, "pp", start_offset_s=0.3, end_offset_s=0.4),
    ), labels=("pp",))
    result = load_experiment_recordings(experiment, _load_only_settings())[0]

    assert result.valid is False
    assert result.loaded_recording is not None
    assert result.loaded_recording.failure_reason == "invalid_recording_offsets"


def test_input_validation_reports_missing_files_and_dynamic_labels(tmp_path: Path) -> None:
    experiment = _experiment((
        _recording(tmp_path / "missing.wav", "pp"),
    ), labels=("pp", "p"))
    validation = validate_experiment_definition(experiment, _load_only_settings())

    assert validation.valid is False
    assert validation.missing_files == (str(tmp_path / "missing.wav"),)
    assert validation.dynamic_labels_missing == ("p",)
    assert "missing_files" in validation.reasons


def test_sample_rate_uniformity_policy_is_explicit(tmp_path: Path) -> None:
    pp = _write_wav(tmp_path / "pp.wav", 300.0, sample_rate=4096)
    p = _write_wav(tmp_path / "p.wav", 301.0, sample_rate=2048)
    experiment = _experiment((_recording(pp, "pp"), _recording(p, "p")), labels=("pp", "p"))

    strict = analyze_experiment(experiment, _load_only_settings(require_uniform_sample_rate=True))
    relaxed = analyze_experiment(experiment, _load_only_settings(require_uniform_sample_rate=False))

    assert strict.status is ExperimentAnalysisStatus.INVALID_INPUT
    assert "nonuniform_sample_rate" in strict.input_validation.reasons
    assert relaxed.input_validation.valid is True


def test_channel_count_uniformity_policy_does_not_downmix_silently(tmp_path: Path) -> None:
    mono = _write_wav(tmp_path / "mono.wav", 300.0, channels=1)
    stereo = _write_wav(tmp_path / "stereo.wav", 301.0, channels=2)
    experiment = _experiment((
        _recording(mono, "pp"),
        _recording(stereo, "p", channel=1),
    ), labels=("pp", "p"))

    result = analyze_experiment(experiment, _load_only_settings(require_uniform_channel_count=True))

    assert result.status is ExperimentAnalysisStatus.INVALID_INPUT
    assert "nonuniform_channel_count" in result.input_validation.reasons
    assert any(
        "selected_channel:1" in recording.loaded_recording.diagnostics
        for recording in result.recording_results
        if recording.loaded_recording is not None
    )


def test_load_only_partial_pipeline_marks_scientific_stages_skipped(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "pp.wav", 300.0)
    result = analyze_experiment(_experiment((_recording(path, "pp"),), labels=("pp",)), _load_only_settings())

    assert result.status is ExperimentAnalysisStatus.COMPLETED
    assert "load" in result.completed_stages
    assert "stft" in result.skipped_stages
    assert result.recording_results[0].spectral_result is None
    assert result.recording_results[0].tracking_result is None


def test_complete_pipeline_with_wavs_reaches_modal_layers_and_provenance(tmp_path: Path) -> None:
    recordings = []
    for index, label in enumerate(("pp", "p", "mf", "f", "ff")):
        path = _write_wav(
            tmp_path / f"{label}.wav",
            300.0 + index,
            second_frequency_hz=520.0 + index,
            duration_s=1.2,
        )
        recordings.append(_recording(path, label))
    result = analyze_experiment(
        _experiment(tuple(recordings), labels=("pp", "p", "mf", "f", "ff")),
        _full_settings(run_dynamic_condition_comparison=False),
    )
    summary = summarize_experiment_analysis(result)

    assert result.status is ExperimentAnalysisStatus.COMPLETED
    assert result.input_validation.valid is True
    assert summary["recording_count"] == 5
    assert summary["candidate_count"] >= 5
    assert summary["chain_count"] >= 1
    assert summary["hypothesis_count"] >= 1
    assert summary["parameter_estimate_count"] >= 1
    assert summary["q_estimate_count"] >= 1
    assert set(result.provenance.file_fingerprints) == {"pp", "p", "mf", "f", "ff"}


def test_missing_dynamic_condition_does_not_create_non_adjacent_association(tmp_path: Path) -> None:
    recordings = []
    for index, label in enumerate(("pp", "p", "f", "ff")):
        path = _write_wav(tmp_path / f"{label}.wav", 300.0 + index, duration_s=1.1)
        recordings.append(_recording(path, label))
    result = analyze_experiment(
        _experiment(tuple(recordings), labels=("pp", "p", "mf", "f", "ff")),
        _full_settings(run_dynamic_condition_comparison=False, run_modal_energy_exchange=False),
    )

    assert result.cross_condition_result is not None
    pairs = tuple(
        (item.lower_dynamic_label, item.higher_dynamic_label)
        for item in result.cross_condition_result.adjacent_pair_results
    )
    assert ("p", "f") not in pairs
    assert pairs == (("pp", "p"), ("f", "ff"))
    assert len(result.cross_condition_result.candidate_chain_results) == 2
    assert any("adjacent_condition_gap" in item for item in result.cross_condition_result.diagnostics)


def test_replicates_are_all_analyzed_and_reference_selection_is_auditable(tmp_path: Path) -> None:
    good = _write_wav(tmp_path / "p1.wav", 300.0, duration_s=1.0)
    clipped = _write_wav(tmp_path / "p2.wav", 300.0, duration_s=1.0, clipping=True)
    short = _write_wav(tmp_path / "p3.wav", 300.0, duration_s=0.25)
    settings = _load_only_settings(
        replicate_policy=ExperimentReplicatePolicy.SELECT_BY_QUALITY_AFTER_ANALYSIS,
        minimum_analysis_duration_s=0.2,
        require_uniform_channel_count=False,
    )
    result = analyze_experiment(
        _experiment((
            _recording(good, "p", recording_id="p_take_1", take_index=0),
            _recording(clipped, "p", recording_id="p_take_2", take_index=1),
            _recording(short, "p", recording_id="p_take_3", take_index=2),
        ), labels=("p",)),
        settings,
    )
    condition = result.condition_results[0]

    assert len(condition.recording_results) == 3
    assert len(condition.replicate_summary) == 3
    assert condition.selected_reference_recording_id == "p_take_1"
    assert "no_waveform_average_performed" in condition.diagnostics
    assert {item.recording_id for item in condition.replicate_summary} == {
        "p_take_1",
        "p_take_2",
        "p_take_3",
    }


def test_failed_recording_is_preserved_when_continuation_is_enabled(tmp_path: Path) -> None:
    valid = _write_wav(tmp_path / "pp.wav", 300.0)
    invalid = tmp_path / "bad.wav"
    invalid.write_bytes(b"not wav")
    experiment = _experiment((
        _recording(valid, "pp"),
        _recording(invalid, "p"),
    ), labels=("pp", "p"))
    result = analyze_experiment(experiment, _load_only_settings(require_uniform_sample_rate=False))

    assert len(result.recording_results) == 2
    assert any(item.failure_reason for item in result.recording_results)
    assert result.status is ExperimentAnalysisStatus.INVALID_INPUT


def test_fail_fast_missing_file_raises_before_silent_continuation(tmp_path: Path) -> None:
    experiment = _experiment((_recording(tmp_path / "missing.wav", "pp"),), labels=("pp",))
    with pytest.raises(ExperimentInputError):
        analyze_experiment(experiment, _load_only_settings(fail_fast_on_invalid_input=True))


def test_explicit_reference_policy_requires_present_recording(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "p.wav", 300.0)
    settings = _load_only_settings(replicate_policy=ExperimentReplicatePolicy.EXPLICIT_REFERENCE)
    result = analyze_experiment(_experiment((_recording(path, "p", recording_id="p1"),), labels=("p",)), settings)
    quality = result.condition_results[0].replicate_summary[0]
    selected, ranked = select_experiment_reference_replicate(
        (quality,),
        settings,
        reference_recording_id="missing",
    )
    assert selected is None
    assert ranked[0].selected is False
    assert ranked[0].reasons == ("replicate_preserved_not_selected",)


def test_precomputed_recording_result_can_be_reused_after_fingerprint_validation(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "pp.wav", 300.0)
    experiment = _experiment((_recording(path, "pp"),), labels=("pp",))
    first = analyze_experiment(experiment, _load_only_settings())
    reused = analyze_experiment(
        experiment,
        _load_only_settings(reuse_precomputed_results=True),
        precomputed_results={"recording_results": first.recording_results},
    )

    assert reused.recording_results[0] == first.recording_results[0]


def test_precomputed_recording_result_rejects_file_fingerprint_mismatch(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "pp.wav", 300.0)
    experiment = _experiment((_recording(path, "pp"),), labels=("pp",))
    first = analyze_experiment(experiment, _load_only_settings())
    _write_wav(path, 310.0)

    with pytest.raises(ExperimentPrecomputedResultError):
        validate_precomputed_experiment_stage(
            ExperimentPipelineStage.SUMMARY,
            first.recording_results[0],
            expected_recording_id="pp",
            expected_file_fingerprint=experiment_file_fingerprint(path),
        )


def test_pipeline_is_deterministic_across_recording_order(tmp_path: Path) -> None:
    recordings = tuple(
        _recording(_write_wav(tmp_path / f"{label}.wav", 300.0 + index), label)
        for index, label in enumerate(("pp", "p", "mf"))
    )
    settings = _full_settings(run_dynamic_condition_comparison=False, run_modal_energy_exchange=False)
    first = analyze_experiment(_experiment(recordings, labels=("pp", "p", "mf")), settings)
    second = analyze_experiment(_experiment(tuple(reversed(recordings)), labels=("mf", "p", "pp")), settings)

    assert first.analysis_id == second.analysis_id
    assert summarize_experiment_analysis(first) == summarize_experiment_analysis(second)


def test_local_file_perturbation_changes_only_related_fingerprint(tmp_path: Path) -> None:
    pp = _recording(_write_wav(tmp_path / "pp.wav", 300.0), "pp")
    p = _recording(_write_wav(tmp_path / "p.wav", 301.0), "p")
    settings = _load_only_settings()
    first = analyze_experiment(_experiment((pp, p), labels=("pp", "p")), settings)
    _write_wav(Path(p.file_path), 302.0)
    second = analyze_experiment(_experiment((pp, p), labels=("pp", "p")), settings)

    assert first.analysis_id != second.analysis_id
    assert first.provenance.file_fingerprints["pp"] == second.provenance.file_fingerprints["pp"]
    assert first.provenance.file_fingerprints["p"] != second.provenance.file_fingerprints["p"]


def test_pipeline_does_not_modify_audio_files(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "pp.wav", 300.0)
    before = experiment_file_fingerprint(path)
    analyze_experiment(_experiment((_recording(path, "pp"),), labels=("pp",)), _load_only_settings())
    after = experiment_file_fingerprint(path)
    assert before == after


def test_energy_exchange_is_grouped_within_each_recording(tmp_path: Path) -> None:
    pp = _recording(
        _write_wav(tmp_path / "pp.wav", 300.0, second_frequency_hz=520.0, duration_s=1.2),
        "pp",
    )
    p = _recording(
        _write_wav(tmp_path / "p.wav", 301.0, second_frequency_hz=521.0, duration_s=1.2),
        "p",
    )
    result = analyze_experiment(
        _experiment((pp, p), labels=("pp", "p")),
        _full_settings(run_dynamic_condition_comparison=False),
    )

    assert result.energy_exchange_results
    assert len(result.energy_exchange_results) == 2
    assert all(item.dynamic_label in {"pp", "p"} for item in result.energy_exchange_results)
    assert "modal_energy_exchange" in result.completed_stages


def test_summary_contains_stable_scientific_sections(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "pp.wav", 300.0)
    result = analyze_experiment(_experiment((_recording(path, "pp"),), labels=("pp",)), _load_only_settings())
    summary = summarize_experiment_analysis(result)

    assert summary["experiment_id"] == result.experiment.experiment_id
    assert summary["recording_count"] == 1
    assert summary["completed_stages"] == result.completed_stages
    assert "provenance" in summary
    assert summary["provenance"]["settings_fingerprint"] == result.provenance.settings_fingerprint


def test_validate_precomputed_stage_rejects_incomplete_stage() -> None:
    blocked = ExperimentPipelineStageResult(
        stage=ExperimentPipelineStage.STFT,
        status=ExperimentPipelineStageStatus.BLOCKED,
        started=False,
        completed=False,
        blocked_reasons=("missing_load_dependency",),
    )
    with pytest.raises(ExperimentPrecomputedResultError):
        validate_precomputed_experiment_stage(ExperimentPipelineStage.STFT, blocked)


@pytest.mark.parametrize(
    ("duration", "minimum", "valid"),
    ((0.2, 0.3, False), (0.3, 0.3, True), (0.4, 0.3, True)),
)
def test_minimum_analysis_duration_limit_is_inclusive(
    tmp_path: Path,
    duration: float,
    minimum: float,
    valid: bool,
) -> None:
    path = _write_wav(tmp_path / "duration.wav", 300.0, duration_s=duration)
    result = load_experiment_recordings(
        _experiment((_recording(path, "pp"),), labels=("pp",)),
        _load_only_settings(minimum_analysis_duration_s=minimum),
    )[0]
    assert result.valid is valid


def test_immutability_of_definition_and_repeat_builds(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "pp.wav", 300.0)
    recording = _recording(path, "pp")
    experiment = _experiment((recording,), labels=("pp",))
    settings = _load_only_settings()

    first = analyze_experiment(experiment, settings)
    second = analyze_experiment(experiment, settings)

    assert experiment.recordings == (recording,)
    assert first == second
    with pytest.raises(AttributeError):
        first.recording_results = ()
