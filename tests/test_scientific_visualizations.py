"""Tests for reproducible BellLab scientific visualizations."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import belllab.scientific_visualizations as viz_module
from belllab import (
    AnalysisSettings,
    CrossConditionCandidateAssociationSettings,
    Envelope,
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
    STFTSettings,
    ScientificColorPolicy,
    ScientificFigureType,
    ScientificVisualizationReason,
    ScientificVisualizationSettings,
    ScientificVisualizationStatus,
    Signal,
    SpectralTrackingSettings,
    SpectrumAnalysisSettings,
    TimeResolvedSpectralCharacterizationSettings,
    WithinConditionAssociationSettings,
    analyze_experiment,
    create_experiment_visualizations,
    experiment_file_fingerprint,
    plot_candidate_chains,
    plot_cross_condition_associations,
    plot_decay_estimate,
    plot_dynamic_condition_comparison,
    plot_frequency_tracks,
    plot_global_spectrum,
    plot_modal_bandwidth,
    plot_modal_candidates,
    plot_modal_energy_exchange_correlation,
    plot_modal_energy_exchange_evidence,
    plot_modal_frequency_trajectories,
    plot_modal_hypotheses,
    plot_modal_parameters,
    plot_modal_q_factors,
    plot_spectral_peaks,
    plot_spectrogram,
    plot_synthetic_validation_campaign,
    plot_synthetic_validation_result,
    plot_temporal_envelope,
    plot_within_condition_associations,
    plot_waveform,
    save_scientific_figure,
    scientific_figure_checksum,
    scientific_visualization_settings_fingerprint,
    summarize_scientific_visualizations,
)
from belllab.modal_q_factors import (
    ModalBandwidthDefinition,
    ModalBandwidthEstimate,
    ModalQFactorEstimateReason,
    SpectralResolutionAssessment,
)
from belllab.synthetic_validation import (
    SyntheticValidationSettings,
    generate_synthetic_validation_scenario,
    run_synthetic_validation_campaign,
    validate_synthetic_scenario,
)


def _write_wav(
    path: Path,
    frequency_hz: float,
    *,
    sample_rate: int = 4096,
    duration_s: float = 1.2,
    tau_s: float = 2.0,
    amplitude: float = 0.35,
    second_frequency_hz: float | None = None,
    channels: int = 1,
    clipping: bool = False,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(int(round(sample_rate * duration_s)), dtype=np.float64) / sample_rate
    signal = amplitude * np.exp(-time / tau_s) * np.sin(2 * np.pi * frequency_hz * time)
    if second_frequency_hz is not None:
        signal = signal + 0.22 * np.exp(-time / (0.75 * tau_s)) * np.sin(
            2 * np.pi * second_frequency_hz * time
        )
    if clipping:
        signal = np.clip(4.0 * signal, -1.0, 1.0)
    if channels == 2:
        signal = np.column_stack((signal, 0.5 * signal))
    sf.write(path, signal.astype(np.float32), sample_rate, subtype="FLOAT")
    return path


def _recording_definition(
    path: Path,
    label: str,
    *,
    recording_id: str | None = None,
    **changes,
) -> ExperimentRecordingDefinition:
    return ExperimentRecordingDefinition(
        file_path=path,
        dynamic_label=label,
        recording_id=recording_id or label,
        **changes,
    )


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
        run_dynamic_condition_comparison=False,
    )
    base.update(changes)
    return ExperimentPipelineSettings(**base)


@pytest.fixture(scope="module")
def full_analysis(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("viz-full")
    recordings = []
    for index, label in enumerate(("pp", "p", "mf", "f", "ff")):
        path = _write_wav(
            tmp_path / f"{label}.wav",
            300.0 + index,
            second_frequency_hz=520.0 + index,
            duration_s=1.2,
        )
        recordings.append(_recording_definition(path, label))
    experiment = ExperimentDefinition(
        name="visualization full experiment",
        recordings=tuple(recordings),
        dynamic_labels=("pp", "p", "mf", "f", "ff"),
    )
    return analyze_experiment(experiment, _full_settings())


@pytest.fixture(scope="module")
def missing_condition_analysis(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("viz-missing")
    recordings = []
    for index, label in enumerate(("pp", "p", "f", "ff")):
        path = _write_wav(tmp_path / f"{label}.wav", 300.0 + index)
        recordings.append(_recording_definition(path, label))
    experiment = ExperimentDefinition(
        name="visualization missing condition",
        recordings=tuple(recordings),
        dynamic_labels=("pp", "p", "mf", "f", "ff"),
    )
    return analyze_experiment(
        experiment,
        _full_settings(run_modal_energy_exchange=False),
    )


@pytest.fixture(scope="module")
def synthetic_result():
    settings = SyntheticValidationSettings(
        sample_rate_hz=2048,
        duration_s=1.0,
        run_modal_q_estimation=False,
        run_energy_exchange_evidence=False,
    )
    scenario = generate_synthetic_validation_scenario("single_ideal", settings)
    return validate_synthetic_scenario(scenario, settings)


def _viz_settings(tmp_path: Path, **changes) -> ScientificVisualizationSettings:
    values = dict(
        output_directory=tmp_path / "figures",
        overwrite_policy=ExportOverwritePolicy.REPLACE,
        close_after_save=False,
    )
    values.update(changes)
    return ScientificVisualizationSettings(**values)


def _saved_settings(tmp_path: Path, **changes) -> ScientificVisualizationSettings:
    values = dict(
        output_directory=tmp_path / "figures",
        overwrite_policy=ExportOverwritePolicy.REPLACE,
        close_after_save=True,
        formats=("png", "svg"),
    )
    values.update(changes)
    return ScientificVisualizationSettings(**values)


def _recording_result(full_analysis, recording_id: str = "pp"):
    return next(item for item in full_analysis.recording_results if item.recording_definition.recording_id == recording_id)


def test_public_contracts_are_importable() -> None:
    assert ScientificVisualizationStatus.CREATED.value == "created"
    assert ScientificVisualizationReason.FIGURE_CREATED.value == "figure_created"
    assert ScientificColorPolicy.DYNAMIC_CONDITION.value == "dynamic_condition"
    assert ScientificFigureType.WAVEFORM.value == "waveform"
    assert callable(plot_waveform)
    assert callable(create_experiment_visualizations)
    assert callable(save_scientific_figure)


@pytest.mark.parametrize(
    "changes",
    (
        {"formats": ()},
        {"formats": ("jpg",)},
        {"dpi": 0},
        {"figure_width_in": 0.0},
        {"figure_height_in": -1.0},
        {"amplitude_scale": "power"},
        {"frequency_scale": "mel"},
        {"maximum_waveform_points": 1},
        {"decimation_method": "random"},
        {"maximum_annotation_count": -1},
    ),
)
def test_settings_reject_invalid_configuration(tmp_path: Path, changes) -> None:
    with pytest.raises(ValueError):
        _viz_settings(tmp_path, **changes)


def test_settings_fingerprint_is_path_independent(tmp_path: Path) -> None:
    first = ScientificVisualizationSettings(output_directory=tmp_path / "a", dpi=120)
    second = ScientificVisualizationSettings(output_directory=tmp_path / "b", dpi=120)
    third = ScientificVisualizationSettings(output_directory=tmp_path / "b", dpi=144)

    assert scientific_visualization_settings_fingerprint(first) == scientific_visualization_settings_fingerprint(second)
    assert scientific_visualization_settings_fingerprint(first) != scientific_visualization_settings_fingerprint(third)


def test_condition_palette_and_markers_are_stable(tmp_path: Path) -> None:
    settings = _viz_settings(tmp_path)
    assert settings.condition_color_mapping["pp"] == ScientificVisualizationSettings().condition_color_mapping["pp"]
    assert settings.condition_color_mapping["pp"] != settings.condition_color_mapping["ff"]
    assert settings.grayscale_compatible is True
    assert settings.colorblind_safe is True


def test_status_styles_do_not_use_color_only(tmp_path: Path) -> None:
    settings = _viz_settings(tmp_path)
    valid = settings.status_style_mapping["valid"]
    rejected = settings.status_style_mapping["rejected"]

    assert valid.marker != rejected.marker
    assert valid.line_style != "" and rejected.line_style != ""


def test_waveform_returns_figure_axes_and_labels(full_analysis, tmp_path: Path) -> None:
    result = plot_waveform(_recording_result(full_analysis), _viz_settings(tmp_path))

    assert result.status is ScientificVisualizationStatus.CREATED
    assert result.figure is not None
    assert len(result.axes) == 1
    assert "Waveform" in result.axes[0].get_title()
    assert result.axes[0].get_xlabel() == "Time (s)"
    assert "no_downmix_or_normalization_performed" in result.diagnostics


def test_waveform_save_png_svg_and_checksums(full_analysis, tmp_path: Path) -> None:
    result = plot_waveform(_recording_result(full_analysis), _saved_settings(tmp_path), save=True)

    assert result.figure is None
    assert {artifact.format for artifact in result.artifacts} == {"png", "svg"}
    for artifact in result.artifacts:
        path = Path(artifact.path)
        assert path.is_file()
        assert artifact.size_bytes and artifact.size_bytes > 0
        assert artifact.checksum == scientific_figure_checksum(path)


def test_waveform_decimation_preserves_extrema_in_provenance(tmp_path: Path) -> None:
    sample_rate = 10_000
    times = tuple(np.arange(sample_rate, dtype=float) / sample_rate)
    samples = tuple(np.sin(2 * np.pi * 20 * np.asarray(times)))
    signal = Signal((samples,), sample_rate, times, 1.0, 1, "normalized")

    result = plot_waveform(signal, _viz_settings(tmp_path, maximum_waveform_points=200))

    assert result.status is ScientificVisualizationStatus.CREATED
    assert result.provenance.decimations_used
    assert any("10000" in item for item in result.provenance.decimations_used)


def test_waveform_multichannel_is_explicit_not_downmixed(tmp_path: Path) -> None:
    times = tuple(np.linspace(0, 1, 64, endpoint=False))
    signal = Signal(
        (tuple(np.sin(np.asarray(times))), tuple(np.cos(np.asarray(times)))),
        64,
        times,
        1.0,
        2,
        "normalized",
    )

    result = plot_waveform(signal, _viz_settings(tmp_path))

    assert len(result.axes[0].lines) == 2
    assert "no_downmix_or_normalization_performed" in result.diagnostics


def test_waveform_clipping_threshold_is_rendered(tmp_path: Path) -> None:
    times = tuple(np.linspace(0, 1, 32, endpoint=False))
    samples = tuple(1.0 if index % 2 else -1.0 for index in range(32))
    signal = Signal((samples,), 32, times, 1.0, 1, "normalized")

    result = plot_waveform(signal, _viz_settings(tmp_path))

    assert "clipping_threshold_rendered_from_loaded_samples" in result.diagnostics
    assert any(line.get_label() == "clipping threshold" for line in result.axes[0].lines)


def test_temporal_envelope_plot_uses_existing_envelope(full_analysis, tmp_path: Path) -> None:
    result = plot_temporal_envelope(_recording_result(full_analysis), _viz_settings(tmp_path))

    assert result.status is ScientificVisualizationStatus.CREATED
    assert "envelope_source_not_recalculated" in result.diagnostics
    assert "Temporal envelope" in result.axes[0].get_title()


def test_temporal_envelope_db_scale_clips_only_for_presentation(tmp_path: Path) -> None:
    envelope = Envelope(
        times_s=(0.0, 0.5, 1.0),
        amplitudes=(1.0, 0.0, 0.25),
        method="fixture",
        unit="linear_amplitude",
    )

    result = plot_temporal_envelope(envelope, _viz_settings(tmp_path, amplitude_scale="db"))

    assert result.status is ScientificVisualizationStatus.CREATED
    assert "log_scale_clipped" in result.diagnostics


def test_decay_estimate_plot_does_not_refit(full_analysis, tmp_path: Path) -> None:
    result = plot_decay_estimate(_recording_result(full_analysis), _viz_settings(tmp_path))

    assert result.status is ScientificVisualizationStatus.CREATED
    assert "decay_plot_uses_existing_fit_or_modal_parameter_only" in result.diagnostics


def test_global_spectrum_plot_uses_existing_spectrum(full_analysis, tmp_path: Path) -> None:
    result = plot_global_spectrum(_recording_result(full_analysis), _viz_settings(tmp_path))

    assert result.status is ScientificVisualizationStatus.CREATED
    assert "no_fft_recalculated" in result.diagnostics
    assert "Frequency" in result.axes[0].get_xlabel()


def test_global_spectrum_log_frequency_does_not_crash_on_zero_bin(full_analysis, tmp_path: Path) -> None:
    result = plot_global_spectrum(_recording_result(full_analysis), _viz_settings(tmp_path, frequency_scale="log"))

    assert result.status is ScientificVisualizationStatus.CREATED
    assert result.axes[0].get_xscale() == "log"


def test_spectral_peaks_render_existing_peaks(full_analysis, tmp_path: Path) -> None:
    result = plot_spectral_peaks(_recording_result(full_analysis), _viz_settings(tmp_path, show_ids=True))

    assert result.status in {ScientificVisualizationStatus.CREATED, ScientificVisualizationStatus.CREATED_WITH_RESERVATIONS}
    assert "no_peak_detection_recomputed" in result.diagnostics
    assert result.axes[0].collections


def test_spectrogram_plot_uses_existing_stft(full_analysis, tmp_path: Path) -> None:
    result = plot_spectrogram(_recording_result(full_analysis), _viz_settings(tmp_path, amplitude_scale="db"))

    assert result.status is ScientificVisualizationStatus.CREATED
    assert "no_stft_recalculated" in result.diagnostics
    assert "Spectrogram" in result.axes[0].get_title()


def test_spectrogram_missing_stft_is_insufficient(full_analysis, tmp_path: Path) -> None:
    partial = replace(_recording_result(full_analysis), stft_result=None)
    result = plot_spectrogram(partial, _viz_settings(tmp_path))

    assert result.status is ScientificVisualizationStatus.INSUFFICIENT_EVIDENCE
    assert result.insufficient_evidence_reasons == (ScientificVisualizationReason.MISSING_STFT,)


def test_frequency_tracks_preserve_gap_semantics(full_analysis, tmp_path: Path) -> None:
    result = plot_frequency_tracks(_recording_result(full_analysis), _viz_settings(tmp_path))

    assert result.status is ScientificVisualizationStatus.CREATED
    assert "track_gaps_not_connected" in result.diagnostics
    assert "Frequency tracks" in result.axes[0].get_title()


def test_modal_candidates_plot_uses_operational_label(full_analysis, tmp_path: Path) -> None:
    result = plot_modal_candidates(full_analysis, _viz_settings(tmp_path, show_ids=True))

    assert result.status is ScientificVisualizationStatus.CREATED
    assert "Candidate representative frequency" in result.axes[0].get_title()


def test_within_condition_associations_plot(full_analysis, tmp_path: Path) -> None:
    result = plot_within_condition_associations(full_analysis, _viz_settings(tmp_path, show_ids=True))

    assert result.status is ScientificVisualizationStatus.CREATED
    assert "Within-condition" in result.axes[0].get_title()


def test_cross_condition_associations_do_not_connect_missing_gap(missing_condition_analysis, tmp_path: Path) -> None:
    result = plot_cross_condition_associations(missing_condition_analysis, _viz_settings(tmp_path))

    assert result.status is ScientificVisualizationStatus.CREATED
    assert "no_non_adjacent_edges_drawn" in result.diagnostics
    pairs = tuple(
        (item.lower_dynamic_label, item.higher_dynamic_label)
        for item in missing_condition_analysis.cross_condition_result.adjacent_pair_results
    )
    assert ("p", "f") not in pairs


def test_candidate_chains_plot_does_not_promote_modes(full_analysis, tmp_path: Path) -> None:
    result = plot_candidate_chains(full_analysis, _viz_settings(tmp_path))

    assert result.status is ScientificVisualizationStatus.CREATED
    assert "candidate_chains_are_operational_not_physical_modes" in result.diagnostics


def test_modal_hypotheses_title_is_conservative(full_analysis, tmp_path: Path) -> None:
    result = plot_modal_hypotheses(full_analysis, _viz_settings(tmp_path, show_ids=True))

    assert result.status in {
        ScientificVisualizationStatus.CREATED,
        ScientificVisualizationStatus.CREATED_WITH_RESERVATIONS,
    }
    assert "not physical modes" in result.axes[0].get_title()


def test_modal_frequency_trajectories_do_not_infer_nonlinearity(full_analysis, tmp_path: Path) -> None:
    result = plot_modal_frequency_trajectories(full_analysis, _viz_settings(tmp_path))

    assert result.status is ScientificVisualizationStatus.CREATED
    assert "frequency_trajectory_not_hardening_or_softening_proof" in result.diagnostics


def test_modal_parameters_have_frequency_tau_and_decay_panels(full_analysis, tmp_path: Path) -> None:
    result = plot_modal_parameters(full_analysis, _viz_settings(tmp_path))

    assert result.status is ScientificVisualizationStatus.CREATED
    assert len(result.axes) == 3


def test_modal_q_factors_show_multiple_methods_without_hiding_disagreement(full_analysis, tmp_path: Path) -> None:
    result = plot_modal_q_factors(full_analysis, _viz_settings(tmp_path))

    assert result.status is ScientificVisualizationStatus.CREATED
    assert "q_methods_rendered_without_hiding_disagreement" in result.diagnostics


def test_modal_bandwidth_direct_estimate_renders_crossings(tmp_path: Path) -> None:
    bandwidth = ModalBandwidthEstimate(
        center_frequency_hz=1000.0,
        lower_frequency_hz=995.0,
        upper_frequency_hz=1005.0,
        bandwidth_hz=10.0,
        bandwidth_definition=ModalBandwidthDefinition.POWER_MINUS_3_DB,
        bandwidth_level_db=-3.0,
        left_crossing_found=True,
        right_crossing_found=True,
        interpolation_method="linear",
        frequency_resolution_hz=1.0,
        resolution_ratio=10.0,
        resolution_assessment=SpectralResolutionAssessment.WELL_RESOLVED,
        neighboring_peak_distance_hz=50.0,
        neighboring_peak_overlap_fraction=0.0,
        isolated_peak=True,
        resolution_limited=False,
        valid=True,
        reasons=(ModalQFactorEstimateReason.BANDWIDTH_METHOD_AVAILABLE,),
    )

    result = plot_modal_bandwidth(bandwidth, _viz_settings(tmp_path))

    assert result.status is ScientificVisualizationStatus.CREATED
    assert "bandwidth_plot_uses_existing_bandwidth_estimates" in result.diagnostics


def test_dynamic_condition_comparison_is_ordinal(full_analysis, tmp_path: Path) -> None:
    result = plot_dynamic_condition_comparison(full_analysis, _viz_settings(tmp_path))

    assert result.status is ScientificVisualizationStatus.CREATED
    assert "dynamic_labels_plotted_as_ordinal_graphic_positions" in result.diagnostics
    assert "ordinal" in result.axes[0].get_xlabel()


def test_energy_exchange_evidence_title_is_conservative(full_analysis, tmp_path: Path) -> None:
    result = plot_modal_energy_exchange_evidence(full_analysis, _viz_settings(tmp_path))

    assert result.status in {ScientificVisualizationStatus.CREATED, ScientificVisualizationStatus.CREATED_WITH_RESERVATIONS}
    assert "Evidencia operacional de possivel redistribuicao" in result.axes[0].get_title()
    assert "energy_evidence_rendered_without_causality" in result.diagnostics


def test_energy_exchange_correlation_lag_is_not_causality(full_analysis, tmp_path: Path) -> None:
    result = plot_modal_energy_exchange_correlation(full_analysis, _viz_settings(tmp_path))

    assert result.status is ScientificVisualizationStatus.CREATED
    assert "not causal direction" in result.axes[0].get_xlabel()


def test_synthetic_validation_result_plot(synthetic_result, tmp_path: Path) -> None:
    result = plot_synthetic_validation_result(synthetic_result, _viz_settings(tmp_path))

    assert result.status in {
        ScientificVisualizationStatus.CREATED,
        ScientificVisualizationStatus.CREATED_WITH_RESERVATIONS,
    }
    assert "synthetic_success_not_real_data_proof" in result.diagnostics
    assert len(result.axes) == 4


def test_synthetic_validation_campaign_plot(synthetic_result, tmp_path: Path) -> None:
    campaign = run_synthetic_validation_campaign((synthetic_result.scenario,), synthetic_result.scenario.settings)
    result = plot_synthetic_validation_campaign(campaign, _viz_settings(tmp_path))

    assert result.status is ScientificVisualizationStatus.CREATED
    assert "synthetic_campaign_not_universal_real_data_validation" in result.diagnostics


def test_create_experiment_visualizations_collection_saves_requested_figures(full_analysis, tmp_path: Path) -> None:
    settings = _saved_settings(
        tmp_path,
        figure_types=(
            ScientificFigureType.WAVEFORM,
            ScientificFigureType.GLOBAL_SPECTRUM,
            ScientificFigureType.MODAL_HYPOTHESES,
            ScientificFigureType.MODAL_PARAMETERS,
            ScientificFigureType.MODAL_Q_FACTORS,
            ScientificFigureType.MODAL_ENERGY_EXCHANGE_EVIDENCE,
            ScientificFigureType.EXPERIMENT_SUMMARY,
        ),
    )

    collection = create_experiment_visualizations(full_analysis, settings)
    summary = summarize_scientific_visualizations(collection)

    assert collection.valid is True
    assert summary["artifact_count"] == len(collection.artifacts)
    assert ScientificFigureType.WAVEFORM in collection.completed_figure_types
    assert all(Path(artifact.path).is_file() for artifact in collection.artifacts if artifact.path)


def test_collection_preserves_insufficient_requested_figures(full_analysis, tmp_path: Path) -> None:
    settings = _saved_settings(
        tmp_path,
        figure_types=(ScientificFigureType.MODAL_BANDWIDTH,),
    )

    collection = create_experiment_visualizations(full_analysis, settings)

    assert ScientificFigureType.MODAL_BANDWIDTH in collection.skipped_figure_types
    assert collection.status is ScientificVisualizationStatus.INSUFFICIENT_EVIDENCE


def test_save_scientific_figure_overwrite_error_and_skip(full_analysis, tmp_path: Path) -> None:
    result = plot_waveform(_recording_result(full_analysis), _viz_settings(tmp_path))
    first = save_scientific_figure(result, _saved_settings(tmp_path, overwrite_policy=ExportOverwritePolicy.REPLACE))
    second = save_scientific_figure(result, _saved_settings(tmp_path, overwrite_policy=ExportOverwritePolicy.ERROR))
    third = save_scientific_figure(result, _saved_settings(tmp_path, overwrite_policy=ExportOverwritePolicy.SKIP))

    assert all(item.status is ScientificVisualizationStatus.CREATED for item in first)
    assert all(item.status is ScientificVisualizationStatus.FAILED for item in second)
    assert all(item.status is ScientificVisualizationStatus.SKIPPED for item in third)


def test_versioned_filename_preserves_existing_artifact(full_analysis, tmp_path: Path) -> None:
    result = plot_waveform(_recording_result(full_analysis), _viz_settings(tmp_path))
    first = save_scientific_figure(result, _saved_settings(tmp_path, overwrite_policy=ExportOverwritePolicy.REPLACE, formats=("png",)))
    second = save_scientific_figure(result, _saved_settings(tmp_path, overwrite_policy=ExportOverwritePolicy.VERSIONED_FILENAME, formats=("png",)))

    assert first[0].relative_path != second[0].relative_path
    assert "_v001" in second[0].relative_path


def test_pdf_format_is_supported_when_requested(full_analysis, tmp_path: Path) -> None:
    result = plot_waveform(_recording_result(full_analysis), _saved_settings(tmp_path, formats=("pdf",)), save=True)

    assert result.artifacts[0].format == "pdf"
    assert Path(result.artifacts[0].path).is_file()


def test_atomic_write_failure_preserves_existing_file(full_analysis, tmp_path: Path, monkeypatch) -> None:
    result = plot_waveform(_recording_result(full_analysis), _viz_settings(tmp_path))
    settings = _saved_settings(tmp_path, formats=("png",), overwrite_policy=ExportOverwritePolicy.REPLACE)
    first = save_scientific_figure(result, settings)[0]
    before = Path(first.path).read_bytes()
    original_replace = viz_module.os.replace

    def failing_replace(src, dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(viz_module.os, "replace", failing_replace)
    failed = save_scientific_figure(result, settings)[0]
    monkeypatch.setattr(viz_module.os, "replace", original_replace)

    assert failed.status is ScientificVisualizationStatus.FAILED
    assert Path(first.path).read_bytes() == before
    assert not Path(first.path).with_name(f".{Path(first.path).name}.tmp").exists()


def test_figure_ids_are_deterministic_across_output_directories(full_analysis, tmp_path: Path) -> None:
    first = plot_waveform(_recording_result(full_analysis), _saved_settings(tmp_path / "a"), save=True)
    second = plot_waveform(_recording_result(full_analysis), _saved_settings(tmp_path / "b"), save=True)

    assert first.figure_id == second.figure_id
    assert tuple(item.relative_path for item in first.artifacts) == tuple(item.relative_path for item in second.artifacts)


def test_local_perturbation_changes_dependent_waveform_checksum(tmp_path: Path) -> None:
    first_path = _write_wav(tmp_path / "first" / "pp.wav", 300.0)
    second_path = _write_wav(tmp_path / "second" / "pp.wav", 330.0)
    first = analyze_experiment(
        ExperimentDefinition(name="first", recordings=(_recording_definition(first_path, "pp"),), dynamic_labels=("pp",)),
        _full_settings(
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
        ),
    )
    second = analyze_experiment(
        ExperimentDefinition(name="second", recordings=(_recording_definition(second_path, "pp"),), dynamic_labels=("pp",)),
        _full_settings(
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
        ),
    )

    first_plot = plot_waveform(_recording_result(first), _saved_settings(tmp_path / "out1", formats=("png",)), save=True)
    second_plot = plot_waveform(_recording_result(second), _saved_settings(tmp_path / "out2", formats=("png",)), save=True)

    assert first_plot.artifacts[0].checksum != second_plot.artifacts[0].checksum


def test_visualization_does_not_modify_analysis_result_or_audio(full_analysis, tmp_path: Path) -> None:
    before_result = full_analysis
    path = Path(_recording_result(full_analysis).recording_definition.file_path)
    before_hash = experiment_file_fingerprint(path)

    create_experiment_visualizations(
        full_analysis,
        _saved_settings(tmp_path, figure_types=(ScientificFigureType.WAVEFORM, ScientificFigureType.GLOBAL_SPECTRUM)),
    )

    assert full_analysis == before_result
    assert experiment_file_fingerprint(path) == before_hash


def test_matplotlib_rcparams_are_not_changed_permanently(full_analysis, tmp_path: Path) -> None:
    matplotlib, _, _ = viz_module._matplotlib()
    before = dict(matplotlib.rcParams)

    plot_waveform(_recording_result(full_analysis), _viz_settings(tmp_path))

    assert dict(matplotlib.rcParams) == before


def test_many_saved_figures_close_without_open_figure_warning(full_analysis, tmp_path: Path) -> None:
    settings = _saved_settings(
        tmp_path,
        figure_types=(ScientificFigureType.WAVEFORM, ScientificFigureType.GLOBAL_SPECTRUM),
        close_after_save=True,
    )

    for _ in range(5):
        collection = create_experiment_visualizations(full_analysis, settings)
        assert all(figure.figure is None for figure in collection.figures if figure.artifacts)


def test_summary_reports_stable_shape(full_analysis, tmp_path: Path) -> None:
    result = plot_global_spectrum(_recording_result(full_analysis), _viz_settings(tmp_path))
    summary = summarize_scientific_visualizations(result)

    assert summary["figure_id"] == result.figure_id
    assert summary["figure_type"] == "global_spectrum"
    assert summary["valid"] is True
