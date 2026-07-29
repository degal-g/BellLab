"""Synthetic end-to-end validation contracts for controlled BellLab scenarios."""

from __future__ import annotations

from dataclasses import replace
from math import pi

import numpy as np
import pytest

from belllab import (
    SyntheticAmplitudeModel,
    SyntheticDampedComponent,
    SyntheticFrequencyModel,
    SyntheticValidationReason,
    SyntheticValidationSettings,
    SyntheticValidationStatus,
    generate_synthetic_ground_truth,
    generate_synthetic_validation_scenario,
    run_synthetic_monte_carlo_validation,
    run_synthetic_pipeline,
    run_synthetic_validation_campaign,
    summarize_synthetic_validation,
    synthetic_validation_settings_fingerprint,
    validate_synthetic_associations,
    validate_synthetic_bandwidth,
    validate_synthetic_candidates,
    validate_synthetic_chains,
    validate_synthetic_decay,
    validate_synthetic_energy_exchange,
    validate_synthetic_frequency,
    validate_synthetic_modal_hypotheses,
    validate_synthetic_q,
    validate_synthetic_scenario,
    validate_synthetic_tracking,
)


BUILT_IN_SCENARIOS = (
    "single_ideal",
    "multiple_isolated",
    "near_modes_resolved",
    "near_modes_marginal",
    "near_modes_unidentifiable",
    "beating",
    "linear_drift",
    "frequency_crossing",
    "emergence_disappearance",
    "apparent_split_merge",
    "energy_exchange",
    "no_energy_exchange",
    "noise",
    "mains_hum",
    "clipping",
    "short_duration",
    "sampling_resolution",
)


def _fast_settings(**changes) -> SyntheticValidationSettings:
    defaults = dict(
        sample_rate_hz=4000,
        duration_s=4.0,
        spectrum_n_fft=32768,
        stft_window_length=512,
        stft_hop_length=128,
        stft_n_fft=1024,
        peak_min_prominence=0.01,
    )
    defaults.update(changes)
    return SyntheticValidationSettings(**defaults)


def test_public_contracts_are_importable_from_package_root() -> None:
    settings = SyntheticValidationSettings()
    scenario = generate_synthetic_validation_scenario("single_ideal", settings)
    truth = generate_synthetic_ground_truth(scenario)

    assert scenario.settings is settings
    assert truth.scenario_id == scenario.scenario_id
    assert synthetic_validation_settings_fingerprint(settings) == truth.settings_fingerprint


@pytest.mark.parametrize("name", BUILT_IN_SCENARIOS)
def test_built_in_scenarios_are_deterministic_and_have_known_truth(name: str) -> None:
    settings = _fast_settings()
    first = generate_synthetic_validation_scenario(name, settings)
    second = generate_synthetic_validation_scenario(name, settings)
    truth = generate_synthetic_ground_truth(first)

    assert first.scenario_id == second.scenario_id
    assert first.valid
    assert len(truth.components) == len(first.components)
    assert len(truth.known_frequencies_hz) == len(first.components)
    assert all(value > 0 for _, value in truth.known_frequencies_hz)
    assert "controlled_validation_not_real_recording_proof" in first.diagnostics
    if "non_identifiable" in " ".join(first.identifiability_notes):
        assert SyntheticValidationReason.NON_IDENTIFIABLE_SCENARIO_REPORTED in first.reasons


def test_single_ideal_scenario_recovers_core_operational_metrics() -> None:
    scenario = generate_synthetic_validation_scenario("single_ideal")
    result = validate_synthetic_scenario(scenario)

    assert result.status is SyntheticValidationStatus.PASSED
    assert result.pipeline_output.pipeline_errors == ()
    assert {"spectrum", "peaks", "stft", "tracking", "candidate_characterization"}.issubset(
        set(result.pipeline_output.pipeline_stages_completed)
    )
    assert result.frequency_validations[0].relative_error < 1e-9
    assert result.decay_validations[0].relative_error < 1e-9
    assert result.q_validations[0].representative_relative_error < 1e-9
    assert result.bandwidth_validations[0].passed
    assert result.tracking_validation.passed
    assert result.candidate_validation.passed
    assert SyntheticValidationReason.NO_GENERAL_PHYSICAL_VALIDITY_CLAIM in result.supporting_reasons
    assert SyntheticValidationReason.NO_GROUND_TRUTH_USED_BY_ESTIMATOR in result.supporting_reasons
    assert SyntheticValidationReason.NO_TRACKING_CORRECTION_FROM_TRUTH in result.supporting_reasons
    assert "no_general_real_recording_validity_claim" in result.diagnostics


def test_ground_truth_uses_documented_q_and_bandwidth_conventions() -> None:
    scenario = generate_synthetic_validation_scenario("single_ideal")
    truth = generate_synthetic_ground_truth(scenario)

    assert truth.known_frequencies_hz == (("mode_500", 500.0),)
    assert truth.known_tau_values_s == (("mode_500", 2.0),)
    assert truth.known_q_values[0][1] == pytest.approx(pi * 500.0 * 2.0)
    assert truth.known_bandwidth_values_hz[0][1] == pytest.approx(1.0 / (pi * 2.0))
    assert max(abs(value) for value in truth.noise_signal.samples[0]) == 0.0
    assert truth.clean_signal.samples == truth.observed_signal.samples


def test_noise_and_clipping_are_seeded_and_auditable() -> None:
    settings = _fast_settings(
        noise_model="white",
        signal_to_noise_ratio_db=40.0,
        clipping_mode="hard",
        clipping_threshold=0.4,
    )
    scenario = generate_synthetic_validation_scenario("single_ideal", settings)
    first = generate_synthetic_ground_truth(scenario)
    second = generate_synthetic_ground_truth(scenario)
    changed = generate_synthetic_ground_truth(
        replace(scenario, settings=replace(settings, random_seed=1234))
    )

    assert first.noise_signal.samples == second.noise_signal.samples
    assert first.observed_signal.samples == second.observed_signal.samples
    assert first.observed_signal.samples != changed.observed_signal.samples
    assert first.clipping_metadata["clipping_fraction_observed"] is not None
    assert first.noise_metadata["signal_to_noise_ratio_db"] == 40.0


@pytest.mark.parametrize(
    "changes",
    [
        {"sample_rate_hz": 0},
        {"duration_s": 0.0},
        {"attack_time_s": -1.0},
        {"signal_start_time_s": 99.0},
        {"global_amplitude": float("inf")},
        {"random_seed": 1.5},
        {"noise_standard_deviation": -0.1},
        {"colored_noise_exponent": -1.0},
        {"include_mains_hum": True, "mains_frequency_hz": 5000.0},
        {"clipping_threshold": -1.0},
        {"clipping_fraction": 1.1},
        {"trial_count": 0},
        {"trial_seed_stride": 0},
        {"stft_hop_length": 2048},
        {"stft_n_fft": 128},
        {"tracking_frequency_distance_unit": "octaves"},
    ],
)
def test_settings_reject_invalid_invariants(changes) -> None:
    with pytest.raises(ValueError):
        SyntheticValidationSettings(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"component_id": ""},
        {"initial_frequency_hz": 0.0},
        {"tau_s": 0.0},
        {"amplitude": float("nan")},
        {"end_time_s": 0.5, "start_time_s": 1.0},
        {"frequency_model": "unknown"},
        {"amplitude_model": "unknown"},
        {"frequency_trajectory": ((0.0, 100.0), (0.0, 110.0))},
        {"frequency_trajectory": ((0.0, -100.0),)},
        {"diagnostics": ("duplicate", "duplicate")},
    ],
)
def test_component_contract_rejects_invalid_values(changes) -> None:
    values = dict(component_id="x", initial_frequency_hz=100.0, amplitude=1.0, tau_s=1.0)
    values.update(changes)
    with pytest.raises(ValueError):
        SyntheticDampedComponent(**values)


@pytest.mark.parametrize(
    ("estimated", "settings", "passed"),
    [
        (101.0, SyntheticValidationSettings(maximum_frequency_absolute_error_hz=1.0, maximum_frequency_relative_error=None), True),
        (101.01, SyntheticValidationSettings(maximum_frequency_absolute_error_hz=1.0, maximum_frequency_relative_error=None), False),
        (101.0, SyntheticValidationSettings(maximum_frequency_absolute_error_hz=None, maximum_frequency_relative_error=0.01), True),
        (101.1, SyntheticValidationSettings(maximum_frequency_absolute_error_hz=None, maximum_frequency_relative_error=0.01), False),
        (None, SyntheticValidationSettings(), False),
    ],
)
def test_frequency_validation_limits_are_inclusive(estimated, settings, passed) -> None:
    result = validate_synthetic_frequency(100.0, estimated, settings)
    assert result.passed is passed
    if passed:
        assert SyntheticValidationReason.FREQUENCY_ERROR_WITHIN_TOLERANCE in result.reasons
    else:
        assert result.reasons


def test_frequency_trajectory_metrics_are_quantitative() -> None:
    result = validate_synthetic_frequency(
        100.0,
        100.5,
        SyntheticValidationSettings(maximum_frequency_absolute_error_hz=1.0),
        true_trajectory_hz=(100.0, 101.0, 102.0),
        estimated_trajectory_hz=(100.1, 100.8, 102.3),
        true_slope_hz_per_s=1.0,
        estimated_slope_hz_per_s=1.1,
    )

    assert result.passed
    assert result.trajectory_rmse_hz == pytest.approx(np.sqrt((0.1**2 + (-0.2) ** 2 + 0.3**2) / 3))
    assert result.trajectory_mae_hz == pytest.approx(0.2)
    assert result.trajectory_slope_error_hz_per_s == pytest.approx(0.1)
    assert result.trajectory_total_change_error_hz == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("estimated", "passed"),
    [(11.0, True), (11.01, False), (None, False), (0.0, False)],
)
def test_decay_validation_preserves_absence_and_uses_relative_limits(estimated, passed) -> None:
    result = validate_synthetic_decay(
        10.0,
        estimated,
        SyntheticValidationSettings(maximum_tau_relative_error=0.1),
    )
    assert result.passed is passed
    if estimated is None or estimated == 0.0:
        assert result.estimated_tau_s is None
        assert SyntheticValidationReason.NO_VALID_ESTIMATE in result.reasons


@pytest.mark.parametrize(
    ("decay", "bandwidth", "representative", "passed"),
    [
        (105.0, None, 105.0, True),
        (None, 95.0, 95.0, True),
        (None, None, None, False),
        (150.0, None, 150.0, False),
        (100.0, 120.0, 110.0, True),
    ],
)
def test_q_validation_uses_same_synthetic_convention_without_calibration(decay, bandwidth, representative, passed) -> None:
    result = validate_synthetic_q(
        100.0,
        decay,
        bandwidth,
        representative,
        SyntheticValidationSettings(maximum_q_relative_error=0.1),
    )
    assert result.passed is passed
    assert result.true_q == 100.0
    assert "q_truth_uses_Q_equals_pi_f_tau_for_compatible_components" in result.diagnostics


@pytest.mark.parametrize(
    ("truth", "estimated", "resolution", "passed", "reason"),
    [
        (10.0, 11.0, 1.0, True, SyntheticValidationReason.BANDWIDTH_ERROR_WITHIN_TOLERANCE),
        (10.0, 16.0, 1.0, False, SyntheticValidationReason.BANDWIDTH_ERROR_EXCEEDS_TOLERANCE),
        (None, 5.0, 1.0, False, SyntheticValidationReason.SCENARIO_NOT_IDENTIFIABLE),
        (10.0, None, 1.0, False, SyntheticValidationReason.NO_VALID_ESTIMATE),
        (10.0, 11.0, 8.0, True, SyntheticValidationReason.RESOLUTION_LIMITED),
    ],
)
def test_bandwidth_validation_handles_limits_absence_and_resolution(truth, estimated, resolution, passed, reason) -> None:
    result = validate_synthetic_bandwidth(
        truth,
        estimated,
        SyntheticValidationSettings(maximum_bandwidth_relative_error=0.2),
        frequency_resolution_hz=resolution,
    )
    assert result.passed is passed
    assert reason in result.reasons


def test_pipeline_reports_configured_partial_stages_without_silent_continuation() -> None:
    settings = _fast_settings(run_tracking=False)
    scenario = generate_synthetic_validation_scenario("single_ideal", settings)
    truth = generate_synthetic_ground_truth(scenario)
    pipeline = run_synthetic_pipeline(scenario, truth)

    assert "stft" in pipeline.pipeline_stages_completed
    assert "tracking" not in pipeline.pipeline_stages_completed
    assert "candidate_characterization_requires_tracking" in pipeline.diagnostics
    assert pipeline.pipeline_errors == ()


def test_tracking_and_candidate_validation_use_posthoc_truth_matching_only() -> None:
    scenario = generate_synthetic_validation_scenario("single_ideal")
    truth = generate_synthetic_ground_truth(scenario)
    pipeline = run_synthetic_pipeline(scenario, truth)

    tracking = validate_synthetic_tracking(truth, pipeline.tracking_results)
    candidates = validate_synthetic_candidates(truth, pipeline.candidate_results)

    assert tracking.passed
    assert candidates.passed
    assert tracking.matched_track_pairs == (("mode_500", 0),)
    assert candidates.matched_candidates == (("mode_500", 0),)
    assert "tracking_algorithm_not_modified" in tracking.diagnostics


def test_association_validation_reports_precision_recall_and_mismatches() -> None:
    result = validate_synthetic_associations(
        (("a", "b"), ("c", "d")),
        (("b", "a"), ("x", "y")),
    )

    assert not result.passed
    assert result.correct_pairs == (("a", "b"),)
    assert result.missing_pairs == (("c", "d"),)
    assert result.incorrect_pairs == (("x", "y"),)
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)


def test_chain_validation_compares_content_not_operational_ids() -> None:
    empty = validate_synthetic_chains((), ())
    recovered = validate_synthetic_chains((("a", "b", "c"),), (("a", "b", "c"),))
    mismatch = validate_synthetic_chains((("a", "b"),), (("a", "x"),))

    assert empty.passed
    assert recovered.passed
    assert recovered.node_recovery_fraction == 1.0
    assert not mismatch.passed
    assert SyntheticValidationReason.CHAIN_MISMATCH in mismatch.reasons


def test_hypothesis_validation_does_not_promote_synthetic_success_to_modal_mode() -> None:
    empty = validate_synthetic_modal_hypotheses((), None)
    matched = validate_synthetic_modal_hypotheses((("chain-a", "accepted"),), (("chain-a", "accepted"),))
    mismatch = validate_synthetic_modal_hypotheses((("chain-a", "accepted"),), (("chain-a", "rejected"),))

    assert empty.passed
    assert matched.passed
    assert matched.recovery_fraction == 1.0
    assert not mismatch.passed
    assert "no_ModalMode_created" in empty.diagnostics


def test_energy_validation_separates_supported_absent_false_positive_and_false_negative() -> None:
    supported = validate_synthetic_energy_exchange((("a", "b"),), (), (("b", "a"),))
    false_positive = validate_synthetic_energy_exchange((), (("a", "b"),), (("a", "b"),))
    false_negative = validate_synthetic_energy_exchange((("a", "b"),), (), ())
    absent = validate_synthetic_energy_exchange((), (("a", "b"),), ())

    assert supported.passed
    assert not false_positive.passed
    assert false_positive.false_positive_pairs == (("a", "b"),)
    assert not false_negative.passed
    assert false_negative.false_negative_pairs == (("a", "b"),)
    assert absent.passed
    assert "no_physical_energy_transfer_or_causality_inferred" in supported.diagnostics


def test_energy_exchange_scenario_recovers_imposed_operational_pattern() -> None:
    scenario = generate_synthetic_validation_scenario("energy_exchange")
    truth = generate_synthetic_ground_truth(scenario)
    pipeline = run_synthetic_pipeline(scenario, truth)
    validation = validate_synthetic_energy_exchange(
        truth.known_energy_exchange_pairs,
        truth.known_non_exchange_pairs,
        pipeline.energy_exchange_results,
    )

    assert validation.passed
    assert validation.supported_pairs == (("exchange_a", "exchange_b"),)
    assert SyntheticValidationReason.ENERGY_EXCHANGE_PATTERN_RECOVERED in validation.reasons


def test_no_exchange_scenario_does_not_require_false_support() -> None:
    scenario = generate_synthetic_validation_scenario("no_energy_exchange")
    result = validate_synthetic_scenario(scenario)

    assert result.energy_exchange_validation.passed
    assert result.energy_exchange_validation.supported_pairs == ()
    assert result.energy_exchange_validation.expected_non_exchange_pairs == (("decay_a", "decay_b"),)


def test_beating_and_crossing_remain_reservation_contexts_only() -> None:
    beating = validate_synthetic_scenario(generate_synthetic_validation_scenario("beating"))
    crossing = generate_synthetic_validation_scenario("frequency_crossing")

    assert SyntheticValidationReason.POSSIBLE_BEATING_CONTEXT in beating.reservation_reasons
    assert "frequency_crossing_context" in " ".join(crossing.identifiability_notes)


def test_custom_component_order_does_not_change_scenario_or_truth_identity() -> None:
    settings = _fast_settings()
    left = SyntheticDampedComponent("a", 300.0, 0.5, tau_s=1.0)
    right = SyntheticDampedComponent("b", 600.0, 0.5, tau_s=2.0)
    forward = generate_synthetic_validation_scenario("custom", settings, components=(left, right))
    reverse = generate_synthetic_validation_scenario("custom", settings, components=(right, left))

    assert forward.scenario_id == reverse.scenario_id
    assert generate_synthetic_ground_truth(forward).known_frequencies_hz == generate_synthetic_ground_truth(reverse).known_frequencies_hz


def test_validation_does_not_mutate_scenario_components_or_global_rng() -> None:
    np.random.seed(12345)
    state_before = np.random.get_state()
    scenario = generate_synthetic_validation_scenario("single_ideal")
    components_before = scenario.components

    first = validate_synthetic_scenario(scenario)
    second = validate_synthetic_scenario(scenario)
    state_after = np.random.get_state()

    assert scenario.components == components_before
    assert summarize_synthetic_validation(first) == summarize_synthetic_validation(second)
    assert state_before[0] == state_after[0]
    assert np.array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]


def test_campaign_has_one_result_per_scenario_and_stable_counts() -> None:
    settings = _fast_settings()
    scenarios = (
        generate_synthetic_validation_scenario("single_ideal", settings),
        generate_synthetic_validation_scenario("no_energy_exchange", settings),
    )
    first = run_synthetic_validation_campaign(scenarios, settings)
    second = run_synthetic_validation_campaign(reversed(scenarios), settings)

    assert first.campaign_id == second.campaign_id
    assert first.scenario_count == 2
    assert first.passed_count + first.passed_with_reservations_count + first.failed_count + first.inconclusive_count + first.insufficient_evidence_count + first.invalid_scenario_count + first.pipeline_error_count == 2
    assert first.metric_summaries["scenario_statuses"] == second.metric_summaries["scenario_statuses"]


def test_monte_carlo_uses_explicit_seed_stride_and_is_reproducible() -> None:
    settings = _fast_settings(trial_count=2, trial_seed_stride=17, store_trial_details=True)
    scenario = generate_synthetic_validation_scenario("single_ideal", settings)
    first = run_synthetic_monte_carlo_validation(scenario, settings)
    second = run_synthetic_monte_carlo_validation(scenario, settings)

    assert first.seeds == (0, 17)
    assert first.pass_fraction == second.pass_fraction
    assert first.metric_distributions == second.metric_distributions
    assert SyntheticValidationReason.DETERMINISTIC_SEED_USED in first.reasons
    assert SyntheticValidationReason.NO_GLOBAL_RNG_MUTATION in first.reasons


def test_local_noise_seed_perturbation_changes_only_observed_signal_not_identity() -> None:
    base_settings = _fast_settings(noise_model="white", signal_to_noise_ratio_db=30.0, random_seed=1)
    changed_settings = replace(base_settings, random_seed=2)
    base = generate_synthetic_validation_scenario("single_ideal", base_settings)
    changed = generate_synthetic_validation_scenario("single_ideal", changed_settings)
    base_truth = generate_synthetic_ground_truth(base)
    changed_truth = generate_synthetic_ground_truth(changed)

    assert base.components == changed.components
    assert base_truth.clean_signal.samples == changed_truth.clean_signal.samples
    assert base_truth.observed_signal.samples != changed_truth.observed_signal.samples
    assert base.scenario_id != changed.scenario_id


def test_linear_drift_and_piecewise_frequency_models_have_known_representatives() -> None:
    settings = _fast_settings(duration_s=4.0)
    drift = generate_synthetic_validation_scenario("linear_drift", settings)
    piecewise = generate_synthetic_validation_scenario(
        "piecewise",
        settings,
        components=(
            SyntheticDampedComponent(
                "piecewise",
                100.0,
                1.0,
                tau_s=1.0,
                frequency_model=SyntheticFrequencyModel.PIECEWISE_LINEAR,
                frequency_trajectory=((0.0, 100.0), (4.0, 140.0)),
            ),
        ),
    )

    drift_truth = generate_synthetic_ground_truth(drift)
    piecewise_truth = generate_synthetic_ground_truth(piecewise)

    assert drift_truth.known_frequencies_hz[0][1] == pytest.approx(504.0)
    assert piecewise_truth.known_frequencies_hz[0][1] == pytest.approx(120.0)


def test_amplitude_models_preserve_none_for_missing_tau_and_known_q() -> None:
    scenario = generate_synthetic_validation_scenario(
        "constant",
        _fast_settings(),
        components=(
            SyntheticDampedComponent(
                "constant",
                220.0,
                0.5,
                tau_s=None,
                amplitude_model=SyntheticAmplitudeModel.CONSTANT_AMPLITUDE,
            ),
        ),
    )
    truth = generate_synthetic_ground_truth(scenario)

    assert truth.known_tau_values_s == (("constant", None),)
    assert truth.known_q_values == (("constant", None),)
    assert truth.known_bandwidth_values_hz == (("constant", None),)


def test_summary_shapes_for_scenario_campaign_and_monte_carlo() -> None:
    settings = _fast_settings(trial_count=1)
    scenario = generate_synthetic_validation_scenario("single_ideal", settings)
    scenario_result = validate_synthetic_scenario(scenario, settings)
    campaign = run_synthetic_validation_campaign((scenario,), settings)
    monte_carlo = run_synthetic_monte_carlo_validation(scenario, settings)

    assert summarize_synthetic_validation(scenario_result)["scenario_id"] == scenario.scenario_id
    assert summarize_synthetic_validation(campaign)["scenario_count"] == 1
    assert summarize_synthetic_validation(monte_carlo)["trial_count"] == 1
