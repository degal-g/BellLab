"""Operational modal-parameter estimates from modal hypotheses."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace
from math import exp, isclose, isfinite, log, sqrt
from random import Random

import pytest

from belllab import (
    AdjacentDynamicConditionPair,
    CandidateReference,
    CrossConditionCandidateAssociationSettings,
    FiniteValuePolicy,
    ModalHypothesisResult,
    ModalParameterEstimateReason,
    ModalParameterEstimateStatus,
    ModalParameterEstimationSettings,
    ModalHypothesisSettings,
    ModalHypothesisStatus,
    ParameterLocationMethod,
    ParameterUncertaintyMethod,
    ParameterWeightingMethod,
    build_cross_condition_candidate_chains,
    build_cross_condition_candidate_matches,
    build_modal_hypotheses,
    estimate_modal_decay,
    estimate_modal_decay_rate,
    estimate_modal_decay_uncertainty,
    estimate_modal_frequency,
    estimate_modal_frequency_trajectory,
    estimate_modal_frequency_uncertainty,
    estimate_modal_parameter_provenance,
    estimate_modal_parameters,
    estimate_modal_parameters_for_hypothesis,
    settings_fingerprint,
    summarize_modal_parameter_estimates,
)


FULL_SEQUENCE = ("pp", "p", "mf", "f", "ff")
RELAXED_HYPOTHESIS_SETTINGS = ModalHypothesisSettings(
    maximum_step_absolute_frequency_change_hz=100.0,
    maximum_total_absolute_frequency_change_hz=500.0,
    maximum_frequency_trajectory_rmse_hz=100.0,
    maximum_match_cost=100.0,
    maximum_mean_match_cost=100.0,
    maximum_log_tau_range=None,
    maximum_log_tau_standard_deviation=None,
)


def _ref(
    label: str,
    name: str,
    frequency: float,
    candidate_id: int,
    *,
    accepted: bool = True,
    tau: float | None = 4.0,
    amplitude_quality: float | None = 0.95,
    coverage: float | None = 1.0,
    rmse: float | None = 0.1,
    drift: float | None = 0.0,
    ambiguous_fraction: float | None = 0.0,
    near_fraction: float | None = 0.0,
    margin: float | None = 1.0,
    impact_excited: bool | None = True,
    classification: str | None = "impact_emergent",
) -> CandidateReference:
    return CandidateReference(
        f"{label}-{name}",
        candidate_id,
        candidate_id,
        frequency,
        label,
        accepted,
        impact_excited,
        classification,
        0.01,
        tau,
        amplitude_quality,
        drift,
        rmse,
        coverage,
        ambiguous_fraction,
        near_fraction,
        margin,
        classification,
        None,
        ("modal_parameter_test_reference",),
    )


def _chain(
    frequencies: tuple[float, ...],
    *,
    labels: tuple[str, ...] = FULL_SEQUENCE,
    prefix: str = "A",
    candidate_start: int = 0,
    association_settings: CrossConditionCandidateAssociationSettings | None = None,
    per_ref: dict[int, dict[str, object]] | None = None,
):
    refs = []
    for index, (label, frequency) in enumerate(zip(labels, frequencies, strict=True)):
        overrides = per_ref.get(index, {}) if per_ref is not None else {}
        refs.append(
            _ref(
                label,
                f"{prefix}{index}",
                frequency,
                candidate_start + index,
                **overrides,
            )
        )
    pair_settings = association_settings or CrossConditionCandidateAssociationSettings(
        maximum_absolute_frequency_difference_hz=100.0,
        maximum_association_cost=100.0,
    )
    pairs = tuple(
        build_cross_condition_candidate_matches(
            (refs[index],),
            (refs[index + 1],),
            AdjacentDynamicConditionPair(labels[index], labels[index + 1]),
            pair_settings,
        )
        for index in range(len(refs) - 1)
    )
    result = build_cross_condition_candidate_chains(pairs, labels)
    assert result.chain_count == 1
    return result.chains[0], result


def _hypothesis(
    frequencies: tuple[float, ...],
    *,
    labels: tuple[str, ...] = FULL_SEQUENCE,
    prefix: str = "H",
    candidate_start: int = 0,
    per_ref: dict[int, dict[str, object]] | None = None,
    hypothesis_settings: ModalHypothesisSettings | None = None,
):
    chain = _chain(
        frequencies,
        labels=labels,
        prefix=prefix,
        candidate_start=candidate_start,
        per_ref=per_ref,
    )[0]
    settings = hypothesis_settings or RELAXED_HYPOTHESIS_SETTINGS
    return build_modal_hypotheses((chain,), settings, labels).hypotheses[0]


def _corrupt_dataclass(instance, **changes):
    corrupted = object.__new__(type(instance))
    for field in fields(instance):
        object.__setattr__(
            corrupted,
            field.name,
            changes.get(field.name, getattr(instance, field.name)),
        )
    return corrupted


def _replace_node_ref(hypothesis, node_index: int, **changes):
    chain = hypothesis.chain
    nodes = list(chain.nodes)
    ref = replace(nodes[node_index].candidate_ref, **changes)
    nodes[node_index] = replace(nodes[node_index], candidate_ref=ref)
    frequencies = tuple(node.candidate_ref.representative_frequency_hz for node in nodes)
    new_chain = replace(
        chain,
        nodes=tuple(nodes),
        frequency_trajectory_hz=frequencies,
        initial_frequency_hz=frequencies[0],
        final_frequency_hz=frequencies[-1],
        total_frequency_change_hz=frequencies[-1] - frequencies[0],
        total_frequency_change_relative=(frequencies[-1] - frequencies[0])
        / (0.5 * (frequencies[-1] + frequencies[0])),
    )
    return replace(hypothesis, chain=new_chain)


def _corrupt_node_ref(hypothesis, node_index: int, **changes):
    chain = hypothesis.chain
    nodes = list(chain.nodes)
    ref = _corrupt_dataclass(nodes[node_index].candidate_ref, **changes)
    nodes[node_index] = _corrupt_dataclass(nodes[node_index], candidate_ref=ref)
    new_chain = _corrupt_dataclass(chain, nodes=tuple(nodes))
    return replace(hypothesis, chain=new_chain)


def _with_hypothesis_status(hypothesis, status: ModalHypothesisStatus):
    return replace(
        hypothesis,
        status=status,
        accepted=status
        in {
            ModalHypothesisStatus.ACCEPTED,
            ModalHypothesisStatus.ACCEPTED_WITH_RESERVATIONS,
        },
        requires_review=status is not ModalHypothesisStatus.ACCEPTED,
        valid=status is not ModalHypothesisStatus.INVALID_INPUT,
    )


def _modal_hypothesis_sort_key(hypothesis) -> tuple:
    chain = hypothesis.chain
    if chain is None:
        return (99, "", hypothesis.hypothesis_id)
    first = chain.nodes[0].candidate_ref
    return (
        FULL_SEQUENCE.index(chain.start_dynamic_label),
        first.representative_frequency_hz,
        tuple(
            (
                node.dynamic_label,
                node.candidate_ref.recording_id,
                node.candidate_ref.candidate_id,
                node.candidate_ref.source_track_id,
            )
            for node in chain.nodes
        ),
        chain.chain_id,
        hypothesis.hypothesis_id,
    )


def _modal_hypothesis_result(hypotheses):
    ordered = tuple(sorted(hypotheses, key=_modal_hypothesis_sort_key))
    return ModalHypothesisResult(
        sequence=FULL_SEQUENCE,
        hypotheses=ordered,
        accepted_hypotheses=tuple(
            item for item in ordered if item.status is ModalHypothesisStatus.ACCEPTED
        ),
        accepted_with_reservations_hypotheses=tuple(
            item for item in ordered if item.status is ModalHypothesisStatus.ACCEPTED_WITH_RESERVATIONS
        ),
        inconclusive_hypotheses=tuple(
            item for item in ordered if item.status is ModalHypothesisStatus.INCONCLUSIVE
        ),
        rejected_hypotheses=tuple(
            item for item in ordered if item.status is ModalHypothesisStatus.REJECTED
        ),
        insufficient_evidence_hypotheses=tuple(
            item for item in ordered if item.status is ModalHypothesisStatus.INSUFFICIENT_EVIDENCE
        ),
        hypothesis_count=len(ordered),
        accepted_count=sum(item.status is ModalHypothesisStatus.ACCEPTED for item in ordered),
        accepted_with_reservations_count=sum(
            item.status is ModalHypothesisStatus.ACCEPTED_WITH_RESERVATIONS
            for item in ordered
        ),
        inconclusive_count=sum(item.status is ModalHypothesisStatus.INCONCLUSIVE for item in ordered),
        rejected_count=sum(item.status is ModalHypothesisStatus.REJECTED for item in ordered),
        insufficient_evidence_count=sum(
            item.status is ModalHypothesisStatus.INSUFFICIENT_EVIDENCE
            for item in ordered
        ),
        source_chain_count=len(ordered),
        settings=RELAXED_HYPOTHESIS_SETTINGS,
        valid=True,
        failure_reason=None,
        diagnostics=("modal_parameter_test_result",),
    )


def _replace_edge_cost(hypothesis, edge_index: int, cost: float):
    chain = hypothesis.chain
    nodes = list(chain.nodes)
    nodes[edge_index] = replace(nodes[edge_index], outgoing_association_cost=cost)
    nodes[edge_index + 1] = replace(nodes[edge_index + 1], incoming_association_cost=cost)
    new_chain = replace(
        chain,
        nodes=tuple(nodes),
        association_costs=tuple(
            cost if index == edge_index else value
            for index, value in enumerate(chain.association_costs)
        ),
        maximum_association_cost=max(
            cost if index == edge_index else value
            for index, value in enumerate(chain.association_costs)
        ),
        minimum_association_cost=min(
            cost if index == edge_index else value
            for index, value in enumerate(chain.association_costs)
        ),
        mean_association_cost=sum(
            cost if index == edge_index else value
            for index, value in enumerate(chain.association_costs)
        )
        / len(chain.association_costs),
        maximum_normalized_association_cost=max(
            cost if index == edge_index else value
            for index, value in enumerate(chain.association_costs)
        ),
    )
    return replace(hypothesis, chain=new_chain)


def _split_hypothesis():
    pp = _ref("pp", "spA", 200.0, 0)
    p = _ref("p", "spB", 200.0, 1)
    mf0 = _ref("mf", "spC", 198.5, 2)
    mf1 = _ref("mf", "spD", 201.5, 3)
    pairs = (
        build_cross_condition_candidate_matches((pp,), (p,), AdjacentDynamicConditionPair("pp", "p")),
        build_cross_condition_candidate_matches((p,), (mf0, mf1), AdjacentDynamicConditionPair("p", "mf")),
    )
    result = build_cross_condition_candidate_chains(pairs, ("pp", "p", "mf"))
    chain = next(item for item in result.chains if item.match_count == 2)
    return build_modal_hypotheses(
        (chain,),
        ModalHypothesisSettings(
            maximum_ambiguous_match_fraction=1.0,
            maximum_step_absolute_frequency_change_hz=10.0,
            maximum_total_absolute_frequency_change_hz=10.0,
            maximum_frequency_trajectory_rmse_hz=10.0,
        ),
        ("pp", "p", "mf"),
    ).hypotheses[0]


def test_basic_frequency_estimate_and_trajectory_are_quantitative_and_deterministic() -> None:
    hypothesis = _hypothesis((100.0, 100.4, 100.8, 101.1, 101.6), prefix="BF")
    settings = ModalParameterEstimationSettings(
        frequency_uncertainty_method=ParameterUncertaintyMethod.DISABLED,
        tau_uncertainty_method=ParameterUncertaintyMethod.DISABLED,
    )

    first = estimate_modal_parameters_for_hypothesis(hypothesis, settings)
    second = estimate_modal_parameters_for_hypothesis(hypothesis, settings)
    frequency = first.frequency_estimate
    trajectory = first.frequency_trajectory

    assert first == second
    assert first.status is ModalParameterEstimateStatus.VALID
    assert frequency.representative_frequency_hz == pytest.approx(100.78)
    assert frequency.frequency_mean_hz == pytest.approx(100.78)
    assert frequency.frequency_median_hz == pytest.approx(100.8)
    assert frequency.minimum_frequency_hz == pytest.approx(100.0)
    assert frequency.maximum_frequency_hz == pytest.approx(101.6)
    assert frequency.frequency_range_hz == pytest.approx(1.6)
    assert frequency.relative_frequency_range == pytest.approx(1.6 / 100.78)
    assert frequency.frequency_standard_deviation_hz == pytest.approx(sqrt(1.528 / 5.0))
    assert frequency.frequency_mad_hz == pytest.approx(0.4)
    assert frequency.frequency_coefficient_of_variation == pytest.approx(sqrt(1.528 / 5.0) / 100.78)
    assert trajectory.signed_step_changes_hz == pytest.approx((0.4, 0.4, 0.3, 0.5))
    assert trajectory.total_signed_change_hz == pytest.approx(1.6)
    assert trajectory.total_absolute_change_hz == pytest.approx(1.6)
    assert trajectory.linear_slope_hz_per_condition_step == pytest.approx(0.39)
    assert trajectory.linear_fit_intercept_hz == pytest.approx(100.0)
    assert trajectory.linear_fit_rmse_hz == pytest.approx(sqrt(0.007 / 5.0))
    assert "condition_variation_is_not_nonlinearity_proof" in first.diagnostics


def test_nonmonotonic_frequency_trajectory_is_accepted_without_physical_interpretation() -> None:
    hypothesis = _hypothesis((100.0, 100.8, 100.2, 101.0, 100.4), prefix="NM")

    estimate = estimate_modal_parameters_for_hypothesis(hypothesis)
    trajectory = estimate.frequency_trajectory

    assert estimate.status is ModalParameterEstimateStatus.VALID
    assert trajectory.valid is True
    assert trajectory.signed_step_changes_hz == pytest.approx((0.8, -0.6, 0.8, -0.6))
    assert trajectory.up_step_count == 2
    assert trajectory.down_step_count == 2
    assert "no_hardening_or_softening_classification" in trajectory.diagnostics
    assert "no_linearity_or_nonlinearity_proof" in trajectory.diagnostics


@pytest.mark.parametrize(
    ("frequencies", "status", "reason"),
    [
        ((100.0, 100.1, 99.9, 100.2), ModalParameterEstimateStatus.VALID, None),
        ((100.0, 110.0, 90.0, 120.0), ModalParameterEstimateStatus.INSUFFICIENT_EVIDENCE, ModalParameterEstimateReason.EXCESSIVE_FREQUENCY_DISPERSION),
    ],
    ids=["low_dispersion", "high_dispersion"],
)
def test_frequency_dispersion_limits_drive_status_and_reasons(frequencies, status, reason) -> None:
    labels = FULL_SEQUENCE[: len(frequencies)]
    hypothesis = _hypothesis(frequencies, labels=labels, prefix="DP")

    estimate = estimate_modal_parameters_for_hypothesis(hypothesis)

    assert estimate.status is status
    if reason is not None:
        assert reason in estimate.insufficient_evidence_reasons
        assert estimate.frequency_estimate.passes_dispersion_limits is False


@pytest.mark.parametrize(
    ("limit_delta", "passes"),
    [(0.001, True), (0.0, True), (-0.001, False)],
    ids=["above", "equal", "below"],
)
def test_frequency_relative_range_limit_is_inclusive(limit_delta: float, passes: bool) -> None:
    hypothesis = _hypothesis((100.0, 101.0), labels=("pp", "p"), prefix="RL")
    exact = 1.0 / 100.5
    settings = ModalParameterEstimationSettings(
        maximum_frequency_relative_range=exact + limit_delta,
        maximum_frequency_coefficient_of_variation=None,
        tau_uncertainty_method=ParameterUncertaintyMethod.DISABLED,
    )

    frequency = estimate_modal_frequency(hypothesis, settings)

    assert frequency.passes_dispersion_limits is passes


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        (ParameterLocationMethod.ARITHMETIC_MEAN, (100.0 + 110.0 + 130.0) / 3.0),
        (ParameterLocationMethod.MEDIAN, 110.0),
        (ParameterLocationMethod.WEIGHTED_MEAN, 121.0),
        (ParameterLocationMethod.WEIGHTED_MEDIAN, 130.0),
    ],
    ids=["mean", "median", "weighted_mean", "weighted_median"],
)
def test_frequency_location_methods_are_explicit(method, expected) -> None:
    hypothesis = _hypothesis(
        (100.0, 110.0, 130.0),
        labels=("pp", "p", "mf"),
        prefix="FL",
        per_ref={
            0: {"coverage": 0.1},
            1: {"coverage": 0.3},
            2: {"coverage": 0.6},
        },
    )
    settings = ModalParameterEstimationSettings(
        frequency_location_method=method,
        frequency_weighting_method=ParameterWeightingMethod.TRACKING_COVERAGE,
        maximum_frequency_coefficient_of_variation=None,
        maximum_frequency_relative_range=None,
    )

    frequency = estimate_modal_frequency(hypothesis, settings)

    assert frequency.representative_frequency_hz == pytest.approx(expected)
    assert frequency.normalized_weights == pytest.approx((0.1, 0.3, 0.6))


@pytest.mark.parametrize(
    ("weighting_method", "expected_weights"),
    [
        (ParameterWeightingMethod.UNIFORM, (1.0, 1.0, 1.0)),
        (ParameterWeightingMethod.TRACKING_COVERAGE, (0.2, 0.3, 0.5)),
        (ParameterWeightingMethod.AMPLITUDE_FIT_QUALITY, (0.5, 0.75, 1.0)),
        (ParameterWeightingMethod.FREQUENCY_FIT_QUALITY, (1.0, 0.5, 0.25)),
        (ParameterWeightingMethod.INVERSE_ASSOCIATION_COST, (100.0 / 101.0, 100.0 / 101.0, 100.0 / 101.0)),
    ],
    ids=["uniform", "coverage", "amplitude_quality", "frequency_fit_quality", "inverse_cost"],
)
def test_weighting_methods_record_original_and_normalized_weights(weighting_method, expected_weights) -> None:
    hypothesis = _hypothesis(
        (100.0, 101.0, 102.0),
        labels=("pp", "p", "mf"),
        prefix="WG",
        per_ref={
            0: {"coverage": 0.2, "amplitude_quality": 0.5, "rmse": 0.0},
            1: {"coverage": 0.3, "amplitude_quality": 0.75, "rmse": 1.0},
            2: {"coverage": 0.5, "amplitude_quality": 1.0, "rmse": 3.0},
        },
    )
    settings = ModalParameterEstimationSettings(
        frequency_weighting_method=weighting_method,
        frequency_location_method=ParameterLocationMethod.WEIGHTED_MEAN,
        maximum_frequency_coefficient_of_variation=None,
        maximum_frequency_relative_range=None,
    )

    frequency = estimate_modal_frequency(hypothesis, settings)
    total = sum(expected_weights)

    assert frequency.weights == pytest.approx(expected_weights)
    assert frequency.normalized_weights == pytest.approx(tuple(value / total for value in expected_weights))
    assert frequency.valid is True


def test_combined_weighting_policy_is_explicit_and_documented() -> None:
    hypothesis = _hypothesis(
        (100.0, 101.0, 102.0),
        labels=("pp", "p", "mf"),
        prefix="CW",
        per_ref={
            0: {"coverage": 0.5, "amplitude_quality": 0.8, "rmse": 0.0},
            1: {"coverage": 1.0, "amplitude_quality": 0.5, "rmse": 1.0},
            2: {"coverage": 0.25, "amplitude_quality": 1.0, "rmse": 3.0},
        },
    )
    settings = ModalParameterEstimationSettings(
        frequency_weighting_method=ParameterWeightingMethod.COMBINED_QUALITY_COVERAGE,
        frequency_location_method=ParameterLocationMethod.WEIGHTED_MEAN,
        maximum_frequency_coefficient_of_variation=None,
        maximum_frequency_relative_range=None,
    )

    frequency = estimate_modal_frequency(hypothesis, settings)

    assert "weighting_method=combined_quality_coverage" in frequency.diagnostics
    assert all(weight is not None and weight >= 0.0 for weight in frequency.weights)
    assert sum(frequency.normalized_weights) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"coverage_fraction": -0.1}, ModalParameterEstimateReason.INVALID_WEIGHT_VALUE),
        ({"coverage_fraction": float("nan")}, ModalParameterEstimateReason.INVALID_WEIGHT_VALUE),
        ({"coverage_fraction": float("inf")}, ModalParameterEstimateReason.INVALID_WEIGHT_VALUE),
        ({"coverage_fraction": None}, ModalParameterEstimateReason.MISSING_WEIGHT_VALUE),
    ],
    ids=["negative", "nan", "infinite", "missing"],
)
def test_invalid_or_missing_weights_fail_explicitly(changes, reason) -> None:
    hypothesis = _hypothesis((100.0, 100.1), labels=("pp", "p"), prefix="IW")
    corrupted = _corrupt_node_ref(hypothesis, 0, **changes)
    settings = ModalParameterEstimationSettings(
        frequency_weighting_method=ParameterWeightingMethod.TRACKING_COVERAGE,
        frequency_location_method=ParameterLocationMethod.WEIGHTED_MEAN,
    )

    estimate = estimate_modal_parameters_for_hypothesis(corrupted, settings)

    assert reason in estimate.frequency_estimate.reasons
    assert estimate.status in {
        ModalParameterEstimateStatus.INVALID_INPUT,
        ModalParameterEstimateStatus.INSUFFICIENT_EVIDENCE,
    }


def test_all_zero_weights_are_insufficient_and_not_normalized_silently() -> None:
    hypothesis = _hypothesis(
        (100.0, 100.1),
        labels=("pp", "p"),
        prefix="ZW",
        per_ref={0: {"coverage": 0.0}, 1: {"coverage": 0.0}},
    )
    settings = ModalParameterEstimationSettings(
        frequency_weighting_method=ParameterWeightingMethod.TRACKING_COVERAGE,
        frequency_location_method=ParameterLocationMethod.WEIGHTED_MEAN,
    )

    frequency = estimate_modal_frequency(hypothesis, settings)

    assert frequency.normalized_weights == (None, None)
    assert ModalParameterEstimateReason.INSUFFICIENT_WEIGHT_VALUES in frequency.reasons
    assert "all_weights_zero" in frequency.diagnostics


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        (ParameterLocationMethod.ARITHMETIC_MEAN, 4.05),
        (ParameterLocationMethod.MEDIAN, 4.05),
        (ParameterLocationMethod.GEOMETRIC_MEAN, exp(sum(log(value) for value in (4.0, 4.2, 3.9, 4.1)) / 4.0)),
        (ParameterLocationMethod.GEOMETRIC_MEDIAN, exp((log(4.0) + log(4.1)) / 2.0)),
    ],
    ids=["mean", "median", "geometric_mean", "geometric_median"],
)
def test_tau_location_methods_use_positive_values_and_log_domain(method, expected) -> None:
    taus = (4.0, 4.2, 3.9, 4.1)
    hypothesis = _hypothesis(
        (100.0, 100.1, 100.2, 100.3),
        labels=("pp", "p", "mf", "f"),
        prefix="TL",
        per_ref={index: {"tau": tau} for index, tau in enumerate(taus)},
    )
    settings = ModalParameterEstimationSettings(tau_location_method=method)

    decay = estimate_modal_decay(hypothesis, settings)
    log_values = tuple(log(value) for value in taus)
    log_median = (sorted(log_values)[1] + sorted(log_values)[2]) / 2.0
    absolute_log_deviations = sorted(abs(value - log_median) for value in log_values)
    expected_log_mad = (
        absolute_log_deviations[1] + absolute_log_deviations[2]
    ) / 2.0

    assert decay.valid is True
    assert decay.representative_tau_s == pytest.approx(expected)
    assert decay.log_tau_mean == pytest.approx(sum(log(value) for value in taus) / 4.0)
    assert decay.log_tau_standard_deviation == pytest.approx(
        sqrt(sum((log(value) - decay.log_tau_mean) ** 2 for value in taus) / 4.0)
    )
    assert decay.log_tau_mad == pytest.approx(expected_log_mad)
    assert "no_q_factor_or_bandwidth_derived" in decay.diagnostics


def test_high_tau_dispersion_is_insufficient_evidence() -> None:
    hypothesis = _hypothesis(
        (100.0, 100.1, 100.2, 100.3),
        labels=("pp", "p", "mf", "f"),
        prefix="TD",
        per_ref={index: {"tau": tau} for index, tau in enumerate((1.0, 8.0, 0.7, 10.0))},
    )

    estimate = estimate_modal_parameters_for_hypothesis(hypothesis)

    assert estimate.decay_estimate.valid is False
    assert ModalParameterEstimateReason.EXCESSIVE_DECAY_DISPERSION in estimate.insufficient_evidence_reasons
    assert estimate.status is ModalParameterEstimateStatus.INSUFFICIENT_EVIDENCE


def test_frequency_valid_tau_absent_and_tau_optional_is_partial() -> None:
    hypothesis = _hypothesis(
        (100.0, 100.1, 100.2),
        labels=("pp", "p", "mf"),
        prefix="PT",
        per_ref={index: {"tau": None} for index in range(3)},
        hypothesis_settings=replace(RELAXED_HYPOTHESIS_SETTINGS, missing_decay_evidence_policy="allow"),
    )

    estimate = estimate_modal_parameters_for_hypothesis(hypothesis)

    assert estimate.frequency_estimate.valid is True
    assert estimate.decay_estimate.valid is False
    assert estimate.decay_estimate.representative_tau_s is None
    assert estimate.status is ModalParameterEstimateStatus.PARTIAL


@pytest.mark.parametrize("bad_frequency", [0.0, -1.0, float("nan"), float("inf")], ids=["zero", "negative", "nan", "inf"])
def test_invalid_frequency_values_are_invalid_input_by_default(bad_frequency: float) -> None:
    hypothesis = _hypothesis((100.0, 100.1, 100.2), labels=("pp", "p", "mf"), prefix="IF")
    corrupted = _corrupt_node_ref(hypothesis, 1, representative_frequency_hz=bad_frequency)

    estimate = estimate_modal_parameters_for_hypothesis(corrupted)

    assert estimate.status is ModalParameterEstimateStatus.INVALID_INPUT
    assert ModalParameterEstimateReason.INVALID_FREQUENCY_VALUE in estimate.invalid_reasons
    assert estimate.frequency_estimate.missing_value_count == 1


@pytest.mark.parametrize("bad_tau", [0.0, -1.0, float("nan"), float("inf")], ids=["zero", "negative", "nan", "inf"])
def test_invalid_tau_values_are_invalid_input_by_default(bad_tau: float) -> None:
    hypothesis = _hypothesis((100.0, 100.1, 100.2), labels=("pp", "p", "mf"), prefix="IT")
    corrupted = _corrupt_node_ref(hypothesis, 1, amplitude_tau_s=bad_tau)

    estimate = estimate_modal_parameters_for_hypothesis(corrupted)

    assert estimate.status is ModalParameterEstimateStatus.INVALID_INPUT
    assert ModalParameterEstimateReason.INVALID_DECAY_VALUE in estimate.invalid_reasons
    assert bad_tau not in estimate.decay_estimate.tau_values_s


def test_invalid_values_can_be_excluded_only_by_explicit_policy() -> None:
    hypothesis = _hypothesis((100.0, 100.1, 100.2), labels=("pp", "p", "mf"), prefix="EX")
    corrupted = _corrupt_node_ref(hypothesis, 1, amplitude_tau_s=0.0)
    settings = ModalParameterEstimationSettings(
        finite_value_policy=FiniteValuePolicy.EXCLUDE_WITH_DIAGNOSTIC,
        allow_missing_tau=True,
    )

    estimate = estimate_modal_parameters_for_hypothesis(corrupted, settings)

    assert estimate.status is ModalParameterEstimateStatus.VALID_WITH_RESERVATIONS
    assert ModalParameterEstimateReason.INVALID_DECAY_VALUE in estimate.decay_estimate.reasons
    assert ModalParameterEstimateReason.INVALID_DECAY_VALUE not in estimate.invalid_reasons


@pytest.mark.parametrize(
    ("method", "standard"),
    [
        (ParameterUncertaintyMethod.SAMPLE_STANDARD_DEVIATION, 1.0),
        (ParameterUncertaintyMethod.STANDARD_ERROR, 1.0 / sqrt(3.0)),
        (ParameterUncertaintyMethod.SCALED_MAD, 1.4826),
    ],
    ids=["sample_std", "standard_error", "scaled_mad"],
)
def test_frequency_uncertainty_methods(method, standard) -> None:
    hypothesis = _hypothesis((100.0, 101.0, 102.0), labels=("pp", "p", "mf"), prefix="FU")
    settings = ModalParameterEstimationSettings(
        frequency_uncertainty_method=method,
        tau_uncertainty_method=ParameterUncertaintyMethod.DISABLED,
        maximum_frequency_coefficient_of_variation=None,
        maximum_frequency_relative_range=None,
    )
    frequency = estimate_modal_frequency(hypothesis, settings)

    uncertainty = estimate_modal_frequency_uncertainty(frequency, hypothesis, settings)

    assert uncertainty.valid is True
    assert uncertainty.standard_uncertainty_hz == pytest.approx(standard)
    assert uncertainty.lower_bound_hz <= frequency.representative_frequency_hz <= uncertainty.upper_bound_hz


def test_frequency_bootstrap_uncertainty_is_seeded_and_operational() -> None:
    hypothesis = _hypothesis((100.0, 104.0, 110.0), labels=("pp", "p", "mf"), prefix="FB")
    base_settings = ModalParameterEstimationSettings(
        frequency_uncertainty_method=ParameterUncertaintyMethod.BOOTSTRAP_PERCENTILE,
        bootstrap_sample_count=200,
        bootstrap_random_seed=11,
        maximum_frequency_coefficient_of_variation=None,
        maximum_frequency_relative_range=None,
    )
    frequency = estimate_modal_frequency(hypothesis, base_settings)

    first = estimate_modal_frequency_uncertainty(frequency, hypothesis, base_settings)
    second = estimate_modal_frequency_uncertainty(frequency, hypothesis, base_settings)
    other_seed = estimate_modal_frequency_uncertainty(
        frequency,
        hypothesis,
        replace(base_settings, bootstrap_random_seed=12),
    )

    assert first == second
    assert first.valid is True
    assert first.bootstrap_sample_count == 200
    assert first.random_seed == 11
    assert first.lower_bound_hz <= frequency.representative_frequency_hz <= first.upper_bound_hz
    assert (
        first.standard_uncertainty_hz,
        first.lower_bound_hz,
        first.upper_bound_hz,
    ) != (
        other_seed.standard_uncertainty_hz,
        other_seed.lower_bound_hz,
        other_seed.upper_bound_hz,
    )


def test_frequency_uncertainty_single_two_identical_and_disabled_cases() -> None:
    one = _corrupt_node_ref(
        _hypothesis((100.0, 101.0), labels=("pp", "p"), prefix="F1"),
        1,
        representative_frequency_hz=float("nan"),
    )
    two = _hypothesis((100.0, 102.0), labels=("pp", "p"), prefix="F2")
    identical = _hypothesis((100.0, 100.0, 100.0), labels=("pp", "p", "mf"), prefix="FI")

    one_frequency = estimate_modal_frequency(
        one,
        ModalParameterEstimationSettings(
            minimum_frequency_value_count=1,
            finite_value_policy=FiniteValuePolicy.EXCLUDE_WITH_DIAGNOSTIC,
        ),
    )
    one_uncertainty = estimate_modal_frequency_uncertainty(
        one_frequency,
        one,
        ModalParameterEstimationSettings(
            minimum_frequency_value_count=1,
            frequency_uncertainty_method=ParameterUncertaintyMethod.SAMPLE_STANDARD_DEVIATION,
        ),
    )
    two_uncertainty = estimate_modal_frequency_uncertainty(
        estimate_modal_frequency(two),
        two,
        ModalParameterEstimationSettings(frequency_uncertainty_method=ParameterUncertaintyMethod.SAMPLE_STANDARD_DEVIATION),
    )
    identical_uncertainty = estimate_modal_frequency_uncertainty(
        estimate_modal_frequency(identical),
        identical,
        ModalParameterEstimationSettings(frequency_uncertainty_method=ParameterUncertaintyMethod.SAMPLE_STANDARD_DEVIATION),
    )
    disabled = estimate_modal_frequency_uncertainty(
        estimate_modal_frequency(identical),
        identical,
        ModalParameterEstimationSettings(frequency_uncertainty_method=ParameterUncertaintyMethod.DISABLED),
    )

    assert one_uncertainty.valid is False
    assert ModalParameterEstimateReason.INSUFFICIENT_FREQUENCY_VALUES in one_uncertainty.reasons
    assert two_uncertainty.valid is True
    assert identical_uncertainty.standard_uncertainty_hz == pytest.approx(0.0)
    assert disabled.valid is False
    assert "frequency_uncertainty_method_disabled" in disabled.diagnostics


@pytest.mark.parametrize(
    ("method", "valid"),
    [
        (ParameterUncertaintyMethod.LOG_STANDARD_DEVIATION, True),
        (ParameterUncertaintyMethod.LOG_STANDARD_ERROR, True),
        (ParameterUncertaintyMethod.LOG_SCALED_MAD, True),
        (ParameterUncertaintyMethod.LOG_BOOTSTRAP_PERCENTILE, True),
        (ParameterUncertaintyMethod.DISABLED, False),
    ],
    ids=["log_std", "log_sem", "log_mad", "log_bootstrap", "disabled"],
)
def test_tau_uncertainty_methods_are_log_domain(method, valid) -> None:
    hypothesis = _hypothesis(
        (100.0, 100.1, 100.2, 100.3),
        labels=("pp", "p", "mf", "f"),
        prefix="DU",
        per_ref={index: {"tau": tau} for index, tau in enumerate((4.0, 4.2, 3.9, 4.1))},
    )
    settings = ModalParameterEstimationSettings(
        tau_uncertainty_method=method,
        bootstrap_sample_count=200,
        bootstrap_random_seed=7,
    )
    decay = estimate_modal_decay(hypothesis, settings)

    uncertainty = estimate_modal_decay_uncertainty(decay, settings)

    assert uncertainty.valid is valid
    if valid:
        assert uncertainty.standard_uncertainty_log_tau is not None
        assert uncertainty.multiplicative_uncertainty_factor >= 1.0
        assert uncertainty.lower_bound_tau_s <= decay.representative_tau_s <= uncertainty.upper_bound_tau_s
        assert uncertainty.lower_bound_tau_s > 0.0
        assert uncertainty.upper_bound_tau_s > 0.0
    else:
        assert "decay_uncertainty_method_disabled" in uncertainty.diagnostics


def test_tau_uncertainty_identical_single_and_seed_cases() -> None:
    identical = _hypothesis(
        (100.0, 100.1, 100.2),
        labels=("pp", "p", "mf"),
        prefix="DI",
        per_ref={index: {"tau": 4.0} for index in range(3)},
    )
    single = _hypothesis(
        (100.0, 100.1),
        labels=("pp", "p"),
        prefix="DS",
        per_ref={0: {"tau": 4.0}, 1: {"tau": None}},
    )
    settings = ModalParameterEstimationSettings(
        tau_uncertainty_method=ParameterUncertaintyMethod.LOG_BOOTSTRAP_PERCENTILE,
        bootstrap_sample_count=200,
        bootstrap_random_seed=3,
        minimum_frequency_value_count=1,
        minimum_tau_value_count=1,
    )

    identical_uncertainty = estimate_modal_decay_uncertainty(estimate_modal_decay(identical, settings), settings)
    single_uncertainty = estimate_modal_decay_uncertainty(estimate_modal_decay(single, settings), settings)
    same_seed = estimate_modal_decay_uncertainty(estimate_modal_decay(identical, settings), settings)
    other_seed = estimate_modal_decay_uncertainty(
        estimate_modal_decay(identical, replace(settings, bootstrap_random_seed=4)),
        replace(settings, bootstrap_random_seed=4),
    )

    assert identical_uncertainty.valid is True
    assert identical_uncertainty.standard_uncertainty_log_tau == pytest.approx(0.0)
    assert single_uncertainty.valid is False
    assert ModalParameterEstimateReason.INSUFFICIENT_DECAY_VALUES in single_uncertainty.reasons
    assert identical_uncertainty == same_seed
    assert identical_uncertainty.standard_uncertainty_log_tau == other_seed.standard_uncertainty_log_tau
    assert identical_uncertainty.lower_bound_tau_s == other_seed.lower_bound_tau_s
    assert identical_uncertainty.upper_bound_tau_s == other_seed.upper_bound_tau_s
    assert other_seed.random_seed == 4


def test_decay_rate_and_db_times_follow_documented_amplitude_convention() -> None:
    rate = estimate_modal_decay_rate(2.0)

    assert rate.valid is True
    assert rate.amplitude_decay_rate_per_s == pytest.approx(0.5)
    assert rate.energy_decay_rate_per_s == pytest.approx(1.0)
    assert rate.time_to_inverse_e_s == pytest.approx(2.0)
    assert rate.time_to_minus_20_db_s == pytest.approx(2.0 * log(10.0))
    assert rate.time_to_minus_40_db_s == pytest.approx(4.0 * log(10.0))
    assert rate.time_to_minus_60_db_s == pytest.approx(6.0 * log(10.0))
    assert "A(t)=A0 exp(-t/tau)" in rate.convention
    assert "no_q_factor_or_bandwidth_derived" in rate.diagnostics


@pytest.mark.parametrize("tau", [1e-9, 1e9], ids=["small", "large"])
def test_decay_rate_accepts_extreme_positive_tau_without_nan_or_infinity(tau: float) -> None:
    rate = estimate_modal_decay_rate(tau)

    assert rate.valid is True
    assert all(
        isfinite(value)
        for value in (
            rate.amplitude_decay_rate_per_s,
            rate.energy_decay_rate_per_s,
            rate.time_to_inverse_e_s,
            rate.time_to_minus_20_db_s,
            rate.time_to_minus_40_db_s,
            rate.time_to_minus_60_db_s,
        )
    )


@pytest.mark.parametrize("tau", [None, 0.0, -1.0, float("nan"), float("inf")], ids=["none", "zero", "negative", "nan", "inf"])
def test_decay_rate_invalid_tau_is_explicit(tau) -> None:
    rate = estimate_modal_decay_rate(tau)

    assert rate.valid is False
    expected_reason = (
        ModalParameterEstimateReason.INSUFFICIENT_DECAY_VALUES
        if tau is None
        else ModalParameterEstimateReason.INVALID_DECAY_VALUE
    )
    assert expected_reason in rate.reasons


def test_status_policy_for_hypothesis_states_and_audit_flags() -> None:
    accepted = _hypothesis((100.0, 100.1, 100.2), labels=("pp", "p", "mf"), prefix="SA")
    reserved = _split_hypothesis()
    inconclusive = _with_hypothesis_status(
        _hypothesis((101.0, 101.1, 101.2), labels=("pp", "p", "mf"), prefix="SI"),
        ModalHypothesisStatus.INCONCLUSIVE,
    )
    rejected = _with_hypothesis_status(
        _hypothesis((100.0, 100.1), labels=("pp", "p"), prefix="SR"),
        ModalHypothesisStatus.REJECTED,
    )
    insufficient = _with_hypothesis_status(
        _hypothesis((100.0, 100.1), labels=("pp", "p"), prefix="SE"),
        ModalHypothesisStatus.INSUFFICIENT_EVIDENCE,
    )
    invalid = estimate_modal_parameters_for_hypothesis(object())
    audit_settings = ModalParameterEstimationSettings(
        allow_inconclusive_hypotheses=True,
        allow_rejected_hypotheses_for_audit=True,
        allow_insufficient_evidence_hypotheses_for_audit=True,
    )

    assert estimate_modal_parameters_for_hypothesis(accepted).status is ModalParameterEstimateStatus.VALID
    assert estimate_modal_parameters_for_hypothesis(reserved).status is ModalParameterEstimateStatus.VALID_WITH_RESERVATIONS
    assert estimate_modal_parameters_for_hypothesis(inconclusive).status is ModalParameterEstimateStatus.INSUFFICIENT_EVIDENCE
    assert estimate_modal_parameters_for_hypothesis(inconclusive, audit_settings).status is ModalParameterEstimateStatus.VALID_WITH_RESERVATIONS
    assert estimate_modal_parameters_for_hypothesis(rejected, audit_settings).status is ModalParameterEstimateStatus.INSUFFICIENT_EVIDENCE
    assert estimate_modal_parameters_for_hypothesis(insufficient, audit_settings).status is ModalParameterEstimateStatus.INSUFFICIENT_EVIDENCE
    assert invalid.status is ModalParameterEstimateStatus.INVALID_INPUT
    assert invalid.valid is False


def test_global_result_partitions_all_hypotheses_and_preserves_provenance() -> None:
    accepted = _hypothesis((100.0, 100.1, 100.2), labels=("pp", "p", "mf"), prefix="GA", candidate_start=0)
    frequency_only = _hypothesis(
        (110.0, 110.1, 110.2),
        labels=("pp", "p", "mf"),
        prefix="GB",
        candidate_start=10,
        per_ref={index: {"tau": None} for index in range(3)},
        hypothesis_settings=replace(RELAXED_HYPOTHESIS_SETTINGS, missing_decay_evidence_policy="allow"),
    )
    reserved = _split_hypothesis()
    inconclusive = _with_hypothesis_status(
        _hypothesis(
            (120.0, 120.1, 120.2),
            labels=("pp", "p", "mf"),
            prefix="GC",
            candidate_start=20,
        ),
        ModalHypothesisStatus.INCONCLUSIVE,
    )
    rejected = _with_hypothesis_status(
        _hypothesis(
            (130.0, 130.1),
            labels=("pp", "p"),
            prefix="GD",
            candidate_start=30,
        ),
        ModalHypothesisStatus.REJECTED,
    )
    invalid_frequency = _corrupt_node_ref(
        _hypothesis((140.0, 140.1, 140.2), labels=("pp", "p", "mf"), prefix="GE", candidate_start=40),
        1,
        representative_frequency_hz=float("nan"),
    )
    high_tau = _hypothesis(
        (150.0, 150.1, 150.2, 150.3),
        labels=("pp", "p", "mf", "f"),
        prefix="GF",
        candidate_start=50,
        per_ref={index: {"tau": tau} for index, tau in enumerate((1.0, 8.0, 0.7, 10.0))},
    )
    settings = ModalParameterEstimationSettings(
        allow_inconclusive_hypotheses=True,
        allow_rejected_hypotheses_for_audit=True,
        allow_insufficient_evidence_hypotheses_for_audit=True,
    )
    hypothesis_result = _modal_hypothesis_result(
        (
            accepted,
            frequency_only,
            reserved,
            inconclusive,
            rejected,
            invalid_frequency,
            high_tau,
        )
    )

    result = estimate_modal_parameters(hypothesis_result, settings)

    assert result.estimate_count == 7
    assert result.source_hypothesis_count == 7
    assert len({item.estimate_id for item in result.estimates}) == 7
    assert result.valid_count >= 1
    assert result.valid_with_reservations_count >= 1
    assert result.partial_count >= 1
    assert result.insufficient_evidence_count >= 2
    assert result.invalid_count == 1
    assert result.valid is False
    assert result.failure_reason == "invalid_estimates_present"
    assert result.valid_estimates == tuple(
        item for item in result.estimates if item.status is ModalParameterEstimateStatus.VALID
    )
    for estimate in result.estimates:
        assert estimate.provenance.hypothesis_id == estimate.hypothesis_id
        assert estimate.provenance.settings_fingerprint == settings_fingerprint(settings)


def test_provenance_links_to_candidates_matches_contexts_and_settings() -> None:
    hypothesis = _split_hypothesis()

    provenance = estimate_modal_parameter_provenance(hypothesis)

    assert provenance.hypothesis_id == hypothesis.hypothesis_id
    assert provenance.source_chain_id == hypothesis.source_chain_id
    assert provenance.candidate_ids == tuple(node.candidate_ref.candidate_id for node in hypothesis.chain.nodes)
    assert provenance.match_ids == tuple(node.outgoing_match_id for node in hypothesis.chain.nodes if node.outgoing_match_id is not None)
    assert provenance.frequency_source_count == len(hypothesis.chain.nodes)
    assert provenance.tau_source_count == len(hypothesis.chain.nodes)
    assert provenance.ambiguous_match_ids == hypothesis.chain.ambiguous_match_ids
    assert provenance.possible_split_context_ids
    assert provenance.settings_fingerprint == settings_fingerprint(ModalParameterEstimationSettings())
    assert "settings_fingerprint_is_deterministic_no_timestamp" in provenance.diagnostics


def test_estimation_is_deterministic_across_input_orders_and_diagnostic_orders() -> None:
    hypotheses = (
        _hypothesis((100.0, 100.1, 100.2), labels=("pp", "p", "mf"), prefix="DA", candidate_start=0),
        _hypothesis((120.0, 120.1, 120.2), labels=("pp", "p", "mf"), prefix="DB", candidate_start=10),
        _split_hypothesis(),
    )
    changed_chain = replace(
        hypotheses[0].chain,
        diagnostics=tuple(reversed(hypotheses[0].chain.diagnostics)),
    )
    changed = replace(hypotheses[0], chain=changed_chain)
    settings = ModalParameterEstimationSettings(
        frequency_uncertainty_method=ParameterUncertaintyMethod.BOOTSTRAP_PERCENTILE,
        tau_uncertainty_method=ParameterUncertaintyMethod.LOG_BOOTSTRAP_PERCENTILE,
        bootstrap_sample_count=200,
        bootstrap_random_seed=17,
    )

    ordered = estimate_modal_parameters(hypotheses, settings)
    reversed_result = estimate_modal_parameters(tuple(reversed(hypotheses)), settings)
    shuffled_items = list(hypotheses)
    Random(5).shuffle(shuffled_items)
    shuffled = estimate_modal_parameters(tuple(shuffled_items), settings)
    changed_result = estimate_modal_parameters((changed,) + hypotheses[1:], settings)

    def normalized(result):
        return tuple(
            (
                item.estimate_id,
                item.hypothesis_id,
                item.status,
                item.frequency_estimate.representative_frequency_hz,
                item.frequency_trajectory.signed_step_changes_hz,
                item.frequency_uncertainty.standard_uncertainty_hz,
                item.decay_estimate.representative_tau_s,
                item.decay_uncertainty.standard_uncertainty_log_tau,
                item.provenance,
                item.supporting_reasons,
                item.reservation_reasons,
                item.insufficient_evidence_reasons,
                item.invalid_reasons,
            )
            for item in result.estimates
        )

    assert normalized(ordered) == normalized(reversed_result) == normalized(shuffled) == normalized(changed_result)
    assert summarize_modal_parameter_estimates(ordered) == summarize_modal_parameter_estimates(reversed_result)


@pytest.mark.parametrize(
    ("mutation", "settings"),
    [
        (
            lambda hypothesis: _replace_node_ref(hypothesis, 1, representative_frequency_hz=101.0),
            ModalParameterEstimationSettings(),
        ),
        (
            lambda hypothesis: _replace_node_ref(hypothesis, 1, amplitude_tau_s=4.8),
            ModalParameterEstimationSettings(),
        ),
        (
            lambda hypothesis: _replace_node_ref(hypothesis, 1, amplitude_fit_r_squared=0.5),
            ModalParameterEstimationSettings(
                frequency_location_method=ParameterLocationMethod.WEIGHTED_MEAN,
                frequency_weighting_method=ParameterWeightingMethod.AMPLITUDE_FIT_QUALITY,
            ),
        ),
        (
            lambda hypothesis: _replace_edge_cost(hypothesis, 0, 0.8),
            ModalParameterEstimationSettings(
                frequency_location_method=ParameterLocationMethod.WEIGHTED_MEAN,
                frequency_weighting_method=ParameterWeightingMethod.INVERSE_ASSOCIATION_COST,
            ),
        ),
        (
            lambda hypothesis: _replace_node_ref(hypothesis, 1, frequency_fit_rmse_hz=0.8),
            ModalParameterEstimationSettings(
                frequency_uncertainty_method=ParameterUncertaintyMethod.CONSERVATIVE,
            ),
        ),
    ],
    ids=["frequency", "tau", "fit_quality", "cost", "individual_uncertainty"],
)
def test_local_perturbation_changes_only_corresponding_estimate(mutation, settings) -> None:
    target = _hypothesis((100.0, 100.1, 100.2), labels=("pp", "p", "mf"), prefix="PA", candidate_start=0)
    other = _hypothesis((120.0, 120.1, 120.2), labels=("pp", "p", "mf"), prefix="PB", candidate_start=10)
    perturbed_target = mutation(target)

    base = estimate_modal_parameters((target, other), settings)
    perturbed = estimate_modal_parameters((perturbed_target, other), settings)
    base_by_hypothesis = {item.hypothesis_id: item for item in base.estimates}
    perturbed_by_hypothesis = {item.hypothesis_id: item for item in perturbed.estimates}

    assert base_by_hypothesis.keys() == perturbed_by_hypothesis.keys()
    assert perturbed_by_hypothesis[target.hypothesis_id] != base_by_hypothesis[target.hypothesis_id]
    assert perturbed_by_hypothesis[other.hypothesis_id] == base_by_hypothesis[other.hypothesis_id]
    assert perturbed_by_hypothesis[other.hypothesis_id].estimate_id == base_by_hypothesis[other.hypothesis_id].estimate_id


def test_inputs_are_immutable_and_repeated_builds_are_identical() -> None:
    hypotheses = (
        _hypothesis((100.0, 100.1, 100.2), labels=("pp", "p", "mf"), prefix="IM", candidate_start=0),
        _split_hypothesis(),
    )
    snapshot = deepcopy(hypotheses)
    settings = ModalParameterEstimationSettings(
        frequency_uncertainty_method=ParameterUncertaintyMethod.BOOTSTRAP_PERCENTILE,
        tau_uncertainty_method=ParameterUncertaintyMethod.LOG_BOOTSTRAP_PERCENTILE,
        bootstrap_sample_count=200,
        bootstrap_random_seed=23,
    )

    first = estimate_modal_parameters(hypotheses, settings)
    second = estimate_modal_parameters(hypotheses, settings)

    assert hypotheses == snapshot
    assert first == second
    assert all(isinstance(hypothesis.chain.nodes, tuple) for hypothesis in hypotheses)


@pytest.mark.parametrize(
    "changes",
    [
        {"minimum_frequency_value_count": 0},
        {"minimum_tau_value_count": 0},
        {"frequency_location_method": "unknown"},
        {"frequency_weighting_method": "unknown"},
        {"frequency_uncertainty_method": "unknown"},
        {"maximum_frequency_coefficient_of_variation": -0.1},
        {"maximum_frequency_relative_range": float("inf")},
        {"maximum_log_tau_range": -0.1},
        {"uncertainty_confidence_level": 1.0},
        {"bootstrap_sample_count": 0},
        {"bootstrap_random_seed": 1.5},
        {"minimum_positive_value": 0.0},
        {"finite_value_policy": "unknown"},
    ],
    ids=[
        "minimum_frequency_zero",
        "minimum_tau_zero",
        "bad_location",
        "bad_weighting",
        "bad_uncertainty",
        "negative_cv",
        "infinite_range",
        "negative_tau_range",
        "confidence_endpoint",
        "bootstrap_zero",
        "seed_float",
        "minimum_positive_zero",
        "bad_finite_policy",
    ],
)
def test_settings_reject_invalid_invariants(changes) -> None:
    with pytest.raises(ValueError):
        ModalParameterEstimationSettings(**changes)


def test_numeric_invariants_have_no_nan_no_infinity_and_no_zero_substitution() -> None:
    hypothesis = _hypothesis(
        (100.0, 100.1, 100.2, 100.3),
        labels=("pp", "p", "mf", "f"),
        prefix="NI",
        per_ref={0: {"tau": None}, 1: {"tau": 4.0}, 2: {"tau": None}, 3: {"tau": 4.2}},
    )
    settings = ModalParameterEstimationSettings()

    estimate = estimate_modal_parameters_for_hypothesis(hypothesis, settings)
    numbers = [
        estimate.frequency_estimate.representative_frequency_hz,
        estimate.frequency_estimate.frequency_range_hz,
        estimate.frequency_trajectory.total_signed_change_hz,
        estimate.frequency_uncertainty.standard_uncertainty_hz,
        estimate.decay_estimate.representative_tau_s,
        estimate.decay_rate_estimate.amplitude_decay_rate_per_s,
        estimate.decay_uncertainty.standard_uncertainty_log_tau,
    ]

    assert all(value is not None and isfinite(value) for value in numbers)
    assert 0.0 not in estimate.decay_estimate.tau_values_s
    assert estimate.decay_estimate.missing_value_count == 2
    assert estimate.decay_estimate.minimum_tau_s <= estimate.decay_estimate.representative_tau_s <= estimate.decay_estimate.maximum_tau_s
    assert estimate.frequency_estimate.minimum_frequency_hz <= estimate.frequency_estimate.representative_frequency_hz <= estimate.frequency_estimate.maximum_frequency_hz
    assert estimate.decay_rate_estimate.time_to_minus_20_db_s > 0.0
    assert estimate.decay_rate_estimate.time_to_minus_40_db_s > estimate.decay_rate_estimate.time_to_minus_20_db_s
    assert estimate.decay_rate_estimate.time_to_minus_60_db_s > estimate.decay_rate_estimate.time_to_minus_40_db_s


def test_public_summary_is_deterministic_and_contains_operational_counts() -> None:
    hypotheses = (
        _hypothesis((100.0, 100.1, 100.2), labels=("pp", "p", "mf"), prefix="SU", candidate_start=0),
        _hypothesis(
            (110.0, 110.1, 110.2),
            labels=("pp", "p", "mf"),
            prefix="SV",
            candidate_start=10,
            per_ref={index: {"tau": None} for index in range(3)},
            hypothesis_settings=replace(RELAXED_HYPOTHESIS_SETTINGS, missing_decay_evidence_policy="allow"),
        ),
    )

    result = estimate_modal_parameters(hypotheses)
    summary = summarize_modal_parameter_estimates(result)

    assert summary["estimate_count"] == 2
    assert summary["source_hypothesis_count"] == 2
    assert summary["statuses"] == tuple(item.status.value for item in result.estimates)
    assert summary["estimate_ids"] == tuple(item.estimate_id for item in result.estimates)
    assert summary["settings_fingerprint"] == settings_fingerprint(ModalParameterEstimationSettings())
