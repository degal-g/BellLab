"""Operational modal hypotheses built from candidate chains."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace
from math import exp, isclose, isfinite
from random import Random

import pytest

from belllab import (
    AdjacentDynamicConditionPair,
    CandidateReference,
    CrossConditionCandidateAssociationSettings,
    ModalHypothesisReason,
    ModalHypothesisSettings,
    ModalHypothesisStatus,
    build_cross_condition_candidate_chains,
    build_cross_condition_candidate_matches,
    build_modal_hypotheses,
    compute_modal_hypothesis_score,
    evaluate_modal_hypothesis,
    evaluate_modal_hypothesis_decay_evidence,
    evaluate_modal_hypothesis_frequency_evidence,
    summarize_modal_hypotheses,
)


FULL_SEQUENCE = ("pp", "p", "mf", "f", "ff")


def _ref(
    label: str,
    name: str,
    frequency: float,
    candidate_id: int,
    *,
    accepted: bool = True,
    impact_excited: bool | None = True,
    classification: str | None = "impact_emergent",
    tau: float | None = 4.0,
    stability: float | None = 0.01,
    drift: float | None = 0.0,
    rmse: float | None = 0.1,
    coverage: float | None = 1.0,
    ambiguous_fraction: float | None = 0.0,
    near_fraction: float | None = 0.0,
    margin: float | None = 1.0,
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
        stability,
        tau,
        0.95,
        drift,
        rmse,
        coverage,
        ambiguous_fraction,
        near_fraction,
        margin,
        classification,
        None,
        ("modal_hypothesis_test_reference",),
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
        maximum_absolute_frequency_difference_hz=10.0,
        maximum_association_cost=10.0,
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


def _corrupt_dataclass(instance, **changes):
    corrupted = object.__new__(type(instance))
    for field in fields(instance):
        object.__setattr__(
            corrupted,
            field.name,
            changes.get(field.name, getattr(instance, field.name)),
        )
    return corrupted


def _replace_node_ref(chain, node_index: int, **changes):
    nodes = list(chain.nodes)
    ref = replace(nodes[node_index].candidate_ref, **changes)
    nodes[node_index] = replace(nodes[node_index], candidate_ref=ref)
    frequencies = tuple(node.candidate_ref.representative_frequency_hz for node in nodes)
    return replace(
        chain,
        nodes=tuple(nodes),
        frequency_trajectory_hz=frequencies,
        initial_frequency_hz=frequencies[0],
        final_frequency_hz=frequencies[-1],
        total_frequency_change_hz=frequencies[-1] - frequencies[0],
        total_frequency_change_relative=(frequencies[-1] - frequencies[0])
        / (0.5 * (frequencies[-1] + frequencies[0])),
    )


def _replace_node_ref_unsafely(chain, node_index: int, **changes):
    nodes = list(chain.nodes)
    ref = _corrupt_dataclass(nodes[node_index].candidate_ref, **changes)
    nodes[node_index] = replace(nodes[node_index], candidate_ref=ref)
    return replace(chain, nodes=tuple(nodes))


def _strong_chain(prefix: str = "S", candidate_start: int = 0):
    return _chain(
        (100.0, 100.2, 100.1, 100.3, 100.2),
        prefix=prefix,
        candidate_start=candidate_start,
    )[0]


def _singleton_chain(prefix: str = "SG", candidate_start: int = 0):
    pair = build_cross_condition_candidate_matches(
        (_ref("pp", f"{prefix}a", 100.0, candidate_start),),
        (_ref("p", f"{prefix}b", 500.0, candidate_start + 1),),
        AdjacentDynamicConditionPair("pp", "p"),
    )
    result = build_cross_condition_candidate_chains((pair,), ("pp", "p"))
    return next(item for item in result.chains if item.start_dynamic_label == "pp")


def _chain_with_costs(costs: tuple[float, ...]):
    labels = FULL_SEQUENCE[: len(costs) + 1]
    base, _ = _chain(
        tuple(100.0 + 0.1 * index for index in range(len(labels))),
        labels=labels,
        prefix="K",
    )
    return replace(
        base,
        association_costs=costs,
        maximum_association_cost=max(costs) if costs else None,
        minimum_association_cost=min(costs) if costs else None,
        mean_association_cost=(sum(costs) / len(costs) if costs else None),
        maximum_normalized_association_cost=max(costs) if costs else None,
    )


def test_complete_chain_is_accepted_and_quantitative_components_are_auditable() -> None:
    chain = _strong_chain()
    hypothesis = evaluate_modal_hypothesis(chain, FULL_SEQUENCE)

    assert hypothesis.status is ModalHypothesisStatus.ACCEPTED
    assert hypothesis.accepted is True
    assert hypothesis.requires_review is False
    assert hypothesis.coverage_evidence.requested_condition_count == 5
    assert hypothesis.coverage_evidence.observed_condition_count == 5
    assert hypothesis.coverage_evidence.condition_coverage_fraction == pytest.approx(1.0)
    assert hypothesis.coverage_evidence.complete_across_requested_sequence is True
    assert hypothesis.frequency_evidence.signed_step_changes_hz == pytest.approx((0.2, -0.1, 0.2, -0.1))
    assert hypothesis.frequency_evidence.total_absolute_change_hz == pytest.approx(0.6)
    assert hypothesis.frequency_evidence.trajectory_rmse_from_mean_hz == pytest.approx(0.10198039027185603)
    assert hypothesis.association_evidence.match_costs == pytest.approx((0.02, 0.01, 0.02, 0.01))
    assert hypothesis.association_evidence.mean_match_cost == pytest.approx(0.015)
    assert hypothesis.tracking_evidence.mean_coverage_fraction == pytest.approx(1.0)
    assert hypothesis.decay_evidence.tau_values_s == pytest.approx((4.0, 4.0, 4.0, 4.0, 4.0))
    assert hypothesis.decay_evidence.log_tau_range == pytest.approx(0.0)
    assert hypothesis.impact_evidence.impact_supported_fraction == pytest.approx(1.0)
    assert hypothesis.structural_context.contains_possible_split_context is False
    assert hypothesis.score.normalized_score > 0.9
    assert ModalHypothesisReason.SUFFICIENT_FREQUENCY_CONTINUITY in hypothesis.supporting_reasons
    assert "no_modal_mode_created" in hypothesis.diagnostics


def test_subsequence_complete_chain_is_partial_for_full_requested_sequence() -> None:
    chain, _ = _chain(
        (100.0, 100.1, 100.2),
        labels=("p", "mf", "f"),
        prefix="P",
    )
    settings = ModalHypothesisSettings(
        require_complete_chain=False,
        allow_partial_chains=True,
        minimum_condition_count=3,
        minimum_match_count=2,
        minimum_condition_coverage_fraction=0.6,
    )

    hypothesis = evaluate_modal_hypothesis(chain, FULL_SEQUENCE, settings)

    assert hypothesis.coverage_evidence.complete_across_requested_sequence is False
    assert hypothesis.coverage_evidence.partial is True
    assert hypothesis.coverage_evidence.condition_coverage_fraction == pytest.approx(3 / 5)
    assert hypothesis.status is ModalHypothesisStatus.ACCEPTED_WITH_RESERVATIONS
    assert ModalHypothesisReason.PARTIAL_BUT_SUPPORTED_CHAIN in hypothesis.reservation_reasons


def test_partial_chain_is_rejected_when_complete_chain_is_required() -> None:
    chain, _ = _chain(
        (100.0, 100.1, 100.2),
        labels=("p", "mf", "f"),
        prefix="R",
    )

    hypothesis = evaluate_modal_hypothesis(chain, FULL_SEQUENCE)

    assert hypothesis.status is ModalHypothesisStatus.REJECTED
    assert ModalHypothesisReason.TOO_FEW_CONDITIONS in hypothesis.rejection_reasons


def test_singleton_chain_is_insufficient_by_default_and_not_silently_accepted() -> None:
    singleton = _singleton_chain()

    hypothesis = evaluate_modal_hypothesis(singleton, ("pp", "p"))

    assert singleton.isolated_candidate
    assert hypothesis.status is ModalHypothesisStatus.INSUFFICIENT_EVIDENCE
    assert hypothesis.accepted is False
    assert ModalHypothesisReason.SINGLETON_CHAIN in hypothesis.missing_evidence_reasons


@pytest.mark.parametrize(
    "frequencies",
    [
        (100.0, 100.2, 100.1, 100.3, 100.2),
        (100.0, 100.5, 101.0, 101.6, 102.1),
        (100.0, 99.5, 99.0, 98.4, 98.0),
        (100.0, 100.8, 100.2, 101.0, 100.4),
    ],
    ids=["stable", "increasing", "decreasing", "nonmonotonic_continuous"],
)
def test_frequency_trajectories_can_pass_without_physical_interpretation(frequencies) -> None:
    chain, _ = _chain(frequencies, prefix="F")
    evidence = evaluate_modal_hypothesis_frequency_evidence(chain)

    assert evidence.passes
    assert ModalHypothesisReason.SUFFICIENT_FREQUENCY_CONTINUITY in evidence.reasons
    assert "no_hardening_or_softening_classification" in evidence.diagnostics
    assert "no_linearity_or_nonlinearity_classification" in evidence.diagnostics
    assert "no_monotonicity_requirement" in evidence.diagnostics


def test_discontinuous_frequency_trajectory_is_rejected_by_frequency_gate() -> None:
    chain, _ = _chain(
        (100.0, 100.2, 150.0, 150.2, 150.4),
        prefix="D",
        association_settings=CrossConditionCandidateAssociationSettings(
            maximum_absolute_frequency_difference_hz=100.0,
            maximum_association_cost=10.0,
        ),
    )

    hypothesis = evaluate_modal_hypothesis(chain, FULL_SEQUENCE)

    assert hypothesis.frequency_evidence.passes is False
    assert hypothesis.status is ModalHypothesisStatus.REJECTED
    assert ModalHypothesisReason.FREQUENCY_DISCONTINUITY in hypothesis.rejection_reasons


@pytest.mark.parametrize(
    ("difference", "passes"),
    [(1.999, True), (2.0, True), (2.001, False)],
    ids=["below", "equal", "above"],
)
def test_frequency_step_limit_is_inclusive(difference: float, passes: bool) -> None:
    chain, _ = _chain(
        (100.0, 100.0 + difference),
        labels=("pp", "p"),
        prefix="L",
    )
    settings = ModalHypothesisSettings(
        maximum_step_absolute_frequency_change_hz=2.0,
        maximum_total_absolute_frequency_change_hz=100.0,
        maximum_frequency_trajectory_rmse_hz=100.0,
    )

    evidence = evaluate_modal_hypothesis_frequency_evidence(chain, settings)

    assert evidence.passes is passes


@pytest.mark.parametrize(
    ("costs", "passes", "reason"),
    [
        ((0.1, 0.2, 0.1, 0.2), True, None),
        ((0.1, 1.1, 0.1, 0.1), False, ModalHypothesisReason.EXCESSIVE_ASSOCIATION_COST),
        ((0.1, 1.1, 0.1), False, ModalHypothesisReason.EXCESSIVE_ASSOCIATION_COST),
        ((1.0,), True, None),
        ((1.01,), False, ModalHypothesisReason.EXCESSIVE_ASSOCIATION_COST),
    ],
    ids=["low_costs", "one_high", "mean_ok_max_bad", "exact_limit", "above_limit"],
)
def test_association_cost_gates_are_separate_from_frequency_gates(costs, passes, reason) -> None:
    labels = FULL_SEQUENCE[: len(costs) + 1]
    chain = _chain_with_costs(costs)
    settings = ModalHypothesisSettings(
        require_complete_chain=False,
        maximum_step_absolute_frequency_change_hz=100.0,
        maximum_total_absolute_frequency_change_hz=100.0,
        maximum_frequency_trajectory_rmse_hz=100.0,
        maximum_match_cost=1.0,
        maximum_mean_match_cost=1.0,
    )

    hypothesis = evaluate_modal_hypothesis(chain, labels, settings)

    assert hypothesis.association_evidence.passes is passes
    if reason is not None:
        assert reason in hypothesis.association_evidence.reasons


def test_ambiguous_match_can_be_preserved_as_reservation_without_branching() -> None:
    pp = _ref("pp", "amb", 100.0, 0)
    p0 = _ref("p", "amb0", 99.9, 1)
    p1 = _ref("p", "amb1", 100.1, 2)
    pair = build_cross_condition_candidate_matches((pp,), (p0, p1), AdjacentDynamicConditionPair("pp", "p"))
    result = build_cross_condition_candidate_chains((pair,), ("pp", "p"))
    chain = next(item for item in result.chains if item.match_count == 1)
    settings = ModalHypothesisSettings(maximum_ambiguous_match_fraction=1.0)

    hypothesis = evaluate_modal_hypothesis(chain, ("pp", "p"), settings)

    assert chain.contains_ambiguous_match
    assert hypothesis.association_evidence.ambiguous_match_fraction == pytest.approx(1.0)
    assert hypothesis.status is ModalHypothesisStatus.ACCEPTED_WITH_RESERVATIONS
    assert ModalHypothesisReason.EXCESSIVE_AMBIGUITY in hypothesis.reservation_reasons


def test_near_threshold_match_can_be_preserved_as_reservation() -> None:
    chain, _ = _chain(
        (100.0, 108.0),
        labels=("pp", "p"),
        prefix="N",
        association_settings=CrossConditionCandidateAssociationSettings(
            maximum_absolute_frequency_difference_hz=10.0,
            maximum_association_cost=1.0,
            near_threshold_ratio=0.8,
        ),
    )
    settings = ModalHypothesisSettings(
        maximum_step_absolute_frequency_change_hz=100.0,
        maximum_total_absolute_frequency_change_hz=100.0,
        maximum_frequency_trajectory_rmse_hz=100.0,
        maximum_near_threshold_match_fraction=1.0,
        maximum_mean_match_cost=1.0,
    )

    hypothesis = evaluate_modal_hypothesis(chain, ("pp", "p"), settings)

    assert chain.contains_near_threshold_match
    assert hypothesis.status is ModalHypothesisStatus.ACCEPTED_WITH_RESERVATIONS
    assert ModalHypothesisReason.EXCESSIVE_NEAR_THRESHOLD_FRACTION in hypothesis.reservation_reasons


def test_missing_and_small_match_margins_are_auditable() -> None:
    simple_chain, _ = _chain((100.0, 100.1), labels=("pp", "p"), prefix="M")
    margin_required = ModalHypothesisSettings(minimum_match_margin=0.1)
    missing = evaluate_modal_hypothesis(simple_chain, ("pp", "p"), margin_required)
    assert missing.association_evidence.minimum_match_margin is None
    assert ModalHypothesisReason.INSUFFICIENT_EVIDENCE in missing.association_evidence.reasons

    pp = _ref("pp", "marg", 100.0, 10)
    p0 = _ref("p", "marg0", 99.9, 11)
    p1 = _ref("p", "marg1", 100.1, 12)
    pair = build_cross_condition_candidate_matches((pp,), (p0, p1), AdjacentDynamicConditionPair("pp", "p"))
    chain = next(item for item in build_cross_condition_candidate_chains((pair,), ("pp", "p")).chains if item.match_count == 1)
    too_small = evaluate_modal_hypothesis(
        chain,
        ("pp", "p"),
        ModalHypothesisSettings(maximum_ambiguous_match_fraction=1.0, minimum_match_margin=0.1),
    )
    assert too_small.association_evidence.minimum_match_margin == pytest.approx(0.0)
    assert ModalHypothesisReason.EXCESSIVE_AMBIGUITY in too_small.rejection_reasons


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({index: {"coverage": 0.2} for index in range(5)}, ModalHypothesisReason.INSUFFICIENT_TRACKING_QUALITY),
        ({index: {"ambiguous_fraction": 0.8} for index in range(5)}, ModalHypothesisReason.INSUFFICIENT_TRACKING_QUALITY),
        ({index: {"near_fraction": 0.8} for index in range(5)}, ModalHypothesisReason.INSUFFICIENT_TRACKING_QUALITY),
        ({index: {"rmse": 3.0} for index in range(5)}, ModalHypothesisReason.INSUFFICIENT_TRACKING_QUALITY),
    ],
    ids=["low_coverage", "high_ambiguity", "high_near_threshold", "high_rmse"],
)
def test_tracking_quality_gates_reuse_candidate_metrics_only(overrides, reason) -> None:
    chain, _ = _chain(
        (100.0, 100.1, 100.2, 100.3, 100.4),
        prefix="T",
        per_ref=overrides,
    )

    hypothesis = evaluate_modal_hypothesis(chain, FULL_SEQUENCE)

    assert hypothesis.tracking_evidence.passes is False
    assert reason in hypothesis.rejection_reasons
    assert "no_tracking_recomputed" in hypothesis.tracking_evidence.diagnostics


@pytest.mark.parametrize(
    ("policy", "status"),
    [
        ("allow", ModalHypothesisStatus.ACCEPTED),
        ("reservation", ModalHypothesisStatus.ACCEPTED_WITH_RESERVATIONS),
        ("insufficient", ModalHypothesisStatus.INSUFFICIENT_EVIDENCE),
        ("reject", ModalHypothesisStatus.REJECTED),
    ],
    ids=["allow", "reservation", "insufficient", "reject"],
)
def test_missing_tracking_evidence_policy_controls_absence(policy: str, status: ModalHypothesisStatus) -> None:
    missing_metrics = {
        index: {
            "coverage": None,
            "ambiguous_fraction": None,
            "near_fraction": None,
            "margin": None,
            "rmse": None,
        }
        for index in range(5)
    }
    chain, _ = _chain((100.0, 100.1, 100.2, 100.3, 100.4), prefix="U", per_ref=missing_metrics)
    settings = ModalHypothesisSettings(missing_tracking_evidence_policy=policy)

    hypothesis = evaluate_modal_hypothesis(chain, FULL_SEQUENCE, settings)

    assert hypothesis.status is status
    assert ("coverage_fraction", 5) in hypothesis.tracking_evidence.missing_value_counts


@pytest.mark.parametrize(
    ("taus", "passes"),
    [
        ((4.0, 4.2, 3.9, 4.1, 4.0), True),
        ((4.0, 4.8, 3.7, 4.5, 4.2), True),
        ((1.0, 8.0, 0.7, 10.0, 2.0), False),
    ],
    ids=["consistent", "moderate", "inconsistent"],
)
def test_decay_tau_consistency_uses_log_domain(taus, passes) -> None:
    chain, _ = _chain(
        (100.0, 100.1, 100.2, 100.3, 100.4),
        prefix="Q",
        per_ref={index: {"tau": tau} for index, tau in enumerate(taus)},
    )

    hypothesis = evaluate_modal_hypothesis(chain, FULL_SEQUENCE)

    assert hypothesis.decay_evidence.passes is passes
    assert all(isfinite(value) for value in hypothesis.decay_evidence.log_tau_values)
    assert "no_q_factor_or_bandwidth_derived" in hypothesis.decay_evidence.diagnostics
    if not passes:
        assert ModalHypothesisReason.INCONSISTENT_DECAY in hypothesis.rejection_reasons


def test_missing_required_decay_is_insufficient_even_when_score_is_high() -> None:
    chain, _ = _chain(
        (100.0, 100.1, 100.2, 100.3, 100.4),
        prefix="Z",
        per_ref={index: {"tau": None} for index in range(5)},
    )
    settings = ModalHypothesisSettings(require_decay_evidence=True)

    hypothesis = evaluate_modal_hypothesis(chain, FULL_SEQUENCE, settings)

    assert hypothesis.status is ModalHypothesisStatus.INSUFFICIENT_EVIDENCE
    assert ModalHypothesisReason.MISSING_REQUIRED_DECAY in hypothesis.missing_evidence_reasons
    assert hypothesis.score.normalized_score > 0.7


@pytest.mark.parametrize("invalid_tau", [0.0, -1.0, float("inf")], ids=["zero", "negative", "infinite"])
def test_invalid_tau_values_are_not_used_as_zero(invalid_tau: float) -> None:
    chain = _strong_chain("I")
    corrupted = _replace_node_ref_unsafely(chain, 0, amplitude_tau_s=invalid_tau)

    evidence = evaluate_modal_hypothesis_decay_evidence(corrupted)

    assert invalid_tau not in evidence.tau_values_s
    assert evidence.missing_tau_count == 1
    assert "invalid_tau_values_ignored:1" in evidence.diagnostics


def test_decay_log_range_limit_is_inclusive() -> None:
    chain, _ = _chain(
        (100.0, 100.1),
        labels=("pp", "p"),
        prefix="E",
        per_ref={0: {"tau": 1.0}, 1: {"tau": exp(0.5)}},
    )
    settings = ModalHypothesisSettings(
        maximum_log_tau_range=0.5,
        maximum_log_tau_standard_deviation=1.0,
    )

    evidence = evaluate_modal_hypothesis_decay_evidence(chain, settings)

    assert evidence.log_tau_range == pytest.approx(0.5)
    assert evidence.passes


@pytest.mark.parametrize(
    ("supported", "require", "status"),
    [
        ((True, True, True, True, True), True, ModalHypothesisStatus.ACCEPTED),
        ((True, True, True, False, False), True, ModalHypothesisStatus.ACCEPTED),
        ((True, False, False, False, False), True, ModalHypothesisStatus.REJECTED),
        ((False, False, False, False, False), False, ModalHypothesisStatus.ACCEPTED),
    ],
    ids=["all_supported", "majority_supported", "minority_supported", "require_disabled"],
)
def test_impact_evidence_requirement_uses_existing_classifications(supported, require, status) -> None:
    per_ref = {
        index: {
            "impact_excited": value,
            "classification": "impact_emergent" if value else "persistent_background_tone",
        }
        for index, value in enumerate(supported)
    }
    chain, _ = _chain((100.0, 100.1, 100.2, 100.3, 100.4), prefix="X", per_ref=per_ref)
    settings = ModalHypothesisSettings(require_impact_excitation=require, minimum_impact_supported_fraction=0.6)

    hypothesis = evaluate_modal_hypothesis(chain, FULL_SEQUENCE, settings)

    assert hypothesis.status is status
    assert hypothesis.impact_evidence.background_persistent_count == supported.count(False)
    assert "preimpact_evidence_is_operational_not_causality" in hypothesis.impact_evidence.diagnostics


def test_missing_required_impact_evidence_is_insufficient() -> None:
    chain, _ = _chain(
        (100.0, 100.1, 100.2, 100.3, 100.4),
        prefix="Y",
        per_ref={index: {"impact_excited": None, "classification": None} for index in range(5)},
    )

    hypothesis = evaluate_modal_hypothesis(
        chain,
        FULL_SEQUENCE,
        ModalHypothesisSettings(require_impact_excitation=True),
    )

    assert hypothesis.status is ModalHypothesisStatus.INSUFFICIENT_EVIDENCE
    assert ModalHypothesisReason.MISSING_REQUIRED_IMPACT_EVIDENCE in hypothesis.missing_evidence_reasons


def _split_chain(prefix: str = "sp", candidate_start: int = 0):
    pp_a = _ref("pp", f"{prefix}A", 200.0, candidate_start)
    p_b = _ref("p", f"{prefix}B", 200.0, candidate_start + 1)
    mf_c = _ref("mf", f"{prefix}C", 198.5, candidate_start + 2)
    mf_d = _ref("mf", f"{prefix}D", 201.5, candidate_start + 3)
    pairs = (
        build_cross_condition_candidate_matches((pp_a,), (p_b,), AdjacentDynamicConditionPair("pp", "p")),
        build_cross_condition_candidate_matches((p_b,), (mf_c, mf_d), AdjacentDynamicConditionPair("p", "mf")),
    )
    result = build_cross_condition_candidate_chains(pairs, ("pp", "p", "mf"))
    return next(item for item in result.chains if item.match_count == 2)


def _merge_chain():
    mf_a = _ref("mf", "mgA", 298.5, 0)
    mf_b = _ref("mf", "mgB", 301.5, 1)
    f_c = _ref("f", "mgC", 300.0, 2)
    pair = build_cross_condition_candidate_matches((mf_a, mf_b), (f_c,), AdjacentDynamicConditionPair("mf", "f"))
    result = build_cross_condition_candidate_chains((pair,), ("mf", "f"))
    return next(item for item in result.chains if item.match_count == 1)


@pytest.mark.parametrize(
    ("chain_factory", "settings", "status", "reason"),
    [
        (_split_chain, ModalHypothesisSettings(reserve_possible_split_context=True, maximum_ambiguous_match_fraction=1.0), ModalHypothesisStatus.ACCEPTED_WITH_RESERVATIONS, ModalHypothesisReason.POSSIBLE_SPLIT_CONTEXT),
        (_split_chain, ModalHypothesisSettings(reject_possible_split_context=True, reserve_possible_split_context=False, maximum_ambiguous_match_fraction=1.0), ModalHypothesisStatus.REJECTED, ModalHypothesisReason.POSSIBLE_SPLIT_CONTEXT),
        (_merge_chain, ModalHypothesisSettings(reserve_possible_merge_context=True, maximum_ambiguous_match_fraction=1.0), ModalHypothesisStatus.ACCEPTED_WITH_RESERVATIONS, ModalHypothesisReason.POSSIBLE_MERGE_CONTEXT),
        (_merge_chain, ModalHypothesisSettings(reject_possible_merge_context=True, reserve_possible_merge_context=False, maximum_ambiguous_match_fraction=1.0), ModalHypothesisStatus.REJECTED, ModalHypothesisReason.POSSIBLE_MERGE_CONTEXT),
    ],
    ids=["split_reservation", "split_reject", "merge_reservation", "merge_reject"],
)
def test_split_and_merge_context_policies_do_not_branch_or_fuse(chain_factory, settings, status, reason) -> None:
    chain = chain_factory()

    hypothesis = evaluate_modal_hypothesis(chain, tuple(node.dynamic_label for node in chain.nodes), settings)

    assert hypothesis.status is status
    assert reason in hypothesis.structural_context.reasons
    assert "no_branching_or_fusion_performed" in hypothesis.structural_context.diagnostics
    assert len(hypothesis.chain.nodes) == chain.condition_count


def test_structural_context_can_be_preserved_only_as_diagnostic() -> None:
    chain = _split_chain()
    settings = ModalHypothesisSettings(
        reject_possible_split_context=False,
        reserve_possible_split_context=False,
        maximum_ambiguous_match_fraction=1.0,
    )

    hypothesis = evaluate_modal_hypothesis(chain, ("pp", "p", "mf"), settings)

    assert hypothesis.structural_context.contains_possible_split_context
    assert hypothesis.structural_context.requires_reservation is False
    assert ModalHypothesisReason.POSSIBLE_SPLIT_CONTEXT not in hypothesis.reservation_reasons
    assert hypothesis.status is ModalHypothesisStatus.ACCEPTED_WITH_RESERVATIONS


def test_all_statuses_are_reachable_and_mutually_exclusive() -> None:
    accepted = evaluate_modal_hypothesis(_strong_chain("SA"), FULL_SEQUENCE)
    reserved = evaluate_modal_hypothesis(
        _split_chain(),
        ("pp", "p", "mf"),
        ModalHypothesisSettings(maximum_ambiguous_match_fraction=1.0),
    )
    inconclusive = evaluate_modal_hypothesis(
        _strong_chain("SI"),
        FULL_SEQUENCE,
        ModalHypothesisSettings(minimum_acceptance_score=0.99, minimum_reservation_score=0.98),
    )
    rejected = evaluate_modal_hypothesis(
        _chain(
            (100.0, 150.0),
            labels=("pp", "p"),
            prefix="SR",
            association_settings=CrossConditionCandidateAssociationSettings(
                maximum_absolute_frequency_difference_hz=100.0,
                maximum_association_cost=10.0,
            ),
        )[0],
        ("pp", "p"),
    )
    insufficient = evaluate_modal_hypothesis(
        _chain(
            (100.0, 100.1),
            labels=("pp", "p"),
            prefix="SE",
            per_ref={0: {"tau": None}, 1: {"tau": None}},
        )[0],
        ("pp", "p"),
        ModalHypothesisSettings(require_decay_evidence=True),
    )
    invalid = evaluate_modal_hypothesis(object())
    statuses = (
        accepted.status,
        reserved.status,
        inconclusive.status,
        rejected.status,
        insufficient.status,
        invalid.status,
    )

    assert statuses == tuple(ModalHypothesisStatus)
    for hypothesis in (accepted, reserved, inconclusive, rejected, insufficient, invalid):
        assert sum(hypothesis.status is status for status in ModalHypothesisStatus) == 1


def test_score_components_are_explicit_and_sum_is_auditable() -> None:
    hypothesis = evaluate_modal_hypothesis(_strong_chain("SC"), FULL_SEQUENCE)
    components = hypothesis.score.components
    numerator = sum(component.weighted_value or 0.0 for component in components if component.available and component.weight > 0.0)
    denominator = sum(component.weight for component in components if component.available and component.weight > 0.0)

    assert hypothesis.score.raw_score == pytest.approx(numerator / denominator)
    assert hypothesis.score.normalized_score == pytest.approx(hypothesis.score.raw_score)
    assert [component.name for component in components] == [
        "coverage",
        "frequency",
        "association",
        "tracking",
        "decay",
        "impact",
    ]


def test_score_thresholds_are_inclusive_and_do_not_replace_gates() -> None:
    chain = _strong_chain("TH")
    base_score = evaluate_modal_hypothesis(chain, FULL_SEQUENCE).score.normalized_score
    equal = evaluate_modal_hypothesis(
        chain,
        FULL_SEQUENCE,
        ModalHypothesisSettings(
            minimum_acceptance_score=base_score,
            minimum_reservation_score=base_score,
        ),
    )
    above = evaluate_modal_hypothesis(
        chain,
        FULL_SEQUENCE,
        ModalHypothesisSettings(
            minimum_acceptance_score=min(1.0, base_score + 0.01),
            minimum_reservation_score=min(1.0, base_score + 0.005),
        ),
    )

    assert equal.status is ModalHypothesisStatus.ACCEPTED
    assert above.status is ModalHypothesisStatus.INCONCLUSIVE

    missing_decay = _chain(
        (100.0, 100.1, 100.2, 100.3, 100.4),
        prefix="TG",
        per_ref={index: {"tau": None} for index in range(5)},
    )[0]
    gated = evaluate_modal_hypothesis(
        missing_decay,
        FULL_SEQUENCE,
        ModalHypothesisSettings(
            require_decay_evidence=True,
            minimum_acceptance_score=0.1,
            minimum_reservation_score=0.1,
        ),
    )
    assert gated.score.passes_acceptance_threshold
    assert gated.status is ModalHypothesisStatus.INSUFFICIENT_EVIDENCE


def test_score_weights_zero_and_optional_disabled_criteria_are_supported() -> None:
    chain = _strong_chain("W")
    zero_weight = ModalHypothesisSettings(
        tracking_quality_weight=0.0,
        decay_consistency_weight=0.0,
        impact_evidence_weight=0.0,
    )
    disabled = ModalHypothesisSettings(
        maximum_step_absolute_frequency_change_hz=None,
        maximum_total_absolute_frequency_change_hz=None,
        maximum_frequency_trajectory_rmse_hz=None,
        maximum_match_cost=None,
        maximum_mean_match_cost=None,
        maximum_ambiguous_match_fraction=None,
        maximum_near_threshold_match_fraction=None,
        minimum_mean_coverage_fraction=None,
        maximum_mean_ambiguous_assignment_fraction=None,
        maximum_mean_near_threshold_assignment_fraction=None,
        maximum_mean_frequency_fit_rmse_hz=None,
        maximum_log_tau_range=None,
        maximum_log_tau_standard_deviation=None,
        minimum_impact_supported_fraction=None,
    )

    zero_hypothesis = evaluate_modal_hypothesis(chain, FULL_SEQUENCE, zero_weight)
    disabled_hypothesis = evaluate_modal_hypothesis(chain, FULL_SEQUENCE, disabled)

    assert dict((component.name, component.weight) for component in zero_hypothesis.score.components)["tracking"] == 0.0
    assert disabled_hypothesis.status is ModalHypothesisStatus.ACCEPTED


def test_compute_score_can_be_reused_without_hidden_formula() -> None:
    hypothesis = evaluate_modal_hypothesis(_strong_chain("CS"), FULL_SEQUENCE)

    recomputed = compute_modal_hypothesis_score(
        hypothesis.coverage_evidence,
        hypothesis.frequency_evidence,
        hypothesis.association_evidence,
        hypothesis.tracking_evidence,
        hypothesis.decay_evidence,
        hypothesis.impact_evidence,
        hypothesis.structural_context,
        hypothesis.missing_evidence_reasons,
        ModalHypothesisSettings(),
    )

    assert recomputed.normalized_score == pytest.approx(hypothesis.score.normalized_score)
    assert recomputed.components == hypothesis.score.components


def test_build_modal_hypotheses_partitions_all_source_chains() -> None:
    strong = _strong_chain("BA", 0)
    partial = _chain((120.0, 120.1, 120.2), labels=("p", "mf", "f"), prefix="BB", candidate_start=10)[0]
    ambiguous = _split_chain()
    singleton = _singleton_chain("BD", 30)
    discontinuous = _chain(
        (400.0, 450.0),
        labels=("pp", "p"),
        prefix="BE",
        candidate_start=40,
        association_settings=CrossConditionCandidateAssociationSettings(
            maximum_absolute_frequency_difference_hz=100.0,
            maximum_association_cost=10.0,
        ),
    )[0]
    split = _split_chain("BF", 60)
    missing_tau = _chain(
        (500.0, 500.1, 500.2, 500.3, 500.4),
        prefix="BG",
        candidate_start=50,
        per_ref={index: {"tau": None} for index in range(5)},
    )[0]
    settings = ModalHypothesisSettings(
        require_complete_chain=False,
        allow_partial_chains=True,
        minimum_condition_coverage_fraction=0.4,
        require_decay_evidence=True,
        maximum_ambiguous_match_fraction=1.0,
    )

    result = build_modal_hypotheses(
        (strong, partial, ambiguous, singleton, discontinuous, split, missing_tau),
        settings,
        FULL_SEQUENCE,
    )

    assert result.hypothesis_count == 7
    assert result.source_chain_count == 7
    assert len({item.hypothesis_id for item in result.hypotheses}) == 7
    assert result.accepted_count == 1
    assert result.accepted_with_reservations_count >= 1
    assert result.rejected_count >= 1
    assert result.insufficient_evidence_count >= 2
    assert result.accepted_hypotheses == tuple(
        item for item in result.hypotheses if item.status is ModalHypothesisStatus.ACCEPTED
    )


def test_build_modal_hypotheses_is_deterministic_across_input_orders_and_diagnostics_order() -> None:
    chains = (
        _strong_chain("DA", 0),
        _chain((120.0, 120.1, 120.2), labels=("p", "mf", "f"), prefix="DB", candidate_start=10)[0],
        _split_chain(),
    )
    changed_diagnostics = replace(chains[0], diagnostics=tuple(reversed(chains[0].diagnostics)))
    settings = ModalHypothesisSettings(
        require_complete_chain=False,
        allow_partial_chains=True,
        minimum_condition_coverage_fraction=0.4,
        maximum_ambiguous_match_fraction=1.0,
    )

    ordered = build_modal_hypotheses(chains, settings, FULL_SEQUENCE)
    reversed_result = build_modal_hypotheses(tuple(reversed(chains)), settings, FULL_SEQUENCE)
    shuffled_items = list(chains)
    Random(7).shuffle(shuffled_items)
    shuffled = build_modal_hypotheses(tuple(shuffled_items), settings, FULL_SEQUENCE)
    equivalent = build_modal_hypotheses((changed_diagnostics,) + chains[1:], settings, FULL_SEQUENCE)

    def normalized(result):
        return tuple(
            (
                item.hypothesis_id,
                item.source_chain_id,
                item.status,
                round(item.score.normalized_score, 12),
                item.supporting_reasons,
                item.reservation_reasons,
                item.rejection_reasons,
                item.missing_evidence_reasons,
                item.diagnostics,
            )
            for item in result.hypotheses
        )

    assert normalized(ordered) == normalized(reversed_result) == normalized(shuffled) == normalized(equivalent)


def test_local_perturbation_changes_only_corresponding_hypothesis_and_keeps_id_stable() -> None:
    target = _strong_chain("PA", 0)
    other = _strong_chain("PB", 10)
    perturbed_target = _replace_node_ref(target, 2, representative_frequency_hz=100.8)
    settings = ModalHypothesisSettings()

    base = build_modal_hypotheses((target, other), settings, FULL_SEQUENCE)
    perturbed = build_modal_hypotheses((perturbed_target, other), settings, FULL_SEQUENCE)
    base_by_chain = {item.source_chain_id: item for item in base.hypotheses}
    perturbed_by_chain = {item.source_chain_id: item for item in perturbed.hypotheses}

    assert base_by_chain.keys() == perturbed_by_chain.keys()
    assert perturbed_by_chain[target.chain_id].hypothesis_id == base_by_chain[target.chain_id].hypothesis_id
    assert perturbed_by_chain[target.chain_id].frequency_evidence.frequencies_hz != base_by_chain[target.chain_id].frequency_evidence.frequencies_hz
    assert perturbed_by_chain[other.chain_id] == base_by_chain[other.chain_id]


def test_inputs_are_immutable_and_repeated_builds_are_identical() -> None:
    chains = (_strong_chain("IM", 0), _split_chain())
    snapshot = deepcopy(chains)
    settings = ModalHypothesisSettings(
        require_complete_chain=False,
        allow_partial_chains=True,
        minimum_condition_coverage_fraction=0.4,
        maximum_ambiguous_match_fraction=1.0,
    )

    first = build_modal_hypotheses(chains, settings, FULL_SEQUENCE)
    second = build_modal_hypotheses(chains, settings, FULL_SEQUENCE)

    assert chains == snapshot
    assert first == second
    assert summarize_modal_hypotheses(first)["hypothesis_ids"] == tuple(
        item.hypothesis_id for item in first.hypotheses
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"minimum_condition_count": 0},
        {"minimum_condition_coverage_fraction": 1.1},
        {"frequency_continuity_weight": -1.0},
        {"maximum_match_cost": float("inf")},
        {"minimum_decay_value_count": 0},
        {"missing_tracking_evidence_policy": "unknown"},
        {"require_complete_chain": True, "allow_partial_chains": True},
        {"reject_possible_split_context": True, "reserve_possible_split_context": True},
        {"minimum_acceptance_score": 0.4, "minimum_reservation_score": 0.5},
    ],
    ids=[
        "zero_min_conditions",
        "fraction_above_one",
        "negative_weight",
        "infinite_cost",
        "zero_decay_count",
        "unknown_missing_policy",
        "contradictory_partial",
        "contradictory_split",
        "threshold_order",
    ],
)
def test_settings_reject_invalid_invariants(changes) -> None:
    with pytest.raises(ValueError):
        ModalHypothesisSettings(**changes)


def test_numeric_invariants_have_no_nan_or_infinity_and_absence_is_none() -> None:
    hypothesis = evaluate_modal_hypothesis(_strong_chain("NI"), FULL_SEQUENCE)
    numeric_values = [
        hypothesis.coverage_evidence.condition_coverage_fraction,
        hypothesis.frequency_evidence.maximum_step_change_hz,
        hypothesis.frequency_evidence.trajectory_rmse_from_mean_hz,
        hypothesis.association_evidence.mean_match_cost,
        hypothesis.tracking_evidence.mean_coverage_fraction,
        hypothesis.decay_evidence.log_tau_range,
        hypothesis.impact_evidence.impact_supported_fraction,
        hypothesis.score.normalized_score,
    ]

    assert all(value is not None and isfinite(value) for value in numeric_values)
    assert all(0.0 <= component.value <= 1.0 for component in hypothesis.score.components if component.available)

    singleton = _singleton_chain("none", 99)
    singleton_hypothesis = evaluate_modal_hypothesis(singleton, ("pp", "p"))
    assert singleton_hypothesis.association_evidence.mean_match_cost is None
    assert singleton_hypothesis.association_evidence.maximum_match_cost is None
