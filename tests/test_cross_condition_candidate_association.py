"""Associação operacional entre candidatos de condições dinâmicas adjacentes."""

from __future__ import annotations

from dataclasses import replace
from math import log2

import pytest

from belllab import (
    AdjacentDynamicConditionPair,
    CandidateReference,
    CrossConditionCandidateAssociationSettings,
    associate_candidates_across_adjacent_conditions,
    build_cross_condition_candidate_matches,
)
from tests.test_within_condition import _recording


def _ref(
    label: str,
    frequency: float,
    candidate_id: int = 0,
    *,
    recording_id: str | None = None,
    accepted: bool = True,
    impact_excited: bool | None = None,
    classification: str | None = None,
    tau: float | None = 1.0,
    stability: float | None = 0.01,
    drift: float | None = 0.0,
    rmse: float | None = 0.1,
    coverage: float | None = 1.0,
    ambiguous_fraction: float | None = 0.0,
    near_fraction: float | None = 0.0,
    margin: float | None = 1.0,
) -> CandidateReference:
    return CandidateReference(
        recording_id or f"{label}-{candidate_id}",
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
        ("test_candidate_reference",),
    )


def _pair(lower: str = "pp", higher: str = "p") -> AdjacentDynamicConditionPair:
    return AdjacentDynamicConditionPair(lower, higher)


def _associate(
    lower_frequencies: tuple[float, ...],
    higher_frequencies: tuple[float, ...],
    settings: CrossConditionCandidateAssociationSettings | None = None,
    *,
    lower_label: str = "pp",
    higher_label: str = "p",
):
    lower = tuple(
        _ref(lower_label, frequency, index)
        for index, frequency in enumerate(lower_frequencies)
    )
    higher = tuple(
        _ref(higher_label, frequency, index)
        for index, frequency in enumerate(higher_frequencies)
    )
    return build_cross_condition_candidate_matches(
        lower,
        higher,
        _pair(lower_label, higher_label),
        settings,
    )


def _selected(result):
    return tuple(item for item in result.association_diagnostics if item.selected)


def _normalized(result) -> tuple:
    return (
        tuple(
            (
                match.match_id,
                match.lower_candidate_ref.recording_id,
                match.lower_candidate_ref.candidate_id,
                match.higher_candidate_ref.recording_id,
                match.higher_candidate_ref.candidate_id,
                round(match.association_diagnostic.total_cost, 12),
                match.frequency_change_classification,
                match.ambiguous,
                match.near_threshold,
            )
            for match in result.matches
        ),
        tuple(
            (item.reference.recording_id, item.reference.candidate_id, item.reason)
            for item in result.disappearing_candidates
        ),
        tuple(
            (item.reference.recording_id, item.reference.candidate_id, item.reason)
            for item in result.emerging_candidates
        ),
        tuple(
            (
                split.source_candidate_ref.candidate_id,
                tuple(ref.candidate_id for ref in split.target_candidate_refs),
                tuple(round(cost, 12) for cost in split.costs),
            )
            for split in result.possible_splits
        ),
        tuple(
            (
                tuple(ref.candidate_id for ref in merge.source_candidate_refs),
                merge.target_candidate_ref.candidate_id,
                tuple(round(cost, 12) for cost in merge.costs),
            )
            for merge in result.possible_merges
        ),
    )


def test_basic_pp_to_p_matches_and_preserves_unmatched_candidates() -> None:
    result = associate_candidates_across_adjacent_conditions(
        _recording("pp-r0", 0, (100.0, 200.0, 300.0), label="pp"),
        _recording("p-r0", 0, (100.4, 199.5, 450.0), label="p"),
    )

    assert [
        (
            match.lower_candidate_ref.representative_frequency_hz,
            match.higher_candidate_ref.representative_frequency_hz,
        )
        for match in result.matches
    ] == [(100.0, 100.4), (200.0, 199.5)]
    assert [item.reference.representative_frequency_hz for item in result.disappearing_candidates] == [300.0]
    assert [item.reference.representative_frequency_hz for item in result.emerging_candidates] == [450.0]
    assert [item.total_cost for item in _selected(result)] == pytest.approx([0.2, 0.25])
    assert result.valid
    assert "no_modal_mode_conversion_was_performed" in result.diagnostics


def test_frequency_distances_are_absolute_symmetric_relative_and_logarithmic() -> None:
    result = _associate((100.0,), (101.0,))
    diagnostic = result.association_diagnostics[0]
    assert diagnostic.frequency_difference_hz == pytest.approx(1.0)
    assert diagnostic.relative_frequency_difference == pytest.approx(1.0 / 100.5)
    assert diagnostic.log_frequency_difference == pytest.approx(abs(log2(101.0 / 100.0)))
    assert diagnostic.frequency_cost_component == pytest.approx(0.5)
    assert diagnostic.total_cost == pytest.approx(sum(value for _, value in diagnostic.cost_components))


@pytest.mark.parametrize(
    ("higher", "classification"),
    [
        (100.4, "frequency_preserved"),
        (100.6, "frequency_shifted_up"),
        (99.4, "frequency_shifted_down"),
        (100.5, "frequency_preserved"),
        (100.500001, "frequency_shifted_up"),
        (100.499999, "frequency_preserved"),
    ],
    ids=[
        "preserved",
        "shifted_up",
        "shifted_down",
        "exact_limit",
        "above_limit",
        "below_limit",
    ],
)
def test_frequency_change_classification_is_operational_and_inclusive(
    higher: float,
    classification: str,
) -> None:
    result = _associate((100.0,), (higher,))
    assert result.matches[0].frequency_change_classification == classification


def test_ambiguity_tie_keeps_one_to_one_match_and_preserves_alternative() -> None:
    result = _associate((100.0,), (99.9, 100.1))
    match = result.matches[0]
    assert match.higher_candidate_ref.candidate_id == 0
    assert match.association_diagnostic.assignment_margin == pytest.approx(0.0, abs=1e-12)
    assert match.ambiguous
    assert result.emerging_candidates[0].reference.candidate_id == 1
    assert result.emerging_candidates[0].reason == "ambiguous_match"
    assert len(result.possible_splits) == 1


@pytest.mark.parametrize(
    ("alternative_margin", "ambiguous"),
    [(0.05, True), (0.10, True), (0.20, False)],
    ids=["margin_005", "margin_exact_010", "margin_020"],
)
def test_ambiguity_margin_threshold_is_inclusive(
    alternative_margin: float,
    ambiguous: bool,
) -> None:
    alternative_difference = 2.0 * (0.1 + alternative_margin)
    result = _associate((100.0,), (100.2, 100.0 + alternative_difference))
    match = result.matches[0]
    assert match.association_diagnostic.assignment_margin == pytest.approx(alternative_margin)
    assert match.ambiguous is ambiguous


def test_possible_split_is_diagnostic_only_and_matching_remains_one_to_one() -> None:
    result = _associate((200.0,), (198.5, 201.5), lower_label="p", higher_label="mf")
    assert len(result.matches) == 1
    assert len(result.emerging_candidates) == 1
    assert len(result.possible_splits) == 1
    split = result.possible_splits[0]
    assert split.possible_split
    assert split.source_candidate_ref.representative_frequency_hz == 200.0
    assert [ref.representative_frequency_hz for ref in split.target_candidate_refs] == [198.5, 201.5]
    assert split.costs == pytest.approx((0.75, 0.75))
    assert "operational_indication_not_physical_split" in split.diagnostics


def test_possible_merge_is_diagnostic_only_and_matching_remains_one_to_one() -> None:
    result = _associate((298.5, 301.5), (300.0,), lower_label="mf", higher_label="f")
    assert len(result.matches) == 1
    assert len(result.disappearing_candidates) == 1
    assert len(result.possible_merges) == 1
    merge = result.possible_merges[0]
    assert merge.possible_merge
    assert [ref.representative_frequency_hz for ref in merge.source_candidate_refs] == [298.5, 301.5]
    assert merge.target_candidate_ref.representative_frequency_hz == 300.0
    assert merge.costs == pytest.approx((0.75, 0.75))
    assert "operational_indication_not_physical_merge" in merge.diagnostics


@pytest.mark.parametrize(
    ("low_tau", "high_tau", "allow_missing", "limit", "matched", "reason"),
    [
        (1.0, 1.1, True, 0.25, True, None),
        (1.0, 4.0, True, 0.25, False, "tau_gate"),
        (1.0, None, True, 0.25, True, None),
        (None, None, True, 0.25, True, None),
        (1.0, None, False, 0.25, False, "missing_tau"),
        (None, None, False, None, False, "missing_tau"),
    ],
    ids=[
        "similar_tau",
        "different_tau",
        "one_missing_allowed",
        "both_missing_allowed",
        "one_missing_forbidden",
        "both_missing_forbidden",
    ],
)
def test_tau_policy_never_substitutes_missing_values_with_zero(
    low_tau: float | None,
    high_tau: float | None,
    allow_missing: bool,
    limit: float | None,
    matched: bool,
    reason: str | None,
) -> None:
    settings = CrossConditionCandidateAssociationSettings(
        maximum_log_tau_difference=limit,
        allow_missing_tau=allow_missing,
        tau_cost_weight=0.1,
        maximum_association_cost=2.0,
    )
    result = build_cross_condition_candidate_matches(
        (_ref("pp", 100.0, tau=low_tau),),
        (_ref("p", 100.2, tau=high_tau),),
        _pair(),
        settings,
    )
    assert bool(result.matches) is matched
    diagnostic = result.association_diagnostics[0]
    assert diagnostic.tau_log_difference is None if low_tau is None or high_tau is None else diagnostic.tau_log_difference is not None
    if reason is not None:
        assert diagnostic.rejection_reason == reason


@pytest.mark.parametrize(
    ("lower_class", "higher_class", "compatible"),
    [
        ("impact_emergent", "impact_emergent", True),
        ("impact_amplified", "impact_amplified", True),
        ("impact_emergent", "impact_amplified", True),
        ("reexcited_preexisting_component", "impact_amplified", True),
        ("persistent_background_tone", "persistent_background_tone", True),
    ],
    ids=[
        "emergent_emergent",
        "amplified_amplified",
        "emergent_amplified",
        "reexcited_amplified",
        "persistent_background",
    ],
)
def test_preimpact_classifications_are_operational_evidence_not_causality(
    lower_class: str,
    higher_class: str,
    compatible: bool,
) -> None:
    lower_excited = lower_class != "persistent_background_tone"
    higher_excited = higher_class != "persistent_background_tone"
    result = build_cross_condition_candidate_matches(
        (_ref("pp", 100.0, impact_excited=lower_excited, classification=lower_class),),
        (_ref("p", 100.2, impact_excited=higher_excited, classification=higher_class),),
        _pair(),
        CrossConditionCandidateAssociationSettings(impact_evidence_cost_weight=0.5),
    )
    diagnostic = result.matches[0].association_diagnostic
    assert diagnostic.impact_evidence_compatible is compatible
    assert diagnostic.impact_evidence_cost_component == pytest.approx(0.0)


def test_missing_preimpact_evidence_can_be_allowed_or_forbidden() -> None:
    allowed = build_cross_condition_candidate_matches(
        (_ref("pp", 100.0, impact_excited=None),),
        (_ref("p", 100.2, impact_excited=True, classification="impact_emergent"),),
        _pair(),
        CrossConditionCandidateAssociationSettings(allow_missing_preimpact_evidence=True),
    )
    assert allowed.matches
    assert allowed.matches[0].impact_evidence_compatible is None

    forbidden = build_cross_condition_candidate_matches(
        (_ref("pp", 100.0, impact_excited=None),),
        (_ref("p", 100.2, impact_excited=True, classification="impact_emergent"),),
        _pair(),
        CrossConditionCandidateAssociationSettings(allow_missing_preimpact_evidence=False),
    )
    assert not forbidden.matches
    assert forbidden.disappearing_candidates[0].reason == "missing_required_evidence"


def test_required_impact_excitation_filters_candidates_without_promoting_anything() -> None:
    result = build_cross_condition_candidate_matches(
        (_ref("pp", 100.0, impact_excited=False, classification="persistent_background_tone"),),
        (_ref("p", 100.2, impact_excited=True, classification="impact_amplified"),),
        _pair(),
        CrossConditionCandidateAssociationSettings(require_impact_excitation=True),
    )
    assert not result.matches
    assert result.disappearing_candidates[0].reason == "missing_required_evidence"
    assert not result.association_diagnostics


def test_rejected_candidates_are_ignored_by_default_and_preserved() -> None:
    result = build_cross_condition_candidate_matches(
        (_ref("pp", 100.0, accepted=False),),
        (_ref("p", 100.2, accepted=False),),
        _pair(),
    )
    assert not result.matches
    assert result.disappearing_candidates[0].reason == "candidate_rejected_by_policy"
    assert result.emerging_candidates[0].reason == "candidate_rejected_by_policy"
    assert not result.disappearing_candidates[0].reference.accepted


def test_rejected_candidates_can_be_included_for_audit_without_promotion() -> None:
    result = build_cross_condition_candidate_matches(
        (_ref("pp", 100.0, accepted=False),),
        (_ref("p", 100.2, accepted=False),),
        _pair(),
        CrossConditionCandidateAssociationSettings(allow_rejected_candidates=True),
    )
    assert len(result.matches) == 1
    assert not result.matches[0].accepted
    assert not result.matches[0].lower_candidate_ref.accepted
    assert "contains_rejected_candidate_for_audit_only" in result.matches[0].diagnostics


@pytest.mark.parametrize(
    ("difference", "expected"),
    [(1.9, True), (2.0, True), (2.1, False)],
    ids=["absolute_below", "absolute_equal", "absolute_above"],
)
def test_absolute_frequency_limit_is_inclusive(difference: float, expected: bool) -> None:
    result = _associate((100.0,), (100.0 + difference,))
    assert bool(result.matches) is expected


@pytest.mark.parametrize(
    ("relative", "expected"),
    [(0.009, True), (0.010, True), (0.011, False)],
    ids=["relative_below", "relative_equal", "relative_above"],
)
def test_symmetric_relative_frequency_limit_is_inclusive(relative: float, expected: bool) -> None:
    difference = 200.0 * relative / (2.0 - relative)
    settings = CrossConditionCandidateAssociationSettings(
        maximum_absolute_frequency_difference_hz=None,
        maximum_relative_frequency_difference=0.01,
    )
    result = _associate((100.0,), (100.0 + difference,), settings)
    assert bool(result.matches) is expected


@pytest.mark.parametrize(
    ("cost", "expected"),
    [(0.4, True), (0.5, True), (0.6, False)],
    ids=["cost_below", "cost_equal", "cost_above"],
)
def test_total_cost_limit_is_inclusive(cost: float, expected: bool) -> None:
    settings = CrossConditionCandidateAssociationSettings(maximum_association_cost=0.5)
    result = _associate((100.0,), (100.0 + 2.0 * cost,), settings)
    assert bool(result.matches) is expected
    if not expected:
        assert result.association_diagnostics[0].rejection_reason == "cost_above_threshold"


@pytest.mark.parametrize(
    ("cost", "near"),
    [(0.79, False), (0.80, True), (0.90, True)],
    ids=["below_near_threshold", "equal_near_threshold", "above_near_threshold"],
)
def test_near_threshold_uses_inclusive_total_cost_ratio(cost: float, near: bool) -> None:
    settings = CrossConditionCandidateAssociationSettings(
        maximum_association_cost=1.0,
        near_threshold_ratio=0.8,
    )
    result = _associate((100.0,), (100.0 + 2.0 * cost,), settings)
    assert result.matches[0].near_threshold is near


def test_optional_tracking_components_and_gates_are_auditable() -> None:
    lower = _ref("pp", 100.0, ambiguous_fraction=0.4, near_fraction=0.1, margin=0.5)
    higher = _ref("p", 100.1, ambiguous_fraction=0.2, near_fraction=0.3, margin=0.2)
    settings = CrossConditionCandidateAssociationSettings(
        ambiguity_cost_weight=0.5,
        near_threshold_cost_weight=0.25,
        assignment_margin_cost_weight=0.1,
        maximum_association_cost=2.0,
    )
    result = build_cross_condition_candidate_matches((lower,), (higher,), _pair(), settings)
    diagnostic = result.matches[0].association_diagnostic
    assert diagnostic.ambiguity_cost_component == pytest.approx(0.2)
    assert diagnostic.near_threshold_cost_component == pytest.approx(0.075)
    assert diagnostic.assignment_margin_cost_component == pytest.approx(0.03)

    rejected = build_cross_condition_candidate_matches(
        (lower,),
        (higher,),
        _pair(),
        replace(settings, maximum_ambiguous_fraction=0.3),
    )
    assert not rejected.matches
    assert rejected.association_diagnostics[0].rejection_reason == "maximum_ambiguous_fraction"


def test_frequency_stability_drift_and_rmse_components_are_explicit() -> None:
    lower = _ref("pp", 100.0, stability=0.01, drift=0.0, rmse=0.1)
    higher = _ref("p", 100.2, stability=0.03, drift=0.4, rmse=0.3)
    settings = CrossConditionCandidateAssociationSettings(
        frequency_stability_cost_weight=1.0,
        frequency_drift_cost_weight=0.5,
        frequency_fit_rmse_cost_weight=0.25,
        maximum_association_cost=2.0,
    )
    result = build_cross_condition_candidate_matches((lower,), (higher,), _pair(), settings)
    diagnostic = result.matches[0].association_diagnostic
    assert diagnostic.frequency_stability_difference == pytest.approx(0.02)
    assert diagnostic.frequency_drift_difference_hz == pytest.approx(0.4)
    assert diagnostic.frequency_fit_rmse_difference_hz == pytest.approx(0.2)
    assert diagnostic.frequency_stability_cost_component == pytest.approx(0.02)
    assert diagnostic.frequency_drift_cost_component == pytest.approx(0.2)
    assert diagnostic.frequency_fit_rmse_cost_component == pytest.approx(0.05)


@pytest.mark.parametrize(
    ("lower", "higher"),
    [
        ("pp", "mf"),
        ("pp", "ff"),
        ("f", "p"),
        ("mf", "mf"),
        ("unknown", "p"),
    ],
    ids=["skipped_adjacent", "pp_ff", "inverted", "same_label", "unknown"],
)
def test_invalid_condition_pairs_are_rejected(lower: str, higher: str) -> None:
    with pytest.raises(ValueError):
        AdjacentDynamicConditionPair(lower, higher)


def test_function_rejects_non_adjacent_candidate_inputs() -> None:
    with pytest.raises(ValueError):
        associate_candidates_across_adjacent_conditions(
            _ref("pp", 100.0),
            _ref("mf", 100.2),
        )


def test_determinism_is_independent_of_input_order() -> None:
    lower = (
        _ref("pp", 100.0, 0),
        _ref("pp", 200.0, 1),
        _ref("pp", 300.0, 2),
    )
    higher = (
        _ref("p", 100.2, 0),
        _ref("p", 199.8, 1),
        _ref("p", 450.0, 2),
    )
    settings = CrossConditionCandidateAssociationSettings()
    ordered = build_cross_condition_candidate_matches(lower, higher, _pair(), settings)
    reversed_result = build_cross_condition_candidate_matches(
        tuple(reversed(lower)),
        tuple(reversed(higher)),
        _pair(),
        settings,
    )
    shuffled = build_cross_condition_candidate_matches(
        (lower[2], lower[0], lower[1]),
        (higher[1], higher[2], higher[0]),
        _pair(),
        settings,
    )
    assert _normalized(ordered) == _normalized(reversed_result) == _normalized(shuffled)


def test_local_frequency_perturbation_changes_only_related_costs() -> None:
    lower = (_ref("pp", 100.0, 0), _ref("pp", 200.0, 1))
    higher = (_ref("p", 100.2, 0), _ref("p", 200.2, 1))
    base = build_cross_condition_candidate_matches(lower, higher, _pair())
    perturbed = build_cross_condition_candidate_matches(
        lower,
        (replace(higher[0], representative_frequency_hz=100.3), higher[1]),
        _pair(),
    )
    assert [match.match_id for match in perturbed.matches] == [match.match_id for match in base.matches]
    assert perturbed.matches[0].association_diagnostic.total_cost != pytest.approx(
        base.matches[0].association_diagnostic.total_cost
    )
    assert perturbed.matches[1].association_diagnostic.total_cost == pytest.approx(
        base.matches[1].association_diagnostic.total_cost
    )
    assert perturbed.matches[1].higher_candidate_ref.candidate_id == base.matches[1].higher_candidate_ref.candidate_id


def test_no_reliable_correspondence_is_valid_without_forcing_match() -> None:
    result = _associate((100.0,), (120.0,))
    assert result.valid
    assert not result.matches
    assert result.disappearing_candidates[0].reason == "no_candidate_in_frequency_range"
    assert result.emerging_candidates[0].reason == "no_candidate_in_frequency_range"
    assert "no_reliable_correspondence" in result.diagnostics


def test_insufficient_candidate_data_is_explicit() -> None:
    result = build_cross_condition_candidate_matches(
        (),
        (_ref("p", 100.0),),
        _pair(),
    )
    assert not result.valid
    assert result.failure_reason == "insufficient_candidate_data"
    assert result.emerging_candidates[0].reason == "insufficient_data"


@pytest.mark.parametrize(
    "changes",
    [
        {"frequency_cost_weight": 0.0},
        {"maximum_association_cost": 0.0},
        {"near_threshold_ratio": 1.1},
        {"ambiguity_margin_threshold": -0.1},
        {
            "maximum_absolute_frequency_difference_hz": None,
            "maximum_relative_frequency_difference": None,
            "maximum_log_frequency_difference": None,
        },
    ],
    ids=[
        "zero_frequency_weight",
        "zero_max_cost",
        "near_ratio_above_one",
        "negative_margin",
        "no_frequency_gate",
    ],
)
def test_settings_reject_invalid_values(changes) -> None:
    with pytest.raises(ValueError):
        CrossConditionCandidateAssociationSettings(**changes)


def test_allow_unmatched_candidates_false_marks_result_invalid_without_forcing_match() -> None:
    result = _associate(
        (100.0,),
        (120.0,),
        CrossConditionCandidateAssociationSettings(allow_unmatched_candidates=False),
    )
    assert not result.valid
    assert result.failure_reason == "unmatched_candidates_not_allowed_by_configuration"
    assert not result.matches
