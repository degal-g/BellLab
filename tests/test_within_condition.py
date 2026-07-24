"""Associação conservadora de candidatos entre repetições equivalentes."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from belllab import (
    CandidateReference,
    ExcitationCondition,
    RecordingCandidateSet,
    WithinConditionAssociationSettings,
    associate_candidates_within_condition,
    evaluate_modal_candidate,
    group_candidates_by_excitation_condition,
)
from tests.test_modal_candidates import _context, _settings as _candidate_settings
from tests.test_preimpact import _evidence


def _candidate(
    frequency: float,
    candidate_id: int,
    *,
    accepted: bool = True,
    tau: float | None = 1.0,
):
    amplitudes = (
        tuple(1.0 for _ in range(6))
        if tau is None
        else tuple(float(np.exp(-time / tau)) for time in range(6))
    )
    characterization, tracking = _context(
        tuple((100.0,) for _ in range(6)), amplitudes=amplitudes
    )
    settings = (
        _candidate_settings()
        if accepted
        else _candidate_settings(minimum_duration_s=100.0)
    )
    candidate = evaluate_modal_candidate(
        characterization, tracking, settings, candidate_id=candidate_id
    )
    characterization = replace(candidate.characterization, track_id=candidate_id)
    return replace(
        candidate,
        source_track_id=candidate_id,
        characterization=characterization,
        representative_frequency_hz=frequency,
    )


def _recording(
    recording_id: str,
    repeat: int,
    frequencies: tuple[float, ...],
    *,
    label: str = "pp",
    accepted: bool = True,
    taus: tuple[float | None, ...] | None = None,
    evidence=(),
) -> RecordingCandidateSet:
    taus = taus or tuple(1.0 for _ in frequencies)
    return RecordingCandidateSet(
        recording_id,
        ExcitationCondition(label, repeat),
        tuple(
            _candidate(value, index, accepted=accepted, tau=taus[index])
            for index, value in enumerate(frequencies)
        ),
        tuple(evidence),
    )


def _settings(**changes) -> WithinConditionAssociationSettings:
    return WithinConditionAssociationSettings(**changes)


def test_two_pp_repetitions_match_expected_frequencies_and_preserve_unmatched() -> None:
    result = associate_candidates_within_condition((
        _recording("A", 1, (100.0, 200.0, 300.0)),
        _recording("B", 2, (100.4, 199.5, 450.0)),
    ))
    assert [
        tuple(item.representative_frequency_hz for item in cluster.member_candidate_refs)
        for cluster in result.clusters
    ] == [(100.0, 100.4), (200.0, 199.5)]
    assert [(item.reference.recording_id, item.reference.representative_frequency_hz)
            for item in result.unmatched_candidates] == [("A", 300.0), ("B", 450.0)]
    selected = tuple(item for item in result.association_diagnostics if item.selected)
    assert [item.frequency_difference_hz for item in selected] == pytest.approx([0.4, 0.5])
    assert [item.total_cost for item in selected] == pytest.approx([0.2, 0.25])


@pytest.mark.parametrize(
    ("difference", "count"),
    [(1.9, 1), (2.0, 1), (2.1, 0)],
    ids=["absolute_below", "absolute_exact", "absolute_above"],
)
def test_absolute_frequency_gate_is_inclusive(difference, count) -> None:
    result = associate_candidates_within_condition((
        _recording("A", 0, (100.0,)),
        _recording("B", 1, (100.0 + difference,)),
    ))
    assert len(result.clusters) == count


@pytest.mark.parametrize(
    ("relative", "expected"),
    [(0.009, 1), (0.01, 1), (0.011, 0)],
    ids=["relative_below", "relative_exact", "relative_above"],
)
def test_relative_frequency_gate_is_inclusive(relative, expected) -> None:
    # Solve d / ((100 + 100+d)/2) = relative exactly.
    difference = 200.0 * relative / (2.0 - relative)
    cfg = _settings(
        maximum_absolute_frequency_difference_hz=None,
        maximum_relative_frequency_difference=0.01,
    )
    result = associate_candidates_within_condition((
        _recording("A", 0, (100.0,)),
        _recording("B", 1, (100.0 + difference,)),
    ), cfg)
    assert len(result.clusters) == expected


def test_total_cost_equal_to_maximum_is_accepted() -> None:
    result = associate_candidates_within_condition((
        _recording("A", 0, (100.0,)),
        _recording("B", 1, (102.0,)),
    ))
    diagnostic = next(item for item in result.association_diagnostics if item.selected)
    assert diagnostic.total_cost == pytest.approx(1.0)
    assert diagnostic.admissible


def test_frequency_components_are_explicit_and_sum_to_total() -> None:
    result = associate_candidates_within_condition((
        _recording("A", 0, (100.0,), taus=(1.0,)),
        _recording("B", 1, (101.0,), taus=(2.0,)),
    ), _settings(decay_tau_cost_weight=0.25, maximum_association_cost=2.0))
    diagnostic = next(item for item in result.association_diagnostics if item.selected)
    assert diagnostic.frequency_cost_component == pytest.approx(0.5)
    assert diagnostic.tau_cost_component == pytest.approx(0.25)
    assert diagnostic.total_cost == pytest.approx(0.75)


def test_ambiguity_tie_has_zero_margin_and_deterministic_tie_break() -> None:
    result = associate_candidates_within_condition((
        _recording("A", 0, (100.0,)),
        _recording("B", 1, (99.9, 100.1)),
    ))
    selected = next(item for item in result.association_diagnostics if item.selected)
    assert selected.right_candidate_id == 0
    assert selected.assignment_margin == pytest.approx(0.0, abs=1e-12)
    assert selected.ambiguous
    assert result.clusters[0].ambiguous
    assert any(item.reference.candidate_id == 1 for item in result.unmatched_candidates)


@pytest.mark.parametrize(
    ("alternative_cost", "ambiguous"),
    [(0.05, True), (0.10, True), (0.20, False)],
    ids=["margin_005", "margin_exact_010", "margin_020"],
)
def test_ambiguity_margin_is_inclusive(alternative_cost, ambiguous) -> None:
    # Primary difference .2 Hz -> cost .1; alternative cost is primary + margin.
    alternative_difference = 2.0 * (0.1 + alternative_cost)
    result = associate_candidates_within_condition((
        _recording("A", 0, (100.0,)),
        _recording("B", 1, (100.2, 100.0 + alternative_difference)),
    ))
    selected = next(item for item in result.association_diagnostics if item.selected)
    assert selected.assignment_margin == pytest.approx(alternative_cost)
    assert selected.ambiguous is ambiguous


def test_unique_alternative_has_no_margin_and_is_not_ambiguous() -> None:
    result = associate_candidates_within_condition((
        _recording("A", 0, (100.0,)),
        _recording("B", 1, (100.2,)),
    ))
    diagnostic = next(item for item in result.association_diagnostics if item.selected)
    assert diagnostic.row_assignment_margin is None
    assert diagnostic.column_assignment_margin is None
    assert diagnostic.assignment_margin is None
    assert not diagnostic.ambiguous


def test_three_repetitions_form_two_full_coverage_clusters() -> None:
    result = associate_candidates_within_condition((
        _recording("A", 0, (100.0, 200.0), label="mf"),
        _recording("B", 1, (100.3, 199.8), label="mf"),
        _recording("C", 2, (99.7, 201.0), label="mf"),
    ))
    assert len(result.clusters) == 2
    first, second = result.clusters
    assert first.frequency_median_hz == pytest.approx(100.0)
    assert first.frequency_mean_hz == pytest.approx(100.0)
    assert first.frequency_std_hz == pytest.approx(np.std([100.0, 100.3, 99.7]))
    assert first.frequency_span_hz == pytest.approx(0.6)
    assert second.frequency_median_hz == pytest.approx(200.0)
    assert all(item.repeat_coverage_fraction == 1.0 for item in result.clusters)
    assert all(item.reproducible for item in result.clusters)


def test_two_of_three_coverage_is_explicit_and_reproducible_at_exact_limit() -> None:
    result = associate_candidates_within_condition((
        _recording("A", 0, (100.0,), label="mf"),
        _recording("B", 1, (100.2,), label="mf"),
        _recording("C", 2, (), label="mf"),
    ), _settings(minimum_repeat_coverage_fraction=2 / 3))
    cluster = result.clusters[0]
    assert cluster.repeat_coverage_fraction == pytest.approx(2 / 3)
    assert cluster.reproducible


@pytest.mark.parametrize(
    ("minimum", "reproducible"),
    [(0.65, True), (2 / 3, True), (0.68, False)],
    ids=["coverage_below", "coverage_exact", "coverage_above"],
)
def test_repeat_coverage_threshold(minimum, reproducible) -> None:
    result = associate_candidates_within_condition((
        _recording("A", 0, (100.0,)),
        _recording("B", 1, (100.2,)),
        _recording("C", 2, ()),
    ), _settings(minimum_repeat_coverage_fraction=minimum))
    assert result.clusters[0].reproducible is reproducible


def test_single_member_contract_exposes_one_of_three_coverage() -> None:
    # A singleton is returned as unmatched by the algorithm; the quantitative
    # 1/3 coverage is nevertheless the same criterion used before that policy.
    result = associate_candidates_within_condition((
        _recording("A", 0, (100.0,)),
        _recording("B", 1, ()),
        _recording("C", 2, ()),
    ))
    assert not result.clusters
    assert len(result.unmatched_candidates) == 1


def test_progressive_grouping_rejects_incoherent_transitive_chain() -> None:
    result = associate_candidates_within_condition((
        _recording("A", 0, (100.0,)),
        _recording("B", 1, (102.0,)),
        _recording("C", 2, (104.0,)),
    ))
    assert len(result.clusters) == 1
    assert [item.representative_frequency_hz
            for item in result.clusters[0].member_candidate_refs] == [100.0, 102.0]
    assert result.unmatched_candidates[0].reference.representative_frequency_hz == 104.0
    assert any(
        "failed_all_member_consistency" in item.diagnostics
        for item in result.association_diagnostics
    )


@pytest.mark.parametrize(
    ("left_tau", "right_tau", "allow_missing", "expected"),
    [
        (1.0, 1.1, True, 1),
        (1.0, 8.0, True, 0),
    ],
    ids=["similar_tau", "different_tau"],
)
def test_tau_cost_is_optional_and_auditable(left_tau, right_tau, allow_missing, expected) -> None:
    result = associate_candidates_within_condition((
        _recording("A", 0, (100.0,), taus=(left_tau,)),
        _recording("B", 1, (100.0,), taus=(right_tau,)),
    ), _settings(
        decay_tau_cost_weight=0.5,
        maximum_association_cost=1.0,
        allow_missing_tau=allow_missing,
    ))
    assert len(result.clusters) == expected


@pytest.mark.parametrize(
    ("left_tau", "right_tau", "allow_missing", "expected"),
    [
        (1.0, None, True, 1),
        (None, None, True, 1),
        (1.0, None, False, 0),
        (None, None, False, 0),
    ],
    ids=[
        "one_missing_allowed", "both_missing_allowed",
        "one_missing_forbidden", "both_missing_forbidden",
    ],
)
def test_missing_tau_policy(left_tau, right_tau, allow_missing, expected) -> None:
    result = associate_candidates_within_condition((
        _recording("A", 0, (100.0,), taus=(left_tau,)),
        _recording("B", 1, (100.1,), taus=(right_tau,)),
    ), _settings(decay_tau_cost_weight=0.5, allow_missing_tau=allow_missing))
    assert len(result.clusters) == expected
    assert any(
        "tau_not_applicable" in item.diagnostics
        for item in result.association_diagnostics
    )


def _recording_with_evidence(
    recording_id: str,
    repeat: int,
    pre: tuple[float, ...],
    post: tuple[float, ...],
) -> RecordingCandidateSet:
    evidence, _, _ = _evidence(pre, post)
    return _recording(
        recording_id, repeat, (100.0,),
        evidence=(replace(evidence, source_track_id=0),)
    )


@pytest.mark.parametrize(
    ("left_pre", "left_post", "right_pre", "right_post"),
    [
        ((), (4.0, 3.0, 2.0), (), (3.0, 2.0, 1.0)),
        ((1.0, 1.0, 1.0), (4.0, 4.0, 4.0),
         (1.0, 1.0, 1.0), (5.0, 5.0, 5.0)),
        ((), (4.0, 3.0, 2.0),
         (1.0, 1.0, 1.0), (4.0, 4.0, 4.0)),
    ],
    ids=["two_emergent", "two_amplified", "emergent_and_amplified"],
)
def test_impact_excited_classifications_are_compatible_without_identity_requirement(
    left_pre, left_post, right_pre, right_post
) -> None:
    result = associate_candidates_within_condition((
        _recording_with_evidence("A", 0, left_pre, left_post),
        _recording_with_evidence("B", 1, right_pre, right_post),
    ), _settings(require_impact_excitation=True))
    assert len(result.clusters) == 1
    assert all(item.impact_excited for item in result.clusters[0].member_candidate_refs)


def test_persistent_background_is_optional_and_never_erased() -> None:
    recordings = (
        _recording_with_evidence("A", 0, (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
        _recording_with_evidence("B", 1, (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
    )
    allowed = associate_candidates_within_condition(recordings)
    excluded = associate_candidates_within_condition(
        recordings, _settings(reject_persistent_background_tone=True)
    )
    assert len(allowed.clusters) == 1
    assert not excluded.clusters
    assert all(item.reason == "missing_required_evidence"
               for item in excluded.unmatched_candidates)


def test_missing_preimpact_evidence_policy_is_optional() -> None:
    recordings = (
        _recording("A", 0, (100.0,)),
        _recording("B", 1, (100.1,)),
    )
    assert len(associate_candidates_within_condition(recordings).clusters) == 1
    result = associate_candidates_within_condition(
        recordings, _settings(allow_missing_preimpact_evidence=False)
    )
    assert not result.clusters
    assert all(item.reason == "missing_required_evidence"
               for item in result.unmatched_candidates)


def test_require_impact_excitation_rejects_persistent_but_preserves_references() -> None:
    recordings = (
        _recording_with_evidence("A", 0, (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
        _recording_with_evidence("B", 1, (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
    )
    result = associate_candidates_within_condition(
        recordings, _settings(require_impact_excitation=True)
    )
    assert not result.clusters
    assert len(result.unmatched_candidates) == 2
    assert len(result.candidate_references) == 2


def test_rejected_candidates_are_ignored_by_default_and_preserved() -> None:
    result = associate_candidates_within_condition((
        _recording("A", 0, (100.0,), accepted=False),
        _recording("B", 1, (100.1,), accepted=False),
    ))
    assert not result.clusters
    assert all(item.reason == "candidate_rejected_by_policy"
               for item in result.unmatched_candidates)


def test_rejected_candidates_can_be_grouped_for_audit_without_promotion() -> None:
    result = associate_candidates_within_condition((
        _recording("A", 0, (100.0,), accepted=False),
        _recording("B", 1, (100.1,), accepted=False),
    ), _settings(allow_rejected_candidates=True))
    assert len(result.clusters) == 1
    assert all(not item.accepted for item in result.clusters[0].member_candidate_refs)


@pytest.mark.parametrize(
    ("left", "right"),
    [("pp", "ff"), ("p", "mf"), ("unspecified", "pp")],
    ids=["pp_ff", "p_mf", "unspecified_known"],
)
def test_internal_association_rejects_mixed_dynamic_conditions(left, right) -> None:
    with pytest.raises(ValueError, match="cannot mix dynamic labels"):
        associate_candidates_within_condition((
            _recording("A", 0, (100.0,), label=left),
            _recording("B", 1, (100.1,), label=right),
        ))


def test_high_level_api_separates_dynamic_conditions() -> None:
    results = group_candidates_by_excitation_condition((
        _recording("F", 0, (200.0,), label="ff"),
        _recording("P", 0, (100.0,), label="pp"),
    ))
    assert tuple(item.dynamic_label for item in results) == ("ff", "pp")
    assert all(not item.clusters for item in results)


@pytest.mark.parametrize(
    "changes",
    [
        {"maximum_absolute_frequency_difference_hz": -1.0},
        {"maximum_relative_frequency_difference": -0.1},
        {"frequency_cost_weight": -1.0},
        {"frequency_stability_cost_weight": -1.0},
        {"minimum_repeat_coverage_fraction": -0.1},
        {"minimum_repeat_coverage_fraction": 1.1},
        {"minimum_repeat_count": -1},
        {"maximum_association_cost": 0.0},
        {"maximum_association_cost": float("nan")},
        {"maximum_association_cost": float("inf")},
        {"ambiguity_margin_threshold": float("-inf")},
    ],
    ids=[
        "negative_absolute", "negative_relative", "negative_frequency_weight",
        "negative_stability_weight", "negative_coverage", "coverage_above_one",
        "negative_repeat_count", "zero_cost", "nan_cost", "infinite_cost",
        "infinite_margin",
    ],
)
def test_invalid_association_settings_are_rejected(changes) -> None:
    with pytest.raises(ValueError):
        _settings(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"dynamic_label": "forte"},
        {"repeat_index": -1},
        {"measured_rms": -1.0},
        {"measured_energy": -1.0},
        {"measured_peak": float("nan")},
        {"session_id": "  "},
    ],
    ids=[
        "unknown_label", "negative_repeat", "negative_rms", "negative_energy",
        "nan_peak", "blank_session",
    ],
)
def test_invalid_excitation_condition_is_rejected(changes) -> None:
    values = {"dynamic_label": "pp", "repeat_index": 0}
    values.update(changes)
    with pytest.raises(ValueError):
        ExcitationCondition(**values)


def test_duplicate_recording_ids_and_repeat_indices_are_rejected() -> None:
    with pytest.raises(ValueError, match="recording IDs"):
        associate_candidates_within_condition((
            _recording("A", 0, (100.0,)),
            _recording("A", 1, (100.1,)),
        ))
    with pytest.raises(ValueError, match="repeat indices"):
        associate_candidates_within_condition((
            _recording("A", 0, (100.0,)),
            _recording("B", 0, (100.1,)),
        ))


def test_determinism_across_ordered_reversed_and_repeated_inputs() -> None:
    recordings = (
        _recording("A", 0, (100.0, 200.0), label="mf"),
        _recording("B", 1, (100.3, 199.8), label="mf"),
        _recording("C", 2, (99.7, 201.0), label="mf"),
    )
    first = associate_candidates_within_condition(recordings)
    second = associate_candidates_within_condition(tuple(reversed(recordings)))
    third = associate_candidates_within_condition(recordings)
    assert first == second == third
    assert tuple(item.cluster_id for item in first.clusters) == (0, 1)


def test_inputs_are_not_mutated() -> None:
    recordings = (
        _recording("A", 0, (100.0,)),
        _recording("B", 1, (100.2,)),
    )
    snapshot = repr(recordings)
    associate_candidates_within_condition(recordings)
    assert repr(recordings) == snapshot


def test_public_contracts_are_importable_from_package_root() -> None:
    reference = CandidateReference("A", 0, 0, 100.0, "pp", True)
    assert reference.representative_frequency_hz == 100.0
