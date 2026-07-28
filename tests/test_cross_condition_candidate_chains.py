"""Operational chains built from adjacent cross-condition associations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace
from math import isfinite

import pytest

from belllab import (
    AdjacentDynamicConditionPair,
    CandidateReference,
    CrossConditionCandidateAssociationResult,
    CrossConditionCandidateAssociationSettings,
    CrossConditionCandidateMatch,
    build_cross_condition_candidate_chains,
    build_cross_condition_candidate_matches,
    summarize_cross_condition_candidate_chains,
    validate_adjacent_association_sequence,
)


def _ref(
    label: str,
    name: str,
    frequency: float,
    candidate_id: int,
    *,
    accepted: bool = True,
) -> CandidateReference:
    return CandidateReference(
        f"{label}-{name}",
        candidate_id,
        candidate_id,
        frequency,
        label,
        accepted,
        None,
        None,
        0.01,
        1.0,
        0.95,
        0.0,
        0.1,
        1.0,
        0.0,
        0.0,
        1.0,
        None,
        None,
        ("test_candidate_chain_reference",),
    )


def _pair_result(
    lower_refs: tuple[CandidateReference, ...],
    higher_refs: tuple[CandidateReference, ...],
    settings: CrossConditionCandidateAssociationSettings | None = None,
    *,
    lower_label: str | None = None,
    higher_label: str | None = None,
) -> CrossConditionCandidateAssociationResult:
    lower = lower_label or lower_refs[0].dynamic_label
    higher = higher_label or higher_refs[0].dynamic_label
    return build_cross_condition_candidate_matches(
        lower_refs,
        higher_refs,
        AdjacentDynamicConditionPair(lower, higher),
        settings,
    )


def _basic_sequence() -> tuple[
    tuple[CrossConditionCandidateAssociationResult, ...],
    dict[str, CandidateReference],
]:
    refs = {
        "A": _ref("pp", "A", 100.0, 0),
        "B": _ref("pp", "B", 200.0, 1),
        "C": _ref("pp", "C", 300.0, 2),
        "D": _ref("p", "D", 101.0, 3),
        "E": _ref("p", "E", 199.0, 4),
        "F": _ref("p", "F", 400.0, 5),
        "G": _ref("mf", "G", 102.0, 6),
        "H": _ref("mf", "H", 198.0, 7),
        "I": _ref("mf", "I", 401.0, 8),
        "J": _ref("f", "J", 103.0, 9),
        "K": _ref("f", "K", 402.0, 10),
        "L": _ref("ff", "L", 104.0, 11),
        "M": _ref("ff", "M", 403.0, 12),
        "N": _ref("ff", "N", 700.0, 13),
    }
    settings = CrossConditionCandidateAssociationSettings(maximum_association_cost=1.0)
    pairs = (
        _pair_result((refs["A"], refs["B"], refs["C"]), (refs["D"], refs["E"], refs["F"]), settings),
        _pair_result((refs["D"], refs["E"], refs["F"]), (refs["G"], refs["H"], refs["I"]), settings),
        _pair_result((refs["G"], refs["H"], refs["I"]), (refs["J"], refs["K"]), settings),
        _pair_result((refs["J"], refs["K"]), (refs["L"], refs["M"], refs["N"]), settings),
    )
    return pairs, refs


def _names(chain) -> tuple[str, ...]:
    return tuple(node.candidate_ref.recording_id.split("-", 1)[1] for node in chain.nodes)


def _chain_by_names(result, names: tuple[str, ...]):
    for chain in result.chains:
        if _names(chain) == names:
            return chain
    raise AssertionError(f"missing chain {names}")


def _normalized(result) -> tuple:
    return tuple(
        (
            chain.chain_id,
            _names(chain),
            tuple(round(value, 12) for value in chain.frequency_trajectory_hz),
            tuple(round(value, 12) for value in chain.frequency_step_changes_hz),
            tuple(round(value, 12) for value in chain.association_costs),
            chain.complete_across_requested_sequence,
            chain.starts_as_emerging,
            chain.ends_as_disappearing,
            chain.contains_ambiguous_match,
            chain.contains_near_threshold_match,
            chain.contains_possible_split_context,
            chain.contains_possible_merge_context,
        )
        for chain in result.chains
    )


def _corrupt_dataclass(instance, **changes):
    corrupted = object.__new__(type(instance))
    for field in fields(instance):
        object.__setattr__(
            corrupted,
            field.name,
            changes.get(field.name, getattr(instance, field.name)),
        )
    return corrupted


def test_basic_full_sequence_partitions_complete_partial_and_singleton_chains() -> None:
    pairs, _ = _basic_sequence()
    result = build_cross_condition_candidate_chains(pairs)

    assert result.sequence.dynamic_labels == ("pp", "p", "mf", "f", "ff")
    assert result.chain_count == 5
    assert result.complete_chain_count == 1
    assert result.partial_chain_count == 2
    assert result.singleton_chain_count == 2
    assert result.candidate_count == 14
    assert result.matched_candidate_count == 12
    assert result.unmatched_candidate_count == 2
    assert {_names(chain) for chain in result.chains} == {
        ("A", "D", "G", "J", "L"),
        ("B", "E", "H"),
        ("F", "I", "K", "M"),
        ("C",),
        ("N",),
    }
    assert _chain_by_names(result, ("A", "D", "G", "J", "L")).complete_across_requested_sequence
    used_match_ids = {
        node.outgoing_match_id
        for chain in result.chains
        for node in chain.nodes
        if node.outgoing_match_id is not None
    }
    expected_match_ids = {
        match.match_id
        for pair in pairs
        for match in pair.matches
        if match.accepted
    }
    assert used_match_ids == expected_match_ids
    assert len(used_match_ids) == sum(chain.match_count for chain in result.chains)


def test_chain_frequency_trajectory_and_costs_are_auditable() -> None:
    pairs, _ = _basic_sequence()
    result = build_cross_condition_candidate_chains(pairs)
    chain = _chain_by_names(result, ("A", "D", "G", "J", "L"))

    assert chain.frequency_trajectory_hz == (100.0, 101.0, 102.0, 103.0, 104.0)
    assert chain.frequency_step_changes_hz == pytest.approx((1.0, 1.0, 1.0, 1.0))
    assert chain.frequency_step_changes_relative == pytest.approx((
        1.0 / 100.5,
        1.0 / 101.5,
        1.0 / 102.5,
        1.0 / 103.5,
    ))
    assert chain.frequency_change_classifications == (
        "frequency_shifted_up",
        "frequency_shifted_up",
        "frequency_shifted_up",
        "frequency_shifted_up",
    )
    assert chain.total_frequency_change_hz == pytest.approx(4.0)
    assert chain.total_frequency_change_relative == pytest.approx(4.0 / 102.0)
    assert chain.upward_step_count == 4
    assert chain.association_costs == pytest.approx((0.5, 0.5, 0.5, 0.5))
    assert chain.maximum_association_cost == pytest.approx(0.5)
    assert chain.minimum_association_cost == pytest.approx(0.5)
    assert chain.mean_association_cost == pytest.approx(0.5)
    assert chain.minimum_assignment_margin is None


def test_emerging_chain_flags_depend_on_requested_sequence() -> None:
    pairs, _ = _basic_sequence()
    full = build_cross_condition_candidate_chains(pairs)
    full_chain = _chain_by_names(full, ("F", "I", "K", "M"))
    assert full_chain.starts_as_emerging
    assert not full_chain.complete_across_requested_sequence

    subsequence = build_cross_condition_candidate_chains(
        pairs[1:],
        ("p", "mf", "f", "ff"),
    )
    sub_chain = _chain_by_names(subsequence, ("F", "I", "K", "M"))
    assert sub_chain.starts_at_sequence_boundary
    assert not sub_chain.starts_as_emerging
    assert sub_chain.complete_across_requested_sequence


def test_disappearing_chain_flags_depend_on_requested_sequence() -> None:
    pairs, _ = _basic_sequence()
    full = build_cross_condition_candidate_chains(pairs)
    full_chain = _chain_by_names(full, ("B", "E", "H"))
    assert full_chain.ends_as_disappearing
    assert not full_chain.complete_across_requested_sequence

    subsequence = build_cross_condition_candidate_chains(
        pairs[:2],
        ("pp", "p", "mf"),
    )
    sub_chain = _chain_by_names(subsequence, ("B", "E", "H"))
    assert sub_chain.ends_at_sequence_boundary
    assert not sub_chain.ends_as_disappearing
    assert sub_chain.complete_across_requested_sequence


def test_singleton_chains_preserve_isolated_emerging_and_disappearing_context() -> None:
    pairs, _ = _basic_sequence()
    result = build_cross_condition_candidate_chains(pairs)
    disappearing = _chain_by_names(result, ("C",))
    emerging = _chain_by_names(result, ("N",))

    assert disappearing.isolated_candidate
    assert disappearing.ends_as_disappearing
    assert disappearing.association_costs == ()
    assert disappearing.maximum_association_cost is None
    assert disappearing.mean_association_cost is None
    assert emerging.isolated_candidate
    assert emerging.starts_as_emerging
    assert emerging.minimum_assignment_margin is None


def test_gap_is_not_closed_by_frequency_or_non_adjacent_association() -> None:
    pp_a = _ref("pp", "A", 100.0, 0)
    p_b = _ref("p", "B", 100.4, 1)
    mf_x = _ref("mf", "X", 500.0, 2)
    f_c = _ref("f", "C", 100.8, 3)
    ff_d = _ref("ff", "D", 101.2, 4)
    settings = CrossConditionCandidateAssociationSettings(maximum_association_cost=1.0)
    pairs = (
        _pair_result((pp_a,), (p_b,), settings),
        _pair_result((p_b,), (mf_x,), settings),
        _pair_result((mf_x,), (f_c,), settings),
        _pair_result((f_c,), (ff_d,), settings),
    )
    result = build_cross_condition_candidate_chains(pairs)

    assert _chain_by_names(result, ("A", "B"))
    assert _chain_by_names(result, ("C", "D"))
    assert _chain_by_names(result, ("X",))
    assert ("A", "B", "C", "D") not in {_names(chain) for chain in result.chains}
    assert "no_frequency_gap_closure_was_performed" in result.diagnostics


def test_ambiguous_adjacent_match_is_preserved_on_the_linear_chain() -> None:
    pp_a = _ref("pp", "A", 100.0, 0)
    p_b = _ref("p", "B", 99.9, 1)
    p_c = _ref("p", "C", 100.1, 2)
    pair = _pair_result((pp_a,), (p_b, p_c))
    result = build_cross_condition_candidate_chains((pair,), ("pp", "p"))
    matched = _chain_by_names(result, ("A", "B"))
    alternative = _chain_by_names(result, ("C",))

    assert matched.contains_ambiguous_match
    assert matched.ambiguous_match_ids == (pair.matches[0].match_id,)
    assert matched.ambiguous_match_positions == (0,)
    assert matched.ambiguous_assignment_margins == pytest.approx((0.0,))
    assert alternative.starts_as_emerging
    assert result.ambiguous_chain_count == 1


def test_near_threshold_match_keeps_local_cost_and_position() -> None:
    refs = {
        "A": _ref("pp", "A", 100.0, 0),
        "B": _ref("p", "B", 100.2, 1),
        "C": _ref("mf", "C", 102.0, 2),
        "D": _ref("f", "D", 102.2, 3),
    }
    settings = CrossConditionCandidateAssociationSettings(
        maximum_association_cost=1.0,
        near_threshold_ratio=0.8,
    )
    pairs = (
        _pair_result((refs["A"],), (refs["B"],), settings),
        _pair_result((refs["B"],), (refs["C"],), settings),
        _pair_result((refs["C"],), (refs["D"],), settings),
    )
    result = build_cross_condition_candidate_chains(pairs)
    chain = result.chains[0]

    assert chain.contains_near_threshold_match
    assert chain.near_threshold_match_ids == (pairs[1].matches[0].match_id,)
    assert chain.near_threshold_match_positions == (1,)
    assert chain.association_costs == pytest.approx((0.1, 0.9, 0.1))
    assert chain.maximum_association_cost == pytest.approx(0.9)
    assert result.near_threshold_chain_count == 1


def test_possible_split_context_is_attached_without_branching() -> None:
    pp_a = _ref("pp", "A", 200.0, 0)
    p_b = _ref("p", "B", 200.0, 1)
    mf_c = _ref("mf", "C", 198.5, 2)
    mf_d = _ref("mf", "D", 201.5, 3)
    pairs = (
        _pair_result((pp_a,), (p_b,)),
        _pair_result((p_b,), (mf_c, mf_d)),
    )
    result = build_cross_condition_candidate_chains(pairs)
    selected = _chain_by_names(result, ("A", "B", "C"))
    alternative = _chain_by_names(result, ("D",))

    assert selected.contains_possible_split_context
    assert selected.possible_split_contexts == pairs[1].possible_splits
    assert alternative.contains_possible_split_context
    assert _names(selected) == ("A", "B", "C")
    assert "possible_split_context_only" in selected.diagnostics


def test_possible_merge_context_is_attached_without_fusing_chains() -> None:
    mf_a = _ref("mf", "A", 298.5, 0)
    mf_b = _ref("mf", "B", 301.5, 1)
    f_c = _ref("f", "C", 300.0, 2)
    pair = _pair_result((mf_a, mf_b), (f_c,))
    result = build_cross_condition_candidate_chains((pair,), ("mf", "f"))
    selected = _chain_by_names(result, ("A", "C"))
    leftover = _chain_by_names(result, ("B",))

    assert selected.contains_possible_merge_context
    assert selected.possible_merge_contexts == pair.possible_merges
    assert leftover.contains_possible_merge_context
    assert leftover.isolated_candidate
    assert result.merge_context_chain_count == 2


def test_subsequences_use_only_requested_candidate_partition() -> None:
    pairs, _ = _basic_sequence()

    pp_p = build_cross_condition_candidate_chains(pairs[:1], ("pp", "p"))
    p_mf_f = build_cross_condition_candidate_chains(pairs[1:3], ("p", "mf", "f"))
    mf_f_ff = build_cross_condition_candidate_chains(pairs[2:], ("mf", "f", "ff"))
    full = build_cross_condition_candidate_chains(tuple(reversed(pairs)))

    assert pp_p.candidate_count == 6
    assert {_names(chain) for chain in pp_p.complete_chains} == {("A", "D"), ("B", "E")}
    assert p_mf_f.candidate_count == 8
    assert {_names(chain) for chain in p_mf_f.complete_chains} == {("D", "G", "J"), ("F", "I", "K")}
    assert mf_f_ff.candidate_count == 8
    assert {_names(chain) for chain in mf_f_ff.complete_chains} == {("G", "J", "L"), ("I", "K", "M")}
    assert full.sequence.dynamic_labels == ("pp", "p", "mf", "f", "ff")
    assert not any(
        node.dynamic_label not in ("p", "mf", "f")
        for chain in p_mf_f.chains
        for node in chain.nodes
    )


def test_summary_exposes_deterministic_counts_and_ids() -> None:
    pairs, _ = _basic_sequence()
    result = build_cross_condition_candidate_chains(pairs)
    summary = summarize_cross_condition_candidate_chains(result)

    assert summary["dynamic_labels"] == ("pp", "p", "mf", "f", "ff")
    assert summary["candidate_count"] == 14
    assert summary["chain_count"] == 5
    assert summary["chain_ids"] == tuple(chain.chain_id for chain in result.chains)
    assert "complete_candidate_partition" in summary["diagnostics"]


@pytest.mark.parametrize(
    "labels",
    [
        (),
        ("pp",),
        ("p", "pp"),
        ("pp", "mf"),
        ("p", "p"),
        ("pp", "p", "f"),
        ("unknown", "p"),
    ],
    ids=[
        "empty",
        "single_condition",
        "inverted",
        "skipped",
        "repeated",
        "gap",
        "unknown",
    ],
)
def test_invalid_requested_dynamic_label_sequences_are_rejected(labels) -> None:
    pairs, _ = _basic_sequence()
    with pytest.raises(ValueError):
        validate_adjacent_association_sequence(pairs, labels)


def test_missing_duplicate_disconnected_and_mismatched_pairs_are_rejected() -> None:
    pairs, refs = _basic_sequence()
    with pytest.raises(ValueError):
        build_cross_condition_candidate_chains((pairs[0], pairs[2]))
    with pytest.raises(ValueError):
        build_cross_condition_candidate_chains((pairs[0], pairs[0]))
    with pytest.raises(ValueError):
        build_cross_condition_candidate_chains(pairs[:1], ("p", "mf"))

    invalid_pair = _pair_result((), (refs["D"],), lower_label="pp", higher_label="p")
    assert not invalid_pair.valid
    with pytest.raises(ValueError):
        build_cross_condition_candidate_chains((invalid_pair,))


def test_corrupted_duplicate_outgoing_match_is_rejected_explicitly() -> None:
    pairs, _ = _basic_sequence()
    duplicate = replace(pairs[1].matches[0], match_id="p-mf-duplicate")
    corrupted = _corrupt_dataclass(
        pairs[1],
        matches=pairs[1].matches + (duplicate,),
    )

    with pytest.raises(ValueError, match="more than one outgoing"):
        build_cross_condition_candidate_chains((pairs[0], corrupted, pairs[2], pairs[3]))


def test_corrupted_match_pointing_outside_candidate_universe_is_rejected() -> None:
    pairs, _ = _basic_sequence()
    outside = _ref("p", "outside", 101.5, 99)
    corrupted_match = _corrupt_dataclass(
        pairs[0].matches[0],
        higher_candidate_ref=outside,
    )
    corrupted_pair = _corrupt_dataclass(
        pairs[0],
        matches=(corrupted_match,) + pairs[0].matches[1:],
    )

    with pytest.raises(ValueError, match="outside the sequence"):
        build_cross_condition_candidate_chains((corrupted_pair, pairs[1], pairs[2], pairs[3]))


def test_corrupted_duplicate_candidate_reference_is_rejected_by_pair_contract() -> None:
    pairs, _ = _basic_sequence()
    with pytest.raises(ValueError):
        replace(
            pairs[0],
            lower_candidate_references=pairs[0].lower_candidate_references + (pairs[0].lower_candidate_references[0],),
        )


def test_determinism_is_independent_of_pair_and_local_list_order() -> None:
    pairs, _ = _basic_sequence()
    shuffled_pairs = (
        replace(
            pairs[3],
            matches=tuple(reversed(pairs[3].matches)),
            emerging_candidates=tuple(reversed(pairs[3].emerging_candidates)),
        ),
        replace(
            pairs[1],
            matches=tuple(reversed(pairs[1].matches)),
            disappearing_candidates=tuple(reversed(pairs[1].disappearing_candidates)),
            emerging_candidates=tuple(reversed(pairs[1].emerging_candidates)),
        ),
        replace(pairs[0], matches=tuple(reversed(pairs[0].matches))),
        replace(pairs[2], matches=tuple(reversed(pairs[2].matches))),
    )

    ordered = build_cross_condition_candidate_chains(pairs)
    shuffled = build_cross_condition_candidate_chains(shuffled_pairs)
    repeated = build_cross_condition_candidate_chains(tuple(reversed(pairs)))
    assert _normalized(ordered) == _normalized(shuffled) == _normalized(repeated)


def test_local_frequency_perturbation_changes_only_the_related_chain_metrics() -> None:
    pairs, refs = _basic_sequence()
    base = build_cross_condition_candidate_chains(pairs)
    changed_d = replace(refs["D"], representative_frequency_hz=101.2)
    perturbed_refs = dict(refs)
    perturbed_refs["D"] = changed_d
    settings = CrossConditionCandidateAssociationSettings(maximum_association_cost=1.0)
    perturbed_pairs = (
        _pair_result((refs["A"], refs["B"], refs["C"]), (changed_d, refs["E"], refs["F"]), settings),
        _pair_result((changed_d, refs["E"], refs["F"]), (refs["G"], refs["H"], refs["I"]), settings),
        pairs[2],
        pairs[3],
    )
    perturbed = build_cross_condition_candidate_chains(perturbed_pairs)

    base_by_names = {_names(chain): chain for chain in base.chains}
    changed_by_names = {_names(chain): chain for chain in perturbed.chains}
    for names, chain in base_by_names.items():
        changed = changed_by_names[names]
        assert changed.chain_id == chain.chain_id
        if names == ("A", "D", "G", "J", "L"):
            assert changed.frequency_trajectory_hz != chain.frequency_trajectory_hz
            assert changed.association_costs != chain.association_costs
        else:
            assert changed.frequency_trajectory_hz == chain.frequency_trajectory_hz
            assert changed.association_costs == chain.association_costs


def test_inputs_are_not_modified_and_repeated_builds_are_identical() -> None:
    pairs, _ = _basic_sequence()
    snapshot = deepcopy(pairs)

    first = build_cross_condition_candidate_chains(pairs)
    second = build_cross_condition_candidate_chains(pairs)

    assert pairs == snapshot
    assert _normalized(first) == _normalized(second)
    assert pairs[0].matches == snapshot[0].matches
    assert pairs[1].emerging_candidates == snapshot[1].emerging_candidates


def test_numeric_invariants_use_finite_values_and_none_for_absent_costs() -> None:
    pairs, _ = _basic_sequence()
    result = build_cross_condition_candidate_chains(pairs)

    for chain in result.chains:
        assert all(isfinite(value) and value > 0.0 for value in chain.frequency_trajectory_hz)
        assert all(isfinite(value) for value in chain.frequency_step_changes_hz)
        assert all(isfinite(value) for value in chain.frequency_step_changes_relative)
        assert all(isfinite(value) and value >= 0.0 for value in chain.association_costs)
        expected_total_relative = chain.total_frequency_change_hz / (
            0.5 * (chain.initial_frequency_hz + chain.final_frequency_hz)
        )
        assert chain.total_frequency_change_relative == pytest.approx(expected_total_relative)
        if chain.match_count == 0:
            assert chain.maximum_association_cost is None
            assert chain.minimum_association_cost is None
            assert chain.mean_association_cost is None
            assert chain.maximum_normalized_association_cost is None
        else:
            assert chain.maximum_association_cost == max(chain.association_costs)
            assert chain.minimum_association_cost == min(chain.association_costs)
            assert chain.mean_association_cost == pytest.approx(sum(chain.association_costs) / chain.match_count)


def test_rejected_audit_only_matches_are_not_used_as_chain_edges() -> None:
    pp_a = _ref("pp", "A", 100.0, 0, accepted=False)
    p_b = _ref("p", "B", 100.2, 1, accepted=False)
    pair = _pair_result(
        (pp_a,),
        (p_b,),
        CrossConditionCandidateAssociationSettings(allow_rejected_candidates=True),
    )
    assert pair.matches and not pair.matches[0].accepted

    result = build_cross_condition_candidate_chains((pair,), ("pp", "p"))
    assert {_names(chain) for chain in result.chains} == {("A",), ("B",)}
    assert result.singleton_chain_count == 2
    assert "audit_only_rejected_matches_not_used:1" in result.diagnostics
