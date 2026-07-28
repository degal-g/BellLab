"""Operational chains built from adjacent cross-condition candidate matches.

This module links already accepted adjacent candidate correspondences along the
nominal sequence ``pp -> p -> mf -> f -> ff``. A chain is only an operational
sequence of local correspondences; it is not a physical modal identity, a modal
family, or evidence of linear or nonlinear behavior.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha1
from math import isclose, isfinite

from belllab.cross_condition import (
    CrossConditionCandidateAssociationResult,
    CrossConditionCandidateMatch,
    DisappearingCandidate,
    EmergingCandidate,
    PossibleCandidateMerge,
    PossibleCandidateSplit,
)
from belllab.dynamic_comparison import DYNAMIC_LABEL_ORDER
from belllab.within_condition import CandidateReference


_DYNAMIC_LABEL_INDEX = {
    label: index for index, label in enumerate(DYNAMIC_LABEL_ORDER)
}
_FREQUENCY_CHANGE_CLASSIFICATIONS = frozenset({
    "frequency_preserved",
    "frequency_shifted_up",
    "frequency_shifted_down",
    "frequency_shift_indeterminate",
})


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")


def _strings(values: tuple[str, ...], name: str) -> None:
    if not isinstance(values, tuple) or any(
        not isinstance(item, str) or not item.strip() for item in values
    ) or len(values) != len(set(values)):
        raise ValueError(f"{name} must contain unique nonempty strings.")


def _finite_optional(
    value: float | None,
    name: str,
    *,
    nonnegative: bool = False,
    positive: bool = False,
) -> None:
    if value is None:
        return
    if not isfinite(value):
        raise ValueError(f"{name} must be finite when present.")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive when present.")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be non-negative when present.")


@dataclass(frozen=True, slots=True)
class AdjacentAssociationSequence:
    """Validated contiguous sequence of adjacent association results."""

    dynamic_labels: tuple[str, ...]
    pair_results: tuple[CrossConditionCandidateAssociationResult, ...]
    start_dynamic_label: str
    end_dynamic_label: str
    condition_count: int
    pair_count: int
    complete: bool
    valid: bool
    failure_reason: str | None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_dynamic_label_sequence(self.dynamic_labels)
        if not isinstance(self.pair_results, tuple):
            raise ValueError("pair_results must be an immutable tuple.")
        if any(not isinstance(item, CrossConditionCandidateAssociationResult) for item in self.pair_results):
            raise ValueError("pair_results must contain cross-condition association results.")
        if self.start_dynamic_label != self.dynamic_labels[0]:
            raise ValueError("start_dynamic_label must match dynamic_labels.")
        if self.end_dynamic_label != self.dynamic_labels[-1]:
            raise ValueError("end_dynamic_label must match dynamic_labels.")
        if self.condition_count != len(self.dynamic_labels):
            raise ValueError("condition_count must match dynamic_labels.")
        if self.pair_count != len(self.pair_results):
            raise ValueError("pair_count must match pair_results.")
        if self.pair_count != self.condition_count - 1:
            raise ValueError("pair_count must equal condition_count - 1.")
        expected_pairs = tuple(zip(self.dynamic_labels, self.dynamic_labels[1:], strict=False))
        actual_pairs = tuple(
            (result.lower_dynamic_label, result.higher_dynamic_label)
            for result in self.pair_results
        )
        if actual_pairs != expected_pairs:
            raise ValueError("pair_results must follow the requested adjacent sequence.")
        if not all(result.valid for result in self.pair_results):
            raise ValueError("all pair results must be valid.")
        _validate_shared_condition_references(self.pair_results)
        if self.complete is not True:
            raise ValueError("a validated adjacent association sequence must be complete.")
        if self.valid is not True or self.failure_reason is not None:
            raise ValueError("a validated adjacent association sequence must be valid.")
        _strings(self.diagnostics, "adjacent association sequence diagnostics")


@dataclass(frozen=True, slots=True)
class CandidateChainNode:
    """Single candidate reference inside an operational chain."""

    dynamic_label: str
    candidate_ref: CandidateReference
    position_index: int
    incoming_match_id: str | None
    outgoing_match_id: str | None
    incoming_frequency_change: float | None
    outgoing_frequency_change: float | None
    incoming_association_cost: float | None
    outgoing_association_cost: float | None
    ambiguous_incoming: bool
    ambiguous_outgoing: bool
    near_threshold_incoming: bool
    near_threshold_outgoing: bool
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.dynamic_label not in _DYNAMIC_LABEL_INDEX:
            raise ValueError("node dynamic label is not recognized.")
        if self.dynamic_label != self.candidate_ref.dynamic_label:
            raise ValueError("node dynamic label must match candidate reference.")
        if self.position_index != _DYNAMIC_LABEL_INDEX[self.dynamic_label]:
            raise ValueError("node position_index must follow the nominal dynamic order.")
        _finite_optional(
            self.candidate_ref.representative_frequency_hz,
            "node representative_frequency_hz",
            positive=True,
        )
        _validate_match_side(
            self.incoming_match_id,
            self.incoming_frequency_change,
            self.incoming_association_cost,
            self.ambiguous_incoming,
            self.near_threshold_incoming,
            "incoming",
        )
        _validate_match_side(
            self.outgoing_match_id,
            self.outgoing_frequency_change,
            self.outgoing_association_cost,
            self.ambiguous_outgoing,
            self.near_threshold_outgoing,
            "outgoing",
        )
        _strings(self.diagnostics, "candidate chain node diagnostics")


@dataclass(frozen=True, slots=True)
class CrossConditionCandidateChain:
    """Operational chain of candidates linked by adjacent matches only."""

    chain_id: str
    nodes: tuple[CandidateChainNode, ...]
    start_dynamic_label: str
    end_dynamic_label: str
    condition_span: int
    condition_count: int
    match_count: int
    complete_across_requested_sequence: bool
    starts_at_sequence_boundary: bool
    starts_as_emerging: bool
    ends_at_sequence_boundary: bool
    ends_as_disappearing: bool
    isolated_candidate: bool
    partial_chain: bool
    contains_ambiguous_match: bool
    contains_near_threshold_match: bool
    contains_possible_split_context: bool
    contains_possible_merge_context: bool
    frequency_trajectory_hz: tuple[float, ...]
    frequency_step_changes_hz: tuple[float, ...]
    frequency_step_changes_relative: tuple[float, ...]
    frequency_change_classifications: tuple[str, ...]
    initial_frequency_hz: float
    final_frequency_hz: float
    total_frequency_change_hz: float
    total_frequency_change_relative: float
    upward_step_count: int
    downward_step_count: int
    preserved_step_count: int
    indeterminate_step_count: int
    association_costs: tuple[float, ...]
    maximum_association_cost: float | None
    minimum_association_cost: float | None
    mean_association_cost: float | None
    maximum_normalized_association_cost: float | None
    ambiguous_match_count: int
    near_threshold_match_count: int
    minimum_assignment_margin: float | None
    ambiguous_match_ids: tuple[str, ...]
    ambiguous_match_positions: tuple[int, ...]
    ambiguous_assignment_margins: tuple[float, ...]
    near_threshold_match_ids: tuple[str, ...]
    near_threshold_match_positions: tuple[int, ...]
    possible_split_contexts: tuple[PossibleCandidateSplit, ...]
    possible_merge_contexts: tuple[PossibleCandidateMerge, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.chain_id, "chain_id")
        if not isinstance(self.nodes, tuple) or not self.nodes:
            raise ValueError("nodes must be a nonempty immutable tuple.")
        if any(not isinstance(node, CandidateChainNode) for node in self.nodes):
            raise ValueError("nodes must contain CandidateChainNode instances.")
        if self.start_dynamic_label != self.nodes[0].dynamic_label:
            raise ValueError("start_dynamic_label must match the first node.")
        if self.end_dynamic_label != self.nodes[-1].dynamic_label:
            raise ValueError("end_dynamic_label must match the last node.")
        positions = tuple(node.position_index for node in self.nodes)
        if positions != tuple(sorted(positions)):
            raise ValueError("chain node positions must be ordered.")
        if any(right - left != 1 for left, right in zip(positions, positions[1:], strict=False)):
            raise ValueError("chain nodes must be connected by adjacent conditions.")
        ref_keys = tuple(_candidate_identity_key(node.candidate_ref) for node in self.nodes)
        if len(ref_keys) != len(set(ref_keys)):
            raise ValueError("candidate references must not repeat in a chain.")
        if self.condition_count != len(self.nodes):
            raise ValueError("condition_count must equal the number of nodes.")
        if self.match_count != len(self.nodes) - 1:
            raise ValueError("match_count must equal the number of adjacent edges.")
        expected_span = positions[-1] - positions[0] + 1
        if self.condition_span != expected_span:
            raise ValueError("condition_span is incoherent with node positions.")
        for node_index, node in enumerate(self.nodes):
            if node_index == 0 and node.incoming_match_id is not None:
                raise ValueError("first node cannot have an incoming match.")
            if node_index == len(self.nodes) - 1 and node.outgoing_match_id is not None:
                raise ValueError("last node cannot have an outgoing match.")
            if 0 < node_index and node.incoming_match_id is None:
                raise ValueError("intermediate and final matched nodes require incoming matches.")
            if node_index < len(self.nodes) - 1 and node.outgoing_match_id is None:
                raise ValueError("initial and intermediate matched nodes require outgoing matches.")
        if self.isolated_candidate != (self.match_count == 0):
            raise ValueError("isolated_candidate must mirror match_count == 0.")
        if self.partial_chain and self.complete_across_requested_sequence:
            raise ValueError("a complete chain cannot also be marked partial.")
        if self.partial_chain and self.isolated_candidate:
            raise ValueError("singleton chains are tracked separately from partial chains.")
        if self.starts_at_sequence_boundary and self.starts_as_emerging:
            raise ValueError("a sequence-boundary start cannot be marked emerging.")
        if self.ends_at_sequence_boundary and self.ends_as_disappearing:
            raise ValueError("a sequence-boundary end cannot be marked disappearing.")
        _validate_frequency_trajectory(self)
        _validate_cost_summary(self)
        if self.contains_ambiguous_match != bool(self.ambiguous_match_ids):
            raise ValueError("contains_ambiguous_match must mirror ambiguous_match_ids.")
        if self.contains_near_threshold_match != bool(self.near_threshold_match_ids):
            raise ValueError("contains_near_threshold_match must mirror near_threshold_match_ids.")
        if self.contains_possible_split_context != bool(self.possible_split_contexts):
            raise ValueError("contains_possible_split_context must mirror contexts.")
        if self.contains_possible_merge_context != bool(self.possible_merge_contexts):
            raise ValueError("contains_possible_merge_context must mirror contexts.")
        if self.ambiguous_match_count != len(self.ambiguous_match_ids):
            raise ValueError("ambiguous_match_count must match ambiguous_match_ids.")
        if self.near_threshold_match_count != len(self.near_threshold_match_ids):
            raise ValueError("near_threshold_match_count must match near_threshold_match_ids.")
        if len(self.ambiguous_match_positions) != len(self.ambiguous_match_ids):
            raise ValueError("ambiguous match positions and IDs must have equal length.")
        if len(self.ambiguous_assignment_margins) != len(self.ambiguous_match_ids):
            raise ValueError("ambiguous assignment margins and IDs must have equal length.")
        if len(self.near_threshold_match_positions) != len(self.near_threshold_match_ids):
            raise ValueError("near-threshold match positions and IDs must have equal length.")
        if any(position < 0 or position >= self.match_count for position in self.ambiguous_match_positions):
            raise ValueError("ambiguous match positions must refer to chain edges.")
        if any(position < 0 or position >= self.match_count for position in self.near_threshold_match_positions):
            raise ValueError("near-threshold match positions must refer to chain edges.")
        if any(not isfinite(value) or value < 0.0 for value in self.ambiguous_assignment_margins):
            raise ValueError("ambiguous assignment margins must be finite and non-negative.")
        _strings(self.diagnostics, "cross-condition candidate chain diagnostics")


@dataclass(frozen=True, slots=True)
class CrossConditionCandidateChainResult:
    """Complete partition of sequence candidates into operational chains."""

    sequence: AdjacentAssociationSequence
    chains: tuple[CrossConditionCandidateChain, ...]
    complete_chains: tuple[CrossConditionCandidateChain, ...]
    partial_chains: tuple[CrossConditionCandidateChain, ...]
    singleton_chains: tuple[CrossConditionCandidateChain, ...]
    candidate_count: int
    matched_candidate_count: int
    unmatched_candidate_count: int
    chain_count: int
    complete_chain_count: int
    partial_chain_count: int
    singleton_chain_count: int
    ambiguous_chain_count: int
    near_threshold_chain_count: int
    split_context_chain_count: int
    merge_context_chain_count: int
    all_candidate_references: tuple[CandidateReference, ...]
    valid: bool
    failure_reason: str | None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.sequence, AdjacentAssociationSequence):
            raise ValueError("sequence must be an AdjacentAssociationSequence.")
        if not isinstance(self.chains, tuple):
            raise ValueError("chains must be an immutable tuple.")
        if any(not isinstance(chain, CrossConditionCandidateChain) for chain in self.chains):
            raise ValueError("chains must contain CrossConditionCandidateChain instances.")
        if self.chains != tuple(sorted(self.chains, key=_chain_sort_key)):
            raise ValueError("chains must be in deterministic order.")
        for name in ("complete_chains", "partial_chains", "singleton_chains"):
            if not isinstance(getattr(self, name), tuple):
                raise ValueError(f"{name} must be an immutable tuple.")
        expected_complete = tuple(
            chain for chain in self.chains
            if chain.complete_across_requested_sequence
        )
        expected_singletons = tuple(chain for chain in self.chains if chain.isolated_candidate)
        expected_partial = tuple(
            chain for chain in self.chains
            if not chain.complete_across_requested_sequence and not chain.isolated_candidate
        )
        if self.complete_chains != expected_complete:
            raise ValueError("complete_chains must mirror chains.")
        if self.partial_chains != expected_partial:
            raise ValueError("partial_chains must mirror chains.")
        if self.singleton_chains != expected_singletons:
            raise ValueError("singleton_chains must mirror chains.")
        all_refs = tuple(ref for chain in self.chains for ref in _chain_refs(chain))
        if self.all_candidate_references != tuple(sorted(self.all_candidate_references, key=_candidate_sort_key)):
            raise ValueError("all_candidate_references must be in deterministic order.")
        if tuple(sorted(all_refs, key=_candidate_sort_key)) != self.all_candidate_references:
            raise ValueError("chain nodes must partition all_candidate_references.")
        keys = tuple(_candidate_identity_key(ref) for ref in all_refs)
        if len(keys) != len(set(keys)):
            raise ValueError("candidate references appear in more than one chain.")
        expected_match_ids = tuple(
            node.outgoing_match_id
            for chain in self.chains
            for node in chain.nodes
            if node.outgoing_match_id is not None
        )
        if len(expected_match_ids) != len(set(expected_match_ids)):
            raise ValueError("accepted adjacent matches must appear in only one chain.")
        matched_refs = tuple(
            ref for chain in self.chains for ref in _chain_refs(chain)
            if chain.match_count > 0
        )
        if self.candidate_count != len(self.all_candidate_references):
            raise ValueError("candidate_count must match all_candidate_references.")
        if self.matched_candidate_count != len(matched_refs):
            raise ValueError("matched_candidate_count is incoherent.")
        if self.unmatched_candidate_count != self.candidate_count - self.matched_candidate_count:
            raise ValueError("unmatched_candidate_count is incoherent.")
        counts = {
            "chain_count": len(self.chains),
            "complete_chain_count": len(self.complete_chains),
            "partial_chain_count": len(self.partial_chains),
            "singleton_chain_count": len(self.singleton_chains),
            "ambiguous_chain_count": sum(chain.contains_ambiguous_match for chain in self.chains),
            "near_threshold_chain_count": sum(chain.contains_near_threshold_match for chain in self.chains),
            "split_context_chain_count": sum(chain.contains_possible_split_context for chain in self.chains),
            "merge_context_chain_count": sum(chain.contains_possible_merge_context for chain in self.chains),
        }
        for name, expected in counts.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} is incoherent with chains.")
        if self.valid is not True or self.failure_reason is not None:
            raise ValueError("a built candidate-chain result must be valid.")
        _strings(self.diagnostics, "candidate chain result diagnostics")


def validate_adjacent_association_sequence(
    pair_results: AdjacentAssociationSequence
    | Iterable[CrossConditionCandidateAssociationResult],
    dynamic_labels: Iterable[str] | None = None,
) -> AdjacentAssociationSequence:
    """Validate and normalize a contiguous sequence of adjacent pair results."""

    if isinstance(pair_results, AdjacentAssociationSequence):
        if dynamic_labels is not None:
            raise ValueError("dynamic_labels must not be supplied with an existing sequence.")
        return AdjacentAssociationSequence(
            pair_results.dynamic_labels,
            pair_results.pair_results,
            pair_results.start_dynamic_label,
            pair_results.end_dynamic_label,
            pair_results.condition_count,
            pair_results.pair_count,
            pair_results.complete,
            pair_results.valid,
            pair_results.failure_reason,
            pair_results.diagnostics,
        )

    results = tuple(pair_results)
    if not results:
        raise ValueError("association sequence requires at least one adjacent pair result.")
    if any(not isinstance(result, CrossConditionCandidateAssociationResult) for result in results):
        raise ValueError("pair_results must contain cross-condition association results.")
    if not all(result.valid for result in results):
        raise ValueError("all pair results must be valid before chain construction.")
    pair_keys = tuple((result.lower_dynamic_label, result.higher_dynamic_label) for result in results)
    if len(pair_keys) != len(set(pair_keys)):
        raise ValueError("pair_results must not contain duplicate adjacent pairs.")
    pair_lookup = {key: result for key, result in zip(pair_keys, results, strict=True)}

    if dynamic_labels is None:
        ordered_pairs = tuple(sorted(pair_keys, key=lambda item: _DYNAMIC_LABEL_INDEX[item[0]]))
        for previous, current in zip(ordered_pairs, ordered_pairs[1:], strict=False):
            if previous[1] != current[0]:
                raise ValueError("pair_results must form a connected adjacent sequence.")
        labels = (ordered_pairs[0][0],) + tuple(pair[1] for pair in ordered_pairs)
    else:
        labels = tuple(dynamic_labels)
        _validate_dynamic_label_sequence(labels)
        expected_pairs = tuple(zip(labels, labels[1:], strict=False))
        missing = tuple(pair for pair in expected_pairs if pair not in pair_lookup)
        extra = tuple(pair for pair in pair_keys if pair not in expected_pairs)
        if missing or extra:
            raise ValueError("pair_results do not match the requested adjacent sequence.")
        ordered_pairs = expected_pairs

    ordered_results = tuple(pair_lookup[pair] for pair in ordered_pairs)
    diagnostics = (
        "candidate_chain_sequence_is_operational_not_modal_identity",
        "uses_existing_adjacent_association_results_only",
        "no_non_adjacent_candidate_association_created",
        "association_settings_preserved_per_pair",
        "sequence_order=" + "->".join(labels),
    )
    return AdjacentAssociationSequence(
        dynamic_labels=labels,
        pair_results=ordered_results,
        start_dynamic_label=labels[0],
        end_dynamic_label=labels[-1],
        condition_count=len(labels),
        pair_count=len(ordered_results),
        complete=True,
        valid=True,
        failure_reason=None,
        diagnostics=diagnostics,
    )


def build_cross_condition_candidate_chains(
    pair_results: AdjacentAssociationSequence
    | Iterable[CrossConditionCandidateAssociationResult],
    dynamic_labels: Iterable[str] | None = None,
) -> CrossConditionCandidateChainResult:
    """Build deterministic maximal operational chains from adjacent matches."""

    sequence = validate_adjacent_association_sequence(pair_results, dynamic_labels)
    refs_by_key = _collect_candidate_references(sequence)
    all_refs = tuple(sorted(refs_by_key.values(), key=_candidate_sort_key))
    incoming: dict[tuple[str, str, int, int], CrossConditionCandidateMatch] = {}
    outgoing: dict[tuple[str, str, int, int], CrossConditionCandidateMatch] = {}
    accepted_match_ids: set[str] = set()
    skipped_rejected_match_count = 0

    for result in sequence.pair_results:
        for match in sorted(result.matches, key=_match_sort_key):
            if not match.accepted:
                skipped_rejected_match_count += 1
                continue
            source_key = _candidate_identity_key(match.lower_candidate_ref)
            target_key = _candidate_identity_key(match.higher_candidate_ref)
            if source_key not in refs_by_key or target_key not in refs_by_key:
                raise ValueError("match points to a candidate outside the sequence references.")
            if source_key in outgoing:
                raise ValueError("candidate has more than one outgoing adjacent match.")
            if target_key in incoming:
                raise ValueError("candidate has more than one incoming adjacent match.")
            if match.match_id in accepted_match_ids:
                raise ValueError("accepted match IDs must be unique across the sequence.")
            if _DYNAMIC_LABEL_INDEX[match.higher_candidate_ref.dynamic_label] - _DYNAMIC_LABEL_INDEX[match.lower_candidate_ref.dynamic_label] != 1:
                raise ValueError("chain construction accepts only adjacent match edges.")
            outgoing[source_key] = match
            incoming[target_key] = match
            accepted_match_ids.add(match.match_id)

    emerging_by_key, disappearing_by_key = _unmatched_contexts(sequence)
    split_contexts = _split_contexts_by_candidate(sequence)
    merge_contexts = _merge_contexts_by_candidate(sequence)
    visited: set[tuple[str, str, int, int]] = set()
    chains: list[CrossConditionCandidateChain] = []
    for key in tuple(sorted(refs_by_key, key=lambda item: _candidate_sort_key(refs_by_key[item]))):
        if key in visited or key in incoming:
            continue
        chains.append(_build_chain_from_start(
            key,
            refs_by_key,
            outgoing,
            sequence,
            emerging_by_key,
            disappearing_by_key,
            split_contexts,
            merge_contexts,
            visited,
        ))
    for key in tuple(sorted(refs_by_key, key=lambda item: _candidate_sort_key(refs_by_key[item]))):
        if key in visited:
            continue
        chains.append(_build_chain_from_start(
            key,
            refs_by_key,
            outgoing,
            sequence,
            emerging_by_key,
            disappearing_by_key,
            split_contexts,
            merge_contexts,
            visited,
        ))

    chains_tuple = tuple(sorted(chains, key=_chain_sort_key))
    matched_candidate_count = sum(
        len(chain.nodes) for chain in chains_tuple
        if chain.match_count > 0
    )
    diagnostics = [
        "candidate_chains_are_operational_not_physical_modal_identity",
        "no_modal_mode_conversion_was_performed",
        "no_non_adjacent_association_created",
        "no_frequency_gap_closure_was_performed",
        "split_and_merge_contexts_are_diagnostic_only",
        "deterministic_content_based_chain_ids",
        "complete_candidate_partition",
    ]
    if skipped_rejected_match_count:
        diagnostics.append(f"audit_only_rejected_matches_not_used:{skipped_rejected_match_count}")
    complete_chains = tuple(chain for chain in chains_tuple if chain.complete_across_requested_sequence)
    singleton_chains = tuple(chain for chain in chains_tuple if chain.isolated_candidate)
    partial_chains = tuple(
        chain for chain in chains_tuple
        if not chain.complete_across_requested_sequence and not chain.isolated_candidate
    )
    return CrossConditionCandidateChainResult(
        sequence=sequence,
        chains=chains_tuple,
        complete_chains=complete_chains,
        partial_chains=partial_chains,
        singleton_chains=singleton_chains,
        candidate_count=len(all_refs),
        matched_candidate_count=matched_candidate_count,
        unmatched_candidate_count=len(all_refs) - matched_candidate_count,
        chain_count=len(chains_tuple),
        complete_chain_count=len(complete_chains),
        partial_chain_count=len(partial_chains),
        singleton_chain_count=len(singleton_chains),
        ambiguous_chain_count=sum(chain.contains_ambiguous_match for chain in chains_tuple),
        near_threshold_chain_count=sum(chain.contains_near_threshold_match for chain in chains_tuple),
        split_context_chain_count=sum(chain.contains_possible_split_context for chain in chains_tuple),
        merge_context_chain_count=sum(chain.contains_possible_merge_context for chain in chains_tuple),
        all_candidate_references=all_refs,
        valid=True,
        failure_reason=None,
        diagnostics=tuple(diagnostics),
    )


def summarize_cross_condition_candidate_chains(
    result: CrossConditionCandidateChainResult,
) -> dict[str, object]:
    """Return a compact deterministic summary for audit reports."""

    if not isinstance(result, CrossConditionCandidateChainResult):
        raise ValueError("result must be a CrossConditionCandidateChainResult.")
    return {
        "dynamic_labels": result.sequence.dynamic_labels,
        "candidate_count": result.candidate_count,
        "matched_candidate_count": result.matched_candidate_count,
        "unmatched_candidate_count": result.unmatched_candidate_count,
        "chain_count": result.chain_count,
        "complete_chain_count": result.complete_chain_count,
        "partial_chain_count": result.partial_chain_count,
        "singleton_chain_count": result.singleton_chain_count,
        "ambiguous_chain_count": result.ambiguous_chain_count,
        "near_threshold_chain_count": result.near_threshold_chain_count,
        "split_context_chain_count": result.split_context_chain_count,
        "merge_context_chain_count": result.merge_context_chain_count,
        "chain_ids": tuple(chain.chain_id for chain in result.chains),
        "diagnostics": result.diagnostics,
    }


def _build_chain_from_start(
    start_key: tuple[str, str, int, int],
    refs_by_key: dict[tuple[str, str, int, int], CandidateReference],
    outgoing: dict[tuple[str, str, int, int], CrossConditionCandidateMatch],
    sequence: AdjacentAssociationSequence,
    emerging_by_key: dict[tuple[str, str, int, int], tuple[EmergingCandidate, ...]],
    disappearing_by_key: dict[tuple[str, str, int, int], tuple[DisappearingCandidate, ...]],
    split_contexts: dict[tuple[str, str, int, int], tuple[PossibleCandidateSplit, ...]],
    merge_contexts: dict[tuple[str, str, int, int], tuple[PossibleCandidateMerge, ...]],
    visited: set[tuple[str, str, int, int]],
) -> CrossConditionCandidateChain:
    keys = [start_key]
    matches: list[CrossConditionCandidateMatch] = []
    current = start_key
    while current in outgoing:
        match = outgoing[current]
        target = _candidate_identity_key(match.higher_candidate_ref)
        if target in keys:
            raise ValueError("candidate chain graph contains a cycle.")
        matches.append(match)
        keys.append(target)
        current = target
    visited.update(keys)

    refs = tuple(refs_by_key[key] for key in keys)
    nodes = tuple(
        _node_for_ref(index, refs, tuple(matches))
        for index in range(len(refs))
    )
    split_tuple = _contexts_for_refs(refs, split_contexts, _split_sort_key)
    merge_tuple = _contexts_for_refs(refs, merge_contexts, _merge_sort_key)
    return _assemble_chain(
        nodes,
        tuple(matches),
        sequence,
        emerging_by_key,
        disappearing_by_key,
        split_tuple,
        merge_tuple,
    )


def _assemble_chain(
    nodes: tuple[CandidateChainNode, ...],
    matches: tuple[CrossConditionCandidateMatch, ...],
    sequence: AdjacentAssociationSequence,
    emerging_by_key: dict[tuple[str, str, int, int], tuple[EmergingCandidate, ...]],
    disappearing_by_key: dict[tuple[str, str, int, int], tuple[DisappearingCandidate, ...]],
    split_contexts: tuple[PossibleCandidateSplit, ...],
    merge_contexts: tuple[PossibleCandidateMerge, ...],
) -> CrossConditionCandidateChain:
    refs = tuple(node.candidate_ref for node in nodes)
    frequencies = tuple(ref.representative_frequency_hz for ref in refs)
    step_changes = tuple(match.frequency_change_hz for match in matches)
    relative_changes = tuple(match.frequency_change_relative for match in matches)
    classifications = tuple(match.frequency_change_classification for match in matches)
    costs = tuple(match.association_diagnostic.total_cost for match in matches)
    margins = tuple(
        match.association_diagnostic.assignment_margin
        for match in matches
        if match.association_diagnostic.assignment_margin is not None
    )
    ambiguous_ids = tuple(match.match_id for match in matches if match.ambiguous)
    ambiguous_positions = tuple(
        index for index, match in enumerate(matches)
        if match.ambiguous
    )
    ambiguous_margins = tuple(
        match.association_diagnostic.assignment_margin
        for match in matches
        if match.ambiguous and match.association_diagnostic.assignment_margin is not None
    )
    near_ids = tuple(match.match_id for match in matches if match.near_threshold)
    near_positions = tuple(
        index for index, match in enumerate(matches)
        if match.near_threshold
    )
    first_key = _candidate_identity_key(refs[0])
    last_key = _candidate_identity_key(refs[-1])
    starts_at_boundary = refs[0].dynamic_label == sequence.start_dynamic_label
    ends_at_boundary = refs[-1].dynamic_label == sequence.end_dynamic_label
    starts_as_emerging = (
        not starts_at_boundary
        and first_key in emerging_by_key
        and _has_previous_pair_in_sequence(refs[0].dynamic_label, sequence)
    )
    ends_as_disappearing = (
        not ends_at_boundary
        and last_key in disappearing_by_key
        and _has_next_pair_in_sequence(refs[-1].dynamic_label, sequence)
    )
    complete = tuple(ref.dynamic_label for ref in refs) == sequence.dynamic_labels
    total_change = frequencies[-1] - frequencies[0]
    total_relative = total_change / (0.5 * (frequencies[-1] + frequencies[0]))
    chain_id = _chain_id(sequence.dynamic_labels, refs, matches)
    diagnostics = [
        "operational_candidate_chain_not_modal_identity",
        "built_from_adjacent_matches_only",
        "no_global_route_optimization",
        "no_gap_closure",
    ]
    if starts_as_emerging:
        diagnostics.append("starts_from_existing_emerging_candidate_record")
    if ends_as_disappearing:
        diagnostics.append("ends_with_existing_disappearing_candidate_record")
    if split_contexts:
        diagnostics.append("possible_split_context_only")
    if merge_contexts:
        diagnostics.append("possible_merge_context_only")
    return CrossConditionCandidateChain(
        chain_id=chain_id,
        nodes=nodes,
        start_dynamic_label=refs[0].dynamic_label,
        end_dynamic_label=refs[-1].dynamic_label,
        condition_span=_DYNAMIC_LABEL_INDEX[refs[-1].dynamic_label] - _DYNAMIC_LABEL_INDEX[refs[0].dynamic_label] + 1,
        condition_count=len(nodes),
        match_count=len(matches),
        complete_across_requested_sequence=complete,
        starts_at_sequence_boundary=starts_at_boundary,
        starts_as_emerging=starts_as_emerging,
        ends_at_sequence_boundary=ends_at_boundary,
        ends_as_disappearing=ends_as_disappearing,
        isolated_candidate=not matches,
        partial_chain=bool(matches) and not complete,
        contains_ambiguous_match=bool(ambiguous_ids),
        contains_near_threshold_match=bool(near_ids),
        contains_possible_split_context=bool(split_contexts),
        contains_possible_merge_context=bool(merge_contexts),
        frequency_trajectory_hz=frequencies,
        frequency_step_changes_hz=step_changes,
        frequency_step_changes_relative=relative_changes,
        frequency_change_classifications=classifications,
        initial_frequency_hz=frequencies[0],
        final_frequency_hz=frequencies[-1],
        total_frequency_change_hz=total_change,
        total_frequency_change_relative=total_relative,
        upward_step_count=classifications.count("frequency_shifted_up"),
        downward_step_count=classifications.count("frequency_shifted_down"),
        preserved_step_count=classifications.count("frequency_preserved"),
        indeterminate_step_count=classifications.count("frequency_shift_indeterminate"),
        association_costs=costs,
        maximum_association_cost=max(costs) if costs else None,
        minimum_association_cost=min(costs) if costs else None,
        mean_association_cost=sum(costs) / len(costs) if costs else None,
        maximum_normalized_association_cost=max(costs) if costs else None,
        ambiguous_match_count=len(ambiguous_ids),
        near_threshold_match_count=len(near_ids),
        minimum_assignment_margin=min(margins) if margins else None,
        ambiguous_match_ids=ambiguous_ids,
        ambiguous_match_positions=ambiguous_positions,
        ambiguous_assignment_margins=ambiguous_margins,
        near_threshold_match_ids=near_ids,
        near_threshold_match_positions=near_positions,
        possible_split_contexts=split_contexts,
        possible_merge_contexts=merge_contexts,
        diagnostics=tuple(diagnostics),
    )


def _node_for_ref(
    index: int,
    refs: tuple[CandidateReference, ...],
    matches: tuple[CrossConditionCandidateMatch, ...],
) -> CandidateChainNode:
    ref = refs[index]
    incoming = matches[index - 1] if index > 0 else None
    outgoing = matches[index] if index < len(matches) else None
    diagnostics = [
        "candidate_reference_not_copied",
        "position_index_is_nominal_dynamic_order_index",
    ]
    return CandidateChainNode(
        dynamic_label=ref.dynamic_label,
        candidate_ref=ref,
        position_index=_DYNAMIC_LABEL_INDEX[ref.dynamic_label],
        incoming_match_id=incoming.match_id if incoming is not None else None,
        outgoing_match_id=outgoing.match_id if outgoing is not None else None,
        incoming_frequency_change=incoming.frequency_change_hz if incoming is not None else None,
        outgoing_frequency_change=outgoing.frequency_change_hz if outgoing is not None else None,
        incoming_association_cost=incoming.association_diagnostic.total_cost if incoming is not None else None,
        outgoing_association_cost=outgoing.association_diagnostic.total_cost if outgoing is not None else None,
        ambiguous_incoming=incoming.ambiguous if incoming is not None else False,
        ambiguous_outgoing=outgoing.ambiguous if outgoing is not None else False,
        near_threshold_incoming=incoming.near_threshold if incoming is not None else False,
        near_threshold_outgoing=outgoing.near_threshold if outgoing is not None else False,
        diagnostics=tuple(diagnostics),
    )


def _validate_dynamic_label_sequence(labels: tuple[str, ...]) -> None:
    if not isinstance(labels, tuple):
        raise ValueError("dynamic_labels must be an immutable tuple.")
    if len(labels) < 2:
        raise ValueError("dynamic_labels must contain at least two adjacent conditions.")
    if any(label not in _DYNAMIC_LABEL_INDEX for label in labels):
        raise ValueError("dynamic_labels contains an unknown condition.")
    if len(labels) != len(set(labels)):
        raise ValueError("dynamic_labels must not contain repeated conditions.")
    indices = tuple(_DYNAMIC_LABEL_INDEX[label] for label in labels)
    if indices != tuple(sorted(indices)):
        raise ValueError("dynamic_labels must follow the nominal dynamic order.")
    if any(right - left != 1 for left, right in zip(indices, indices[1:], strict=False)):
        raise ValueError("dynamic_labels must form a contiguous adjacent sequence.")


def _validate_shared_condition_references(
    results: tuple[CrossConditionCandidateAssociationResult, ...],
) -> None:
    by_label: dict[str, dict[tuple[str, str, int, int], CandidateReference]] = {}
    for result in results:
        for ref in result.lower_candidate_references + result.higher_candidate_references:
            key = _candidate_identity_key(ref)
            current = by_label.setdefault(ref.dynamic_label, {})
            existing = current.get(key)
            if existing is None:
                current[key] = ref
            elif existing != ref:
                raise ValueError("shared candidate references are inconsistent across pair results.")
    for previous, current in zip(results, results[1:], strict=False):
        previous_keys = {
            _candidate_identity_key(ref) for ref in previous.higher_candidate_references
        }
        current_keys = {
            _candidate_identity_key(ref) for ref in current.lower_candidate_references
        }
        if previous_keys != current_keys:
            raise ValueError("shared condition candidate references must match exactly.")


def _validate_match_side(
    match_id: str | None,
    frequency_change: float | None,
    association_cost: float | None,
    ambiguous: bool,
    near_threshold: bool,
    side: str,
) -> None:
    if match_id is None:
        if frequency_change is not None or association_cost is not None:
            raise ValueError(f"{side} match metrics require a match ID.")
        if ambiguous or near_threshold:
            raise ValueError(f"{side} flags require a match ID.")
        return
    _text(match_id, f"{side}_match_id")
    _finite_optional(frequency_change, f"{side}_frequency_change")
    if frequency_change is None:
        raise ValueError(f"{side} frequency change is required with a match ID.")
    _finite_optional(association_cost, f"{side}_association_cost", nonnegative=True)
    if association_cost is None:
        raise ValueError(f"{side} association cost is required with a match ID.")
    if not isinstance(ambiguous, bool) or not isinstance(near_threshold, bool):
        raise ValueError(f"{side} flags must be booleans.")


def _validate_frequency_trajectory(chain: CrossConditionCandidateChain) -> None:
    if len(chain.frequency_trajectory_hz) != len(chain.nodes):
        raise ValueError("frequency trajectory must match chain nodes.")
    expected = tuple(node.candidate_ref.representative_frequency_hz for node in chain.nodes)
    if chain.frequency_trajectory_hz != expected:
        raise ValueError("frequency trajectory must mirror candidate references.")
    if any(not isfinite(value) or value <= 0.0 for value in chain.frequency_trajectory_hz):
        raise ValueError("chain frequencies must be finite and positive.")
    vectors = (
        chain.frequency_step_changes_hz,
        chain.frequency_step_changes_relative,
        chain.frequency_change_classifications,
    )
    if any(len(vector) != chain.match_count for vector in vectors):
        raise ValueError("frequency step vectors must match match_count.")
    if any(not isfinite(value) for value in chain.frequency_step_changes_hz + chain.frequency_step_changes_relative):
        raise ValueError("frequency step changes must be finite.")
    if any(item not in _FREQUENCY_CHANGE_CLASSIFICATIONS for item in chain.frequency_change_classifications):
        raise ValueError("frequency change classifications contain an unknown value.")
    if chain.initial_frequency_hz != chain.frequency_trajectory_hz[0]:
        raise ValueError("initial_frequency_hz must match trajectory.")
    if chain.final_frequency_hz != chain.frequency_trajectory_hz[-1]:
        raise ValueError("final_frequency_hz must match trajectory.")
    expected_total = chain.final_frequency_hz - chain.initial_frequency_hz
    if not isclose(chain.total_frequency_change_hz, expected_total, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("total_frequency_change_hz is incoherent.")
    expected_relative = expected_total / (
        0.5 * (chain.final_frequency_hz + chain.initial_frequency_hz)
    )
    if not isclose(chain.total_frequency_change_relative, expected_relative, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("total frequency change must use the symmetric relative denominator.")
    expected_counts = {
        "upward_step_count": chain.frequency_change_classifications.count("frequency_shifted_up"),
        "downward_step_count": chain.frequency_change_classifications.count("frequency_shifted_down"),
        "preserved_step_count": chain.frequency_change_classifications.count("frequency_preserved"),
        "indeterminate_step_count": chain.frequency_change_classifications.count("frequency_shift_indeterminate"),
    }
    for name, expected_count in expected_counts.items():
        if getattr(chain, name) != expected_count:
            raise ValueError(f"{name} is incoherent with frequency classifications.")


def _validate_cost_summary(chain: CrossConditionCandidateChain) -> None:
    if len(chain.association_costs) != chain.match_count:
        raise ValueError("association_costs must match match_count.")
    if any(not isfinite(value) or value < 0.0 for value in chain.association_costs):
        raise ValueError("association costs must be finite and non-negative.")
    if not chain.association_costs:
        none_fields = (
            chain.maximum_association_cost,
            chain.minimum_association_cost,
            chain.mean_association_cost,
            chain.maximum_normalized_association_cost,
            chain.minimum_assignment_margin,
        )
        if any(value is not None for value in none_fields):
            raise ValueError("chains without matches must expose absent cost aggregates as None.")
        return
    expected_max = max(chain.association_costs)
    expected_min = min(chain.association_costs)
    expected_mean = sum(chain.association_costs) / len(chain.association_costs)
    if not isclose(chain.maximum_association_cost, expected_max, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("maximum_association_cost is incoherent.")
    if not isclose(chain.minimum_association_cost, expected_min, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("minimum_association_cost is incoherent.")
    if not isclose(chain.mean_association_cost, expected_mean, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("mean_association_cost is incoherent.")
    if not isclose(chain.maximum_normalized_association_cost, expected_max, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("maximum_normalized_association_cost is incoherent.")
    _finite_optional(chain.minimum_assignment_margin, "minimum_assignment_margin", nonnegative=True)


def _collect_candidate_references(
    sequence: AdjacentAssociationSequence,
) -> dict[tuple[str, str, int, int], CandidateReference]:
    refs: dict[tuple[str, str, int, int], CandidateReference] = {}
    for result in sequence.pair_results:
        for ref in result.lower_candidate_references + result.higher_candidate_references:
            key = _candidate_identity_key(ref)
            existing = refs.get(key)
            if existing is None:
                refs[key] = ref
            elif existing != ref:
                raise ValueError("candidate references are inconsistent across pair results.")
    expected_labels = set(sequence.dynamic_labels)
    if any(ref.dynamic_label not in expected_labels for ref in refs.values()):
        raise ValueError("candidate references must belong to the requested sequence.")
    return refs


def _unmatched_contexts(
    sequence: AdjacentAssociationSequence,
) -> tuple[
    dict[tuple[str, str, int, int], tuple[EmergingCandidate, ...]],
    dict[tuple[str, str, int, int], tuple[DisappearingCandidate, ...]],
]:
    emerging: dict[tuple[str, str, int, int], list[EmergingCandidate]] = {}
    disappearing: dict[tuple[str, str, int, int], list[DisappearingCandidate]] = {}
    for result in sequence.pair_results:
        for item in result.emerging_candidates:
            emerging.setdefault(_candidate_identity_key(item.reference), []).append(item)
        for item in result.disappearing_candidates:
            disappearing.setdefault(_candidate_identity_key(item.reference), []).append(item)
    return (
        {key: tuple(sorted(value, key=lambda item: _candidate_sort_key(item.reference))) for key, value in emerging.items()},
        {key: tuple(sorted(value, key=lambda item: _candidate_sort_key(item.reference))) for key, value in disappearing.items()},
    )


def _split_contexts_by_candidate(
    sequence: AdjacentAssociationSequence,
) -> dict[tuple[str, str, int, int], tuple[PossibleCandidateSplit, ...]]:
    contexts: dict[tuple[str, str, int, int], list[PossibleCandidateSplit]] = {}
    for result in sequence.pair_results:
        for split in result.possible_splits:
            refs = (split.source_candidate_ref,) + split.target_candidate_refs
            for ref in refs:
                contexts.setdefault(_candidate_identity_key(ref), []).append(split)
    return {
        key: tuple(sorted(tuple(dict.fromkeys(value)), key=_split_sort_key))
        for key, value in contexts.items()
    }


def _merge_contexts_by_candidate(
    sequence: AdjacentAssociationSequence,
) -> dict[tuple[str, str, int, int], tuple[PossibleCandidateMerge, ...]]:
    contexts: dict[tuple[str, str, int, int], list[PossibleCandidateMerge]] = {}
    for result in sequence.pair_results:
        for merge in result.possible_merges:
            refs = merge.source_candidate_refs + (merge.target_candidate_ref,)
            for ref in refs:
                contexts.setdefault(_candidate_identity_key(ref), []).append(merge)
    return {
        key: tuple(sorted(tuple(dict.fromkeys(value)), key=_merge_sort_key))
        for key, value in contexts.items()
    }


def _contexts_for_refs(
    refs: tuple[CandidateReference, ...],
    contexts: dict[tuple[str, str, int, int], tuple[object, ...]],
    key,
) -> tuple:
    unique = []
    seen: set[object] = set()
    for ref in refs:
        for context in contexts.get(_candidate_identity_key(ref), ()):
            if context in seen:
                continue
            seen.add(context)
            unique.append(context)
    return tuple(sorted(unique, key=key))


def _has_previous_pair_in_sequence(
    label: str,
    sequence: AdjacentAssociationSequence,
) -> bool:
    index = sequence.dynamic_labels.index(label)
    return index > 0


def _has_next_pair_in_sequence(
    label: str,
    sequence: AdjacentAssociationSequence,
) -> bool:
    index = sequence.dynamic_labels.index(label)
    return index < len(sequence.dynamic_labels) - 1


def _chain_id(
    dynamic_labels: tuple[str, ...],
    refs: tuple[CandidateReference, ...],
    matches: tuple[CrossConditionCandidateMatch, ...],
) -> str:
    ref_tokens = tuple(
        f"{ref.dynamic_label}:{ref.recording_id}:{ref.candidate_id}:{ref.source_track_id}"
        for ref in refs
    )
    match_tokens = tuple(
        f"{match.lower_candidate_ref.dynamic_label}->{match.higher_candidate_ref.dynamic_label}:{match.match_id}"
        for match in matches
    )
    payload = "|".join((
        "sequence=" + ",".join(dynamic_labels),
        "refs=" + ",".join(ref_tokens),
        "matches=" + ",".join(match_tokens),
    ))
    return "candidate-chain-" + sha1(payload.encode("utf-8")).hexdigest()[:16]


def _chain_refs(chain: CrossConditionCandidateChain) -> tuple[CandidateReference, ...]:
    return tuple(node.candidate_ref for node in chain.nodes)


def _chain_sort_key(chain: CrossConditionCandidateChain) -> tuple:
    first = chain.nodes[0].candidate_ref
    return (
        _DYNAMIC_LABEL_INDEX[chain.start_dynamic_label],
        first.representative_frequency_hz,
        tuple(_candidate_identity_key(ref) for ref in _chain_refs(chain)),
        chain.chain_id,
    )


def _candidate_identity_key(ref: CandidateReference) -> tuple[str, str, int, int]:
    return ref.dynamic_label, ref.recording_id, ref.candidate_id, ref.source_track_id


def _candidate_sort_key(ref: CandidateReference) -> tuple[int, float, str, int, int]:
    return (
        _DYNAMIC_LABEL_INDEX[ref.dynamic_label],
        ref.representative_frequency_hz,
        ref.recording_id,
        ref.candidate_id,
        ref.source_track_id,
    )


def _match_sort_key(match: CrossConditionCandidateMatch) -> tuple[int, str, int, str, int, str]:
    return (
        _DYNAMIC_LABEL_INDEX[match.lower_candidate_ref.dynamic_label],
        match.lower_candidate_ref.recording_id,
        match.lower_candidate_ref.candidate_id,
        match.higher_candidate_ref.recording_id,
        match.higher_candidate_ref.candidate_id,
        match.match_id,
    )


def _split_sort_key(split: PossibleCandidateSplit) -> tuple:
    return (
        _candidate_sort_key(split.source_candidate_ref),
        tuple(_candidate_sort_key(ref) for ref in split.target_candidate_refs),
        split.costs,
    )


def _merge_sort_key(merge: PossibleCandidateMerge) -> tuple:
    return (
        tuple(_candidate_sort_key(ref) for ref in merge.source_candidate_refs),
        _candidate_sort_key(merge.target_candidate_ref),
        merge.costs,
    )
