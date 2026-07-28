"""Associação operacional de candidatos entre condições dinâmicas adjacentes.

Esta camada compara candidatos já caracterizados em pares nominais adjacentes
``pp -> p -> mf -> f -> ff``. Uma correspondência é apenas compatibilidade
operacional configurável entre dois candidatos; não é identidade modal física,
modo preservado, prova de linearidade ou prova de não linearidade.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isclose, isfinite, log2

import numpy as np
from scipy.optimize import linear_sum_assignment

from belllab.dynamic_comparison import (
    DYNAMIC_LABEL_ORDER,
    DynamicConditionSpectralSummary,
)
from belllab.types import ModalCandidate, PreImpactEvidence
from belllab.within_condition import CandidateReference, RecordingCandidateSet


_DYNAMIC_LABEL_INDEX = {
    label: index for index, label in enumerate(DYNAMIC_LABEL_ORDER)
}
_FREQUENCY_CHANGE_CLASSIFICATIONS = frozenset({
    "frequency_preserved",
    "frequency_shifted_up",
    "frequency_shifted_down",
    "frequency_shift_indeterminate",
})
_UNMATCHED_REASONS = frozenset({
    "no_candidate_in_frequency_range",
    "cost_above_threshold",
    "ambiguous_match",
    "missing_required_evidence",
    "candidate_rejected_by_policy",
    "no_compatible_candidate",
    "insufficient_data",
})
_DIAGNOSTIC_REJECTION_REASONS = frozenset({
    "absolute_frequency_gate",
    "relative_frequency_gate",
    "log_frequency_gate",
    "tau_gate",
    "maximum_ambiguous_fraction",
    "maximum_near_threshold_fraction",
    "minimum_assignment_margin",
    "missing_tau",
    "missing_preimpact_evidence",
    "impact_excitation_required",
    "missing_tracking_evidence",
    "cost_above_threshold",
})
_FREQUENCY_GATE_REASONS = frozenset({
    "absolute_frequency_gate",
    "relative_frequency_gate",
    "log_frequency_gate",
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
        raise ValueError(f"{name} must be finite when provided.")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive when provided.")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be non-negative when provided.")


def _fraction(value: float | None, name: str) -> None:
    if value is not None and (not isfinite(value) or not 0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be finite and in [0, 1] when provided.")


@dataclass(frozen=True, slots=True)
class AdjacentDynamicConditionPair:
    """Contrato público para um par nominal dinâmico estritamente adjacente."""

    lower_dynamic_label: str
    higher_dynamic_label: str
    step_count: int = 1
    adjacent: bool = True
    lower_condition_summary: DynamicConditionSpectralSummary | None = None
    higher_condition_summary: DynamicConditionSpectralSummary | None = None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            self.lower_dynamic_label not in _DYNAMIC_LABEL_INDEX
            or self.higher_dynamic_label not in _DYNAMIC_LABEL_INDEX
        ):
            raise ValueError("dynamic labels must be one of pp, p, mf, f, ff.")
        lower = _DYNAMIC_LABEL_INDEX[self.lower_dynamic_label]
        higher = _DYNAMIC_LABEL_INDEX[self.higher_dynamic_label]
        if lower >= higher:
            raise ValueError("condition pair must be passed in nominal lower-to-higher order.")
        expected_steps = higher - lower
        if self.step_count != expected_steps:
            raise ValueError("step_count is incoherent with the dynamic labels.")
        if expected_steps != 1 or not self.adjacent:
            raise ValueError("cross-condition candidate association accepts only adjacent pairs.")
        if self.lower_condition_summary is not None:
            if self.lower_condition_summary.dynamic_label != self.lower_dynamic_label:
                raise ValueError("lower_condition_summary label is incoherent.")
        if self.higher_condition_summary is not None:
            if self.higher_condition_summary.dynamic_label != self.higher_dynamic_label:
                raise ValueError("higher_condition_summary label is incoherent.")
        _strings(self.diagnostics, "condition pair diagnostics")


@dataclass(frozen=True, slots=True)
class CrossConditionCandidateAssociationSettings:
    """Configuração conservadora para associação entre condições adjacentes."""

    maximum_absolute_frequency_difference_hz: float | None = 2.0
    maximum_relative_frequency_difference: float | None = None
    maximum_log_frequency_difference: float | None = None
    frequency_cost_weight: float = 1.0
    frequency_stability_cost_weight: float = 0.0
    frequency_drift_cost_weight: float = 0.0
    frequency_fit_rmse_cost_weight: float = 0.0
    tau_cost_weight: float = 0.0
    maximum_log_tau_difference: float | None = None
    amplitude_fit_quality_cost_weight: float = 0.0
    allow_missing_tau: bool = True
    ambiguity_cost_weight: float = 0.0
    near_threshold_cost_weight: float = 0.0
    assignment_margin_cost_weight: float = 0.0
    maximum_ambiguous_fraction: float | None = None
    maximum_near_threshold_fraction: float | None = None
    minimum_assignment_margin: float | None = None
    impact_evidence_cost_weight: float = 0.0
    require_impact_excitation: bool = False
    allow_missing_preimpact_evidence: bool = True
    maximum_association_cost: float = 1.0
    ambiguity_margin_threshold: float = 0.1
    near_threshold_ratio: float = 0.9
    allow_rejected_candidates: bool = False
    allow_unmatched_candidates: bool = True
    detect_split_candidates: bool = True
    detect_merge_candidates: bool = True
    frequency_preserved_absolute_tolerance_hz: float | None = 0.5
    frequency_preserved_relative_tolerance: float | None = None

    def __post_init__(self) -> None:
        frequency_limits = (
            self.maximum_absolute_frequency_difference_hz,
            self.maximum_relative_frequency_difference,
            self.maximum_log_frequency_difference,
        )
        for name in (
            "maximum_absolute_frequency_difference_hz",
            "maximum_relative_frequency_difference",
            "maximum_log_frequency_difference",
        ):
            _finite_optional(getattr(self, name), name, positive=True)
        if not any(limit is not None for limit in frequency_limits):
            raise ValueError("at least one positive frequency difference limit is required.")
        _finite_optional(
            self.maximum_log_tau_difference,
            "maximum_log_tau_difference",
            nonnegative=True,
        )
        for name in (
            "frequency_cost_weight",
            "frequency_stability_cost_weight",
            "frequency_drift_cost_weight",
            "frequency_fit_rmse_cost_weight",
            "tau_cost_weight",
            "amplitude_fit_quality_cost_weight",
            "ambiguity_cost_weight",
            "near_threshold_cost_weight",
            "assignment_margin_cost_weight",
            "impact_evidence_cost_weight",
            "ambiguity_margin_threshold",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.frequency_cost_weight <= 0.0:
            raise ValueError("frequency_cost_weight must be positive.")
        for name in (
            "maximum_ambiguous_fraction",
            "maximum_near_threshold_fraction",
            "frequency_preserved_relative_tolerance",
        ):
            _fraction(getattr(self, name), name)
        _finite_optional(
            self.minimum_assignment_margin,
            "minimum_assignment_margin",
            nonnegative=True,
        )
        _finite_optional(
            self.frequency_preserved_absolute_tolerance_hz,
            "frequency_preserved_absolute_tolerance_hz",
            nonnegative=True,
        )
        if not isfinite(self.maximum_association_cost) or self.maximum_association_cost <= 0.0:
            raise ValueError("maximum_association_cost must be finite and positive.")
        if not isfinite(self.near_threshold_ratio) or not 0.0 <= self.near_threshold_ratio <= 1.0:
            raise ValueError("near_threshold_ratio must be finite and in [0, 1].")
        for name in (
            "allow_missing_tau",
            "require_impact_excitation",
            "allow_missing_preimpact_evidence",
            "allow_rejected_candidates",
            "allow_unmatched_candidates",
            "detect_split_candidates",
            "detect_merge_candidates",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")


@dataclass(frozen=True, slots=True)
class CrossConditionCandidateAssociationDiagnostic:
    """Custo, admissibilidade e seleção para um par candidato-candidato."""

    lower_recording_id: str
    lower_candidate_id: int
    higher_recording_id: str
    higher_candidate_id: int
    lower_dynamic_label: str
    higher_dynamic_label: str
    frequency_difference_hz: float
    relative_frequency_difference: float
    log_frequency_difference: float
    frequency_stability_difference: float | None
    frequency_drift_difference_hz: float | None
    frequency_fit_rmse_difference_hz: float | None
    tau_log_difference: float | None
    amplitude_fit_quality_difference: float | None
    ambiguous_fraction_difference: float | None
    near_threshold_fraction_difference: float | None
    assignment_margin_difference: float | None
    impact_evidence_compatible: bool | None
    frequency_cost_component: float
    frequency_stability_cost_component: float
    frequency_drift_cost_component: float
    frequency_fit_rmse_cost_component: float
    tau_cost_component: float
    amplitude_fit_quality_cost_component: float
    ambiguity_cost_component: float
    near_threshold_cost_component: float
    assignment_margin_cost_component: float
    impact_evidence_cost_component: float
    cost_components: tuple[tuple[str, float], ...]
    total_cost: float
    admissible: bool
    selected: bool
    row_assignment_margin: float | None
    column_assignment_margin: float | None
    assignment_margin: float | None
    ambiguous: bool
    near_threshold: bool
    frequency_change_classification: str
    rejection_reason: str | None = None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.lower_recording_id, "lower_recording_id")
        _text(self.higher_recording_id, "higher_recording_id")
        if self.lower_candidate_id < 0 or self.higher_candidate_id < 0:
            raise ValueError("candidate IDs must not be negative.")
        AdjacentDynamicConditionPair(
            self.lower_dynamic_label,
            self.higher_dynamic_label,
        )
        required = (
            self.frequency_difference_hz,
            self.relative_frequency_difference,
            self.log_frequency_difference,
            self.frequency_cost_component,
            self.frequency_stability_cost_component,
            self.frequency_drift_cost_component,
            self.frequency_fit_rmse_cost_component,
            self.tau_cost_component,
            self.amplitude_fit_quality_cost_component,
            self.ambiguity_cost_component,
            self.near_threshold_cost_component,
            self.assignment_margin_cost_component,
            self.impact_evidence_cost_component,
            self.total_cost,
        )
        if any(not isfinite(value) or value < 0.0 for value in required):
            raise ValueError("diagnostic costs and distances must be finite and non-negative.")
        for name in (
            "frequency_stability_difference",
            "frequency_drift_difference_hz",
            "frequency_fit_rmse_difference_hz",
            "tau_log_difference",
            "amplitude_fit_quality_difference",
            "ambiguous_fraction_difference",
            "near_threshold_fraction_difference",
            "assignment_margin_difference",
        ):
            _finite_optional(getattr(self, name), name, nonnegative=True)
        if not isinstance(self.cost_components, tuple) or not self.cost_components:
            raise ValueError("cost_components must be a nonempty immutable tuple.")
        names = tuple(name for name, _ in self.cost_components)
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("cost component names must be nonempty strings.")
        if len(names) != len(set(names)):
            raise ValueError("cost component names must be unique.")
        if any(not isfinite(value) or value < 0.0 for _, value in self.cost_components):
            raise ValueError("cost component values must be finite and non-negative.")
        expected = sum(value for _, value in self.cost_components)
        if not isclose(self.total_cost, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("total_cost must equal the sum of cost_components.")
        explicit = {
            "frequency": self.frequency_cost_component,
            "frequency_stability": self.frequency_stability_cost_component,
            "frequency_drift": self.frequency_drift_cost_component,
            "frequency_fit_rmse": self.frequency_fit_rmse_cost_component,
            "tau": self.tau_cost_component,
            "amplitude_fit_quality": self.amplitude_fit_quality_cost_component,
            "ambiguity": self.ambiguity_cost_component,
            "near_threshold": self.near_threshold_cost_component,
            "assignment_margin": self.assignment_margin_cost_component,
            "impact_evidence": self.impact_evidence_cost_component,
        }
        if dict(self.cost_components) != explicit:
            raise ValueError("cost_components must match the explicit component fields.")
        if self.selected and not self.admissible:
            raise ValueError("selected diagnostic must be admissible.")
        margins = (
            self.row_assignment_margin,
            self.column_assignment_margin,
            self.assignment_margin,
        )
        if any(value is not None and (not isfinite(value) or value < 0.0) for value in margins):
            raise ValueError("assignment margins must be finite and non-negative.")
        available = tuple(value for value in margins[:2] if value is not None)
        expected_margin = min(available) if available else None
        if self.assignment_margin != expected_margin:
            raise ValueError("assignment_margin must equal the minimum available margin.")
        if self.ambiguous and (not self.selected or self.assignment_margin is None):
            raise ValueError("ambiguous requires a selected association with a margin.")
        if self.near_threshold and not self.selected:
            raise ValueError("near_threshold is defined only for selected associations.")
        if self.frequency_change_classification not in _FREQUENCY_CHANGE_CLASSIFICATIONS:
            raise ValueError("frequency_change_classification is not recognized.")
        if self.rejection_reason is not None:
            if self.rejection_reason not in _DIAGNOSTIC_REJECTION_REASONS:
                raise ValueError("rejection_reason is not recognized.")
        _strings(self.diagnostics, "association diagnostic diagnostics")


@dataclass(frozen=True, slots=True)
class DisappearingCandidate:
    """Candidato da condição inferior preservado sem correspondência superior."""

    reference: CandidateReference
    condition: str
    reason: str
    best_alternatives: tuple[CandidateReference, ...] = ()
    minimum_cost_observed: float | None = None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.condition != self.reference.dynamic_label:
            raise ValueError("disappearing candidate condition is incoherent.")
        if self.reason not in _UNMATCHED_REASONS:
            raise ValueError("unknown disappearing candidate reason.")
        if not isinstance(self.best_alternatives, tuple):
            raise ValueError("best_alternatives must be an immutable tuple.")
        _finite_optional(self.minimum_cost_observed, "minimum_cost_observed", nonnegative=True)
        _strings(self.diagnostics, "disappearing candidate diagnostics")


@dataclass(frozen=True, slots=True)
class EmergingCandidate:
    """Candidato da condição superior preservado sem correspondência inferior."""

    reference: CandidateReference
    condition: str
    reason: str
    best_alternatives: tuple[CandidateReference, ...] = ()
    minimum_cost_observed: float | None = None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.condition != self.reference.dynamic_label:
            raise ValueError("emerging candidate condition is incoherent.")
        if self.reason not in _UNMATCHED_REASONS:
            raise ValueError("unknown emerging candidate reason.")
        if not isinstance(self.best_alternatives, tuple):
            raise ValueError("best_alternatives must be an immutable tuple.")
        _finite_optional(self.minimum_cost_observed, "minimum_cost_observed", nonnegative=True)
        _strings(self.diagnostics, "emerging candidate diagnostics")


@dataclass(frozen=True, slots=True)
class PossibleCandidateSplit:
    """Indício operacional de divisão; não resolve associação um-para-muitos."""

    source_candidate_ref: CandidateReference
    target_candidate_refs: tuple[CandidateReference, ...]
    costs: tuple[float, ...]
    assignment_margins: tuple[float, ...]
    target_frequencies_hz: tuple[float, ...]
    possible_split: bool = True
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.target_candidate_refs) < 2:
            raise ValueError("possible split requires at least two targets.")
        if not (
            len(self.target_candidate_refs)
            == len(self.costs)
            == len(self.assignment_margins)
            == len(self.target_frequencies_hz)
        ):
            raise ValueError("possible split vectors must have equal length.")
        if any(item.dynamic_label == self.source_candidate_ref.dynamic_label for item in self.target_candidate_refs):
            raise ValueError("possible split must refer to a higher adjacent condition.")
        if any(not isfinite(value) or value < 0.0 for value in self.costs + self.assignment_margins):
            raise ValueError("possible split costs and margins must be finite and non-negative.")
        if any(not isfinite(value) or value <= 0.0 for value in self.target_frequencies_hz):
            raise ValueError("possible split target frequencies must be finite and positive.")
        if self.possible_split is not True:
            raise ValueError("possible_split diagnostic must be True.")
        _strings(self.diagnostics, "possible split diagnostics")


@dataclass(frozen=True, slots=True)
class PossibleCandidateMerge:
    """Indício operacional de fusão; não resolve associação muitos-para-um."""

    source_candidate_refs: tuple[CandidateReference, ...]
    target_candidate_ref: CandidateReference
    costs: tuple[float, ...]
    assignment_margins: tuple[float, ...]
    source_frequencies_hz: tuple[float, ...]
    possible_merge: bool = True
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.source_candidate_refs) < 2:
            raise ValueError("possible merge requires at least two sources.")
        if not (
            len(self.source_candidate_refs)
            == len(self.costs)
            == len(self.assignment_margins)
            == len(self.source_frequencies_hz)
        ):
            raise ValueError("possible merge vectors must have equal length.")
        if any(item.dynamic_label == self.target_candidate_ref.dynamic_label for item in self.source_candidate_refs):
            raise ValueError("possible merge must refer to a lower adjacent condition.")
        if any(not isfinite(value) or value < 0.0 for value in self.costs + self.assignment_margins):
            raise ValueError("possible merge costs and margins must be finite and non-negative.")
        if any(not isfinite(value) or value <= 0.0 for value in self.source_frequencies_hz):
            raise ValueError("possible merge source frequencies must be finite and positive.")
        if self.possible_merge is not True:
            raise ValueError("possible_merge diagnostic must be True.")
        _strings(self.diagnostics, "possible merge diagnostics")


@dataclass(frozen=True, slots=True)
class CrossConditionCandidateMatch:
    """Correspondência operacional entre dois candidatos, não um modo preservado."""

    match_id: str
    lower_candidate_ref: CandidateReference
    higher_candidate_ref: CandidateReference
    association_diagnostic: CrossConditionCandidateAssociationDiagnostic
    frequency_change_classification: str
    frequency_change_hz: float
    frequency_change_relative: float
    tau_change: float | None
    impact_evidence_compatible: bool | None
    ambiguous: bool
    near_threshold: bool
    accepted: bool
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.match_id, "match_id")
        pair = AdjacentDynamicConditionPair(
            self.lower_candidate_ref.dynamic_label,
            self.higher_candidate_ref.dynamic_label,
        )
        if (
            pair.lower_dynamic_label != self.association_diagnostic.lower_dynamic_label
            or pair.higher_dynamic_label != self.association_diagnostic.higher_dynamic_label
        ):
            raise ValueError("match diagnostic labels are incoherent.")
        if not self.association_diagnostic.selected:
            raise ValueError("match requires a selected association diagnostic.")
        if (
            self.association_diagnostic.lower_recording_id,
            self.association_diagnostic.lower_candidate_id,
        ) != (
            self.lower_candidate_ref.recording_id,
            self.lower_candidate_ref.candidate_id,
        ):
            raise ValueError("lower match reference does not match its diagnostic.")
        if (
            self.association_diagnostic.higher_recording_id,
            self.association_diagnostic.higher_candidate_id,
        ) != (
            self.higher_candidate_ref.recording_id,
            self.higher_candidate_ref.candidate_id,
        ):
            raise ValueError("higher match reference does not match its diagnostic.")
        if self.frequency_change_classification not in _FREQUENCY_CHANGE_CLASSIFICATIONS:
            raise ValueError("frequency_change_classification is not recognized.")
        if self.frequency_change_classification != self.association_diagnostic.frequency_change_classification:
            raise ValueError("match and diagnostic frequency classifications differ.")
        if not isfinite(self.frequency_change_hz) or not isfinite(self.frequency_change_relative):
            raise ValueError("frequency changes must be finite.")
        _finite_optional(self.tau_change, "tau_change")
        if self.impact_evidence_compatible != self.association_diagnostic.impact_evidence_compatible:
            raise ValueError("impact evidence compatibility must mirror the diagnostic.")
        if self.ambiguous != self.association_diagnostic.ambiguous:
            raise ValueError("match ambiguity must mirror the diagnostic.")
        if self.near_threshold != self.association_diagnostic.near_threshold:
            raise ValueError("match near-threshold state must mirror the diagnostic.")
        if self.accepted and (
            not self.lower_candidate_ref.accepted
            or not self.higher_candidate_ref.accepted
        ):
            raise ValueError("accepted match cannot promote rejected candidates.")
        _strings(self.diagnostics, "match diagnostics")


@dataclass(frozen=True, slots=True)
class CrossConditionCandidateAssociationResult:
    """Resultado completo para um par adjacente de condições dinâmicas."""

    lower_dynamic_label: str
    higher_dynamic_label: str
    lower_candidate_references: tuple[CandidateReference, ...]
    higher_candidate_references: tuple[CandidateReference, ...]
    matches: tuple[CrossConditionCandidateMatch, ...]
    disappearing_candidates: tuple[DisappearingCandidate, ...]
    emerging_candidates: tuple[EmergingCandidate, ...]
    possible_splits: tuple[PossibleCandidateSplit, ...]
    possible_merges: tuple[PossibleCandidateMerge, ...]
    association_diagnostics: tuple[CrossConditionCandidateAssociationDiagnostic, ...]
    settings: CrossConditionCandidateAssociationSettings
    valid: bool
    failure_reason: str | None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        AdjacentDynamicConditionPair(self.lower_dynamic_label, self.higher_dynamic_label)
        if self.lower_candidate_references != tuple(sorted(self.lower_candidate_references, key=_ref_key)):
            raise ValueError("lower_candidate_references must be in deterministic order.")
        if self.higher_candidate_references != tuple(sorted(self.higher_candidate_references, key=_ref_key)):
            raise ValueError("higher_candidate_references must be in deterministic order.")
        lower_source = tuple(_ref_key(item) for item in self.lower_candidate_references)
        higher_source = tuple(_ref_key(item) for item in self.higher_candidate_references)
        if len(lower_source) != len(set(lower_source)) or len(higher_source) != len(set(higher_source)):
            raise ValueError("candidate references must be unique within each condition.")
        lower_placed = tuple(
            _ref_key(match.lower_candidate_ref) for match in self.matches
        ) + tuple(_ref_key(item.reference) for item in self.disappearing_candidates)
        higher_placed = tuple(
            _ref_key(match.higher_candidate_ref) for match in self.matches
        ) + tuple(_ref_key(item.reference) for item in self.emerging_candidates)
        if sorted(lower_placed) != sorted(lower_source):
            raise ValueError("every lower reference must appear exactly once.")
        if sorted(higher_placed) != sorted(higher_source):
            raise ValueError("every higher reference must appear exactly once.")
        if len(lower_placed) != len(set(lower_placed)) or len(higher_placed) != len(set(higher_placed)):
            raise ValueError("candidate references are duplicated in the result.")
        match_ids = tuple(match.match_id for match in self.matches)
        if len(match_ids) != len(set(match_ids)):
            raise ValueError("match IDs must be unique.")
        diagnostic_keys = tuple(_diagnostic_key(item) for item in self.association_diagnostics)
        if len(diagnostic_keys) != len(set(diagnostic_keys)):
            raise ValueError("association diagnostics must not contain duplicate pairs.")
        for split in self.possible_splits:
            if _ref_key(split.source_candidate_ref) not in lower_source:
                raise ValueError("possible split source is not a lower candidate.")
            if any(_ref_key(ref) not in higher_source for ref in split.target_candidate_refs):
                raise ValueError("possible split target is not a higher candidate.")
        for merge in self.possible_merges:
            if _ref_key(merge.target_candidate_ref) not in higher_source:
                raise ValueError("possible merge target is not a higher candidate.")
            if any(_ref_key(ref) not in lower_source for ref in merge.source_candidate_refs):
                raise ValueError("possible merge source is not a lower candidate.")
        if self.valid:
            if self.failure_reason is not None:
                raise ValueError("valid result must not have failure_reason.")
        elif not self.failure_reason:
            raise ValueError("invalid result requires failure_reason.")
        _strings(self.diagnostics, "cross-condition result diagnostics")


def associate_candidates_across_adjacent_conditions(
    lower_condition: RecordingCandidateSet
    | CandidateReference
    | Iterable[RecordingCandidateSet]
    | Iterable[CandidateReference],
    higher_condition: RecordingCandidateSet
    | CandidateReference
    | Iterable[RecordingCandidateSet]
    | Iterable[CandidateReference],
    settings: CrossConditionCandidateAssociationSettings | None = None,
    *,
    lower_condition_summary: DynamicConditionSpectralSummary | None = None,
    higher_condition_summary: DynamicConditionSpectralSummary | None = None,
) -> CrossConditionCandidateAssociationResult:
    """Associa candidatos já caracterizados entre duas condições adjacentes."""

    lower_refs, lower_label = _coerce_condition_input(lower_condition)
    higher_refs, higher_label = _coerce_condition_input(higher_condition)
    if lower_label is None and lower_condition_summary is not None:
        lower_label = lower_condition_summary.dynamic_label
    if higher_label is None and higher_condition_summary is not None:
        higher_label = higher_condition_summary.dynamic_label
    if lower_label is None or higher_label is None:
        raise ValueError("dynamic labels cannot be inferred from empty inputs.")
    pair = AdjacentDynamicConditionPair(
        lower_label,
        higher_label,
        lower_condition_summary=lower_condition_summary,
        higher_condition_summary=higher_condition_summary,
        diagnostics=("adjacent_dynamic_condition_pair",),
    )
    return build_cross_condition_candidate_matches(
        lower_refs,
        higher_refs,
        pair,
        settings,
    )


def build_cross_condition_candidate_matches(
    lower_references: Iterable[CandidateReference],
    higher_references: Iterable[CandidateReference],
    pair: AdjacentDynamicConditionPair,
    settings: CrossConditionCandidateAssociationSettings | None = None,
) -> CrossConditionCandidateAssociationResult:
    """Resolve matching um-a-um e preserva candidatos sem par."""

    cfg = settings or CrossConditionCandidateAssociationSettings()
    AdjacentDynamicConditionPair(
        pair.lower_dynamic_label,
        pair.higher_dynamic_label,
        pair.step_count,
        pair.adjacent,
        pair.lower_condition_summary,
        pair.higher_condition_summary,
        pair.diagnostics,
    )
    lower_all = tuple(sorted(lower_references, key=_ref_key))
    higher_all = tuple(sorted(higher_references, key=_ref_key))
    if any(ref.dynamic_label != pair.lower_dynamic_label for ref in lower_all):
        raise ValueError("lower references must match the lower dynamic label.")
    if any(ref.dynamic_label != pair.higher_dynamic_label for ref in higher_all):
        raise ValueError("higher references must match the higher dynamic label.")

    lower_policy: list[DisappearingCandidate] = []
    higher_policy: list[EmergingCandidate] = []
    lower_eligible: list[CandidateReference] = []
    higher_eligible: list[CandidateReference] = []
    for ref in lower_all:
        reason = _policy_rejection(ref, cfg)
        if reason is None:
            lower_eligible.append(ref)
        else:
            lower_policy.append(DisappearingCandidate(
                ref,
                ref.dynamic_label,
                reason,
                diagnostics=("preserved_without_cross_condition_match",),
            ))
    for ref in higher_all:
        reason = _policy_rejection(ref, cfg)
        if reason is None:
            higher_eligible.append(ref)
        else:
            higher_policy.append(EmergingCandidate(
                ref,
                ref.dynamic_label,
                reason,
                diagnostics=("preserved_without_cross_condition_match",),
            ))

    diagnostics: list[CrossConditionCandidateAssociationDiagnostic] = []
    selected_pairs: set[tuple[int, int]] = set()
    costs = np.full((len(lower_eligible), len(higher_eligible)), np.inf)
    drafts: dict[tuple[int, int], CrossConditionCandidateAssociationDiagnostic] = {}
    for row, lower_ref in enumerate(lower_eligible):
        for column, higher_ref in enumerate(higher_eligible):
            draft = _compare_candidates(lower_ref, higher_ref, pair, cfg)
            drafts[row, column] = draft
            if draft.admissible:
                costs[row, column] = draft.total_cost
    if costs.size and np.isfinite(costs).any():
        safe = np.where(np.isfinite(costs), costs, 1e15)
        rows, columns = linear_sum_assignment(safe)
        selected_pairs = {
            (int(row), int(column))
            for row, column in zip(rows, columns, strict=True)
            if np.isfinite(costs[row, column])
        }

    selected_by_key: dict[tuple[str, int, str, int], CrossConditionCandidateAssociationDiagnostic] = {}
    for key, draft in drafts.items():
        row, column = key
        if key in selected_pairs:
            row_margin, column_margin, assignment_margin = _margins(costs, row, column)
            ambiguous = assignment_margin is not None and _inclusive_le(
                assignment_margin,
                cfg.ambiguity_margin_threshold,
            )
            near_threshold = _inclusive_ge(
                draft.total_cost,
                cfg.near_threshold_ratio * cfg.maximum_association_cost,
            )
            final = _replace_diagnostic(
                draft,
                selected=True,
                row_assignment_margin=row_margin,
                column_assignment_margin=column_margin,
                assignment_margin=assignment_margin,
                ambiguous=ambiguous,
                near_threshold=near_threshold,
                diagnostics=draft.diagnostics + ("selected_by_hungarian",),
            )
            diagnostics.append(final)
            selected_by_key[_diagnostic_key(final)] = final
        else:
            diagnostics.append(draft)
    diagnostics = sorted(diagnostics, key=_diagnostic_key)

    selected_lower = {row for row, _ in selected_pairs}
    selected_higher = {column for _, column in selected_pairs}
    matches: list[CrossConditionCandidateMatch] = []
    for match_index, (row, column) in enumerate(sorted(selected_pairs)):
        lower_ref = lower_eligible[row]
        higher_ref = higher_eligible[column]
        diagnostic = selected_by_key[
            (
                lower_ref.recording_id,
                lower_ref.candidate_id,
                higher_ref.recording_id,
                higher_ref.candidate_id,
            )
        ]
        matches.append(_match(match_index, lower_ref, higher_ref, diagnostic))

    disappearing = list(lower_policy)
    for row, ref in enumerate(lower_eligible):
        if row in selected_lower:
            continue
        reason = _unmatched_reason(ref, "lower", diagnostics, lower_eligible, higher_eligible)
        disappearing.append(DisappearingCandidate(
            ref,
            ref.dynamic_label,
            reason,
            _best_alternatives(ref, "lower", diagnostics, higher_eligible),
            _minimum_cost(ref, "lower", diagnostics),
            ("preserved_without_cross_condition_match",),
        ))

    emerging = list(higher_policy)
    for column, ref in enumerate(higher_eligible):
        if column in selected_higher:
            continue
        reason = _unmatched_reason(ref, "higher", diagnostics, lower_eligible, higher_eligible)
        emerging.append(EmergingCandidate(
            ref,
            ref.dynamic_label,
            reason,
            _best_alternatives(ref, "higher", diagnostics, lower_eligible),
            _minimum_cost(ref, "higher", diagnostics),
            ("preserved_without_cross_condition_match",),
        ))

    possible_splits = (
        detect_possible_candidate_splits(lower_eligible, higher_eligible, diagnostics, cfg)
        if cfg.detect_split_candidates
        else ()
    )
    possible_merges = (
        detect_possible_candidate_merges(lower_eligible, higher_eligible, diagnostics, cfg)
        if cfg.detect_merge_candidates
        else ()
    )
    result_diagnostics = [
        "cross_condition_candidate_correspondence_is_operational_not_modal_identity",
        "not_a_modal_mode_tracking_result",
        "no_modal_mode_conversion_was_performed",
        "adjacent_dynamic_condition_pair_only",
        "association_method=hungarian",
    ]
    if not matches:
        result_diagnostics.append("no_reliable_correspondence")
    if possible_splits:
        result_diagnostics.append("possible_split_detected")
    if possible_merges:
        result_diagnostics.append("possible_merge_detected")
    insufficient = not lower_all or not higher_all
    if insufficient:
        result_diagnostics.append("insufficient_candidate_data")
    unmatched_present = bool(disappearing or emerging)
    valid = not insufficient and (
        cfg.allow_unmatched_candidates or not unmatched_present
    )
    failure_reason = None
    if insufficient:
        failure_reason = "insufficient_candidate_data"
    elif not cfg.allow_unmatched_candidates and unmatched_present:
        failure_reason = "unmatched_candidates_not_allowed_by_configuration"
    return CrossConditionCandidateAssociationResult(
        lower_dynamic_label=pair.lower_dynamic_label,
        higher_dynamic_label=pair.higher_dynamic_label,
        lower_candidate_references=lower_all,
        higher_candidate_references=higher_all,
        matches=tuple(sorted(matches, key=lambda item: item.match_id)),
        disappearing_candidates=tuple(sorted(disappearing, key=lambda item: _ref_key(item.reference))),
        emerging_candidates=tuple(sorted(emerging, key=lambda item: _ref_key(item.reference))),
        possible_splits=possible_splits,
        possible_merges=possible_merges,
        association_diagnostics=tuple(diagnostics),
        settings=cfg,
        valid=valid,
        failure_reason=failure_reason,
        diagnostics=tuple(dict.fromkeys(result_diagnostics)),
    )


def detect_possible_candidate_splits(
    lower_references: Iterable[CandidateReference],
    higher_references: Iterable[CandidateReference],
    association_diagnostics: Iterable[CrossConditionCandidateAssociationDiagnostic],
    settings: CrossConditionCandidateAssociationSettings | None = None,
) -> tuple[PossibleCandidateSplit, ...]:
    """Detecta indício de divisão sem modificar o matching um-a-um."""

    cfg = settings or CrossConditionCandidateAssociationSettings()
    lower = tuple(sorted(lower_references, key=_ref_key))
    higher_by_key = {_short_ref_key(ref): ref for ref in higher_references}
    by_lower: dict[tuple[str, int], list[CrossConditionCandidateAssociationDiagnostic]] = {}
    for diagnostic in association_diagnostics:
        if diagnostic.admissible:
            by_lower.setdefault(
                (diagnostic.lower_recording_id, diagnostic.lower_candidate_id),
                [],
            ).append(diagnostic)
    splits: list[PossibleCandidateSplit] = []
    for source in lower:
        alternatives = sorted(
            by_lower.get(_short_ref_key(source), ()),
            key=lambda item: (item.total_cost, item.higher_recording_id, item.higher_candidate_id),
        )
        if len(alternatives) < 2:
            continue
        best = alternatives[0].total_cost
        close = tuple(
            item for item in alternatives
            if _inclusive_le(item.total_cost - best, cfg.ambiguity_margin_threshold)
        )
        if len(close) < 2 or not _split_frequency_neighborhood(source, close, cfg):
            continue
        target_refs = tuple(
            higher_by_key[(item.higher_recording_id, item.higher_candidate_id)]
            for item in close
        )
        splits.append(PossibleCandidateSplit(
            source_candidate_ref=source,
            target_candidate_refs=target_refs,
            costs=tuple(item.total_cost for item in close),
            assignment_margins=tuple(item.total_cost - best for item in close),
            target_frequencies_hz=tuple(item.representative_frequency_hz for item in target_refs),
            diagnostics=(
                "possible_split",
                "operational_indication_not_physical_split",
            ),
        ))
    return tuple(sorted(splits, key=lambda item: _ref_key(item.source_candidate_ref)))


def detect_possible_candidate_merges(
    lower_references: Iterable[CandidateReference],
    higher_references: Iterable[CandidateReference],
    association_diagnostics: Iterable[CrossConditionCandidateAssociationDiagnostic],
    settings: CrossConditionCandidateAssociationSettings | None = None,
) -> tuple[PossibleCandidateMerge, ...]:
    """Detecta indício de fusão sem modificar o matching um-a-um."""

    cfg = settings or CrossConditionCandidateAssociationSettings()
    lower_by_key = {_short_ref_key(ref): ref for ref in lower_references}
    higher = tuple(sorted(higher_references, key=_ref_key))
    by_higher: dict[tuple[str, int], list[CrossConditionCandidateAssociationDiagnostic]] = {}
    for diagnostic in association_diagnostics:
        if diagnostic.admissible:
            by_higher.setdefault(
                (diagnostic.higher_recording_id, diagnostic.higher_candidate_id),
                [],
            ).append(diagnostic)
    merges: list[PossibleCandidateMerge] = []
    for target in higher:
        alternatives = sorted(
            by_higher.get(_short_ref_key(target), ()),
            key=lambda item: (item.total_cost, item.lower_recording_id, item.lower_candidate_id),
        )
        if len(alternatives) < 2:
            continue
        best = alternatives[0].total_cost
        close = tuple(
            item for item in alternatives
            if _inclusive_le(item.total_cost - best, cfg.ambiguity_margin_threshold)
        )
        if len(close) < 2 or not _merge_frequency_neighborhood(target, close, cfg):
            continue
        source_refs = tuple(
            lower_by_key[(item.lower_recording_id, item.lower_candidate_id)]
            for item in close
        )
        merges.append(PossibleCandidateMerge(
            source_candidate_refs=source_refs,
            target_candidate_ref=target,
            costs=tuple(item.total_cost for item in close),
            assignment_margins=tuple(item.total_cost - best for item in close),
            source_frequencies_hz=tuple(item.representative_frequency_hz for item in source_refs),
            diagnostics=(
                "possible_merge",
                "operational_indication_not_physical_merge",
            ),
        ))
    return tuple(sorted(merges, key=lambda item: _ref_key(item.target_candidate_ref)))


def _coerce_condition_input(
    items: RecordingCandidateSet
    | CandidateReference
    | Iterable[RecordingCandidateSet]
    | Iterable[CandidateReference],
) -> tuple[tuple[CandidateReference, ...], str | None]:
    if isinstance(items, (RecordingCandidateSet, CandidateReference)):
        sequence = (items,)
    else:
        sequence = tuple(items)
    if not sequence:
        return (), None
    if all(isinstance(item, RecordingCandidateSet) for item in sequence):
        recordings = tuple(sorted(sequence, key=lambda item: item.recording_id))
        labels = {recording.condition.dynamic_label for recording in recordings}
        if len(labels) != 1:
            raise ValueError("one side of cross-condition association cannot mix labels.")
        refs: list[CandidateReference] = []
        for recording in recordings:
            evidence = {
                item.source_track_id: item for item in recording.preimpact_evidence
            }
            for candidate in sorted(recording.candidates, key=lambda item: item.candidate_id):
                if candidate.representative_frequency_hz is None:
                    raise ValueError("cross-condition association requires representative frequencies.")
                refs.append(_reference(recording, candidate, evidence.get(candidate.source_track_id)))
        return tuple(sorted(refs, key=_ref_key)), next(iter(labels))
    if all(isinstance(item, CandidateReference) for item in sequence):
        refs = tuple(sorted(sequence, key=_ref_key))
        labels = {ref.dynamic_label for ref in refs}
        if len(labels) != 1:
            raise ValueError("one side of cross-condition association cannot mix labels.")
        return refs, next(iter(labels))
    raise TypeError("condition input must contain only RecordingCandidateSet or CandidateReference objects.")


def _reference(
    recording: RecordingCandidateSet,
    candidate: ModalCandidate,
    evidence: PreImpactEvidence | None,
) -> CandidateReference:
    return CandidateReference(
        recording.recording_id,
        candidate.candidate_id,
        candidate.source_track_id,
        candidate.representative_frequency_hz,
        recording.condition.dynamic_label,
        candidate.accepted,
        evidence.impact_excited if evidence is not None else None,
        evidence.classification if evidence is not None else None,
        candidate.frequency_stability,
        candidate.amplitude_tau_s,
        candidate.amplitude_fit_r_squared,
        candidate.frequency_drift_hz,
        candidate.frequency_fit_rmse_hz,
        candidate.coverage_fraction,
        candidate.ambiguous_assignment_fraction,
        candidate.near_threshold_assignment_fraction,
        candidate.minimum_assignment_margin,
        evidence.classification if evidence is not None else None,
        None,
        ("cross_condition_candidate_reference",),
    )


def _policy_rejection(
    ref: CandidateReference,
    cfg: CrossConditionCandidateAssociationSettings,
) -> str | None:
    if not ref.accepted and not cfg.allow_rejected_candidates:
        return "candidate_rejected_by_policy"
    if cfg.require_impact_excitation and ref.impact_excited is not True:
        return "missing_required_evidence"
    if not cfg.allow_missing_preimpact_evidence and ref.impact_excited is None:
        return "missing_required_evidence"
    return None


def _compare_candidates(
    lower: CandidateReference,
    higher: CandidateReference,
    pair: AdjacentDynamicConditionPair,
    cfg: CrossConditionCandidateAssociationSettings,
) -> CrossConditionCandidateAssociationDiagnostic:
    if lower.dynamic_label != pair.lower_dynamic_label or higher.dynamic_label != pair.higher_dynamic_label:
        raise ValueError("candidate labels must match the adjacent pair.")
    frequency_difference = abs(
        higher.representative_frequency_hz - lower.representative_frequency_hz
    )
    denominator = 0.5 * (
        higher.representative_frequency_hz + lower.representative_frequency_hz
    )
    relative_difference = frequency_difference / denominator
    log_difference = abs(log2(
        higher.representative_frequency_hz / lower.representative_frequency_hz
    ))
    gate_values = (
        (
            frequency_difference,
            cfg.maximum_absolute_frequency_difference_hz,
            "absolute_frequency_gate",
        ),
        (
            relative_difference,
            cfg.maximum_relative_frequency_difference,
            "relative_frequency_gate",
        ),
        (
            log_difference,
            cfg.maximum_log_frequency_difference,
            "log_frequency_gate",
        ),
    )
    active_gates = tuple(
        (value, limit, name) for value, limit, name in gate_values
        if limit is not None
    )
    canonical_value, canonical_limit, canonical_name = active_gates[0]
    frequency_cost = cfg.frequency_cost_weight * canonical_value / canonical_limit
    failed_gate = next(
        (name for value, limit, name in active_gates if value > limit),
        None,
    )

    stability_difference = _difference(
        lower.frequency_stability,
        higher.frequency_stability,
    )
    drift_difference = _difference(
        lower.frequency_drift_hz,
        higher.frequency_drift_hz,
    )
    rmse_difference = _difference(
        lower.frequency_fit_rmse_hz,
        higher.frequency_fit_rmse_hz,
    )
    tau_difference = _log_difference(lower.amplitude_tau_s, higher.amplitude_tau_s)
    quality_difference = _difference(
        lower.amplitude_fit_r_squared,
        higher.amplitude_fit_r_squared,
    )
    ambiguity_difference = _difference(
        lower.ambiguous_assignment_fraction,
        higher.ambiguous_assignment_fraction,
    )
    near_difference = _difference(
        lower.near_threshold_assignment_fraction,
        higher.near_threshold_assignment_fraction,
    )
    margin_difference = _difference(
        lower.minimum_assignment_margin,
        higher.minimum_assignment_margin,
    )
    impact_compatible = _impact_evidence_compatible(lower, higher)

    stability_cost = cfg.frequency_stability_cost_weight * (stability_difference or 0.0)
    drift_cost = cfg.frequency_drift_cost_weight * (drift_difference or 0.0)
    rmse_cost = cfg.frequency_fit_rmse_cost_weight * (rmse_difference or 0.0)
    tau_cost = cfg.tau_cost_weight * (tau_difference or 0.0)
    quality_cost = cfg.amplitude_fit_quality_cost_weight * (quality_difference or 0.0)
    ambiguity_cost = cfg.ambiguity_cost_weight * max(
        lower.ambiguous_assignment_fraction or 0.0,
        higher.ambiguous_assignment_fraction or 0.0,
    )
    near_cost = cfg.near_threshold_cost_weight * max(
        lower.near_threshold_assignment_fraction or 0.0,
        higher.near_threshold_assignment_fraction or 0.0,
    )
    assignment_margin_cost = cfg.assignment_margin_cost_weight * (margin_difference or 0.0)
    impact_cost = (
        cfg.impact_evidence_cost_weight
        * (0.0 if impact_compatible is not False else 1.0)
    )
    components = (
        ("frequency", frequency_cost),
        ("frequency_stability", stability_cost),
        ("frequency_drift", drift_cost),
        ("frequency_fit_rmse", rmse_cost),
        ("tau", tau_cost),
        ("amplitude_fit_quality", quality_cost),
        ("ambiguity", ambiguity_cost),
        ("near_threshold", near_cost),
        ("assignment_margin", assignment_margin_cost),
        ("impact_evidence", impact_cost),
    )
    total = sum(value for _, value in components)
    reason = failed_gate
    notes = [f"frequency_cost_normalized_by={canonical_name}"]

    if reason is None and not cfg.allow_missing_tau and (
        lower.amplitude_tau_s is None or higher.amplitude_tau_s is None
    ):
        reason = "missing_tau"
    if (
        reason is None
        and cfg.maximum_log_tau_difference is not None
        and tau_difference is not None
        and tau_difference > cfg.maximum_log_tau_difference
    ):
        reason = "tau_gate"
    if tau_difference is None and (
        cfg.tau_cost_weight > 0.0
        or cfg.maximum_log_tau_difference is not None
        or not cfg.allow_missing_tau
    ):
        notes.append("tau_not_applicable")
    if reason is None and not cfg.allow_missing_preimpact_evidence and (
        lower.impact_excited is None or higher.impact_excited is None
    ):
        reason = "missing_preimpact_evidence"
    if reason is None and cfg.require_impact_excitation and (
        lower.impact_excited is not True or higher.impact_excited is not True
    ):
        reason = "impact_excitation_required"
    if (
        lower.impact_excited is None or higher.impact_excited is None
    ) and (
        cfg.impact_evidence_cost_weight > 0.0
        or not cfg.allow_missing_preimpact_evidence
    ):
        notes.append("preimpact_evidence_not_applicable")
    reason = _tracking_gate_reason(lower, higher, cfg, reason, notes)
    if reason is None and total > cfg.maximum_association_cost:
        reason = "cost_above_threshold"
    admissible = reason is None
    return CrossConditionCandidateAssociationDiagnostic(
        lower.recording_id,
        lower.candidate_id,
        higher.recording_id,
        higher.candidate_id,
        lower.dynamic_label,
        higher.dynamic_label,
        frequency_difference,
        relative_difference,
        log_difference,
        stability_difference,
        drift_difference,
        rmse_difference,
        tau_difference,
        quality_difference,
        ambiguity_difference,
        near_difference,
        margin_difference,
        impact_compatible,
        frequency_cost,
        stability_cost,
        drift_cost,
        rmse_cost,
        tau_cost,
        quality_cost,
        ambiguity_cost,
        near_cost,
        assignment_margin_cost,
        impact_cost,
        components,
        total,
        admissible,
        False,
        None,
        None,
        None,
        False,
        False,
        _frequency_change_classification(lower, higher, cfg),
        reason,
        tuple(dict.fromkeys(notes)),
    )


def _tracking_gate_reason(
    lower: CandidateReference,
    higher: CandidateReference,
    cfg: CrossConditionCandidateAssociationSettings,
    current_reason: str | None,
    diagnostics: list[str],
) -> str | None:
    reason = current_reason
    if cfg.maximum_ambiguous_fraction is not None:
        values = (lower.ambiguous_assignment_fraction, higher.ambiguous_assignment_fraction)
        if any(value is None for value in values):
            diagnostics.append("ambiguity_fraction_not_applicable")
            reason = reason or "missing_tracking_evidence"
        elif max(values) > cfg.maximum_ambiguous_fraction:
            reason = reason or "maximum_ambiguous_fraction"
    if cfg.maximum_near_threshold_fraction is not None:
        values = (lower.near_threshold_assignment_fraction, higher.near_threshold_assignment_fraction)
        if any(value is None for value in values):
            diagnostics.append("near_threshold_fraction_not_applicable")
            reason = reason or "missing_tracking_evidence"
        elif max(values) > cfg.maximum_near_threshold_fraction:
            reason = reason or "maximum_near_threshold_fraction"
    if cfg.minimum_assignment_margin is not None:
        values = (lower.minimum_assignment_margin, higher.minimum_assignment_margin)
        if any(value is None for value in values):
            diagnostics.append("assignment_margin_not_applicable")
            reason = reason or "missing_tracking_evidence"
        elif min(values) < cfg.minimum_assignment_margin:
            reason = reason or "minimum_assignment_margin"
    return reason


def _replace_diagnostic(
    item: CrossConditionCandidateAssociationDiagnostic,
    **changes: object,
) -> CrossConditionCandidateAssociationDiagnostic:
    values = {
        field: getattr(item, field)
        for field in item.__dataclass_fields__
    }
    values.update(changes)
    return CrossConditionCandidateAssociationDiagnostic(**values)


def _margins(
    costs: np.ndarray,
    row: int,
    column: int,
) -> tuple[float | None, float | None, float | None]:
    selected = costs[row, column]
    row_values = np.delete(costs[row], column)
    column_values = np.delete(costs[:, column], row)
    row_valid = row_values[np.isfinite(row_values)]
    column_valid = column_values[np.isfinite(column_values)]
    row_margin = float(np.min(row_valid) - selected) if row_valid.size else None
    column_margin = float(np.min(column_valid) - selected) if column_valid.size else None
    available = tuple(value for value in (row_margin, column_margin) if value is not None)
    return row_margin, column_margin, min(available) if available else None


def _match(
    match_index: int,
    lower: CandidateReference,
    higher: CandidateReference,
    diagnostic: CrossConditionCandidateAssociationDiagnostic,
) -> CrossConditionCandidateMatch:
    frequency_change = higher.representative_frequency_hz - lower.representative_frequency_hz
    relative_change = frequency_change / (
        0.5 * (lower.representative_frequency_hz + higher.representative_frequency_hz)
    )
    tau_change = (
        higher.amplitude_tau_s - lower.amplitude_tau_s
        if lower.amplitude_tau_s is not None and higher.amplitude_tau_s is not None
        else None
    )
    diagnostics = [
        "operational_candidate_correspondence_not_modal_identity",
        "not_a_preserved_physical_mode",
    ]
    if not lower.accepted or not higher.accepted:
        diagnostics.append("contains_rejected_candidate_for_audit_only")
    return CrossConditionCandidateMatch(
        match_id=f"{lower.dynamic_label}-{higher.dynamic_label}-{match_index:03d}",
        lower_candidate_ref=lower,
        higher_candidate_ref=higher,
        association_diagnostic=diagnostic,
        frequency_change_classification=diagnostic.frequency_change_classification,
        frequency_change_hz=frequency_change,
        frequency_change_relative=relative_change,
        tau_change=tau_change,
        impact_evidence_compatible=diagnostic.impact_evidence_compatible,
        ambiguous=diagnostic.ambiguous,
        near_threshold=diagnostic.near_threshold,
        accepted=lower.accepted and higher.accepted,
        diagnostics=tuple(diagnostics),
    )


def _unmatched_reason(
    ref: CandidateReference,
    side: str,
    diagnostics: list[CrossConditionCandidateAssociationDiagnostic],
    lower_refs: list[CandidateReference],
    higher_refs: list[CandidateReference],
) -> str:
    opposite = higher_refs if side == "lower" else lower_refs
    if not opposite:
        return "insufficient_data"
    relevant = _relevant_diagnostics(ref, side, diagnostics)
    if not relevant:
        return "no_compatible_candidate"
    if _competing_selected_ambiguous(ref, side, relevant, diagnostics):
        return "ambiguous_match"
    reasons = tuple(item.rejection_reason for item in relevant if item.rejection_reason)
    if reasons and all(reason in _FREQUENCY_GATE_REASONS for reason in reasons):
        return "no_candidate_in_frequency_range"
    if any(reason in {"missing_tau", "missing_preimpact_evidence", "impact_excitation_required", "missing_tracking_evidence"} for reason in reasons):
        return "missing_required_evidence"
    if any(reason == "cost_above_threshold" for reason in reasons):
        return "cost_above_threshold"
    if any(item.admissible for item in relevant):
        return "no_compatible_candidate"
    return "no_compatible_candidate"


def _competing_selected_ambiguous(
    ref: CandidateReference,
    side: str,
    relevant: list[CrossConditionCandidateAssociationDiagnostic],
    diagnostics: list[CrossConditionCandidateAssociationDiagnostic],
) -> bool:
    if side == "lower":
        target_keys = {
            (item.higher_recording_id, item.higher_candidate_id)
            for item in relevant
        }
        return any(
            item.selected
            and item.ambiguous
            and (item.lower_recording_id, item.lower_candidate_id) == _short_ref_key(ref)
            for item in diagnostics
        ) or any(
            item.selected
            and item.ambiguous
            and (item.higher_recording_id, item.higher_candidate_id) in target_keys
            for item in diagnostics
        )
    source_keys = {
        (item.lower_recording_id, item.lower_candidate_id)
        for item in relevant
    }
    return any(
        item.selected
        and item.ambiguous
        and (item.higher_recording_id, item.higher_candidate_id) == _short_ref_key(ref)
        for item in diagnostics
    ) or any(
        item.selected
        and item.ambiguous
        and (item.lower_recording_id, item.lower_candidate_id) in source_keys
        for item in diagnostics
    )


def _best_alternatives(
    ref: CandidateReference,
    side: str,
    diagnostics: list[CrossConditionCandidateAssociationDiagnostic],
    opposite_refs: list[CandidateReference],
) -> tuple[CandidateReference, ...]:
    relevant = _relevant_diagnostics(ref, side, diagnostics)
    if not relevant:
        return ()
    minimum = min(item.total_cost for item in relevant)
    keys: set[tuple[str, int]] = set()
    for item in relevant:
        if isclose(item.total_cost, minimum, rel_tol=1e-12, abs_tol=1e-12):
            if side == "lower":
                keys.add((item.higher_recording_id, item.higher_candidate_id))
            else:
                keys.add((item.lower_recording_id, item.lower_candidate_id))
    lookup = {_short_ref_key(item): item for item in opposite_refs}
    return tuple(lookup[key] for key in sorted(keys) if key in lookup)


def _minimum_cost(
    ref: CandidateReference,
    side: str,
    diagnostics: list[CrossConditionCandidateAssociationDiagnostic],
) -> float | None:
    relevant = _relevant_diagnostics(ref, side, diagnostics)
    return min((item.total_cost for item in relevant), default=None)


def _relevant_diagnostics(
    ref: CandidateReference,
    side: str,
    diagnostics: list[CrossConditionCandidateAssociationDiagnostic],
) -> list[CrossConditionCandidateAssociationDiagnostic]:
    key = _short_ref_key(ref)
    if side == "lower":
        return [
            item for item in diagnostics
            if (item.lower_recording_id, item.lower_candidate_id) == key
        ]
    return [
        item for item in diagnostics
        if (item.higher_recording_id, item.higher_candidate_id) == key
    ]


def _split_frequency_neighborhood(
    source: CandidateReference,
    alternatives: tuple[CrossConditionCandidateAssociationDiagnostic, ...],
    cfg: CrossConditionCandidateAssociationSettings,
) -> bool:
    if cfg.maximum_absolute_frequency_difference_hz is None:
        return True
    return all(
        diagnostic.frequency_difference_hz <= cfg.maximum_absolute_frequency_difference_hz
        for diagnostic in alternatives
    )


def _merge_frequency_neighborhood(
    target: CandidateReference,
    alternatives: tuple[CrossConditionCandidateAssociationDiagnostic, ...],
    cfg: CrossConditionCandidateAssociationSettings,
) -> bool:
    del target
    if cfg.maximum_absolute_frequency_difference_hz is None:
        return True
    return all(
        diagnostic.frequency_difference_hz <= cfg.maximum_absolute_frequency_difference_hz
        for diagnostic in alternatives
    )


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return abs(right - left)


def _log_difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return abs(log2(right / left))


def _impact_evidence_compatible(
    lower: CandidateReference,
    higher: CandidateReference,
) -> bool | None:
    if lower.impact_excited is None or higher.impact_excited is None:
        return None
    if lower.impact_excited and higher.impact_excited:
        return True
    lower_class = lower.preimpact_classification or lower.classification
    higher_class = higher.preimpact_classification or higher.classification
    if lower_class == higher_class:
        return True
    return lower.impact_excited == higher.impact_excited


def _frequency_change_classification(
    lower: CandidateReference,
    higher: CandidateReference,
    cfg: CrossConditionCandidateAssociationSettings,
) -> str:
    signed = higher.representative_frequency_hz - lower.representative_frequency_hz
    absolute = abs(signed)
    relative = absolute / (
        0.5 * (lower.representative_frequency_hz + higher.representative_frequency_hz)
    )
    if isclose(absolute, 0.0, rel_tol=0.0, abs_tol=1e-12):
        return "frequency_preserved"
    if cfg.frequency_preserved_absolute_tolerance_hz is not None and _inclusive_le(
        absolute,
        cfg.frequency_preserved_absolute_tolerance_hz,
    ):
        return "frequency_preserved"
    if cfg.frequency_preserved_relative_tolerance is not None and _inclusive_le(
        relative,
        cfg.frequency_preserved_relative_tolerance,
    ):
        return "frequency_preserved"
    if signed > 0.0:
        return "frequency_shifted_up"
    if signed < 0.0:
        return "frequency_shifted_down"
    return "frequency_shift_indeterminate"


def _inclusive_le(left: float, right: float) -> bool:
    return left <= right or isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def _inclusive_ge(left: float, right: float) -> bool:
    return left >= right or isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def _ref_key(ref: CandidateReference) -> tuple[str, float, str, int, int]:
    return (
        ref.dynamic_label,
        ref.representative_frequency_hz,
        ref.recording_id,
        ref.candidate_id,
        ref.source_track_id,
    )


def _short_ref_key(ref: CandidateReference) -> tuple[str, int]:
    return ref.recording_id, ref.candidate_id


def _diagnostic_key(
    item: CrossConditionCandidateAssociationDiagnostic,
) -> tuple[str, int, str, int]:
    return (
        item.lower_recording_id,
        item.lower_candidate_id,
        item.higher_recording_id,
        item.higher_candidate_id,
    )
