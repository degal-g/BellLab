"""Operational modal hypotheses built from candidate chains only.

This layer evaluates an existing :class:`CrossConditionCandidateChain` against
explicit criteria. A positive result is an operational modal hypothesis under
the active settings; it is not a proved physical mode, not a `ModalMode`, and
not proof of physical identity, hardening, softening, linearity, nonlinearity,
split, or merge.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from hashlib import sha1
from math import isclose, isfinite, log, sqrt

from belllab.candidate_chains import (
    CrossConditionCandidateChain,
    CrossConditionCandidateChainResult,
)
from belllab.dynamic_comparison import DYNAMIC_LABEL_ORDER


_DYNAMIC_LABEL_INDEX = {
    label: index for index, label in enumerate(DYNAMIC_LABEL_ORDER)
}
_MISSING_POLICIES = frozenset({"allow", "reservation", "insufficient", "reject"})


class ModalHypothesisStatus(str, Enum):
    """Mutually exclusive decision states for an operational modal hypothesis."""

    ACCEPTED = "accepted"
    ACCEPTED_WITH_RESERVATIONS = "accepted_with_reservations"
    INCONCLUSIVE = "inconclusive"
    REJECTED = "rejected"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    INVALID_INPUT = "invalid_input"


class ModalHypothesisReason(str, Enum):
    """Typed decision reasons preserved separately by role."""

    SUFFICIENT_CROSS_CONDITION_PERSISTENCE = "sufficient_cross_condition_persistence"
    SUFFICIENT_FREQUENCY_CONTINUITY = "sufficient_frequency_continuity"
    SUFFICIENT_TRACKING_QUALITY = "sufficient_tracking_quality"
    SUFFICIENT_DECAY_CONSISTENCY = "sufficient_decay_consistency"
    SUFFICIENT_IMPACT_EVIDENCE = "sufficient_impact_evidence"
    COMPLETE_CHAIN = "complete_chain"
    PARTIAL_BUT_SUPPORTED_CHAIN = "partial_but_supported_chain"
    SINGLETON_CHAIN = "singleton_chain"
    TOO_FEW_CONDITIONS = "too_few_conditions"
    TOO_FEW_MATCHES = "too_few_matches"
    FREQUENCY_DISCONTINUITY = "frequency_discontinuity"
    EXCESSIVE_FREQUENCY_VARIATION = "excessive_frequency_variation"
    EXCESSIVE_ASSOCIATION_COST = "excessive_association_cost"
    EXCESSIVE_AMBIGUITY = "excessive_ambiguity"
    EXCESSIVE_NEAR_THRESHOLD_FRACTION = "excessive_near_threshold_fraction"
    INSUFFICIENT_TRACKING_QUALITY = "insufficient_tracking_quality"
    INCONSISTENT_DECAY = "inconsistent_decay"
    MISSING_REQUIRED_DECAY = "missing_required_decay"
    MISSING_REQUIRED_IMPACT_EVIDENCE = "missing_required_impact_evidence"
    POSSIBLE_SPLIT_CONTEXT = "possible_split_context"
    POSSIBLE_MERGE_CONTEXT = "possible_merge_context"
    REJECTED_CANDIDATE_PRESENT = "rejected_candidate_present"
    INVALID_CHAIN = "invalid_chain"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True, slots=True)
class ModalHypothesisSettings:
    """Conservative configurable criteria for operational modal hypotheses.

    ``None`` disables optional numeric criteria. Weights are non-negative and
    absent evidence is handled by explicit policy fields; it is never converted
    silently to zero. Frequency is intentionally dominant by default.
    """

    minimum_condition_count: int | None = 2
    minimum_match_count: int | None = 1
    minimum_condition_coverage_fraction: float | None = 1.0
    require_complete_chain: bool = True
    allow_partial_chains: bool = False
    allow_singleton_chains: bool = False
    coverage_weight: float = 1.0

    maximum_step_absolute_frequency_change_hz: float | None = 2.0
    maximum_step_relative_frequency_change: float | None = None
    maximum_total_absolute_frequency_change_hz: float | None = 5.0
    maximum_total_relative_frequency_change: float | None = None
    maximum_frequency_trajectory_rmse_hz: float | None = 1.0
    maximum_frequency_trajectory_relative_rmse: float | None = None
    frequency_continuity_weight: float = 4.0

    maximum_match_cost: float | None = 1.0
    maximum_mean_match_cost: float | None = 0.75
    maximum_ambiguous_match_fraction: float | None = 0.25
    maximum_near_threshold_match_fraction: float | None = 0.25
    minimum_match_margin: float | None = None
    association_quality_weight: float = 1.0

    minimum_mean_coverage_fraction: float | None = 0.8
    maximum_mean_ambiguous_assignment_fraction: float | None = 0.2
    maximum_mean_near_threshold_assignment_fraction: float | None = 0.25
    minimum_mean_assignment_margin: float | None = None
    maximum_mean_frequency_fit_rmse_hz: float | None = 1.0
    tracking_quality_weight: float = 1.0
    missing_tracking_evidence_policy: str = "reservation"

    require_decay_evidence: bool = False
    allow_missing_decay: bool = True
    minimum_decay_value_count: int = 2
    maximum_log_tau_range: float | None = 0.5
    maximum_log_tau_standard_deviation: float | None = 0.25
    decay_consistency_weight: float = 1.0
    missing_decay_evidence_policy: str = "reservation"

    require_impact_excitation: bool = False
    allow_missing_preimpact_evidence: bool = True
    minimum_impact_supported_fraction: float | None = 0.5
    impact_evidence_weight: float = 1.0
    missing_impact_evidence_policy: str = "reservation"

    reject_possible_split_context: bool = False
    reject_possible_merge_context: bool = False
    reserve_possible_split_context: bool = True
    reserve_possible_merge_context: bool = True

    minimum_acceptance_score: float = 0.75
    minimum_reservation_score: float = 0.55
    maximum_rejection_count: int = 0
    allow_accepted_with_reservations: bool = True
    structural_context_penalty: float = 0.05
    missing_evidence_penalty: float = 0.05

    def __post_init__(self) -> None:
        for name in ("minimum_condition_count", "minimum_match_count"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must not be negative when provided.")
        if self.minimum_condition_count == 0:
            raise ValueError("minimum_condition_count must be positive when provided.")
        if self.minimum_decay_value_count < 1:
            raise ValueError("minimum_decay_value_count must be positive.")
        if self.maximum_rejection_count < 0:
            raise ValueError("maximum_rejection_count must not be negative.")
        for name in (
            "minimum_condition_coverage_fraction",
            "maximum_step_relative_frequency_change",
            "maximum_total_relative_frequency_change",
            "maximum_frequency_trajectory_relative_rmse",
            "maximum_ambiguous_match_fraction",
            "maximum_near_threshold_match_fraction",
            "minimum_mean_coverage_fraction",
            "maximum_mean_ambiguous_assignment_fraction",
            "maximum_mean_near_threshold_assignment_fraction",
            "minimum_impact_supported_fraction",
            "minimum_acceptance_score",
            "minimum_reservation_score",
        ):
            _fraction(getattr(self, name), name)
        for name in (
            "maximum_step_absolute_frequency_change_hz",
            "maximum_total_absolute_frequency_change_hz",
            "maximum_frequency_trajectory_rmse_hz",
            "maximum_match_cost",
            "maximum_mean_match_cost",
            "minimum_match_margin",
            "minimum_mean_assignment_margin",
            "maximum_mean_frequency_fit_rmse_hz",
            "maximum_log_tau_range",
            "maximum_log_tau_standard_deviation",
            "structural_context_penalty",
            "missing_evidence_penalty",
        ):
            _finite_optional(getattr(self, name), name, nonnegative=True)
        for name in (
            "coverage_weight",
            "frequency_continuity_weight",
            "association_quality_weight",
            "tracking_quality_weight",
            "decay_consistency_weight",
            "impact_evidence_weight",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        for name in (
            "require_complete_chain",
            "allow_partial_chains",
            "allow_singleton_chains",
            "require_decay_evidence",
            "allow_missing_decay",
            "require_impact_excitation",
            "allow_missing_preimpact_evidence",
            "reject_possible_split_context",
            "reject_possible_merge_context",
            "reserve_possible_split_context",
            "reserve_possible_merge_context",
            "allow_accepted_with_reservations",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")
        for name in (
            "missing_tracking_evidence_policy",
            "missing_decay_evidence_policy",
            "missing_impact_evidence_policy",
        ):
            if getattr(self, name) not in _MISSING_POLICIES:
                raise ValueError(f"{name} must be one of allow, reservation, insufficient, reject.")
        if self.require_complete_chain and self.allow_partial_chains:
            raise ValueError("require_complete_chain conflicts with allow_partial_chains.")
        if self.reject_possible_split_context and self.reserve_possible_split_context:
            raise ValueError("split context cannot be both rejected and reserved.")
        if self.reject_possible_merge_context and self.reserve_possible_merge_context:
            raise ValueError("merge context cannot be both rejected and reserved.")
        if self.minimum_reservation_score > self.minimum_acceptance_score:
            raise ValueError("minimum_reservation_score must not exceed minimum_acceptance_score.")


@dataclass(frozen=True, slots=True)
class ModalHypothesisCoverageEvidence:
    requested_condition_count: int
    observed_condition_count: int
    condition_coverage_fraction: float
    match_count: int
    complete_across_requested_sequence: bool
    starts_as_emerging: bool
    ends_as_disappearing: bool
    singleton: bool
    partial: bool
    passes: bool
    reasons: tuple[ModalHypothesisReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.requested_condition_count <= 0 or self.observed_condition_count < 0:
            raise ValueError("coverage condition counts are incoherent.")
        if self.match_count < 0:
            raise ValueError("match_count must not be negative.")
        _fraction(self.condition_coverage_fraction, "condition_coverage_fraction")
        _reason_tuple(self.reasons, "coverage reasons")
        _strings(self.diagnostics, "coverage diagnostics")


@dataclass(frozen=True, slots=True)
class ModalHypothesisFrequencyEvidence:
    frequencies_hz: tuple[float, ...]
    condition_labels: tuple[str, ...]
    signed_step_changes_hz: tuple[float, ...]
    absolute_step_changes_hz: tuple[float, ...]
    signed_step_relative_changes: tuple[float, ...]
    absolute_step_relative_changes: tuple[float, ...]
    maximum_step_change_hz: float | None
    maximum_step_relative_change: float | None
    total_signed_change_hz: float
    total_absolute_change_hz: float
    total_relative_symmetric_change: float
    trajectory_mean_hz: float
    trajectory_standard_deviation_hz: float
    trajectory_rmse_from_mean_hz: float
    trajectory_relative_rmse: float
    up_step_count: int
    down_step_count: int
    preserved_step_count: int
    indeterminate_step_count: int
    passes: bool
    reasons: tuple[ModalHypothesisReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.frequencies_hz or len(self.frequencies_hz) != len(self.condition_labels):
            raise ValueError("frequency evidence requires aligned frequencies and labels.")
        if any(not isfinite(value) or value <= 0.0 for value in self.frequencies_hz):
            raise ValueError("frequencies_hz must be finite and positive.")
        for values in (
            self.signed_step_changes_hz,
            self.absolute_step_changes_hz,
            self.signed_step_relative_changes,
            self.absolute_step_relative_changes,
        ):
            if len(values) != len(self.frequencies_hz) - 1:
                raise ValueError("frequency step vectors must match trajectory length.")
            if any(not isfinite(value) for value in values):
                raise ValueError("frequency step vectors must be finite.")
        for name in (
            "maximum_step_change_hz",
            "maximum_step_relative_change",
        ):
            _finite_optional(getattr(self, name), name, nonnegative=True)
        for name in (
            "total_signed_change_hz",
            "total_absolute_change_hz",
            "total_relative_symmetric_change",
            "trajectory_mean_hz",
            "trajectory_standard_deviation_hz",
            "trajectory_rmse_from_mean_hz",
            "trajectory_relative_rmse",
        ):
            value = getattr(self, name)
            if not isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if self.trajectory_mean_hz <= 0.0:
            raise ValueError("trajectory_mean_hz must be positive.")
        for name in ("up_step_count", "down_step_count", "preserved_step_count", "indeterminate_step_count"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative.")
        _reason_tuple(self.reasons, "frequency reasons")
        _strings(self.diagnostics, "frequency diagnostics")


@dataclass(frozen=True, slots=True)
class ModalHypothesisAssociationEvidence:
    match_ids: tuple[str, ...]
    match_costs: tuple[float, ...]
    mean_match_cost: float | None
    maximum_match_cost: float | None
    minimum_match_cost: float | None
    ambiguous_match_count: int
    ambiguous_match_fraction: float | None
    near_threshold_match_count: int
    near_threshold_match_fraction: float | None
    match_margins: tuple[float | None, ...]
    minimum_match_margin: float | None
    passes: bool
    reasons: tuple[ModalHypothesisReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _strings(self.match_ids, "association match IDs", allow_empty=True)
        if len(self.match_ids) != len(self.match_costs) or len(self.match_ids) != len(self.match_margins):
            raise ValueError("association vectors must align with match_ids.")
        if any(not isfinite(value) or value < 0.0 for value in self.match_costs):
            raise ValueError("match costs must be finite and non-negative.")
        for name in ("mean_match_cost", "maximum_match_cost", "minimum_match_cost", "minimum_match_margin"):
            _finite_optional(getattr(self, name), name, nonnegative=True)
        for value in self.match_margins:
            _finite_optional(value, "match_margins item", nonnegative=True)
        for name in ("ambiguous_match_count", "near_threshold_match_count"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative.")
        _fraction(self.ambiguous_match_fraction, "ambiguous_match_fraction")
        _fraction(self.near_threshold_match_fraction, "near_threshold_match_fraction")
        _reason_tuple(self.reasons, "association reasons")
        _strings(self.diagnostics, "association diagnostics")


@dataclass(frozen=True, slots=True)
class ModalHypothesisTrackingEvidence:
    candidate_count: int
    coverage_values: tuple[float | None, ...]
    mean_coverage_fraction: float | None
    minimum_coverage_fraction: float | None
    ambiguous_assignment_fractions: tuple[float | None, ...]
    mean_ambiguous_assignment_fraction: float | None
    near_threshold_assignment_fractions: tuple[float | None, ...]
    mean_near_threshold_assignment_fraction: float | None
    assignment_margins: tuple[float | None, ...]
    minimum_assignment_margin: float | None
    frequency_fit_rmse_values_hz: tuple[float | None, ...]
    mean_frequency_fit_rmse_hz: float | None
    missing_value_counts: tuple[tuple[str, int], ...]
    passes: bool
    reasons: tuple[ModalHypothesisReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.candidate_count < 0:
            raise ValueError("candidate_count must not be negative.")
        vectors = (
            self.coverage_values,
            self.ambiguous_assignment_fractions,
            self.near_threshold_assignment_fractions,
            self.assignment_margins,
            self.frequency_fit_rmse_values_hz,
        )
        if any(len(vector) != self.candidate_count for vector in vectors):
            raise ValueError("tracking vectors must align with candidate_count.")
        for vector_name in (
            "coverage_values",
            "ambiguous_assignment_fractions",
            "near_threshold_assignment_fractions",
        ):
            for value in getattr(self, vector_name):
                _fraction(value, vector_name)
        for vector_name in ("assignment_margins", "frequency_fit_rmse_values_hz"):
            for value in getattr(self, vector_name):
                _finite_optional(value, vector_name, nonnegative=True)
        for name in (
            "mean_coverage_fraction",
            "minimum_coverage_fraction",
            "mean_ambiguous_assignment_fraction",
            "mean_near_threshold_assignment_fraction",
        ):
            _fraction(getattr(self, name), name)
        for name in ("minimum_assignment_margin", "mean_frequency_fit_rmse_hz"):
            _finite_optional(getattr(self, name), name, nonnegative=True)
        _missing_counts(self.missing_value_counts)
        _reason_tuple(self.reasons, "tracking reasons")
        _strings(self.diagnostics, "tracking diagnostics")


@dataclass(frozen=True, slots=True)
class ModalHypothesisDecayEvidence:
    tau_values_s: tuple[float, ...]
    available_tau_count: int
    missing_tau_count: int
    log_tau_values: tuple[float, ...]
    minimum_tau_s: float | None
    maximum_tau_s: float | None
    log_tau_range: float | None
    log_tau_mean: float | None
    log_tau_standard_deviation: float | None
    fit_quality_values: tuple[float, ...]
    mean_fit_quality: float | None
    passes: bool
    reasons: tuple[ModalHypothesisReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.available_tau_count != len(self.tau_values_s):
            raise ValueError("available_tau_count must match tau_values_s.")
        if self.missing_tau_count < 0:
            raise ValueError("missing_tau_count must not be negative.")
        if any(not isfinite(value) or value <= 0.0 for value in self.tau_values_s):
            raise ValueError("tau_values_s must be finite and positive.")
        if len(self.log_tau_values) != len(self.tau_values_s):
            raise ValueError("log_tau_values must match tau_values_s.")
        if any(not isfinite(value) for value in self.log_tau_values):
            raise ValueError("log_tau_values must be finite.")
        for name in (
            "minimum_tau_s",
            "maximum_tau_s",
            "log_tau_range",
            "log_tau_standard_deviation",
            "mean_fit_quality",
        ):
            _finite_optional(getattr(self, name), name, nonnegative=True)
        _finite_optional(self.log_tau_mean, "log_tau_mean")
        if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in self.fit_quality_values):
            raise ValueError("fit_quality_values must be finite fractions.")
        _reason_tuple(self.reasons, "decay reasons")
        _strings(self.diagnostics, "decay diagnostics")


@dataclass(frozen=True, slots=True)
class ModalHypothesisImpactEvidence:
    candidate_count: int
    available_evidence_count: int
    missing_evidence_count: int
    impact_supported_count: int
    impact_supported_fraction: float | None
    classifications: tuple[str | None, ...]
    background_persistent_count: int
    passes: bool
    reasons: tuple[ModalHypothesisReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.candidate_count < 0:
            raise ValueError("candidate_count must not be negative.")
        if len(self.classifications) != self.candidate_count:
            raise ValueError("classifications must align with candidate_count.")
        if self.available_evidence_count + self.missing_evidence_count != self.candidate_count:
            raise ValueError("impact evidence counts must partition candidates.")
        if self.impact_supported_count < 0 or self.background_persistent_count < 0:
            raise ValueError("impact counts must not be negative.")
        if self.impact_supported_count > self.candidate_count:
            raise ValueError("impact_supported_count exceeds candidate_count.")
        _fraction(self.impact_supported_fraction, "impact_supported_fraction")
        for item in self.classifications:
            if item is not None:
                _text(item, "impact classification")
        _reason_tuple(self.reasons, "impact reasons")
        _strings(self.diagnostics, "impact diagnostics")


@dataclass(frozen=True, slots=True)
class ModalHypothesisStructuralContext:
    possible_split_contexts: tuple[object, ...]
    possible_merge_contexts: tuple[object, ...]
    split_context_count: int
    merge_context_count: int
    contains_possible_split_context: bool
    contains_possible_merge_context: bool
    passes: bool
    requires_reservation: bool
    reasons: tuple[ModalHypothesisReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.split_context_count != len(self.possible_split_contexts):
            raise ValueError("split_context_count must match contexts.")
        if self.merge_context_count != len(self.possible_merge_contexts):
            raise ValueError("merge_context_count must match contexts.")
        if self.contains_possible_split_context != bool(self.possible_split_contexts):
            raise ValueError("contains_possible_split_context is incoherent.")
        if self.contains_possible_merge_context != bool(self.possible_merge_contexts):
            raise ValueError("contains_possible_merge_context is incoherent.")
        if self.requires_reservation and not self.passes:
            raise ValueError("rejected structural context cannot also require reservation.")
        _reason_tuple(self.reasons, "structural reasons")
        _strings(self.diagnostics, "structural diagnostics")


@dataclass(frozen=True, slots=True)
class ModalHypothesisScoreComponent:
    name: str
    value: float | None
    weight: float
    weighted_value: float | None
    available: bool
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.name, "score component name")
        if not isfinite(self.weight) or self.weight < 0.0:
            raise ValueError("score component weight must be finite and non-negative.")
        if self.available:
            if self.value is None or not isfinite(self.value) or not 0.0 <= self.value <= 1.0:
                raise ValueError("available score component value must be in [0, 1].")
            expected = self.value * self.weight
            if self.weighted_value is None or not isclose(self.weighted_value, expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("weighted_value must equal value * weight.")
        elif self.value is not None or self.weighted_value is not None:
            raise ValueError("unavailable score components must not carry values.")
        _strings(self.diagnostics, "score component diagnostics", allow_empty=True)


@dataclass(frozen=True, slots=True)
class ModalHypothesisScore:
    coverage_component: float | None
    frequency_component: float | None
    association_component: float | None
    tracking_component: float | None
    decay_component: float | None
    impact_component: float | None
    structural_penalty: float
    missing_evidence_penalty: float
    raw_score: float
    normalized_score: float
    passes_acceptance_threshold: bool
    passes_reservation_threshold: bool
    components: tuple[ModalHypothesisScoreComponent, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "coverage_component",
            "frequency_component",
            "association_component",
            "tracking_component",
            "decay_component",
            "impact_component",
        ):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or not 0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1] when present.")
        for name in ("structural_penalty", "missing_evidence_penalty", "raw_score", "normalized_score"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.normalized_score > 1.0:
            raise ValueError("normalized_score must not exceed 1.")
        names = tuple(component.name for component in self.components)
        if len(names) != len(set(names)):
            raise ValueError("score components must have unique names.")
        _strings(self.diagnostics, "score diagnostics", allow_empty=True)


@dataclass(frozen=True, slots=True)
class ModalHypothesis:
    hypothesis_id: str
    source_chain_id: str | None
    chain: CrossConditionCandidateChain | None
    status: ModalHypothesisStatus
    score: ModalHypothesisScore
    coverage_evidence: ModalHypothesisCoverageEvidence
    frequency_evidence: ModalHypothesisFrequencyEvidence
    association_evidence: ModalHypothesisAssociationEvidence
    tracking_evidence: ModalHypothesisTrackingEvidence
    decay_evidence: ModalHypothesisDecayEvidence
    impact_evidence: ModalHypothesisImpactEvidence
    structural_context: ModalHypothesisStructuralContext
    supporting_reasons: tuple[ModalHypothesisReason, ...]
    reservation_reasons: tuple[ModalHypothesisReason, ...]
    rejection_reasons: tuple[ModalHypothesisReason, ...]
    missing_evidence_reasons: tuple[ModalHypothesisReason, ...]
    accepted: bool
    requires_review: bool
    valid: bool
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.hypothesis_id, "hypothesis_id")
        if self.source_chain_id is not None:
            _text(self.source_chain_id, "source_chain_id")
        if not isinstance(self.status, ModalHypothesisStatus):
            raise ValueError("status must be a ModalHypothesisStatus.")
        accepted_status = self.status in {
            ModalHypothesisStatus.ACCEPTED,
            ModalHypothesisStatus.ACCEPTED_WITH_RESERVATIONS,
        }
        if self.accepted != accepted_status:
            raise ValueError("accepted must mirror accepted hypothesis states.")
        if self.valid != (self.status is not ModalHypothesisStatus.INVALID_INPUT):
            raise ValueError("valid must be false only for invalid_input.")
        if self.requires_review != (self.status is not ModalHypothesisStatus.ACCEPTED):
            raise ValueError("requires_review must be false only for clean accepted hypotheses.")
        for name in (
            "supporting_reasons",
            "reservation_reasons",
            "rejection_reasons",
            "missing_evidence_reasons",
        ):
            _reason_tuple(getattr(self, name), name)
        if self.status is ModalHypothesisStatus.ACCEPTED and (
            self.reservation_reasons or self.rejection_reasons or self.missing_evidence_reasons
        ):
            raise ValueError("accepted hypotheses must not carry reservations, rejections or missing evidence.")
        if self.accepted and self.rejection_reasons:
            raise ValueError("accepted hypotheses cannot carry rejection reasons.")
        _strings(self.diagnostics, "hypothesis diagnostics", allow_empty=True)


@dataclass(frozen=True, slots=True)
class ModalHypothesisResult:
    sequence: tuple[str, ...]
    hypotheses: tuple[ModalHypothesis, ...]
    accepted_hypotheses: tuple[ModalHypothesis, ...]
    accepted_with_reservations_hypotheses: tuple[ModalHypothesis, ...]
    inconclusive_hypotheses: tuple[ModalHypothesis, ...]
    rejected_hypotheses: tuple[ModalHypothesis, ...]
    insufficient_evidence_hypotheses: tuple[ModalHypothesis, ...]
    hypothesis_count: int
    accepted_count: int
    accepted_with_reservations_count: int
    inconclusive_count: int
    rejected_count: int
    insufficient_evidence_count: int
    source_chain_count: int
    settings: ModalHypothesisSettings
    valid: bool
    failure_reason: str | None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_sequence(self.sequence, allow_single=True)
        if not isinstance(self.settings, ModalHypothesisSettings):
            raise ValueError("settings must be ModalHypothesisSettings.")
        if self.hypotheses != tuple(sorted(self.hypotheses, key=_hypothesis_sort_key)):
            raise ValueError("hypotheses must be in deterministic order.")
        ids = tuple(item.hypothesis_id for item in self.hypotheses)
        if len(ids) != len(set(ids)):
            raise ValueError("hypothesis IDs must be unique.")
        chain_ids = tuple(
            item.source_chain_id for item in self.hypotheses
            if item.source_chain_id is not None
        )
        if len(chain_ids) != len(set(chain_ids)):
            raise ValueError("source chains must not appear in more than one hypothesis.")
        subsets = {
            ModalHypothesisStatus.ACCEPTED: self.accepted_hypotheses,
            ModalHypothesisStatus.ACCEPTED_WITH_RESERVATIONS: self.accepted_with_reservations_hypotheses,
            ModalHypothesisStatus.INCONCLUSIVE: self.inconclusive_hypotheses,
            ModalHypothesisStatus.REJECTED: self.rejected_hypotheses,
            ModalHypothesisStatus.INSUFFICIENT_EVIDENCE: self.insufficient_evidence_hypotheses,
        }
        for status, subset in subsets.items():
            expected = tuple(item for item in self.hypotheses if item.status is status)
            if subset != expected:
                raise ValueError("status subsets must mirror hypotheses.")
        counts = {
            "hypothesis_count": len(self.hypotheses),
            "accepted_count": len(self.accepted_hypotheses),
            "accepted_with_reservations_count": len(self.accepted_with_reservations_hypotheses),
            "inconclusive_count": len(self.inconclusive_hypotheses),
            "rejected_count": len(self.rejected_hypotheses),
            "insufficient_evidence_count": len(self.insufficient_evidence_hypotheses),
        }
        for name, expected in counts.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} is incoherent with hypotheses.")
        if self.source_chain_count != self.hypothesis_count:
            raise ValueError("each source chain must produce exactly one hypothesis.")
        if self.valid and self.failure_reason is not None:
            raise ValueError("valid result must not have failure_reason.")
        if not self.valid and not self.failure_reason:
            raise ValueError("invalid result requires failure_reason.")
        _strings(self.diagnostics, "hypothesis result diagnostics", allow_empty=True)


def evaluate_modal_hypothesis_frequency_evidence(
    chain: CrossConditionCandidateChain,
    settings: ModalHypothesisSettings | None = None,
) -> ModalHypothesisFrequencyEvidence:
    """Evaluate frequency continuity without classifying physical behavior."""

    cfg = settings or ModalHypothesisSettings()
    frequencies = tuple(chain.frequency_trajectory_hz)
    labels = tuple(node.dynamic_label for node in chain.nodes)
    signed = tuple(chain.frequency_step_changes_hz)
    absolute = tuple(abs(value) for value in signed)
    signed_relative = tuple(chain.frequency_step_changes_relative)
    absolute_relative = tuple(abs(value) for value in signed_relative)
    maximum_step = max(absolute) if absolute else None
    maximum_relative = max(absolute_relative) if absolute_relative else None
    total_signed = frequencies[-1] - frequencies[0]
    total_absolute = sum(absolute)
    total_relative = abs(total_signed / (0.5 * (frequencies[-1] + frequencies[0]))) if len(frequencies) > 1 else 0.0
    trajectory_mean = sum(frequencies) / len(frequencies)
    rmse = sqrt(sum((value - trajectory_mean) ** 2 for value in frequencies) / len(frequencies))
    stdev = rmse
    relative_rmse = rmse / trajectory_mean
    reasons: list[ModalHypothesisReason] = []
    diagnostics = [
        "frequency_continuity_is_operational_not_physical_identity",
        "no_hardening_or_softening_classification",
        "no_linearity_or_nonlinearity_classification",
        "no_monotonicity_requirement",
    ]
    passes = len(frequencies) >= 2
    if not passes:
        reasons.append(ModalHypothesisReason.INSUFFICIENT_EVIDENCE)
    if _exceeds(maximum_step, cfg.maximum_step_absolute_frequency_change_hz):
        passes = False
        reasons.append(ModalHypothesisReason.FREQUENCY_DISCONTINUITY)
    if _exceeds(maximum_relative, cfg.maximum_step_relative_frequency_change):
        passes = False
        reasons.append(ModalHypothesisReason.FREQUENCY_DISCONTINUITY)
    if _exceeds(total_absolute, cfg.maximum_total_absolute_frequency_change_hz):
        passes = False
        reasons.append(ModalHypothesisReason.EXCESSIVE_FREQUENCY_VARIATION)
    if _exceeds(total_relative, cfg.maximum_total_relative_frequency_change):
        passes = False
        reasons.append(ModalHypothesisReason.EXCESSIVE_FREQUENCY_VARIATION)
    if _exceeds(rmse, cfg.maximum_frequency_trajectory_rmse_hz):
        passes = False
        reasons.append(ModalHypothesisReason.EXCESSIVE_FREQUENCY_VARIATION)
    if _exceeds(relative_rmse, cfg.maximum_frequency_trajectory_relative_rmse):
        passes = False
        reasons.append(ModalHypothesisReason.EXCESSIVE_FREQUENCY_VARIATION)
    if passes:
        reasons.append(ModalHypothesisReason.SUFFICIENT_FREQUENCY_CONTINUITY)
    return ModalHypothesisFrequencyEvidence(
        frequencies_hz=frequencies,
        condition_labels=labels,
        signed_step_changes_hz=signed,
        absolute_step_changes_hz=absolute,
        signed_step_relative_changes=signed_relative,
        absolute_step_relative_changes=absolute_relative,
        maximum_step_change_hz=maximum_step,
        maximum_step_relative_change=maximum_relative,
        total_signed_change_hz=total_signed,
        total_absolute_change_hz=total_absolute,
        total_relative_symmetric_change=total_relative,
        trajectory_mean_hz=trajectory_mean,
        trajectory_standard_deviation_hz=stdev,
        trajectory_rmse_from_mean_hz=rmse,
        trajectory_relative_rmse=relative_rmse,
        up_step_count=chain.upward_step_count,
        down_step_count=chain.downward_step_count,
        preserved_step_count=chain.preserved_step_count,
        indeterminate_step_count=chain.indeterminate_step_count,
        passes=passes,
        reasons=_ordered_reasons(reasons),
        diagnostics=tuple(diagnostics),
    )


def evaluate_modal_hypothesis_decay_evidence(
    chain: CrossConditionCandidateChain,
    settings: ModalHypothesisSettings | None = None,
) -> ModalHypothesisDecayEvidence:
    """Evaluate tau availability and log-domain consistency from references."""

    cfg = settings or ModalHypothesisSettings()
    tau_values: list[float] = []
    fit_quality_values: list[float] = []
    invalid_tau_count = 0
    for node in chain.nodes:
        tau = node.candidate_ref.amplitude_tau_s
        if tau is None:
            continue
        if isfinite(tau) and tau > 0.0:
            tau_values.append(tau)
        else:
            invalid_tau_count += 1
        quality = node.candidate_ref.amplitude_fit_r_squared
        if quality is not None and isfinite(quality) and 0.0 <= quality <= 1.0:
            fit_quality_values.append(quality)
    missing = chain.condition_count - len(tau_values)
    logs = tuple(log(value) for value in tau_values)
    log_range = max(logs) - min(logs) if len(logs) >= 2 else None
    log_mean = sum(logs) / len(logs) if logs else None
    log_std = (
        sqrt(sum((value - log_mean) ** 2 for value in logs) / len(logs))
        if logs and log_mean is not None
        else None
    )
    reasons: list[ModalHypothesisReason] = []
    diagnostics = [
        "decay_tau_consistency_is_operational_not_physical_identity",
        "no_q_factor_or_bandwidth_derived",
        "missing_tau_values_preserved",
    ]
    if invalid_tau_count:
        diagnostics.append(f"invalid_tau_values_ignored:{invalid_tau_count}")
    passes = True
    enough = len(tau_values) >= cfg.minimum_decay_value_count
    if cfg.require_decay_evidence and not enough:
        passes = False
        reasons.append(ModalHypothesisReason.MISSING_REQUIRED_DECAY)
    elif not cfg.allow_missing_decay and missing:
        passes = False
        reasons.append(ModalHypothesisReason.MISSING_REQUIRED_DECAY)
    elif missing and cfg.missing_decay_evidence_policy != "allow":
        reasons.append(ModalHypothesisReason.INSUFFICIENT_EVIDENCE)
    if enough:
        if _exceeds(log_range, cfg.maximum_log_tau_range):
            passes = False
            reasons.append(ModalHypothesisReason.INCONSISTENT_DECAY)
        if _exceeds(log_std, cfg.maximum_log_tau_standard_deviation):
            passes = False
            reasons.append(ModalHypothesisReason.INCONSISTENT_DECAY)
        if passes:
            reasons.append(ModalHypothesisReason.SUFFICIENT_DECAY_CONSISTENCY)
    elif not cfg.require_decay_evidence and cfg.allow_missing_decay:
        diagnostics.append("decay_consistency_not_applicable")
    return ModalHypothesisDecayEvidence(
        tau_values_s=tuple(tau_values),
        available_tau_count=len(tau_values),
        missing_tau_count=missing,
        log_tau_values=logs,
        minimum_tau_s=min(tau_values) if tau_values else None,
        maximum_tau_s=max(tau_values) if tau_values else None,
        log_tau_range=log_range,
        log_tau_mean=log_mean,
        log_tau_standard_deviation=log_std,
        fit_quality_values=tuple(fit_quality_values),
        mean_fit_quality=(sum(fit_quality_values) / len(fit_quality_values) if fit_quality_values else None),
        passes=passes,
        reasons=_ordered_reasons(reasons),
        diagnostics=tuple(diagnostics),
    )


def evaluate_modal_hypothesis(
    chain: CrossConditionCandidateChain | object,
    sequence: Iterable[str] | None = None,
    settings: ModalHypothesisSettings | None = None,
) -> ModalHypothesis:
    """Evaluate one ready candidate chain as an operational modal hypothesis."""

    cfg = settings or ModalHypothesisSettings()
    if not isinstance(chain, CrossConditionCandidateChain):
        seq = tuple(sequence) if sequence is not None else ("pp",)
        if not seq:
            seq = ("pp",)
        return _invalid_hypothesis(seq, cfg, "input_is_not_cross_condition_candidate_chain")
    seq = tuple(sequence) if sequence is not None else tuple(node.dynamic_label for node in chain.nodes)
    try:
        _validate_sequence(seq, allow_single=True)
    except ValueError:
        return _invalid_hypothesis(("pp",), cfg, "invalid_requested_sequence")
    if any(node.dynamic_label not in seq for node in chain.nodes):
        return _invalid_hypothesis(seq, cfg, "chain_nodes_outside_requested_sequence")

    coverage = _coverage_evidence(chain, seq, cfg)
    frequency = evaluate_modal_hypothesis_frequency_evidence(chain, cfg)
    association = _association_evidence(chain, cfg)
    tracking = _tracking_evidence(chain, cfg)
    decay = evaluate_modal_hypothesis_decay_evidence(chain, cfg)
    impact = _impact_evidence(chain, cfg)
    structural = _structural_context(chain, cfg)

    supporting: list[ModalHypothesisReason] = []
    reservations: list[ModalHypothesisReason] = []
    rejections: list[ModalHypothesisReason] = []
    missing: list[ModalHypothesisReason] = []
    insufficient = False
    mandatory_failed = False

    _extend_supporting(supporting, coverage.reasons)
    _extend_supporting(supporting, frequency.reasons)
    _extend_supporting(supporting, tracking.reasons)
    _extend_supporting(supporting, decay.reasons)
    _extend_supporting(supporting, impact.reasons)

    if any(not node.candidate_ref.accepted for node in chain.nodes):
        rejections.append(ModalHypothesisReason.REJECTED_CANDIDATE_PRESENT)
    if coverage.singleton and not cfg.allow_singleton_chains:
        insufficient = True
        missing.extend((ModalHypothesisReason.SINGLETON_CHAIN, ModalHypothesisReason.TOO_FEW_MATCHES))
    elif not coverage.passes:
        if cfg.require_complete_chain or (coverage.partial and not cfg.allow_partial_chains):
            mandatory_failed = True
            rejections.extend(
                reason for reason in coverage.reasons
                if reason in {
                    ModalHypothesisReason.TOO_FEW_CONDITIONS,
                    ModalHypothesisReason.TOO_FEW_MATCHES,
                }
            )
        else:
            insufficient = True
            missing.extend(
                reason for reason in coverage.reasons
                if reason in {
                    ModalHypothesisReason.TOO_FEW_CONDITIONS,
                    ModalHypothesisReason.TOO_FEW_MATCHES,
                    ModalHypothesisReason.INSUFFICIENT_EVIDENCE,
                }
            )
    if coverage.partial and coverage.passes:
        reservations.append(ModalHypothesisReason.PARTIAL_BUT_SUPPORTED_CHAIN)
    if frequency.reasons and not frequency.passes:
        rejections.extend(
            reason for reason in frequency.reasons
            if reason is not ModalHypothesisReason.INSUFFICIENT_EVIDENCE
        )
    if association.reasons and not association.passes:
        if ModalHypothesisReason.TOO_FEW_MATCHES in association.reasons:
            insufficient = True
            missing.append(ModalHypothesisReason.TOO_FEW_MATCHES)
        rejections.extend(
            reason for reason in association.reasons
            if reason is not ModalHypothesisReason.TOO_FEW_MATCHES
        )
    if association.ambiguous_match_count:
        reservations.append(ModalHypothesisReason.EXCESSIVE_AMBIGUITY)
    if association.near_threshold_match_count:
        reservations.append(ModalHypothesisReason.EXCESSIVE_NEAR_THRESHOLD_FRACTION)
    _apply_tracking_policy(tracking, cfg, reservations, rejections, missing)
    _apply_decay_policy(decay, cfg, reservations, rejections, missing)
    _apply_impact_policy(impact, cfg, reservations, rejections, missing)
    if (
        cfg.missing_tracking_evidence_policy == "insufficient"
        and ModalHypothesisReason.INSUFFICIENT_EVIDENCE in tracking.reasons
    ):
        insufficient = True
    if (
        cfg.missing_decay_evidence_policy == "insufficient"
        and ModalHypothesisReason.INSUFFICIENT_EVIDENCE in decay.reasons
    ):
        insufficient = True
    if (
        cfg.missing_impact_evidence_policy == "insufficient"
        and ModalHypothesisReason.INSUFFICIENT_EVIDENCE in impact.reasons
    ):
        insufficient = True
    if not structural.passes:
        rejections.extend(structural.reasons)
    elif structural.requires_reservation:
        reservations.extend(structural.reasons)
    if missing:
        insufficient = any(
            reason in {
                ModalHypothesisReason.MISSING_REQUIRED_DECAY,
                ModalHypothesisReason.MISSING_REQUIRED_IMPACT_EVIDENCE,
                ModalHypothesisReason.TOO_FEW_MATCHES,
                ModalHypothesisReason.SINGLETON_CHAIN,
            }
            for reason in missing
        ) or insufficient

    score = compute_modal_hypothesis_score(
        coverage,
        frequency,
        association,
        tracking,
        decay,
        impact,
        structural,
        _ordered_reasons(missing),
        cfg,
    )
    rejections_tuple = _ordered_reasons(rejections)
    reservations_tuple = _ordered_reasons(reservations)
    missing_tuple = _ordered_reasons(missing)
    supporting_tuple = _ordered_reasons(supporting)

    if mandatory_failed:
        status = ModalHypothesisStatus.REJECTED
    elif insufficient or any(
        reason in {
            ModalHypothesisReason.MISSING_REQUIRED_DECAY,
            ModalHypothesisReason.MISSING_REQUIRED_IMPACT_EVIDENCE,
        }
        for reason in missing_tuple
    ):
        status = ModalHypothesisStatus.INSUFFICIENT_EVIDENCE
    elif len(rejections_tuple) > cfg.maximum_rejection_count:
        status = ModalHypothesisStatus.REJECTED
    elif rejections_tuple:
        status = (
            ModalHypothesisStatus.ACCEPTED_WITH_RESERVATIONS
            if cfg.allow_accepted_with_reservations and score.passes_reservation_threshold
            else ModalHypothesisStatus.INCONCLUSIVE
        )
    elif reservations_tuple or missing_tuple:
        status = (
            ModalHypothesisStatus.ACCEPTED_WITH_RESERVATIONS
            if cfg.allow_accepted_with_reservations and score.passes_reservation_threshold
            else ModalHypothesisStatus.INCONCLUSIVE
        )
    elif score.passes_acceptance_threshold:
        status = ModalHypothesisStatus.ACCEPTED
    else:
        status = ModalHypothesisStatus.INCONCLUSIVE

    diagnostics = [
        "modal_hypothesis_is_operational_not_physical_mode",
        "accepted_hypothesis_is_not_modal_identity_proof",
        "built_from_existing_candidate_chain_only",
        "no_audio_fft_stft_tracking_or_association_recomputed",
        "no_non_adjacent_association_created",
        "no_gap_closure_performed",
        "no_split_or_merge_resolved",
        "no_modal_mode_created",
    ]
    diagnostics.extend(sorted(chain.diagnostics))
    hypothesis_id = _hypothesis_id(seq, chain)
    return ModalHypothesis(
        hypothesis_id=hypothesis_id,
        source_chain_id=chain.chain_id,
        chain=chain,
        status=status,
        score=score,
        coverage_evidence=coverage,
        frequency_evidence=frequency,
        association_evidence=association,
        tracking_evidence=tracking,
        decay_evidence=decay,
        impact_evidence=impact,
        structural_context=structural,
        supporting_reasons=supporting_tuple,
        reservation_reasons=reservations_tuple,
        rejection_reasons=rejections_tuple,
        missing_evidence_reasons=missing_tuple,
        accepted=status in {
            ModalHypothesisStatus.ACCEPTED,
            ModalHypothesisStatus.ACCEPTED_WITH_RESERVATIONS,
        },
        requires_review=status is not ModalHypothesisStatus.ACCEPTED,
        valid=True,
        diagnostics=tuple(dict.fromkeys(diagnostics)),
    )


def compute_modal_hypothesis_score(
    coverage: ModalHypothesisCoverageEvidence,
    frequency: ModalHypothesisFrequencyEvidence,
    association: ModalHypothesisAssociationEvidence,
    tracking: ModalHypothesisTrackingEvidence,
    decay: ModalHypothesisDecayEvidence,
    impact: ModalHypothesisImpactEvidence,
    structural: ModalHypothesisStructuralContext,
    missing_evidence_reasons: Iterable[ModalHypothesisReason],
    settings: ModalHypothesisSettings | None = None,
) -> ModalHypothesisScore:
    """Compute an auditable normalized score in [0, 1]."""

    cfg = settings or ModalHypothesisSettings()
    coverage_value = coverage.condition_coverage_fraction
    frequency_value = _frequency_score_value(frequency, cfg)
    association_value = _association_score_value(association, cfg)
    tracking_value = _tracking_score_value(tracking, cfg)
    decay_value = _decay_score_value(decay, cfg)
    impact_value = _impact_score_value(impact)
    values = (
        ("coverage", coverage_value, cfg.coverage_weight),
        ("frequency", frequency_value, cfg.frequency_continuity_weight),
        ("association", association_value, cfg.association_quality_weight),
        ("tracking", tracking_value, cfg.tracking_quality_weight),
        ("decay", decay_value, cfg.decay_consistency_weight),
        ("impact", impact_value, cfg.impact_evidence_weight),
    )
    components = tuple(
        _score_component(name, value, weight)
        for name, value, weight in values
    )
    numerator = sum(
        component.weighted_value or 0.0
        for component in components
        if component.available and component.weight > 0.0
    )
    denominator = sum(
        component.weight
        for component in components
        if component.available and component.weight > 0.0
    )
    weighted_average = numerator / denominator if denominator else 0.0
    structural_penalty = (
        cfg.structural_context_penalty
        * float(structural.requires_reservation or not structural.passes)
    )
    missing_penalty = cfg.missing_evidence_penalty * len(set(missing_evidence_reasons))
    raw = max(0.0, weighted_average - structural_penalty - missing_penalty)
    normalized = min(1.0, raw)
    diagnostics = [
        "score_is_audit_metric_not_decision_override",
        "mandatory_gates_evaluated_before_score",
        "frequency_weight_is_dominant_by_default",
    ]
    return ModalHypothesisScore(
        coverage_component=coverage_value,
        frequency_component=frequency_value,
        association_component=association_value,
        tracking_component=tracking_value,
        decay_component=decay_value,
        impact_component=impact_value,
        structural_penalty=structural_penalty,
        missing_evidence_penalty=missing_penalty,
        raw_score=raw,
        normalized_score=normalized,
        passes_acceptance_threshold=_inclusive_ge(normalized, cfg.minimum_acceptance_score),
        passes_reservation_threshold=_inclusive_ge(normalized, cfg.minimum_reservation_score),
        components=components,
        diagnostics=tuple(diagnostics),
    )


def build_modal_hypotheses(
    chains: CrossConditionCandidateChainResult | Iterable[CrossConditionCandidateChain],
    settings: ModalHypothesisSettings | None = None,
    sequence: Iterable[str] | None = None,
) -> ModalHypothesisResult:
    """Evaluate every chain exactly once in deterministic order."""

    cfg = settings or ModalHypothesisSettings()
    if isinstance(chains, CrossConditionCandidateChainResult):
        chain_tuple = chains.chains
        seq = tuple(sequence) if sequence is not None else chains.sequence.dynamic_labels
    else:
        chain_tuple = tuple(chains)
        if sequence is None:
            labels = tuple(
                label
                for chain in chain_tuple
                if isinstance(chain, CrossConditionCandidateChain)
                for label in (node.dynamic_label for node in chain.nodes)
            )
            seq = _minimal_sequence(labels) if labels else ("pp",)
        else:
            seq = tuple(sequence)
    _validate_sequence(seq, allow_single=True)
    ordered = tuple(sorted(chain_tuple, key=_chain_sort_key_for_hypothesis))
    hypotheses = tuple(
        evaluate_modal_hypothesis(chain, seq, cfg)
        for chain in ordered
    )
    hypotheses = tuple(sorted(hypotheses, key=_hypothesis_sort_key))
    accepted = tuple(
        item for item in hypotheses
        if item.status is ModalHypothesisStatus.ACCEPTED
    )
    reservations = tuple(
        item for item in hypotheses
        if item.status is ModalHypothesisStatus.ACCEPTED_WITH_RESERVATIONS
    )
    inconclusive = tuple(
        item for item in hypotheses
        if item.status is ModalHypothesisStatus.INCONCLUSIVE
    )
    rejected = tuple(
        item for item in hypotheses
        if item.status is ModalHypothesisStatus.REJECTED
    )
    insufficient = tuple(
        item for item in hypotheses
        if item.status is ModalHypothesisStatus.INSUFFICIENT_EVIDENCE
    )
    diagnostics = (
        "modal_hypotheses_are_operational_not_physical_modes",
        "one_hypothesis_per_source_chain",
        "deterministic_order_and_content_based_ids",
        "no_chain_was_dropped",
        "no_non_adjacent_association_created",
        "no_gap_closure_performed",
        "split_and_merge_contexts_preserved_without_resolution",
    )
    return ModalHypothesisResult(
        sequence=seq,
        hypotheses=hypotheses,
        accepted_hypotheses=accepted,
        accepted_with_reservations_hypotheses=reservations,
        inconclusive_hypotheses=inconclusive,
        rejected_hypotheses=rejected,
        insufficient_evidence_hypotheses=insufficient,
        hypothesis_count=len(hypotheses),
        accepted_count=len(accepted),
        accepted_with_reservations_count=len(reservations),
        inconclusive_count=len(inconclusive),
        rejected_count=len(rejected),
        insufficient_evidence_count=len(insufficient),
        source_chain_count=len(chain_tuple),
        settings=cfg,
        valid=True,
        failure_reason=None,
        diagnostics=diagnostics,
    )


def summarize_modal_hypotheses(result: ModalHypothesisResult) -> dict[str, object]:
    """Return a compact deterministic summary for audit reports."""

    if not isinstance(result, ModalHypothesisResult):
        raise ValueError("result must be a ModalHypothesisResult.")
    return {
        "sequence": result.sequence,
        "hypothesis_count": result.hypothesis_count,
        "accepted_count": result.accepted_count,
        "accepted_with_reservations_count": result.accepted_with_reservations_count,
        "inconclusive_count": result.inconclusive_count,
        "rejected_count": result.rejected_count,
        "insufficient_evidence_count": result.insufficient_evidence_count,
        "source_chain_count": result.source_chain_count,
        "hypothesis_ids": tuple(item.hypothesis_id for item in result.hypotheses),
        "source_chain_ids": tuple(item.source_chain_id for item in result.hypotheses),
        "statuses": tuple(item.status.value for item in result.hypotheses),
        "normalized_scores": tuple(item.score.normalized_score for item in result.hypotheses),
        "diagnostics": result.diagnostics,
    }


def _coverage_evidence(
    chain: CrossConditionCandidateChain,
    sequence: tuple[str, ...],
    cfg: ModalHypothesisSettings,
) -> ModalHypothesisCoverageEvidence:
    labels = tuple(node.dynamic_label for node in chain.nodes)
    requested = len(sequence)
    observed = len(labels)
    coverage = observed / requested
    complete = labels == sequence
    singleton = chain.match_count == 0
    partial = not complete and not singleton
    reasons: list[ModalHypothesisReason] = []
    diagnostics = [
        "coverage_is_relative_to_requested_sequence",
        "candidate_chain_is_not_modal_hypothesis_by_itself",
    ]
    passes = True
    if cfg.minimum_condition_count is not None and observed < cfg.minimum_condition_count:
        passes = False
        reasons.append(ModalHypothesisReason.TOO_FEW_CONDITIONS)
    if cfg.minimum_match_count is not None and chain.match_count < cfg.minimum_match_count:
        passes = False
        reasons.append(ModalHypothesisReason.TOO_FEW_MATCHES)
    if cfg.minimum_condition_coverage_fraction is not None and coverage < cfg.minimum_condition_coverage_fraction and not isclose(coverage, cfg.minimum_condition_coverage_fraction, rel_tol=1e-12, abs_tol=1e-12):
        passes = False
        reasons.append(ModalHypothesisReason.TOO_FEW_CONDITIONS)
    if cfg.require_complete_chain and not complete:
        passes = False
        reasons.append(ModalHypothesisReason.TOO_FEW_CONDITIONS)
    if partial and not cfg.allow_partial_chains and not cfg.require_complete_chain:
        passes = False
        reasons.append(ModalHypothesisReason.TOO_FEW_CONDITIONS)
    if singleton:
        reasons.append(ModalHypothesisReason.SINGLETON_CHAIN)
        if not cfg.allow_singleton_chains:
            passes = False
    if passes:
        reasons.append(ModalHypothesisReason.SUFFICIENT_CROSS_CONDITION_PERSISTENCE)
        reasons.append(
            ModalHypothesisReason.COMPLETE_CHAIN
            if complete
            else ModalHypothesisReason.PARTIAL_BUT_SUPPORTED_CHAIN
        )
    return ModalHypothesisCoverageEvidence(
        requested_condition_count=requested,
        observed_condition_count=observed,
        condition_coverage_fraction=coverage,
        match_count=chain.match_count,
        complete_across_requested_sequence=complete,
        starts_as_emerging=chain.starts_as_emerging,
        ends_as_disappearing=chain.ends_as_disappearing,
        singleton=singleton,
        partial=partial,
        passes=passes,
        reasons=_ordered_reasons(reasons),
        diagnostics=tuple(diagnostics),
    )


def _association_evidence(
    chain: CrossConditionCandidateChain,
    cfg: ModalHypothesisSettings,
) -> ModalHypothesisAssociationEvidence:
    match_ids = tuple(
        node.outgoing_match_id
        for node in chain.nodes
        if node.outgoing_match_id is not None
    )
    costs = tuple(chain.association_costs)
    margins: list[float | None] = [None] * len(match_ids)
    for position, margin in zip(chain.ambiguous_match_positions, chain.ambiguous_assignment_margins, strict=False):
        if 0 <= position < len(margins):
            margins[position] = margin
    mean_cost = sum(costs) / len(costs) if costs else None
    max_cost = max(costs) if costs else None
    min_cost = min(costs) if costs else None
    ambiguous_fraction = chain.ambiguous_match_count / chain.match_count if chain.match_count else None
    near_fraction = chain.near_threshold_match_count / chain.match_count if chain.match_count else None
    reasons: list[ModalHypothesisReason] = []
    diagnostics = [
        "association_quality_uses_existing_adjacent_matches_only",
        "match_costs_preserved_per_original_match",
        "assignment_margins_preserved_when_available_from_chain",
    ]
    if chain.minimum_assignment_margin is not None and all(value is None for value in margins):
        diagnostics.append("chain_contract_exposes_minimum_margin_only")
    passes = bool(costs)
    if not passes:
        reasons.append(ModalHypothesisReason.TOO_FEW_MATCHES)
    if _exceeds(max_cost, cfg.maximum_match_cost):
        passes = False
        reasons.append(ModalHypothesisReason.EXCESSIVE_ASSOCIATION_COST)
    if _exceeds(mean_cost, cfg.maximum_mean_match_cost):
        passes = False
        reasons.append(ModalHypothesisReason.EXCESSIVE_ASSOCIATION_COST)
    if _exceeds(ambiguous_fraction, cfg.maximum_ambiguous_match_fraction):
        passes = False
        reasons.append(ModalHypothesisReason.EXCESSIVE_AMBIGUITY)
    if _exceeds(near_fraction, cfg.maximum_near_threshold_match_fraction):
        passes = False
        reasons.append(ModalHypothesisReason.EXCESSIVE_NEAR_THRESHOLD_FRACTION)
    if cfg.minimum_match_margin is not None:
        if chain.minimum_assignment_margin is None:
            passes = False
            reasons.append(ModalHypothesisReason.INSUFFICIENT_EVIDENCE)
        elif chain.minimum_assignment_margin < cfg.minimum_match_margin and not isclose(chain.minimum_assignment_margin, cfg.minimum_match_margin, rel_tol=1e-12, abs_tol=1e-12):
            passes = False
            reasons.append(ModalHypothesisReason.EXCESSIVE_AMBIGUITY)
    return ModalHypothesisAssociationEvidence(
        match_ids=match_ids,
        match_costs=costs,
        mean_match_cost=mean_cost,
        maximum_match_cost=max_cost,
        minimum_match_cost=min_cost,
        ambiguous_match_count=chain.ambiguous_match_count,
        ambiguous_match_fraction=ambiguous_fraction,
        near_threshold_match_count=chain.near_threshold_match_count,
        near_threshold_match_fraction=near_fraction,
        match_margins=tuple(margins),
        minimum_match_margin=chain.minimum_assignment_margin,
        passes=passes,
        reasons=_ordered_reasons(reasons),
        diagnostics=tuple(diagnostics),
    )


def _tracking_evidence(
    chain: CrossConditionCandidateChain,
    cfg: ModalHypothesisSettings,
) -> ModalHypothesisTrackingEvidence:
    refs = tuple(node.candidate_ref for node in chain.nodes)
    coverage_values = tuple(ref.coverage_fraction for ref in refs)
    ambiguous_values = tuple(ref.ambiguous_assignment_fraction for ref in refs)
    near_values = tuple(ref.near_threshold_assignment_fraction for ref in refs)
    margins = tuple(ref.minimum_assignment_margin for ref in refs)
    rmse_values = tuple(ref.frequency_fit_rmse_hz for ref in refs)
    missing_counts = (
        ("coverage_fraction", _missing_count(coverage_values)),
        ("ambiguous_assignment_fraction", _missing_count(ambiguous_values)),
        ("near_threshold_assignment_fraction", _missing_count(near_values)),
        ("assignment_margin", _missing_count(margins)),
        ("frequency_fit_rmse_hz", _missing_count(rmse_values)),
    )
    mean_coverage = _mean_present(coverage_values)
    min_coverage = _min_present(coverage_values)
    mean_ambiguous = _mean_present(ambiguous_values)
    mean_near = _mean_present(near_values)
    min_margin = _min_present(margins)
    mean_rmse = _mean_present(rmse_values)
    reasons: list[ModalHypothesisReason] = []
    diagnostics = [
        "tracking_evidence_reuses_candidate_reference_metrics",
        "no_tracking_recomputed",
        "missing_tracking_values_preserved",
    ]
    passes = True
    missing_required = False
    for metric, value, limit, operator in (
        ("coverage_fraction", mean_coverage, cfg.minimum_mean_coverage_fraction, ">="),
        ("ambiguous_assignment_fraction", mean_ambiguous, cfg.maximum_mean_ambiguous_assignment_fraction, "<="),
        ("near_threshold_assignment_fraction", mean_near, cfg.maximum_mean_near_threshold_assignment_fraction, "<="),
        ("assignment_margin", min_margin, cfg.minimum_mean_assignment_margin, ">="),
        ("frequency_fit_rmse_hz", mean_rmse, cfg.maximum_mean_frequency_fit_rmse_hz, "<="),
    ):
        if limit is None:
            continue
        if value is None:
            missing_required = True
            diagnostics.append(f"{metric}_not_available")
            continue
        if operator == ">=":
            failed = value < limit and not isclose(value, limit, rel_tol=1e-12, abs_tol=1e-12)
        else:
            failed = value > limit and not isclose(value, limit, rel_tol=1e-12, abs_tol=1e-12)
        if failed:
            passes = False
            reasons.append(ModalHypothesisReason.INSUFFICIENT_TRACKING_QUALITY)
    if missing_required:
        reasons.append(ModalHypothesisReason.INSUFFICIENT_EVIDENCE)
        if cfg.missing_tracking_evidence_policy in {"insufficient", "reject"}:
            passes = False
    if passes and not missing_required:
        reasons.append(ModalHypothesisReason.SUFFICIENT_TRACKING_QUALITY)
    return ModalHypothesisTrackingEvidence(
        candidate_count=len(refs),
        coverage_values=coverage_values,
        mean_coverage_fraction=mean_coverage,
        minimum_coverage_fraction=min_coverage,
        ambiguous_assignment_fractions=ambiguous_values,
        mean_ambiguous_assignment_fraction=mean_ambiguous,
        near_threshold_assignment_fractions=near_values,
        mean_near_threshold_assignment_fraction=mean_near,
        assignment_margins=margins,
        minimum_assignment_margin=min_margin,
        frequency_fit_rmse_values_hz=rmse_values,
        mean_frequency_fit_rmse_hz=mean_rmse,
        missing_value_counts=missing_counts,
        passes=passes,
        reasons=_ordered_reasons(reasons),
        diagnostics=tuple(dict.fromkeys(diagnostics)),
    )


def _impact_evidence(
    chain: CrossConditionCandidateChain,
    cfg: ModalHypothesisSettings,
) -> ModalHypothesisImpactEvidence:
    refs = tuple(node.candidate_ref for node in chain.nodes)
    classifications = tuple(ref.preimpact_classification or ref.classification for ref in refs)
    available = tuple(
        ref.impact_excited is not None or classification is not None
        for ref, classification in zip(refs, classifications, strict=True)
    )
    available_count = sum(available)
    supported = sum(ref.impact_excited is True for ref in refs)
    missing = len(refs) - available_count
    fraction = supported / len(refs) if refs else None
    background = sum(classification == "persistent_background_tone" for classification in classifications)
    reasons: list[ModalHypothesisReason] = []
    diagnostics = [
        "preimpact_evidence_is_operational_not_causality",
        "classifications_reused_without_physical_relabeling",
        "missing_preimpact_values_preserved",
    ]
    passes = True
    if cfg.require_impact_excitation:
        if available_count == 0 or (missing and not cfg.allow_missing_preimpact_evidence):
            passes = False
            reasons.append(ModalHypothesisReason.MISSING_REQUIRED_IMPACT_EVIDENCE)
        elif cfg.minimum_impact_supported_fraction is not None and fraction is not None and fraction < cfg.minimum_impact_supported_fraction and not isclose(fraction, cfg.minimum_impact_supported_fraction, rel_tol=1e-12, abs_tol=1e-12):
            passes = False
            reasons.append(ModalHypothesisReason.MISSING_REQUIRED_IMPACT_EVIDENCE)
    elif missing and cfg.missing_impact_evidence_policy != "allow":
        reasons.append(ModalHypothesisReason.INSUFFICIENT_EVIDENCE)
    if not cfg.allow_missing_preimpact_evidence and missing:
        passes = False
        reasons.append(ModalHypothesisReason.MISSING_REQUIRED_IMPACT_EVIDENCE)
    if passes and available_count and supported:
        reasons.append(ModalHypothesisReason.SUFFICIENT_IMPACT_EVIDENCE)
    return ModalHypothesisImpactEvidence(
        candidate_count=len(refs),
        available_evidence_count=available_count,
        missing_evidence_count=missing,
        impact_supported_count=supported,
        impact_supported_fraction=fraction,
        classifications=classifications,
        background_persistent_count=background,
        passes=passes,
        reasons=_ordered_reasons(reasons),
        diagnostics=tuple(diagnostics),
    )


def _structural_context(
    chain: CrossConditionCandidateChain,
    cfg: ModalHypothesisSettings,
) -> ModalHypothesisStructuralContext:
    reasons: list[ModalHypothesisReason] = []
    diagnostics = [
        "split_and_merge_contexts_are_diagnostic_only",
        "no_branching_or_fusion_performed",
        "no_physical_split_or_merge_concluded",
    ]
    passes = True
    requires_reservation = False
    if chain.contains_possible_split_context:
        reasons.append(ModalHypothesisReason.POSSIBLE_SPLIT_CONTEXT)
        if cfg.reject_possible_split_context:
            passes = False
        elif cfg.reserve_possible_split_context:
            requires_reservation = True
    if chain.contains_possible_merge_context:
        reasons.append(ModalHypothesisReason.POSSIBLE_MERGE_CONTEXT)
        if cfg.reject_possible_merge_context:
            passes = False
        elif cfg.reserve_possible_merge_context:
            requires_reservation = True
    return ModalHypothesisStructuralContext(
        possible_split_contexts=chain.possible_split_contexts,
        possible_merge_contexts=chain.possible_merge_contexts,
        split_context_count=len(chain.possible_split_contexts),
        merge_context_count=len(chain.possible_merge_contexts),
        contains_possible_split_context=chain.contains_possible_split_context,
        contains_possible_merge_context=chain.contains_possible_merge_context,
        passes=passes,
        requires_reservation=requires_reservation and passes,
        reasons=_ordered_reasons(reasons),
        diagnostics=tuple(diagnostics),
    )


def _apply_tracking_policy(
    tracking: ModalHypothesisTrackingEvidence,
    cfg: ModalHypothesisSettings,
    reservations: list[ModalHypothesisReason],
    rejections: list[ModalHypothesisReason],
    missing: list[ModalHypothesisReason],
) -> None:
    if ModalHypothesisReason.INSUFFICIENT_TRACKING_QUALITY in tracking.reasons:
        rejections.append(ModalHypothesisReason.INSUFFICIENT_TRACKING_QUALITY)
    if ModalHypothesisReason.INSUFFICIENT_EVIDENCE in tracking.reasons:
        if cfg.missing_tracking_evidence_policy == "reservation":
            reservations.append(ModalHypothesisReason.INSUFFICIENT_EVIDENCE)
            missing.append(ModalHypothesisReason.INSUFFICIENT_EVIDENCE)
        elif cfg.missing_tracking_evidence_policy == "insufficient":
            missing.append(ModalHypothesisReason.INSUFFICIENT_EVIDENCE)
        elif cfg.missing_tracking_evidence_policy == "reject":
            rejections.append(ModalHypothesisReason.INSUFFICIENT_TRACKING_QUALITY)


def _apply_decay_policy(
    decay: ModalHypothesisDecayEvidence,
    cfg: ModalHypothesisSettings,
    reservations: list[ModalHypothesisReason],
    rejections: list[ModalHypothesisReason],
    missing: list[ModalHypothesisReason],
) -> None:
    if ModalHypothesisReason.MISSING_REQUIRED_DECAY in decay.reasons:
        if cfg.require_decay_evidence:
            missing.append(ModalHypothesisReason.MISSING_REQUIRED_DECAY)
        else:
            rejections.append(ModalHypothesisReason.MISSING_REQUIRED_DECAY)
    if ModalHypothesisReason.INCONSISTENT_DECAY in decay.reasons:
        rejections.append(ModalHypothesisReason.INCONSISTENT_DECAY)
    if (
        ModalHypothesisReason.INSUFFICIENT_EVIDENCE in decay.reasons
        and ModalHypothesisReason.MISSING_REQUIRED_DECAY not in decay.reasons
    ):
        if cfg.missing_decay_evidence_policy == "reservation":
            reservations.append(ModalHypothesisReason.INSUFFICIENT_EVIDENCE)
            missing.append(ModalHypothesisReason.INSUFFICIENT_EVIDENCE)
        elif cfg.missing_decay_evidence_policy == "insufficient":
            missing.append(ModalHypothesisReason.INSUFFICIENT_EVIDENCE)
        elif cfg.missing_decay_evidence_policy == "reject":
            rejections.append(ModalHypothesisReason.MISSING_REQUIRED_DECAY)


def _apply_impact_policy(
    impact: ModalHypothesisImpactEvidence,
    cfg: ModalHypothesisSettings,
    reservations: list[ModalHypothesisReason],
    rejections: list[ModalHypothesisReason],
    missing: list[ModalHypothesisReason],
) -> None:
    if ModalHypothesisReason.MISSING_REQUIRED_IMPACT_EVIDENCE in impact.reasons:
        if cfg.require_impact_excitation and impact.available_evidence_count == 0:
            missing.append(ModalHypothesisReason.MISSING_REQUIRED_IMPACT_EVIDENCE)
        elif not cfg.allow_missing_preimpact_evidence and impact.missing_evidence_count:
            missing.append(ModalHypothesisReason.MISSING_REQUIRED_IMPACT_EVIDENCE)
        else:
            rejections.append(ModalHypothesisReason.MISSING_REQUIRED_IMPACT_EVIDENCE)
    if ModalHypothesisReason.INSUFFICIENT_EVIDENCE in impact.reasons:
        if cfg.missing_impact_evidence_policy == "reservation":
            reservations.append(ModalHypothesisReason.INSUFFICIENT_EVIDENCE)
            missing.append(ModalHypothesisReason.INSUFFICIENT_EVIDENCE)
        elif cfg.missing_impact_evidence_policy == "insufficient":
            missing.append(ModalHypothesisReason.INSUFFICIENT_EVIDENCE)
        elif cfg.missing_impact_evidence_policy == "reject":
            rejections.append(ModalHypothesisReason.MISSING_REQUIRED_IMPACT_EVIDENCE)


def _frequency_score_value(
    evidence: ModalHypothesisFrequencyEvidence,
    cfg: ModalHypothesisSettings,
) -> float | None:
    if len(evidence.frequencies_hz) < 2:
        return None
    scores = (
        _upper_limit_score(evidence.maximum_step_change_hz, cfg.maximum_step_absolute_frequency_change_hz),
        _upper_limit_score(evidence.maximum_step_relative_change, cfg.maximum_step_relative_frequency_change),
        _upper_limit_score(evidence.total_absolute_change_hz, cfg.maximum_total_absolute_frequency_change_hz),
        _upper_limit_score(evidence.total_relative_symmetric_change, cfg.maximum_total_relative_frequency_change),
        _upper_limit_score(evidence.trajectory_rmse_from_mean_hz, cfg.maximum_frequency_trajectory_rmse_hz),
        _upper_limit_score(evidence.trajectory_relative_rmse, cfg.maximum_frequency_trajectory_relative_rmse),
    )
    return _mean_present(scores)


def _association_score_value(
    evidence: ModalHypothesisAssociationEvidence,
    cfg: ModalHypothesisSettings,
) -> float | None:
    if not evidence.match_costs:
        return None
    scores = [
        _upper_limit_score(evidence.maximum_match_cost, cfg.maximum_match_cost),
        _upper_limit_score(evidence.mean_match_cost, cfg.maximum_mean_match_cost),
        _upper_limit_score(evidence.ambiguous_match_fraction, cfg.maximum_ambiguous_match_fraction),
        _upper_limit_score(evidence.near_threshold_match_fraction, cfg.maximum_near_threshold_match_fraction),
    ]
    if cfg.minimum_match_margin is not None:
        scores.append(_lower_limit_score(evidence.minimum_match_margin, cfg.minimum_match_margin))
    return _mean_present(tuple(scores))


def _tracking_score_value(
    evidence: ModalHypothesisTrackingEvidence,
    cfg: ModalHypothesisSettings,
) -> float | None:
    scores = (
        _lower_limit_score(evidence.mean_coverage_fraction, cfg.minimum_mean_coverage_fraction),
        _upper_limit_score(evidence.mean_ambiguous_assignment_fraction, cfg.maximum_mean_ambiguous_assignment_fraction),
        _upper_limit_score(evidence.mean_near_threshold_assignment_fraction, cfg.maximum_mean_near_threshold_assignment_fraction),
        _lower_limit_score(evidence.minimum_assignment_margin, cfg.minimum_mean_assignment_margin),
        _upper_limit_score(evidence.mean_frequency_fit_rmse_hz, cfg.maximum_mean_frequency_fit_rmse_hz),
    )
    return _mean_present(scores)


def _decay_score_value(
    evidence: ModalHypothesisDecayEvidence,
    cfg: ModalHypothesisSettings,
) -> float | None:
    if evidence.available_tau_count < cfg.minimum_decay_value_count:
        return None
    scores = (
        _upper_limit_score(evidence.log_tau_range, cfg.maximum_log_tau_range),
        _upper_limit_score(evidence.log_tau_standard_deviation, cfg.maximum_log_tau_standard_deviation),
    )
    value = _mean_present(scores)
    return 1.0 if value is None else value


def _impact_score_value(evidence: ModalHypothesisImpactEvidence) -> float | None:
    if evidence.available_evidence_count == 0:
        return None
    return evidence.impact_supported_fraction


def _score_component(
    name: str,
    value: float | None,
    weight: float,
) -> ModalHypothesisScoreComponent:
    if value is None:
        return ModalHypothesisScoreComponent(name, None, weight, None, False, ("component_unavailable",))
    clipped = min(1.0, max(0.0, value))
    return ModalHypothesisScoreComponent(name, clipped, weight, clipped * weight, True)


def _invalid_hypothesis(
    sequence: tuple[str, ...],
    settings: ModalHypothesisSettings,
    diagnostic: str,
) -> ModalHypothesis:
    coverage = ModalHypothesisCoverageEvidence(
        requested_condition_count=max(1, len(sequence)),
        observed_condition_count=0,
        condition_coverage_fraction=0.0,
        match_count=0,
        complete_across_requested_sequence=False,
        starts_as_emerging=False,
        ends_as_disappearing=False,
        singleton=False,
        partial=False,
        passes=False,
        reasons=(ModalHypothesisReason.INVALID_CHAIN,),
        diagnostics=("invalid_input",),
    )
    frequency = ModalHypothesisFrequencyEvidence(
        frequencies_hz=(1.0,),
        condition_labels=("invalid",),
        signed_step_changes_hz=(),
        absolute_step_changes_hz=(),
        signed_step_relative_changes=(),
        absolute_step_relative_changes=(),
        maximum_step_change_hz=None,
        maximum_step_relative_change=None,
        total_signed_change_hz=0.0,
        total_absolute_change_hz=0.0,
        total_relative_symmetric_change=0.0,
        trajectory_mean_hz=1.0,
        trajectory_standard_deviation_hz=0.0,
        trajectory_rmse_from_mean_hz=0.0,
        trajectory_relative_rmse=0.0,
        up_step_count=0,
        down_step_count=0,
        preserved_step_count=0,
        indeterminate_step_count=0,
        passes=False,
        reasons=(ModalHypothesisReason.INVALID_CHAIN,),
        diagnostics=("invalid_input",),
    )
    association = ModalHypothesisAssociationEvidence(
        match_ids=(),
        match_costs=(),
        mean_match_cost=None,
        maximum_match_cost=None,
        minimum_match_cost=None,
        ambiguous_match_count=0,
        ambiguous_match_fraction=None,
        near_threshold_match_count=0,
        near_threshold_match_fraction=None,
        match_margins=(),
        minimum_match_margin=None,
        passes=False,
        reasons=(ModalHypothesisReason.INVALID_CHAIN,),
        diagnostics=("invalid_input",),
    )
    tracking = ModalHypothesisTrackingEvidence(
        candidate_count=0,
        coverage_values=(),
        mean_coverage_fraction=None,
        minimum_coverage_fraction=None,
        ambiguous_assignment_fractions=(),
        mean_ambiguous_assignment_fraction=None,
        near_threshold_assignment_fractions=(),
        mean_near_threshold_assignment_fraction=None,
        assignment_margins=(),
        minimum_assignment_margin=None,
        frequency_fit_rmse_values_hz=(),
        mean_frequency_fit_rmse_hz=None,
        missing_value_counts=(),
        passes=False,
        reasons=(ModalHypothesisReason.INVALID_CHAIN,),
        diagnostics=("invalid_input",),
    )
    decay = ModalHypothesisDecayEvidence(
        tau_values_s=(),
        available_tau_count=0,
        missing_tau_count=0,
        log_tau_values=(),
        minimum_tau_s=None,
        maximum_tau_s=None,
        log_tau_range=None,
        log_tau_mean=None,
        log_tau_standard_deviation=None,
        fit_quality_values=(),
        mean_fit_quality=None,
        passes=False,
        reasons=(ModalHypothesisReason.INVALID_CHAIN,),
        diagnostics=("invalid_input",),
    )
    impact = ModalHypothesisImpactEvidence(
        candidate_count=0,
        available_evidence_count=0,
        missing_evidence_count=0,
        impact_supported_count=0,
        impact_supported_fraction=None,
        classifications=(),
        background_persistent_count=0,
        passes=False,
        reasons=(ModalHypothesisReason.INVALID_CHAIN,),
        diagnostics=("invalid_input",),
    )
    structural = ModalHypothesisStructuralContext(
        possible_split_contexts=(),
        possible_merge_contexts=(),
        split_context_count=0,
        merge_context_count=0,
        contains_possible_split_context=False,
        contains_possible_merge_context=False,
        passes=False,
        requires_reservation=False,
        reasons=(ModalHypothesisReason.INVALID_CHAIN,),
        diagnostics=("invalid_input",),
    )
    score = compute_modal_hypothesis_score(
        coverage,
        frequency,
        association,
        tracking,
        decay,
        impact,
        structural,
        (ModalHypothesisReason.INVALID_CHAIN,),
        settings,
    )
    hypothesis_id = "modal-hypothesis-" + sha1(f"invalid:{diagnostic}".encode("utf-8")).hexdigest()[:16]
    return ModalHypothesis(
        hypothesis_id=hypothesis_id,
        source_chain_id=None,
        chain=None,
        status=ModalHypothesisStatus.INVALID_INPUT,
        score=score,
        coverage_evidence=coverage,
        frequency_evidence=frequency,
        association_evidence=association,
        tracking_evidence=tracking,
        decay_evidence=decay,
        impact_evidence=impact,
        structural_context=structural,
        supporting_reasons=(),
        reservation_reasons=(),
        rejection_reasons=(ModalHypothesisReason.INVALID_CHAIN,),
        missing_evidence_reasons=(),
        accepted=False,
        requires_review=True,
        valid=False,
        diagnostics=(diagnostic, "invalid_chain_not_promoted_to_hypothesis"),
    )


def _extend_supporting(
    target: list[ModalHypothesisReason],
    reasons: tuple[ModalHypothesisReason, ...],
) -> None:
    target.extend(
        reason for reason in reasons
        if reason in {
            ModalHypothesisReason.SUFFICIENT_CROSS_CONDITION_PERSISTENCE,
            ModalHypothesisReason.SUFFICIENT_FREQUENCY_CONTINUITY,
            ModalHypothesisReason.SUFFICIENT_TRACKING_QUALITY,
            ModalHypothesisReason.SUFFICIENT_DECAY_CONSISTENCY,
            ModalHypothesisReason.SUFFICIENT_IMPACT_EVIDENCE,
            ModalHypothesisReason.COMPLETE_CHAIN,
            ModalHypothesisReason.PARTIAL_BUT_SUPPORTED_CHAIN,
        }
    )


def _upper_limit_score(value: float | None, limit: float | None) -> float | None:
    if value is None or limit is None:
        return None
    if limit == 0.0:
        return 1.0 if value == 0.0 else 0.0
    return 1.0 - min(1.0, max(0.0, value / limit))


def _lower_limit_score(value: float | None, limit: float | None) -> float | None:
    if value is None or limit is None:
        return None
    if limit == 0.0:
        return 1.0
    return min(1.0, max(0.0, value / limit))


def _exceeds(value: float | None, limit: float | None) -> bool:
    return (
        value is not None
        and limit is not None
        and value > limit
        and not isclose(value, limit, rel_tol=1e-12, abs_tol=1e-12)
    )


def _inclusive_ge(left: float, right: float) -> bool:
    return left >= right or isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def _mean_present(values: Iterable[float | None]) -> float | None:
    present = tuple(value for value in values if value is not None)
    if not present:
        return None
    return sum(present) / len(present)


def _min_present(values: Iterable[float | None]) -> float | None:
    present = tuple(value for value in values if value is not None)
    return min(present) if present else None


def _missing_count(values: Iterable[object | None]) -> int:
    return sum(value is None for value in values)


def _missing_counts(values: tuple[tuple[str, int], ...]) -> None:
    names = tuple(name for name, _ in values)
    if len(names) != len(set(names)):
        raise ValueError("missing value count names must be unique.")
    for name, count in values:
        _text(name, "missing count name")
        if count < 0:
            raise ValueError("missing value counts must not be negative.")


def _hypothesis_id(sequence: tuple[str, ...], chain: CrossConditionCandidateChain) -> str:
    ref_tokens = tuple(
        f"{node.dynamic_label}:{node.candidate_ref.recording_id}:{node.candidate_ref.candidate_id}:{node.candidate_ref.source_track_id}"
        for node in chain.nodes
    )
    match_tokens = tuple(
        node.outgoing_match_id for node in chain.nodes
        if node.outgoing_match_id is not None
    )
    payload = "|".join((
        "sequence=" + ",".join(sequence),
        "source_chain_id=" + chain.chain_id,
        "refs=" + ",".join(ref_tokens),
        "matches=" + ",".join(match_tokens),
    ))
    return "modal-hypothesis-" + sha1(payload.encode("utf-8")).hexdigest()[:16]


def _chain_sort_key_for_hypothesis(chain: CrossConditionCandidateChain | object) -> tuple:
    if not isinstance(chain, CrossConditionCandidateChain):
        return (99, "", "", "")
    first = chain.nodes[0].candidate_ref
    return (
        _DYNAMIC_LABEL_INDEX[chain.start_dynamic_label],
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
    )


def _hypothesis_sort_key(hypothesis: ModalHypothesis) -> tuple:
    chain = hypothesis.chain
    if chain is None:
        return (99, "", hypothesis.hypothesis_id)
    return _chain_sort_key_for_hypothesis(chain) + (hypothesis.hypothesis_id,)


def _minimal_sequence(labels: tuple[str, ...]) -> tuple[str, ...]:
    indices = sorted({_DYNAMIC_LABEL_INDEX[label] for label in labels})
    return tuple(DYNAMIC_LABEL_ORDER[index] for index in range(indices[0], indices[-1] + 1))


def _validate_sequence(labels: tuple[str, ...], *, allow_single: bool) -> None:
    if not isinstance(labels, tuple) or not labels:
        raise ValueError("sequence must be a nonempty tuple.")
    if not allow_single and len(labels) < 2:
        raise ValueError("sequence must contain at least two labels.")
    if any(label not in _DYNAMIC_LABEL_INDEX for label in labels):
        raise ValueError("sequence contains an unknown dynamic label.")
    if len(labels) != len(set(labels)):
        raise ValueError("sequence must not contain repeated labels.")
    indices = tuple(_DYNAMIC_LABEL_INDEX[label] for label in labels)
    if indices != tuple(sorted(indices)):
        raise ValueError("sequence must follow nominal dynamic order.")
    if any(right - left != 1 for left, right in zip(indices, indices[1:], strict=False)):
        raise ValueError("sequence must be contiguous.")


def _ordered_reasons(
    reasons: Iterable[ModalHypothesisReason],
) -> tuple[ModalHypothesisReason, ...]:
    unique = {
        reason if isinstance(reason, ModalHypothesisReason) else ModalHypothesisReason(reason)
        for reason in reasons
    }
    return tuple(sorted(unique, key=lambda item: item.value))


def _reason_tuple(values: tuple[ModalHypothesisReason, ...], name: str) -> None:
    if not isinstance(values, tuple):
        raise ValueError(f"{name} must be an immutable tuple.")
    converted = _ordered_reasons(values)
    if values != converted:
        raise ValueError(f"{name} must contain unique reasons in deterministic order.")


def _finite_optional(value: float | None, name: str, *, nonnegative: bool = False) -> None:
    if value is None:
        return
    if not isfinite(value):
        raise ValueError(f"{name} must be finite when provided.")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be non-negative when provided.")


def _fraction(value: float | None, name: str) -> None:
    if value is not None and (not isfinite(value) or not 0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be finite and in [0, 1] when provided.")


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")


def _strings(values: tuple[str, ...], name: str, *, allow_empty: bool = False) -> None:
    if not isinstance(values, tuple):
        raise ValueError(f"{name} must be an immutable tuple.")
    if not allow_empty and not values:
        raise ValueError(f"{name} must not be empty.")
    if any(not isinstance(item, str) or not item.strip() for item in values):
        raise ValueError(f"{name} must contain nonempty strings.")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must contain unique strings.")
