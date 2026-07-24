"""Operational response-regime descriptors from already computed metrics.

The descriptors in this module are auditable labels produced by explicit
thresholds applied to observed metrics.  They are not physical regime proofs,
nonlinearity tests, modal identification, chaos detection, or conversions to
``ModalMode``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

from belllab.dynamic_comparison import (
    DYNAMIC_LABEL_ORDER,
    DynamicConditionComparisonResult,
    DynamicConditionSpectralSummary,
    AggregatedMetric,
)


STRUCTURE_DESCRIPTORS = frozenset({
    "discrete_line_dominated",
    "mixed_line_and_continuum",
    "dense_spectrum",
    "broadband_dominated",
    "indeterminate",
})
TEMPORAL_DESCRIPTORS = frozenset({
    "stable_spectral_character",
    "broadband_to_tonal",
    "tonal_to_broadband",
    "progressive_spectral_densification",
    "progressive_spectral_sparsification",
    "mixed_temporal_evolution",
    "indeterminate",
})
LINE_IDENTITY_DESCRIPTORS = frozenset({
    "line_identity_preserved",
    "line_identity_partially_preserved",
    "line_identity_not_resolved",
    "not_evaluated",
})
CONFIDENCE_DESCRIPTORS = frozenset({"high", "moderate", "low", "insufficient"})
DIMENSIONS = frozenset({"structure", "temporal_evolution", "line_identity", "confidence"})
OPERATORS = frozenset({">=", "<=", "between", "abs>=", "abs<=", "==", "present"})
SUPPORT_DIRECTIONS = frozenset({"support", "oppose", "neutral"})
CHANGE_NAMES = frozenset({
    "descriptor_preserved",
    "descriptor_changed",
    "became_more_broadband",
    "became_more_tonal",
    "became_denser",
    "became_sparser",
    "insufficient_evidence",
})


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")


def _unique_strings(values: tuple[str, ...], name: str) -> None:
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{name} must contain nonempty strings.")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates.")


def _finite_optional(*values: float | None) -> bool:
    return all(value is None or isfinite(value) for value in values)


def _fraction(value: float | None, name: str) -> None:
    if value is not None and (not isfinite(value) or value < 0.0 or value > 1.0):
        raise ValueError(f"{name} must be in [0, 1] when provided.")


def _descriptor_known(descriptor: str) -> bool:
    return (
        descriptor in STRUCTURE_DESCRIPTORS
        or descriptor in TEMPORAL_DESCRIPTORS
        or descriptor in LINE_IDENTITY_DESCRIPTORS
        or descriptor in CONFIDENCE_DESCRIPTORS
    )


@dataclass(frozen=True, slots=True)
class RegimeCriterionWeight:
    """Optional explicit weight override for one criterion."""

    criterion_name: str
    weight: float

    def __post_init__(self) -> None:
        _text(self.criterion_name, "criterion_name")
        if not isfinite(self.weight) or self.weight < 0.0:
            raise ValueError("criterion weight must be finite and non-negative.")


@dataclass(frozen=True, slots=True)
class ResponseRegimeDescriptorSettings:
    """Explicit thresholds for operational response-regime descriptors."""

    maximum_flatness_for_line_dominated: float | None = 0.25
    minimum_flatness_for_broadband: float | None = 0.55
    maximum_entropy_for_line_dominated: float | None = 0.55
    minimum_entropy_for_broadband: float | None = 0.75
    minimum_tonal_fraction_for_line_dominated: float | None = 0.65
    maximum_tonal_fraction_for_broadband: float | None = 0.35
    maximum_peak_density_for_sparse: float | None = 0.010
    minimum_peak_density_for_dense: float | None = 0.020
    maximum_occupied_fraction_for_narrow: float | None = 0.35
    minimum_occupied_fraction_for_broadband: float | None = 0.65
    minimum_spectral_crest_for_line_dominated: float | None = 5.0
    maximum_spectral_crest_for_broadband: float | None = 3.0
    minimum_peak_count_for_dense: float | None = 8.0
    maximum_median_peak_spacing_for_dense_hz: float | None = None
    minimum_tonal_fraction_for_mixed: float | None = 0.25
    maximum_tonal_fraction_for_mixed: float | None = 0.75
    minimum_residual_fraction_for_mixed: float | None = 0.25
    minimum_flatness_for_mixed: float | None = 0.20
    maximum_flatness_for_mixed: float | None = 0.60

    minimum_flatness_drop_for_broadband_to_tonal: float | None = 0.15
    minimum_entropy_drop_for_broadband_to_tonal: float | None = 0.15
    minimum_tonal_fraction_increase: float | None = 0.20
    minimum_density_change: float | None = 0.005
    minimum_bandwidth_change: float | None = None
    minimum_occupied_fraction_change: float | None = 0.15
    minimum_persistent_change_point_count: int | None = 1
    maximum_flatness_change_for_stable: float | None = 0.05
    maximum_entropy_change_for_stable: float | None = 0.05
    maximum_tonal_fraction_change_for_stable: float | None = 0.08
    maximum_density_change_for_stable: float | None = 0.003
    maximum_bandwidth_change_for_stable: float | None = None

    minimum_valid_repeat_count: int = 1
    minimum_valid_frame_coverage: float | None = 0.50
    maximum_within_condition_variability: float | None = 0.50
    reject_clipped_conditions: bool = False
    minimum_signal_to_background_db: float | None = None
    require_spectral_comparability: bool = False
    allow_missing_metrics: bool = True
    maximum_missing_metric_fraction_for_moderate_confidence: float = 0.50
    maximum_centroid_slope_for_preserved_identity_hz_per_s: float | None = 10.0
    minimum_tonal_fraction_for_line_identity: float | None = 0.50
    minimum_support_fraction_for_descriptor: float = 0.55
    maximum_opposition_fraction_for_descriptor: float = 0.45
    minimum_available_weight_for_descriptor: float = 3.0
    high_confidence_minimum_support_fraction: float = 0.70
    density_resolution_limit_spacing_factor: float = 2.0
    resolution_limited_density_weight_factor: float = 0.25
    score_tie_tolerance: float = 1e-12
    criterion_weights: tuple[RegimeCriterionWeight, ...] = ()
    numerical_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        fraction_fields = (
            "maximum_flatness_for_line_dominated",
            "minimum_flatness_for_broadband",
            "maximum_entropy_for_line_dominated",
            "minimum_entropy_for_broadband",
            "minimum_tonal_fraction_for_line_dominated",
            "maximum_tonal_fraction_for_broadband",
            "maximum_occupied_fraction_for_narrow",
            "minimum_occupied_fraction_for_broadband",
            "minimum_tonal_fraction_for_mixed",
            "maximum_tonal_fraction_for_mixed",
            "minimum_residual_fraction_for_mixed",
            "minimum_flatness_for_mixed",
            "maximum_flatness_for_mixed",
            "minimum_flatness_drop_for_broadband_to_tonal",
            "minimum_entropy_drop_for_broadband_to_tonal",
            "minimum_tonal_fraction_increase",
            "minimum_occupied_fraction_change",
            "maximum_flatness_change_for_stable",
            "maximum_entropy_change_for_stable",
            "maximum_tonal_fraction_change_for_stable",
            "minimum_valid_frame_coverage",
            "maximum_within_condition_variability",
            "maximum_missing_metric_fraction_for_moderate_confidence",
            "minimum_tonal_fraction_for_line_identity",
            "minimum_support_fraction_for_descriptor",
            "maximum_opposition_fraction_for_descriptor",
            "high_confidence_minimum_support_fraction",
            "resolution_limited_density_weight_factor",
        )
        for field in fraction_fields:
            _fraction(getattr(self, field), field)
        nonnegative_fields = (
            "maximum_peak_density_for_sparse",
            "minimum_peak_density_for_dense",
            "minimum_spectral_crest_for_line_dominated",
            "maximum_spectral_crest_for_broadband",
            "minimum_peak_count_for_dense",
            "maximum_median_peak_spacing_for_dense_hz",
            "minimum_density_change",
            "minimum_bandwidth_change",
            "maximum_density_change_for_stable",
            "maximum_bandwidth_change_for_stable",
            "minimum_signal_to_background_db",
            "maximum_centroid_slope_for_preserved_identity_hz_per_s",
            "minimum_available_weight_for_descriptor",
            "density_resolution_limit_spacing_factor",
            "score_tie_tolerance",
            "numerical_tolerance",
        )
        for field in nonnegative_fields:
            value = getattr(self, field)
            if value is not None and (not isfinite(value) or value < 0.0):
                raise ValueError(f"{field} must be finite and non-negative when provided.")
        if self.minimum_persistent_change_point_count is not None and self.minimum_persistent_change_point_count < 0:
            raise ValueError("minimum_persistent_change_point_count must not be negative.")
        if self.minimum_valid_repeat_count < 0:
            raise ValueError("minimum_valid_repeat_count must not be negative.")
        if self.maximum_flatness_for_line_dominated is not None and self.minimum_flatness_for_broadband is not None:
            if self.maximum_flatness_for_line_dominated >= self.minimum_flatness_for_broadband:
                raise ValueError("line flatness maximum must be strictly below broadband flatness minimum.")
        if self.maximum_entropy_for_line_dominated is not None and self.minimum_entropy_for_broadband is not None:
            if self.maximum_entropy_for_line_dominated >= self.minimum_entropy_for_broadband:
                raise ValueError("line entropy maximum must be strictly below broadband entropy minimum.")
        if self.maximum_tonal_fraction_for_broadband is not None and self.minimum_tonal_fraction_for_line_dominated is not None:
            if self.maximum_tonal_fraction_for_broadband >= self.minimum_tonal_fraction_for_line_dominated:
                raise ValueError("broadband tonal maximum must be strictly below line tonal minimum.")
        if self.maximum_peak_density_for_sparse is not None and self.minimum_peak_density_for_dense is not None:
            if self.maximum_peak_density_for_sparse >= self.minimum_peak_density_for_dense:
                raise ValueError("sparse density maximum must be strictly below dense density minimum.")
        if self.maximum_occupied_fraction_for_narrow is not None and self.minimum_occupied_fraction_for_broadband is not None:
            if self.maximum_occupied_fraction_for_narrow >= self.minimum_occupied_fraction_for_broadband:
                raise ValueError("narrow occupied maximum must be strictly below broadband occupied minimum.")
        if self.minimum_tonal_fraction_for_mixed is not None and self.maximum_tonal_fraction_for_mixed is not None:
            if self.minimum_tonal_fraction_for_mixed > self.maximum_tonal_fraction_for_mixed:
                raise ValueError("mixed tonal fraction bounds are inverted.")
        if self.minimum_flatness_for_mixed is not None and self.maximum_flatness_for_mixed is not None:
            if self.minimum_flatness_for_mixed > self.maximum_flatness_for_mixed:
                raise ValueError("mixed flatness bounds are inverted.")
        if self.minimum_support_fraction_for_descriptor + self.numerical_tolerance < 0.0:
            raise ValueError("minimum support fraction is invalid.")
        if not isinstance(self.reject_clipped_conditions, bool) or not isinstance(self.require_spectral_comparability, bool):
            raise ValueError("boolean settings must be booleans.")
        if not isinstance(self.allow_missing_metrics, bool):
            raise ValueError("allow_missing_metrics must be a boolean.")
        names = [item.criterion_name for item in self.criterion_weights]
        if len(names) != len(set(names)):
            raise ValueError("criterion weights must not contain duplicate names.")
        if self.criterion_weights and all(item.weight == 0.0 for item in self.criterion_weights):
            raise ValueError("criterion weights must contain at least one positive weight.")


@dataclass(frozen=True, slots=True)
class RegimeDescriptorCriterionResult:
    """Audit trail for one criterion used by an operational descriptor."""

    criterion_name: str
    metric_name: str
    observed_value: float | None
    operator: Literal[">=", "<=", "between", "abs>=", "abs<=", "==", "present"]
    threshold: float | tuple[float, float] | None
    passed: bool | None
    applicable: bool
    weight: float
    support_direction: Literal["support", "oppose", "neutral"]
    reason: str
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.criterion_name, "criterion_name")
        _text(self.metric_name, "metric_name")
        _text(self.reason, "reason")
        if self.operator not in OPERATORS:
            raise ValueError("criterion operator is not recognized.")
        if not _finite_optional(self.observed_value):
            raise ValueError("observed value must be finite when present.")
        if isinstance(self.threshold, tuple):
            if len(self.threshold) != 2 or any(not isfinite(value) for value in self.threshold):
                raise ValueError("tuple threshold must contain two finite values.")
            if self.threshold[0] > self.threshold[1]:
                raise ValueError("tuple threshold bounds are inverted.")
        elif self.threshold is not None and not isfinite(self.threshold):
            raise ValueError("threshold must be finite when present.")
        if not isfinite(self.weight) or self.weight < 0.0:
            raise ValueError("criterion weight must be finite and non-negative.")
        if self.support_direction not in SUPPORT_DIRECTIONS:
            raise ValueError("support_direction is not recognized.")
        if not self.applicable:
            if self.passed is not None:
                raise ValueError("non-applicable criterion must not pass or fail.")
            if self.support_direction != "neutral":
                raise ValueError("non-applicable criterion must be neutral.")
        else:
            if self.passed is None:
                raise ValueError("applicable criterion requires passed=True or False.")
            expected = "support" if self.passed else "oppose"
            if self.support_direction != expected:
                raise ValueError("support_direction must follow passed state.")
        _unique_strings(self.diagnostics, "criterion diagnostics")


@dataclass(frozen=True, slots=True)
class RegimeDescriptorScore:
    """Weighted evidence score for one descriptor; not a probability."""

    descriptor_name: str
    support_score: float
    opposition_score: float
    available_weight: float
    support_fraction: float | None
    opposition_fraction: float | None
    selected: bool
    indeterminate: bool
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.descriptor_name, "descriptor_name")
        if not _descriptor_known(self.descriptor_name):
            raise ValueError("descriptor_name is not recognized.")
        if any(
            not isfinite(value) or value < 0.0
            for value in (self.support_score, self.opposition_score, self.available_weight)
        ):
            raise ValueError("descriptor scores must be finite and non-negative.")
        if self.support_score > self.available_weight + 1e-12:
            raise ValueError("support score cannot exceed available weight.")
        if self.opposition_score > self.available_weight + 1e-12:
            raise ValueError("opposition score cannot exceed available weight.")
        if abs((self.support_score + self.opposition_score) - self.available_weight) > 1e-12:
            raise ValueError("support and opposition scores must sum to available weight.")
        _fraction(self.support_fraction, "support_fraction")
        _fraction(self.opposition_fraction, "opposition_fraction")
        if self.available_weight == 0.0:
            if self.support_fraction is not None or self.opposition_fraction is not None:
                raise ValueError("zero available weight requires undefined fractions.")
        else:
            if self.support_fraction is None or self.opposition_fraction is None:
                raise ValueError("positive available weight requires score fractions.")
            expected_support = self.support_score / self.available_weight
            expected_opposition = self.opposition_score / self.available_weight
            if abs(self.support_fraction - expected_support) > 1e-12:
                raise ValueError("support fraction is incoherent with scores.")
            if abs(self.opposition_fraction - expected_opposition) > 1e-12:
                raise ValueError("opposition fraction is incoherent with scores.")
        if self.selected and self.available_weight <= 0.0:
            raise ValueError("selected score requires positive available weight.")
        if self.selected and self.support_score <= 0.0:
            raise ValueError("selected score requires positive support.")
        if self.selected and self.indeterminate:
            raise ValueError("score cannot be selected and indeterminate.")
        _unique_strings(self.diagnostics, "score diagnostics")


@dataclass(frozen=True, slots=True)
class DescriptorEvaluation:
    """Full criterion-level evaluation for one descriptor."""

    dimension: str
    descriptor_name: str
    score: RegimeDescriptorScore
    criteria: tuple[RegimeDescriptorCriterionResult, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.dimension not in DIMENSIONS - {"confidence"}:
            raise ValueError("descriptor evaluation dimension is not recognized.")
        if not _descriptor_known(self.descriptor_name):
            raise ValueError("descriptor name is not recognized.")
        if self.score.descriptor_name != self.descriptor_name:
            raise ValueError("descriptor score name is incoherent.")
        names = [criterion.criterion_name for criterion in self.criteria]
        if len(names) != len(set(names)):
            raise ValueError("criteria must not contain duplicates.")
        _unique_strings(self.diagnostics, "descriptor evaluation diagnostics")


@dataclass(frozen=True, slots=True)
class ResponseRegimeDescription:
    """Operational response-regime description for one dynamic condition."""

    dynamic_label: str
    structure_descriptor: str
    temporal_evolution_descriptor: str
    line_identity_descriptor: str
    confidence_descriptor: Literal["high", "moderate", "low", "insufficient"]
    descriptor_results: tuple[DescriptorEvaluation, ...]
    supporting_metrics: tuple[str, ...]
    conflicting_metrics: tuple[str, ...]
    unavailable_metrics: tuple[str, ...]
    limitations: tuple[str, ...]
    valid: bool
    failure_reason: str | None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.dynamic_label not in DYNAMIC_LABEL_ORDER:
            raise ValueError("dynamic_label is not recognized.")
        if self.structure_descriptor not in STRUCTURE_DESCRIPTORS:
            raise ValueError("structure_descriptor is not recognized.")
        if self.temporal_evolution_descriptor not in TEMPORAL_DESCRIPTORS:
            raise ValueError("temporal_evolution_descriptor is not recognized.")
        if self.line_identity_descriptor not in LINE_IDENTITY_DESCRIPTORS:
            raise ValueError("line_identity_descriptor is not recognized.")
        if self.confidence_descriptor not in CONFIDENCE_DESCRIPTORS:
            raise ValueError("confidence_descriptor is not recognized.")
        _unique_strings(self.supporting_metrics, "supporting_metrics")
        _unique_strings(self.conflicting_metrics, "conflicting_metrics")
        _unique_strings(self.unavailable_metrics, "unavailable_metrics")
        _unique_strings(self.limitations, "limitations")
        if self.confidence_descriptor == "insufficient" and not self.limitations:
            raise ValueError("insufficient confidence requires at least one limitation.")
        if self.valid and self.structure_descriptor == "indeterminate":
            selected_structure = [
                item for item in self.descriptor_results
                if item.dimension == "structure" and item.score.selected
            ]
            if selected_structure:
                raise ValueError("indeterminate structure cannot coexist with selected structure score.")
        if self.valid and self.structure_descriptor != "indeterminate":
            if not _description_has_selected_score(self.descriptor_results, "structure", self.structure_descriptor):
                raise ValueError("selected structure descriptor must have a selected score.")
        if (
            self.valid
            and self.temporal_evolution_descriptor not in {"indeterminate", "mixed_temporal_evolution"}
            and not _description_has_selected_score(
                self.descriptor_results,
                "temporal_evolution",
                self.temporal_evolution_descriptor,
            )
        ):
            raise ValueError("selected temporal descriptor must have a selected score.")
        if (
            self.valid
            and self.line_identity_descriptor == "line_identity_preserved"
            and not _description_has_selected_score(
                self.descriptor_results,
                "line_identity",
                "line_identity_preserved",
            )
        ):
            raise ValueError("preserved line identity requires a selected score.")
        if self.valid:
            if self.failure_reason is not None:
                raise ValueError("valid description must not have failure_reason.")
        elif not self.failure_reason:
            raise ValueError("invalid description requires failure_reason.")
        _unique_strings(self.diagnostics, "description diagnostics")


@dataclass(frozen=True, slots=True)
class RegimeDescriptorSequence:
    """Descriptor sequence over the canonical dynamic-label order."""

    dimension: str
    labels: tuple[str, ...]
    descriptors: tuple[str | None, ...]
    missing_labels: tuple[str, ...]
    changes: tuple[str, ...]
    stable_segments: tuple[tuple[str, str, str], ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.dimension not in DIMENSIONS - {"confidence"}:
            raise ValueError("sequence dimension is not recognized.")
        if self.labels != DYNAMIC_LABEL_ORDER:
            raise ValueError("sequence labels must preserve canonical dynamic order.")
        if len(self.descriptors) != len(self.labels):
            raise ValueError("sequence descriptors must align with labels.")
        _unique_strings(self.missing_labels, "missing_labels")
        if any(change not in CHANGE_NAMES for change in self.changes):
            raise ValueError("sequence changes contain unknown names.")
        for descriptor in self.descriptors:
            if descriptor is not None and not _descriptor_known(descriptor):
                raise ValueError("sequence descriptor is not recognized.")
        _unique_strings(self.diagnostics, "sequence diagnostics")


@dataclass(frozen=True, slots=True)
class EmergentResponsePattern:
    """Structured descriptive pattern observed across dynamic conditions."""

    pattern_name: str
    start_label: str
    end_label: str
    supporting_conditions: tuple[str, ...]
    supporting_metrics: tuple[str, ...]
    limitations: tuple[str, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.pattern_name, "pattern_name")
        if self.start_label not in DYNAMIC_LABEL_ORDER or self.end_label not in DYNAMIC_LABEL_ORDER:
            raise ValueError("pattern labels must be recognized.")
        if DYNAMIC_LABEL_ORDER.index(self.start_label) > DYNAMIC_LABEL_ORDER.index(self.end_label):
            raise ValueError("pattern labels must be ordered.")
        _unique_strings(self.supporting_conditions, "supporting_conditions")
        _unique_strings(self.supporting_metrics, "supporting_metrics")
        _unique_strings(self.limitations, "limitations")
        _unique_strings(self.diagnostics, "pattern diagnostics")


@dataclass(frozen=True, slots=True)
class DynamicResponseRegimeDescription:
    """Top-level operational descriptor result over dynamic conditions."""

    condition_descriptions: tuple[ResponseRegimeDescription, ...]
    ordered_labels: tuple[str, ...]
    descriptor_sequences: tuple[RegimeDescriptorSequence, ...]
    descriptor_changes: tuple[str, ...]
    emergent_patterns: tuple[EmergentResponsePattern, ...]
    limitations: tuple[str, ...]
    valid: bool
    failure_reason: str | None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if tuple(sorted(self.ordered_labels, key=DYNAMIC_LABEL_ORDER.index)) != self.ordered_labels:
            raise ValueError("ordered_labels must preserve canonical order.")
        _unique_strings(self.ordered_labels, "ordered_labels")
        description_labels = tuple(item.dynamic_label for item in self.condition_descriptions)
        if description_labels != self.ordered_labels:
            raise ValueError("condition descriptions must match ordered labels.")
        if any(change not in CHANGE_NAMES for change in self.descriptor_changes):
            raise ValueError("descriptor_changes contain unknown names.")
        _unique_strings(self.limitations, "limitations")
        if self.valid:
            if self.failure_reason is not None:
                raise ValueError("valid dynamic regime result must not have failure_reason.")
        elif not self.failure_reason:
            raise ValueError("invalid dynamic regime result requires failure_reason.")
        _unique_strings(self.diagnostics, "dynamic regime diagnostics")


def _description_has_selected_score(
    evaluations: tuple[DescriptorEvaluation, ...],
    dimension: str,
    descriptor_name: str,
) -> bool:
    return any(
        evaluation.dimension == dimension
        and evaluation.descriptor_name == descriptor_name
        and evaluation.score.selected
        for evaluation in evaluations
    )


def describe_response_regime(
    condition_summary: DynamicConditionSpectralSummary,
    settings: ResponseRegimeDescriptorSettings | None = None,
) -> ResponseRegimeDescription:
    """Describe one condition using only its already computed summary metrics."""

    cfg = settings or ResponseRegimeDescriptorSettings()
    metrics = _metric_map(condition_summary)
    limitations = list(_base_limitations(condition_summary, cfg, metrics))
    invalid_reason = _invalid_reason(condition_summary, cfg, limitations)
    evaluations = tuple(_evaluate_all_descriptors(condition_summary, cfg, metrics, limitations))
    selected_structure = _select_structure_descriptor(evaluations)
    selected_temporal = _select_temporal_descriptor(evaluations)
    selected_identity = _select_line_identity_descriptor(evaluations)
    supporting, conflicting, unavailable = _metric_evidence_lists(evaluations)
    confidence = _confidence_descriptor(condition_summary, cfg, evaluations, limitations, unavailable)
    diagnostics = (
        "operational_response_regime_descriptor_not_physical_regime_proof",
        "no_non_linearity_proof_was_performed",
        "no_cross_condition_candidate_association_was_performed",
        "no_modal_mode_conversion_was_performed",
    )
    if invalid_reason is not None:
        selected_structure = "indeterminate"
        selected_temporal = "indeterminate"
        selected_identity = "not_evaluated"
        confidence = "insufficient"
        if not limitations:
            limitations.append(invalid_reason)
    return ResponseRegimeDescription(
        dynamic_label=condition_summary.dynamic_label,
        structure_descriptor=selected_structure,
        temporal_evolution_descriptor=selected_temporal,
        line_identity_descriptor=selected_identity,
        confidence_descriptor=confidence,
        descriptor_results=evaluations,
        supporting_metrics=tuple(supporting),
        conflicting_metrics=tuple(conflicting),
        unavailable_metrics=tuple(unavailable),
        limitations=tuple(dict.fromkeys(limitations)),
        valid=invalid_reason is None,
        failure_reason=invalid_reason,
        diagnostics=diagnostics,
    )


def evaluate_regime_descriptor(
    condition_summary: DynamicConditionSpectralSummary,
    descriptor_name: str,
    settings: ResponseRegimeDescriptorSettings | None = None,
) -> RegimeDescriptorScore:
    """Return the weighted score for one descriptor."""

    cfg = settings or ResponseRegimeDescriptorSettings()
    if not _descriptor_known(descriptor_name):
        raise ValueError("descriptor_name is not recognized.")
    metrics = _metric_map(condition_summary)
    limitations = list(_base_limitations(condition_summary, cfg, metrics))
    for evaluation in _evaluate_all_descriptors(condition_summary, cfg, metrics, limitations):
        if evaluation.descriptor_name == descriptor_name:
            return evaluation.score
    raise ValueError("descriptor_name is not scored by this function.")


def build_regime_descriptor_sequence(
    descriptions: tuple[ResponseRegimeDescription, ...] | list[ResponseRegimeDescription],
    dimension: Literal["structure", "temporal_evolution", "line_identity"],
) -> RegimeDescriptorSequence:
    """Build a canonical pp-to-ff descriptor sequence for one dimension."""

    if dimension not in DIMENSIONS - {"confidence"}:
        raise ValueError("dimension is not recognized.")
    by_label = {description.dynamic_label: description for description in descriptions}
    descriptors: list[str | None] = []
    missing: list[str] = []
    for label in DYNAMIC_LABEL_ORDER:
        description = by_label.get(label)
        if description is None:
            descriptors.append(None)
            missing.append(label)
            continue
        descriptors.append(_descriptor_for_dimension(description, dimension))
    changes = _sequence_changes(tuple(descriptors))
    stable_segments = _stable_segments(tuple(descriptors))
    diagnostics = ("descriptor_sequence_is_operational_not_physical_transition",)
    return RegimeDescriptorSequence(
        dimension=dimension,
        labels=DYNAMIC_LABEL_ORDER,
        descriptors=tuple(descriptors),
        missing_labels=tuple(missing),
        changes=changes,
        stable_segments=stable_segments,
        diagnostics=diagnostics,
    )


def describe_dynamic_response_regimes(
    comparison_or_summaries: DynamicConditionComparisonResult
    | tuple[DynamicConditionSpectralSummary, ...]
    | list[DynamicConditionSpectralSummary],
    settings: ResponseRegimeDescriptorSettings | None = None,
) -> DynamicResponseRegimeDescription:
    """Describe operational regime descriptors over available dynamic conditions."""

    cfg = settings or ResponseRegimeDescriptorSettings()
    if isinstance(comparison_or_summaries, DynamicConditionComparisonResult):
        summaries = comparison_or_summaries.condition_summaries
    else:
        summaries = tuple(comparison_or_summaries)
    summaries = tuple(sorted(summaries, key=lambda summary: DYNAMIC_LABEL_ORDER.index(summary.dynamic_label)))
    descriptions = tuple(describe_response_regime(summary, cfg) for summary in summaries)
    ordered_labels = tuple(description.dynamic_label for description in descriptions)
    sequences = tuple(
        build_regime_descriptor_sequence(descriptions, dimension)
        for dimension in ("structure", "temporal_evolution", "line_identity")
    )
    descriptor_changes = tuple(dict.fromkeys(change for sequence in sequences for change in sequence.changes))
    limitations = tuple(dict.fromkeys(limitation for description in descriptions for limitation in description.limitations))
    patterns = _emergent_patterns(descriptions, sequences)
    valid_descriptions = tuple(description for description in descriptions if description.valid)
    diagnostics = (
        "dynamic_response_regime_description_is_descriptive_not_regime_transition_proof",
        "no_non_linearity_proof_was_performed",
        "no_cross_condition_candidate_association_was_performed",
        "no_modal_mode_conversion_was_performed",
    )
    valid = len(valid_descriptions) >= 1
    return DynamicResponseRegimeDescription(
        condition_descriptions=descriptions,
        ordered_labels=ordered_labels,
        descriptor_sequences=sequences,
        descriptor_changes=descriptor_changes,
        emergent_patterns=patterns,
        limitations=limitations,
        valid=valid,
        failure_reason=None if valid else "no_valid_condition_descriptions",
        diagnostics=diagnostics,
    )


def _metric_map(summary: DynamicConditionSpectralSummary) -> dict[str, AggregatedMetric]:
    return {metric.metric_name: metric for metric in summary.all_metrics}


def _metric_value(metrics: dict[str, AggregatedMetric], name: str) -> float | None:
    metric = metrics.get(name)
    if metric is None or not metric.valid:
        return None
    return metric.median


def _metric_cv(metrics: dict[str, AggregatedMetric], name: str) -> float | None:
    metric = metrics.get(name)
    if metric is None or not metric.valid:
        return None
    return metric.coefficient_of_variation


def _weight(settings: ResponseRegimeDescriptorSettings, criterion_name: str, *, factor: float = 1.0) -> float:
    for item in settings.criterion_weights:
        if item.criterion_name == criterion_name:
            return item.weight * factor
    return 1.0 * factor


def _criterion(
    metrics: dict[str, AggregatedMetric],
    settings: ResponseRegimeDescriptorSettings,
    criterion_name: str,
    metric_name: str,
    operator: Literal[">=", "<=", "between", "abs>=", "abs<=", "==", "present"],
    threshold: float | tuple[float, float] | None,
    *,
    weight_factor: float = 1.0,
    disabled_when_threshold_missing: bool = True,
    diagnostics: tuple[str, ...] = (),
) -> RegimeDescriptorCriterionResult:
    observed = _metric_value(metrics, metric_name)
    if threshold is None and disabled_when_threshold_missing:
        return RegimeDescriptorCriterionResult(
            criterion_name,
            metric_name,
            observed,
            operator,
            None,
            None,
            False,
            _weight(settings, criterion_name, factor=weight_factor),
            "neutral",
            f"{criterion_name}: disabled",
            diagnostics,
        )
    if observed is None:
        return RegimeDescriptorCriterionResult(
            criterion_name,
            metric_name,
            None,
            operator,
            threshold,
            None,
            False,
            _weight(settings, criterion_name, factor=weight_factor),
            "neutral",
            f"{criterion_name}: metric_unavailable",
            diagnostics,
        )
    passed = _evaluate_operator(observed, operator, threshold, settings.numerical_tolerance)
    return RegimeDescriptorCriterionResult(
        criterion_name=criterion_name,
        metric_name=metric_name,
        observed_value=observed,
        operator=operator,
        threshold=threshold,
        passed=passed,
        applicable=True,
        weight=_weight(settings, criterion_name, factor=weight_factor),
        support_direction="support" if passed else "oppose",
        reason=f"{criterion_name}: {'passed' if passed else 'failed'}",
        diagnostics=diagnostics,
    )


def _evaluate_operator(
    observed: float,
    operator: str,
    threshold: float | tuple[float, float] | None,
    tolerance: float,
) -> bool:
    if operator == "present":
        return True
    if threshold is None:
        raise ValueError("threshold is required for this operator.")
    if operator == ">=":
        return observed + tolerance >= float(threshold)
    if operator == "<=":
        return observed <= float(threshold) + tolerance
    if operator == "abs>=":
        return abs(observed) + tolerance >= float(threshold)
    if operator == "abs<=":
        return abs(observed) <= float(threshold) + tolerance
    if operator == "==":
        return abs(observed - float(threshold)) <= tolerance
    if operator == "between":
        lower, upper = threshold  # type: ignore[misc]
        return float(lower) - tolerance <= observed <= float(upper) + tolerance
    raise ValueError("operator is not recognized.")


def _score(
    descriptor_name: str,
    criteria: tuple[RegimeDescriptorCriterionResult, ...],
    settings: ResponseRegimeDescriptorSettings,
    diagnostics: tuple[str, ...] = (),
) -> RegimeDescriptorScore:
    support_score = sum(criterion.weight for criterion in criteria if criterion.applicable and criterion.passed)
    opposition_score = sum(criterion.weight for criterion in criteria if criterion.applicable and criterion.passed is False)
    available_weight = support_score + opposition_score
    if available_weight > 0.0:
        support_fraction = support_score / available_weight
        opposition_fraction = opposition_score / available_weight
    else:
        support_fraction = None
        opposition_fraction = None
    available_weight_sufficient = (
        available_weight + settings.score_tie_tolerance >= settings.minimum_available_weight_for_descriptor
    )
    selected = (
        available_weight_sufficient
        and support_fraction is not None
        and opposition_fraction is not None
        and support_fraction + settings.score_tie_tolerance >= settings.minimum_support_fraction_for_descriptor
        and opposition_fraction <= settings.maximum_opposition_fraction_for_descriptor + settings.score_tie_tolerance
    )
    indeterminate = not available_weight_sufficient
    return RegimeDescriptorScore(
        descriptor_name=descriptor_name,
        support_score=support_score,
        opposition_score=opposition_score,
        available_weight=available_weight,
        support_fraction=support_fraction,
        opposition_fraction=opposition_fraction,
        selected=selected,
        indeterminate=indeterminate,
        diagnostics=diagnostics,
    )


def _evaluate_all_descriptors(
    summary: DynamicConditionSpectralSummary,
    settings: ResponseRegimeDescriptorSettings,
    metrics: dict[str, AggregatedMetric],
    limitations: list[str],
) -> tuple[DescriptorEvaluation, ...]:
    density_factor = _density_weight_factor(summary, settings, metrics, limitations)
    evaluations: list[DescriptorEvaluation] = []
    for dimension, descriptor, criteria in (
        ("structure", "discrete_line_dominated", _line_dominated_criteria(metrics, settings, density_factor)),
        ("structure", "broadband_dominated", _broadband_criteria(metrics, settings, density_factor)),
        ("structure", "dense_spectrum", _dense_criteria(metrics, settings, density_factor)),
        ("structure", "mixed_line_and_continuum", _mixed_criteria(metrics, settings)),
        ("temporal_evolution", "broadband_to_tonal", _broadband_to_tonal_criteria(metrics, settings, density_factor)),
        ("temporal_evolution", "tonal_to_broadband", _tonal_to_broadband_criteria(metrics, settings, density_factor)),
        ("temporal_evolution", "progressive_spectral_densification", _densification_criteria(metrics, settings, density_factor)),
        ("temporal_evolution", "progressive_spectral_sparsification", _sparsification_criteria(metrics, settings, density_factor)),
        ("temporal_evolution", "stable_spectral_character", _stable_temporal_criteria(metrics, settings, density_factor)),
        ("line_identity", "line_identity_preserved", _line_identity_criteria(metrics, settings, summary)),
    ):
        score = _score(descriptor, criteria, settings)
        evaluations.append(
            DescriptorEvaluation(
                dimension=dimension,
                descriptor_name=descriptor,
                score=score,
                criteria=criteria,
                diagnostics=(),
            )
        )
    return tuple(evaluations)


def _line_dominated_criteria(
    metrics: dict[str, AggregatedMetric],
    settings: ResponseRegimeDescriptorSettings,
    density_factor: float,
) -> tuple[RegimeDescriptorCriterionResult, ...]:
    return (
        _criterion(metrics, settings, "line_flatness_low", "global_spectral_flatness", "<=", settings.maximum_flatness_for_line_dominated),
        _criterion(metrics, settings, "line_entropy_low", "global_spectral_entropy", "<=", settings.maximum_entropy_for_line_dominated),
        _criterion(metrics, settings, "line_tonal_fraction_high", "global_tonal_energy_fraction", ">=", settings.minimum_tonal_fraction_for_line_dominated),
        _criterion(metrics, settings, "line_spectral_crest_high", "global_spectral_crest_factor", ">=", settings.minimum_spectral_crest_for_line_dominated),
        _criterion(metrics, settings, "line_peak_density_sparse", "global_peak_density_per_hz", "<=", settings.maximum_peak_density_for_sparse, weight_factor=density_factor),
        _criterion(metrics, settings, "line_occupied_fraction_narrow", "global_occupied_frequency_fraction", "<=", settings.maximum_occupied_fraction_for_narrow),
    )


def _broadband_criteria(
    metrics: dict[str, AggregatedMetric],
    settings: ResponseRegimeDescriptorSettings,
    density_factor: float,
) -> tuple[RegimeDescriptorCriterionResult, ...]:
    return (
        _criterion(metrics, settings, "broadband_flatness_high", "global_spectral_flatness", ">=", settings.minimum_flatness_for_broadband),
        _criterion(metrics, settings, "broadband_entropy_high", "global_spectral_entropy", ">=", settings.minimum_entropy_for_broadband),
        _criterion(metrics, settings, "broadband_tonal_fraction_low", "global_tonal_energy_fraction", "<=", settings.maximum_tonal_fraction_for_broadband),
        _criterion(metrics, settings, "broadband_residual_fraction_high", "global_residual_energy_fraction", ">=", None if settings.maximum_tonal_fraction_for_broadband is None else 1.0 - settings.maximum_tonal_fraction_for_broadband),
        _criterion(metrics, settings, "broadband_occupied_fraction_high", "global_occupied_frequency_fraction", ">=", settings.minimum_occupied_fraction_for_broadband),
        _criterion(metrics, settings, "broadband_spectral_crest_low", "global_spectral_crest_factor", "<=", settings.maximum_spectral_crest_for_broadband),
        _criterion(metrics, settings, "broadband_peak_density_dense", "global_peak_density_per_hz", ">=", settings.minimum_peak_density_for_dense, weight_factor=density_factor),
    )


def _dense_criteria(
    metrics: dict[str, AggregatedMetric],
    settings: ResponseRegimeDescriptorSettings,
    density_factor: float,
) -> tuple[RegimeDescriptorCriterionResult, ...]:
    return (
        _criterion(metrics, settings, "dense_peak_density_high", "global_peak_density_per_hz", ">=", settings.minimum_peak_density_for_dense, weight_factor=density_factor),
        _criterion(metrics, settings, "dense_peak_count_high", "global_significant_peak_count", ">=", settings.minimum_peak_count_for_dense, weight_factor=density_factor),
        _criterion(metrics, settings, "dense_peak_spacing_low", "global_median_peak_spacing_hz", "<=", settings.maximum_median_peak_spacing_for_dense_hz, weight_factor=density_factor),
        _criterion(metrics, settings, "dense_occupied_fraction_high", "global_occupied_frequency_fraction", ">=", settings.minimum_occupied_fraction_for_broadband),
        _criterion(metrics, settings, "dense_tonal_fraction_present", "global_tonal_energy_fraction", ">=", settings.minimum_tonal_fraction_for_mixed),
    )


def _mixed_criteria(
    metrics: dict[str, AggregatedMetric],
    settings: ResponseRegimeDescriptorSettings,
) -> tuple[RegimeDescriptorCriterionResult, ...]:
    tonal_bounds = None
    if settings.minimum_tonal_fraction_for_mixed is not None and settings.maximum_tonal_fraction_for_mixed is not None:
        tonal_bounds = (settings.minimum_tonal_fraction_for_mixed, settings.maximum_tonal_fraction_for_mixed)
    flatness_bounds = None
    if settings.minimum_flatness_for_mixed is not None and settings.maximum_flatness_for_mixed is not None:
        flatness_bounds = (settings.minimum_flatness_for_mixed, settings.maximum_flatness_for_mixed)
    return (
        _criterion(metrics, settings, "mixed_tonal_fraction_intermediate", "global_tonal_energy_fraction", "between", tonal_bounds),
        _criterion(metrics, settings, "mixed_residual_fraction_present", "global_residual_energy_fraction", ">=", settings.minimum_residual_fraction_for_mixed),
        _criterion(metrics, settings, "mixed_flatness_intermediate", "global_spectral_flatness", "between", flatness_bounds),
        _criterion(metrics, settings, "mixed_density_intermediate_or_dense", "global_peak_density_per_hz", ">=", settings.maximum_peak_density_for_sparse),
    )


def _broadband_to_tonal_criteria(
    metrics: dict[str, AggregatedMetric],
    settings: ResponseRegimeDescriptorSettings,
    density_factor: float,
) -> tuple[RegimeDescriptorCriterionResult, ...]:
    return (
        _criterion(metrics, settings, "b2t_flatness_drop", "region_late_minus_early_spectral_flatness", "<=", _negative(settings.minimum_flatness_drop_for_broadband_to_tonal)),
        _criterion(metrics, settings, "b2t_entropy_drop", "region_late_minus_early_spectral_entropy", "<=", _negative(settings.minimum_entropy_drop_for_broadband_to_tonal)),
        _criterion(metrics, settings, "b2t_tonal_fraction_increase", "region_late_minus_early_tonal_energy_fraction", ">=", settings.minimum_tonal_fraction_increase),
        _criterion(metrics, settings, "b2t_residual_fraction_drop", "region_late_minus_early_residual_energy_fraction", "<=", _negative(settings.minimum_tonal_fraction_increase)),
        _criterion(metrics, settings, "b2t_bandwidth_drop", "region_late_minus_early_occupied_bandwidth_hz", "<=", _negative(settings.minimum_bandwidth_change)),
        _criterion(metrics, settings, "b2t_change_points", "time_change_point_count", ">=", float(settings.minimum_persistent_change_point_count) if settings.minimum_persistent_change_point_count is not None else None),
        _criterion(metrics, settings, "b2t_density_drop", "region_late_minus_early_peak_density_per_hz", "<=", _negative(settings.minimum_density_change), weight_factor=density_factor),
    )


def _tonal_to_broadband_criteria(
    metrics: dict[str, AggregatedMetric],
    settings: ResponseRegimeDescriptorSettings,
    density_factor: float,
) -> tuple[RegimeDescriptorCriterionResult, ...]:
    return (
        _criterion(metrics, settings, "t2b_flatness_increase", "region_late_minus_early_spectral_flatness", ">=", settings.minimum_flatness_drop_for_broadband_to_tonal),
        _criterion(metrics, settings, "t2b_entropy_increase", "region_late_minus_early_spectral_entropy", ">=", settings.minimum_entropy_drop_for_broadband_to_tonal),
        _criterion(metrics, settings, "t2b_tonal_fraction_drop", "region_late_minus_early_tonal_energy_fraction", "<=", _negative(settings.minimum_tonal_fraction_increase)),
        _criterion(metrics, settings, "t2b_residual_fraction_increase", "region_late_minus_early_residual_energy_fraction", ">=", settings.minimum_tonal_fraction_increase),
        _criterion(metrics, settings, "t2b_bandwidth_increase", "region_late_minus_early_occupied_bandwidth_hz", ">=", settings.minimum_bandwidth_change),
        _criterion(metrics, settings, "t2b_density_increase", "region_late_minus_early_peak_density_per_hz", ">=", settings.minimum_density_change, weight_factor=density_factor),
    )


def _densification_criteria(
    metrics: dict[str, AggregatedMetric],
    settings: ResponseRegimeDescriptorSettings,
    density_factor: float,
) -> tuple[RegimeDescriptorCriterionResult, ...]:
    return (
        _criterion(metrics, settings, "densification_density_increase", "region_late_minus_early_peak_density_per_hz", ">=", settings.minimum_density_change, weight_factor=density_factor),
        _criterion(metrics, settings, "densification_peak_count_increase", "region_late_minus_early_significant_peak_count", ">=", 1.0, weight_factor=density_factor),
        _criterion(metrics, settings, "densification_bandwidth_increase", "region_late_minus_early_occupied_bandwidth_hz", ">=", settings.minimum_bandwidth_change),
    )


def _sparsification_criteria(
    metrics: dict[str, AggregatedMetric],
    settings: ResponseRegimeDescriptorSettings,
    density_factor: float,
) -> tuple[RegimeDescriptorCriterionResult, ...]:
    return (
        _criterion(metrics, settings, "sparsification_density_drop", "region_late_minus_early_peak_density_per_hz", "<=", _negative(settings.minimum_density_change), weight_factor=density_factor),
        _criterion(metrics, settings, "sparsification_peak_count_drop", "region_late_minus_early_significant_peak_count", "<=", -1.0, weight_factor=density_factor),
        _criterion(metrics, settings, "sparsification_bandwidth_drop", "region_late_minus_early_occupied_bandwidth_hz", "<=", _negative(settings.minimum_bandwidth_change)),
    )


def _stable_temporal_criteria(
    metrics: dict[str, AggregatedMetric],
    settings: ResponseRegimeDescriptorSettings,
    density_factor: float,
) -> tuple[RegimeDescriptorCriterionResult, ...]:
    return (
        _criterion(metrics, settings, "stable_flatness_change_small", "region_late_minus_early_spectral_flatness", "abs<=", settings.maximum_flatness_change_for_stable),
        _criterion(metrics, settings, "stable_entropy_change_small", "region_late_minus_early_spectral_entropy", "abs<=", settings.maximum_entropy_change_for_stable),
        _criterion(metrics, settings, "stable_tonal_fraction_change_small", "region_late_minus_early_tonal_energy_fraction", "abs<=", settings.maximum_tonal_fraction_change_for_stable),
        _criterion(metrics, settings, "stable_density_change_small", "region_late_minus_early_peak_density_per_hz", "abs<=", settings.maximum_density_change_for_stable, weight_factor=density_factor),
        _criterion(metrics, settings, "stable_bandwidth_change_small", "region_late_minus_early_occupied_bandwidth_hz", "abs<=", settings.maximum_bandwidth_change_for_stable),
    )


def _line_identity_criteria(
    metrics: dict[str, AggregatedMetric],
    settings: ResponseRegimeDescriptorSettings,
    summary: DynamicConditionSpectralSummary,
) -> tuple[RegimeDescriptorCriterionResult, ...]:
    return (
        _criterion(metrics, settings, "identity_temporal_coverage_high", "time_temporal_coverage_fraction", ">=", settings.minimum_valid_frame_coverage),
        _criterion(metrics, settings, "identity_centroid_slope_small", "time_spectral_centroid_hz_slope_per_s", "abs<=", settings.maximum_centroid_slope_for_preserved_identity_hz_per_s),
        _criterion(metrics, settings, "identity_late_tonal_fraction_high", "late_tonal_energy_fraction", ">=", settings.minimum_tonal_fraction_for_line_identity),
        _summary_count_criterion(summary, settings, "identity_valid_repeats", "valid_repeat_count", float(settings.minimum_valid_repeat_count)),
        _variability_criterion(metrics, settings, "identity_variability_low"),
    )


def _summary_count_criterion(
    summary: DynamicConditionSpectralSummary,
    settings: ResponseRegimeDescriptorSettings,
    criterion_name: str,
    metric_name: str,
    threshold: float,
) -> RegimeDescriptorCriterionResult:
    observed = float(summary.valid_repeat_count)
    passed = observed + settings.numerical_tolerance >= threshold
    return RegimeDescriptorCriterionResult(
        criterion_name,
        metric_name,
        observed,
        ">=",
        threshold,
        passed,
        True,
        _weight(settings, criterion_name),
        "support" if passed else "oppose",
        f"{criterion_name}: {'passed' if passed else 'failed'}",
    )


def _variability_criterion(
    metrics: dict[str, AggregatedMetric],
    settings: ResponseRegimeDescriptorSettings,
    criterion_name: str,
) -> RegimeDescriptorCriterionResult:
    if settings.maximum_within_condition_variability is None:
        return RegimeDescriptorCriterionResult(
            criterion_name,
            "within_condition_coefficient_of_variation",
            None,
            "<=",
            None,
            None,
            False,
            _weight(settings, criterion_name),
            "neutral",
            f"{criterion_name}: disabled",
        )
    cvs = [
        metric.coefficient_of_variation
        for metric in metrics.values()
        if metric.coefficient_of_variation is not None
    ]
    if not cvs:
        return RegimeDescriptorCriterionResult(
            criterion_name,
            "within_condition_coefficient_of_variation",
            None,
            "<=",
            settings.maximum_within_condition_variability,
            None,
            False,
            _weight(settings, criterion_name),
            "neutral",
            f"{criterion_name}: metric_unavailable",
        )
    observed = max(cvs)
    passed = observed <= settings.maximum_within_condition_variability + settings.numerical_tolerance
    return RegimeDescriptorCriterionResult(
        criterion_name,
        "within_condition_coefficient_of_variation",
        observed,
        "<=",
        settings.maximum_within_condition_variability,
        passed,
        True,
        _weight(settings, criterion_name),
        "support" if passed else "oppose",
        f"{criterion_name}: {'passed' if passed else 'failed'}",
    )


def _negative(value: float | None) -> float | None:
    return None if value is None else -value


def _base_limitations(
    summary: DynamicConditionSpectralSummary,
    settings: ResponseRegimeDescriptorSettings,
    metrics: dict[str, AggregatedMetric],
) -> tuple[str, ...]:
    limitations: list[str] = []
    if not summary.valid:
        limitations.append("condition_summary_invalid")
    if summary.valid_repeat_count < settings.minimum_valid_repeat_count:
        limitations.append("insufficient_valid_repeats")
    if (summary.clipped_repeat_fraction or 0.0) > 0.0:
        limitations.append("spectral_metrics_potentially_distorted_by_clipping")
    elif (summary.near_clipped_repeat_fraction or 0.0) > 0.0:
        limitations.append("spectral_metrics_potentially_affected_by_near_clipping")
    if settings.reject_clipped_conditions and (summary.clipped_repeat_fraction or 0.0) > 0.0:
        limitations.append("clipped_condition_rejected_by_configuration")
    snr = _metric_value(metrics, "excitation_signal_to_background_db")
    if settings.minimum_signal_to_background_db is not None:
        if snr is None:
            limitations.append("signal_to_background_unavailable")
        elif snr < settings.minimum_signal_to_background_db:
            limitations.append("low_signal_to_background_ratio")
    coverage = _metric_value(metrics, "time_temporal_coverage_fraction")
    if settings.minimum_valid_frame_coverage is not None:
        if coverage is None:
            limitations.append("temporal_coverage_unavailable")
        elif coverage < settings.minimum_valid_frame_coverage:
            limitations.append("low_valid_temporal_coverage")
    if summary.comparability_status:
        limitations.append("comparability_limitations_present")
    if settings.require_spectral_comparability and summary.comparability_status:
        limitations.append("spectral_comparability_required_but_unavailable")
    if _missing_fraction(metrics) > settings.maximum_missing_metric_fraction_for_moderate_confidence:
        limitations.append("many_descriptor_metrics_unavailable")
    if _max_cv(metrics) is not None and settings.maximum_within_condition_variability is not None:
        if _max_cv(metrics) > settings.maximum_within_condition_variability:
            limitations.append("high_within_condition_variability")
    return tuple(dict.fromkeys(limitations))


def _invalid_reason(
    summary: DynamicConditionSpectralSummary,
    settings: ResponseRegimeDescriptorSettings,
    limitations: list[str],
) -> str | None:
    if not summary.valid:
        return "condition_summary_invalid"
    if settings.reject_clipped_conditions and (summary.clipped_repeat_fraction or 0.0) > 0.0:
        return "clipped_condition_rejected_by_configuration"
    if settings.require_spectral_comparability and summary.comparability_status:
        return "spectral_comparability_required_but_unavailable"
    if not settings.allow_missing_metrics and "many_descriptor_metrics_unavailable" in limitations:
        return "missing_metrics_not_allowed"
    return None


def _missing_fraction(metrics: dict[str, AggregatedMetric]) -> float:
    if not metrics:
        return 1.0
    missing = sum(1 for metric in metrics.values() if not metric.valid)
    return missing / len(metrics)


def _max_cv(metrics: dict[str, AggregatedMetric]) -> float | None:
    cvs = [metric.coefficient_of_variation for metric in metrics.values() if metric.coefficient_of_variation is not None]
    return max(cvs) if cvs else None


def _density_weight_factor(
    summary: DynamicConditionSpectralSummary,
    settings: ResponseRegimeDescriptorSettings,
    metrics: dict[str, AggregatedMetric],
    limitations: list[str],
) -> float:
    resolution = _fingerprint_value(summary.global_spectral_fingerprint, "frequency_resolution_hz")
    spacing = _metric_value(metrics, "global_median_peak_spacing_hz")
    if resolution is None or spacing is None or spacing <= 0.0:
        return 1.0
    if spacing <= settings.density_resolution_limit_spacing_factor * resolution:
        limitations.append("density_metrics_resolution_limited")
        return settings.resolution_limited_density_weight_factor
    return 1.0


def _fingerprint_value(fingerprint: tuple[tuple[str, object | None], ...], key: str) -> float | None:
    value = dict(fingerprint).get(key)
    if isinstance(value, (int, float)) and isfinite(float(value)):
        return float(value)
    return None


def _select_structure_descriptor(evaluations: tuple[DescriptorEvaluation, ...]) -> str:
    scores = {evaluation.descriptor_name: evaluation.score for evaluation in evaluations if evaluation.dimension == "structure"}
    selected = {name for name, score in scores.items() if score.selected}
    if "mixed_line_and_continuum" in selected:
        return "mixed_line_and_continuum"
    if "discrete_line_dominated" in selected and "broadband_dominated" in selected:
        return "indeterminate"
    if "broadband_dominated" in selected:
        return "broadband_dominated"
    if "dense_spectrum" in selected:
        return "dense_spectrum"
    if "discrete_line_dominated" in selected:
        return "discrete_line_dominated"
    return "indeterminate"


def _select_temporal_descriptor(evaluations: tuple[DescriptorEvaluation, ...]) -> str:
    scores = {evaluation.descriptor_name: evaluation.score for evaluation in evaluations if evaluation.dimension == "temporal_evolution"}
    selected = {name for name, score in scores.items() if score.selected}
    if "broadband_to_tonal" in selected and "tonal_to_broadband" not in selected:
        return "broadband_to_tonal"
    if "tonal_to_broadband" in selected and "broadband_to_tonal" not in selected:
        return "tonal_to_broadband"
    if "progressive_spectral_densification" in selected and "progressive_spectral_sparsification" not in selected:
        return "progressive_spectral_densification"
    if "progressive_spectral_sparsification" in selected and "progressive_spectral_densification" not in selected:
        return "progressive_spectral_sparsification"
    if "stable_spectral_character" in selected and len(selected) == 1:
        return "stable_spectral_character"
    if selected:
        return "mixed_temporal_evolution"
    return "indeterminate"


def _select_line_identity_descriptor(evaluations: tuple[DescriptorEvaluation, ...]) -> str:
    score = next(
        evaluation.score
        for evaluation in evaluations
        if evaluation.descriptor_name == "line_identity_preserved"
    )
    if score.available_weight <= 0.0:
        return "not_evaluated"
    if score.selected:
        return "line_identity_preserved"
    if score.available_weight >= 2.0 and (score.support_fraction or 0.0) >= 0.35:
        return "line_identity_partially_preserved"
    return "line_identity_not_resolved"


def _score_present(scores: dict[str, RegimeDescriptorScore], name: str) -> bool:
    score = scores.get(name)
    return score is not None and score.available_weight > 0.0 and (score.support_fraction or 0.0) >= 0.4


def _metric_evidence_lists(
    evaluations: tuple[DescriptorEvaluation, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    supporting: list[str] = []
    conflicting: list[str] = []
    unavailable: list[str] = []
    for evaluation in evaluations:
        for criterion in evaluation.criteria:
            if not criterion.applicable:
                unavailable.append(criterion.metric_name)
            elif criterion.passed:
                supporting.append(criterion.metric_name)
            else:
                conflicting.append(criterion.metric_name)
    return tuple(dict.fromkeys(supporting)), tuple(dict.fromkeys(conflicting)), tuple(dict.fromkeys(unavailable))


def _confidence_descriptor(
    summary: DynamicConditionSpectralSummary,
    settings: ResponseRegimeDescriptorSettings,
    evaluations: tuple[DescriptorEvaluation, ...],
    limitations: list[str],
    unavailable_metrics: tuple[str, ...],
) -> Literal["high", "moderate", "low", "insufficient"]:
    if not summary.valid:
        return "insufficient"
    if settings.reject_clipped_conditions and (summary.clipped_repeat_fraction or 0.0) > 0.0:
        return "insufficient"
    if "low_signal_to_background_ratio" in limitations or "signal_to_background_unavailable" in limitations:
        return "low"
    if "low_valid_temporal_coverage" in limitations or "condition_summary_invalid" in limitations:
        return "low"
    if "spectral_metrics_potentially_affected_by_near_clipping" in limitations:
        return "low"
    if "spectral_comparability_required_but_unavailable" in limitations:
        return "insufficient"
    selected_scores = [evaluation.score for evaluation in evaluations if evaluation.score.selected]
    if not selected_scores:
        return "low" if unavailable_metrics else "insufficient"
    best_support = max(score.support_fraction or 0.0 for score in selected_scores)
    conflicts = sum(score.opposition_fraction or 0.0 for score in selected_scores)
    relevant_evaluations = tuple(evaluation for evaluation in evaluations if evaluation.score.selected)
    if not relevant_evaluations:
        relevant_evaluations = evaluations
    relevant_criteria = tuple(criterion for evaluation in relevant_evaluations for criterion in evaluation.criteria)
    missing_fraction = (
        sum(1 for criterion in relevant_criteria if not criterion.applicable)
        / max(1, len(relevant_criteria))
    )
    if (
        best_support >= settings.high_confidence_minimum_support_fraction
        and not limitations
        and missing_fraction <= 0.25
        and conflicts <= 0.25
    ):
        return "high"
    if (
        best_support >= settings.minimum_support_fraction_for_descriptor
        and missing_fraction <= settings.maximum_missing_metric_fraction_for_moderate_confidence
        and not any(item in limitations for item in (
            "spectral_metrics_potentially_distorted_by_clipping",
            "spectral_metrics_potentially_affected_by_near_clipping",
            "high_within_condition_variability",
            "density_metrics_resolution_limited",
        ))
    ):
        return "moderate"
    return "low"


def _descriptor_for_dimension(description: ResponseRegimeDescription, dimension: str) -> str:
    if dimension == "structure":
        return description.structure_descriptor
    if dimension == "temporal_evolution":
        return description.temporal_evolution_descriptor
    if dimension == "line_identity":
        return description.line_identity_descriptor
    raise ValueError("dimension is not recognized.")


def _sequence_changes(descriptors: tuple[str | None, ...]) -> tuple[str, ...]:
    changes: list[str] = []
    previous: str | None = None
    for descriptor in descriptors:
        if descriptor is None:
            continue
        if previous is None:
            previous = descriptor
            continue
        if descriptor == previous:
            changes.append("descriptor_preserved")
        else:
            changes.append("descriptor_changed")
            if descriptor in {"broadband_dominated", "broadband_to_tonal", "tonal_to_broadband"}:
                changes.append("became_more_broadband")
            if descriptor in {"discrete_line_dominated", "broadband_to_tonal", "line_identity_preserved"}:
                changes.append("became_more_tonal")
            if descriptor == "dense_spectrum":
                changes.append("became_denser")
            if descriptor in {"discrete_line_dominated", "progressive_spectral_sparsification"}:
                changes.append("became_sparser")
        previous = descriptor
    if not changes:
        changes.append("insufficient_evidence")
    return tuple(dict.fromkeys(changes))


def _stable_segments(descriptors: tuple[str | None, ...]) -> tuple[tuple[str, str, str], ...]:
    segments: list[tuple[str, str, str]] = []
    current_descriptor: str | None = None
    start_label: str | None = None
    previous_label: str | None = None
    for label, descriptor in zip(DYNAMIC_LABEL_ORDER, descriptors):
        if descriptor is None:
            continue
        if descriptor != current_descriptor:
            if current_descriptor is not None and start_label is not None and previous_label is not None:
                segments.append((current_descriptor, start_label, previous_label))
            current_descriptor = descriptor
            start_label = label
        previous_label = label
    if current_descriptor is not None and start_label is not None and previous_label is not None:
        segments.append((current_descriptor, start_label, previous_label))
    return tuple(segments)


def _emergent_patterns(
    descriptions: tuple[ResponseRegimeDescription, ...],
    sequences: tuple[RegimeDescriptorSequence, ...],
) -> tuple[EmergentResponsePattern, ...]:
    patterns: list[EmergentResponsePattern] = []
    by_label = {description.dynamic_label: description for description in descriptions}
    tonal_labels = tuple(
        description.dynamic_label
        for description in descriptions
        if description.structure_descriptor == "discrete_line_dominated"
    )
    if len(tonal_labels) >= 2:
        patterns.append(EmergentResponsePattern(
            "tonal_structure_preserved_over_multiple_conditions",
            tonal_labels[0],
            tonal_labels[-1],
            tonal_labels,
            ("global_spectral_flatness", "global_tonal_energy_fraction"),
            (),
            ("operational_pattern_not_physical_mode_family",),
        ))
    dense_labels = tuple(
        description.dynamic_label
        for description in descriptions
        if description.structure_descriptor in {"dense_spectrum", "broadband_dominated"}
    )
    if len(dense_labels) >= 2:
        patterns.append(EmergentResponsePattern(
            "increased_density_or_broadband_character_over_dynamic_order",
            dense_labels[0],
            dense_labels[-1],
            dense_labels,
            ("global_peak_density_per_hz", "global_occupied_frequency_fraction"),
            (),
            ("operational_pattern_not_nonlinear_transition",),
        ))
    broadband_to_tonal = tuple(
        description.dynamic_label
        for description in descriptions
        if description.temporal_evolution_descriptor == "broadband_to_tonal"
    )
    if broadband_to_tonal:
        patterns.append(EmergentResponsePattern(
            "broadband_attack_followed_by_more_tonal_tail",
            broadband_to_tonal[0],
            broadband_to_tonal[-1],
            broadband_to_tonal,
            ("region_late_minus_early_spectral_flatness", "region_late_minus_early_tonal_energy_fraction"),
            (),
            ("operational_change_point_not_regime_transition_proof",),
        ))
    low_confidence = tuple(
        description.dynamic_label
        for description in descriptions
        if description.confidence_descriptor in {"low", "insufficient"}
    )
    if low_confidence:
        patterns.append(EmergentResponsePattern(
            "increased_uncertainty_or_limited_evidence",
            low_confidence[0],
            low_confidence[-1],
            low_confidence,
            ("clipping", "snr", "missing_metrics", "resolution"),
            tuple(dict.fromkeys(limitation for label in low_confidence for limitation in by_label[label].limitations)),
            ("uncertainty_pattern_is_not_physical_interpretation",),
        ))
    return tuple(patterns)
