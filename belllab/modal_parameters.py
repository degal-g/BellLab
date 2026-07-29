"""Operational parameter estimates for modal hypotheses.

This layer summarizes values already present in :class:`ModalHypothesis`
objects. It never reads audio, recomputes FFT/STFT/tracking/candidates, creates
new matches, closes gaps, resolves split/merge contexts, or promotes any result
to ``ModalMode``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha1
import json
from math import exp, isclose, isfinite, log, sqrt
from random import Random

from belllab.dynamic_comparison import DYNAMIC_LABEL_ORDER
from belllab.modal_hypotheses import (
    ModalHypothesis,
    ModalHypothesisResult,
    ModalHypothesisStatus,
)
from belllab.within_condition import CandidateReference


_DYNAMIC_LABEL_INDEX = {
    label: index for index, label in enumerate(DYNAMIC_LABEL_ORDER)
}
_MAD_NORMAL_CONSISTENCY_FACTOR = 1.4826


class ModalParameterEstimateStatus(str, Enum):
    """Mutually exclusive states for an operational parameter estimate."""

    VALID = "valid"
    VALID_WITH_RESERVATIONS = "valid_with_reservations"
    PARTIAL = "partial"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    INVALID_INPUT = "invalid_input"


class ModalParameterEstimateReason(str, Enum):
    """Typed reasons separated by support, reservation, insufficiency or invalidity."""

    SUFFICIENT_FREQUENCY_EVIDENCE = "sufficient_frequency_evidence"
    SUFFICIENT_DECAY_EVIDENCE = "sufficient_decay_evidence"
    SUFFICIENT_CONDITION_COVERAGE = "sufficient_condition_coverage"
    ACCEPTED_MODAL_HYPOTHESIS = "accepted_modal_hypothesis"
    HYPOTHESIS_WITH_RESERVATIONS = "hypothesis_with_reservations"
    INCONCLUSIVE_HYPOTHESIS = "inconclusive_hypothesis"
    REJECTED_HYPOTHESIS = "rejected_hypothesis"
    INSUFFICIENT_FREQUENCY_VALUES = "insufficient_frequency_values"
    INSUFFICIENT_DECAY_VALUES = "insufficient_decay_values"
    MISSING_FREQUENCY_UNCERTAINTY = "missing_frequency_uncertainty"
    MISSING_DECAY_UNCERTAINTY = "missing_decay_uncertainty"
    EXCESSIVE_FREQUENCY_DISPERSION = "excessive_frequency_dispersion"
    EXCESSIVE_DECAY_DISPERSION = "excessive_decay_dispersion"
    AMBIGUOUS_SOURCE_MATCH = "ambiguous_source_match"
    NEAR_THRESHOLD_SOURCE_MATCH = "near_threshold_source_match"
    POSSIBLE_SPLIT_CONTEXT = "possible_split_context"
    POSSIBLE_MERGE_CONTEXT = "possible_merge_context"
    REJECTED_CANDIDATE_PRESENT = "rejected_candidate_present"
    INVALID_FREQUENCY_VALUE = "invalid_frequency_value"
    INVALID_DECAY_VALUE = "invalid_decay_value"
    INVALID_HYPOTHESIS = "invalid_hypothesis"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    INVALID_WEIGHT_VALUE = "invalid_weight_value"
    MISSING_WEIGHT_VALUE = "missing_weight_value"
    INSUFFICIENT_WEIGHT_VALUES = "insufficient_weight_values"


class ParameterLocationMethod(str, Enum):
    """Explicit location policies for representative parameter values."""

    ARITHMETIC_MEAN = "arithmetic_mean"
    MEDIAN = "median"
    WEIGHTED_MEAN = "weighted_mean"
    WEIGHTED_MEDIAN = "weighted_median"
    GEOMETRIC_MEAN = "geometric_mean"
    GEOMETRIC_MEDIAN = "geometric_median"


class ParameterWeightingMethod(str, Enum):
    """Explicit non-negative weighting policies based on existing diagnostics."""

    UNIFORM = "uniform"
    TRACKING_COVERAGE = "tracking_coverage"
    FREQUENCY_FIT_QUALITY = "frequency_fit_quality"
    AMPLITUDE_FIT_QUALITY = "amplitude_fit_quality"
    INVERSE_ASSOCIATION_COST = "inverse_association_cost"
    COMBINED_QUALITY_COVERAGE = "combined_quality_coverage"


class ParameterUncertaintyMethod(str, Enum):
    """Operational uncertainty summaries, not complete physical confidence intervals."""

    DISABLED = "disabled"
    SAMPLE_STANDARD_DEVIATION = "sample_standard_deviation"
    STANDARD_ERROR = "standard_error"
    SCALED_MAD = "scaled_mad"
    BOOTSTRAP_PERCENTILE = "bootstrap_percentile"
    CONSERVATIVE = "conservative"
    LOG_STANDARD_DEVIATION = "log_standard_deviation"
    LOG_STANDARD_ERROR = "log_standard_error"
    LOG_SCALED_MAD = "log_scaled_mad"
    LOG_BOOTSTRAP_PERCENTILE = "log_bootstrap_percentile"


class FiniteValuePolicy(str, Enum):
    """Policy for non-finite or non-positive source values."""

    INVALIDATE = "invalidate"
    EXCLUDE_WITH_DIAGNOSTIC = "exclude_with_diagnostic"


@dataclass(frozen=True, slots=True)
class ModalParameterEstimationSettings:
    """Conservative, explicit settings for modal-parameter estimation."""

    allow_accepted_hypotheses: bool = True
    allow_accepted_with_reservations: bool = True
    allow_inconclusive_hypotheses: bool = False
    allow_rejected_hypotheses_for_audit: bool = False
    allow_insufficient_evidence_hypotheses_for_audit: bool = False

    minimum_frequency_value_count: int = 2
    frequency_location_method: ParameterLocationMethod = ParameterLocationMethod.ARITHMETIC_MEAN
    frequency_weighting_method: ParameterWeightingMethod = ParameterWeightingMethod.UNIFORM
    frequency_uncertainty_method: ParameterUncertaintyMethod = ParameterUncertaintyMethod.CONSERVATIVE
    maximum_frequency_coefficient_of_variation: float | None = 0.02
    maximum_frequency_relative_range: float | None = 0.05
    allow_missing_frequency_uncertainty: bool = True

    minimum_tau_value_count: int = 2
    tau_location_method: ParameterLocationMethod = ParameterLocationMethod.GEOMETRIC_MEAN
    tau_weighting_method: ParameterWeightingMethod = ParameterWeightingMethod.UNIFORM
    tau_uncertainty_method: ParameterUncertaintyMethod = ParameterUncertaintyMethod.LOG_STANDARD_DEVIATION
    maximum_log_tau_range: float | None = 0.5
    maximum_log_tau_standard_deviation: float | None = 0.25
    allow_missing_tau: bool = True
    allow_missing_tau_uncertainty: bool = True

    reserve_ambiguous_matches: bool = True
    reserve_near_threshold_matches: bool = True
    reserve_possible_split_context: bool = True
    reserve_possible_merge_context: bool = True

    uncertainty_confidence_level: float = 0.95
    bootstrap_sample_count: int = 1000
    bootstrap_random_seed: int | None = 0
    minimum_positive_value: float = 1e-12
    finite_value_policy: FiniteValuePolicy = FiniteValuePolicy.INVALIDATE

    def __post_init__(self) -> None:
        for name in (
            "frequency_location_method",
            "tau_location_method",
        ):
            _coerce_enum(self, name, ParameterLocationMethod)
        for name in (
            "frequency_weighting_method",
            "tau_weighting_method",
        ):
            _coerce_enum(self, name, ParameterWeightingMethod)
        for name in (
            "frequency_uncertainty_method",
            "tau_uncertainty_method",
        ):
            _coerce_enum(self, name, ParameterUncertaintyMethod)
        _coerce_enum(self, "finite_value_policy", FiniteValuePolicy)
        for name in (
            "allow_accepted_hypotheses",
            "allow_accepted_with_reservations",
            "allow_inconclusive_hypotheses",
            "allow_rejected_hypotheses_for_audit",
            "allow_insufficient_evidence_hypotheses_for_audit",
            "allow_missing_frequency_uncertainty",
            "allow_missing_tau",
            "allow_missing_tau_uncertainty",
            "reserve_ambiguous_matches",
            "reserve_near_threshold_matches",
            "reserve_possible_split_context",
            "reserve_possible_merge_context",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")
        for name in ("minimum_frequency_value_count", "minimum_tau_value_count"):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be positive.")
        for name in (
            "maximum_frequency_coefficient_of_variation",
            "maximum_frequency_relative_range",
            "maximum_log_tau_range",
            "maximum_log_tau_standard_deviation",
        ):
            _finite_optional(getattr(self, name), name, nonnegative=True)
        if (
            not isfinite(self.uncertainty_confidence_level)
            or not 0.0 < self.uncertainty_confidence_level < 1.0
        ):
            raise ValueError("uncertainty_confidence_level must be finite and in (0, 1).")
        if self.bootstrap_sample_count <= 0:
            raise ValueError("bootstrap_sample_count must be positive.")
        if self.bootstrap_random_seed is not None and not isinstance(self.bootstrap_random_seed, int):
            raise ValueError("bootstrap_random_seed must be an int or None.")
        if (
            not isfinite(self.minimum_positive_value)
            or self.minimum_positive_value <= 0.0
        ):
            raise ValueError("minimum_positive_value must be finite and positive.")


@dataclass(frozen=True, slots=True)
class ModalFrequencyEstimate:
    values_hz: tuple[float, ...]
    condition_labels: tuple[str, ...]
    candidate_ids: tuple[int, ...]
    recording_ids: tuple[str, ...]
    source_track_ids: tuple[int, ...]
    frequency_fit_rmse_values_hz: tuple[float | None, ...]
    source_frequency_drifts_hz: tuple[float | None, ...]
    coverage_values: tuple[float | None, ...]
    weights: tuple[float | None, ...]
    normalized_weights: tuple[float | None, ...]
    location_method: ParameterLocationMethod
    weighting_method: ParameterWeightingMethod
    representative_frequency_hz: float | None
    minimum_frequency_hz: float | None
    maximum_frequency_hz: float | None
    frequency_range_hz: float | None
    relative_frequency_range: float | None
    frequency_mean_hz: float | None
    frequency_median_hz: float | None
    frequency_standard_deviation_hz: float | None
    frequency_mad_hz: float | None
    frequency_coefficient_of_variation: float | None
    available_value_count: int
    missing_value_count: int
    valid: bool
    passes_dispersion_limits: bool
    reasons: tuple[ModalParameterEstimateReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _enum(self.location_method, ParameterLocationMethod, "location_method")
        _enum(self.weighting_method, ParameterWeightingMethod, "weighting_method")
        aligned = (
            self.condition_labels,
            self.candidate_ids,
            self.recording_ids,
            self.source_track_ids,
            self.frequency_fit_rmse_values_hz,
            self.source_frequency_drifts_hz,
            self.coverage_values,
            self.weights,
            self.normalized_weights,
        )
        if any(len(values) != len(self.values_hz) for values in aligned):
            raise ValueError("frequency estimate source vectors must align.")
        if self.available_value_count != len(self.values_hz):
            raise ValueError("available_value_count must match values_hz.")
        if self.missing_value_count < 0:
            raise ValueError("missing_value_count must not be negative.")
        if any(not isfinite(value) or value <= 0.0 for value in self.values_hz):
            raise ValueError("frequency values must be finite and positive.")
        _strings(self.condition_labels, "frequency condition labels", allow_empty=True)
        _strings(self.recording_ids, "frequency recording ids", allow_empty=True)
        if any(value < 0 for value in self.candidate_ids):
            raise ValueError("frequency candidate IDs must not be negative.")
        if any(value < 0 for value in self.source_track_ids):
            raise ValueError("frequency candidate and track IDs must not be negative.")
        for value in self.frequency_fit_rmse_values_hz:
            _finite_optional(value, "frequency fit RMSE", nonnegative=True)
        for value in self.source_frequency_drifts_hz:
            _finite_optional(value, "source frequency drift")
        for value in self.coverage_values:
            _fraction(value, "coverage value")
        _validate_weights(self.weights, self.normalized_weights, require_normalized_sum=self.valid)
        for name in (
            "representative_frequency_hz",
            "minimum_frequency_hz",
            "maximum_frequency_hz",
            "frequency_range_hz",
            "relative_frequency_range",
            "frequency_mean_hz",
            "frequency_median_hz",
            "frequency_standard_deviation_hz",
            "frequency_mad_hz",
            "frequency_coefficient_of_variation",
        ):
            _finite_optional(getattr(self, name), name, nonnegative=True)
        if self.valid and self.representative_frequency_hz is None:
            raise ValueError("valid frequency estimate requires a representative frequency.")
        if self.representative_frequency_hz is not None and self.minimum_frequency_hz is not None and self.maximum_frequency_hz is not None:
            if not self.minimum_frequency_hz <= self.representative_frequency_hz <= self.maximum_frequency_hz:
                raise ValueError("representative frequency must lie within min/max.")
        _reason_tuple(self.reasons, "frequency estimate reasons")
        _strings(self.diagnostics, "frequency estimate diagnostics", allow_empty=True)


@dataclass(frozen=True, slots=True)
class ModalFrequencyTrajectoryEstimate:
    condition_labels: tuple[str, ...]
    frequencies_hz: tuple[float, ...]
    signed_step_changes_hz: tuple[float, ...] | None
    signed_step_relative_changes: tuple[float, ...] | None
    step_count: int
    total_signed_change_hz: float | None
    total_absolute_change_hz: float | None
    total_relative_symmetric_change: float | None
    mean_signed_step_change_hz: float | None
    mean_absolute_step_change_hz: float | None
    maximum_absolute_step_change_hz: float | None
    up_step_count: int
    down_step_count: int
    preserved_step_count: int
    indeterminate_step_count: int
    linear_slope_hz_per_condition_step: float | None
    linear_fit_intercept_hz: float | None
    linear_fit_rmse_hz: float | None
    linear_fit_r_squared: float | None
    valid: bool
    reasons: tuple[ModalParameterEstimateReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.condition_labels) != len(self.frequencies_hz):
            raise ValueError("trajectory labels and frequencies must align.")
        _strings(self.condition_labels, "trajectory labels", allow_empty=True)
        if any(not isfinite(value) or value <= 0.0 for value in self.frequencies_hz):
            raise ValueError("trajectory frequencies must be finite and positive.")
        if self.step_count != max(0, len(self.frequencies_hz) - 1):
            raise ValueError("step_count must match trajectory length.")
        if self.step_count < 1:
            empty_fields = (
                self.signed_step_changes_hz,
                self.signed_step_relative_changes,
                self.total_signed_change_hz,
                self.total_absolute_change_hz,
                self.total_relative_symmetric_change,
                self.mean_signed_step_change_hz,
                self.mean_absolute_step_change_hz,
                self.maximum_absolute_step_change_hz,
                self.linear_slope_hz_per_condition_step,
                self.linear_fit_intercept_hz,
                self.linear_fit_rmse_hz,
                self.linear_fit_r_squared,
            )
            if any(value is not None for value in empty_fields):
                raise ValueError("single-point trajectories must keep trajectory metrics as None.")
        else:
            if self.signed_step_changes_hz is None or self.signed_step_relative_changes is None:
                raise ValueError("multi-point trajectories require step vectors.")
            if len(self.signed_step_changes_hz) != self.step_count or len(self.signed_step_relative_changes) != self.step_count:
                raise ValueError("trajectory step vectors must match step_count.")
            if any(not isfinite(value) for value in self.signed_step_changes_hz + self.signed_step_relative_changes):
                raise ValueError("trajectory step vectors must be finite.")
            for name in (
                "total_signed_change_hz",
                "total_absolute_change_hz",
                "total_relative_symmetric_change",
                "mean_signed_step_change_hz",
                "mean_absolute_step_change_hz",
                "maximum_absolute_step_change_hz",
                "linear_slope_hz_per_condition_step",
                "linear_fit_intercept_hz",
                "linear_fit_rmse_hz",
                "linear_fit_r_squared",
            ):
                _finite_optional(getattr(self, name), name)
        for name in ("up_step_count", "down_step_count", "preserved_step_count", "indeterminate_step_count"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative.")
        _reason_tuple(self.reasons, "frequency trajectory reasons")
        _strings(self.diagnostics, "frequency trajectory diagnostics", allow_empty=True)


@dataclass(frozen=True, slots=True)
class ModalFrequencyUncertainty:
    method: ParameterUncertaintyMethod
    standard_uncertainty_hz: float | None
    lower_bound_hz: float | None
    upper_bound_hz: float | None
    confidence_level: float
    sample_count: int
    bootstrap_sample_count: int | None
    random_seed: int | None
    individual_uncertainties_hz: tuple[float | None, ...]
    dispersion_component_hz: float | None
    measurement_component_hz: float | None
    valid: bool
    reasons: tuple[ModalParameterEstimateReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _enum(self.method, ParameterUncertaintyMethod, "frequency uncertainty method")
        if self.sample_count < 0:
            raise ValueError("sample_count must not be negative.")
        if self.bootstrap_sample_count is not None and self.bootstrap_sample_count <= 0:
            raise ValueError("bootstrap_sample_count must be positive when present.")
        if not isfinite(self.confidence_level) or not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be finite and in (0, 1).")
        for name in (
            "standard_uncertainty_hz",
            "lower_bound_hz",
            "upper_bound_hz",
            "dispersion_component_hz",
            "measurement_component_hz",
        ):
            _finite_optional(getattr(self, name), name, nonnegative=True)
        if self.lower_bound_hz is not None and self.upper_bound_hz is not None:
            if self.lower_bound_hz > self.upper_bound_hz:
                raise ValueError("frequency uncertainty bounds are inverted.")
        for value in self.individual_uncertainties_hz:
            _finite_optional(value, "individual frequency uncertainty", nonnegative=True)
        _reason_tuple(self.reasons, "frequency uncertainty reasons")
        _strings(self.diagnostics, "frequency uncertainty diagnostics", allow_empty=True)


@dataclass(frozen=True, slots=True)
class ModalDecayEstimate:
    tau_values_s: tuple[float, ...]
    condition_labels: tuple[str, ...]
    candidate_ids: tuple[int, ...]
    recording_ids: tuple[str, ...]
    source_track_ids: tuple[int, ...]
    weights: tuple[float | None, ...]
    normalized_weights: tuple[float | None, ...]
    location_method: ParameterLocationMethod
    weighting_method: ParameterWeightingMethod
    representative_tau_s: float | None
    minimum_tau_s: float | None
    maximum_tau_s: float | None
    log_tau_values: tuple[float, ...]
    log_tau_mean: float | None
    log_tau_median: float | None
    log_tau_standard_deviation: float | None
    log_tau_mad: float | None
    log_tau_range: float | None
    available_value_count: int
    missing_value_count: int
    fit_quality_values: tuple[float, ...]
    mean_fit_quality: float | None
    valid: bool
    passes_dispersion_limits: bool
    reasons: tuple[ModalParameterEstimateReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _enum(self.location_method, ParameterLocationMethod, "tau location method")
        _enum(self.weighting_method, ParameterWeightingMethod, "tau weighting method")
        aligned = (
            self.condition_labels,
            self.candidate_ids,
            self.recording_ids,
            self.source_track_ids,
            self.weights,
            self.normalized_weights,
        )
        if any(len(values) != len(self.tau_values_s) for values in aligned):
            raise ValueError("decay estimate source vectors must align.")
        if self.available_value_count != len(self.tau_values_s):
            raise ValueError("available_value_count must match tau_values_s.")
        if self.missing_value_count < 0:
            raise ValueError("missing_value_count must not be negative.")
        if any(not isfinite(value) or value <= 0.0 for value in self.tau_values_s):
            raise ValueError("tau values must be finite and positive.")
        if len(self.log_tau_values) != len(self.tau_values_s):
            raise ValueError("log_tau_values must match tau_values_s.")
        if any(not isfinite(value) for value in self.log_tau_values):
            raise ValueError("log tau values must be finite.")
        _strings(self.condition_labels, "decay condition labels", allow_empty=True)
        _strings(self.recording_ids, "decay recording ids", allow_empty=True)
        if any(value < 0 for value in self.candidate_ids):
            raise ValueError("decay candidate IDs must not be negative.")
        if any(value < 0 for value in self.source_track_ids):
            raise ValueError("decay candidate and track IDs must not be negative.")
        _validate_weights(self.weights, self.normalized_weights, require_normalized_sum=self.valid)
        for name in (
            "representative_tau_s",
            "minimum_tau_s",
            "maximum_tau_s",
            "log_tau_standard_deviation",
            "log_tau_mad",
            "log_tau_range",
            "mean_fit_quality",
        ):
            _finite_optional(getattr(self, name), name, nonnegative=True)
        for name in ("log_tau_mean", "log_tau_median"):
            _finite_optional(getattr(self, name), name)
        for value in self.fit_quality_values:
            _fraction(value, "fit quality value")
        if self.valid and self.representative_tau_s is None:
            raise ValueError("valid decay estimate requires representative_tau_s.")
        if self.representative_tau_s is not None and self.minimum_tau_s is not None and self.maximum_tau_s is not None:
            if not self.minimum_tau_s <= self.representative_tau_s <= self.maximum_tau_s:
                raise ValueError("representative tau must lie within min/max.")
        _reason_tuple(self.reasons, "decay estimate reasons")
        _strings(self.diagnostics, "decay estimate diagnostics", allow_empty=True)


@dataclass(frozen=True, slots=True)
class ModalDecayRateEstimate:
    representative_tau_s: float | None
    amplitude_decay_rate_per_s: float | None
    energy_decay_rate_per_s: float | None
    time_to_inverse_e_s: float | None
    time_to_minus_20_db_s: float | None
    time_to_minus_40_db_s: float | None
    time_to_minus_60_db_s: float | None
    valid: bool
    convention: str
    reasons: tuple[ModalParameterEstimateReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.convention, "decay-rate convention")
        for name in (
            "representative_tau_s",
            "amplitude_decay_rate_per_s",
            "energy_decay_rate_per_s",
            "time_to_inverse_e_s",
            "time_to_minus_20_db_s",
            "time_to_minus_40_db_s",
            "time_to_minus_60_db_s",
        ):
            _finite_optional(getattr(self, name), name, positive=self.valid)
        if self.valid and self.representative_tau_s is None:
            raise ValueError("valid decay-rate estimate requires tau.")
        _reason_tuple(self.reasons, "decay-rate reasons")
        _strings(self.diagnostics, "decay-rate diagnostics", allow_empty=True)


@dataclass(frozen=True, slots=True)
class ModalDecayUncertainty:
    method: ParameterUncertaintyMethod
    standard_uncertainty_log_tau: float | None
    multiplicative_uncertainty_factor: float | None
    lower_bound_tau_s: float | None
    upper_bound_tau_s: float | None
    confidence_level: float
    sample_count: int
    bootstrap_sample_count: int | None
    random_seed: int | None
    valid: bool
    reasons: tuple[ModalParameterEstimateReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _enum(self.method, ParameterUncertaintyMethod, "decay uncertainty method")
        if self.sample_count < 0:
            raise ValueError("sample_count must not be negative.")
        if self.bootstrap_sample_count is not None and self.bootstrap_sample_count <= 0:
            raise ValueError("bootstrap_sample_count must be positive when present.")
        if not isfinite(self.confidence_level) or not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be finite and in (0, 1).")
        for name in (
            "standard_uncertainty_log_tau",
            "multiplicative_uncertainty_factor",
            "lower_bound_tau_s",
            "upper_bound_tau_s",
        ):
            _finite_optional(getattr(self, name), name, positive=False, nonnegative=True)
        if self.multiplicative_uncertainty_factor is not None and self.multiplicative_uncertainty_factor < 1.0:
            raise ValueError("multiplicative_uncertainty_factor must be at least one.")
        if self.lower_bound_tau_s is not None and self.upper_bound_tau_s is not None:
            if self.lower_bound_tau_s <= 0.0 or self.upper_bound_tau_s <= 0.0:
                raise ValueError("tau uncertainty bounds must be positive.")
            if self.lower_bound_tau_s > self.upper_bound_tau_s:
                raise ValueError("tau uncertainty bounds are inverted.")
        _reason_tuple(self.reasons, "decay uncertainty reasons")
        _strings(self.diagnostics, "decay uncertainty diagnostics", allow_empty=True)


@dataclass(frozen=True, slots=True)
class ModalParameterProvenance:
    hypothesis_id: str
    source_chain_id: str | None
    candidate_ids: tuple[int, ...]
    match_ids: tuple[str, ...]
    condition_labels: tuple[str, ...]
    recording_ids: tuple[str, ...]
    frequency_source_count: int
    tau_source_count: int
    ambiguous_match_ids: tuple[str, ...]
    near_threshold_match_ids: tuple[str, ...]
    possible_split_context_ids: tuple[str, ...]
    possible_merge_context_ids: tuple[str, ...]
    settings_fingerprint: str
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.hypothesis_id, "provenance hypothesis_id")
        if self.source_chain_id is not None:
            _text(self.source_chain_id, "provenance source_chain_id")
        _strings(self.match_ids, "provenance match ids", allow_empty=True)
        _strings(self.condition_labels, "provenance condition labels", allow_empty=True)
        _strings(self.recording_ids, "provenance recording ids", allow_empty=True)
        _strings(self.ambiguous_match_ids, "provenance ambiguous match ids", allow_empty=True)
        _strings(self.near_threshold_match_ids, "provenance near-threshold match ids", allow_empty=True)
        _strings(self.possible_split_context_ids, "provenance split context ids", allow_empty=True)
        _strings(self.possible_merge_context_ids, "provenance merge context ids", allow_empty=True)
        _text(self.settings_fingerprint, "settings_fingerprint")
        if self.frequency_source_count < 0 or self.tau_source_count < 0:
            raise ValueError("source counts must not be negative.")
        if min(self.candidate_ids or (0,)) < 0:
            raise ValueError("candidate IDs must not be negative.")
        _strings(self.diagnostics, "provenance diagnostics", allow_empty=True)


@dataclass(frozen=True, slots=True)
class ModalParameterEstimate:
    estimate_id: str
    hypothesis_id: str
    status: ModalParameterEstimateStatus
    frequency_estimate: ModalFrequencyEstimate
    frequency_trajectory: ModalFrequencyTrajectoryEstimate
    frequency_uncertainty: ModalFrequencyUncertainty
    decay_estimate: ModalDecayEstimate
    decay_rate_estimate: ModalDecayRateEstimate
    decay_uncertainty: ModalDecayUncertainty
    provenance: ModalParameterProvenance
    supporting_reasons: tuple[ModalParameterEstimateReason, ...]
    reservation_reasons: tuple[ModalParameterEstimateReason, ...]
    insufficient_evidence_reasons: tuple[ModalParameterEstimateReason, ...]
    invalid_reasons: tuple[ModalParameterEstimateReason, ...]
    valid: bool
    requires_review: bool
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.estimate_id, "estimate_id")
        _text(self.hypothesis_id, "estimate hypothesis_id")
        _enum(self.status, ModalParameterEstimateStatus, "estimate status")
        expected_valid = self.status in {
            ModalParameterEstimateStatus.VALID,
            ModalParameterEstimateStatus.VALID_WITH_RESERVATIONS,
            ModalParameterEstimateStatus.PARTIAL,
        }
        if self.valid != expected_valid:
            raise ValueError("estimate valid flag must mirror usable statuses.")
        if self.requires_review != (self.status is not ModalParameterEstimateStatus.VALID):
            raise ValueError("requires_review must be false only for valid estimates.")
        for name in (
            "supporting_reasons",
            "reservation_reasons",
            "insufficient_evidence_reasons",
            "invalid_reasons",
        ):
            _reason_tuple(getattr(self, name), name)
        if self.status is ModalParameterEstimateStatus.VALID and (
            self.reservation_reasons
            or self.insufficient_evidence_reasons
            or self.invalid_reasons
        ):
            raise ValueError("valid parameter estimates must not carry reservations, insufficiency or invalidity.")
        if self.status is ModalParameterEstimateStatus.INVALID_INPUT and not self.invalid_reasons:
            raise ValueError("invalid input estimates require invalid reasons.")
        _strings(self.diagnostics, "parameter estimate diagnostics", allow_empty=True)


@dataclass(frozen=True, slots=True)
class ModalParameterEstimationResult:
    sequence: tuple[str, ...]
    estimates: tuple[ModalParameterEstimate, ...]
    valid_estimates: tuple[ModalParameterEstimate, ...]
    valid_with_reservations_estimates: tuple[ModalParameterEstimate, ...]
    partial_estimates: tuple[ModalParameterEstimate, ...]
    insufficient_evidence_estimates: tuple[ModalParameterEstimate, ...]
    invalid_estimates: tuple[ModalParameterEstimate, ...]
    estimate_count: int
    valid_count: int
    valid_with_reservations_count: int
    partial_count: int
    insufficient_evidence_count: int
    invalid_count: int
    source_hypothesis_count: int
    settings: ModalParameterEstimationSettings
    valid: bool
    failure_reason: str | None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_sequence(self.sequence)
        if not isinstance(self.settings, ModalParameterEstimationSettings):
            raise ValueError("settings must be ModalParameterEstimationSettings.")
        if self.estimates != tuple(sorted(self.estimates, key=_estimate_sort_key)):
            raise ValueError("estimates must be in deterministic order.")
        ids = tuple(item.estimate_id for item in self.estimates)
        if len(ids) != len(set(ids)):
            raise ValueError("estimate IDs must be unique.")
        hypothesis_ids = tuple(item.hypothesis_id for item in self.estimates)
        if len(hypothesis_ids) != len(set(hypothesis_ids)):
            raise ValueError("hypotheses must not produce duplicate estimates.")
        subsets = {
            ModalParameterEstimateStatus.VALID: self.valid_estimates,
            ModalParameterEstimateStatus.VALID_WITH_RESERVATIONS: self.valid_with_reservations_estimates,
            ModalParameterEstimateStatus.PARTIAL: self.partial_estimates,
            ModalParameterEstimateStatus.INSUFFICIENT_EVIDENCE: self.insufficient_evidence_estimates,
            ModalParameterEstimateStatus.INVALID_INPUT: self.invalid_estimates,
        }
        for status, subset in subsets.items():
            expected = tuple(item for item in self.estimates if item.status is status)
            if subset != expected:
                raise ValueError("status subsets must mirror estimates.")
        counts = {
            "estimate_count": len(self.estimates),
            "valid_count": len(self.valid_estimates),
            "valid_with_reservations_count": len(self.valid_with_reservations_estimates),
            "partial_count": len(self.partial_estimates),
            "insufficient_evidence_count": len(self.insufficient_evidence_estimates),
            "invalid_count": len(self.invalid_estimates),
        }
        for name, expected in counts.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} is incoherent with estimates.")
        if self.source_hypothesis_count != self.estimate_count:
            raise ValueError("each source hypothesis must produce exactly one estimate.")
        if self.valid != (self.invalid_count == 0):
            raise ValueError("result valid flag must mirror absence of invalid estimates.")
        if self.valid and self.failure_reason is not None:
            raise ValueError("valid result must not have failure_reason.")
        if not self.valid and not self.failure_reason:
            raise ValueError("invalid result requires failure_reason.")
        _strings(self.diagnostics, "parameter result diagnostics", allow_empty=True)


def estimate_modal_parameters(
    hypotheses: ModalHypothesisResult | Iterable[ModalHypothesis],
    settings: ModalParameterEstimationSettings | None = None,
) -> ModalParameterEstimationResult:
    """Estimate parameters for every hypothesis exactly once."""

    cfg = settings or ModalParameterEstimationSettings()
    if isinstance(hypotheses, ModalHypothesisResult):
        source = hypotheses.hypotheses
        sequence = hypotheses.sequence
    else:
        source = tuple(hypotheses)
        sequence = _sequence_from_hypotheses(source)
    ordered = tuple(sorted(source, key=_hypothesis_sort_key_for_parameters))
    estimates = tuple(
        estimate_modal_parameters_for_hypothesis(hypothesis, cfg)
        for hypothesis in ordered
    )
    estimates = tuple(sorted(estimates, key=_estimate_sort_key))
    valid = tuple(item for item in estimates if item.status is ModalParameterEstimateStatus.VALID)
    reserved = tuple(item for item in estimates if item.status is ModalParameterEstimateStatus.VALID_WITH_RESERVATIONS)
    partial = tuple(item for item in estimates if item.status is ModalParameterEstimateStatus.PARTIAL)
    insufficient = tuple(item for item in estimates if item.status is ModalParameterEstimateStatus.INSUFFICIENT_EVIDENCE)
    invalid = tuple(item for item in estimates if item.status is ModalParameterEstimateStatus.INVALID_INPUT)
    diagnostics = (
        "modal_parameter_estimates_are_operational_summaries",
        "one_estimate_per_source_hypothesis",
        "deterministic_content_and_settings_based_ids",
        "no_audio_fft_stft_tracking_candidate_or_association_recomputed",
        "no_non_adjacent_association_created",
        "no_gap_closure_performed",
        "no_split_or_merge_resolved",
        "no_modal_mode_created",
        "no_q_factor_or_bandwidth_calculated",
    )
    return ModalParameterEstimationResult(
        sequence=sequence,
        estimates=estimates,
        valid_estimates=valid,
        valid_with_reservations_estimates=reserved,
        partial_estimates=partial,
        insufficient_evidence_estimates=insufficient,
        invalid_estimates=invalid,
        estimate_count=len(estimates),
        valid_count=len(valid),
        valid_with_reservations_count=len(reserved),
        partial_count=len(partial),
        insufficient_evidence_count=len(insufficient),
        invalid_count=len(invalid),
        source_hypothesis_count=len(source),
        settings=cfg,
        valid=not invalid,
        failure_reason=None if not invalid else "invalid_estimates_present",
        diagnostics=diagnostics,
    )


def estimate_modal_parameters_for_hypothesis(
    hypothesis: ModalHypothesis | object,
    settings: ModalParameterEstimationSettings | None = None,
) -> ModalParameterEstimate:
    """Estimate all operational parameters for a single modal hypothesis."""

    cfg = settings or ModalParameterEstimationSettings()
    if not isinstance(hypothesis, ModalHypothesis):
        return _invalid_parameter_estimate(
            "invalid-hypothesis",
            cfg,
            (ModalParameterEstimateReason.INVALID_HYPOTHESIS,),
            ("input_is_not_modal_hypothesis",),
        )
    provenance = estimate_modal_parameter_provenance(hypothesis, cfg)
    if hypothesis.status is ModalHypothesisStatus.INVALID_INPUT or hypothesis.chain is None:
        return _invalid_parameter_estimate(
            hypothesis.hypothesis_id,
            cfg,
            (ModalParameterEstimateReason.INVALID_HYPOTHESIS,),
            ("invalid_modal_hypothesis_not_estimated",),
            provenance=provenance,
        )

    frequency = estimate_modal_frequency(hypothesis, cfg)
    trajectory = estimate_modal_frequency_trajectory(frequency, cfg)
    frequency_uncertainty = estimate_modal_frequency_uncertainty(frequency, hypothesis, cfg)
    decay = estimate_modal_decay(hypothesis, cfg)
    decay_rate = estimate_modal_decay_rate(decay, cfg)
    decay_uncertainty = estimate_modal_decay_uncertainty(decay, cfg)

    supporting: list[ModalParameterEstimateReason] = []
    reservations: list[ModalParameterEstimateReason] = []
    insufficient: list[ModalParameterEstimateReason] = []
    invalid: list[ModalParameterEstimateReason] = []
    diagnostics = [
        "hypothesis_modal_parameter_estimate_is_operational_not_physical_mode",
        "representative_frequency_is_not_exact_modal_frequency",
        "estimated_decay_time_is_not_invariant_physical_constant",
        "condition_variation_is_not_nonlinearity_proof",
        "operational_uncertainty_is_not_complete_physical_confidence_interval",
        "no_q_factor_or_bandwidth_calculated",
        "no_split_or_merge_resolved",
        "no_modal_mode_created",
    ]

    supporting.extend(
        reason for reason in frequency.reasons
        if reason is ModalParameterEstimateReason.SUFFICIENT_FREQUENCY_EVIDENCE
    )
    supporting.extend(
        reason for reason in decay.reasons
        if reason is ModalParameterEstimateReason.SUFFICIENT_DECAY_EVIDENCE
    )
    if hypothesis.coverage_evidence.passes:
        supporting.append(ModalParameterEstimateReason.SUFFICIENT_CONDITION_COVERAGE)
    _apply_hypothesis_policy(hypothesis, cfg, supporting, reservations, insufficient)

    _extend_by_role(frequency.reasons, cfg, reservations, insufficient, invalid)
    _extend_by_role(decay.reasons, cfg, reservations, insufficient, invalid)
    _extend_by_role(frequency_uncertainty.reasons, cfg, reservations, insufficient, invalid)
    _extend_by_role(decay_uncertainty.reasons, cfg, reservations, insufficient, invalid)
    if decay_rate.valid:
        supporting.extend(
            reason for reason in decay_rate.reasons
            if reason is ModalParameterEstimateReason.SUFFICIENT_DECAY_EVIDENCE
        )
    else:
        _extend_by_role(decay_rate.reasons, cfg, reservations, insufficient, invalid)

    chain = hypothesis.chain
    if chain.contains_ambiguous_match and cfg.reserve_ambiguous_matches:
        reservations.append(ModalParameterEstimateReason.AMBIGUOUS_SOURCE_MATCH)
    if chain.contains_near_threshold_match and cfg.reserve_near_threshold_matches:
        reservations.append(ModalParameterEstimateReason.NEAR_THRESHOLD_SOURCE_MATCH)
    if chain.contains_possible_split_context and cfg.reserve_possible_split_context:
        reservations.append(ModalParameterEstimateReason.POSSIBLE_SPLIT_CONTEXT)
    if chain.contains_possible_merge_context and cfg.reserve_possible_merge_context:
        reservations.append(ModalParameterEstimateReason.POSSIBLE_MERGE_CONTEXT)
    if any(not node.candidate_ref.accepted for node in chain.nodes):
        reservations.append(ModalParameterEstimateReason.REJECTED_CANDIDATE_PRESENT)

    status = _estimate_status(
        hypothesis,
        cfg,
        frequency,
        decay,
        frequency_uncertainty,
        decay_uncertainty,
        reservations,
        insufficient,
        invalid,
    )
    supporting_tuple = _ordered_reasons(supporting)
    reservations_tuple = _ordered_reasons(reservations)
    insufficient_tuple = _ordered_reasons(insufficient)
    invalid_tuple = _ordered_reasons(invalid)
    estimate_id = _estimate_id(
        hypothesis.hypothesis_id,
        provenance,
        frequency,
        decay,
        frequency_uncertainty,
        decay_uncertainty,
        status,
    )
    return ModalParameterEstimate(
        estimate_id=estimate_id,
        hypothesis_id=hypothesis.hypothesis_id,
        status=status,
        frequency_estimate=frequency,
        frequency_trajectory=trajectory,
        frequency_uncertainty=frequency_uncertainty,
        decay_estimate=decay,
        decay_rate_estimate=decay_rate,
        decay_uncertainty=decay_uncertainty,
        provenance=provenance,
        supporting_reasons=supporting_tuple,
        reservation_reasons=reservations_tuple,
        insufficient_evidence_reasons=insufficient_tuple,
        invalid_reasons=invalid_tuple,
        valid=status in {
            ModalParameterEstimateStatus.VALID,
            ModalParameterEstimateStatus.VALID_WITH_RESERVATIONS,
            ModalParameterEstimateStatus.PARTIAL,
        },
        requires_review=status is not ModalParameterEstimateStatus.VALID,
        diagnostics=tuple(dict.fromkeys(diagnostics)),
    )


def estimate_modal_frequency(
    hypothesis: ModalHypothesis,
    settings: ModalParameterEstimationSettings | None = None,
) -> ModalFrequencyEstimate:
    """Estimate a representative operational frequency from hypothesis candidates."""

    cfg = settings or ModalParameterEstimationSettings()
    refs = _hypothesis_refs(hypothesis)
    values: list[float] = []
    labels: list[str] = []
    candidate_ids: list[int] = []
    recording_ids: list[str] = []
    track_ids: list[int] = []
    rmse_values: list[float | None] = []
    drift_values: list[float | None] = []
    coverage_values: list[float | None] = []
    reasons: list[ModalParameterEstimateReason] = []
    diagnostics = [
        "frequency_values_reused_from_candidate_reference",
        "representative_frequency_is_operational_not_exact_modal_frequency",
        "no_frequency_tracking_or_association_recomputed",
    ]
    invalid_count = 0
    for ref in refs:
        value = ref.representative_frequency_hz
        if _positive_finite(value, cfg):
            values.append(float(value))
            labels.append(ref.dynamic_label)
            candidate_ids.append(ref.candidate_id)
            recording_ids.append(ref.recording_id)
            track_ids.append(ref.source_track_id)
            rmse_values.append(_finite_nonnegative_or_none(ref.frequency_fit_rmse_hz))
            drift_values.append(_finite_or_none(ref.frequency_drift_hz))
            coverage_values.append(_fraction_or_none(ref.coverage_fraction))
        else:
            invalid_count += 1
    if invalid_count:
        reasons.append(ModalParameterEstimateReason.INVALID_FREQUENCY_VALUE)
        diagnostics.append(f"invalid_frequency_values:{invalid_count}")
    missing_count = len(refs) - len(values)
    if missing_count:
        diagnostics.append("missing_or_invalid_frequency_values_preserved")
    local_costs = _local_association_costs(hypothesis)
    valid_ref_costs = tuple(
        (ref, cost)
        for ref, cost in zip(refs, local_costs, strict=True)
        if _positive_finite(ref.representative_frequency_hz, cfg)
    )
    weights, normalized, weight_reasons, weight_diagnostics = _weights_for_refs(
        tuple(ref for ref, _ in valid_ref_costs),
        cfg.frequency_weighting_method,
        tuple(cost for _, cost in valid_ref_costs),
    )
    reasons.extend(weight_reasons)
    diagnostics.extend(weight_diagnostics)
    stats = _basic_stats(tuple(values))
    representative = None
    if len(values) >= cfg.minimum_frequency_value_count and not weight_reasons:
        representative = _location(
            tuple(values),
            tuple(value for value in normalized if value is not None),
            cfg.frequency_location_method,
        )
    if len(values) < cfg.minimum_frequency_value_count:
        reasons.append(ModalParameterEstimateReason.INSUFFICIENT_FREQUENCY_VALUES)
    relative_range = (
        stats["range"] / representative
        if representative is not None and representative > 0.0 and stats["range"] is not None
        else None
    )
    coefficient = (
        stats["std"] / stats["mean"]
        if stats["mean"] is not None and stats["mean"] > 0.0 and stats["std"] is not None
        else None
    )
    passes_dispersion = True
    if _exceeds(coefficient, cfg.maximum_frequency_coefficient_of_variation):
        passes_dispersion = False
        reasons.append(ModalParameterEstimateReason.EXCESSIVE_FREQUENCY_DISPERSION)
    if _exceeds(relative_range, cfg.maximum_frequency_relative_range):
        passes_dispersion = False
        reasons.append(ModalParameterEstimateReason.EXCESSIVE_FREQUENCY_DISPERSION)
    valid = (
        representative is not None
        and len(values) >= cfg.minimum_frequency_value_count
        and passes_dispersion
        and not _contains_invalid_reason(weight_reasons)
        and (
            cfg.finite_value_policy is FiniteValuePolicy.EXCLUDE_WITH_DIAGNOSTIC
            or ModalParameterEstimateReason.INVALID_FREQUENCY_VALUE not in reasons
        )
    )
    if valid:
        reasons.append(ModalParameterEstimateReason.SUFFICIENT_FREQUENCY_EVIDENCE)
    return ModalFrequencyEstimate(
        values_hz=tuple(values),
        condition_labels=tuple(labels),
        candidate_ids=tuple(candidate_ids),
        recording_ids=tuple(recording_ids),
        source_track_ids=tuple(track_ids),
        frequency_fit_rmse_values_hz=tuple(rmse_values),
        source_frequency_drifts_hz=tuple(drift_values),
        coverage_values=tuple(coverage_values),
        weights=weights,
        normalized_weights=normalized,
        location_method=cfg.frequency_location_method,
        weighting_method=cfg.frequency_weighting_method,
        representative_frequency_hz=representative,
        minimum_frequency_hz=stats["min"],
        maximum_frequency_hz=stats["max"],
        frequency_range_hz=stats["range"],
        relative_frequency_range=relative_range,
        frequency_mean_hz=stats["mean"],
        frequency_median_hz=stats["median"],
        frequency_standard_deviation_hz=stats["std"],
        frequency_mad_hz=stats["mad"],
        frequency_coefficient_of_variation=coefficient,
        available_value_count=len(values),
        missing_value_count=missing_count,
        valid=valid,
        passes_dispersion_limits=passes_dispersion,
        reasons=_ordered_reasons(reasons),
        diagnostics=tuple(dict.fromkeys(diagnostics)),
    )


def estimate_modal_frequency_trajectory(
    source: ModalHypothesis | ModalFrequencyEstimate,
    settings: ModalParameterEstimationSettings | None = None,
) -> ModalFrequencyTrajectoryEstimate:
    """Summarize inter-condition frequency trajectory descriptively."""

    del settings
    frequency = source if isinstance(source, ModalFrequencyEstimate) else estimate_modal_frequency(source)
    values = frequency.values_hz
    labels = frequency.condition_labels
    diagnostics = [
        "frequency_trajectory_is_descriptive_not_physical_law",
        "linear_fit_uses_condition_ordinal_index_only",
        "no_hardening_or_softening_classification",
        "no_linearity_or_nonlinearity_proof",
        "no_causal_intensity_dependence_inferred",
    ]
    if len(values) < 2:
        return ModalFrequencyTrajectoryEstimate(
            condition_labels=labels,
            frequencies_hz=values,
            signed_step_changes_hz=None,
            signed_step_relative_changes=None,
            step_count=0,
            total_signed_change_hz=None,
            total_absolute_change_hz=None,
            total_relative_symmetric_change=None,
            mean_signed_step_change_hz=None,
            mean_absolute_step_change_hz=None,
            maximum_absolute_step_change_hz=None,
            up_step_count=0,
            down_step_count=0,
            preserved_step_count=0,
            indeterminate_step_count=0,
            linear_slope_hz_per_condition_step=None,
            linear_fit_intercept_hz=None,
            linear_fit_rmse_hz=None,
            linear_fit_r_squared=None,
            valid=False,
            reasons=(ModalParameterEstimateReason.INSUFFICIENT_FREQUENCY_VALUES,),
            diagnostics=tuple(diagnostics),
        )
    signed = tuple(right - left for left, right in zip(values, values[1:], strict=False))
    signed_relative = tuple(
        change / (0.5 * (left + right))
        for change, left, right in zip(signed, values, values[1:], strict=False)
    )
    absolute = tuple(abs(value) for value in signed)
    total_signed = values[-1] - values[0]
    total_absolute = sum(absolute)
    total_relative = total_signed / (0.5 * (values[-1] + values[0]))
    up = sum(value > 1e-12 for value in signed)
    down = sum(value < -1e-12 for value in signed)
    preserved = sum(isclose(value, 0.0, rel_tol=0.0, abs_tol=1e-12) for value in signed)
    indeterminate = len(signed) - up - down - preserved
    slope, intercept, rmse, r_squared = _linear_fit_by_ordinal(values)
    return ModalFrequencyTrajectoryEstimate(
        condition_labels=labels,
        frequencies_hz=values,
        signed_step_changes_hz=signed,
        signed_step_relative_changes=signed_relative,
        step_count=len(signed),
        total_signed_change_hz=total_signed,
        total_absolute_change_hz=total_absolute,
        total_relative_symmetric_change=total_relative,
        mean_signed_step_change_hz=sum(signed) / len(signed),
        mean_absolute_step_change_hz=sum(absolute) / len(absolute),
        maximum_absolute_step_change_hz=max(absolute),
        up_step_count=up,
        down_step_count=down,
        preserved_step_count=preserved,
        indeterminate_step_count=indeterminate,
        linear_slope_hz_per_condition_step=slope,
        linear_fit_intercept_hz=intercept,
        linear_fit_rmse_hz=rmse,
        linear_fit_r_squared=r_squared,
        valid=True,
        reasons=(ModalParameterEstimateReason.SUFFICIENT_FREQUENCY_EVIDENCE,),
        diagnostics=tuple(diagnostics),
    )


def estimate_modal_frequency_uncertainty(
    frequency: ModalFrequencyEstimate,
    hypothesis: ModalHypothesis | None = None,
    settings: ModalParameterEstimationSettings | None = None,
) -> ModalFrequencyUncertainty:
    """Estimate operational frequency uncertainty from dispersion and fit RMSE."""

    cfg = settings or ModalParameterEstimationSettings()
    values = frequency.values_hz
    individual = frequency.frequency_fit_rmse_values_hz
    reasons: list[ModalParameterEstimateReason] = []
    diagnostics = [
        "frequency_uncertainty_is_operational_not_complete_physical_confidence_interval",
        "frequency_fit_rmse_reused_as_individual_operational_uncertainty_when_available",
    ]
    if hypothesis is not None:
        del hypothesis
    if cfg.frequency_uncertainty_method is ParameterUncertaintyMethod.DISABLED:
        diagnostics.append("frequency_uncertainty_method_disabled")
        return ModalFrequencyUncertainty(
            cfg.frequency_uncertainty_method,
            None,
            None,
            None,
            cfg.uncertainty_confidence_level,
            len(values),
            None,
            cfg.bootstrap_random_seed,
            individual,
            None,
            None,
            False,
            (),
            tuple(diagnostics),
        )
    missing_individual = sum(value is None for value in individual)
    valid_individual = tuple(
        float(value) for value in individual
        if value is not None and isfinite(value) and value >= 0.0
    )
    if missing_individual and not cfg.allow_missing_frequency_uncertainty:
        reasons.append(ModalParameterEstimateReason.MISSING_FREQUENCY_UNCERTAINTY)
        diagnostics.append(f"missing_frequency_uncertainty_values:{missing_individual}")
    if len(values) < 2 and cfg.frequency_uncertainty_method in {
        ParameterUncertaintyMethod.SAMPLE_STANDARD_DEVIATION,
        ParameterUncertaintyMethod.STANDARD_ERROR,
        ParameterUncertaintyMethod.BOOTSTRAP_PERCENTILE,
        ParameterUncertaintyMethod.CONSERVATIVE,
    }:
        reasons.append(ModalParameterEstimateReason.INSUFFICIENT_FREQUENCY_VALUES)
    dispersion = _sample_standard_deviation(values) if len(values) >= 2 else None
    measurement = _rms(valid_individual) if valid_individual else None
    standard: float | None = None
    lower: float | None = None
    upper: float | None = None
    bootstrap_count: int | None = None
    method = cfg.frequency_uncertainty_method
    center = frequency.representative_frequency_hz
    if center is not None and not reasons:
        if method is ParameterUncertaintyMethod.SAMPLE_STANDARD_DEVIATION:
            standard = dispersion
        elif method is ParameterUncertaintyMethod.STANDARD_ERROR:
            standard = dispersion / sqrt(len(values)) if dispersion is not None else None
        elif method is ParameterUncertaintyMethod.SCALED_MAD:
            standard = (
                _MAD_NORMAL_CONSISTENCY_FACTOR * (frequency.frequency_mad_hz or 0.0)
                if values
                else None
            )
        elif method is ParameterUncertaintyMethod.BOOTSTRAP_PERCENTILE:
            if len(values) < 2:
                diagnostics.append("bootstrap_requires_at_least_two_frequency_values")
            else:
                bootstrap_count = cfg.bootstrap_sample_count
                samples = _bootstrap_locations(
                    values,
                    frequency.normalized_weights,
                    frequency.location_method,
                    cfg.bootstrap_sample_count,
                    cfg.bootstrap_random_seed,
                )
                lower = _quantile(samples, (1.0 - cfg.uncertainty_confidence_level) / 2.0)
                upper = _quantile(samples, 1.0 - (1.0 - cfg.uncertainty_confidence_level) / 2.0)
                lower = min(lower, center)
                upper = max(upper, center)
                standard = _sample_standard_deviation(samples) if len(samples) >= 2 else 0.0
        elif method is ParameterUncertaintyMethod.CONSERVATIVE:
            candidates = tuple(value for value in (dispersion, measurement) if value is not None)
            standard = max(candidates) if candidates else None
        else:
            reasons.append(ModalParameterEstimateReason.INSUFFICIENT_EVIDENCE)
            diagnostics.append("frequency_uncertainty_method_is_not_frequency_domain")
        if standard is not None and lower is None and upper is None:
            lower = max(cfg.minimum_positive_value, center - standard)
            upper = center + standard
    valid = standard is not None and lower is not None and upper is not None and not reasons
    if not valid and not reasons and method is not ParameterUncertaintyMethod.DISABLED:
        reasons.append(ModalParameterEstimateReason.MISSING_FREQUENCY_UNCERTAINTY)
    return ModalFrequencyUncertainty(
        method=method,
        standard_uncertainty_hz=standard,
        lower_bound_hz=lower,
        upper_bound_hz=upper,
        confidence_level=cfg.uncertainty_confidence_level,
        sample_count=len(values),
        bootstrap_sample_count=bootstrap_count,
        random_seed=cfg.bootstrap_random_seed,
        individual_uncertainties_hz=individual,
        dispersion_component_hz=dispersion,
        measurement_component_hz=measurement,
        valid=valid,
        reasons=_ordered_reasons(reasons),
        diagnostics=tuple(dict.fromkeys(diagnostics)),
    )


def estimate_modal_decay(
    hypothesis: ModalHypothesis,
    settings: ModalParameterEstimationSettings | None = None,
) -> ModalDecayEstimate:
    """Estimate a representative operational tau from candidate references."""

    cfg = settings or ModalParameterEstimationSettings()
    refs = _hypothesis_refs(hypothesis)
    values: list[float] = []
    labels: list[str] = []
    candidate_ids: list[int] = []
    recording_ids: list[str] = []
    track_ids: list[int] = []
    fit_qualities: list[float] = []
    reasons: list[ModalParameterEstimateReason] = []
    diagnostics = [
        "tau_values_reused_from_candidate_reference_amplitude_tau_s",
        "representative_tau_is_operational_not_invariant_physical_constant",
        "tau_summary_prefers_log_domain",
        "no_q_factor_or_bandwidth_derived",
    ]
    invalid_count = 0
    for ref in refs:
        tau = ref.amplitude_tau_s
        if tau is None:
            continue
        if _positive_finite(tau, cfg):
            values.append(float(tau))
            labels.append(ref.dynamic_label)
            candidate_ids.append(ref.candidate_id)
            recording_ids.append(ref.recording_id)
            track_ids.append(ref.source_track_id)
            quality = ref.amplitude_fit_r_squared
            if quality is not None and isfinite(quality) and 0.0 <= quality <= 1.0:
                fit_qualities.append(float(quality))
        else:
            invalid_count += 1
    if invalid_count:
        reasons.append(ModalParameterEstimateReason.INVALID_DECAY_VALUE)
        diagnostics.append(f"invalid_tau_values:{invalid_count}")
    missing_count = len(refs) - len(values)
    if missing_count:
        diagnostics.append("missing_or_invalid_tau_values_preserved")
    valid_refs = tuple(
        ref for ref in refs
        if ref.amplitude_tau_s is not None and _positive_finite(ref.amplitude_tau_s, cfg)
    )
    local_costs = _local_association_costs(hypothesis)
    valid_ref_costs = tuple(
        (ref, cost)
        for ref, cost in zip(refs, local_costs, strict=True)
        if ref.amplitude_tau_s is not None and _positive_finite(ref.amplitude_tau_s, cfg)
    )
    weights, normalized, weight_reasons, weight_diagnostics = _weights_for_refs(
        valid_refs,
        cfg.tau_weighting_method,
        tuple(cost for _, cost in valid_ref_costs),
    )
    reasons.extend(weight_reasons)
    diagnostics.extend(weight_diagnostics)
    logs = tuple(log(value) for value in values)
    log_stats = _basic_stats(logs)
    representative = None
    if len(values) >= cfg.minimum_tau_value_count and not weight_reasons:
        representative = _location(
            tuple(values),
            tuple(value for value in normalized if value is not None),
            cfg.tau_location_method,
        )
    if len(values) < cfg.minimum_tau_value_count:
        reasons.append(ModalParameterEstimateReason.INSUFFICIENT_DECAY_VALUES)
    passes_dispersion = True
    if _exceeds(log_stats["range"], cfg.maximum_log_tau_range):
        passes_dispersion = False
        reasons.append(ModalParameterEstimateReason.EXCESSIVE_DECAY_DISPERSION)
    if _exceeds(log_stats["std"], cfg.maximum_log_tau_standard_deviation):
        passes_dispersion = False
        reasons.append(ModalParameterEstimateReason.EXCESSIVE_DECAY_DISPERSION)
    valid = (
        representative is not None
        and len(values) >= cfg.minimum_tau_value_count
        and passes_dispersion
        and not _contains_invalid_reason(weight_reasons)
        and (
            cfg.finite_value_policy is FiniteValuePolicy.EXCLUDE_WITH_DIAGNOSTIC
            or ModalParameterEstimateReason.INVALID_DECAY_VALUE not in reasons
        )
    )
    if valid:
        reasons.append(ModalParameterEstimateReason.SUFFICIENT_DECAY_EVIDENCE)
    return ModalDecayEstimate(
        tau_values_s=tuple(values),
        condition_labels=tuple(labels),
        candidate_ids=tuple(candidate_ids),
        recording_ids=tuple(recording_ids),
        source_track_ids=tuple(track_ids),
        weights=weights,
        normalized_weights=normalized,
        location_method=cfg.tau_location_method,
        weighting_method=cfg.tau_weighting_method,
        representative_tau_s=representative,
        minimum_tau_s=min(values) if values else None,
        maximum_tau_s=max(values) if values else None,
        log_tau_values=logs,
        log_tau_mean=log_stats["mean"],
        log_tau_median=log_stats["median"],
        log_tau_standard_deviation=log_stats["std"],
        log_tau_mad=log_stats["mad"],
        log_tau_range=log_stats["range"],
        available_value_count=len(values),
        missing_value_count=missing_count,
        fit_quality_values=tuple(fit_qualities),
        mean_fit_quality=sum(fit_qualities) / len(fit_qualities) if fit_qualities else None,
        valid=valid,
        passes_dispersion_limits=passes_dispersion,
        reasons=_ordered_reasons(reasons),
        diagnostics=tuple(dict.fromkeys(diagnostics)),
    )


def estimate_modal_decay_rate(
    decay: ModalDecayEstimate | float | None,
    settings: ModalParameterEstimationSettings | None = None,
) -> ModalDecayRateEstimate:
    """Derive only documented mathematical rates from ``A(t)=A0 exp(-t/tau)``."""

    cfg = settings or ModalParameterEstimationSettings()
    tau = decay.representative_tau_s if isinstance(decay, ModalDecayEstimate) else decay
    convention = "A(t)=A0 exp(-t/tau); amplitude_decay_rate=1/tau; energy_rate=2/tau for energy proportional to A^2"
    diagnostics = [
        "decay_rate_is_mathematical_conversion_from_representative_tau",
        "amplitude_and_energy_decay_conventions_are_not_confused",
        "no_q_factor_or_bandwidth_derived",
    ]
    if tau is None or not _positive_finite(tau, cfg):
        reason = (
            ModalParameterEstimateReason.INSUFFICIENT_DECAY_VALUES
            if tau is None
            else ModalParameterEstimateReason.INVALID_DECAY_VALUE
        )
        return ModalDecayRateEstimate(
            representative_tau_s=tau if isinstance(tau, float) and isfinite(tau) and tau > 0.0 else None,
            amplitude_decay_rate_per_s=None,
            energy_decay_rate_per_s=None,
            time_to_inverse_e_s=None,
            time_to_minus_20_db_s=None,
            time_to_minus_40_db_s=None,
            time_to_minus_60_db_s=None,
            valid=False,
            convention=convention,
            reasons=(reason,),
            diagnostics=tuple(diagnostics),
        )
    tau = float(tau)
    return ModalDecayRateEstimate(
        representative_tau_s=tau,
        amplitude_decay_rate_per_s=1.0 / tau,
        energy_decay_rate_per_s=2.0 / tau,
        time_to_inverse_e_s=tau,
        time_to_minus_20_db_s=tau * log(10.0),
        time_to_minus_40_db_s=2.0 * tau * log(10.0),
        time_to_minus_60_db_s=3.0 * tau * log(10.0),
        valid=True,
        convention=convention,
        reasons=(ModalParameterEstimateReason.SUFFICIENT_DECAY_EVIDENCE,),
        diagnostics=tuple(diagnostics),
    )


def estimate_modal_decay_uncertainty(
    decay: ModalDecayEstimate,
    settings: ModalParameterEstimationSettings | None = None,
) -> ModalDecayUncertainty:
    """Estimate tau uncertainty in log domain and transform bounds to seconds."""

    cfg = settings or ModalParameterEstimationSettings()
    method = cfg.tau_uncertainty_method
    logs = decay.log_tau_values
    reasons: list[ModalParameterEstimateReason] = []
    diagnostics = [
        "decay_uncertainty_is_operational_log_domain_summary",
        "tau_uncertainty_bounds_are_transformed_back_to_seconds",
        "not_a_complete_physical_confidence_interval",
    ]
    if method is ParameterUncertaintyMethod.DISABLED:
        diagnostics.append("decay_uncertainty_method_disabled")
        return ModalDecayUncertainty(
            method,
            None,
            None,
            None,
            None,
            cfg.uncertainty_confidence_level,
            len(logs),
            None,
            cfg.bootstrap_random_seed,
            False,
            (),
            tuple(diagnostics),
        )
    if len(logs) < 2 and method in {
        ParameterUncertaintyMethod.LOG_STANDARD_DEVIATION,
        ParameterUncertaintyMethod.LOG_STANDARD_ERROR,
        ParameterUncertaintyMethod.LOG_BOOTSTRAP_PERCENTILE,
        ParameterUncertaintyMethod.SAMPLE_STANDARD_DEVIATION,
        ParameterUncertaintyMethod.STANDARD_ERROR,
        ParameterUncertaintyMethod.BOOTSTRAP_PERCENTILE,
        ParameterUncertaintyMethod.CONSERVATIVE,
    }:
        reasons.append(ModalParameterEstimateReason.INSUFFICIENT_DECAY_VALUES)
    center = decay.representative_tau_s
    standard: float | None = None
    factor: float | None = None
    lower: float | None = None
    upper: float | None = None
    bootstrap_count: int | None = None
    if center is not None and not reasons:
        if method in {
            ParameterUncertaintyMethod.LOG_STANDARD_DEVIATION,
            ParameterUncertaintyMethod.SAMPLE_STANDARD_DEVIATION,
        }:
            standard = _sample_standard_deviation(logs)
        elif method in {
            ParameterUncertaintyMethod.LOG_STANDARD_ERROR,
            ParameterUncertaintyMethod.STANDARD_ERROR,
        }:
            std = _sample_standard_deviation(logs)
            standard = std / sqrt(len(logs)) if std is not None else None
        elif method in {
            ParameterUncertaintyMethod.LOG_SCALED_MAD,
            ParameterUncertaintyMethod.SCALED_MAD,
        }:
            standard = _MAD_NORMAL_CONSISTENCY_FACTOR * (decay.log_tau_mad or 0.0)
        elif method in {
            ParameterUncertaintyMethod.LOG_BOOTSTRAP_PERCENTILE,
            ParameterUncertaintyMethod.BOOTSTRAP_PERCENTILE,
        }:
            if len(logs) < 2:
                diagnostics.append("bootstrap_requires_at_least_two_tau_values")
            else:
                bootstrap_count = cfg.bootstrap_sample_count
                bootstrap_logs = _bootstrap_log_tau_locations(
                    decay.tau_values_s,
                    decay.normalized_weights,
                    decay.location_method,
                    cfg.bootstrap_sample_count,
                    cfg.bootstrap_random_seed,
                )
                low_log = _quantile(bootstrap_logs, (1.0 - cfg.uncertainty_confidence_level) / 2.0)
                high_log = _quantile(bootstrap_logs, 1.0 - (1.0 - cfg.uncertainty_confidence_level) / 2.0)
                lower = min(exp(low_log), center)
                upper = max(exp(high_log), center)
                standard = _sample_standard_deviation(bootstrap_logs) if len(bootstrap_logs) >= 2 else 0.0
        elif method is ParameterUncertaintyMethod.CONSERVATIVE:
            std = _sample_standard_deviation(logs)
            mad = _MAD_NORMAL_CONSISTENCY_FACTOR * (decay.log_tau_mad or 0.0)
            standard = max(value for value in (std, mad) if value is not None)
        else:
            reasons.append(ModalParameterEstimateReason.MISSING_DECAY_UNCERTAINTY)
            diagnostics.append("tau_uncertainty_method_is_not_log_domain")
        if standard is not None:
            factor = exp(standard)
            if lower is None or upper is None:
                lower = center / factor
                upper = center * factor
    valid = standard is not None and factor is not None and lower is not None and upper is not None and not reasons
    if not valid and not reasons and method is not ParameterUncertaintyMethod.DISABLED:
        reasons.append(ModalParameterEstimateReason.MISSING_DECAY_UNCERTAINTY)
    return ModalDecayUncertainty(
        method=method,
        standard_uncertainty_log_tau=standard,
        multiplicative_uncertainty_factor=factor,
        lower_bound_tau_s=lower,
        upper_bound_tau_s=upper,
        confidence_level=cfg.uncertainty_confidence_level,
        sample_count=len(logs),
        bootstrap_sample_count=bootstrap_count,
        random_seed=cfg.bootstrap_random_seed,
        valid=valid,
        reasons=_ordered_reasons(reasons),
        diagnostics=tuple(dict.fromkeys(diagnostics)),
    )


def estimate_modal_parameter_provenance(
    hypothesis: ModalHypothesis,
    settings: ModalParameterEstimationSettings | None = None,
) -> ModalParameterProvenance:
    """Build deterministic provenance back to hypothesis, chain, candidates and matches."""

    cfg = settings or ModalParameterEstimationSettings()
    refs = _hypothesis_refs(hypothesis)
    chain = hypothesis.chain
    match_ids: tuple[str, ...] = ()
    ambiguous_ids: tuple[str, ...] = ()
    near_ids: tuple[str, ...] = ()
    split_ids: tuple[str, ...] = ()
    merge_ids: tuple[str, ...] = ()
    source_chain_id = None
    if chain is not None:
        source_chain_id = chain.chain_id
        match_ids = tuple(
            node.outgoing_match_id
            for node in chain.nodes
            if node.outgoing_match_id is not None
        )
        ambiguous_ids = tuple(chain.ambiguous_match_ids)
        near_ids = tuple(chain.near_threshold_match_ids)
        split_ids = tuple(_split_context_id(item) for item in chain.possible_split_contexts)
        merge_ids = tuple(_merge_context_id(item) for item in chain.possible_merge_contexts)
    diagnostics = (
        "provenance_links_to_existing_hypothesis_chain_candidates_and_matches",
        "settings_fingerprint_is_deterministic_no_timestamp",
    )
    return ModalParameterProvenance(
        hypothesis_id=hypothesis.hypothesis_id,
        source_chain_id=source_chain_id,
        candidate_ids=tuple(ref.candidate_id for ref in refs),
        match_ids=match_ids,
        condition_labels=tuple(ref.dynamic_label for ref in refs),
        recording_ids=tuple(ref.recording_id for ref in refs),
        frequency_source_count=sum(_positive_finite(ref.representative_frequency_hz, cfg) for ref in refs),
        tau_source_count=sum(ref.amplitude_tau_s is not None and _positive_finite(ref.amplitude_tau_s, cfg) for ref in refs),
        ambiguous_match_ids=ambiguous_ids,
        near_threshold_match_ids=near_ids,
        possible_split_context_ids=split_ids,
        possible_merge_context_ids=merge_ids,
        settings_fingerprint=settings_fingerprint(cfg),
        diagnostics=diagnostics,
    )


def summarize_modal_parameter_estimates(
    result: ModalParameterEstimationResult,
) -> dict[str, object]:
    """Return a compact deterministic summary for audit reports."""

    if not isinstance(result, ModalParameterEstimationResult):
        raise ValueError("result must be a ModalParameterEstimationResult.")
    return {
        "sequence": result.sequence,
        "estimate_count": result.estimate_count,
        "valid_count": result.valid_count,
        "valid_with_reservations_count": result.valid_with_reservations_count,
        "partial_count": result.partial_count,
        "insufficient_evidence_count": result.insufficient_evidence_count,
        "invalid_count": result.invalid_count,
        "source_hypothesis_count": result.source_hypothesis_count,
        "estimate_ids": tuple(item.estimate_id for item in result.estimates),
        "hypothesis_ids": tuple(item.hypothesis_id for item in result.estimates),
        "source_chain_ids": tuple(item.provenance.source_chain_id for item in result.estimates),
        "statuses": tuple(item.status.value for item in result.estimates),
        "representative_frequency_hz": tuple(
            item.frequency_estimate.representative_frequency_hz
            for item in result.estimates
        ),
        "representative_tau_s": tuple(
            item.decay_estimate.representative_tau_s
            for item in result.estimates
        ),
        "settings_fingerprint": settings_fingerprint(result.settings),
        "diagnostics": result.diagnostics,
    }


def settings_fingerprint(settings: ModalParameterEstimationSettings) -> str:
    """Return a stable settings fingerprint without timestamps or global state."""

    payload = json.dumps(_canonicalize(asdict(settings)), sort_keys=True, separators=(",", ":"))
    return "modal-parameter-settings-" + sha1(payload.encode("utf-8")).hexdigest()[:16]


def _invalid_parameter_estimate(
    hypothesis_id: str,
    settings: ModalParameterEstimationSettings,
    invalid_reasons: tuple[ModalParameterEstimateReason, ...],
    diagnostics: tuple[str, ...],
    *,
    provenance: ModalParameterProvenance | None = None,
) -> ModalParameterEstimate:
    frequency = ModalFrequencyEstimate(
        values_hz=(),
        condition_labels=(),
        candidate_ids=(),
        recording_ids=(),
        source_track_ids=(),
        frequency_fit_rmse_values_hz=(),
        source_frequency_drifts_hz=(),
        coverage_values=(),
        weights=(),
        normalized_weights=(),
        location_method=settings.frequency_location_method,
        weighting_method=settings.frequency_weighting_method,
        representative_frequency_hz=None,
        minimum_frequency_hz=None,
        maximum_frequency_hz=None,
        frequency_range_hz=None,
        relative_frequency_range=None,
        frequency_mean_hz=None,
        frequency_median_hz=None,
        frequency_standard_deviation_hz=None,
        frequency_mad_hz=None,
        frequency_coefficient_of_variation=None,
        available_value_count=0,
        missing_value_count=0,
        valid=False,
        passes_dispersion_limits=False,
        reasons=(ModalParameterEstimateReason.INVALID_HYPOTHESIS,),
        diagnostics=("invalid_hypothesis_no_frequency_estimate",),
    )
    trajectory = estimate_modal_frequency_trajectory(frequency, settings)
    uncertainty = ModalFrequencyUncertainty(
        settings.frequency_uncertainty_method,
        None,
        None,
        None,
        settings.uncertainty_confidence_level,
        0,
        None,
        settings.bootstrap_random_seed,
        (),
        None,
        None,
        False,
        (),
        ("invalid_hypothesis_no_frequency_uncertainty",),
    )
    decay = ModalDecayEstimate(
        tau_values_s=(),
        condition_labels=(),
        candidate_ids=(),
        recording_ids=(),
        source_track_ids=(),
        weights=(),
        normalized_weights=(),
        location_method=settings.tau_location_method,
        weighting_method=settings.tau_weighting_method,
        representative_tau_s=None,
        minimum_tau_s=None,
        maximum_tau_s=None,
        log_tau_values=(),
        log_tau_mean=None,
        log_tau_median=None,
        log_tau_standard_deviation=None,
        log_tau_mad=None,
        log_tau_range=None,
        available_value_count=0,
        missing_value_count=0,
        fit_quality_values=(),
        mean_fit_quality=None,
        valid=False,
        passes_dispersion_limits=False,
        reasons=(ModalParameterEstimateReason.INVALID_HYPOTHESIS,),
        diagnostics=("invalid_hypothesis_no_decay_estimate",),
    )
    decay_rate = estimate_modal_decay_rate(decay, settings)
    decay_uncertainty = ModalDecayUncertainty(
        settings.tau_uncertainty_method,
        None,
        None,
        None,
        None,
        settings.uncertainty_confidence_level,
        0,
        None,
        settings.bootstrap_random_seed,
        False,
        (),
        ("invalid_hypothesis_no_decay_uncertainty",),
    )
    if provenance is None:
        provenance = ModalParameterProvenance(
            hypothesis_id=hypothesis_id,
            source_chain_id=None,
            candidate_ids=(),
            match_ids=(),
            condition_labels=(),
            recording_ids=(),
            frequency_source_count=0,
            tau_source_count=0,
            ambiguous_match_ids=(),
            near_threshold_match_ids=(),
            possible_split_context_ids=(),
            possible_merge_context_ids=(),
            settings_fingerprint=settings_fingerprint(settings),
            diagnostics=("invalid_hypothesis_no_source_provenance",),
        )
    invalid_tuple = _ordered_reasons(invalid_reasons)
    estimate_id = _estimate_id(
        hypothesis_id,
        provenance,
        frequency,
        decay,
        uncertainty,
        decay_uncertainty,
        ModalParameterEstimateStatus.INVALID_INPUT,
    )
    return ModalParameterEstimate(
        estimate_id=estimate_id,
        hypothesis_id=hypothesis_id,
        status=ModalParameterEstimateStatus.INVALID_INPUT,
        frequency_estimate=frequency,
        frequency_trajectory=trajectory,
        frequency_uncertainty=uncertainty,
        decay_estimate=decay,
        decay_rate_estimate=decay_rate,
        decay_uncertainty=decay_uncertainty,
        provenance=provenance,
        supporting_reasons=(),
        reservation_reasons=(),
        insufficient_evidence_reasons=(),
        invalid_reasons=invalid_tuple,
        valid=False,
        requires_review=True,
        diagnostics=diagnostics,
    )


def _apply_hypothesis_policy(
    hypothesis: ModalHypothesis,
    cfg: ModalParameterEstimationSettings,
    supporting: list[ModalParameterEstimateReason],
    reservations: list[ModalParameterEstimateReason],
    insufficient: list[ModalParameterEstimateReason],
) -> None:
    status = hypothesis.status
    if status is ModalHypothesisStatus.ACCEPTED:
        if cfg.allow_accepted_hypotheses:
            supporting.append(ModalParameterEstimateReason.ACCEPTED_MODAL_HYPOTHESIS)
        else:
            insufficient.append(ModalParameterEstimateReason.INSUFFICIENT_EVIDENCE)
    elif status is ModalHypothesisStatus.ACCEPTED_WITH_RESERVATIONS:
        if cfg.allow_accepted_with_reservations:
            reservations.append(ModalParameterEstimateReason.HYPOTHESIS_WITH_RESERVATIONS)
        else:
            insufficient.append(ModalParameterEstimateReason.HYPOTHESIS_WITH_RESERVATIONS)
    elif status is ModalHypothesisStatus.INCONCLUSIVE:
        if cfg.allow_inconclusive_hypotheses:
            reservations.append(ModalParameterEstimateReason.INCONCLUSIVE_HYPOTHESIS)
        else:
            insufficient.append(ModalParameterEstimateReason.INCONCLUSIVE_HYPOTHESIS)
    elif status is ModalHypothesisStatus.REJECTED:
        if cfg.allow_rejected_hypotheses_for_audit:
            reservations.append(ModalParameterEstimateReason.REJECTED_HYPOTHESIS)
        else:
            insufficient.append(ModalParameterEstimateReason.REJECTED_HYPOTHESIS)
    elif status is ModalHypothesisStatus.INSUFFICIENT_EVIDENCE:
        if cfg.allow_insufficient_evidence_hypotheses_for_audit:
            reservations.append(ModalParameterEstimateReason.INSUFFICIENT_EVIDENCE)
        else:
            insufficient.append(ModalParameterEstimateReason.INSUFFICIENT_EVIDENCE)


def _estimate_status(
    hypothesis: ModalHypothesis,
    cfg: ModalParameterEstimationSettings,
    frequency: ModalFrequencyEstimate,
    decay: ModalDecayEstimate,
    frequency_uncertainty: ModalFrequencyUncertainty,
    decay_uncertainty: ModalDecayUncertainty,
    reservations: list[ModalParameterEstimateReason],
    insufficient: list[ModalParameterEstimateReason],
    invalid: list[ModalParameterEstimateReason],
) -> ModalParameterEstimateStatus:
    del frequency_uncertainty, decay_uncertainty
    if (
        invalid
        or _contains_invalid_reason(frequency.reasons, cfg)
        or _contains_invalid_reason(decay.reasons, cfg)
    ):
        return ModalParameterEstimateStatus.INVALID_INPUT
    if _hypothesis_not_allowed(hypothesis, cfg):
        return ModalParameterEstimateStatus.INSUFFICIENT_EVIDENCE
    if hypothesis.status in {
        ModalHypothesisStatus.REJECTED,
        ModalHypothesisStatus.INSUFFICIENT_EVIDENCE,
    }:
        insufficient.append(ModalParameterEstimateReason.INSUFFICIENT_EVIDENCE)
        return ModalParameterEstimateStatus.INSUFFICIENT_EVIDENCE
    if not frequency.valid:
        insufficient.append(ModalParameterEstimateReason.INSUFFICIENT_EVIDENCE)
        return ModalParameterEstimateStatus.INSUFFICIENT_EVIDENCE
    if not decay.valid and not cfg.allow_missing_tau:
        insufficient.append(ModalParameterEstimateReason.INSUFFICIENT_EVIDENCE)
        return ModalParameterEstimateStatus.INSUFFICIENT_EVIDENCE
    if (
        not frequency.passes_dispersion_limits
        or (
            decay.available_value_count >= cfg.minimum_tau_value_count
            and not decay.passes_dispersion_limits
        )
    ):
        insufficient.append(ModalParameterEstimateReason.INSUFFICIENT_EVIDENCE)
        return ModalParameterEstimateStatus.INSUFFICIENT_EVIDENCE
    if not decay.valid:
        return ModalParameterEstimateStatus.PARTIAL
    if reservations or insufficient:
        return ModalParameterEstimateStatus.VALID_WITH_RESERVATIONS
    return ModalParameterEstimateStatus.VALID


def _hypothesis_not_allowed(
    hypothesis: ModalHypothesis,
    cfg: ModalParameterEstimationSettings,
) -> bool:
    return (
        (hypothesis.status is ModalHypothesisStatus.ACCEPTED and not cfg.allow_accepted_hypotheses)
        or (
            hypothesis.status is ModalHypothesisStatus.ACCEPTED_WITH_RESERVATIONS
            and not cfg.allow_accepted_with_reservations
        )
        or (
            hypothesis.status is ModalHypothesisStatus.INCONCLUSIVE
            and not cfg.allow_inconclusive_hypotheses
        )
        or (
            hypothesis.status is ModalHypothesisStatus.REJECTED
            and not cfg.allow_rejected_hypotheses_for_audit
        )
        or (
            hypothesis.status is ModalHypothesisStatus.INSUFFICIENT_EVIDENCE
            and not cfg.allow_insufficient_evidence_hypotheses_for_audit
        )
    )


def _extend_by_role(
    reasons: Iterable[ModalParameterEstimateReason],
    cfg: ModalParameterEstimationSettings,
    reservations: list[ModalParameterEstimateReason],
    insufficient: list[ModalParameterEstimateReason],
    invalid: list[ModalParameterEstimateReason],
) -> None:
    for reason in reasons:
        if reason in {
            ModalParameterEstimateReason.INVALID_HYPOTHESIS,
            ModalParameterEstimateReason.INVALID_WEIGHT_VALUE,
        }:
            invalid.append(reason)
        elif reason in {
            ModalParameterEstimateReason.INVALID_FREQUENCY_VALUE,
            ModalParameterEstimateReason.INVALID_DECAY_VALUE,
        }:
            if cfg.finite_value_policy is FiniteValuePolicy.EXCLUDE_WITH_DIAGNOSTIC:
                reservations.append(reason)
            else:
                invalid.append(reason)
        elif reason in {
            ModalParameterEstimateReason.INSUFFICIENT_FREQUENCY_VALUES,
            ModalParameterEstimateReason.INSUFFICIENT_DECAY_VALUES,
            ModalParameterEstimateReason.EXCESSIVE_FREQUENCY_DISPERSION,
            ModalParameterEstimateReason.EXCESSIVE_DECAY_DISPERSION,
            ModalParameterEstimateReason.INSUFFICIENT_EVIDENCE,
            ModalParameterEstimateReason.INSUFFICIENT_WEIGHT_VALUES,
        }:
            insufficient.append(reason)
        elif reason in {
            ModalParameterEstimateReason.MISSING_FREQUENCY_UNCERTAINTY,
            ModalParameterEstimateReason.MISSING_DECAY_UNCERTAINTY,
            ModalParameterEstimateReason.MISSING_WEIGHT_VALUE,
        }:
            reservations.append(reason)


def _hypothesis_refs(hypothesis: ModalHypothesis) -> tuple[CandidateReference, ...]:
    if hypothesis.chain is None:
        return ()
    return tuple(node.candidate_ref for node in hypothesis.chain.nodes)


def _weights_for_refs(
    refs: tuple[CandidateReference, ...],
    method: ParameterWeightingMethod,
    local_association_costs: tuple[float | None, ...] | None = None,
) -> tuple[
    tuple[float | None, ...],
    tuple[float | None, ...],
    tuple[ModalParameterEstimateReason, ...],
    tuple[str, ...],
]:
    reasons: list[ModalParameterEstimateReason] = []
    diagnostics = [
        f"weighting_method={method.value}",
        "weights_are_non_negative_and_normalized_for_audit",
    ]
    if local_association_costs is None:
        local_association_costs = tuple(None for _ in refs)
    if len(local_association_costs) != len(refs):
        raise ValueError("local association costs must align with references.")
    raw_weights: list[float | None] = []
    for ref, local_cost in zip(refs, local_association_costs, strict=True):
        if method is ParameterWeightingMethod.UNIFORM:
            weight = 1.0
        elif method is ParameterWeightingMethod.TRACKING_COVERAGE:
            weight = ref.coverage_fraction
        elif method is ParameterWeightingMethod.FREQUENCY_FIT_QUALITY:
            weight = (
                1.0 / (1.0 + ref.frequency_fit_rmse_hz)
                if ref.frequency_fit_rmse_hz is not None
                else None
            )
        elif method is ParameterWeightingMethod.AMPLITUDE_FIT_QUALITY:
            weight = ref.amplitude_fit_r_squared
        elif method is ParameterWeightingMethod.INVERSE_ASSOCIATION_COST:
            weight = 1.0 / (1.0 + local_cost) if local_cost is not None else None
        else:
            inverse_cost = 1.0 / (1.0 + local_cost) if local_cost is not None else None
            pieces = (
                ref.coverage_fraction,
                (
                    1.0 / (1.0 + ref.frequency_fit_rmse_hz)
                    if ref.frequency_fit_rmse_hz is not None
                    else None
                ),
                ref.amplitude_fit_r_squared,
                inverse_cost,
            )
            if any(value is None for value in pieces):
                weight = None
            else:
                weight = pieces[0] * pieces[1] * pieces[2] * pieces[3]
        raw_weights.append(weight)
    weights: list[float | None] = []
    for value in raw_weights:
        if value is None:
            reasons.append(ModalParameterEstimateReason.MISSING_WEIGHT_VALUE)
            weights.append(None)
        elif not isfinite(value) or value < 0.0:
            reasons.append(ModalParameterEstimateReason.INVALID_WEIGHT_VALUE)
            weights.append(None)
        else:
            weights.append(float(value))
    finite_weights = tuple(
        value for value in weights
        if value is not None and isfinite(value) and value >= 0.0
    )
    total = sum(finite_weights)
    if weights and not finite_weights:
        reasons.append(ModalParameterEstimateReason.INSUFFICIENT_WEIGHT_VALUES)
    if finite_weights and total <= 0.0:
        reasons.append(ModalParameterEstimateReason.INSUFFICIENT_WEIGHT_VALUES)
        diagnostics.append("all_weights_zero")
    normalized: tuple[float | None, ...]
    if not reasons and total > 0.0:
        normalized = tuple(value / total if value is not None else None for value in weights)
    else:
        normalized = tuple(None for _ in weights)
    if ModalParameterEstimateReason.MISSING_WEIGHT_VALUE in reasons:
        diagnostics.append("weight_source_value_missing")
    if ModalParameterEstimateReason.INVALID_WEIGHT_VALUE in reasons:
        diagnostics.append("weight_source_value_invalid")
    return (
        tuple(weights),
        normalized,
        _ordered_reasons(reasons),
        tuple(dict.fromkeys(diagnostics)),
    )


def _local_association_costs(hypothesis: ModalHypothesis) -> tuple[float | None, ...]:
    if hypothesis.chain is None:
        return ()
    costs: list[float | None] = []
    for node in hypothesis.chain.nodes:
        available = tuple(
            value for value in (
                node.incoming_association_cost,
                node.outgoing_association_cost,
            )
            if value is not None
        )
        costs.append(sum(available) / len(available) if available else None)
    return tuple(costs)


def _location(
    values: tuple[float, ...],
    normalized_weights: tuple[float, ...],
    method: ParameterLocationMethod,
) -> float:
    if method is ParameterLocationMethod.ARITHMETIC_MEAN:
        return sum(values) / len(values)
    if method is ParameterLocationMethod.MEDIAN:
        return _median(values)
    if method is ParameterLocationMethod.WEIGHTED_MEAN:
        return sum(value * weight for value, weight in zip(values, normalized_weights, strict=True))
    if method is ParameterLocationMethod.WEIGHTED_MEDIAN:
        return _weighted_median(values, normalized_weights)
    if method is ParameterLocationMethod.GEOMETRIC_MEAN:
        return exp(sum(log(value) for value in values) / len(values))
    if method is ParameterLocationMethod.GEOMETRIC_MEDIAN:
        return exp(_median(tuple(log(value) for value in values)))
    raise ValueError("unknown location method.")


def _basic_stats(values: tuple[float, ...]) -> dict[str, float | None]:
    if not values:
        return {
            "min": None,
            "max": None,
            "range": None,
            "mean": None,
            "median": None,
            "std": None,
            "mad": None,
        }
    minimum = min(values)
    maximum = max(values)
    mean = sum(values) / len(values)
    median = _median(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    mad = _median(tuple(abs(value - median) for value in values))
    return {
        "min": minimum,
        "max": maximum,
        "range": maximum - minimum,
        "mean": mean,
        "median": median,
        "std": sqrt(variance),
        "mad": mad,
    }


def _median(values: tuple[float, ...]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _weighted_median(values: tuple[float, ...], normalized_weights: tuple[float, ...]) -> float:
    if len(values) != len(normalized_weights):
        raise ValueError("weighted median requires aligned values and weights.")
    pairs = sorted(zip(values, normalized_weights, strict=True), key=lambda item: item[0])
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= 0.5 or isclose(cumulative, 0.5, rel_tol=1e-12, abs_tol=1e-12):
            return value
    return pairs[-1][0]


def _sample_standard_deviation(values: tuple[float, ...]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _rms(values: tuple[float, ...]) -> float | None:
    if not values:
        return None
    return sqrt(sum(value * value for value in values) / len(values))


def _linear_fit_by_ordinal(values: tuple[float, ...]) -> tuple[float, float, float, float | None]:
    xs = tuple(float(index) for index in range(len(values)))
    x_mean = sum(xs) / len(xs)
    y_mean = sum(values) / len(values)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values, strict=True)) / denominator
    intercept = y_mean - slope * x_mean
    fitted = tuple(intercept + slope * x for x in xs)
    residual = sum((y - f) ** 2 for y, f in zip(values, fitted, strict=True))
    total = sum((y - y_mean) ** 2 for y in values)
    rmse = sqrt(residual / len(values))
    r_squared = 1.0 - residual / total if total > 0.0 else None
    return slope, intercept, rmse, r_squared


def _bootstrap_locations(
    values: tuple[float, ...],
    normalized_weights: tuple[float | None, ...],
    method: ParameterLocationMethod,
    sample_count: int,
    seed: int | None,
) -> tuple[float, ...]:
    rng = Random(seed)
    count = len(values)
    clean_weights = tuple(
        weight if weight is not None else 1.0 / count
        for weight in normalized_weights
    )
    estimates = []
    for _ in range(sample_count):
        indices = tuple(rng.randrange(count) for _ in range(count))
        sample_values = tuple(values[index] for index in indices)
        sample_weights_raw = tuple(clean_weights[index] for index in indices)
        total = sum(sample_weights_raw)
        sample_weights = tuple(value / total for value in sample_weights_raw) if total > 0.0 else tuple(1.0 / count for _ in indices)
        estimates.append(_location(sample_values, sample_weights, method))
    return tuple(estimates)


def _bootstrap_log_tau_locations(
    values: tuple[float, ...],
    normalized_weights: tuple[float | None, ...],
    method: ParameterLocationMethod,
    sample_count: int,
    seed: int | None,
) -> tuple[float, ...]:
    estimates = _bootstrap_locations(values, normalized_weights, method, sample_count, seed)
    return tuple(log(value) for value in estimates)


def _quantile(values: tuple[float, ...], q: float) -> float:
    if not values:
        raise ValueError("quantile requires values.")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _positive_finite(value: object, cfg: ModalParameterEstimationSettings) -> bool:
    return (
        isinstance(value, (int, float))
        and isfinite(float(value))
        and float(value) > cfg.minimum_positive_value
    )


def _finite_or_none(value: float | None) -> float | None:
    if value is None or not isinstance(value, (int, float)) or not isfinite(float(value)):
        return None
    return float(value)


def _finite_nonnegative_or_none(value: float | None) -> float | None:
    value = _finite_or_none(value)
    if value is None or value < 0.0:
        return None
    return value


def _fraction_or_none(value: float | None) -> float | None:
    value = _finite_or_none(value)
    if value is None or not 0.0 <= value <= 1.0:
        return None
    return value


def _contains_invalid_reason(
    reasons: Iterable[ModalParameterEstimateReason],
    cfg: ModalParameterEstimationSettings | None = None,
) -> bool:
    value_reasons = {
        ModalParameterEstimateReason.INVALID_FREQUENCY_VALUE,
        ModalParameterEstimateReason.INVALID_DECAY_VALUE,
    }
    if cfg is not None and cfg.finite_value_policy is FiniteValuePolicy.EXCLUDE_WITH_DIAGNOSTIC:
        value_reasons = set()
    return any(
        reason in {
            ModalParameterEstimateReason.INVALID_HYPOTHESIS,
            ModalParameterEstimateReason.INVALID_WEIGHT_VALUE,
        }
        or reason in value_reasons
        for reason in reasons
    )


def _exceeds(value: float | None, limit: float | None) -> bool:
    return (
        value is not None
        and limit is not None
        and value > limit
        and not isclose(value, limit, rel_tol=1e-12, abs_tol=1e-12)
    )


def _hypothesis_sort_key_for_parameters(hypothesis: ModalHypothesis | object) -> tuple:
    if not isinstance(hypothesis, ModalHypothesis) or hypothesis.chain is None:
        return (99, "", "", "")
    refs = _hypothesis_refs(hypothesis)
    return (
        _DYNAMIC_LABEL_INDEX[hypothesis.chain.start_dynamic_label],
        refs[0].representative_frequency_hz if refs else 0.0,
        tuple(
            (
                ref.dynamic_label,
                ref.recording_id,
                ref.candidate_id,
                ref.source_track_id,
            )
            for ref in refs
        ),
        hypothesis.hypothesis_id,
    )


def _estimate_sort_key(estimate: ModalParameterEstimate) -> tuple:
    labels = estimate.provenance.condition_labels
    first_index = _DYNAMIC_LABEL_INDEX.get(labels[0], 99) if labels else 99
    first_frequency = (
        estimate.frequency_estimate.values_hz[0]
        if estimate.frequency_estimate.values_hz
        else 0.0
    )
    return (
        first_index,
        first_frequency,
        estimate.provenance.source_chain_id or "",
        estimate.hypothesis_id,
        estimate.estimate_id,
    )


def _sequence_from_hypotheses(hypotheses: tuple[ModalHypothesis | object, ...]) -> tuple[str, ...]:
    labels = tuple(
        ref.dynamic_label
        for hypothesis in hypotheses
        if isinstance(hypothesis, ModalHypothesis)
        for ref in _hypothesis_refs(hypothesis)
        if ref.dynamic_label in _DYNAMIC_LABEL_INDEX
    )
    if not labels:
        return ("pp",)
    indices = sorted({_DYNAMIC_LABEL_INDEX[label] for label in labels})
    return tuple(DYNAMIC_LABEL_ORDER[index] for index in range(indices[0], indices[-1] + 1))


def _validate_sequence(labels: tuple[str, ...]) -> None:
    if not isinstance(labels, tuple) or not labels:
        raise ValueError("sequence must be a nonempty tuple.")
    if any(label not in _DYNAMIC_LABEL_INDEX for label in labels):
        raise ValueError("sequence contains an unknown dynamic label.")
    if len(labels) != len(set(labels)):
        raise ValueError("sequence must not contain repeated labels.")
    indices = tuple(_DYNAMIC_LABEL_INDEX[label] for label in labels)
    if indices != tuple(sorted(indices)):
        raise ValueError("sequence must follow nominal dynamic order.")
    if any(right - left != 1 for left, right in zip(indices, indices[1:], strict=False)):
        raise ValueError("sequence must be contiguous.")


def _estimate_id(
    hypothesis_id: str,
    provenance: ModalParameterProvenance,
    frequency: ModalFrequencyEstimate,
    decay: ModalDecayEstimate,
    frequency_uncertainty: ModalFrequencyUncertainty,
    decay_uncertainty: ModalDecayUncertainty,
    status: ModalParameterEstimateStatus,
) -> str:
    payload = {
        "hypothesis_id": hypothesis_id,
        "source_chain_id": provenance.source_chain_id,
        "candidate_ids": provenance.candidate_ids,
        "recording_ids": provenance.recording_ids,
        "condition_labels": provenance.condition_labels,
        "match_ids": provenance.match_ids,
        "frequency_values_hz": frequency.values_hz,
        "representative_frequency_hz": frequency.representative_frequency_hz,
        "tau_values_s": decay.tau_values_s,
        "representative_tau_s": decay.representative_tau_s,
        "frequency_uncertainty_method": frequency_uncertainty.method.value,
        "frequency_uncertainty": frequency_uncertainty.standard_uncertainty_hz,
        "decay_uncertainty_method": decay_uncertainty.method.value,
        "decay_uncertainty": decay_uncertainty.standard_uncertainty_log_tau,
        "status": status.value,
        "settings_fingerprint": provenance.settings_fingerprint,
    }
    encoded = json.dumps(_canonicalize(payload), sort_keys=True, separators=(",", ":"))
    return "modal-parameter-estimate-" + sha1(encoded.encode("utf-8")).hexdigest()[:16]


def _split_context_id(item: object) -> str:
    payload = {
        "source": _ref_identity(getattr(item, "source_candidate_ref", None)),
        "targets": tuple(_ref_identity(ref) for ref in getattr(item, "target_candidate_refs", ())),
        "costs": getattr(item, "costs", ()),
    }
    encoded = json.dumps(_canonicalize(payload), sort_keys=True, separators=(",", ":"))
    return "possible-split-" + sha1(encoded.encode("utf-8")).hexdigest()[:16]


def _merge_context_id(item: object) -> str:
    payload = {
        "sources": tuple(_ref_identity(ref) for ref in getattr(item, "source_candidate_refs", ())),
        "target": _ref_identity(getattr(item, "target_candidate_ref", None)),
        "costs": getattr(item, "costs", ()),
    }
    encoded = json.dumps(_canonicalize(payload), sort_keys=True, separators=(",", ":"))
    return "possible-merge-" + sha1(encoded.encode("utf-8")).hexdigest()[:16]


def _ref_identity(ref: CandidateReference | object | None) -> tuple | None:
    if not isinstance(ref, CandidateReference):
        return None
    return (
        ref.dynamic_label,
        ref.recording_id,
        ref.candidate_id,
        ref.source_track_id,
    )


def _canonicalize(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _canonicalize(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, float):
        return format(value, ".17g")
    return value


def _coerce_enum(instance: object, name: str, enum_type: type[Enum]) -> None:
    value = getattr(instance, name)
    if isinstance(value, enum_type):
        return
    try:
        object.__setattr__(instance, name, enum_type(value))
    except ValueError as exc:
        raise ValueError(f"{name} must be a recognized {enum_type.__name__}.") from exc


def _enum(value: object, enum_type: type[Enum], name: str) -> None:
    if not isinstance(value, enum_type):
        raise ValueError(f"{name} must be a {enum_type.__name__}.")


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
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be non-negative when provided.")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive when provided.")


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


def _validate_weights(
    weights: tuple[float | None, ...],
    normalized: tuple[float | None, ...],
    *,
    require_normalized_sum: bool,
) -> None:
    if len(weights) != len(normalized):
        raise ValueError("weights and normalized_weights must align.")
    for value in weights + normalized:
        if value is not None and (not isfinite(value) or value < 0.0):
            raise ValueError("weights must be finite and non-negative when present.")
    if require_normalized_sum and normalized:
        finite = tuple(value for value in normalized if value is not None)
        if len(finite) != len(normalized):
            raise ValueError("valid estimates require complete normalized weights.")
        if not isclose(sum(finite), 1.0, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("normalized weights must sum to one.")


def _ordered_reasons(
    reasons: Iterable[ModalParameterEstimateReason],
) -> tuple[ModalParameterEstimateReason, ...]:
    unique = {
        reason if isinstance(reason, ModalParameterEstimateReason) else ModalParameterEstimateReason(reason)
        for reason in reasons
    }
    return tuple(sorted(unique, key=lambda item: item.value))


def _reason_tuple(values: tuple[ModalParameterEstimateReason, ...], name: str) -> None:
    if not isinstance(values, tuple):
        raise ValueError(f"{name} must be an immutable tuple.")
    converted = _ordered_reasons(values)
    if values != converted:
        raise ValueError(f"{name} must contain unique reasons in deterministic order.")
