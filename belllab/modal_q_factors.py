"""Operational Q-factor and bandwidth estimates for modal hypotheses.

This layer consumes already-built :class:`ModalParameterEstimate` objects and,
optionally, already-calculated spectral peaks or spectra. It never reads audio,
recomputes FFT/STFT/tracking/candidates, creates new associations, closes gaps,
resolves split/merge contexts, or promotes any result to ``ModalMode``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha1
import json
from math import exp, isclose, isfinite, log, pi, sqrt
from random import Random

from belllab.global_spectrum import GlobalSpectralCharacterization, GlobalSpectralPeakMetric
from belllab.modal_parameters import (
    ModalDecayUncertainty,
    ModalParameterEstimate,
    ModalParameterEstimateStatus,
    ModalParameterEstimationResult,
)
from belllab.types import SpectralPeak, Spectrum


_DECAY_CONVENTION_TEXT = (
    "A(t)=A0 exp(-t/tau); Q_decay=pi*f*tau for an approximately "
    "exponential amplitude decay, weak damping, isolated component, and "
    "approximately stable frequency."
)


class ModalQFactorEstimateStatus(str, Enum):
    """Mutually exclusive states for an operational Q-factor estimate."""

    VALID = "valid"
    VALID_WITH_RESERVATIONS = "valid_with_reservations"
    PARTIAL = "partial"
    INCONCLUSIVE = "inconclusive"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    INVALID_INPUT = "invalid_input"


class ModalQFactorEstimateReason(str, Enum):
    """Typed reasons separated by support, reservation, insufficiency, or invalidity."""

    SUFFICIENT_DECAY_EVIDENCE = "sufficient_decay_evidence"
    SUFFICIENT_BANDWIDTH_EVIDENCE = "sufficient_bandwidth_evidence"
    DECAY_METHOD_AVAILABLE = "decay_method_available"
    BANDWIDTH_METHOD_AVAILABLE = "bandwidth_method_available"
    METHODS_CONSISTENT = "methods_consistent"
    METHODS_PARTIALLY_CONSISTENT = "methods_partially_consistent"
    METHODS_INCONSISTENT = "methods_inconsistent"
    MISSING_FREQUENCY = "missing_frequency"
    MISSING_TAU = "missing_tau"
    MISSING_BANDWIDTH = "missing_bandwidth"
    MISSING_FREQUENCY_UNCERTAINTY = "missing_frequency_uncertainty"
    MISSING_TAU_UNCERTAINTY = "missing_tau_uncertainty"
    MISSING_BANDWIDTH_UNCERTAINTY = "missing_bandwidth_uncertainty"
    INVALID_FREQUENCY = "invalid_frequency"
    INVALID_TAU = "invalid_tau"
    INVALID_BANDWIDTH = "invalid_bandwidth"
    INVALID_PARAMETER_ESTIMATE = "invalid_parameter_estimate"
    INSUFFICIENT_SPECTRAL_RESOLUTION = "insufficient_spectral_resolution"
    PEAK_NOT_ISOLATED = "peak_not_isolated"
    NEIGHBORING_PEAK_INTERFERENCE = "neighboring_peak_interference"
    BANDWIDTH_AT_RESOLUTION_LIMIT = "bandwidth_at_resolution_limit"
    EXCESSIVE_METHOD_DISAGREEMENT = "excessive_method_disagreement"
    AMBIGUOUS_SOURCE_MATCH = "ambiguous_source_match"
    NEAR_THRESHOLD_SOURCE_MATCH = "near_threshold_source_match"
    POSSIBLE_SPLIT_CONTEXT = "possible_split_context"
    POSSIBLE_MERGE_CONTEXT = "possible_merge_context"
    UNSUPPORTED_PARAMETER_STATUS = "unsupported_parameter_status"
    REQUIRED_DECAY_METHOD_MISSING = "required_decay_method_missing"
    REQUIRED_BANDWIDTH_METHOD_MISSING = "required_bandwidth_method_missing"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    WELL_RESOLVED = "well_resolved"
    MARGINALLY_RESOLVED = "marginally_resolved"
    RESOLUTION_LIMITED = "resolution_limited"
    UNRESOLVED = "unresolved"


class ModalQDecayConvention(str, Enum):
    """Explicit mathematical convention for decay-derived Q."""

    AMPLITUDE_EXPONENTIAL = "amplitude_exponential"


class ModalBandwidthDefinition(str, Enum):
    """Explicit bandwidth definitions; amplitude and power are not mixed."""

    AMPLITUDE_MINUS_3_DB = "amplitude_minus_3_db"
    POWER_MINUS_3_DB = "power_minus_3_db"
    HALF_PROMINENCE_AMPLITUDE = "half_prominence_amplitude"
    HALF_PROMINENCE_POWER = "half_prominence_power"


class ModalQUncertaintyMethod(str, Enum):
    """Operational Q uncertainty summaries, not physical confidence intervals."""

    DISABLED = "disabled"
    LINEAR_PROPAGATION = "linear_propagation"
    PARAMETRIC_BOOTSTRAP = "parametric_bootstrap"


class ModalQConsistencyPolicy(str, Enum):
    """Decision policy when two valid Q methods disagree."""

    INCONCLUSIVE_ON_DISAGREEMENT = "inconclusive_on_disagreement"
    RESERVATION_ON_DISAGREEMENT = "reservation_on_disagreement"


class ModalQCombinationMethod(str, Enum):
    """Explicit policy for choosing or combining method-specific Q values."""

    NONE = "none"
    ARITHMETIC_MEAN = "arithmetic_mean"
    GEOMETRIC_MEAN = "geometric_mean"
    INVERSE_UNCERTAINTY_WEIGHTED = "inverse_uncertainty_weighted"
    PREFER_DECAY = "prefer_decay"
    PREFER_BANDWIDTH = "prefer_bandwidth"


class SpectralResolutionAssessment(str, Enum):
    """Operational relationship between bandwidth and frequency resolution."""

    WELL_RESOLVED = "well_resolved"
    MARGINALLY_RESOLVED = "marginally_resolved"
    RESOLUTION_LIMITED = "resolution_limited"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class ModalQFactorEstimationSettings:
    """Conservative, explicit settings for operational Q-factor estimation."""

    allow_valid_parameter_estimates: bool = True
    allow_valid_with_reservations: bool = True
    allow_partial_parameter_estimates: bool = True
    allow_insufficient_evidence_for_audit: bool = False
    allow_invalid_input_for_audit: bool = False

    enable_decay_method: bool = True
    require_decay_method: bool = False
    decay_convention: ModalQDecayConvention = ModalQDecayConvention.AMPLITUDE_EXPONENTIAL
    minimum_frequency_hz: float = 1e-12
    minimum_tau_s: float = 1e-12
    maximum_decay_q: float | None = None
    allow_missing_decay_uncertainty: bool = True

    enable_bandwidth_method: bool = True
    require_bandwidth_method: bool = False
    bandwidth_definition: ModalBandwidthDefinition = ModalBandwidthDefinition.AMPLITUDE_MINUS_3_DB
    bandwidth_level_db: float = -3.0
    minimum_bandwidth_hz: float = 1e-12
    minimum_spectral_resolution_ratio: float | None = 2.0
    unresolved_spectral_resolution_ratio: float = 1.0
    maximum_neighboring_peak_overlap_fraction: float | None = 0.0
    require_isolated_peak: bool = True
    allow_resolution_limited_bandwidth: bool = False
    allow_missing_bandwidth_uncertainty: bool = True

    maximum_relative_q_disagreement: float | None = 0.20
    maximum_log_q_difference: float | None = log(1.25)
    partial_consistency_multiplier: float = 2.0
    consistency_policy: ModalQConsistencyPolicy = ModalQConsistencyPolicy.INCONCLUSIVE_ON_DISAGREEMENT
    prefer_decay_method: bool = False
    prefer_bandwidth_method: bool = False
    combine_consistent_methods: ModalQCombinationMethod = ModalQCombinationMethod.GEOMETRIC_MEAN

    uncertainty_method: ModalQUncertaintyMethod = ModalQUncertaintyMethod.LINEAR_PROPAGATION
    uncertainty_confidence_level: float = 0.95
    bootstrap_sample_count: int = 1000
    bootstrap_random_seed: int | None = 0

    reserve_ambiguous_matches: bool = True
    reserve_near_threshold_matches: bool = True
    reserve_possible_split_context: bool = True
    reserve_possible_merge_context: bool = True

    minimum_positive_value: float = 1e-12

    def __post_init__(self) -> None:
        for name, enum_type in (
            ("decay_convention", ModalQDecayConvention),
            ("bandwidth_definition", ModalBandwidthDefinition),
            ("uncertainty_method", ModalQUncertaintyMethod),
            ("consistency_policy", ModalQConsistencyPolicy),
            ("combine_consistent_methods", ModalQCombinationMethod),
        ):
            _coerce_enum(self, name, enum_type)
        for name in (
            "allow_valid_parameter_estimates",
            "allow_valid_with_reservations",
            "allow_partial_parameter_estimates",
            "allow_insufficient_evidence_for_audit",
            "allow_invalid_input_for_audit",
            "enable_decay_method",
            "require_decay_method",
            "allow_missing_decay_uncertainty",
            "enable_bandwidth_method",
            "require_bandwidth_method",
            "require_isolated_peak",
            "allow_resolution_limited_bandwidth",
            "allow_missing_bandwidth_uncertainty",
            "prefer_decay_method",
            "prefer_bandwidth_method",
            "reserve_ambiguous_matches",
            "reserve_near_threshold_matches",
            "reserve_possible_split_context",
            "reserve_possible_merge_context",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")
        if not self.enable_decay_method and not self.enable_bandwidth_method:
            raise ValueError("at least one Q estimation method must be enabled.")
        if self.require_decay_method and not self.enable_decay_method:
            raise ValueError("require_decay_method requires enable_decay_method.")
        if self.require_bandwidth_method and not self.enable_bandwidth_method:
            raise ValueError("require_bandwidth_method requires enable_bandwidth_method.")
        if self.prefer_decay_method and self.prefer_bandwidth_method:
            raise ValueError("decay and bandwidth methods cannot both be preferred.")
        _finite_optional(self.minimum_frequency_hz, "minimum_frequency_hz", positive=True)
        _finite_optional(self.minimum_tau_s, "minimum_tau_s", positive=True)
        _finite_optional(self.maximum_decay_q, "maximum_decay_q", positive=True)
        if not isfinite(self.bandwidth_level_db) or self.bandwidth_level_db >= 0.0:
            raise ValueError("bandwidth_level_db must be finite and negative.")
        _finite_optional(self.minimum_bandwidth_hz, "minimum_bandwidth_hz", positive=True)
        _finite_optional(
            self.minimum_spectral_resolution_ratio,
            "minimum_spectral_resolution_ratio",
            nonnegative=True,
        )
        _finite_optional(
            self.unresolved_spectral_resolution_ratio,
            "unresolved_spectral_resolution_ratio",
            nonnegative=True,
        )
        if (
            self.minimum_spectral_resolution_ratio is not None
            and self.unresolved_spectral_resolution_ratio > self.minimum_spectral_resolution_ratio
        ):
            raise ValueError("unresolved_spectral_resolution_ratio must not exceed minimum_spectral_resolution_ratio.")
        _fraction(
            self.maximum_neighboring_peak_overlap_fraction,
            "maximum_neighboring_peak_overlap_fraction",
        )
        _finite_optional(
            self.maximum_relative_q_disagreement,
            "maximum_relative_q_disagreement",
            nonnegative=True,
        )
        _finite_optional(self.maximum_log_q_difference, "maximum_log_q_difference", nonnegative=True)
        if (
            not isfinite(self.partial_consistency_multiplier)
            or self.partial_consistency_multiplier < 1.0
        ):
            raise ValueError("partial_consistency_multiplier must be finite and at least one.")
        if (
            not isfinite(self.uncertainty_confidence_level)
            or not 0.0 < self.uncertainty_confidence_level < 1.0
        ):
            raise ValueError("uncertainty_confidence_level must be finite and in (0, 1).")
        if self.bootstrap_sample_count <= 0:
            raise ValueError("bootstrap_sample_count must be positive.")
        if self.bootstrap_random_seed is not None and not isinstance(self.bootstrap_random_seed, int):
            raise ValueError("bootstrap_random_seed must be an int or None.")
        _finite_optional(self.minimum_positive_value, "minimum_positive_value", positive=True)


@dataclass(frozen=True, slots=True)
class ModalBandwidthSource:
    """Explicit already-calculated spectral source for bandwidth estimation."""

    spectrum_id: str | None = None
    center_frequency_hz: float | None = None
    frequency_axis_hz: tuple[float, ...] = ()
    magnitude_values: tuple[float, ...] = ()
    peak_frequencies_hz: tuple[float, ...] = ()
    frequency_resolution_hz: float | None = None
    bandwidth_uncertainty_hz: float | None = None
    precomputed_peak: SpectralPeak | GlobalSpectralPeakMetric | None = None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.spectrum_id is not None:
            _text(self.spectrum_id, "spectrum_id")
        _strings(self.diagnostics, "bandwidth source diagnostics", allow_empty=True)
        _finite_optional(self.center_frequency_hz, "source center_frequency_hz", positive=True)
        _finite_optional(self.frequency_resolution_hz, "source frequency_resolution_hz", positive=True)
        _finite_optional(self.bandwidth_uncertainty_hz, "source bandwidth_uncertainty_hz", nonnegative=True)
        for name, values in (
            ("frequency_axis_hz", self.frequency_axis_hz),
            ("magnitude_values", self.magnitude_values),
            ("peak_frequencies_hz", self.peak_frequencies_hz),
        ):
            if not isinstance(values, tuple):
                raise ValueError(f"{name} must be an immutable tuple.")


@dataclass(frozen=True, slots=True)
class ModalDecayQEstimate:
    representative_frequency_hz: float | None
    representative_tau_s: float | None
    q_decay: float | None
    decay_convention: str
    frequency_uncertainty_hz: float | None
    tau_uncertainty_s: float | None
    standard_uncertainty_q: float | None
    lower_bound_q: float | None
    upper_bound_q: float | None
    relative_uncertainty: float | None
    assumptions: tuple[str, ...]
    valid: bool
    passes_limits: bool
    reasons: tuple[ModalQFactorEstimateReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.decay_convention, "decay_convention")
        _strings(self.assumptions, "decay Q assumptions")
        _validate_optional_positive(self.representative_frequency_hz, "representative_frequency_hz")
        _validate_optional_positive(self.representative_tau_s, "representative_tau_s")
        _validate_optional_positive(self.q_decay, "q_decay")
        _finite_optional(self.frequency_uncertainty_hz, "frequency_uncertainty_hz", nonnegative=True)
        _finite_optional(self.tau_uncertainty_s, "tau_uncertainty_s", nonnegative=True)
        _finite_optional(self.standard_uncertainty_q, "standard_uncertainty_q", nonnegative=True)
        _validate_bounds(self.lower_bound_q, self.upper_bound_q, "decay Q bounds", positive=True)
        _finite_optional(self.relative_uncertainty, "relative_uncertainty", nonnegative=True)
        _reason_tuple(self.reasons, "decay Q reasons")
        _strings(self.diagnostics, "decay Q diagnostics", allow_empty=True)
        if self.valid and self.q_decay is None:
            raise ValueError("valid decay Q estimates require q_decay.")


@dataclass(frozen=True, slots=True)
class ModalPeakIsolationEvidence:
    target_peak_frequency_hz: float | None
    nearest_lower_peak_frequency_hz: float | None
    nearest_upper_peak_frequency_hz: float | None
    nearest_peak_distance_hz: float | None
    bandwidth_hz: float | None
    overlap_fraction: float | None
    isolated: bool | None
    passes: bool
    reasons: tuple[ModalQFactorEstimateReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("target_peak_frequency_hz", self.target_peak_frequency_hz),
            ("nearest_lower_peak_frequency_hz", self.nearest_lower_peak_frequency_hz),
            ("nearest_upper_peak_frequency_hz", self.nearest_upper_peak_frequency_hz),
        ):
            _validate_optional_positive(value, name)
        _finite_optional(self.nearest_peak_distance_hz, "nearest_peak_distance_hz", positive=True)
        _validate_optional_positive(self.bandwidth_hz, "bandwidth_hz")
        _fraction(self.overlap_fraction, "overlap_fraction")
        if self.isolated is not None and not isinstance(self.isolated, bool):
            raise ValueError("isolated must be a boolean or None.")
        _reason_tuple(self.reasons, "peak isolation reasons")
        _strings(self.diagnostics, "peak isolation diagnostics", allow_empty=True)


@dataclass(frozen=True, slots=True)
class ModalBandwidthEstimate:
    center_frequency_hz: float | None
    lower_frequency_hz: float | None
    upper_frequency_hz: float | None
    bandwidth_hz: float | None
    bandwidth_definition: ModalBandwidthDefinition
    bandwidth_level_db: float | None
    left_crossing_found: bool
    right_crossing_found: bool
    interpolation_method: str | None
    frequency_resolution_hz: float | None
    resolution_ratio: float | None
    resolution_assessment: SpectralResolutionAssessment | None
    neighboring_peak_distance_hz: float | None
    neighboring_peak_overlap_fraction: float | None
    isolated_peak: bool | None
    resolution_limited: bool
    valid: bool
    reasons: tuple[ModalQFactorEstimateReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _enum(self.bandwidth_definition, ModalBandwidthDefinition, "bandwidth_definition")
        if self.resolution_assessment is not None:
            _enum(self.resolution_assessment, SpectralResolutionAssessment, "resolution_assessment")
        _validate_optional_positive(self.center_frequency_hz, "center_frequency_hz")
        _validate_optional_nonnegative(self.lower_frequency_hz, "lower_frequency_hz")
        _validate_optional_nonnegative(self.upper_frequency_hz, "upper_frequency_hz")
        _validate_optional_positive(self.bandwidth_hz, "bandwidth_hz")
        _finite_optional(self.bandwidth_level_db, "bandwidth_level_db")
        _finite_optional(self.frequency_resolution_hz, "frequency_resolution_hz", positive=True)
        _finite_optional(self.resolution_ratio, "resolution_ratio", nonnegative=True)
        _finite_optional(self.neighboring_peak_distance_hz, "neighboring_peak_distance_hz", positive=True)
        _fraction(self.neighboring_peak_overlap_fraction, "neighboring_peak_overlap_fraction")
        if self.lower_frequency_hz is not None and self.upper_frequency_hz is not None:
            if self.upper_frequency_hz <= self.lower_frequency_hz:
                raise ValueError("bandwidth crossings must be ordered.")
            if self.center_frequency_hz is not None and not self.lower_frequency_hz <= self.center_frequency_hz <= self.upper_frequency_hz:
                raise ValueError("center_frequency_hz must lie within bandwidth crossings.")
        _reason_tuple(self.reasons, "bandwidth reasons")
        _strings(self.diagnostics, "bandwidth diagnostics", allow_empty=True)
        if self.valid and self.bandwidth_hz is None:
            raise ValueError("valid bandwidth estimates require bandwidth_hz.")


@dataclass(frozen=True, slots=True)
class ModalBandwidthQEstimate:
    center_frequency_hz: float | None
    bandwidth_hz: float | None
    q_bandwidth: float | None
    frequency_resolution_hz: float | None
    bandwidth_uncertainty_hz: float | None
    frequency_uncertainty_hz: float | None
    resolution_uncertainty_component_hz: float | None
    standard_uncertainty_q: float | None
    lower_bound_q: float | None
    upper_bound_q: float | None
    relative_uncertainty: float | None
    resolution_limited: bool
    isolated_peak: bool | None
    valid: bool
    reasons: tuple[ModalQFactorEstimateReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_optional_positive(self.center_frequency_hz, "center_frequency_hz")
        _validate_optional_positive(self.bandwidth_hz, "bandwidth_hz")
        _validate_optional_positive(self.q_bandwidth, "q_bandwidth")
        _finite_optional(self.frequency_resolution_hz, "frequency_resolution_hz", positive=True)
        _finite_optional(self.bandwidth_uncertainty_hz, "bandwidth_uncertainty_hz", nonnegative=True)
        _finite_optional(self.frequency_uncertainty_hz, "frequency_uncertainty_hz", nonnegative=True)
        _finite_optional(self.resolution_uncertainty_component_hz, "resolution_uncertainty_component_hz", nonnegative=True)
        _finite_optional(self.standard_uncertainty_q, "standard_uncertainty_q", nonnegative=True)
        _validate_bounds(self.lower_bound_q, self.upper_bound_q, "bandwidth Q bounds", positive=True)
        _finite_optional(self.relative_uncertainty, "relative_uncertainty", nonnegative=True)
        _reason_tuple(self.reasons, "bandwidth Q reasons")
        _strings(self.diagnostics, "bandwidth Q diagnostics", allow_empty=True)
        if self.valid and self.q_bandwidth is None:
            raise ValueError("valid bandwidth Q estimates require q_bandwidth.")


@dataclass(frozen=True, slots=True)
class ModalQMethodCombination:
    method: ModalQCombinationMethod
    combined_q: float | None
    combined_uncertainty: float | None
    weights: tuple[float | None, ...]
    normalized_weights: tuple[float | None, ...]
    valid: bool
    reasons: tuple[ModalQFactorEstimateReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _enum(self.method, ModalQCombinationMethod, "combination method")
        _validate_optional_positive(self.combined_q, "combined_q")
        _finite_optional(self.combined_uncertainty, "combined_uncertainty", nonnegative=True)
        _validate_weights(self.weights, self.normalized_weights, require_normalized_sum=self.valid)
        _reason_tuple(self.reasons, "combination reasons")
        _strings(self.diagnostics, "combination diagnostics", allow_empty=True)
        if self.valid and self.combined_q is None:
            raise ValueError("valid combinations require combined_q.")


@dataclass(frozen=True, slots=True)
class ModalQMethodComparison:
    q_decay: float | None
    q_bandwidth: float | None
    absolute_difference: float | None
    relative_symmetric_difference: float | None
    log_q_difference: float | None
    ratio_decay_to_bandwidth: float | None
    consistent: bool
    partially_consistent: bool
    inconsistent: bool
    preferred_method: str | None
    combined_q: float | None
    combined_uncertainty: float | None
    reasons: tuple[ModalQFactorEstimateReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_optional_positive(self.q_decay, "q_decay")
        _validate_optional_positive(self.q_bandwidth, "q_bandwidth")
        _finite_optional(self.absolute_difference, "absolute_difference", nonnegative=True)
        _finite_optional(self.relative_symmetric_difference, "relative_symmetric_difference", nonnegative=True)
        _finite_optional(self.log_q_difference, "log_q_difference", nonnegative=True)
        _validate_optional_positive(self.ratio_decay_to_bandwidth, "ratio_decay_to_bandwidth")
        if sum((self.consistent, self.partially_consistent, self.inconsistent)) > 1:
            raise ValueError("method consistency states must be exclusive.")
        _validate_optional_positive(self.combined_q, "combined_q")
        _finite_optional(self.combined_uncertainty, "combined_uncertainty", nonnegative=True)
        _reason_tuple(self.reasons, "method comparison reasons")
        _strings(self.diagnostics, "method comparison diagnostics", allow_empty=True)


@dataclass(frozen=True, slots=True)
class ModalQFactorProvenance:
    modal_parameter_estimate_id: str | None
    hypothesis_id: str | None
    source_chain_id: str | None
    candidate_ids: tuple[int, ...]
    recording_ids: tuple[str, ...]
    spectrum_ids: tuple[str, ...]
    frequency_source: str | None
    tau_source: str | None
    bandwidth_source: str | None
    settings_fingerprint: str
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("modal_parameter_estimate_id", self.modal_parameter_estimate_id),
            ("hypothesis_id", self.hypothesis_id),
            ("source_chain_id", self.source_chain_id),
            ("frequency_source", self.frequency_source),
            ("tau_source", self.tau_source),
            ("bandwidth_source", self.bandwidth_source),
        ):
            if value is not None:
                _text(value, name)
        if any(not isinstance(item, int) or item < 0 for item in self.candidate_ids):
            raise ValueError("candidate_ids must contain non-negative integers.")
        _strings(self.recording_ids, "recording_ids", allow_empty=True)
        _strings(self.spectrum_ids, "spectrum_ids", allow_empty=True)
        _text(self.settings_fingerprint, "settings_fingerprint")
        _strings(self.diagnostics, "Q provenance diagnostics", allow_empty=True)


@dataclass(frozen=True, slots=True)
class ModalQFactorEstimate:
    estimate_id: str
    modal_parameter_estimate_id: str | None
    hypothesis_id: str | None
    status: ModalQFactorEstimateStatus
    decay_q_estimate: ModalDecayQEstimate | None
    bandwidth_estimate: ModalBandwidthEstimate | None
    bandwidth_q_estimate: ModalBandwidthQEstimate | None
    peak_isolation_evidence: ModalPeakIsolationEvidence | None
    method_comparison: ModalQMethodComparison | None
    representative_q: float | None
    representative_q_method: str | None
    representative_q_uncertainty: float | None
    supporting_reasons: tuple[ModalQFactorEstimateReason, ...]
    reservation_reasons: tuple[ModalQFactorEstimateReason, ...]
    inconclusive_reasons: tuple[ModalQFactorEstimateReason, ...]
    insufficient_evidence_reasons: tuple[ModalQFactorEstimateReason, ...]
    invalid_reasons: tuple[ModalQFactorEstimateReason, ...]
    valid: bool
    requires_review: bool
    provenance: ModalQFactorProvenance
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.estimate_id, "estimate_id")
        if self.modal_parameter_estimate_id is not None:
            _text(self.modal_parameter_estimate_id, "modal_parameter_estimate_id")
        if self.hypothesis_id is not None:
            _text(self.hypothesis_id, "hypothesis_id")
        _enum(self.status, ModalQFactorEstimateStatus, "status")
        _validate_optional_positive(self.representative_q, "representative_q")
        _finite_optional(self.representative_q_uncertainty, "representative_q_uncertainty", nonnegative=True)
        for name in (
            "supporting_reasons",
            "reservation_reasons",
            "inconclusive_reasons",
            "insufficient_evidence_reasons",
            "invalid_reasons",
        ):
            _reason_tuple(getattr(self, name), name)
        expected_valid = self.status in {
            ModalQFactorEstimateStatus.VALID,
            ModalQFactorEstimateStatus.VALID_WITH_RESERVATIONS,
        }
        if self.valid != expected_valid:
            raise ValueError("valid flag must agree with status.")
        if self.requires_review != (self.status is not ModalQFactorEstimateStatus.VALID):
            raise ValueError("requires_review must agree with status.")
        _strings(self.diagnostics, "Q estimate diagnostics", allow_empty=True)


@dataclass(frozen=True, slots=True)
class ModalQFactorEstimationResult:
    sequence: tuple[str, ...]
    estimates: tuple[ModalQFactorEstimate, ...]
    valid_estimates: tuple[ModalQFactorEstimate, ...]
    valid_with_reservations_estimates: tuple[ModalQFactorEstimate, ...]
    partial_estimates: tuple[ModalQFactorEstimate, ...]
    inconclusive_estimates: tuple[ModalQFactorEstimate, ...]
    insufficient_evidence_estimates: tuple[ModalQFactorEstimate, ...]
    invalid_estimates: tuple[ModalQFactorEstimate, ...]
    estimate_count: int
    valid_count: int
    valid_with_reservations_count: int
    partial_count: int
    inconclusive_count: int
    insufficient_evidence_count: int
    invalid_count: int
    source_parameter_estimate_count: int
    settings: ModalQFactorEstimationSettings
    valid: bool
    failure_reason: str | None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.sequence, tuple) or any(not isinstance(label, str) or not label for label in self.sequence):
            raise ValueError("sequence must be a tuple of non-empty labels.")
        if len({item.estimate_id for item in self.estimates}) != len(self.estimates):
            raise ValueError("Q estimate IDs must be unique.")
        subsets = {
            ModalQFactorEstimateStatus.VALID: self.valid_estimates,
            ModalQFactorEstimateStatus.VALID_WITH_RESERVATIONS: self.valid_with_reservations_estimates,
            ModalQFactorEstimateStatus.PARTIAL: self.partial_estimates,
            ModalQFactorEstimateStatus.INCONCLUSIVE: self.inconclusive_estimates,
            ModalQFactorEstimateStatus.INSUFFICIENT_EVIDENCE: self.insufficient_evidence_estimates,
            ModalQFactorEstimateStatus.INVALID_INPUT: self.invalid_estimates,
        }
        for status, subset in subsets.items():
            expected = tuple(item for item in self.estimates if item.status is status)
            if subset != expected:
                raise ValueError("Q estimate subsets are incoherent.")
        counts = {
            "estimate_count": len(self.estimates),
            "valid_count": len(self.valid_estimates),
            "valid_with_reservations_count": len(self.valid_with_reservations_estimates),
            "partial_count": len(self.partial_estimates),
            "inconclusive_count": len(self.inconclusive_estimates),
            "insufficient_evidence_count": len(self.insufficient_evidence_estimates),
            "invalid_count": len(self.invalid_estimates),
        }
        for name, expected in counts.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} is incoherent.")
        if self.source_parameter_estimate_count != len(self.estimates):
            raise ValueError("one Q estimate is required per source parameter estimate.")
        if self.valid != (self.failure_reason is None):
            raise ValueError("valid and failure_reason are incoherent.")
        _strings(self.diagnostics, "Q result diagnostics", allow_empty=True)


def estimate_modal_q_factors(
    parameter_estimates: ModalParameterEstimationResult | Iterable[ModalParameterEstimate],
    settings: ModalQFactorEstimationSettings | None = None,
    *,
    bandwidth_sources: Mapping[str, ModalBandwidthSource | ModalBandwidthEstimate | Spectrum | SpectralPeak | GlobalSpectralPeakMetric | GlobalSpectralCharacterization] | None = None,
) -> ModalQFactorEstimationResult:
    """Estimate one operational Q result for every modal-parameter estimate."""
    cfg = settings or ModalQFactorEstimationSettings()
    estimates = _parameter_estimates(parameter_estimates)
    ordered = tuple(sorted(estimates, key=_parameter_sort_key_for_q))
    results = tuple(
        estimate_modal_q_factor_for_parameter_estimate(
            estimate,
            cfg,
            bandwidth_source=_select_bandwidth_source(estimate, bandwidth_sources),
        )
        for estimate in ordered
    )
    valid_estimates = tuple(item for item in results if item.status is ModalQFactorEstimateStatus.VALID)
    reserved = tuple(item for item in results if item.status is ModalQFactorEstimateStatus.VALID_WITH_RESERVATIONS)
    partial = tuple(item for item in results if item.status is ModalQFactorEstimateStatus.PARTIAL)
    inconclusive = tuple(item for item in results if item.status is ModalQFactorEstimateStatus.INCONCLUSIVE)
    insufficient = tuple(item for item in results if item.status is ModalQFactorEstimateStatus.INSUFFICIENT_EVIDENCE)
    invalid = tuple(item for item in results if item.status is ModalQFactorEstimateStatus.INVALID_INPUT)
    failure = None
    if invalid:
        failure = "invalid_q_estimates_present"
    elif inconclusive:
        failure = "inconclusive_q_estimates_present"
    elif insufficient:
        failure = "insufficient_q_evidence_present"
    return ModalQFactorEstimationResult(
        sequence=_sequence_from_parameter_estimates(ordered),
        estimates=results,
        valid_estimates=valid_estimates,
        valid_with_reservations_estimates=reserved,
        partial_estimates=partial,
        inconclusive_estimates=inconclusive,
        insufficient_evidence_estimates=insufficient,
        invalid_estimates=invalid,
        estimate_count=len(results),
        valid_count=len(valid_estimates),
        valid_with_reservations_count=len(reserved),
        partial_count=len(partial),
        inconclusive_count=len(inconclusive),
        insufficient_evidence_count=len(insufficient),
        invalid_count=len(invalid),
        source_parameter_estimate_count=len(ordered),
        settings=cfg,
        valid=failure is None,
        failure_reason=failure,
        diagnostics=(
            "one_q_estimate_per_modal_parameter_estimate",
            "no_audio_or_spectral_analysis_recomputed",
            "q_estimates_are_operational_not_modal_mode_proof",
        ),
    )


def estimate_modal_q_factor_for_parameter_estimate(
    parameter_estimate: ModalParameterEstimate | object,
    settings: ModalQFactorEstimationSettings | None = None,
    *,
    bandwidth_source: ModalBandwidthSource | ModalBandwidthEstimate | Spectrum | SpectralPeak | GlobalSpectralPeakMetric | GlobalSpectralCharacterization | None = None,
) -> ModalQFactorEstimate:
    """Estimate operational Q for one modal-parameter estimate."""
    cfg = settings or ModalQFactorEstimationSettings()
    if not isinstance(parameter_estimate, ModalParameterEstimate):
        provenance = _invalid_provenance(cfg)
        return _build_q_factor_estimate(
            modal_parameter_estimate_id=None,
            hypothesis_id=None,
            status=ModalQFactorEstimateStatus.INVALID_INPUT,
            decay_q=None,
            bandwidth=None,
            bandwidth_q=None,
            isolation=None,
            comparison=None,
            representative_q=None,
            representative_method=None,
            representative_uncertainty=None,
            provenance=provenance,
            supporting=(),
            reservations=(),
            inconclusive=(),
            insufficient=(),
            invalid=(ModalQFactorEstimateReason.INVALID_PARAMETER_ESTIMATE,),
            diagnostics=(
                "invalid_modal_parameter_estimate",
                "no_modal_mode_created",
                "no_physical_q_claim_declared",
            ),
            cfg=cfg,
        )

    provenance = estimate_modal_q_factor_provenance(parameter_estimate, cfg, bandwidth_source)
    supporting: list[ModalQFactorEstimateReason] = []
    reservations: list[ModalQFactorEstimateReason] = []
    inconclusive: list[ModalQFactorEstimateReason] = []
    insufficient: list[ModalQFactorEstimateReason] = []
    invalid: list[ModalQFactorEstimateReason] = []
    diagnostics: list[str] = [
        "q_estimate_is_operational_not_physical_mode",
        "q_decay_not_exact_physical_q",
        "q_bandwidth_not_exact_physical_q",
        "method_agreement_not_physical_validation",
        "method_disagreement_not_nonlinearity_proof",
        "no_modal_mode_created",
        "no_split_or_merge_resolution",
        "no_non_adjacent_association_created",
        "no_gap_closure_performed",
    ]

    source_allowed = _parameter_status_allowed(parameter_estimate.status, cfg)
    if not source_allowed:
        insufficient.append(ModalQFactorEstimateReason.UNSUPPORTED_PARAMETER_STATUS)
        diagnostics.append(f"unsupported_parameter_status:{parameter_estimate.status.value}")
    if parameter_estimate.status is ModalParameterEstimateStatus.VALID_WITH_RESERVATIONS:
        reservations.append(ModalQFactorEstimateReason.UNSUPPORTED_PARAMETER_STATUS)
    if parameter_estimate.provenance.ambiguous_match_ids and cfg.reserve_ambiguous_matches:
        reservations.append(ModalQFactorEstimateReason.AMBIGUOUS_SOURCE_MATCH)
    if parameter_estimate.provenance.near_threshold_match_ids and cfg.reserve_near_threshold_matches:
        reservations.append(ModalQFactorEstimateReason.NEAR_THRESHOLD_SOURCE_MATCH)
    if parameter_estimate.provenance.possible_split_context_ids and cfg.reserve_possible_split_context:
        reservations.append(ModalQFactorEstimateReason.POSSIBLE_SPLIT_CONTEXT)
    if parameter_estimate.provenance.possible_merge_context_ids and cfg.reserve_possible_merge_context:
        reservations.append(ModalQFactorEstimateReason.POSSIBLE_MERGE_CONTEXT)

    decay_q = estimate_q_from_decay(parameter_estimate, cfg) if cfg.enable_decay_method else None
    if decay_q is None:
        if cfg.require_decay_method:
            insufficient.append(ModalQFactorEstimateReason.REQUIRED_DECAY_METHOD_MISSING)
    else:
        _extend_by_role(decay_q.reasons, supporting, reservations, insufficient, inconclusive, invalid)

    bandwidth = None
    bandwidth_q = None
    isolation = None
    if cfg.enable_bandwidth_method:
        if bandwidth_source is None:
            insufficient.append(ModalQFactorEstimateReason.MISSING_BANDWIDTH)
        else:
            bandwidth = _bandwidth_from_source(
                bandwidth_source,
                parameter_estimate.frequency_estimate.representative_frequency_hz,
                cfg,
            )
            _extend_by_role(bandwidth.reasons, supporting, reservations, insufficient, inconclusive, invalid)
            isolation = ModalPeakIsolationEvidence(
                target_peak_frequency_hz=bandwidth.center_frequency_hz,
                nearest_lower_peak_frequency_hz=None,
                nearest_upper_peak_frequency_hz=None,
                nearest_peak_distance_hz=bandwidth.neighboring_peak_distance_hz,
                bandwidth_hz=bandwidth.bandwidth_hz,
                overlap_fraction=bandwidth.neighboring_peak_overlap_fraction,
                isolated=bandwidth.isolated_peak,
                passes=bandwidth.isolated_peak is not False,
                reasons=_ordered_reasons(
                    reason
                    for reason in bandwidth.reasons
                    if reason
                    in {
                        ModalQFactorEstimateReason.SUFFICIENT_BANDWIDTH_EVIDENCE,
                        ModalQFactorEstimateReason.PEAK_NOT_ISOLATED,
                        ModalQFactorEstimateReason.NEIGHBORING_PEAK_INTERFERENCE,
                    }
                ),
                diagnostics=("derived_from_bandwidth_estimate",),
            )
            if bandwidth.valid:
                bandwidth_q = estimate_q_from_bandwidth(
                    bandwidth,
                    frequency_uncertainty_hz=_frequency_uncertainty_hz(parameter_estimate),
                    bandwidth_uncertainty_hz=_source_bandwidth_uncertainty(bandwidth_source),
                    settings=cfg,
                )
                _extend_by_role(bandwidth_q.reasons, supporting, reservations, insufficient, inconclusive, invalid)
    if cfg.require_bandwidth_method and (bandwidth_q is None or not bandwidth_q.valid):
        insufficient.append(ModalQFactorEstimateReason.REQUIRED_BANDWIDTH_METHOD_MISSING)

    comparison = None
    if decay_q is not None and bandwidth_q is not None and decay_q.valid and bandwidth_q.valid:
        comparison = compare_modal_q_methods(decay_q, bandwidth_q, cfg)
        _extend_by_role(comparison.reasons, supporting, reservations, insufficient, inconclusive, invalid)

    representative_q, representative_method, representative_uncertainty = _representative_q(
        decay_q,
        bandwidth_q,
        comparison,
        cfg,
    )
    status = _q_status(
        parameter_estimate,
        source_allowed,
        decay_q,
        bandwidth,
        bandwidth_q,
        comparison,
        supporting,
        reservations,
        inconclusive,
        insufficient,
        invalid,
        cfg,
    )
    return _build_q_factor_estimate(
        modal_parameter_estimate_id=parameter_estimate.estimate_id,
        hypothesis_id=parameter_estimate.hypothesis_id,
        status=status,
        decay_q=decay_q,
        bandwidth=bandwidth,
        bandwidth_q=bandwidth_q,
        isolation=isolation,
        comparison=comparison,
        representative_q=representative_q,
        representative_method=representative_method,
        representative_uncertainty=representative_uncertainty,
        provenance=provenance,
        supporting=supporting,
        reservations=reservations,
        inconclusive=inconclusive,
        insufficient=insufficient,
        invalid=invalid,
        diagnostics=tuple(diagnostics),
        cfg=cfg,
    )


def estimate_q_from_decay(
    parameter_estimate: ModalParameterEstimate | None = None,
    settings: ModalQFactorEstimationSettings | None = None,
    *,
    representative_frequency_hz: float | None = None,
    representative_tau_s: float | None = None,
    frequency_uncertainty_hz: float | None = None,
    tau_uncertainty_s: float | None = None,
) -> ModalDecayQEstimate:
    """Estimate ``Q_decay = pi * f * tau`` under the amplitude-decay convention."""
    cfg = settings or ModalQFactorEstimationSettings()
    if parameter_estimate is not None:
        representative_frequency_hz = parameter_estimate.frequency_estimate.representative_frequency_hz
        representative_tau_s = parameter_estimate.decay_estimate.representative_tau_s
        frequency_uncertainty_hz = _frequency_uncertainty_hz(parameter_estimate)
        tau_uncertainty_s = _tau_uncertainty_s(
            representative_tau_s,
            parameter_estimate.decay_uncertainty,
        )

    reasons: list[ModalQFactorEstimateReason] = []
    diagnostics: list[str] = [
        "decay_convention:" + cfg.decay_convention.value,
        "formula:Q_decay=pi*f*tau",
        "tau_is_amplitude_decay_time_not_energy_decay_time",
        "no_physical_oscillator_fit_performed",
    ]
    assumptions = (
        "approximately_exponential_amplitude_decay",
        "weak_damping_approximation",
        "component_sufficiently_isolated",
        "frequency_approximately_stable_in_analyzed_interval",
    )
    f_valid = _positive_finite(representative_frequency_hz, cfg.minimum_frequency_hz)
    tau_valid = _positive_finite(representative_tau_s, cfg.minimum_tau_s)
    if representative_frequency_hz is None:
        reasons.append(ModalQFactorEstimateReason.MISSING_FREQUENCY)
    elif not f_valid:
        reasons.append(ModalQFactorEstimateReason.INVALID_FREQUENCY)
    if representative_tau_s is None:
        reasons.append(ModalQFactorEstimateReason.MISSING_TAU)
    elif not tau_valid:
        reasons.append(ModalQFactorEstimateReason.INVALID_TAU)

    q_value = None
    passes_limits = False
    if f_valid and tau_valid:
        q_value = pi * float(representative_frequency_hz) * float(representative_tau_s)
        if isfinite(q_value) and q_value > 0.0:
            passes_limits = not _exceeds(q_value, cfg.maximum_decay_q)
            reasons.extend((
                ModalQFactorEstimateReason.SUFFICIENT_DECAY_EVIDENCE,
                ModalQFactorEstimateReason.DECAY_METHOD_AVAILABLE,
            ))
            if not passes_limits:
                reasons.append(ModalQFactorEstimateReason.INVALID_TAU)
                diagnostics.append("maximum_decay_q_exceeded")
        else:
            reasons.append(ModalQFactorEstimateReason.INVALID_TAU)

    if q_value is not None:
        standard, lower, upper, relative, uncertainty_reasons, uncertainty_diagnostics = _decay_q_uncertainty(
            q_value,
            float(representative_frequency_hz),
            float(representative_tau_s),
            frequency_uncertainty_hz,
            tau_uncertainty_s,
            cfg,
        )
        reasons.extend(uncertainty_reasons)
        diagnostics.extend(uncertainty_diagnostics)
    else:
        standard = lower = upper = relative = None
    valid = q_value is not None and passes_limits and not _has_invalid_q_reason(reasons)
    if not cfg.allow_missing_decay_uncertainty and (
        ModalQFactorEstimateReason.MISSING_FREQUENCY_UNCERTAINTY in reasons
        or ModalQFactorEstimateReason.MISSING_TAU_UNCERTAINTY in reasons
    ):
        valid = False
    return ModalDecayQEstimate(
        representative_frequency_hz=_positive_or_none(representative_frequency_hz),
        representative_tau_s=_positive_or_none(representative_tau_s),
        q_decay=q_value,
        decay_convention=_DECAY_CONVENTION_TEXT,
        frequency_uncertainty_hz=_nonnegative_or_none(frequency_uncertainty_hz),
        tau_uncertainty_s=_nonnegative_or_none(tau_uncertainty_s),
        standard_uncertainty_q=standard,
        lower_bound_q=lower,
        upper_bound_q=upper,
        relative_uncertainty=relative,
        assumptions=assumptions,
        valid=valid,
        passes_limits=passes_limits,
        reasons=_ordered_reasons(reasons or (ModalQFactorEstimateReason.INSUFFICIENT_EVIDENCE,)),
        diagnostics=tuple(dict.fromkeys(diagnostics)),
    )


def estimate_modal_bandwidth(
    center_frequency_hz: float | None,
    frequency_axis_hz: Sequence[float] | None = None,
    magnitude_values: Sequence[float] | None = None,
    *,
    peak_frequencies_hz: Sequence[float] = (),
    frequency_resolution_hz: float | None = None,
    precomputed_peak: SpectralPeak | GlobalSpectralPeakMetric | None = None,
    settings: ModalQFactorEstimationSettings | None = None,
) -> ModalBandwidthEstimate:
    """Estimate bandwidth from already-calculated spectral values or peak metrics."""
    cfg = settings or ModalQFactorEstimationSettings()
    if precomputed_peak is not None:
        return _bandwidth_from_peak(precomputed_peak, center_frequency_hz, frequency_resolution_hz, cfg)

    diagnostics: list[str] = [
        "bandwidth_extracted_from_existing_spectral_arrays",
        "no_fft_or_stft_recomputed",
        "linear_interpolation_between_existing_bins",
    ]
    reasons: list[ModalQFactorEstimateReason] = []
    center = _float_or_none(center_frequency_hz)
    if not _positive_finite(center, cfg.minimum_frequency_hz):
        reason = ModalQFactorEstimateReason.MISSING_FREQUENCY if center is None else ModalQFactorEstimateReason.INVALID_FREQUENCY
        return _invalid_bandwidth(center, cfg, reason, diagnostics)
    frequencies = tuple(float(value) for value in (frequency_axis_hz or ()))
    magnitudes = tuple(float(value) for value in (magnitude_values or ()))
    if len(frequencies) != len(magnitudes) or len(frequencies) < 3:
        return _invalid_bandwidth(center, cfg, ModalQFactorEstimateReason.MISSING_BANDWIDTH, diagnostics + ["insufficient_spectral_bins"])
    if any(not isfinite(value) or value < 0.0 for value in frequencies) or any(
        later <= earlier for earlier, later in zip(frequencies, frequencies[1:])
    ):
        return _invalid_bandwidth(center, cfg, ModalQFactorEstimateReason.INVALID_BANDWIDTH, diagnostics + ["invalid_frequency_axis"])
    if any(not isfinite(value) for value in magnitudes):
        return _invalid_bandwidth(center, cfg, ModalQFactorEstimateReason.INVALID_BANDWIDTH, diagnostics + ["invalid_magnitude_value"])

    peak_index = _target_peak_index(frequencies, magnitudes, center)
    if peak_index is None or peak_index == 0 or peak_index == len(frequencies) - 1:
        return _invalid_bandwidth(center, cfg, ModalQFactorEstimateReason.MISSING_BANDWIDTH, diagnostics + ["target_peak_at_or_beyond_edge"])
    peak_value = magnitudes[peak_index]
    if not isfinite(peak_value) or peak_value <= 0.0:
        return _invalid_bandwidth(center, cfg, ModalQFactorEstimateReason.INVALID_BANDWIDTH, diagnostics + ["target_peak_nonpositive"])
    cutoff = _bandwidth_cutoff(peak_value, cfg)
    if cutoff is None:
        return _invalid_bandwidth(center, cfg, ModalQFactorEstimateReason.INVALID_BANDWIDTH, diagnostics + ["unsupported_bandwidth_cutoff"])
    left = _left_crossing(frequencies, magnitudes, peak_index, cutoff)
    right = _right_crossing(frequencies, magnitudes, peak_index, cutoff)
    if left is None or right is None:
        missing = []
        if left is None:
            missing.append("left_crossing_missing")
        if right is None:
            missing.append("right_crossing_missing")
        return ModalBandwidthEstimate(
            center_frequency_hz=center,
            lower_frequency_hz=left,
            upper_frequency_hz=right,
            bandwidth_hz=None,
            bandwidth_definition=cfg.bandwidth_definition,
            bandwidth_level_db=cfg.bandwidth_level_db,
            left_crossing_found=left is not None,
            right_crossing_found=right is not None,
            interpolation_method="linear",
            frequency_resolution_hz=_resolution_from_axis(frequencies, frequency_resolution_hz),
            resolution_ratio=None,
            resolution_assessment=None,
            neighboring_peak_distance_hz=None,
            neighboring_peak_overlap_fraction=None,
            isolated_peak=None,
            resolution_limited=False,
            valid=False,
            reasons=_ordered_reasons((ModalQFactorEstimateReason.MISSING_BANDWIDTH,)),
            diagnostics=tuple(dict.fromkeys(diagnostics + missing)),
        )
    bandwidth = right - left
    if not isfinite(bandwidth) or bandwidth <= cfg.minimum_bandwidth_hz:
        return _invalid_bandwidth(center, cfg, ModalQFactorEstimateReason.INVALID_BANDWIDTH, diagnostics + ["bandwidth_not_positive"])

    resolution = _resolution_from_axis(frequencies, frequency_resolution_hz)
    assessment, resolution_reason, resolution_limited, resolution_passes = _assess_resolution(bandwidth, resolution, cfg)
    if resolution_reason is not None:
        reasons.append(resolution_reason)
    isolation = evaluate_modal_peak_isolation(center, peak_frequencies_hz, bandwidth, cfg)
    reasons.extend(isolation.reasons)
    reasons.append(ModalQFactorEstimateReason.SUFFICIENT_BANDWIDTH_EVIDENCE)
    reasons.append(ModalQFactorEstimateReason.BANDWIDTH_METHOD_AVAILABLE)
    if not resolution_passes:
        reasons.append(ModalQFactorEstimateReason.INSUFFICIENT_SPECTRAL_RESOLUTION)
    if isolation.isolated is False and cfg.require_isolated_peak:
        reasons.append(ModalQFactorEstimateReason.PEAK_NOT_ISOLATED)
    valid = resolution_passes and (isolation.isolated is not False or not cfg.require_isolated_peak)
    return ModalBandwidthEstimate(
        center_frequency_hz=center,
        lower_frequency_hz=left,
        upper_frequency_hz=right,
        bandwidth_hz=bandwidth,
        bandwidth_definition=cfg.bandwidth_definition,
        bandwidth_level_db=cfg.bandwidth_level_db,
        left_crossing_found=True,
        right_crossing_found=True,
        interpolation_method="linear",
        frequency_resolution_hz=resolution,
        resolution_ratio=(bandwidth / resolution if resolution is not None else None),
        resolution_assessment=assessment,
        neighboring_peak_distance_hz=isolation.nearest_peak_distance_hz,
        neighboring_peak_overlap_fraction=isolation.overlap_fraction,
        isolated_peak=isolation.isolated,
        resolution_limited=resolution_limited,
        valid=valid,
        reasons=_ordered_reasons(reasons),
        diagnostics=tuple(dict.fromkeys(diagnostics + [
            "bandwidth_definition:" + cfg.bandwidth_definition.value,
            "no_overlapped_peak_separation",
            "no_lorentzian_fit_performed",
        ])),
    )


def estimate_q_from_bandwidth(
    bandwidth_estimate: ModalBandwidthEstimate | None = None,
    *,
    center_frequency_hz: float | None = None,
    bandwidth_hz: float | None = None,
    frequency_uncertainty_hz: float | None = None,
    bandwidth_uncertainty_hz: float | None = None,
    settings: ModalQFactorEstimationSettings | None = None,
) -> ModalBandwidthQEstimate:
    """Estimate ``Q_bandwidth = f_center / bandwidth`` from a bandwidth estimate."""
    cfg = settings or ModalQFactorEstimationSettings()
    reasons: list[ModalQFactorEstimateReason] = []
    diagnostics: list[str] = [
        "formula:Q_bandwidth=f_center/bandwidth",
        "bandwidth_q_is_operational_not_physical_q",
    ]
    resolution = None
    resolution_limited = False
    isolated_peak = None
    if bandwidth_estimate is not None:
        center_frequency_hz = center_frequency_hz if center_frequency_hz is not None else bandwidth_estimate.center_frequency_hz
        bandwidth_hz = bandwidth_hz if bandwidth_hz is not None else bandwidth_estimate.bandwidth_hz
        resolution = bandwidth_estimate.frequency_resolution_hz
        resolution_limited = bandwidth_estimate.resolution_limited
        isolated_peak = bandwidth_estimate.isolated_peak
        reasons.extend(bandwidth_estimate.reasons)
        diagnostics.extend(bandwidth_estimate.diagnostics)
        if not bandwidth_estimate.valid:
            reasons.append(ModalQFactorEstimateReason.INVALID_BANDWIDTH)
    f_valid = _positive_finite(center_frequency_hz, cfg.minimum_frequency_hz)
    b_valid = _positive_finite(bandwidth_hz, cfg.minimum_bandwidth_hz)
    if center_frequency_hz is None:
        reasons.append(ModalQFactorEstimateReason.MISSING_FREQUENCY)
    elif not f_valid:
        reasons.append(ModalQFactorEstimateReason.INVALID_FREQUENCY)
    if bandwidth_hz is None:
        reasons.append(ModalQFactorEstimateReason.MISSING_BANDWIDTH)
    elif not b_valid:
        reasons.append(ModalQFactorEstimateReason.INVALID_BANDWIDTH)
    q_value = None
    if f_valid and b_valid:
        q_value = float(center_frequency_hz) / float(bandwidth_hz)
        if not isfinite(q_value) or q_value <= 0.0:
            q_value = None
            reasons.append(ModalQFactorEstimateReason.INVALID_BANDWIDTH)
        else:
            reasons.extend((
                ModalQFactorEstimateReason.SUFFICIENT_BANDWIDTH_EVIDENCE,
                ModalQFactorEstimateReason.BANDWIDTH_METHOD_AVAILABLE,
            ))
    resolution_component = _nonnegative_or_none(resolution)
    effective_bandwidth_uncertainty = _effective_bandwidth_uncertainty(
        bandwidth_uncertainty_hz,
        resolution_component,
    )
    if q_value is not None:
        standard, lower, upper, relative, uncertainty_reasons, uncertainty_diagnostics = _bandwidth_q_uncertainty(
            q_value,
            float(center_frequency_hz),
            float(bandwidth_hz),
            frequency_uncertainty_hz,
            effective_bandwidth_uncertainty,
            cfg,
        )
        reasons.extend(uncertainty_reasons)
        diagnostics.extend(uncertainty_diagnostics)
    else:
        standard = lower = upper = relative = None
    valid = q_value is not None and not _has_invalid_q_reason(reasons)
    if not cfg.allow_missing_bandwidth_uncertainty and ModalQFactorEstimateReason.MISSING_BANDWIDTH_UNCERTAINTY in reasons:
        valid = False
    return ModalBandwidthQEstimate(
        center_frequency_hz=_float_or_none(center_frequency_hz),
        bandwidth_hz=_float_or_none(bandwidth_hz),
        q_bandwidth=q_value,
        frequency_resolution_hz=resolution,
        bandwidth_uncertainty_hz=effective_bandwidth_uncertainty,
        frequency_uncertainty_hz=_nonnegative_or_none(frequency_uncertainty_hz),
        resolution_uncertainty_component_hz=resolution_component,
        standard_uncertainty_q=standard,
        lower_bound_q=lower,
        upper_bound_q=upper,
        relative_uncertainty=relative,
        resolution_limited=resolution_limited,
        isolated_peak=isolated_peak,
        valid=valid,
        reasons=_ordered_reasons(reasons or (ModalQFactorEstimateReason.INSUFFICIENT_EVIDENCE,)),
        diagnostics=tuple(dict.fromkeys(diagnostics)),
    )


def evaluate_modal_peak_isolation(
    target_peak_frequency_hz: float | None,
    peak_frequencies_hz: Sequence[float],
    bandwidth_hz: float | None,
    settings: ModalQFactorEstimationSettings | None = None,
) -> ModalPeakIsolationEvidence:
    """Evaluate neighboring-peak interference without attempting separation."""
    cfg = settings or ModalQFactorEstimationSettings()
    target = _float_or_none(target_peak_frequency_hz)
    bandwidth = _float_or_none(bandwidth_hz)
    reasons: list[ModalQFactorEstimateReason] = []
    diagnostics = ["peak_isolation_is_diagnostic_only", "no_overlapped_peak_separation"]
    if not _positive_finite(target, cfg.minimum_frequency_hz):
        reason = ModalQFactorEstimateReason.MISSING_FREQUENCY if target is None else ModalQFactorEstimateReason.INVALID_FREQUENCY
        return ModalPeakIsolationEvidence(
            target,
            None,
            None,
            None,
            bandwidth,
            None,
            None,
            False,
            _ordered_reasons((reason,)),
            tuple(diagnostics),
        )
    if not _positive_finite(bandwidth, cfg.minimum_bandwidth_hz):
        reason = ModalQFactorEstimateReason.MISSING_BANDWIDTH if bandwidth is None else ModalQFactorEstimateReason.INVALID_BANDWIDTH
        return ModalPeakIsolationEvidence(
            target,
            None,
            None,
            None,
            bandwidth,
            None,
            None,
            False,
            _ordered_reasons((reason,)),
            tuple(diagnostics),
        )
    clean_peaks = tuple(
        sorted(
            {
                float(value)
                for value in peak_frequencies_hz
                if isinstance(value, (int, float)) and isfinite(float(value)) and float(value) > 0.0
            }
        )
    )
    tolerance = max(cfg.minimum_positive_value, bandwidth * 1e-9)
    lower_candidates = tuple(value for value in clean_peaks if value < target - tolerance)
    upper_candidates = tuple(value for value in clean_peaks if value > target + tolerance)
    nearest_lower = max(lower_candidates) if lower_candidates else None
    nearest_upper = min(upper_candidates) if upper_candidates else None
    distances = tuple(
        value
        for value in (
            target - nearest_lower if nearest_lower is not None else None,
            nearest_upper - target if nearest_upper is not None else None,
        )
        if value is not None and value > 0.0
    )
    if not distances:
        diagnostics.append("no_neighboring_peaks_detected")
        return ModalPeakIsolationEvidence(
            target,
            nearest_lower,
            nearest_upper,
            None,
            bandwidth,
            0.0,
            True,
            True,
            _ordered_reasons((ModalQFactorEstimateReason.SUFFICIENT_BANDWIDTH_EVIDENCE,)),
            tuple(diagnostics),
        )
    nearest_distance = min(distances)
    overlap = max(0.0, min(1.0, (bandwidth - nearest_distance) / bandwidth))
    isolated = (
        cfg.maximum_neighboring_peak_overlap_fraction is None
        or overlap <= cfg.maximum_neighboring_peak_overlap_fraction
        or isclose(overlap, cfg.maximum_neighboring_peak_overlap_fraction, rel_tol=1e-12, abs_tol=1e-12)
    )
    if isolated:
        reasons.append(ModalQFactorEstimateReason.SUFFICIENT_BANDWIDTH_EVIDENCE)
    else:
        reasons.extend((
            ModalQFactorEstimateReason.PEAK_NOT_ISOLATED,
            ModalQFactorEstimateReason.NEIGHBORING_PEAK_INTERFERENCE,
        ))
    return ModalPeakIsolationEvidence(
        target,
        nearest_lower,
        nearest_upper,
        nearest_distance,
        bandwidth,
        overlap,
        isolated,
        isolated,
        _ordered_reasons(reasons),
        tuple(diagnostics),
    )


def compare_modal_q_methods(
    decay_q_estimate: ModalDecayQEstimate | float | None = None,
    bandwidth_q_estimate: ModalBandwidthQEstimate | float | None = None,
    settings: ModalQFactorEstimationSettings | None = None,
    *,
    q_decay: float | None = None,
    q_bandwidth: float | None = None,
) -> ModalQMethodComparison:
    """Compare positive Q estimates by symmetric relative and log differences."""
    cfg = settings or ModalQFactorEstimationSettings()
    qd = _q_value(decay_q_estimate, q_decay, "decay")
    qb = _q_value(bandwidth_q_estimate, q_bandwidth, "bandwidth")
    reasons: list[ModalQFactorEstimateReason] = []
    diagnostics: list[str] = [
        "relative_difference_is_symmetric",
        "comparison_is_operational_not_physical_validation",
    ]
    if not _positive_finite(qd, cfg.minimum_positive_value) or not _positive_finite(qb, cfg.minimum_positive_value):
        reasons.append(ModalQFactorEstimateReason.INSUFFICIENT_EVIDENCE)
        return ModalQMethodComparison(
            qd,
            qb,
            None,
            None,
            None,
            None,
            False,
            False,
            False,
            None,
            None,
            None,
            _ordered_reasons(reasons),
            tuple(diagnostics),
        )
    absolute = abs(qd - qb)
    relative = absolute / ((qd + qb) / 2.0)
    log_difference = abs(log(qd / qb))
    ratio = qd / qb
    strict = _passes_limit(relative, cfg.maximum_relative_q_disagreement) and _passes_limit(log_difference, cfg.maximum_log_q_difference)
    partial = (
        not strict
        and _passes_scaled_limit(relative, cfg.maximum_relative_q_disagreement, cfg.partial_consistency_multiplier)
        and _passes_scaled_limit(log_difference, cfg.maximum_log_q_difference, cfg.partial_consistency_multiplier)
    )
    inconsistent = not strict and not partial
    if strict:
        reasons.append(ModalQFactorEstimateReason.METHODS_CONSISTENT)
    elif partial:
        reasons.append(ModalQFactorEstimateReason.METHODS_PARTIALLY_CONSISTENT)
    else:
        reasons.extend((
            ModalQFactorEstimateReason.METHODS_INCONSISTENT,
            ModalQFactorEstimateReason.EXCESSIVE_METHOD_DISAGREEMENT,
        ))
    combination = combine_modal_q_estimates(
        decay_q_estimate,
        bandwidth_q_estimate,
        cfg,
        q_decay=qd,
        q_bandwidth=qb,
        methods_consistent=strict,
    )
    reasons.extend(combination.reasons)
    preferred = _preferred_method(decay_q_estimate, bandwidth_q_estimate, cfg, both_available=True)
    return ModalQMethodComparison(
        qd,
        qb,
        absolute,
        relative,
        log_difference,
        ratio,
        strict,
        partial,
        inconsistent,
        "combined" if combination.valid else preferred,
        combination.combined_q,
        combination.combined_uncertainty,
        _ordered_reasons(reasons),
        tuple(dict.fromkeys(diagnostics + list(combination.diagnostics))),
    )


def combine_modal_q_estimates(
    decay_q_estimate: ModalDecayQEstimate | float | None = None,
    bandwidth_q_estimate: ModalBandwidthQEstimate | float | None = None,
    settings: ModalQFactorEstimationSettings | None = None,
    *,
    q_decay: float | None = None,
    q_bandwidth: float | None = None,
    decay_uncertainty_q: float | None = None,
    bandwidth_uncertainty_q: float | None = None,
    methods_consistent: bool = True,
) -> ModalQMethodCombination:
    """Combine method-specific Q values only under an explicit policy."""
    cfg = settings or ModalQFactorEstimationSettings()
    qd = _q_value(decay_q_estimate, q_decay, "decay")
    qb = _q_value(bandwidth_q_estimate, q_bandwidth, "bandwidth")
    ud = _q_uncertainty(decay_q_estimate, decay_uncertainty_q)
    ub = _q_uncertainty(bandwidth_q_estimate, bandwidth_uncertainty_q)
    method = cfg.combine_consistent_methods
    reasons: list[ModalQFactorEstimateReason] = []
    diagnostics: list[str] = ["combination_policy:" + method.value]
    if not methods_consistent and method not in {
        ModalQCombinationMethod.PREFER_DECAY,
        ModalQCombinationMethod.PREFER_BANDWIDTH,
    }:
        diagnostics.append("methods_inconsistent_no_combination")
        return ModalQMethodCombination(method, None, None, (None, None), (None, None), False, _ordered_reasons((ModalQFactorEstimateReason.METHODS_INCONSISTENT,)), tuple(diagnostics))
    if not _positive_finite(qd, cfg.minimum_positive_value) or not _positive_finite(qb, cfg.minimum_positive_value):
        diagnostics.append("combination_requires_two_positive_q_values")
        return ModalQMethodCombination(method, None, None, (None, None), (None, None), False, _ordered_reasons((ModalQFactorEstimateReason.INSUFFICIENT_EVIDENCE,)), tuple(diagnostics))
    if method is ModalQCombinationMethod.NONE:
        diagnostics.append("combination_disabled")
        return ModalQMethodCombination(method, None, None, (None, None), (None, None), False, _ordered_reasons(()), tuple(diagnostics))
    if method is ModalQCombinationMethod.PREFER_DECAY:
        return ModalQMethodCombination(method, qd, ud, (1.0, 0.0), (1.0, 0.0), True, _ordered_reasons((ModalQFactorEstimateReason.DECAY_METHOD_AVAILABLE,)), tuple(diagnostics))
    if method is ModalQCombinationMethod.PREFER_BANDWIDTH:
        return ModalQMethodCombination(method, qb, ub, (0.0, 1.0), (0.0, 1.0), True, _ordered_reasons((ModalQFactorEstimateReason.BANDWIDTH_METHOD_AVAILABLE,)), tuple(diagnostics))
    if method is ModalQCombinationMethod.ARITHMETIC_MEAN:
        combined = (qd + qb) / 2.0
        uncertainty = _combined_uncertainty((ud, ub), (0.5, 0.5))
        return ModalQMethodCombination(method, combined, uncertainty, (1.0, 1.0), (0.5, 0.5), True, _ordered_reasons((ModalQFactorEstimateReason.METHODS_CONSISTENT,)), tuple(diagnostics))
    if method is ModalQCombinationMethod.GEOMETRIC_MEAN:
        combined = sqrt(qd * qb)
        uncertainty = _combined_relative_uncertainty(combined, (qd, qb), (ud, ub))
        return ModalQMethodCombination(method, combined, uncertainty, (1.0, 1.0), (0.5, 0.5), True, _ordered_reasons((ModalQFactorEstimateReason.METHODS_CONSISTENT,)), tuple(diagnostics))
    if method is ModalQCombinationMethod.INVERSE_UNCERTAINTY_WEIGHTED:
        if ud is None or ub is None:
            diagnostics.append("uncertainty_weighted_combination_requires_both_uncertainties")
            return ModalQMethodCombination(method, None, None, (None, None), (None, None), False, _ordered_reasons((ModalQFactorEstimateReason.INSUFFICIENT_EVIDENCE,)), tuple(diagnostics))
        floor = cfg.minimum_positive_value
        if ud == 0.0 or ub == 0.0:
            diagnostics.append("zero_uncertainty_weight_floor_applied")
        weights = (1.0 / max(ud, floor) ** 2, 1.0 / max(ub, floor) ** 2)
        total = sum(weights)
        normalized = tuple(value / total for value in weights)
        combined = (normalized[0] * qd) + (normalized[1] * qb)
        uncertainty = sqrt(1.0 / total)
        return ModalQMethodCombination(method, combined, uncertainty, weights, normalized, True, _ordered_reasons((ModalQFactorEstimateReason.METHODS_CONSISTENT,)), tuple(diagnostics))
    raise ValueError("unsupported Q combination method.")


def estimate_modal_q_factor_provenance(
    parameter_estimate: ModalParameterEstimate,
    settings: ModalQFactorEstimationSettings | None = None,
    bandwidth_source: object | None = None,
) -> ModalQFactorProvenance:
    """Build deterministic provenance back to parameter, candidate, and spectrum sources."""
    cfg = settings or ModalQFactorEstimationSettings()
    spectrum_ids = _source_spectrum_ids(bandwidth_source)
    bandwidth_label = _bandwidth_source_label(bandwidth_source)
    return ModalQFactorProvenance(
        modal_parameter_estimate_id=parameter_estimate.estimate_id,
        hypothesis_id=parameter_estimate.hypothesis_id,
        source_chain_id=parameter_estimate.provenance.source_chain_id,
        candidate_ids=parameter_estimate.provenance.candidate_ids,
        recording_ids=parameter_estimate.provenance.recording_ids,
        spectrum_ids=spectrum_ids,
        frequency_source="modal_parameter_frequency_estimate.representative_frequency_hz",
        tau_source=(
            "modal_parameter_decay_estimate.representative_tau_s"
            if parameter_estimate.decay_estimate.representative_tau_s is not None
            else None
        ),
        bandwidth_source=bandwidth_label,
        settings_fingerprint=modal_q_settings_fingerprint(cfg),
        diagnostics=(
            "settings_fingerprint_is_deterministic_no_timestamp",
            "frequency_tau_and_bandwidth_sources_preserved",
        ),
    )


def summarize_modal_q_factor_estimates(
    result: ModalQFactorEstimationResult,
) -> dict[str, object]:
    """Return a deterministic public summary of operational Q estimates."""
    return {
        "estimate_count": result.estimate_count,
        "source_parameter_estimate_count": result.source_parameter_estimate_count,
        "valid_count": result.valid_count,
        "valid_with_reservations_count": result.valid_with_reservations_count,
        "partial_count": result.partial_count,
        "inconclusive_count": result.inconclusive_count,
        "insufficient_evidence_count": result.insufficient_evidence_count,
        "invalid_count": result.invalid_count,
        "statuses": tuple(item.status.value for item in result.estimates),
        "estimate_ids": tuple(item.estimate_id for item in result.estimates),
        "representative_q_values": tuple(item.representative_q for item in result.estimates),
        "settings_fingerprint": modal_q_settings_fingerprint(result.settings),
        "valid": result.valid,
        "failure_reason": result.failure_reason,
    }


def modal_q_settings_fingerprint(
    settings: ModalQFactorEstimationSettings | None = None,
) -> str:
    """Deterministic settings fingerprint; timestamps are deliberately excluded."""
    cfg = settings or ModalQFactorEstimationSettings()
    encoded = json.dumps(_canonicalize(asdict(cfg)), sort_keys=True, separators=(",", ":"))
    return "modal-q-settings-" + sha1(encoded.encode("utf-8")).hexdigest()[:16]


def estimate_decay_q_uncertainty(
    q_decay: float,
    representative_frequency_hz: float,
    representative_tau_s: float,
    frequency_uncertainty_hz: float | None,
    tau_uncertainty_s: float | None,
    settings: ModalQFactorEstimationSettings | None = None,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Public wrapper for the configured operational decay-Q uncertainty."""
    standard, lower, upper, relative, _, _ = _decay_q_uncertainty(
        q_decay,
        representative_frequency_hz,
        representative_tau_s,
        frequency_uncertainty_hz,
        tau_uncertainty_s,
        settings or ModalQFactorEstimationSettings(),
    )
    return standard, lower, upper, relative


def estimate_bandwidth_q_uncertainty(
    q_bandwidth: float,
    center_frequency_hz: float,
    bandwidth_hz: float,
    frequency_uncertainty_hz: float | None,
    bandwidth_uncertainty_hz: float | None,
    settings: ModalQFactorEstimationSettings | None = None,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Public wrapper for the configured operational bandwidth-Q uncertainty."""
    standard, lower, upper, relative, _, _ = _bandwidth_q_uncertainty(
        q_bandwidth,
        center_frequency_hz,
        bandwidth_hz,
        frequency_uncertainty_hz,
        bandwidth_uncertainty_hz,
        settings or ModalQFactorEstimationSettings(),
    )
    return standard, lower, upper, relative


def _parameter_estimates(
    source: ModalParameterEstimationResult | Iterable[ModalParameterEstimate],
) -> tuple[ModalParameterEstimate, ...]:
    if isinstance(source, ModalParameterEstimationResult):
        estimates = source.estimates
    else:
        estimates = tuple(source)
    ids = tuple(item.estimate_id for item in estimates if isinstance(item, ModalParameterEstimate))
    if len(ids) != len(set(ids)):
        raise ValueError("source parameter estimates must not contain duplicate IDs.")
    return estimates


def _parameter_status_allowed(
    status: ModalParameterEstimateStatus,
    cfg: ModalQFactorEstimationSettings,
) -> bool:
    return (
        (status is ModalParameterEstimateStatus.VALID and cfg.allow_valid_parameter_estimates)
        or (status is ModalParameterEstimateStatus.VALID_WITH_RESERVATIONS and cfg.allow_valid_with_reservations)
        or (status is ModalParameterEstimateStatus.PARTIAL and cfg.allow_partial_parameter_estimates)
        or (status is ModalParameterEstimateStatus.INSUFFICIENT_EVIDENCE and cfg.allow_insufficient_evidence_for_audit)
        or (status is ModalParameterEstimateStatus.INVALID_INPUT and cfg.allow_invalid_input_for_audit)
    )


def _parameter_sort_key_for_q(estimate: ModalParameterEstimate | object) -> tuple:
    if not isinstance(estimate, ModalParameterEstimate):
        return (99, "", "", "")
    labels = estimate.provenance.condition_labels
    first_label = labels[0] if labels else ""
    first_frequency = estimate.frequency_estimate.values_hz[0] if estimate.frequency_estimate.values_hz else 0.0
    return (first_label, first_frequency, estimate.hypothesis_id, estimate.estimate_id)


def _sequence_from_parameter_estimates(estimates: tuple[ModalParameterEstimate, ...]) -> tuple[str, ...]:
    labels = tuple(
        label
        for estimate in estimates
        for label in estimate.provenance.condition_labels
    )
    return tuple(dict.fromkeys(labels)) if labels else ("pp",)


def _select_bandwidth_source(
    estimate: ModalParameterEstimate,
    sources: Mapping[str, object] | None,
) -> object | None:
    if sources is None:
        return None
    keys = (
        estimate.estimate_id,
        estimate.hypothesis_id,
        estimate.provenance.source_chain_id,
    )
    for key in keys:
        if key is not None and key in sources:
            return sources[key]
    return None


def _bandwidth_from_source(
    source: ModalBandwidthSource | ModalBandwidthEstimate | Spectrum | SpectralPeak | GlobalSpectralPeakMetric | GlobalSpectralCharacterization,
    center_frequency_hz: float | None,
    cfg: ModalQFactorEstimationSettings,
) -> ModalBandwidthEstimate:
    if isinstance(source, ModalBandwidthEstimate):
        return source
    if isinstance(source, ModalBandwidthSource):
        center = source.center_frequency_hz if source.center_frequency_hz is not None else center_frequency_hz
        return estimate_modal_bandwidth(
            center,
            source.frequency_axis_hz,
            source.magnitude_values,
            peak_frequencies_hz=source.peak_frequencies_hz,
            frequency_resolution_hz=source.frequency_resolution_hz,
            precomputed_peak=source.precomputed_peak,
            settings=cfg,
        )
    if isinstance(source, Spectrum):
        return estimate_modal_bandwidth(
            center_frequency_hz,
            source.frequencies_hz,
            source.magnitudes,
            frequency_resolution_hz=source.frequency_resolution_hz,
            settings=cfg,
        )
    if isinstance(source, GlobalSpectralCharacterization):
        target_peak = _nearest_global_peak(center_frequency_hz, source.peak_metrics)
        return estimate_modal_bandwidth(
            center_frequency_hz,
            precomputed_peak=target_peak,
            frequency_resolution_hz=source.frequency_resolution_hz,
            settings=cfg,
        )
    if isinstance(source, (SpectralPeak, GlobalSpectralPeakMetric)):
        return estimate_modal_bandwidth(center_frequency_hz, precomputed_peak=source, settings=cfg)
    raise TypeError("bandwidth_source must be a recognized already-calculated spectral source.")


def _bandwidth_from_peak(
    peak: SpectralPeak | GlobalSpectralPeakMetric,
    center_frequency_hz: float | None,
    frequency_resolution_hz: float | None,
    cfg: ModalQFactorEstimationSettings,
) -> ModalBandwidthEstimate:
    if isinstance(peak, GlobalSpectralPeakMetric):
        center = float(center_frequency_hz or peak.representative_frequency_hz)
        bandwidth = float(peak.width_hz)
        lower = float(peak.left_frequency_hz)
        upper = float(peak.right_frequency_hz)
        definition = ModalBandwidthDefinition.HALF_PROMINENCE_POWER
        resolution = frequency_resolution_hz
        reasons: list[ModalQFactorEstimateReason] = [
            ModalQFactorEstimateReason.SUFFICIENT_BANDWIDTH_EVIDENCE,
            ModalQFactorEstimateReason.BANDWIDTH_METHOD_AVAILABLE,
        ]
        diagnostics = [
            "bandwidth_reused_from_global_spectral_peak_metric",
            "global_peak_width_is_half_prominence_in_canonical_power",
            "width_not_formal_uncertainty",
        ]
        if peak.resolution_limited:
            reasons.append(ModalQFactorEstimateReason.BANDWIDTH_AT_RESOLUTION_LIMIT)
            diagnostics.append("source_peak_resolution_limited")
        if peak.isolated:
            isolated = True
        elif peak.overlap_classification == "indeterminate":
            isolated = None
            diagnostics.append("singleton_peak_isolation_indeterminate")
        else:
            isolated = False
            reasons.extend((
                ModalQFactorEstimateReason.PEAK_NOT_ISOLATED,
                ModalQFactorEstimateReason.NEIGHBORING_PEAK_INTERFERENCE,
            ))
        assessment, resolution_reason, resolution_limited, resolution_passes = _assess_resolution(bandwidth, resolution, cfg)
        if resolution_reason is not None:
            reasons.append(resolution_reason)
        valid = (
            _positive_finite(center, cfg.minimum_frequency_hz)
            and _positive_finite(bandwidth, cfg.minimum_bandwidth_hz)
            and resolution_passes
            and (isolated is not False or not cfg.require_isolated_peak)
        )
        return ModalBandwidthEstimate(
            center,
            lower,
            upper,
            bandwidth,
            definition,
            None,
            True,
            True,
            "source_global_peak_half_prominence",
            resolution,
            bandwidth / resolution if resolution is not None else None,
            assessment,
            None,
            None,
            isolated,
            resolution_limited or peak.resolution_limited,
            valid,
            _ordered_reasons(reasons),
            tuple(dict.fromkeys(diagnostics)),
        )

    center = float(center_frequency_hz or peak.refined_frequency_hz or peak.bin_frequency_hz)
    bandwidth = _float_or_none(peak.width_hz)
    reasons = [
        ModalQFactorEstimateReason.BANDWIDTH_METHOD_AVAILABLE,
        ModalQFactorEstimateReason.SUFFICIENT_BANDWIDTH_EVIDENCE,
    ]
    diagnostics = [
        "bandwidth_reused_from_spectral_peak",
        "spectral_peak_width_is_half_prominence_in_source_spectrum_scale",
        "precomputed_width_without_crossing_bounds",
    ]
    if bandwidth is None or not _positive_finite(bandwidth, cfg.minimum_bandwidth_hz):
        reasons.append(ModalQFactorEstimateReason.INVALID_BANDWIDTH)
        valid = False
    else:
        valid = _positive_finite(center, cfg.minimum_frequency_hz)
    assessment, resolution_reason, resolution_limited, resolution_passes = _assess_resolution(bandwidth, frequency_resolution_hz, cfg)
    if resolution_reason is not None:
        reasons.append(resolution_reason)
    valid = valid and resolution_passes
    return ModalBandwidthEstimate(
        center,
        None,
        None,
        bandwidth,
        ModalBandwidthDefinition.HALF_PROMINENCE_AMPLITUDE,
        None,
        False,
        False,
        peak.width_method,
        frequency_resolution_hz,
        bandwidth / frequency_resolution_hz if bandwidth is not None and frequency_resolution_hz is not None else None,
        assessment,
        None,
        None,
        None,
        resolution_limited,
        valid,
        _ordered_reasons(reasons),
        tuple(dict.fromkeys(diagnostics)),
    )


def _nearest_global_peak(
    center_frequency_hz: float | None,
    peaks: tuple[GlobalSpectralPeakMetric, ...],
) -> GlobalSpectralPeakMetric | None:
    if not peaks:
        return None
    center = _float_or_none(center_frequency_hz)
    if center is None:
        return peaks[0]
    return min(peaks, key=lambda peak: abs(peak.representative_frequency_hz - center))


def _invalid_bandwidth(
    center: float | None,
    cfg: ModalQFactorEstimationSettings,
    reason: ModalQFactorEstimateReason,
    diagnostics: Sequence[str],
) -> ModalBandwidthEstimate:
    return ModalBandwidthEstimate(
        center,
        None,
        None,
        None,
        cfg.bandwidth_definition,
        cfg.bandwidth_level_db,
        False,
        False,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        False,
        False,
        _ordered_reasons((reason,)),
        tuple(dict.fromkeys(diagnostics)),
    )


def _target_peak_index(
    frequencies: tuple[float, ...],
    magnitudes: tuple[float, ...],
    target: float,
) -> int | None:
    local_peaks = tuple(
        index
        for index in range(1, len(magnitudes) - 1)
        if magnitudes[index] >= magnitudes[index - 1]
        and magnitudes[index] >= magnitudes[index + 1]
    )
    candidates = local_peaks or tuple(range(len(magnitudes)))
    return min(candidates, key=lambda index: (abs(frequencies[index] - target), -magnitudes[index]))


def _bandwidth_cutoff(
    peak_value: float,
    cfg: ModalQFactorEstimationSettings,
) -> float | None:
    if cfg.bandwidth_definition is ModalBandwidthDefinition.AMPLITUDE_MINUS_3_DB:
        return peak_value * (10.0 ** (cfg.bandwidth_level_db / 20.0))
    if cfg.bandwidth_definition is ModalBandwidthDefinition.POWER_MINUS_3_DB:
        return peak_value * (10.0 ** (cfg.bandwidth_level_db / 10.0))
    return None


def _left_crossing(
    frequencies: tuple[float, ...],
    magnitudes: tuple[float, ...],
    peak_index: int,
    cutoff: float,
) -> float | None:
    for index in range(peak_index - 1, -1, -1):
        left_value = magnitudes[index]
        right_value = magnitudes[index + 1]
        if isclose(left_value, cutoff, rel_tol=1e-12, abs_tol=1e-12):
            return frequencies[index]
        if left_value <= cutoff <= right_value:
            return _interpolate_crossing(
                frequencies[index],
                left_value,
                frequencies[index + 1],
                right_value,
                cutoff,
            )
    return None


def _right_crossing(
    frequencies: tuple[float, ...],
    magnitudes: tuple[float, ...],
    peak_index: int,
    cutoff: float,
) -> float | None:
    for index in range(peak_index, len(magnitudes) - 1):
        left_value = magnitudes[index]
        right_value = magnitudes[index + 1]
        if isclose(right_value, cutoff, rel_tol=1e-12, abs_tol=1e-12):
            return frequencies[index + 1]
        if left_value >= cutoff >= right_value:
            return _interpolate_crossing(
                frequencies[index],
                left_value,
                frequencies[index + 1],
                right_value,
                cutoff,
            )
    return None


def _interpolate_crossing(
    f0: float,
    y0: float,
    f1: float,
    y1: float,
    target: float,
) -> float:
    if isclose(y1, y0, rel_tol=0.0, abs_tol=1e-18):
        return f0
    fraction = (target - y0) / (y1 - y0)
    return f0 + (fraction * (f1 - f0))


def _resolution_from_axis(
    frequencies: tuple[float, ...],
    explicit: float | None,
) -> float | None:
    if explicit is not None and isfinite(explicit) and explicit > 0.0:
        return float(explicit)
    differences = tuple(
        later - earlier
        for earlier, later in zip(frequencies, frequencies[1:])
        if later > earlier
    )
    if not differences:
        return None
    ordered = tuple(sorted(differences))
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _assess_resolution(
    bandwidth_hz: float | None,
    frequency_resolution_hz: float | None,
    cfg: ModalQFactorEstimationSettings,
) -> tuple[SpectralResolutionAssessment | None, ModalQFactorEstimateReason | None, bool, bool]:
    if bandwidth_hz is None or frequency_resolution_hz is None:
        return None, None, False, True
    ratio = bandwidth_hz / frequency_resolution_hz
    minimum = cfg.minimum_spectral_resolution_ratio
    if minimum is None:
        return SpectralResolutionAssessment.WELL_RESOLVED, ModalQFactorEstimateReason.WELL_RESOLVED, False, True
    if ratio < cfg.unresolved_spectral_resolution_ratio and not isclose(ratio, cfg.unresolved_spectral_resolution_ratio, rel_tol=1e-12, abs_tol=1e-12):
        return SpectralResolutionAssessment.UNRESOLVED, ModalQFactorEstimateReason.UNRESOLVED, True, False
    if ratio <= cfg.unresolved_spectral_resolution_ratio or isclose(ratio, cfg.unresolved_spectral_resolution_ratio, rel_tol=1e-12, abs_tol=1e-12):
        passes = cfg.allow_resolution_limited_bandwidth
        return SpectralResolutionAssessment.RESOLUTION_LIMITED, ModalQFactorEstimateReason.BANDWIDTH_AT_RESOLUTION_LIMIT, True, passes
    if ratio < minimum and not isclose(ratio, minimum, rel_tol=1e-12, abs_tol=1e-12):
        passes = cfg.allow_resolution_limited_bandwidth
        return SpectralResolutionAssessment.MARGINALLY_RESOLVED, ModalQFactorEstimateReason.MARGINALLY_RESOLVED, True, passes
    return SpectralResolutionAssessment.WELL_RESOLVED, ModalQFactorEstimateReason.WELL_RESOLVED, False, True


def _decay_q_uncertainty(
    q_value: float,
    frequency_hz: float,
    tau_s: float,
    frequency_uncertainty_hz: float | None,
    tau_uncertainty_s: float | None,
    cfg: ModalQFactorEstimationSettings,
) -> tuple[
    float | None,
    float | None,
    float | None,
    float | None,
    tuple[ModalQFactorEstimateReason, ...],
    tuple[str, ...],
]:
    diagnostics = ["decay_q_uncertainty_method:" + cfg.uncertainty_method.value]
    reasons: list[ModalQFactorEstimateReason] = []
    uf = _nonnegative_or_none(frequency_uncertainty_hz)
    ut = _nonnegative_or_none(tau_uncertainty_s)
    if uf is None:
        reasons.append(ModalQFactorEstimateReason.MISSING_FREQUENCY_UNCERTAINTY)
    if ut is None:
        reasons.append(ModalQFactorEstimateReason.MISSING_TAU_UNCERTAINTY)
    if cfg.uncertainty_method is ModalQUncertaintyMethod.DISABLED:
        diagnostics.append("q_uncertainty_method_disabled")
        return None, None, None, None, _ordered_reasons(reasons), tuple(diagnostics)
    if cfg.uncertainty_method is ModalQUncertaintyMethod.LINEAR_PROPAGATION:
        if uf is None or ut is None:
            diagnostics.append("linear_uncertainty_requires_frequency_and_tau_uncertainty")
            return None, None, None, None, _ordered_reasons(reasons), tuple(diagnostics)
        relative = sqrt((uf / frequency_hz) ** 2 + (ut / tau_s) ** 2)
        standard = q_value * relative
        return standard, max(0.0, q_value - standard), q_value + standard, relative, _ordered_reasons(reasons), tuple(diagnostics)
    if uf is None or ut is None:
        diagnostics.append("parametric_bootstrap_requires_frequency_and_tau_uncertainty")
        return None, None, None, None, _ordered_reasons(reasons), tuple(diagnostics)
    samples = _bootstrap_decay_q(q_value, frequency_hz, tau_s, uf, ut, cfg)
    if len(samples) < 2:
        reasons.append(ModalQFactorEstimateReason.INSUFFICIENT_EVIDENCE)
        return None, None, None, None, _ordered_reasons(reasons), tuple(diagnostics)
    standard = _sample_standard_deviation(samples)
    lower = _quantile(samples, (1.0 - cfg.uncertainty_confidence_level) / 2.0)
    upper = _quantile(samples, 1.0 - (1.0 - cfg.uncertainty_confidence_level) / 2.0)
    return standard, lower, upper, standard / q_value if q_value > 0 else None, _ordered_reasons(reasons), tuple(diagnostics)


def _bandwidth_q_uncertainty(
    q_value: float,
    frequency_hz: float,
    bandwidth_hz: float,
    frequency_uncertainty_hz: float | None,
    bandwidth_uncertainty_hz: float | None,
    cfg: ModalQFactorEstimationSettings,
) -> tuple[
    float | None,
    float | None,
    float | None,
    float | None,
    tuple[ModalQFactorEstimateReason, ...],
    tuple[str, ...],
]:
    diagnostics = [
        "bandwidth_q_uncertainty_method:" + cfg.uncertainty_method.value,
        "bandwidth_uncertainty_includes_resolution_component_when_available",
    ]
    reasons: list[ModalQFactorEstimateReason] = []
    uf = _nonnegative_or_none(frequency_uncertainty_hz)
    ub = _nonnegative_or_none(bandwidth_uncertainty_hz)
    if uf is None:
        reasons.append(ModalQFactorEstimateReason.MISSING_FREQUENCY_UNCERTAINTY)
    if ub is None:
        reasons.append(ModalQFactorEstimateReason.MISSING_BANDWIDTH_UNCERTAINTY)
    if cfg.uncertainty_method is ModalQUncertaintyMethod.DISABLED:
        diagnostics.append("q_uncertainty_method_disabled")
        return None, None, None, None, _ordered_reasons(reasons), tuple(diagnostics)
    if uf is None or ub is None:
        diagnostics.append("linear_uncertainty_requires_frequency_and_bandwidth_uncertainty")
        return None, None, None, None, _ordered_reasons(reasons), tuple(diagnostics)
    relative = sqrt((uf / frequency_hz) ** 2 + (ub / bandwidth_hz) ** 2)
    standard = q_value * relative
    return standard, max(0.0, q_value - standard), q_value + standard, relative, _ordered_reasons(reasons), tuple(diagnostics)


def _bootstrap_decay_q(
    q_value: float,
    frequency_hz: float,
    tau_s: float,
    frequency_uncertainty_hz: float,
    tau_uncertainty_s: float,
    cfg: ModalQFactorEstimationSettings,
) -> tuple[float, ...]:
    del q_value
    rng = Random(cfg.bootstrap_random_seed)
    sigma_log_tau = tau_uncertainty_s / tau_s if tau_s > 0 else 0.0
    samples: list[float] = []
    attempts = 0
    max_attempts = cfg.bootstrap_sample_count * 10
    while len(samples) < cfg.bootstrap_sample_count and attempts < max_attempts:
        attempts += 1
        frequency = rng.gauss(frequency_hz, frequency_uncertainty_hz)
        tau = exp(rng.gauss(log(tau_s), sigma_log_tau))
        if frequency > cfg.minimum_positive_value and tau > cfg.minimum_positive_value:
            value = pi * frequency * tau
            if isfinite(value) and value > 0.0:
                samples.append(value)
    return tuple(samples)


def _source_bandwidth_uncertainty(source: object | None) -> float | None:
    if isinstance(source, ModalBandwidthSource):
        return source.bandwidth_uncertainty_hz
    return None


def _effective_bandwidth_uncertainty(
    bandwidth_uncertainty_hz: float | None,
    resolution_component_hz: float | None,
) -> float | None:
    values = tuple(
        value
        for value in (
            _nonnegative_or_none(bandwidth_uncertainty_hz),
            _nonnegative_or_none(resolution_component_hz),
        )
        if value is not None
    )
    return max(values) if values else None


def _tau_uncertainty_s(
    representative_tau_s: float | None,
    uncertainty: ModalDecayUncertainty,
) -> float | None:
    tau = _float_or_none(representative_tau_s)
    if tau is None or tau <= 0.0 or not uncertainty.valid:
        return None
    candidates: list[float] = []
    if uncertainty.lower_bound_tau_s is not None:
        candidates.append(abs(tau - uncertainty.lower_bound_tau_s))
    if uncertainty.upper_bound_tau_s is not None:
        candidates.append(abs(uncertainty.upper_bound_tau_s - tau))
    if uncertainty.multiplicative_uncertainty_factor is not None:
        candidates.append(tau * (uncertainty.multiplicative_uncertainty_factor - 1.0))
    return max(candidates) if candidates else None


def _frequency_uncertainty_hz(parameter_estimate: ModalParameterEstimate) -> float | None:
    uncertainty = parameter_estimate.frequency_uncertainty
    if not uncertainty.valid:
        return None
    return _nonnegative_or_none(uncertainty.standard_uncertainty_hz)


def _representative_q(
    decay_q: ModalDecayQEstimate | None,
    bandwidth_q: ModalBandwidthQEstimate | None,
    comparison: ModalQMethodComparison | None,
    cfg: ModalQFactorEstimationSettings,
) -> tuple[float | None, str | None, float | None]:
    if comparison is not None and comparison.combined_q is not None and comparison.consistent:
        return comparison.combined_q, comparison.preferred_method, comparison.combined_uncertainty
    if cfg.prefer_decay_method and decay_q is not None and decay_q.valid:
        return decay_q.q_decay, "decay", decay_q.standard_uncertainty_q
    if cfg.prefer_bandwidth_method and bandwidth_q is not None and bandwidth_q.valid:
        return bandwidth_q.q_bandwidth, "bandwidth", bandwidth_q.standard_uncertainty_q
    valid_decay = decay_q is not None and decay_q.valid
    valid_bandwidth = bandwidth_q is not None and bandwidth_q.valid
    if valid_decay and not valid_bandwidth:
        return decay_q.q_decay, "decay", decay_q.standard_uncertainty_q
    if valid_bandwidth and not valid_decay:
        return bandwidth_q.q_bandwidth, "bandwidth", bandwidth_q.standard_uncertainty_q
    if valid_decay and valid_bandwidth and comparison is not None and comparison.partially_consistent:
        if cfg.combine_consistent_methods is ModalQCombinationMethod.PREFER_BANDWIDTH:
            return bandwidth_q.q_bandwidth, "bandwidth", bandwidth_q.standard_uncertainty_q
        return decay_q.q_decay, "decay", decay_q.standard_uncertainty_q
    return None, None, None


def _q_status(
    parameter_estimate: ModalParameterEstimate,
    source_allowed: bool,
    decay_q: ModalDecayQEstimate | None,
    bandwidth: ModalBandwidthEstimate | None,
    bandwidth_q: ModalBandwidthQEstimate | None,
    comparison: ModalQMethodComparison | None,
    supporting: Sequence[ModalQFactorEstimateReason],
    reservations: Sequence[ModalQFactorEstimateReason],
    inconclusive: Sequence[ModalQFactorEstimateReason],
    insufficient: Sequence[ModalQFactorEstimateReason],
    invalid: Sequence[ModalQFactorEstimateReason],
    cfg: ModalQFactorEstimationSettings,
) -> ModalQFactorEstimateStatus:
    del supporting
    decay_valid = decay_q is not None and decay_q.valid
    bandwidth_valid = bandwidth_q is not None and bandwidth_q.valid
    method_valid_count = int(decay_valid) + int(bandwidth_valid)
    method_invalid = (
        (decay_q is not None and not decay_q.valid and _has_invalid_q_reason(decay_q.reasons))
        or (bandwidth is not None and not bandwidth.valid and _has_invalid_q_reason(bandwidth.reasons))
        or (bandwidth_q is not None and not bandwidth_q.valid and _has_invalid_q_reason(bandwidth_q.reasons))
    )
    if invalid or parameter_estimate.status is ModalParameterEstimateStatus.INVALID_INPUT:
        return ModalQFactorEstimateStatus.INVALID_INPUT
    if not source_allowed:
        return ModalQFactorEstimateStatus.INSUFFICIENT_EVIDENCE
    if method_valid_count == 0 and not method_invalid:
        return ModalQFactorEstimateStatus.INSUFFICIENT_EVIDENCE
    if cfg.require_decay_method and not decay_valid:
        return ModalQFactorEstimateStatus.INSUFFICIENT_EVIDENCE
    if cfg.require_bandwidth_method and not bandwidth_valid:
        return ModalQFactorEstimateStatus.INSUFFICIENT_EVIDENCE
    if method_invalid and method_valid_count == 0:
        return ModalQFactorEstimateStatus.INVALID_INPUT
    if comparison is not None and comparison.inconsistent:
        if cfg.consistency_policy is ModalQConsistencyPolicy.RESERVATION_ON_DISAGREEMENT:
            return ModalQFactorEstimateStatus.VALID_WITH_RESERVATIONS
        return ModalQFactorEstimateStatus.INCONCLUSIVE
    if inconclusive:
        return ModalQFactorEstimateStatus.INCONCLUSIVE
    if method_valid_count == 1:
        return ModalQFactorEstimateStatus.PARTIAL
    if reservations or (comparison is not None and comparison.partially_consistent):
        return ModalQFactorEstimateStatus.VALID_WITH_RESERVATIONS
    if insufficient and method_valid_count == 0:
        return ModalQFactorEstimateStatus.INSUFFICIENT_EVIDENCE
    return ModalQFactorEstimateStatus.VALID


def _build_q_factor_estimate(
    *,
    modal_parameter_estimate_id: str | None,
    hypothesis_id: str | None,
    status: ModalQFactorEstimateStatus,
    decay_q: ModalDecayQEstimate | None,
    bandwidth: ModalBandwidthEstimate | None,
    bandwidth_q: ModalBandwidthQEstimate | None,
    isolation: ModalPeakIsolationEvidence | None,
    comparison: ModalQMethodComparison | None,
    representative_q: float | None,
    representative_method: str | None,
    representative_uncertainty: float | None,
    provenance: ModalQFactorProvenance,
    supporting: Iterable[ModalQFactorEstimateReason],
    reservations: Iterable[ModalQFactorEstimateReason],
    inconclusive: Iterable[ModalQFactorEstimateReason],
    insufficient: Iterable[ModalQFactorEstimateReason],
    invalid: Iterable[ModalQFactorEstimateReason],
    diagnostics: Sequence[str],
    cfg: ModalQFactorEstimationSettings,
) -> ModalQFactorEstimate:
    supporting_reasons = _ordered_reasons(supporting)
    reservation_reasons = _ordered_reasons(reservations)
    inconclusive_reasons = _ordered_reasons(inconclusive)
    insufficient_reasons = _ordered_reasons(insufficient)
    invalid_reasons = _ordered_reasons(invalid)
    estimate_id = _q_estimate_id(
        modal_parameter_estimate_id,
        hypothesis_id,
        status,
        decay_q,
        bandwidth,
        bandwidth_q,
        comparison,
        representative_q,
        provenance,
        cfg,
    )
    return ModalQFactorEstimate(
        estimate_id,
        modal_parameter_estimate_id,
        hypothesis_id,
        status,
        decay_q,
        bandwidth,
        bandwidth_q,
        isolation,
        comparison,
        representative_q,
        representative_method,
        representative_uncertainty,
        supporting_reasons,
        reservation_reasons,
        inconclusive_reasons,
        insufficient_reasons,
        invalid_reasons,
        status in {
            ModalQFactorEstimateStatus.VALID,
            ModalQFactorEstimateStatus.VALID_WITH_RESERVATIONS,
        },
        status is not ModalQFactorEstimateStatus.VALID,
        provenance,
        tuple(dict.fromkeys(diagnostics)),
    )


def _q_estimate_id(
    modal_parameter_estimate_id: str | None,
    hypothesis_id: str | None,
    status: ModalQFactorEstimateStatus,
    decay_q: ModalDecayQEstimate | None,
    bandwidth: ModalBandwidthEstimate | None,
    bandwidth_q: ModalBandwidthQEstimate | None,
    comparison: ModalQMethodComparison | None,
    representative_q: float | None,
    provenance: ModalQFactorProvenance,
    cfg: ModalQFactorEstimationSettings,
) -> str:
    payload = {
        "modal_parameter_estimate_id": modal_parameter_estimate_id,
        "hypothesis_id": hypothesis_id,
        "status": status.value,
        "q_decay": decay_q.q_decay if decay_q is not None else None,
        "q_bandwidth": bandwidth_q.q_bandwidth if bandwidth_q is not None else None,
        "bandwidth_hz": bandwidth.bandwidth_hz if bandwidth is not None else None,
        "bandwidth_definition": bandwidth.bandwidth_definition.value if bandwidth is not None else None,
        "method_comparison": (
            {
                "relative": comparison.relative_symmetric_difference,
                "log": comparison.log_q_difference,
                "combined_q": comparison.combined_q,
            }
            if comparison is not None
            else None
        ),
        "representative_q": representative_q,
        "candidate_ids": provenance.candidate_ids,
        "recording_ids": provenance.recording_ids,
        "spectrum_ids": provenance.spectrum_ids,
        "settings_fingerprint": modal_q_settings_fingerprint(cfg),
    }
    encoded = json.dumps(_canonicalize(payload), sort_keys=True, separators=(",", ":"))
    return "modal-q-factor-estimate-" + sha1(encoded.encode("utf-8")).hexdigest()[:16]


def _invalid_provenance(cfg: ModalQFactorEstimationSettings) -> ModalQFactorProvenance:
    return ModalQFactorProvenance(
        None,
        None,
        None,
        (),
        (),
        (),
        None,
        None,
        None,
        modal_q_settings_fingerprint(cfg),
        ("invalid_input_no_source_provenance",),
    )


def _source_spectrum_ids(source: object | None) -> tuple[str, ...]:
    if isinstance(source, ModalBandwidthSource) and source.spectrum_id is not None:
        return (source.spectrum_id,)
    if isinstance(source, GlobalSpectralCharacterization):
        return (source.recording_id,)
    return ()


def _bandwidth_source_label(source: object | None) -> str | None:
    if source is None:
        return None
    if isinstance(source, ModalBandwidthEstimate):
        return "modal_bandwidth_estimate"
    if isinstance(source, ModalBandwidthSource):
        return source.spectrum_id or "modal_bandwidth_source"
    if isinstance(source, Spectrum):
        return "spectrum_frequency_axis_and_magnitudes"
    if isinstance(source, SpectralPeak):
        return "spectral_peak_width_hz"
    if isinstance(source, GlobalSpectralPeakMetric):
        return "global_spectral_peak_metric_width_hz"
    if isinstance(source, GlobalSpectralCharacterization):
        return source.recording_id
    return "unknown_bandwidth_source"


def _q_value(
    estimate: ModalDecayQEstimate | ModalBandwidthQEstimate | float | None,
    fallback: float | None,
    method: str,
) -> float | None:
    if fallback is not None:
        return _float_or_none(fallback)
    if isinstance(estimate, ModalDecayQEstimate):
        return estimate.q_decay
    if isinstance(estimate, ModalBandwidthQEstimate):
        return estimate.q_bandwidth
    if isinstance(estimate, (int, float)):
        return _float_or_none(float(estimate))
    if estimate is None:
        return None
    raise TypeError(f"{method} Q estimate is not recognized.")


def _q_uncertainty(
    estimate: ModalDecayQEstimate | ModalBandwidthQEstimate | float | None,
    fallback: float | None,
) -> float | None:
    if fallback is not None:
        return _nonnegative_or_none(fallback)
    if isinstance(estimate, (ModalDecayQEstimate, ModalBandwidthQEstimate)):
        return estimate.standard_uncertainty_q
    return None


def _preferred_method(
    decay_q: ModalDecayQEstimate | float | None,
    bandwidth_q: ModalBandwidthQEstimate | float | None,
    cfg: ModalQFactorEstimationSettings,
    *,
    both_available: bool,
) -> str | None:
    del decay_q, bandwidth_q
    if cfg.prefer_decay_method:
        return "decay"
    if cfg.prefer_bandwidth_method:
        return "bandwidth"
    if both_available:
        return "decay"
    return None


def _extend_by_role(
    reasons: Iterable[ModalQFactorEstimateReason],
    supporting: list[ModalQFactorEstimateReason],
    reservations: list[ModalQFactorEstimateReason],
    insufficient: list[ModalQFactorEstimateReason],
    inconclusive: list[ModalQFactorEstimateReason],
    invalid: list[ModalQFactorEstimateReason],
) -> None:
    for reason in reasons:
        if reason in _SUPPORTING_REASONS:
            supporting.append(reason)
        elif reason in _RESERVATION_REASONS:
            reservations.append(reason)
        elif reason in _INSUFFICIENT_REASONS:
            insufficient.append(reason)
        elif reason in _INCONCLUSIVE_REASONS:
            inconclusive.append(reason)
        elif reason in _INVALID_REASONS:
            invalid.append(reason)


_SUPPORTING_REASONS = {
    ModalQFactorEstimateReason.SUFFICIENT_DECAY_EVIDENCE,
    ModalQFactorEstimateReason.SUFFICIENT_BANDWIDTH_EVIDENCE,
    ModalQFactorEstimateReason.DECAY_METHOD_AVAILABLE,
    ModalQFactorEstimateReason.BANDWIDTH_METHOD_AVAILABLE,
    ModalQFactorEstimateReason.METHODS_CONSISTENT,
    ModalQFactorEstimateReason.WELL_RESOLVED,
}
_RESERVATION_REASONS = {
    ModalQFactorEstimateReason.METHODS_PARTIALLY_CONSISTENT,
    ModalQFactorEstimateReason.MISSING_FREQUENCY_UNCERTAINTY,
    ModalQFactorEstimateReason.MISSING_TAU_UNCERTAINTY,
    ModalQFactorEstimateReason.MISSING_BANDWIDTH_UNCERTAINTY,
    ModalQFactorEstimateReason.AMBIGUOUS_SOURCE_MATCH,
    ModalQFactorEstimateReason.NEAR_THRESHOLD_SOURCE_MATCH,
    ModalQFactorEstimateReason.POSSIBLE_SPLIT_CONTEXT,
    ModalQFactorEstimateReason.POSSIBLE_MERGE_CONTEXT,
    ModalQFactorEstimateReason.BANDWIDTH_AT_RESOLUTION_LIMIT,
    ModalQFactorEstimateReason.MARGINALLY_RESOLVED,
    ModalQFactorEstimateReason.RESOLUTION_LIMITED,
    ModalQFactorEstimateReason.PEAK_NOT_ISOLATED,
    ModalQFactorEstimateReason.NEIGHBORING_PEAK_INTERFERENCE,
    ModalQFactorEstimateReason.UNSUPPORTED_PARAMETER_STATUS,
}
_INSUFFICIENT_REASONS = {
    ModalQFactorEstimateReason.MISSING_FREQUENCY,
    ModalQFactorEstimateReason.MISSING_TAU,
    ModalQFactorEstimateReason.MISSING_BANDWIDTH,
    ModalQFactorEstimateReason.INSUFFICIENT_SPECTRAL_RESOLUTION,
    ModalQFactorEstimateReason.REQUIRED_DECAY_METHOD_MISSING,
    ModalQFactorEstimateReason.REQUIRED_BANDWIDTH_METHOD_MISSING,
    ModalQFactorEstimateReason.INSUFFICIENT_EVIDENCE,
    ModalQFactorEstimateReason.UNRESOLVED,
}
_INCONCLUSIVE_REASONS = {
    ModalQFactorEstimateReason.METHODS_INCONSISTENT,
    ModalQFactorEstimateReason.EXCESSIVE_METHOD_DISAGREEMENT,
}
_INVALID_REASONS = {
    ModalQFactorEstimateReason.INVALID_FREQUENCY,
    ModalQFactorEstimateReason.INVALID_TAU,
    ModalQFactorEstimateReason.INVALID_BANDWIDTH,
    ModalQFactorEstimateReason.INVALID_PARAMETER_ESTIMATE,
}


def _has_invalid_q_reason(reasons: Iterable[ModalQFactorEstimateReason]) -> bool:
    return any(reason in _INVALID_REASONS for reason in reasons)


def _passes_limit(value: float, limit: float | None) -> bool:
    return limit is None or value <= limit or isclose(value, limit, rel_tol=1e-12, abs_tol=1e-12)


def _passes_scaled_limit(value: float, limit: float | None, multiplier: float) -> bool:
    return limit is None or value <= limit * multiplier or isclose(value, limit * multiplier, rel_tol=1e-12, abs_tol=1e-12)


def _sample_standard_deviation(values: tuple[float, ...]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _quantile(values: tuple[float, ...], probability: float) -> float:
    ordered = tuple(sorted(values))
    if not ordered:
        raise ValueError("quantile requires values.")
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return (ordered[lower] * (1.0 - fraction)) + (ordered[upper] * fraction)


def _combined_uncertainty(
    uncertainties: tuple[float | None, float | None],
    weights: tuple[float, float],
) -> float | None:
    if uncertainties[0] is None or uncertainties[1] is None:
        return None
    return sqrt((weights[0] * uncertainties[0]) ** 2 + (weights[1] * uncertainties[1]) ** 2)


def _combined_relative_uncertainty(
    combined_q: float,
    q_values: tuple[float, float],
    uncertainties: tuple[float | None, float | None],
) -> float | None:
    if uncertainties[0] is None or uncertainties[1] is None:
        return None
    relatives = tuple(
        uncertainty / q
        for uncertainty, q in zip(uncertainties, q_values, strict=True)
    )
    return combined_q * sqrt(sum(value**2 for value in relatives)) / 2.0


def _float_or_none(value: object) -> float | None:
    if isinstance(value, (int, float)) and isfinite(float(value)):
        return float(value)
    return None


def _nonnegative_or_none(value: object) -> float | None:
    value = _float_or_none(value)
    if value is None or value < 0.0:
        return None
    return value


def _positive_or_none(value: object) -> float | None:
    value = _float_or_none(value)
    if value is None or value <= 0.0:
        return None
    return value


def _positive_finite(value: object, minimum: float) -> bool:
    return isinstance(value, (int, float)) and isfinite(float(value)) and float(value) > minimum


def _exceeds(value: float | None, limit: float | None) -> bool:
    return value is not None and limit is not None and value > limit and not isclose(value, limit, rel_tol=1e-12, abs_tol=1e-12)


def _validate_optional_positive(value: float | None, name: str) -> None:
    if value is not None and (not isfinite(value) or value <= 0.0):
        raise ValueError(f"{name} must be finite and positive when provided.")


def _validate_optional_nonnegative(value: float | None, name: str) -> None:
    if value is not None and (not isfinite(value) or value < 0.0):
        raise ValueError(f"{name} must be finite and non-negative when provided.")


def _validate_bounds(
    lower: float | None,
    upper: float | None,
    name: str,
    *,
    positive: bool = False,
) -> None:
    if (lower is None) != (upper is None):
        raise ValueError(f"{name} must provide both bounds or neither.")
    if lower is None or upper is None:
        return
    _finite_optional(lower, name + " lower", positive=positive)
    _finite_optional(upper, name + " upper", positive=positive)
    if upper < lower:
        raise ValueError(f"{name} are inverted.")


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
            raise ValueError("weights must be finite and non-negative.")
    if require_normalized_sum:
        if any(value is None for value in normalized):
            raise ValueError("valid weights require complete normalized values.")
        if not isclose(sum(value for value in normalized if value is not None), 1.0, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("normalized weights must sum to one.")


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
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates.")


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


def _ordered_reasons(
    reasons: Iterable[ModalQFactorEstimateReason],
) -> tuple[ModalQFactorEstimateReason, ...]:
    unique = {
        reason if isinstance(reason, ModalQFactorEstimateReason) else ModalQFactorEstimateReason(reason)
        for reason in reasons
    }
    return tuple(sorted(unique, key=lambda item: item.value))


def _reason_tuple(values: tuple[ModalQFactorEstimateReason, ...], name: str) -> None:
    if not isinstance(values, tuple):
        raise ValueError(f"{name} must be an immutable tuple.")
    if values != _ordered_reasons(values):
        raise ValueError(f"{name} must contain unique reasons in deterministic order.")


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
