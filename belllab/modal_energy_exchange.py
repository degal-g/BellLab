"""Operational evidence for possible modal energy redistribution.

This module compares already computed amplitude envelopes.  It does not read
audio, recompute spectra, rebuild tracks, infer causality, fit coupled
oscillators, resolve split/merge contexts, or promote any estimate to a
physical modal mode.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from math import isfinite, sqrt
import hashlib
import json
import random
from typing import Any

from belllab.modal_hypotheses import ModalHypothesis, ModalHypothesisStatus
from belllab.modal_parameters import (
    ModalParameterEstimate,
    ModalParameterEstimateStatus,
)
from belllab.types import Envelope, SpectralTrack


class ModalEnergyExchangeStatus(Enum):
    """Mutually exclusive status for operational energy-exchange evidence."""

    SUPPORTED = "supported"
    SUPPORTED_WITH_RESERVATIONS = "supported_with_reservations"
    INCONCLUSIVE = "inconclusive"
    NOT_SUPPORTED = "not_supported"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    INVALID_INPUT = "invalid_input"


class ModalEnergyExchangeReason(Enum):
    """Typed reasons grouped by support, reservation, absence, and invalidity."""

    OPPOSED_ENVELOPE_TRENDS = "opposed_envelope_trends"
    SIGNIFICANT_NEGATIVE_CORRELATION = "significant_negative_correlation"
    LAGGED_NEGATIVE_CORRELATION = "lagged_negative_correlation"
    DELAYED_GROWTH = "delayed_growth"
    LATE_AMPLITUDE_RECOVERY = "late_amplitude_recovery"
    ALTERNATING_DOMINANCE = "alternating_dominance"
    APPROXIMATELY_CONSERVED_PAIR_ENERGY = "approximately_conserved_pair_energy"
    TEMPORAL_OVERLAP_SUFFICIENT = "temporal_overlap_sufficient"
    TRACKING_QUALITY_SUFFICIENT = "tracking_quality_sufficient"
    AMBIGUOUS_TRACKING = "ambiguous_tracking"
    NEAR_THRESHOLD_TRACKING = "near_threshold_tracking"
    POSSIBLE_BEATING = "possible_beating"
    POSSIBLE_FREQUENCY_CROSSING = "possible_frequency_crossing"
    POSSIBLE_PEAK_OVERLAP = "possible_peak_overlap"
    SHORT_OVERLAP_WINDOW = "short_overlap_window"
    UNEQUAL_SAMPLING = "unequal_sampling"
    INTERPOLATION_REQUIRED = "interpolation_required"
    BACKGROUND_CONTAMINATION = "background_contamination"
    CLIPPING_CONTEXT = "clipping_context"
    POSSIBLE_SPLIT_CONTEXT = "possible_split_context"
    POSSIBLE_MERGE_CONTEXT = "possible_merge_context"
    CORRELATION_NOT_SIGNIFICANT = "correlation_not_significant"
    SAME_DIRECTION_TRENDS = "same_direction_trends"
    NO_DELAYED_GROWTH = "no_delayed_growth"
    PAIR_ENERGY_NOT_CONSERVED = "pair_energy_not_conserved"
    DECAY_ONLY_BEHAVIOR = "decay_only_behavior"
    INSUFFICIENT_TEMPORAL_OVERLAP = "insufficient_temporal_overlap"
    INSUFFICIENT_DYNAMIC_RANGE = "insufficient_dynamic_range"
    INSUFFICIENT_ENVELOPE_SAMPLES = "insufficient_envelope_samples"
    MISSING_ENVELOPE = "missing_envelope"
    MISSING_TIME_AXIS = "missing_time_axis"
    INCOMPATIBLE_TIME_AXES = "incompatible_time_axes"
    INVALID_AMPLITUDE_VALUES = "invalid_amplitude_values"
    INVALID_TIME_VALUES = "invalid_time_values"
    UNSUPPORTED_SOURCE_STATUS = "unsupported_source_status"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ModalAmplitudeRepresentation(Enum):
    """Amplitude representation used by the evidence layer."""

    LINEAR_AMPLITUDE = "linear_amplitude"
    NORMALIZED_AMPLITUDE = "normalized_amplitude"
    AMPLITUDE_DB = "amplitude_db"
    RELATIVE_POWER = "relative_power"
    OPERATIONAL_ENERGY = "operational_energy"


class ModalEnvelopeResamplingPolicy(Enum):
    """Temporal alignment policy for two prepared envelope series."""

    REQUIRE_IDENTICAL = "require_identical"
    INTERSECTION = "intersection"
    LINEAR_INTERPOLATION = "linear_interpolation"
    DOWNSAMPLE = "downsample"


class ModalEnvelopeNormalizationMethod(Enum):
    """Envelope normalization method."""

    NONE = "none"
    PEAK = "peak"
    INITIAL = "initial"
    MEAN = "mean"


class ModalEnvelopeSmoothingMethod(Enum):
    """Optional smoothing applied to a copy of the envelope."""

    NONE = "none"
    MOVING_AVERAGE = "moving_average"


class ModalEnvelopeTrendMethod(Enum):
    """Explicit descriptive trend estimator."""

    LINEAR_REGRESSION = "linear_regression"
    SEGMENT_DIFFERENCE = "segment_difference"
    MEDIAN_DERIVATIVE = "median_derivative"
    START_END = "start_end"


class ModalEnvelopeCorrelationMethod(Enum):
    """Correlation method for aligned envelopes."""

    PEARSON = "pearson"
    SPEARMAN = "spearman"


class ModalEnvelopeSignificanceMethod(Enum):
    """Deterministic operational significance method."""

    DISABLED = "disabled"
    CIRCULAR_SHIFT = "circular_shift"
    BLOCK_PERMUTATION = "block_permutation"


class ModalEnergyProxy(Enum):
    """Operational pair-energy proxy convention."""

    AMPLITUDE_SQUARED = "amplitude_squared"
    NORMALIZED_AMPLITUDE_SQUARED = "normalized_amplitude_squared"
    FREQUENCY_WEIGHTED_AMPLITUDE_SQUARED = "frequency_weighted_amplitude_squared"


@dataclass(frozen=True, slots=True)
class ModalEnergyExchangeSettings:
    """Explicit configuration for operational energy-exchange evidence."""

    allow_accepted_hypotheses: bool = True
    allow_accepted_with_reservations: bool = True
    allow_partial_parameter_estimates: bool = True
    allow_inconclusive_sources_for_audit: bool = False
    analysis_window_start_s: float | None = None
    analysis_window_end_s: float | None = None
    exclude_attack_duration_s: float = 0.0
    minimum_overlap_duration_s: float | None = 0.0
    minimum_overlap_sample_count: int = 5
    maximum_time_step_mismatch_fraction: float = 0.05
    resampling_policy: ModalEnvelopeResamplingPolicy = (
        ModalEnvelopeResamplingPolicy.REQUIRE_IDENTICAL
    )
    resampling_step_s: float | None = None
    amplitude_representation: ModalAmplitudeRepresentation = (
        ModalAmplitudeRepresentation.LINEAR_AMPLITUDE
    )
    minimum_positive_amplitude: float = 1e-12
    normalize_envelopes: bool = True
    normalization_method: ModalEnvelopeNormalizationMethod = (
        ModalEnvelopeNormalizationMethod.PEAK
    )
    smoothing_method: ModalEnvelopeSmoothingMethod = ModalEnvelopeSmoothingMethod.NONE
    smoothing_window_s: float | None = None
    allow_log_amplitude: bool = True
    trend_method: ModalEnvelopeTrendMethod = ModalEnvelopeTrendMethod.LINEAR_REGRESSION
    minimum_negative_slope: float = 1e-9
    minimum_positive_slope: float = 1e-9
    minimum_delayed_growth_fraction: float = 0.15
    minimum_recovery_fraction: float = 0.15
    minimum_alternating_dominance_count: int = 2
    minimum_dynamic_range_fraction: float | None = 0.02
    growth_minimum_duration_s: float | None = 0.0
    recovery_minimum_duration_s: float | None = 0.0
    dominance_hysteresis_ratio: float = 1.05
    correlation_method: ModalEnvelopeCorrelationMethod = (
        ModalEnvelopeCorrelationMethod.PEARSON
    )
    minimum_negative_correlation_magnitude: float = 0.6
    maximum_zero_lag_correlation: float = 0.0
    maximum_lag_s: float | None = 0.25
    lag_step_s: float | None = None
    minimum_lagged_correlation_magnitude: float = 0.6
    significance_method: ModalEnvelopeSignificanceMethod = (
        ModalEnvelopeSignificanceMethod.CIRCULAR_SHIFT
    )
    significance_level: float = 0.05
    permutation_count: int = 200
    random_seed: int | None = 0
    significance_block_size: int = 2
    energy_proxy: ModalEnergyProxy = ModalEnergyProxy.NORMALIZED_AMPLITUDE_SQUARED
    pair_energy_variation_limit: float | None = 0.35
    minimum_pair_energy_stability_fraction: float = 0.7
    allow_frequency_weighted_energy_proxy: bool = False
    detect_possible_beating: bool = True
    maximum_frequency_separation_for_beating_hz: float | None = 5.0
    minimum_beating_cycles: float = 2.0
    beating_period_tolerance_fraction: float = 0.25
    reserve_frequency_crossing: bool = True
    reserve_peak_overlap: bool = True
    minimum_support_score: float = 0.6
    minimum_reservation_score: float = 0.4
    require_opposed_trends: bool = True
    require_negative_correlation: bool = True
    require_delayed_response: bool = False
    require_pair_energy_stability: bool = False
    opposed_trend_score_weight: float = 1.0
    negative_correlation_score_weight: float = 1.0
    lagged_correlation_score_weight: float = 0.75
    delayed_growth_score_weight: float = 0.75
    recovery_score_weight: float = 0.5
    alternating_dominance_score_weight: float = 0.5
    pair_energy_score_weight: float = 0.5
    tracking_quality_score_weight: float = 0.25
    beating_reservation_penalty: float = 0.2
    missing_evidence_penalty: float = 0.1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resampling_policy",
            _coerce_enum(self.resampling_policy, ModalEnvelopeResamplingPolicy),
        )
        object.__setattr__(
            self,
            "amplitude_representation",
            _coerce_enum(self.amplitude_representation, ModalAmplitudeRepresentation),
        )
        object.__setattr__(
            self,
            "normalization_method",
            _coerce_enum(
                self.normalization_method,
                ModalEnvelopeNormalizationMethod,
            ),
        )
        object.__setattr__(
            self,
            "smoothing_method",
            _coerce_enum(self.smoothing_method, ModalEnvelopeSmoothingMethod),
        )
        object.__setattr__(
            self,
            "trend_method",
            _coerce_enum(self.trend_method, ModalEnvelopeTrendMethod),
        )
        object.__setattr__(
            self,
            "correlation_method",
            _coerce_enum(self.correlation_method, ModalEnvelopeCorrelationMethod),
        )
        object.__setattr__(
            self,
            "significance_method",
            _coerce_enum(self.significance_method, ModalEnvelopeSignificanceMethod),
        )
        object.__setattr__(
            self,
            "energy_proxy",
            _coerce_enum(self.energy_proxy, ModalEnergyProxy),
        )
        for name in (
            "allow_accepted_hypotheses",
            "allow_accepted_with_reservations",
            "allow_partial_parameter_estimates",
            "allow_inconclusive_sources_for_audit",
            "normalize_envelopes",
            "allow_log_amplitude",
            "allow_frequency_weighted_energy_proxy",
            "detect_possible_beating",
            "reserve_frequency_crossing",
            "reserve_peak_overlap",
            "require_opposed_trends",
            "require_negative_correlation",
            "require_delayed_response",
            "require_pair_energy_stability",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")
        nonnegative_optional = (
            "analysis_window_start_s",
            "analysis_window_end_s",
            "exclude_attack_duration_s",
            "minimum_overlap_duration_s",
            "resampling_step_s",
            "smoothing_window_s",
            "minimum_negative_slope",
            "minimum_positive_slope",
            "growth_minimum_duration_s",
            "recovery_minimum_duration_s",
            "maximum_lag_s",
            "lag_step_s",
            "maximum_frequency_separation_for_beating_hz",
        )
        for name in nonnegative_optional:
            value = getattr(self, name)
            if value is not None and (not _finite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative.")
        positive_optional = ("resampling_step_s", "lag_step_s", "smoothing_window_s")
        for name in positive_optional:
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided.")
        if self.minimum_positive_amplitude <= 0 or not _finite(
            self.minimum_positive_amplitude
        ):
            raise ValueError("minimum_positive_amplitude must be finite and positive.")
        if self.minimum_overlap_sample_count <= 0:
            raise ValueError("minimum_overlap_sample_count must be positive.")
        if self.permutation_count <= 0:
            raise ValueError("permutation_count must be positive.")
        if self.significance_block_size <= 0:
            raise ValueError("significance_block_size must be positive.")
        if self.random_seed is not None and not isinstance(self.random_seed, int):
            raise ValueError("random_seed must be an integer or None.")
        if (
            self.analysis_window_start_s is not None
            and self.analysis_window_end_s is not None
            and self.analysis_window_end_s <= self.analysis_window_start_s
        ):
            raise ValueError("analysis_window_end_s must be above start.")
        for name in (
            "maximum_time_step_mismatch_fraction",
            "minimum_delayed_growth_fraction",
            "minimum_recovery_fraction",
            "minimum_dynamic_range_fraction",
            "minimum_negative_correlation_magnitude",
            "maximum_zero_lag_correlation",
            "minimum_lagged_correlation_magnitude",
            "significance_level",
            "pair_energy_variation_limit",
            "minimum_pair_energy_stability_fraction",
            "beating_period_tolerance_fraction",
            "minimum_support_score",
            "minimum_reservation_score",
        ):
            value = getattr(self, name)
            if value is not None and (not _finite(value) or not 0 <= value <= 1):
                raise ValueError(f"{name} must be finite and in [0, 1].")
        if self.minimum_alternating_dominance_count < 0:
            raise ValueError("minimum_alternating_dominance_count must be non-negative.")
        if not _finite(self.dominance_hysteresis_ratio) or self.dominance_hysteresis_ratio < 1:
            raise ValueError("dominance_hysteresis_ratio must be finite and >= 1.")
        if not _finite(self.minimum_beating_cycles) or self.minimum_beating_cycles < 0:
            raise ValueError("minimum_beating_cycles must be finite and non-negative.")
        score_weights = (
            self.opposed_trend_score_weight,
            self.negative_correlation_score_weight,
            self.lagged_correlation_score_weight,
            self.delayed_growth_score_weight,
            self.recovery_score_weight,
            self.alternating_dominance_score_weight,
            self.pair_energy_score_weight,
            self.tracking_quality_score_weight,
            self.beating_reservation_penalty,
            self.missing_evidence_penalty,
        )
        if any(not _finite(value) or value < 0 for value in score_weights):
            raise ValueError("score weights and penalties must be finite and non-negative.")
        if sum(score_weights[:8]) <= 0:
            raise ValueError("at least one positive support score weight is required.")


@dataclass(frozen=True, slots=True)
class ModalEnvelopeSeries:
    """Prepared amplitude envelope copied from an already computed source."""

    source_id: str
    hypothesis_id: str | None
    candidate_id: str | int | None
    track_id: str | int | None
    recording_id: str | None
    dynamic_label: str | None
    times_s: tuple[float, ...]
    amplitudes: tuple[float, ...]
    normalized_amplitudes: tuple[float, ...]
    energy_proxy: tuple[float, ...]
    valid_mask: tuple[bool, ...]
    sample_count: int
    time_start_s: float | None
    time_end_s: float | None
    duration_s: float | None
    sampling_step_s: float | None
    interpolated: bool
    smoothed: bool
    normalization_method: ModalEnvelopeNormalizationMethod
    valid: bool
    reasons: tuple[ModalEnergyExchangeReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "normalization_method",
            _coerce_enum(
                self.normalization_method,
                ModalEnvelopeNormalizationMethod,
            ),
        )
        if not self.source_id.strip():
            raise ValueError("source_id must be nonempty.")
        if self.sample_count != len(self.times_s):
            raise ValueError("sample_count must match times_s.")
        for values in (
            self.amplitudes,
            self.normalized_amplitudes,
            self.energy_proxy,
            self.valid_mask,
        ):
            if len(values) != self.sample_count:
                raise ValueError("series vectors must have coherent lengths.")
        _validate_reasons(self.reasons)
        _validate_texts(self.diagnostics, "diagnostics")
        if any(not isinstance(value, bool) for value in self.valid_mask):
            raise ValueError("valid_mask entries must be booleans.")
        numeric = (
            *self.times_s,
            *self.amplitudes,
            *self.normalized_amplitudes,
            *self.energy_proxy,
        )
        if any(not _finite(value) for value in numeric):
            raise ValueError("series numeric values must be finite.")
        if any(value < 0 for value in self.energy_proxy):
            raise ValueError("energy proxy values must be non-negative.")
        if any(
            later <= earlier for earlier, later in zip(self.times_s, self.times_s[1:])
        ):
            raise ValueError("series times_s must be strictly increasing.")
        if self.sample_count:
            if self.time_start_s != self.times_s[0] or self.time_end_s != self.times_s[-1]:
                raise ValueError("series time bounds must match times_s.")
            expected_duration = self.times_s[-1] - self.times_s[0]
            if not _close_optional(self.duration_s, expected_duration):
                raise ValueError("duration_s must match the series span.")
            if self.sampling_step_s is not None and (
                not _finite(self.sampling_step_s) or self.sampling_step_s <= 0
            ):
                raise ValueError("sampling_step_s must be positive when present.")
        elif self.valid:
            raise ValueError("valid envelope series requires at least one sample.")


@dataclass(frozen=True, slots=True)
class ModalEnvelopeAlignment:
    """Two prepared envelopes aligned on a common time axis."""

    source_a_id: str
    source_b_id: str
    common_times_s: tuple[float, ...]
    aligned_amplitudes_a: tuple[float, ...]
    aligned_amplitudes_b: tuple[float, ...]
    aligned_energy_proxy_a: tuple[float, ...]
    aligned_energy_proxy_b: tuple[float, ...]
    overlap_start_s: float | None
    overlap_end_s: float | None
    overlap_duration_s: float | None
    sample_count: int
    resampling_applied: bool
    resampling_step_s: float | None
    valid: bool
    reasons: tuple[ModalEnergyExchangeReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.source_a_id.strip() or not self.source_b_id.strip():
            raise ValueError("alignment source IDs must be nonempty.")
        if self.sample_count != len(self.common_times_s):
            raise ValueError("alignment sample_count must match common_times_s.")
        for values in (
            self.aligned_amplitudes_a,
            self.aligned_amplitudes_b,
            self.aligned_energy_proxy_a,
            self.aligned_energy_proxy_b,
        ):
            if len(values) != self.sample_count:
                raise ValueError("aligned vectors must have coherent lengths.")
        _validate_reasons(self.reasons)
        _validate_texts(self.diagnostics, "diagnostics")
        if any(
            not _finite(value)
            for value in (
                *self.common_times_s,
                *self.aligned_amplitudes_a,
                *self.aligned_amplitudes_b,
                *self.aligned_energy_proxy_a,
                *self.aligned_energy_proxy_b,
            )
        ):
            raise ValueError("aligned numeric values must be finite.")
        if any(value < 0 for value in (*self.aligned_energy_proxy_a, *self.aligned_energy_proxy_b)):
            raise ValueError("aligned energy proxies must be non-negative.")
        if any(
            later <= earlier
            for earlier, later in zip(self.common_times_s, self.common_times_s[1:])
        ):
            raise ValueError("common_times_s must be strictly increasing.")
        if self.sample_count:
            if self.overlap_start_s != self.common_times_s[0] or self.overlap_end_s != self.common_times_s[-1]:
                raise ValueError("alignment bounds must match common_times_s.")
            if not _close_optional(
                self.overlap_duration_s,
                self.common_times_s[-1] - self.common_times_s[0],
            ):
                raise ValueError("overlap_duration_s must match common span.")
        elif self.valid:
            raise ValueError("valid alignment requires samples.")
        if self.resampling_step_s is not None and (
            not _finite(self.resampling_step_s) or self.resampling_step_s <= 0
        ):
            raise ValueError("resampling_step_s must be positive when present.")


@dataclass(frozen=True, slots=True)
class ModalEnvelopeTrendEvidence:
    """Descriptive trend evidence for two aligned envelopes."""

    slope_a: float | None
    slope_b: float | None
    normalized_slope_a: float | None
    normalized_slope_b: float | None
    trend_a: str
    trend_b: str
    opposed_trends: bool
    trend_overlap_fraction: float | None
    change_point_times_a: tuple[float, ...]
    change_point_times_b: tuple[float, ...]
    late_growth_detected_a: bool
    late_growth_detected_b: bool
    late_recovery_detected_a: bool
    late_recovery_detected_b: bool
    passes: bool
    reasons: tuple[ModalEnergyExchangeReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_optional_finite(
            self.slope_a,
            self.slope_b,
            self.normalized_slope_a,
            self.normalized_slope_b,
            self.trend_overlap_fraction,
            *self.change_point_times_a,
            *self.change_point_times_b,
        )
        if self.trend_overlap_fraction is not None and not 0 <= self.trend_overlap_fraction <= 1:
            raise ValueError("trend_overlap_fraction must be in [0, 1].")
        _validate_reasons(self.reasons)
        _validate_texts(self.diagnostics, "diagnostics")


@dataclass(frozen=True, slots=True)
class ModalDelayedGrowthEvidence:
    """Evidence for late growth in one envelope."""

    source_id: str
    minimum_time_s: float | None
    growth_start_time_s: float | None
    growth_peak_time_s: float | None
    baseline_amplitude: float | None
    peak_amplitude: float | None
    growth_absolute: float | None
    growth_relative: float | None
    growth_duration_s: float | None
    supported: bool
    reasons: tuple[ModalEnergyExchangeReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id must be nonempty.")
        _validate_optional_finite(
            self.minimum_time_s,
            self.growth_start_time_s,
            self.growth_peak_time_s,
            self.baseline_amplitude,
            self.peak_amplitude,
            self.growth_absolute,
            self.growth_relative,
            self.growth_duration_s,
        )
        if self.growth_relative is not None and self.growth_relative < 0:
            raise ValueError("growth_relative must not be negative.")
        if self.growth_duration_s is not None and self.growth_duration_s < 0:
            raise ValueError("growth_duration_s must not be negative.")
        _validate_reasons(self.reasons)
        _validate_texts(self.diagnostics, "diagnostics")


@dataclass(frozen=True, slots=True)
class ModalAmplitudeRecoveryEvidence:
    """Evidence for late recovery after an initial decline."""

    source_id: str
    minimum_time_s: float | None
    recovery_start_time_s: float | None
    recovery_peak_time_s: float | None
    minimum_amplitude: float | None
    recovered_amplitude: float | None
    recovery_absolute: float | None
    recovery_relative: float | None
    recovery_duration_s: float | None
    initial_decay_detected: bool
    supported: bool
    reasons: tuple[ModalEnergyExchangeReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id must be nonempty.")
        _validate_optional_finite(
            self.minimum_time_s,
            self.recovery_start_time_s,
            self.recovery_peak_time_s,
            self.minimum_amplitude,
            self.recovered_amplitude,
            self.recovery_absolute,
            self.recovery_relative,
            self.recovery_duration_s,
        )
        if self.recovery_relative is not None and self.recovery_relative < 0:
            raise ValueError("recovery_relative must not be negative.")
        if self.recovery_duration_s is not None and self.recovery_duration_s < 0:
            raise ValueError("recovery_duration_s must not be negative.")
        _validate_reasons(self.reasons)
        _validate_texts(self.diagnostics, "diagnostics")


@dataclass(frozen=True, slots=True)
class ModalEnvelopeCorrelationEvidence:
    """Correlation and lag evidence for two aligned envelopes."""

    method: ModalEnvelopeCorrelationMethod
    zero_lag_correlation: float | None
    zero_lag_p_value: float | None
    lag_values_s: tuple[float, ...]
    lagged_correlations: tuple[float | None, ...]
    best_negative_lag_s: float | None
    best_negative_correlation: float | None
    best_negative_p_value: float | None
    significant_negative_correlation: bool
    effective_sample_count: int
    passes: bool
    reasons: tuple[ModalEnergyExchangeReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "method",
            _coerce_enum(self.method, ModalEnvelopeCorrelationMethod),
        )
        if self.effective_sample_count < 0:
            raise ValueError("effective_sample_count must not be negative.")
        if len(self.lag_values_s) != len(self.lagged_correlations):
            raise ValueError("lag values and correlations must have coherent lengths.")
        _validate_optional_finite(
            self.zero_lag_correlation,
            self.zero_lag_p_value,
            *self.lag_values_s,
            *tuple(value for value in self.lagged_correlations if value is not None),
            self.best_negative_lag_s,
            self.best_negative_correlation,
            self.best_negative_p_value,
        )
        for value in (
            self.zero_lag_correlation,
            self.best_negative_correlation,
            *tuple(value for value in self.lagged_correlations if value is not None),
        ):
            if value is not None and not -1 <= value <= 1:
                raise ValueError("correlations must be in [-1, 1].")
        for value in (
            self.zero_lag_p_value,
            self.best_negative_p_value,
        ):
            if value is not None and not 0 <= value <= 1:
                raise ValueError("p-values must be in [0, 1].")
        _validate_reasons(self.reasons)
        _validate_texts(self.diagnostics, "diagnostics")


@dataclass(frozen=True, slots=True)
class ModalPairEnergyEvidence:
    """Auxiliary evidence from an apparent relative pair-energy proxy."""

    energy_a: tuple[float, ...]
    energy_b: tuple[float, ...]
    pair_energy: tuple[float, ...]
    normalized_pair_energy: tuple[float, ...]
    pair_energy_mean: float | None
    pair_energy_standard_deviation: float | None
    pair_energy_relative_range: float | None
    pair_energy_coefficient_of_variation: float | None
    stable_pair_energy_fraction: float | None
    approximately_conserved: bool
    passes: bool
    reasons: tuple[ModalEnergyExchangeReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        length = len(self.energy_a)
        for values in (
            self.energy_b,
            self.pair_energy,
            self.normalized_pair_energy,
        ):
            if len(values) != length:
                raise ValueError("pair-energy vectors must have coherent lengths.")
        if any(
            not _finite(value)
            for value in (
                *self.energy_a,
                *self.energy_b,
                *self.pair_energy,
                *self.normalized_pair_energy,
            )
        ):
            raise ValueError("pair-energy values must be finite.")
        if any(value < 0 for value in (*self.energy_a, *self.energy_b, *self.pair_energy)):
            raise ValueError("energy values must be non-negative.")
        _validate_optional_finite(
            self.pair_energy_mean,
            self.pair_energy_standard_deviation,
            self.pair_energy_relative_range,
            self.pair_energy_coefficient_of_variation,
            self.stable_pair_energy_fraction,
        )
        if self.stable_pair_energy_fraction is not None and not 0 <= self.stable_pair_energy_fraction <= 1:
            raise ValueError("stable_pair_energy_fraction must be in [0, 1].")
        _validate_reasons(self.reasons)
        _validate_texts(self.diagnostics, "diagnostics")


@dataclass(frozen=True, slots=True)
class ModalAlternatingDominanceEvidence:
    """Dominance alternation between two envelopes with hysteresis."""

    dominance_series: tuple[str, ...]
    dominance_change_times_s: tuple[float, ...]
    dominance_change_count: int
    minimum_dominance_ratio: float
    mean_dominance_duration_s: float | None
    alternating_dominance: bool
    passes: bool
    reasons: tuple[ModalEnergyExchangeReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.dominance_change_count != len(self.dominance_change_times_s):
            raise ValueError("dominance_change_count must match change times.")
        if self.dominance_change_count < 0:
            raise ValueError("dominance_change_count must not be negative.")
        if any(item not in {"a", "b", "tie"} for item in self.dominance_series):
            raise ValueError("dominance_series contains an unknown label.")
        if not _finite(self.minimum_dominance_ratio) or self.minimum_dominance_ratio < 1:
            raise ValueError("minimum_dominance_ratio must be finite and >= 1.")
        _validate_optional_finite(
            self.mean_dominance_duration_s,
            *self.dominance_change_times_s,
        )
        if self.mean_dominance_duration_s is not None and self.mean_dominance_duration_s < 0:
            raise ValueError("mean_dominance_duration_s must not be negative.")
        _validate_reasons(self.reasons)
        _validate_texts(self.diagnostics, "diagnostics")


@dataclass(frozen=True, slots=True)
class ModalBeatingEvidence:
    """Compatibility of observed modulation with an apparent beating period."""

    frequency_a_hz: float | None
    frequency_b_hz: float | None
    frequency_separation_hz: float | None
    expected_beating_period_s: float | None
    observed_modulation_period_s: float | None
    modulation_period_difference: float | None
    sufficient_cycles: bool
    possible_beating: bool
    passes: bool
    reasons: tuple[ModalEnergyExchangeReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_optional_finite(
            self.frequency_a_hz,
            self.frequency_b_hz,
            self.frequency_separation_hz,
            self.expected_beating_period_s,
            self.observed_modulation_period_s,
            self.modulation_period_difference,
        )
        for value in (
            self.frequency_a_hz,
            self.frequency_b_hz,
            self.frequency_separation_hz,
            self.expected_beating_period_s,
            self.observed_modulation_period_s,
        ):
            if value is not None and value <= 0:
                raise ValueError("beating frequencies and periods must be positive.")
        if self.modulation_period_difference is not None and self.modulation_period_difference < 0:
            raise ValueError("modulation_period_difference must not be negative.")
        _validate_reasons(self.reasons)
        _validate_texts(self.diagnostics, "diagnostics")


@dataclass(frozen=True, slots=True)
class ModalEnergyExchangeScore:
    """Audit-friendly score assembled from explicit components."""

    opposed_trend_component: float | None
    negative_correlation_component: float | None
    lagged_correlation_component: float | None
    delayed_growth_component: float | None
    recovery_component: float | None
    alternating_dominance_component: float | None
    pair_energy_component: float | None
    tracking_quality_component: float | None
    beating_penalty_or_reservation: float
    missing_evidence_penalty: float
    raw_score: float
    normalized_score: float
    passes_support_threshold: bool
    passes_reservation_threshold: bool
    components: tuple[tuple[str, float | None, float, float | None], ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_optional_finite(
            self.opposed_trend_component,
            self.negative_correlation_component,
            self.lagged_correlation_component,
            self.delayed_growth_component,
            self.recovery_component,
            self.alternating_dominance_component,
            self.pair_energy_component,
            self.tracking_quality_component,
            self.beating_penalty_or_reservation,
            self.missing_evidence_penalty,
            self.raw_score,
            self.normalized_score,
        )
        if not 0 <= self.normalized_score <= 1:
            raise ValueError("normalized_score must be in [0, 1].")
        for _, value, weight, contribution in self.components:
            if value is not None and not _finite(value):
                raise ValueError("score component values must be finite.")
            if not _finite(weight) or weight < 0:
                raise ValueError("score component weights must be non-negative.")
            if contribution is not None and not _finite(contribution):
                raise ValueError("score component contributions must be finite.")
        _validate_texts(self.diagnostics, "diagnostics")


@dataclass(frozen=True, slots=True)
class ModalEnergyExchangeProvenance:
    """Provenance for a pair evidence object."""

    source_a_id: str
    source_b_id: str
    hypothesis_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    track_ids: tuple[str, ...]
    recording_id: str | None
    dynamic_label: str | None
    time_window: tuple[float | None, float | None]
    settings_fingerprint: str
    input_sample_counts: tuple[int, int]
    aligned_sample_count: int
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.source_a_id.strip() or not self.source_b_id.strip():
            raise ValueError("provenance source IDs must be nonempty.")
        for collection, name in (
            (self.hypothesis_ids, "hypothesis_ids"),
            (self.candidate_ids, "candidate_ids"),
            (self.track_ids, "track_ids"),
        ):
            _validate_texts(collection, name)
        _validate_optional_finite(*self.time_window)
        if min(self.input_sample_counts) < 0 or self.aligned_sample_count < 0:
            raise ValueError("sample counts must not be negative.")
        if not self.settings_fingerprint.strip():
            raise ValueError("settings_fingerprint must be nonempty.")
        _validate_texts(self.diagnostics, "diagnostics")


@dataclass(frozen=True, slots=True)
class ModalEnergyExchangeEvidence:
    """Operational evidence for one unordered source pair."""

    evidence_id: str
    source_a_id: str
    source_b_id: str
    hypothesis_a_id: str | None
    hypothesis_b_id: str | None
    dynamic_label: str | None
    status: ModalEnergyExchangeStatus
    alignment: ModalEnvelopeAlignment
    trend_evidence: ModalEnvelopeTrendEvidence
    delayed_growth_evidence: tuple[ModalDelayedGrowthEvidence, ModalDelayedGrowthEvidence]
    recovery_evidence: tuple[ModalAmplitudeRecoveryEvidence, ModalAmplitudeRecoveryEvidence]
    correlation_evidence: ModalEnvelopeCorrelationEvidence
    pair_energy_evidence: ModalPairEnergyEvidence
    alternating_dominance_evidence: ModalAlternatingDominanceEvidence
    beating_evidence: ModalBeatingEvidence
    score: ModalEnergyExchangeScore
    supporting_reasons: tuple[ModalEnergyExchangeReason, ...]
    reservation_reasons: tuple[ModalEnergyExchangeReason, ...]
    inconclusive_reasons: tuple[ModalEnergyExchangeReason, ...]
    not_supported_reasons: tuple[ModalEnergyExchangeReason, ...]
    insufficient_evidence_reasons: tuple[ModalEnergyExchangeReason, ...]
    invalid_reasons: tuple[ModalEnergyExchangeReason, ...]
    valid: bool
    requires_review: bool
    provenance: ModalEnergyExchangeProvenance
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "status",
            _coerce_enum(self.status, ModalEnergyExchangeStatus),
        )
        if not self.evidence_id.strip():
            raise ValueError("evidence_id must be nonempty.")
        if self.source_a_id != self.alignment.source_a_id or self.source_b_id != self.alignment.source_b_id:
            raise ValueError("evidence source IDs must match alignment.")
        if len(self.delayed_growth_evidence) != 2 or len(self.recovery_evidence) != 2:
            raise ValueError("pair evidence requires two single-source evidence objects.")
        for reasons in (
            self.supporting_reasons,
            self.reservation_reasons,
            self.inconclusive_reasons,
            self.not_supported_reasons,
            self.insufficient_evidence_reasons,
            self.invalid_reasons,
        ):
            _validate_reasons(reasons)
        _validate_texts(self.diagnostics, "diagnostics")
        expected_valid = self.status in {
            ModalEnergyExchangeStatus.SUPPORTED,
            ModalEnergyExchangeStatus.SUPPORTED_WITH_RESERVATIONS,
        }
        if self.valid != expected_valid:
            raise ValueError("valid flag must match supported statuses.")


@dataclass(frozen=True, slots=True)
class ModalEnergyExchangeResult:
    """Global result with one evidence per eligible canonical pair."""

    dynamic_label: str | None
    pair_evidences: tuple[ModalEnergyExchangeEvidence, ...]
    supported_pairs: tuple[ModalEnergyExchangeEvidence, ...]
    supported_with_reservations_pairs: tuple[ModalEnergyExchangeEvidence, ...]
    inconclusive_pairs: tuple[ModalEnergyExchangeEvidence, ...]
    not_supported_pairs: tuple[ModalEnergyExchangeEvidence, ...]
    insufficient_evidence_pairs: tuple[ModalEnergyExchangeEvidence, ...]
    invalid_pairs: tuple[ModalEnergyExchangeEvidence, ...]
    pair_count: int
    supported_count: int
    supported_with_reservations_count: int
    inconclusive_count: int
    not_supported_count: int
    insufficient_evidence_count: int
    invalid_count: int
    source_count: int
    settings: ModalEnergyExchangeSettings
    valid: bool
    failure_reason: str | None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.pair_count != len(self.pair_evidences):
            raise ValueError("pair_count must match pair_evidences.")
        ids = tuple(item.evidence_id for item in self.pair_evidences)
        if len(ids) != len(set(ids)):
            raise ValueError("evidence IDs must be unique.")
        subsets = {
            ModalEnergyExchangeStatus.SUPPORTED: self.supported_pairs,
            ModalEnergyExchangeStatus.SUPPORTED_WITH_RESERVATIONS: self.supported_with_reservations_pairs,
            ModalEnergyExchangeStatus.INCONCLUSIVE: self.inconclusive_pairs,
            ModalEnergyExchangeStatus.NOT_SUPPORTED: self.not_supported_pairs,
            ModalEnergyExchangeStatus.INSUFFICIENT_EVIDENCE: self.insufficient_evidence_pairs,
            ModalEnergyExchangeStatus.INVALID_INPUT: self.invalid_pairs,
        }
        counts = {
            ModalEnergyExchangeStatus.SUPPORTED: self.supported_count,
            ModalEnergyExchangeStatus.SUPPORTED_WITH_RESERVATIONS: self.supported_with_reservations_count,
            ModalEnergyExchangeStatus.INCONCLUSIVE: self.inconclusive_count,
            ModalEnergyExchangeStatus.NOT_SUPPORTED: self.not_supported_count,
            ModalEnergyExchangeStatus.INSUFFICIENT_EVIDENCE: self.insufficient_evidence_count,
            ModalEnergyExchangeStatus.INVALID_INPUT: self.invalid_count,
        }
        for status, subset in subsets.items():
            expected = tuple(item for item in self.pair_evidences if item.status is status)
            if subset != expected or counts[status] != len(expected):
                raise ValueError("status subsets and counts are inconsistent.")
        if sum(counts.values()) != self.pair_count:
            raise ValueError("status counts must sum to pair_count.")
        if self.source_count < 0:
            raise ValueError("source_count must not be negative.")
        if self.valid and self.failure_reason is not None:
            raise ValueError("valid result must not carry failure_reason.")
        if not self.valid and not self.failure_reason:
            raise ValueError("invalid result requires failure_reason.")
        _validate_texts(self.diagnostics, "diagnostics")


def modal_energy_exchange_settings_fingerprint(
    settings: ModalEnergyExchangeSettings | None = None,
) -> str:
    """Return a deterministic fingerprint for the effective settings."""
    cfg = settings or ModalEnergyExchangeSettings()
    payload = json.dumps(_canonicalize(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prepare_modal_envelope_series(
    source: object | None = None,
    settings: ModalEnergyExchangeSettings | None = None,
    *,
    times_s: Sequence[float] | None = None,
    amplitudes: Sequence[float] | None = None,
    source_id: str | None = None,
    hypothesis_id: str | None = None,
    candidate_id: str | int | None = None,
    track_id: str | int | None = None,
    recording_id: str | None = None,
    dynamic_label: str | None = None,
    amplitude_unit: str | None = None,
    representative_frequency_hz: float | None = None,
    diagnostics: Sequence[str] = (),
) -> ModalEnvelopeSeries:
    """Prepare an immutable envelope series from an already computed source."""
    cfg = settings or ModalEnergyExchangeSettings()
    if isinstance(source, ModalEnvelopeSeries):
        if not source.valid and times_s is None and amplitudes is None:
            return source
        raw_times = times_s if times_s is not None else source.times_s
        raw_amplitudes = amplitudes if amplitudes is not None else source.amplitudes
        source_id = source_id or source.source_id
        hypothesis_id = hypothesis_id if hypothesis_id is not None else source.hypothesis_id
        candidate_id = candidate_id if candidate_id is not None else source.candidate_id
        track_id = track_id if track_id is not None else source.track_id
        recording_id = recording_id if recording_id is not None else source.recording_id
        dynamic_label = dynamic_label if dynamic_label is not None else source.dynamic_label
        diagnostics = (*source.diagnostics, *tuple(diagnostics))
    elif isinstance(source, SpectralTrack):
        raw_times = source.times_s
        raw_amplitudes = source.amplitudes
        source_id = source_id or f"track:{source.track_id}"
        track_id = track_id if track_id is not None else source.track_id
        amplitude_unit = amplitude_unit or source.amplitude_unit
        representative_frequency_hz = (
            representative_frequency_hz or source.median_frequency_hz
        )
        diagnostics = (*source.diagnostics, *tuple(diagnostics))
    elif isinstance(source, Envelope):
        raw_times = source.times_s
        raw_amplitudes = source.amplitudes
        source_id = source_id or _stable_id("envelope", source.times_s, source.amplitudes)
        amplitude_unit = amplitude_unit or source.unit
    elif source is not None and hasattr(source, "times_s") and hasattr(source, "amplitudes"):
        raw_times = getattr(source, "times_s")
        raw_amplitudes = getattr(source, "amplitudes")
        source_id = source_id or _stable_id("series", raw_times, raw_amplitudes)
    else:
        raw_times = times_s
        raw_amplitudes = amplitudes
        source_id = source_id or _stable_id("series", raw_times, raw_amplitudes)

    if raw_times is None:
        return _invalid_series(
            source_id,
            hypothesis_id,
            candidate_id,
            track_id,
            recording_id,
            dynamic_label,
            (ModalEnergyExchangeReason.MISSING_TIME_AXIS,),
            ("missing_time_axis", *tuple(diagnostics)),
            cfg,
        )
    if raw_amplitudes is None:
        return _invalid_series(
            source_id,
            hypothesis_id,
            candidate_id,
            track_id,
            recording_id,
            dynamic_label,
            (ModalEnergyExchangeReason.MISSING_ENVELOPE,),
            ("missing_envelope", *tuple(diagnostics)),
            cfg,
        )
    try:
        times = tuple(float(value) for value in raw_times)
        amps = tuple(float(value) for value in raw_amplitudes)
    except (TypeError, ValueError):
        return _invalid_series(
            source_id,
            hypothesis_id,
            candidate_id,
            track_id,
            recording_id,
            dynamic_label,
            (
                ModalEnergyExchangeReason.INVALID_TIME_VALUES,
                ModalEnergyExchangeReason.INVALID_AMPLITUDE_VALUES,
            ),
            ("non_numeric_envelope_values", *tuple(diagnostics)),
            cfg,
        )
    if not times:
        return _invalid_series(
            source_id,
            hypothesis_id,
            candidate_id,
            track_id,
            recording_id,
            dynamic_label,
            (ModalEnergyExchangeReason.INSUFFICIENT_ENVELOPE_SAMPLES,),
            ("empty_envelope_series", *tuple(diagnostics)),
            cfg,
        )
    if len(times) != len(amps):
        return _invalid_series(
            source_id,
            hypothesis_id,
            candidate_id,
            track_id,
            recording_id,
            dynamic_label,
            (
                ModalEnergyExchangeReason.MISSING_TIME_AXIS,
                ModalEnergyExchangeReason.MISSING_ENVELOPE,
            ),
            ("envelope_length_mismatch", *tuple(diagnostics)),
            cfg,
        )
    if any(not _finite(value) for value in times) or any(
        later <= earlier for earlier, later in zip(times, times[1:])
    ):
        return _invalid_series(
            source_id,
            hypothesis_id,
            candidate_id,
            track_id,
            recording_id,
            dynamic_label,
            (ModalEnergyExchangeReason.INVALID_TIME_VALUES,),
            ("invalid_time_values", *tuple(diagnostics)),
            cfg,
        )
    if any(not _finite(value) for value in amps):
        return _invalid_series(
            source_id,
            hypothesis_id,
            candidate_id,
            track_id,
            recording_id,
            dynamic_label,
            (ModalEnergyExchangeReason.INVALID_AMPLITUDE_VALUES,),
            ("invalid_amplitude_values", *tuple(diagnostics)),
            cfg,
        )
    if _uses_linear_nonnegative_amplitude(cfg, amplitude_unit) and any(value < 0 for value in amps):
        return _invalid_series(
            source_id,
            hypothesis_id,
            candidate_id,
            track_id,
            recording_id,
            dynamic_label,
            (ModalEnergyExchangeReason.INVALID_AMPLITUDE_VALUES,),
            ("negative_linear_amplitude", *tuple(diagnostics)),
            cfg,
        )

    times, amps = _apply_analysis_window(times, amps, cfg)
    if not times:
        return _invalid_series(
            source_id,
            hypothesis_id,
            candidate_id,
            track_id,
            recording_id,
            dynamic_label,
            (ModalEnergyExchangeReason.INSUFFICIENT_ENVELOPE_SAMPLES,),
            ("analysis_window_removed_all_samples", *tuple(diagnostics)),
            cfg,
        )
    smoothed = cfg.smoothing_method is ModalEnvelopeSmoothingMethod.MOVING_AVERAGE
    if smoothed:
        amps = _smooth_moving_average(times, amps, cfg.smoothing_window_s)
    normalized, normalization = _normalize_amplitudes(amps, cfg)
    linear_for_energy = _linear_amplitude_values(amps, amplitude_unit, cfg)
    if cfg.energy_proxy is ModalEnergyProxy.NORMALIZED_AMPLITUDE_SQUARED:
        energy_amplitudes = normalized
    elif cfg.energy_proxy is ModalEnergyProxy.FREQUENCY_WEIGHTED_AMPLITUDE_SQUARED:
        if (
            cfg.allow_frequency_weighted_energy_proxy
            and representative_frequency_hz is not None
            and _finite(representative_frequency_hz)
            and representative_frequency_hz > 0
        ):
            energy = tuple(value * value * representative_frequency_hz for value in linear_for_energy)
            valid_mask = tuple(value > cfg.minimum_positive_amplitude for value in linear_for_energy)
            return ModalEnvelopeSeries(
                source_id=source_id,
                hypothesis_id=hypothesis_id,
                candidate_id=candidate_id,
                track_id=track_id,
                recording_id=recording_id,
                dynamic_label=dynamic_label,
                times_s=times,
                amplitudes=amps,
                normalized_amplitudes=normalized,
                energy_proxy=energy,
                valid_mask=valid_mask,
                sample_count=len(times),
                time_start_s=times[0],
                time_end_s=times[-1],
                duration_s=times[-1] - times[0],
                sampling_step_s=_representative_step(times),
                interpolated=False,
                smoothed=smoothed,
                normalization_method=normalization,
                valid=True,
                reasons=(),
                diagnostics=_ordered_texts(("operational_envelope_series", *tuple(diagnostics))),
            )
        energy_amplitudes = linear_for_energy
        diagnostics = (
            *tuple(diagnostics),
            "frequency_weighted_energy_proxy_unavailable",
        )
    else:
        energy_amplitudes = linear_for_energy
    energy = tuple(value * value for value in energy_amplitudes)
    valid_mask = tuple(value > cfg.minimum_positive_amplitude for value in linear_for_energy)
    return ModalEnvelopeSeries(
        source_id=source_id,
        hypothesis_id=hypothesis_id,
        candidate_id=candidate_id,
        track_id=track_id,
        recording_id=recording_id,
        dynamic_label=dynamic_label,
        times_s=times,
        amplitudes=amps,
        normalized_amplitudes=normalized,
        energy_proxy=energy,
        valid_mask=valid_mask,
        sample_count=len(times),
        time_start_s=times[0],
        time_end_s=times[-1],
        duration_s=times[-1] - times[0],
        sampling_step_s=_representative_step(times),
        interpolated=False,
        smoothed=smoothed,
        normalization_method=normalization,
        valid=True,
        reasons=(),
        diagnostics=_ordered_texts(("operational_envelope_series", *tuple(diagnostics))),
    )


def align_modal_envelope_series(
    source_a: ModalEnvelopeSeries,
    source_b: ModalEnvelopeSeries,
    settings: ModalEnergyExchangeSettings | None = None,
) -> ModalEnvelopeAlignment:
    """Align two prepared envelopes without extrapolating outside overlap."""
    cfg = settings or ModalEnergyExchangeSettings()
    if not source_a.valid or not source_b.valid:
        invalid = tuple(source_a.reasons + source_b.reasons)
        return _invalid_alignment(
            source_a.source_id,
            source_b.source_id,
            invalid or (ModalEnergyExchangeReason.MISSING_ENVELOPE,),
            ("invalid_source_envelope",),
        )
    overlap_start = max(source_a.time_start_s, source_b.time_start_s)  # type: ignore[arg-type]
    overlap_end = min(source_a.time_end_s, source_b.time_end_s)  # type: ignore[arg-type]
    if overlap_end <= overlap_start:
        return _invalid_alignment(
            source_a.source_id,
            source_b.source_id,
            (ModalEnergyExchangeReason.INSUFFICIENT_TEMPORAL_OVERLAP,),
            ("no_common_time_overlap",),
        )
    overlap_duration = overlap_end - overlap_start
    reasons: list[ModalEnergyExchangeReason] = []
    diagnostics: list[str] = []
    if (
        cfg.minimum_overlap_duration_s is not None
        and overlap_duration < cfg.minimum_overlap_duration_s
    ):
        return _invalid_alignment(
            source_a.source_id,
            source_b.source_id,
            (
                ModalEnergyExchangeReason.INSUFFICIENT_TEMPORAL_OVERLAP,
                ModalEnergyExchangeReason.SHORT_OVERLAP_WINDOW,
            ),
            ("overlap_duration_below_minimum",),
        )
    times_a, amps_a, energy_a = _slice_overlap(source_a, overlap_start, overlap_end)
    times_b, amps_b, energy_b = _slice_overlap(source_b, overlap_start, overlap_end)
    if times_a and times_b:
        effective_overlap_start = max(times_a[0], times_b[0])
        effective_overlap_end = min(times_a[-1], times_b[-1])
    else:
        effective_overlap_start = overlap_start
        effective_overlap_end = overlap_end
    resampling_applied = False
    resampling_step = None
    if _axes_compatible(times_a, times_b, cfg.maximum_time_step_mismatch_fraction):
        common_times = times_a
        aligned_a = amps_a
        aligned_b = amps_b
        aligned_energy_a = energy_a
        aligned_energy_b = energy_b
        if times_a != times_b:
            diagnostics.append("axes_within_time_step_mismatch_fraction")
    elif cfg.resampling_policy is ModalEnvelopeResamplingPolicy.REQUIRE_IDENTICAL:
        return _invalid_alignment(
            source_a.source_id,
            source_b.source_id,
            (ModalEnergyExchangeReason.INCOMPATIBLE_TIME_AXES,),
            ("time_axes_not_identical",),
        )
    elif cfg.resampling_policy is ModalEnvelopeResamplingPolicy.INTERSECTION:
        common_times, aligned_a, aligned_b, aligned_energy_a, aligned_energy_b = (
            _intersect_axes(times_a, amps_a, energy_a, times_b, amps_b, energy_b)
        )
    else:
        step = cfg.resampling_step_s or max(
            source_a.sampling_step_s or overlap_duration,
            source_b.sampling_step_s or overlap_duration,
        )
        common_times = _resampling_grid(effective_overlap_start, effective_overlap_end, step)
        aligned_a = tuple(_interpolate(times_a, amps_a, value) for value in common_times)
        aligned_b = tuple(_interpolate(times_b, amps_b, value) for value in common_times)
        aligned_energy_a = tuple(_interpolate(times_a, energy_a, value) for value in common_times)
        aligned_energy_b = tuple(_interpolate(times_b, energy_b, value) for value in common_times)
        resampling_applied = True
        resampling_step = step
        reasons.append(ModalEnergyExchangeReason.INTERPOLATION_REQUIRED)
        reasons.append(ModalEnergyExchangeReason.UNEQUAL_SAMPLING)
        diagnostics.append(f"resampling_policy:{cfg.resampling_policy.value}")
    if not common_times:
        return _invalid_alignment(
            source_a.source_id,
            source_b.source_id,
            (
                ModalEnergyExchangeReason.INSUFFICIENT_TEMPORAL_OVERLAP,
                ModalEnergyExchangeReason.INCOMPATIBLE_TIME_AXES,
            ),
            ("empty_common_time_axis",),
        )
    if len(common_times) < cfg.minimum_overlap_sample_count:
        return _invalid_alignment(
            source_a.source_id,
            source_b.source_id,
            (ModalEnergyExchangeReason.INSUFFICIENT_ENVELOPE_SAMPLES,),
            ("aligned_sample_count_below_minimum",),
        )
    reasons.append(ModalEnergyExchangeReason.TEMPORAL_OVERLAP_SUFFICIENT)
    return ModalEnvelopeAlignment(
        source_a_id=source_a.source_id,
        source_b_id=source_b.source_id,
        common_times_s=common_times,
        aligned_amplitudes_a=aligned_a,
        aligned_amplitudes_b=aligned_b,
        aligned_energy_proxy_a=aligned_energy_a,
        aligned_energy_proxy_b=aligned_energy_b,
        overlap_start_s=common_times[0],
        overlap_end_s=common_times[-1],
        overlap_duration_s=common_times[-1] - common_times[0],
        sample_count=len(common_times),
        resampling_applied=resampling_applied,
        resampling_step_s=resampling_step,
        valid=True,
        reasons=_ordered_reasons(reasons),
        diagnostics=_ordered_texts(diagnostics),
    )


def evaluate_modal_envelope_trends(
    alignment: ModalEnvelopeAlignment,
    settings: ModalEnergyExchangeSettings | None = None,
) -> ModalEnvelopeTrendEvidence:
    """Evaluate opposed descriptive envelope trends on aligned samples."""
    cfg = settings or ModalEnergyExchangeSettings()
    if not alignment.valid or alignment.sample_count < 2:
        return ModalEnvelopeTrendEvidence(
            None,
            None,
            None,
            None,
            "indeterminate",
            "indeterminate",
            False,
            None,
            (),
            (),
            False,
            False,
            False,
            False,
            False,
            (ModalEnergyExchangeReason.INSUFFICIENT_ENVELOPE_SAMPLES,),
            ("trend_requires_two_aligned_samples",),
        )
    times = alignment.common_times_s
    a = alignment.aligned_amplitudes_a
    b = alignment.aligned_amplitudes_b
    if _dynamic_range_fraction(a) < (cfg.minimum_dynamic_range_fraction or 0) and _dynamic_range_fraction(b) < (cfg.minimum_dynamic_range_fraction or 0):
        return ModalEnvelopeTrendEvidence(
            None,
            None,
            None,
            None,
            "indeterminate",
            "indeterminate",
            False,
            None,
            (),
            (),
            False,
            False,
            False,
            False,
            False,
            (ModalEnergyExchangeReason.INSUFFICIENT_DYNAMIC_RANGE,),
            ("insufficient_dynamic_range_for_trend",),
        )
    slope_a = _trend_slope(times, a, cfg.trend_method)
    slope_b = _trend_slope(times, b, cfg.trend_method)
    trend_a = _classify_trend(slope_a, cfg)
    trend_b = _classify_trend(slope_b, cfg)
    opposed = (trend_a, trend_b) in {
        ("decreasing", "increasing"),
        ("increasing", "decreasing"),
    }
    overlap_fraction = _opposed_derivative_fraction(a, b)
    reasons: list[ModalEnergyExchangeReason] = []
    if opposed:
        reasons.append(ModalEnergyExchangeReason.OPPOSED_ENVELOPE_TRENDS)
    elif trend_a == trend_b and trend_a in {"increasing", "decreasing"}:
        reasons.append(ModalEnergyExchangeReason.SAME_DIRECTION_TRENDS)
    else:
        reasons.append(ModalEnergyExchangeReason.INSUFFICIENT_EVIDENCE)
    return ModalEnvelopeTrendEvidence(
        slope_a,
        slope_b,
        _normalized_slope(slope_a, a),
        _normalized_slope(slope_b, b),
        trend_a,
        trend_b,
        opposed,
        overlap_fraction,
        _change_points(times, a, cfg),
        _change_points(times, b, cfg),
        False,
        False,
        False,
        False,
        opposed,
        _ordered_reasons(reasons),
        (),
    )


def evaluate_modal_delayed_growth(
    series: ModalEnvelopeSeries | ModalEnvelopeAlignment,
    settings: ModalEnergyExchangeSettings | None = None,
    *,
    source: str = "a",
) -> ModalDelayedGrowthEvidence:
    """Detect sustained late growth in one envelope."""
    cfg = settings or ModalEnergyExchangeSettings()
    source_id, times, amplitudes = _single_series(series, source)
    if len(times) < cfg.minimum_overlap_sample_count:
        return ModalDelayedGrowthEvidence(
            source_id,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            False,
            (ModalEnergyExchangeReason.INSUFFICIENT_ENVELOPE_SAMPLES,),
            ("delayed_growth_requires_more_samples",),
        )
    duration = times[-1] - times[0]
    minimum_time = times[0] + 0.25 * duration
    early_end = max(1, len(times) // 3)
    baseline_index = min(range(early_end), key=lambda index: amplitudes[index])
    late_indices = tuple(index for index, time in enumerate(times) if time >= minimum_time)
    if not late_indices:
        return ModalDelayedGrowthEvidence(
            source_id,
            minimum_time,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            False,
            (ModalEnergyExchangeReason.NO_DELAYED_GROWTH,),
            ("no_late_window_for_growth",),
        )
    min_before_late = min(range(0, late_indices[-1] + 1), key=lambda index: amplitudes[index])
    peak_index = max(late_indices, key=lambda index: amplitudes[index])
    baseline_index = min_before_late if min_before_late < peak_index else baseline_index
    baseline = amplitudes[baseline_index]
    peak = amplitudes[peak_index]
    absolute = peak - baseline
    relative = absolute / max(abs(baseline), cfg.minimum_positive_amplitude)
    duration_s = times[peak_index] - times[baseline_index]
    threshold = cfg.minimum_delayed_growth_fraction
    sustained = _sustained_after_threshold(
        times,
        amplitudes,
        baseline_index,
        peak_index,
        baseline + absolute * 0.5,
        cfg.growth_minimum_duration_s,
    )
    supported = (
        peak_index > baseline_index
        and relative >= threshold
        and duration_s >= (cfg.growth_minimum_duration_s or 0)
        and sustained
    )
    return ModalDelayedGrowthEvidence(
        source_id,
        minimum_time,
        times[baseline_index] if supported else None,
        times[peak_index] if supported else None,
        baseline,
        peak,
        absolute if absolute > 0 else 0.0,
        relative if relative > 0 else 0.0,
        duration_s if duration_s > 0 else 0.0,
        supported,
        (
            (ModalEnergyExchangeReason.DELAYED_GROWTH,)
            if supported
            else (ModalEnergyExchangeReason.NO_DELAYED_GROWTH,)
        ),
        (),
    )


def evaluate_modal_amplitude_recovery(
    series: ModalEnvelopeSeries | ModalEnvelopeAlignment,
    settings: ModalEnergyExchangeSettings | None = None,
    *,
    source: str = "a",
) -> ModalAmplitudeRecoveryEvidence:
    """Detect late amplitude recovery after an initial decrease."""
    cfg = settings or ModalEnergyExchangeSettings()
    source_id, times, amplitudes = _single_series(series, source)
    if len(times) < max(5, cfg.minimum_overlap_sample_count):
        return ModalAmplitudeRecoveryEvidence(
            source_id,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            False,
            False,
            (ModalEnergyExchangeReason.INSUFFICIENT_ENVELOPE_SAMPLES,),
            ("recovery_requires_more_samples",),
        )
    initial = amplitudes[0]
    search_end = max(2, (2 * len(times)) // 3)
    minimum_index = min(range(1, search_end), key=lambda index: amplitudes[index])
    late_indices = tuple(range(minimum_index + 1, len(times)))
    if not late_indices:
        return ModalAmplitudeRecoveryEvidence(
            source_id,
            None,
            None,
            None,
            amplitudes[minimum_index],
            None,
            None,
            None,
            None,
            False,
            False,
            (ModalEnergyExchangeReason.NO_DELAYED_GROWTH,),
            ("no_samples_after_minimum_for_recovery",),
        )
    recovery_peak_index = max(late_indices, key=lambda index: amplitudes[index])
    minimum = amplitudes[minimum_index]
    recovered = amplitudes[recovery_peak_index]
    initial_drop = initial - minimum
    absolute = recovered - minimum
    relative = absolute / max(abs(initial_drop), cfg.minimum_positive_amplitude)
    duration_s = times[recovery_peak_index] - times[minimum_index]
    initial_decay = initial_drop > cfg.minimum_positive_amplitude
    supported = (
        initial_decay
        and absolute > 0
        and relative >= cfg.minimum_recovery_fraction
        and duration_s >= (cfg.recovery_minimum_duration_s or 0)
    )
    return ModalAmplitudeRecoveryEvidence(
        source_id,
        times[minimum_index],
        times[minimum_index] if supported else None,
        times[recovery_peak_index] if supported else None,
        minimum,
        recovered,
        absolute if absolute > 0 else 0.0,
        relative if relative > 0 else 0.0,
        duration_s if duration_s > 0 else 0.0,
        initial_decay,
        supported,
        (
            (ModalEnergyExchangeReason.LATE_AMPLITUDE_RECOVERY,)
            if supported
            else (ModalEnergyExchangeReason.NO_DELAYED_GROWTH,)
        ),
        (),
    )


def evaluate_modal_envelope_correlation(
    alignment: ModalEnvelopeAlignment,
    settings: ModalEnergyExchangeSettings | None = None,
) -> ModalEnvelopeCorrelationEvidence:
    """Evaluate zero-lag and lagged envelope anticorrelation.

    Lag convention: ``lag > 0`` means changes in component A precede changes in
    component B on the aligned ordinal time axis.  This is only a descriptive
    convention and does not imply causal direction.
    """
    cfg = settings or ModalEnergyExchangeSettings()
    if not alignment.valid or alignment.sample_count < 3:
        return _empty_correlation(
            cfg.correlation_method,
            (ModalEnergyExchangeReason.INSUFFICIENT_ENVELOPE_SAMPLES,),
            "correlation_requires_three_samples",
        )
    a = alignment.aligned_amplitudes_a
    b = alignment.aligned_amplitudes_b
    if _dynamic_range_fraction(a) < (cfg.minimum_dynamic_range_fraction or 0) or _dynamic_range_fraction(b) < (cfg.minimum_dynamic_range_fraction or 0):
        return _empty_correlation(
            cfg.correlation_method,
            (ModalEnergyExchangeReason.INSUFFICIENT_DYNAMIC_RANGE,),
            "insufficient_dynamic_range_for_correlation",
        )
    transformed_a, transformed_b = _correlation_vectors(a, b, cfg.correlation_method)
    zero = _correlation(transformed_a, transformed_b)
    p_zero = _operational_p_value(zero, transformed_a, transformed_b, cfg)
    lag_values, lagged = _lagged_correlations(
        alignment.common_times_s,
        transformed_a,
        transformed_b,
        cfg,
    )
    candidates = tuple(
        (lag, corr)
        for lag, corr in zip(lag_values, lagged, strict=True)
        if corr is not None
    )
    best_lag: float | None = None
    best_corr: float | None = None
    if candidates:
        best_lag, best_corr = min(candidates, key=lambda item: item[1])
    best_p = (
        _operational_p_value(best_corr, transformed_a, transformed_b, cfg)
        if best_corr is not None
        else None
    )
    zero_significant = (
        zero is not None
        and zero <= -cfg.minimum_negative_correlation_magnitude
        and (p_zero is None or p_zero <= cfg.significance_level)
    )
    lag_significant = (
        best_corr is not None
        and best_corr <= -cfg.minimum_lagged_correlation_magnitude
        and (best_p is None or best_p <= cfg.significance_level)
    )
    passes = zero_significant or lag_significant
    reasons: list[ModalEnergyExchangeReason] = []
    if zero_significant:
        reasons.append(ModalEnergyExchangeReason.SIGNIFICANT_NEGATIVE_CORRELATION)
    if lag_significant:
        reasons.append(ModalEnergyExchangeReason.LAGGED_NEGATIVE_CORRELATION)
    if not passes:
        reasons.append(ModalEnergyExchangeReason.CORRELATION_NOT_SIGNIFICANT)
    return ModalEnvelopeCorrelationEvidence(
        cfg.correlation_method,
        zero,
        p_zero,
        lag_values,
        lagged,
        best_lag,
        best_corr,
        best_p,
        passes,
        alignment.sample_count,
        passes,
        _ordered_reasons(reasons),
        (),
    )


def evaluate_modal_pair_energy(
    alignment: ModalEnvelopeAlignment,
    settings: ModalEnergyExchangeSettings | None = None,
) -> ModalPairEnergyEvidence:
    """Evaluate approximate stability of an operational pair-energy proxy."""
    cfg = settings or ModalEnergyExchangeSettings()
    if not alignment.valid or alignment.sample_count == 0:
        return ModalPairEnergyEvidence(
            (),
            (),
            (),
            (),
            None,
            None,
            None,
            None,
            None,
            False,
            False,
            (ModalEnergyExchangeReason.INSUFFICIENT_ENVELOPE_SAMPLES,),
            ("pair_energy_requires_aligned_samples",),
        )
    energy_a = alignment.aligned_energy_proxy_a
    energy_b = alignment.aligned_energy_proxy_b
    pair = tuple(a + b for a, b in zip(energy_a, energy_b, strict=True))
    mean = _mean(pair)
    if mean is None or mean <= 0:
        return ModalPairEnergyEvidence(
            energy_a,
            energy_b,
            pair,
            (),
            mean,
            None,
            None,
            None,
            None,
            False,
            False,
            (ModalEnergyExchangeReason.INSUFFICIENT_DYNAMIC_RANGE,),
            ("pair_energy_mean_not_positive",),
        )
    normalized = tuple(value / mean for value in pair)
    std = _population_std(pair)
    relative_range = (max(pair) - min(pair)) / mean
    cv = std / mean if std is not None else None
    stable_fraction = sum(
        abs(value - mean) / mean <= (cfg.pair_energy_variation_limit or 0)
        for value in pair
    ) / len(pair)
    approximately = (
        cfg.pair_energy_variation_limit is None
        or relative_range <= cfg.pair_energy_variation_limit
    ) and stable_fraction >= cfg.minimum_pair_energy_stability_fraction
    return ModalPairEnergyEvidence(
        energy_a,
        energy_b,
        pair,
        normalized,
        mean,
        std,
        relative_range,
        cv,
        stable_fraction,
        approximately,
        approximately,
        (
            (ModalEnergyExchangeReason.APPROXIMATELY_CONSERVED_PAIR_ENERGY,)
            if approximately
            else (ModalEnergyExchangeReason.PAIR_ENERGY_NOT_CONSERVED,)
        ),
        (),
    )


def evaluate_modal_alternating_dominance(
    alignment: ModalEnvelopeAlignment,
    settings: ModalEnergyExchangeSettings | None = None,
) -> ModalAlternatingDominanceEvidence:
    """Evaluate alternating dominance with a configurable hysteresis ratio."""
    cfg = settings or ModalEnergyExchangeSettings()
    if not alignment.valid or alignment.sample_count < 2:
        return ModalAlternatingDominanceEvidence(
            (),
            (),
            0,
            cfg.dominance_hysteresis_ratio,
            None,
            False,
            False,
            (ModalEnergyExchangeReason.INSUFFICIENT_ENVELOPE_SAMPLES,),
            ("alternating_dominance_requires_two_samples",),
        )
    labels: list[str] = []
    previous = "tie"
    for a, b in zip(
        alignment.aligned_energy_proxy_a,
        alignment.aligned_energy_proxy_b,
        strict=True,
    ):
        if b <= 0 and a <= 0:
            label = "tie"
        elif b <= 0 or a / b >= cfg.dominance_hysteresis_ratio:
            label = "a"
        elif a <= 0 or b / a >= cfg.dominance_hysteresis_ratio:
            label = "b"
        else:
            label = previous
        labels.append(label)
        if label != "tie":
            previous = label
    change_times: list[float] = []
    last = next((item for item in labels if item != "tie"), "tie")
    for index, label in enumerate(labels[1:], start=1):
        if label != "tie" and last != "tie" and label != last:
            change_times.append(alignment.common_times_s[index])
        if label != "tie":
            last = label
    durations = tuple(
        later - earlier
        for earlier, later in zip(
            (alignment.common_times_s[0], *change_times),
            (*change_times, alignment.common_times_s[-1]),
        )
        if later >= earlier
    )
    mean_duration = _mean(durations) if durations else None
    alternating = len(change_times) >= cfg.minimum_alternating_dominance_count
    return ModalAlternatingDominanceEvidence(
        tuple(labels),
        tuple(change_times),
        len(change_times),
        cfg.dominance_hysteresis_ratio,
        mean_duration,
        alternating,
        alternating,
        (
            (ModalEnergyExchangeReason.ALTERNATING_DOMINANCE,)
            if alternating
            else (ModalEnergyExchangeReason.INSUFFICIENT_EVIDENCE,)
        ),
        (),
    )


def evaluate_modal_beating_context(
    frequency_a_hz: float | None,
    frequency_b_hz: float | None,
    alignment: ModalEnvelopeAlignment | None = None,
    settings: ModalEnergyExchangeSettings | None = None,
) -> ModalBeatingEvidence:
    """Record compatibility with apparent beating; do not infer coupling."""
    cfg = settings or ModalEnergyExchangeSettings()
    if not cfg.detect_possible_beating:
        return ModalBeatingEvidence(
            frequency_a_hz,
            frequency_b_hz,
            None,
            None,
            None,
            None,
            False,
            False,
            False,
            (),
            ("beating_detection_disabled",),
        )
    if (
        frequency_a_hz is None
        or frequency_b_hz is None
        or not _finite(frequency_a_hz)
        or not _finite(frequency_b_hz)
        or frequency_a_hz <= 0
        or frequency_b_hz <= 0
    ):
        return ModalBeatingEvidence(
            frequency_a_hz if frequency_a_hz is not None and _finite(frequency_a_hz) and frequency_a_hz > 0 else None,
            frequency_b_hz if frequency_b_hz is not None and _finite(frequency_b_hz) and frequency_b_hz > 0 else None,
            None,
            None,
            None,
            None,
            False,
            False,
            False,
            (),
            ("beating_requires_two_positive_frequencies",),
        )
    separation = abs(frequency_a_hz - frequency_b_hz)
    if separation <= 0:
        return ModalBeatingEvidence(
            frequency_a_hz,
            frequency_b_hz,
            None,
            None,
            None,
            None,
            False,
            False,
            False,
            (),
            ("equal_frequencies_no_beating_period",),
        )
    expected = 1.0 / separation
    if (
        cfg.maximum_frequency_separation_for_beating_hz is not None
        and separation > cfg.maximum_frequency_separation_for_beating_hz
    ):
        return ModalBeatingEvidence(
            frequency_a_hz,
            frequency_b_hz,
            separation,
            expected,
            None,
            None,
            False,
            False,
            False,
            (),
            ("frequency_separation_above_beating_window",),
        )
    observed = (
        _observed_modulation_period(alignment.common_times_s, alignment.aligned_amplitudes_a)
        if alignment is not None and alignment.valid and alignment.sample_count >= 3
        else None
    )
    cycles = (
        (alignment.overlap_duration_s or 0.0) / expected
        if alignment is not None and expected > 0
        else 0.0
    )
    sufficient_cycles = cycles >= cfg.minimum_beating_cycles
    difference = (
        abs(observed - expected) / expected
        if observed is not None and expected > 0
        else None
    )
    possible = (
        observed is not None
        and difference is not None
        and difference <= cfg.beating_period_tolerance_fraction
        and sufficient_cycles
    )
    return ModalBeatingEvidence(
        frequency_a_hz,
        frequency_b_hz,
        separation,
        expected,
        observed,
        difference,
        sufficient_cycles,
        possible,
        possible,
        (ModalEnergyExchangeReason.POSSIBLE_BEATING,) if possible else (),
        ("apparent_beating_context_only",) if possible else (),
    )


def compute_modal_energy_exchange_score(
    trend_evidence: ModalEnvelopeTrendEvidence,
    correlation_evidence: ModalEnvelopeCorrelationEvidence,
    delayed_growth_evidence: tuple[ModalDelayedGrowthEvidence, ModalDelayedGrowthEvidence],
    recovery_evidence: tuple[ModalAmplitudeRecoveryEvidence, ModalAmplitudeRecoveryEvidence],
    pair_energy_evidence: ModalPairEnergyEvidence,
    alternating_dominance_evidence: ModalAlternatingDominanceEvidence,
    beating_evidence: ModalBeatingEvidence,
    settings: ModalEnergyExchangeSettings | None = None,
    *,
    tracking_quality_sufficient: bool = True,
) -> ModalEnergyExchangeScore:
    """Compute an auditable bounded score from explicit evidence components."""
    cfg = settings or ModalEnergyExchangeSettings()
    values = {
        "opposed_trend": 1.0 if trend_evidence.opposed_trends else 0.0,
        "negative_correlation": (
            1.0 if correlation_evidence.significant_negative_correlation else 0.0
        ),
        "lagged_correlation": (
            1.0
            if ModalEnergyExchangeReason.LAGGED_NEGATIVE_CORRELATION
            in correlation_evidence.reasons
            else 0.0
        ),
        "delayed_growth": (
            1.0 if any(item.supported for item in delayed_growth_evidence) else 0.0
        ),
        "recovery": 1.0 if any(item.supported for item in recovery_evidence) else 0.0,
        "alternating_dominance": (
            1.0 if alternating_dominance_evidence.alternating_dominance else 0.0
        ),
        "pair_energy": (
            1.0 if pair_energy_evidence.approximately_conserved else 0.0
        ),
        "tracking_quality": 1.0 if tracking_quality_sufficient else 0.0,
    }
    weights = {
        "opposed_trend": cfg.opposed_trend_score_weight,
        "negative_correlation": cfg.negative_correlation_score_weight,
        "lagged_correlation": cfg.lagged_correlation_score_weight,
        "delayed_growth": cfg.delayed_growth_score_weight,
        "recovery": cfg.recovery_score_weight,
        "alternating_dominance": cfg.alternating_dominance_score_weight,
        "pair_energy": cfg.pair_energy_score_weight,
        "tracking_quality": cfg.tracking_quality_score_weight,
    }
    components = tuple(
        (name, values[name], weights[name], values[name] * weights[name])
        for name in sorted(values)
    )
    positive_weight = sum(weights.values())
    missing_penalty = 0.0
    if correlation_evidence.zero_lag_correlation is None:
        missing_penalty += cfg.missing_evidence_penalty
    if not any(item.supported for item in delayed_growth_evidence):
        missing_penalty += cfg.missing_evidence_penalty
    beating_penalty = (
        cfg.beating_reservation_penalty if beating_evidence.possible_beating else 0.0
    )
    raw = sum(item[3] or 0.0 for item in components) - missing_penalty - beating_penalty
    normalized = _clamp(raw / positive_weight if positive_weight > 0 else 0.0, 0.0, 1.0)
    return ModalEnergyExchangeScore(
        values["opposed_trend"],
        values["negative_correlation"],
        values["lagged_correlation"],
        values["delayed_growth"],
        values["recovery"],
        values["alternating_dominance"],
        values["pair_energy"],
        values["tracking_quality"],
        beating_penalty,
        missing_penalty,
        raw,
        normalized,
        normalized >= cfg.minimum_support_score,
        normalized >= cfg.minimum_reservation_score,
        components,
        (),
    )


def evaluate_modal_energy_exchange_pair(
    source_a: object,
    source_b: object,
    settings: ModalEnergyExchangeSettings | None = None,
    *,
    parameter_estimate_a: ModalParameterEstimate | None = None,
    parameter_estimate_b: ModalParameterEstimate | None = None,
    hypothesis_a: ModalHypothesis | None = None,
    hypothesis_b: ModalHypothesis | None = None,
    frequency_a_hz: float | None = None,
    frequency_b_hz: float | None = None,
    dynamic_label: str | None = None,
) -> ModalEnergyExchangeEvidence:
    """Evaluate one canonical source pair without mutating any input object."""
    cfg = settings or ModalEnergyExchangeSettings()
    prepared_a = prepare_modal_envelope_series(
        source_a,
        cfg,
        hypothesis_id=_hypothesis_id(hypothesis_a, parameter_estimate_a),
        dynamic_label=dynamic_label,
        representative_frequency_hz=frequency_a_hz or _parameter_frequency(parameter_estimate_a),
    )
    prepared_b = prepare_modal_envelope_series(
        source_b,
        cfg,
        hypothesis_id=_hypothesis_id(hypothesis_b, parameter_estimate_b),
        dynamic_label=dynamic_label,
        representative_frequency_hz=frequency_b_hz or _parameter_frequency(parameter_estimate_b),
    )
    if prepared_b.source_id < prepared_a.source_id:
        prepared_a, prepared_b = prepared_b, prepared_a
        parameter_estimate_a, parameter_estimate_b = parameter_estimate_b, parameter_estimate_a
        hypothesis_a, hypothesis_b = hypothesis_b, hypothesis_a
        frequency_a_hz, frequency_b_hz = frequency_b_hz, frequency_a_hz
    source_status_reasons, source_reservations = _source_status_reasons(
        parameter_estimate_a,
        parameter_estimate_b,
        hypothesis_a,
        hypothesis_b,
        cfg,
    )
    frequency_a = frequency_a_hz or _parameter_frequency(parameter_estimate_a)
    frequency_b = frequency_b_hz or _parameter_frequency(parameter_estimate_b)
    alignment = align_modal_envelope_series(prepared_a, prepared_b, cfg)
    trend = evaluate_modal_envelope_trends(alignment, cfg)
    delayed = (
        evaluate_modal_delayed_growth(alignment, cfg, source="a"),
        evaluate_modal_delayed_growth(alignment, cfg, source="b"),
    )
    recovery = (
        evaluate_modal_amplitude_recovery(alignment, cfg, source="a"),
        evaluate_modal_amplitude_recovery(alignment, cfg, source="b"),
    )
    trend = ModalEnvelopeTrendEvidence(
        trend.slope_a,
        trend.slope_b,
        trend.normalized_slope_a,
        trend.normalized_slope_b,
        trend.trend_a,
        trend.trend_b,
        trend.opposed_trends,
        trend.trend_overlap_fraction,
        trend.change_point_times_a,
        trend.change_point_times_b,
        delayed[0].supported,
        delayed[1].supported,
        recovery[0].supported,
        recovery[1].supported,
        trend.passes,
        trend.reasons,
        trend.diagnostics,
    )
    correlation = evaluate_modal_envelope_correlation(alignment, cfg)
    pair_energy = evaluate_modal_pair_energy(alignment, cfg)
    dominance = evaluate_modal_alternating_dominance(alignment, cfg)
    beating = evaluate_modal_beating_context(frequency_a, frequency_b, alignment, cfg)
    reservations = list(source_reservations)
    if alignment.resampling_applied:
        reservations.extend(
            (
                ModalEnergyExchangeReason.INTERPOLATION_REQUIRED,
                ModalEnergyExchangeReason.UNEQUAL_SAMPLING,
            )
        )
    if beating.possible_beating:
        reservations.append(ModalEnergyExchangeReason.POSSIBLE_BEATING)
    if _diagnostics_contain(prepared_a, prepared_b, "background"):
        reservations.append(ModalEnergyExchangeReason.BACKGROUND_CONTAMINATION)
    if _diagnostics_contain(prepared_a, prepared_b, "clipping"):
        reservations.append(ModalEnergyExchangeReason.CLIPPING_CONTEXT)
    if cfg.reserve_frequency_crossing and _diagnostics_contain(
        prepared_a,
        prepared_b,
        "frequency_crossing",
    ):
        reservations.append(ModalEnergyExchangeReason.POSSIBLE_FREQUENCY_CROSSING)
    if cfg.reserve_peak_overlap and _diagnostics_contain(
        prepared_a,
        prepared_b,
        "peak_overlap",
    ):
        reservations.append(ModalEnergyExchangeReason.POSSIBLE_PEAK_OVERLAP)
    tracking_quality_sufficient = not any(
        reason
        in {
            ModalEnergyExchangeReason.AMBIGUOUS_TRACKING,
            ModalEnergyExchangeReason.NEAR_THRESHOLD_TRACKING,
        }
        for reason in reservations
    )
    score = compute_modal_energy_exchange_score(
        trend,
        correlation,
        delayed,
        recovery,
        pair_energy,
        dominance,
        beating,
        cfg,
        tracking_quality_sufficient=tracking_quality_sufficient,
    )
    supporting = _supporting_reasons(
        alignment,
        trend,
        correlation,
        delayed,
        recovery,
        pair_energy,
        dominance,
        tracking_quality_sufficient,
    )
    inconclusive = _inconclusive_reasons(trend, correlation, pair_energy)
    not_supported = _not_supported_reasons(trend, delayed, recovery, pair_energy)
    insufficient = list(source_status_reasons)
    invalid: list[ModalEnergyExchangeReason] = []
    if not prepared_a.valid or not prepared_b.valid:
        source_reasons = set(prepared_a.reasons + prepared_b.reasons)
        if source_reasons & {
            ModalEnergyExchangeReason.INVALID_TIME_VALUES,
            ModalEnergyExchangeReason.INVALID_AMPLITUDE_VALUES,
        }:
            invalid.extend(source_reasons)
        else:
            insufficient.extend(source_reasons)
    if not alignment.valid:
        alignment_reasons = set(alignment.reasons)
        if alignment_reasons & {ModalEnergyExchangeReason.INCOMPATIBLE_TIME_AXES}:
            invalid.extend(alignment_reasons)
        else:
            insufficient.extend(alignment_reasons)
    gate_failed = _required_gate_failed(cfg, trend, correlation, delayed, pair_energy)
    if gate_failed is not None:
        not_supported.append(gate_failed)
    status = _decide_energy_exchange_status(
        invalid,
        insufficient,
        gate_failed,
        score,
        supporting,
        reservations,
        inconclusive,
        not_supported,
        cfg,
    )
    provenance = _energy_exchange_provenance(
        prepared_a,
        prepared_b,
        alignment,
        parameter_estimate_a,
        parameter_estimate_b,
        cfg,
    )
    evidence_id = _energy_exchange_id(
        prepared_a,
        prepared_b,
        parameter_estimate_a,
        parameter_estimate_b,
        cfg,
    )
    return ModalEnergyExchangeEvidence(
        evidence_id,
        prepared_a.source_id,
        prepared_b.source_id,
        prepared_a.hypothesis_id,
        prepared_b.hypothesis_id,
        dynamic_label or prepared_a.dynamic_label or prepared_b.dynamic_label,
        status,
        alignment,
        trend,
        delayed,
        recovery,
        correlation,
        pair_energy,
        dominance,
        beating,
        score,
        _ordered_reasons(supporting),
        _ordered_reasons(reservations),
        _ordered_reasons(inconclusive),
        _ordered_reasons(not_supported),
        _ordered_reasons(insufficient),
        _ordered_reasons(invalid),
        status
        in {
            ModalEnergyExchangeStatus.SUPPORTED,
            ModalEnergyExchangeStatus.SUPPORTED_WITH_RESERVATIONS,
        },
        status
        in {
            ModalEnergyExchangeStatus.SUPPORTED_WITH_RESERVATIONS,
            ModalEnergyExchangeStatus.INCONCLUSIVE,
        },
        provenance,
        _ordered_texts(
            (
                "operational_energy_exchange_evidence_only",
                "no_physical_transfer_or_causality_inferred",
                "lag_positive_means_a_precedes_b_descriptively",
            )
        ),
    )


def evaluate_modal_energy_exchange(
    sources: Iterable[object],
    settings: ModalEnergyExchangeSettings | None = None,
    *,
    parameter_estimates: Mapping[str, ModalParameterEstimate] | None = None,
    pairs: Iterable[tuple[object, object]] | None = None,
    dynamic_label: str | None = None,
) -> ModalEnergyExchangeResult:
    """Evaluate all canonical unordered pairs, or an explicit pair set."""
    cfg = settings or ModalEnergyExchangeSettings()
    source_tuple = tuple(sources)
    estimate_by_source = dict(parameter_estimates or {})
    if pairs is None:
        prepared = tuple(
            prepare_modal_envelope_series(source, cfg, dynamic_label=dynamic_label)
            for source in source_tuple
        )
        ordered = tuple(sorted(prepared, key=lambda item: item.source_id))
        pair_inputs = tuple(
            (ordered[left], ordered[right])
            for left in range(len(ordered))
            for right in range(left + 1, len(ordered))
        )
    else:
        pair_inputs = tuple(pairs)
    evidences = tuple(
        evaluate_modal_energy_exchange_pair(
            left,
            right,
            cfg,
            parameter_estimate_a=estimate_by_source.get(_source_identifier(left)),
            parameter_estimate_b=estimate_by_source.get(_source_identifier(right)),
            dynamic_label=dynamic_label,
        )
        for left, right in pair_inputs
    )
    evidences = tuple(sorted(evidences, key=lambda item: (item.source_a_id, item.source_b_id)))
    return _energy_exchange_result(
        evidences,
        len(source_tuple),
        cfg,
        dynamic_label,
        (),
    )


def summarize_modal_energy_exchange(
    result: ModalEnergyExchangeResult,
) -> dict[str, int | bool | str | None]:
    """Return compact deterministic summary counters for reporting."""
    return {
        "dynamic_label": result.dynamic_label,
        "pair_count": result.pair_count,
        "source_count": result.source_count,
        "supported_count": result.supported_count,
        "supported_with_reservations_count": result.supported_with_reservations_count,
        "inconclusive_count": result.inconclusive_count,
        "not_supported_count": result.not_supported_count,
        "insufficient_evidence_count": result.insufficient_evidence_count,
        "invalid_count": result.invalid_count,
        "valid": result.valid,
    }


def _energy_exchange_result(
    evidences: tuple[ModalEnergyExchangeEvidence, ...],
    source_count: int,
    settings: ModalEnergyExchangeSettings,
    dynamic_label: str | None,
    diagnostics: tuple[str, ...],
) -> ModalEnergyExchangeResult:
    supported = tuple(
        item for item in evidences if item.status is ModalEnergyExchangeStatus.SUPPORTED
    )
    supported_res = tuple(
        item
        for item in evidences
        if item.status is ModalEnergyExchangeStatus.SUPPORTED_WITH_RESERVATIONS
    )
    inconclusive = tuple(
        item for item in evidences if item.status is ModalEnergyExchangeStatus.INCONCLUSIVE
    )
    not_supported = tuple(
        item for item in evidences if item.status is ModalEnergyExchangeStatus.NOT_SUPPORTED
    )
    insufficient = tuple(
        item
        for item in evidences
        if item.status is ModalEnergyExchangeStatus.INSUFFICIENT_EVIDENCE
    )
    invalid = tuple(
        item for item in evidences if item.status is ModalEnergyExchangeStatus.INVALID_INPUT
    )
    return ModalEnergyExchangeResult(
        dynamic_label,
        evidences,
        supported,
        supported_res,
        inconclusive,
        not_supported,
        insufficient,
        invalid,
        len(evidences),
        len(supported),
        len(supported_res),
        len(inconclusive),
        len(not_supported),
        len(insufficient),
        len(invalid),
        source_count,
        settings,
        True,
        None,
        diagnostics,
    )


def _decide_energy_exchange_status(
    invalid: Sequence[ModalEnergyExchangeReason],
    insufficient: Sequence[ModalEnergyExchangeReason],
    gate_failed: ModalEnergyExchangeReason | None,
    score: ModalEnergyExchangeScore,
    supporting: Sequence[ModalEnergyExchangeReason],
    reservations: Sequence[ModalEnergyExchangeReason],
    inconclusive: Sequence[ModalEnergyExchangeReason],
    not_supported: Sequence[ModalEnergyExchangeReason],
    settings: ModalEnergyExchangeSettings,
) -> ModalEnergyExchangeStatus:
    if invalid:
        return ModalEnergyExchangeStatus.INVALID_INPUT
    if insufficient:
        return ModalEnergyExchangeStatus.INSUFFICIENT_EVIDENCE
    if gate_failed is not None:
        if gate_failed is ModalEnergyExchangeReason.CORRELATION_NOT_SIGNIFICANT:
            return ModalEnergyExchangeStatus.INCONCLUSIVE
        return ModalEnergyExchangeStatus.NOT_SUPPORTED
    if not supporting:
        return ModalEnergyExchangeStatus.INSUFFICIENT_EVIDENCE
    if (
        ModalEnergyExchangeReason.CORRELATION_NOT_SIGNIFICANT in inconclusive
        and score.normalized_score >= settings.minimum_reservation_score
    ):
        return ModalEnergyExchangeStatus.INCONCLUSIVE
    if (
        ModalEnergyExchangeReason.SAME_DIRECTION_TRENDS in not_supported
        and not any(
            reason
            in supporting
            for reason in (
                ModalEnergyExchangeReason.DELAYED_GROWTH,
                ModalEnergyExchangeReason.LATE_AMPLITUDE_RECOVERY,
                ModalEnergyExchangeReason.ALTERNATING_DOMINANCE,
            )
        )
    ):
        return ModalEnergyExchangeStatus.NOT_SUPPORTED
    if score.passes_support_threshold:
        return (
            ModalEnergyExchangeStatus.SUPPORTED_WITH_RESERVATIONS
            if reservations
            else ModalEnergyExchangeStatus.SUPPORTED
        )
    if score.passes_reservation_threshold and supporting:
        return ModalEnergyExchangeStatus.INCONCLUSIVE
    return ModalEnergyExchangeStatus.NOT_SUPPORTED


def _required_gate_failed(
    settings: ModalEnergyExchangeSettings,
    trend: ModalEnvelopeTrendEvidence,
    correlation: ModalEnvelopeCorrelationEvidence,
    delayed: tuple[ModalDelayedGrowthEvidence, ModalDelayedGrowthEvidence],
    pair_energy: ModalPairEnergyEvidence,
) -> ModalEnergyExchangeReason | None:
    if settings.require_opposed_trends and not trend.opposed_trends:
        return ModalEnergyExchangeReason.SAME_DIRECTION_TRENDS
    if settings.require_negative_correlation and not correlation.passes:
        return ModalEnergyExchangeReason.CORRELATION_NOT_SIGNIFICANT
    if settings.require_delayed_response and not any(item.supported for item in delayed):
        return ModalEnergyExchangeReason.NO_DELAYED_GROWTH
    if settings.require_pair_energy_stability and not pair_energy.approximately_conserved:
        return ModalEnergyExchangeReason.PAIR_ENERGY_NOT_CONSERVED
    return None


def _supporting_reasons(
    alignment: ModalEnvelopeAlignment,
    trend: ModalEnvelopeTrendEvidence,
    correlation: ModalEnvelopeCorrelationEvidence,
    delayed: tuple[ModalDelayedGrowthEvidence, ModalDelayedGrowthEvidence],
    recovery: tuple[ModalAmplitudeRecoveryEvidence, ModalAmplitudeRecoveryEvidence],
    pair_energy: ModalPairEnergyEvidence,
    dominance: ModalAlternatingDominanceEvidence,
    tracking_quality_sufficient: bool,
) -> list[ModalEnergyExchangeReason]:
    reasons: list[ModalEnergyExchangeReason] = []
    if alignment.valid:
        reasons.append(ModalEnergyExchangeReason.TEMPORAL_OVERLAP_SUFFICIENT)
    if trend.opposed_trends:
        reasons.append(ModalEnergyExchangeReason.OPPOSED_ENVELOPE_TRENDS)
    if correlation.significant_negative_correlation:
        reasons.append(ModalEnergyExchangeReason.SIGNIFICANT_NEGATIVE_CORRELATION)
    if ModalEnergyExchangeReason.LAGGED_NEGATIVE_CORRELATION in correlation.reasons:
        reasons.append(ModalEnergyExchangeReason.LAGGED_NEGATIVE_CORRELATION)
    if any(item.supported for item in delayed):
        reasons.append(ModalEnergyExchangeReason.DELAYED_GROWTH)
    if any(item.supported for item in recovery):
        reasons.append(ModalEnergyExchangeReason.LATE_AMPLITUDE_RECOVERY)
    if pair_energy.approximately_conserved:
        reasons.append(ModalEnergyExchangeReason.APPROXIMATELY_CONSERVED_PAIR_ENERGY)
    if dominance.alternating_dominance:
        reasons.append(ModalEnergyExchangeReason.ALTERNATING_DOMINANCE)
    if tracking_quality_sufficient:
        reasons.append(ModalEnergyExchangeReason.TRACKING_QUALITY_SUFFICIENT)
    return reasons


def _inconclusive_reasons(
    trend: ModalEnvelopeTrendEvidence,
    correlation: ModalEnvelopeCorrelationEvidence,
    pair_energy: ModalPairEnergyEvidence,
) -> list[ModalEnergyExchangeReason]:
    reasons: list[ModalEnergyExchangeReason] = []
    if ModalEnergyExchangeReason.CORRELATION_NOT_SIGNIFICANT in correlation.reasons:
        reasons.append(ModalEnergyExchangeReason.CORRELATION_NOT_SIGNIFICANT)
    if ModalEnergyExchangeReason.INSUFFICIENT_DYNAMIC_RANGE in trend.reasons:
        reasons.append(ModalEnergyExchangeReason.INSUFFICIENT_DYNAMIC_RANGE)
    if ModalEnergyExchangeReason.PAIR_ENERGY_NOT_CONSERVED in pair_energy.reasons:
        reasons.append(ModalEnergyExchangeReason.PAIR_ENERGY_NOT_CONSERVED)
    return reasons


def _not_supported_reasons(
    trend: ModalEnvelopeTrendEvidence,
    delayed: tuple[ModalDelayedGrowthEvidence, ModalDelayedGrowthEvidence],
    recovery: tuple[ModalAmplitudeRecoveryEvidence, ModalAmplitudeRecoveryEvidence],
    pair_energy: ModalPairEnergyEvidence,
) -> list[ModalEnergyExchangeReason]:
    reasons: list[ModalEnergyExchangeReason] = []
    if ModalEnergyExchangeReason.SAME_DIRECTION_TRENDS in trend.reasons:
        reasons.append(ModalEnergyExchangeReason.SAME_DIRECTION_TRENDS)
        if not any(item.supported for item in delayed) and not any(
            item.supported for item in recovery
        ):
            reasons.append(ModalEnergyExchangeReason.DECAY_ONLY_BEHAVIOR)
    if not any(item.supported for item in delayed):
        reasons.append(ModalEnergyExchangeReason.NO_DELAYED_GROWTH)
    if ModalEnergyExchangeReason.PAIR_ENERGY_NOT_CONSERVED in pair_energy.reasons:
        reasons.append(ModalEnergyExchangeReason.PAIR_ENERGY_NOT_CONSERVED)
    return reasons


def _source_status_reasons(
    parameter_a: ModalParameterEstimate | None,
    parameter_b: ModalParameterEstimate | None,
    hypothesis_a: ModalHypothesis | None,
    hypothesis_b: ModalHypothesis | None,
    settings: ModalEnergyExchangeSettings,
) -> tuple[list[ModalEnergyExchangeReason], list[ModalEnergyExchangeReason]]:
    insufficient: list[ModalEnergyExchangeReason] = []
    reservations: list[ModalEnergyExchangeReason] = []
    for parameter in (parameter_a, parameter_b):
        if parameter is None:
            continue
        status = parameter.status
        if status is ModalParameterEstimateStatus.VALID:
            continue
        if status is ModalParameterEstimateStatus.VALID_WITH_RESERVATIONS:
            reservations.append(ModalEnergyExchangeReason.AMBIGUOUS_TRACKING)
            continue
        if status is ModalParameterEstimateStatus.PARTIAL and settings.allow_partial_parameter_estimates:
            reservations.append(ModalEnergyExchangeReason.NEAR_THRESHOLD_TRACKING)
            continue
        if (
            status is ModalParameterEstimateStatus.INSUFFICIENT_EVIDENCE
            and settings.allow_inconclusive_sources_for_audit
        ):
            reservations.append(ModalEnergyExchangeReason.UNSUPPORTED_SOURCE_STATUS)
            continue
        insufficient.append(ModalEnergyExchangeReason.UNSUPPORTED_SOURCE_STATUS)
    for hypothesis in (hypothesis_a, hypothesis_b):
        if hypothesis is None:
            continue
        status = hypothesis.status
        allowed = (
            status is ModalHypothesisStatus.ACCEPTED
            and settings.allow_accepted_hypotheses
        ) or (
            status is ModalHypothesisStatus.ACCEPTED_WITH_RESERVATIONS
            and settings.allow_accepted_with_reservations
        ) or (
            status is ModalHypothesisStatus.INCONCLUSIVE
            and settings.allow_inconclusive_sources_for_audit
        )
        if not allowed:
            insufficient.append(ModalEnergyExchangeReason.UNSUPPORTED_SOURCE_STATUS)
        if status is ModalHypothesisStatus.ACCEPTED_WITH_RESERVATIONS:
            reservations.append(ModalEnergyExchangeReason.AMBIGUOUS_TRACKING)
    for parameter in (parameter_a, parameter_b):
        if parameter is None:
            continue
        provenance = parameter.provenance
        if provenance.ambiguous_match_ids:
            reservations.append(ModalEnergyExchangeReason.AMBIGUOUS_TRACKING)
        if provenance.near_threshold_match_ids:
            reservations.append(ModalEnergyExchangeReason.NEAR_THRESHOLD_TRACKING)
        if provenance.possible_split_context_ids:
            reservations.append(ModalEnergyExchangeReason.POSSIBLE_SPLIT_CONTEXT)
        if provenance.possible_merge_context_ids:
            reservations.append(ModalEnergyExchangeReason.POSSIBLE_MERGE_CONTEXT)
    return insufficient, reservations


def _energy_exchange_provenance(
    source_a: ModalEnvelopeSeries,
    source_b: ModalEnvelopeSeries,
    alignment: ModalEnvelopeAlignment,
    parameter_a: ModalParameterEstimate | None,
    parameter_b: ModalParameterEstimate | None,
    settings: ModalEnergyExchangeSettings,
) -> ModalEnergyExchangeProvenance:
    hypothesis_ids = _ordered_texts(
        value
        for value in (
            source_a.hypothesis_id,
            source_b.hypothesis_id,
            parameter_a.hypothesis_id if parameter_a is not None else None,
            parameter_b.hypothesis_id if parameter_b is not None else None,
        )
        if value is not None
    )
    candidate_ids = _ordered_texts(
        str(value)
        for value in (
            source_a.candidate_id,
            source_b.candidate_id,
        )
        if value is not None
    )
    track_ids = _ordered_texts(
        str(value)
        for value in (
            source_a.track_id,
            source_b.track_id,
        )
        if value is not None
    )
    recording_id = (
        source_a.recording_id
        if source_a.recording_id == source_b.recording_id
        else None
    )
    dynamic_label = (
        source_a.dynamic_label
        if source_a.dynamic_label == source_b.dynamic_label
        else source_a.dynamic_label or source_b.dynamic_label
    )
    return ModalEnergyExchangeProvenance(
        source_a.source_id,
        source_b.source_id,
        hypothesis_ids,
        candidate_ids,
        track_ids,
        recording_id,
        dynamic_label,
        (alignment.overlap_start_s, alignment.overlap_end_s),
        modal_energy_exchange_settings_fingerprint(settings),
        (source_a.sample_count, source_b.sample_count),
        alignment.sample_count,
        _ordered_texts(
            (
                "provenance_to_precomputed_envelopes",
                "settings_fingerprint_deterministic",
            )
        ),
    )


def _energy_exchange_id(
    source_a: ModalEnvelopeSeries,
    source_b: ModalEnvelopeSeries,
    parameter_a: ModalParameterEstimate | None,
    parameter_b: ModalParameterEstimate | None,
    settings: ModalEnergyExchangeSettings,
) -> str:
    payload = {
        "source_a": _series_identity(source_a),
        "source_b": _series_identity(source_b),
        "parameter_a": parameter_a.estimate_id if parameter_a is not None else None,
        "parameter_b": parameter_b.estimate_id if parameter_b is not None else None,
        "settings": modal_energy_exchange_settings_fingerprint(settings),
    }
    digest = hashlib.sha256(
        json.dumps(_canonicalize(payload), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return f"modal-energy-exchange-{digest[:20]}"


def _series_identity(series: ModalEnvelopeSeries) -> dict[str, object]:
    return {
        "source_id": series.source_id,
        "hypothesis_id": series.hypothesis_id,
        "candidate_id": series.candidate_id,
        "track_id": series.track_id,
        "recording_id": series.recording_id,
        "times_s": series.times_s,
        "amplitudes": series.amplitudes,
        "energy_proxy": series.energy_proxy,
    }


def _single_series(
    series: ModalEnvelopeSeries | ModalEnvelopeAlignment,
    source: str,
) -> tuple[str, tuple[float, ...], tuple[float, ...]]:
    if isinstance(series, ModalEnvelopeSeries):
        return series.source_id, series.times_s, series.normalized_amplitudes
    if source == "a":
        return series.source_a_id, series.common_times_s, series.aligned_amplitudes_a
    if source == "b":
        return series.source_b_id, series.common_times_s, series.aligned_amplitudes_b
    raise ValueError("source must be 'a' or 'b'.")


def _invalid_series(
    source_id: str | None,
    hypothesis_id: str | None,
    candidate_id: str | int | None,
    track_id: str | int | None,
    recording_id: str | None,
    dynamic_label: str | None,
    reasons: Sequence[ModalEnergyExchangeReason],
    diagnostics: Sequence[str],
    settings: ModalEnergyExchangeSettings,
) -> ModalEnvelopeSeries:
    return ModalEnvelopeSeries(
        source_id=source_id or "missing-envelope-source",
        hypothesis_id=hypothesis_id,
        candidate_id=candidate_id,
        track_id=track_id,
        recording_id=recording_id,
        dynamic_label=dynamic_label,
        times_s=(),
        amplitudes=(),
        normalized_amplitudes=(),
        energy_proxy=(),
        valid_mask=(),
        sample_count=0,
        time_start_s=None,
        time_end_s=None,
        duration_s=None,
        sampling_step_s=None,
        interpolated=False,
        smoothed=False,
        normalization_method=(
            settings.normalization_method
            if settings.normalize_envelopes
            else ModalEnvelopeNormalizationMethod.NONE
        ),
        valid=False,
        reasons=_ordered_reasons(reasons),
        diagnostics=_ordered_texts(diagnostics),
    )


def _invalid_alignment(
    source_a_id: str,
    source_b_id: str,
    reasons: Sequence[ModalEnergyExchangeReason],
    diagnostics: Sequence[str],
) -> ModalEnvelopeAlignment:
    return ModalEnvelopeAlignment(
        source_a_id,
        source_b_id,
        (),
        (),
        (),
        (),
        (),
        None,
        None,
        None,
        0,
        False,
        None,
        False,
        _ordered_reasons(reasons),
        _ordered_texts(diagnostics),
    )


def _empty_correlation(
    method: ModalEnvelopeCorrelationMethod,
    reasons: Sequence[ModalEnergyExchangeReason],
    diagnostic: str,
) -> ModalEnvelopeCorrelationEvidence:
    return ModalEnvelopeCorrelationEvidence(
        method,
        None,
        None,
        (),
        (),
        None,
        None,
        None,
        False,
        0,
        False,
        _ordered_reasons(reasons),
        (diagnostic,),
    )


def _apply_analysis_window(
    times: tuple[float, ...],
    amplitudes: tuple[float, ...],
    settings: ModalEnergyExchangeSettings,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    start = settings.analysis_window_start_s if settings.analysis_window_start_s is not None else times[0]
    if settings.exclude_attack_duration_s:
        start = max(start, times[0] + settings.exclude_attack_duration_s)
    end = settings.analysis_window_end_s if settings.analysis_window_end_s is not None else times[-1]
    selected = tuple(
        (time, amplitude)
        for time, amplitude in zip(times, amplitudes, strict=True)
        if start <= time <= end
    )
    if not selected:
        return (), ()
    return tuple(item[0] for item in selected), tuple(item[1] for item in selected)


def _smooth_moving_average(
    times: tuple[float, ...],
    amplitudes: tuple[float, ...],
    window_s: float | None,
) -> tuple[float, ...]:
    if window_s is None or window_s <= 0:
        return amplitudes
    half = window_s / 2.0
    smoothed: list[float] = []
    for center in times:
        values = tuple(
            value
            for time, value in zip(times, amplitudes, strict=True)
            if abs(time - center) <= half
        )
        smoothed.append(sum(values) / len(values))
    return tuple(smoothed)


def _normalize_amplitudes(
    amplitudes: tuple[float, ...],
    settings: ModalEnergyExchangeSettings,
) -> tuple[tuple[float, ...], ModalEnvelopeNormalizationMethod]:
    if not settings.normalize_envelopes:
        return amplitudes, ModalEnvelopeNormalizationMethod.NONE
    method = settings.normalization_method
    if method is ModalEnvelopeNormalizationMethod.NONE:
        return amplitudes, method
    reference: float | None
    if method is ModalEnvelopeNormalizationMethod.PEAK:
        reference = max(abs(value) for value in amplitudes)
    elif method is ModalEnvelopeNormalizationMethod.INITIAL:
        reference = abs(amplitudes[0])
    else:
        reference = _mean(tuple(abs(value) for value in amplitudes))
    if reference is None or reference <= settings.minimum_positive_amplitude:
        return amplitudes, ModalEnvelopeNormalizationMethod.NONE
    return tuple(value / reference for value in amplitudes), method


def _linear_amplitude_values(
    amplitudes: tuple[float, ...],
    amplitude_unit: str | None,
    settings: ModalEnergyExchangeSettings,
) -> tuple[float, ...]:
    if settings.amplitude_representation is ModalAmplitudeRepresentation.AMPLITUDE_DB or (
        amplitude_unit is not None and "db" in amplitude_unit.lower()
    ):
        return tuple(10.0 ** (value / 20.0) for value in amplitudes)
    if settings.amplitude_representation is ModalAmplitudeRepresentation.RELATIVE_POWER:
        return tuple(sqrt(max(value, 0.0)) for value in amplitudes)
    if settings.amplitude_representation is ModalAmplitudeRepresentation.OPERATIONAL_ENERGY:
        return tuple(sqrt(max(value, 0.0)) for value in amplitudes)
    return amplitudes


def _uses_linear_nonnegative_amplitude(
    settings: ModalEnergyExchangeSettings,
    amplitude_unit: str | None,
) -> bool:
    if settings.amplitude_representation is ModalAmplitudeRepresentation.AMPLITUDE_DB:
        return False
    if amplitude_unit is not None and "db" in amplitude_unit.lower():
        return False
    return True


def _slice_overlap(
    series: ModalEnvelopeSeries,
    start: float,
    end: float,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    selected = tuple(
        (time, amp, energy)
        for time, amp, energy in zip(
            series.times_s,
            series.normalized_amplitudes,
            series.energy_proxy,
            strict=True,
        )
        if start <= time <= end
    )
    return (
        tuple(item[0] for item in selected),
        tuple(item[1] for item in selected),
        tuple(item[2] for item in selected),
    )


def _axes_compatible(
    a: tuple[float, ...],
    b: tuple[float, ...],
    mismatch_fraction: float,
) -> bool:
    if len(a) != len(b):
        return False
    if a == b:
        return True
    step = _representative_step(a)
    if step is None or step <= 0:
        return False
    return all(abs(left - right) <= mismatch_fraction * step for left, right in zip(a, b, strict=True))


def _intersect_axes(
    times_a: tuple[float, ...],
    amps_a: tuple[float, ...],
    energy_a: tuple[float, ...],
    times_b: tuple[float, ...],
    amps_b: tuple[float, ...],
    energy_b: tuple[float, ...],
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    index_b = {round(time, 12): index for index, time in enumerate(times_b)}
    rows = tuple(
        (
            time,
            amps_a[index_a],
            amps_b[index_b[round(time, 12)]],
            energy_a[index_a],
            energy_b[index_b[round(time, 12)]],
        )
        for index_a, time in enumerate(times_a)
        if round(time, 12) in index_b
    )
    return (
        tuple(row[0] for row in rows),
        tuple(row[1] for row in rows),
        tuple(row[2] for row in rows),
        tuple(row[3] for row in rows),
        tuple(row[4] for row in rows),
    )


def _resampling_grid(start: float, end: float, step: float) -> tuple[float, ...]:
    values: list[float] = []
    current = start
    while current <= end + (step * 1e-9):
        values.append(round(current, 12))
        current += step
    if values and values[-1] > end:
        values[-1] = end
    elif values and abs(values[-1] - end) > step * 1e-6:
        values.append(end)
    return tuple(values)


def _interpolate(
    times: tuple[float, ...],
    values: tuple[float, ...],
    target: float,
) -> float:
    if target < times[0] - 1e-12 or target > times[-1] + 1e-12:
        raise ValueError("interpolation target outside source range.")
    if target <= times[0]:
        return values[0]
    if target >= times[-1]:
        return values[-1]
    for index in range(1, len(times)):
        if times[index] >= target:
            left_t, right_t = times[index - 1], times[index]
            left_v, right_v = values[index - 1], values[index]
            fraction = (target - left_t) / (right_t - left_t)
            return left_v + fraction * (right_v - left_v)
    return values[-1]


def _trend_slope(
    times: tuple[float, ...],
    values: tuple[float, ...],
    method: ModalEnvelopeTrendMethod,
) -> float | None:
    if len(times) < 2:
        return None
    if method is ModalEnvelopeTrendMethod.LINEAR_REGRESSION:
        return _linear_slope(times, values)
    if method is ModalEnvelopeTrendMethod.START_END:
        return (values[-1] - values[0]) / (times[-1] - times[0])
    if method is ModalEnvelopeTrendMethod.SEGMENT_DIFFERENCE:
        midpoint = max(1, len(values) // 2)
        before = _mean(values[:midpoint])
        after = _mean(values[midpoint:])
        if before is None or after is None:
            return None
        return (after - before) / (times[-1] - times[0])
    derivatives = _derivatives(times, values)
    return _median(derivatives)


def _linear_slope(times: tuple[float, ...], values: tuple[float, ...]) -> float | None:
    x_mean = _mean(times)
    y_mean = _mean(values)
    if x_mean is None or y_mean is None:
        return None
    denominator = sum((x - x_mean) ** 2 for x in times)
    if denominator <= 0:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(times, values, strict=True)) / denominator


def _classify_trend(
    slope: float | None,
    settings: ModalEnergyExchangeSettings,
) -> str:
    if slope is None:
        return "indeterminate"
    if slope <= -settings.minimum_negative_slope:
        return "decreasing"
    if slope >= settings.minimum_positive_slope:
        return "increasing"
    return "stable"


def _normalized_slope(slope: float | None, values: tuple[float, ...]) -> float | None:
    if slope is None:
        return None
    scale = _mean(tuple(abs(value) for value in values))
    if scale is None or scale <= 0:
        return None
    return slope / scale


def _opposed_derivative_fraction(
    values_a: tuple[float, ...],
    values_b: tuple[float, ...],
) -> float | None:
    if len(values_a) < 2:
        return None
    pairs = tuple(
        (a2 - a1, b2 - b1)
        for a1, a2, b1, b2 in zip(
            values_a[:-1],
            values_a[1:],
            values_b[:-1],
            values_b[1:],
            strict=True,
        )
    )
    informative = tuple((da, db) for da, db in pairs if da != 0 or db != 0)
    if not informative:
        return None
    return sum(da * db < 0 for da, db in informative) / len(informative)


def _change_points(
    times: tuple[float, ...],
    values: tuple[float, ...],
    settings: ModalEnergyExchangeSettings,
) -> tuple[float, ...]:
    derivatives = _derivatives(times, values)
    if len(derivatives) < 2:
        return ()
    signs = tuple(
        -1
        if value <= -settings.minimum_negative_slope
        else 1
        if value >= settings.minimum_positive_slope
        else 0
        for value in derivatives
    )
    points = tuple(
        times[index + 1]
        for index, (left, right) in enumerate(zip(signs, signs[1:]))
        if left and right and left != right
    )
    return points


def _derivatives(times: tuple[float, ...], values: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(
        (v2 - v1) / (t2 - t1)
        for t1, t2, v1, v2 in zip(
            times[:-1],
            times[1:],
            values[:-1],
            values[1:],
            strict=True,
        )
    )


def _sustained_after_threshold(
    times: tuple[float, ...],
    amplitudes: tuple[float, ...],
    start_index: int,
    peak_index: int,
    threshold: float,
    duration_s: float | None,
) -> bool:
    if duration_s is None or duration_s <= 0:
        return True
    first: int | None = None
    last: int | None = None
    for index in range(start_index, peak_index + 1):
        if amplitudes[index] >= threshold:
            first = index if first is None else first
            last = index
    return first is not None and last is not None and times[last] - times[first] >= duration_s


def _correlation_vectors(
    a: tuple[float, ...],
    b: tuple[float, ...],
    method: ModalEnvelopeCorrelationMethod,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if method is ModalEnvelopeCorrelationMethod.SPEARMAN:
        return _ranks(a), _ranks(b)
    return a, b


def _ranks(values: tuple[float, ...]) -> tuple[float, ...]:
    ordered = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][0] == ordered[cursor][0]:
            end += 1
        rank = (cursor + end + 1) / 2.0
        for _, index in ordered[cursor:end]:
            ranks[index] = rank
        cursor = end
    return tuple(ranks)


def _correlation(a: tuple[float, ...], b: tuple[float, ...]) -> float | None:
    if len(a) != len(b) or len(a) < 2:
        return None
    mean_a = _mean(a)
    mean_b = _mean(b)
    if mean_a is None or mean_b is None:
        return None
    centered_a = tuple(value - mean_a for value in a)
    centered_b = tuple(value - mean_b for value in b)
    denom = sqrt(sum(value * value for value in centered_a) * sum(value * value for value in centered_b))
    if denom <= 0:
        return None
    return _clamp(
        sum(x * y for x, y in zip(centered_a, centered_b, strict=True)) / denom,
        -1.0,
        1.0,
    )


def _lagged_correlations(
    times: tuple[float, ...],
    a: tuple[float, ...],
    b: tuple[float, ...],
    settings: ModalEnergyExchangeSettings,
) -> tuple[tuple[float, ...], tuple[float | None, ...]]:
    if settings.maximum_lag_s is None or settings.maximum_lag_s <= 0:
        return (), ()
    step = settings.lag_step_s or _representative_step(times)
    if step is None or step <= 0:
        return (), ()
    max_lag_steps = int(settings.maximum_lag_s / step)
    lag_values: list[float] = []
    correlations: list[float | None] = []
    for shift in range(-max_lag_steps, max_lag_steps + 1):
        if shift == 0:
            lag_values.append(0.0)
            correlations.append(_correlation(a, b))
        elif shift > 0:
            lag_values.append(shift * step)
            correlations.append(_correlation(a[:-shift], b[shift:]))
        else:
            lag_values.append(shift * step)
            correlations.append(_correlation(a[-shift:], b[:shift]))
    return tuple(lag_values), tuple(correlations)


def _operational_p_value(
    observed: float | None,
    a: tuple[float, ...],
    b: tuple[float, ...],
    settings: ModalEnergyExchangeSettings,
) -> float | None:
    if observed is None or settings.significance_method is ModalEnvelopeSignificanceMethod.DISABLED:
        return None
    if len(a) < 4:
        return None
    rng = random.Random(settings.random_seed)
    null: list[float] = []
    for _ in range(settings.permutation_count):
        if settings.significance_method is ModalEnvelopeSignificanceMethod.CIRCULAR_SHIFT:
            shift = rng.randrange(1, len(b))
            sampled = b[shift:] + b[:shift]
        else:
            sampled = _block_permutation(b, settings.significance_block_size, rng)
        corr = _correlation(a, sampled)
        if corr is not None:
            null.append(corr)
    if not null:
        return None
    return (1 + sum(value <= observed for value in null)) / (len(null) + 1)


def _block_permutation(
    values: tuple[float, ...],
    block_size: int,
    rng: random.Random,
) -> tuple[float, ...]:
    blocks = [values[index:index + block_size] for index in range(0, len(values), block_size)]
    rng.shuffle(blocks)
    return tuple(value for block in blocks for value in block)


def _observed_modulation_period(
    times: tuple[float, ...],
    values: tuple[float, ...],
) -> float | None:
    peaks = tuple(
        times[index]
        for index in range(1, len(values) - 1)
        if values[index] >= values[index - 1] and values[index] > values[index + 1]
    )
    if len(peaks) < 2:
        return None
    intervals = tuple(later - earlier for earlier, later in zip(peaks, peaks[1:]))
    return _median(intervals)


def _parameter_frequency(parameter: ModalParameterEstimate | None) -> float | None:
    if parameter is None or parameter.frequency_estimate is None:
        return None
    value = parameter.frequency_estimate.representative_frequency_hz
    return value if value is not None and _finite(value) and value > 0 else None


def _hypothesis_id(
    hypothesis: ModalHypothesis | None,
    parameter: ModalParameterEstimate | None,
) -> str | None:
    if hypothesis is not None:
        return hypothesis.hypothesis_id
    if parameter is not None:
        return parameter.hypothesis_id
    return None


def _source_identifier(source: object) -> str:
    if isinstance(source, ModalEnvelopeSeries):
        return source.source_id
    if isinstance(source, SpectralTrack):
        return f"track:{source.track_id}"
    if isinstance(source, Envelope):
        return _stable_id("envelope", source.times_s, source.amplitudes)
    if hasattr(source, "source_id"):
        return str(getattr(source, "source_id"))
    if hasattr(source, "times_s") and hasattr(source, "amplitudes"):
        return _stable_id("series", getattr(source, "times_s"), getattr(source, "amplitudes"))
    return _stable_id("object", repr(source))


def _diagnostics_contain(
    source_a: ModalEnvelopeSeries,
    source_b: ModalEnvelopeSeries,
    pattern: str,
) -> bool:
    return any(pattern in item for item in (*source_a.diagnostics, *source_b.diagnostics))


def _dynamic_range_fraction(values: tuple[float, ...]) -> float:
    if not values:
        return 0.0
    scale = max(abs(value) for value in values)
    if scale <= 0:
        return 0.0
    return (max(values) - min(values)) / scale


def _representative_step(times: tuple[float, ...]) -> float | None:
    if len(times) < 2:
        return None
    return _median(tuple(later - earlier for earlier, later in zip(times, times[1:])))


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _population_std(values: Sequence[float]) -> float | None:
    if not values:
        return None
    mean = _mean(values)
    if mean is None:
        return None
    return sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and isfinite(float(value))


def _close_optional(actual: float | None, expected: float | None) -> bool:
    if actual is None or expected is None:
        return actual is expected
    return abs(actual - expected) <= 1e-12


def _validate_optional_finite(*values: float | None) -> None:
    if any(value is not None and not _finite(value) for value in values):
        raise ValueError("optional numeric values must be finite.")


def _validate_texts(values: Sequence[str], name: str) -> None:
    if not isinstance(values, tuple):
        raise ValueError(f"{name} must be a tuple.")
    if any(not isinstance(item, str) or not item.strip() for item in values):
        raise ValueError(f"{name} must contain nonempty strings.")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates.")


def _validate_reasons(reasons: Sequence[ModalEnergyExchangeReason]) -> None:
    if not isinstance(reasons, tuple):
        raise ValueError("reasons must be a tuple.")
    coerced = tuple(_coerce_enum(reason, ModalEnergyExchangeReason) for reason in reasons)
    if len(set(coerced)) != len(coerced):
        raise ValueError("reasons must not contain duplicates.")


def _ordered_texts(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(dict.fromkeys(str(value) for value in values if str(value).strip())))


def _ordered_reasons(
    reasons: Iterable[ModalEnergyExchangeReason],
) -> tuple[ModalEnergyExchangeReason, ...]:
    coerced = tuple(_coerce_enum(reason, ModalEnergyExchangeReason) for reason in reasons)
    return tuple(sorted(set(coerced), key=lambda item: item.value))


def _coerce_enum(value: Any, enum_type: type[Enum]) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(f"unknown {enum_type.__name__}: {value!r}") from exc


def _canonicalize(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _canonicalize(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, tuple | list):
        return [_canonicalize(item) for item in value]
    return value


def _stable_id(prefix: str, *parts: object) -> str:
    payload = json.dumps(_canonicalize(parts), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest[:16]}"
