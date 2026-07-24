"""Descriptive comparison across dynamic conditions.

This module compares already computed excitation, global spectral, and
time-resolved spectral characterizations for the nominal musical dynamic order
``pp < p < mf < f < ff``.  It measures differences between conditions; it does
not classify regimes, prove nonlinearity, associate candidates across
conditions, or promote any result to ``ModalMode``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite, log10, sqrt
from typing import Literal

import numpy as np

from belllab.excitation import ExcitationCharacterization
from belllab.global_spectrum import (
    GlobalSpectralCharacterization,
    evaluate_spectral_characterization_comparability,
)
from belllab.time_resolved_spectrum import (
    TimeResolvedSpectralCharacterization,
    evaluate_time_resolved_spectral_comparability,
)
from belllab.within_condition import ExcitationCondition


DYNAMIC_LABEL_ORDER: tuple[str, ...] = ("pp", "p", "mf", "f", "ff")
_DYNAMIC_LABEL_INDEX = {label: index for index, label in enumerate(DYNAMIC_LABEL_ORDER)}

_COMPARISON_POLICIES = frozenset({"available_pairs", "adjacent_only"})
_REFERENCE_POLICIES = frozenset({"configured", "configured_or_lowest_available", "none"})
_REPRESENTATIVE_STATISTICS = frozenset({"median", "mean"})
_MISSING_METRIC_POLICIES = frozenset({"preserve"})

_BASE_METRICS = frozenset(
    {
        "excitation_peak_absolute_amplitude",
        "excitation_rms_amplitude",
        "excitation_signal_energy",
        "excitation_equivalent_level_dbfs",
        "excitation_crest_factor",
        "excitation_impulse_duration_s",
        "excitation_attack_duration_s",
        "excitation_signal_to_background_db",
        "excitation_clipped_recording_fraction",
        "excitation_near_clipped_recording_fraction",
        "excitation_clipped_sample_fraction",
        "excitation_near_clipping_sample_fraction",
        "global_total_spectral_energy",
        "global_spectral_centroid_hz",
        "global_spectral_spread_hz",
        "global_spectral_rolloff_50_hz",
        "global_spectral_rolloff_85_hz",
        "global_spectral_rolloff_95_hz",
        "global_spectral_flatness",
        "global_spectral_entropy",
        "global_spectral_crest_factor",
        "global_significant_peak_count",
        "global_peak_density_per_hz",
        "global_peak_density_per_octave",
        "global_median_peak_spacing_hz",
        "global_tonal_energy_fraction",
        "global_residual_energy_fraction",
        "global_occupied_bandwidth_hz",
        "global_occupied_frequency_fraction",
        "time_energy_db_slope_per_s",
        "time_spectral_centroid_hz_slope_per_s",
        "time_spectral_spread_hz_slope_per_s",
        "time_spectral_flatness_slope_per_s",
        "time_spectral_entropy_slope_per_s",
        "time_significant_peak_count_slope_per_s",
        "time_peak_density_per_hz_slope_per_s",
        "time_tonal_energy_fraction_slope_per_s",
        "time_residual_energy_fraction_slope_per_s",
        "time_occupied_bandwidth_hz_slope_per_s",
        "time_change_point_count",
        "time_temporal_coverage_fraction",
    }
)

_REGION_NAMES = ("early", "middle", "late")
_REGION_METRIC_FIELDS = {
    "spectral_energy": ("median_energy", "energy"),
    "spectral_centroid_hz": ("median_centroid_hz", "Hz"),
    "spectral_spread_hz": ("median_spread_hz", "Hz"),
    "spectral_flatness": ("median_flatness", "fraction"),
    "spectral_entropy": ("median_entropy", "fraction"),
    "significant_peak_count": ("median_peak_count", "count"),
    "peak_density_per_hz": ("median_peak_density_per_hz", "1/Hz"),
    "tonal_energy_fraction": ("median_tonal_energy_fraction", "fraction"),
    "residual_energy_fraction": ("median_residual_energy_fraction", "fraction"),
    "occupied_bandwidth_hz": ("median_occupied_bandwidth_hz", "Hz"),
}

_TREND_UNITS = {
    "energy_db": "dB/s",
    "spectral_centroid_hz": "Hz/s",
    "spectral_spread_hz": "Hz/s",
    "spectral_flatness": "fraction/s",
    "spectral_entropy": "fraction/s",
    "significant_peak_count": "count/s",
    "peak_density_per_hz": "1/Hz/s",
    "tonal_energy_fraction": "fraction/s",
    "residual_energy_fraction": "fraction/s",
    "occupied_bandwidth_hz": "Hz/s",
}


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")


def _unique_strings(values: tuple[str, ...], name: str) -> None:
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{name} must contain nonempty strings.")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates.")


def _diagnostics(values: tuple[str, ...]) -> tuple[str, ...]:
    _unique_strings(values, "diagnostics")
    return values


def _finite_optional(*values: float | None) -> bool:
    return all(value is None or isfinite(value) for value in values)


def _fraction(value: float | None, name: str) -> None:
    if value is not None and (not isfinite(value) or value < 0.0 or value > 1.0):
        raise ValueError(f"{name} must lie in [0, 1] when present.")


def _recognized_metric_name(metric_name: str) -> bool:
    if metric_name in _BASE_METRICS:
        return True
    prefixes = (
        "early_",
        "middle_",
        "late_",
        "region_middle_minus_early_",
        "region_late_minus_early_",
        "region_late_minus_middle_",
        "global_band_",
        "time_band_",
        "time_first_change_point_",
    )
    return metric_name.startswith(prefixes)


def _clean_label(label: str) -> str:
    cleaned = "".join(char if char.isalnum() else "_" for char in label.strip().lower())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "unnamed"


def _sorted_unique(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _as_float(value: object | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return 1.0 if bool(value) else 0.0
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


@dataclass(frozen=True, slots=True)
class MetricTolerance:
    """Absolute and relative tolerances used to classify metric changes."""

    metric_name: str
    absolute_tolerance: float = 0.0
    relative_tolerance: float = 0.0

    def __post_init__(self) -> None:
        _text(self.metric_name, "metric_name")
        if not _recognized_metric_name(self.metric_name):
            raise ValueError("metric tolerance references an unknown metric.")
        if (
            not isfinite(self.absolute_tolerance)
            or not isfinite(self.relative_tolerance)
            or self.absolute_tolerance < 0.0
            or self.relative_tolerance < 0.0
        ):
            raise ValueError("metric tolerances must be finite and non-negative.")


@dataclass(frozen=True, slots=True)
class AggregatedMetric:
    """Robust descriptive aggregation of one metric across repeats."""

    metric_name: str
    unit: str
    available_count: int
    finite_count: int
    discarded_count: int
    median: float | None
    mean: float | None
    standard_deviation: float | None
    minimum: float | None
    maximum: float | None
    range: float | None
    coefficient_of_variation: float | None
    median_absolute_deviation: float | None
    valid: bool
    failure_reason: str | None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.metric_name, "metric_name")
        _text(self.unit, "unit")
        if self.available_count < 0 or self.finite_count < 0 or self.discarded_count < 0:
            raise ValueError("metric counts must not be negative.")
        if self.finite_count > self.available_count:
            raise ValueError("finite_count cannot exceed available_count.")
        if self.discarded_count != self.available_count - self.finite_count:
            raise ValueError("discarded_count is incoherent.")
        if not _finite_optional(
            self.median,
            self.mean,
            self.standard_deviation,
            self.minimum,
            self.maximum,
            self.range,
            self.coefficient_of_variation,
            self.median_absolute_deviation,
        ):
            raise ValueError("aggregated metric values must be finite when present.")
        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum:
                raise ValueError("minimum cannot exceed maximum.")
            if self.median is not None and not self.minimum <= self.median <= self.maximum:
                raise ValueError("median must lie inside the observed range.")
        if self.standard_deviation is not None and self.standard_deviation < 0.0:
            raise ValueError("standard deviation must not be negative.")
        if self.range is not None and self.range < 0.0:
            raise ValueError("range must not be negative.")
        if self.coefficient_of_variation is not None and self.coefficient_of_variation < 0.0:
            raise ValueError("coefficient of variation must not be negative.")
        if self.median_absolute_deviation is not None and self.median_absolute_deviation < 0.0:
            raise ValueError("median absolute deviation must not be negative.")
        if self.valid:
            if self.failure_reason is not None:
                raise ValueError("valid metric must not have failure_reason.")
            if self.finite_count <= 0:
                raise ValueError("valid metric requires at least one finite value.")
        else:
            if not self.failure_reason:
                raise ValueError("invalid metric requires failure_reason.")
        _diagnostics(self.diagnostics)


@dataclass(frozen=True, slots=True)
class DynamicConditionRecordingAnalysis:
    """Already computed analyses for one recording of one dynamic condition."""

    recording_id: str
    condition: ExcitationCondition
    excitation: ExcitationCharacterization | None = None
    global_spectrum: GlobalSpectralCharacterization | None = None
    time_resolved: TimeResolvedSpectralCharacterization | None = None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.recording_id, "recording_id")
        if self.condition.dynamic_label not in _DYNAMIC_LABEL_INDEX:
            raise ValueError("dynamic label must be one of pp, p, mf, f, ff.")
        _diagnostics(self.diagnostics)


@dataclass(frozen=True, slots=True)
class DynamicConditionComparisonSettings:
    """Policies for descriptive dynamic-condition comparison."""

    enabled_metrics: tuple[str, ...] = ()
    representative_statistic: Literal["median", "mean"] = "median"
    minimum_valid_repeats: int = 1
    metric_tolerances: tuple[MetricTolerance, ...] = ()
    require_same_session_for_amplitude: bool = False
    require_same_microphone_for_amplitude: bool = True
    require_same_interface_for_amplitude: bool = True
    require_same_gain_for_amplitude: bool = True
    require_same_distance_for_amplitude: bool = True
    require_same_channel_for_amplitude: bool = True
    require_same_orientation_for_amplitude: bool = False
    require_same_unit_for_amplitude: bool = True
    require_no_clipping_for_amplitude: bool = True
    exclude_clipped_conditions: bool = False
    pair_comparison_policy: Literal["available_pairs", "adjacent_only"] = "available_pairs"
    reference_policy: Literal["configured", "configured_or_lowest_available", "none"] = (
        "configured_or_lowest_available"
    )
    reference_dynamic_label: str = "pp"
    missing_metric_policy: Literal["preserve"] = "preserve"
    allow_band_definition_mismatch: bool = False
    required_bands: tuple[str, ...] = ()
    required_regions: tuple[str, ...] = ("early", "middle", "late")
    numerical_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if self.representative_statistic not in _REPRESENTATIVE_STATISTICS:
            raise ValueError("representative statistic is not recognized.")
        if self.minimum_valid_repeats < 0:
            raise ValueError("minimum_valid_repeats must not be negative.")
        if self.pair_comparison_policy not in _COMPARISON_POLICIES:
            raise ValueError("pair comparison policy is not recognized.")
        if self.reference_policy not in _REFERENCE_POLICIES:
            raise ValueError("reference policy is not recognized.")
        if self.reference_dynamic_label not in _DYNAMIC_LABEL_INDEX:
            raise ValueError("reference_dynamic_label is not recognized.")
        if self.missing_metric_policy not in _MISSING_METRIC_POLICIES:
            raise ValueError("missing metric policy is not recognized.")
        if not isfinite(self.numerical_tolerance) or self.numerical_tolerance < 0.0:
            raise ValueError("numerical_tolerance must be finite and non-negative.")
        _unique_strings(self.enabled_metrics, "enabled_metrics")
        _unique_strings(self.required_bands, "required_bands")
        _unique_strings(self.required_regions, "required_regions")
        if any(not _recognized_metric_name(name) for name in self.enabled_metrics):
            raise ValueError("enabled_metrics contains an unknown metric.")
        if any(region not in _REGION_NAMES for region in self.required_regions):
            raise ValueError("required_regions contains an unknown region.")
        metric_names = [tolerance.metric_name for tolerance in self.metric_tolerances]
        if len(metric_names) != len(set(metric_names)):
            raise ValueError("metric_tolerances must not contain duplicates.")
        for field in (
            self.require_same_session_for_amplitude,
            self.require_same_microphone_for_amplitude,
            self.require_same_interface_for_amplitude,
            self.require_same_gain_for_amplitude,
            self.require_same_distance_for_amplitude,
            self.require_same_channel_for_amplitude,
            self.require_same_orientation_for_amplitude,
            self.require_same_unit_for_amplitude,
            self.require_no_clipping_for_amplitude,
            self.exclude_clipped_conditions,
            self.allow_band_definition_mismatch,
        ):
            if not isinstance(field, bool):
                raise ValueError("boolean settings must be booleans.")


@dataclass(frozen=True, slots=True)
class DynamicConditionSpectralSummary:
    """Aggregated descriptive metrics for one nominal dynamic label."""

    dynamic_label: str
    recording_ids: tuple[str, ...]
    repeat_count: int
    valid_repeat_count: int
    discarded_repeat_count: int
    excitation_metrics: tuple[AggregatedMetric, ...]
    global_spectral_metrics: tuple[AggregatedMetric, ...]
    time_resolved_metrics: tuple[AggregatedMetric, ...]
    early_region_metrics: tuple[AggregatedMetric, ...]
    middle_region_metrics: tuple[AggregatedMetric, ...]
    late_region_metrics: tuple[AggregatedMetric, ...]
    region_change_metrics: tuple[AggregatedMetric, ...]
    band_metrics: tuple[AggregatedMetric, ...]
    within_condition_variability: tuple[AggregatedMetric, ...]
    comparability_status: tuple[str, ...]
    instrumental_fingerprint: tuple[tuple[str, object | None], ...]
    global_spectral_fingerprint: tuple[tuple[str, object | None], ...]
    time_resolved_fingerprint: tuple[tuple[str, object | None], ...]
    band_definitions: tuple[tuple[str, float, float], ...]
    region_definitions: tuple[tuple[str, float, float], ...]
    clipped_repeat_fraction: float | None
    near_clipped_repeat_fraction: float | None
    valid: bool
    failure_reason: str | None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.dynamic_label not in _DYNAMIC_LABEL_INDEX:
            raise ValueError("dynamic_label is not recognized.")
        _unique_strings(self.recording_ids, "recording_ids")
        if self.repeat_count != len(self.recording_ids):
            raise ValueError("repeat_count must match recording_ids.")
        if self.valid_repeat_count < 0 or self.discarded_repeat_count < 0:
            raise ValueError("repeat counts must not be negative.")
        if self.valid_repeat_count + self.discarded_repeat_count != self.repeat_count:
            raise ValueError("repeat counts are incoherent.")
        _fraction(self.clipped_repeat_fraction, "clipped_repeat_fraction")
        _fraction(self.near_clipped_repeat_fraction, "near_clipped_repeat_fraction")
        _unique_strings(self.comparability_status, "comparability_status")
        metric_names = [metric.metric_name for metric in self.all_metrics]
        if len(metric_names) != len(set(metric_names)):
            raise ValueError("summary metrics must not contain duplicate names.")
        if self.valid:
            if self.failure_reason is not None:
                raise ValueError("valid summary must not have failure_reason.")
        elif not self.failure_reason:
            raise ValueError("invalid summary requires failure_reason.")
        _diagnostics(self.diagnostics)

    @property
    def all_metrics(self) -> tuple[AggregatedMetric, ...]:
        return (
            self.excitation_metrics
            + self.global_spectral_metrics
            + self.time_resolved_metrics
            + self.early_region_metrics
            + self.middle_region_metrics
            + self.late_region_metrics
            + self.region_change_metrics
            + self.band_metrics
        )


@dataclass(frozen=True, slots=True)
class MetricComparison:
    """Descriptive change of one metric between two nominal dynamic labels."""

    metric_name: str
    unit: str
    lower_value: float | None
    higher_value: float | None
    absolute_change: float | None
    relative_change: float | None
    ratio: float | None
    change_db: float | None
    direction: Literal["increase", "decrease", "approximately_equal", "unavailable", "not_comparable"]
    comparable: bool
    not_applicable_reason: str | None
    change_to_within_condition_variability_ratio: float | None = None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.metric_name, "metric_name")
        _text(self.unit, "unit")
        if not _finite_optional(
            self.lower_value,
            self.higher_value,
            self.absolute_change,
            self.relative_change,
            self.ratio,
            self.change_db,
            self.change_to_within_condition_variability_ratio,
        ):
            raise ValueError("metric comparison values must be finite when present.")
        if self.ratio is not None and self.ratio < 0.0:
            raise ValueError("ratio must not be negative.")
        if (
            self.change_to_within_condition_variability_ratio is not None
            and self.change_to_within_condition_variability_ratio < 0.0
        ):
            raise ValueError("variability ratio must not be negative.")
        if self.comparable:
            if self.direction in {"unavailable", "not_comparable"}:
                raise ValueError("comparable metric needs an available direction.")
            if self.not_applicable_reason is not None:
                raise ValueError("comparable metric must not have not_applicable_reason.")
        else:
            if self.direction not in {"unavailable", "not_comparable"}:
                raise ValueError("non-comparable metric direction is incoherent.")
            if not self.not_applicable_reason:
                raise ValueError("non-comparable metric requires not_applicable_reason.")
            if any(
                value is not None
                for value in (
                    self.absolute_change,
                    self.relative_change,
                    self.ratio,
                    self.change_db,
                    self.change_to_within_condition_variability_ratio,
                )
            ):
                raise ValueError("non-comparable metric must not contain change values.")
        _diagnostics(self.diagnostics)


@dataclass(frozen=True, slots=True)
class DynamicConditionPairComparison:
    """Comparison between two present nominal dynamic labels."""

    lower_dynamic_label: str
    higher_dynamic_label: str
    label_step_count: int
    lower_summary: DynamicConditionSpectralSummary
    higher_summary: DynamicConditionSpectralSummary
    metric_comparisons: tuple[MetricComparison, ...]
    comparable: bool
    incompatibilities: tuple[str, ...]
    valid: bool
    failure_reason: str | None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.lower_dynamic_label not in _DYNAMIC_LABEL_INDEX or self.higher_dynamic_label not in _DYNAMIC_LABEL_INDEX:
            raise ValueError("pair labels must be recognized.")
        if _DYNAMIC_LABEL_INDEX[self.lower_dynamic_label] >= _DYNAMIC_LABEL_INDEX[self.higher_dynamic_label]:
            raise ValueError("pair labels must follow nominal dynamic order.")
        expected_steps = _DYNAMIC_LABEL_INDEX[self.higher_dynamic_label] - _DYNAMIC_LABEL_INDEX[self.lower_dynamic_label]
        if self.label_step_count != expected_steps or self.label_step_count <= 0:
            raise ValueError("label_step_count is incoherent.")
        if self.lower_summary.dynamic_label != self.lower_dynamic_label:
            raise ValueError("lower_summary label is incoherent.")
        if self.higher_summary.dynamic_label != self.higher_dynamic_label:
            raise ValueError("higher_summary label is incoherent.")
        _unique_strings(self.incompatibilities, "incompatibilities")
        names = [metric.metric_name for metric in self.metric_comparisons]
        if len(names) != len(set(names)):
            raise ValueError("metric comparisons must not contain duplicates.")
        if self.valid:
            if self.failure_reason is not None:
                raise ValueError("valid pair comparison must not have failure_reason.")
        elif not self.failure_reason:
            raise ValueError("invalid pair comparison requires failure_reason.")
        _diagnostics(self.diagnostics)


@dataclass(frozen=True, slots=True)
class DynamicMetricMonotonicityResult:
    """Operational monotonicity of one metric over the dynamic order."""

    metric_name: str
    unit: str
    monotonicity: Literal[
        "monotonically_increasing",
        "monotonically_decreasing",
        "non_decreasing",
        "non_increasing",
        "constant",
        "non_monotonic",
        "insufficient",
    ]
    available_count: int
    inversion_count: int
    tie_count: int
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.metric_name, "metric_name")
        _text(self.unit, "unit")
        if self.available_count < 0 or self.inversion_count < 0 or self.tie_count < 0:
            raise ValueError("monotonicity counts must not be negative.")
        _diagnostics(self.diagnostics)


@dataclass(frozen=True, slots=True)
class DynamicMetricSequence:
    """Ordered sequence of one metric over pp, p, mf, f, ff."""

    metric_name: str
    unit: str
    labels: tuple[str, ...]
    values: tuple[float | None, ...]
    valid_mask: tuple[bool, ...]
    pairwise_changes: tuple[MetricComparison, ...]
    monotonicity: str
    inversion_count: int
    tie_count: int
    missing_count: int
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.metric_name, "metric_name")
        _text(self.unit, "unit")
        if self.labels != DYNAMIC_LABEL_ORDER:
            raise ValueError("metric sequence labels must preserve canonical dynamic order.")
        if not (len(self.values) == len(self.valid_mask) == len(self.labels)):
            raise ValueError("metric sequence fields must have matching lengths.")
        if not _finite_optional(*self.values):
            raise ValueError("metric sequence values must be finite when present.")
        if self.inversion_count < 0 or self.tie_count < 0 or self.missing_count < 0:
            raise ValueError("metric sequence counts must not be negative.")
        if self.missing_count != sum(1 for valid in self.valid_mask if not valid):
            raise ValueError("metric sequence missing_count is incoherent.")
        _diagnostics(self.diagnostics)


@dataclass(frozen=True, slots=True)
class DynamicConditionComparisonResult:
    """Top-level descriptive comparison across dynamic conditions."""

    condition_summaries: tuple[DynamicConditionSpectralSummary, ...]
    pairwise_comparisons: tuple[DynamicConditionPairComparison, ...]
    reference_comparisons: tuple[DynamicConditionPairComparison, ...]
    reference_dynamic_label: str | None
    ordered_dynamic_labels: tuple[str, ...]
    missing_dynamic_labels: tuple[str, ...]
    metric_sequences: tuple[DynamicMetricSequence, ...]
    monotonicity_results: tuple[DynamicMetricMonotonicityResult, ...]
    valid: bool
    failure_reason: str | None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if any(label not in _DYNAMIC_LABEL_INDEX for label in self.ordered_dynamic_labels):
            raise ValueError("ordered_dynamic_labels contains an unknown label.")
        if tuple(sorted(self.ordered_dynamic_labels, key=_DYNAMIC_LABEL_INDEX.__getitem__)) != self.ordered_dynamic_labels:
            raise ValueError("ordered_dynamic_labels must follow canonical order.")
        _unique_strings(self.ordered_dynamic_labels, "ordered_dynamic_labels")
        _unique_strings(self.missing_dynamic_labels, "missing_dynamic_labels")
        if set(self.ordered_dynamic_labels).intersection(self.missing_dynamic_labels):
            raise ValueError("present and missing labels overlap.")
        if self.reference_dynamic_label is not None and self.reference_dynamic_label not in _DYNAMIC_LABEL_INDEX:
            raise ValueError("reference_dynamic_label is not recognized.")
        summary_labels = tuple(summary.dynamic_label for summary in self.condition_summaries)
        if summary_labels != self.ordered_dynamic_labels:
            raise ValueError("condition summaries must match ordered_dynamic_labels.")
        if self.valid:
            if self.failure_reason is not None:
                raise ValueError("valid result must not have failure_reason.")
        elif not self.failure_reason:
            raise ValueError("invalid result requires failure_reason.")
        _diagnostics(self.diagnostics)


def aggregate_metric_values(
    metric_name: str,
    values: tuple[float | None, ...] | list[float | None],
    *,
    unit: str,
    allow_coefficient_of_variation: bool = True,
    diagnostics: tuple[str, ...] = (),
) -> AggregatedMetric:
    """Aggregate finite values without inventing sentinels for missing data."""

    if not _recognized_metric_name(metric_name):
        raise ValueError("metric_name is not recognized.")
    finite_values = [_as_float(value) for value in values]
    finite_values = [value for value in finite_values if value is not None]
    available_count = len(values)
    finite_count = len(finite_values)
    discarded_count = available_count - finite_count
    if finite_count == 0:
        return AggregatedMetric(
            metric_name=metric_name,
            unit=unit,
            available_count=available_count,
            finite_count=0,
            discarded_count=discarded_count,
            median=None,
            mean=None,
            standard_deviation=None,
            minimum=None,
            maximum=None,
            range=None,
            coefficient_of_variation=None,
            median_absolute_deviation=None,
            valid=False,
            failure_reason="no_finite_values",
            diagnostics=diagnostics,
        )
    array = np.asarray(finite_values, dtype=float)
    median = float(np.median(array))
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=0))
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    value_range = maximum - minimum
    mad = float(np.median(np.abs(array - median)))
    coefficient_of_variation = None
    if allow_coefficient_of_variation and _allows_coefficient_of_variation(metric_name, unit) and not isclose(
        mean,
        0.0,
        abs_tol=1e-15,
    ):
        coefficient_of_variation = abs(std / mean)
    return AggregatedMetric(
        metric_name=metric_name,
        unit=unit,
        available_count=available_count,
        finite_count=finite_count,
        discarded_count=discarded_count,
        median=median,
        mean=mean,
        standard_deviation=std,
        minimum=minimum,
        maximum=maximum,
        range=value_range,
        coefficient_of_variation=coefficient_of_variation,
        median_absolute_deviation=mad,
        valid=True,
        failure_reason=None,
        diagnostics=diagnostics,
    )


def summarize_dynamic_condition(
    analyses: tuple[DynamicConditionRecordingAnalysis, ...] | list[DynamicConditionRecordingAnalysis],
    settings: DynamicConditionComparisonSettings | None = None,
) -> DynamicConditionSpectralSummary:
    """Aggregate repeats that share the same nominal dynamic label."""

    cfg = settings or DynamicConditionComparisonSettings()
    if not analyses:
        raise ValueError("summarize_dynamic_condition requires at least one analysis.")
    ordered = tuple(sorted(analyses, key=lambda analysis: analysis.recording_id))
    labels = {analysis.condition.dynamic_label for analysis in ordered}
    if len(labels) != 1:
        raise ValueError("all analyses must belong to the same dynamic condition.")
    label = ordered[0].condition.dynamic_label
    recording_ids = tuple(analysis.recording_id for analysis in ordered)
    if len(recording_ids) != len(set(recording_ids)):
        raise ValueError("recording_ids must be unique inside a condition.")

    included = tuple(analysis for analysis in ordered if not _excluded_by_clipping(analysis, cfg))
    valid_repeat_count = sum(1 for analysis in included if _analysis_has_valid_component(analysis))
    repeat_count = len(ordered)
    discarded_repeat_count = repeat_count - valid_repeat_count
    diagnostics: list[str] = [
        "nominal_dynamic_label_not_a_measured_intensity",
        "dynamic_condition_summary_is_descriptive_not_regime_classification",
    ]
    if len(included) < len(ordered):
        diagnostics.append("clipped_repeats_excluded_by_configuration")

    metric_groups = _collect_condition_metrics(ordered, included)
    selected = set(cfg.enabled_metrics) if cfg.enabled_metrics else None
    excitation_metrics = _aggregate_group(metric_groups["excitation"], selected)
    global_metrics = _aggregate_group(metric_groups["global"], selected)
    time_metrics = _aggregate_group(metric_groups["time"], selected)
    early_metrics = _aggregate_group(metric_groups["early"], selected)
    middle_metrics = _aggregate_group(metric_groups["middle"], selected)
    late_metrics = _aggregate_group(metric_groups["late"], selected)
    region_change_metrics = _aggregate_group(metric_groups["region_change"], selected)
    band_metrics = _aggregate_group(metric_groups["band"], selected)
    all_aggregates = (
        excitation_metrics
        + global_metrics
        + time_metrics
        + early_metrics
        + middle_metrics
        + late_metrics
        + region_change_metrics
        + band_metrics
    )
    variability = tuple(metric for metric in all_aggregates if metric.finite_count >= 2)

    comparability_status = list(_within_condition_comparability_status(included, cfg))
    clipped_fraction = _clipped_fraction(ordered, near=False)
    near_clipped_fraction = _clipped_fraction(ordered, near=True)
    if clipped_fraction is not None and clipped_fraction > 0.0:
        diagnostics.append("spectral_metrics_potentially_distorted_by_clipping")
        comparability_status.append("condition_contains_clipped_recordings")

    band_definitions = _band_definitions(ordered)
    for required in cfg.required_bands:
        if required not in {label for label, _, _ in band_definitions}:
            comparability_status.append(f"required_band_missing:{required}")
    region_definitions = _region_definitions(ordered)
    for required in cfg.required_regions:
        if required not in {region for region, _, _ in region_definitions}:
            comparability_status.append(f"required_region_missing:{required}")

    valid = valid_repeat_count >= cfg.minimum_valid_repeats and any(metric.valid for metric in all_aggregates)
    return DynamicConditionSpectralSummary(
        dynamic_label=label,
        recording_ids=recording_ids,
        repeat_count=repeat_count,
        valid_repeat_count=valid_repeat_count,
        discarded_repeat_count=discarded_repeat_count,
        excitation_metrics=excitation_metrics,
        global_spectral_metrics=global_metrics,
        time_resolved_metrics=time_metrics,
        early_region_metrics=early_metrics,
        middle_region_metrics=middle_metrics,
        late_region_metrics=late_metrics,
        region_change_metrics=region_change_metrics,
        band_metrics=band_metrics,
        within_condition_variability=variability,
        comparability_status=tuple(dict.fromkeys(comparability_status)),
        instrumental_fingerprint=_instrumental_fingerprint(ordered),
        global_spectral_fingerprint=_global_fingerprint(ordered),
        time_resolved_fingerprint=_time_fingerprint(ordered),
        band_definitions=band_definitions,
        region_definitions=region_definitions,
        clipped_repeat_fraction=clipped_fraction,
        near_clipped_repeat_fraction=near_clipped_fraction,
        valid=valid,
        failure_reason=None if valid else "insufficient_valid_repeats_or_metrics",
        diagnostics=tuple(dict.fromkeys(diagnostics)),
    )


def compare_dynamic_condition_pair(
    lower_summary: DynamicConditionSpectralSummary,
    higher_summary: DynamicConditionSpectralSummary,
    settings: DynamicConditionComparisonSettings | None = None,
) -> DynamicConditionPairComparison:
    """Compare two summaries in nominal dynamic-label order."""

    cfg = settings or DynamicConditionComparisonSettings()
    lower_index = _DYNAMIC_LABEL_INDEX[lower_summary.dynamic_label]
    higher_index = _DYNAMIC_LABEL_INDEX[higher_summary.dynamic_label]
    if lower_index >= higher_index:
        raise ValueError("summaries must be passed in nominal lower-to-higher order.")

    incompatibilities = list(_pair_incompatibilities(lower_summary, higher_summary, cfg))
    lower_metrics = {metric.metric_name: metric for metric in lower_summary.all_metrics}
    higher_metrics = {metric.metric_name: metric for metric in higher_summary.all_metrics}
    names = tuple(
        sorted(
            (set(lower_metrics) | set(higher_metrics))
            if not cfg.enabled_metrics
            else set(cfg.enabled_metrics),
        )
    )
    comparisons = tuple(
        _compare_metric(
            name,
            lower_metrics.get(name),
            higher_metrics.get(name),
            lower_summary,
            higher_summary,
            cfg,
            tuple(incompatibilities),
        )
        for name in names
    )
    step_count = higher_index - lower_index
    diagnostics: list[str] = [
        "lower_and_higher_refer_to_nominal_dynamic_labels_not_measured_intensity",
        "pair_comparison_is_descriptive_not_non_linearity_proof",
    ]
    if step_count > 1:
        missing_between = DYNAMIC_LABEL_ORDER[lower_index + 1 : higher_index]
        diagnostics.append("non_adjacent_dynamic_comparison")
        diagnostics.append("missing_intermediate_dynamic_labels:" + ",".join(missing_between))
    comparable = any(comparison.comparable for comparison in comparisons)
    return DynamicConditionPairComparison(
        lower_dynamic_label=lower_summary.dynamic_label,
        higher_dynamic_label=higher_summary.dynamic_label,
        label_step_count=step_count,
        lower_summary=lower_summary,
        higher_summary=higher_summary,
        metric_comparisons=comparisons,
        comparable=comparable,
        incompatibilities=tuple(dict.fromkeys(incompatibilities)),
        valid=lower_summary.valid and higher_summary.valid and comparable,
        failure_reason=None if lower_summary.valid and higher_summary.valid and comparable else "no_comparable_metric_or_invalid_summary",
        diagnostics=tuple(dict.fromkeys(diagnostics)),
    )


def compare_dynamic_conditions(
    analyses: tuple[DynamicConditionRecordingAnalysis, ...] | list[DynamicConditionRecordingAnalysis],
    settings: DynamicConditionComparisonSettings | None = None,
) -> DynamicConditionComparisonResult:
    """Compare all available dynamic conditions without reading files or recomputing spectra."""

    cfg = settings or DynamicConditionComparisonSettings()
    ordered_input = tuple(
        sorted(
            analyses,
            key=lambda analysis: (_DYNAMIC_LABEL_INDEX[analysis.condition.dynamic_label], analysis.recording_id),
        )
    )
    grouped: dict[str, list[DynamicConditionRecordingAnalysis]] = {label: [] for label in DYNAMIC_LABEL_ORDER}
    for analysis in ordered_input:
        grouped[analysis.condition.dynamic_label].append(analysis)

    summaries = tuple(
        summarize_dynamic_condition(tuple(grouped[label]), cfg)
        for label in DYNAMIC_LABEL_ORDER
        if grouped[label]
    )
    ordered_labels = tuple(summary.dynamic_label for summary in summaries)
    missing_labels = tuple(label for label in DYNAMIC_LABEL_ORDER if label not in ordered_labels)
    valid_summaries = tuple(summary for summary in summaries if summary.valid)

    pairwise = _build_pairwise_comparisons(valid_summaries, cfg)
    reference_label, reference_pairs, reference_diagnostics = _build_reference_comparisons(valid_summaries, cfg)
    sequences = _build_metric_sequences(summaries, cfg)
    monotonicity_results = tuple(
        evaluate_dynamic_metric_monotonicity(
            sequence.metric_name,
            sequence.unit,
            sequence.labels,
            sequence.values,
            settings=cfg,
        )
        for sequence in sequences
    )
    diagnostics = [
        "dynamic_condition_comparison_is_descriptive_not_regime_classification",
        "no_cross_condition_candidate_association_was_performed",
        "no_modal_mode_conversion_was_performed",
    ]
    diagnostics.extend(reference_diagnostics)
    if missing_labels:
        diagnostics.append("missing_dynamic_labels:" + ",".join(missing_labels))
    valid = len(valid_summaries) >= 2 and any(pair.valid for pair in pairwise + reference_pairs)
    return DynamicConditionComparisonResult(
        condition_summaries=summaries,
        pairwise_comparisons=pairwise,
        reference_comparisons=reference_pairs,
        reference_dynamic_label=reference_label,
        ordered_dynamic_labels=ordered_labels,
        missing_dynamic_labels=missing_labels,
        metric_sequences=sequences,
        monotonicity_results=monotonicity_results,
        valid=valid,
        failure_reason=None if valid else "insufficient_comparable_dynamic_conditions",
        diagnostics=tuple(dict.fromkeys(diagnostics)),
    )


def evaluate_dynamic_metric_monotonicity(
    metric_name: str,
    unit: str,
    labels: tuple[str, ...],
    values: tuple[float | None, ...],
    settings: DynamicConditionComparisonSettings | None = None,
) -> DynamicMetricMonotonicityResult:
    """Evaluate descriptive monotonicity over the nominal dynamic order."""

    cfg = settings or DynamicConditionComparisonSettings()
    if labels != DYNAMIC_LABEL_ORDER:
        raise ValueError("labels must preserve canonical dynamic order.")
    finite_pairs = [(label, value) for label, value in zip(labels, values) if value is not None and isfinite(value)]
    if len(finite_pairs) < 2:
        return DynamicMetricMonotonicityResult(
            metric_name=metric_name,
            unit=unit,
            monotonicity="insufficient",
            available_count=len(finite_pairs),
            inversion_count=0,
            tie_count=0,
            diagnostics=("fewer_than_two_finite_values",),
        )
    tolerance = _tolerance_for(metric_name, cfg)
    signs: list[int] = []
    tie_count = 0
    for (_, left), (_, right) in zip(finite_pairs, finite_pairs[1:]):
        sign = _change_sign(left, right, tolerance)
        signs.append(sign)
        if sign == 0:
            tie_count += 1
    if all(sign == 0 for sign in signs):
        monotonicity = "constant"
    elif all(sign > 0 for sign in signs):
        monotonicity = "monotonically_increasing"
    elif all(sign < 0 for sign in signs):
        monotonicity = "monotonically_decreasing"
    elif all(sign >= 0 for sign in signs):
        monotonicity = "non_decreasing"
    elif all(sign <= 0 for sign in signs):
        monotonicity = "non_increasing"
    else:
        monotonicity = "non_monotonic"
    nonzero = [sign for sign in signs if sign != 0]
    inversion_count = 0
    if nonzero:
        reference_sign = nonzero[0]
        inversion_count = sum(1 for sign in nonzero[1:] if sign != reference_sign)
    return DynamicMetricMonotonicityResult(
        metric_name=metric_name,
        unit=unit,
        monotonicity=monotonicity,
        available_count=len(finite_pairs),
        inversion_count=inversion_count,
        tie_count=tie_count,
        diagnostics=(),
    )


def _aggregate_group(
    metric_values: dict[str, tuple[str, list[float | None], bool, tuple[str, ...]]],
    selected: set[str] | None,
) -> tuple[AggregatedMetric, ...]:
    metrics: list[AggregatedMetric] = []
    for name in sorted(metric_values):
        if selected is not None and name not in selected:
            continue
        unit, values, allow_cv, diagnostics = metric_values[name]
        metrics.append(
            aggregate_metric_values(
                name,
                values,
                unit=unit,
                allow_coefficient_of_variation=allow_cv,
                diagnostics=diagnostics,
            )
        )
    return tuple(metrics)


def _add_metric(
    group: dict[str, tuple[str, list[float | None], bool, tuple[str, ...]]],
    name: str,
    unit: str,
    values: list[float | None],
    *,
    allow_cv: bool = True,
    diagnostics: tuple[str, ...] = (),
) -> None:
    if not _recognized_metric_name(name):
        raise ValueError(f"unrecognized metric name: {name}")
    group[name] = (unit, values, allow_cv, diagnostics)


def _collect_condition_metrics(
    all_records: tuple[DynamicConditionRecordingAnalysis, ...],
    included_records: tuple[DynamicConditionRecordingAnalysis, ...],
) -> dict[str, dict[str, tuple[str, list[float | None], bool, tuple[str, ...]]]]:
    groups: dict[str, dict[str, tuple[str, list[float | None], bool, tuple[str, ...]]]] = {
        "excitation": {},
        "global": {},
        "time": {},
        "early": {},
        "middle": {},
        "late": {},
        "region_change": {},
        "band": {},
    }
    records = tuple(all_records)
    included_set = {record.recording_id for record in included_records}

    def active(record: DynamicConditionRecordingAnalysis) -> bool:
        return record.recording_id in included_set

    amplitude_unit = _common_excitation_unit(records)
    _add_metric(groups["excitation"], "excitation_peak_absolute_amplitude", amplitude_unit, [
        _get_excitation_value(record, "peak_absolute_amplitude") if active(record) else None for record in records
    ])
    _add_metric(groups["excitation"], "excitation_rms_amplitude", amplitude_unit, [
        _get_excitation_value(record, "rms_amplitude") if active(record) else None for record in records
    ])
    _add_metric(groups["excitation"], "excitation_signal_energy", "energy", [
        _get_excitation_value(record, "signal_energy") if active(record) else None for record in records
    ])
    _add_metric(groups["excitation"], "excitation_equivalent_level_dbfs", "dBFS", [
        _get_excitation_value(record, "equivalent_level_dbfs") if active(record) else None for record in records
    ], allow_cv=False)
    _add_metric(groups["excitation"], "excitation_crest_factor", "ratio", [
        _get_excitation_value(record, "crest_factor") if active(record) else None for record in records
    ])
    _add_metric(groups["excitation"], "excitation_impulse_duration_s", "s", [
        _get_excitation_value(record, "impulse_duration_s") if active(record) else None for record in records
    ])
    _add_metric(groups["excitation"], "excitation_attack_duration_s", "s", [
        _get_excitation_value(record, "attack_duration_s") if active(record) else None for record in records
    ])
    _add_metric(groups["excitation"], "excitation_signal_to_background_db", "dB", [
        _get_excitation_value(record, "signal_to_background_db") if active(record) else None for record in records
    ], allow_cv=False)
    _add_metric(groups["excitation"], "excitation_clipped_recording_fraction", "fraction", [
        _get_clipping_value(record, near=False) if active(record) else None for record in records
    ], allow_cv=False)
    _add_metric(groups["excitation"], "excitation_near_clipped_recording_fraction", "fraction", [
        _get_clipping_value(record, near=True) if active(record) else None for record in records
    ], allow_cv=False)
    _add_metric(groups["excitation"], "excitation_clipped_sample_fraction", "fraction", [
        _get_excitation_value(record, "clipped_sample_fraction") if active(record) else None for record in records
    ], allow_cv=False)
    _add_metric(groups["excitation"], "excitation_near_clipping_sample_fraction", "fraction", [
        _get_excitation_value(record, "near_clipping_sample_fraction") if active(record) else None for record in records
    ], allow_cv=False)

    global_fields = {
        "global_total_spectral_energy": ("total_spectral_energy", "energy", True),
        "global_spectral_centroid_hz": ("spectral_centroid_hz", "Hz", True),
        "global_spectral_spread_hz": ("spectral_spread_hz", "Hz", True),
        "global_spectral_rolloff_50_hz": ("spectral_rolloff_50_hz", "Hz", True),
        "global_spectral_rolloff_85_hz": ("spectral_rolloff_85_hz", "Hz", True),
        "global_spectral_rolloff_95_hz": ("spectral_rolloff_95_hz", "Hz", True),
        "global_spectral_flatness": ("spectral_flatness", "fraction", False),
        "global_spectral_entropy": ("spectral_entropy", "fraction", False),
        "global_spectral_crest_factor": ("spectral_crest_factor", "ratio", True),
        "global_significant_peak_count": ("significant_peak_count", "count", True),
        "global_peak_density_per_hz": ("peak_density_per_hz", "1/Hz", True),
        "global_peak_density_per_octave": ("peak_density_per_octave", "1/octave", True),
        "global_median_peak_spacing_hz": ("median_peak_spacing_hz", "Hz", True),
        "global_tonal_energy_fraction": ("tonal_energy_fraction", "fraction", False),
        "global_residual_energy_fraction": ("residual_energy_fraction", "fraction", False),
        "global_occupied_bandwidth_hz": ("occupied_bandwidth_hz", "Hz", True),
        "global_occupied_frequency_fraction": ("occupied_frequency_fraction", "fraction", False),
    }
    for metric_name, (attribute, unit, allow_cv) in global_fields.items():
        _add_metric(groups["global"], metric_name, unit, [
            _get_global_value(record, attribute) if active(record) else None for record in records
        ], allow_cv=allow_cv)

    for band_label in sorted(_global_band_labels(records)):
        clean = _clean_label(band_label)
        _add_metric(groups["band"], f"global_band_{clean}_energy", "energy", [
            _get_global_band_value(record, band_label, "energy") if active(record) else None for record in records
        ])
        _add_metric(groups["band"], f"global_band_{clean}_energy_fraction", "fraction", [
            _get_global_band_value(record, band_label, "energy_fraction") if active(record) else None for record in records
        ], allow_cv=False)

    for trend_name, unit in _TREND_UNITS.items():
        _add_metric(groups["time"], f"time_{trend_name}_slope_per_s", unit, [
            _get_time_trend_value(record, trend_name) if active(record) else None for record in records
        ], allow_cv=False)
    _add_metric(groups["time"], "time_change_point_count", "count", [
        _get_time_change_point_count(record) if active(record) else None for record in records
    ])
    _add_metric(groups["time"], "time_temporal_coverage_fraction", "fraction", [
        _get_time_summary_value(record, "temporal_coverage_fraction") if active(record) else None for record in records
    ], allow_cv=False)
    for change_metric in sorted(_time_change_metric_names(records)):
        clean = _clean_label(change_metric)
        _add_metric(groups["time"], f"time_first_change_point_{clean}_s", "s", [
            _get_first_change_point_time(record, change_metric) if active(record) else None for record in records
        ])

    for region in _REGION_NAMES:
        for suffix, (attribute, unit) in _REGION_METRIC_FIELDS.items():
            metric_name = f"{region}_{suffix}"
            _add_metric(groups[region], metric_name, unit, [
                _get_region_value(record, region, attribute) if active(record) else None for record in records
            ], allow_cv=unit not in {"fraction"})

    for left, right in (("early", "middle"), ("early", "late"), ("middle", "late")):
        comparison_prefix = {
            ("early", "middle"): "region_middle_minus_early",
            ("early", "late"): "region_late_minus_early",
            ("middle", "late"): "region_late_minus_middle",
        }[(left, right)]
        for suffix, (attribute, unit) in _REGION_METRIC_FIELDS.items():
            _add_metric(groups["region_change"], f"{comparison_prefix}_{suffix}", unit, [
                _get_region_change(record, left, right, attribute) if active(record) else None for record in records
            ], allow_cv=False)

    for band_label in sorted(_time_band_labels(records)):
        clean = _clean_label(band_label)
        _add_metric(groups["band"], f"time_band_{clean}_coverage_fraction", "fraction", [
            _get_time_band_value(record, band_label, "coverage_fraction") if active(record) else None for record in records
        ], allow_cv=False)
        _add_metric(groups["band"], f"time_band_{clean}_time_until_below_threshold_s", "s", [
            _get_time_band_value(record, band_label, "time_until_below_threshold_s") if active(record) else None for record in records
        ])
        _add_metric(groups["band"], f"time_band_{clean}_energy_fraction_slope_per_s", "fraction/s", [
            _get_time_band_trend_value(record, band_label) if active(record) else None for record in records
        ], allow_cv=False)

    return groups


def _analysis_has_valid_component(analysis: DynamicConditionRecordingAnalysis) -> bool:
    return any(
        component is not None and getattr(component, "valid", False)
        for component in (analysis.excitation, analysis.global_spectrum, analysis.time_resolved)
    )


def _excluded_by_clipping(
    analysis: DynamicConditionRecordingAnalysis,
    settings: DynamicConditionComparisonSettings,
) -> bool:
    if not settings.exclude_clipped_conditions:
        return False
    return bool(analysis.excitation and analysis.excitation.clipping_detected)


def _get_excitation_value(record: DynamicConditionRecordingAnalysis, attribute: str) -> float | None:
    excitation = record.excitation
    if excitation is None or not excitation.valid:
        return None
    return _as_float(getattr(excitation, attribute))


def _get_clipping_value(record: DynamicConditionRecordingAnalysis, *, near: bool) -> float | None:
    excitation = record.excitation
    if excitation is None or not excitation.valid:
        return None
    return 1.0 if bool(excitation.near_clipping_detected if near else excitation.clipping_detected) else 0.0


def _get_global_value(record: DynamicConditionRecordingAnalysis, attribute: str) -> float | None:
    global_result = record.global_spectrum
    if global_result is None or not global_result.valid:
        return None
    if attribute.startswith("spectral_rolloff_"):
        fraction = float(attribute.removeprefix("spectral_rolloff_").removesuffix("_hz")) / 100.0
        for rolloff_fraction, frequency in global_result.rolloff_frequencies_hz:
            if isclose(rolloff_fraction, fraction, rel_tol=0.0, abs_tol=1e-12):
                return _as_float(frequency)
        return None
    return _as_float(getattr(global_result, attribute))


def _get_global_band_value(record: DynamicConditionRecordingAnalysis, label: str, attribute: str) -> float | None:
    global_result = record.global_spectrum
    if global_result is None or not global_result.valid:
        return None
    for band in global_result.band_energy_metrics:
        if band.label == label:
            return _as_float(getattr(band, attribute))
    return None


def _get_time_summary_value(record: DynamicConditionRecordingAnalysis, attribute: str) -> float | None:
    time_result = record.time_resolved
    if time_result is None or not time_result.valid:
        return None
    return _as_float(getattr(time_result.summary, attribute))


def _get_time_trend_value(record: DynamicConditionRecordingAnalysis, metric_name: str) -> float | None:
    time_result = record.time_resolved
    if time_result is None or not time_result.valid:
        return None
    for trend in time_result.temporal_trends:
        if trend.metric_name == metric_name and trend.success:
            return _as_float(trend.slope_per_s)
    return None


def _get_time_change_point_count(record: DynamicConditionRecordingAnalysis) -> float | None:
    time_result = record.time_resolved
    if time_result is None or not time_result.valid:
        return None
    return float(len(time_result.change_points))


def _get_first_change_point_time(record: DynamicConditionRecordingAnalysis, metric_name: str) -> float | None:
    time_result = record.time_resolved
    if time_result is None or not time_result.valid:
        return None
    times = [point.time_s for point in time_result.change_points if point.metric_name == metric_name]
    return min(times) if times else None


def _get_region(record: DynamicConditionRecordingAnalysis, region_name: str):
    time_result = record.time_resolved
    if time_result is None or not time_result.valid:
        return None
    for region in time_result.summary.regions:
        if region.region == region_name and region.valid:
            return region
    return None


def _get_region_value(record: DynamicConditionRecordingAnalysis, region_name: str, attribute: str) -> float | None:
    region = _get_region(record, region_name)
    if region is None:
        return None
    return _as_float(getattr(region, attribute))


def _get_region_change(record: DynamicConditionRecordingAnalysis, left: str, right: str, attribute: str) -> float | None:
    left_value = _get_region_value(record, left, attribute)
    right_value = _get_region_value(record, right, attribute)
    if left_value is None or right_value is None:
        return None
    return right_value - left_value


def _get_time_band_value(record: DynamicConditionRecordingAnalysis, label: str, attribute: str) -> float | None:
    time_result = record.time_resolved
    if time_result is None or not time_result.valid:
        return None
    for band in time_result.summary.band_summaries:
        if band.label == label:
            return _as_float(getattr(band, attribute))
    return None


def _get_time_band_trend_value(record: DynamicConditionRecordingAnalysis, label: str) -> float | None:
    time_result = record.time_resolved
    if time_result is None or not time_result.valid:
        return None
    for band in time_result.summary.band_summaries:
        if band.label == label and band.energy_fraction_trend.success:
            return _as_float(band.energy_fraction_trend.slope_per_s)
    return None


def _global_band_labels(records: tuple[DynamicConditionRecordingAnalysis, ...]) -> tuple[str, ...]:
    labels: list[str] = []
    for record in records:
        if record.global_spectrum is not None:
            labels.extend(band.label for band in record.global_spectrum.band_energy_metrics)
    return tuple(sorted(set(labels)))


def _time_band_labels(records: tuple[DynamicConditionRecordingAnalysis, ...]) -> tuple[str, ...]:
    labels: list[str] = []
    for record in records:
        if record.time_resolved is not None:
            labels.extend(band.label for band in record.time_resolved.summary.band_summaries)
    return tuple(sorted(set(labels)))


def _time_change_metric_names(records: tuple[DynamicConditionRecordingAnalysis, ...]) -> tuple[str, ...]:
    names: list[str] = []
    for record in records:
        if record.time_resolved is not None:
            names.extend(point.metric_name for point in record.time_resolved.change_points)
    return tuple(sorted(set(names)))


def _common_excitation_unit(records: tuple[DynamicConditionRecordingAnalysis, ...]) -> str:
    units = {
        record.excitation.amplitude_unit
        for record in records
        if record.excitation is not None and record.excitation.valid
    }
    if len(units) == 1:
        return next(iter(units))
    return "mixed" if units else "amplitude"


def _clipped_fraction(records: tuple[DynamicConditionRecordingAnalysis, ...], *, near: bool) -> float | None:
    values = [_get_clipping_value(record, near=near) for record in records]
    finite = [value for value in values if value is not None]
    if not finite:
        return None
    return float(np.mean(finite))


def _allows_coefficient_of_variation(metric_name: str, unit: str) -> bool:
    if unit in {"dB", "dBFS", "dB/s"}:
        return False
    if "slope_per_s" in metric_name:
        return False
    if metric_name.endswith("_fraction") or unit == "fraction":
        return False
    return True


def _instrumental_fingerprint(records: tuple[DynamicConditionRecordingAnalysis, ...]) -> tuple[tuple[str, object | None], ...]:
    for record in records:
        excitation = record.excitation
        condition = record.condition
        if excitation is not None:
            return (
                ("session_id", excitation.session_id),
                ("microphone_id", excitation.microphone_id),
                ("interface_id", excitation.interface_id),
                ("acquisition_gain", excitation.acquisition_gain),
                ("microphone_distance_m", excitation.microphone_distance_m),
                ("microphone_orientation", excitation.microphone_orientation),
                ("channel_index", excitation.channel_index),
                ("amplitude_unit", excitation.amplitude_unit),
            )
        return (
            ("session_id", condition.session_id),
            ("microphone_id", condition.microphone_id),
            ("interface_id", condition.interface_id),
            ("acquisition_gain", condition.acquisition_gain),
            ("microphone_distance_m", condition.microphone_distance_m),
            ("microphone_orientation", condition.microphone_orientation),
            ("channel_index", condition.channel_index),
            ("amplitude_unit", None),
        )
    return ()


def _global_fingerprint(records: tuple[DynamicConditionRecordingAnalysis, ...]) -> tuple[tuple[str, object | None], ...]:
    for record in records:
        result = record.global_spectrum
        if result is not None:
            settings = result.settings
            return (
                ("sample_rate_hz", result.sample_rate_hz),
                ("analysis_duration_s", result.analysis_end_time_s - result.analysis_start_time_s),
                ("frequency_min_hz", result.frequency_min_hz),
                ("frequency_max_hz", result.frequency_max_hz),
                ("fft_size", result.fft_size),
                ("window_name", settings.window_name),
                ("detrend_policy", settings.detrend_policy),
                ("frequency_resolution_hz", result.frequency_resolution_hz),
                ("peak_min_power", settings.peak_min_power),
                ("peak_min_prominence", settings.peak_min_prominence),
                ("peak_distance_bins", settings.peak_distance_bins),
                ("spectral_domain", result.canonical_spectral_domain),
                ("spectral_normalization", result.spectral_normalization),
                ("bands", tuple((band.label, band.frequency_start_hz, band.frequency_end_hz) for band in result.band_energy_metrics)),
            )
    return ()


def _time_fingerprint(records: tuple[DynamicConditionRecordingAnalysis, ...]) -> tuple[tuple[str, object | None], ...]:
    for record in records:
        result = record.time_resolved
        if result is not None:
            settings = result.settings
            return (
                ("sample_rate_hz", result.sample_rate_hz),
                ("frame_duration_s", result.frame_duration_s),
                ("hop_duration_s", result.hop_duration_s),
                ("frequency_min_hz", result.frequency_min_hz),
                ("frequency_max_hz", result.frequency_max_hz),
                ("fft_size", result.fft_size),
                ("window_name", settings.window_name),
                ("detrend_policy", settings.detrend_policy),
                ("analysis_window", (settings.analysis_window_start_s, settings.analysis_window_end_s)),
                ("impact_time_s", result.impact_time_s),
                ("peak_min_power", settings.peak_min_power),
                ("peak_min_prominence", settings.peak_min_prominence),
                ("peak_distance_bins", settings.peak_distance_bins),
                ("normalization_between_frames", settings.normalization_between_frames),
                ("regions", tuple((region.region, region.start_relative_s, region.end_relative_s) for region in result.summary.regions)),
                ("smoothing", (settings.smoothing_method, settings.smoothing_window_frames)),
                ("change_points", (settings.change_point_method, settings.change_point_window_frames, settings.change_point_thresholds)),
                ("bands", tuple((band.label, band.frequency_start_hz, band.frequency_end_hz) for band in result.summary.band_summaries)),
            )
    return ()


def _band_definitions(records: tuple[DynamicConditionRecordingAnalysis, ...]) -> tuple[tuple[str, float, float], ...]:
    definitions: dict[str, tuple[float, float]] = {}
    for record in records:
        if record.global_spectrum is not None:
            for band in record.global_spectrum.band_energy_metrics:
                definitions.setdefault(band.label, (band.frequency_start_hz, band.frequency_end_hz))
        if record.time_resolved is not None:
            for band in record.time_resolved.summary.band_summaries:
                definitions.setdefault(band.label, (band.frequency_start_hz, band.frequency_end_hz))
    return tuple((label, start, end) for label, (start, end) in sorted(definitions.items()))


def _region_definitions(records: tuple[DynamicConditionRecordingAnalysis, ...]) -> tuple[tuple[str, float, float], ...]:
    definitions: dict[str, tuple[float, float]] = {}
    for record in records:
        if record.time_resolved is not None:
            for region in record.time_resolved.summary.regions:
                definitions.setdefault(region.region, (region.start_relative_s, region.end_relative_s))
    return tuple((label, start, end) for label, (start, end) in sorted(definitions.items()))


def _within_condition_comparability_status(
    records: tuple[DynamicConditionRecordingAnalysis, ...],
    settings: DynamicConditionComparisonSettings,
) -> tuple[str, ...]:
    diagnostics: list[str] = []
    if _fingerprint_has_mixed_values(records, "instrumental"):
        diagnostics.append("instrumental_metadata_varies_within_condition")
    globals_ = [record.global_spectrum for record in records if record.global_spectrum is not None]
    for left, right in zip(globals_, globals_[1:]):
        result = evaluate_spectral_characterization_comparability(left, right)
        if not result.comparable:
            diagnostics.extend(f"within_condition_global:{item}" for item in result.incompatibilities)
    times = [record.time_resolved for record in records if record.time_resolved is not None]
    for left, right in zip(times, times[1:]):
        result = evaluate_time_resolved_spectral_comparability(left, right)
        if not result.comparable:
            diagnostics.extend(f"within_condition_time:{item}" for item in result.incompatibilities)
    if settings.exclude_clipped_conditions:
        diagnostics.append("clipped_condition_exclusion_policy_enabled")
    return tuple(dict.fromkeys(diagnostics))


def _fingerprint_has_mixed_values(records: tuple[DynamicConditionRecordingAnalysis, ...], kind: str) -> bool:
    fingerprints = []
    for record in records:
        if kind == "instrumental":
            fingerprints.append(_instrumental_fingerprint((record,)))
    return len(set(fingerprints)) > 1


def _pair_incompatibilities(
    lower: DynamicConditionSpectralSummary,
    higher: DynamicConditionSpectralSummary,
    settings: DynamicConditionComparisonSettings,
) -> tuple[str, ...]:
    incompatibilities: list[str] = []
    lower_instrument = dict(lower.instrumental_fingerprint)
    higher_instrument = dict(higher.instrumental_fingerprint)
    checks = (
        ("session_id", settings.require_same_session_for_amplitude),
        ("microphone_id", settings.require_same_microphone_for_amplitude),
        ("interface_id", settings.require_same_interface_for_amplitude),
        ("acquisition_gain", settings.require_same_gain_for_amplitude),
        ("microphone_distance_m", settings.require_same_distance_for_amplitude),
        ("microphone_orientation", settings.require_same_orientation_for_amplitude),
        ("channel_index", settings.require_same_channel_for_amplitude),
        ("amplitude_unit", settings.require_same_unit_for_amplitude),
    )
    for field, required in checks:
        if required and lower_instrument.get(field) != higher_instrument.get(field):
            incompatibilities.append(f"instrumental_{field}_mismatch")
    if settings.require_no_clipping_for_amplitude and (
        (lower.clipped_repeat_fraction or 0.0) > 0.0 or (higher.clipped_repeat_fraction or 0.0) > 0.0
    ):
        incompatibilities.append("amplitude_metrics_incompatible_with_clipping")
    if lower.global_spectral_fingerprint and higher.global_spectral_fingerprint and lower.global_spectral_fingerprint != higher.global_spectral_fingerprint:
        incompatibilities.append("global_spectral_configuration_mismatch")
    if lower.time_resolved_fingerprint and higher.time_resolved_fingerprint and lower.time_resolved_fingerprint != higher.time_resolved_fingerprint:
        incompatibilities.append("time_resolved_configuration_mismatch")
    if not settings.allow_band_definition_mismatch and lower.band_definitions and higher.band_definitions and lower.band_definitions != higher.band_definitions:
        incompatibilities.append("band_definition_mismatch")
    if lower.comparability_status:
        incompatibilities.extend(f"{lower.dynamic_label}:{item}" for item in lower.comparability_status)
    if higher.comparability_status:
        incompatibilities.extend(f"{higher.dynamic_label}:{item}" for item in higher.comparability_status)
    return tuple(dict.fromkeys(incompatibilities))


def _compare_metric(
    metric_name: str,
    lower_metric: AggregatedMetric | None,
    higher_metric: AggregatedMetric | None,
    lower_summary: DynamicConditionSpectralSummary,
    higher_summary: DynamicConditionSpectralSummary,
    settings: DynamicConditionComparisonSettings,
    pair_incompatibilities: tuple[str, ...],
) -> MetricComparison:
    unit = lower_metric.unit if lower_metric is not None else (higher_metric.unit if higher_metric is not None else "unknown")
    if lower_metric is None or higher_metric is None:
        return _not_comparable(metric_name, unit, "metric_missing_in_one_condition")
    lower_value = _representative_value(lower_metric, settings)
    higher_value = _representative_value(higher_metric, settings)
    if lower_value is None or higher_value is None:
        return MetricComparison(
            metric_name=metric_name,
            unit=unit,
            lower_value=lower_value,
            higher_value=higher_value,
            absolute_change=None,
            relative_change=None,
            ratio=None,
            change_db=None,
            direction="unavailable",
            comparable=False,
            not_applicable_reason="representative_value_unavailable",
            diagnostics=(),
        )
    metric_incompatibility = _metric_incompatibility_reason(metric_name, pair_incompatibilities)
    diagnostics: list[str] = []
    if metric_incompatibility is not None:
        return MetricComparison(
            metric_name=metric_name,
            unit=unit,
            lower_value=lower_value,
            higher_value=higher_value,
            absolute_change=None,
            relative_change=None,
            ratio=None,
            change_db=None,
            direction="not_comparable",
            comparable=False,
            not_applicable_reason=metric_incompatibility,
            diagnostics=(),
        )
    if (
        (lower_summary.clipped_repeat_fraction or 0.0) > 0.0
        or (higher_summary.clipped_repeat_fraction or 0.0) > 0.0
    ) and _is_spectral_metric(metric_name):
        diagnostics.append("spectral_metrics_potentially_distorted_by_clipping")

    absolute_change = higher_value - lower_value
    tolerance = _tolerance_for(metric_name, settings)
    direction = _direction(lower_value, higher_value, tolerance)
    relative_change = None
    ratio = None
    change_db = None
    if _allows_relative_change(metric_name, unit) and not isclose(lower_value, 0.0, abs_tol=settings.numerical_tolerance):
        relative_change = absolute_change / abs(lower_value)
    if _allows_ratio(metric_name, unit) and lower_value > 0.0 and higher_value >= 0.0:
        ratio = higher_value / lower_value
        if ratio > 0.0 and _allows_change_db(metric_name, unit):
            factor = 20.0 if _is_amplitude_metric(metric_name) and "energy" not in metric_name else 10.0
            change_db = factor * log10(ratio)
    variability_ratio = _change_to_variability_ratio(absolute_change, lower_metric, higher_metric)
    return MetricComparison(
        metric_name=metric_name,
        unit=unit,
        lower_value=lower_value,
        higher_value=higher_value,
        absolute_change=absolute_change,
        relative_change=relative_change,
        ratio=ratio,
        change_db=change_db,
        direction=direction,
        comparable=True,
        not_applicable_reason=None,
        change_to_within_condition_variability_ratio=variability_ratio,
        diagnostics=tuple(dict.fromkeys(diagnostics)),
    )


def _not_comparable(metric_name: str, unit: str, reason: str) -> MetricComparison:
    return MetricComparison(
        metric_name=metric_name,
        unit=unit,
        lower_value=None,
        higher_value=None,
        absolute_change=None,
        relative_change=None,
        ratio=None,
        change_db=None,
        direction="unavailable",
        comparable=False,
        not_applicable_reason=reason,
        diagnostics=(),
    )


def _representative_value(metric: AggregatedMetric, settings: DynamicConditionComparisonSettings) -> float | None:
    return metric.median if settings.representative_statistic == "median" else metric.mean


def _metric_incompatibility_reason(metric_name: str, pair_incompatibilities: tuple[str, ...]) -> str | None:
    if metric_name.startswith("global_band_") or metric_name.startswith("time_band_"):
        for reason in pair_incompatibilities:
            if reason == "band_definition_mismatch":
                return reason
    if _is_amplitude_metric(metric_name) or _is_absolute_band_energy_metric(metric_name):
        for reason in pair_incompatibilities:
            if reason.startswith("instrumental_") or "clipping" in reason:
                return reason
    if _is_global_metric(metric_name):
        for reason in pair_incompatibilities:
            if reason == "global_spectral_configuration_mismatch":
                return reason
    if _is_time_metric(metric_name):
        for reason in pair_incompatibilities:
            if reason == "time_resolved_configuration_mismatch":
                return reason
    return None


def _is_amplitude_metric(metric_name: str) -> bool:
    return metric_name in {
        "excitation_peak_absolute_amplitude",
        "excitation_rms_amplitude",
        "excitation_signal_energy",
        "excitation_equivalent_level_dbfs",
        "global_total_spectral_energy",
    }


def _is_absolute_band_energy_metric(metric_name: str) -> bool:
    return (metric_name.startswith("global_band_") or metric_name.startswith("time_band_")) and metric_name.endswith("_energy")


def _is_global_metric(metric_name: str) -> bool:
    return metric_name.startswith("global_")


def _is_time_metric(metric_name: str) -> bool:
    return (
        metric_name.startswith("time_")
        or metric_name.startswith("early_")
        or metric_name.startswith("middle_")
        or metric_name.startswith("late_")
        or metric_name.startswith("region_")
    )


def _is_spectral_metric(metric_name: str) -> bool:
    return _is_global_metric(metric_name) or _is_time_metric(metric_name)


def _allows_relative_change(metric_name: str, unit: str) -> bool:
    if unit in {"dB", "dBFS", "dB/s"}:
        return False
    if "slope_per_s" in metric_name:
        return False
    return True


def _allows_ratio(metric_name: str, unit: str) -> bool:
    if unit in {"dB", "dBFS", "dB/s", "fraction/s"}:
        return False
    if "slope_per_s" in metric_name:
        return False
    return True


def _allows_change_db(metric_name: str, unit: str) -> bool:
    if unit in {"dB", "dBFS", "fraction", "count", "1/Hz", "1/octave"}:
        return False
    if metric_name.endswith("_fraction"):
        return False
    return _is_amplitude_metric(metric_name) or _is_absolute_band_energy_metric(metric_name)


def _change_to_variability_ratio(
    absolute_change: float,
    lower_metric: AggregatedMetric,
    higher_metric: AggregatedMetric,
) -> float | None:
    lower_std = lower_metric.standard_deviation if lower_metric.finite_count >= 2 else None
    higher_std = higher_metric.standard_deviation if higher_metric.finite_count >= 2 else None
    variances = [value * value for value in (lower_std, higher_std) if value is not None]
    if not variances:
        return None
    combined = sqrt(sum(variances))
    if isclose(combined, 0.0, abs_tol=1e-15):
        return None
    return abs(absolute_change) / combined


def _tolerance_for(metric_name: str, settings: DynamicConditionComparisonSettings) -> MetricTolerance:
    for tolerance in settings.metric_tolerances:
        if tolerance.metric_name == metric_name:
            return tolerance
    return MetricTolerance(metric_name=metric_name, absolute_tolerance=0.0, relative_tolerance=0.0)


def _direction(left: float, right: float, tolerance: MetricTolerance) -> Literal["increase", "decrease", "approximately_equal"]:
    sign = _change_sign(left, right, tolerance)
    if sign > 0:
        return "increase"
    if sign < 0:
        return "decrease"
    return "approximately_equal"


def _change_sign(left: float, right: float, tolerance: MetricTolerance) -> int:
    delta = right - left
    if abs(delta) <= tolerance.absolute_tolerance:
        return 0
    denominator = max(abs(left), abs(right), 1e-300)
    if abs(delta) / denominator <= tolerance.relative_tolerance:
        return 0
    return 1 if delta > 0.0 else -1


def _build_pairwise_comparisons(
    summaries: tuple[DynamicConditionSpectralSummary, ...],
    settings: DynamicConditionComparisonSettings,
) -> tuple[DynamicConditionPairComparison, ...]:
    if len(summaries) < 2:
        return ()
    pairs: list[DynamicConditionPairComparison] = []
    if settings.pair_comparison_policy == "adjacent_only":
        by_label = {summary.dynamic_label: summary for summary in summaries}
        for left, right in zip(DYNAMIC_LABEL_ORDER, DYNAMIC_LABEL_ORDER[1:]):
            if left in by_label and right in by_label:
                pairs.append(compare_dynamic_condition_pair(by_label[left], by_label[right], settings))
    else:
        for left, right in zip(summaries, summaries[1:]):
            pairs.append(compare_dynamic_condition_pair(left, right, settings))
    return tuple(pairs)


def _build_reference_comparisons(
    summaries: tuple[DynamicConditionSpectralSummary, ...],
    settings: DynamicConditionComparisonSettings,
) -> tuple[str | None, tuple[DynamicConditionPairComparison, ...], tuple[str, ...]]:
    if settings.reference_policy == "none" or not summaries:
        return None, (), ()
    by_label = {summary.dynamic_label: summary for summary in summaries}
    diagnostics: list[str] = []
    reference = by_label.get(settings.reference_dynamic_label)
    if reference is None and settings.reference_policy == "configured_or_lowest_available":
        reference = summaries[0]
        diagnostics.append(f"reference_dynamic_label_fallback:{reference.dynamic_label}")
    if reference is None:
        diagnostics.append(f"reference_dynamic_label_missing:{settings.reference_dynamic_label}")
        return None, (), tuple(diagnostics)
    pairs = []
    for summary in summaries:
        if _DYNAMIC_LABEL_INDEX[summary.dynamic_label] > _DYNAMIC_LABEL_INDEX[reference.dynamic_label]:
            pairs.append(compare_dynamic_condition_pair(reference, summary, settings))
    return reference.dynamic_label, tuple(pairs), tuple(diagnostics)


def _build_metric_sequences(
    summaries: tuple[DynamicConditionSpectralSummary, ...],
    settings: DynamicConditionComparisonSettings,
) -> tuple[DynamicMetricSequence, ...]:
    by_label = {summary.dynamic_label: summary for summary in summaries}
    metric_names = sorted(
        set(settings.enabled_metrics)
        if settings.enabled_metrics
        else {metric.metric_name for summary in summaries for metric in summary.all_metrics}
    )
    sequences: list[DynamicMetricSequence] = []
    for metric_name in metric_names:
        values: list[float | None] = []
        valid_mask: list[bool] = []
        unit = "unknown"
        for label in DYNAMIC_LABEL_ORDER:
            summary = by_label.get(label)
            metric = _metric_from_summary(summary, metric_name) if summary is not None else None
            if metric is not None:
                unit = metric.unit
                value = _representative_value(metric, settings)
            else:
                value = None
            values.append(value)
            valid_mask.append(value is not None)
        changes: list[MetricComparison] = []
        previous_summary = None
        for label in DYNAMIC_LABEL_ORDER:
            summary = by_label.get(label)
            if summary is None:
                continue
            if previous_summary is not None:
                lower_metric = _metric_from_summary(previous_summary, metric_name)
                higher_metric = _metric_from_summary(summary, metric_name)
                changes.append(
                    _compare_metric(
                        metric_name,
                        lower_metric,
                        higher_metric,
                        previous_summary,
                        summary,
                        settings,
                        _pair_incompatibilities(previous_summary, summary, settings),
                    )
                )
            previous_summary = summary
        monotonicity = evaluate_dynamic_metric_monotonicity(
            metric_name,
            unit,
            DYNAMIC_LABEL_ORDER,
            tuple(values),
            settings=settings,
        )
        sequences.append(
            DynamicMetricSequence(
                metric_name=metric_name,
                unit=unit,
                labels=DYNAMIC_LABEL_ORDER,
                values=tuple(values),
                valid_mask=tuple(valid_mask),
                pairwise_changes=tuple(changes),
                monotonicity=monotonicity.monotonicity,
                inversion_count=monotonicity.inversion_count,
                tie_count=monotonicity.tie_count,
                missing_count=sum(1 for valid in valid_mask if not valid),
                diagnostics=(),
            )
        )
    return tuple(sequences)


def _metric_from_summary(
    summary: DynamicConditionSpectralSummary | None,
    metric_name: str,
) -> AggregatedMetric | None:
    if summary is None:
        return None
    for metric in summary.all_metrics:
        if metric.metric_name == metric_name:
            return metric
    return None
