"""Time-resolved descriptive spectral characterization.

This module describes how global spectral metrics evolve after an impact. It
does not classify regimes, prove nonlinearity, identify chaos, or promote any
observation to a physical mode.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isclose, isfinite, log10
from typing import Literal

import numpy as np

from belllab.config import SpectrumAnalysisSettings
from belllab.global_spectrum import (
    GlobalSpectralCharacterizationSettings,
    SpectralBand,
    SpectralBandEnergy,
    characterize_global_spectrum,
)
from belllab.recording import Recording
from belllab.spectrum import analyze_spectrum
from belllab.types import Signal, Spectrum


_FRAME_POLICIES = frozenset({"invalidate", "process"})
_SMOOTHING_METHODS = frozenset({"none", "median", "mean"})
_CHANGE_METHODS = frozenset({"median_adjacent"})
_NORMALIZATION_POLICIES = frozenset({"none"})
_DEFAULT_CHANGE_THRESHOLDS = (
    ("energy_db", 6.0),
    ("spectral_flatness", 0.15),
    ("spectral_entropy", 0.15),
    ("tonal_energy_fraction", 0.20),
    ("peak_density_per_hz", 0.01),
    ("occupied_bandwidth_hz", 100.0),
)
_TREND_METRICS = (
    "energy_db",
    "spectral_centroid_hz",
    "spectral_spread_hz",
    "spectral_flatness",
    "spectral_entropy",
    "significant_peak_count",
    "peak_density_per_hz",
    "tonal_energy_fraction",
    "residual_energy_fraction",
    "occupied_bandwidth_hz",
)


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")


def _unique_strings(values: tuple[str, ...], name: str) -> None:
    if any(not isinstance(item, str) or not item.strip() for item in values):
        raise ValueError(f"{name} must contain nonempty strings.")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates.")


def _finite_optional(*values: float | None) -> bool:
    return all(value is None or isfinite(value) for value in values)


def _in_unit_interval(value: float | None) -> bool:
    return value is None or (isfinite(value) and 0.0 <= value <= 1.0)


@dataclass(frozen=True, slots=True)
class TimeResolvedSpectralCharacterizationSettings:
    """Explicit settings for frame-wise spectral characterization."""

    analysis_window_start_s: float = 0.0
    analysis_window_end_s: float | None = None
    frame_duration_s: float = 0.100
    hop_duration_s: float = 0.025
    fft_size: int | None = None
    window_name: Literal["rectangular", "hann"] = "hann"
    detrend_policy: Literal["none", "mean"] = "mean"
    pad_end: bool = False
    frequency_min_hz: float = 0.0
    frequency_max_hz: float | None = None
    minimum_bin_count: int = 3
    minimum_frame_energy: float = 0.0
    silent_frame_policy: Literal["invalidate", "process"] = "invalidate"
    normalization_between_frames: Literal["none"] = "none"
    peak_min_power: float | None = None
    peak_min_prominence: float | None = None
    peak_distance_bins: int | None = None
    peak_min_width_bins: float | None = None
    peak_max_width_bins: float | None = None
    tonal_neighborhood_width_factor: float = 1.0
    rolloff_fractions: tuple[float, ...] = (0.05, 0.50, 0.85, 0.90, 0.95)
    occupied_lower_fraction: float = 0.05
    occupied_upper_fraction: float = 0.95
    bands: tuple[SpectralBand, ...] = ()
    tonal_dominance_fraction_threshold: float = 0.50
    high_residual_fraction_threshold: float = 0.50
    band_presence_energy_fraction_threshold: float = 0.05
    early_window_s: tuple[float, float] | None = (0.0, 0.15)
    middle_window_s: tuple[float, float] | None = (0.15, 0.45)
    late_window_s: tuple[float, float] | None = (0.45, 0.90)
    smoothing_method: Literal["none", "median", "mean"] = "none"
    smoothing_window_frames: int = 3
    minimum_regression_frame_count: int = 3
    change_point_method: Literal["median_adjacent"] = "median_adjacent"
    change_point_window_frames: int = 2
    change_point_minimum_persistence_frames: int = 2
    change_point_thresholds: tuple[tuple[str, float], ...] = _DEFAULT_CHANGE_THRESHOLDS
    numerical_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        numeric = (
            self.analysis_window_start_s,
            self.frame_duration_s,
            self.hop_duration_s,
            self.frequency_min_hz,
            self.minimum_frame_energy,
            self.tonal_neighborhood_width_factor,
            self.occupied_lower_fraction,
            self.occupied_upper_fraction,
            self.tonal_dominance_fraction_threshold,
            self.high_residual_fraction_threshold,
            self.band_presence_energy_fraction_threshold,
            self.numerical_tolerance,
        )
        if any(not isfinite(value) for value in numeric):
            raise ValueError("time-resolved settings values must be finite.")
        if self.analysis_window_end_s is not None and not isfinite(self.analysis_window_end_s):
            raise ValueError("analysis_window_end_s must be finite when provided.")
        if (
            self.analysis_window_end_s is not None
            and self.analysis_window_end_s <= self.analysis_window_start_s
        ):
            raise ValueError("analysis window end must be above start.")
        if self.frame_duration_s <= 0 or self.hop_duration_s <= 0:
            raise ValueError("frame_duration_s and hop_duration_s must be positive.")
        if self.fft_size is not None and self.fft_size <= 0:
            raise ValueError("fft_size must be positive when provided.")
        if self.window_name not in {"rectangular", "hann"}:
            raise ValueError("unsupported window_name.")
        if self.detrend_policy not in {"none", "mean"}:
            raise ValueError("detrend_policy must be 'none' or 'mean'.")
        if not isinstance(self.pad_end, bool):
            raise ValueError("pad_end must be a boolean.")
        if self.frequency_min_hz < 0:
            raise ValueError("frequency_min_hz must not be negative.")
        if self.frequency_max_hz is not None and (
            not isfinite(self.frequency_max_hz)
            or self.frequency_max_hz <= self.frequency_min_hz
        ):
            raise ValueError("frequency_max_hz must be finite and above minimum.")
        if self.minimum_bin_count <= 0:
            raise ValueError("minimum_bin_count must be positive.")
        if self.minimum_frame_energy < 0:
            raise ValueError("minimum_frame_energy must not be negative.")
        if self.silent_frame_policy not in _FRAME_POLICIES:
            raise ValueError("silent_frame_policy is not recognized.")
        if self.normalization_between_frames not in _NORMALIZATION_POLICIES:
            raise ValueError("normalization_between_frames is not supported.")
        optional_nonnegative = (
            self.peak_min_power,
            self.peak_min_prominence,
            self.peak_min_width_bins,
            self.peak_max_width_bins,
        )
        if any(value is not None and (not isfinite(value) or value < 0) for value in optional_nonnegative):
            raise ValueError("peak thresholds and widths must be finite and non-negative.")
        if self.peak_distance_bins is not None and self.peak_distance_bins <= 0:
            raise ValueError("peak_distance_bins must be positive.")
        if (
            self.peak_min_width_bins is not None
            and self.peak_max_width_bins is not None
            and self.peak_min_width_bins > self.peak_max_width_bins
        ):
            raise ValueError("peak_min_width_bins must not exceed peak_max_width_bins.")
        if self.tonal_neighborhood_width_factor < 0 or self.numerical_tolerance < 0:
            raise ValueError("width factor and tolerance must not be negative.")
        if not 0 < self.occupied_lower_fraction < self.occupied_upper_fraction < 1:
            raise ValueError("occupied fractions must be strictly ordered in (0, 1).")
        if not self.rolloff_fractions or any(
            not isfinite(value) or not 0 < value < 1 for value in self.rolloff_fractions
        ):
            raise ValueError("rolloff fractions must be finite and in (0, 1).")
        if tuple(sorted(set(self.rolloff_fractions))) != self.rolloff_fractions:
            raise ValueError("rolloff fractions must be unique and ordered.")
        labels = [band.label for band in self.bands]
        if len(labels) != len(set(labels)):
            raise ValueError("band labels must be unique.")
        if tuple(sorted(self.bands, key=lambda band: band.frequency_start_hz)) != self.bands:
            raise ValueError("bands must be ordered.")
        if any(a.frequency_end_hz > b.frequency_start_hz for a, b in zip(self.bands, self.bands[1:])):
            raise ValueError("bands must not overlap.")
        for threshold in (
            self.tonal_dominance_fraction_threshold,
            self.high_residual_fraction_threshold,
            self.band_presence_energy_fraction_threshold,
        ):
            if not 0 <= threshold <= 1:
                raise ValueError("summary thresholds must be in [0, 1].")
        _validate_regions(self.early_window_s, self.middle_window_s, self.late_window_s)
        if self.smoothing_method not in _SMOOTHING_METHODS:
            raise ValueError("smoothing_method is not recognized.")
        if self.smoothing_window_frames <= 0:
            raise ValueError("smoothing_window_frames must be positive.")
        if self.smoothing_method == "median" and self.smoothing_window_frames % 2 == 0:
            raise ValueError("median smoothing requires an odd window.")
        if self.minimum_regression_frame_count < 0:
            raise ValueError("minimum_regression_frame_count must not be negative.")
        if self.change_point_method not in _CHANGE_METHODS:
            raise ValueError("change_point_method is not recognized.")
        if self.change_point_window_frames <= 0:
            raise ValueError("change_point_window_frames must be positive.")
        if self.change_point_minimum_persistence_frames <= 0:
            raise ValueError("change_point_minimum_persistence_frames must be positive.")
        metric_names = [name for name, _ in self.change_point_thresholds]
        if len(metric_names) != len(set(metric_names)):
            raise ValueError("change point metric names must be unique.")
        if any(
            not name.strip() or not isfinite(threshold) or threshold < 0
            for name, threshold in self.change_point_thresholds
        ):
            raise ValueError("change point thresholds must be finite and non-negative.")


@dataclass(frozen=True, slots=True)
class SpectralMetricTemporalFit:
    """Linear descriptive fit for one time-varying spectral metric."""

    metric_name: str
    success: bool
    method: str | None
    slope_per_s: float | None
    intercept: float | None
    r_squared: float | None
    rmse: float | None
    available_point_count: int
    finite_point_count: int
    used_point_count: int
    start_time_s: float | None
    end_time_s: float | None
    failure_reason: str | None = None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.metric_name, "metric_name")
        counts = (
            self.available_point_count,
            self.finite_point_count,
            self.used_point_count,
        )
        if min(counts) < 0 or self.used_point_count > self.finite_point_count:
            raise ValueError("fit point counts are incoherent.")
        if self.finite_point_count > self.available_point_count:
            raise ValueError("fit finite count exceeds available count.")
        if not _finite_optional(
            self.slope_per_s,
            self.intercept,
            self.r_squared,
            self.rmse,
            self.start_time_s,
            self.end_time_s,
        ):
            raise ValueError("fit numeric values must be finite when present.")
        if self.rmse is not None and self.rmse < 0:
            raise ValueError("fit RMSE must not be negative.")
        if self.r_squared is not None and self.r_squared < -self._r2_tolerance():
            raise ValueError("fit r_squared is invalid.")
        if self.start_time_s is not None and self.end_time_s is not None and self.end_time_s < self.start_time_s:
            raise ValueError("fit times are inverted.")
        if self.success:
            if self.failure_reason is not None:
                raise ValueError("successful fit must not have failure_reason.")
            if self.method is None or self.slope_per_s is None or self.intercept is None or self.rmse is None:
                raise ValueError("successful fit lacks regression fields.")
        else:
            if not self.failure_reason:
                raise ValueError("failed fit requires failure_reason.")
        _unique_strings(self.diagnostics, "fit diagnostics")

    @staticmethod
    def _r2_tolerance() -> float:
        return 1e-12


@dataclass(frozen=True, slots=True)
class SpectralMetricChangePoint:
    """Operational change point in a descriptive metric series."""

    metric_name: str
    frame_index: int
    time_s: float
    value_before: float
    value_after: float
    difference: float
    relative_difference: float | None
    threshold: float
    direction: Literal["increase", "decrease", "unchanged"]
    persistence_frame_count: int
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.metric_name, "metric_name")
        if self.frame_index < 0 or self.persistence_frame_count <= 0:
            raise ValueError("change point frame and persistence counts are invalid.")
        values = (
            self.time_s,
            self.value_before,
            self.value_after,
            self.difference,
            self.threshold,
        )
        if not all(isfinite(value) for value in values) or self.threshold < 0:
            raise ValueError("change point values must be finite and threshold non-negative.")
        if self.relative_difference is not None and not isfinite(self.relative_difference):
            raise ValueError("relative_difference must be finite when present.")
        if self.direction not in {"increase", "decrease", "unchanged"}:
            raise ValueError("change point direction is not recognized.")
        _unique_strings(self.diagnostics, "change point diagnostics")


@dataclass(frozen=True, slots=True)
class SpectralTemporalRegionSummary:
    """Robust median summary for a configured relative-time region."""

    region: str
    start_relative_s: float
    end_relative_s: float
    frame_count: int
    valid_frame_count: int
    median_energy: float | None
    median_centroid_hz: float | None
    median_spread_hz: float | None
    median_flatness: float | None
    median_entropy: float | None
    median_peak_count: float | None
    median_peak_density_per_hz: float | None
    median_tonal_energy_fraction: float | None
    median_residual_energy_fraction: float | None
    median_occupied_bandwidth_hz: float | None
    valid: bool
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.region, "region")
        if not all(isfinite(value) for value in (self.start_relative_s, self.end_relative_s)):
            raise ValueError("region times must be finite.")
        if self.end_relative_s <= self.start_relative_s:
            raise ValueError("region times must be ordered.")
        if self.frame_count < 0 or self.valid_frame_count < 0 or self.valid_frame_count > self.frame_count:
            raise ValueError("region frame counts are incoherent.")
        if self.valid != (self.valid_frame_count > 0):
            raise ValueError("region validity must follow valid_frame_count.")
        if not _finite_optional(
            self.median_energy,
            self.median_centroid_hz,
            self.median_spread_hz,
            self.median_flatness,
            self.median_entropy,
            self.median_peak_count,
            self.median_peak_density_per_hz,
            self.median_tonal_energy_fraction,
            self.median_residual_energy_fraction,
            self.median_occupied_bandwidth_hz,
        ):
            raise ValueError("region metrics must be finite when present.")
        for value in (
            self.median_flatness,
            self.median_entropy,
            self.median_tonal_energy_fraction,
            self.median_residual_energy_fraction,
        ):
            if not _in_unit_interval(value):
                raise ValueError("region fractions must lie in [0, 1].")
        _unique_strings(self.diagnostics, "region diagnostics")


@dataclass(frozen=True, slots=True)
class TimeResolvedSpectralBandSummary:
    """Frame-wise persistence summary for one configured spectral band."""

    label: str
    frequency_start_hz: float
    frequency_end_hz: float
    initial_energy: float | None
    final_energy: float | None
    maximum_energy: float | None
    initial_energy_fraction: float | None
    final_energy_fraction: float | None
    energy_fraction_trend: SpectralMetricTemporalFit
    coverage_fraction: float
    time_until_below_threshold_s: float | None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.label, "label")
        if not all(isfinite(value) for value in (self.frequency_start_hz, self.frequency_end_hz, self.coverage_fraction)):
            raise ValueError("band summary values must be finite.")
        if self.frequency_end_hz <= self.frequency_start_hz:
            raise ValueError("band summary frequencies are inverted.")
        if not 0 <= self.coverage_fraction <= 1:
            raise ValueError("band coverage must be in [0, 1].")
        if not _finite_optional(
            self.initial_energy,
            self.final_energy,
            self.maximum_energy,
            self.initial_energy_fraction,
            self.final_energy_fraction,
            self.time_until_below_threshold_s,
        ):
            raise ValueError("band summary optional values must be finite.")
        for value in (
            self.initial_energy,
            self.final_energy,
            self.maximum_energy,
        ):
            if value is not None and value < 0:
                raise ValueError("band energies must not be negative.")
        for value in (self.initial_energy_fraction, self.final_energy_fraction):
            if not _in_unit_interval(value):
                raise ValueError("band fractions must lie in [0, 1].")
        _unique_strings(self.diagnostics, "band summary diagnostics")


@dataclass(frozen=True, slots=True)
class TimeResolvedSpectralFrame:
    """Descriptive spectral metrics for one complete analysis frame."""

    frame_index: int
    center_time_s: float
    relative_time_s: float
    start_time_s: float
    end_time_s: float
    sample_count: int
    spectral_energy: float | None
    spectral_centroid_hz: float | None
    spectral_spread_hz: float | None
    spectral_rolloff_50_hz: float | None
    spectral_rolloff_85_hz: float | None
    spectral_rolloff_95_hz: float | None
    spectral_flatness: float | None
    spectral_entropy: float | None
    spectral_crest_factor: float | None
    significant_peak_count: int
    peak_density_per_hz: float | None
    peak_density_per_octave: float | None
    tonal_energy_fraction: float | None
    residual_energy_fraction: float | None
    occupied_bandwidth_hz: float | None
    occupied_frequency_fraction: float | None
    dominant_frequency_hz: float | None
    dominant_bin_frequency_hz: float | None
    dominant_peak_frequency_hz: float | None
    dominant_peak_power_fraction: float | None
    band_energy_metrics: tuple[SpectralBandEnergy, ...]
    valid: bool
    failure_reason: str | None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index must not be negative.")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive.")
        if not all(isfinite(value) for value in (self.center_time_s, self.relative_time_s, self.start_time_s, self.end_time_s)):
            raise ValueError("frame times must be finite.")
        if not self.start_time_s <= self.center_time_s <= self.end_time_s:
            raise ValueError("frame center must lie inside frame bounds.")
        if self.end_time_s <= self.start_time_s:
            raise ValueError("frame bounds must be ordered.")
        if self.significant_peak_count < 0:
            raise ValueError("peak count must not be negative.")
        if not _finite_optional(
            self.spectral_energy,
            self.spectral_centroid_hz,
            self.spectral_spread_hz,
            self.spectral_rolloff_50_hz,
            self.spectral_rolloff_85_hz,
            self.spectral_rolloff_95_hz,
            self.spectral_flatness,
            self.spectral_entropy,
            self.spectral_crest_factor,
            self.peak_density_per_hz,
            self.peak_density_per_octave,
            self.tonal_energy_fraction,
            self.residual_energy_fraction,
            self.occupied_bandwidth_hz,
            self.occupied_frequency_fraction,
            self.dominant_frequency_hz,
            self.dominant_bin_frequency_hz,
            self.dominant_peak_frequency_hz,
            self.dominant_peak_power_fraction,
        ):
            raise ValueError("frame numeric metrics must be finite when present.")
        for value in (
            self.spectral_flatness,
            self.spectral_entropy,
            self.tonal_energy_fraction,
            self.residual_energy_fraction,
            self.occupied_frequency_fraction,
            self.dominant_peak_power_fraction,
        ):
            if not _in_unit_interval(value):
                raise ValueError("frame fractions must lie in [0, 1].")
        if self.spectral_energy is not None and self.spectral_energy < 0:
            raise ValueError("spectral_energy must not be negative.")
        if self.spectral_spread_hz is not None and self.spectral_spread_hz < 0:
            raise ValueError("spectral_spread_hz must not be negative.")
        if self.spectral_crest_factor is not None and self.spectral_crest_factor < 1:
            raise ValueError("spectral_crest_factor must be at least one.")
        if self.tonal_energy_fraction is not None and not isclose(
            self.tonal_energy_fraction + (self.residual_energy_fraction or 0.0),
            1.0,
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise ValueError("tonal and residual fractions must sum to one.")
        if self.valid == (self.failure_reason is not None):
            raise ValueError("frame validity and failure_reason are incoherent.")
        _unique_strings(self.diagnostics, "frame diagnostics")


@dataclass(frozen=True, slots=True)
class TimeResolvedSpectralSummary:
    """Summary of valid frame metrics without physical regime classification."""

    valid_start_time_s: float | None
    valid_end_time_s: float | None
    covered_duration_s: float
    valid_frame_count: int
    temporal_coverage_fraction: float
    initial_energy: float | None
    final_energy: float | None
    maximum_energy: float | None
    maximum_energy_time_s: float | None
    initial_flatness: float | None
    final_flatness: float | None
    minimum_flatness: float | None
    maximum_flatness: float | None
    initial_entropy: float | None
    final_entropy: float | None
    minimum_entropy: float | None
    maximum_entropy: float | None
    initial_centroid_hz: float | None
    final_centroid_hz: float | None
    minimum_centroid_hz: float | None
    maximum_centroid_hz: float | None
    initial_tonal_energy_fraction: float | None
    final_tonal_energy_fraction: float | None
    initial_peak_density_per_hz: float | None
    final_peak_density_per_hz: float | None
    initial_occupied_bandwidth_hz: float | None
    final_occupied_bandwidth_hz: float | None
    tonal_dominated_frame_fraction: float | None
    high_residual_frame_fraction: float | None
    tonal_fraction_change: float | None
    residual_fraction_change: float | None
    flatness_change: float | None
    entropy_change: float | None
    peak_density_change: float | None
    regions: tuple[SpectralTemporalRegionSummary, ...]
    band_summaries: tuple[TimeResolvedSpectralBandSummary, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.valid_frame_count < 0:
            raise ValueError("valid_frame_count must not be negative.")
        if self.covered_duration_s < 0:
            raise ValueError("covered_duration_s must not be negative.")
        if not 0 <= self.temporal_coverage_fraction <= 1:
            raise ValueError("temporal_coverage_fraction must be in [0, 1].")
        if not _finite_optional(
            self.valid_start_time_s,
            self.valid_end_time_s,
            self.covered_duration_s,
            self.initial_energy,
            self.final_energy,
            self.maximum_energy,
            self.maximum_energy_time_s,
            self.initial_flatness,
            self.final_flatness,
            self.minimum_flatness,
            self.maximum_flatness,
            self.initial_entropy,
            self.final_entropy,
            self.minimum_entropy,
            self.maximum_entropy,
            self.initial_centroid_hz,
            self.final_centroid_hz,
            self.minimum_centroid_hz,
            self.maximum_centroid_hz,
            self.initial_tonal_energy_fraction,
            self.final_tonal_energy_fraction,
            self.initial_peak_density_per_hz,
            self.final_peak_density_per_hz,
            self.initial_occupied_bandwidth_hz,
            self.final_occupied_bandwidth_hz,
            self.tonal_dominated_frame_fraction,
            self.high_residual_frame_fraction,
            self.tonal_fraction_change,
            self.residual_fraction_change,
            self.flatness_change,
            self.entropy_change,
            self.peak_density_change,
        ):
            raise ValueError("summary numeric fields must be finite when present.")
        for value in (
            self.initial_flatness,
            self.final_flatness,
            self.minimum_flatness,
            self.maximum_flatness,
            self.initial_entropy,
            self.final_entropy,
            self.minimum_entropy,
            self.maximum_entropy,
            self.initial_tonal_energy_fraction,
            self.final_tonal_energy_fraction,
            self.tonal_dominated_frame_fraction,
            self.high_residual_frame_fraction,
        ):
            if not _in_unit_interval(value):
                raise ValueError("summary fractions must lie in [0, 1].")
        if self.valid_start_time_s is not None and self.valid_end_time_s is not None and self.valid_end_time_s < self.valid_start_time_s:
            raise ValueError("summary valid times are inverted.")
        _unique_strings(self.diagnostics, "summary diagnostics")


@dataclass(frozen=True, slots=True)
class TimeResolvedSpectralCharacterization:
    """Top-level immutable result for time-resolved spectral evolution."""

    recording_id: str
    impact_time_s: float
    analysis_start_time_s: float
    analysis_end_time_s: float
    sample_rate_hz: int
    fft_size: int
    frame_count: int
    valid_frame_count: int
    discarded_frame_count: int
    frame_duration_s: float
    hop_duration_s: float
    frequency_resolution_hz: float
    frequency_min_hz: float
    frequency_max_hz: float | None
    frames: tuple[TimeResolvedSpectralFrame, ...]
    summary: TimeResolvedSpectralSummary
    temporal_trends: tuple[SpectralMetricTemporalFit, ...]
    change_points: tuple[SpectralMetricChangePoint, ...]
    settings: TimeResolvedSpectralCharacterizationSettings
    valid: bool
    failure_reason: str | None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.recording_id, "recording_id")
        if not all(isfinite(value) for value in (
            self.impact_time_s,
            self.analysis_start_time_s,
            self.analysis_end_time_s,
            self.frame_duration_s,
            self.hop_duration_s,
            self.frequency_resolution_hz,
            self.frequency_min_hz,
        )):
            raise ValueError("time-resolved result values must be finite.")
        if self.analysis_end_time_s < self.analysis_start_time_s:
            raise ValueError("analysis interval is inverted.")
        if self.sample_rate_hz <= 0 or self.fft_size <= 0:
            raise ValueError("sample rate and fft size must be positive.")
        if self.frame_count != len(self.frames):
            raise ValueError("frame_count must match frames.")
        if self.valid_frame_count + self.discarded_frame_count != self.frame_count:
            raise ValueError("valid plus discarded frame counts must equal total.")
        if self.valid_frame_count != sum(frame.valid for frame in self.frames):
            raise ValueError("valid_frame_count does not match frames.")
        if self.discarded_frame_count != sum(not frame.valid for frame in self.frames):
            raise ValueError("discarded_frame_count does not match frames.")
        indices = tuple(frame.frame_index for frame in self.frames)
        if indices != tuple(range(len(self.frames))):
            raise ValueError("frame indices must be unique and sequential.")
        if any(later < earlier for earlier, later in zip(
            (frame.center_time_s for frame in self.frames),
            (frame.center_time_s for frame in self.frames[1:]),
        )):
            raise ValueError("frame times must be ordered.")
        if self.valid != (self.failure_reason is None):
            raise ValueError("result validity and failure_reason are incoherent.")
        _unique_strings(self.diagnostics, "result diagnostics")


@dataclass(frozen=True, slots=True)
class TimeResolvedSpectralComparabilityResult:
    """Compatibility diagnostics only; no metric comparison is performed."""

    comparable: bool
    incompatibilities: tuple[str, ...]

    def __post_init__(self) -> None:
        _unique_strings(self.incompatibilities, "comparability incompatibilities")
        if self.comparable == bool(self.incompatibilities):
            raise ValueError("comparability state is incoherent.")


def characterize_time_resolved_spectrum(
    signal: Signal,
    impact_time_s: float,
    settings: TimeResolvedSpectralCharacterizationSettings | None = None,
    *,
    recording_id: str = "signal",
) -> TimeResolvedSpectralCharacterization:
    """Characterize frame-wise spectral evolution around an impact."""
    cfg = settings or TimeResolvedSpectralCharacterizationSettings()
    _text(recording_id, "recording_id")
    if not isfinite(impact_time_s):
        raise ValueError("impact_time_s must be finite.")
    if signal.sample_rate <= 0:
        raise ValueError("signal sample rate must be positive.")
    frame_samples = int(round(cfg.frame_duration_s * signal.sample_rate))
    hop_samples = int(round(cfg.hop_duration_s * signal.sample_rate))
    if frame_samples <= 0 or hop_samples <= 0:
        raise ValueError("frame and hop durations must contain at least one sample.")
    fft_size = cfg.fft_size or frame_samples
    if fft_size < frame_samples:
        raise ValueError("fft_size must be at least the frame sample count.")

    layout = _frame_layout(signal, impact_time_s, cfg, frame_samples, hop_samples)
    diagnostics: list[str] = [
        "evolucao_metricas_espectrais_nao_e_transicao_fisica_comprovada",
        "time_resolved_metrics_do_not_prove_nonlinearity",
        "time_resolved_peaks_are_not_modal_modes",
        "canonical_domain:linear_power_via_global_characterization",
        "frame_policy:complete_frames_without_temporal_padding_by_default",
        "normalization_between_frames:none",
    ]
    diagnostics.extend(layout.diagnostics)
    if cfg.hop_duration_s > cfg.frame_duration_s:
        diagnostics.append("hop_duration_exceeds_frame_duration")
    if cfg.smoothing_method != "none":
        diagnostics.append(f"smoothing:{cfg.smoothing_method}_window_{cfg.smoothing_window_frames}")

    frames = tuple(
        _characterize_frame(
            signal,
            cfg,
            recording_id,
            impact_time_s,
            frame_index,
            start_index,
            frame_samples,
            fft_size,
            padded,
        )
        for frame_index, (start_index, padded) in enumerate(layout.frame_starts)
    )
    summary = _build_summary(frames, cfg)
    trends = tuple(_fit_metric(frames, metric, cfg) for metric in _TREND_METRICS)
    change_points = _detect_change_points(frames, cfg)
    valid_count = sum(frame.valid for frame in frames)
    failure_reason = None
    if not frames:
        failure_reason = "no_complete_frames_in_analysis_window"
        diagnostics.append("no_complete_frames_in_analysis_window")
    elif valid_count == 0:
        failure_reason = "all_frames_invalid"
        diagnostics.append("all_frames_invalid")
    return TimeResolvedSpectralCharacterization(
        recording_id=recording_id,
        impact_time_s=impact_time_s,
        analysis_start_time_s=layout.analysis_start_s,
        analysis_end_time_s=layout.analysis_end_s,
        sample_rate_hz=signal.sample_rate,
        fft_size=fft_size,
        frame_count=len(frames),
        valid_frame_count=valid_count,
        discarded_frame_count=len(frames) - valid_count,
        frame_duration_s=frame_samples / signal.sample_rate,
        hop_duration_s=hop_samples / signal.sample_rate,
        frequency_resolution_hz=signal.sample_rate / frame_samples,
        frequency_min_hz=cfg.frequency_min_hz,
        frequency_max_hz=cfg.frequency_max_hz,
        frames=frames,
        summary=summary,
        temporal_trends=trends,
        change_points=change_points,
        settings=cfg,
        valid=failure_reason is None,
        failure_reason=failure_reason,
        diagnostics=tuple(dict.fromkeys(diagnostics)),
    )


def characterize_recording_time_resolved_spectrum(
    recording: Recording,
    impact_time_s: float,
    settings: TimeResolvedSpectralCharacterizationSettings | None = None,
    *,
    recording_id: str | None = None,
) -> TimeResolvedSpectralCharacterization:
    """Adapter for an already loaded Recording."""
    return characterize_time_resolved_spectrum(
        recording.signal,
        impact_time_s,
        settings,
        recording_id=recording_id or recording.bell_id,
    )


def evaluate_time_resolved_spectral_comparability(
    first: TimeResolvedSpectralCharacterization,
    second: TimeResolvedSpectralCharacterization,
) -> TimeResolvedSpectralComparabilityResult:
    """Report direct-comparability issues without comparing signals."""
    checks = (
        ("sample_rate", first.sample_rate_hz, second.sample_rate_hz),
        ("frame_duration", first.frame_duration_s, second.frame_duration_s),
        ("hop_duration", first.hop_duration_s, second.hop_duration_s),
        ("fft_size", first.fft_size, second.fft_size),
        ("window", first.settings.window_name, second.settings.window_name),
        ("detrending", first.settings.detrend_policy, second.settings.detrend_policy),
        ("frequency_range", (first.settings.frequency_min_hz, first.settings.frequency_max_hz), (second.settings.frequency_min_hz, second.settings.frequency_max_hz)),
        ("analysis_window", (first.settings.analysis_window_start_s, first.settings.analysis_window_end_s), (second.settings.analysis_window_start_s, second.settings.analysis_window_end_s)),
        ("impact_time", first.impact_time_s, second.impact_time_s),
        ("peak_criteria", _peak_criteria(first.settings), _peak_criteria(second.settings)),
        ("normalization", first.settings.normalization_between_frames, second.settings.normalization_between_frames),
        ("regions", _regions(first.settings), _regions(second.settings)),
        ("smoothing", (first.settings.smoothing_method, first.settings.smoothing_window_frames), (second.settings.smoothing_method, second.settings.smoothing_window_frames)),
        ("change_points", (first.settings.change_point_method, first.settings.change_point_window_frames, first.settings.change_point_thresholds), (second.settings.change_point_method, second.settings.change_point_window_frames, second.settings.change_point_thresholds)),
    )
    issues = [f"incompatible_{label}" for label, left, right in checks if left != right]
    return TimeResolvedSpectralComparabilityResult(not issues, tuple(issues))


@dataclass(frozen=True, slots=True)
class _FrameLayout:
    analysis_start_s: float
    analysis_end_s: float
    frame_starts: tuple[tuple[int, bool], ...]
    diagnostics: tuple[str, ...]


def _frame_layout(
    signal: Signal,
    impact_time_s: float,
    cfg: TimeResolvedSpectralCharacterizationSettings,
    frame_samples: int,
    hop_samples: int,
) -> _FrameLayout:
    requested_start = impact_time_s + cfg.analysis_window_start_s
    requested_end = (
        signal.duration
        if cfg.analysis_window_end_s is None
        else impact_time_s + cfg.analysis_window_end_s
    )
    diagnostics: list[str] = []
    if requested_start < 0.0:
        diagnostics.append("analysis_start_clipped_to_signal")
    if requested_end > signal.duration:
        diagnostics.append("analysis_end_clipped_to_signal")
    if requested_end <= 0.0:
        return _FrameLayout(0.0, 0.0, (), ("analysis_window_outside_signal",))
    if requested_start >= signal.duration:
        return _FrameLayout(
            signal.duration,
            signal.duration,
            (),
            ("analysis_window_outside_signal",),
        )
    start_s = max(0.0, requested_start)
    end_s = min(signal.duration, requested_end)
    if end_s <= start_s:
        return _FrameLayout(start_s, end_s, (), ("analysis_window_outside_signal",))
    start_index = int(np.ceil(start_s * signal.sample_rate - cfg.numerical_tolerance))
    end_index = int(np.floor(end_s * signal.sample_rate + cfg.numerical_tolerance))
    frame_starts: list[tuple[int, bool]] = []
    cursor = start_index
    discarded_samples = 0
    while cursor < end_index:
        complete = cursor + frame_samples <= end_index
        if complete:
            frame_starts.append((cursor, False))
            cursor += hop_samples
            continue
        if cfg.pad_end:
            frame_starts.append((cursor, True))
            diagnostics.append("temporal_padding_applied")
        else:
            discarded_samples = end_index - cursor
            diagnostics.append(f"final_incomplete_frame_discarded_samples={discarded_samples}")
        break
    if not frame_starts and not cfg.pad_end:
        diagnostics.append("segment_shorter_than_frame_duration")
    return _FrameLayout(
        start_index / signal.sample_rate,
        end_index / signal.sample_rate,
        tuple(frame_starts),
        tuple(dict.fromkeys(diagnostics)),
    )


def _characterize_frame(
    signal: Signal,
    cfg: TimeResolvedSpectralCharacterizationSettings,
    recording_id: str,
    impact_time_s: float,
    frame_index: int,
    start_index: int,
    frame_samples: int,
    fft_size: int,
    padded: bool,
) -> TimeResolvedSpectralFrame:
    start_s = start_index / signal.sample_rate
    end_s = (start_index + frame_samples) / signal.sample_rate
    center_s = start_s + 0.5 * frame_samples / signal.sample_rate
    diagnostics: list[str] = [
        "frame_metric_not_regime_classification",
        "dominant_bin_is_not_modal_identity",
    ]
    if padded:
        diagnostics.append("temporal_zero_padding_applied")
    matrix = np.asarray(signal.samples)
    stop_index = min(start_index + frame_samples, matrix.shape[1])
    segment = matrix[:, start_index:stop_index]
    if segment.size == 0:
        return _invalid_frame(
            frame_index, center_s, impact_time_s, start_s, end_s, frame_samples,
            "insufficient_frame_samples", tuple(diagnostics),
        )
    if not np.all(np.isfinite(segment.astype(float))):
        diagnostics.append("nonfinite_frame_samples")
        return _invalid_frame(
            frame_index, center_s, impact_time_s, start_s, end_s, frame_samples,
            "nonfinite_frame_samples", tuple(diagnostics),
        )
    if padded:
        pad_width = frame_samples - segment.shape[1]
        segment = np.pad(segment.astype(float), ((0, 0), (0, pad_width)))
    frame_signal = Signal(
        samples=tuple(tuple(float(value) for value in channel) for channel in segment),
        sample_rate=signal.sample_rate,
        time=tuple(index / signal.sample_rate for index in range(frame_samples)),
        duration=frame_samples / signal.sample_rate,
        channels=signal.channels,
        unit=signal.unit,
    )
    try:
        spectrum = analyze_spectrum(
            frame_signal,
            SpectrumAnalysisSettings(
                remove_mean=cfg.detrend_policy == "mean",
                window_name=cfg.window_name,
                n_fft=fft_size,
                scale="linear_amplitude",
            ),
        ).spectrum
        adjusted = replace(
            spectrum,
            timestamp=center_s,
            interval_start_s=start_s,
            interval_end_s=end_s,
        )
        global_result = characterize_global_spectrum(
            adjusted,
            _global_settings(cfg),
            recording_id=f"{recording_id}:frame:{frame_index}",
        )
    except ValueError as exc:
        diagnostics.append("frame_global_characterization_failed")
        return _invalid_frame(
            frame_index,
            center_s,
            impact_time_s,
            start_s,
            end_s,
            frame_samples,
            str(exc),
            tuple(dict.fromkeys(diagnostics)),
        )
    diagnostics.extend(global_result.diagnostics)
    low_energy = (
        global_result.valid
        and global_result.total_spectral_energy < cfg.minimum_frame_energy
        and cfg.silent_frame_policy == "invalidate"
    )
    if not global_result.valid or low_energy:
        reason = (
            "frame_energy_below_threshold"
            if low_energy else global_result.failure_reason or "invalid_spectral_frame"
        )
        if low_energy:
            diagnostics.append("frame_energy_below_threshold")
        return _invalid_frame(
            frame_index,
            center_s,
            impact_time_s,
            start_s,
            end_s,
            frame_samples,
            reason,
            tuple(dict.fromkeys(diagnostics)),
            spectral_energy=global_result.total_spectral_energy,
        )
    dominant_bin, dominant_peak, dominant_fraction, dominant_diagnostics = _dominant_frequencies(
        adjusted,
        global_result,
        cfg,
    )
    diagnostics.extend(dominant_diagnostics)
    return TimeResolvedSpectralFrame(
        frame_index=frame_index,
        center_time_s=center_s,
        relative_time_s=center_s - impact_time_s,
        start_time_s=start_s,
        end_time_s=end_s,
        sample_count=frame_samples,
        spectral_energy=global_result.total_spectral_energy,
        spectral_centroid_hz=global_result.spectral_centroid_hz,
        spectral_spread_hz=global_result.spectral_spread_hz,
        spectral_rolloff_50_hz=global_result.spectral_rolloff_50_hz,
        spectral_rolloff_85_hz=global_result.spectral_rolloff_85_hz,
        spectral_rolloff_95_hz=global_result.spectral_rolloff_95_hz,
        spectral_flatness=global_result.spectral_flatness,
        spectral_entropy=global_result.spectral_entropy,
        spectral_crest_factor=global_result.spectral_crest_factor,
        significant_peak_count=global_result.significant_peak_count,
        peak_density_per_hz=global_result.peak_density_per_hz,
        peak_density_per_octave=global_result.peak_density_per_octave,
        tonal_energy_fraction=global_result.tonal_energy_fraction,
        residual_energy_fraction=global_result.residual_energy_fraction,
        occupied_bandwidth_hz=global_result.occupied_bandwidth_hz,
        occupied_frequency_fraction=global_result.occupied_frequency_fraction,
        dominant_frequency_hz=dominant_bin,
        dominant_bin_frequency_hz=dominant_bin,
        dominant_peak_frequency_hz=dominant_peak,
        dominant_peak_power_fraction=dominant_fraction,
        band_energy_metrics=global_result.band_energy_metrics,
        valid=True,
        failure_reason=None,
        diagnostics=tuple(dict.fromkeys(diagnostics)),
    )


def _invalid_frame(
    frame_index: int,
    center_s: float,
    impact_time_s: float,
    start_s: float,
    end_s: float,
    sample_count: int,
    reason: str,
    diagnostics: tuple[str, ...],
    *,
    spectral_energy: float | None = None,
) -> TimeResolvedSpectralFrame:
    _text(reason, "failure_reason")
    return TimeResolvedSpectralFrame(
        frame_index=frame_index,
        center_time_s=center_s,
        relative_time_s=center_s - impact_time_s,
        start_time_s=start_s,
        end_time_s=end_s,
        sample_count=sample_count,
        spectral_energy=spectral_energy,
        spectral_centroid_hz=None,
        spectral_spread_hz=None,
        spectral_rolloff_50_hz=None,
        spectral_rolloff_85_hz=None,
        spectral_rolloff_95_hz=None,
        spectral_flatness=None,
        spectral_entropy=None,
        spectral_crest_factor=None,
        significant_peak_count=0,
        peak_density_per_hz=None,
        peak_density_per_octave=None,
        tonal_energy_fraction=None,
        residual_energy_fraction=None,
        occupied_bandwidth_hz=None,
        occupied_frequency_fraction=None,
        dominant_frequency_hz=None,
        dominant_bin_frequency_hz=None,
        dominant_peak_frequency_hz=None,
        dominant_peak_power_fraction=None,
        band_energy_metrics=(),
        valid=False,
        failure_reason=reason,
        diagnostics=tuple(dict.fromkeys((*diagnostics, reason))),
    )


def _global_settings(
    cfg: TimeResolvedSpectralCharacterizationSettings,
) -> GlobalSpectralCharacterizationSettings:
    return GlobalSpectralCharacterizationSettings(
        frequency_min_hz=cfg.frequency_min_hz,
        frequency_max_hz=cfg.frequency_max_hz,
        detrend_policy=cfg.detrend_policy,
        window_name=cfg.window_name,
        fft_size=cfg.fft_size,
        spectral_input_domain="linear_amplitude",
        minimum_bin_count=cfg.minimum_bin_count,
        peak_min_power=cfg.peak_min_power,
        peak_min_prominence=cfg.peak_min_prominence,
        peak_distance_bins=cfg.peak_distance_bins,
        peak_min_width_bins=cfg.peak_min_width_bins,
        peak_max_width_bins=cfg.peak_max_width_bins,
        tonal_neighborhood_width_factor=cfg.tonal_neighborhood_width_factor,
        rolloff_fractions=cfg.rolloff_fractions,
        occupied_lower_fraction=cfg.occupied_lower_fraction,
        occupied_upper_fraction=cfg.occupied_upper_fraction,
        bands=cfg.bands,
        numerical_tolerance=cfg.numerical_tolerance,
    )


def _dominant_frequencies(
    spectrum: Spectrum,
    global_result,
    cfg: TimeResolvedSpectralCharacterizationSettings,
) -> tuple[float | None, float | None, float | None, tuple[str, ...]]:
    frequencies = np.asarray(spectrum.frequencies_hz, dtype=float)
    amplitudes = np.asarray(spectrum.magnitudes, dtype=float)
    upper = cfg.frequency_max_hz if cfg.frequency_max_hz is not None else float(frequencies[-1])
    mask = (frequencies >= cfg.frequency_min_hz) & (frequencies <= upper) & np.isfinite(amplitudes)
    if not np.any(mask):
        return None, None, None, ("dominant_bin_unavailable",)
    selected_indices = np.flatnonzero(mask)
    power = amplitudes[selected_indices] ** 2
    local = int(np.argmax(power))
    dominant_index = int(selected_indices[local])
    dominant_bin = float(frequencies[dominant_index])
    if not global_result.peak_metrics:
        return dominant_bin, None, None, ("dominant_bin_not_significant_peak",)
    peak = max(global_result.peak_metrics, key=lambda item: item.power)
    diagnostics: tuple[str, ...] = ()
    if peak.peak_index != local:
        diagnostics = ("dominant_bin_not_significant_peak",)
    return (
        dominant_bin,
        peak.representative_frequency_hz,
        peak.relative_power,
        diagnostics,
    )


def _build_summary(
    frames: tuple[TimeResolvedSpectralFrame, ...],
    cfg: TimeResolvedSpectralCharacterizationSettings,
) -> TimeResolvedSpectralSummary:
    valid = tuple(frame for frame in frames if frame.valid)
    diagnostics: list[str] = [
        "summary_is_descriptive_not_regime_classification",
        "late_minus_early_changes_are_operational",
    ]
    regions = _build_regions(frames, cfg)
    bands = _build_band_summaries(frames, cfg)
    if not valid:
        diagnostics.append("no_valid_frames_for_summary")
        return TimeResolvedSpectralSummary(
            None, None, 0.0, 0, 0.0,
            None, None, None, None,
            None, None, None, None,
            None, None, None, None,
            None, None, None, None,
            None, None,
            None, None,
            None, None,
            None, None,
            None, None, None, None, None,
            regions, bands, tuple(diagnostics),
        )
    energies = _finite_values(valid, "spectral_energy")
    max_energy_frame = max(
        (frame for frame in valid if frame.spectral_energy is not None),
        key=lambda frame: frame.spectral_energy or 0.0,
    )
    early = next((region for region in regions if region.region == "early" and region.valid), None)
    late = next((region for region in regions if region.region == "late" and region.valid), None)
    return TimeResolvedSpectralSummary(
        valid_start_time_s=valid[0].start_time_s,
        valid_end_time_s=valid[-1].end_time_s,
        covered_duration_s=valid[-1].end_time_s - valid[0].start_time_s,
        valid_frame_count=len(valid),
        temporal_coverage_fraction=len(valid) / len(frames) if frames else 0.0,
        initial_energy=valid[0].spectral_energy,
        final_energy=valid[-1].spectral_energy,
        maximum_energy=float(np.max(energies)) if energies.size else None,
        maximum_energy_time_s=max_energy_frame.center_time_s,
        initial_flatness=valid[0].spectral_flatness,
        final_flatness=valid[-1].spectral_flatness,
        minimum_flatness=_min_metric(valid, "spectral_flatness"),
        maximum_flatness=_max_metric(valid, "spectral_flatness"),
        initial_entropy=valid[0].spectral_entropy,
        final_entropy=valid[-1].spectral_entropy,
        minimum_entropy=_min_metric(valid, "spectral_entropy"),
        maximum_entropy=_max_metric(valid, "spectral_entropy"),
        initial_centroid_hz=valid[0].spectral_centroid_hz,
        final_centroid_hz=valid[-1].spectral_centroid_hz,
        minimum_centroid_hz=_min_metric(valid, "spectral_centroid_hz"),
        maximum_centroid_hz=_max_metric(valid, "spectral_centroid_hz"),
        initial_tonal_energy_fraction=valid[0].tonal_energy_fraction,
        final_tonal_energy_fraction=valid[-1].tonal_energy_fraction,
        initial_peak_density_per_hz=valid[0].peak_density_per_hz,
        final_peak_density_per_hz=valid[-1].peak_density_per_hz,
        initial_occupied_bandwidth_hz=valid[0].occupied_bandwidth_hz,
        final_occupied_bandwidth_hz=valid[-1].occupied_bandwidth_hz,
        tonal_dominated_frame_fraction=_fraction_at_least(
            valid, "tonal_energy_fraction", cfg.tonal_dominance_fraction_threshold
        ),
        high_residual_frame_fraction=_fraction_at_least(
            valid, "residual_energy_fraction", cfg.high_residual_fraction_threshold
        ),
        tonal_fraction_change=_region_change(early, late, "median_tonal_energy_fraction"),
        residual_fraction_change=_region_change(early, late, "median_residual_energy_fraction"),
        flatness_change=_region_change(early, late, "median_flatness"),
        entropy_change=_region_change(early, late, "median_entropy"),
        peak_density_change=_region_change(early, late, "median_peak_density_per_hz"),
        regions=regions,
        band_summaries=bands,
        diagnostics=tuple(dict.fromkeys(diagnostics)),
    )


def _build_regions(
    frames: tuple[TimeResolvedSpectralFrame, ...],
    cfg: TimeResolvedSpectralCharacterizationSettings,
) -> tuple[SpectralTemporalRegionSummary, ...]:
    configured = (
        ("early", cfg.early_window_s),
        ("middle", cfg.middle_window_s),
        ("late", cfg.late_window_s),
    )
    return tuple(
        _region_summary(label, window, frames)
        for label, window in configured
        if window is not None
    )


def _region_summary(
    label: str,
    window: tuple[float, float],
    frames: tuple[TimeResolvedSpectralFrame, ...],
) -> SpectralTemporalRegionSummary:
    selected = tuple(
        frame for frame in frames
        if window[0] <= frame.relative_time_s < window[1]
    )
    valid = tuple(frame for frame in selected if frame.valid)
    diagnostics = ["region_median_uses_valid_frames_only"]
    if not selected:
        diagnostics.append("region_contains_no_frames")
    elif not valid:
        diagnostics.append("region_contains_no_valid_frames")
    return SpectralTemporalRegionSummary(
        region=label,
        start_relative_s=window[0],
        end_relative_s=window[1],
        frame_count=len(selected),
        valid_frame_count=len(valid),
        median_energy=_median_metric(valid, "spectral_energy"),
        median_centroid_hz=_median_metric(valid, "spectral_centroid_hz"),
        median_spread_hz=_median_metric(valid, "spectral_spread_hz"),
        median_flatness=_median_metric(valid, "spectral_flatness"),
        median_entropy=_median_metric(valid, "spectral_entropy"),
        median_peak_count=_median_metric(valid, "significant_peak_count"),
        median_peak_density_per_hz=_median_metric(valid, "peak_density_per_hz"),
        median_tonal_energy_fraction=_median_metric(valid, "tonal_energy_fraction"),
        median_residual_energy_fraction=_median_metric(valid, "residual_energy_fraction"),
        median_occupied_bandwidth_hz=_median_metric(valid, "occupied_bandwidth_hz"),
        valid=bool(valid),
        diagnostics=tuple(diagnostics),
    )


def _build_band_summaries(
    frames: tuple[TimeResolvedSpectralFrame, ...],
    cfg: TimeResolvedSpectralCharacterizationSettings,
) -> tuple[TimeResolvedSpectralBandSummary, ...]:
    valid = tuple(frame for frame in frames if frame.valid)
    summaries: list[TimeResolvedSpectralBandSummary] = []
    for band in cfg.bands:
        frame_band_pairs = [
            (frame, _band_metric(frame, band.label))
            for frame in valid
            if _band_metric(frame, band.label) is not None
        ]
        if not frame_band_pairs:
            fit = _failed_fit(
                f"band:{band.label}:energy_fraction",
                len(frames),
                "insufficient_points",
                ("no_valid_band_points",),
            )
            summaries.append(TimeResolvedSpectralBandSummary(
                band.label,
                band.frequency_start_hz,
                band.frequency_end_hz,
                None,
                None,
                None,
                None,
                None,
                fit,
                0.0,
                None,
                ("no_valid_band_points",),
            ))
            continue
        band_metrics = tuple(metric for _, metric in frame_band_pairs if metric is not None)
        fractions = tuple(metric.energy_fraction for metric in band_metrics)
        coverage = sum(
            fraction >= cfg.band_presence_energy_fraction_threshold
            for fraction in fractions
        ) / len(frame_band_pairs)
        below = next(
            (
                frame.relative_time_s
                for frame, metric in frame_band_pairs
                if metric is not None
                and metric.energy_fraction < cfg.band_presence_energy_fraction_threshold
            ),
            None,
        )
        fit = _fit_values(
            f"band:{band.label}:energy_fraction",
            np.asarray([frame.relative_time_s for frame, _ in frame_band_pairs], dtype=float),
            np.asarray(fractions, dtype=float),
            available_count=len(frames),
            minimum_count=max(2, cfg.minimum_regression_frame_count),
        )
        summaries.append(TimeResolvedSpectralBandSummary(
            label=band.label,
            frequency_start_hz=band.frequency_start_hz,
            frequency_end_hz=band.frequency_end_hz,
            initial_energy=band_metrics[0].energy,
            final_energy=band_metrics[-1].energy,
            maximum_energy=float(np.max([metric.energy for metric in band_metrics])),
            initial_energy_fraction=band_metrics[0].energy_fraction,
            final_energy_fraction=band_metrics[-1].energy_fraction,
            energy_fraction_trend=fit,
            coverage_fraction=coverage,
            time_until_below_threshold_s=below,
            diagnostics=("band_presence_uses_energy_fraction_threshold",),
        ))
    return tuple(summaries)


def _fit_metric(
    frames: tuple[TimeResolvedSpectralFrame, ...],
    metric: str,
    cfg: TimeResolvedSpectralCharacterizationSettings,
) -> SpectralMetricTemporalFit:
    times, values, diagnostics = _metric_series(frames, metric)
    if diagnostics:
        return _failed_fit(
            metric,
            len(frames),
            diagnostics[0],
            tuple(diagnostics),
            finite_count=len(values),
        )
    return _fit_values(
        metric,
        times,
        values,
        available_count=len(frames),
        minimum_count=max(2, cfg.minimum_regression_frame_count),
    )


def _fit_values(
    metric: str,
    times: np.ndarray,
    values: np.ndarray,
    *,
    available_count: int,
    minimum_count: int,
) -> SpectralMetricTemporalFit:
    finite = np.isfinite(times) & np.isfinite(values)
    used_times = times[finite]
    used_values = values[finite]
    diagnostics: list[str] = []
    if int((~finite).sum()):
        diagnostics.append("nonfinite_points_discarded")
    if used_values.size < minimum_count:
        diagnostics.append("insufficient_points")
        return _failed_fit(
            metric,
            available_count,
            "insufficient_points",
            tuple(diagnostics),
            finite_count=int(used_values.size),
        )
    if np.ptp(used_times) <= 0:
        diagnostics.append("non_distinct_times")
        return _failed_fit(
            metric,
            available_count,
            "non_distinct_times",
            tuple(diagnostics),
            finite_count=int(used_values.size),
        )
    coefficients = np.polyfit(used_times, used_values, 1)
    fitted = np.polyval(coefficients, used_times)
    residual = float(np.sum((used_values - fitted) ** 2))
    total = float(np.sum((used_values - np.mean(used_values)) ** 2))
    r_squared = 1.0 if total <= 1e-24 else 1.0 - residual / total
    return SpectralMetricTemporalFit(
        metric_name=metric,
        success=True,
        method="linear_least_squares",
        slope_per_s=float(coefficients[0]),
        intercept=float(coefficients[1]),
        r_squared=float(r_squared),
        rmse=float(np.sqrt(np.mean((used_values - fitted) ** 2))),
        available_point_count=available_count,
        finite_point_count=int(used_values.size),
        used_point_count=int(used_values.size),
        start_time_s=float(used_times[0]),
        end_time_s=float(used_times[-1]),
        diagnostics=tuple(diagnostics),
    )


def _failed_fit(
    metric: str,
    available_count: int,
    reason: str,
    diagnostics: tuple[str, ...],
    *,
    finite_count: int = 0,
) -> SpectralMetricTemporalFit:
    return SpectralMetricTemporalFit(
        metric_name=metric,
        success=False,
        method=None,
        slope_per_s=None,
        intercept=None,
        r_squared=None,
        rmse=None,
        available_point_count=available_count,
        finite_point_count=finite_count,
        used_point_count=finite_count,
        start_time_s=None,
        end_time_s=None,
        failure_reason=reason,
        diagnostics=tuple(dict.fromkeys(diagnostics)),
    )


def _metric_series(
    frames: tuple[TimeResolvedSpectralFrame, ...],
    metric: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    valid = tuple(frame for frame in frames if frame.valid)
    if not valid:
        return np.array([]), np.array([]), ("no_valid_frames",)
    if metric == "energy_db":
        energies = np.asarray([
            frame.spectral_energy
            for frame in valid
            if frame.spectral_energy is not None and frame.spectral_energy > 0
        ], dtype=float)
        energy_frames = tuple(
            frame
            for frame in valid
            if frame.spectral_energy is not None and frame.spectral_energy > 0
        )
        if not energy_frames:
            return np.array([]), np.array([]), ("no_positive_energy_frames",)
        reference = float(np.max(energies))
        values = 10.0 * np.log10(energies / reference)
        times = np.asarray([frame.relative_time_s for frame in energy_frames], dtype=float)
        return times, values, ()
    values: list[float] = []
    times: list[float] = []
    for frame in valid:
        value = getattr(frame, metric)
        if value is not None and isfinite(float(value)):
            values.append(float(value))
            times.append(frame.relative_time_s)
    if not values:
        return np.array([]), np.array([]), (f"metric_unavailable:{metric}",)
    return np.asarray(times, dtype=float), np.asarray(values, dtype=float), ()


def _detect_change_points(
    frames: tuple[TimeResolvedSpectralFrame, ...],
    cfg: TimeResolvedSpectralCharacterizationSettings,
) -> tuple[SpectralMetricChangePoint, ...]:
    points: list[SpectralMetricChangePoint] = []
    thresholds = dict(cfg.change_point_thresholds)
    for metric, threshold in thresholds.items():
        times, values, diagnostics = _metric_series(frames, metric)
        if diagnostics or values.size < 2 * cfg.change_point_window_frames:
            continue
        values_for_detection = _smooth(values, cfg)
        for index in range(cfg.change_point_window_frames, values_for_detection.size - cfg.change_point_window_frames + 1):
            before_values = values_for_detection[index - cfg.change_point_window_frames:index]
            after_values = values_for_detection[index:index + cfg.change_point_window_frames]
            before = float(np.median(before_values))
            after = float(np.median(after_values))
            if np.max(np.abs(before_values - before)) > threshold + cfg.numerical_tolerance:
                continue
            difference = after - before
            if abs(difference) + cfg.numerical_tolerance < threshold:
                continue
            direction = "increase" if difference > 0 else "decrease" if difference < 0 else "unchanged"
            persistence = int(np.sum(np.abs(after_values - before) + cfg.numerical_tolerance >= threshold))
            if persistence < min(cfg.change_point_minimum_persistence_frames, after_values.size):
                continue
            relative = difference / abs(before) if abs(before) > cfg.numerical_tolerance else None
            points.append(SpectralMetricChangePoint(
                metric_name=metric,
                frame_index=index,
                time_s=float(times[index]),
                value_before=before,
                value_after=after,
                difference=difference,
                relative_difference=relative,
                threshold=threshold,
                direction=direction,
                persistence_frame_count=persistence,
                diagnostics=("operational_change_point_not_physical_regime_transition",),
            ))
    return tuple(points)


def _smooth(
    values: np.ndarray,
    cfg: TimeResolvedSpectralCharacterizationSettings,
) -> np.ndarray:
    if cfg.smoothing_method == "none" or values.size == 0:
        return values.copy()
    radius = cfg.smoothing_window_frames // 2
    output = np.empty_like(values)
    for index in range(values.size):
        lower = max(0, index - radius)
        upper = min(values.size, index + radius + 1)
        window = values[lower:upper]
        if cfg.smoothing_method == "median":
            output[index] = np.median(window)
        else:
            output[index] = np.mean(window)
    return output


def _finite_values(frames: tuple[TimeResolvedSpectralFrame, ...], metric: str) -> np.ndarray:
    return np.asarray([
        float(value)
        for frame in frames
        for value in (getattr(frame, metric),)
        if value is not None and isfinite(float(value))
    ], dtype=float)


def _median_metric(frames: tuple[TimeResolvedSpectralFrame, ...], metric: str) -> float | None:
    values = _finite_values(frames, metric)
    return float(np.median(values)) if values.size else None


def _min_metric(frames: tuple[TimeResolvedSpectralFrame, ...], metric: str) -> float | None:
    values = _finite_values(frames, metric)
    return float(np.min(values)) if values.size else None


def _max_metric(frames: tuple[TimeResolvedSpectralFrame, ...], metric: str) -> float | None:
    values = _finite_values(frames, metric)
    return float(np.max(values)) if values.size else None


def _fraction_at_least(
    frames: tuple[TimeResolvedSpectralFrame, ...],
    metric: str,
    threshold: float,
) -> float | None:
    values = _finite_values(frames, metric)
    if not values.size:
        return None
    return float(np.sum(values >= threshold) / values.size)


def _region_change(
    early: SpectralTemporalRegionSummary | None,
    late: SpectralTemporalRegionSummary | None,
    metric: str,
) -> float | None:
    if early is None or late is None:
        return None
    early_value = getattr(early, metric)
    late_value = getattr(late, metric)
    if early_value is None or late_value is None:
        return None
    return float(late_value - early_value)


def _band_metric(
    frame: TimeResolvedSpectralFrame,
    label: str,
) -> SpectralBandEnergy | None:
    return next((metric for metric in frame.band_energy_metrics if metric.label == label), None)


def _validate_regions(
    early: tuple[float, float] | None,
    middle: tuple[float, float] | None,
    late: tuple[float, float] | None,
) -> None:
    regions = [region for region in (early, middle, late) if region is not None]
    for region in regions:
        if len(region) != 2 or not all(isfinite(value) for value in region):
            raise ValueError("temporal regions must contain finite bounds.")
        if region[1] <= region[0]:
            raise ValueError("temporal regions must be ordered.")
    ordered = sorted(regions, key=lambda item: item[0])
    if ordered != regions:
        raise ValueError("temporal regions must be ordered early to late.")
    if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError("temporal regions must not overlap.")


def _peak_criteria(cfg: TimeResolvedSpectralCharacterizationSettings) -> tuple[object, ...]:
    return (
        cfg.peak_min_power,
        cfg.peak_min_prominence,
        cfg.peak_distance_bins,
        cfg.peak_min_width_bins,
        cfg.peak_max_width_bins,
    )


def _regions(cfg: TimeResolvedSpectralCharacterizationSettings) -> tuple[object, ...]:
    return (cfg.early_window_s, cfg.middle_window_s, cfg.late_window_s)
