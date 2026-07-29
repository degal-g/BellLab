"""Controlled synthetic validation for BellLab operational layers.

The contracts in this module compare known synthetic truth with values
recovered by BellLab APIs. A passing synthetic scenario is evidence about a
controlled numerical setup only; it is not a proof of physical validity on real
recordings.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
from math import isfinite, log, pi, sqrt
from types import MappingProxyType
from typing import Any

import numpy as np

from belllab.config import (
    FramePeakDetectionSettings,
    ModalCandidateSettings,
    PeakDetectionSettings,
    SpectralTrackingSettings,
    SpectrumAnalysisSettings,
    STFTSettings,
    TemporalAnalysisSettings,
)
from belllab.modal_candidates import select_modal_candidates
from belllab.modal_energy_exchange import (
    ModalEnergyExchangeResult,
    ModalEnergyExchangeSettings,
    ModalEnergyExchangeStatus,
    evaluate_modal_energy_exchange,
    prepare_modal_envelope_series,
)
from belllab.modal_hypotheses import ModalHypothesisResult, ModalHypothesisStatus
from belllab.modal_parameters import (
    ModalParameterEstimationResult,
    ModalParameterEstimate,
)
from belllab.modal_q_factors import (
    ModalBandwidthSource,
    ModalQFactorEstimationResult,
    estimate_modal_bandwidth,
    estimate_q_from_bandwidth,
    estimate_q_from_decay,
)
from belllab.results import (
    PeakDetectionResults,
    SpectrumResults,
    SpectralTrackingResults,
    STFTResults,
    TemporalResults,
    TimeFrequencyPeakResults,
)
from belllab.spectrum import analyze_spectrum, analyze_stft, detect_spectral_peaks
from belllab.temporal import analyze_temporal
from belllab.tracking import (
    characterize_spectral_track,
    detect_stft_peaks,
    track_spectral_peaks,
)
from belllab.types import Signal, SpectralTrack


class SyntheticValidationStatus(str, Enum):
    """Exclusive status for one synthetic validation result."""

    PASSED = "passed"
    PASSED_WITH_RESERVATIONS = "passed_with_reservations"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    INVALID_SCENARIO = "invalid_scenario"
    PIPELINE_ERROR = "pipeline_error"


class SyntheticValidationReason(str, Enum):
    """Typed reasons for synthetic validation support, reservations and failure."""

    FREQUENCY_ERROR_WITHIN_TOLERANCE = "frequency_error_within_tolerance"
    TAU_ERROR_WITHIN_TOLERANCE = "tau_error_within_tolerance"
    Q_ERROR_WITHIN_TOLERANCE = "q_error_within_tolerance"
    BANDWIDTH_ERROR_WITHIN_TOLERANCE = "bandwidth_error_within_tolerance"
    TRACKING_RECOVERED = "tracking_recovered"
    CANDIDATE_COUNT_RECOVERED = "candidate_count_recovered"
    ASSOCIATION_RECOVERED = "association_recovered"
    CHAIN_RECOVERED = "chain_recovered"
    HYPOTHESIS_STATUS_RECOVERED = "hypothesis_status_recovered"
    ENERGY_EXCHANGE_PATTERN_RECOVERED = "energy_exchange_pattern_recovered"
    NO_FALSE_ENERGY_EXCHANGE_SUPPORT = "no_false_energy_exchange_support"
    NO_FALSE_ASSOCIATION = "no_false_association"
    NO_FALSE_CANDIDATE_EXCESS = "no_false_candidate_excess"
    NO_MISSED_CANDIDATE_EXCESS = "no_missed_candidate_excess"
    NO_TRACK_GAP_EXCESS = "no_track_gap_excess"
    NO_CHAIN_RECOVERY_DEFICIT = "no_chain_recovery_deficit"
    NO_HYPOTHESIS_RECOVERY_DEFICIT = "no_hypothesis_recovery_deficit"
    NO_GENERAL_PHYSICAL_VALIDITY_CLAIM = "no_general_physical_validity_claim"
    NO_GROUND_TRUTH_USED_BY_ESTIMATOR = "no_ground_truth_used_by_estimator"
    NO_THRESHOLD_CALIBRATION_FROM_RESULT = "no_threshold_calibration_from_result"
    NO_TRACKING_CORRECTION_FROM_TRUTH = "no_tracking_correction_from_truth"
    NO_PHYSICAL_SPLIT_OR_MERGE_RESOLUTION = "no_physical_split_or_merge_resolution"
    NO_CAUSALITY_INFERRED = "no_causality_inferred"
    NO_MODAL_MODE_PROMOTION = "no_modal_mode_promotion"
    NO_AUDIO_FILE_READ = "no_audio_file_read"
    NO_EXTERNAL_DATA_READ = "no_external_data_read"
    NO_HIDDEN_TRUE_VALUE_INPUT = "no_hidden_true_value_input"
    NO_GLOBAL_RNG_MUTATION = "no_global_rng_mutation"
    NO_INPUT_MUTATION = "no_input_mutation"
    DETERMINISTIC_SEED_USED = "deterministic_seed_used"
    PIPELINE_STAGE_COMPLETED = "pipeline_stage_completed"
    PARTIAL_PIPELINE_ALLOWED = "partial_pipeline_allowed"
    IDENTIFIABLE_SYNTHETIC_SCENARIO = "identifiable_synthetic_scenario"
    NON_IDENTIFIABLE_SCENARIO_REPORTED = "non_identifiable_scenario_reported"
    NOISE_ROBUSTNESS_SUPPORTED = "noise_robustness_supported"
    CLIPPING_ROBUSTNESS_SUPPORTED = "clipping_robustness_supported"
    MONTE_CARLO_PASS_FRACTION_WITHIN_LIMIT = "monte_carlo_pass_fraction_within_limit"

    RESOLUTION_LIMITED = "resolution_limited"
    SHORT_DURATION = "short_duration"
    LOW_SIGNAL_TO_NOISE_RATIO = "low_signal_to_noise_ratio"
    MILD_CLIPPING = "mild_clipping"
    MODERATE_CLIPPING = "moderate_clipping"
    SEVERE_CLIPPING = "severe_clipping"
    NEIGHBORING_MODE_INTERFERENCE = "neighboring_mode_interference"
    POSSIBLE_BEATING_CONTEXT = "possible_beating_context"
    FREQUENCY_CROSSING_CONTEXT = "frequency_crossing_context"
    AMBIGUOUS_TRACKING = "ambiguous_tracking"
    NEAR_THRESHOLD_RESULT = "near_threshold_result"
    PARTIAL_RECOVERY = "partial_recovery"
    SEED_SENSITIVE_RESULT = "seed_sensitive_result"
    APPARENT_SPLIT_CONTEXT = "apparent_split_context"
    APPARENT_MERGE_CONTEXT = "apparent_merge_context"
    EMERGING_COMPONENT_CONTEXT = "emerging_component_context"
    DISAPPEARING_COMPONENT_CONTEXT = "disappearing_component_context"

    FREQUENCY_ERROR_EXCEEDS_TOLERANCE = "frequency_error_exceeds_tolerance"
    TAU_ERROR_EXCEEDS_TOLERANCE = "tau_error_exceeds_tolerance"
    Q_ERROR_EXCEEDS_TOLERANCE = "q_error_exceeds_tolerance"
    BANDWIDTH_ERROR_EXCEEDS_TOLERANCE = "bandwidth_error_exceeds_tolerance"
    TRACK_FRAGMENTATION = "track_fragmentation"
    TRACK_SWAP = "track_swap"
    CANDIDATE_MISSED = "candidate_missed"
    FALSE_CANDIDATE_DETECTED = "false_candidate_detected"
    ASSOCIATION_MISMATCH = "association_mismatch"
    CHAIN_MISMATCH = "chain_mismatch"
    HYPOTHESIS_MISMATCH = "hypothesis_mismatch"
    ENERGY_EXCHANGE_FALSE_POSITIVE = "energy_exchange_false_positive"
    ENERGY_EXCHANGE_FALSE_NEGATIVE = "energy_exchange_false_negative"
    UNEXPECTED_PIPELINE_EXCEPTION = "unexpected_pipeline_exception"

    DURATION_INSUFFICIENT = "duration_insufficient"
    RESOLUTION_INSUFFICIENT = "resolution_insufficient"
    TOO_FEW_DECAY_CYCLES = "too_few_decay_cycles"
    TOO_FEW_BEATING_CYCLES = "too_few_beating_cycles"
    SIGNAL_BELOW_DETECTION_LIMIT = "signal_below_detection_limit"
    NO_VALID_ESTIMATE = "no_valid_estimate"
    SCENARIO_NOT_IDENTIFIABLE = "scenario_not_identifiable"
    INVALID_COMPONENT = "invalid_component"
    INVALID_SETTINGS = "invalid_settings"
    PIPELINE_ERROR = "pipeline_error"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class SyntheticFrequencyModel(str, Enum):
    """Synthetic instantaneous-frequency policies."""

    CONSTANT = "constant"
    LINEAR_DRIFT = "linear_drift"
    PIECEWISE_LINEAR = "piecewise_linear"
    SINUSOIDAL_MODULATION = "sinusoidal_modulation"
    CROSSING_PAIR_MEMBER = "crossing_pair_member"
    CUSTOM_SAMPLES = "custom_samples"


class SyntheticAmplitudeModel(str, Enum):
    """Synthetic envelope policies."""

    EXPONENTIAL_DECAY = "exponential_decay"
    DELAYED_ONSET = "delayed_onset"
    DELAYED_GROWTH = "delayed_growth"
    DECAY_THEN_RECOVERY = "decay_then_recovery"
    PIECEWISE_ENVELOPE = "piecewise_envelope"
    BEATING_PAIR_MEMBER = "beating_pair_member"
    CONSTANT_AMPLITUDE = "constant_amplitude"
    CUSTOM_SAMPLES = "custom_samples"


class SyntheticNoiseModel(str, Enum):
    """Noise model used only by the synthetic generator."""

    NONE = "none"
    WHITE = "white"
    PINK = "pink"


class SyntheticClippingMode(str, Enum):
    """Clipping policy used only by the synthetic generator."""

    NONE = "none"
    HARD = "hard"
    SOFT = "soft"


@dataclass(frozen=True, slots=True)
class SyntheticValidationSettings:
    """Explicit settings for controlled synthetic validation."""

    sample_rate_hz: int = 8000
    duration_s: float = 8.0
    attack_time_s: float = 0.0
    signal_start_time_s: float = 0.0
    global_amplitude: float = 0.6
    random_seed: int | None = 0

    noise_model: SyntheticNoiseModel = SyntheticNoiseModel.NONE
    signal_to_noise_ratio_db: float | None = None
    noise_standard_deviation: float | None = None
    include_colored_noise: bool = False
    colored_noise_exponent: float = 1.0
    include_mains_hum: bool = False
    mains_frequency_hz: float = 60.0
    mains_amplitude: float = 0.0

    clipping_mode: SyntheticClippingMode = SyntheticClippingMode.NONE
    clipping_threshold: float | None = None
    clipping_fraction: float | None = None
    allow_soft_clipping: bool = False

    run_temporal_analysis: bool = True
    run_global_spectrum: bool = True
    run_stft: bool = True
    run_tracking: bool = True
    run_candidate_characterization: bool = True
    run_within_condition_association: bool = True
    run_cross_condition_association: bool = True
    run_candidate_chains: bool = True
    run_modal_hypotheses: bool = True
    run_modal_parameter_estimation: bool = True
    run_modal_q_estimation: bool = True
    run_energy_exchange_evidence: bool = True

    maximum_frequency_absolute_error_hz: float | None = 2.0
    maximum_frequency_relative_error: float | None = 0.01
    maximum_tau_relative_error: float | None = 0.35
    maximum_q_relative_error: float | None = 0.35
    maximum_bandwidth_relative_error: float | None = 0.50
    maximum_tracking_gap_fraction: float = 0.25
    maximum_false_candidate_count: int = 0
    maximum_missed_candidate_count: int = 0
    minimum_chain_recovery_fraction: float = 1.0
    minimum_hypothesis_recovery_fraction: float = 1.0

    trial_count: int = 5
    trial_seed_stride: int = 9973
    minimum_pass_fraction: float = 0.8
    minimum_pass_with_reservation_fraction: float = 1.0
    store_trial_details: bool = True

    spectrum_window_name: str = "hann"
    spectrum_n_fft: int | None = 131072
    stft_window_length: int = 1024
    stft_hop_length: int = 256
    stft_n_fft: int | None = 2048
    peak_min_prominence: float | None = 0.02
    peak_min_amplitude: float | None = None
    peak_distance_bins: int | None = None
    peak_max_peaks: int | None = None
    tracking_frequency_tolerance: float = 0.02
    tracking_frequency_distance_unit: str = "relative"
    candidate_minimum_observation_count: int | None = 2

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "noise_model",
            _coerce_enum(self.noise_model, SyntheticNoiseModel),
        )
        object.__setattr__(
            self,
            "clipping_mode",
            _coerce_enum(self.clipping_mode, SyntheticClippingMode),
        )
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive.")
        for name in ("duration_s", "global_amplitude"):
            _finite_positive(getattr(self, name), name)
        _finite_nonnegative(self.attack_time_s, "attack_time_s")
        _finite_nonnegative(self.signal_start_time_s, "signal_start_time_s")
        if self.signal_start_time_s >= self.duration_s:
            raise ValueError("signal_start_time_s must be inside the signal duration.")
        if self.random_seed is not None and not isinstance(self.random_seed, int):
            raise ValueError("random_seed must be an int or None.")
        _finite_optional(self.signal_to_noise_ratio_db, "signal_to_noise_ratio_db")
        _finite_optional(self.noise_standard_deviation, "noise_standard_deviation", nonnegative=True)
        _finite_nonnegative(self.colored_noise_exponent, "colored_noise_exponent")
        _finite_positive(self.mains_frequency_hz, "mains_frequency_hz")
        _finite_nonnegative(self.mains_amplitude, "mains_amplitude")
        _finite_optional(self.clipping_threshold, "clipping_threshold", positive=True)
        _fraction(self.clipping_fraction, "clipping_fraction")
        for name in (
            "include_colored_noise",
            "include_mains_hum",
            "allow_soft_clipping",
            "run_temporal_analysis",
            "run_global_spectrum",
            "run_stft",
            "run_tracking",
            "run_candidate_characterization",
            "run_within_condition_association",
            "run_cross_condition_association",
            "run_candidate_chains",
            "run_modal_hypotheses",
            "run_modal_parameter_estimation",
            "run_modal_q_estimation",
            "run_energy_exchange_evidence",
            "store_trial_details",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")
        for name in (
            "maximum_frequency_absolute_error_hz",
            "maximum_tau_relative_error",
            "maximum_q_relative_error",
            "maximum_bandwidth_relative_error",
            "maximum_tracking_gap_fraction",
            "minimum_chain_recovery_fraction",
            "minimum_hypothesis_recovery_fraction",
            "minimum_pass_fraction",
            "minimum_pass_with_reservation_fraction",
        ):
            _fraction_or_nonnegative(getattr(self, name), name)
        _fraction(self.maximum_frequency_relative_error, "maximum_frequency_relative_error")
        if self.maximum_false_candidate_count < 0 or self.maximum_missed_candidate_count < 0:
            raise ValueError("candidate count tolerances must not be negative.")
        if self.trial_count <= 0:
            raise ValueError("trial_count must be positive.")
        if self.trial_seed_stride <= 0:
            raise ValueError("trial_seed_stride must be positive.")
        _text(self.spectrum_window_name, "spectrum_window_name")
        if self.spectrum_n_fft is not None and self.spectrum_n_fft <= 0:
            raise ValueError("spectrum_n_fft must be positive when provided.")
        if self.stft_window_length <= 0 or self.stft_hop_length <= 0:
            raise ValueError("stft window and hop lengths must be positive.")
        if self.stft_hop_length > self.stft_window_length:
            raise ValueError("stft_hop_length must not exceed stft_window_length.")
        if self.stft_n_fft is not None and self.stft_n_fft < self.stft_window_length:
            raise ValueError("stft_n_fft must be at least stft_window_length.")
        _finite_optional(self.peak_min_prominence, "peak_min_prominence", nonnegative=True)
        _finite_optional(self.peak_min_amplitude, "peak_min_amplitude", nonnegative=True)
        if self.peak_distance_bins is not None and self.peak_distance_bins <= 0:
            raise ValueError("peak_distance_bins must be positive when provided.")
        if self.peak_max_peaks is not None and self.peak_max_peaks <= 0:
            raise ValueError("peak_max_peaks must be positive when provided.")
        _finite_positive(self.tracking_frequency_tolerance, "tracking_frequency_tolerance")
        if self.tracking_frequency_distance_unit not in {"hz", "relative", "cents"}:
            raise ValueError("tracking_frequency_distance_unit must be hz, relative or cents.")
        if (
            self.candidate_minimum_observation_count is not None
            and self.candidate_minimum_observation_count < 0
        ):
            raise ValueError("candidate_minimum_observation_count must not be negative.")
        nyquist = 0.5 * self.sample_rate_hz
        if self.include_mains_hum and self.mains_frequency_hz >= nyquist:
            raise ValueError("mains_frequency_hz must be below Nyquist.")


@dataclass(frozen=True, slots=True)
class SyntheticDampedComponent:
    """One synthetic component with known operational parameters."""

    component_id: str
    initial_frequency_hz: float
    amplitude: float
    phase_rad: float = 0.0
    tau_s: float | None = None
    start_time_s: float = 0.0
    end_time_s: float | None = None
    frequency_model: SyntheticFrequencyModel = SyntheticFrequencyModel.CONSTANT
    frequency_drift_hz_per_s: float = 0.0
    frequency_trajectory: tuple[tuple[float, float], ...] = ()
    amplitude_model: SyntheticAmplitudeModel = SyntheticAmplitudeModel.EXPONENTIAL_DECAY
    delayed_growth: tuple[tuple[float, float], ...] = ()
    amplitude_recovery: tuple[tuple[float, float], ...] = ()
    dynamic_label: str = "mf"
    expected_detectable: bool = True
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.component_id, "component_id")
        _finite_positive(self.initial_frequency_hz, "initial_frequency_hz")
        _finite_optional(self.tau_s, "tau_s", positive=True)
        _finite_optional(self.end_time_s, "end_time_s", positive=True)
        _finite_nonnegative(self.start_time_s, "start_time_s")
        _finite_optional(self.amplitude, "amplitude")
        _finite_optional(self.phase_rad, "phase_rad")
        _finite_optional(self.frequency_drift_hz_per_s, "frequency_drift_hz_per_s")
        object.__setattr__(
            self,
            "frequency_model",
            _coerce_enum(self.frequency_model, SyntheticFrequencyModel),
        )
        object.__setattr__(
            self,
            "amplitude_model",
            _coerce_enum(self.amplitude_model, SyntheticAmplitudeModel),
        )
        if self.end_time_s is not None and self.end_time_s <= self.start_time_s:
            raise ValueError("end_time_s must be greater than start_time_s when provided.")
        if not isinstance(self.expected_detectable, bool):
            raise ValueError("expected_detectable must be a boolean.")
        _text(self.dynamic_label, "dynamic_label")
        _series_points(self.frequency_trajectory, "frequency_trajectory", positive_values=True)
        _series_points(self.delayed_growth, "delayed_growth", positive_values=False)
        _series_points(self.amplitude_recovery, "amplitude_recovery", positive_values=False)
        _strings(self.diagnostics, "component diagnostics", allow_empty=True)


@dataclass(frozen=True, slots=True)
class SyntheticGroundTruth:
    """Known truth generated before the BellLab pipeline is run."""

    scenario_id: str
    components: tuple[SyntheticDampedComponent, ...]
    sample_rate_hz: int
    duration_s: float
    time_axis_s: tuple[float, ...]
    clean_signal: Signal
    noise_signal: Signal
    observed_signal: Signal
    known_frequencies_hz: tuple[tuple[str, float], ...]
    known_tau_values_s: tuple[tuple[str, float | None], ...]
    known_q_values: tuple[tuple[str, float | None], ...]
    known_bandwidth_values_hz: tuple[tuple[str, float | None], ...]
    known_component_presence: tuple[tuple[str, float, float], ...]
    known_associations: tuple[tuple[str, str], ...]
    known_chains: tuple[tuple[str, ...], ...]
    known_energy_exchange_pairs: tuple[tuple[str, str], ...]
    known_non_exchange_pairs: tuple[tuple[str, str], ...]
    noise_metadata: Mapping[str, object]
    clipping_metadata: Mapping[str, object]
    settings_fingerprint: str
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.scenario_id, "ground truth scenario_id")
        if self.sample_rate_hz <= 0:
            raise ValueError("ground truth sample_rate_hz must be positive.")
        _finite_positive(self.duration_s, "ground truth duration_s")
        if len(self.components) != len({item.component_id for item in self.components}):
            raise ValueError("component IDs must be unique in ground truth.")
        if not self.time_axis_s:
            raise ValueError("time_axis_s must not be empty.")
        if any(not isfinite(value) for value in self.time_axis_s):
            raise ValueError("time_axis_s values must be finite.")
        if any(later <= earlier for earlier, later in zip(self.time_axis_s, self.time_axis_s[1:])):
            raise ValueError("time_axis_s must be strictly increasing.")
        for signal in (self.clean_signal, self.noise_signal, self.observed_signal):
            if signal.sample_rate != self.sample_rate_hz:
                raise ValueError("ground-truth signals must share the sample rate.")
            if signal.time != self.time_axis_s:
                raise ValueError("ground-truth signals must share the time axis.")
        _truth_pairs(self.known_frequencies_hz, "known_frequencies_hz", positive=True)
        _truth_pairs(self.known_tau_values_s, "known_tau_values_s", positive=True, allow_none=True)
        _truth_pairs(self.known_q_values, "known_q_values", positive=True, allow_none=True)
        _truth_pairs(self.known_bandwidth_values_hz, "known_bandwidth_values_hz", positive=True, allow_none=True)
        _text(self.settings_fingerprint, "settings_fingerprint")
        _strings(self.diagnostics, "ground truth diagnostics", allow_empty=True)
        object.__setattr__(self, "noise_metadata", MappingProxyType(dict(self.noise_metadata)))
        object.__setattr__(self, "clipping_metadata", MappingProxyType(dict(self.clipping_metadata)))


@dataclass(frozen=True, slots=True)
class SyntheticValidationScenario:
    """A controlled scenario and its expected operational outcomes."""

    scenario_id: str
    name: str
    description: str
    components: tuple[SyntheticDampedComponent, ...]
    dynamic_labels: tuple[str, ...]
    recording_layout: Mapping[str, object]
    settings: SyntheticValidationSettings
    expected_outcomes: Mapping[str, object]
    identifiability_notes: tuple[str, ...]
    valid: bool
    reasons: tuple[SyntheticValidationReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.scenario_id, "scenario_id")
        _text(self.name, "scenario name")
        _text(self.description, "description")
        if not self.components:
            raise ValueError("scenario requires at least one component.")
        if len({item.component_id for item in self.components}) != len(self.components):
            raise ValueError("scenario component IDs must be unique.")
        _strings(self.dynamic_labels, "dynamic_labels")
        if not isinstance(self.settings, SyntheticValidationSettings):
            raise ValueError("settings must be SyntheticValidationSettings.")
        if not isinstance(self.valid, bool):
            raise ValueError("valid must be a boolean.")
        _reason_tuple(self.reasons, "scenario reasons")
        _strings(self.identifiability_notes, "identifiability_notes", allow_empty=True)
        _strings(self.diagnostics, "scenario diagnostics", allow_empty=True)
        object.__setattr__(self, "recording_layout", MappingProxyType(dict(self.recording_layout)))
        object.__setattr__(self, "expected_outcomes", MappingProxyType(dict(self.expected_outcomes)))


@dataclass(frozen=True, slots=True)
class SyntheticPipelineOutput:
    """Outputs and errors from public BellLab pipeline stages."""

    scenario_id: str
    signals: tuple[Signal, ...]
    temporal_results: TemporalResults | None
    spectral_results: SpectrumResults | None
    peak_results: PeakDetectionResults | None
    stft_results: STFTResults | None
    time_frequency_peak_results: TimeFrequencyPeakResults | None
    tracking_results: SpectralTrackingResults | None
    candidate_results: tuple[object, ...]
    within_condition_results: tuple[object, ...]
    cross_condition_results: tuple[object, ...]
    chain_results: object | None
    modal_hypothesis_results: ModalHypothesisResult | None
    modal_parameter_results: ModalParameterEstimationResult | None
    modal_q_results: ModalQFactorEstimationResult | None
    energy_exchange_results: ModalEnergyExchangeResult | None
    pipeline_stages_completed: tuple[str, ...]
    pipeline_errors: tuple[tuple[str, str], ...]
    valid: bool
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.scenario_id, "pipeline scenario_id")
        if not self.signals:
            raise ValueError("pipeline output requires at least one signal.")
        _strings(self.pipeline_stages_completed, "pipeline stages", allow_empty=True)
        if len(self.pipeline_stages_completed) != len(set(self.pipeline_stages_completed)):
            raise ValueError("pipeline stages must be unique.")
        for stage, message in self.pipeline_errors:
            _text(stage, "pipeline error stage")
            _text(message, "pipeline error message")
        if self.valid != (not self.pipeline_errors):
            raise ValueError("pipeline valid flag must mirror absence of pipeline_errors.")
        _strings(self.diagnostics, "pipeline diagnostics", allow_empty=True)


@dataclass(frozen=True, slots=True)
class SyntheticFrequencyValidation:
    true_frequency_hz: float
    estimated_frequency_hz: float | None
    absolute_error_hz: float | None
    relative_error: float | None
    signed_error_hz: float | None
    within_absolute_tolerance: bool | None
    within_relative_tolerance: bool | None
    trajectory_rmse_hz: float | None
    trajectory_mae_hz: float | None
    trajectory_max_error_hz: float | None
    trajectory_slope_error_hz_per_s: float | None
    trajectory_total_change_error_hz: float | None
    passed: bool
    reasons: tuple[SyntheticValidationReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _finite_positive(self.true_frequency_hz, "true_frequency_hz")
        _finite_optional(self.estimated_frequency_hz, "estimated_frequency_hz", positive=True)
        for name in (
            "absolute_error_hz",
            "relative_error",
            "trajectory_rmse_hz",
            "trajectory_mae_hz",
            "trajectory_max_error_hz",
            "trajectory_total_change_error_hz",
        ):
            _finite_optional(getattr(self, name), name, nonnegative=True)
        _finite_optional(self.signed_error_hz, "signed_error_hz")
        _finite_optional(self.trajectory_slope_error_hz_per_s, "trajectory_slope_error_hz_per_s", nonnegative=True)
        _reason_tuple(self.reasons, "frequency validation reasons")
        _strings(self.diagnostics, "frequency validation diagnostics", allow_empty=True)


@dataclass(frozen=True, slots=True)
class SyntheticDecayValidation:
    true_tau_s: float
    estimated_tau_s: float | None
    absolute_error_s: float | None
    relative_error: float | None
    log_error: float | None
    within_tolerance: bool | None
    fit_quality: float | None
    passed: bool
    reasons: tuple[SyntheticValidationReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _finite_positive(self.true_tau_s, "true_tau_s")
        _finite_optional(self.estimated_tau_s, "estimated_tau_s", positive=True)
        _finite_optional(self.absolute_error_s, "absolute_error_s", nonnegative=True)
        _finite_optional(self.relative_error, "relative_error", nonnegative=True)
        _finite_optional(self.log_error, "log_error", nonnegative=True)
        _fraction(self.fit_quality, "fit_quality")
        _reason_tuple(self.reasons, "decay validation reasons")
        _strings(self.diagnostics, "decay validation diagnostics", allow_empty=True)


@dataclass(frozen=True, slots=True)
class SyntheticQValidation:
    true_q: float
    estimated_q_decay: float | None
    estimated_q_bandwidth: float | None
    representative_q: float | None
    decay_relative_error: float | None
    bandwidth_relative_error: float | None
    representative_relative_error: float | None
    method_consistency: str | None
    within_tolerance: bool | None
    passed: bool
    reasons: tuple[SyntheticValidationReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _finite_positive(self.true_q, "true_q")
        for name in ("estimated_q_decay", "estimated_q_bandwidth", "representative_q"):
            _finite_optional(getattr(self, name), name, positive=True)
        for name in ("decay_relative_error", "bandwidth_relative_error", "representative_relative_error"):
            _finite_optional(getattr(self, name), name, nonnegative=True)
        if self.method_consistency is not None:
            _text(self.method_consistency, "method_consistency")
        _reason_tuple(self.reasons, "Q validation reasons")
        _strings(self.diagnostics, "Q validation diagnostics", allow_empty=True)


@dataclass(frozen=True, slots=True)
class SyntheticBandwidthValidation:
    true_bandwidth_hz: float | None
    estimated_bandwidth_hz: float | None
    absolute_error_hz: float | None
    relative_error: float | None
    frequency_resolution_hz: float | None
    resolution_ratio: float | None
    within_tolerance: bool | None
    passed: bool
    reasons: tuple[SyntheticValidationReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _finite_optional(self.true_bandwidth_hz, "true_bandwidth_hz", positive=True)
        _finite_optional(self.estimated_bandwidth_hz, "estimated_bandwidth_hz", positive=True)
        for name in ("absolute_error_hz", "relative_error", "resolution_ratio"):
            _finite_optional(getattr(self, name), name, nonnegative=True)
        _finite_optional(self.frequency_resolution_hz, "frequency_resolution_hz", positive=True)
        _reason_tuple(self.reasons, "bandwidth validation reasons")
        _strings(self.diagnostics, "bandwidth validation diagnostics", allow_empty=True)


@dataclass(frozen=True, slots=True)
class SyntheticTrackingValidation:
    expected_track_count: int
    recovered_track_count: int
    matched_track_pairs: tuple[tuple[str, int], ...]
    missed_true_tracks: tuple[str, ...]
    false_tracks: tuple[int, ...]
    fragmented_tracks: tuple[str, ...]
    track_swaps: tuple[tuple[str, str], ...]
    coverage_fractions: tuple[tuple[str, float], ...]
    frequency_rmse_values_hz: tuple[tuple[str, float | None], ...]
    gap_fractions: tuple[tuple[str, float], ...]
    passed: bool
    reasons: tuple[SyntheticValidationReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.expected_track_count < 0 or self.recovered_track_count < 0:
            raise ValueError("track counts must not be negative.")
        _strings(self.missed_true_tracks, "missed_true_tracks", allow_empty=True)
        if any(track_id < 0 for track_id in self.false_tracks):
            raise ValueError("false track IDs must not be negative.")
        _strings(self.fragmented_tracks, "fragmented_tracks", allow_empty=True)
        for _, value in self.coverage_fractions + self.gap_fractions:
            _fraction(value, "tracking fraction")
        for _, value in self.frequency_rmse_values_hz:
            _finite_optional(value, "frequency rmse", nonnegative=True)
        _reason_tuple(self.reasons, "tracking validation reasons")
        _strings(self.diagnostics, "tracking validation diagnostics", allow_empty=True)


@dataclass(frozen=True, slots=True)
class SyntheticCandidateValidation:
    expected_candidate_count: int
    recovered_candidate_count: int
    matched_candidates: tuple[tuple[str, int], ...]
    missed_candidates: tuple[str, ...]
    false_candidates: tuple[int, ...]
    rejected_expected_candidates: tuple[str, ...]
    unexpected_accepted_candidates: tuple[int, ...]
    passed: bool
    reasons: tuple[SyntheticValidationReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.expected_candidate_count < 0 or self.recovered_candidate_count < 0:
            raise ValueError("candidate counts must not be negative.")
        _strings(self.missed_candidates, "missed_candidates", allow_empty=True)
        _strings(self.rejected_expected_candidates, "rejected_expected_candidates", allow_empty=True)
        if any(value < 0 for value in self.false_candidates + self.unexpected_accepted_candidates):
            raise ValueError("candidate IDs must not be negative.")
        _reason_tuple(self.reasons, "candidate validation reasons")
        _strings(self.diagnostics, "candidate validation diagnostics", allow_empty=True)


@dataclass(frozen=True, slots=True)
class SyntheticAssociationValidation:
    expected_pairs: tuple[tuple[str, str], ...]
    recovered_pairs: tuple[tuple[str, str], ...]
    correct_pairs: tuple[tuple[str, str], ...]
    missing_pairs: tuple[tuple[str, str], ...]
    incorrect_pairs: tuple[tuple[str, str], ...]
    emerging_candidates_expected: tuple[str, ...]
    emerging_candidates_recovered: tuple[str, ...]
    disappearing_candidates_expected: tuple[str, ...]
    disappearing_candidates_recovered: tuple[str, ...]
    precision: float | None
    recall: float | None
    f1_score: float | None
    passed: bool
    reasons: tuple[SyntheticValidationReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _pair_tuple(self.expected_pairs, "expected_pairs")
        _pair_tuple(self.recovered_pairs, "recovered_pairs")
        _pair_tuple(self.correct_pairs, "correct_pairs")
        _pair_tuple(self.missing_pairs, "missing_pairs")
        _pair_tuple(self.incorrect_pairs, "incorrect_pairs")
        for name in ("emerging_candidates_expected", "emerging_candidates_recovered", "disappearing_candidates_expected", "disappearing_candidates_recovered"):
            _strings(getattr(self, name), name, allow_empty=True)
        for value in (self.precision, self.recall, self.f1_score):
            _fraction(value, "association metric")
        _reason_tuple(self.reasons, "association validation reasons")
        _strings(self.diagnostics, "association validation diagnostics", allow_empty=True)


@dataclass(frozen=True, slots=True)
class SyntheticChainValidation:
    expected_chains: tuple[tuple[str, ...], ...]
    recovered_chains: tuple[tuple[str, ...], ...]
    exact_chain_matches: tuple[tuple[str, ...], ...]
    partial_chain_matches: tuple[tuple[str, ...], ...]
    missing_chains: tuple[tuple[str, ...], ...]
    spurious_chains: tuple[tuple[str, ...], ...]
    node_recovery_fraction: float | None
    edge_recovery_fraction: float | None
    complete_chain_recovery_fraction: float | None
    passed: bool
    reasons: tuple[SyntheticValidationReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("node_recovery_fraction", "edge_recovery_fraction", "complete_chain_recovery_fraction"):
            _fraction(getattr(self, name), name)
        _reason_tuple(self.reasons, "chain validation reasons")
        _strings(self.diagnostics, "chain validation diagnostics", allow_empty=True)


@dataclass(frozen=True, slots=True)
class SyntheticModalHypothesisValidation:
    expected_statuses: tuple[tuple[str, str], ...]
    recovered_statuses: tuple[tuple[str, str], ...]
    correct_status_count: int
    status_confusion: tuple[tuple[str, str, int], ...]
    accepted_true_components: tuple[str, ...]
    rejected_true_components: tuple[str, ...]
    accepted_false_components: tuple[str, ...]
    recovery_fraction: float | None
    passed: bool
    reasons: tuple[SyntheticValidationReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.correct_status_count < 0:
            raise ValueError("correct_status_count must not be negative.")
        _fraction(self.recovery_fraction, "hypothesis recovery_fraction")
        for name in ("accepted_true_components", "rejected_true_components", "accepted_false_components"):
            _strings(getattr(self, name), name, allow_empty=True)
        _reason_tuple(self.reasons, "hypothesis validation reasons")
        _strings(self.diagnostics, "hypothesis validation diagnostics", allow_empty=True)


@dataclass(frozen=True, slots=True)
class SyntheticEnergyExchangeValidation:
    expected_exchange_pairs: tuple[tuple[str, str], ...]
    expected_non_exchange_pairs: tuple[tuple[str, str], ...]
    supported_pairs: tuple[tuple[str, str], ...]
    not_supported_pairs: tuple[tuple[str, str], ...]
    false_positive_pairs: tuple[tuple[str, str], ...]
    false_negative_pairs: tuple[tuple[str, str], ...]
    inconclusive_pairs: tuple[tuple[str, str], ...]
    precision: float | None
    recall: float | None
    f1_score: float | None
    lag_errors_s: tuple[tuple[str, str, float], ...]
    passed: bool
    reasons: tuple[SyntheticValidationReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "expected_exchange_pairs",
            "expected_non_exchange_pairs",
            "supported_pairs",
            "not_supported_pairs",
            "false_positive_pairs",
            "false_negative_pairs",
            "inconclusive_pairs",
        ):
            _pair_tuple(getattr(self, name), name)
        for value in (self.precision, self.recall, self.f1_score):
            _fraction(value, "energy exchange metric")
        for left, right, error in self.lag_errors_s:
            _text(left, "lag error left")
            _text(right, "lag error right")
            _finite_nonnegative(error, "lag error")
        _reason_tuple(self.reasons, "energy validation reasons")
        _strings(self.diagnostics, "energy validation diagnostics", allow_empty=True)


@dataclass(frozen=True, slots=True)
class SyntheticScenarioValidationResult:
    scenario: SyntheticValidationScenario
    ground_truth: SyntheticGroundTruth
    pipeline_output: SyntheticPipelineOutput
    frequency_validations: tuple[SyntheticFrequencyValidation, ...]
    decay_validations: tuple[SyntheticDecayValidation, ...]
    q_validations: tuple[SyntheticQValidation, ...]
    bandwidth_validations: tuple[SyntheticBandwidthValidation, ...]
    tracking_validation: SyntheticTrackingValidation
    candidate_validation: SyntheticCandidateValidation
    association_validation: SyntheticAssociationValidation
    chain_validation: SyntheticChainValidation
    hypothesis_validation: SyntheticModalHypothesisValidation
    energy_exchange_validation: SyntheticEnergyExchangeValidation
    status: SyntheticValidationStatus
    supporting_reasons: tuple[SyntheticValidationReason, ...]
    reservation_reasons: tuple[SyntheticValidationReason, ...]
    failure_reasons: tuple[SyntheticValidationReason, ...]
    insufficient_evidence_reasons: tuple[SyntheticValidationReason, ...]
    pipeline_error_reasons: tuple[SyntheticValidationReason, ...]
    passed_metric_count: int
    failed_metric_count: int
    inconclusive_metric_count: int
    valid: bool
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "status",
            _coerce_enum(self.status, SyntheticValidationStatus),
        )
        if self.scenario.scenario_id != self.ground_truth.scenario_id or self.scenario.scenario_id != self.pipeline_output.scenario_id:
            raise ValueError("scenario, ground truth and pipeline IDs must agree.")
        for name in (
            "supporting_reasons",
            "reservation_reasons",
            "failure_reasons",
            "insufficient_evidence_reasons",
            "pipeline_error_reasons",
        ):
            _reason_tuple(getattr(self, name), name)
        if min(self.passed_metric_count, self.failed_metric_count, self.inconclusive_metric_count) < 0:
            raise ValueError("metric counts must not be negative.")
        if self.valid != (self.status in {SyntheticValidationStatus.PASSED, SyntheticValidationStatus.PASSED_WITH_RESERVATIONS}):
            raise ValueError("valid must mirror passed statuses.")
        _strings(self.diagnostics, "scenario result diagnostics", allow_empty=True)


@dataclass(frozen=True, slots=True)
class SyntheticValidationCampaignResult:
    campaign_id: str
    scenario_results: tuple[SyntheticScenarioValidationResult, ...]
    scenario_count: int
    passed_count: int
    passed_with_reservations_count: int
    failed_count: int
    inconclusive_count: int
    insufficient_evidence_count: int
    invalid_scenario_count: int
    pipeline_error_count: int
    pass_fraction: float
    metric_summaries: Mapping[str, object]
    settings: SyntheticValidationSettings
    valid: bool
    failure_reason: str | None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.campaign_id, "campaign_id")
        if self.scenario_count != len(self.scenario_results):
            raise ValueError("scenario_count must match scenario_results.")
        ids = tuple(item.scenario.scenario_id for item in self.scenario_results)
        if len(ids) != len(set(ids)):
            raise ValueError("campaign scenario IDs must be unique.")
        expected = {
            "passed_count": SyntheticValidationStatus.PASSED,
            "passed_with_reservations_count": SyntheticValidationStatus.PASSED_WITH_RESERVATIONS,
            "failed_count": SyntheticValidationStatus.FAILED,
            "inconclusive_count": SyntheticValidationStatus.INCONCLUSIVE,
            "insufficient_evidence_count": SyntheticValidationStatus.INSUFFICIENT_EVIDENCE,
            "invalid_scenario_count": SyntheticValidationStatus.INVALID_SCENARIO,
            "pipeline_error_count": SyntheticValidationStatus.PIPELINE_ERROR,
        }
        for field, status in expected.items():
            if getattr(self, field) != sum(item.status is status for item in self.scenario_results):
                raise ValueError(f"{field} is incoherent with scenario results.")
        total = sum(getattr(self, field) for field in expected)
        if total != self.scenario_count:
            raise ValueError("campaign status counts must sum to scenario_count.")
        _fraction(self.pass_fraction, "pass_fraction")
        if self.valid and self.failure_reason is not None:
            raise ValueError("valid campaign must not have failure_reason.")
        if not self.valid and not self.failure_reason:
            raise ValueError("invalid campaign requires failure_reason.")
        _strings(self.diagnostics, "campaign diagnostics", allow_empty=True)
        object.__setattr__(self, "metric_summaries", MappingProxyType(dict(self.metric_summaries)))


@dataclass(frozen=True, slots=True)
class SyntheticMonteCarloValidation:
    base_scenario_id: str
    trial_results: tuple[SyntheticScenarioValidationResult, ...]
    trial_count: int
    seeds: tuple[int | None, ...]
    pass_count: int
    pass_fraction: float
    reservation_count: int
    failure_count: int
    metric_distributions: Mapping[str, tuple[float, ...]]
    frequency_error_quantiles: tuple[tuple[float, float], ...]
    tau_error_quantiles: tuple[tuple[float, float], ...]
    q_error_quantiles: tuple[tuple[float, float], ...]
    tracking_recovery_quantiles: tuple[tuple[float, float], ...]
    candidate_recovery_quantiles: tuple[tuple[float, float], ...]
    valid: bool
    reasons: tuple[SyntheticValidationReason, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.base_scenario_id, "base_scenario_id")
        if self.trial_count != len(self.seeds):
            raise ValueError("trial_count must match seeds.")
        if self.trial_results and self.trial_count != len(self.trial_results):
            raise ValueError("trial_count must match stored trial_results.")
        if self.pass_count < 0 or self.reservation_count < 0 or self.failure_count < 0:
            raise ValueError("Monte Carlo counts must not be negative.")
        _fraction(self.pass_fraction, "Monte Carlo pass_fraction")
        _reason_tuple(self.reasons, "Monte Carlo reasons")
        _strings(self.diagnostics, "Monte Carlo diagnostics", allow_empty=True)
        object.__setattr__(self, "metric_distributions", MappingProxyType(dict(self.metric_distributions)))


def generate_synthetic_validation_scenario(
    scenario_name: str = "single_ideal",
    settings: SyntheticValidationSettings | None = None,
    *,
    components: Iterable[SyntheticDampedComponent] | None = None,
    scenario_id: str | None = None,
    name: str | None = None,
) -> SyntheticValidationScenario:
    """Create a deterministic validation scenario with known truth."""

    cfg = settings or SyntheticValidationSettings()
    scenario_key = scenario_name.strip().lower().replace("-", "_")
    if components is None:
        component_tuple, description, outcomes, notes = _built_in_scenario(scenario_key, cfg)
    else:
        component_tuple = tuple(sorted(components, key=lambda item: item.component_id))
        description = "Custom controlled synthetic validation scenario."
        outcomes = {
            "expected_candidate_count": sum(item.expected_detectable for item in component_tuple),
            "expected_track_count": sum(item.expected_detectable for item in component_tuple),
        }
        notes = ("custom_scenario_identifiability_must_be_declared_by_user",)
    labels = tuple(sorted({item.dynamic_label for item in component_tuple}, key=_dynamic_sort_key))
    reasons = [SyntheticValidationReason.IDENTIFIABLE_SYNTHETIC_SCENARIO]
    if any("non_identifiable" in item for item in notes):
        reasons.append(SyntheticValidationReason.NON_IDENTIFIABLE_SCENARIO_REPORTED)
    if any("beating" in item for item in notes):
        reasons.append(SyntheticValidationReason.POSSIBLE_BEATING_CONTEXT)
    if any("crossing" in item for item in notes):
        reasons.append(SyntheticValidationReason.FREQUENCY_CROSSING_CONTEXT)
    sid = scenario_id or _stable_id(
        "syn-scenario",
        scenario_key,
        component_tuple,
        synthetic_validation_settings_fingerprint(cfg),
    )
    return SyntheticValidationScenario(
        scenario_id=sid,
        name=name or scenario_key,
        description=description,
        components=component_tuple,
        dynamic_labels=labels,
        recording_layout=MappingProxyType(
            {
                "recording_count": 1,
                "condition_count": len(labels),
                "synthetic_only": True,
            }
        ),
        settings=cfg,
        expected_outcomes=MappingProxyType(outcomes),
        identifiability_notes=tuple(notes),
        valid=True,
        reasons=_ordered_reasons(reasons),
        diagnostics=(
            "synthetic_truth_defined_before_analysis",
            "controlled_validation_not_real_recording_proof",
            "no_threshold_calibration_from_this_result",
        ),
    )


def generate_synthetic_ground_truth(
    scenario: SyntheticValidationScenario,
    settings: SyntheticValidationSettings | None = None,
) -> SyntheticGroundTruth:
    """Generate the signal and known values for a scenario before analysis."""

    cfg = settings or scenario.settings
    sample_count = int(round(cfg.sample_rate_hz * cfg.duration_s))
    if sample_count <= 0:
        raise ValueError("synthetic signal must contain at least one sample.")
    times = np.arange(sample_count, dtype=np.float64) / cfg.sample_rate_hz
    clean = np.zeros(sample_count, dtype=np.float64)
    for component in sorted(scenario.components, key=lambda item: item.component_id):
        clean += _component_waveform(component, times, cfg)
    clean *= cfg.global_amplitude

    rng = np.random.default_rng(cfg.random_seed)
    noise = _noise_series(clean, times, cfg, rng)
    observed = clean + noise
    clipping_metadata: dict[str, object] = {
        "mode": cfg.clipping_mode.value,
        "threshold": cfg.clipping_threshold,
        "clipping_fraction_target": cfg.clipping_fraction,
    }
    if cfg.clipping_mode is not SyntheticClippingMode.NONE:
        observed, actual_fraction = _apply_clipping(observed, cfg)
        clipping_metadata["clipping_fraction_observed"] = actual_fraction
    clean_signal = _signal_from_array(clean, cfg.sample_rate_hz)
    noise_signal = _signal_from_array(noise, cfg.sample_rate_hz)
    observed_signal = _signal_from_array(observed, cfg.sample_rate_hz)
    known_freq = tuple(
        (item.component_id, _representative_frequency(item, cfg.duration_s))
        for item in sorted(scenario.components, key=lambda value: value.component_id)
    )
    known_tau = tuple(
        (item.component_id, item.tau_s)
        for item in sorted(scenario.components, key=lambda value: value.component_id)
    )
    known_q = tuple(
        (
            component_id,
            (pi * freq * tau if tau is not None else None),
        )
        for (component_id, freq), (_, tau) in zip(known_freq, known_tau, strict=True)
    )
    known_bandwidth = tuple(
        (
            component_id,
            (freq / q if q is not None and q > 0 else None),
        )
        for (component_id, freq), (_, q) in zip(known_freq, known_q, strict=True)
    )
    exchange_pairs = _canonical_pair_tuple(
        scenario.expected_outcomes.get("known_energy_exchange_pairs", ())
    )
    non_exchange_pairs = _canonical_pair_tuple(
        scenario.expected_outcomes.get("known_non_exchange_pairs", _all_non_exchange_pairs(scenario.components, exchange_pairs))
    )
    return SyntheticGroundTruth(
        scenario_id=scenario.scenario_id,
        components=scenario.components,
        sample_rate_hz=cfg.sample_rate_hz,
        duration_s=sample_count / cfg.sample_rate_hz,
        time_axis_s=tuple(float(value) for value in times),
        clean_signal=clean_signal,
        noise_signal=noise_signal,
        observed_signal=observed_signal,
        known_frequencies_hz=known_freq,
        known_tau_values_s=known_tau,
        known_q_values=known_q,
        known_bandwidth_values_hz=known_bandwidth,
        known_component_presence=tuple(
            (
                item.component_id,
                item.start_time_s,
                item.end_time_s if item.end_time_s is not None else sample_count / cfg.sample_rate_hz,
            )
            for item in sorted(scenario.components, key=lambda value: value.component_id)
        ),
        known_associations=tuple(scenario.expected_outcomes.get("known_associations", ())),
        known_chains=tuple(scenario.expected_outcomes.get("known_chains", ())),
        known_energy_exchange_pairs=exchange_pairs,
        known_non_exchange_pairs=non_exchange_pairs,
        noise_metadata=MappingProxyType(
            {
                "noise_model": cfg.noise_model.value,
                "signal_to_noise_ratio_db": cfg.signal_to_noise_ratio_db,
                "noise_standard_deviation": cfg.noise_standard_deviation,
                "include_colored_noise": cfg.include_colored_noise,
                "include_mains_hum": cfg.include_mains_hum,
            }
        ),
        clipping_metadata=MappingProxyType(clipping_metadata),
        settings_fingerprint=synthetic_validation_settings_fingerprint(cfg),
        diagnostics=(
            "truth_constructed_before_belllab_pipeline",
            "synthetic_truth_not_physical_experimental_truth",
            "ground_truth_not_used_inside_estimators",
        ),
    )


def run_synthetic_pipeline(
    scenario: SyntheticValidationScenario,
    ground_truth: SyntheticGroundTruth | None = None,
    settings: SyntheticValidationSettings | None = None,
) -> SyntheticPipelineOutput:
    """Run configured public BellLab stages on the synthetic observed signal."""

    cfg = settings or scenario.settings
    truth = ground_truth or generate_synthetic_ground_truth(scenario, cfg)
    signal = truth.observed_signal
    stages: list[str] = []
    errors: list[tuple[str, str]] = []
    diagnostics = [
        "public_belllab_apis_only",
        "ground_truth_not_passed_to_estimators",
        "no_tracking_correction_from_truth",
        "partial_pipeline_stages_reported_explicitly",
    ]
    temporal = None
    spectrum = None
    peaks = None
    stft = None
    frame_peaks = None
    tracking = None
    candidates: tuple[object, ...] = ()
    within: tuple[object, ...] = ()
    cross: tuple[object, ...] = ()
    chains = None
    hypotheses = None
    parameters = None
    q_result = None
    energy_result = None

    if cfg.run_temporal_analysis:
        temporal = _run_stage("temporal", errors, lambda: analyze_temporal(
            signal,
            _temporal_analysis_settings(cfg),
        ))
        if temporal is not None:
            stages.append("temporal")
    if cfg.run_global_spectrum:
        spectrum = _run_stage("spectrum", errors, lambda: analyze_spectrum(
            signal,
            _spectrum_settings(cfg),
        ))
        if spectrum is not None:
            stages.append("spectrum")
            peaks = _run_stage(
                "peaks",
                errors,
                lambda: detect_spectral_peaks(spectrum.spectrum, _peak_settings(cfg)),  # type: ignore[union-attr]
            )
            if peaks is not None:
                stages.append("peaks")
    if cfg.run_stft:
        stft = _run_stage("stft", errors, lambda: analyze_stft(signal, _stft_settings(cfg)))
        if stft is not None:
            stages.append("stft")
            if cfg.run_tracking:
                frame_peaks = _run_stage(
                    "frame_peaks",
                    errors,
                    lambda: detect_stft_peaks(stft.time_frequency, _frame_peak_settings(cfg)),
                )
                if frame_peaks is not None:
                    stages.append("frame_peaks")
                    tracking = _run_stage(
                        "tracking",
                        errors,
                        lambda: track_spectral_peaks(frame_peaks, _tracking_settings(cfg)),
                    )
                    if tracking is not None:
                        stages.append("tracking")
    if cfg.run_candidate_characterization and tracking is not None:
        def _candidates() -> tuple[object, ...]:
            characterizations = tuple(
                characterize_spectral_track(track)
                for track in tracking.tracks
            )
            return select_modal_candidates(
                characterizations,
                tracking,
                ModalCandidateSettings(
                    minimum_observation_count=cfg.candidate_minimum_observation_count,
                    minimum_coverage_fraction=None,
                    minimum_duration_s=None,
                    maximum_relative_frequency_stability=None,
                    maximum_absolute_frequency_drift_hz=None,
                    maximum_frequency_fit_rmse_hz=None,
                    require_successful_frequency_fit=False,
                    require_amplitude_decay=False,
                ),
            )

        candidates = _run_stage("candidate_characterization", errors, _candidates) or ()
        if candidates:
            stages.append("candidate_characterization")
    elif cfg.run_candidate_characterization:
        diagnostics.append("candidate_characterization_requires_tracking")

    if cfg.run_within_condition_association:
        diagnostics.append("within_condition_association_requires_recording_sets")
    if cfg.run_cross_condition_association:
        diagnostics.append("cross_condition_association_requires_adjacent_condition_sets")
    if cfg.run_candidate_chains:
        diagnostics.append("candidate_chains_require_adjacent_association_results")
    if cfg.run_modal_hypotheses:
        diagnostics.append("modal_hypotheses_require_candidate_chains")
    if cfg.run_modal_parameter_estimation:
        diagnostics.append("modal_parameter_estimation_requires_modal_hypotheses")
    if cfg.run_modal_q_estimation:
        diagnostics.append("modal_q_estimation_requires_modal_parameter_estimates")
    if cfg.run_energy_exchange_evidence:
        energy_sources = _component_envelope_sources(scenario, truth)
        if len(energy_sources) >= 2:
            energy_settings = ModalEnergyExchangeSettings(
                resampling_policy="linear_interpolation",
                maximum_lag_s=min(0.5, cfg.duration_s / 2.0),
                permutation_count=50,
                random_seed=cfg.random_seed,
                minimum_overlap_sample_count=5,
                require_pair_energy_stability=False,
            )
            energy_result = _run_stage(
                "energy_exchange",
                errors,
                lambda: evaluate_modal_energy_exchange(
                    energy_sources,
                    energy_settings,
                    dynamic_label=scenario.dynamic_labels[0] if scenario.dynamic_labels else None,
                ),
            )
            if energy_result is not None:
                stages.append("energy_exchange")
        else:
            diagnostics.append("energy_exchange_requires_at_least_two_sources")

    return SyntheticPipelineOutput(
        scenario_id=scenario.scenario_id,
        signals=(signal,),
        temporal_results=temporal,
        spectral_results=spectrum,
        peak_results=peaks,
        stft_results=stft,
        time_frequency_peak_results=frame_peaks,
        tracking_results=tracking,
        candidate_results=tuple(candidates),
        within_condition_results=within,
        cross_condition_results=cross,
        chain_results=chains,
        modal_hypothesis_results=hypotheses,
        modal_parameter_results=parameters,
        modal_q_results=q_result,
        energy_exchange_results=energy_result,
        pipeline_stages_completed=tuple(stages),
        pipeline_errors=tuple(errors),
        valid=not errors,
        diagnostics=tuple(dict.fromkeys(diagnostics)),
    )


def validate_synthetic_frequency(
    true_frequency_hz: float,
    estimated_frequency_hz: float | None,
    settings: SyntheticValidationSettings | None = None,
    *,
    true_trajectory_hz: Sequence[float] | None = None,
    estimated_trajectory_hz: Sequence[float] | None = None,
    true_slope_hz_per_s: float | None = None,
    estimated_slope_hz_per_s: float | None = None,
) -> SyntheticFrequencyValidation:
    """Validate recovered frequency and optional trajectory metrics."""

    cfg = settings or SyntheticValidationSettings()
    _finite_positive(true_frequency_hz, "true_frequency_hz")
    reasons: list[SyntheticValidationReason] = []
    diagnostics = ["frequency_validation_is_operational"]
    if estimated_frequency_hz is None or not isfinite(estimated_frequency_hz) or estimated_frequency_hz <= 0:
        reasons.append(SyntheticValidationReason.NO_VALID_ESTIMATE)
        return SyntheticFrequencyValidation(
            true_frequency_hz,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            False,
            _ordered_reasons(reasons),
            tuple(diagnostics),
        )
    signed = estimated_frequency_hz - true_frequency_hz
    absolute = abs(signed)
    relative = absolute / true_frequency_hz
    absolute_ok = (
        True
        if cfg.maximum_frequency_absolute_error_hz is None
        else absolute <= cfg.maximum_frequency_absolute_error_hz
    )
    relative_ok = (
        True
        if cfg.maximum_frequency_relative_error is None
        else relative <= cfg.maximum_frequency_relative_error
    )
    trajectory = _trajectory_error_metrics(
        true_trajectory_hz,
        estimated_trajectory_hz,
        true_slope_hz_per_s,
        estimated_slope_hz_per_s,
    )
    passed = absolute_ok and relative_ok
    reasons.append(
        SyntheticValidationReason.FREQUENCY_ERROR_WITHIN_TOLERANCE
        if passed
        else SyntheticValidationReason.FREQUENCY_ERROR_EXCEEDS_TOLERANCE
    )
    return SyntheticFrequencyValidation(
        true_frequency_hz,
        estimated_frequency_hz,
        absolute,
        relative,
        signed,
        absolute_ok,
        relative_ok,
        trajectory["rmse"],
        trajectory["mae"],
        trajectory["max_error"],
        trajectory["slope_error"],
        trajectory["total_change_error"],
        passed,
        _ordered_reasons(reasons),
        tuple(diagnostics),
    )


def validate_synthetic_decay(
    true_tau_s: float,
    estimated_tau_s: float | None,
    settings: SyntheticValidationSettings | None = None,
    *,
    fit_quality: float | None = None,
) -> SyntheticDecayValidation:
    """Validate recovered amplitude-decay tau without substituting absence."""

    cfg = settings or SyntheticValidationSettings()
    _finite_positive(true_tau_s, "true_tau_s")
    if estimated_tau_s is None or not isfinite(estimated_tau_s) or estimated_tau_s <= 0:
        return SyntheticDecayValidation(
            true_tau_s,
            None,
            None,
            None,
            None,
            None,
            fit_quality,
            False,
            (SyntheticValidationReason.NO_VALID_ESTIMATE,),
            ("decay_validation_preserved_missing_tau",),
        )
    absolute = abs(estimated_tau_s - true_tau_s)
    relative = absolute / true_tau_s
    log_error = abs(log(estimated_tau_s / true_tau_s))
    within = (
        True
        if cfg.maximum_tau_relative_error is None
        else relative <= cfg.maximum_tau_relative_error
    )
    return SyntheticDecayValidation(
        true_tau_s,
        estimated_tau_s,
        absolute,
        relative,
        log_error,
        within,
        fit_quality,
        within,
        (
            SyntheticValidationReason.TAU_ERROR_WITHIN_TOLERANCE
            if within
            else SyntheticValidationReason.TAU_ERROR_EXCEEDS_TOLERANCE,
        ),
        ("decay_tau_is_operational_synthetic_metric",),
    )


def validate_synthetic_q(
    true_q: float,
    estimated_q_decay: float | None = None,
    estimated_q_bandwidth: float | None = None,
    representative_q: float | None = None,
    settings: SyntheticValidationSettings | None = None,
    *,
    method_consistency: str | None = None,
) -> SyntheticQValidation:
    """Validate Q values generated by the documented synthetic convention."""

    cfg = settings or SyntheticValidationSettings()
    _finite_positive(true_q, "true_q")
    decay_error = _relative_error(true_q, estimated_q_decay)
    bandwidth_error = _relative_error(true_q, estimated_q_bandwidth)
    rep_error = _relative_error(true_q, representative_q)
    available = tuple(
        value for value in (decay_error, bandwidth_error, rep_error)
        if value is not None
    )
    if not available:
        return SyntheticQValidation(
            true_q,
            None,
            None,
            None,
            None,
            None,
            None,
            method_consistency,
            None,
            False,
            (SyntheticValidationReason.NO_VALID_ESTIMATE,),
            (
                "q_validation_preserved_missing_values",
                "q_truth_uses_Q_equals_pi_f_tau_for_compatible_components",
            ),
        )
    limit = cfg.maximum_q_relative_error
    within = True if limit is None else min(available) <= limit
    return SyntheticQValidation(
        true_q,
        estimated_q_decay,
        estimated_q_bandwidth,
        representative_q,
        decay_error,
        bandwidth_error,
        rep_error,
        method_consistency,
        within,
        within,
        (
            SyntheticValidationReason.Q_ERROR_WITHIN_TOLERANCE
            if within
            else SyntheticValidationReason.Q_ERROR_EXCEEDS_TOLERANCE,
        ),
        ("q_truth_uses_Q_equals_pi_f_tau_for_compatible_components",),
    )


def validate_synthetic_bandwidth(
    true_bandwidth_hz: float | None,
    estimated_bandwidth_hz: float | None,
    settings: SyntheticValidationSettings | None = None,
    *,
    frequency_resolution_hz: float | None = None,
) -> SyntheticBandwidthValidation:
    """Validate bandwidth when the scenario is identifiable by bandwidth."""

    cfg = settings or SyntheticValidationSettings()
    if true_bandwidth_hz is None:
        return SyntheticBandwidthValidation(
            None,
            estimated_bandwidth_hz,
            None,
            None,
            frequency_resolution_hz,
            None,
            None,
            False,
            (SyntheticValidationReason.SCENARIO_NOT_IDENTIFIABLE,),
            ("bandwidth_validation_not_forced",),
        )
    _finite_positive(true_bandwidth_hz, "true_bandwidth_hz")
    if estimated_bandwidth_hz is None or not isfinite(estimated_bandwidth_hz) or estimated_bandwidth_hz <= 0:
        return SyntheticBandwidthValidation(
            true_bandwidth_hz,
            None,
            None,
            None,
            frequency_resolution_hz,
            None,
            None,
            False,
            (SyntheticValidationReason.NO_VALID_ESTIMATE,),
            ("missing_bandwidth_not_replaced_by_zero",),
        )
    absolute = abs(estimated_bandwidth_hz - true_bandwidth_hz)
    relative = absolute / true_bandwidth_hz
    ratio = (
        estimated_bandwidth_hz / frequency_resolution_hz
        if frequency_resolution_hz is not None and frequency_resolution_hz > 0
        else None
    )
    within = (
        True
        if cfg.maximum_bandwidth_relative_error is None
        else relative <= cfg.maximum_bandwidth_relative_error
    )
    reasons = [
        SyntheticValidationReason.BANDWIDTH_ERROR_WITHIN_TOLERANCE
        if within
        else SyntheticValidationReason.BANDWIDTH_ERROR_EXCEEDS_TOLERANCE
    ]
    if ratio is not None and ratio < 2.0:
        reasons.append(SyntheticValidationReason.RESOLUTION_LIMITED)
    return SyntheticBandwidthValidation(
        true_bandwidth_hz,
        estimated_bandwidth_hz,
        absolute,
        relative,
        frequency_resolution_hz,
        ratio,
        within,
        within,
        _ordered_reasons(reasons),
        ("bandwidth_validation_is_operational",),
    )


def validate_synthetic_tracking(
    ground_truth: SyntheticGroundTruth,
    tracking_results: SpectralTrackingResults | None,
    settings: SyntheticValidationSettings | None = None,
) -> SyntheticTrackingValidation:
    """Validate tracking by an explicit nearest-frequency matching policy."""

    cfg = settings or SyntheticValidationSettings()
    expected = tuple(item for item in ground_truth.components if item.expected_detectable)
    tracks = tuple(tracking_results.tracks) if tracking_results is not None else ()
    matches, missed, false = _match_tracks_to_components(expected, tracks, cfg)
    coverage: list[tuple[str, float]] = []
    rmse: list[tuple[str, float | None]] = []
    gaps: list[tuple[str, float]] = []
    matched_track_ids = {track_id for _, track_id in matches}
    track_by_id = {track.track_id: track for track in tracks}
    for component_id, track_id in matches:
        track = track_by_id[track_id]
        coverage.append((component_id, min(1.0, track.duration_s / ground_truth.duration_s)))
        rmse.append((component_id, None))
        denominator = max(1, track.last_frame - track.first_frame)
        gaps.append((component_id, track.total_missing_frames / denominator))
    false_tracks = tuple(track.track_id for track in tracks if track.track_id not in matched_track_ids)
    gap_ok = all(value <= cfg.maximum_tracking_gap_fraction for _, value in gaps)
    passed = not missed and len(false_tracks) == 0 and gap_ok
    reasons: list[SyntheticValidationReason] = []
    if passed:
        reasons.extend((SyntheticValidationReason.TRACKING_RECOVERED, SyntheticValidationReason.NO_TRACK_GAP_EXCESS))
    else:
        if missed:
            reasons.append(SyntheticValidationReason.TRACK_FRAGMENTATION)
        if false_tracks:
            reasons.append(SyntheticValidationReason.FALSE_CANDIDATE_DETECTED)
        if not gap_ok:
            reasons.append(SyntheticValidationReason.TRACK_FRAGMENTATION)
    return SyntheticTrackingValidation(
        expected_track_count=len(expected),
        recovered_track_count=len(tracks),
        matched_track_pairs=tuple(matches),
        missed_true_tracks=tuple(missed),
        false_tracks=tuple(false_tracks),
        fragmented_tracks=tuple(missed if missed else ()),
        track_swaps=(),
        coverage_fractions=tuple(coverage),
        frequency_rmse_values_hz=tuple(rmse),
        gap_fractions=tuple(gaps),
        passed=passed,
        reasons=_ordered_reasons(reasons or [SyntheticValidationReason.INSUFFICIENT_EVIDENCE]),
        diagnostics=(
            "tracking_validation_uses_truth_only_for_posthoc_matching",
            "tracking_algorithm_not_modified",
        ),
    )


def validate_synthetic_candidates(
    ground_truth: SyntheticGroundTruth,
    candidates: Iterable[object],
    settings: SyntheticValidationSettings | None = None,
) -> SyntheticCandidateValidation:
    """Validate candidate recovery counts and nearest-frequency matches."""

    cfg = settings or SyntheticValidationSettings()
    candidate_tuple = tuple(candidates)
    expected = tuple(item for item in ground_truth.components if item.expected_detectable)
    matched: list[tuple[str, int]] = []
    used: set[int] = set()
    for component in expected:
        true_freq = _truth_frequency(ground_truth, component.component_id)
        best_id = None
        best_error = None
        for index, candidate in enumerate(candidate_tuple):
            if index in used:
                continue
            frequency = getattr(candidate, "representative_frequency_hz", None)
            if frequency is None or not isfinite(frequency) or frequency <= 0:
                continue
            error = abs(frequency - true_freq)
            if best_error is None or error < best_error:
                best_error = error
                best_id = getattr(candidate, "candidate_id", index)
        if best_id is not None and _frequency_error_within_limits(best_error or 0.0, true_freq, cfg):
            matched.append((component.component_id, int(best_id)))
            used.add(int(best_id))
    matched_components = {component_id for component_id, _ in matched}
    missed = tuple(component.component_id for component in expected if component.component_id not in matched_components)
    false_ids = tuple(
        int(getattr(candidate, "candidate_id", index))
        for index, candidate in enumerate(candidate_tuple)
        if int(getattr(candidate, "candidate_id", index)) not in used
        and bool(getattr(candidate, "accepted", True))
    )
    rejected_expected = tuple(
        component_id
        for component_id, candidate_id in matched
        for candidate in candidate_tuple
        if int(getattr(candidate, "candidate_id", -1)) == candidate_id
        and not bool(getattr(candidate, "accepted", True))
    )
    passed = (
        len(missed) <= cfg.maximum_missed_candidate_count
        and len(false_ids) <= cfg.maximum_false_candidate_count
        and not rejected_expected
    )
    reasons: list[SyntheticValidationReason] = []
    if passed:
        reasons.extend(
            (
                SyntheticValidationReason.CANDIDATE_COUNT_RECOVERED,
                SyntheticValidationReason.NO_FALSE_CANDIDATE_EXCESS,
                SyntheticValidationReason.NO_MISSED_CANDIDATE_EXCESS,
            )
        )
    else:
        if missed:
            reasons.append(SyntheticValidationReason.CANDIDATE_MISSED)
        if false_ids:
            reasons.append(SyntheticValidationReason.FALSE_CANDIDATE_DETECTED)
    return SyntheticCandidateValidation(
        len(expected),
        len(candidate_tuple),
        tuple(matched),
        missed,
        false_ids,
        rejected_expected,
        (),
        passed,
        _ordered_reasons(reasons or [SyntheticValidationReason.INSUFFICIENT_EVIDENCE]),
        ("candidate_validation_uses_operational_candidate_contracts",),
    )


def validate_synthetic_associations(
    expected_pairs: Iterable[tuple[str, str]],
    recovered_pairs: Iterable[tuple[str, str]],
    settings: SyntheticValidationSettings | None = None,
    *,
    emerging_expected: Iterable[str] = (),
    emerging_recovered: Iterable[str] = (),
    disappearing_expected: Iterable[str] = (),
    disappearing_recovered: Iterable[str] = (),
) -> SyntheticAssociationValidation:
    """Validate within- or cross-condition association pairs by canonical content."""

    _ = settings or SyntheticValidationSettings()
    expected = _canonical_pair_tuple(expected_pairs)
    recovered = _canonical_pair_tuple(recovered_pairs)
    correct, missing, incorrect, precision, recall, f1 = _set_metrics(expected, recovered)
    passed = not missing and not incorrect
    reasons = (
        (SyntheticValidationReason.ASSOCIATION_RECOVERED,)
        if passed
        else (SyntheticValidationReason.ASSOCIATION_MISMATCH,)
    )
    return SyntheticAssociationValidation(
        expected,
        recovered,
        correct,
        missing,
        incorrect,
        tuple(sorted(emerging_expected)),
        tuple(sorted(emerging_recovered)),
        tuple(sorted(disappearing_expected)),
        tuple(sorted(disappearing_recovered)),
        precision,
        recall,
        f1,
        passed,
        reasons,
        ("association_validation_does_not_modify_belllab_matching",),
    )


def validate_synthetic_chains(
    expected_chains: Iterable[Sequence[str]],
    recovered_chains: Iterable[Sequence[str]],
    settings: SyntheticValidationSettings | None = None,
) -> SyntheticChainValidation:
    """Validate chain content without assuming operational IDs match synthetic IDs."""

    cfg = settings or SyntheticValidationSettings()
    expected = tuple(sorted(tuple(chain) for chain in expected_chains))
    recovered = tuple(sorted(tuple(chain) for chain in recovered_chains))
    if not expected and not recovered:
        return SyntheticChainValidation(
            (),
            (),
            (),
            (),
            (),
            (),
            1.0,
            1.0,
            1.0,
            True,
            (SyntheticValidationReason.CHAIN_RECOVERED,),
            ("no_chain_expected_or_recovered",),
        )
    expected_set = set(expected)
    recovered_set = set(recovered)
    exact = tuple(sorted(expected_set & recovered_set))
    missing = tuple(sorted(expected_set - recovered_set))
    spurious = tuple(sorted(recovered_set - expected_set))
    expected_nodes = {node for chain in expected for node in chain}
    recovered_nodes = {node for chain in recovered for node in chain}
    expected_edges = {
        (left, right)
        for chain in expected
        for left, right in zip(chain, chain[1:])
    }
    recovered_edges = {
        (left, right)
        for chain in recovered
        for left, right in zip(chain, chain[1:])
    }
    node_fraction = _safe_fraction(len(expected_nodes & recovered_nodes), len(expected_nodes))
    edge_fraction = _safe_fraction(len(expected_edges & recovered_edges), len(expected_edges))
    chain_fraction = _safe_fraction(len(exact), len(expected))
    passed = (
        chain_fraction is not None
        and chain_fraction >= cfg.minimum_chain_recovery_fraction
        and not spurious
    )
    reasons = (
        (SyntheticValidationReason.CHAIN_RECOVERED, SyntheticValidationReason.NO_CHAIN_RECOVERY_DEFICIT)
        if passed
        else (SyntheticValidationReason.CHAIN_MISMATCH,)
    )
    return SyntheticChainValidation(
        expected,
        recovered,
        exact,
        (),
        missing,
        spurious,
        node_fraction,
        edge_fraction,
        chain_fraction,
        passed,
        _ordered_reasons(reasons),
        ("chain_validation_compares_canonical_content_only",),
    )


def validate_synthetic_modal_hypotheses(
    expected_statuses: Iterable[tuple[str, str]],
    recovered_statuses: Iterable[tuple[str, str]] | ModalHypothesisResult | None,
    settings: SyntheticValidationSettings | None = None,
) -> SyntheticModalHypothesisValidation:
    """Validate modal-hypothesis status recovery without physical promotion."""

    cfg = settings or SyntheticValidationSettings()
    expected = tuple(sorted((str(key), str(value)) for key, value in expected_statuses))
    if isinstance(recovered_statuses, ModalHypothesisResult):
        recovered = tuple(
            sorted(
                (
                    item.source_chain_id or item.hypothesis_id,
                    item.status.value,
                )
                for item in recovered_statuses.hypotheses
            )
        )
    elif recovered_statuses is None:
        recovered = ()
    else:
        recovered = tuple(sorted((str(key), str(value)) for key, value in recovered_statuses))
    if not expected and not recovered:
        return SyntheticModalHypothesisValidation(
            (),
            (),
            0,
            (),
            (),
            (),
            (),
            1.0,
            True,
            (SyntheticValidationReason.HYPOTHESIS_STATUS_RECOVERED,),
            (
                "no_hypothesis_expected_or_recovered",
                "no_ModalMode_created",
            ),
        )
    lookup = dict(recovered)
    correct = sum(lookup.get(key) == value for key, value in expected)
    status_pairs = tuple(
        sorted(
            (expected_value, lookup.get(key, "missing"), 1)
            for key, expected_value in expected
        )
    )
    fraction = _safe_fraction(correct, len(expected))
    passed = fraction is not None and fraction >= cfg.minimum_hypothesis_recovery_fraction
    reasons = (
        (SyntheticValidationReason.HYPOTHESIS_STATUS_RECOVERED, SyntheticValidationReason.NO_HYPOTHESIS_RECOVERY_DEFICIT)
        if passed
        else (SyntheticValidationReason.HYPOTHESIS_MISMATCH,)
    )
    accepted_true = tuple(key for key, value in recovered if value in {ModalHypothesisStatus.ACCEPTED.value, ModalHypothesisStatus.ACCEPTED_WITH_RESERVATIONS.value})
    rejected_true = tuple(key for key, value in recovered if value == ModalHypothesisStatus.REJECTED.value)
    return SyntheticModalHypothesisValidation(
        expected,
        recovered,
        correct,
        status_pairs,
        accepted_true,
        rejected_true,
        (),
        fraction,
        passed,
        _ordered_reasons(reasons),
        (
            "accepted_synthetic_hypothesis_is_not_physical_modal_identity",
            "no_ModalMode_created",
        ),
    )


def validate_synthetic_energy_exchange(
    expected_exchange_pairs: Iterable[tuple[str, str]],
    expected_non_exchange_pairs: Iterable[tuple[str, str]],
    recovered: ModalEnergyExchangeResult | Iterable[tuple[str, str]] | None,
    settings: SyntheticValidationSettings | None = None,
    *,
    expected_lags_s: Mapping[tuple[str, str], float] | None = None,
) -> SyntheticEnergyExchangeValidation:
    """Validate operational possible redistribution evidence by pair status."""

    _ = settings or SyntheticValidationSettings()
    expected_exchange = _canonical_pair_tuple(expected_exchange_pairs)
    expected_non_exchange = _canonical_pair_tuple(expected_non_exchange_pairs)
    if isinstance(recovered, ModalEnergyExchangeResult):
        supported = _canonical_pair_tuple(
            (item.source_a_id, item.source_b_id)
            for item in recovered.pair_evidences
            if item.status
            in {
                ModalEnergyExchangeStatus.SUPPORTED,
                ModalEnergyExchangeStatus.SUPPORTED_WITH_RESERVATIONS,
            }
        )
        not_supported = _canonical_pair_tuple(
            (item.source_a_id, item.source_b_id)
            for item in recovered.pair_evidences
            if item.status is ModalEnergyExchangeStatus.NOT_SUPPORTED
        )
        inconclusive = _canonical_pair_tuple(
            (item.source_a_id, item.source_b_id)
            for item in recovered.pair_evidences
            if item.status is ModalEnergyExchangeStatus.INCONCLUSIVE
        )
        lag_errors = _lag_errors_from_energy_result(recovered, expected_lags_s or {})
    elif recovered is None:
        supported = ()
        not_supported = ()
        inconclusive = ()
        lag_errors = ()
    else:
        supported = _canonical_pair_tuple(recovered)
        not_supported = ()
        inconclusive = ()
        lag_errors = ()
    false_positive = tuple(pair for pair in supported if pair in expected_non_exchange)
    false_negative = tuple(pair for pair in expected_exchange if pair not in supported)
    correct, _, _, precision, recall, f1 = _set_metrics(expected_exchange, supported)
    passed = not false_positive and not false_negative
    reasons: list[SyntheticValidationReason] = []
    if passed:
        reasons.append(SyntheticValidationReason.ENERGY_EXCHANGE_PATTERN_RECOVERED)
        if not false_positive:
            reasons.append(SyntheticValidationReason.NO_FALSE_ENERGY_EXCHANGE_SUPPORT)
    else:
        if false_positive:
            reasons.append(SyntheticValidationReason.ENERGY_EXCHANGE_FALSE_POSITIVE)
        if false_negative:
            reasons.append(SyntheticValidationReason.ENERGY_EXCHANGE_FALSE_NEGATIVE)
    return SyntheticEnergyExchangeValidation(
        expected_exchange,
        expected_non_exchange,
        supported,
        not_supported,
        false_positive,
        false_negative,
        inconclusive,
        precision if correct or supported else (1.0 if not expected_exchange and not supported else precision),
        recall if expected_exchange else 1.0,
        f1,
        lag_errors,
        passed,
        _ordered_reasons(reasons or [SyntheticValidationReason.INSUFFICIENT_EVIDENCE]),
        (
            "operational_possible_redistribution_only",
            "no_physical_energy_transfer_or_causality_inferred",
        ),
    )


def validate_synthetic_scenario(
    scenario: SyntheticValidationScenario,
    settings: SyntheticValidationSettings | None = None,
) -> SyntheticScenarioValidationResult:
    """Run and validate one complete synthetic scenario."""

    cfg = settings or scenario.settings
    if not scenario.valid:
        truth = generate_synthetic_ground_truth(scenario, cfg)
        pipeline = run_synthetic_pipeline(scenario, truth, cfg)
        empty_tracking = validate_synthetic_tracking(truth, None, cfg)
        empty_candidates = validate_synthetic_candidates(truth, (), cfg)
        empty_assoc = validate_synthetic_associations((), (), cfg)
        empty_chains = validate_synthetic_chains((), (), cfg)
        empty_hyp = validate_synthetic_modal_hypotheses((), None, cfg)
        empty_energy = validate_synthetic_energy_exchange((), (), None, cfg)
        return SyntheticScenarioValidationResult(
            scenario,
            truth,
            pipeline,
            (),
            (),
            (),
            (),
            empty_tracking,
            empty_candidates,
            empty_assoc,
            empty_chains,
            empty_hyp,
            empty_energy,
            SyntheticValidationStatus.INVALID_SCENARIO,
            (),
            (),
            (),
            (SyntheticValidationReason.INVALID_COMPONENT,),
            (),
            0,
            0,
            0,
            False,
            ("invalid_scenario_not_forced",),
        )

    truth = generate_synthetic_ground_truth(scenario, cfg)
    pipeline = run_synthetic_pipeline(scenario, truth, cfg)
    frequency_validations: list[SyntheticFrequencyValidation] = []
    decay_validations: list[SyntheticDecayValidation] = []
    q_validations: list[SyntheticQValidation] = []
    bandwidth_validations: list[SyntheticBandwidthValidation] = []
    component_lookup = {component.component_id: component for component in truth.components}
    tracks = tuple(pipeline.tracking_results.tracks) if pipeline.tracking_results is not None else ()
    characterizations = {
        track.track_id: characterize_spectral_track(track)
        for track in tracks
    }
    for component_id, true_frequency in truth.known_frequencies_hz:
        estimated_frequency = _estimated_frequency(component_id, true_frequency, pipeline)
        component = component_lookup[component_id]
        frequency_validations.append(
            validate_synthetic_frequency(
                true_frequency,
                estimated_frequency,
                cfg,
                true_slope_hz_per_s=component.frequency_drift_hz_per_s,
                estimated_slope_hz_per_s=_estimated_track_slope(true_frequency, tracks, characterizations),
            )
        )
    for component_id, true_tau in truth.known_tau_values_s:
        if true_tau is None:
            continue
        estimated_tau, fit_quality = _estimated_tau(component_id, _truth_frequency(truth, component_id), tracks, characterizations)
        decay_validations.append(
            validate_synthetic_decay(true_tau, estimated_tau, cfg, fit_quality=fit_quality)
        )
    for component_id, true_q in truth.known_q_values:
        if true_q is None:
            continue
        frequency = _estimated_frequency(component_id, _truth_frequency(truth, component_id), pipeline)
        tau, _quality = _estimated_tau(component_id, _truth_frequency(truth, component_id), tracks, characterizations)
        q_decay = None
        if frequency is not None and tau is not None:
            decay_q = estimate_q_from_decay(
                representative_frequency_hz=frequency,
                representative_tau_s=tau,
            )
            q_decay = decay_q.q_decay
        q_validations.append(validate_synthetic_q(true_q, q_decay, None, q_decay, cfg))
    for component_id, true_bandwidth in truth.known_bandwidth_values_hz:
        estimated_bandwidth = None
        resolution = None
        if pipeline.spectral_results is not None and pipeline.spectral_results.spectrum is not None:
            spectrum = pipeline.spectral_results.spectrum
            resolution = spectrum.bin_spacing_hz
            source = ModalBandwidthSource(
                spectrum_id=f"{scenario.scenario_id}:spectrum",
                center_frequency_hz=_truth_frequency(truth, component_id),
                frequency_axis_hz=spectrum.frequencies_hz,
                magnitude_values=spectrum.magnitudes,
                peak_frequencies_hz=tuple(freq for _, freq in truth.known_frequencies_hz),
                frequency_resolution_hz=spectrum.bin_spacing_hz,
                diagnostics=("synthetic_validation_bandwidth_source",),
            )
            bandwidth = estimate_modal_bandwidth(
                source.center_frequency_hz,
                source.frequency_axis_hz,
                source.magnitude_values,
                peak_frequencies_hz=source.peak_frequencies_hz,
                frequency_resolution_hz=source.frequency_resolution_hz,
            )
            if bandwidth.valid:
                estimated_bandwidth = bandwidth.bandwidth_hz
                q_bw = estimate_q_from_bandwidth(bandwidth)
                if q_bw.valid and q_bw.q_bandwidth is not None:
                    pass
        bandwidth_validations.append(
            validate_synthetic_bandwidth(true_bandwidth, estimated_bandwidth, cfg, frequency_resolution_hz=resolution)
        )
    tracking_validation = validate_synthetic_tracking(truth, pipeline.tracking_results, cfg)
    candidate_validation = validate_synthetic_candidates(truth, pipeline.candidate_results, cfg)
    association_validation = validate_synthetic_associations(
        truth.known_associations,
        (),
        cfg,
        emerging_expected=tuple(scenario.expected_outcomes.get("emerging_candidates_expected", ())),
        disappearing_expected=tuple(scenario.expected_outcomes.get("disappearing_candidates_expected", ())),
    )
    chain_validation = validate_synthetic_chains(truth.known_chains, (), cfg)
    hypothesis_validation = validate_synthetic_modal_hypotheses(
        tuple(scenario.expected_outcomes.get("expected_hypothesis_statuses", ())),
        pipeline.modal_hypothesis_results,
        cfg,
    )
    energy_validation = validate_synthetic_energy_exchange(
        truth.known_energy_exchange_pairs,
        truth.known_non_exchange_pairs,
        pipeline.energy_exchange_results,
        cfg,
        expected_lags_s=scenario.expected_outcomes.get("known_energy_exchange_lags_s", {}),
    )
    all_metric_states = [
        *(item.passed for item in frequency_validations),
        *(item.passed for item in decay_validations),
        *(item.passed for item in q_validations),
        *(item.passed for item in bandwidth_validations if SyntheticValidationReason.SCENARIO_NOT_IDENTIFIABLE not in item.reasons),
        tracking_validation.passed,
        candidate_validation.passed,
    ]
    optional_metrics = (
        association_validation,
        chain_validation,
        hypothesis_validation,
        energy_validation,
    )
    all_metric_states.extend(
        metric.passed
        for metric in optional_metrics
        if metric.reasons
        and SyntheticValidationReason.INSUFFICIENT_EVIDENCE not in metric.reasons
    )
    passed_count = sum(all_metric_states)
    failed_count = sum(not value for value in all_metric_states)
    inconclusive_count = sum(
        SyntheticValidationReason.SCENARIO_NOT_IDENTIFIABLE in item.reasons
        for item in bandwidth_validations
    )
    support: list[SyntheticValidationReason] = [
        SyntheticValidationReason.NO_GENERAL_PHYSICAL_VALIDITY_CLAIM,
        SyntheticValidationReason.NO_GROUND_TRUTH_USED_BY_ESTIMATOR,
        SyntheticValidationReason.NO_THRESHOLD_CALIBRATION_FROM_RESULT,
        SyntheticValidationReason.NO_TRACKING_CORRECTION_FROM_TRUTH,
        SyntheticValidationReason.NO_PHYSICAL_SPLIT_OR_MERGE_RESOLUTION,
        SyntheticValidationReason.NO_CAUSALITY_INFERRED,
        SyntheticValidationReason.NO_MODAL_MODE_PROMOTION,
        SyntheticValidationReason.NO_AUDIO_FILE_READ,
        SyntheticValidationReason.NO_EXTERNAL_DATA_READ,
        SyntheticValidationReason.NO_HIDDEN_TRUE_VALUE_INPUT,
        SyntheticValidationReason.NO_GLOBAL_RNG_MUTATION,
        SyntheticValidationReason.NO_INPUT_MUTATION,
    ]
    reservations: list[SyntheticValidationReason] = []
    insufficiencies: list[SyntheticValidationReason] = []
    failures: list[SyntheticValidationReason] = []
    if "noise" in scenario.name and (cfg.signal_to_noise_ratio_db is not None and cfg.signal_to_noise_ratio_db <= 20):
        reservations.append(SyntheticValidationReason.LOW_SIGNAL_TO_NOISE_RATIO)
    if "beating" in scenario.name:
        reservations.append(SyntheticValidationReason.POSSIBLE_BEATING_CONTEXT)
    if "near_modes" in scenario.name:
        reservations.append(SyntheticValidationReason.NEIGHBORING_MODE_INTERFERENCE)
    if "crossing" in scenario.name:
        reservations.append(SyntheticValidationReason.FREQUENCY_CROSSING_CONTEXT)
    if "clipping" in scenario.name:
        reservations.append(SyntheticValidationReason.MILD_CLIPPING)
    if any("non_identifiable" in note for note in scenario.identifiability_notes):
        insufficiencies.append(SyntheticValidationReason.SCENARIO_NOT_IDENTIFIABLE)
    if pipeline.pipeline_errors:
        status = SyntheticValidationStatus.PIPELINE_ERROR
        pipeline_error_reasons = (SyntheticValidationReason.UNEXPECTED_PIPELINE_EXCEPTION,)
    elif failures:
        status = SyntheticValidationStatus.FAILED
        pipeline_error_reasons = ()
    elif failed_count:
        status = SyntheticValidationStatus.FAILED
        failures.append(SyntheticValidationReason.NO_VALID_ESTIMATE)
        pipeline_error_reasons = ()
    elif insufficiencies and not passed_count:
        status = SyntheticValidationStatus.INSUFFICIENT_EVIDENCE
        pipeline_error_reasons = ()
    elif reservations or insufficiencies or inconclusive_count:
        status = SyntheticValidationStatus.PASSED_WITH_RESERVATIONS
        pipeline_error_reasons = ()
    else:
        status = SyntheticValidationStatus.PASSED
        pipeline_error_reasons = ()
    return SyntheticScenarioValidationResult(
        scenario,
        truth,
        pipeline,
        tuple(frequency_validations),
        tuple(decay_validations),
        tuple(q_validations),
        tuple(bandwidth_validations),
        tracking_validation,
        candidate_validation,
        association_validation,
        chain_validation,
        hypothesis_validation,
        energy_validation,
        status,
        _ordered_reasons(support),
        _ordered_reasons(reservations),
        _ordered_reasons(failures),
        _ordered_reasons(insufficiencies),
        _ordered_reasons(pipeline_error_reasons),
        passed_count,
        failed_count,
        inconclusive_count,
        status in {
            SyntheticValidationStatus.PASSED,
            SyntheticValidationStatus.PASSED_WITH_RESERVATIONS,
        },
        (
            "synthetic_validation_is_controlled_operational_evidence_only",
            "truth_not_used_to_calibrate_thresholds_or_repair_pipeline",
            "no_general_real_recording_validity_claim",
        ),
    )


def run_synthetic_validation_campaign(
    scenarios: Iterable[SyntheticValidationScenario] | None = None,
    settings: SyntheticValidationSettings | None = None,
) -> SyntheticValidationCampaignResult:
    """Run a deterministic campaign over one result per scenario."""

    cfg = settings or SyntheticValidationSettings()
    scenario_tuple = (
        tuple(_default_campaign_scenarios(cfg))
        if scenarios is None
        else tuple(sorted(scenarios, key=lambda item: item.scenario_id))
    )
    results = tuple(validate_synthetic_scenario(scenario, cfg) for scenario in scenario_tuple)
    results = tuple(sorted(results, key=lambda item: item.scenario.scenario_id))
    passed = tuple(item for item in results if item.status is SyntheticValidationStatus.PASSED)
    reserved = tuple(item for item in results if item.status is SyntheticValidationStatus.PASSED_WITH_RESERVATIONS)
    pass_fraction = _safe_fraction(len(passed) + len(reserved), len(results)) or 0.0
    valid = pass_fraction >= cfg.minimum_pass_fraction
    failure_reason = None if valid else "pass_fraction_below_minimum"
    return SyntheticValidationCampaignResult(
        campaign_id=_stable_id(
            "syn-campaign",
            tuple(item.scenario_id for item in scenario_tuple),
            synthetic_validation_settings_fingerprint(cfg),
        ),
        scenario_results=results,
        scenario_count=len(results),
        passed_count=len(passed),
        passed_with_reservations_count=len(reserved),
        failed_count=sum(item.status is SyntheticValidationStatus.FAILED for item in results),
        inconclusive_count=sum(item.status is SyntheticValidationStatus.INCONCLUSIVE for item in results),
        insufficient_evidence_count=sum(item.status is SyntheticValidationStatus.INSUFFICIENT_EVIDENCE for item in results),
        invalid_scenario_count=sum(item.status is SyntheticValidationStatus.INVALID_SCENARIO for item in results),
        pipeline_error_count=sum(item.status is SyntheticValidationStatus.PIPELINE_ERROR for item in results),
        pass_fraction=pass_fraction,
        metric_summaries=MappingProxyType(_metric_summaries(results)),
        settings=cfg,
        valid=valid,
        failure_reason=failure_reason,
        diagnostics=(
            "one_result_per_scenario",
            "deterministic_campaign_order",
            "synthetic_success_not_general_validity_proof",
        ),
    )


def run_synthetic_monte_carlo_validation(
    base_scenario: SyntheticValidationScenario,
    settings: SyntheticValidationSettings | None = None,
) -> SyntheticMonteCarloValidation:
    """Run repeated deterministic trials using explicit seed progression."""

    cfg = settings or base_scenario.settings
    base_seed = cfg.random_seed or 0
    seeds = tuple(base_seed + index * cfg.trial_seed_stride for index in range(cfg.trial_count))
    trial_results: list[SyntheticScenarioValidationResult] = []
    for index, seed in enumerate(seeds):
        trial_settings = replace(cfg, random_seed=seed)
        scenario = replace(
            base_scenario,
            scenario_id=_stable_id(base_scenario.scenario_id, "trial", index, seed),
            settings=trial_settings,
        )
        trial_results.append(validate_synthetic_scenario(scenario, trial_settings))
    stored_results = tuple(trial_results) if cfg.store_trial_details else ()
    pass_count = sum(
        item.status
        in {
            SyntheticValidationStatus.PASSED,
            SyntheticValidationStatus.PASSED_WITH_RESERVATIONS,
        }
        for item in trial_results
    )
    reservation_count = sum(
        item.status is SyntheticValidationStatus.PASSED_WITH_RESERVATIONS
        for item in trial_results
    )
    failure_count = cfg.trial_count - pass_count
    pass_fraction = pass_count / cfg.trial_count
    distributions = _monte_carlo_distributions(tuple(trial_results))
    reasons = [
        SyntheticValidationReason.DETERMINISTIC_SEED_USED,
        SyntheticValidationReason.NO_GLOBAL_RNG_MUTATION,
    ]
    valid = pass_fraction >= cfg.minimum_pass_fraction
    reasons.append(
        SyntheticValidationReason.MONTE_CARLO_PASS_FRACTION_WITHIN_LIMIT
        if valid
        else SyntheticValidationReason.SEED_SENSITIVE_RESULT
    )
    return SyntheticMonteCarloValidation(
        base_scenario_id=base_scenario.scenario_id,
        trial_results=stored_results,
        trial_count=cfg.trial_count,
        seeds=seeds,
        pass_count=pass_count,
        pass_fraction=pass_fraction,
        reservation_count=reservation_count,
        failure_count=failure_count,
        metric_distributions=MappingProxyType(distributions),
        frequency_error_quantiles=_quantiles(distributions.get("frequency_relative_error", ())),
        tau_error_quantiles=_quantiles(distributions.get("tau_relative_error", ())),
        q_error_quantiles=_quantiles(distributions.get("q_relative_error", ())),
        tracking_recovery_quantiles=_quantiles(distributions.get("tracking_recovery", ())),
        candidate_recovery_quantiles=_quantiles(distributions.get("candidate_recovery", ())),
        valid=valid,
        reasons=_ordered_reasons(reasons),
        diagnostics=(
            "monte_carlo_uses_explicit_seed_stride",
            "no_manual_trial_selection",
        ),
    )


def summarize_synthetic_validation(
    result: (
        SyntheticScenarioValidationResult
        | SyntheticValidationCampaignResult
        | SyntheticMonteCarloValidation
    ),
) -> dict[str, object]:
    """Return deterministic compact counters for reports."""

    if isinstance(result, SyntheticScenarioValidationResult):
        return {
            "scenario_id": result.scenario.scenario_id,
            "status": result.status.value,
            "passed_metric_count": result.passed_metric_count,
            "failed_metric_count": result.failed_metric_count,
            "inconclusive_metric_count": result.inconclusive_metric_count,
            "valid": result.valid,
            "diagnostics": result.diagnostics,
        }
    if isinstance(result, SyntheticValidationCampaignResult):
        return {
            "campaign_id": result.campaign_id,
            "scenario_count": result.scenario_count,
            "passed_count": result.passed_count,
            "passed_with_reservations_count": result.passed_with_reservations_count,
            "failed_count": result.failed_count,
            "pass_fraction": result.pass_fraction,
            "valid": result.valid,
            "diagnostics": result.diagnostics,
        }
    if isinstance(result, SyntheticMonteCarloValidation):
        return {
            "base_scenario_id": result.base_scenario_id,
            "trial_count": result.trial_count,
            "pass_count": result.pass_count,
            "pass_fraction": result.pass_fraction,
            "valid": result.valid,
            "reasons": tuple(reason.value for reason in result.reasons),
        }
    raise TypeError("unsupported synthetic validation result.")


def synthetic_validation_settings_fingerprint(
    settings: SyntheticValidationSettings | None = None,
) -> str:
    """Return a deterministic settings fingerprint with no timestamps."""

    cfg = settings or SyntheticValidationSettings()
    payload = json.dumps(_canonicalize(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _built_in_scenario(
    name: str,
    settings: SyntheticValidationSettings,
) -> tuple[tuple[SyntheticDampedComponent, ...], str, dict[str, object], tuple[str, ...]]:
    duration = settings.duration_s
    if name == "single_ideal":
        components = (
            SyntheticDampedComponent("mode_500", 500.0, 1.0, tau_s=2.0),
        )
        return components, "Single isolated damped component without noise or clipping.", _basic_outcomes(1), ()
    if name == "multiple_isolated":
        components = (
            SyntheticDampedComponent("mode_250", 250.0, 0.8, tau_s=3.0),
            SyntheticDampedComponent("mode_500", 500.0, 0.7, tau_s=2.0),
            SyntheticDampedComponent("mode_900", 900.0, 0.6, tau_s=1.2),
        )
        return components, "Three isolated damped components.", _basic_outcomes(3), ()
    if name in {"near_modes", "near_modes_resolved"}:
        components = (
            SyntheticDampedComponent("near_a", 500.0, 0.8, tau_s=2.0),
            SyntheticDampedComponent("near_b", 540.0, 0.7, tau_s=2.1),
        )
        return components, "Nearby modes above nominal resolution.", _basic_outcomes(2), ("neighboring_modes_resolved",)
    if name == "near_modes_marginal":
        components = (
            SyntheticDampedComponent("near_a", 500.0, 0.8, tau_s=2.0),
            SyntheticDampedComponent("near_b", 503.0, 0.7, tau_s=2.1),
        )
        return components, "Nearby modes close to spectral resolution.", _basic_outcomes(2), ("neighboring_mode_interference", "resolution_limited")
    if name == "near_modes_unidentifiable":
        components = (
            SyntheticDampedComponent("near_a", 500.0, 0.8, tau_s=2.0),
            SyntheticDampedComponent("near_b", 500.4, 0.7, tau_s=2.1),
        )
        return components, "Nearby modes below the operational resolution.", _basic_outcomes(2), ("non_identifiable_by_bandwidth", "neighboring_mode_interference")
    if name == "beating":
        components = (
            SyntheticDampedComponent(
                "beat_a",
                500.0,
                0.7,
                tau_s=4.0,
                amplitude_model=SyntheticAmplitudeModel.BEATING_PAIR_MEMBER,
            ),
            SyntheticDampedComponent(
                "beat_b",
                502.0,
                0.7,
                tau_s=4.0,
                phase_rad=0.2,
                amplitude_model=SyntheticAmplitudeModel.BEATING_PAIR_MEMBER,
            ),
        )
        outcomes = _basic_outcomes(2)
        outcomes["known_non_exchange_pairs"] = (("beat_a", "beat_b"),)
        return components, "Two close components with apparent beating context.", outcomes, ("beating_context",)
    if name == "linear_drift":
        components = (
            SyntheticDampedComponent(
                "drift_500",
                500.0,
                0.9,
                tau_s=2.0,
                frequency_model=SyntheticFrequencyModel.LINEAR_DRIFT,
                frequency_drift_hz_per_s=2.0,
            ),
        )
        return components, "One component with linear synthetic drift.", _basic_outcomes(1), ("drift_is_synthetic_not_hardening_or_softening",)
    if name == "frequency_crossing":
        components = (
            SyntheticDampedComponent(
                "cross_a",
                490.0,
                0.7,
                tau_s=3.0,
                frequency_model=SyntheticFrequencyModel.LINEAR_DRIFT,
                frequency_drift_hz_per_s=5.0,
            ),
            SyntheticDampedComponent(
                "cross_b",
                510.0,
                0.7,
                tau_s=3.0,
                frequency_model=SyntheticFrequencyModel.LINEAR_DRIFT,
                frequency_drift_hz_per_s=-5.0,
            ),
        )
        return components, "Two trajectories with apparent frequency crossing.", _basic_outcomes(2), ("frequency_crossing_context", "track_swap_risk")
    if name == "emergence_disappearance":
        components = (
            SyntheticDampedComponent("persistent", 500.0, 0.8, tau_s=2.0),
            SyntheticDampedComponent("emerging", 750.0, 0.7, tau_s=1.5, start_time_s=duration * 0.35),
            SyntheticDampedComponent("disappearing", 300.0, 0.7, tau_s=1.5, end_time_s=duration * 0.55),
        )
        outcomes = _basic_outcomes(3)
        outcomes["emerging_candidates_expected"] = ("emerging",)
        outcomes["disappearing_candidates_expected"] = ("disappearing",)
        return components, "Components emerge and disappear inside the synthetic window.", outcomes, ("emerging_component_context", "disappearing_component_context")
    if name == "apparent_split_merge":
        components = (
            SyntheticDampedComponent("split_parent", 500.0, 0.8, tau_s=2.0, dynamic_label="p"),
            SyntheticDampedComponent("split_child_a", 497.0, 0.5, tau_s=2.0, dynamic_label="mf"),
            SyntheticDampedComponent("split_child_b", 503.0, 0.5, tau_s=2.0, dynamic_label="mf"),
        )
        return components, "Operational apparent split/merge context only.", _basic_outcomes(3), ("apparent_split_context", "apparent_merge_context")
    if name == "energy_exchange":
        components = (
            SyntheticDampedComponent(
                "exchange_a",
                500.0,
                1.0,
                tau_s=4.0,
                amplitude_model=SyntheticAmplitudeModel.CUSTOM_SAMPLES,
                frequency_trajectory=(),
                delayed_growth=(),
                amplitude_recovery=((0.0, 1.0), (duration, 0.25)),
            ),
            SyntheticDampedComponent(
                "exchange_b",
                700.0,
                1.0,
                tau_s=4.0,
                amplitude_model=SyntheticAmplitudeModel.DELAYED_GROWTH,
                delayed_growth=((0.0, 0.15), (duration * 0.35, 0.2), (duration, 0.95)),
            ),
        )
        outcomes = _basic_outcomes(2)
        outcomes["known_energy_exchange_pairs"] = (("exchange_a", "exchange_b"),)
        outcomes["known_energy_exchange_lags_s"] = {("exchange_a", "exchange_b"): duration * 0.20}
        return components, "Imposed apparent redistribution pattern with known lag.", outcomes, ("imposed_operational_energy_pattern",)
    if name == "no_energy_exchange":
        components = (
            SyntheticDampedComponent("decay_a", 500.0, 0.8, tau_s=1.5),
            SyntheticDampedComponent("decay_b", 700.0, 0.7, tau_s=2.0),
        )
        outcomes = _basic_outcomes(2)
        outcomes["known_non_exchange_pairs"] = (("decay_a", "decay_b"),)
        return components, "Two independent synthetic decays.", outcomes, ()
    if name == "noise":
        components = (
            SyntheticDampedComponent("noisy_mode", 500.0, 1.0, tau_s=2.0),
        )
        return components, "Single component with configured additive noise.", _basic_outcomes(1), ("noise_robustness_context",)
    if name == "mains_hum":
        components = (
            SyntheticDampedComponent("mode_500", 500.0, 0.9, tau_s=2.0),
            SyntheticDampedComponent("mains_60", 60.0, 0.1, tau_s=None, amplitude_model=SyntheticAmplitudeModel.CONSTANT_AMPLITUDE, expected_detectable=False, diagnostics=("background_hum_component",)),
        )
        return components, "Modal component plus independent 60 Hz hum.", _basic_outcomes(1), ("background_hum_context",)
    if name == "clipping":
        components = (
            SyntheticDampedComponent("clipped_mode", 500.0, 1.2, tau_s=2.0),
        )
        return components, "Single component under configured clipping.", _basic_outcomes(1), ("clipping_context",)
    if name == "short_duration":
        components = (
            SyntheticDampedComponent("long_tau_short_window", 500.0, 0.9, tau_s=5.0),
        )
        return components, "Duration short relative to tau.", _basic_outcomes(1), ("short_duration", "too_few_decay_cycles")
    if name == "sampling_resolution":
        components = (
            SyntheticDampedComponent("resolution_mode", 500.0, 0.9, tau_s=2.0),
        )
        return components, "Sampling and spectral resolution stress scenario.", _basic_outcomes(1), ("resolution_limited",)
    raise ValueError(f"unknown synthetic validation scenario: {name}")


def _default_campaign_scenarios(
    settings: SyntheticValidationSettings,
) -> tuple[SyntheticValidationScenario, ...]:
    names = (
        "single_ideal",
        "multiple_isolated",
        "near_modes_resolved",
        "near_modes_marginal",
        "near_modes_unidentifiable",
        "beating",
        "linear_drift",
        "frequency_crossing",
        "emergence_disappearance",
        "apparent_split_merge",
        "energy_exchange",
        "no_energy_exchange",
        "noise",
        "mains_hum",
        "clipping",
        "short_duration",
        "sampling_resolution",
    )
    return tuple(generate_synthetic_validation_scenario(name, settings) for name in names)


def _basic_outcomes(count: int) -> dict[str, object]:
    return {
        "expected_candidate_count": count,
        "expected_track_count": count,
        "known_associations": (),
        "known_chains": (),
        "expected_hypothesis_statuses": (),
    }


def _component_waveform(
    component: SyntheticDampedComponent,
    times: np.ndarray,
    settings: SyntheticValidationSettings,
) -> np.ndarray:
    active = times >= component.start_time_s
    if component.end_time_s is not None:
        active &= times < component.end_time_s
    local = np.maximum(0.0, times - component.start_time_s)
    frequencies = _component_frequency_values(component, times, local)
    phase = _component_phase(component, frequencies, active, settings.sample_rate_hz)
    envelope = _component_envelope(component, times, local, settings)
    values = component.amplitude * envelope * np.sin(phase)
    return np.where(active, values, 0.0)


def _component_frequency_values(
    component: SyntheticDampedComponent,
    times: np.ndarray,
    local: np.ndarray,
) -> np.ndarray:
    if component.frequency_model is SyntheticFrequencyModel.CONSTANT:
        return np.full(times.shape, component.initial_frequency_hz, dtype=np.float64)
    if component.frequency_model in {
        SyntheticFrequencyModel.LINEAR_DRIFT,
        SyntheticFrequencyModel.CROSSING_PAIR_MEMBER,
    }:
        return component.initial_frequency_hz + component.frequency_drift_hz_per_s * local
    if component.frequency_model is SyntheticFrequencyModel.PIECEWISE_LINEAR and component.frequency_trajectory:
        points = tuple(sorted(component.frequency_trajectory))
        xp = np.asarray([point[0] for point in points], dtype=np.float64)
        fp = np.asarray([point[1] for point in points], dtype=np.float64)
        return np.interp(times, xp, fp, left=fp[0], right=fp[-1])
    if component.frequency_model is SyntheticFrequencyModel.SINUSOIDAL_MODULATION:
        return component.initial_frequency_hz + component.frequency_drift_hz_per_s * np.sin(2.0 * pi * local)
    if component.frequency_model is SyntheticFrequencyModel.CUSTOM_SAMPLES and component.frequency_trajectory:
        points = tuple(sorted(component.frequency_trajectory))
        xp = np.asarray([point[0] for point in points], dtype=np.float64)
        fp = np.asarray([point[1] for point in points], dtype=np.float64)
        return np.interp(times, xp, fp, left=fp[0], right=fp[-1])
    return np.full(times.shape, component.initial_frequency_hz, dtype=np.float64)


def _component_phase(
    component: SyntheticDampedComponent,
    frequencies: np.ndarray,
    active: np.ndarray,
    sample_rate_hz: int,
) -> np.ndarray:
    increments = 2.0 * pi * frequencies / sample_rate_hz
    phase = np.cumsum(increments)
    if np.any(active):
        start_index = int(np.argmax(active))
        phase = phase - phase[start_index] + component.phase_rad
    return phase


def _component_envelope(
    component: SyntheticDampedComponent,
    times: np.ndarray,
    local: np.ndarray,
    settings: SyntheticValidationSettings,
) -> np.ndarray:
    if component.amplitude_model is SyntheticAmplitudeModel.CONSTANT_AMPLITUDE:
        return np.ones(times.shape, dtype=np.float64)
    if component.amplitude_model in {
        SyntheticAmplitudeModel.PIECEWISE_ENVELOPE,
        SyntheticAmplitudeModel.CUSTOM_SAMPLES,
    } and component.amplitude_recovery:
        points = tuple(sorted(component.amplitude_recovery))
        xp = np.asarray([point[0] for point in points], dtype=np.float64)
        fp = np.asarray([point[1] for point in points], dtype=np.float64)
        return np.maximum(0.0, np.interp(times, xp, fp, left=fp[0], right=fp[-1]))
    if component.amplitude_model is SyntheticAmplitudeModel.DELAYED_ONSET:
        onset = component.start_time_s + max(settings.attack_time_s, 0.0)
        envelope = np.where(times >= onset, 1.0, 0.0)
        if component.tau_s is not None:
            envelope *= np.exp(-np.maximum(0.0, times - onset) / component.tau_s)
        return envelope
    if component.amplitude_model is SyntheticAmplitudeModel.DELAYED_GROWTH:
        points = component.delayed_growth or (
            (component.start_time_s, 0.05),
            (component.start_time_s + 0.5 * settings.duration_s, 0.15),
            (settings.duration_s, 1.0),
        )
        xp = np.asarray([point[0] for point in points], dtype=np.float64)
        fp = np.asarray([point[1] for point in points], dtype=np.float64)
        return np.maximum(0.0, np.interp(times, xp, fp, left=fp[0], right=fp[-1]))
    if component.amplitude_model is SyntheticAmplitudeModel.DECAY_THEN_RECOVERY:
        midpoint = component.start_time_s + 0.45 * settings.duration_s
        envelope = np.exp(-local / (component.tau_s or settings.duration_s))
        recovery = np.clip((times - midpoint) / max(settings.duration_s - midpoint, 1e-12), 0.0, 1.0)
        return np.maximum(envelope, 0.25 + 0.55 * recovery)
    if component.amplitude_model is SyntheticAmplitudeModel.BEATING_PAIR_MEMBER:
        return np.exp(-local / (component.tau_s or settings.duration_s))
    tau = component.tau_s
    if tau is None:
        return np.ones(times.shape, dtype=np.float64)
    return np.exp(-local / tau)


def _noise_series(
    clean: np.ndarray,
    times: np.ndarray,
    settings: SyntheticValidationSettings,
    rng: np.random.Generator,
) -> np.ndarray:
    noise = np.zeros(clean.shape, dtype=np.float64)
    model = settings.noise_model
    if settings.include_colored_noise:
        model = SyntheticNoiseModel.PINK
    if model is SyntheticNoiseModel.WHITE:
        noise = rng.normal(0.0, 1.0, size=clean.size)
    elif model is SyntheticNoiseModel.PINK:
        white = rng.normal(0.0, 1.0, size=clean.size)
        spectrum = np.fft.rfft(white)
        frequencies = np.fft.rfftfreq(clean.size, d=1.0 / settings.sample_rate_hz)
        scale = np.ones_like(frequencies)
        scale[1:] = 1.0 / np.power(frequencies[1:], settings.colored_noise_exponent / 2.0)
        noise = np.fft.irfft(spectrum * scale, n=clean.size)
    if model is not SyntheticNoiseModel.NONE:
        std = settings.noise_standard_deviation
        if std is None and settings.signal_to_noise_ratio_db is not None:
            signal_rms = float(np.sqrt(np.mean(clean * clean))) if clean.size else 0.0
            std = signal_rms / (10.0 ** (settings.signal_to_noise_ratio_db / 20.0))
        std = 0.0 if std is None else std
        current = float(np.std(noise))
        noise = noise * (std / current) if current > 0 else np.zeros_like(noise)
    if settings.include_mains_hum or settings.mains_amplitude > 0:
        noise = noise + settings.mains_amplitude * np.sin(2.0 * pi * settings.mains_frequency_hz * times)
    return noise


def _apply_clipping(
    values: np.ndarray,
    settings: SyntheticValidationSettings,
) -> tuple[np.ndarray, float]:
    threshold = settings.clipping_threshold
    if threshold is None:
        threshold = float(np.quantile(np.abs(values), 1.0 - (settings.clipping_fraction or 0.0)))
        threshold = max(threshold, 1e-12)
    if settings.clipping_mode is SyntheticClippingMode.SOFT or settings.allow_soft_clipping:
        clipped = threshold * np.tanh(values / threshold)
    else:
        clipped = np.clip(values, -threshold, threshold)
    fraction = float(np.mean(np.abs(values) >= threshold))
    return clipped, fraction


def _signal_from_array(values: np.ndarray, sample_rate_hz: int) -> Signal:
    sample_tuple = tuple(float(value) for value in values)
    time = tuple(index / sample_rate_hz for index in range(len(sample_tuple)))
    return Signal(
        samples=(sample_tuple,),
        sample_rate=sample_rate_hz,
        time=time,
        duration=len(sample_tuple) / sample_rate_hz,
        channels=1,
        unit="normalized",
    )


def _representative_frequency(
    component: SyntheticDampedComponent,
    duration_s: float,
) -> float:
    end = component.end_time_s if component.end_time_s is not None else duration_s
    midpoint = component.start_time_s + 0.5 * max(0.0, end - component.start_time_s)
    if component.frequency_model is SyntheticFrequencyModel.CONSTANT:
        return component.initial_frequency_hz
    if component.frequency_model in {
        SyntheticFrequencyModel.LINEAR_DRIFT,
        SyntheticFrequencyModel.CROSSING_PAIR_MEMBER,
    }:
        return component.initial_frequency_hz + component.frequency_drift_hz_per_s * (midpoint - component.start_time_s)
    if component.frequency_trajectory:
        points = tuple(sorted(component.frequency_trajectory))
        xp = np.asarray([point[0] for point in points], dtype=np.float64)
        fp = np.asarray([point[1] for point in points], dtype=np.float64)
        return float(np.interp(midpoint, xp, fp, left=fp[0], right=fp[-1]))
    return component.initial_frequency_hz


def _run_stage(
    stage: str,
    errors: list[tuple[str, str]],
    fn: Any,
) -> Any:
    try:
        return fn()
    except Exception as exc:  # pragma: no cover - validated through output contract
        errors.append((stage, f"{exc.__class__.__name__}: {exc}"))
        return None


def _temporal_analysis_settings(settings: SyntheticValidationSettings) -> Any:
    class _Wrapper:
        temporal = TemporalAnalysisSettings(envelope_method="hilbert")

    return _Wrapper()


def _spectrum_settings(settings: SyntheticValidationSettings) -> SpectrumAnalysisSettings:
    return SpectrumAnalysisSettings(
        window_name=settings.spectrum_window_name,  # type: ignore[arg-type]
        n_fft=settings.spectrum_n_fft,
        scale="linear_amplitude",
    )


def _peak_settings(settings: SyntheticValidationSettings) -> PeakDetectionSettings:
    return PeakDetectionSettings(
        min_prominence=settings.peak_min_prominence,
        min_amplitude=settings.peak_min_amplitude,
        distance_bins=settings.peak_distance_bins,
        max_peaks=settings.peak_max_peaks,
        sort_by="frequency",
    )


def _stft_settings(settings: SyntheticValidationSettings) -> STFTSettings:
    return STFTSettings(
        window_length=settings.stft_window_length,
        hop_length=settings.stft_hop_length,
        n_fft=settings.stft_n_fft,
        scale="linear_amplitude",
        pad_end=False,
    )


def _frame_peak_settings(settings: SyntheticValidationSettings) -> FramePeakDetectionSettings:
    return FramePeakDetectionSettings(
        peak_settings=_peak_settings(settings),
        max_peaks_per_frame=settings.peak_max_peaks,
    )


def _tracking_settings(settings: SyntheticValidationSettings) -> SpectralTrackingSettings:
    return SpectralTrackingSettings(
        frequency_tolerance=settings.tracking_frequency_tolerance,
        frequency_distance_unit=settings.tracking_frequency_distance_unit,  # type: ignore[arg-type]
        max_gap_frames=1,
        min_track_length=2,
        use_refined_frequency=True,
    )


def _component_envelope_sources(
    scenario: SyntheticValidationScenario,
    truth: SyntheticGroundTruth,
) -> tuple[object, ...]:
    times = tuple(value for index, value in enumerate(truth.time_axis_s) if index % max(1, truth.sample_rate_hz // 50) == 0)
    if not times:
        return ()
    sources = []
    for component in scenario.components:
        time_array = np.asarray(times, dtype=np.float64)
        local = np.maximum(0.0, time_array - component.start_time_s)
        envelope = tuple(float(value) for value in component.amplitude * _component_envelope(component, time_array, local, scenario.settings))
        sources.append(
            prepare_modal_envelope_series(
                times_s=times,
                amplitudes=envelope,
                source_id=component.component_id,
                candidate_id=component.component_id,
                dynamic_label=component.dynamic_label,
                diagnostics=component.diagnostics,
            )
        )
    return tuple(sources)


def _estimated_frequency(
    component_id: str,
    true_frequency: float,
    pipeline: SyntheticPipelineOutput,
) -> float | None:
    if pipeline.peak_results is not None and pipeline.peak_results.peaks:
        peak = min(
            pipeline.peak_results.peaks,
            key=lambda item: abs((item.refined_frequency_hz or item.bin_frequency_hz) - true_frequency),
        )
        frequency = peak.refined_frequency_hz or peak.bin_frequency_hz
        if frequency > 0 and isfinite(frequency):
            return frequency
    if pipeline.tracking_results is not None and pipeline.tracking_results.tracks:
        track = min(
            pipeline.tracking_results.tracks,
            key=lambda item: abs(item.median_frequency_hz - true_frequency),
        )
        if track.median_frequency_hz > 0 and isfinite(track.median_frequency_hz):
            return track.median_frequency_hz
    return None


def _estimated_tau(
    component_id: str,
    true_frequency: float,
    tracks: tuple[SpectralTrack, ...],
    characterizations: Mapping[int, object],
) -> tuple[float | None, float | None]:
    if not tracks:
        return None, None
    track = min(tracks, key=lambda item: abs(item.median_frequency_hz - true_frequency))
    characterization = characterizations.get(track.track_id)
    fit = getattr(characterization, "amplitude_fit", None)
    tau = getattr(fit, "tau_s", None)
    quality = getattr(fit, "r_squared", None)
    if tau is not None and isfinite(tau) and tau > 0:
        return float(tau), quality
    return None, quality


def _estimated_track_slope(
    true_frequency: float,
    tracks: tuple[SpectralTrack, ...],
    characterizations: Mapping[int, object],
) -> float | None:
    if not tracks:
        return None
    track = min(tracks, key=lambda item: abs(item.median_frequency_hz - true_frequency))
    characterization = characterizations.get(track.track_id)
    fit = getattr(characterization, "frequency_fit", None)
    slope = getattr(fit, "slope_hz_per_s", None)
    return float(slope) if slope is not None and isfinite(slope) else None


def _match_tracks_to_components(
    components: tuple[SyntheticDampedComponent, ...],
    tracks: tuple[SpectralTrack, ...],
    settings: SyntheticValidationSettings,
) -> tuple[tuple[tuple[str, int], ...], tuple[str, ...], tuple[int, ...]]:
    matches: list[tuple[str, int]] = []
    used: set[int] = set()
    missed: list[str] = []
    for component in components:
        true_frequency = _representative_frequency(component, settings.duration_s)
        best: SpectralTrack | None = None
        best_error: float | None = None
        for track in tracks:
            if track.track_id in used:
                continue
            error = abs(track.median_frequency_hz - true_frequency)
            if best_error is None or error < best_error:
                best_error = error
                best = track
        if best is not None and _frequency_error_within_limits(best_error or 0.0, true_frequency, settings):
            matches.append((component.component_id, best.track_id))
            used.add(best.track_id)
        else:
            missed.append(component.component_id)
    false = tuple(track.track_id for track in tracks if track.track_id not in used)
    return tuple(matches), tuple(missed), false


def _frequency_error_within_limits(
    absolute_error: float,
    true_frequency: float,
    settings: SyntheticValidationSettings,
) -> bool:
    absolute_ok = (
        True
        if settings.maximum_frequency_absolute_error_hz is None
        else absolute_error <= settings.maximum_frequency_absolute_error_hz
    )
    relative_ok = (
        True
        if settings.maximum_frequency_relative_error is None
        else absolute_error / true_frequency <= settings.maximum_frequency_relative_error
    )
    return absolute_ok and relative_ok


def _trajectory_error_metrics(
    true_values: Sequence[float] | None,
    estimated_values: Sequence[float] | None,
    true_slope: float | None,
    estimated_slope: float | None,
) -> dict[str, float | None]:
    rmse = mae = max_error = total_change_error = None
    if true_values is not None and estimated_values is not None:
        true_tuple = tuple(float(value) for value in true_values)
        estimated_tuple = tuple(float(value) for value in estimated_values)
        if true_tuple and len(true_tuple) == len(estimated_tuple):
            errors = tuple(estimated - true for true, estimated in zip(true_tuple, estimated_tuple, strict=True))
            rmse = sqrt(sum(value * value for value in errors) / len(errors))
            mae = sum(abs(value) for value in errors) / len(errors)
            max_error = max(abs(value) for value in errors)
            if len(true_tuple) >= 2:
                total_change_error = abs(
                    (estimated_tuple[-1] - estimated_tuple[0])
                    - (true_tuple[-1] - true_tuple[0])
                )
    slope_error = (
        abs(estimated_slope - true_slope)
        if estimated_slope is not None and true_slope is not None
        else None
    )
    return {
        "rmse": rmse,
        "mae": mae,
        "max_error": max_error,
        "slope_error": slope_error,
        "total_change_error": total_change_error,
    }


def _truth_frequency(truth: SyntheticGroundTruth, component_id: str) -> float:
    for key, value in truth.known_frequencies_hz:
        if key == component_id:
            return value
    raise KeyError(component_id)


def _relative_error(true_value: float, estimated: float | None) -> float | None:
    if estimated is None or not isfinite(estimated) or estimated <= 0:
        return None
    return abs(estimated - true_value) / true_value


def _set_metrics(
    expected_pairs: tuple[tuple[str, str], ...],
    recovered_pairs: tuple[tuple[str, str], ...],
) -> tuple[
    tuple[tuple[str, str], ...],
    tuple[tuple[str, str], ...],
    tuple[tuple[str, str], ...],
    float | None,
    float | None,
    float | None,
]:
    expected_set = set(expected_pairs)
    recovered_set = set(recovered_pairs)
    correct = tuple(sorted(expected_set & recovered_set))
    missing = tuple(sorted(expected_set - recovered_set))
    incorrect = tuple(sorted(recovered_set - expected_set))
    precision = _safe_fraction(len(correct), len(recovered_set))
    recall = _safe_fraction(len(correct), len(expected_set))
    if precision is None and recall is None:
        f1 = None
    elif not precision or not recall:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return correct, missing, incorrect, precision, recall, f1


def _safe_fraction(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _lag_errors_from_energy_result(
    result: ModalEnergyExchangeResult,
    expected: Mapping[tuple[str, str], float],
) -> tuple[tuple[str, str, float], ...]:
    errors: list[tuple[str, str, float]] = []
    canonical_expected = {
        tuple(sorted((left, right))): value
        for (left, right), value in expected.items()
    }
    for evidence in result.pair_evidences:
        key = tuple(sorted((evidence.source_a_id, evidence.source_b_id)))
        if key not in canonical_expected:
            continue
        lag = evidence.correlation_evidence.best_negative_lag_s
        if lag is not None:
            errors.append((key[0], key[1], abs(lag - canonical_expected[key])))
    return tuple(sorted(errors))


def _metric_summaries(
    results: tuple[SyntheticScenarioValidationResult, ...],
) -> dict[str, object]:
    frequency_errors = tuple(
        item.relative_error
        for result in results
        for item in result.frequency_validations
        if item.relative_error is not None
    )
    tau_errors = tuple(
        item.relative_error
        for result in results
        for item in result.decay_validations
        if item.relative_error is not None
    )
    q_errors = tuple(
        item.representative_relative_error
        for result in results
        for item in result.q_validations
        if item.representative_relative_error is not None
    )
    return {
        "frequency_relative_error_mean": _mean(frequency_errors),
        "tau_relative_error_mean": _mean(tau_errors),
        "q_relative_error_mean": _mean(q_errors),
        "scenario_statuses": tuple((item.scenario.scenario_id, item.status.value) for item in results),
    }


def _monte_carlo_distributions(
    results: tuple[SyntheticScenarioValidationResult, ...],
) -> dict[str, tuple[float, ...]]:
    return {
        "frequency_relative_error": tuple(
            value
            for result in results
            for item in result.frequency_validations
            for value in (item.relative_error,)
            if value is not None
        ),
        "tau_relative_error": tuple(
            value
            for result in results
            for item in result.decay_validations
            for value in (item.relative_error,)
            if value is not None
        ),
        "q_relative_error": tuple(
            value
            for result in results
            for item in result.q_validations
            for value in (item.representative_relative_error,)
            if value is not None
        ),
        "tracking_recovery": tuple(
            _safe_fraction(
                len(result.tracking_validation.matched_track_pairs),
                result.tracking_validation.expected_track_count,
            )
            or 0.0
            for result in results
        ),
        "candidate_recovery": tuple(
            _safe_fraction(
                len(result.candidate_validation.matched_candidates),
                result.candidate_validation.expected_candidate_count,
            )
            or 0.0
            for result in results
        ),
    }


def _quantiles(values: Sequence[float]) -> tuple[tuple[float, float], ...]:
    if not values:
        return ()
    arr = np.asarray(values, dtype=np.float64)
    return tuple(
        (probability, float(np.quantile(arr, probability)))
        for probability in (0.05, 0.5, 0.95)
    )


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _all_non_exchange_pairs(
    components: tuple[SyntheticDampedComponent, ...],
    exchange_pairs: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    exchange_set = set(exchange_pairs)
    ids = tuple(sorted(item.component_id for item in components))
    return tuple(
        pair
        for left_index, left in enumerate(ids)
        for pair in ((left, right) for right in ids[left_index + 1:])
        if pair not in exchange_set
    )


def _canonical_pair_tuple(
    pairs: Iterable[tuple[str, str]] | object,
) -> tuple[tuple[str, str], ...]:
    if pairs is None:
        return ()
    result = []
    for left, right in pairs:  # type: ignore[union-attr]
        _text(str(left), "pair left")
        _text(str(right), "pair right")
        if str(left) == str(right):
            continue
        result.append(tuple(sorted((str(left), str(right)))))
    return tuple(sorted(set(result)))


def _dynamic_sort_key(label: str) -> tuple[int, str]:
    order = {"pp": 0, "p": 1, "mf": 2, "f": 3, "ff": 4}
    return order.get(label, 99), label


def _stable_id(prefix: str, *parts: object) -> str:
    payload = json.dumps(_canonicalize(parts), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _canonicalize(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {
            field: _canonicalize(getattr(value, field))
            for field in sorted(value.__dataclass_fields__)
        }
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (tuple, list)):
        return tuple(_canonicalize(item) for item in value)
    if isinstance(value, Signal):
        return {
            "sample_rate": value.sample_rate,
            "duration": value.duration,
            "sample_count": len(value.time),
            "sha1": hashlib.sha1(repr(value.samples).encode("utf-8")).hexdigest(),
        }
    return value


def _coerce_enum(value: object, enum_type: type[Enum]) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)  # type: ignore[call-arg]
    except ValueError as exc:
        raise ValueError(f"value must be a recognized {enum_type.__name__}.") from exc


def _ordered_reasons(
    reasons: Iterable[SyntheticValidationReason],
) -> tuple[SyntheticValidationReason, ...]:
    ordered = []
    seen = set()
    for reason in reasons:
        coerced = _coerce_enum(reason, SyntheticValidationReason)
        if coerced not in seen:
            ordered.append(coerced)
            seen.add(coerced)
    return tuple(sorted(ordered, key=lambda item: item.value))


def _reason_tuple(values: tuple[SyntheticValidationReason, ...], name: str) -> None:
    if not isinstance(values, tuple):
        raise ValueError(f"{name} must be a tuple.")
    for value in values:
        _coerce_enum(value, SyntheticValidationReason)
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates.")


def _strings(values: tuple[str, ...], name: str, *, allow_empty: bool = False) -> None:
    if not isinstance(values, tuple):
        raise ValueError(f"{name} must be a tuple.")
    if not allow_empty and not values:
        raise ValueError(f"{name} must not be empty.")
    if any(not isinstance(item, str) or not item.strip() for item in values):
        raise ValueError(f"{name} must contain nonempty strings.")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates.")


def _pair_tuple(values: tuple[tuple[str, str], ...], name: str) -> None:
    if not isinstance(values, tuple):
        raise ValueError(f"{name} must be a tuple.")
    for left, right in values:
        _text(left, f"{name} left")
        _text(right, f"{name} right")
        if left == right:
            raise ValueError(f"{name} pairs must not contain identical endpoints.")


def _truth_pairs(
    values: tuple[tuple[str, float | None], ...],
    name: str,
    *,
    positive: bool,
    allow_none: bool = False,
) -> None:
    if not isinstance(values, tuple):
        raise ValueError(f"{name} must be a tuple.")
    ids = []
    for key, value in values:
        _text(key, f"{name} key")
        ids.append(key)
        if value is None:
            if not allow_none:
                raise ValueError(f"{name} values must not be None.")
            continue
        if not isfinite(value) or (positive and value <= 0):
            raise ValueError(f"{name} values must be finite and positive.")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name} IDs must be unique.")


def _series_points(
    values: tuple[tuple[float, float], ...],
    name: str,
    *,
    positive_values: bool,
) -> None:
    previous_time = None
    for time, value in values:
        _finite_nonnegative(time, f"{name} time")
        if positive_values:
            _finite_positive(value, f"{name} value")
        else:
            _finite_optional(value, f"{name} value")
        if previous_time is not None and time <= previous_time:
            raise ValueError(f"{name} times must be strictly increasing.")
        previous_time = time


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")


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
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive when provided.")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must not be negative when provided.")


def _finite_positive(value: float, name: str) -> None:
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")


def _finite_nonnegative(value: float, name: str) -> None:
    if not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative.")


def _fraction(value: float | None, name: str) -> None:
    if value is not None and (not isfinite(value) or not 0 <= value <= 1):
        raise ValueError(f"{name} must be finite and in [0, 1] when provided.")


def _fraction_or_nonnegative(value: float | None, name: str) -> None:
    if value is not None and (not isfinite(value) or value < 0):
        raise ValueError(f"{name} must be finite and non-negative when provided.")
