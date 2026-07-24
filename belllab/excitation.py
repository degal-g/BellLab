"""Caracterização instrumental de uma excitação registrada, sem inferência modal."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite, log10, sqrt
from typing import Iterable

import numpy as np

from belllab.recording import Recording
from belllab.types import Signal
from belllab.within_condition import DYNAMIC_LABELS, ExcitationCondition


_DIGITAL_UNITS = frozenset({"normalized", "digital_normalized", "linear_amplitude"})
_DYNAMIC_ORDER = ("pp", "p", "mf", "f", "ff")
_METRICS = frozenset({"rms_amplitude", "signal_energy", "peak_absolute_amplitude"})


def _nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")


def _unique_strings(values: tuple[str, ...], name: str) -> None:
    if not isinstance(values, tuple) or any(
        not isinstance(item, str) or not item.strip() for item in values
    ) or len(values) != len(set(values)):
        raise ValueError(f"{name} must contain unique nonempty strings.")


@dataclass(frozen=True, slots=True)
class ExcitationCharacterizationSettings:
    """Janelas relativas ao impacto e políticas numéricas explícitas."""

    analysis_window_start_s: float = -0.01
    analysis_window_end_s: float = 0.20
    background_window_start_s: float = -1.0
    background_window_end_s: float = -0.1
    clipping_threshold: float = 0.999
    near_clipping_threshold: float = 0.95
    impulse_energy_start_fraction: float = 0.05
    impulse_energy_end_fraction: float = 0.95
    attack_threshold_fraction: float = 0.10
    channel_index: int = 0
    minimum_sample_count: int = 1
    minimum_background_sample_count: int = 1
    remove_dc_for_power: bool = False
    pcm_full_scale: float | None = None
    pcm_zero_point: float | None = None
    normalized_reference_amplitude: float = 1.0
    time_tolerance_s: float = 1e-12

    def __post_init__(self) -> None:
        numeric = (
            self.analysis_window_start_s, self.analysis_window_end_s,
            self.background_window_start_s, self.background_window_end_s,
            self.clipping_threshold, self.near_clipping_threshold,
            self.impulse_energy_start_fraction, self.impulse_energy_end_fraction,
            self.attack_threshold_fraction, self.normalized_reference_amplitude,
            self.time_tolerance_s,
        )
        if any(not isfinite(value) for value in numeric):
            raise ValueError("excitation settings values must be finite.")
        if self.analysis_window_start_s >= self.analysis_window_end_s:
            raise ValueError("analysis window start must precede its end.")
        if self.background_window_start_s >= self.background_window_end_s:
            raise ValueError("background window start must precede its end.")
        if self.background_window_end_s >= self.analysis_window_start_s:
            raise ValueError("background window must end before the analysis window.")
        if not 0 < self.near_clipping_threshold <= self.clipping_threshold:
            raise ValueError("clipping threshold must be at least near-clipping and both positive.")
        if not 0 <= self.impulse_energy_start_fraction < self.impulse_energy_end_fraction <= 1:
            raise ValueError("impulse energy fractions must be ordered in [0, 1].")
        if not 0 < self.attack_threshold_fraction <= 1:
            raise ValueError("attack_threshold_fraction must be in (0, 1].")
        if self.channel_index < 0:
            raise ValueError("channel_index must not be negative.")
        if min(self.minimum_sample_count, self.minimum_background_sample_count) < 0:
            raise ValueError("minimum sample counts must not be negative.")
        if not isinstance(self.remove_dc_for_power, bool):
            raise ValueError("remove_dc_for_power must be a boolean.")
        if self.pcm_full_scale is not None and (
            not isfinite(self.pcm_full_scale) or self.pcm_full_scale <= 0
        ):
            raise ValueError("pcm_full_scale must be finite and positive.")
        if self.pcm_zero_point is not None and not isfinite(self.pcm_zero_point):
            raise ValueError("pcm_zero_point must be finite when provided.")
        if self.normalized_reference_amplitude <= 0:
            raise ValueError("normalized_reference_amplitude must be positive.")
        if self.time_tolerance_s < 0:
            raise ValueError("time_tolerance_s must not be negative.")


@dataclass(frozen=True, slots=True)
class ExcitationCharacterization:
    """Métricas relativas do impacto; não é intensidade acústica calibrada."""

    recording_id: str
    dynamic_label: str
    session_id: str | None
    impact_time_s: float
    analysis_window_start_s: float
    analysis_window_end_s: float
    amplitude_unit: str
    channel_index: int
    power_dc_removed: bool
    sample_count: int
    finite_sample_count: int
    discarded_sample_count: int
    peak_absolute_amplitude: float | None
    peak_signed_amplitude: float | None
    peak_time_s: float | None
    positive_peak: float | None
    negative_peak: float | None
    peak_asymmetry: float | None
    mean_amplitude: float | None
    dc_offset: float | None
    dc_offset_to_peak: float | None
    dc_offset_to_rms: float | None
    rms_amplitude: float | None
    mean_square_amplitude: float | None
    signal_energy: float | None
    equivalent_level_dbfs: float | None
    crest_factor: float | None
    crest_factor_db: float | None
    impulse_start_time_s: float | None
    impulse_end_time_s: float | None
    impulse_duration_s: float | None
    attack_start_time_s: float | None
    attack_duration_s: float | None
    time_to_peak_s: float | None
    clipping_detected: bool
    clipped_sample_count: int
    clipped_sample_fraction: float
    longest_clipped_run: int
    near_clipping_detected: bool
    near_clipping_sample_fraction: float
    background_sample_count: int
    background_finite_sample_count: int
    background_rms: float | None
    background_energy: float | None
    background_peak: float | None
    background_variability: float | None
    background_clipping_detected: bool
    signal_to_background_ratio: float | None
    signal_to_background_db: float | None
    microphone_id: str | None
    interface_id: str | None
    acquisition_gain: float | None
    microphone_distance_m: float | None
    microphone_orientation: str | None
    impact_location: str | None
    exciter_type: str | None
    exciter_mass_kg: float | None
    valid: bool
    failure_reason: str | None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.recording_id, "recording_id")
        if self.dynamic_label not in DYNAMIC_LABELS:
            raise ValueError("dynamic_label is not recognized.")
        if not self.amplitude_unit.strip():
            raise ValueError("amplitude_unit must not be empty.")
        if self.channel_index < 0:
            raise ValueError("channel_index must not be negative.")
        if not (
            isfinite(self.impact_time_s)
            and isfinite(self.analysis_window_start_s)
            and isfinite(self.analysis_window_end_s)
            and self.analysis_window_start_s < self.analysis_window_end_s
        ):
            raise ValueError("characterization time interval is invalid.")
        counts = (
            self.sample_count, self.finite_sample_count, self.discarded_sample_count,
            self.clipped_sample_count, self.longest_clipped_run,
            self.background_sample_count, self.background_finite_sample_count,
        )
        if min(counts) < 0:
            raise ValueError("characterization counts must not be negative.")
        if self.finite_sample_count + self.discarded_sample_count != self.sample_count:
            raise ValueError("excitation sample counts are inconsistent.")
        if self.background_finite_sample_count > self.background_sample_count:
            raise ValueError("background sample counts are inconsistent.")
        optional = (
            self.peak_absolute_amplitude, self.peak_signed_amplitude, self.peak_time_s,
            self.positive_peak, self.negative_peak, self.peak_asymmetry,
            self.mean_amplitude, self.dc_offset, self.dc_offset_to_peak,
            self.dc_offset_to_rms, self.rms_amplitude, self.mean_square_amplitude,
            self.signal_energy, self.equivalent_level_dbfs, self.crest_factor,
            self.crest_factor_db, self.impulse_start_time_s, self.impulse_end_time_s,
            self.impulse_duration_s, self.attack_start_time_s,
            self.attack_duration_s, self.time_to_peak_s, self.background_rms,
            self.background_energy, self.background_peak, self.background_variability,
            self.signal_to_background_ratio, self.signal_to_background_db,
            self.acquisition_gain, self.microphone_distance_m, self.exciter_mass_kg,
        )
        if any(value is not None and not isfinite(value) for value in optional):
            raise ValueError("characterization optional numeric values must be finite.")
        for value, name in (
            (self.peak_absolute_amplitude, "peak"), (self.rms_amplitude, "RMS"),
            (self.mean_square_amplitude, "mean square"), (self.signal_energy, "energy"),
            (self.impulse_duration_s, "impulse duration"),
            (self.attack_duration_s, "attack duration"),
            (self.background_rms, "background RMS"),
            (self.background_energy, "background energy"),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must not be negative.")
        if self.rms_amplitude is not None and not isclose(
            self.mean_square_amplitude, self.rms_amplitude ** 2,
            rel_tol=1e-10, abs_tol=1e-14,
        ):
            raise ValueError("mean_square_amplitude must equal rms_amplitude squared.")
        if self.peak_absolute_amplitude is not None:
            signed = tuple(
                abs(value) for value in (self.positive_peak, self.negative_peak)
                if value is not None
            )
            if signed and self.peak_absolute_amplitude + 1e-14 < max(signed):
                raise ValueError("absolute peak must bound signed peaks.")
        if self.crest_factor is not None and self.crest_factor < 1 - 1e-12:
            raise ValueError("crest_factor must be at least one.")
        for value in (
            self.clipped_sample_fraction, self.near_clipping_sample_fraction
        ):
            if not isfinite(value) or not 0 <= value <= 1:
                raise ValueError("clipping fractions must lie in [0, 1].")
        if self.clipping_detected != (self.clipped_sample_count > 0):
            raise ValueError("clipping flag must agree with clipped sample count.")
        if self.near_clipping_detected != (self.near_clipping_sample_fraction > 0):
            raise ValueError("near-clipping flag must agree with its fraction.")
        if self.longest_clipped_run > self.clipped_sample_count:
            raise ValueError("longest clipped run exceeds clipped sample count.")
        if self.signal_to_background_ratio is not None and self.signal_to_background_ratio <= 0:
            raise ValueError("signal-to-background ratio must be positive.")
        if self.valid == (self.failure_reason is not None):
            raise ValueError("validity and failure_reason are inconsistent.")
        _unique_strings(self.diagnostics, "characterization diagnostics")


@dataclass(frozen=True, slots=True)
class DynamicOrderPairResult:
    """Comparação adjacente auditável na ordem musical esperada."""

    lower_label: str
    upper_label: str
    lower_value: float
    upper_value: float
    tolerance: float
    result: str

    def __post_init__(self) -> None:
        if self.lower_label not in _DYNAMIC_ORDER or self.upper_label not in _DYNAMIC_ORDER:
            raise ValueError("pair labels are not recognized.")
        if _DYNAMIC_ORDER.index(self.lower_label) >= _DYNAMIC_ORDER.index(self.upper_label):
            raise ValueError("pair labels are not in increasing dynamic order.")
        if any(not isfinite(value) for value in (
            self.lower_value, self.upper_value, self.tolerance
        )) or self.tolerance < 0:
            raise ValueError("pair values and tolerance must be finite.")
        if self.result not in {"ordered", "tie", "inversion"}:
            raise ValueError("dynamic pair result is not recognized.")


@dataclass(frozen=True, slots=True)
class DynamicOrderConsistencyResult:
    """Consistência ordinal sem renomear categorias nem comparar modos."""

    session_id: str
    metric_name: str
    ordered_labels: tuple[str, ...]
    observed_values: tuple[float, ...]
    pairwise_results: tuple[DynamicOrderPairResult, ...]
    inversion_count: int
    tie_count: int
    consistent: bool
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.session_id, "session_id")
        if self.metric_name not in _METRICS:
            raise ValueError("metric_name is not supported.")
        if len(self.ordered_labels) != len(self.observed_values):
            raise ValueError("dynamic labels and observed values must align.")
        if len(self.ordered_labels) != len(set(self.ordered_labels)):
            raise ValueError("dynamic labels must not be duplicated.")
        if any(label not in _DYNAMIC_ORDER for label in self.ordered_labels):
            raise ValueError("dynamic ordering excludes unknown labels.")
        if self.ordered_labels != tuple(sorted(
            self.ordered_labels, key=_DYNAMIC_ORDER.index
        )):
            raise ValueError("dynamic labels must use normalized musical order.")
        if any(not isfinite(value) for value in self.observed_values):
            raise ValueError("observed dynamic values must be finite.")
        if min(self.inversion_count, self.tie_count) < 0:
            raise ValueError("dynamic result counts must not be negative.")
        expected_inversions = sum(item.result == "inversion" for item in self.pairwise_results)
        expected_ties = sum(item.result == "tie" for item in self.pairwise_results)
        if (self.inversion_count, self.tie_count) != (expected_inversions, expected_ties):
            raise ValueError("dynamic result counts are inconsistent.")
        expected_consistent = bool(self.pairwise_results) and self.inversion_count == 0
        if self.consistent != expected_consistent:
            raise ValueError("dynamic consistency must agree with inversions.")
        _unique_strings(self.diagnostics, "dynamic consistency diagnostics")


def characterize_excitation_signal(
    signal: Signal,
    recording_id: str,
    condition: ExcitationCondition,
    impact_time_s: float,
    settings: ExcitationCharacterizationSettings | None = None,
) -> ExcitationCharacterization:
    """Caracteriza numericamente um canal já carregado, sem ler arquivos."""
    cfg = settings or ExcitationCharacterizationSettings()
    _nonempty(recording_id, "recording_id")
    if not isfinite(impact_time_s):
        raise ValueError("impact_time_s must be finite.")
    if cfg.channel_index >= signal.channels:
        raise ValueError("channel_index is outside the signal.")
    if condition.channel is not None and condition.channel != cfg.channel_index:
        raise ValueError("condition channel must match the analyzed channel_index.")
    channel_samples = signal.samples[cfg.channel_index]
    raw = np.asarray(channel_samples)
    if raw.dtype.kind in {"b", "O", "S", "U", "c", "V"}:
        raise ValueError("signal sample dtype must be real integer PCM or floating point.")
    integer_pcm = raw.dtype.kind in {"i", "u"} or all(
        isinstance(item, int) and not isinstance(item, bool)
        for item in channel_samples
    )
    if integer_pcm:
        plain_python_integers = bool(channel_samples) and all(
            type(item) is int for item in channel_samples
        )
        if plain_python_integers and cfg.pcm_full_scale is None:
            raise ValueError("pcm_full_scale is required for integer PCM signals.")
        signed = raw.dtype.kind == "i"
        limits = None if plain_python_integers else np.iinfo(raw.dtype)
        inferred_zero = 0.0 if signed else float(
            limits.min + (int(limits.max) - int(limits.min) + 1) // 2
        )
        if signed and cfg.pcm_zero_point not in {None, 0, 0.0}:
            raise ValueError("signed PCM requires pcm_zero_point=0.")
        zero_point = (
            inferred_zero if cfg.pcm_zero_point is None else float(cfg.pcm_zero_point)
        )
        if limits is not None and not limits.min <= zero_point <= limits.max:
            raise ValueError("pcm_zero_point must lie within the integer dtype limits.")
        required_scale = (
            float(cfg.pcm_full_scale)
            if limits is None
            else max(
                abs(float(limits.min) - zero_point),
                abs(float(limits.max) - zero_point),
            )
        )
        scale = required_scale if cfg.pcm_full_scale is None else float(cfg.pcm_full_scale)
        if scale < required_scale:
            raise ValueError(
                "pcm_full_scale must cover both integer dtype extremes for the zero point."
            )
        # Convert before subtraction: unsigned arithmetic would otherwise wrap.
        waveform = (raw.astype(np.float64) - zero_point) / scale
        amplitude_unit = "normalized_from_pcm"
        digital_reference: float | None = cfg.normalized_reference_amplitude
    else:
        if raw.dtype.kind != "f":
            raise ValueError("signal sample dtype must be real integer PCM or floating point.")
        if cfg.pcm_full_scale is not None or cfg.pcm_zero_point is not None:
            raise ValueError("PCM normalization parameters are invalid for floating-point input.")
        waveform = raw.astype(float)
        amplitude_unit = signal.unit
        digital_reference = (
            cfg.normalized_reference_amplitude if signal.unit in _DIGITAL_UNITS else None
        )
    times = np.asarray(signal.time, dtype=float)
    relative = times - impact_time_s
    analysis_mask = (
        (relative >= cfg.analysis_window_start_s - cfg.time_tolerance_s)
        & (relative <= cfg.analysis_window_end_s + cfg.time_tolerance_s)
    )
    background_mask = (
        (relative >= cfg.background_window_start_s - cfg.time_tolerance_s)
        & (relative <= cfg.background_window_end_s + cfg.time_tolerance_s)
    )
    selected_times = times[analysis_mask]
    selected_all = waveform[analysis_mask]
    finite_mask = np.isfinite(selected_all)
    values = selected_all[finite_mask]
    value_times = selected_times[finite_mask]
    background_all = waveform[background_mask]
    background_finite = background_all[np.isfinite(background_all)]
    diagnostics: list[str] = ["relative_amplitude_not_absolute_physical_intensity"]
    if integer_pcm:
        diagnostics.extend((
            (
                "pcm_original_dtype=python_int_unknown_width"
                if plain_python_integers
                else f"pcm_original_dtype={raw.dtype.name}"
            ),
            f"pcm_signed={'true' if signed else 'false'}",
            "pcm_normalization=zero_point_linear_single_scale",
            f"pcm_zero_point={zero_point:g}",
            f"pcm_full_scale={scale:g}",
            (
                "pcm_parameters=configured"
                if cfg.pcm_full_scale is not None or cfg.pcm_zero_point is not None
                else "pcm_parameters=inferred_from_dtype"
            ),
            "pcm_converted_to_float_before_centering",
        ))
    if len(selected_all) - len(values):
        diagnostics.append("nonfinite_excitation_samples_discarded")
    if len(background_all) - len(background_finite):
        diagnostics.append("nonfinite_background_samples_discarded")
    if cfg.remove_dc_for_power:
        diagnostics.append("dc_removed_for_rms_and_energy")
    else:
        diagnostics.append("raw_signal_used_for_rms_and_energy")

    valid = len(values) >= cfg.minimum_sample_count
    failure = None if valid else "insufficient_excitation_samples"
    if not valid:
        diagnostics.append("insufficient_excitation_samples")
    mean_value = float(np.mean(values)) if len(values) else None
    power_values = values - mean_value if len(values) and cfg.remove_dc_for_power else values
    mean_square = float(np.mean(np.square(power_values))) if len(values) else None
    rms = sqrt(mean_square) if mean_square is not None else None
    energy = float(np.sum(np.square(power_values)) / signal.sample_rate) if len(values) else None
    absolute = np.abs(values)
    if len(values):
        peak_index = int(np.argmax(absolute))
        peak = float(absolute[peak_index])
        peak_signed = float(values[peak_index])
        peak_time = float(value_times[peak_index])
        positive = float(max(np.max(values), 0.0))
        negative = float(min(np.min(values), 0.0))
    else:
        peak = peak_signed = peak_time = positive = negative = None
    asymmetry = (
        (abs(positive) - abs(negative)) / peak
        if peak is not None and peak > 0 else None
    )
    crest = peak / rms if peak is not None and rms is not None and rms > 0 else None
    crest_db = 20 * log10(crest) if crest is not None else None
    level_dbfs = (
        20 * log10(rms / digital_reference)
        if rms is not None and rms > 0 and digital_reference is not None else None
    )
    if rms == 0:
        diagnostics.append("silent_window")
    if digital_reference is None:
        diagnostics.append("dbfs_unavailable_without_digital_reference")

    impulse_start = impulse_end = impulse_duration = None
    attack_start = attack_duration = time_to_peak = None
    if len(values) and energy is not None and energy > 0:
        cumulative = np.cumsum(np.square(power_values))
        total = float(cumulative[-1])
        start_index = int(np.searchsorted(
            cumulative, cfg.impulse_energy_start_fraction * total, side="left"
        ))
        end_index = int(np.searchsorted(
            cumulative, cfg.impulse_energy_end_fraction * total, side="left"
        ))
        impulse_start = float(value_times[start_index])
        impulse_end = float(value_times[end_index])
        impulse_duration = impulse_end - impulse_start
        attack_candidates = np.flatnonzero(
            absolute >= cfg.attack_threshold_fraction * peak
        )
        if attack_candidates.size:
            attack_start = float(value_times[int(attack_candidates[0])])
            time_to_peak = peak_time - impact_time_s
            attack_duration = peak_time - attack_start
            if time_to_peak < 0:
                diagnostics.append("peak_precedes_impact_time")
            if attack_duration < 0:
                diagnostics.append("attack_alignment_unavailable")
                attack_duration = None
    else:
        diagnostics.append("impulse_duration_unavailable")

    clipping_mask = (
        absolute >= cfg.clipping_threshold
        if digital_reference is not None else np.zeros(len(values), dtype=bool)
    )
    near_mask = (
        absolute >= cfg.near_clipping_threshold
        if digital_reference is not None else np.zeros(len(values), dtype=bool)
    )
    clipped_count = int(np.sum(clipping_mask))
    near_count = int(np.sum(near_mask))
    longest = _longest_true_run(clipping_mask)
    denominator = len(values)
    clipped_fraction = clipped_count / denominator if denominator else 0.0
    near_fraction = near_count / denominator if denominator else 0.0

    background_sufficient = (
        len(background_finite) >= cfg.minimum_background_sample_count
    )
    background_rms = (
        float(np.sqrt(np.mean(np.square(background_finite))))
        if background_sufficient else None
    )
    background_energy = (
        float(np.sum(np.square(background_finite)) / signal.sample_rate)
        if background_sufficient else None
    )
    background_peak = (
        float(np.max(np.abs(background_finite))) if background_sufficient else None
    )
    background_variability = (
        float(np.std(background_finite)) if background_sufficient else None
    )
    background_clipping = bool(
        background_sufficient and digital_reference is not None
        and np.any(np.abs(background_finite) >= cfg.clipping_threshold)
    )
    ratio = (
        rms / background_rms
        if rms is not None and rms > 0 and background_rms is not None
        and background_rms > 0 else None
    )
    ratio_db = 20 * log10(ratio) if ratio is not None else None
    if not background_sufficient:
        diagnostics.append("insufficient_background_samples")
    elif background_rms == 0:
        diagnostics.append("zero_background_rms")
    comparability_fields = (
        condition.microphone_id, condition.interface_id,
        condition.acquisition_gain, condition.microphone_distance_m,
    )
    if any(value is None for value in comparability_fields):
        diagnostics.append("cross_recording_amplitude_comparability_unverified")
    if clipped_count:
        diagnostics.append("cross_recording_amplitude_comparability_compromised_by_clipping")

    return ExcitationCharacterization(
        recording_id, condition.dynamic_label, condition.session_id,
        float(impact_time_s), float(impact_time_s + cfg.analysis_window_start_s),
        float(impact_time_s + cfg.analysis_window_end_s), amplitude_unit,
        cfg.channel_index, cfg.remove_dc_for_power, len(selected_all), len(values),
        len(selected_all) - len(values), peak, peak_signed, peak_time, positive,
        negative, asymmetry, mean_value, mean_value,
        mean_value / peak if mean_value is not None and peak else None,
        mean_value / rms if mean_value is not None and rms else None,
        rms, mean_square, energy, level_dbfs, crest, crest_db,
        impulse_start, impulse_end, impulse_duration, attack_start,
        attack_duration, time_to_peak, clipped_count > 0, clipped_count,
        clipped_fraction, longest, near_count > 0, near_fraction,
        len(background_all), len(background_finite), background_rms,
        background_energy, background_peak, background_variability,
        background_clipping, ratio, ratio_db, condition.microphone_id,
        condition.interface_id, condition.acquisition_gain,
        condition.microphone_distance_m, condition.microphone_orientation,
        condition.impact_location, condition.exciter_type,
        condition.exciter_mass_kg, valid, failure, tuple(diagnostics),
    )


def characterize_excitation(
    recording: Recording,
    condition: ExcitationCondition,
    impact_time_s: float,
    settings: ExcitationCharacterizationSettings | None = None,
    *,
    recording_id: str | None = None,
) -> ExcitationCharacterization:
    """Adapta uma Recording já carregada sem fazer leitura implícita."""
    identity = recording_id or recording.signal.sha256 or recording.signal.filename
    identity = identity or str(recording.path)
    return characterize_excitation_signal(
        recording.signal, identity, condition, impact_time_s, settings
    )


def evaluate_dynamic_order_consistency(
    characterizations: Iterable[ExcitationCharacterization],
    *,
    metric: str = "rms_amplitude",
    tolerance: float = 0.0,
) -> DynamicOrderConsistencyResult:
    """Avalia a ordem musical por uma métrica sem alterar nenhum rótulo."""
    if metric not in _METRICS:
        raise ValueError("metric is not supported.")
    if not isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative.")
    items = tuple(characterizations)
    if not items:
        raise ValueError("at least one characterization is required.")
    sessions = {item.session_id for item in items}
    if None in sessions or len(sessions) != 1:
        raise ValueError("dynamic consistency requires one explicit session_id.")
    labels = tuple(item.dynamic_label for item in items)
    if any(label == "unspecified" for label in labels):
        raise ValueError("dynamic consistency requires musical dynamic labels.")
    if len(labels) != len(set(labels)):
        raise ValueError("dynamic labels must not be duplicated within a session.")
    measured = []
    for item in items:
        value = getattr(item, metric)
        if value is None or not isfinite(value):
            raise ValueError("dynamic consistency metric values must be finite.")
        measured.append((item.dynamic_label, float(value), item))
    measured.sort(key=lambda item: _DYNAMIC_ORDER.index(item[0]))
    pairs: list[DynamicOrderPairResult] = []
    for lower, upper in zip(measured, measured[1:]):
        difference = upper[1] - lower[1]
        result = (
            "tie" if abs(difference) <= tolerance
            else "ordered" if difference > 0
            else "inversion"
        )
        pairs.append(DynamicOrderPairResult(
            lower[0], upper[0], lower[1], upper[1], tolerance, result
        ))
    diagnostics: list[str] = []
    if len(measured) < len(_DYNAMIC_ORDER):
        diagnostics.append("dynamic_conditions_missing")
    if len(measured) < 2:
        diagnostics.append("insufficient_dynamic_conditions")
    metadata = {
        (
            item.microphone_id, item.interface_id, item.acquisition_gain,
            item.microphone_distance_m, item.channel_index, item.amplitude_unit,
        )
        for _, _, item in measured
    }
    if (
        len(metadata) != 1
        or any(value is None for value in next(iter(metadata))[:4])
        or any(item.clipping_detected for _, _, item in measured)
    ):
        diagnostics.append("cross_recording_amplitude_comparability_unverified")
    inversions = sum(item.result == "inversion" for item in pairs)
    ties = sum(item.result == "tie" for item in pairs)
    return DynamicOrderConsistencyResult(
        next(iter(sessions)), metric,
        tuple(item[0] for item in measured),
        tuple(item[1] for item in measured), tuple(pairs),
        inversions, ties, bool(pairs) and inversions == 0, tuple(diagnostics),
    )


def _longest_true_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest
