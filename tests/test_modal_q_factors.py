"""Operational Q-factor and bandwidth estimates from modal parameters."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace
from math import exp, isfinite, log, pi, sqrt
from random import Random

import pytest

from belllab import (
    AdjacentDynamicConditionPair,
    CandidateReference,
    CrossConditionCandidateAssociationSettings,
    GlobalSpectralPeakMetric,
    ModalBandwidthDefinition,
    ModalBandwidthEstimate,
    ModalBandwidthSource,
    ModalParameterEstimateStatus,
    ModalParameterEstimateReason,
    ModalParameterEstimationSettings,
    ModalQCombinationMethod,
    ModalQConsistencyPolicy,
    ModalQFactorEstimateReason,
    ModalQFactorEstimateStatus,
    ModalQFactorEstimationSettings,
    ModalQUncertaintyMethod,
    ParameterLocationMethod,
    SpectralPeak,
    SpectralResolutionAssessment,
    build_cross_condition_candidate_chains,
    build_cross_condition_candidate_matches,
    estimate_modal_bandwidth,
    estimate_modal_parameters_for_hypothesis,
    estimate_modal_q_factor_for_parameter_estimate,
    estimate_modal_q_factor_provenance,
    estimate_modal_q_factors,
    estimate_q_from_bandwidth,
    estimate_q_from_decay,
    evaluate_modal_peak_isolation,
    compare_modal_q_methods,
    combine_modal_q_estimates,
    modal_q_settings_fingerprint,
    summarize_modal_q_factor_estimates,
)


FULL_SEQUENCE = ("pp", "p", "mf", "f", "ff")
PARAMETER_SETTINGS = ModalParameterEstimationSettings(
    tau_location_method=ParameterLocationMethod.ARITHMETIC_MEAN,
    maximum_frequency_coefficient_of_variation=None,
    maximum_frequency_relative_range=None,
    maximum_log_tau_range=None,
    maximum_log_tau_standard_deviation=None,
)


def _ref(
    label: str,
    name: str,
    frequency: float,
    candidate_id: int,
    *,
    tau: float | None = 2.0,
    amplitude_quality: float | None = 0.95,
    rmse: float | None = 0.1,
    coverage: float | None = 1.0,
    ambiguous_fraction: float | None = 0.0,
    near_fraction: float | None = 0.0,
    margin: float | None = 1.0,
) -> CandidateReference:
    return CandidateReference(
        f"{label}-{name}",
        candidate_id,
        candidate_id,
        frequency,
        label,
        True,
        True,
        "impact_emergent",
        0.01,
        tau,
        amplitude_quality,
        0.0,
        rmse,
        coverage,
        ambiguous_fraction,
        near_fraction,
        margin,
        "impact_emergent",
        None,
        ("modal_q_factor_test_reference",),
    )


def _hypothesis(
    frequency_hz: float = 1000.0,
    tau_s: float | None = 2.0,
    *,
    prefix: str = "Q",
    candidate_start: int = 0,
    count: int = 3,
):
    labels = FULL_SEQUENCE[:count]
    refs = tuple(
        _ref(label, f"{prefix}{index}", frequency_hz, candidate_start + index, tau=tau_s)
        for index, label in enumerate(labels)
    )
    pairs = tuple(
        build_cross_condition_candidate_matches(
            (refs[index],),
            (refs[index + 1],),
            AdjacentDynamicConditionPair(labels[index], labels[index + 1]),
            CrossConditionCandidateAssociationSettings(
                maximum_absolute_frequency_difference_hz=100.0,
                maximum_association_cost=100.0,
            ),
        )
        for index in range(len(refs) - 1)
    )
    chain_result = build_cross_condition_candidate_chains(pairs, labels)
    chain = chain_result.chains[0]
    from belllab import build_modal_hypotheses, ModalHypothesisSettings

    hypothesis = build_modal_hypotheses(
        (chain,),
        ModalHypothesisSettings(
            maximum_step_absolute_frequency_change_hz=100.0,
            maximum_total_absolute_frequency_change_hz=100.0,
            maximum_frequency_trajectory_rmse_hz=100.0,
            maximum_match_cost=100.0,
            maximum_mean_match_cost=100.0,
            maximum_log_tau_range=None,
            maximum_log_tau_standard_deviation=None,
            missing_decay_evidence_policy="allow",
        ),
        labels,
    ).hypotheses[0]
    return hypothesis


def _parameter_estimate(
    frequency_hz: float = 1000.0,
    tau_s: float | None = 2.0,
    *,
    prefix: str = "P",
    candidate_start: int = 0,
):
    return estimate_modal_parameters_for_hypothesis(
        _hypothesis(frequency_hz, tau_s, prefix=prefix, candidate_start=candidate_start),
        PARAMETER_SETTINGS,
    )


def _corrupt_dataclass(instance, **changes):
    corrupted = object.__new__(type(instance))
    for field in fields(instance):
        object.__setattr__(
            corrupted,
            field.name,
            changes.get(field.name, getattr(instance, field.name)),
        )
    return corrupted


def _corrupt_frequency(parameter_estimate, value: float):
    frequency = _corrupt_dataclass(
        parameter_estimate.frequency_estimate,
        representative_frequency_hz=value,
        values_hz=(value,),
        valid=False,
        reasons=(parameter_estimate.frequency_estimate.reasons),
    )
    return replace(
        parameter_estimate,
        status=ModalParameterEstimateStatus.INVALID_INPUT,
        frequency_estimate=frequency,
        invalid_reasons=(ModalParameterEstimateReason.INVALID_FREQUENCY_VALUE,),
        valid=False,
        requires_review=True,
    )


def _bandwidth_source(
    *,
    center: float = 1000.0,
    width: float = 10.0,
    resolution: float = 1.0,
    peaks: tuple[float, ...] = (1000.0,),
    spectrum_id: str = "synthetic-spectrum",
    uncertainty: float | None = None,
) -> ModalBandwidthSource:
    cutoff = 10.0 ** (-3.0 / 20.0)
    return ModalBandwidthSource(
        spectrum_id=spectrum_id,
        center_frequency_hz=center,
        frequency_axis_hz=(
            center - width,
            center - (width / 2.0),
            center,
            center + (width / 2.0),
            center + width,
        ),
        magnitude_values=(0.2, cutoff, 1.0, cutoff, 0.2),
        peak_frequencies_hz=peaks,
        frequency_resolution_hz=resolution,
        bandwidth_uncertainty_hz=uncertainty,
    )


def test_decay_q_basic_formula_uncertainty_and_convention_are_explicit() -> None:
    estimate = estimate_q_from_decay(
        representative_frequency_hz=1000.0,
        representative_tau_s=2.0,
        frequency_uncertainty_hz=1.0,
        tau_uncertainty_s=0.1,
    )
    expected_q = pi * 1000.0 * 2.0
    expected_relative = sqrt((1.0 / 1000.0) ** 2 + (0.1 / 2.0) ** 2)

    assert estimate.valid is True
    assert estimate.q_decay == pytest.approx(expected_q)
    assert estimate.standard_uncertainty_q == pytest.approx(expected_q * expected_relative)
    assert estimate.lower_bound_q == pytest.approx(expected_q - estimate.standard_uncertainty_q)
    assert estimate.upper_bound_q == pytest.approx(expected_q + estimate.standard_uncertainty_q)
    assert "A(t)=A0 exp(-t/tau)" in estimate.decay_convention
    assert "tau_is_amplitude_decay_time_not_energy_decay_time" in estimate.diagnostics
    assert "no_physical_oscillator_fit_performed" in estimate.diagnostics


@pytest.mark.parametrize(
    ("frequency", "tau", "reason"),
    [
        (0.0, 2.0, ModalQFactorEstimateReason.INVALID_FREQUENCY),
        (-1.0, 2.0, ModalQFactorEstimateReason.INVALID_FREQUENCY),
        (float("nan"), 2.0, ModalQFactorEstimateReason.INVALID_FREQUENCY),
        (float("inf"), 2.0, ModalQFactorEstimateReason.INVALID_FREQUENCY),
        (1000.0, 0.0, ModalQFactorEstimateReason.INVALID_TAU),
        (1000.0, -1.0, ModalQFactorEstimateReason.INVALID_TAU),
        (1000.0, float("nan"), ModalQFactorEstimateReason.INVALID_TAU),
        (1000.0, float("inf"), ModalQFactorEstimateReason.INVALID_TAU),
    ],
    ids=["zero_f", "negative_f", "nan_f", "inf_f", "zero_tau", "negative_tau", "nan_tau", "inf_tau"],
)
def test_decay_q_invalid_values_are_explicit(frequency: float, tau: float, reason: ModalQFactorEstimateReason) -> None:
    estimate = estimate_q_from_decay(
        representative_frequency_hz=frequency,
        representative_tau_s=tau,
    )

    assert estimate.valid is False
    assert estimate.q_decay is None
    assert reason in estimate.reasons


def test_decay_q_bootstrap_is_seeded_and_operational() -> None:
    settings = ModalQFactorEstimationSettings(
        uncertainty_method=ModalQUncertaintyMethod.PARAMETRIC_BOOTSTRAP,
        bootstrap_sample_count=200,
        bootstrap_random_seed=9,
    )

    first = estimate_q_from_decay(
        representative_frequency_hz=1000.0,
        representative_tau_s=2.0,
        frequency_uncertainty_hz=2.0,
        tau_uncertainty_s=0.2,
        settings=settings,
    )
    second = estimate_q_from_decay(
        representative_frequency_hz=1000.0,
        representative_tau_s=2.0,
        frequency_uncertainty_hz=2.0,
        tau_uncertainty_s=0.2,
        settings=settings,
    )
    other = estimate_q_from_decay(
        representative_frequency_hz=1000.0,
        representative_tau_s=2.0,
        frequency_uncertainty_hz=2.0,
        tau_uncertainty_s=0.2,
        settings=replace(settings, bootstrap_random_seed=10),
    )

    assert first == second
    assert first.valid is True
    assert first.lower_bound_q > 0.0
    assert first.upper_bound_q > first.lower_bound_q
    assert (
        first.standard_uncertainty_q,
        first.lower_bound_q,
        first.upper_bound_q,
    ) != (
        other.standard_uncertainty_q,
        other.lower_bound_q,
        other.upper_bound_q,
    )


def test_synthetic_bandwidth_and_q_are_extracted_without_recomputing_spectrum() -> None:
    source = _bandwidth_source(width=10.0, resolution=1.0)

    bandwidth = estimate_modal_bandwidth(
        source.center_frequency_hz,
        source.frequency_axis_hz,
        source.magnitude_values,
        peak_frequencies_hz=source.peak_frequencies_hz,
        frequency_resolution_hz=source.frequency_resolution_hz,
    )
    q_estimate = estimate_q_from_bandwidth(
        bandwidth,
        frequency_uncertainty_hz=1.0,
        bandwidth_uncertainty_hz=0.5,
    )

    assert bandwidth.valid is True
    assert bandwidth.lower_frequency_hz == pytest.approx(995.0)
    assert bandwidth.upper_frequency_hz == pytest.approx(1005.0)
    assert bandwidth.bandwidth_hz == pytest.approx(10.0)
    assert bandwidth.resolution_ratio == pytest.approx(10.0)
    assert bandwidth.resolution_assessment is SpectralResolutionAssessment.WELL_RESOLVED
    assert bandwidth.isolated_peak is True
    assert q_estimate.valid is True
    assert q_estimate.q_bandwidth == pytest.approx(100.0)
    assert "formula:Q_bandwidth=f_center/bandwidth" in q_estimate.diagnostics


def test_bandwidth_interpolation_is_not_rounded_to_bins() -> None:
    cutoff = 10.0 ** (-3.0 / 20.0)
    frequencies = (990.0, 994.0, 996.0, 1000.0, 1004.0, 1006.0, 1010.0)
    magnitudes = (0.1, 0.5, 0.9, 1.0, 0.9, 0.5, 0.1)
    expected_left = 994.0 + ((cutoff - 0.5) / (0.9 - 0.5)) * 2.0
    expected_right = 1004.0 + ((cutoff - 0.9) / (0.5 - 0.9)) * 2.0

    bandwidth = estimate_modal_bandwidth(
        1000.0,
        frequencies,
        magnitudes,
        peak_frequencies_hz=(1000.0,),
        frequency_resolution_hz=1.0,
    )

    assert bandwidth.valid is True
    assert bandwidth.lower_frequency_hz == pytest.approx(expected_left)
    assert bandwidth.upper_frequency_hz == pytest.approx(expected_right)
    assert bandwidth.bandwidth_hz != pytest.approx(10.0)
    assert bandwidth.interpolation_method == "linear"


@pytest.mark.parametrize(
    ("frequencies", "magnitudes", "left", "right", "valid"),
    [
        ((995.0, 1000.0, 1005.0), (0.707945784, 1.0, 0.707945784), True, True, True),
        ((990.0, 995.0, 1000.0, 1005.0, 1010.0), (0.8, 0.9, 1.0, 0.5, 0.1), False, True, False),
        ((990.0, 995.0, 1000.0, 1005.0, 1010.0), (0.1, 0.5, 1.0, 0.9, 0.8), True, False, False),
        ((1000.0, 1005.0, 1010.0), (1.0, 0.5, 0.1), False, False, False),
    ],
    ids=["exact_crossings", "missing_left", "missing_right", "peak_at_edge"],
)
def test_bandwidth_crossing_edge_cases_are_explicit(frequencies, magnitudes, left, right, valid) -> None:
    bandwidth = estimate_modal_bandwidth(
        1000.0,
        frequencies,
        magnitudes,
        peak_frequencies_hz=(1000.0,),
        frequency_resolution_hz=1.0,
    )

    assert bandwidth.left_crossing_found is left
    assert bandwidth.right_crossing_found is right
    assert bandwidth.valid is valid
    if not valid:
        assert ModalQFactorEstimateReason.MISSING_BANDWIDTH in bandwidth.reasons


@pytest.mark.parametrize(
    ("ratio", "assessment", "valid"),
    [
        (0.5, SpectralResolutionAssessment.UNRESOLVED, False),
        (1.0, SpectralResolutionAssessment.RESOLUTION_LIMITED, True),
        (1.5, SpectralResolutionAssessment.MARGINALLY_RESOLVED, True),
        (2.0, SpectralResolutionAssessment.WELL_RESOLVED, True),
        (5.0, SpectralResolutionAssessment.WELL_RESOLVED, True),
    ],
    ids=["half_bin", "one_bin", "one_half_bins", "two_bins", "five_bins"],
)
def test_resolution_assessment_is_configurable_and_inclusive(ratio, assessment, valid) -> None:
    settings = ModalQFactorEstimationSettings(allow_resolution_limited_bandwidth=True)
    bandwidth = estimate_modal_bandwidth(
        1000.0,
        _bandwidth_source(width=10.0, resolution=10.0 / ratio).frequency_axis_hz,
        _bandwidth_source(width=10.0, resolution=10.0 / ratio).magnitude_values,
        peak_frequencies_hz=(1000.0,),
        frequency_resolution_hz=10.0 / ratio,
        settings=settings,
    )

    assert bandwidth.resolution_assessment is assessment
    assert bandwidth.valid is valid
    assert bandwidth.resolution_ratio == pytest.approx(ratio)


@pytest.mark.parametrize(
    ("neighbors", "isolated", "distance", "overlap"),
    [
        ((), True, None, 0.0),
        ((1030.0,), True, 30.0, 0.0),
        ((1005.0,), False, 5.0, 0.5),
        ((992.0, 1007.0), False, 7.0, 0.3),
        ((990.0, 1010.0), True, 10.0, 0.0),
    ],
    ids=["none", "distant", "inside_width", "partially_overlapped", "touching"],
)
def test_peak_isolation_evidence_diagnoses_neighbors_without_separation(neighbors, isolated, distance, overlap) -> None:
    evidence = evaluate_modal_peak_isolation(
        1000.0,
        (1000.0,) + neighbors,
        10.0,
        ModalQFactorEstimationSettings(maximum_neighboring_peak_overlap_fraction=0.25),
    )

    assert evidence.isolated is isolated
    assert evidence.nearest_peak_distance_hz == pytest.approx(distance) if distance is not None else evidence.nearest_peak_distance_hz is None
    assert evidence.overlap_fraction == pytest.approx(overlap)
    assert "no_overlapped_peak_separation" in evidence.diagnostics


@pytest.mark.parametrize(
    ("q_decay", "q_bandwidth", "consistent", "partial", "inconsistent"),
    [
        (100.0, 105.0, True, False, False),
        (100.0, 120.0, False, True, False),
        (100.0, 250.0, False, False, True),
    ],
    ids=["consistent", "partially_consistent", "inconsistent"],
)
def test_method_comparison_uses_symmetric_difference_and_log_ratio(q_decay, q_bandwidth, consistent, partial, inconsistent) -> None:
    settings = ModalQFactorEstimationSettings(
        maximum_relative_q_disagreement=0.10,
        maximum_log_q_difference=None,
        combine_consistent_methods=ModalQCombinationMethod.GEOMETRIC_MEAN,
    )

    comparison = compare_modal_q_methods(q_decay, q_bandwidth, settings)

    assert comparison.absolute_difference == pytest.approx(abs(q_decay - q_bandwidth))
    assert comparison.relative_symmetric_difference == pytest.approx(abs(q_decay - q_bandwidth) / ((q_decay + q_bandwidth) / 2.0))
    assert comparison.log_q_difference == pytest.approx(abs(log(q_decay / q_bandwidth)))
    assert comparison.ratio_decay_to_bandwidth == pytest.approx(q_decay / q_bandwidth)
    assert comparison.consistent is consistent
    assert comparison.partially_consistent is partial
    assert comparison.inconsistent is inconsistent


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        (ModalQCombinationMethod.ARITHMETIC_MEAN, 102.0),
        (ModalQCombinationMethod.GEOMETRIC_MEAN, sqrt(100.0 * 104.0)),
        (ModalQCombinationMethod.PREFER_DECAY, 100.0),
        (ModalQCombinationMethod.PREFER_BANDWIDTH, 104.0),
        (ModalQCombinationMethod.NONE, None),
    ],
    ids=["mean", "geometric", "prefer_decay", "prefer_bandwidth", "none"],
)
def test_combination_methods_are_explicit(method, expected) -> None:
    combination = combine_modal_q_estimates(
        100.0,
        104.0,
        ModalQFactorEstimationSettings(combine_consistent_methods=method),
        decay_uncertainty_q=2.0,
        bandwidth_uncertainty_q=3.0,
    )

    assert combination.combined_q == pytest.approx(expected) if expected is not None else combination.combined_q is None
    if expected is None:
        assert combination.valid is False


def test_uncertainty_weighted_combination_records_finite_weights_and_zero_floor() -> None:
    combination = combine_modal_q_estimates(
        100.0,
        104.0,
        ModalQFactorEstimationSettings(
            combine_consistent_methods=ModalQCombinationMethod.INVERSE_UNCERTAINTY_WEIGHTED,
        ),
        decay_uncertainty_q=0.0,
        bandwidth_uncertainty_q=2.0,
    )

    assert combination.valid is True
    assert all(isfinite(value) and value >= 0.0 for value in combination.weights)
    assert sum(combination.normalized_weights) == pytest.approx(1.0)
    assert "zero_uncertainty_weight_floor_applied" in combination.diagnostics


def test_inconsistent_methods_are_not_combined_by_default() -> None:
    comparison = compare_modal_q_methods(
        100.0,
        250.0,
        ModalQFactorEstimationSettings(
            maximum_relative_q_disagreement=0.10,
            maximum_log_q_difference=None,
            combine_consistent_methods=ModalQCombinationMethod.ARITHMETIC_MEAN,
        ),
    )

    assert comparison.inconsistent is True
    assert comparison.combined_q is None
    assert ModalQFactorEstimateReason.EXCESSIVE_METHOD_DISAGREEMENT in comparison.reasons


def test_q_factor_disagreement_is_inconclusive_and_preserves_provenance() -> None:
    parameter = _parameter_estimate(prefix="VA")
    source = _bandwidth_source(spectrum_id="spectrum-valid")

    estimate = estimate_modal_q_factor_for_parameter_estimate(parameter, bandwidth_source=source)

    assert estimate.decay_q_estimate.q_decay == pytest.approx(pi * 1000.0 * 2.0)
    assert estimate.bandwidth_q_estimate.q_bandwidth == pytest.approx(100.0)
    assert estimate.method_comparison.consistent is False
    assert estimate.status is ModalQFactorEstimateStatus.INCONCLUSIVE
    assert estimate.provenance.modal_parameter_estimate_id == parameter.estimate_id
    assert estimate.provenance.spectrum_ids == ("spectrum-valid",)


def test_q_factor_valid_when_methods_consistent_by_constructed_tau() -> None:
    tau_for_q100 = 100.0 / (pi * 1000.0)
    parameter = _parameter_estimate(1000.0, tau_for_q100, prefix="VC")

    estimate = estimate_modal_q_factor_for_parameter_estimate(
        parameter,
        bandwidth_source=_bandwidth_source(width=10.0, spectrum_id="spectrum-consistent"),
    )

    assert estimate.status is ModalQFactorEstimateStatus.VALID
    assert estimate.representative_q == pytest.approx(100.0)
    assert estimate.representative_q_method == "combined"
    assert estimate.method_comparison.consistent is True


def test_q_factor_partial_when_only_decay_method_is_available() -> None:
    parameter = _parameter_estimate(prefix="OD")

    estimate = estimate_modal_q_factor_for_parameter_estimate(parameter)

    assert estimate.status is ModalQFactorEstimateStatus.PARTIAL
    assert estimate.decay_q_estimate.valid is True
    assert estimate.bandwidth_q_estimate is None
    assert ModalQFactorEstimateReason.MISSING_BANDWIDTH in estimate.insufficient_evidence_reasons


def test_q_factor_partial_when_only_bandwidth_method_is_available() -> None:
    parameter = _parameter_estimate(tau_s=None, prefix="OB")

    estimate = estimate_modal_q_factor_for_parameter_estimate(
        parameter,
        bandwidth_source=_bandwidth_source(width=10.0),
    )

    assert parameter.status is ModalParameterEstimateStatus.PARTIAL
    assert estimate.status is ModalQFactorEstimateStatus.PARTIAL
    assert estimate.decay_q_estimate.valid is False
    assert estimate.bandwidth_q_estimate.valid is True


def test_q_factor_inconclusive_for_strong_method_disagreement() -> None:
    parameter = _parameter_estimate(1000.0, 100.0 / (pi * 1000.0), prefix="ID")

    estimate = estimate_modal_q_factor_for_parameter_estimate(
        parameter,
        bandwidth_source=_bandwidth_source(width=4.0),
        settings=ModalQFactorEstimationSettings(maximum_relative_q_disagreement=0.10, maximum_log_q_difference=None),
    )

    assert estimate.status is ModalQFactorEstimateStatus.INCONCLUSIVE
    assert ModalQFactorEstimateReason.EXCESSIVE_METHOD_DISAGREEMENT in estimate.inconclusive_reasons
    assert estimate.representative_q is None


def test_resolution_limited_and_nonisolated_bandwidths_create_reservations_when_allowed() -> None:
    tau_for_q100 = 100.0 / (pi * 1000.0)
    parameter = _parameter_estimate(1000.0, tau_for_q100, prefix="RV")
    settings = ModalQFactorEstimationSettings(
        allow_resolution_limited_bandwidth=True,
        require_isolated_peak=False,
    )

    limited = estimate_modal_q_factor_for_parameter_estimate(
        parameter,
        settings,
        bandwidth_source=_bandwidth_source(width=10.0, resolution=8.0, spectrum_id="limited"),
    )
    nonisolated = estimate_modal_q_factor_for_parameter_estimate(
        parameter,
        settings,
        bandwidth_source=_bandwidth_source(width=10.0, peaks=(1000.0, 1005.0), spectrum_id="nonisolated"),
    )

    assert limited.status is ModalQFactorEstimateStatus.VALID_WITH_RESERVATIONS
    assert ModalQFactorEstimateReason.MARGINALLY_RESOLVED in limited.reservation_reasons
    assert nonisolated.status is ModalQFactorEstimateStatus.VALID_WITH_RESERVATIONS
    assert ModalQFactorEstimateReason.PEAK_NOT_ISOLATED in nonisolated.reservation_reasons


def test_no_method_available_is_insufficient_and_invalid_origin_does_not_produce_valid_q() -> None:
    missing_tau = _parameter_estimate(tau_s=None, prefix="MT")
    invalid_frequency = _corrupt_frequency(_parameter_estimate(prefix="IF"), float("nan"))

    no_method = estimate_modal_q_factor_for_parameter_estimate(missing_tau)
    invalid = estimate_modal_q_factor_for_parameter_estimate(invalid_frequency)

    assert no_method.status is ModalQFactorEstimateStatus.INSUFFICIENT_EVIDENCE
    assert invalid.status is ModalQFactorEstimateStatus.INVALID_INPUT
    assert invalid.valid is False


def test_source_parameter_status_policy_and_audit_flags() -> None:
    insufficient = replace(
        _parameter_estimate(prefix="PS"),
        status=ModalParameterEstimateStatus.INSUFFICIENT_EVIDENCE,
        valid=False,
        requires_review=True,
    )
    default = estimate_modal_q_factor_for_parameter_estimate(insufficient)
    audit = estimate_modal_q_factor_for_parameter_estimate(
        insufficient,
        ModalQFactorEstimationSettings(allow_insufficient_evidence_for_audit=True),
    )

    assert default.status is ModalQFactorEstimateStatus.INSUFFICIENT_EVIDENCE
    assert ModalQFactorEstimateReason.UNSUPPORTED_PARAMETER_STATUS in default.insufficient_evidence_reasons
    assert audit.status is ModalQFactorEstimateStatus.PARTIAL


def test_global_result_partitions_all_parameter_estimates_and_keeps_unique_ids() -> None:
    tau_for_q100 = 100.0 / (pi * 1000.0)
    valid = _parameter_estimate(1000.0, tau_for_q100, prefix="GA", candidate_start=0)
    decay_only = _parameter_estimate(1000.0, 2.0, prefix="GB", candidate_start=10)
    bandwidth_only = _parameter_estimate(1000.0, None, prefix="GC", candidate_start=20)
    discordant = _parameter_estimate(1000.0, tau_for_q100, prefix="GD", candidate_start=30)
    limited = _parameter_estimate(1000.0, tau_for_q100, prefix="GE", candidate_start=40)
    nonisolated = _parameter_estimate(1000.0, tau_for_q100, prefix="GF", candidate_start=50)
    no_tau = _parameter_estimate(1000.0, None, prefix="GG", candidate_start=60)
    invalid = _corrupt_frequency(_parameter_estimate(1000.0, 2.0, prefix="GH", candidate_start=70), float("nan"))
    settings = ModalQFactorEstimationSettings(
        allow_resolution_limited_bandwidth=True,
        require_isolated_peak=False,
        maximum_relative_q_disagreement=0.10,
        maximum_log_q_difference=None,
    )
    sources = {
        valid.estimate_id: _bandwidth_source(center=1000.0, width=10.0, spectrum_id="valid"),
        bandwidth_only.estimate_id: _bandwidth_source(center=1000.0, width=10.0, spectrum_id="bandwidth-only"),
        discordant.estimate_id: _bandwidth_source(center=1000.0, width=4.0, spectrum_id="discordant"),
        limited.estimate_id: _bandwidth_source(center=1000.0, width=10.0, resolution=8.0, spectrum_id="limited"),
        nonisolated.estimate_id: _bandwidth_source(center=1000.0, width=10.0, peaks=(1000.0, 1005.0), spectrum_id="nonisolated"),
    }

    result = estimate_modal_q_factors(
        (valid, decay_only, bandwidth_only, discordant, limited, nonisolated, no_tau, invalid),
        settings,
        bandwidth_sources=sources,
    )

    assert result.estimate_count == 8
    assert result.source_parameter_estimate_count == 8
    assert len({item.estimate_id for item in result.estimates}) == 8
    assert result.valid_count == 1
    assert result.partial_count >= 2
    assert result.inconclusive_count == 1
    assert result.valid_with_reservations_count >= 2
    assert result.invalid_count == 1
    assert result.valid_estimates == tuple(item for item in result.estimates if item.status is ModalQFactorEstimateStatus.VALID)


def test_estimation_is_deterministic_across_input_orders_and_diagnostics_order() -> None:
    tau_for_q100 = 100.0 / (pi * 1000.0)
    estimates = (
        _parameter_estimate(1000.0, tau_for_q100, prefix="DA", candidate_start=0),
        _parameter_estimate(1100.0, 100.0 / (pi * 1100.0), prefix="DB", candidate_start=10),
        _parameter_estimate(1200.0, None, prefix="DC", candidate_start=20),
    )
    changed = replace(estimates[0], diagnostics=tuple(reversed(estimates[0].diagnostics)))
    sources = {
        estimates[0].estimate_id: _bandwidth_source(center=1000.0, width=10.0, spectrum_id="da"),
        estimates[1].estimate_id: _bandwidth_source(center=1100.0, width=11.0, spectrum_id="db"),
        estimates[2].estimate_id: _bandwidth_source(center=1200.0, width=12.0, spectrum_id="dc"),
    }
    settings = ModalQFactorEstimationSettings(
        uncertainty_method=ModalQUncertaintyMethod.PARAMETRIC_BOOTSTRAP,
        bootstrap_sample_count=200,
        bootstrap_random_seed=17,
    )

    ordered = estimate_modal_q_factors(estimates, settings, bandwidth_sources=sources)
    reversed_result = estimate_modal_q_factors(tuple(reversed(estimates)), settings, bandwidth_sources=sources)
    shuffled_items = list(estimates)
    Random(4).shuffle(shuffled_items)
    shuffled = estimate_modal_q_factors(tuple(shuffled_items), settings, bandwidth_sources=sources)
    changed_result = estimate_modal_q_factors((changed,) + estimates[1:], settings, bandwidth_sources=sources)

    def normalized(result):
        return tuple(
            (
                item.estimate_id,
                item.modal_parameter_estimate_id,
                item.status,
                item.decay_q_estimate.q_decay if item.decay_q_estimate else None,
                item.bandwidth_estimate.bandwidth_hz if item.bandwidth_estimate else None,
                item.bandwidth_q_estimate.q_bandwidth if item.bandwidth_q_estimate else None,
                item.method_comparison.combined_q if item.method_comparison else None,
                item.representative_q,
                item.provenance,
                item.supporting_reasons,
                item.reservation_reasons,
                item.insufficient_evidence_reasons,
                item.invalid_reasons,
            )
            for item in result.estimates
        )

    assert normalized(ordered) == normalized(reversed_result) == normalized(shuffled) == normalized(changed_result)
    assert summarize_modal_q_factor_estimates(ordered) == summarize_modal_q_factor_estimates(reversed_result)


def test_local_perturbation_changes_only_corresponding_q_estimate() -> None:
    tau_for_q100 = 100.0 / (pi * 1000.0)
    target = _parameter_estimate(1000.0, tau_for_q100, prefix="PA", candidate_start=0)
    other = _parameter_estimate(1100.0, 100.0 / (pi * 1100.0), prefix="PB", candidate_start=10)
    settings = ModalQFactorEstimationSettings()
    base_sources = {
        target.estimate_id: _bandwidth_source(center=1000.0, width=10.0, spectrum_id="target"),
        other.estimate_id: _bandwidth_source(center=1100.0, width=11.0, spectrum_id="other"),
    }
    perturbed_sources = {
        target.estimate_id: _bandwidth_source(center=1000.0, width=12.0, spectrum_id="target"),
        other.estimate_id: _bandwidth_source(center=1100.0, width=11.0, spectrum_id="other"),
    }

    base = estimate_modal_q_factors((target, other), settings, bandwidth_sources=base_sources)
    perturbed = estimate_modal_q_factors((target, other), settings, bandwidth_sources=perturbed_sources)
    base_by_source = {item.modal_parameter_estimate_id: item for item in base.estimates}
    perturbed_by_source = {item.modal_parameter_estimate_id: item for item in perturbed.estimates}

    assert perturbed_by_source[target.estimate_id] != base_by_source[target.estimate_id]
    assert perturbed_by_source[other.estimate_id] == base_by_source[other.estimate_id]
    assert perturbed_by_source[other.estimate_id].estimate_id == base_by_source[other.estimate_id].estimate_id


def test_inputs_are_immutable_and_repeated_builds_are_identical() -> None:
    tau_for_q100 = 100.0 / (pi * 1000.0)
    estimates = (
        _parameter_estimate(1000.0, tau_for_q100, prefix="IA", candidate_start=0),
        _parameter_estimate(1100.0, None, prefix="IB", candidate_start=10),
    )
    snapshot = deepcopy(estimates)
    sources = {
        estimates[0].estimate_id: _bandwidth_source(center=1000.0, width=10.0, spectrum_id="ia"),
        estimates[1].estimate_id: _bandwidth_source(center=1100.0, width=11.0, spectrum_id="ib"),
    }

    first = estimate_modal_q_factors(estimates, bandwidth_sources=sources)
    second = estimate_modal_q_factors(estimates, bandwidth_sources=sources)

    assert estimates == snapshot
    assert first == second
    assert all(isinstance(item.provenance.candidate_ids, tuple) for item in first.estimates)


@pytest.mark.parametrize(
    "changes",
    [
        {"decay_convention": "unknown"},
        {"enable_decay_method": False, "enable_bandwidth_method": False},
        {"require_decay_method": True, "enable_decay_method": False},
        {"require_bandwidth_method": True, "enable_bandwidth_method": False},
        {"minimum_frequency_hz": 0.0},
        {"minimum_tau_s": -1.0},
        {"maximum_decay_q": float("inf")},
        {"bandwidth_definition": "unknown"},
        {"bandwidth_level_db": 0.0},
        {"minimum_bandwidth_hz": 0.0},
        {"minimum_spectral_resolution_ratio": -0.1},
        {"maximum_neighboring_peak_overlap_fraction": 1.1},
        {"maximum_relative_q_disagreement": -0.1},
        {"uncertainty_method": "unknown"},
        {"uncertainty_confidence_level": 1.0},
        {"bootstrap_sample_count": 0},
        {"bootstrap_random_seed": 1.5},
        {"prefer_decay_method": True, "prefer_bandwidth_method": True},
    ],
    ids=[
        "bad_decay_convention",
        "no_methods",
        "required_decay_disabled",
        "required_bandwidth_disabled",
        "minimum_frequency_zero",
        "minimum_tau_negative",
        "maximum_q_infinite",
        "bad_bandwidth_definition",
        "bandwidth_level_nonnegative",
        "minimum_bandwidth_zero",
        "negative_resolution_ratio",
        "overlap_fraction",
        "negative_disagreement",
        "bad_uncertainty_method",
        "confidence_endpoint",
        "bootstrap_zero",
        "seed_float",
        "two_preferences",
    ],
)
def test_settings_reject_invalid_invariants(changes) -> None:
    with pytest.raises(ValueError):
        ModalQFactorEstimationSettings(**changes)


def test_precomputed_peak_widths_are_reused_without_lorentzian_fit() -> None:
    global_peak = GlobalSpectralPeakMetric(
        peak_index=10,
        bin_frequency_hz=1000.0,
        refined_frequency_hz=None,
        representative_frequency_hz=1000.0,
        power=1.0,
        relative_power=1.0,
        prominence=1.0,
        width_bins=10.0,
        width_hz=10.0,
        left_frequency_hz=995.0,
        right_frequency_hz=1005.0,
        isolation_index=None,
        overlap_classification="indeterminate",
        isolated=False,
        overlapping=False,
        resolution_limited=False,
        diagnostics=("width_not_a_formal_uncertainty",),
    )
    spectral_peak = SpectralPeak(
        bin_index=10,
        bin_frequency_hz=1000.0,
        bin_amplitude=1.0,
        amplitude_unit="normalized amplitude (peak)",
        width_hz=10.0,
        width_method="half-prominence",
    )

    global_bandwidth = estimate_modal_bandwidth(
        1000.0,
        precomputed_peak=global_peak,
        frequency_resolution_hz=1.0,
        settings=ModalQFactorEstimationSettings(require_isolated_peak=False),
    )
    peak_bandwidth = estimate_modal_bandwidth(
        1000.0,
        precomputed_peak=spectral_peak,
        frequency_resolution_hz=1.0,
    )

    assert global_bandwidth.bandwidth_definition is ModalBandwidthDefinition.HALF_PROMINENCE_POWER
    assert global_bandwidth.bandwidth_hz == pytest.approx(10.0)
    assert "global_peak_width_is_half_prominence_in_canonical_power" in global_bandwidth.diagnostics
    assert peak_bandwidth.bandwidth_definition is ModalBandwidthDefinition.HALF_PROMINENCE_AMPLITUDE
    assert "precomputed_width_without_crossing_bounds" in peak_bandwidth.diagnostics


def test_numeric_invariants_have_no_nan_no_infinity_and_no_zero_substitution() -> None:
    tau_for_q100 = 100.0 / (pi * 1000.0)
    estimate = estimate_modal_q_factor_for_parameter_estimate(
        _parameter_estimate(1000.0, tau_for_q100, prefix="NI"),
        bandwidth_source=_bandwidth_source(width=10.0, uncertainty=0.5),
    )
    numbers = [
        estimate.decay_q_estimate.q_decay,
        estimate.bandwidth_estimate.bandwidth_hz,
        estimate.bandwidth_q_estimate.q_bandwidth,
        estimate.method_comparison.relative_symmetric_difference,
        estimate.representative_q,
    ]

    assert estimate.status is ModalQFactorEstimateStatus.VALID
    assert all(value is not None and isfinite(value) for value in numbers)
    assert all(value > 0.0 for value in numbers if value != 0.0)
    assert estimate.bandwidth_estimate.lower_frequency_hz < estimate.bandwidth_estimate.center_frequency_hz < estimate.bandwidth_estimate.upper_frequency_hz
    assert estimate.decay_q_estimate.lower_bound_q <= estimate.decay_q_estimate.q_decay <= estimate.decay_q_estimate.upper_bound_q
    assert estimate.bandwidth_q_estimate.lower_bound_q <= estimate.bandwidth_q_estimate.q_bandwidth <= estimate.bandwidth_q_estimate.upper_bound_q


def test_public_provenance_and_summary_are_deterministic() -> None:
    parameter = _parameter_estimate(prefix="PR")
    source = _bandwidth_source(spectrum_id="provenance-spectrum")

    provenance = estimate_modal_q_factor_provenance(parameter, bandwidth_source=source)
    result = estimate_modal_q_factors((parameter,), bandwidth_sources={parameter.estimate_id: source})
    summary = summarize_modal_q_factor_estimates(result)

    assert provenance.modal_parameter_estimate_id == parameter.estimate_id
    assert provenance.hypothesis_id == parameter.hypothesis_id
    assert provenance.candidate_ids == parameter.provenance.candidate_ids
    assert provenance.recording_ids == parameter.provenance.recording_ids
    assert provenance.spectrum_ids == ("provenance-spectrum",)
    assert provenance.settings_fingerprint == modal_q_settings_fingerprint()
    assert summary["estimate_count"] == 1
    assert summary["estimate_ids"] == tuple(item.estimate_id for item in result.estimates)
    assert summary["settings_fingerprint"] == modal_q_settings_fingerprint()
