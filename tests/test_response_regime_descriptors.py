"""Tests for operational response-regime descriptors."""

from __future__ import annotations

import pytest

from belllab import (
    AggregatedMetric,
    DescriptorEvaluation,
    DynamicConditionComparisonResult,
    DynamicConditionSpectralSummary,
    DynamicResponseRegimeDescription,
    RegimeCriterionWeight,
    RegimeDescriptorCriterionResult,
    RegimeDescriptorScore,
    ResponseRegimeDescription,
    ResponseRegimeDescriptorSettings,
    describe_dynamic_response_regimes,
    describe_response_regime,
    evaluate_regime_descriptor,
)


def _agg(
    name: str,
    value: float | None,
    *,
    unit: str = "fraction",
    cv: float | None = None,
) -> AggregatedMetric:
    if value is None:
        return AggregatedMetric(
            metric_name=name,
            unit=unit,
            available_count=1,
            finite_count=0,
            discarded_count=1,
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
            diagnostics=(),
        )
    return AggregatedMetric(
        metric_name=name,
        unit=unit,
        available_count=3,
        finite_count=3,
        discarded_count=0,
        median=value,
        mean=value,
        standard_deviation=0.0,
        minimum=value,
        maximum=value,
        range=0.0,
        coefficient_of_variation=cv,
        median_absolute_deviation=0.0,
        valid=True,
        failure_reason=None,
        diagnostics=(),
    )


def _summary(
    label: str,
    values: dict[str, float | None],
    *,
    repeat_count: int = 3,
    valid_repeat_count: int | None = None,
    clipped_fraction: float | None = 0.0,
    near_clipped_fraction: float | None = None,
    comparability_status: tuple[str, ...] = (),
    resolution_hz: float = 10.0,
    cv_by_metric: dict[str, float] | None = None,
    valid: bool = True,
) -> DynamicConditionSpectralSummary:
    cv_by_metric = cv_by_metric or {}
    metrics = tuple(
        _agg(name, value, unit=_unit(name), cv=cv_by_metric.get(name))
        for name, value in values.items()
    )
    excitation = tuple(metric for metric in metrics if metric.metric_name.startswith("excitation_"))
    global_metrics = tuple(metric for metric in metrics if metric.metric_name.startswith("global_"))
    time_metrics = tuple(metric for metric in metrics if metric.metric_name.startswith("time_"))
    early = tuple(metric for metric in metrics if metric.metric_name.startswith("early_"))
    middle = tuple(metric for metric in metrics if metric.metric_name.startswith("middle_"))
    late = tuple(metric for metric in metrics if metric.metric_name.startswith("late_"))
    region_change = tuple(metric for metric in metrics if metric.metric_name.startswith("region_"))
    valid_repeat_count = repeat_count if valid_repeat_count is None else valid_repeat_count
    return DynamicConditionSpectralSummary(
        dynamic_label=label,
        recording_ids=tuple(f"{label}-{index}" for index in range(repeat_count)),
        repeat_count=repeat_count,
        valid_repeat_count=valid_repeat_count,
        discarded_repeat_count=repeat_count - valid_repeat_count,
        excitation_metrics=excitation,
        global_spectral_metrics=global_metrics,
        time_resolved_metrics=time_metrics,
        early_region_metrics=early,
        middle_region_metrics=middle,
        late_region_metrics=late,
        region_change_metrics=region_change,
        band_metrics=(),
        within_condition_variability=tuple(metric for metric in metrics if metric.coefficient_of_variation is not None),
        comparability_status=comparability_status,
        instrumental_fingerprint=(),
        global_spectral_fingerprint=(("frequency_resolution_hz", resolution_hz),),
        time_resolved_fingerprint=(),
        band_definitions=(),
        region_definitions=(("early", 0.0, 0.2), ("middle", 0.2, 0.5), ("late", 0.7, 1.0)),
        clipped_repeat_fraction=clipped_fraction,
        near_clipped_repeat_fraction=clipped_fraction if near_clipped_fraction is None else near_clipped_fraction,
        valid=valid,
        failure_reason=None if valid else "test_invalid",
        diagnostics=(),
    )


def _unit(name: str) -> str:
    if name.endswith("_hz") or name.endswith("_bandwidth_hz") or "centroid" in name:
        return "Hz"
    if "density" in name:
        return "1/Hz"
    if "count" in name:
        return "count"
    if "crest" in name:
        return "ratio"
    if name.endswith("_db") or "db" in name:
        return "dB"
    return "fraction"


def _line_metrics() -> dict[str, float]:
    return {
        "global_spectral_flatness": 0.04,
        "global_spectral_entropy": 0.20,
        "global_tonal_energy_fraction": 0.90,
        "global_residual_energy_fraction": 0.10,
        "global_spectral_crest_factor": 12.0,
        "global_peak_density_per_hz": 0.004,
        "global_significant_peak_count": 3.0,
        "global_median_peak_spacing_hz": 120.0,
        "global_occupied_frequency_fraction": 0.20,
        "time_temporal_coverage_fraction": 0.95,
        "time_spectral_centroid_hz_slope_per_s": 2.0,
        "late_tonal_energy_fraction": 0.88,
        "excitation_signal_to_background_db": 30.0,
    }


def _broadband_metrics() -> dict[str, float]:
    return {
        "global_spectral_flatness": 0.82,
        "global_spectral_entropy": 0.93,
        "global_tonal_energy_fraction": 0.15,
        "global_residual_energy_fraction": 0.85,
        "global_spectral_crest_factor": 1.8,
        "global_peak_density_per_hz": 0.030,
        "global_significant_peak_count": 12.0,
        "global_median_peak_spacing_hz": 20.0,
        "global_occupied_frequency_fraction": 0.88,
        "time_temporal_coverage_fraction": 0.92,
        "excitation_signal_to_background_db": 25.0,
    }


def _dense_tonal_metrics() -> dict[str, float]:
    return {
        "global_spectral_flatness": 0.25,
        "global_spectral_entropy": 0.62,
        "global_tonal_energy_fraction": 0.82,
        "global_residual_energy_fraction": 0.18,
        "global_spectral_crest_factor": 6.0,
        "global_peak_density_per_hz": 0.035,
        "global_significant_peak_count": 18.0,
        "global_median_peak_spacing_hz": 30.0,
        "global_occupied_frequency_fraction": 0.75,
        "time_temporal_coverage_fraction": 0.90,
        "excitation_signal_to_background_db": 25.0,
    }


def _mixed_metrics() -> dict[str, float]:
    return {
        "global_spectral_flatness": 0.42,
        "global_spectral_entropy": 0.66,
        "global_tonal_energy_fraction": 0.48,
        "global_residual_energy_fraction": 0.52,
        "global_spectral_crest_factor": 4.0,
        "global_peak_density_per_hz": 0.015,
        "global_significant_peak_count": 7.0,
        "global_median_peak_spacing_hz": 45.0,
        "global_occupied_frequency_fraction": 0.55,
        "time_temporal_coverage_fraction": 0.88,
        "excitation_signal_to_background_db": 22.0,
    }


def _broadband_to_tonal_metrics() -> dict[str, float]:
    metrics = _mixed_metrics()
    metrics.update({
        "early_spectral_flatness": 0.72,
        "late_spectral_flatness": 0.18,
        "region_late_minus_early_spectral_flatness": -0.54,
        "region_late_minus_early_spectral_entropy": -0.35,
        "region_late_minus_early_tonal_energy_fraction": 0.45,
        "region_late_minus_early_residual_energy_fraction": -0.45,
        "region_late_minus_early_occupied_bandwidth_hz": -220.0,
        "region_late_minus_early_peak_density_per_hz": -0.020,
        "region_late_minus_early_significant_peak_count": -5.0,
        "time_change_point_count": 2.0,
        "time_temporal_coverage_fraction": 0.95,
        "time_spectral_centroid_hz_slope_per_s": 4.0,
        "late_tonal_energy_fraction": 0.82,
    })
    return metrics


def _tonal_to_broadband_metrics() -> dict[str, float]:
    metrics = _mixed_metrics()
    metrics.update({
        "region_late_minus_early_spectral_flatness": 0.35,
        "region_late_minus_early_spectral_entropy": 0.30,
        "region_late_minus_early_tonal_energy_fraction": -0.35,
        "region_late_minus_early_residual_energy_fraction": 0.35,
        "region_late_minus_early_occupied_bandwidth_hz": 180.0,
        "region_late_minus_early_peak_density_per_hz": 0.010,
        "region_late_minus_early_significant_peak_count": 4.0,
        "time_change_point_count": 1.0,
    })
    return metrics


def test_settings_validate_threshold_coherence() -> None:
    with pytest.raises(ValueError, match="line flatness maximum"):
        ResponseRegimeDescriptorSettings(
            maximum_flatness_for_line_dominated=0.8,
            minimum_flatness_for_broadband=0.6,
        )


def test_criterion_invariant_rejects_non_applicable_pass_state() -> None:
    with pytest.raises(ValueError, match="non-applicable"):
        RegimeDescriptorCriterionResult(
            "bad",
            "global_spectral_flatness",
            None,
            "<=",
            0.2,
            True,
            False,
            1.0,
            "support",
            "bad",
        )


def test_discrete_line_dominated_descriptor_is_auditable() -> None:
    description = describe_response_regime(_summary("pp", _line_metrics()))
    score = evaluate_regime_descriptor(_summary("pp", _line_metrics()), "discrete_line_dominated")

    assert description.structure_descriptor == "discrete_line_dominated"
    assert description.confidence_descriptor in {"high", "moderate"}
    assert score.selected
    assert score.support_fraction == pytest.approx(1.0)
    assert "no_non_linearity_proof_was_performed" in description.diagnostics
    assert "ModalMode" not in " ".join(description.diagnostics)


def test_broadband_dominated_uses_multiple_criteria_not_white_noise_label() -> None:
    description = describe_response_regime(_summary("ff", _broadband_metrics()))
    broadband = next(item for item in description.descriptor_results if item.descriptor_name == "broadband_dominated")

    assert description.structure_descriptor == "broadband_dominated"
    assert broadband.score.selected
    assert broadband.score.support_score >= 5.0
    assert "white_noise" not in description.structure_descriptor
    assert "no_non_linearity_proof_was_performed" in description.diagnostics


def test_dense_tonal_spectrum_is_separate_from_broadband() -> None:
    description = describe_response_regime(_summary("f", _dense_tonal_metrics()))

    assert description.structure_descriptor == "dense_spectrum"
    assert evaluate_regime_descriptor(_summary("f", _dense_tonal_metrics()), "dense_spectrum").selected
    assert not evaluate_regime_descriptor(_summary("f", _dense_tonal_metrics()), "broadband_dominated").selected


def test_mixed_line_and_continuum_preserves_conflicts() -> None:
    description = describe_response_regime(_summary("mf", _mixed_metrics()))

    assert description.structure_descriptor == "mixed_line_and_continuum"
    assert "global_tonal_energy_fraction" in description.supporting_metrics
    assert description.conflicting_metrics


def test_central_pp_ff_scenario_broadband_attack_tonal_tail() -> None:
    pp = _summary("pp", _line_metrics())
    ff = _summary("ff", _broadband_to_tonal_metrics())

    pp_description = describe_response_regime(pp)
    ff_description = describe_response_regime(ff)

    assert pp_description.structure_descriptor == "discrete_line_dominated"
    assert ff_description.structure_descriptor in {"mixed_line_and_continuum", "dense_spectrum"}
    assert ff_description.temporal_evolution_descriptor == "broadband_to_tonal"
    assert ff_description.line_identity_descriptor in {"line_identity_preserved", "line_identity_partially_preserved"}
    assert "no_non_linearity_proof_was_performed" in ff_description.diagnostics


def test_tonal_to_broadband_direction_is_allowed() -> None:
    description = describe_response_regime(_summary("f", _tonal_to_broadband_metrics()))

    assert description.temporal_evolution_descriptor == "tonal_to_broadband"
    assert "operational_response_regime_descriptor_not_physical_regime_proof" in description.diagnostics


def test_conflicting_metrics_reduce_to_mixed_or_indeterminate() -> None:
    metrics = {
        "global_spectral_flatness": 0.05,
        "global_spectral_entropy": 0.90,
        "global_tonal_energy_fraction": 0.20,
        "global_residual_energy_fraction": 0.80,
        "global_spectral_crest_factor": 12.0,
        "global_peak_density_per_hz": 0.015,
        "global_occupied_frequency_fraction": 0.50,
        "time_temporal_coverage_fraction": 0.90,
    }

    description = describe_response_regime(_summary("mf", metrics))

    assert description.structure_descriptor in {"mixed_line_and_continuum", "indeterminate"}
    assert "global_spectral_entropy" in description.conflicting_metrics
    assert description.confidence_descriptor in {"low", "moderate"}


def test_clipping_reduces_confidence_but_preserves_description_when_allowed() -> None:
    description = describe_response_regime(_summary("ff", _broadband_metrics(), clipped_fraction=1.0))

    assert description.valid
    assert description.structure_descriptor == "broadband_dominated"
    assert description.confidence_descriptor == "low"
    assert "spectral_metrics_potentially_distorted_by_clipping" in description.limitations


def test_clipping_can_be_rejected_by_configuration() -> None:
    settings = ResponseRegimeDescriptorSettings(reject_clipped_conditions=True)
    description = describe_response_regime(_summary("ff", _broadband_metrics(), clipped_fraction=1.0), settings)

    assert not description.valid
    assert description.failure_reason == "clipped_condition_rejected_by_configuration"
    assert description.confidence_descriptor == "insufficient"


def test_low_snr_marks_low_confidence_without_erasing_condition() -> None:
    settings = ResponseRegimeDescriptorSettings(minimum_signal_to_background_db=20.0)
    metrics = _line_metrics()
    metrics["excitation_signal_to_background_db"] = 5.0

    description = describe_response_regime(_summary("pp", metrics), settings)

    assert description.valid
    assert description.confidence_descriptor == "low"
    assert "low_signal_to_background_ratio" in description.limitations


def test_missing_metrics_are_non_applicable_and_can_be_indeterminate() -> None:
    metrics = {
        "global_spectral_flatness": None,
        "global_spectral_entropy": None,
        "global_tonal_energy_fraction": None,
    }

    description = describe_response_regime(_summary("p", metrics))
    score = evaluate_regime_descriptor(_summary("p", metrics), "discrete_line_dominated")

    assert description.structure_descriptor == "indeterminate"
    assert score.available_weight == pytest.approx(0.0)
    assert "global_spectral_flatness" in description.unavailable_metrics


def test_missing_metrics_can_invalidate_when_policy_requires_them() -> None:
    settings = ResponseRegimeDescriptorSettings(allow_missing_metrics=False)
    description = describe_response_regime(_summary("p", {"global_spectral_flatness": None}), settings)

    assert not description.valid
    assert description.failure_reason == "missing_metrics_not_allowed"


def test_resolution_limited_density_reduces_density_weight_and_confidence() -> None:
    metrics = _dense_tonal_metrics()
    metrics["global_median_peak_spacing_hz"] = 15.0
    description = describe_response_regime(_summary("f", metrics, resolution_hz=10.0))
    dense = next(item for item in description.descriptor_results if item.descriptor_name == "dense_spectrum")
    density_criterion = next(item for item in dense.criteria if item.criterion_name == "dense_peak_density_high")

    assert density_criterion.weight == pytest.approx(0.25)
    assert "density_metrics_resolution_limited" in description.limitations
    assert description.confidence_descriptor in {"low", "moderate"}


def test_high_within_condition_variability_reduces_confidence() -> None:
    metrics = _line_metrics()
    summary = _summary(
        "pp",
        metrics,
        cv_by_metric={"global_spectral_flatness": 1.5},
    )

    description = describe_response_regime(summary)

    assert "high_within_condition_variability" in description.limitations
    assert description.confidence_descriptor == "low"


def test_line_identity_not_evaluated_when_temporal_metrics_absent() -> None:
    metrics = _line_metrics()
    metrics.pop("time_temporal_coverage_fraction")
    metrics.pop("time_spectral_centroid_hz_slope_per_s")
    metrics.pop("late_tonal_energy_fraction")

    description = describe_response_regime(_summary("pp", metrics))

    assert description.line_identity_descriptor in {"not_evaluated", "line_identity_not_resolved"}


def test_line_identity_preserved_with_coverage_stability_and_tonality() -> None:
    description = describe_response_regime(_summary("pp", _line_metrics()))

    assert description.line_identity_descriptor == "line_identity_preserved"


def test_stable_temporal_character_descriptor() -> None:
    metrics = _line_metrics()
    metrics.update({
        "region_late_minus_early_spectral_flatness": 0.01,
        "region_late_minus_early_spectral_entropy": -0.02,
        "region_late_minus_early_tonal_energy_fraction": 0.01,
        "region_late_minus_early_peak_density_per_hz": 0.001,
        "region_late_minus_early_occupied_bandwidth_hz": 5.0,
    })
    settings = ResponseRegimeDescriptorSettings(maximum_bandwidth_change_for_stable=10.0)

    description = describe_response_regime(_summary("pp", metrics), settings)

    assert description.temporal_evolution_descriptor == "stable_spectral_character"


def test_progressive_densification_and_sparsification_are_independent() -> None:
    densifying = _mixed_metrics()
    densifying.update({
        "region_late_minus_early_peak_density_per_hz": 0.020,
        "region_late_minus_early_significant_peak_count": 5.0,
        "region_late_minus_early_occupied_bandwidth_hz": 100.0,
    })
    sparsifying = _mixed_metrics()
    sparsifying.update({
        "region_late_minus_early_peak_density_per_hz": -0.020,
        "region_late_minus_early_significant_peak_count": -5.0,
        "region_late_minus_early_occupied_bandwidth_hz": -100.0,
    })
    settings = ResponseRegimeDescriptorSettings(minimum_bandwidth_change=50.0)

    assert describe_response_regime(_summary("mf", densifying), settings).temporal_evolution_descriptor == "progressive_spectral_densification"
    assert describe_response_regime(_summary("mf", sparsifying), settings).temporal_evolution_descriptor == "progressive_spectral_sparsification"


def test_descriptor_sequence_pp_to_ff_and_emergent_patterns() -> None:
    summaries = (
        _summary("pp", _line_metrics()),
        _summary("p", _line_metrics()),
        _summary("mf", _mixed_metrics()),
        _summary("f", _dense_tonal_metrics()),
        _summary("ff", _broadband_metrics()),
    )

    result = describe_dynamic_response_regimes(summaries)
    structure_sequence = next(item for item in result.descriptor_sequences if item.dimension == "structure")

    assert isinstance(result, DynamicResponseRegimeDescription)
    assert structure_sequence.descriptors == (
        "discrete_line_dominated",
        "discrete_line_dominated",
        "mixed_line_and_continuum",
        "dense_spectrum",
        "broadband_dominated",
    )
    assert "descriptor_preserved" in structure_sequence.changes
    assert "descriptor_changed" in structure_sequence.changes
    assert result.emergent_patterns
    assert "no_non_linearity_proof_was_performed" in result.diagnostics


def test_descriptor_sequence_preserves_missing_conditions() -> None:
    result = describe_dynamic_response_regimes((
        _summary("pp", _line_metrics()),
        _summary("mf", _mixed_metrics()),
        _summary("ff", _broadband_metrics()),
    ))
    structure_sequence = next(item for item in result.descriptor_sequences if item.dimension == "structure")

    assert structure_sequence.missing_labels == ("p", "f")
    assert structure_sequence.descriptors[1] is None
    assert structure_sequence.descriptors[3] is None


def test_dynamic_result_input_is_supported() -> None:
    summaries = (
        _summary("pp", _line_metrics()),
        _summary("ff", _broadband_metrics()),
    )
    dynamic_result = DynamicConditionComparisonResult(
        condition_summaries=summaries,
        pairwise_comparisons=(),
        reference_comparisons=(),
        reference_dynamic_label=None,
        ordered_dynamic_labels=("pp", "ff"),
        missing_dynamic_labels=("p", "mf", "f"),
        metric_sequences=(),
        monotonicity_results=(),
        valid=True,
        failure_reason=None,
        diagnostics=(),
    )

    result = describe_dynamic_response_regimes(dynamic_result)

    assert result.ordered_labels == ("pp", "ff")


def test_only_invalid_condition_yields_structured_dynamic_failure() -> None:
    result = describe_dynamic_response_regimes((_summary("ff", {}, valid=False, valid_repeat_count=0),))

    assert not result.valid
    assert result.failure_reason == "no_valid_condition_descriptions"


def test_determinism_with_shuffled_input() -> None:
    summaries = (
        _summary("ff", _broadband_metrics()),
        _summary("pp", _line_metrics()),
        _summary("mf", _mixed_metrics()),
    )

    first = describe_dynamic_response_regimes(summaries)
    second = describe_dynamic_response_regimes(tuple(reversed(summaries)))

    assert first == second


def test_no_forbidden_physical_operations_are_reported() -> None:
    description = describe_response_regime(_summary("ff", _broadband_to_tonal_metrics()))

    joined = " ".join(description.diagnostics)
    assert "no_non_linearity_proof_was_performed" in joined
    assert "no_cross_condition_candidate_association_was_performed" in joined
    assert "no_modal_mode_conversion_was_performed" in joined


def _evaluation(description: ResponseRegimeDescription, descriptor_name: str) -> DescriptorEvaluation:
    return next(item for item in description.descriptor_results if item.descriptor_name == descriptor_name)


def _criterion_result(
    metric_name: str,
    value: float | None,
    descriptor_name: str,
    criterion_name: str,
    *,
    settings: ResponseRegimeDescriptorSettings | None = None,
    extra_metrics: dict[str, float | None] | None = None,
    resolution_hz: float = 10.0,
):
    metrics = {metric_name: value}
    if extra_metrics:
        metrics.update(extra_metrics)
    description = describe_response_regime(_summary("pp", metrics, resolution_hz=resolution_hz), settings)
    criterion = next(
        item for item in _evaluation(description, descriptor_name).criteria
        if item.criterion_name == criterion_name
    )
    return criterion


@pytest.mark.parametrize(
    ("metric_name", "descriptor_name", "criterion_name", "threshold", "below_pass", "at_pass", "above_pass"),
    (
        ("global_spectral_flatness", "discrete_line_dominated", "line_flatness_low", 0.25, True, True, False),
        ("global_spectral_flatness", "broadband_dominated", "broadband_flatness_high", 0.55, False, True, True),
        ("global_spectral_entropy", "discrete_line_dominated", "line_entropy_low", 0.55, True, True, False),
        ("global_spectral_entropy", "broadband_dominated", "broadband_entropy_high", 0.75, False, True, True),
        ("global_tonal_energy_fraction", "discrete_line_dominated", "line_tonal_fraction_high", 0.65, False, True, True),
        ("global_tonal_energy_fraction", "broadband_dominated", "broadband_tonal_fraction_low", 0.35, True, True, False),
        ("global_residual_energy_fraction", "broadband_dominated", "broadband_residual_fraction_high", 0.65, False, True, True),
        ("global_peak_density_per_hz", "discrete_line_dominated", "line_peak_density_sparse", 0.010, True, True, False),
        ("global_peak_density_per_hz", "dense_spectrum", "dense_peak_density_high", 0.020, False, True, True),
        ("global_occupied_frequency_fraction", "discrete_line_dominated", "line_occupied_fraction_narrow", 0.35, True, True, False),
        ("global_occupied_frequency_fraction", "broadband_dominated", "broadband_occupied_fraction_high", 0.65, False, True, True),
        ("global_spectral_crest_factor", "discrete_line_dominated", "line_spectral_crest_high", 5.0, False, True, True),
        ("global_spectral_crest_factor", "broadband_dominated", "broadband_spectral_crest_low", 3.0, True, True, False),
        ("global_significant_peak_count", "dense_spectrum", "dense_peak_count_high", 8.0, False, True, True),
    ),
)
def test_primary_threshold_boundaries_are_inclusive(
    metric_name: str,
    descriptor_name: str,
    criterion_name: str,
    threshold: float,
    below_pass: bool,
    at_pass: bool,
    above_pass: bool,
) -> None:
    delta = 1e-6

    below = _criterion_result(metric_name, threshold - delta, descriptor_name, criterion_name)
    at = _criterion_result(metric_name, threshold, descriptor_name, criterion_name)
    above = _criterion_result(metric_name, threshold + delta, descriptor_name, criterion_name)

    assert below.passed is below_pass
    assert at.passed is at_pass
    assert above.passed is above_pass


@pytest.mark.parametrize(
    ("metric_name", "descriptor_name", "criterion_name", "threshold", "below_pass", "at_pass", "above_pass"),
    (
        ("region_late_minus_early_spectral_flatness", "broadband_to_tonal", "b2t_flatness_drop", -0.15, True, True, False),
        ("region_late_minus_early_spectral_entropy", "broadband_to_tonal", "b2t_entropy_drop", -0.15, True, True, False),
        ("region_late_minus_early_tonal_energy_fraction", "broadband_to_tonal", "b2t_tonal_fraction_increase", 0.20, False, True, True),
        ("region_late_minus_early_residual_energy_fraction", "broadband_to_tonal", "b2t_residual_fraction_drop", -0.20, True, True, False),
        ("region_late_minus_early_spectral_flatness", "tonal_to_broadband", "t2b_flatness_increase", 0.15, False, True, True),
        ("region_late_minus_early_spectral_entropy", "tonal_to_broadband", "t2b_entropy_increase", 0.15, False, True, True),
        ("region_late_minus_early_tonal_energy_fraction", "tonal_to_broadband", "t2b_tonal_fraction_drop", -0.20, True, True, False),
        ("region_late_minus_early_residual_energy_fraction", "tonal_to_broadband", "t2b_residual_fraction_increase", 0.20, False, True, True),
        ("region_late_minus_early_peak_density_per_hz", "progressive_spectral_densification", "densification_density_increase", 0.005, False, True, True),
        ("region_late_minus_early_peak_density_per_hz", "progressive_spectral_sparsification", "sparsification_density_drop", -0.005, True, True, False),
        ("time_change_point_count", "broadband_to_tonal", "b2t_change_points", 1.0, False, True, True),
    ),
)
def test_temporal_threshold_boundaries_are_inclusive(
    metric_name: str,
    descriptor_name: str,
    criterion_name: str,
    threshold: float,
    below_pass: bool,
    at_pass: bool,
    above_pass: bool,
) -> None:
    delta = 1e-6

    assert _criterion_result(metric_name, threshold - delta, descriptor_name, criterion_name).passed is below_pass
    assert _criterion_result(metric_name, threshold, descriptor_name, criterion_name).passed is at_pass
    assert _criterion_result(metric_name, threshold + delta, descriptor_name, criterion_name).passed is above_pass


def test_stable_change_threshold_is_inclusive_on_both_sides() -> None:
    criterion_name = "stable_flatness_change_small"
    descriptor_name = "stable_spectral_character"
    metric_name = "region_late_minus_early_spectral_flatness"

    assert _criterion_result(metric_name, -0.050001, descriptor_name, criterion_name).passed is False
    assert _criterion_result(metric_name, -0.05, descriptor_name, criterion_name).passed is True
    assert _criterion_result(metric_name, 0.05, descriptor_name, criterion_name).passed is True
    assert _criterion_result(metric_name, 0.050001, descriptor_name, criterion_name).passed is False


def test_quality_threshold_boundaries_for_snr_coverage_repeats_and_variability() -> None:
    snr_settings = ResponseRegimeDescriptorSettings(minimum_signal_to_background_db=20.0)
    assert "low_signal_to_background_ratio" in describe_response_regime(
        _summary("pp", {**_line_metrics(), "excitation_signal_to_background_db": 19.999999}),
        snr_settings,
    ).limitations
    assert "low_signal_to_background_ratio" not in describe_response_regime(
        _summary("pp", {**_line_metrics(), "excitation_signal_to_background_db": 20.0}),
        snr_settings,
    ).limitations
    assert "low_signal_to_background_ratio" not in describe_response_regime(
        _summary("pp", {**_line_metrics(), "excitation_signal_to_background_db": 20.000001}),
        snr_settings,
    ).limitations

    assert "low_valid_temporal_coverage" in describe_response_regime(
        _summary("pp", {**_line_metrics(), "time_temporal_coverage_fraction": 0.499999})
    ).limitations
    assert "low_valid_temporal_coverage" not in describe_response_regime(
        _summary("pp", {**_line_metrics(), "time_temporal_coverage_fraction": 0.5})
    ).limitations

    repeat_settings = ResponseRegimeDescriptorSettings(minimum_valid_repeat_count=2)
    assert "insufficient_valid_repeats" in describe_response_regime(
        _summary("pp", _line_metrics(), repeat_count=2, valid_repeat_count=1),
        repeat_settings,
    ).limitations
    assert "insufficient_valid_repeats" not in describe_response_regime(
        _summary("pp", _line_metrics(), repeat_count=2, valid_repeat_count=2),
        repeat_settings,
    ).limitations

    assert "high_within_condition_variability" not in describe_response_regime(
        _summary("pp", _line_metrics(), cv_by_metric={"global_spectral_flatness": 0.5})
    ).limitations
    assert "high_within_condition_variability" in describe_response_regime(
        _summary("pp", _line_metrics(), cv_by_metric={"global_spectral_flatness": 0.500001})
    ).limitations


def test_score_formula_excludes_missing_and_disabled_criteria_from_available_weight() -> None:
    settings = ResponseRegimeDescriptorSettings(
        maximum_peak_density_for_sparse=None,
        minimum_available_weight_for_descriptor=3.0,
        criterion_weights=(
            RegimeCriterionWeight("line_flatness_low", 2.0),
            RegimeCriterionWeight("line_entropy_low", 1.0),
        ),
    )
    summary = _summary("pp", {
        "global_spectral_flatness": 0.10,
        "global_spectral_entropy": 0.80,
        "global_tonal_energy_fraction": None,
        "global_peak_density_per_hz": 0.0,
    })

    score = evaluate_regime_descriptor(summary, "discrete_line_dominated", settings)
    criteria = _evaluation(describe_response_regime(summary, settings), "discrete_line_dominated").criteria

    assert score.support_score == pytest.approx(2.0)
    assert score.opposition_score == pytest.approx(1.0)
    assert score.available_weight == pytest.approx(3.0)
    assert score.support_fraction == pytest.approx(2.0 / 3.0)
    assert score.opposition_fraction == pytest.approx(1.0 / 3.0)
    assert next(item for item in criteria if item.criterion_name == "line_tonal_fraction_high").applicable is False
    assert next(item for item in criteria if item.criterion_name == "line_peak_density_sparse").applicable is False


def test_zero_available_weight_is_indeterminate_without_fractions_or_division() -> None:
    score = evaluate_regime_descriptor(
        _summary("pp", {"global_spectral_flatness": None, "global_spectral_entropy": None}),
        "discrete_line_dominated",
    )

    assert score.available_weight == pytest.approx(0.0)
    assert score.support_fraction is None
    assert score.opposition_fraction is None
    assert not score.selected
    assert score.indeterminate


def test_selection_thresholds_for_support_opposition_and_available_weight_are_inclusive() -> None:
    metrics = {
        "global_spectral_flatness": 0.10,
        "global_spectral_entropy": 0.20,
        "global_tonal_energy_fraction": 0.90,
        "global_spectral_crest_factor": 1.0,
    }
    exact = ResponseRegimeDescriptorSettings(
        minimum_support_fraction_for_descriptor=0.75,
        maximum_opposition_fraction_for_descriptor=0.25,
        minimum_available_weight_for_descriptor=4.0,
    )

    assert evaluate_regime_descriptor(_summary("pp", metrics), "discrete_line_dominated", exact).selected
    assert not evaluate_regime_descriptor(
        _summary("pp", metrics),
        "discrete_line_dominated",
        ResponseRegimeDescriptorSettings(minimum_support_fraction_for_descriptor=0.750001, minimum_available_weight_for_descriptor=4.0),
    ).selected
    assert not evaluate_regime_descriptor(
        _summary("pp", metrics),
        "discrete_line_dominated",
        ResponseRegimeDescriptorSettings(maximum_opposition_fraction_for_descriptor=0.249999, minimum_available_weight_for_descriptor=4.0),
    ).selected
    assert not evaluate_regime_descriptor(
        _summary("pp", metrics),
        "discrete_line_dominated",
        ResponseRegimeDescriptorSettings(minimum_available_weight_for_descriptor=4.000001),
    ).selected
    assert evaluate_regime_descriptor(
        _summary("pp", metrics),
        "discrete_line_dominated",
        ResponseRegimeDescriptorSettings(minimum_available_weight_for_descriptor=3.999999),
    ).selected


def test_score_tie_tolerance_closes_near_selection_boundaries() -> None:
    metrics = {
        "global_spectral_flatness": 0.10,
        "global_spectral_entropy": 0.80,
    }

    def score_for(support_weight: float) -> bool:
        settings = ResponseRegimeDescriptorSettings(
            minimum_support_fraction_for_descriptor=0.55,
            maximum_opposition_fraction_for_descriptor=0.45,
            minimum_available_weight_for_descriptor=1.0,
            score_tie_tolerance=1e-12,
            minimum_tonal_fraction_for_line_dominated=None,
            minimum_spectral_crest_for_line_dominated=None,
            maximum_peak_density_for_sparse=None,
            maximum_occupied_fraction_for_narrow=None,
            criterion_weights=(
                RegimeCriterionWeight("line_flatness_low", support_weight),
                RegimeCriterionWeight("line_entropy_low", 1.0 - support_weight),
            ),
        )
        return evaluate_regime_descriptor(_summary("pp", metrics), "discrete_line_dominated", settings).selected

    assert score_for(0.55 - 0.5e-12)
    assert score_for(0.55 - 1.0e-12)
    assert not score_for(0.55 - 2.0e-12)


def test_weight_scaling_changes_absolute_scores_not_fractions_or_selection() -> None:
    metrics = _line_metrics()
    base_settings = ResponseRegimeDescriptorSettings()
    scaled_settings = ResponseRegimeDescriptorSettings(
        criterion_weights=(
            RegimeCriterionWeight("line_flatness_low", 3.0),
            RegimeCriterionWeight("line_entropy_low", 3.0),
            RegimeCriterionWeight("line_tonal_fraction_high", 3.0),
            RegimeCriterionWeight("line_spectral_crest_high", 3.0),
            RegimeCriterionWeight("line_peak_density_sparse", 3.0),
            RegimeCriterionWeight("line_occupied_fraction_narrow", 3.0),
        )
    )

    base = evaluate_regime_descriptor(_summary("pp", metrics), "discrete_line_dominated", base_settings)
    scaled = evaluate_regime_descriptor(_summary("pp", metrics), "discrete_line_dominated", scaled_settings)

    assert scaled.support_score == pytest.approx(base.support_score * 3.0)
    assert scaled.available_weight == pytest.approx(base.available_weight * 3.0)
    assert scaled.support_fraction == pytest.approx(base.support_fraction)
    assert scaled.opposition_fraction == pytest.approx(base.opposition_fraction)
    assert scaled.selected == base.selected


def test_asymmetric_weights_make_weighted_support_predictable() -> None:
    settings = ResponseRegimeDescriptorSettings(
        criterion_weights=(
            RegimeCriterionWeight("line_flatness_low", 4.0),
            RegimeCriterionWeight("line_entropy_low", 1.0),
        )
    )
    metrics = {"global_spectral_flatness": 0.10, "global_spectral_entropy": 0.80}

    score = evaluate_regime_descriptor(_summary("pp", metrics), "discrete_line_dominated", settings)

    assert score.support_score == pytest.approx(4.0)
    assert score.opposition_score == pytest.approx(1.0)
    assert score.support_fraction == pytest.approx(0.8)


def test_score_contract_rejects_incoherent_manual_scores() -> None:
    with pytest.raises(ValueError, match="sum to available weight"):
        RegimeDescriptorScore("broadband_dominated", 1.0, 1.0, 3.0, 1 / 3, 1 / 3, False, False)
    with pytest.raises(ValueError, match="support fraction"):
        RegimeDescriptorScore("broadband_dominated", 1.0, 1.0, 2.0, 0.9, 0.5, False, False)
    with pytest.raises(ValueError, match="zero available weight"):
        RegimeDescriptorScore("broadband_dominated", 0.0, 0.0, 0.0, 0.0, None, False, True)
    with pytest.raises(ValueError, match="positive support"):
        RegimeDescriptorScore("broadband_dominated", 0.0, 1.0, 1.0, 0.0, 1.0, True, False)


def test_invalid_settings_cover_boundaries_weights_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="strictly below broadband flatness"):
        ResponseRegimeDescriptorSettings(maximum_flatness_for_line_dominated=0.55, minimum_flatness_for_broadband=0.55)
    with pytest.raises(ValueError, match="strictly below dense density"):
        ResponseRegimeDescriptorSettings(maximum_peak_density_for_sparse=0.02, minimum_peak_density_for_dense=0.02)
    with pytest.raises(ValueError, match="minimum_support_fraction"):
        ResponseRegimeDescriptorSettings(minimum_support_fraction_for_descriptor=float("nan"))
    with pytest.raises(ValueError, match="score_tie_tolerance"):
        ResponseRegimeDescriptorSettings(score_tie_tolerance=-1e-12)
    with pytest.raises(ValueError, match="criterion weight"):
        ResponseRegimeDescriptorSettings(criterion_weights=(RegimeCriterionWeight("line_flatness_low", -1.0),))
    with pytest.raises(ValueError, match="at least one positive"):
        ResponseRegimeDescriptorSettings(criterion_weights=(RegimeCriterionWeight("line_flatness_low", 0.0),))
    with pytest.raises(ValueError, match="minimum_density_change"):
        ResponseRegimeDescriptorSettings(minimum_density_change=float("inf"))


def test_structure_tie_policy_is_deterministic_and_metric_order_independent() -> None:
    forward = _broadband_metrics()
    reversed_metrics = dict(reversed(tuple(forward.items())))

    forward_description = describe_response_regime(_summary("ff", forward))
    reversed_description = describe_response_regime(_summary("ff", reversed_metrics))

    assert forward_description.structure_descriptor == "broadband_dominated"
    assert reversed_description.structure_descriptor == "broadband_dominated"
    assert forward_description == reversed_description


def test_mixed_requires_simultaneous_evidence_and_is_not_indeterminate_fallback() -> None:
    mixed = describe_response_regime(_summary("mf", _mixed_metrics()))
    indeterminate = describe_response_regime(_summary("mf", {
        "global_spectral_flatness": None,
        "global_spectral_entropy": None,
        "global_tonal_energy_fraction": None,
        "global_residual_energy_fraction": None,
    }))

    assert mixed.structure_descriptor == "mixed_line_and_continuum"
    assert evaluate_regime_descriptor(_summary("mf", _mixed_metrics()), "mixed_line_and_continuum").selected
    assert indeterminate.structure_descriptor == "indeterminate"


def test_dense_spacing_and_resolution_boundaries_are_explicit() -> None:
    settings = ResponseRegimeDescriptorSettings(maximum_median_peak_spacing_for_dense_hz=50.0)

    assert _criterion_result("global_median_peak_spacing_hz", 49.999999, "dense_spectrum", "dense_peak_spacing_low", settings=settings).passed is True
    assert _criterion_result("global_median_peak_spacing_hz", 50.0, "dense_spectrum", "dense_peak_spacing_low", settings=settings).passed is True
    assert _criterion_result("global_median_peak_spacing_hz", 50.000001, "dense_spectrum", "dense_peak_spacing_low", settings=settings).passed is False

    limited = describe_response_regime(_summary("f", {**_dense_tonal_metrics(), "global_median_peak_spacing_hz": 20.0}, resolution_hz=10.0))
    adequate = describe_response_regime(_summary("f", {**_dense_tonal_metrics(), "global_median_peak_spacing_hz": 20.000001}, resolution_hz=10.0))
    missing = describe_response_regime(_summary("f", _dense_tonal_metrics(), resolution_hz=10.0))
    missing_summary = _summary("f", {**_dense_tonal_metrics(), "global_median_peak_spacing_hz": None}, resolution_hz=10.0)

    assert "density_metrics_resolution_limited" in limited.limitations
    assert "density_metrics_resolution_limited" not in adequate.limitations
    assert "density_metrics_resolution_limited" not in describe_response_regime(missing_summary).limitations
    assert missing.valid


def test_clipping_near_clipping_and_rejection_boundaries_are_explicit() -> None:
    clean = describe_response_regime(_summary("ff", _broadband_metrics(), clipped_fraction=0.0, near_clipped_fraction=0.0))
    near = describe_response_regime(_summary("ff", _broadband_metrics(), clipped_fraction=0.0, near_clipped_fraction=1 / 3))
    clipped = describe_response_regime(_summary("ff", _broadband_metrics(), clipped_fraction=1 / 3, near_clipped_fraction=1 / 3))
    rejected = describe_response_regime(
        _summary("ff", _broadband_metrics(), clipped_fraction=1 / 3),
        ResponseRegimeDescriptorSettings(reject_clipped_conditions=True),
    )

    assert not any("clipping" in limitation for limitation in clean.limitations)
    assert "spectral_metrics_potentially_affected_by_near_clipping" in near.limitations
    assert near.confidence_descriptor == "low"
    assert "spectral_metrics_potentially_distorted_by_clipping" in clipped.limitations
    assert clipped.confidence_descriptor == "low"
    assert not rejected.valid
    assert rejected.failure_reason == "clipped_condition_rejected_by_configuration"


def test_missing_metric_combinations_affect_available_weight_and_confidence_without_zero_fill() -> None:
    one_missing = _summary("pp", {**_line_metrics(), "global_spectral_entropy": None})
    all_structure_missing = _summary("pp", {
        "global_spectral_flatness": None,
        "global_spectral_entropy": None,
        "global_tonal_energy_fraction": None,
        "global_residual_energy_fraction": None,
        "global_spectral_crest_factor": None,
        "global_peak_density_per_hz": None,
        "global_occupied_frequency_fraction": None,
    })
    one_favorable = _summary("pp", {"global_spectral_flatness": 0.1})

    assert evaluate_regime_descriptor(one_missing, "discrete_line_dominated").available_weight == pytest.approx(5.0)
    assert describe_response_regime(all_structure_missing).structure_descriptor == "indeterminate"
    score = evaluate_regime_descriptor(one_favorable, "discrete_line_dominated")
    assert score.support_score == pytest.approx(1.0)
    assert score.available_weight == pytest.approx(1.0)
    assert not score.selected
    assert score.indeterminate


def test_description_contract_requires_principal_descriptor_to_have_selected_score() -> None:
    description = describe_response_regime(_summary("pp", _line_metrics()))

    with pytest.raises(ValueError, match="selected structure descriptor"):
        ResponseRegimeDescription(
            dynamic_label=description.dynamic_label,
            structure_descriptor="broadband_dominated",
            temporal_evolution_descriptor=description.temporal_evolution_descriptor,
            line_identity_descriptor=description.line_identity_descriptor,
            confidence_descriptor=description.confidence_descriptor,
            descriptor_results=description.descriptor_results,
            supporting_metrics=description.supporting_metrics,
            conflicting_metrics=description.conflicting_metrics,
            unavailable_metrics=description.unavailable_metrics,
            limitations=description.limitations,
            valid=True,
            failure_reason=None,
            diagnostics=description.diagnostics,
        )


def test_reproducibility_with_reordered_weights_metrics_and_conditions() -> None:
    weights = (
        RegimeCriterionWeight("line_flatness_low", 2.0),
        RegimeCriterionWeight("line_entropy_low", 1.0),
    )
    reversed_weights = tuple(reversed(weights))
    metrics = _line_metrics()
    reversed_metrics = dict(reversed(tuple(metrics.items())))

    first = describe_response_regime(_summary("pp", metrics), ResponseRegimeDescriptorSettings(criterion_weights=weights))
    second = describe_response_regime(_summary("pp", reversed_metrics), ResponseRegimeDescriptorSettings(criterion_weights=reversed_weights))
    dynamic_first = describe_dynamic_response_regimes((
        _summary("ff", _broadband_metrics()),
        _summary("pp", metrics),
        _summary("mf", _mixed_metrics()),
    ))
    dynamic_second = describe_dynamic_response_regimes((
        _summary("mf", _mixed_metrics()),
        _summary("pp", metrics),
        _summary("ff", _broadband_metrics()),
    ))

    assert first == second
    assert dynamic_first == dynamic_second


def test_local_metric_perturbation_only_changes_dependent_criterion() -> None:
    below = describe_response_regime(_summary("pp", {**_line_metrics(), "global_spectral_flatness": 0.249999}))
    above = describe_response_regime(_summary("pp", {**_line_metrics(), "global_spectral_flatness": 0.250001}))
    below_line = _evaluation(below, "discrete_line_dominated").criteria
    above_line = _evaluation(above, "discrete_line_dominated").criteria

    changed = tuple(
        left.criterion_name
        for left, right in zip(below_line, above_line)
        if left.passed != right.passed
    )

    assert changed == ("line_flatness_low",)
    assert below.temporal_evolution_descriptor == above.temporal_evolution_descriptor
    assert below.line_identity_descriptor == above.line_identity_descriptor
