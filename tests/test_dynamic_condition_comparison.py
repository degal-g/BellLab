"""Tests for descriptive dynamic-condition comparison."""

from __future__ import annotations

from math import log10

import numpy as np
import pytest

from belllab import (
    DynamicConditionComparisonSettings,
    DynamicConditionRecordingAnalysis,
    ExcitationCharacterization,
    ExcitationCondition,
    GlobalSpectralCharacterizationSettings,
    MetricTolerance,
    Signal,
    SpectralBand,
    Spectrum,
    TimeResolvedSpectralCharacterizationSettings,
    aggregate_metric_values,
    characterize_global_spectrum,
    characterize_time_resolved_spectrum,
    compare_dynamic_condition_pair,
    compare_dynamic_conditions,
    evaluate_dynamic_metric_monotonicity,
    summarize_dynamic_condition,
)


def _condition(
    label: str,
    repeat: int = 0,
    *,
    gain: float = 1.0,
    microphone_id: str = "mic-a",
    interface_id: str = "if-a",
    distance_m: float = 0.30,
    channel: int = 0,
    amplitude_unit: str = "normalized",
) -> ExcitationCondition:
    return ExcitationCondition(
        dynamic_label=label,
        repeat_index=repeat,
        measured_rms=1.0,
        measured_energy=1.0,
        amplitude_unit=amplitude_unit,
        acquisition_gain=gain,
        session_id="session-a",
        microphone_id=microphone_id,
        interface_id=interface_id,
        channel=channel,
        microphone_distance_m=distance_m,
        microphone_orientation="front",
        impact_location="rim",
        exciter_type="mallet",
    )


def _excitation(
    recording_id: str,
    label: str,
    *,
    rms: float,
    energy: float | None = None,
    peak: float | None = None,
    gain: float = 1.0,
    microphone_id: str = "mic-a",
    interface_id: str = "if-a",
    distance_m: float = 0.30,
    clipped: bool = False,
    near_clipped: bool = False,
    amplitude_unit: str = "normalized",
) -> ExcitationCharacterization:
    peak = peak if peak is not None else max(rms * 2.0, rms)
    energy = energy if energy is not None else rms * rms
    crest = peak / rms if rms > 0 else None
    clipped_count = 1 if clipped else 0
    near_fraction = 0.1 if near_clipped or clipped else 0.0
    clipped_fraction = 0.1 if clipped else 0.0
    dbfs = 20.0 * log10(rms) if rms > 0 else None
    return ExcitationCharacterization(
        recording_id=recording_id,
        dynamic_label=label,
        session_id="session-a",
        impact_time_s=0.0,
        analysis_window_start_s=0.0,
        analysis_window_end_s=0.1,
        amplitude_unit=amplitude_unit,
        channel_index=0,
        power_dc_removed=False,
        sample_count=10,
        finite_sample_count=10,
        discarded_sample_count=0,
        peak_absolute_amplitude=peak,
        peak_signed_amplitude=peak,
        peak_time_s=0.0,
        positive_peak=peak,
        negative_peak=-peak,
        peak_asymmetry=0.0,
        mean_amplitude=0.0,
        dc_offset=0.0,
        dc_offset_to_peak=0.0,
        dc_offset_to_rms=0.0 if rms > 0 else None,
        rms_amplitude=rms,
        mean_square_amplitude=rms * rms,
        signal_energy=energy,
        equivalent_level_dbfs=dbfs,
        crest_factor=crest,
        crest_factor_db=20.0 * log10(crest) if crest else None,
        impulse_start_time_s=0.0,
        impulse_end_time_s=0.02,
        impulse_duration_s=0.02,
        attack_start_time_s=0.0,
        attack_duration_s=0.01,
        time_to_peak_s=0.0,
        clipping_detected=clipped,
        clipped_sample_count=clipped_count,
        clipped_sample_fraction=clipped_fraction,
        longest_clipped_run=clipped_count,
        near_clipping_detected=near_clipped or clipped,
        near_clipping_sample_fraction=near_fraction,
        background_sample_count=10,
        background_finite_sample_count=10,
        background_rms=0.01,
        background_energy=0.0001,
        background_peak=0.02,
        background_variability=0.01,
        background_clipping_detected=False,
        signal_to_background_ratio=10.0,
        signal_to_background_db=20.0,
        microphone_id=microphone_id,
        interface_id=interface_id,
        acquisition_gain=gain,
        microphone_distance_m=distance_m,
        microphone_orientation="front",
        impact_location="rim",
        exciter_type="mallet",
        exciter_mass_kg=0.01,
        valid=True,
        failure_reason=None,
        diagnostics=(),
    )


def _spectrum(power: tuple[float, ...], *, start_hz: float = 0.0) -> Spectrum:
    frequencies = tuple(start_hz + index * 100.0 for index in range(len(power)))
    return Spectrum(
        frequencies_hz=frequencies,
        magnitudes=power,
        magnitude_unit="linear power",
        window_name="rectangular",
        fft_size=max(2, 2 * (len(power) - 1)),
        sample_rate_hz=1000,
        original_size=max(2, 2 * (len(power) - 1)),
        bin_spacing_hz=100.0,
        normalization="test_power",
        interval_start_s=0.0,
        interval_end_s=1.0,
        remove_mean=False,
    )


def _global(
    recording_id: str,
    power: tuple[float, ...],
    *,
    bands: tuple[SpectralBand, ...] = (),
):
    settings = GlobalSpectralCharacterizationSettings(
        spectral_input_domain="linear_power",
        minimum_bin_count=3,
        frequency_min_hz=0.0,
        frequency_max_hz=400.0,
        peak_min_prominence=0.1,
        bands=bands,
    )
    return characterize_global_spectrum(_spectrum(power), settings, recording_id=recording_id)


def _signal(samples: np.ndarray, sample_rate: int = 2048) -> Signal:
    return Signal(
        samples=(tuple(float(value) for value in samples),),
        sample_rate=sample_rate,
        time=tuple(index / sample_rate for index in range(samples.size)),
        duration=samples.size / sample_rate,
        channels=1,
        unit="normalized",
    )


def _time_result(recording_id: str, samples: np.ndarray):
    settings = TimeResolvedSpectralCharacterizationSettings(
        analysis_window_end_s=1.0,
        frame_duration_s=0.125,
        hop_duration_s=0.125,
        fft_size=256,
        window_name="rectangular",
        detrend_policy="none",
        frequency_min_hz=16.0,
        frequency_max_hz=900.0,
        peak_min_prominence=1e-3,
        early_window_s=(0.0, 0.25),
        middle_window_s=(0.25, 0.50),
        late_window_s=(0.75, 1.0),
        bands=(
            SpectralBand("low", 16.0, 300.0),
            SpectralBand("high", 300.0, 900.0),
        ),
        change_point_thresholds=(("spectral_flatness", 0.05), ("tonal_energy_fraction", 0.05)),
    )
    return characterize_time_resolved_spectrum(_signal(samples), 0.0, settings, recording_id=recording_id)


def _analysis(
    label: str,
    repeat: int,
    *,
    rms: float,
    global_power: tuple[float, ...] | None = None,
    time_samples: np.ndarray | None = None,
    gain: float = 1.0,
    microphone_id: str = "mic-a",
    interface_id: str = "if-a",
    distance_m: float = 0.30,
    clipped: bool = False,
    bands: tuple[SpectralBand, ...] = (),
) -> DynamicConditionRecordingAnalysis:
    recording_id = f"{label}-{repeat}"
    return DynamicConditionRecordingAnalysis(
        recording_id=recording_id,
        condition=_condition(
            label,
            repeat,
            gain=gain,
            microphone_id=microphone_id,
            interface_id=interface_id,
            distance_m=distance_m,
        ),
        excitation=_excitation(
            recording_id,
            label,
            rms=rms,
            gain=gain,
            microphone_id=microphone_id,
            interface_id=interface_id,
            distance_m=distance_m,
            clipped=clipped,
        ),
        global_spectrum=_global(recording_id, global_power, bands=bands) if global_power is not None else None,
        time_resolved=_time_result(recording_id, time_samples) if time_samples is not None else None,
    )


def _metric(summary, name: str):
    return next(metric for metric in summary.all_metrics if metric.metric_name == name)


def _comparison(pair, name: str):
    return next(metric for metric in pair.metric_comparisons if metric.metric_name == name)


def _sequence(result, name: str):
    return next(sequence for sequence in result.metric_sequences if sequence.metric_name == name)


def _monotonicity(result, name: str):
    return next(item for item in result.monotonicity_results if item.metric_name == name)


def test_aggregated_metric_exact_statistics_and_missing_values() -> None:
    metric = aggregate_metric_values(
        "excitation_rms_amplitude",
        (1.0, None, 3.0, 5.0),
        unit="normalized",
    )

    assert metric.available_count == 4
    assert metric.finite_count == 3
    assert metric.discarded_count == 1
    assert metric.median == pytest.approx(3.0)
    assert metric.mean == pytest.approx(3.0)
    assert metric.standard_deviation == pytest.approx(np.std([1.0, 3.0, 5.0]))
    assert metric.minimum == pytest.approx(1.0)
    assert metric.maximum == pytest.approx(5.0)
    assert metric.range == pytest.approx(4.0)
    assert metric.median_absolute_deviation == pytest.approx(2.0)
    assert metric.coefficient_of_variation == pytest.approx(np.std([1.0, 3.0, 5.0]) / 3.0)


def test_aggregated_metric_without_finite_values_is_structured_failure() -> None:
    metric = aggregate_metric_values(
        "global_spectral_entropy",
        (None, None),
        unit="fraction",
    )

    assert not metric.valid
    assert metric.failure_reason == "no_finite_values"
    assert metric.median is None


def test_summarize_dynamic_condition_aggregates_repeats_by_median() -> None:
    analyses = (
        _analysis("pp", 0, rms=1.0),
        _analysis("pp", 1, rms=3.0),
        _analysis("pp", 2, rms=5.0),
    )

    summary = summarize_dynamic_condition(analyses)
    rms = _metric(summary, "excitation_rms_amplitude")

    assert summary.dynamic_label == "pp"
    assert summary.repeat_count == 3
    assert summary.valid_repeat_count == 3
    assert rms.median == pytest.approx(3.0)
    assert rms.mean == pytest.approx(3.0)
    assert rms.standard_deviation == pytest.approx(np.std([1.0, 3.0, 5.0]))
    assert "dynamic_condition_summary_is_descriptive_not_regime_classification" in summary.diagnostics


def test_summarize_dynamic_condition_requires_single_label() -> None:
    with pytest.raises(ValueError, match="same dynamic condition"):
        summarize_dynamic_condition((_analysis("pp", 0, rms=1.0), _analysis("p", 0, rms=2.0)))


def test_complete_dynamic_order_builds_adjacent_pairs_and_increasing_sequence() -> None:
    analyses = tuple(
        _analysis(label, 0, rms=float(index + 1))
        for index, label in enumerate(("pp", "p", "mf", "f", "ff"))
    )

    result = compare_dynamic_conditions(analyses, DynamicConditionComparisonSettings(enabled_metrics=("excitation_rms_amplitude",)))
    sequence = _sequence(result, "excitation_rms_amplitude")

    assert result.ordered_dynamic_labels == ("pp", "p", "mf", "f", "ff")
    assert result.missing_dynamic_labels == ()
    assert [pair.lower_dynamic_label + ">" + pair.higher_dynamic_label for pair in result.pairwise_comparisons] == [
        "pp>p",
        "p>mf",
        "mf>f",
        "f>ff",
    ]
    assert sequence.values == (1.0, 2.0, 3.0, 4.0, 5.0)
    assert sequence.monotonicity == "monotonically_increasing"
    assert _monotonicity(result, "excitation_rms_amplitude").inversion_count == 0


def test_dynamic_inversion_is_preserved_without_renaming_labels() -> None:
    analyses = (
        _analysis("pp", 0, rms=1.0),
        _analysis("p", 0, rms=4.0),
        _analysis("mf", 0, rms=2.0),
        _analysis("f", 0, rms=5.0),
        _analysis("ff", 0, rms=6.0),
    )

    result = compare_dynamic_conditions(analyses, DynamicConditionComparisonSettings(enabled_metrics=("excitation_rms_amplitude",)))
    p_to_mf = result.pairwise_comparisons[1]

    assert p_to_mf.lower_dynamic_label == "p"
    assert p_to_mf.higher_dynamic_label == "mf"
    assert _comparison(p_to_mf, "excitation_rms_amplitude").direction == "decrease"
    assert _sequence(result, "excitation_rms_amplitude").monotonicity == "non_monotonic"
    assert _sequence(result, "excitation_rms_amplitude").inversion_count == 1


def test_missing_conditions_are_preserved_and_available_pairs_record_jumps() -> None:
    analyses = (
        _analysis("pp", 0, rms=1.0),
        _analysis("mf", 0, rms=3.0),
        _analysis("ff", 0, rms=5.0),
    )

    result = compare_dynamic_conditions(analyses, DynamicConditionComparisonSettings(enabled_metrics=("excitation_rms_amplitude",)))

    assert result.ordered_dynamic_labels == ("pp", "mf", "ff")
    assert result.missing_dynamic_labels == ("p", "f")
    assert [pair.label_step_count for pair in result.pairwise_comparisons] == [2, 2]
    assert "missing_intermediate_dynamic_labels:p" in result.pairwise_comparisons[0].diagnostics
    assert _sequence(result, "excitation_rms_amplitude").valid_mask == (True, False, True, False, True)


def test_adjacent_only_policy_does_not_bridge_missing_conditions() -> None:
    analyses = (
        _analysis("pp", 0, rms=1.0),
        _analysis("mf", 0, rms=3.0),
        _analysis("ff", 0, rms=5.0),
    )

    result = compare_dynamic_conditions(
        analyses,
        DynamicConditionComparisonSettings(
            enabled_metrics=("excitation_rms_amplitude",),
            pair_comparison_policy="adjacent_only",
        ),
    )

    assert result.pairwise_comparisons == ()
    assert result.reference_comparisons


def test_single_condition_is_structured_global_failure() -> None:
    result = compare_dynamic_conditions(
        (_analysis("pp", 0, rms=1.0),),
        DynamicConditionComparisonSettings(enabled_metrics=("excitation_rms_amplitude",)),
    )

    assert not result.valid
    assert result.failure_reason == "insufficient_comparable_dynamic_conditions"
    assert result.pairwise_comparisons == ()


def test_instrumental_gain_mismatch_blocks_amplitude_but_not_fraction_metrics() -> None:
    settings = DynamicConditionComparisonSettings(
        enabled_metrics=("excitation_rms_amplitude", "global_spectral_flatness"),
    )
    lower = summarize_dynamic_condition((_analysis("pp", 0, rms=1.0, global_power=(10, 1, 1, 1, 1), gain=1.0),), settings)
    higher = summarize_dynamic_condition((_analysis("p", 0, rms=2.0, global_power=(2, 2, 2, 2, 2), gain=2.0),), settings)

    pair = compare_dynamic_condition_pair(lower, higher, settings)

    assert _comparison(pair, "excitation_rms_amplitude").direction == "not_comparable"
    assert _comparison(pair, "excitation_rms_amplitude").not_applicable_reason == "instrumental_acquisition_gain_mismatch"
    assert _comparison(pair, "global_spectral_flatness").comparable
    assert _comparison(pair, "global_spectral_flatness").direction == "increase"


def test_microphone_distance_and_unit_mismatch_are_specific_incompatibilities() -> None:
    settings = DynamicConditionComparisonSettings(enabled_metrics=("excitation_rms_amplitude",))
    lower = summarize_dynamic_condition((_analysis("pp", 0, rms=1.0, distance_m=0.3),), settings)
    higher_record = DynamicConditionRecordingAnalysis(
        recording_id="p-0",
        condition=_condition("p", 0, distance_m=0.5, amplitude_unit="Pa"),
        excitation=_excitation("p-0", "p", rms=2.0, distance_m=0.5, amplitude_unit="Pa"),
    )
    higher = summarize_dynamic_condition((higher_record,), settings)

    pair = compare_dynamic_condition_pair(lower, higher, settings)

    assert "instrumental_microphone_distance_m_mismatch" in pair.incompatibilities
    assert "instrumental_amplitude_unit_mismatch" in pair.incompatibilities
    assert not _comparison(pair, "excitation_rms_amplitude").comparable


def test_clipped_condition_is_preserved_and_flags_amplitude_and_spectral_metrics() -> None:
    settings = DynamicConditionComparisonSettings(
        enabled_metrics=("excitation_rms_amplitude", "global_spectral_entropy"),
    )
    lower = summarize_dynamic_condition((_analysis("pp", 0, rms=1.0, global_power=(10, 1, 1, 1, 1)),), settings)
    higher = summarize_dynamic_condition((_analysis("ff", 0, rms=5.0, global_power=(2, 2, 2, 2, 2), clipped=True),), settings)

    pair = compare_dynamic_condition_pair(lower, higher, settings)

    assert higher.valid
    assert higher.clipped_repeat_fraction == pytest.approx(1.0)
    assert "spectral_metrics_potentially_distorted_by_clipping" in higher.diagnostics
    assert _comparison(pair, "excitation_rms_amplitude").not_applicable_reason == "amplitude_metrics_incompatible_with_clipping"
    assert "spectral_metrics_potentially_distorted_by_clipping" in _comparison(pair, "global_spectral_entropy").diagnostics


def test_clipped_condition_can_be_excluded_by_configuration() -> None:
    settings = DynamicConditionComparisonSettings(
        enabled_metrics=("excitation_rms_amplitude",),
        exclude_clipped_conditions=True,
    )
    summary = summarize_dynamic_condition((_analysis("ff", 0, rms=5.0, clipped=True),), settings)

    assert not summary.valid
    assert summary.valid_repeat_count == 0
    assert "clipped_repeats_excluded_by_configuration" in summary.diagnostics


def test_global_spectral_metrics_compare_without_regime_classification() -> None:
    settings = DynamicConditionComparisonSettings(
        enabled_metrics=(
            "global_spectral_centroid_hz",
            "global_spectral_flatness",
            "global_spectral_entropy",
            "global_tonal_energy_fraction",
            "global_occupied_bandwidth_hz",
        )
    )
    analyses = (
        _analysis("pp", 0, rms=1.0, global_power=(10, 1, 1, 1, 1)),
        _analysis("ff", 0, rms=2.0, global_power=(1, 1, 1, 1, 10)),
    )

    result = compare_dynamic_conditions(analyses, settings)
    pair = result.pairwise_comparisons[0]

    assert _comparison(pair, "global_spectral_centroid_hz").direction == "increase"
    assert _comparison(pair, "global_spectral_entropy").comparable
    assert "dynamic_condition_comparison_is_descriptive_not_regime_classification" in result.diagnostics


def test_pp_reference_comparisons_are_exposed_for_all_higher_conditions() -> None:
    analyses = tuple(_analysis(label, 0, rms=float(index + 1)) for index, label in enumerate(("pp", "p", "mf", "f", "ff")))

    result = compare_dynamic_conditions(analyses, DynamicConditionComparisonSettings(enabled_metrics=("excitation_rms_amplitude",)))

    assert result.reference_dynamic_label == "pp"
    assert [pair.higher_dynamic_label for pair in result.reference_comparisons] == ["p", "mf", "f", "ff"]
    assert _comparison(result.reference_comparisons[-1], "excitation_rms_amplitude").ratio == pytest.approx(5.0)


def test_reference_falls_back_to_lowest_available_only_when_configured() -> None:
    analyses = (
        _analysis("mf", 0, rms=3.0),
        _analysis("ff", 0, rms=5.0),
    )

    fallback = compare_dynamic_conditions(
        analyses,
        DynamicConditionComparisonSettings(enabled_metrics=("excitation_rms_amplitude",)),
    )
    strict = compare_dynamic_conditions(
        analyses,
        DynamicConditionComparisonSettings(
            enabled_metrics=("excitation_rms_amplitude",),
            reference_policy="configured",
        ),
    )

    assert fallback.reference_dynamic_label == "mf"
    assert "reference_dynamic_label_fallback:mf" in fallback.diagnostics
    assert strict.reference_dynamic_label is None
    assert "reference_dynamic_label_missing:pp" in strict.diagnostics


def test_change_to_within_condition_variability_ratio_is_descriptive() -> None:
    settings = DynamicConditionComparisonSettings(enabled_metrics=("excitation_rms_amplitude",))
    lower = summarize_dynamic_condition((
        _analysis("pp", 0, rms=1.0),
        _analysis("pp", 1, rms=1.1),
        _analysis("pp", 2, rms=0.9),
    ), settings)
    higher = summarize_dynamic_condition((
        _analysis("p", 0, rms=4.0),
        _analysis("p", 1, rms=4.1),
        _analysis("p", 2, rms=3.9),
    ), settings)

    comparison = _comparison(compare_dynamic_condition_pair(lower, higher, settings), "excitation_rms_amplitude")

    assert comparison.change_to_within_condition_variability_ratio is not None
    assert comparison.change_to_within_condition_variability_ratio > 20.0
    assert "p_value" not in comparison.diagnostics


def test_unit_policies_for_db_fraction_count_and_slope_metrics() -> None:
    settings = DynamicConditionComparisonSettings(
        enabled_metrics=(
            "excitation_equivalent_level_dbfs",
            "global_tonal_energy_fraction",
            "global_significant_peak_count",
            "time_spectral_flatness_slope_per_s",
        )
    )
    lower = summarize_dynamic_condition((_analysis("pp", 0, rms=0.25, global_power=(1, 10, 1, 1, 1)),), settings)
    higher = summarize_dynamic_condition((_analysis("p", 0, rms=0.50, global_power=(1, 10, 1, 10, 1)),), settings)
    pair = compare_dynamic_condition_pair(lower, higher, settings)

    db_metric = _comparison(pair, "excitation_equivalent_level_dbfs")
    fraction_metric = _comparison(pair, "global_tonal_energy_fraction")
    count_metric = _comparison(pair, "global_significant_peak_count")
    slope_metric = _comparison(pair, "time_spectral_flatness_slope_per_s")

    assert db_metric.absolute_change == pytest.approx(20.0 * log10(0.50 / 0.25))
    assert db_metric.ratio is None
    assert fraction_metric.change_db is None
    assert count_metric.ratio is not None
    assert not slope_metric.comparable
    assert slope_metric.not_applicable_reason == "representative_value_unavailable"


def test_tolerances_mark_approximately_equal_changes() -> None:
    settings = DynamicConditionComparisonSettings(
        enabled_metrics=("excitation_rms_amplitude",),
        metric_tolerances=(MetricTolerance("excitation_rms_amplitude", absolute_tolerance=0.2),),
    )
    lower = summarize_dynamic_condition((_analysis("pp", 0, rms=1.0),), settings)
    higher = summarize_dynamic_condition((_analysis("p", 0, rms=1.1),), settings)

    comparison = _comparison(compare_dynamic_condition_pair(lower, higher, settings), "excitation_rms_amplitude")

    assert comparison.direction == "approximately_equal"


def test_invalid_settings_reject_unknown_metrics_and_bad_reference() -> None:
    with pytest.raises(ValueError, match="unknown metric"):
        DynamicConditionComparisonSettings(enabled_metrics=("unknown_metric",))
    with pytest.raises(ValueError, match="reference_dynamic_label"):
        DynamicConditionComparisonSettings(reference_dynamic_label="mp")


def test_metric_monotonicity_handles_constant_insufficient_and_decreasing() -> None:
    settings = DynamicConditionComparisonSettings()

    constant = evaluate_dynamic_metric_monotonicity(
        "global_spectral_flatness",
        "fraction",
        ("pp", "p", "mf", "f", "ff"),
        (0.5, 0.5, 0.5, 0.5, 0.5),
        settings=settings,
    )
    insufficient = evaluate_dynamic_metric_monotonicity(
        "global_spectral_flatness",
        "fraction",
        ("pp", "p", "mf", "f", "ff"),
        (None, None, 0.5, None, None),
        settings=settings,
    )
    decreasing = evaluate_dynamic_metric_monotonicity(
        "global_spectral_flatness",
        "fraction",
        ("pp", "p", "mf", "f", "ff"),
        (0.9, 0.7, 0.5, 0.3, 0.1),
        settings=settings,
    )

    assert constant.monotonicity == "constant"
    assert insufficient.monotonicity == "insufficient"
    assert decreasing.monotonicity == "monotonically_decreasing"


def test_band_fraction_comparison_survives_gain_mismatch_but_absolute_energy_does_not() -> None:
    bands = (SpectralBand("low", 0.0, 200.0), SpectralBand("high", 200.0, 500.0))
    settings = DynamicConditionComparisonSettings(
        enabled_metrics=("global_band_high_energy", "global_band_high_energy_fraction"),
    )
    lower = summarize_dynamic_condition((_analysis("pp", 0, rms=1.0, global_power=(10, 10, 1, 1, 1), bands=bands),), settings)
    higher = summarize_dynamic_condition((_analysis("ff", 0, rms=2.0, global_power=(1, 1, 10, 10, 10), gain=2.0, bands=bands),), settings)

    pair = compare_dynamic_condition_pair(lower, higher, settings)

    assert not _comparison(pair, "global_band_high_energy").comparable
    assert _comparison(pair, "global_band_high_energy_fraction").comparable
    assert _comparison(pair, "global_band_high_energy_fraction").direction == "increase"


def test_band_definition_mismatch_blocks_band_comparison() -> None:
    lower_bands = (SpectralBand("high", 200.0, 500.0),)
    higher_bands = (SpectralBand("high", 300.0, 500.0),)
    settings = DynamicConditionComparisonSettings(enabled_metrics=("global_band_high_energy_fraction",))
    lower = summarize_dynamic_condition((_analysis("pp", 0, rms=1.0, global_power=(10, 10, 1, 1, 1), bands=lower_bands),), settings)
    higher = summarize_dynamic_condition((_analysis("ff", 0, rms=2.0, global_power=(1, 1, 10, 10, 10), bands=higher_bands),), settings)

    pair = compare_dynamic_condition_pair(lower, higher, settings)

    assert "band_definition_mismatch" in pair.incompatibilities
    assert _comparison(pair, "global_band_high_energy_fraction").not_applicable_reason == "band_definition_mismatch"


def test_time_region_metrics_describe_ff_broad_attack_and_tonal_tail() -> None:
    sample_rate = 2048
    frame = np.arange(256) / sample_rate
    rng = np.random.default_rng(1234)
    pp_samples = np.tile(np.sin(2.0 * np.pi * 128.0 * frame), 8)
    ff_early = rng.normal(0.0, 0.6, size=512)
    ff_tail_time = np.arange(1536) / sample_rate
    ff_tail = 0.7 * np.sin(2.0 * np.pi * 128.0 * ff_tail_time)
    ff_samples = np.concatenate([ff_early, ff_tail])
    settings = DynamicConditionComparisonSettings(
        enabled_metrics=(
            "early_spectral_flatness",
            "late_spectral_flatness",
            "early_tonal_energy_fraction",
            "late_tonal_energy_fraction",
            "region_late_minus_early_tonal_energy_fraction",
        )
    )

    result = compare_dynamic_conditions((
        _analysis("pp", 0, rms=1.0, time_samples=pp_samples),
        _analysis("ff", 0, rms=2.0, time_samples=ff_samples),
    ), settings)
    ff_summary = result.condition_summaries[-1]
    pair = result.pairwise_comparisons[0]

    assert _metric(ff_summary, "early_spectral_flatness").median > _metric(ff_summary, "late_spectral_flatness").median
    assert _metric(ff_summary, "region_late_minus_early_tonal_energy_fraction").median > 0.0
    assert _comparison(pair, "early_spectral_flatness").direction == "increase"


def test_time_change_points_are_aggregated_descriptively() -> None:
    sample_rate = 2048
    rng = np.random.default_rng(7)
    early = rng.normal(0.0, 0.5, size=512)
    tail_t = np.arange(1536) / sample_rate
    tail = 0.6 * np.sin(2.0 * np.pi * 160.0 * tail_t)
    summary = summarize_dynamic_condition((
        _analysis("ff", 0, rms=2.0, time_samples=np.concatenate([early, tail])),
    ))

    assert _metric(summary, "time_change_point_count").median >= 1.0
    assert any(metric.metric_name.startswith("time_first_change_point_") for metric in summary.time_resolved_metrics)


def test_time_band_persistence_metrics_are_exposed() -> None:
    sample_rate = 2048
    t = np.arange(2048) / sample_rate
    samples = np.sin(2.0 * np.pi * 128.0 * t) + np.exp(-6.0 * t) * np.sin(2.0 * np.pi * 640.0 * t)
    summary = summarize_dynamic_condition((_analysis("mf", 0, rms=1.0, time_samples=samples),))

    assert _metric(summary, "time_band_low_coverage_fraction").median is not None
    assert _metric(summary, "time_band_high_energy_fraction_slope_per_s").median is not None


def test_temporal_configuration_mismatch_blocks_region_metric_but_not_global_fraction() -> None:
    sample_rate = 2048
    t = np.arange(2048) / sample_rate
    samples = np.sin(2.0 * np.pi * 128.0 * t)
    lower_record = _analysis("pp", 0, rms=1.0, global_power=(10, 1, 1, 1, 1), time_samples=samples)
    higher_time = characterize_time_resolved_spectrum(
        _signal(samples),
        0.0,
        TimeResolvedSpectralCharacterizationSettings(
            analysis_window_end_s=1.0,
            frame_duration_s=0.25,
            hop_duration_s=0.25,
            fft_size=512,
            window_name="rectangular",
            detrend_policy="none",
            frequency_min_hz=16.0,
            frequency_max_hz=900.0,
            peak_min_prominence=1e-3,
            early_window_s=(0.0, 0.25),
            middle_window_s=(0.25, 0.50),
            late_window_s=(0.75, 1.0),
        ),
        recording_id="p-0",
    )
    higher_record = DynamicConditionRecordingAnalysis(
        recording_id="p-0",
        condition=_condition("p", 0),
        excitation=_excitation("p-0", "p", rms=2.0),
        global_spectrum=_global("p-0", (10, 1, 1, 1, 1)),
        time_resolved=higher_time,
    )
    settings = DynamicConditionComparisonSettings(enabled_metrics=("early_spectral_flatness", "global_spectral_flatness"))
    lower = summarize_dynamic_condition((lower_record,), settings)
    higher = summarize_dynamic_condition((higher_record,), settings)

    pair = compare_dynamic_condition_pair(lower, higher, settings)

    assert _comparison(pair, "early_spectral_flatness").not_applicable_reason == "time_resolved_configuration_mismatch"
    assert _comparison(pair, "global_spectral_flatness").comparable


def test_determinism_with_shuffled_conditions_and_repeats() -> None:
    analyses = (
        _analysis("mf", 1, rms=3.1),
        _analysis("pp", 1, rms=1.1),
        _analysis("ff", 0, rms=5.0),
        _analysis("pp", 0, rms=0.9),
        _analysis("mf", 0, rms=2.9),
    )
    settings = DynamicConditionComparisonSettings(enabled_metrics=("excitation_rms_amplitude",))

    first = compare_dynamic_conditions(analyses, settings)
    second = compare_dynamic_conditions(tuple(reversed(analyses)), settings)

    assert first == second


def test_public_result_confirms_no_forbidden_operations() -> None:
    result = compare_dynamic_conditions((
        _analysis("pp", 0, rms=1.0),
        _analysis("ff", 0, rms=2.0),
    ), DynamicConditionComparisonSettings(enabled_metrics=("excitation_rms_amplitude",)))

    assert "no_cross_condition_candidate_association_was_performed" in result.diagnostics
    assert "no_modal_mode_conversion_was_performed" in result.diagnostics
    assert all("regime" not in comparison.direction for pair in result.pairwise_comparisons for comparison in pair.metric_comparisons)
