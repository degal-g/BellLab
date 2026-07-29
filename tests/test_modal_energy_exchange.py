"""Operational evidence for possible modal energy redistribution."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from math import cos, exp, pi, sqrt
import random

import pytest

from belllab import (
    ModalAmplitudeRepresentation,
    ModalEnergyExchangeReason,
    ModalEnergyExchangeSettings,
    ModalEnergyExchangeStatus,
    ModalEnergyProxy,
    ModalEnvelopeCorrelationMethod,
    ModalEnvelopeResamplingPolicy,
    ModalEnvelopeSignificanceMethod,
    align_modal_envelope_series,
    evaluate_modal_alternating_dominance,
    evaluate_modal_amplitude_recovery,
    evaluate_modal_beating_context,
    evaluate_modal_delayed_growth,
    evaluate_modal_energy_exchange,
    evaluate_modal_energy_exchange_pair,
    evaluate_modal_envelope_correlation,
    evaluate_modal_pair_energy,
    prepare_modal_envelope_series,
    summarize_modal_energy_exchange,
)


def _settings(**kwargs) -> ModalEnergyExchangeSettings:
    values = {"significance_method": ModalEnvelopeSignificanceMethod.DISABLED}
    values.update(kwargs)
    return ModalEnergyExchangeSettings(**values)


def _times(count: int, step: float = 0.1) -> tuple[float, ...]:
    return tuple(index * step for index in range(count))


def _series(
    source_id: str,
    amplitudes: tuple[float, ...],
    *,
    times: tuple[float, ...] | None = None,
    settings: ModalEnergyExchangeSettings | None = None,
    diagnostics: tuple[str, ...] = (),
):
    return prepare_modal_envelope_series(
        settings=settings or _settings(),
        times_s=times or _times(len(amplitudes)),
        amplitudes=amplitudes,
        source_id=source_id,
        diagnostics=diagnostics,
    )


def _exchange_pair(
    settings: ModalEnergyExchangeSettings | None = None,
):
    cfg = settings or _settings(
        normalize_envelopes=False,
        energy_proxy=ModalEnergyProxy.AMPLITUDE_SQUARED,
    )
    times = _times(21, 0.05)
    a = tuple(1.0 - 0.6 * time for time in times)
    b = tuple(sqrt(max(0.0, 1.0 - value * value)) for value in a)
    return (
        _series("A", a, times=times, settings=cfg),
        _series("B", b, times=times, settings=cfg),
        cfg,
    )


def _normalized_result(result):
    return tuple(
        (
            item.evidence_id,
            item.source_a_id,
            item.source_b_id,
            item.status,
            round(item.score.normalized_score, 12),
            item.supporting_reasons,
            item.reservation_reasons,
            item.inconclusive_reasons,
            item.not_supported_reasons,
        )
        for item in result.pair_evidences
    )


def test_synthetic_apparent_exchange_is_supported_with_multiple_evidence_channels() -> None:
    source_a, source_b, settings = _exchange_pair()
    evidence = evaluate_modal_energy_exchange_pair(source_a, source_b, settings)

    assert evidence.status is ModalEnergyExchangeStatus.SUPPORTED
    assert evidence.valid is True
    assert evidence.trend_evidence.opposed_trends is True
    assert evidence.correlation_evidence.zero_lag_correlation < -0.9
    assert evidence.correlation_evidence.significant_negative_correlation is True
    assert evidence.delayed_growth_evidence[1].supported is True
    assert evidence.pair_energy_evidence.approximately_conserved is True
    assert evidence.score.passes_support_threshold is True
    assert evidence.score.normalized_score >= settings.minimum_support_score
    assert ModalEnergyExchangeReason.OPPOSED_ENVELOPE_TRENDS in evidence.supporting_reasons
    assert ModalEnergyExchangeReason.DELAYED_GROWTH in evidence.supporting_reasons
    assert ModalEnergyExchangeReason.APPROXIMATELY_CONSERVED_PAIR_ENERGY in evidence.supporting_reasons
    assert "no_physical_transfer_or_causality_inferred" in evidence.diagnostics


def test_two_independent_decay_envelopes_are_not_supported() -> None:
    settings = _settings()
    times = _times(21, 0.05)
    source_a = _series("A", tuple(exp(-time / 0.8) for time in times), times=times)
    source_b = _series("B", tuple(0.8 * exp(-time / 1.2) for time in times), times=times)

    evidence = evaluate_modal_energy_exchange_pair(source_a, source_b, settings)

    assert evidence.status is ModalEnergyExchangeStatus.NOT_SUPPORTED
    assert ModalEnergyExchangeReason.SAME_DIRECTION_TRENDS in evidence.not_supported_reasons
    assert ModalEnergyExchangeReason.DECAY_ONLY_BEHAVIOR in evidence.not_supported_reasons
    assert ModalEnergyExchangeReason.NO_DELAYED_GROWTH in evidence.not_supported_reasons


def test_possible_beating_is_recorded_only_as_reservation_context() -> None:
    settings = _settings(
        maximum_frequency_separation_for_beating_hz=2.0,
        minimum_beating_cycles=2.0,
        beating_period_tolerance_fraction=0.15,
    )
    times = _times(81, 0.05)
    source_a = _series(
        "A",
        tuple(1.0 + 0.2 * cos(2 * pi * time) for time in times),
        times=times,
        settings=settings,
    )
    source_b = _series(
        "B",
        tuple(1.0 - 0.2 * cos(2 * pi * time) for time in times),
        times=times,
        settings=settings,
    )
    alignment = align_modal_envelope_series(source_a, source_b, settings)

    beating = evaluate_modal_beating_context(100.0, 101.0, alignment, settings)

    assert beating.frequency_separation_hz == pytest.approx(1.0)
    assert beating.expected_beating_period_s == pytest.approx(1.0)
    assert beating.observed_modulation_period_s == pytest.approx(1.0, abs=0.05)
    assert beating.possible_beating is True
    assert ModalEnergyExchangeReason.POSSIBLE_BEATING in beating.reasons


def test_lagged_negative_correlation_uses_documented_direction_convention() -> None:
    settings = _settings(maximum_lag_s=0.5, lag_step_s=0.1)
    times = _times(31, 0.1)
    source_a = _series(
        "A",
        tuple(1.0 / (1.0 + exp(6.0 * (time - 1.0))) for time in times),
        times=times,
        settings=settings,
    )
    source_b = _series(
        "B",
        tuple(1.0 / (1.0 + exp(-6.0 * (time - 1.2))) for time in times),
        times=times,
        settings=settings,
    )
    alignment = align_modal_envelope_series(source_a, source_b, settings)

    correlation = evaluate_modal_envelope_correlation(alignment, settings)

    assert correlation.best_negative_lag_s == pytest.approx(0.2, abs=0.11)
    assert correlation.best_negative_correlation < -0.9
    assert ModalEnergyExchangeReason.LAGGED_NEGATIVE_CORRELATION in correlation.reasons


@pytest.mark.parametrize(
    ("values", "threshold", "duration", "supported"),
    [
        ((0.2, 0.2, 0.2, 0.22, 0.4, 0.5, 0.52), 0.15, 0.1, True),
        ((0.2, 0.2, 0.2, 0.2, 0.6, 0.2, 0.2), 0.15, 0.2, False),
        ((0.2, 0.2, 0.2, 0.23, 0.23, 0.23), 0.15, 0.0, True),
        ((0.2, 0.2, 0.2, 0.229, 0.229, 0.229), 0.15, 0.0, False),
        ((0.2, 0.2, 0.2, 0.231, 0.231, 0.231), 0.15, 0.0, True),
    ],
    ids=["sustained", "single_spike", "at_threshold", "below", "above"],
)
def test_delayed_growth_thresholds_are_inclusive_and_noise_aware(
    values, threshold, duration, supported
) -> None:
    settings = _settings(
        minimum_delayed_growth_fraction=threshold,
        growth_minimum_duration_s=duration,
        minimum_overlap_sample_count=5,
    )
    evidence = evaluate_modal_delayed_growth(
        _series("G", values, settings=settings),
        settings,
    )

    assert evidence.supported is supported
    if supported:
        assert ModalEnergyExchangeReason.DELAYED_GROWTH in evidence.reasons
    else:
        assert ModalEnergyExchangeReason.NO_DELAYED_GROWTH in evidence.reasons


@pytest.mark.parametrize(
    ("values", "supported"),
    [
        ((1.0, 0.8, 0.4, 0.5, 0.7, 0.6), True),
        ((1.0, 0.8, 0.4, 0.42, 0.43, 0.42), False),
        ((1.0, 0.8, 0.4, 0.9, 0.45, 0.42), True),
    ],
    ids=["recovery", "too_small", "brief_recovery"],
)
def test_amplitude_recovery_is_distinct_from_delayed_growth(values, supported) -> None:
    settings = _settings(minimum_recovery_fraction=0.25)
    evidence = evaluate_modal_amplitude_recovery(
        _series("R", values, settings=settings),
        settings,
    )

    assert evidence.initial_decay_detected is True
    assert evidence.supported is supported
    if supported:
        assert ModalEnergyExchangeReason.LATE_AMPLITUDE_RECOVERY in evidence.reasons


@pytest.mark.parametrize(
    ("pair_energy", "expected"),
    [
        ((1.0, 1.0, 1.0, 1.0, 1.0), True),
        ((1.0, 1.1, 1.2, 1.3, 1.4), False),
        ((1.4, 1.3, 1.2, 1.1, 1.0), False),
        ((0.5, 1.5, 0.5, 1.5, 1.0), False),
        ((0.9, 1.0, 1.1, 1.0, 1.0), True),
    ],
    ids=["constant", "growing", "decreasing", "oscillating", "inclusive_limit"],
)
def test_pair_energy_proxy_reports_stability_without_physical_conservation(
    pair_energy, expected
) -> None:
    settings = _settings(
        normalize_envelopes=False,
        energy_proxy=ModalEnergyProxy.AMPLITUDE_SQUARED,
        pair_energy_variation_limit=0.2,
    )
    amplitudes = tuple(sqrt(value / 2.0) for value in pair_energy)
    alignment = align_modal_envelope_series(
        _series("A", amplitudes, settings=settings),
        _series("B", amplitudes, settings=settings),
        settings,
    )

    evidence = evaluate_modal_pair_energy(alignment, settings)

    assert evidence.approximately_conserved is expected
    assert evidence.pair_energy_relative_range == pytest.approx(
        (max(pair_energy) - min(pair_energy)) / (sum(pair_energy) / len(pair_energy))
    )


def test_alternating_dominance_uses_hysteresis() -> None:
    settings = _settings(
        normalize_envelopes=False,
        energy_proxy=ModalEnergyProxy.AMPLITUDE_SQUARED,
        dominance_hysteresis_ratio=1.2,
        minimum_alternating_dominance_count=2,
    )
    alignment = align_modal_envelope_series(
        _series("A", (2.0, 2.0, 0.5, 0.5, 2.0, 2.0), settings=settings),
        _series("B", (0.5, 0.5, 2.0, 2.0, 0.5, 0.5), settings=settings),
        settings,
    )
    near_equal = align_modal_envelope_series(
        _series("C", (1.0, 1.05, 0.96, 1.04, 0.98, 1.0), settings=settings),
        _series("D", (1.0,) * 6, settings=settings),
        settings,
    )

    alternating = evaluate_modal_alternating_dominance(alignment, settings)
    noisy = evaluate_modal_alternating_dominance(near_equal, settings)

    assert alternating.alternating_dominance is True
    assert alternating.dominance_change_count == 2
    assert noisy.alternating_dominance is False


@pytest.mark.parametrize(
    ("times_a", "times_b", "policy", "valid", "reason"),
    [
        (_times(6), _times(6), ModalEnvelopeResamplingPolicy.REQUIRE_IDENTICAL, True, ModalEnergyExchangeReason.TEMPORAL_OVERLAP_SUFFICIENT),
        (_times(6), tuple(time + 0.01 for time in _times(6)), ModalEnvelopeResamplingPolicy.REQUIRE_IDENTICAL, False, ModalEnergyExchangeReason.INCOMPATIBLE_TIME_AXES),
        (_times(6), tuple(time + 0.05 for time in _times(6)), ModalEnvelopeResamplingPolicy.LINEAR_INTERPOLATION, True, ModalEnergyExchangeReason.INTERPOLATION_REQUIRED),
        (_times(6), tuple(2.0 + time for time in _times(6)), ModalEnvelopeResamplingPolicy.LINEAR_INTERPOLATION, False, ModalEnergyExchangeReason.INSUFFICIENT_TEMPORAL_OVERLAP),
    ],
    ids=["identical", "different_rejected", "interpolated", "no_overlap"],
)
def test_temporal_alignment_policies(times_a, times_b, policy, valid, reason) -> None:
    settings = _settings(resampling_policy=policy, resampling_step_s=0.1)
    alignment = align_modal_envelope_series(
        _series("A", (1.0, 0.9, 0.8, 0.7, 0.6, 0.5), times=times_a, settings=settings),
        _series("B", (0.5, 0.6, 0.7, 0.8, 0.9, 1.0), times=times_b, settings=settings),
        settings,
    )

    assert alignment.valid is valid
    assert reason in alignment.reasons
    if policy is ModalEnvelopeResamplingPolicy.LINEAR_INTERPOLATION and valid:
        assert alignment.resampling_applied is True


@pytest.mark.parametrize(
    ("times", "amplitudes", "reason"),
    [
        ((0.0, 0.1, 0.1), (1.0, 0.8, 0.6), ModalEnergyExchangeReason.INVALID_TIME_VALUES),
        ((0.2, 0.1, 0.0), (1.0, 0.8, 0.6), ModalEnergyExchangeReason.INVALID_TIME_VALUES),
        ((0.0, float("nan"), 0.2), (1.0, 0.8, 0.6), ModalEnergyExchangeReason.INVALID_TIME_VALUES),
        ((0.0, float("inf"), 0.2), (1.0, 0.8, 0.6), ModalEnergyExchangeReason.INVALID_TIME_VALUES),
        ((0.0, 0.1, 0.2), (1.0, float("nan"), 0.6), ModalEnergyExchangeReason.INVALID_AMPLITUDE_VALUES),
        ((0.0, 0.1, 0.2), (1.0, float("inf"), 0.6), ModalEnergyExchangeReason.INVALID_AMPLITUDE_VALUES),
        ((0.0, 0.1, 0.2), (1.0, -0.1, 0.6), ModalEnergyExchangeReason.INVALID_AMPLITUDE_VALUES),
        ((0.0, 0.1), (1.0,), ModalEnergyExchangeReason.MISSING_ENVELOPE),
        ((), (), ModalEnergyExchangeReason.INSUFFICIENT_ENVELOPE_SAMPLES),
    ],
    ids=[
        "repeated_times",
        "inverted_times",
        "nan_time",
        "inf_time",
        "nan_amplitude",
        "inf_amplitude",
        "negative_linear_amplitude",
        "length_mismatch",
        "empty_series",
    ],
)
def test_invalid_time_and_amplitude_inputs_fail_explicitly(times, amplitudes, reason) -> None:
    series = prepare_modal_envelope_series(
        times_s=times,
        amplitudes=amplitudes,
        source_id="bad",
        settings=_settings(),
    )

    assert series.valid is False
    assert reason in series.reasons


def test_db_amplitude_representation_and_zero_db_are_valid() -> None:
    settings = _settings(amplitude_representation=ModalAmplitudeRepresentation.AMPLITUDE_DB)
    series = prepare_modal_envelope_series(
        times_s=_times(5),
        amplitudes=(0.0, -3.0, -6.0, -9.0, -12.0),
        source_id="db",
        settings=settings,
    )

    assert series.valid is True
    assert all(value >= 0 for value in series.energy_proxy)


@pytest.mark.parametrize(
    ("method", "values_a", "values_b", "expected_sign"),
    [
        (ModalEnvelopeCorrelationMethod.PEARSON, (1.0, 0.8, 0.6, 0.4, 0.2), (0.2, 0.4, 0.6, 0.8, 1.0), -1),
        (ModalEnvelopeCorrelationMethod.SPEARMAN, (1.0, 0.8, 0.6, 0.4, 0.2), (0.2, 0.4, 0.6, 0.8, 1.0), -1),
        (ModalEnvelopeCorrelationMethod.PEARSON, (0.2, 0.4, 0.6, 0.8, 1.0), (0.2, 0.4, 0.6, 0.8, 1.0), 1),
    ],
    ids=["pearson_negative", "spearman_negative", "positive"],
)
def test_correlation_methods_report_sign_and_bounds(
    method, values_a, values_b, expected_sign
) -> None:
    settings = _settings(correlation_method=method)
    alignment = align_modal_envelope_series(
        _series("A", values_a, settings=settings),
        _series("B", values_b, settings=settings),
        settings,
    )

    evidence = evaluate_modal_envelope_correlation(alignment, settings)

    assert -1.0 <= evidence.zero_lag_correlation <= 1.0
    assert evidence.zero_lag_correlation * expected_sign > 0


def test_constant_and_short_correlation_inputs_are_insufficient() -> None:
    constant_settings = _settings(minimum_dynamic_range_fraction=0.01)
    constant = align_modal_envelope_series(
        _series("A", (1.0,) * 5, settings=constant_settings),
        _series("B", (0.5,) * 5, settings=constant_settings),
        constant_settings,
    )
    short_settings = _settings(minimum_overlap_sample_count=2)
    short = align_modal_envelope_series(
        _series("C", (1.0, 0.5), settings=short_settings),
        _series("D", (0.5, 1.0), settings=short_settings),
        short_settings,
    )

    assert ModalEnergyExchangeReason.INSUFFICIENT_DYNAMIC_RANGE in evaluate_modal_envelope_correlation(constant, constant_settings).reasons
    assert ModalEnergyExchangeReason.INSUFFICIENT_ENVELOPE_SAMPLES in evaluate_modal_envelope_correlation(short, short_settings).reasons


def test_circular_shift_significance_is_deterministic_with_seed() -> None:
    settings = ModalEnergyExchangeSettings(
        significance_method=ModalEnvelopeSignificanceMethod.CIRCULAR_SHIFT,
        permutation_count=40,
        random_seed=7,
    )
    other_seed = replace(settings, random_seed=8)
    source_a, source_b, _ = _exchange_pair(settings)
    alignment = align_modal_envelope_series(source_a, source_b, settings)

    first = evaluate_modal_envelope_correlation(alignment, settings)
    second = evaluate_modal_envelope_correlation(alignment, settings)
    third = evaluate_modal_envelope_correlation(alignment, other_seed)

    assert first == second
    assert first.zero_lag_p_value is not None
    assert third.zero_lag_p_value is not None


def test_supported_with_reservations_uses_context_without_claiming_physics() -> None:
    source_a, source_b, settings = _exchange_pair()
    source_b = prepare_modal_envelope_series(
        source_b,
        settings,
        diagnostics=("background_contamination_context",),
    )

    evidence = evaluate_modal_energy_exchange_pair(source_a, source_b, settings)

    assert evidence.status is ModalEnergyExchangeStatus.SUPPORTED_WITH_RESERVATIONS
    assert ModalEnergyExchangeReason.BACKGROUND_CONTAMINATION in evidence.reservation_reasons
    assert "no_physical_transfer_or_causality_inferred" in evidence.diagnostics


def test_peak_overlap_and_frequency_crossing_contexts_are_reservations_only() -> None:
    source_a, source_b, settings = _exchange_pair()
    source_b = prepare_modal_envelope_series(
        source_b,
        settings,
        diagnostics=("possible_peak_overlap_context", "possible_frequency_crossing_context"),
    )

    evidence = evaluate_modal_energy_exchange_pair(source_a, source_b, settings)

    assert evidence.status is ModalEnergyExchangeStatus.SUPPORTED_WITH_RESERVATIONS
    assert ModalEnergyExchangeReason.POSSIBLE_PEAK_OVERLAP in evidence.reservation_reasons
    assert ModalEnergyExchangeReason.POSSIBLE_FREQUENCY_CROSSING in evidence.reservation_reasons


def test_invalid_modal_parameter_source_cannot_generate_supported_evidence() -> None:
    from tests.test_modal_q_factors import _corrupt_frequency, _parameter_estimate

    source_a, source_b, settings = _exchange_pair()
    invalid_parameter = _corrupt_frequency(_parameter_estimate(prefix="energy-invalid"), -1.0)

    evidence = evaluate_modal_energy_exchange_pair(
        source_a,
        source_b,
        settings,
        parameter_estimate_a=invalid_parameter,
    )

    assert evidence.status is ModalEnergyExchangeStatus.INSUFFICIENT_EVIDENCE
    assert ModalEnergyExchangeReason.UNSUPPORTED_SOURCE_STATUS in evidence.insufficient_evidence_reasons
    assert evidence.valid is False


def test_partial_modal_parameter_source_is_auditable_when_allowed() -> None:
    from tests.test_modal_q_factors import _parameter_estimate

    source_a, source_b, settings = _exchange_pair()
    partial_parameter = _parameter_estimate(tau_s=None, prefix="energy-partial")

    evidence = evaluate_modal_energy_exchange_pair(
        source_a,
        source_b,
        settings,
        parameter_estimate_a=partial_parameter,
    )

    assert evidence.status is ModalEnergyExchangeStatus.SUPPORTED_WITH_RESERVATIONS
    assert ModalEnergyExchangeReason.NEAR_THRESHOLD_TRACKING in evidence.reservation_reasons


def test_global_result_has_one_canonical_pair_per_explicit_pair_and_consistent_counts() -> None:
    supported_a, supported_b, settings = _exchange_pair()
    reserved_b = prepare_modal_envelope_series(
        supported_b,
        settings,
        source_id="B-reserved",
        diagnostics=("background_contamination_context",),
    )
    inconclusive_a = _series("I-A", (1.0, 0.9, 0.8, 0.7, 0.6, 0.5), settings=settings)
    inconclusive_b = _series("I-B", (0.4, 0.9, 0.45, 0.95, 0.5, 1.0), settings=settings)
    decay_a = _series("D-A", (1.0, 0.8, 0.7, 0.6, 0.5, 0.4), settings=settings)
    decay_b = _series("D-B", (0.9, 0.7, 0.6, 0.5, 0.4, 0.3), settings=settings)
    no_overlap_a = _series("S-A", (1.0, 0.8, 0.6, 0.4, 0.2), times=_times(5), settings=settings)
    no_overlap_b = _series("S-B", (0.2, 0.4, 0.6, 0.8, 1.0), times=tuple(2.0 + time for time in _times(5)), settings=settings)
    invalid = prepare_modal_envelope_series(
        times_s=_times(5),
        amplitudes=(1.0, -0.1, 0.8, 0.7, 0.6),
        source_id="invalid",
        settings=settings,
    )
    result = evaluate_modal_energy_exchange(
        (
            supported_a,
            supported_b,
            reserved_b,
            inconclusive_a,
            inconclusive_b,
            decay_a,
            decay_b,
            no_overlap_a,
            no_overlap_b,
            invalid,
        ),
        settings,
        pairs=(
            (supported_a, supported_b),
            (supported_a, reserved_b),
            (inconclusive_a, inconclusive_b),
            (decay_a, decay_b),
            (no_overlap_a, no_overlap_b),
            (invalid, supported_a),
        ),
    )

    assert result.pair_count == 6
    assert result.supported_count == 1
    assert result.supported_with_reservations_count == 1
    assert result.inconclusive_count == 1
    assert result.not_supported_count == 1
    assert result.insufficient_evidence_count == 1
    assert result.invalid_count == 1
    assert len({item.evidence_id for item in result.pair_evidences}) == 6
    assert summarize_modal_energy_exchange(result)["pair_count"] == 6


def test_pair_and_global_evaluation_are_deterministic_under_reordering() -> None:
    source_a, source_b, settings = _exchange_pair()
    source_c = _series("C", (0.9, 0.75, 0.62, 0.5, 0.42, 0.35), settings=settings)

    first = evaluate_modal_energy_exchange((source_a, source_b, source_c), settings)
    second = evaluate_modal_energy_exchange((source_c, source_b, source_a), settings)
    pair_ab = evaluate_modal_energy_exchange_pair(source_a, source_b, settings)
    pair_ba = evaluate_modal_energy_exchange_pair(source_b, source_a, settings)

    assert _normalized_result(first) == _normalized_result(second)
    assert pair_ab == pair_ba


def test_local_perturbation_changes_only_pairs_that_include_perturbed_source() -> None:
    source_a, source_b, settings = _exchange_pair()
    source_c = _series("C", (0.9, 0.75, 0.62, 0.5, 0.42, 0.35), settings=settings)
    perturbed_a = prepare_modal_envelope_series(
        source_a,
        settings,
        amplitudes=tuple(
            value + (0.03 if index == 3 else 0.0)
            for index, value in enumerate(source_a.amplitudes)
        ),
        source_id=source_a.source_id,
    )
    base = evaluate_modal_energy_exchange((source_a, source_b, source_c), settings)
    changed = evaluate_modal_energy_exchange((perturbed_a, source_b, source_c), settings)
    by_pair = {
        (item.source_a_id, item.source_b_id): item.evidence_id
        for item in base.pair_evidences
    }
    changed_by_pair = {
        (item.source_a_id, item.source_b_id): item.evidence_id
        for item in changed.pair_evidences
    }

    assert changed_by_pair[("B", "C")] == by_pair[("B", "C")]
    assert changed_by_pair[("A", "B")] != by_pair[("A", "B")]
    assert changed_by_pair[("A", "C")] != by_pair[("A", "C")]


def test_inputs_and_global_rng_are_not_mutated() -> None:
    source_a, source_b, settings = _exchange_pair()
    before_a = deepcopy(source_a)
    before_b = deepcopy(source_b)
    random.seed(1234)
    state_before = random.getstate()

    first = evaluate_modal_energy_exchange_pair(source_a, source_b, settings)
    state_after = random.getstate()
    second = evaluate_modal_energy_exchange_pair(source_a, source_b, settings)

    assert source_a == before_a
    assert source_b == before_b
    assert state_after == state_before
    assert first == second


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum_overlap_sample_count": 0},
        {"permutation_count": 0},
        {"minimum_positive_amplitude": 0.0},
        {"maximum_time_step_mismatch_fraction": -0.1},
        {"maximum_time_step_mismatch_fraction": 1.1},
        {"significance_level": -0.1},
        {"significance_level": 1.1},
        {"analysis_window_start_s": 1.0, "analysis_window_end_s": 0.5},
        {"resampling_step_s": 0.0},
        {"lag_step_s": 0.0},
        {"dominance_hysteresis_ratio": 0.9},
        {"opposed_trend_score_weight": -1.0},
        {"random_seed": 1.5},
        {"resampling_policy": "unknown"},
    ],
    ids=[
        "sample_count",
        "permutation_count",
        "minimum_amplitude",
        "fraction_negative",
        "fraction_above_one",
        "significance_negative",
        "significance_above_one",
        "window_order",
        "resampling_step",
        "lag_step",
        "hysteresis",
        "negative_weight",
        "seed",
        "unknown_policy",
    ],
)
def test_settings_reject_invalid_values(kwargs) -> None:
    with pytest.raises(ValueError):
        ModalEnergyExchangeSettings(**kwargs)


def test_evidence_contract_rejects_incoherent_valid_flag() -> None:
    source_a, source_b, settings = _exchange_pair()
    evidence = evaluate_modal_energy_exchange_pair(source_a, source_b, settings)

    with pytest.raises(ValueError, match="valid flag"):
        replace(evidence, valid=False)
