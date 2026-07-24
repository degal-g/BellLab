"""Validação da evidência operacional de excitação por impacto."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from belllab import (
    ModalCandidateSettings,
    PreImpactAnalysisSettings,
    PreImpactEvidence,
    analyze_candidates_preimpact,
    analyze_preimpact_evidence,
    characterize_spectral_track,
    evaluate_modal_candidate,
    select_modal_candidates,
)
from tests.test_modal_candidates import _context, _settings
from tests.test_track_characterization_complete import _track


PRE_TIMES = (-0.9, -0.5, -0.1)
POST_TIMES = (0.05, 0.15, 0.25)


def _evidence(
    pre: tuple[float, ...],
    post: tuple[float, ...],
    *,
    unit: str = "linear_amplitude",
    settings: PreImpactAnalysisSettings | None = None,
) -> tuple[PreImpactEvidence, object, object]:
    times = PRE_TIMES[: len(pre)] + POST_TIMES[: len(post)]
    amplitudes = pre + post
    track = _track(
        tuple(100.0 for _ in times),
        amplitudes=amplitudes,
        times=times,
        frames=tuple(range(len(times))),
        unit=unit,
    )
    characterization = characterize_spectral_track(track)
    return (
        analyze_preimpact_evidence(track, characterization, 0.0, settings),
        track,
        characterization,
    )


def test_absent_preimpact_line_is_emergent_without_invalid_ratio() -> None:
    evidence, _, _ = _evidence((), (4.0, 3.0, 2.0))
    assert evidence.preimpact_available_point_count == 0
    assert evidence.preimpact_detected is False
    assert evidence.impact_excited is True
    assert evidence.classification == "impact_emergent"
    assert evidence.impact_level_change_db is None
    assert evidence.post_to_pre_ratio is None
    assert evidence.success


def test_absent_preimpact_excitation_policy_is_configurable() -> None:
    evidence, _, _ = _evidence(
        (),
        (4.0, 3.0, 2.0),
        settings=PreImpactAnalysisSettings(absent_preimpact_is_excited=False),
    )
    assert not evidence.impact_excited
    assert evidence.classification == "not_detected_preimpact"


def test_preimpact_level_detection_threshold_is_configurable() -> None:
    evidence, _, _ = _evidence(
        (0.1, 0.1, 0.1),
        (1.0, 1.0, 1.0),
        settings=PreImpactAnalysisSettings(minimum_preimpact_level=0.2),
    )
    assert not evidence.preimpact_detected
    assert evidence.impact_excited
    assert evidence.classification == "impact_emergent"


@pytest.mark.parametrize(
    ("increase_db", "threshold", "excited"),
    [
        (12.0, 6.0, True),
        (6.0, 6.0, True),
        (5.9, 6.0, False),
    ],
    ids=["twelve_db", "exactly_six_db", "below_six_db"],
)
def test_linear_preexisting_line_uses_inclusive_db_increase(
    increase_db, threshold, excited
) -> None:
    ratio = 10 ** (increase_db / 20)
    settings = PreImpactAnalysisSettings(
        minimum_impact_level_increase_db=threshold
    )
    evidence, _, _ = _evidence(
        (1.0, 1.0, 1.0), (ratio, ratio, ratio), settings=settings
    )
    assert evidence.preimpact_detected
    assert evidence.impact_level_change_db == pytest.approx(increase_db)
    assert evidence.post_to_pre_ratio == pytest.approx(ratio)
    assert evidence.impact_excited is excited
    assert evidence.classification == (
        "impact_amplified" if excited else "indeterminate"
    )


@pytest.mark.parametrize(
    ("increase_db", "threshold", "excited"),
    [
        (12.0, 6.0, True),
        (6.0, 6.0, True),
        (5.9, 6.0, False),
    ],
    ids=["twelve_db", "exactly_six_db", "below_six_db"],
)
def test_dbfs_preexisting_line_uses_level_difference(
    increase_db, threshold, excited
) -> None:
    settings = PreImpactAnalysisSettings(
        minimum_impact_level_increase_db=threshold
    )
    evidence, _, _ = _evidence(
        (-30.0, -30.0, -30.0),
        tuple(-30.0 + increase_db for _ in range(3)),
        unit="dbfs_amplitude",
        settings=settings,
    )
    assert evidence.impact_level_change_db == pytest.approx(increase_db)
    assert evidence.post_to_pre_ratio == pytest.approx(10 ** (increase_db / 20))
    assert evidence.impact_excited is excited


def test_persistent_background_tone_is_not_automatically_excited() -> None:
    evidence, _, _ = _evidence((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    assert evidence.preimpact_detected
    assert evidence.impact_level_change_db == pytest.approx(0.0)
    assert not evidence.impact_excited
    assert evidence.background_contaminated
    assert evidence.classification == "persistent_background_tone"


def test_weaker_postimpact_line_has_negative_change_and_no_false_excitation() -> None:
    evidence, _, _ = _evidence((2.0, 2.0, 2.0), (1.0, 1.0, 1.0))
    assert evidence.impact_level_change_db == pytest.approx(-6.0205999133)
    assert evidence.post_to_pre_ratio == pytest.approx(0.5)
    assert not evidence.impact_excited
    assert evidence.classification == "indeterminate"


def test_preexisting_decay_without_reexcitation_is_distinguished() -> None:
    evidence, _, _ = _evidence((3.0, 2.0, 1.0), (0.9, 0.8, 0.7))
    assert evidence.preimpact_slope_per_s == pytest.approx(-2.5)
    assert evidence.preimpact_decay_detected
    assert not evidence.impact_excited
    assert evidence.background_contaminated
    assert evidence.classification == "preexisting_decay"


def test_preexisting_decay_can_be_reexcited_by_new_impact() -> None:
    evidence, _, _ = _evidence((3.0, 2.0, 1.0), (8.0, 6.0, 4.0))
    assert evidence.preimpact_decay_detected
    assert evidence.postimpact_decay_detected
    assert evidence.impact_level_change_db == pytest.approx(
        20 * np.log10(3.0)
    )
    assert evidence.impact_excited
    assert not evidence.background_contaminated
    assert evidence.classification == "reexcited_preexisting_component"


def test_postimpact_decay_can_be_required() -> None:
    settings = PreImpactAnalysisSettings(require_postimpact_decay=True)
    constant, _, _ = _evidence(
        (1.0, 1.0, 1.0), (4.0, 4.0, 4.0), settings=settings
    )
    decaying, _, _ = _evidence(
        (1.0, 1.0, 1.0), (6.0, 4.0, 2.0), settings=settings
    )
    assert not constant.impact_excited
    assert "required_postimpact_decay_not_detected" in constant.diagnostics
    assert decaying.impact_excited and decaying.postimpact_decay_detected


def test_window_statistics_use_median_and_auditable_counts() -> None:
    evidence, _, _ = _evidence((1.0, 1.0, 100.0), (4.0, 4.0, 4.0))
    assert evidence.preimpact_level == pytest.approx(34.0)
    assert evidence.preimpact_median_level == pytest.approx(1.0)
    assert evidence.preimpact_variability == pytest.approx(np.std((1, 1, 100)))
    assert evidence.impact_level_change_db == pytest.approx(20 * np.log10(4))
    assert (evidence.preimpact_available_point_count, evidence.preimpact_finite_point_count) == (3, 3)


def test_linear_nonpositive_prelevel_never_enters_logarithm() -> None:
    evidence, _, _ = _evidence((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    assert evidence.preimpact_detected
    assert evidence.impact_level_change == pytest.approx(1.0)
    assert evidence.impact_level_change_db is None
    assert evidence.post_to_pre_ratio is None
    assert "linear_level_change_db_unavailable" in evidence.diagnostics


def test_nonfinite_values_are_discarded_without_propagation() -> None:
    evidence, _, _ = _evidence(
        (1.0, np.nan, 1.0), (4.0, np.inf, 4.0)
    )
    assert (evidence.preimpact_available_point_count, evidence.preimpact_finite_point_count) == (3, 2)
    assert (evidence.postimpact_available_point_count, evidence.postimpact_finite_point_count) == (3, 2)
    assert evidence.preimpact_median_level == 1.0
    assert evidence.postimpact_median_level == 4.0
    assert evidence.impact_level_change_db == pytest.approx(20 * np.log10(4))
    assert "nonfinite_preimpact_values_discarded" in evidence.diagnostics
    assert "nonfinite_postimpact_values_discarded" in evidence.diagnostics


@pytest.mark.parametrize(
    ("pre", "post", "classification", "failure"),
    [
        ((np.nan, np.inf, -np.inf), (4.0, 3.0, 2.0), "insufficient_preimpact_data", "insufficient_preimpact_data"),
        ((1.0,), (4.0, 3.0, 2.0), "insufficient_preimpact_data", "insufficient_preimpact_data"),
        ((1.0, 1.0, 1.0), (4.0,), "insufficient_postimpact_data", "insufficient_postimpact_data"),
        ((1.0, 1.0, 1.0), (), "insufficient_postimpact_data", "insufficient_postimpact_data"),
    ],
    ids=["all_pre_nonfinite", "one_pre_point", "one_post_point", "empty_post_window"],
)
def test_insufficient_windows_fail_structurally(
    pre, post, classification, failure
) -> None:
    evidence, _, _ = _evidence(pre, post)
    assert not evidence.success
    assert evidence.classification == classification
    assert evidence.failure_reason == failure
    assert not evidence.impact_excited


def test_preimpact_coverage_threshold_is_explicit() -> None:
    settings = PreImpactAnalysisSettings(
        minimum_preimpact_coverage_fraction=0.9
    )
    evidence, _, _ = _evidence(
        (1.0, 1.0, 1.0), (4.0, 3.0, 2.0), settings=settings
    )
    assert evidence.preimpact_coverage_fraction == pytest.approx(8 / 9)
    assert not evidence.success
    assert evidence.classification == "insufficient_preimpact_data"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"preimpact_window_start_s": -0.1, "preimpact_window_end_s": -1.0},
        {"preimpact_window_end_s": 0.0},
        {"postimpact_window_start_s": -0.1},
        {"postimpact_window_start_s": 0.3, "postimpact_window_end_s": 0.2},
        {"minimum_preimpact_point_count": -1},
        {"minimum_postimpact_point_count": -1},
        {"minimum_preimpact_coverage_fraction": -0.1},
        {"minimum_preimpact_coverage_fraction": 1.1},
        {"minimum_impact_level_increase_db": -0.1},
        {"unchanged_level_tolerance_db": -0.1},
        {"preimpact_decay_slope_tolerance": -0.1},
        {"postimpact_decay_slope_tolerance": -0.1},
        {"preimpact_window_start_s": float("nan")},
        {"postimpact_window_end_s": float("inf")},
        {"minimum_impact_level_increase_db": float("nan")},
        {"unchanged_level_tolerance_db": float("inf")},
    ],
    ids=[
        "inverted_pre", "pre_after_impact", "post_before_impact", "inverted_post",
        "negative_pre_count", "negative_post_count", "negative_coverage",
        "coverage_above_one", "negative_increase", "negative_level_tolerance",
        "negative_pre_slope_tolerance", "negative_post_slope_tolerance",
        "nan_window", "infinite_window", "nan_increase", "infinite_tolerance",
    ],
)
def test_preimpact_settings_reject_invalid_configuration(kwargs) -> None:
    with pytest.raises(ValueError):
        PreImpactAnalysisSettings(**kwargs)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"source_track_id": -1}, "source_track_id"),
        ({"amplitude_unit": "power"}, "amplitude_unit"),
        ({"preimpact_finite_point_count": 99}, "finite counts"),
        ({"preimpact_level": float("nan")}, "finite"),
        ({"post_to_pre_ratio": 0.0}, "positive"),
        ({"impact_level_change_db": float("inf")}, "finite"),
        ({"classification": "physical_mode"}, "classification"),
        ({"success": False, "failure_reason": None}, "failure_reason"),
        ({"success": False, "failure_reason": "other"}, "insufficiency"),
        ({"classification": "insufficient_preimpact_data"}, "successful"),
        ({"background_contaminated": True, "preimpact_detected": False}, "contamination"),
        ({"diagnostics": ("repeat", "repeat")}, "diagnostics"),
    ],
    ids=[
        "negative_track", "unknown_unit", "count_mismatch", "nan_level",
        "zero_ratio", "infinite_db", "unknown_classification", "failure_without_reason",
        "failure_classification_mismatch", "successful_insufficiency",
        "background_without_preimpact", "duplicate_diagnostics",
    ],
)
def test_preimpact_evidence_rejects_invalid_invariants(changes, message) -> None:
    evidence, _, _ = _evidence((1.0, 1.0, 1.0), (4.0, 3.0, 2.0))
    with pytest.raises(ValueError, match=message):
        replace(evidence, **changes)


def test_impact_excited_requires_structured_excitation_classification() -> None:
    evidence, _, _ = _evidence((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="excitation classification"):
        replace(evidence, impact_excited=True)


def test_candidate_preimpact_criteria_are_disabled_by_default() -> None:
    evidence, _, characterization = _evidence(
        (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)
    )
    _, tracking = _context()
    candidate = evaluate_modal_candidate(
        characterization, tracking, _settings(), preimpact_evidence=evidence
    )
    assert candidate.accepted
    for name in (
        "require_impact_excitation",
        "reject_persistent_background_tone",
        "minimum_impact_level_increase_db",
    ):
        criterion = next(item for item in candidate.criteria_results if item.criterion == name)
        assert not criterion.enabled


def test_emergent_and_amplified_lines_pass_required_excitation() -> None:
    emergent, _, emergent_characterization = _evidence((), (4.0, 3.0, 2.0))
    amplified, _, amplified_characterization = _evidence(
        (1.0, 1.0, 1.0), (4.0, 4.0, 4.0)
    )
    _, tracking = _context()
    settings = _settings(require_impact_excitation=True)
    assert evaluate_modal_candidate(
        emergent_characterization, tracking, settings,
        preimpact_evidence=emergent,
    ).accepted
    assert evaluate_modal_candidate(
        amplified_characterization, tracking, settings,
        preimpact_evidence=amplified,
    ).accepted


def test_persistent_tone_rejects_only_when_optional_criterion_enabled() -> None:
    evidence, _, characterization = _evidence(
        (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)
    )
    _, tracking = _context()
    optional = evaluate_modal_candidate(
        characterization, tracking, _settings(), preimpact_evidence=evidence
    )
    rejected = evaluate_modal_candidate(
        characterization,
        tracking,
        _settings(reject_persistent_background_tone=True),
        preimpact_evidence=evidence,
    )
    assert optional.accepted
    assert not rejected.accepted
    assert any("reject_persistent_background_tone" in reason for reason in rejected.rejection_reasons)


def test_candidate_minimum_increase_is_inclusive() -> None:
    evidence, _, characterization = _evidence(
        (1.0, 1.0, 1.0),
        tuple(10 ** (6 / 20) for _ in range(3)),
    )
    _, tracking = _context()
    candidate = evaluate_modal_candidate(
        characterization,
        tracking,
        _settings(minimum_impact_level_increase_db=6.0),
        preimpact_evidence=evidence,
    )
    assert candidate.accepted


def test_missing_evidence_fails_only_enabled_candidate_criterion() -> None:
    characterization, tracking = _context()
    optional = evaluate_modal_candidate(characterization, tracking, _settings())
    required = evaluate_modal_candidate(
        characterization,
        tracking,
        _settings(require_impact_excitation=True),
    )
    assert optional.accepted
    assert not required.accepted
    assert any("require_impact_excitation" in reason for reason in required.rejection_reasons)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum_impact_level_increase_db": -0.1},
        {"minimum_impact_level_increase_db": float("nan")},
        {"minimum_impact_level_increase_db": float("inf")},
        {"require_impact_excitation": 1},
        {"reject_persistent_background_tone": "yes"},
    ],
    ids=[
        "candidate_negative_increase",
        "candidate_nan_increase",
        "candidate_infinite_increase",
        "candidate_nonboolean_excitation",
        "candidate_nonboolean_background",
    ],
)
def test_candidate_preimpact_settings_reject_invalid_values(kwargs) -> None:
    with pytest.raises(ValueError):
        ModalCandidateSettings(**kwargs)


def test_batch_analysis_preserves_candidate_order_and_rejections() -> None:
    tracking = _context()[1]
    tracks = tracking.tracks
    characterizations = tuple(characterize_spectral_track(track) for track in tracks)
    candidates = select_modal_candidates(
        characterizations,
        tracking,
        ModalCandidateSettings(minimum_observation_count=99),
    )
    evidence_first = analyze_candidates_preimpact(candidates, tracks, 0.0)
    evidence_second = analyze_candidates_preimpact(candidates, tracks, 0.0)
    assert evidence_first == evidence_second
    assert tuple(item.source_track_id for item in evidence_first) == tuple(
        candidate.source_track_id for candidate in candidates
    )
    assert all(not candidate.accepted for candidate in candidates)


def test_preimpact_analysis_is_deterministic_and_does_not_mutate_inputs() -> None:
    _, track, characterization = _evidence(
        (1.0, 1.0, 1.0), (4.0, 3.0, 2.0)
    )
    first = analyze_preimpact_evidence(track, characterization, 0.0)
    second = analyze_preimpact_evidence(track, characterization, 0.0)
    assert first == second
    assert track.amplitudes == (1.0, 1.0, 1.0, 4.0, 3.0, 2.0)
