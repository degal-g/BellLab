"""Testes comportamentais da promoção operacional para candidatos modais."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from belllab import (
    CandidateCriterionResult,
    ModalCandidate,
    ModalCandidateSettings,
    SpectralTrackingSettings,
    TrackAssignmentDiagnostic,
    characterize_spectral_track,
    evaluate_modal_candidate,
    select_modal_candidates,
    track_spectral_peaks,
)
from tests.test_track_characterization_complete import _track
from tests.test_tracking import _manual_results


def _context(
    frequencies: tuple[tuple[float, ...], ...] | None = None,
    *,
    amplitudes: tuple[float, ...] | None = None,
    max_gap_frames: int = 2,
):
    frequencies = frequencies or tuple((100.0,) for _ in range(6))
    tracking = track_spectral_peaks(
        _manual_results(frequencies),
        SpectralTrackingSettings(
            frequency_tolerance=10.0,
            frequency_distance_unit="hz",
            max_gap_frames=max_gap_frames,
            min_track_length=1,
        ),
    )
    track = tracking.tracks[0]
    if amplitudes is None:
        amplitudes = tuple(float(np.exp(-time)) for time in track.times_s)
    return characterize_spectral_track(replace(track, amplitudes=amplitudes)), tracking


def _settings(**kwargs) -> ModalCandidateSettings:
    return ModalCandidateSettings(minimum_observation_count=None, **kwargs)


def _criterion(candidate: ModalCandidate, name: str) -> CandidateCriterionResult:
    return next(item for item in candidate.criteria_results if item.criterion == name)


def _tracking_with_diagnostics(tracking, diagnostics):
    return replace(
        tracking,
        assignment_diagnostics=tuple(diagnostics),
        ambiguous_assignment_count=sum(item.ambiguous for item in diagnostics),
        near_threshold_assignment_count=sum(item.near_threshold for item in diagnostics),
    )


def _diagnostic(
    *,
    ambiguous: bool = False,
    near: bool = False,
    margin: float | None = 0.5,
    frame: int = 1,
) -> TrackAssignmentDiagnostic:
    return TrackAssignmentDiagnostic(
        frame, 0, 0, 0.1, margin, None, margin, ambiguous, near,
        1.0, "hz", None, 0.1, 0.0,
    )


def test_adequate_characterization_is_accepted_with_auditable_criteria() -> None:
    characterization, tracking = _context()
    settings = ModalCandidateSettings(
        minimum_observation_count=5,
        minimum_coverage_fraction=0.9,
        minimum_duration_s=4.0,
        maximum_relative_frequency_stability=0.01,
        maximum_absolute_frequency_drift_hz=1.0,
        maximum_frequency_fit_rmse_hz=0.1,
        require_successful_frequency_fit=True,
        require_amplitude_decay=True,
        minimum_amplitude_fit_r_squared=0.99,
        minimum_decay_tau_s=0.5,
        maximum_decay_tau_s=2.0,
        maximum_ambiguous_assignment_fraction=0.1,
        maximum_near_threshold_assignment_fraction=0.1,
        minimum_assignment_margin=0.2,
    )
    tracking = _tracking_with_diagnostics(
        tracking, tuple(_diagnostic(frame=index) for index in range(1, 6))
    )
    candidate = evaluate_modal_candidate(
        characterization, tracking, settings, candidate_id=4
    )
    assert candidate.candidate_id == 4
    assert candidate.source_track_id == characterization.track_id
    assert candidate.accepted and candidate.rejection_reasons == ()
    assert candidate.representative_frequency_hz == pytest.approx(100.0)
    assert candidate.frequency_source == "interpolated"
    assert candidate.amplitude_unit == "linear_amplitude"
    assert candidate.observation_count == 6
    assert candidate.coverage_fraction == 1.0
    assert candidate.duration_s == 5.0
    assert candidate.frequency_stability == 0.0
    assert candidate.frequency_drift_hz == 0.0
    assert candidate.frequency_fit_rmse_hz == pytest.approx(0.0, abs=1e-12)
    assert candidate.amplitude_decay_detected
    assert candidate.amplitude_tau_s == pytest.approx(1.0)
    assert candidate.amplitude_fit_r_squared == pytest.approx(1.0)
    assert candidate.accepted_assignment_count == 5
    assert all(
        item.passed is True
        for item in candidate.criteria_results
        if item.enabled and item.applicable
    )


@pytest.mark.parametrize(
    ("setting", "value", "criterion"),
    [
        ("minimum_observation_count", 7, "minimum_observation_count"),
        ("minimum_duration_s", 6.0, "minimum_duration_s"),
    ],
    ids=["few_observations", "short_duration"],
)
def test_persistence_criteria_reject_independently(setting, value, criterion) -> None:
    characterization, tracking = _context()
    candidate = evaluate_modal_candidate(
        characterization, tracking, ModalCandidateSettings(**{setting: value})
    )
    assert not candidate.accepted
    assert _criterion(candidate, criterion).passed is False
    assert any(criterion in reason for reason in candidate.rejection_reasons)


def test_low_coverage_rejects_independently() -> None:
    characterization, tracking = _context(((100.0,), (), (100.0,)))
    assert characterization.coverage_fraction == pytest.approx(2 / 3)
    candidate = evaluate_modal_candidate(
        characterization,
        tracking,
        _settings(minimum_coverage_fraction=0.8),
    )
    assert not candidate.accepted
    assert _criterion(candidate, "minimum_coverage_fraction").passed is False


@pytest.mark.parametrize(
    ("frequencies", "setting", "threshold", "criterion"),
    [
        (
            tuple((value,) for value in (100.0, 102.0, 98.0, 103.0)),
            "maximum_relative_frequency_stability", 0.005,
            "maximum_relative_frequency_stability",
        ),
        (
            tuple((value,) for value in (100.0, 101.0, 102.0, 103.0)),
            "maximum_absolute_frequency_drift_hz", 2.0,
            "maximum_absolute_frequency_drift_hz",
        ),
        (
            tuple((value,) for value in (100.0, 102.0, 99.0, 103.0)),
            "maximum_frequency_fit_rmse_hz", 0.5,
            "maximum_frequency_fit_rmse_hz",
        ),
    ],
    ids=["unstable", "excessive_drift", "excessive_rmse"],
)
def test_frequency_limits_reject_exact_criterion(
    frequencies, setting, threshold, criterion
) -> None:
    characterization, tracking = _context(frequencies)
    candidate = evaluate_modal_candidate(
        characterization,
        tracking,
        _settings(**{setting: threshold}),
    )
    assert not candidate.accepted
    assert _criterion(candidate, criterion).passed is False


def test_failed_frequency_fit_rejected_only_when_required() -> None:
    characterization, tracking = _context(((100.0,),))
    required = evaluate_modal_candidate(
        characterization,
        tracking,
        _settings(require_successful_frequency_fit=True),
    )
    optional = evaluate_modal_candidate(characterization, tracking, _settings())
    assert not required.accepted
    assert _criterion(required, "require_successful_frequency_fit").passed is False
    assert optional.accepted
    assert not _criterion(optional, "require_successful_frequency_fit").enabled


def test_missing_representative_frequency_is_always_rejected() -> None:
    characterization = characterize_spectral_track(
        _track((np.nan, np.inf), refined=(None, None))
    )
    _, tracking = _context()
    candidate = evaluate_modal_candidate(characterization, tracking, _settings())
    assert not candidate.accepted
    assert candidate.representative_frequency_hz is None
    assert _criterion(candidate, "representative_frequency_hz").passed is False


def test_mixed_frequency_source_is_configurable() -> None:
    characterization = characterize_spectral_track(
        _track((100.0, 101.0, 102.0), refined=(100.1, None, 102.1))
    )
    _, tracking = _context()
    allowed = evaluate_modal_candidate(
        characterization, tracking, _settings(allow_mixed_frequency_source=True)
    )
    rejected = evaluate_modal_candidate(
        characterization, tracking, _settings(allow_mixed_frequency_source=False)
    )
    assert allowed.accepted
    assert not _criterion(allowed, "allow_mixed_frequency_source").enabled
    assert not rejected.accepted
    assert _criterion(rejected, "allow_mixed_frequency_source").passed is False


def test_bin_frequency_source_is_not_rejected_by_default() -> None:
    characterization = characterize_spectral_track(
        _track((100.0, 101.0), refined=(None, None))
    )
    _, tracking = _context()
    candidate = evaluate_modal_candidate(characterization, tracking, _settings())
    assert candidate.accepted and candidate.frequency_source == "bin"


def test_median_is_representative_frequency() -> None:
    characterization = characterize_spectral_track(_track((100.0, 101.0, 400.0)))
    _, tracking = _context()
    candidate = evaluate_modal_candidate(characterization, tracking, _settings())
    assert candidate.representative_frequency_hz == pytest.approx(101.0)


def test_amplitude_decay_is_optional_by_default() -> None:
    characterization, tracking = _context(amplitudes=(1.0,) * 6)
    optional = evaluate_modal_candidate(characterization, tracking, _settings())
    required = evaluate_modal_candidate(
        characterization, tracking, _settings(require_amplitude_decay=True)
    )
    assert optional.accepted and optional.amplitude_tau_s is None
    assert not _criterion(optional, "require_amplitude_decay").enabled
    assert not required.accepted
    assert _criterion(required, "require_amplitude_decay").passed is False


@pytest.mark.parametrize(
    ("amplitudes", "setting", "threshold", "criterion"),
    [
        (
            tuple(float(np.exp(-time) * factor) for time, factor in enumerate((1, 1.2, 0.8, 1.1, 0.9, 1))),
            "minimum_amplitude_fit_r_squared", 0.999,
            "minimum_amplitude_fit_r_squared",
        ),
        (
            tuple(float(np.exp(-time / 0.2)) for time in range(6)),
            "minimum_decay_tau_s", 0.5,
            "minimum_decay_tau_s",
        ),
        (
            tuple(float(np.exp(-time / 3.0)) for time in range(6)),
            "maximum_decay_tau_s", 2.0,
            "maximum_decay_tau_s",
        ),
    ],
    ids=["low_r_squared", "tau_below_minimum", "tau_above_maximum"],
)
def test_amplitude_limits_reject_exact_criterion(
    amplitudes, setting, threshold, criterion
) -> None:
    characterization, tracking = _context(amplitudes=amplitudes)
    candidate = evaluate_modal_candidate(
        characterization, tracking, _settings(**{setting: threshold})
    )
    assert not candidate.accepted
    assert _criterion(candidate, criterion).passed is False


def test_amplitude_limits_do_not_reject_when_disabled() -> None:
    characterization, tracking = _context(amplitudes=(1.0,) * 6)
    candidate = evaluate_modal_candidate(characterization, tracking, _settings())
    assert candidate.accepted
    for name in (
        "minimum_amplitude_fit_r_squared",
        "minimum_decay_tau_s",
        "maximum_decay_tau_s",
    ):
        item = _criterion(candidate, name)
        assert not item.enabled and item.passed is None


def test_unsuccessful_amplitude_fit_fails_enabled_r_squared_criterion() -> None:
    characterization, tracking = _context(
        ((100.0,),), amplitudes=(1.0,)
    )
    candidate = evaluate_modal_candidate(
        characterization,
        tracking,
        _settings(minimum_amplitude_fit_r_squared=0.8),
    )
    item = _criterion(candidate, "minimum_amplitude_fit_r_squared")
    assert not candidate.accepted
    assert characterization.amplitude_fit.success is False
    assert item.observed is None and item.passed is False
    assert "unavailable" in item.reason


@pytest.mark.parametrize(
    ("diagnostics", "setting", "criterion"),
    [
        (
            (_diagnostic(ambiguous=True), _diagnostic(ambiguous=True, frame=2), _diagnostic(frame=3)),
            {"maximum_ambiguous_assignment_fraction": 0.5},
            "maximum_ambiguous_assignment_fraction",
        ),
        (
            (_diagnostic(near=True), _diagnostic(near=True, frame=2), _diagnostic(frame=3)),
            {"maximum_near_threshold_assignment_fraction": 0.5},
            "maximum_near_threshold_assignment_fraction",
        ),
        (
            (_diagnostic(margin=0.1), _diagnostic(margin=0.5, frame=2)),
            {"minimum_assignment_margin": 0.2},
            "minimum_assignment_margin",
        ),
    ],
    ids=["ambiguous_fraction", "near_threshold_fraction", "minimum_margin"],
)
def test_tracking_quality_criteria_reject_independently(
    diagnostics, setting, criterion
) -> None:
    characterization, tracking = _context()
    tracking = _tracking_with_diagnostics(tracking, diagnostics)
    candidate = evaluate_modal_candidate(
        characterization, tracking, _settings(**setting)
    )
    assert not candidate.accepted
    assert _criterion(candidate, criterion).passed is False
    assert candidate.accepted_assignment_count == len(diagnostics)


@pytest.mark.parametrize(
    ("setting", "criterion"),
    [
        (
            {"maximum_ambiguous_assignment_fraction": 0.2},
            "maximum_ambiguous_assignment_fraction",
        ),
        (
            {"maximum_near_threshold_assignment_fraction": 0.2},
            "maximum_near_threshold_assignment_fraction",
        ),
    ],
)
def test_enabled_assignment_fraction_fails_without_auditable_assignments(
    setting, criterion
) -> None:
    characterization, tracking = _context(((100.0,),))
    candidate = evaluate_modal_candidate(
        characterization, tracking, _settings(**setting)
    )
    assert not candidate.accepted
    assert _criterion(candidate, criterion).passed is False
    assert "no_auditable_assignments" in candidate.diagnostics


def test_missing_margin_is_not_applicable_and_does_not_reject() -> None:
    characterization, tracking = _context()
    diagnostics = tuple(
        _diagnostic(margin=None, frame=index) for index in range(1, 4)
    )
    candidate = evaluate_modal_candidate(
        characterization,
        _tracking_with_diagnostics(tracking, diagnostics),
        _settings(minimum_assignment_margin=0.2),
    )
    item = _criterion(candidate, "minimum_assignment_margin")
    assert candidate.accepted
    assert item.enabled and not item.applicable and item.passed is None
    assert item.reason.endswith("not_applicable")
    assert candidate.minimum_assignment_margin is None
    assert "assignment_margin_not_applicable" in candidate.diagnostics


def test_reaches_final_frame_is_optional_and_auditable() -> None:
    characterization, tracking = _context()
    ended = replace(characterization, reached_analysis_final_frame=False)
    optional = evaluate_modal_candidate(ended, tracking, _settings())
    required = evaluate_modal_candidate(
        ended, tracking, _settings(require_reaches_final_frame=True)
    )
    assert optional.accepted
    assert not required.accepted
    assert _criterion(required, "require_reaches_final_frame").passed is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum_observation_count": -1},
        {"minimum_coverage_fraction": -0.1},
        {"minimum_coverage_fraction": 1.1},
        {"minimum_duration_s": -0.1},
        {"maximum_relative_frequency_stability": -0.1},
        {"maximum_absolute_frequency_drift_hz": -0.1},
        {"maximum_frequency_fit_rmse_hz": -0.1},
        {"minimum_amplitude_fit_r_squared": -0.1},
        {"minimum_amplitude_fit_r_squared": 1.1},
        {"minimum_decay_tau_s": 0.0},
        {"minimum_decay_tau_s": -1.0},
        {"maximum_decay_tau_s": 0.0},
        {"minimum_decay_tau_s": 2.0, "maximum_decay_tau_s": 1.0},
        {"maximum_ambiguous_assignment_fraction": -0.1},
        {"maximum_ambiguous_assignment_fraction": 1.1},
        {"maximum_near_threshold_assignment_fraction": -0.1},
        {"maximum_near_threshold_assignment_fraction": 1.1},
        {"minimum_assignment_margin": -0.1},
        {"minimum_duration_s": float("nan")},
        {"maximum_frequency_fit_rmse_hz": float("inf")},
        {"minimum_decay_tau_s": -float("inf")},
    ],
    ids=[
        "negative_count", "negative_coverage", "coverage_above_one",
        "negative_duration", "negative_stability", "negative_drift",
        "negative_rmse", "negative_r_squared", "r_squared_above_one",
        "zero_min_tau", "negative_min_tau", "zero_max_tau", "tau_order",
        "negative_ambiguous", "ambiguous_above_one", "negative_near",
        "near_above_one", "negative_margin", "nan", "positive_inf",
        "negative_inf",
    ],
)
def test_modal_candidate_settings_reject_invalid_values(kwargs) -> None:
    with pytest.raises(ValueError):
        ModalCandidateSettings(**kwargs)


def test_candidate_and_criterion_invariants_reject_contradictions() -> None:
    characterization, tracking = _context()
    candidate = evaluate_modal_candidate(characterization, tracking)
    with pytest.raises(ValueError, match="source_track_id"):
        replace(candidate, source_track_id=99)
    with pytest.raises(ValueError, match="representative"):
        replace(candidate, representative_frequency_hz=-1.0)
    with pytest.raises(ValueError, match="fraction"):
        replace(candidate, ambiguous_assignment_fraction=0.5)
    with pytest.raises(ValueError, match="duplicate"):
        replace(candidate, criteria_results=candidate.criteria_results * 2)
    with pytest.raises(ValueError, match="failed enabled applicable"):
        replace(candidate, accepted=False, rejection_reasons=())
    with pytest.raises(ValueError, match="failed criteria"):
        replace(candidate, rejection_reasons=("unexpected",))
    with pytest.raises(ValueError, match="disabled"):
        CandidateCriterionResult("x", 1.0, ">=", 0.0, False, True, True, "bad")


def test_selection_preserves_rejected_candidates_and_has_stable_ids() -> None:
    tracking = track_spectral_peaks(
        _manual_results(tuple((100.0, 200.0) for _ in range(4))),
        SpectralTrackingSettings(
            frequency_tolerance=5.0,
            frequency_distance_unit="hz",
            min_track_length=1,
        ),
    )
    characterizations = tuple(
        characterize_spectral_track(track) for track in reversed(tracking.tracks)
    )
    settings = ModalCandidateSettings(minimum_observation_count=5)
    first = select_modal_candidates(characterizations, tracking, settings)
    second = select_modal_candidates(characterizations, tracking, settings)
    assert first == second
    assert tuple(item.candidate_id for item in first) == (0, 1)
    assert tuple(item.source_track_id for item in first) == (0, 1)
    assert all(not item.accepted for item in first)
    assert all(item.rejection_reasons for item in first)


def test_changing_one_setting_changes_only_its_criterion() -> None:
    characterization, tracking = _context()
    base = evaluate_modal_candidate(characterization, tracking, _settings())
    changed = evaluate_modal_candidate(
        characterization,
        tracking,
        _settings(minimum_duration_s=6.0),
    )
    base_by_name = {item.criterion: item for item in base.criteria_results}
    changed_by_name = {item.criterion: item for item in changed.criteria_results}
    differing = tuple(
        name for name in base_by_name if base_by_name[name] != changed_by_name[name]
    )
    assert differing == ("minimum_duration_s",)


def test_selection_rejects_duplicate_track_characterizations() -> None:
    characterization, tracking = _context()
    with pytest.raises(ValueError, match="unique track IDs"):
        select_modal_candidates((characterization, characterization), tracking)


def _valid_accepted_candidate() -> ModalCandidate:
    characterization, tracking = _context()
    return evaluate_modal_candidate(characterization, tracking)


def _valid_rejected_candidate() -> ModalCandidate:
    characterization, tracking = _context()
    return evaluate_modal_candidate(
        characterization,
        tracking,
        ModalCandidateSettings(minimum_observation_count=7),
    )


def test_accepted_candidate_rejects_failed_criterion() -> None:
    rejected = _valid_rejected_candidate()
    with pytest.raises(ValueError, match="accepted candidate|failed"):
        replace(rejected, accepted=True)


def test_accepted_candidate_rejects_rejection_reason() -> None:
    accepted = _valid_accepted_candidate()
    with pytest.raises(ValueError, match="rejection_reasons"):
        replace(accepted, rejection_reasons=("arbitrary rejection",))


def test_rejected_candidate_requires_rejection_reason() -> None:
    rejected = _valid_rejected_candidate()
    with pytest.raises(ValueError, match="rejection_reasons"):
        replace(rejected, rejection_reasons=())


def test_rejected_candidate_rejects_all_criteria_passed() -> None:
    accepted = _valid_accepted_candidate()
    with pytest.raises(ValueError, match="failed enabled applicable"):
        replace(accepted, accepted=False)


def test_rejected_candidate_cannot_rely_on_disabled_criterion() -> None:
    accepted = _valid_accepted_candidate()
    disabled = next(item for item in accepted.criteria_results if not item.enabled)
    with pytest.raises(ValueError, match="rejection_reasons"):
        replace(
            accepted,
            accepted=False,
            rejection_reasons=(disabled.reason,),
        )


def test_rejected_candidate_cannot_rely_on_not_applicable_criterion() -> None:
    characterization, tracking = _context()
    diagnostics = tuple(
        _diagnostic(margin=None, frame=index) for index in range(1, 3)
    )
    candidate = evaluate_modal_candidate(
        characterization,
        _tracking_with_diagnostics(tracking, diagnostics),
        _settings(minimum_assignment_margin=0.2),
    )
    not_applicable = _criterion(candidate, "minimum_assignment_margin")
    assert candidate.accepted and not not_applicable.applicable
    with pytest.raises(ValueError, match="rejection_reasons"):
        replace(
            candidate,
            accepted=False,
            rejection_reasons=(not_applicable.reason,),
        )


def test_rejection_reason_requires_matching_failed_criterion() -> None:
    rejected = _valid_rejected_candidate()
    with pytest.raises(ValueError, match="rejection_reasons"):
        replace(
            rejected,
            rejection_reasons=rejected.rejection_reasons + ("unmatched reason",),
        )


def test_acceptance_reason_requires_matching_passed_criterion() -> None:
    accepted = _valid_accepted_candidate()
    with pytest.raises(ValueError, match="acceptance_reasons"):
        replace(
            accepted,
            acceptance_reasons=("contradictory acceptance",),
        )


@pytest.mark.parametrize(
    ("field", "reasons"),
    [
        ("acceptance_reasons", ("",)),
        ("acceptance_reasons", ("   ",)),
        ("acceptance_reasons", ("duplicate", "duplicate")),
        ("rejection_reasons", ("",)),
        ("rejection_reasons", ("   ",)),
        ("rejection_reasons", ("duplicate", "duplicate")),
    ],
    ids=[
        "empty_acceptance_reason",
        "whitespace_acceptance_reason",
        "duplicate_acceptance_reasons",
        "empty_rejection_reason",
        "whitespace_rejection_reason",
        "duplicate_rejection_reasons",
    ],
)
def test_candidate_reasons_require_unique_nonempty_strings(field, reasons) -> None:
    candidate = (
        _valid_accepted_candidate()
        if field == "acceptance_reasons"
        else _valid_rejected_candidate()
    )
    with pytest.raises(ValueError, match="unique nonempty"):
        replace(candidate, **{field: reasons})


def test_reason_order_follows_criterion_order_deterministically() -> None:
    characterization, tracking = _context()
    candidate = evaluate_modal_candidate(
        characterization,
        tracking,
        ModalCandidateSettings(
            minimum_observation_count=7,
            minimum_duration_s=6.0,
        ),
    )
    expected = tuple(
        item.reason
        for item in candidate.criteria_results
        if item.enabled and item.applicable and item.passed is False
    )
    assert candidate.rejection_reasons == expected
    with pytest.raises(ValueError, match="criterion order"):
        replace(candidate, rejection_reasons=tuple(reversed(expected)))


def test_valid_accepted_and_rejected_candidates_remain_constructible() -> None:
    accepted = _valid_accepted_candidate()
    rejected = _valid_rejected_candidate()
    assert accepted.accepted and not accepted.rejection_reasons
    assert not rejected.accepted and rejected.rejection_reasons


def test_missing_frequency_is_an_auditable_structural_criterion() -> None:
    characterization = characterize_spectral_track(
        _track((np.nan, np.inf), refined=(None, None))
    )
    _, tracking = _context()
    candidate = evaluate_modal_candidate(characterization, tracking, _settings())
    structural = _criterion(candidate, "representative_frequency_hz")
    assert not candidate.accepted
    assert structural.enabled and structural.applicable and structural.passed is False
    assert candidate.rejection_reasons == (structural.reason,)


def test_accepted_candidate_rejects_structural_rejection_diagnostic() -> None:
    accepted = _valid_accepted_candidate()
    with pytest.raises(ValueError, match="rejection diagnostics"):
        replace(
            accepted,
            diagnostics=accepted.diagnostics + ("structural_rejection:invalid",),
        )


def test_public_evaluation_and_selection_satisfy_closed_contract() -> None:
    characterization, tracking = _context()
    evaluated = evaluate_modal_candidate(characterization, tracking)
    first = select_modal_candidates((characterization,), tracking)
    second = select_modal_candidates((characterization,), tracking)
    assert evaluated.accepted
    assert evaluated.acceptance_reasons == tuple(
        item.reason
        for item in evaluated.criteria_results
        if item.enabled and item.applicable and item.passed is True
    )
    assert first == second
    assert first[0].candidate_id == 0
