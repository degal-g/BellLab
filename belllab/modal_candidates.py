"""Promoção operacional auditável de trajetórias, sem construir modos físicos."""

from __future__ import annotations

from collections.abc import Iterable
from math import isclose

from belllab.config import ModalCandidateSettings
from belllab.results import SpectralTrackingResults
from belllab.types import (
    CandidateCriterionResult,
    ModalCandidate,
    PreImpactEvidence,
    SpectralTrackCharacterization,
)


def evaluate_modal_candidate(
    characterization: SpectralTrackCharacterization,
    tracking_results: SpectralTrackingResults,
    settings: ModalCandidateSettings | None = None,
    *,
    candidate_id: int = 0,
    preimpact_evidence: PreImpactEvidence | None = None,
) -> ModalCandidate:
    """Avalia uma caracterização por critérios explícitos e preserva rejeições."""
    cfg = settings or ModalCandidateSettings()
    if not isinstance(characterization, SpectralTrackCharacterization):
        raise TypeError("characterization must be SpectralTrackCharacterization.")
    if (
        preimpact_evidence is not None
        and preimpact_evidence.source_track_id != characterization.track_id
    ):
        raise ValueError("preimpact evidence track ID must match characterization.")
    diagnostics_for_track = tuple(
        item
        for item in tracking_results.assignment_diagnostics
        if item.track_id == characterization.track_id
    )
    assignment_count = len(diagnostics_for_track)
    ambiguous_count = sum(item.ambiguous for item in diagnostics_for_track)
    near_count = sum(item.near_threshold for item in diagnostics_for_track)
    ambiguous_fraction = ambiguous_count / assignment_count if assignment_count else None
    near_fraction = near_count / assignment_count if assignment_count else None
    margins = tuple(
        item.assignment_margin
        for item in diagnostics_for_track
        if item.assignment_margin is not None
    )
    minimum_margin = min(margins) if margins else None
    representative = characterization.frequency_median_hz
    if representative is None or representative <= 0:
        representative = characterization.frequency_mean_hz
    if representative is not None and representative <= 0:
        representative = None

    criteria: list[CandidateCriterionResult] = []
    criteria.append(
        _criterion(
            "representative_frequency_hz",
            representative,
            ">",
            0.0,
            enabled=True,
            passed=representative is not None,
            missing_is_failure=True,
        )
    )
    criteria.extend((
        _limit(
            "minimum_observation_count",
            characterization.observation_count,
            ">=",
            cfg.minimum_observation_count,
        ),
        _limit(
            "minimum_coverage_fraction",
            characterization.coverage_fraction,
            ">=",
            cfg.minimum_coverage_fraction,
        ),
        _limit(
            "minimum_duration_s",
            characterization.observed_duration_s,
            ">=",
            cfg.minimum_duration_s,
        ),
        _limit(
            "maximum_relative_frequency_stability",
            characterization.relative_frequency_stability,
            "<=",
            cfg.maximum_relative_frequency_stability,
        ),
        _limit(
            "maximum_absolute_frequency_drift_hz",
            (
                abs(characterization.frequency_total_drift_hz)
                if characterization.frequency_total_drift_hz is not None
                else None
            ),
            "<=",
            cfg.maximum_absolute_frequency_drift_hz,
        ),
        _limit(
            "maximum_frequency_fit_rmse_hz",
            characterization.frequency_fit.rmse_hz,
            "<=",
            cfg.maximum_frequency_fit_rmse_hz,
        ),
        _boolean_requirement(
            "require_successful_frequency_fit",
            characterization.frequency_fit.success,
            cfg.require_successful_frequency_fit,
        ),
        _boolean_requirement(
            "require_amplitude_decay",
            characterization.amplitude_fit.decay_detected,
            cfg.require_amplitude_decay,
        ),
        _limit(
            "minimum_amplitude_fit_r_squared",
            characterization.amplitude_fit.r_squared,
            ">=",
            cfg.minimum_amplitude_fit_r_squared,
        ),
        _limit(
            "minimum_decay_tau_s",
            characterization.amplitude_fit.tau_s,
            ">=",
            cfg.minimum_decay_tau_s,
        ),
        _limit(
            "maximum_decay_tau_s",
            characterization.amplitude_fit.tau_s,
            "<=",
            cfg.maximum_decay_tau_s,
        ),
        _limit(
            "maximum_ambiguous_assignment_fraction",
            ambiguous_fraction,
            "<=",
            cfg.maximum_ambiguous_assignment_fraction,
        ),
        _limit(
            "maximum_near_threshold_assignment_fraction",
            near_fraction,
            "<=",
            cfg.maximum_near_threshold_assignment_fraction,
        ),
        _limit(
            "minimum_assignment_margin",
            minimum_margin,
            ">=",
            cfg.minimum_assignment_margin,
            missing_is_not_applicable=True,
        ),
        _criterion(
            "allow_mixed_frequency_source",
            characterization.frequency_source,
            "!=",
            "mixed",
            enabled=not cfg.allow_mixed_frequency_source,
            passed=characterization.frequency_source != "mixed",
        ),
        _boolean_requirement(
            "require_reaches_final_frame",
            characterization.reached_analysis_final_frame,
            cfg.require_reaches_final_frame,
        ),
        _boolean_requirement(
            "require_impact_excitation",
            (
                preimpact_evidence.impact_excited
                if preimpact_evidence is not None
                else None
            ),
            cfg.require_impact_excitation,
        ),
        _criterion(
            "reject_persistent_background_tone",
            (
                preimpact_evidence.classification
                if preimpact_evidence is not None
                else None
            ),
            "!=",
            "persistent_background_tone",
            enabled=cfg.reject_persistent_background_tone,
            passed=(
                preimpact_evidence is not None
                and preimpact_evidence.classification
                != "persistent_background_tone"
            ),
            missing_is_failure=preimpact_evidence is None,
        ),
        _minimum_impact_increase_criterion(
            (
                preimpact_evidence.impact_level_change_db
                if preimpact_evidence is not None
                else None
            ),
            cfg.minimum_impact_level_increase_db,
        ),
    ))

    failed = tuple(
        item for item in criteria
        if item.enabled and item.applicable and item.passed is False
    )
    passed = tuple(
        item for item in criteria
        if item.enabled and item.applicable and item.passed is True
    )
    diagnostics = ["not_a_physical_mode", "operational_modal_candidate"]
    if assignment_count == 0:
        diagnostics.append("no_auditable_assignments")
    if cfg.minimum_assignment_margin is not None and minimum_margin is None:
        diagnostics.append("assignment_margin_not_applicable")
    accepted = not failed
    return ModalCandidate(
        candidate_id=candidate_id,
        source_track_id=characterization.track_id,
        characterization=characterization,
        representative_frequency_hz=representative,
        accepted_assignment_count=assignment_count,
        ambiguous_assignment_count=ambiguous_count,
        near_threshold_assignment_count=near_count,
        ambiguous_assignment_fraction=ambiguous_fraction,
        near_threshold_assignment_fraction=near_fraction,
        minimum_assignment_margin=minimum_margin,
        accepted=accepted,
        criteria_results=tuple(criteria),
        acceptance_reasons=tuple(item.reason for item in passed),
        rejection_reasons=tuple(item.reason for item in failed),
        diagnostics=tuple(diagnostics),
    )


def select_modal_candidates(
    characterizations: Iterable[SpectralTrackCharacterization],
    tracking_results: SpectralTrackingResults,
    settings: ModalCandidateSettings | None = None,
    preimpact_evidence_by_track: dict[int, PreImpactEvidence] | None = None,
) -> tuple[ModalCandidate, ...]:
    """Avalia todas as trajetórias em ordem estável de ``track_id``."""
    ordered = tuple(sorted(characterizations, key=lambda item: item.track_id))
    track_ids = tuple(item.track_id for item in ordered)
    if len(track_ids) != len(set(track_ids)):
        raise ValueError("characterizations must have unique track IDs.")
    return tuple(
        evaluate_modal_candidate(
            characterization,
            tracking_results,
            settings,
            candidate_id=candidate_id,
            preimpact_evidence=(
                preimpact_evidence_by_track.get(characterization.track_id)
                if preimpact_evidence_by_track is not None
                else None
            ),
        )
        for candidate_id, characterization in enumerate(ordered)
    )


def _limit(
    name: str,
    observed: float | int | None,
    operator: str,
    threshold: float | int | None,
    *,
    missing_is_not_applicable: bool = False,
) -> CandidateCriterionResult:
    """Avalia um limite opcional sem esconder ausência do valor observado."""
    if threshold is None:
        return _criterion(name, observed, operator, None, enabled=False, passed=None)
    if observed is None:
        if missing_is_not_applicable:
            return CandidateCriterionResult(
                name, None, operator, threshold, True, False, None,
                f"{name}: not_applicable",
            )
        return _criterion(
            name, None, operator, threshold, enabled=True, passed=False,
            missing_is_failure=True,
        )
    passed = observed >= threshold if operator == ">=" else observed <= threshold
    return _criterion(name, observed, operator, threshold, enabled=True, passed=passed)


def _boolean_requirement(
    name: str,
    observed: bool | None,
    enabled: bool,
) -> CandidateCriterionResult:
    """Avalia requisitos booleanos somente quando explicitamente habilitados."""
    if not enabled:
        return _criterion(name, observed, "is", True, enabled=False, passed=None)
    return _criterion(
        name,
        observed,
        "is",
        True,
        enabled=True,
        passed=observed is True,
        missing_is_failure=observed is None,
    )


def _minimum_impact_increase_criterion(
    observed: float | None,
    threshold: float | None,
) -> CandidateCriterionResult:
    """Aplica inclusão numérica estável ao limite de aumento em dB."""
    name = "minimum_impact_level_increase_db"
    if threshold is None:
        return _criterion(name, observed, ">=", None, enabled=False, passed=None)
    if observed is None:
        return _criterion(
            name, None, ">=", threshold, enabled=True, passed=False,
            missing_is_failure=True,
        )
    passed = observed >= threshold or isclose(
        observed, threshold, rel_tol=1e-12, abs_tol=1e-12
    )
    return _criterion(
        name, observed, ">=", threshold, enabled=True, passed=passed
    )


def _criterion(
    name: str,
    observed: float | int | bool | str | None,
    operator: str,
    threshold: float | int | bool | str | None,
    *,
    enabled: bool,
    passed: bool | None,
    missing_is_failure: bool = False,
) -> CandidateCriterionResult:
    """Constrói texto curto a partir do resultado estruturado canônico."""
    if not enabled:
        return CandidateCriterionResult(
            name, observed, operator, threshold, False, False, None,
            f"{name}: disabled",
        )
    if missing_is_failure and observed is None:
        reason = f"{name}: unavailable"
    else:
        status = "passed" if passed else "failed"
        reason = f"{name}: {status} ({observed!r} {operator} {threshold!r})"
    return CandidateCriterionResult(
        name, observed, operator, threshold, True, True, passed, reason
    )
