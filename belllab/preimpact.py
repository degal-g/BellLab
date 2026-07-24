"""Evidência operacional pré/pós-impacto para trajetórias espectrais."""

from __future__ import annotations

from collections.abc import Iterable
from math import isclose, isfinite, log10

import numpy as np

from belllab.config import PreImpactAnalysisSettings
from belllab.types import ModalCandidate, PreImpactEvidence, SpectralTrack, SpectralTrackCharacterization


def analyze_preimpact_evidence(
    track: SpectralTrack,
    characterization: SpectralTrackCharacterization,
    impact_time_s: float,
    settings: PreImpactAnalysisSettings | None = None,
) -> PreImpactEvidence:
    """Compara níveis robustos em janelas relativas ao impacto."""
    cfg = settings or PreImpactAnalysisSettings()
    if track.track_id != characterization.track_id:
        raise ValueError("track and characterization IDs must match.")
    if track.amplitude_unit != characterization.amplitude_unit:
        raise ValueError("track and characterization amplitude units must match.")
    if not isfinite(impact_time_s):
        raise ValueError("impact_time_s must be finite.")

    times = np.asarray(track.times_s, dtype=float)
    amplitudes = np.asarray(track.amplitudes, dtype=float)
    relative_times = times - impact_time_s
    pre_mask = (
        (relative_times >= cfg.preimpact_window_start_s)
        & (relative_times <= cfg.preimpact_window_end_s)
    )
    post_mask = (
        (relative_times >= cfg.postimpact_window_start_s)
        & (relative_times <= cfg.postimpact_window_end_s)
    )
    pre_times_all, pre_values_all = relative_times[pre_mask], amplitudes[pre_mask]
    post_times_all, post_values_all = relative_times[post_mask], amplitudes[post_mask]
    pre_finite = np.isfinite(pre_times_all) & np.isfinite(pre_values_all)
    post_finite = np.isfinite(post_times_all) & np.isfinite(post_values_all)
    pre_times, pre_values = pre_times_all[pre_finite], pre_values_all[pre_finite]
    post_times, post_values = post_times_all[post_finite], post_values_all[post_finite]
    diagnostics: list[str] = []
    if int((~pre_finite).sum()):
        diagnostics.append("nonfinite_preimpact_values_discarded")
    if int((~post_finite).sum()):
        diagnostics.append("nonfinite_postimpact_values_discarded")

    pre_mean, pre_median, pre_variability, pre_slope = _window_statistics(
        pre_times, pre_values
    )
    post_mean, post_median, post_variability, post_slope = _window_statistics(
        post_times, post_values
    )
    coverage = _window_coverage(
        pre_times,
        cfg.preimpact_window_start_s,
        cfg.preimpact_window_end_s,
    )
    pre_sufficient = (
        len(pre_values) >= cfg.minimum_preimpact_point_count
        and coverage >= cfg.minimum_preimpact_coverage_fraction
    )
    post_sufficient = len(post_values) >= cfg.minimum_postimpact_point_count
    level_detected = (
        pre_median is not None
        and (
            cfg.minimum_preimpact_level is None
            or pre_median >= cfg.minimum_preimpact_level
        )
    )
    pre_detected = pre_sufficient and level_detected
    pre_decay = (
        pre_slope is not None
        and pre_slope < -cfg.preimpact_decay_slope_tolerance
    )
    post_decay = (
        post_slope is not None
        and post_slope < -cfg.postimpact_decay_slope_tolerance
    )

    level_change = (
        post_median - pre_median
        if pre_median is not None and post_median is not None
        else None
    )
    change_db: float | None = None
    ratio: float | None = None
    if pre_median is not None and post_median is not None:
        if track.amplitude_unit == "dbfs_amplitude":
            change_db = post_median - pre_median
            try:
                derived_ratio = 10.0 ** (change_db / 20.0)
            except OverflowError:
                derived_ratio = float("inf")
            if isfinite(derived_ratio) and derived_ratio > 0:
                ratio = derived_ratio
            else:
                diagnostics.append("dbfs_derived_ratio_unavailable")
        elif pre_median > 0 and post_median > 0:
            ratio = post_median / pre_median
            change_db = 20.0 * log10(ratio)
        else:
            diagnostics.append("linear_level_change_db_unavailable")

    if not post_sufficient:
        success = False
        failure_reason = "insufficient_postimpact_data"
        classification = "insufficient_postimpact_data"
        excited = False
        background = pre_detected
        diagnostics.append("insufficient_postimpact_data")
    elif pre_values_all.size > 0 and not pre_sufficient:
        success = False
        failure_reason = "insufficient_preimpact_data"
        classification = "insufficient_preimpact_data"
        excited = False
        background = False
        diagnostics.append("insufficient_preimpact_data")
    else:
        success = True
        failure_reason = None
        absent_preimpact = pre_values_all.size == 0 or not pre_detected
        increase_excited = (
            change_db is not None
            and (
                change_db >= cfg.minimum_impact_level_increase_db
                or isclose(
                    change_db,
                    cfg.minimum_impact_level_increase_db,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
        )
        excited = (
            (absent_preimpact and cfg.absent_preimpact_is_excited)
            or increase_excited
        )
        if cfg.require_postimpact_decay and not post_decay:
            excited = False
            diagnostics.append("required_postimpact_decay_not_detected")
        if absent_preimpact:
            classification = (
                "impact_emergent" if excited else "not_detected_preimpact"
            )
            background = False
        elif excited and pre_decay:
            classification = "reexcited_preexisting_component"
            background = False
        elif excited:
            classification = "impact_amplified"
            background = False
        elif pre_decay:
            classification = "preexisting_decay"
            background = True
        elif change_db is not None and abs(change_db) <= cfg.unchanged_level_tolerance_db:
            classification = "persistent_background_tone"
            background = True
        else:
            classification = "indeterminate"
            background = pre_detected

    return PreImpactEvidence(
        source_track_id=track.track_id,
        impact_time_s=float(impact_time_s),
        amplitude_unit=track.amplitude_unit,
        preimpact_available_point_count=len(pre_values_all),
        preimpact_finite_point_count=len(pre_values),
        postimpact_available_point_count=len(post_values_all),
        postimpact_finite_point_count=len(post_values),
        preimpact_coverage_fraction=coverage,
        preimpact_detected=pre_detected,
        preimpact_level=pre_mean,
        preimpact_median_level=pre_median,
        preimpact_variability=pre_variability,
        preimpact_slope_per_s=pre_slope,
        postimpact_initial_level=float(post_values[0]) if len(post_values) else None,
        postimpact_level=post_mean,
        postimpact_median_level=post_median,
        postimpact_variability=post_variability,
        postimpact_slope_per_s=post_slope,
        postimpact_decay_detected=post_decay,
        impact_level_change=level_change,
        impact_level_change_db=change_db,
        post_to_pre_ratio=ratio,
        impact_excited=excited,
        background_contaminated=background,
        preimpact_decay_detected=pre_decay,
        classification=classification,
        success=success,
        failure_reason=failure_reason,
        diagnostics=tuple(diagnostics),
    )


def analyze_candidates_preimpact(
    candidates: Iterable[ModalCandidate],
    tracks: Iterable[SpectralTrack],
    impact_time_s: float,
    settings: PreImpactAnalysisSettings | None = None,
) -> tuple[PreImpactEvidence, ...]:
    """Analisa candidatos aceitos e rejeitados sem alterar sua ordem ou estado."""
    by_id = {track.track_id: track for track in tracks}
    ordered_candidates = tuple(candidates)
    evidence: list[PreImpactEvidence] = []
    for candidate in ordered_candidates:
        track = by_id.get(candidate.source_track_id)
        if track is None:
            raise ValueError(
                f"missing source track {candidate.source_track_id} for candidate."
            )
        evidence.append(
            analyze_preimpact_evidence(
                track,
                candidate.characterization,
                impact_time_s,
                settings,
            )
        )
    return tuple(evidence)


def _window_statistics(
    times: np.ndarray,
    values: np.ndarray,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Retorna média, mediana, desvio padrão e inclinação no domínio original."""
    if not len(values):
        return None, None, None, None
    slope = (
        float(np.polyfit(times, values, 1)[0])
        if len(values) >= 2 and np.ptp(times) > 0
        else None
    )
    return (
        float(np.mean(values)),
        float(np.median(values)),
        float(np.std(values)),
        slope,
    )


def _window_coverage(times: np.ndarray, start: float, end: float) -> float:
    """Mede a fração temporal abrangida por observações finitas na janela."""
    if len(times) < 2:
        return 0.0
    return min(1.0, max(0.0, float((times[-1] - times[0]) / (end - start))))
