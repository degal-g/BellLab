"""Associação auditável de candidatos entre repetições da mesma condição."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite, log2
from statistics import mean, median, pstdev
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from belllab.types import CandidateCriterionResult, ModalCandidate, PreImpactEvidence


DYNAMIC_LABELS = frozenset({"pp", "p", "mf", "f", "ff", "unspecified"})
_UNMATCHED_REASONS = frozenset({
    "no_candidate_in_frequency_range",
    "cost_above_threshold",
    "ambiguous_match",
    "candidate_rejected_by_policy",
    "missing_required_evidence",
    "no_compatible_cluster",
})


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")


def _finite_optional(value: float | None, name: str, *, nonnegative: bool = False) -> None:
    if value is not None and (
        not isfinite(value) or (nonnegative and value < 0)
    ):
        qualifier = "finite and non-negative" if nonnegative else "finite"
        raise ValueError(f"{name} must be {qualifier} when provided.")


def _strings(values: tuple[str, ...], name: str) -> None:
    if not isinstance(values, tuple) or any(
        not isinstance(item, str) or not item.strip() for item in values
    ) or len(values) != len(set(values)):
        raise ValueError(f"{name} must contain unique nonempty strings.")


@dataclass(frozen=True, slots=True)
class ExcitationCondition:
    """Categoria experimental; ``dynamic_label`` não é uma medida absoluta."""

    dynamic_label: str
    repeat_index: int
    measured_peak: float | None = None
    measured_rms: float | None = None
    measured_energy: float | None = None
    amplitude_unit: str | None = None
    impact_location: str | None = None
    exciter_type: str | None = None
    acquisition_gain: float | None = None
    session_id: str | None = None
    microphone_id: str | None = None
    interface_id: str | None = None
    channel: int | None = None
    microphone_distance_m: float | None = None
    microphone_orientation: str | None = None
    exciter_mass_kg: float | None = None
    operator_label: str | None = None
    notes: str | None = None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.dynamic_label not in DYNAMIC_LABELS:
            raise ValueError(f"dynamic_label must be one of {sorted(DYNAMIC_LABELS)}.")
        if self.repeat_index < 0:
            raise ValueError("repeat_index must not be negative.")
        for name in ("measured_peak", "measured_rms", "measured_energy", "acquisition_gain"):
            _finite_optional(getattr(self, name), name)
        if self.measured_rms is not None and self.measured_rms < 0:
            raise ValueError("measured_rms must not be negative.")
        if self.measured_energy is not None and self.measured_energy < 0:
            raise ValueError("measured_energy must not be negative.")
        if self.channel is not None and self.channel < 0:
            raise ValueError("channel must not be negative.")
        for name in ("microphone_distance_m", "exciter_mass_kg"):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive when provided.")
        for name in (
            "amplitude_unit", "impact_location", "exciter_type", "session_id",
            "microphone_id", "interface_id", "microphone_orientation",
            "operator_label", "notes",
        ):
            value = getattr(self, name)
            if value is not None:
                _text(value, name)
        _strings(self.diagnostics, "condition diagnostics")


@dataclass(frozen=True, slots=True)
class RecordingCandidateSet:
    """Candidatos e evidências pertencentes a uma gravação identificável."""

    recording_id: str
    condition: ExcitationCondition
    candidates: tuple[ModalCandidate, ...]
    preimpact_evidence: tuple[PreImpactEvidence, ...] = ()

    def __post_init__(self) -> None:
        _text(self.recording_id, "recording_id")
        if not isinstance(self.candidates, tuple) or not isinstance(
            self.preimpact_evidence, tuple
        ):
            raise ValueError("candidates and preimpact_evidence must be immutable tuples.")
        candidate_ids = tuple(item.candidate_id for item in self.candidates)
        track_ids = tuple(item.source_track_id for item in self.candidates)
        evidence_ids = tuple(item.source_track_id for item in self.preimpact_evidence)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate IDs must be unique within a recording.")
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("source track IDs must be unique within a recording.")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("preimpact evidence track IDs must be unique.")
        if not set(evidence_ids).issubset(track_ids):
            raise ValueError("preimpact evidence must refer to a candidate source track.")


@dataclass(frozen=True, slots=True)
class CandidateReference:
    """Referência mínima e suficiente para localizar e comparar um candidato."""

    recording_id: str
    candidate_id: int
    source_track_id: int
    representative_frequency_hz: float
    dynamic_label: str
    accepted: bool
    impact_excited: bool | None = None
    classification: str | None = None
    frequency_stability: float | None = None
    amplitude_tau_s: float | None = None
    amplitude_fit_r_squared: float | None = None
    frequency_drift_hz: float | None = None
    frequency_fit_rmse_hz: float | None = None
    coverage_fraction: float | None = None
    ambiguous_assignment_fraction: float | None = None
    near_threshold_assignment_fraction: float | None = None
    minimum_assignment_margin: float | None = None
    preimpact_classification: str | None = None
    structure_descriptor: str | None = None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.recording_id, "recording_id")
        if self.dynamic_label not in DYNAMIC_LABELS:
            raise ValueError("candidate reference has an unknown dynamic_label.")
        if self.candidate_id < 0 or self.source_track_id < 0:
            raise ValueError("candidate and source track IDs must not be negative.")
        if not isfinite(self.representative_frequency_hz) or self.representative_frequency_hz <= 0:
            raise ValueError("representative_frequency_hz must be finite and positive.")
        for name in ("frequency_stability", "amplitude_tau_s", "amplitude_fit_r_squared"):
            _finite_optional(getattr(self, name), name, nonnegative=True)
        for name in (
            "frequency_fit_rmse_hz",
            "minimum_assignment_margin",
        ):
            _finite_optional(getattr(self, name), name, nonnegative=True)
        _finite_optional(self.frequency_drift_hz, "frequency_drift_hz")
        if self.amplitude_tau_s is not None and self.amplitude_tau_s <= 0:
            raise ValueError("amplitude_tau_s must be positive when provided.")
        for name in (
            "coverage_fraction",
            "ambiguous_assignment_fraction",
            "near_threshold_assignment_fraction",
        ):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{name} must be finite and in [0, 1].")
        if self.classification is not None:
            _text(self.classification, "classification")
        if self.preimpact_classification is not None:
            _text(self.preimpact_classification, "preimpact_classification")
        if self.structure_descriptor is not None:
            _text(self.structure_descriptor, "structure_descriptor")
        _strings(self.diagnostics, "candidate reference diagnostics")


@dataclass(frozen=True, slots=True)
class WithinConditionAssociationSettings:
    """Limites e pesos explícitos da associação dentro de uma condição."""

    maximum_absolute_frequency_difference_hz: float | None = 2.0
    maximum_relative_frequency_difference: float | None = None
    maximum_log_frequency_difference: float | None = None
    frequency_cost_weight: float = 1.0
    frequency_stability_cost_weight: float = 0.0
    decay_tau_cost_weight: float = 0.0
    amplitude_decay_quality_cost_weight: float = 0.0
    impact_evidence_cost_weight: float = 0.0
    maximum_association_cost: float = 1.0
    ambiguity_margin_threshold: float = 0.1
    minimum_repeat_count: int = 2
    minimum_repeat_coverage_fraction: float = 0.5
    maximum_cluster_relative_frequency_dispersion: float | None = None
    allow_rejected_candidates: bool = False
    require_impact_excitation: bool = False
    allow_missing_tau: bool = True
    allow_missing_preimpact_evidence: bool = True
    reject_persistent_background_tone: bool = False

    def __post_init__(self) -> None:
        gates = (
            self.maximum_absolute_frequency_difference_hz,
            self.maximum_relative_frequency_difference,
            self.maximum_log_frequency_difference,
        )
        for name in (
            "maximum_absolute_frequency_difference_hz",
            "maximum_relative_frequency_difference",
            "maximum_log_frequency_difference",
        ):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive when provided.")
        _finite_optional(
            self.maximum_cluster_relative_frequency_dispersion,
            "maximum_cluster_relative_frequency_dispersion",
            nonnegative=True,
        )
        if not any(value is not None and value > 0 for value in gates):
            raise ValueError("at least one positive frequency difference limit is required.")
        for name in (
            "frequency_cost_weight",
            "frequency_stability_cost_weight",
            "decay_tau_cost_weight",
            "amplitude_decay_quality_cost_weight",
            "impact_evidence_cost_weight",
            "ambiguity_margin_threshold",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.frequency_cost_weight <= 0:
            raise ValueError("frequency_cost_weight must be positive.")
        if not isfinite(self.maximum_association_cost) or self.maximum_association_cost <= 0:
            raise ValueError("maximum_association_cost must be finite and positive.")
        if self.minimum_repeat_count < 0:
            raise ValueError("minimum_repeat_count must not be negative.")
        if not isfinite(self.minimum_repeat_coverage_fraction) or not (
            0 <= self.minimum_repeat_coverage_fraction <= 1
        ):
            raise ValueError("minimum_repeat_coverage_fraction must be finite and in [0, 1].")
        for name in (
            "allow_rejected_candidates",
            "require_impact_excitation",
            "allow_missing_tau",
            "allow_missing_preimpact_evidence",
            "reject_persistent_background_tone",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")


@dataclass(frozen=True, slots=True)
class CrossRecordingAssociationDiagnostic:
    """Custo, gates e margens de um par entre gravações distintas."""

    left_recording_id: str
    left_candidate_id: int
    right_recording_id: str
    right_candidate_id: int
    frequency_difference_hz: float
    relative_frequency_difference: float
    log_frequency_difference: float
    frequency_cost_component: float
    stability_cost_component: float
    tau_cost_component: float
    amplitude_decay_quality_cost_component: float
    impact_evidence_cost_component: float
    total_cost: float
    admissible: bool
    selected: bool
    row_assignment_margin: float | None
    column_assignment_margin: float | None
    assignment_margin: float | None
    ambiguous: bool
    rejection_reason: str | None = None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.left_recording_id == self.right_recording_id:
            raise ValueError("cross-recording association requires different recordings.")
        for name in (
            "frequency_difference_hz", "relative_frequency_difference",
            "log_frequency_difference", "frequency_cost_component",
            "stability_cost_component", "tau_cost_component",
            "amplitude_decay_quality_cost_component",
            "impact_evidence_cost_component", "total_cost",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        expected = (
            self.frequency_cost_component + self.stability_cost_component
            + self.tau_cost_component + self.amplitude_decay_quality_cost_component
            + self.impact_evidence_cost_component
        )
        if not isclose(self.total_cost, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("total_cost must equal the sum of cost components.")
        if self.selected and not self.admissible:
            raise ValueError("selected association must be admissible.")
        margins = (self.row_assignment_margin, self.column_assignment_margin, self.assignment_margin)
        if any(value is not None and (not isfinite(value) or value < 0) for value in margins):
            raise ValueError("assignment margins must be finite and non-negative.")
        available = tuple(value for value in margins[:2] if value is not None)
        expected_margin = min(available) if available else None
        if self.assignment_margin != expected_margin:
            raise ValueError("assignment_margin must equal the minimum available margin.")
        if self.ambiguous and (not self.selected or self.assignment_margin is None):
            raise ValueError("ambiguous requires a selected association with a margin.")
        if self.rejection_reason is not None:
            _text(self.rejection_reason, "rejection_reason")
        _strings(self.diagnostics, "association diagnostics")


@dataclass(frozen=True, slots=True)
class UnmatchedCandidate:
    """Candidato preservado sem correspondência operacional."""

    reference: CandidateReference
    reason: str
    best_alternatives: tuple[CandidateReference, ...] = ()
    minimum_cost_observed: float | None = None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.reason not in _UNMATCHED_REASONS:
            raise ValueError("unknown unmatched candidate reason.")
        if not isinstance(self.best_alternatives, tuple):
            raise ValueError("best_alternatives must be an immutable tuple.")
        _finite_optional(self.minimum_cost_observed, "minimum_cost_observed", nonnegative=True)
        _strings(self.diagnostics, "unmatched diagnostics")


@dataclass(frozen=True, slots=True)
class WithinConditionCandidateCluster:
    """Agrupamento reproduzível operacional, ainda não um modo físico."""

    cluster_id: int
    dynamic_label: str
    member_candidate_refs: tuple[CandidateReference, ...]
    recording_ids: tuple[str, ...]
    representative_frequency_hz: float
    frequency_mean_hz: float
    frequency_median_hz: float
    frequency_std_hz: float
    frequency_min_hz: float
    frequency_max_hz: float
    frequency_span_hz: float
    relative_frequency_dispersion: float
    member_count: int
    recording_count: int
    repeat_coverage_fraction: float
    reproducible: bool
    ambiguous: bool
    accepted: bool
    criteria_results: tuple[CandidateCriterionResult, ...]
    rejection_reasons: tuple[str, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.cluster_id < 0 or not self.member_candidate_refs:
            raise ValueError("cluster requires a non-negative ID and at least one member.")
        if self.dynamic_label not in DYNAMIC_LABELS:
            raise ValueError("cluster has an unknown dynamic_label.")
        if len({item.recording_id for item in self.member_candidate_refs}) != len(
            self.member_candidate_refs
        ):
            raise ValueError("cluster cannot contain two candidates from one recording.")
        if any(item.dynamic_label != self.dynamic_label for item in self.member_candidate_refs):
            raise ValueError("cluster members must share its dynamic_label.")
        expected_ids = tuple(sorted(item.recording_id for item in self.member_candidate_refs))
        if self.recording_ids != expected_ids:
            raise ValueError("recording_ids must be sorted and match cluster members.")
        if self.member_count != len(self.member_candidate_refs) or self.recording_count != len(expected_ids):
            raise ValueError("cluster counts must match its members.")
        values = (
            self.representative_frequency_hz, self.frequency_mean_hz,
            self.frequency_median_hz, self.frequency_std_hz, self.frequency_min_hz,
            self.frequency_max_hz, self.frequency_span_hz,
            self.relative_frequency_dispersion,
        )
        if any(not isfinite(value) or value < 0 for value in values):
            raise ValueError("cluster frequency metrics must be finite and non-negative.")
        if not self.frequency_min_hz <= self.frequency_median_hz <= self.frequency_max_hz:
            raise ValueError("cluster frequency median must lie within its range.")
        if self.representative_frequency_hz != self.frequency_median_hz:
            raise ValueError("cluster representative frequency must be the median.")
        if not isclose(self.frequency_span_hz, self.frequency_max_hz - self.frequency_min_hz):
            raise ValueError("frequency_span_hz is inconsistent with min/max.")
        if not isfinite(self.repeat_coverage_fraction) or not 0 <= self.repeat_coverage_fraction <= 1:
            raise ValueError("repeat_coverage_fraction must be finite and in [0, 1].")
        names = tuple(item.criterion for item in self.criteria_results)
        if len(names) != len(set(names)):
            raise ValueError("cluster criteria must have unique names.")
        failed = tuple(
            item.reason for item in self.criteria_results
            if item.enabled and item.applicable and item.passed is False
        )
        if self.rejection_reasons != failed:
            raise ValueError("cluster rejection reasons must project failed criteria.")
        if self.reproducible != (not failed) or self.accepted != self.reproducible:
            raise ValueError("cluster reproducibility must follow explicit criteria.")
        _strings(self.rejection_reasons, "cluster rejection reasons")
        _strings(self.diagnostics, "cluster diagnostics")


@dataclass(frozen=True, slots=True)
class WithinConditionAssociationResult:
    """Resultado completo, incluindo agrupamentos, perdas evitadas e custos."""

    dynamic_label: str
    recording_ids: tuple[str, ...]
    candidate_references: tuple[CandidateReference, ...]
    clusters: tuple[WithinConditionCandidateCluster, ...]
    unmatched_candidates: tuple[UnmatchedCandidate, ...]
    association_diagnostics: tuple[CrossRecordingAssociationDiagnostic, ...]
    settings: WithinConditionAssociationSettings
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.dynamic_label not in DYNAMIC_LABELS:
            raise ValueError("result has an unknown dynamic_label.")
        if self.recording_ids != tuple(sorted(self.recording_ids)) or len(
            self.recording_ids
        ) != len(set(self.recording_ids)):
            raise ValueError("recording_ids must be sorted and unique.")
        source_keys = tuple(_key(item) for item in self.candidate_references)
        if len(source_keys) != len(set(source_keys)):
            raise ValueError("candidate references must be unique.")
        placed = tuple(
            _key(item)
            for cluster in self.clusters
            for item in cluster.member_candidate_refs
        ) + tuple(_key(item.reference) for item in self.unmatched_candidates)
        if sorted(placed) != sorted(source_keys):
            raise ValueError("every candidate reference must appear exactly once.")
        if tuple(item.cluster_id for item in self.clusters) != tuple(range(len(self.clusters))):
            raise ValueError("cluster IDs must be contiguous and deterministic.")
        pairs = tuple(
            (item.left_recording_id, item.left_candidate_id,
             item.right_recording_id, item.right_candidate_id)
            for item in self.association_diagnostics
        )
        if len(pairs) != len(set(pairs)):
            raise ValueError("association diagnostic pairs must be unique.")
        _strings(self.diagnostics, "result diagnostics")


def associate_candidates_within_condition(
    recordings: Iterable[RecordingCandidateSet],
    settings: WithinConditionAssociationSettings | None = None,
) -> WithinConditionAssociationResult:
    """Associa progressivamente repetições de uma única condição dinâmica."""
    cfg = settings or WithinConditionAssociationSettings()
    ordered = tuple(sorted(recordings, key=lambda item: item.recording_id))
    if not ordered:
        raise ValueError("at least one recording is required.")
    recording_ids = tuple(item.recording_id for item in ordered)
    if len(recording_ids) != len(set(recording_ids)):
        raise ValueError("recording IDs must be unique.")
    labels = {item.condition.dynamic_label for item in ordered}
    if len(labels) != 1:
        raise ValueError("within-condition association cannot mix dynamic labels.")
    repeats = tuple(item.condition.repeat_index for item in ordered)
    if len(repeats) != len(set(repeats)):
        raise ValueError("repeat indices must be unique within a condition.")
    label = next(iter(labels))

    all_refs: list[CandidateReference] = []
    eligible_by_recording: list[list[CandidateReference]] = []
    unmatched: list[UnmatchedCandidate] = []
    for item in ordered:
        evidence = {value.source_track_id: value for value in item.preimpact_evidence}
        eligible: list[CandidateReference] = []
        for candidate in sorted(item.candidates, key=lambda value: value.candidate_id):
            if candidate.representative_frequency_hz is None:
                raise ValueError(
                    "within-condition association requires representative frequencies."
                )
            ref = _reference(item, candidate, evidence.get(candidate.source_track_id))
            all_refs.append(ref)
            reason = _policy_rejection(ref, cfg)
            if reason is None:
                eligible.append(ref)
            else:
                unmatched.append(UnmatchedCandidate(
                    ref, reason, diagnostics=("preserved_without_association",)
                ))
        eligible_by_recording.append(eligible)

    working: list[tuple[list[CandidateReference], bool]] = [
        ([item], False) for item in eligible_by_recording[0]
    ]
    diagnostics: list[CrossRecordingAssociationDiagnostic] = []
    for incoming in eligible_by_recording[1:]:
        if not working:
            working.extend(([item], False) for item in incoming)
            continue
        costs = np.full((len(working), len(incoming)), np.inf)
        drafts: dict[tuple[int, int], CrossRecordingAssociationDiagnostic] = {}
        for row, (members, _) in enumerate(working):
            medoid = _medoid(members)
            for column, ref in enumerate(incoming):
                draft = _compare(medoid, ref, cfg)
                internally_consistent = all(
                    _compare(member, ref, cfg).admissible for member in members
                )
                if not internally_consistent:
                    draft = _replace_diagnostic(
                        draft, admissible=False,
                        rejection_reason="cluster_internal_inconsistency",
                        diagnostics=draft.diagnostics + ("failed_all_member_consistency",),
                    )
                drafts[row, column] = draft
                if draft.admissible:
                    costs[row, column] = draft.total_cost
        selected: set[tuple[int, int]] = set()
        if costs.size and np.isfinite(costs).any():
            safe = np.where(np.isfinite(costs), costs, 1e15)
            rows, columns = linear_sum_assignment(safe)
            selected = {
                (int(row), int(column))
                for row, column in zip(rows, columns)
                if np.isfinite(costs[row, column])
            }
        assigned_columns: set[int] = set()
        for key, draft in drafts.items():
            row, column = key
            if key in selected:
                row_margin, column_margin, margin = _margins(costs, row, column)
                ambiguous = margin is not None and (
                    margin <= cfg.ambiguity_margin_threshold
                    or isclose(
                        margin,
                        cfg.ambiguity_margin_threshold,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                )
                final = _replace_diagnostic(
                    draft, selected=True, row_assignment_margin=row_margin,
                    column_assignment_margin=column_margin, assignment_margin=margin,
                    ambiguous=ambiguous,
                )
                working[row][0].append(incoming[column])
                working[row] = (working[row][0], working[row][1] or ambiguous)
                assigned_columns.add(column)
                diagnostics.append(final)
            else:
                diagnostics.append(draft)
        for column, ref in enumerate(incoming):
            if column not in assigned_columns:
                working.append(([ref], False))

    clusters: list[WithinConditionCandidateCluster] = []
    for members, ambiguous in working:
        if len(members) == 1:
            ref = members[0]
            alternatives = _best_alternatives(ref, diagnostics, all_refs)
            costs = _costs_for(ref, diagnostics)
            reason = (
                "ambiguous_match"
                if any(item.ambiguous and _diagnostic_has(item, ref) for item in diagnostics)
                else "no_compatible_cluster"
            )
            unmatched.append(UnmatchedCandidate(
                ref, reason, alternatives, min(costs) if costs else None,
                ("singleton_not_promoted_to_cluster",),
            ))
        else:
            clusters.append(_cluster(
                len(clusters), label, tuple(members), len(ordered), ambiguous, cfg
            ))
    clusters.sort(key=lambda item: (
        item.representative_frequency_hz,
        tuple(_key(ref) for ref in item.member_candidate_refs),
    ))
    clusters = [
        _cluster(index, item.dynamic_label, item.member_candidate_refs, len(ordered),
                 item.ambiguous, cfg)
        for index, item in enumerate(clusters)
    ]
    return WithinConditionAssociationResult(
        dynamic_label=label,
        recording_ids=recording_ids,
        candidate_references=tuple(sorted(all_refs, key=_key)),
        clusters=tuple(clusters),
        unmatched_candidates=tuple(sorted(unmatched, key=lambda item: _key(item.reference))),
        association_diagnostics=tuple(sorted(diagnostics, key=_diagnostic_key)),
        settings=cfg,
        diagnostics=("within_condition_only", "not_a_validated_physical_mode"),
    )


def group_candidates_by_excitation_condition(
    recordings: Iterable[RecordingCandidateSet],
    settings: WithinConditionAssociationSettings | None = None,
) -> tuple[WithinConditionAssociationResult, ...]:
    """Separa rótulos dinamicamente antes de executar associações independentes."""
    groups: dict[str, list[RecordingCandidateSet]] = {}
    for item in recordings:
        groups.setdefault(item.condition.dynamic_label, []).append(item)
    return tuple(
        associate_candidates_within_condition(groups[label], settings)
        for label in sorted(groups)
    )


def _reference(
    recording: RecordingCandidateSet,
    candidate: ModalCandidate,
    evidence: PreImpactEvidence | None,
) -> CandidateReference:
    return CandidateReference(
        recording.recording_id, candidate.candidate_id, candidate.source_track_id,
        candidate.representative_frequency_hz, recording.condition.dynamic_label,
        candidate.accepted,
        evidence.impact_excited if evidence is not None else None,
        evidence.classification if evidence is not None else None,
        candidate.frequency_stability,
        candidate.amplitude_tau_s,
        candidate.amplitude_fit_r_squared,
        candidate.frequency_drift_hz,
        candidate.frequency_fit_rmse_hz,
        candidate.coverage_fraction,
        candidate.ambiguous_assignment_fraction,
        candidate.near_threshold_assignment_fraction,
        candidate.minimum_assignment_margin,
        evidence.classification if evidence is not None else None,
        None,
        ("within_condition_candidate_reference",),
    )


def _policy_rejection(
    ref: CandidateReference, cfg: WithinConditionAssociationSettings
) -> str | None:
    if not ref.accepted and not cfg.allow_rejected_candidates:
        return "candidate_rejected_by_policy"
    if cfg.require_impact_excitation and ref.impact_excited is not True:
        return "missing_required_evidence"
    if (
        cfg.reject_persistent_background_tone
        and ref.classification == "persistent_background_tone"
    ):
        return "missing_required_evidence"
    if (
        not cfg.allow_missing_preimpact_evidence
        and ref.impact_excited is None
    ):
        return "missing_required_evidence"
    return None


def _compare(
    left: CandidateReference,
    right: CandidateReference,
    cfg: WithinConditionAssociationSettings,
) -> CrossRecordingAssociationDiagnostic:
    difference = abs(left.representative_frequency_hz - right.representative_frequency_hz)
    reference = 0.5 * (
        left.representative_frequency_hz + right.representative_frequency_hz
    )
    relative = difference / reference
    logarithmic = abs(log2(
        left.representative_frequency_hz / right.representative_frequency_hz
    ))
    gate_pairs = (
        (difference, cfg.maximum_absolute_frequency_difference_hz, "absolute_frequency_gate"),
        (relative, cfg.maximum_relative_frequency_difference, "relative_frequency_gate"),
        (logarithmic, cfg.maximum_log_frequency_difference, "log_frequency_gate"),
    )
    active = tuple((value, limit, name) for value, limit, name in gate_pairs if limit is not None)
    canonical_value, canonical_limit, _ = active[0]
    frequency_component = cfg.frequency_cost_weight * canonical_value / canonical_limit
    stability = (
        cfg.frequency_stability_cost_weight
        * abs(left.frequency_stability - right.frequency_stability)
        if left.frequency_stability is not None and right.frequency_stability is not None
        else 0.0
    )
    tau = (
        cfg.decay_tau_cost_weight * abs(log2(left.amplitude_tau_s / right.amplitude_tau_s))
        if left.amplitude_tau_s is not None and right.amplitude_tau_s is not None
        else 0.0
    )
    quality = (
        cfg.amplitude_decay_quality_cost_weight
        * abs(left.amplitude_fit_r_squared - right.amplitude_fit_r_squared)
        if left.amplitude_fit_r_squared is not None
        and right.amplitude_fit_r_squared is not None
        else 0.0
    )
    impact = (
        cfg.impact_evidence_cost_weight
        * float(left.impact_excited != right.impact_excited)
        if left.impact_excited is not None and right.impact_excited is not None
        else 0.0
    )
    total = frequency_component + stability + tau + quality + impact
    failed_gate = next((name for value, limit, name in active if value > limit), None)
    missing_tau = (
        cfg.decay_tau_cost_weight > 0
        and (left.amplitude_tau_s is None or right.amplitude_tau_s is None)
        and not cfg.allow_missing_tau
    )
    missing_impact = (
        cfg.impact_evidence_cost_weight > 0
        and (left.impact_excited is None or right.impact_excited is None)
        and not cfg.allow_missing_preimpact_evidence
    )
    admissible = (
        failed_gate is None and not missing_tau and not missing_impact
        and total <= cfg.maximum_association_cost
    )
    reason = None
    if failed_gate is not None:
        reason = failed_gate
    elif missing_tau:
        reason = "missing_tau"
    elif missing_impact:
        reason = "missing_preimpact_evidence"
    elif total > cfg.maximum_association_cost:
        reason = "cost_above_threshold"
    notes = []
    if cfg.decay_tau_cost_weight > 0 and (
        left.amplitude_tau_s is None or right.amplitude_tau_s is None
    ):
        notes.append("tau_not_applicable")
    if cfg.impact_evidence_cost_weight > 0 and (
        left.impact_excited is None or right.impact_excited is None
    ):
        notes.append("preimpact_evidence_not_applicable")
    return CrossRecordingAssociationDiagnostic(
        left.recording_id, left.candidate_id, right.recording_id, right.candidate_id,
        difference, relative, logarithmic, frequency_component, stability, tau,
        quality, impact, total, admissible, False, None, None, None, False,
        reason, tuple(notes),
    )


def _replace_diagnostic(
    item: CrossRecordingAssociationDiagnostic, **changes: object
) -> CrossRecordingAssociationDiagnostic:
    values = {
        field: getattr(item, field)
        for field in item.__dataclass_fields__
    }
    values.update(changes)
    return CrossRecordingAssociationDiagnostic(**values)


def _margins(
    costs: np.ndarray, row: int, column: int
) -> tuple[float | None, float | None, float | None]:
    selected = costs[row, column]
    row_values = np.delete(costs[row], column)
    column_values = np.delete(costs[:, column], row)
    row_valid = row_values[np.isfinite(row_values)]
    column_valid = column_values[np.isfinite(column_values)]
    row_margin = float(np.min(row_valid) - selected) if row_valid.size else None
    column_margin = float(np.min(column_valid) - selected) if column_valid.size else None
    available = tuple(value for value in (row_margin, column_margin) if value is not None)
    return row_margin, column_margin, min(available) if available else None


def _medoid(members: list[CandidateReference]) -> CandidateReference:
    center = median(item.representative_frequency_hz for item in members)
    return min(members, key=lambda item: (
        abs(item.representative_frequency_hz - center), _key(item)
    ))


def _criterion(
    name: str, observed: float | int | bool, operator: str,
    threshold: float | int | bool, passed: bool,
) -> CandidateCriterionResult:
    return CandidateCriterionResult(
        name, observed, operator, threshold, True, True, passed,
        f"{name}:{'passed' if passed else 'failed'}",
    )


def _cluster(
    cluster_id: int,
    label: str,
    members: tuple[CandidateReference, ...],
    total_recordings: int,
    ambiguous: bool,
    cfg: WithinConditionAssociationSettings,
) -> WithinConditionCandidateCluster:
    members = tuple(sorted(members, key=_key))
    frequencies = tuple(item.representative_frequency_hz for item in members)
    center = float(median(frequencies))
    std = float(pstdev(frequencies))
    dispersion = std / center
    coverage = len({item.recording_id for item in members}) / total_recordings
    criteria = [
        _criterion("minimum_repeat_count", len(members), ">=", cfg.minimum_repeat_count,
                   len(members) >= cfg.minimum_repeat_count),
        _criterion("minimum_repeat_coverage_fraction", coverage, ">=",
                   cfg.minimum_repeat_coverage_fraction,
                   coverage >= cfg.minimum_repeat_coverage_fraction),
        _criterion("unambiguous_association", ambiguous, "==", False, not ambiguous),
        _criterion("internally_consistent", True, "==", True, True),
    ]
    if cfg.maximum_cluster_relative_frequency_dispersion is None:
        criteria.append(CandidateCriterionResult(
            "maximum_cluster_relative_frequency_dispersion", dispersion, "<=", None,
            False, False, None, "maximum_cluster_relative_frequency_dispersion:disabled",
        ))
    else:
        criteria.append(_criterion(
            "maximum_cluster_relative_frequency_dispersion", dispersion, "<=",
            cfg.maximum_cluster_relative_frequency_dispersion,
            dispersion <= cfg.maximum_cluster_relative_frequency_dispersion,
        ))
    failed = tuple(
        item.reason for item in criteria
        if item.enabled and item.applicable and item.passed is False
    )
    reproducible = not failed
    return WithinConditionCandidateCluster(
        cluster_id, label, members,
        tuple(sorted(item.recording_id for item in members)),
        center, float(mean(frequencies)), center, std, min(frequencies),
        max(frequencies), max(frequencies) - min(frequencies), dispersion,
        len(members), len(members), coverage, reproducible, ambiguous,
        reproducible, tuple(criteria), failed,
        ("within_condition_cluster", "not_a_validated_physical_mode"),
    )


def _key(item: CandidateReference) -> tuple[str, int]:
    return item.recording_id, item.candidate_id


def _diagnostic_key(
    item: CrossRecordingAssociationDiagnostic,
) -> tuple[str, int, str, int]:
    return (
        item.left_recording_id, item.left_candidate_id,
        item.right_recording_id, item.right_candidate_id,
    )


def _diagnostic_has(
    item: CrossRecordingAssociationDiagnostic, ref: CandidateReference
) -> bool:
    return (
        (item.left_recording_id, item.left_candidate_id) == _key(ref)
        or (item.right_recording_id, item.right_candidate_id) == _key(ref)
    )


def _costs_for(
    ref: CandidateReference,
    diagnostics: list[CrossRecordingAssociationDiagnostic],
) -> list[float]:
    return [item.total_cost for item in diagnostics if _diagnostic_has(item, ref)]


def _best_alternatives(
    ref: CandidateReference,
    diagnostics: list[CrossRecordingAssociationDiagnostic],
    references: list[CandidateReference],
) -> tuple[CandidateReference, ...]:
    relevant = [item for item in diagnostics if _diagnostic_has(item, ref)]
    if not relevant:
        return ()
    minimum = min(item.total_cost for item in relevant)
    keys: set[tuple[str, int]] = set()
    for item in relevant:
        if isclose(item.total_cost, minimum):
            left = (item.left_recording_id, item.left_candidate_id)
            right = (item.right_recording_id, item.right_candidate_id)
            keys.add(right if left == _key(ref) else left)
    lookup = {_key(item): item for item in references}
    return tuple(lookup[key] for key in sorted(keys) if key in lookup)
