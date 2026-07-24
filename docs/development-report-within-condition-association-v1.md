# Development report — within-condition candidate association v1

Date: 2026-07-24

## Initial state and scientific scope

The validated initial suite contained 366 passing tests. This work associates
operational `ModalCandidate` observations only across repetitions of the same
excitation condition. A candidate in one recording, a reproducible cluster and
a physically validated mode remain different concepts. No `ModalMode` is
constructed.

Dynamic labels `pp`, `p`, `mf`, `f`, `ff` and `unspecified` are categorical
experimental metadata. They are not absolute force, pressure or energy
measurements. The internal association rejects mixed labels; the high-level API
partitions inputs by label before independent association.

## Files changed

- `belllab/within_condition.py`
- `belllab/__init__.py`
- `tests/test_within_condition.py`
- `README.md`
- `docs/RFC-0001-scientific-specification.md`
- `docs/development-report-within-condition-association-v1.md`

## Public contracts and API

The new immutable public contracts are `ExcitationCondition`,
`RecordingCandidateSet`, `CandidateReference`,
`WithinConditionAssociationSettings`,
`CrossRecordingAssociationDiagnostic`,
`WithinConditionCandidateCluster`, `UnmatchedCandidate` and
`WithinConditionAssociationResult`.

`associate_candidates_within_condition` performs the strict single-condition
operation. `group_candidates_by_excitation_condition` partitions labels first.
Both preserve input objects, stable recording identities, rejected candidates,
unmatched candidates and all evaluated pair diagnostics.

## Frequency and association cost

For positive frequencies `f1` and `f2`, the implementation reports:

- absolute distance: `abs(f1 - f2)`;
- symmetric relative distance:
  `abs(f1 - f2) / (0.5 * (f1 + f2))`;
- logarithmic distance: `abs(log2(f1 / f2))`.

Every configured frequency limit is an admissibility gate. The first enabled
metric in absolute, relative, logarithmic order is the canonical normalization
of the frequency cost. Optional components are explicit: relative stability,
logarithmic tau distance, amplitude-decay fit quality and impact evidence. The
total is the exact sum of all components. Gates and
`maximum_association_cost` are inclusive.

In the two-recording test, 100.0 Hz matched 100.4 Hz with 0.4 Hz distance and
cost 0.20; 200.0 Hz matched 199.5 Hz with 0.5 Hz distance and cost 0.25.
Candidates at 300.0 and 450.0 Hz remained unmatched. A 2.0 Hz distance with a
2.0 Hz limit produced cost 1.0 and was accepted exactly at the total-cost
limit. With tau 1.0 s versus 2.0 s and tau weight 0.25, the components were
0.50 frequency plus 0.25 tau, total 0.75.

Tau is optional by default. When both values exist, its distance is
`abs(log2(tau1 / tau2))`; missing tau is never replaced by zero. It is marked
not applicable when allowed and makes the pair inadmissible when
`allow_missing_tau=False`.

## Pair matching, ambiguity and multiple repetitions

Two recordings use deterministic one-to-one Hungarian assignment over
admissible costs. No candidate can be used twice. Row, column and effective
margins exclude inadmissible alternatives. Ambiguity is inclusive:
`assignment_margin <= ambiguity_margin_threshold`. Tests cover a unique
alternative, a zero-margin tie, margins 0.05, exactly 0.10 and 0.20, and stable
tie-breaking by candidate order.

Three or more recordings are sorted by `recording_id` and added progressively.
Each new candidate is compared with the current median-frequency medoid and
must also be compatible with every existing member. The representative is
updated robustly through the member median. For A=100 Hz, B=102 Hz and C=104 Hz
under a 2 Hz inclusive gate, A and B form a cluster but C is rejected from it
because A and C differ by 4 Hz. Thus compatibility is not transitively chained.

For 100.0, 100.3 and 99.7 Hz, the group median and mean are 100.0 Hz, population
standard deviation is approximately 0.244949 Hz and span is 0.6 Hz. Group
statistics also expose minimum, maximum and relative dispersion.

## Coverage and reproducibility

Repeat coverage is
`recording_count / total_recordings_in_condition`. The tests validate 3/3 =
1.0, 2/3 = 0.666667 and the inclusive coverage limit. Singleton observations
are preserved as `UnmatchedCandidate` rather than presented as an associated
cluster.

Reproducibility follows only explicit criterion results: minimum repeat count,
minimum repeat coverage, absence of assignment ambiguity, internal
all-member consistency and optional maximum relative frequency dispersion.
A rejected cluster retains its failed criteria and reasons.

## Pre-impact evidence and rejected candidates

Impact excitation is optional. Tests cover two emergent components, two
amplified components, and an emergent/amplified pair; classifications need not
be identical when both observations have `impact_excited=True`. Persistent
background lines remain associable by default and can be excluded explicitly.
Missing evidence follows `allow_missing_preimpact_evidence`.

Only accepted candidates are eligible by default. With
`allow_rejected_candidates=True`, rejected candidates may be grouped for audit,
but each member keeps `accepted=False`; no implicit candidate promotion occurs.

## Unmatched observations, invariants and determinism

`UnmatchedCandidate` records the original reference, a controlled reason,
best alternatives, minimum observed cost and diagnostics. A result invariant
requires every input reference to occur exactly once, either in one cluster or
in the unmatched collection. Contracts additionally validate unique
recordings, distinct repeat indices, positive frequencies, finite non-negative
costs, exact cost-component sums, selected-implies-admissible, unique pairs,
one member per recording, coherent cluster statistics and contiguous stable
cluster IDs.

Repeated runs with ordered and reversed input produced equal results, cluster
IDs, member order, frequencies, diagnostics and unmatched collections.

## Validation

Sixty tests were added. The final suite contains 426 passing tests.

- `pytest`: 426 passed
- `pytest -W error`: 426 passed
- `python3 -m compileall -q belllab tests`: passed
- `git diff --check`: passed

No additional static checker is configured in `pyproject.toml`.

## Limitations and next steps

This first version is intentionally frequency-dominant and conservative. It
does not support cross-dynamic comparison, split/merge events, dense spectral
bands, nonlinear regimes, factor Q, global modal families, directory batch
processing or final visualization. Future work may compare independently
formed within-condition clusters across dynamic levels under a separate
scientific contract. No automatic or manual conversion to `ModalMode` was
implemented here.
