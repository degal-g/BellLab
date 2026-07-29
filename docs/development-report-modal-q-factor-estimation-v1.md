# Development report - modal Q-factor estimation v1

Date: 2026-07-29

Branch: `feature/modal-q-factor-estimation`

## Initial State

The branch was confirmed with `git branch --show-current` before edits and was
not `main`. The initial worktree was clean. The baseline suite contained 946
passing tests:

- `pytest`: 946 passed;
- `pytest -W error`: 946 passed.

No direct change was made on `main`.

## Files Created

- `belllab/modal_q_factors.py`;
- `tests/test_modal_q_factors.py`;
- `docs/development-report-modal-q-factor-estimation-v1.md`.

## Files Altered

- `README.md`;
- `docs/RFC-0001-scientific-specification.md`;
- `belllab/__init__.py`.

## Scientific Principle

This layer preserves:

```text
hipotese modal != modo fisico comprovado
Q estimado por decaimento != Q fisico exato
Q estimado por largura de banda != Q fisico exato
concordancia entre metodos != validacao fisica definitiva
discordancia entre metodos != prova de erro ou nao linearidade
```

The estimate is an operational calculation from already available modal
parameters, spectral peaks, spectra or precomputed widths. It does not create a
physical oscillator model, infer coupling, infer energy exchange, prove
linearity or prove nonlinearity.

## Status

`ModalQFactorEstimateStatus` is public and mutually exclusive:

- `valid`;
- `valid_with_reservations`;
- `partial`;
- `inconclusive`;
- `insufficient_evidence`;
- `invalid_input`.

## Reasons

`ModalQFactorEstimateReason` separates support, reservations, insufficiency,
inconclusion and invalidity. It includes sufficient decay/bandwidth evidence,
method availability, method consistency states, missing or invalid frequency,
tau and bandwidth, spectral-resolution limitations, peak isolation risks,
source ambiguity, near-threshold matches, possible split/merge context,
unsupported source status and insufficient evidence.

## Configuration

`ModalQFactorEstimationSettings` exposes explicit policy for:

- accepted, reserved, partial, insufficient and invalid source parameter
  statuses;
- enabling and requiring decay or bandwidth methods;
- decay convention and positive frequency/tau limits;
- bandwidth definition, -3 dB level, minimum bandwidth, resolution ratio,
  peak-isolation policy and missing uncertainty policy;
- consistency thresholds, disagreement policy and combination policy;
- uncertainty method, confidence level, bootstrap count and deterministic seed;
- reservations for ambiguity, near-threshold, split and merge contexts.

All enum fields are validated. Fractions remain in `[0, 1]`, confidence in
`(0, 1)`, bootstrap counts are positive, seeds are deterministic integers or
`None`, and optional `None` disables the corresponding criterion.

## Decay Convention

The implemented convention is:

```text
A(t) = A0 exp(-t/tau)
Q_decay = pi * f * tau
```

`tau` is an amplitude decay time. The relation is recorded as an operational
weak-damping summary requiring approximate exponential decay, weak damping,
sufficient isolation and approximately stable frequency. It is not an energy
decay convention.

Example: with `f = 1000 Hz` and `tau = 2.0 s`,
`Q_decay = 6283.185307179586`.

## Q By Decay

`ModalDecayQEstimate` records representative frequency, tau, Q, convention,
frequency and tau uncertainties, propagated Q uncertainty, bounds, relative
uncertainty, assumptions, limits, reasons and diagnostics.

With `u_f = 1 Hz` and `u_tau = 0.1 s`, linear independent propagation gives:

```text
relative uncertainty = 0.05000999900019996
standard uncertainty Q = 314.22209093012214
lower Q = 5968.963216249464
upper Q = 6597.407398109708
```

## Decay Uncertainty

The default uncertainty method is linear propagation:

```text
(u_Q/Q)^2 = (u_f/f)^2 + (u_tau/tau)^2
```

A deterministic parametric bootstrap is also implemented for decay Q using a
fixed local random generator. It does not alter global random state and rejects
nonpositive samples.

## Bandwidth Definition

The default bandwidth definition is full amplitude width at -3 dB:

```text
cutoff = peak_amplitude * 10**(-3/20)
```

Power -3 dB is available separately. Existing `SpectralPeak.width_hz` is reused
as half-prominence width in the source spectrum scale, and
`GlobalSpectralPeakMetric.width_hz` is reused as half-prominence width in
canonical linear power. These conventions are preserved in diagnostics.

## Bandwidth Extraction

`estimate_modal_bandwidth` receives an already calculated frequency axis and
magnitude vector. It locates the target peak, finds left and right cutoff
crossings and interpolates linearly between existing bins. It does not
recalculate spectra, extrapolate outside the axis, invent missing crossings or
use edge bins silently.

Example: center `1000 Hz`, crossings at `995 Hz` and `1005 Hz` produce
`bandwidth = 10 Hz`.

## Spectral Resolution

Resolution is diagnosed by:

```text
bandwidth_hz / frequency_resolution_hz
```

The public `SpectralResolutionAssessment` classifies `well_resolved`,
`marginally_resolved`, `resolution_limited` and `unresolved`. Thresholds are
configurable. Tests cover 0.5, 1.0, 1.5, 2.0 and 5.0 bin-equivalent widths.

## Peak Isolation

`ModalPeakIsolationEvidence` records target frequency, nearest lower/upper
neighbor, nearest distance, bandwidth, overlap fraction, isolation state,
reasons and diagnostics. The layer does not split overlapped peaks or fit
multiple Lorentzians.

## Q By Bandwidth

`ModalBandwidthQEstimate` uses:

```text
Q_bandwidth = f_center / bandwidth
```

For `f_center = 1000 Hz` and `bandwidth = 10 Hz`, `Q_bandwidth = 100`.
Bandwidth zero, nonpositive frequency or nonfinite values are rejected.

## Bandwidth Uncertainty

The bandwidth Q uncertainty uses:

```text
(u_Q/Q)^2 = (u_f/f)^2 + (u_bandwidth/bandwidth)^2
```

The bandwidth uncertainty includes the supplied bandwidth uncertainty and a
minimum resolution component when available. Missing uncertainty remains
explicit.

## Method Comparison

`ModalQMethodComparison` records absolute difference, symmetric relative
difference, log difference and the decay/bandwidth ratio.

Examples tested:

- `Q_decay=100`, `Q_bandwidth=105`: consistent;
- `Q_decay=100`, `Q_bandwidth=120`: partially consistent under a 10% strict
  and 2x partial policy;
- `Q_decay=100`, `Q_bandwidth=250`: inconsistent.

## Combination

`combine_modal_q_estimates` supports no combination, arithmetic mean,
geometric mean, inverse-uncertainty weighted mean, preference for decay and
preference for bandwidth. Inconsistent methods are not combined by default.
Zero uncertainty uses a finite floor so no infinite weight is produced.

## Status Policy

The decision precedence is explicit:

1. invalid input;
2. unsupported source parameter status;
3. absence of both methods;
4. required method missing;
5. available method invalid;
6. strong method inconsistency;
7. only one valid method;
8. valid methods with reservations;
9. two consistent valid methods.

The status is not inferred from scores.

## Determinism

IDs use deterministic SHA-1 payloads based on source parameter estimate,
hypothesis, candidate and recording provenance, bandwidth source, numerical
outputs, status and settings fingerprint. There is no UUID, global counter or
timestamp. Bootstrap uses a local seeded RNG.

## Immutability

Inputs are never mutated. Lists are not sorted in place. The global result
sorts copies into deterministic order and keeps one estimate per source
`ModalParameterEstimate`. Repeated builds produce identical results.

## Tests

Sixty-nine tests were added in `tests/test_modal_q_factors.py`. They cover:

- decay Q formula, convention and invalid values;
- linear and seeded-bootstrap uncertainty;
- synthetic bandwidth and interpolation;
- missing crossings and edge peaks;
- resolution classification and inclusive limits;
- peak isolation and neighbor interference;
- method comparison, partial consistency and inconsistency;
- combination policies and finite uncertainty weights;
- source-status policies and audit flags;
- global partitioning, deterministic ordering, local perturbation and
  immutability;
- settings invariants, precomputed peak widths, numeric invariants, provenance
  and public summaries.

## Result

After adding the layer, the suite contains 1015 tests:

- `pytest`: 1015 passed.
- `pytest -W error`: 1015 passed;
- `python3 -m compileall -q belllab tests`: passed;
- `git diff --check`: passed.

## Limitations

This version does not implement `ModalMode`, full damped-oscillator fitting,
multiple-Lorentzian fitting, separation of overlapping peaks, hardening,
softening, proof of linearity, proof of nonlinearity, split/merge resolution,
gap closure, non-adjacent association, global matching, modal energy exchange,
modal coupling, causality, machine learning, final visualizations, full
experiment pipelines, audio reading or final scientific report export.

## Next Steps

Future work can validate bandwidth conventions on curated real recordings,
attach richer spectral-source registries to candidate provenance, add optional
window-specific resolution diagnostics and develop report exporters that quote
the operational assumptions next to each Q estimate.
