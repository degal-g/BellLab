# Development report — response-regime descriptors v1

**Date:** 2026-07-24  
**Initial validated state:** 644 tests passing  
**Final validated state:** 670 tests passing

## 1. Files changed

- `belllab/response_regime.py`
- `belllab/__init__.py`
- `tests/test_response_regime_descriptors.py`
- `README.md`
- `docs/RFC-0001-scientific-specification.md`
- `docs/development-report-response-regime-descriptors-v1.md`

## 2. Motivation

This round adds operational, auditable descriptors for the observed spectral
character of a dynamic condition. The descriptors consume metrics already
computed by excitation characterization, global spectral characterization,
time-resolved spectral characterization and dynamic-condition comparison.

No FFT, STFT, peak detection, tracking or candidate association is recomputed.

## 3. Descriptor versus physical regime

The implementation preserves:

```text
operational response-regime descriptor
≠ proven physical regime
≠ proof of nonlinearity
≠ physical mode
```

Descriptors are threshold-based labels attached to observed metrics. They do
not assert causality. A `broadband_dominated` result may reflect excitation,
noise, clipping, leakage, short attacks, resolution limits, overlap or physical
behavior; the code records limitations instead of proving a cause.

## 4. Independent dimensions

The main contract is `ResponseRegimeDescription`. It separates:

- spectral structure;
- temporal evolution;
- operational line identity;
- evidence confidence.

The design does not force a single monolithic label. Scores for multiple
descriptors remain available even when the selected descriptor for one
dimension is mixed or indeterminate.

## 5. Criteria

Each criterion is represented by `RegimeDescriptorCriterionResult`:

- criterion name;
- metric name;
- observed value;
- operator;
- threshold;
- applicable state;
- pass/fail state;
- weight;
- support direction;
- reason;
- diagnostics.

Non-applicable criteria do not pass or fail and never use zero as a replacement
for a missing metric.

## 6. Scores

`RegimeDescriptorScore` stores:

- support score;
- opposition score;
- available weight;
- support fraction;
- opposition fraction;
- selected state;
- indeterminate state.

Scores are not probabilities. The default decision policy requires enough
available criterion weight, enough support fraction and limited opposition.

## 7. Spectral structure

Implemented structural descriptors:

- `discrete_line_dominated`;
- `mixed_line_and_continuum`;
- `dense_spectrum`;
- `broadband_dominated`;
- `indeterminate`.

Line-dominated evidence uses low flatness, low entropy, high tonal fraction,
high spectral crest factor, sparse peak density and narrow occupied fraction.
Broadband evidence uses high flatness, high entropy, low tonal fraction, high
residual fraction, broad occupied fraction, low crest factor and optionally high
peak density. Dense spectrum is evaluated separately from broadband behavior.

## 8. Temporal evolution

Implemented temporal descriptors:

- `stable_spectral_character`;
- `broadband_to_tonal`;
- `tonal_to_broadband`;
- `progressive_spectral_densification`;
- `progressive_spectral_sparsification`;
- `mixed_temporal_evolution`;
- `indeterminate`.

All changes use the existing convention:

```text
late - early
```

For example, `broadband_to_tonal` receives support from flatness drop, entropy
drop, tonal fraction increase, residual fraction drop, bandwidth drop, density
drop and persistent change points.

## 9. Operational line identity

Implemented line-identity descriptors:

- `line_identity_preserved`;
- `line_identity_partially_preserved`;
- `line_identity_not_resolved`;
- `not_evaluated`.

The implementation uses only within-condition/time metrics such as temporal
coverage, centroid-slope stability, late tonal fraction, valid repeat count and
repeat variability. It does not associate individual candidates between `pp`
and `ff` and does not assert physical modal identity.

## 10. Confidence

Confidence categories are operational evidence-quality classes:

- `high`;
- `moderate`;
- `low`;
- `insufficient`.

They are not statistical confidence or probability. Confidence considers valid
repeat count, temporal coverage, clipping, SNR threshold, comparability status,
missing metrics, within-condition variability, resolution limitation and
criteria conflicts.

## 11. Conflicts

The description preserves:

- supporting metrics;
- conflicting metrics;
- unavailable metrics;
- limitations.

Contradictory evidence can lead to `mixed_line_and_continuum`,
`mixed_temporal_evolution` or `indeterminate`, rather than forcing a strong
exclusive label.

## 12. Clipping

Clipped conditions are preserved by default and receive:

```text
spectral_metrics_potentially_distorted_by_clipping
```

Confidence drops. If `reject_clipped_conditions=True`, the condition
description becomes invalid with structured failure. Metrics are not corrected.

## 13. SNR

When `minimum_signal_to_background_db` is configured, low or missing SNR reduces
evidence quality. The condition remains auditable and is not erased.

## 14. Resolution

Density-related criteria inspect the global spectral fingerprint and median
peak spacing. When apparent spacing is comparable to physical resolution, the
description records:

```text
density_metrics_resolution_limited
```

Density criterion weights are reduced, preventing high-confidence dense-spectrum
claims from resolution-limited evidence alone.

## 15. Variability

The code uses aggregated metric coefficients of variation when available. High
within-condition variability reduces confidence and can prevent strong claims.
No p-value or inferential test is computed.

## 16. Sequence `pp` to `ff`

`RegimeDescriptorSequence` exposes descriptor sequences in canonical order:

```text
pp → p → mf → f → ff
```

Missing labels remain `None`, missing labels are listed, changes are named, and
stable descriptor segments are preserved.

## 17. Emergent patterns

`EmergentResponsePattern` captures structured patterns such as:

- tonal structure preserved over multiple conditions;
- increased density or broadband character over the dynamic order;
- broadband attack followed by a more tonal tail;
- increased uncertainty or limited evidence.

These are operational patterns, not proofs of physical transition.

## 18. Public contracts created

- `RegimeCriterionWeight`
- `ResponseRegimeDescriptorSettings`
- `RegimeDescriptorCriterionResult`
- `RegimeDescriptorScore`
- `DescriptorEvaluation`
- `ResponseRegimeDescription`
- `RegimeDescriptorSequence`
- `EmergentResponsePattern`
- `DynamicResponseRegimeDescription`

## 19. Public functions created

- `describe_response_regime`
- `evaluate_regime_descriptor`
- `build_regime_descriptor_sequence`
- `describe_dynamic_response_regimes`

## 20. Tests added

Added `tests/test_response_regime_descriptors.py` with 26 tests covering:

- settings invariants;
- criterion invariants;
- discrete-line descriptor;
- broadband descriptor;
- dense tonal spectrum separate from broadband;
- mixed line/continuum;
- central `pp` versus `ff` synthetic scenario;
- tonal-to-broadband direction;
- conflicting metrics;
- clipping allowed and rejected;
- low SNR;
- missing metrics and strict missing-metric policy;
- resolution-limited density;
- high within-condition variability;
- line identity preserved and not evaluated;
- stable temporal character;
- densification and sparsification;
- descriptor sequence `pp` to `ff`;
- missing conditions;
- `DynamicConditionComparisonResult` input;
- invalid-condition global failure;
- deterministic shuffled input;
- explicit absence of forbidden physical operations.

## 21. Quantitative examples from tests

Discrete-line synthetic condition:

```text
flatness = 0.04
entropy = 0.20
tonal fraction = 0.90
spectral crest factor = 12.0
peak density = 0.004 1/Hz
occupied fraction = 0.20
descriptor = discrete_line_dominated
```

Broadband synthetic condition:

```text
flatness = 0.82
entropy = 0.93
tonal fraction = 0.15
residual fraction = 0.85
occupied fraction = 0.88
descriptor = broadband_dominated
```

Central synthetic scenario:

```text
pp: discrete_line_dominated
ff: mixed or dense structural character
ff temporal evolution: broadband_to_tonal
```

The scenario describes observed metrics only and does not classify a nonlinear
transition.

## 22. Result of the suite

Initial state confirmed before changes:

```text
pytest
644 passed
```

Final validation:

```text
pytest
670 passed

pytest -W error
670 passed

python3 -m compileall -q belllab tests
OK

git diff --check
OK
```

## 23. Limitations

- No proof of linearity.
- No proof of nonlinearity.
- No bifurcation test.
- No chaos detection.
- No Lyapunov exponents.
- No fractal dimension.
- No cross-condition individual candidate association.
- No modal split/fusion.
- No modal families.
- No `ModalMode`.
- No factor Q.
- No machine learning.
- No supervised classification.
- No final visualization layer.

## 24. Next steps

- Add report/export formatting for descriptor results.
- Add visualizations only after data contracts remain stable.
- Extend line-identity descriptors when cross-condition modal-family contracts
  exist, without retrofitting that interpretation into this layer.

## 25. Threshold and score closure

Date: 2026-07-24.

This section records the surgical closure pass that started from the validated
state of 670 tests.  The goal was not to calibrate universal physical constants.
The goal was to make the operational threshold semantics explicit, deterministic
and test-covered.

### 25.1 Threshold inventory

`None` disables an optional criterion.  Otherwise all default thresholds are
finite and configurable through `ResponseRegimeDescriptorSettings`.

| Name | Default | Unit | Operator | Inclusive boundary | Affected descriptor | Operational rationale |
| --- | ---: | --- | --- | --- | --- | --- |
| `maximum_flatness_for_line_dominated` | 0.25 | fraction | `<=` | yes | `discrete_line_dominated` | Low flatness supports concentrated line structure. |
| `minimum_flatness_for_broadband` | 0.55 | fraction | `>=` | yes | `broadband_dominated` | High flatness supports a more uniform spectral distribution. |
| `maximum_entropy_for_line_dominated` | 0.55 | normalized entropy | `<=` | yes | `discrete_line_dominated` | Low entropy supports energy concentration. |
| `minimum_entropy_for_broadband` | 0.75 | normalized entropy | `>=` | yes | `broadband_dominated` | High entropy supports broad spectral distribution. |
| `minimum_tonal_fraction_for_line_dominated` | 0.65 | fraction | `>=` | yes | `discrete_line_dominated` | High tonal fraction supports line dominance. |
| `maximum_tonal_fraction_for_broadband` | 0.35 | fraction | `<=` | yes | `broadband_dominated` | Low tonal fraction supports broadband dominance. |
| residual fraction for broadband | 0.65 | fraction | `>=` | yes | `broadband_dominated` | Derived as `1 - maximum_tonal_fraction_for_broadband`. |
| `maximum_peak_density_for_sparse` | 0.010 | 1/Hz | `<=` | yes | `discrete_line_dominated`, `mixed_line_and_continuum` | Sparse detected peaks support isolated-line behavior. |
| `minimum_peak_density_for_dense` | 0.020 | 1/Hz | `>=` | yes | `dense_spectrum`, `broadband_dominated` | Dense detected peaks support dense spectral structure. |
| `maximum_occupied_fraction_for_narrow` | 0.35 | fraction | `<=` | yes | `discrete_line_dominated` | Narrow occupied band supports concentrated line structure. |
| `minimum_occupied_fraction_for_broadband` | 0.65 | fraction | `>=` | yes | `broadband_dominated`, `dense_spectrum` | Wide occupied band supports broadband or dense structure. |
| `minimum_spectral_crest_for_line_dominated` | 5.0 | power ratio | `>=` | yes | `discrete_line_dominated` | High spectral crest supports dominant spectral lines. |
| `maximum_spectral_crest_for_broadband` | 3.0 | power ratio | `<=` | yes | `broadband_dominated` | Lower spectral crest supports less line-dominated energy. |
| `minimum_peak_count_for_dense` | 8.0 | count | `>=` | yes | `dense_spectrum` | Many significant peaks support dense spectra. |
| `maximum_median_peak_spacing_for_dense_hz` | disabled | Hz | `<=` | yes when enabled | `dense_spectrum` | Small spacing supports density when configured. |
| `minimum_tonal_fraction_for_mixed` | 0.25 | fraction | `between` lower | yes | `mixed_line_and_continuum`, `dense_spectrum` | Requires a meaningful tonal component. |
| `maximum_tonal_fraction_for_mixed` | 0.75 | fraction | `between` upper | yes | `mixed_line_and_continuum` | Avoids treating fully tonal cases as mixed. |
| `minimum_residual_fraction_for_mixed` | 0.25 | fraction | `>=` | yes | `mixed_line_and_continuum` | Requires a meaningful residual component. |
| `minimum_flatness_for_mixed` | 0.20 | fraction | `between` lower | yes | `mixed_line_and_continuum` | Allows intermediate continuum evidence. |
| `maximum_flatness_for_mixed` | 0.60 | fraction | `between` upper | yes | `mixed_line_and_continuum` | Avoids using mixed as universal broadband fallback. |
| `minimum_flatness_drop_for_broadband_to_tonal` | 0.15 | fraction change | `<= -threshold` | yes | `broadband_to_tonal`; inverse for `tonal_to_broadband` | Late flatness must drop by at least the configured amount. |
| `minimum_entropy_drop_for_broadband_to_tonal` | 0.15 | entropy change | `<= -threshold` | yes | `broadband_to_tonal`; inverse for `tonal_to_broadband` | Late entropy must drop by at least the configured amount. |
| `minimum_tonal_fraction_increase` | 0.20 | fraction change | `>=` | yes | `broadband_to_tonal`; inverse for `tonal_to_broadband` | Late tonal fraction must increase by at least the configured amount. |
| residual fraction temporal change | 0.20 | fraction change | `<= -threshold` or `>= threshold` | yes | `broadband_to_tonal`, `tonal_to_broadband` | Derived from tonal-fraction change threshold. |
| `minimum_density_change` | 0.005 | 1/Hz change | `>=` or `<= -threshold` | yes | densification, sparsification, broadband-to-tonal, tonal-to-broadband | Requires a measurable change in detected peak density. |
| `minimum_bandwidth_change` | disabled | Hz change | `>=` or `<= -threshold` | yes when enabled | temporal descriptors | Optional absolute occupied-bandwidth change threshold. |
| `minimum_occupied_fraction_change` | 0.15 | fraction change | reserved | yes when used | temporal occupancy policy | Configured threshold retained for occupied-fraction change policy. |
| `minimum_persistent_change_point_count` | 1 | count | `>=` | yes | `broadband_to_tonal` | Requires at least one operational change point when available. |
| `maximum_flatness_change_for_stable` | 0.05 | fraction change | `abs<=` | yes | `stable_spectral_character` | Small early-late flatness change supports stability. |
| `maximum_entropy_change_for_stable` | 0.05 | entropy change | `abs<=` | yes | `stable_spectral_character` | Small early-late entropy change supports stability. |
| `maximum_tonal_fraction_change_for_stable` | 0.08 | fraction change | `abs<=` | yes | `stable_spectral_character` | Small tonal-fraction change supports stability. |
| `maximum_density_change_for_stable` | 0.003 | 1/Hz change | `abs<=` | yes | `stable_spectral_character` | Small density change supports stability. |
| `maximum_bandwidth_change_for_stable` | disabled | Hz change | `abs<=` | yes when enabled | `stable_spectral_character` | Optional absolute bandwidth stability threshold. |
| `minimum_valid_repeat_count` | 1 | count | `>=` | yes | line identity and confidence | Requires at least one valid repetition by default. |
| `minimum_valid_frame_coverage` | 0.50 | fraction | `>=` | yes | line identity and confidence | Requires enough valid temporal coverage. |
| `maximum_within_condition_variability` | 0.50 | coefficient of variation | `<=` | yes | line identity and confidence | High internal dispersion lowers confidence. |
| `minimum_signal_to_background_db` | disabled | dB | `>=` | yes when enabled | confidence | Low SNR limits descriptor reliability. |
| `maximum_missing_metric_fraction_for_moderate_confidence` | 0.50 | fraction | `<=` | yes | confidence | Too many unavailable metrics prevents moderate confidence. |
| `maximum_centroid_slope_for_preserved_identity_hz_per_s` | 10.0 | Hz/s | `abs<=` | yes | `line_identity_preserved` | Stable centroid trend supports operational line identity. |
| `minimum_tonal_fraction_for_line_identity` | 0.50 | fraction | `>=` | yes | `line_identity_preserved` | Late tonal fraction supports resolved line identity. |
| `minimum_support_fraction_for_descriptor` | 0.55 | fraction | `>=` | yes | all scored descriptors | Minimum favorable evidence fraction for selection. |
| `maximum_opposition_fraction_for_descriptor` | 0.45 | fraction | `<=` | yes | all scored descriptors | Maximum opposing evidence fraction for selection. |
| `minimum_available_weight_for_descriptor` | 3.0 | score weight | `>=` | yes | all scored descriptors | Prevents selection from too little evidence. |
| `high_confidence_minimum_support_fraction` | 0.70 | fraction | `>=` | yes | confidence | Requires stronger selected support for high confidence. |
| `density_resolution_limit_spacing_factor` | 2.0 | multiplier | `<= factor * resolution` | yes | density criteria | Marks peak-density evidence as resolution limited. |
| `resolution_limited_density_weight_factor` | 0.25 | weight multiplier | exact multiplier | n/a | density criteria | Reduces density evidence weight when resolution limited. |
| `score_tie_tolerance` | 1e-12 | fraction/weight tolerance | applied to selection inequalities | yes | all scored descriptors | Prevents tiny floating-point score perturbations from changing selection. |
| `numerical_tolerance` | 1e-12 | metric tolerance | applied to criterion operators | yes | all criteria | Prevents boundary decisions from depending on floating-point roundoff. |

### 25.2 Inclusivity and intermediate zones

Criterion boundaries are inclusive:

```text
minimum criterion: observed + numerical_tolerance >= threshold
maximum criterion: observed <= threshold + numerical_tolerance
absolute minimum: abs(observed) + numerical_tolerance >= threshold
absolute maximum: abs(observed) <= threshold + numerical_tolerance
between: lower - numerical_tolerance <= observed <= upper + numerical_tolerance
```

Opposed structural thresholds now require strict separation at configuration
time:

```text
maximum_flatness_for_line_dominated < minimum_flatness_for_broadband
maximum_entropy_for_line_dominated < minimum_entropy_for_broadband
maximum_tonal_fraction_for_broadband < minimum_tonal_fraction_for_line_dominated
maximum_peak_density_for_sparse < minimum_peak_density_for_dense
maximum_occupied_fraction_for_narrow < minimum_occupied_fraction_for_broadband
```

The defaults therefore create explicit intermediate zones.  For example,
flatness in `(0.25, 0.55)` is neither line-dominated nor broadband-dominated by
that criterion alone.

### 25.3 Score formulas and available weight

For each descriptor:

```text
support_score = sum(weight of applicable criteria that passed)
opposition_score = sum(weight of applicable criteria that failed)
available_weight = support_score + opposition_score
support_fraction = support_score / available_weight
opposition_fraction = opposition_score / available_weight
```

Criteria that are disabled, unavailable or not applicable do not contribute to
`available_weight`.  When `available_weight == 0`, both fractions are `None`,
the score is indeterminate, and no descriptor can be selected.

The canonical selection rule is inclusive:

```text
available_weight + score_tie_tolerance >= minimum_available_weight_for_descriptor
support_fraction + score_tie_tolerance >= minimum_support_fraction_for_descriptor
opposition_fraction <= maximum_opposition_fraction_for_descriptor + score_tie_tolerance
```

The score contract now rejects incoherent manual scores:

- negative or non-finite scores;
- support or opposition larger than available weight;
- support plus opposition not equal to available weight;
- fractions inconsistent with scores;
- selected scores with zero available weight or zero support;
- simultaneous `selected` and `indeterminate`.

### 25.4 Ties, mixed and indeterminate

Descriptor selection is deterministic and does not depend on dictionary order,
metric order or input condition order.

Within structure:

1. `mixed_line_and_continuum` is selected only when its own mixed criteria score
   is selected.
2. simultaneous `discrete_line_dominated` and `broadband_dominated` selection is
   treated as `indeterminate` unless the explicit mixed descriptor is selected;
3. otherwise the canonical order is broadband, dense, then discrete line.

Within temporal evolution, mutually exclusive directions remain separated.  If
multiple temporal directions are selected simultaneously, the principal temporal
descriptor is `mixed_temporal_evolution`.

The documented distinction is:

```text
mixed = simultaneous evidence of line and continuum character
indeterminate = insufficient, unavailable or conflicting evidence without a selected mixed score
```

### 25.5 Confidence, clipping, SNR, resolution and variability

Confidence is categorical operational evidence quality, not probability.

Precedence used in this closure:

- invalid summaries produce `insufficient`;
- `reject_clipped_conditions=True` with clipping produces `insufficient`;
- low or unavailable SNR under an enabled SNR requirement produces `low`;
- low temporal coverage produces `low`;
- near-clipping produces an explicit limitation and `low`;
- required but unavailable spectral comparability produces `insufficient`;
- selected descriptors with strong support, low missing metrics and no
  limitations can reach `high`;
- selected descriptors with adequate support but clipping, near-clipping,
  high variability or resolution-limited density cannot reach `moderate`;
- unresolved evidence falls back to `low` or `insufficient`, depending on data
  availability.

Clipping behavior remains descriptive:

- actual clipping adds `spectral_metrics_potentially_distorted_by_clipping`;
- near-clipping adds `spectral_metrics_potentially_affected_by_near_clipping`;
- clipping is rejected only when configured;
- metrics are not corrected or discarded silently.

SNR behavior remains explicit:

- missing SNR is not converted to zero;
- when an SNR minimum is configured, missing SNR produces
  `signal_to_background_unavailable`;
- SNR below the configured minimum produces `low_signal_to_background_ratio`;
- equality at the configured threshold passes.

Resolution-limited density behavior remains non-destructive:

- if median peak spacing is less than or equal to
  `density_resolution_limit_spacing_factor * frequency_resolution_hz`, density
  evidence receives the reduced weight factor;
- the limitation `density_metrics_resolution_limited` is recorded;
- the condition is not rejected automatically.

Within-condition variability behavior remains descriptive:

- coefficient of variation at the configured maximum passes;
- values above the maximum add `high_within_condition_variability`;
- missing coefficients are not substituted by zero.

### 25.6 Quantitative boundary examples

The added tests explicitly verify below, exact and above-threshold behavior.
Representative examples:

```text
maximum_flatness_for_line_dominated = 0.25
0.249999 passes
0.250000 passes
0.250001 fails

minimum_flatness_for_broadband = 0.55
0.549999 fails
0.550000 passes
0.550001 passes

minimum_tonal_fraction_increase = 0.20
0.199999 fails
0.200000 passes
0.200001 passes

maximum_within_condition_variability = 0.50
0.500000 passes
0.500001 adds high_within_condition_variability
```

Score closure examples:

```text
support_score = 2.0
opposition_score = 1.0
available_weight = 3.0
support_fraction = 2 / 3
opposition_fraction = 1 / 3

available_weight = 0.0
support_fraction = None
opposition_fraction = None
selected = False
indeterminate = True
```

Near score-tie behavior is covered with `score_tie_tolerance = 1e-12`:

```text
support_fraction = 0.55 - 0.5e-12 -> selected
support_fraction = 0.55 - 1.0e-12 -> selected
support_fraction = 0.55 - 2.0e-12 -> not selected
```

### 25.7 Tests added in the closure pass

The descriptor test file grew from 26 to 69 tests.  Relative to the previous
validated project state, the full suite grew from 670 to 713 tests, adding 43
tests in this closure pass.

New coverage includes:

- exact inclusivity for structural thresholds;
- exact inclusivity for temporal thresholds;
- exact inclusivity for stability thresholds;
- SNR, coverage, repeat-count and variability boundaries;
- score support, opposition and available-weight formulas;
- zero available weight with no division by zero;
- inclusive selection thresholds;
- near score-tie tolerance;
- weight scaling and asymmetric weights;
- invalid threshold, non-finite and weight configurations;
- deterministic tie behavior;
- `mixed` versus `indeterminate`;
- dense spacing and resolution-limited density;
- clipping, near-clipping and rejection policy;
- systematic missing-metric combinations;
- strengthened score and description invariants;
- deterministic evaluations under reordered metrics, weights and conditions;
- local perturbation tests.

### 25.8 Final validation for this closure

The final expected validation state after this section is:

```text
pytest
713 passed

pytest -W error
713 passed

python3 -m compileall -q belllab tests
OK

git diff --check
OK
```

### 25.9 Limitations

- The defaults remain provisional operational thresholds, not universal
  physical constants.
- No new descriptor was added.
- No formal proof of linearity or nonlinearity was added.
- No cross-condition individual candidate association was added.
- No modal split, fusion, family construction or `ModalMode` conversion was
  added.
- The score-tie policy closes numerical selection boundaries; it does not turn
  the scores into probabilities.
