# Development report — dynamic-condition comparison v1

**Date:** 2026-07-24  
**Initial validated state:** 615 tests passing  
**Final validated state:** 644 tests passing

## 1. Files changed

- `belllab/dynamic_comparison.py`
- `belllab/__init__.py`
- `tests/test_dynamic_condition_comparison.py`
- `README.md`
- `docs/RFC-0001-scientific-specification.md`
- `docs/development-report-dynamic-condition-comparison-v1.md`

## 2. Motivation

This round implements the first descriptive comparison layer across nominal
dynamic conditions (`pp`, `p`, `mf`, `f`, `ff`). The implementation answers how
already computed excitation, global spectral and time-resolved spectral metrics
change as the nominal excitation label increases.

The layer intentionally preserves the scientific distinction:

```text
difference between dynamic conditions
≠ proof of nonlinearity
≠ physical mode change
≠ confirmed regime transition
```

## 3. Musical label versus measured intensity

The canonical order is:

```text
pp < p < mf < f < ff
```

The code treats this as nominal musical ordering only. `lower_dynamic_label` and
`higher_dynamic_label` in pair comparisons do not assert that measured acoustic
or excitation intensity is physically lower or higher.

## 4. Aggregation of repeats

The public summary contract is `DynamicConditionSpectralSummary`. It aggregates
all repeats of one dynamic label without mixing labels. Metrics use
`AggregatedMetric`, which reports:

- available, finite and discarded counts;
- median;
- mean;
- population standard deviation;
- minimum;
- maximum;
- range;
- coefficient of variation when meaningful;
- median absolute deviation;
- structured validity and diagnostics.

The default representative statistic for comparisons is the median. Missing and
invalid values are preserved as discarded values; no `NaN` or infinity is used
as a sentinel.

## 5. Comparability instrumental

Amplitude-dependent metrics are compared only when instrumental metadata are
compatible under the active settings. The implemented checks cover:

- session, if required;
- microphone;
- interface;
- acquisition gain;
- microphone distance;
- channel;
- microphone orientation, if required;
- amplitude unit;
- clipping policy.

If these fail, amplitude metrics become `not_comparable` while scale-invariant
spectral metrics can remain comparable.

## 6. Comparability spectral

Global and time-resolved comparability are represented by summary fingerprints
and within-condition checks reuse:

- `evaluate_spectral_characterization_comparability`;
- `evaluate_time_resolved_spectral_comparability`.

The comparison is granular. A time-region metric can be blocked by frame/hop
or time-resolution mismatch while a global flatness metric remains available.
Band definitions must match unless explicitly allowed.

## 7. Excitation metrics

The summary aggregates:

- peak absolute amplitude;
- RMS amplitude;
- signal energy;
- equivalent level in dBFS;
- crest factor;
- impulse duration;
- attack duration;
- SNR in dB;
- clipping and near-clipping fractions;
- clipped and near-clipped recording fractions.

Amplitude metrics are not directly compared when gain, unit, microphone,
distance or clipping make them incompatible.

## 8. Global spectral metrics

The summary aggregates:

- total spectral energy;
- centroid;
- spread;
- rolloff 50%, 85% and 95%;
- flatness;
- entropy;
- spectral crest factor;
- significant peak count;
- peak density per Hz;
- peak density per octave;
- median peak spacing;
- tonal energy fraction;
- residual energy fraction;
- occupied bandwidth;
- occupied frequency fraction;
- energy and fraction by configured bands.

These are descriptive comparisons only. Increased flatness, entropy, density or
occupied bandwidth is not interpreted automatically as white noise, chaos or
nonlinearity.

## 9. Time-resolved metrics

The summary aggregates time-resolved descriptors already computed by
`TimeResolvedSpectralCharacterization`:

- energy trend in dB/s;
- centroid trend;
- spread trend;
- flatness trend;
- entropy trend;
- significant peak count trend;
- peak density trend;
- tonal fraction trend;
- residual fraction trend;
- occupied bandwidth trend;
- change-point count;
- first change-point time per metric;
- valid temporal coverage.

Failed temporal fits remain missing values. They are not replaced with zero.

## 10. Early, middle and late regions

The implementation aggregates region medians for:

- energy;
- centroid;
- spread;
- flatness;
- entropy;
- significant peak count;
- peak density;
- tonal fraction;
- residual fraction;
- occupied bandwidth.

It also exposes descriptive changes:

```text
middle - early
late - early
late - middle
```

The sign convention is explicit and no direction is interpreted as automatic
modalization or regime transition.

## 11. Bands

Global band energy and energy fraction are summarized per condition. Temporal
band persistence metrics include coverage fraction, time until crossing the
configured threshold and trend of energy fraction.

Absolute band energy is amplitude-dependent and can become unavailable when
instrumental comparability fails. Band energy fraction can remain comparable
when spectral definitions are identical.

## 12. Adjacent pairs

`compare_dynamic_conditions` builds comparisons between adjacent available
conditions. With complete data this yields:

```text
pp → p
p → mf
mf → f
f → ff
```

When conditions are missing, available pairs can bridge the gap and record the
nominal jump. Example:

```text
pp → mf
```

records a two-step comparison and notes that `p` is absent.

## 13. Reference `pp`

Reference comparisons are exposed separately. The default reference is `pp`.
If `pp` is absent, the configured policy can fall back to the lowest available
condition and records the fallback explicitly. Strict reference mode preserves
the missing reference instead of substituting it.

## 14. Monotonicity

`DynamicMetricSequence` preserves values in canonical label order:

```text
pp → p → mf → f → ff
```

`DynamicMetricMonotonicityResult` classifies sequences descriptively as:

- monotonically increasing;
- monotonically decreasing;
- non-decreasing;
- non-increasing;
- constant;
- non-monotonic;
- insufficient.

This does not imply physical linearity or nonlinearity.

## 15. Variability within condition

Pairwise metric comparisons expose
`change_to_within_condition_variability_ratio` when both conditions have enough
repeat dispersion. It is a descriptive ratio between median change and combined
within-condition standard deviation. It is not a p-value and not a statistical
significance test.

## 16. Clipping

Clipped conditions are preserved by default. Diagnostics mark:

```text
spectral_metrics_potentially_distorted_by_clipping
```

Amplitude comparisons can become unavailable under the clipping policy. A
separate setting allows excluding clipped repeats from aggregation, in which
case the discarded repeats remain visible in counts and diagnostics.

## 17. Conditions absent

Missing dynamic labels are preserved in the top-level result and in metric
sequences. No interpolation is performed. A result with fewer than two
comparable conditions fails structurally with
`insufficient_comparable_dynamic_conditions`.

## 18. Public contracts created

- `MetricTolerance`
- `AggregatedMetric`
- `DynamicConditionRecordingAnalysis`
- `DynamicConditionComparisonSettings`
- `DynamicConditionSpectralSummary`
- `MetricComparison`
- `DynamicConditionPairComparison`
- `DynamicMetricMonotonicityResult`
- `DynamicMetricSequence`
- `DynamicConditionComparisonResult`

## 19. Public functions created

- `aggregate_metric_values`
- `summarize_dynamic_condition`
- `compare_dynamic_condition_pair`
- `compare_dynamic_conditions`
- `evaluate_dynamic_metric_monotonicity`

All functions are numerical/data-structural. They do not read files and do not
recompute FFT or STFT.

## 20. Tests added

Added `tests/test_dynamic_condition_comparison.py` with 29 tests covering:

- exact aggregation statistics and missing values;
- invalid aggregation failure;
- repeat aggregation by dynamic label;
- complete canonical order;
- local inversion;
- missing conditions and non-adjacent jumps;
- adjacent-only policy;
- insufficient condition count;
- instrumental comparability;
- clipping preservation and configured exclusion;
- global spectral comparisons;
- reference `pp` and fallback;
- within-condition variability ratio;
- unit policies for dB, fractions, counts and slopes;
- metric tolerances;
- monotonicity categories;
- band energy and band fraction policies;
- band definition mismatch;
- early/middle/late temporal metrics;
- change-point aggregation;
- temporal band persistence;
- temporal comparability mismatch;
- deterministic output under shuffled input;
- explicit diagnostics that no forbidden operation occurred.

## 21. Quantitative examples from tests

Aggregation test:

```text
values: 1.0, missing, 3.0, 5.0
median: 3.0
mean: 3.0
finite_count: 3
discarded_count: 1
MAD: 2.0
```

Complete dynamic RMS sequence:

```text
pp, p, mf, f, ff = 1, 2, 3, 4, 5
monotonicity = monotonically_increasing
```

Synthetic inversion:

```text
pp = 1
p = 4
mf = 2
f = 5
ff = 6
p → mf direction = decrease
global monotonicity = non_monotonic
```

Synthetic broad attack and tonal tail:

```text
ff early flatness > ff late flatness
ff late tonal fraction - ff early tonal fraction > 0
```

This example describes an operational change in spectral metrics; it does not
classify a physical transition.

## 22. Result of the suite

Initial state confirmed before changes:

```text
pytest
615 passed
```

After implementation and tests:

```text
pytest -q
644 passed
```

Final validation:

```text
pytest
644 passed

pytest -W error
644 passed

python3 -m compileall -q belllab tests
OK

git diff --check
OK
```

## 23. Limitations

- No automatic classification of linear or nonlinear behavior.
- No formal regime-transition detector.
- No chaos detection.
- No association of individual candidates between dynamic conditions.
- No split/fusion modal analysis.
- No global modal families.
- No factor Q.
- No inferential statistics or p-values.
- No machine learning.
- No visualization layer.
- No conversion to `ModalMode`.

## 24. Next steps

- Add reporting/export utilities for these comparison contracts.
- Add plotting only after the data contracts stabilize.
- Add optional calibrated amplitude workflows when acquisition metadata support
  physically meaningful intensity comparisons.
- Later, and only as a separate layer, define formal regime descriptors if the
  scientific criteria are explicit and testable.
