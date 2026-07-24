# Development report — time-resolved spectral characterization v1

Date: 2026-07-24

## 1. Initial state

The validated initial suite contained 574 passing tests. The implemented scope
adds descriptive time-resolved spectral metrics after impact while preserving:

`spectral metric evolution` != `proven physical regime transition` !=
`proof of nonlinearity` != `modal identification`.

No direct comparison between `pp`, `p`, `mf`, `f` and `ff`, no formal regime
classification, no chaos detection, no cross-condition candidate association,
no modal split/merge, no Q factor, no modal family and no `ModalMode`
conversion was introduced.

## 2. Files changed

- `belllab/time_resolved_spectrum.py` (new);
- `belllab/__init__.py`;
- `tests/test_time_resolved_spectral_characterization.py` (new);
- `README.md`;
- `docs/RFC-0001-scientific-specification.md`;
- this report.

## 3. Motivation

The previous global characterization describes a recording-wide spectrum. This
work adds the missing temporal layer: how energy distribution, flatness,
entropy, peak density, tonal fraction, residual fraction, occupied bandwidth,
dominant-bin frequency and band energy evolve frame by frame after impact.

The result is designed for future objective comparison of attacks, decays and
dynamic-level evolution, but it does not perform those comparisons in this
round.

## 4. Frame policy

Frames are aligned to the configured analysis window relative to
`impact_time_s`. The default window starts at the impact and ends at the signal
end. Frame and hop durations are converted to deterministic integer sample
counts with `round(duration * sample_rate)`.

By default only complete frames fully contained in the effective analysis
interval are processed. Final incomplete frames are discarded with diagnostics.
Temporal zero padding is optional through `pad_end=True` and is explicitly
diagnosed. Hop duration may exceed frame duration; this creates gaps and is
reported as a diagnostic.

Each frame exposes start, center, end and relative time. Center time is
deterministic:

`center = start + frame_duration / 2`.

## 5. Spectral domain

Each frame is analyzed as a stationary one-sided FFT using the existing
`analyze_spectrum` normalization. The frame `Spectrum` is then passed to
`characterize_global_spectrum`.

Therefore the canonical distribution domain is still linear power. Linear
amplitude is squared, dB values are not used for centroid, entropy, rolloff or
energy fractions, and zero padding only changes FFT bin spacing, not physical
resolution.

## 6. Reuse of global characterization

The time-resolved layer does not maintain an independent implementation of
centroid, spread, rolloff, flatness, entropy, peak detection, peak width, tonal
fraction, residual fraction, occupied bandwidth or band energy.

For each valid frame, `characterize_global_spectrum` supplies those metrics.
The new module only adapts frame timing, invalid-frame policy, dominant-bin
diagnostics, summaries, temporal fits, regions, band persistence, change points
and comparability.

The test `test_single_frame_matches_global_characterization` confirms equality
or numerical equivalence for energy, centroid, spread, rolloff, flatness,
entropy, crest factor, peak count, tonal fraction and occupied bandwidth when
the complete signal is one frame with equivalent configuration.

## 7. Silent and weak frames

Frames remain in the sequence even when invalid. Reasons include:

- `zero_total_spectral_energy`;
- `frame_energy_below_threshold`;
- `nonfinite_frame_samples`;
- `insufficient_frame_samples`;
- failed global frame characterization.

Invalid frames keep timing and sample count. Metrics that require valid
spectral energy remain unavailable rather than using NaN or infinity.

## 8. Metrics per frame

`TimeResolvedSpectralFrame` includes energy, centroid, spread, 50/85/95%
rolloffs, flatness, entropy, spectral crest factor, significant peak count,
peak density per hertz and per octave, tonal and residual fractions, occupied
bandwidth, occupied frequency fraction, dominant bin frequency, dominant peak
frequency when available, dominant peak power fraction, configured band energy
metrics, validity and diagnostics.

The dominant frequency is explicitly the strongest FFT bin in the analyzed
range. If that bin is not a significant detected peak, the frame records
`dominant_bin_not_significant_peak`.

## 9. Temporal summary

`TimeResolvedSpectralSummary` reports valid time coverage, initial/final/max
energy, time of maximum energy, initial/final/min/max flatness, entropy and
centroid, initial/final tonal fraction, density and occupied bandwidth, frame
fractions above configured tonal/residual thresholds, and operational
late-minus-early changes.

The default change orientation is:

`change = late_region_median - early_region_median`.

Negative flatness change means the late median is lower than the early median;
it is not interpreted as modalization.

## 10. Temporal trends

`SpectralMetricTemporalFit` implements simple linear least-squares regression.
It reports slope, intercept, R2, RMSE, available/finite/used counts, time span,
success/failure and diagnostics.

Fits are computed for:

- energy in dB relative to the maximum positive frame energy;
- centroid;
- spread;
- flatness;
- entropy;
- significant peak count;
- peak density;
- tonal fraction;
- residual fraction;
- occupied bandwidth.

Regression is not forced when there are too few points or non-distinct times.

## 11. Energy in log domain

Energy trend uses:

`10 * log10(frame_energy / max_positive_frame_energy)`.

Zero or negative energy frames are discarded from this trend without hidden
epsilon. The slope is a descriptive global spectral-energy trend, not modal
decay rate and not a Q factor.

## 12. Smoothing

Optional smoothing is configured through `smoothing_method` (`none`, `median`,
`mean`) and `smoothing_window_frames`. It is disabled by default.

Raw per-frame metrics remain stored in `frames`. Smoothing is used only by the
operational change-point detector and is diagnosed; it does not replace the
original metric series.

## 13. Change points

`SpectralMetricChangePoint` records metric name, frame, time, median value
before and after, difference, relative difference, threshold, direction,
persistence and diagnostics.

The implemented policy is deterministic adjacent-window median difference.
A point is accepted when the before window is stable enough, the median
difference reaches the inclusive threshold, and enough after-window frames
persist in that direction. These are operational markers, not physical regime
transitions.

## 14. Early, middle and late regions

`SpectralTemporalRegionSummary` computes robust medians for configured
relative-time regions. Regions are ordered and non-overlapping. Empty regions
and regions containing only invalid frames are preserved with diagnostics.

The default regions are early `[0.0, 0.15) s`, middle `[0.15, 0.45) s` and late
`[0.45, 0.90) s`, but tests configure them explicitly for each synthetic case.

## 15. Tonality evolution

The summary exposes:

- `tonal_fraction_change`;
- `residual_fraction_change`;
- `flatness_change`;
- `entropy_change`;
- `peak_density_change`.

All are `late - early` medians when both regions are valid. They are
descriptive and carry no automatic direction of interpretation.

## 16. Centroid and rolloff evolution

Centroid and rolloff are inherited from the per-frame global characterization.
A controlled two-component signal with a fast-decaying 640 Hz component and a
slower 128 Hz component produced a negative centroid trend of approximately
`-194.45 Hz/s`. Its 95% rolloff moved from 640 Hz to 128 Hz.

## 17. Density

Peak density per hertz is frame-local significant peak count divided by the
analyzed bandwidth. Peak density per octave is available only when the lower
frequency bound is positive.

Controlled successive frames containing one, three and six sinusoidal
components produced significant peak counts `[1, 3, 6]` and increasing density
per hertz.

## 18. Bands

Frame band energy reuses `SpectralBandEnergy` from the global characterization.
`TimeResolvedSpectralBandSummary` reports initial/final/max energy,
initial/final fraction, fraction trend, coverage above a configured energy
fraction threshold, and the first time until the band falls below that
threshold.

No IEC or normative octave filtering is implemented.

## 19. Comparability

`evaluate_time_resolved_spectral_comparability` reports incompatibilities in:

- sample rate;
- frame duration;
- hop duration;
- FFT size;
- window;
- detrending;
- frequency range;
- analysis window;
- impact time;
- peak criteria;
- normalization policy;
- early/middle/late regions;
- smoothing;
- change-point policy.

It does not compare the signals and does not normalize recordings.

## 20. Public contracts and functions

The public immutable contracts are:

- `TimeResolvedSpectralCharacterizationSettings`;
- `TimeResolvedSpectralFrame`;
- `TimeResolvedSpectralCharacterization`;
- `TimeResolvedSpectralSummary`;
- `SpectralMetricTemporalFit`;
- `SpectralMetricChangePoint`;
- `SpectralTemporalRegionSummary`;
- `TimeResolvedSpectralBandSummary`;
- `TimeResolvedSpectralComparabilityResult`.

The public functions are:

- `characterize_time_resolved_spectrum`;
- `characterize_recording_time_resolved_spectrum`;
- `evaluate_time_resolved_spectral_comparability`.

## 21. Tests added

Forty-one tests were added. They cover:

- one-frame equivalence with global characterization;
- exponentially decaying sine;
- fixed-seed decaying white noise;
- broadband attack followed by tonal tail;
- tonal attack followed by noisy tail;
- descending centroid and rolloff;
- peak density;
- zero lower frequency for octave density;
- silent, weak, nonfinite and out-of-window frames;
- partial-frame discard and opt-in temporal padding;
- hop greater than frame duration;
- early/middle/late regions;
- linear temporal trends and insufficient-point failure;
- inclusive persistent change points;
- isolated outlier rejection;
- band persistence and energy conservation;
- invalid settings;
- time-resolved comparability;
- deterministic equality.

## 22. Synthetic broadband attack example

The central synthetic test combines fixed-seed broadband noise before
approximately 0.22 s with a persistent damped 128 Hz sine tail starting at
0.15 s.

With 125 ms frames and 62.5 ms hop:

- early median flatness was approximately `0.4843`;
- late median flatness was approximately `0.000110`;
- early median entropy was approximately `0.8607`;
- late median entropy was approximately `0.00938`;
- early median tonal fraction was approximately `0.733`;
- late median tonal fraction was approximately `0.995`;
- operational change points were reported near the constructed transition.

These values describe the constructed signal only. They do not classify a
physical regime.

## 23. Validation result

The suite grew from 574 to 615 tests:

- `pytest`: 615 passed.
- `pytest -W error`: 615 passed.
- `python3 -m compileall -q belllab tests`: passed.
- `git diff --check`: passed.

No additional static checker is configured in `pyproject.toml`.

## 24. Limitations

Frame spectral energy remains relative to the existing FFT amplitude
normalization, not calibrated acoustic energy or PSD. Window-specific main-lobe
resolution is not yet folded into `frequency_resolution_hz`; the frame duration
sets the explicit conservative physical-resolution limit.

Change points are simple operational median differences. They are not
statistical structural-break tests and do not infer physical causality.

Band summaries are simple spectral-bin aggregations. They are not IEC octave or
third-octave analyses. Dominant frequency is the strongest analyzed bin, not a
tracked candidate and not a mode.

## 25. Next steps

Future layers may compare time-resolved characterizations between dynamic
levels only after explicit compatibility checks. Such a layer should remain
separate from regime classification, modal identification and `ModalMode`
construction.
