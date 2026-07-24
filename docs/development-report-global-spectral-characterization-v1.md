# Development report — global spectral characterization v1

Date: 2026-07-24

## Initial state and scientific boundary

The validated initial suite contained 529 passing tests. This work implements
global descriptive spectral metrics while preserving:

`global spectral metric` ≠ `definitive physical diagnosis` ≠
`proof of nonlinearity` ≠ `physical mode`.

No dynamic-level comparison, regime classification, cross-condition candidate
association, modal split/merge, Q, modal family or `ModalMode` conversion was
introduced.

## Files changed

- `belllab/global_spectrum.py` (new);
- `belllab/__init__.py`;
- `tests/test_global_spectral_characterization.py` (new);
- `README.md`;
- `docs/RFC-0001-scientific-specification.md`;
- this report.

## Public contracts and functions

The immutable contracts are:

- `GlobalSpectralCharacterizationSettings`;
- `SpectralBand`;
- `GlobalSpectralPeakMetric`;
- `SpectralBandEnergy`;
- `GlobalSpectralCharacterization`;
- `SpectralComparabilityResult`.

The numerical entry point is `characterize_global_spectrum`. It receives an
existing `Spectrum` and performs no I/O. `characterize_signal_spectrum`
calculates the stationary FFT from a loaded `Signal`, and
`characterize_recording_spectrum` adapts an already loaded `Recording`.
`evaluate_spectral_characterization_comparability` only enumerates
incompatibilities.

## Canonical spectral domain

Linear power is canonical. BellLab stationary linear amplitude is converted
with `P=A²`. dBFS amplitude is recovered with `A=10**(dB/20)` and then
squared. A direct controlled linear-power input must be declared explicitly.
Negative power is rejected. Centroid, moments, rolloff, flatness, entropy,
energy fractions and bands are never calculated from dB values.

The result records original domain, canonical domain and normalization. A
positive configurable power reference is required; no amplitude/power
conversion is inferred from the sign of values.

## Window, detrending and resolution

The signal adapter supports explicit mean removal or no detrending, rectangular
or Hann windows, interval selection and optional FFT size. It reuses the
stationary FFT and its coherent-gain amplitude normalization.

`bin_spacing_hz = sample_rate / fft_size` describes the FFT grid.
`frequency_resolution_hz` is conservatively `1 / effective_duration`. Zero
padding reduces bin spacing but leaves physical resolution unchanged. Peak
widths at or below that resolution are marked `resolution_limited`.

## Distribution metrics

Centroid is `sum(fP)/sum(P)`. Variance, spread, skewness and kurtosis are
weighted population moments. Skewness and kurtosis remain `None` when variance
is numerically zero. Zero total energy makes distribution metrics unavailable
and produces a coherent invalid result rather than NaN or infinity.

Rolloffs are configurable and include 5%, 50%, 85%, 90% and 95% by default.
The result is the frequency of the first bin satisfying
`cumulative >= fraction * total`, without interpolation. The occupied band is
the inclusive interval between 5% and 95% rolloffs by default; its width and
fraction of the analyzed frequency span are reported separately from spread.

## Flatness, entropy and spectral crest factor

Flatness is the geometric-to-arithmetic mean ratio over strictly positive power
bins. Zeros are excluded, their count is reported, at least two positive bins
are required by default, and no hidden epsilon or floor is introduced.

Normalized entropy uses only positive probabilities:
`-sum(p log p) / log(N_positive)`. Zero terms contribute zero and `log(0)` is
never evaluated. A single positive bin has entropy zero; no positive bin makes
entropy unavailable. Both measures are bounded to `[0,1]` only after their
mathematical computation to absorb floating-point roundoff.

Spectral crest factor is `max(P)/mean(P)` over finite analyzed bins. It is
distinct from temporal crest factor. Uniform power produced exactly 1; the
controlled single-bin case across five bins produced 5.

## Peaks, widths and isolation

Peak detection reuses `detect_spectral_peaks` and therefore the existing
deterministic SciPy `find_peaks` infrastructure. Threshold, prominence,
distance and width criteria are explicit and operate in canonical linear
power. Significant peaks are the accepted detector outputs; they are not modal
candidates.

Frequency refinement remains the existing three-point log-parabolic estimate.
Peak boundaries and widths use half prominence in canonical power, with simple
linear interpolation of fractional bin locations. This is deliberately named
half prominence, not half power or uncertainty. Metrics expose bin/refined/
representative frequency, power, relative power, prominence, width in bins and
hertz, both boundaries, resolution limitation and diagnostics.

Isolation is nearest representative-frequency distance divided by peak width.
Intervals with positive intersection are operationally overlapped, touching
intervals partially overlapped, disjoint intervals isolated, and a singleton
indeterminate. Overlap is not interpreted as nonlinear coupling.

## Density and spacing

Peak density per hertz divides significant peak count by analyzed bandwidth.
Density per octave uses `log2(fmax/fmin)` and is unavailable when the lower
frequency is non-positive. Per-band peak counts and densities are also
reported.

Representative peak frequencies are sorted. Their minimum, maximum, mean,
median and population standard deviation of adjacent spacing are computed.
With fewer than two peaks they remain `None` and a diagnostic is recorded.
Spacing is not interpreted as harmonicity.

## Tonal and residual energy

Each significant peak defines a configurable neighborhood based on its
half-prominence width. A boolean union of all neighborhoods is integrated, so
overlapping intervals cannot count a bin twice.

`tonal_energy_fraction = tonal_energy / total_energy` and the residual is its
complement within numerical tolerance. The residual is explicitly not named
noise: it can contain leakage, spectral tails, undetected components,
overlap, background or continuous response. With no peaks, the controlled
nonzero spectrum returned tonal fraction 0 and residual fraction 1.

## Configurable band energy

Bands are ordered, non-overlapping and uniquely labelled. Their boundary policy
is start-inclusive/end-exclusive, preventing double counting between adjacent
bands. Each result contains energy, fraction, square-root energy as an
operational RMS equivalent, bin count, peak count, density and diagnostics.

For controlled power `[1,2,3,4,5]`, adjacent bands `[0,2)` and `[2,5)`
contained energies 3 and 12. Their sum equalled total energy 15 and their
fractions summed to 1.

## Comparability

The diagnostic checks sample rate, frequency range, effective duration, FFT
size, window, detrending, physical resolution, all peak criteria, original and
canonical domain, and normalization. It neither compares metric values nor
normalizes signals. Tests cover identical inputs and simultaneous changes in
FFT size, window, detrending and detector criteria.

## Tests and quantitative examples

Forty-five tests were added. They cover uniform, concentrated, two-peak,
dominant-background, triangular, zero, nonfinite and negative-power spectra;
inclusive rolloffs; flatness and entropy scaling; zeros and insufficient
positive bins; peak widths; density and regular spacing; tonal interval union;
bands and invalid overlaps; settings invariants; dBFS recovery; deterministic
white noise; sine, three sines, sine plus noise and impulse; zero padding; and
comparability.

Examples:

- uniform power on 0–4 Hz: centroid 2 Hz, variance 2 Hz², spread `sqrt(2)`,
  flatness 1, normalized entropy 1 and crest factor 1;
- one powered bin among five: centroid at that bin, spread 0, entropy 0,
  crest factor 5, and undefined skewness/kurtosis;
- three peaks at 2 Hz spacing: minimum, median and maximum spacing all 2 Hz;
- 0 dBFS, -6.0206 dBFS and silence recover total power 1.25;
- fixed-seed white noise has higher flatness and entropy than a sine, while a
  sine-plus-noise mixture lies between them;
- an impulse has greater normalized entropy and occupied bandwidth than a sine;
- 4× zero padding reduces bin spacing from 1 Hz to 0.25 Hz while physical
  resolution remains 1 Hz.

## Validation result

The suite grew from 529 to 574 tests:

- `pytest`: 574 passed.
- `pytest -W error`: 574 passed;
- `python3 -m compileall -q belllab tests`: passed;
- `git diff --check`: passed.

No additional static checker is configured in `pyproject.toml`.

## Limitations and next steps

Power is a relative spectral measure under the stationary FFT normalization,
not calibrated acoustic energy or PSD per hertz. Window main-lobe width is not
yet folded into a window-specific resolution number; `1/duration` is a
conservative explicit lower limit. Half-prominence width is operational.
Edge bins are not peaks under the reused detector. No normative octave or
fractional-octave filter bank was implemented.

Peak counts remain sensitive to window, duration, leakage, resolution,
threshold, clipping and overlap. Flatness and entropy do not prove white noise;
dense or broad spectra do not prove chaos or nonlinearity. Future work may use
these metrics in an explicitly separate cross-dynamic comparison layer, but
this version performs no such comparison and creates no physical mode.
