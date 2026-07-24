# Development Report — Spectral Peaks v1

**Date:** 2026-07-23  
**Initial test state:** 52 tests passed.

## Changes made

### Bin-spacing terminology

`Spectrum.bin_spacing_hz` is now the canonical name for the FFT-grid spacing,
equal to `sample_rate / n_fft`. `frequency_resolution_hz` remains a compatible
read-only legacy property returning the same value. No deprecation warning was
emitted because the project has no deprecation policy yet.

Bin spacing is not spectral resolution. The window main-lobe width and selected
time interval determine practical separation of nearby components. Zero padding
creates a denser interpolated FFT grid but neither shortens the observation nor
creates information; it therefore does not improve physical separability.

### Contracts and API

- `SpectralPeak` represents a mathematical peak observation, not `ModalMode`.
- `PeakDetectionSettings` contains only frequency, height, prominence, distance,
  width, count, interpolation, local-floor and ordering parameters currently
  used by the detector.
- `PeakDetectionResults` retains the analyzed `Spectrum` by reference, ordered
  peak contracts, effective settings, method, diagnostics, warnings, candidate
  count, accepted count, and analyzed frequency range.
- `detect_spectral_peaks(spectrum, settings=None)` is publicly exported.

## Detection method

Detection operates on an existing `Spectrum`; it never recalculates the FFT.
It restricts the requested frequency range, calls `scipy.signal.find_peaks`,
then uses SciPy prominence and `peak_widths(..., rel_height=0.5)`. Width is
therefore explicitly **half-prominence width**, in bins and Hz. No minimum
prominence is applied by default; users must provide a threshold in the
appropriate linear or dB scale when their use case requires one.

For linear amplitude spectra, height and prominence thresholds are linear
amplitudes. For dBFS spectra, they are dB level/difference values. Local SNR is
always returned in dB: a linear amplitude ratio converted with `20 log10`, or a
dBFS level difference. The local floor is a median of a configurable local bin
neighborhood excluding the peak and immediate neighbors. It is an operational
spectral floor, not physical noise measurement.

Three-point log-magnitude parabolic interpolation refines interior peaks only.
It returns no refinement and records a diagnostic for DC/Nyquist/border,
nonpositive neighbors, flat curvature, or an out-of-neighbor result. Refined
frequency and amplitude are operational estimates, not exact frequencies or
formal uncertainties.

## Tests

New quantitative tests cover bin-centred and off-bin tones, separated and close
tones, disparate amplitudes, white/pink noise, DC, silence, impulse, damped
signal, frequency restriction, dB spectra, zero padding, maximum/invalid
settings, local floor/SNR, and interpolation improvement. Existing FFT tests
now assert canonical bin spacing and the legacy alias.

## Validation

Final suite: **63 passed**. `compileall`, public import verification and
`git diff --check` were also run. Ruff is not installed in the environment.

## Limitations

- No STFT, tracking, peak association across time, modal classification, Q,
  physical calibration, formal uncertainty, PSD/power scale, or real-recording
  validation is included.
- Local-floor and prominence defaults are heuristics needing validation with
  representative historic-bell and other idiophone recordings.
- Only a local median floor and log-parabolic interpolation are supported.
- Edge bins are not interpolated; peaks at DC/Nyquist are intentionally not
  promoted to refined estimates.

## Recommendation

Validate the stationary spectrum and peak layer against curated real WAV
recordings plus independently annotated reference peaks before adding STFT or
modal analysis. The next algorithmic module should be spectral-peak diagnostics
or validation tooling, not physical mode classification.
