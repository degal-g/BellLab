# Development Report — Spectrum v1

**Date:** 2026-07-23  
**Scope:** minimum temporal-contract consolidation and first stationary FFT
implementation.  
**Excluded:** STFT, modal analysis, graphics, reports, batch processing,
campaign infrastructure, and broad architectural redesign.

## 1. Initial test state

Before changes, the complete suite passed with **32 tests**. The repository had
no Git commits and all project files were untracked, so no partial commits were
created: doing so would have staged unrelated initial project content.

## 2. Files changed

- `belllab/config.py`
- `belllab/types.py`
- `belllab/results.py`
- `belllab/recording.py`
- `belllab/comparison.py`
- `belllab/temporal.py`
- `belllab/spectrum.py`
- `belllab/__init__.py`
- `pyproject.toml`
- `README.md`
- `tests/test_models.py`
- `tests/test_temporal.py`
- `tests/test_types.py`
- `tests/test_spectrum.py` (new)

No new runtime dependency was introduced. The existing NumPy dependency
provides the real FFT implementation.

## 3. Minimum architectural changes

### Temporal contracts

`ImpactReport`, `NoiseReport`, and `TemporalMetrics` were moved to `types.py`.
They are data contracts and can therefore be imported by `results.py` without a
circular import. `temporal.py` continues to re-export those imported names for
reasonable backward compatibility with existing imports.

`TemporalResults` is now the canonical aggregate for a complete temporal run:

- `impact: ImpactReport | None`
- `noise: NoiseReport | None`
- `metrics: TemporalMetrics | None`
- `envelope: Envelope | None`
- `decay_fit: DecayFit | None`
- `settings: TemporalAnalysisSettings | None`

`None` means the quantity was not calculated or was unavailable in that run.
The current full temporal execution calculates impact, noise, global metrics and
one configured envelope; exponential decay fitting remains unavailable.

`NoiseReport` is the canonical temporal-noise representation. `NoiseMetrics`
was deliberately retained as a documented legacy public type rather than being
silently removed, but it is no longer used by `TemporalResults`.

### Impact baseline bug

The audit finding was confirmed. `Signal.samples` is arranged as
`(channels, samples)`, and `detect_impact` reduced it to a one-dimensional
temporal amplitude vector. The old baseline bound used `samples.shape[0]`, which
is the number of channels; mono and stereo data consequently used a one-sample
baseline. The implementation now uses `amplitude.size`, the explicit temporal
axis length.

New synthetic tests run the detector for mono and stereo inputs constructed so
the former implementation reports sample zero incorrectly while the corrected
implementation reports the onset at sample 100.

### Settings

`AnalysisSettings` now groups only two implemented configurations:

- `TemporalAnalysisSettings`: noise window duration and envelope method/window;
- `SpectrumAnalysisSettings`: channel policy, interval, DC removal, window,
  FFT length, output scale and normalization.

Defaults preserve temporal behavior where possible: a 50 ms noise window and a
Hilbert envelope. `analyze_temporal` no longer discards settings and stores the
effective temporal settings in its result.

### Recording and context

`Recording` remains the domain object and `ProcessingContext` remains the
immutable description of an analytic execution. Their documentation now states
that the context owns execution results while running; a `Recording` may later
receive an associated snapshot, but callers must not mutate both as concurrent
canonical state.

`instrument_id` was added as an optional generic identifier. The required
legacy `bell_id`, `BellRecording`, and `BellComparison` APIs remain intact.
`Experiment` is documented accurately as the current binary-comparison model;
campaigns remain out of scope.

### Contract invariants

`Signal` now validates positive sample rate/channels, channel count and length,
time-axis length/order, and duration consistency. `Envelope` validates paired
ordered finite series. `Spectrum` validates paired ordered non-negative
frequencies, rejects NaN magnitudes, and defines `overlap` as a fraction in
`[0, 1)`. Parameter mappings are copied into read-only mappings for `Envelope`
and `Spectrum`.

## 4. FFT convention and implementation

`analyze_spectrum(signal, settings=None) -> SpectrumResults` implements one
stationary real FFT. It does not implement STFT.

1. Select the configured half-open interval `[start_time_s, end_time_s)`.
2. Select `channel_index` (default zero) or compute a channel mean only when
   `channel_policy="mean"` is explicit.
3. Remove the segment mean by default.
4. Apply a rectangular or Hann window.
5. Compute `numpy.fft.rfft` with `n_fft >= original_size`; a larger value is
   zero padding.
6. Normalize magnitude by `original_size * coherent_gain`.
7. Double only internal one-sided bins. DC and Nyquist for even `n_fft` remain
   undoubled.

The resulting linear scale is **normalized peak amplitude**. Thus a real sine
centred in a bin recovers its peak amplitude after coherent-gain correction.
`dbfs` applies `20 log10(amplitude)` with amplitude reference one; it is an
amplitude dBFS scale, not power dBFS. Exact zero is represented by `-inf`, not
NaN. The documentation and `Spectrum.parameters` record this reference,
coherent gain and one-sided convention.

Finite Hann windows cause a few parts-per-million numerical deviation for the
synthetic bin-centred sine cases. Amplitude recovery tests therefore use a
documented relative tolerance of `1e-5`, which is narrow enough to expose
missing coherent-gain correction or incorrect one-sided doubling.

`Spectrum` now records the original segment length, FFT size, sample rate,
frequency resolution, channel policy, normalization, interval, DC-removal flag
and effective magnitude parameters. `SpectrumResults` holds the spectrum,
effective settings and any minimal diagnostics (currently zero-padding notice).

## 5. Multichannel decision

The default policy is `select`, channel zero. It intentionally avoids silent
phase cancellation. A synthetic stereo test uses two equal, opposite-phase
channels: the default recovers amplitude 0.6, while the explicit `mean` policy
returns zero at the same bin. Per-channel multi-spectrum output is not included
in this incremental version.

## 6. Test additions

The suite now covers:

- mono and stereo regression cases for the impact baseline axis;
- complete `TemporalResults` aggregation and effective temporal settings;
- `Signal`, `Envelope`, and `Spectrum` invariants;
- generic `instrument_id` together with legacy `bell_id`;
- bin-centred and between-bin sine waves;
- two-tone amplitude recovery;
- DC removal, null signal and dBFS zero behavior;
- impulse DC/interior/Nyquist normalization;
- damped sine behavior;
- explicit multichannel phase cancellation;
- rectangular and Hann windows;
- zero padding;
- invalid channel, FFT size and time interval inputs.

## 7. Validation

Final validation commands and results:

```text
python3 -m pytest -q                 52 passed
python3 -m compileall -q belllab tests  passed
git diff --check                     passed
```

The optional Ruff static analyzer is not installed in the environment;
`compileall` supplied syntax validation. A manual example generated a 128 Hz,
0.5-amplitude sine and recovered:

```text
peak_hz=128.000000
peak_amplitude=0.499999995
resolution_hz=1.000000
```

Existing public imports remain available, and `analyze_temporal` accepts its
former `AnalysisSettings` argument while also permitting `None`.

## 8. Known limitations

- `NoiseMetrics` remains a legacy duplicate until a future deprecation policy;
  it is not converted automatically for callers.
- `Signal` source fields (`sha256`, `loaded_at`, and source path metadata) are
  not yet populated by `load_wav`.
- Temporal confidence is heuristic, not calibrated uncertainty.
- The FFT supports only mono selection or explicit temporal mean, rectangular
  and Hann windows, amplitude linear/dBFS scales, and one stationary segment.
- No spectral phase, power/PSD, STFT, uncertainty, calibration to physical
  pressure, modal interpretation or comparison is implemented.
- Long recordings are still represented in memory as tuples; no batch or
  persistent-results infrastructure was added.

## 9. Recommended next module

Implement a narrowly scoped **spectral peak/diagnostic layer** on top of the
validated stationary `Spectrum` contract before STFT or modal analysis. It
should first define peak-selection criteria, threshold provenance and frequency
uncertainty. This exercises the new spectrum result model with real scientific
use while avoiding premature modal assumptions.
