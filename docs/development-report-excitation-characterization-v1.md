# Development report — excitation characterization v1

Date: 2026-07-24

## Initial state and scope

The validated initial suite contained 426 passing tests. This work separates
three layers that must not be conflated:

`musical dynamic label` ≠ `absolute physical intensity` ≠ `calibrated acoustic level`.

`pp`, `p`, `mf`, `f` and `ff` remain supplied experimental categories. The
implementation measures the effective recorded waveform but does not rename
labels, normalize recordings, compare modal candidates across dynamic levels or
construct `ModalMode`.

## Files changed

- `belllab/excitation.py`
- `belllab/within_condition.py`
- `belllab/__init__.py`
- `tests/test_excitation_characterization.py`
- `README.md`
- `docs/RFC-0001-scientific-specification.md`
- `docs/development-report-excitation-characterization-v1.md`

## Public contracts and functions

The public immutable contracts are:

- `ExcitationCharacterizationSettings`;
- `ExcitationCharacterization`;
- `DynamicOrderPairResult`;
- `DynamicOrderConsistencyResult`.

The public functions are `characterize_excitation_signal`,
`characterize_excitation` and `evaluate_dynamic_order_consistency`. The
signal-level function is purely numerical and performs no file access. The
recording adapter uses an already loaded `Signal`. All results are deterministic
and inputs are not mutated.

`ExcitationCondition` now optionally preserves microphone and interface IDs,
channel, acquisition gain, microphone distance and orientation, exciter mass,
operator label and notes in addition to the previous session, location and
exciter fields.

## Windows and channel policy

Default relative windows are configurable:

- background: `[-1.0, -0.1] s`;
- excitation: `[-0.01, +0.20] s`.

Bounds are relative to `impact_time_s` and inclusive with a configurable
`1e-12 s` numerical tolerance. The background window must end before the
excitation window begins. Multichannel signals require an explicit
`channel_index`; the default is channel 0 and channels are never averaged
silently.

## Units and power convention

Normalized floating-point units use digital reference amplitude 1 by default.
Integer PCM uses dtype-aware full scale when its NumPy dtype is preserved;
plain Python integers, which contain no bit-depth information, still require
explicit `pcm_full_scale`. Signed 16-bit samples use scale 32768, yielding
-32768 → -1.0. This prevents direct comparison of raw integer values from
different bit depths. A physical unit such as `Pa` remains unchanged and does
not produce dBFS or digital-clipping claims.

No dB SPL is produced. `equivalent_level_dbfs` is
`20 log10(RMS / digital reference)` only when that reference exists. A silent
window returns `None`, not `-inf`, and records `silent_window`.

## Peak, RMS, energy and offset

Peak metrics preserve the original waveform: absolute peak, signed dominant
peak, peak time, positive peak and negative peak. Peak asymmetry is

`(|positive peak| - |negative peak|) / absolute peak`.

It is descriptive and is not interpreted as physical nonlinearity.

Power metrics use:

- `mean_square_amplitude = mean(x²)`;
- `rms_amplitude = sqrt(mean_square_amplitude)`;
- `signal_energy = sum(x²) / sample_rate`.

For ten samples of amplitude 0.5 at 100 Hz, peak and RMS were 0.5, mean square
was 0.25, energy was 0.025 amplitude-unit²·s, level was -6.0206 dBFS and crest
factor was 1. A twenty-sample window kept RMS at 0.5 and doubled energy to 0.05.
This energy is a discrete relative signal measure, not calibrated acoustic
energy.

The raw mean/DC offset is always reported. By default RMS and energy use the
original samples. With `remove_dc_for_power=True`, only power metrics use the
centered waveform and the policy is diagnostic. For alternating 0.25/0.75,
raw RMS was `sqrt(0.3125) = 0.559017`; centered RMS was 0.25.

## Crest factor, impulse and attack

Crest factor is `absolute peak / RMS`; its dB form is `20 log10(crest factor)`.
A unit impulse among ten samples produced RMS 0.316228, energy 0.01 and crest
factor 3.162278.

Impulse duration uses cumulative squared-amplitude percentiles, defaulting to
5% and 95%. With four equal-energy samples and configured 25%/75% percentiles,
the measured interval was 0.02 s. Scaling every amplitude by three preserved
the start, end and duration.

Attack begins at the first sample whose absolute amplitude reaches the
configured fraction of the peak. `time_to_peak_s` is referenced to
`impact_time_s`; attack duration is peak time minus operational attack start.
A peak before the estimated impact is preserved as a negative time-to-peak and
diagnosed rather than forced positive.

## Clipping and near-clipping

For digitally referenced signals:

- clipping: `abs(x) >= clipping_threshold`;
- near clipping: `abs(x) >= near_clipping_threshold`.

Defaults are 0.999 and 0.95. Equality is inclusive. The result exposes flags,
counts, fractions and longest consecutive clipped run. Three consecutive unit
samples among five produced count 3, fraction 0.6 and longest run 3.
Near-clipping without clipping is distinguished. Positive and negative limits
and normalized integer PCM are covered. No reconstruction or correction is
performed.

## Background and signal-to-background ratio

The pre-impact window reports available/finite sample counts, RMS, discrete
energy, peak, standard deviation and prior clipping. Nonfinite samples are
discarded with diagnostics. The ratios are:

- `signal_to_background_ratio = excitation RMS / background RMS`;
- `signal_to_background_db = 20 log10(ratio)`.

Excitation twice a 0.1 background produced ratio 2 and 6.0206 dB; excitation ten
times that background produced ratio 10 and 20 dB. Zero background and missing
background never cause division by zero or invalidate an otherwise valid
excitation; they return unavailable ratios with explicit diagnostics.

## Acquisition metadata and comparability

Optional acquisition metadata are validated only when present: strings must be
nonempty, channel non-negative, gain finite, and microphone distance and
exciter mass positive. No missing value is invented.

An individual result records
`cross_recording_amplitude_comparability_unverified` when core metadata are
incomplete. Dynamic-order evaluation also diagnoses differing or incomplete
microphone, interface, gain, distance, selected channel or amplitude unit.
Direct physical comparison is therefore supported only when these acquisition
conditions are compatible and signals are not clipped. Samples are never
automatically normalized between recordings.

## Dynamic-label consistency

`evaluate_dynamic_order_consistency` normalizes input order to
`pp < p < mf < f < ff` and compares adjacent available labels using RMS,
signal energy or absolute peak. Each pair records values, tolerance and
`ordered`, `tie` or `inversion`.

The controlled RMS sequence 0.05, 0.09, 0.16, 0.28 and 0.50 was consistent.
The sequence with p=0.20 and mf=0.16 produced one explicit inversion. Exact
ties and differences within a configured tolerance are distinguished.
Missing conditions are diagnosed, and a single condition is reported as
insufficient rather than consistent. Mixed sessions, duplicate labels,
nonfinite metrics and unspecified labels are rejected. No label is renamed.

## Invariants and validation

Contracts validate finite ordered windows, coherent percentiles and clipping
thresholds, explicit PCM scale, coherent sample counts, non-negative power and
duration metrics, `mean_square = RMS²`, peak bounds, crest factor, clipping
counts/fractions, positive ratios, success/failure state and unique diagnostics.
Dynamic results validate normalized unique labels, finite values and exact
agreement of inversion/tie counts with pairwise results.

Sixty-five tests were added. The final suite contains 491 tests.

- `pytest`: 491 passed
- `pytest -W error`: 491 passed
- `python3 -m compileall -q belllab tests`: passed
- `git diff --check`: passed

No additional static checker is configured in `pyproject.toml`.

## PCM normalization closure

The PCM input policy was closed on 2026-07-24 without changing any excitation
metric or downstream scientific criterion.

Signed integer PCM uses:

`normalized = (sample - 0) / max(abs(dtype_min), abs(dtype_max))`.

Thus `int16` maps -32768 to -1.0, 0 exactly to 0, and 32767 to
32767/32768 = 0.9999694824. `int8` analogously maps -128 to -1.0 and 127 to
127/128 = 0.9921875. The single scale preserves linearity and explicitly keeps
the unavoidable extra negative two's-complement code.

Unsigned integer PCM uses:

`normalized = (float(sample) - zero_point) / full_scale`,

where the default zero point is the midpoint inferred from `numpy.iinfo(dtype)`
and full scale is the larger distance from that midpoint to either dtype
extreme. For `uint8`, zero point and scale are both 128:

- 0 → -1.0;
- 64 → -0.5;
- 127 → -1/128;
- 128 → 0;
- 129 → +1/128;
- 192 → +0.5;
- 255 → 127/128 = 0.9921875.

This is one linear scale rather than a piecewise positive/negative
transformation. Consequently the positive endpoint is slightly below +1,
which is the documented counterpart of the integer-code asymmetry. The code
converts the complete array to `float64` before subtracting the unsigned zero
point, preventing `uint8 - 128` wraparound.

`pcm_full_scale` remains a compatible public override and `pcm_zero_point` is
available when an explicit unsigned midpoint is required. Overrides must be
finite, the zero point must lie inside dtype limits, signed PCM requires zero
point 0, and the scale must cover both extrema. PCM parameters supplied for
floating-point data, boolean/object/complex samples, zero or negative scales
and nonfinite values are rejected clearly.

Audit diagnostics record the input numeric dtype, signedness, linear
normalization policy, zero point, scale, whether parameters were inferred or
configured, and conversion-before-centering. The WAV loader's `PCM_U8` path was
also verified: libsndfile returns centered `int16` codes, which are then
normalized by the signed input-dtype policy without artificial DC offset.

Clipping and near-clipping continue to operate after normalization. With the
default 0.999 clipping threshold, uint8 code 0 clips at -1.0; code 255 is
near-clipping but not clipping because it maps to 0.9921875. With a configured
0.99 threshold, code 255 clips. Code 192 maps exactly to 0.5 and is included by
an inclusive 0.5 threshold, while code 191 remains below it.

A centered uint8 sequence `[64, 192, 64, 192]` produced DC offset 0, RMS 0.5,
mean square 0.25, energy 0.01 at 100 Hz and equivalent level -6.0206 dBFS.
Equivalent float, int16 and uint8 encodings of `[-0.5, 0, +0.5]` produced equal
RMS, energy, DC offset and dBFS within numerical tolerance.

Thirty-eight PCM closure tests were added. The complete suite grew from 491 to
529 tests:

- `pytest`: 529 passed;
- `pytest -W error`: 529 passed;
- `python3 -m compileall -q belllab tests`: passed;
- `git diff --check`: passed.

## Limitations and next steps

The characterization is relative to the recorded digital or declared signal
unit. It does not calibrate pressure, compensate microphone response, infer
impact force, correct clipping, classify nonlinear regimes, analyze spectral
flatness, associate different dynamics, split or merge modal observations,
estimate Q or construct global modal families. A future calibrated workflow
may use the preserved acquisition metadata and raw metrics to establish
traceable cross-recording normalization. No conversion to `ModalMode` was
implemented.
