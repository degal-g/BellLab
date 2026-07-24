# BellLab

BellLab is an open-source scientific framework for temporal, spectral and modal
analysis of struck idiophones.

Historical bells are the project's first application domain. The framework is,
however, designed to analyse any idiophone excited by impact, including gongs,
singing bowls, cymbals, plates, lithophones and archaeological sounding objects.

## Project philosophy

The BellLab core must remain independent of instrument type. Differences between
instruments must be implemented only through domain specializations, never
through changes to the fundamental algorithms.

## Current scope

The package provides stable scientific data contracts, temporal analysis,
stationary one-sided FFT, spectral-peak detection, and STFT. Tracking, modal
interpretation, visualization and reporting capabilities will be expanded
incrementally while preserving reproducibility and public APIs.

## Minimal example

```python
from belllab import AnalysisSettings, SpectrumAnalysisSettings
from belllab.spectrum import analyze_spectrum, detect_spectral_peaks
from belllab.synthetic import pure_sine
from belllab.temporal import analyze_temporal

signal = pure_sine(256.0, duration_s=1.0, sample_rate=4_096, amplitude=0.5)
temporal = analyze_temporal(signal)
spectrum = analyze_spectrum(
    signal,
    SpectrumAnalysisSettings(window_name="hann", scale="linear_amplitude"),
)

peak_index = max(
    range(len(spectrum.spectrum.magnitudes)),
    key=spectrum.spectrum.magnitudes.__getitem__,
)
peak_frequency_hz = spectrum.spectrum.frequencies_hz[peak_index]
peak_amplitude = spectrum.spectrum.magnitudes[peak_index]

peaks = detect_spectral_peaks(spectrum.spectrum)
for peak in peaks.peaks:
    print(
        peak.bin_frequency_hz,
        peak.refined_frequency_hz,
        peak.bin_amplitude,
        peak.prominence,
        peak.local_snr_db,
    )
```

The FFT is a stationary, one-sided amplitude spectrum. It selects channel zero
by default, does not average channels implicitly, removes DC by default, and
uses coherent-gain normalization. See `SpectrumAnalysisSettings` for the
explicit channel, interval, window, scale and zero-padding policies.

## STFT example

```python
from belllab import STFTSettings, analyze_stft
from belllab.synthetic import linear_chirp

signal = linear_chirp(100.0, 500.0, duration_s=1.0, sample_rate=2_048)
result = analyze_stft(
    signal,
    STFTSettings(
        channel_policy="select",
        channel_index=0,
        window_length=256,
        hop_length=64,
        window_name="hann",
        remove_mean=True,
    ),
)
time_frequency = result.time_frequency

# values[frequency_index, time_index]; times are window centres.
first_frame = [row[0] for row in time_frequency.values]
peak_index = max(range(len(first_frame)), key=first_frame.__getitem__)
print(time_frequency.times_s[0], time_frequency.frequencies_hz[peak_index])
print(time_frequency.parameters["detrend_method"])  # "frame_mean"
```

STFT channel selection is explicit: ``"select"`` is the default and
``"mean"`` combines channels only by request. With ``remove_mean=True``,
BellLab subtracts the mean of each frame (``frame_mean``), rather than the mean
of the complete selected interval. The matrix convention is
``values[frequency_index, time_index]``. Window length trades temporal against
frequency selectivity; hop length sets the time spacing. Zero padding refines
FFT bin spacing but does not improve physical spectral resolution or create
new information. STFT is not peak tracking or modal analysis.

## Spectral trajectories

```python
from belllab import (
    FramePeakDetectionSettings,
    PeakDetectionSettings,
    STFTSettings,
    SpectralTrackingSettings,
    analyze_stft,
    detect_stft_peaks,
    track_spectral_peaks,
)
from belllab.synthetic import linear_chirp

signal = linear_chirp(80.0, 240.0, duration_s=2.0, sample_rate=2_048)
stft = analyze_stft(signal, STFTSettings(window_length=256, hop_length=64))
frame_peaks = detect_stft_peaks(
    stft.time_frequency,
    FramePeakDetectionSettings(
        peak_settings=PeakDetectionSettings(min_prominence=0.02),
    ),
)
tracks = track_spectral_peaks(
    frame_peaks,
    SpectralTrackingSettings(
        frequency_tolerance=0.10,
        frequency_distance_unit="relative",
        max_gap_frames=1,
    ),
)
for track in tracks.tracks:
    print(
        track.track_id,
        track.duration_s,
        track.initial_frequency_hz,
        track.final_frequency_hz,
        track.observation_count,
        track.gap_count,
    )
```

A frame peak is an observation in one STFT frame. A spectral trajectory is a
deterministic one-to-one association of such observations over time. Neither is
a physical mode. The initial tracker uses a frequency-tolerance gate and a
Hungarian assignment; it can retain a track through a bounded number of absent
frames, but does not interpolate missing observations. Crossings are inherently
ambiguous for this instantaneous association and can exchange identities.

`characterize_spectral_track(track)` provides descriptive frequency slope,
coverage and an operational amplitude-fit result. For positive linear amplitude
it uses a log-linear fit and may report `decay_tau_s`; for dBFS it fits levels
directly. This is a property of the tracked spectral amplitude, not validated
modal damping and not evidence that a track is a physical mode.

Each track preserves `amplitude_unit` as either `linear_amplitude` or
`dbfs_amplitude`; it is never inferred from whether numerical values are
positive or negative. Linear amplitude uses `ln(A)` versus time. dBFS uses a
linear fit in dB/s and converts a negative slope with
`tau = -20 / (slope_db_per_s * ln(10))`. This operational `tau` is neither Q
nor validated physical modal damping. Association margins are heuristic costs,
not probabilities.

Tracking compares alternatives by row (other peaks for a trajectory) and column
(other trajectories for a peak); its operational margin is the minimum
available margin. These diagnostics do not establish physical identity.

`maximum_association_cost` is an inclusive gate on the total weighted
frequency-plus-amplitude cost. For accepted associations, `near_threshold`
indicates that this absolute cost is at or above the configured fraction of the
maximum; it does not mean that the association is ambiguous. Ambiguity follows
the margin between alternatives, so margin and absolute cost are distinct
diagnostics.

Track characterization remains purely descriptive:

```python
from belllab import characterize_spectral_track

description = characterize_spectral_track(tracks.tracks[0])
print(
    description.frequency_mean_hz,
    description.frequency_fit.slope_hz_per_s,
    description.amplitude_mean,
    description.amplitude_fit.method,
    description.coverage_fraction,
    description.gap_count,
    description.diagnostics,
)
```

The canonical frequency series uses a finite positive interpolated frequency
when available and otherwise falls back to the bin frequency; its global source
is reported as `interpolated`, `bin`, or `mixed`. Frequency and amplitude fits,
coverage, gaps and diagnostics describe a spectral trajectory, not confidence
or physical modal validity. dBFS descriptive averages remain averages in the
level domain and are not converted to linear amplitude.

An operational modal candidate is one further, reversible selection step:

```python
from belllab import ModalCandidateSettings, select_modal_candidates

settings = ModalCandidateSettings(
    minimum_observation_count=5,
    minimum_coverage_fraction=0.7,
    maximum_relative_frequency_stability=0.02,
    require_amplitude_decay=False,
)
characterizations = tuple(
    characterize_spectral_track(track) for track in tracks.tracks
)
candidates = select_modal_candidates(characterizations, tracks, settings)
accepted = tuple(item for item in candidates if item.accepted)
rejected = tuple(item for item in candidates if not item.accepted)
for item in rejected:
    print(item.source_track_id, item.rejection_reasons)
```

Every enabled criterion records its observed value, operator, threshold and
result. Rejected trajectories remain in the result for audit and threshold
revision. A `ModalCandidate` is not converted to `ModalMode` and is not evidence
of a validated physical mode, natural frequency, damping or modal confidence.

Pre-impact evidence can be evaluated independently from candidate selection:

```python
from belllab import PreImpactAnalysisSettings, analyze_preimpact_evidence

evidence = analyze_preimpact_evidence(
    tracks.tracks[0],
    characterizations[0],
    temporal_result.impact.impact_time_s,
    PreImpactAnalysisSettings(
        preimpact_window_start_s=-1.0,
        preimpact_window_end_s=-0.1,
        postimpact_window_start_s=0.02,
        postimpact_window_end_s=0.30,
        minimum_impact_level_increase_db=6.0,
    ),
)
print(evidence.classification, evidence.impact_level_change_db)
```

Window bounds are relative to `impact_time_s`. A pre-existing line is not
rejected automatically: it may be amplified or re-excited by the impact.
Optional candidate criteria can require impact excitation or reject a persistent
background tone, but remain disabled by default.

Candidates from repeated recordings of one dynamic condition can be associated
without interpreting the result as a physical mode:

```python
from belllab import (
    ExcitationCondition,
    RecordingCandidateSet,
    WithinConditionAssociationSettings,
    associate_candidates_within_condition,
)

recordings = (
    RecordingCandidateSet(
        "pp-r1", ExcitationCondition("pp", repeat_index=1), candidates_r1
    ),
    RecordingCandidateSet(
        "pp-r2", ExcitationCondition("pp", repeat_index=2), candidates_r2
    ),
)
association = associate_candidates_within_condition(
    recordings,
    WithinConditionAssociationSettings(
        maximum_absolute_frequency_difference_hz=2.0,
        minimum_repeat_count=2,
    ),
)
print(association.clusters, association.unmatched_candidates)
```

The internal function rejects mixed dynamic labels; the high-level grouping
function separates them first. Cluster frequency is the member median,
unmatched candidates and rejected candidates remain auditable, and no cluster
is converted to `ModalMode`.

Recorded excitation intensity can be characterized independently of the
musical label:

```python
from belllab import (
    ExcitationCharacterizationSettings,
    ExcitationCondition,
    characterize_excitation,
    evaluate_dynamic_order_consistency,
)

condition = ExcitationCondition(
    "mf",
    repeat_index=1,
    session_id="session-01",
    microphone_id="mic-01",
    interface_id="interface-01",
    acquisition_gain=12.0,
    microphone_distance_m=0.75,
)
excitation = characterize_excitation(
    recording,
    condition,
    temporal_result.impact.impact_time_s,
    ExcitationCharacterizationSettings(channel_index=0),
    recording_id="mf-repeat-01",
)
print(
    excitation.peak_absolute_amplitude,
    excitation.rms_amplitude,
    excitation.signal_energy,
    excitation.clipping_detected,
    excitation.background_rms,
    excitation.signal_to_background_db,
)

order = evaluate_dynamic_order_consistency(
    session_characterizations,
    metric="rms_amplitude",
)
print(order.consistent, order.inversion_count, order.diagnostics)
```

These are relative digital or declared-unit measurements. dBFS is computed only
when a digital reference exists and is never reported as dB SPL. Musical labels
are neither renamed nor normalized automatically; direct comparison requires
compatible microphone, interface, gain, distance, channel and amplitude unit.
Integer PCM normalization is dtype-aware: signed PCM uses zero point 0, while
unsigned PCM is centered at its inferred midpoint before one linear full-scale
division. Conversion to floating point precedes unsigned subtraction.

## Global spectral characterization

```python
from belllab import (
    GlobalSpectralCharacterizationSettings,
    SpectralBand,
    characterize_signal_spectrum,
)

global_spectrum = characterize_signal_spectrum(
    signal,
    GlobalSpectralCharacterizationSettings(
        frequency_min_hz=20.0,
        frequency_max_hz=8_000.0,
        peak_min_prominence=1e-5,
        bands=(
            SpectralBand("low", 20.0, 500.0),
            SpectralBand("mid", 500.0, 2_000.0),
            SpectralBand("high", 2_000.0, 8_000.0),
        ),
    ),
    recording_id="recording-01",
)
print(
    global_spectrum.spectral_centroid_hz,
    global_spectrum.spectral_flatness,
    global_spectrum.spectral_entropy,
    global_spectrum.tonal_energy_fraction,
)
```

The canonical distribution domain is linear power, obtained explicitly by
squaring linear amplitude or recovering amplitude from dBFS before squaring.
Centroid, spread, rolloff, entropy and energy fractions are never computed in
dB. Flatness uses only strictly positive bins and no hidden epsilon; entropy
omits zero-probability terms. Peak widths use half prominence in canonical
power. Overlapping tonal intervals are united before integration.

`bin_spacing_hz` is the FFT sampling grid, while
`frequency_resolution_hz` is limited by the effective analyzed duration.
Zero padding changes only the first. Flatness, entropy, peak density, residual
energy and occupied bandwidth are descriptive metrics: they do not establish
white noise, chaos, structural nonlinearity, modal identity or any other
physical regime. `evaluate_spectral_characterization_comparability` only lists
configuration incompatibilities; it does not compare dynamic levels or
normalize recordings.

## Time-resolved spectral characterization

```python
from belllab import (
    SpectralBand,
    TimeResolvedSpectralCharacterizationSettings,
    characterize_time_resolved_spectrum,
)

time_resolved = characterize_time_resolved_spectrum(
    signal,
    impact_time_s=0.0,
    settings=TimeResolvedSpectralCharacterizationSettings(
        frame_duration_s=0.125,
        hop_duration_s=0.0625,
        fft_size=512,
        frequency_min_hz=20.0,
        frequency_max_hz=8_000.0,
        peak_min_prominence=1e-5,
        bands=(
            SpectralBand("low", 20.0, 500.0),
            SpectralBand("mid", 500.0, 2_000.0),
            SpectralBand("high", 2_000.0, 8_000.0),
        ),
    ),
    recording_id="recording-01",
)
print(
    time_resolved.summary.initial_entropy,
    time_resolved.summary.final_entropy,
    time_resolved.summary.tonal_fraction_change,
)
```

Frames are complete windows inside the requested analysis interval by default;
temporal padding is opt-in and diagnostic. Each valid frame reuses the global
spectral characterization, so centroid, rolloff, flatness, entropy, peak
density, tonal fraction, residual fraction, occupied bandwidth and band energy
use the same canonical linear-power policy. Silent, weak or nonfinite frames
remain in the sequence with `valid=False` and an explicit failure reason.

The summary exposes initial/final values, robust early/middle/late medians,
linear temporal fits, simple persistent change points and band persistence.
These are operational descriptors of spectral evolution after impact. They are
not formal regime transitions, proof of nonlinearity, chaos detection, modal
identification or conversion to `ModalMode`.

## Dynamic-condition comparison

```python
from belllab import (
    DynamicConditionComparisonSettings,
    DynamicConditionRecordingAnalysis,
    compare_dynamic_conditions,
)

analyses = (
    DynamicConditionRecordingAnalysis(
        recording_id="bell-pp-01",
        condition=pp_condition,
        excitation=pp_excitation,
        global_spectrum=pp_global_spectrum,
        time_resolved=pp_time_resolved,
    ),
    DynamicConditionRecordingAnalysis(
        recording_id="bell-ff-01",
        condition=ff_condition,
        excitation=ff_excitation,
        global_spectrum=ff_global_spectrum,
        time_resolved=ff_time_resolved,
    ),
)

comparison = compare_dynamic_conditions(
    analyses,
    DynamicConditionComparisonSettings(
        enabled_metrics=(
            "excitation_rms_amplitude",
            "global_spectral_centroid_hz",
            "global_spectral_flatness",
            "early_spectral_flatness",
            "late_tonal_energy_fraction",
        ),
    ),
)
```

Dynamic comparison operates on summaries of repeats within each nominal dynamic
label (`pp`, `p`, `mf`, `f`, `ff`). Repeats are aggregated by median, mean,
standard deviation, minimum, maximum and missing-value counts. Pairwise
comparisons preserve the musical order of labels, report missing conditions,
expose optional reference comparisons against `pp`, and evaluate descriptive
monotonicity per metric.

Instrumental comparability is granular: amplitude metrics can be marked
`not_comparable` when gain, microphone, distance, unit or clipping differ,
while scale-invariant spectral metrics can remain comparable if their spectral
settings match. The result describes changes in excitation, global spectrum,
time-resolved trends, early/middle/late regions, bands and clipping diagnostics.
It does not classify linearity, prove nonlinearity, associate individual
candidates across dynamic conditions or convert anything to `ModalMode`.

## Operational response-regime descriptors

```python
from belllab import (
    ResponseRegimeDescriptorSettings,
    describe_dynamic_response_regimes,
    describe_response_regime,
)

pp_description = describe_response_regime(
    pp_condition_summary,
    ResponseRegimeDescriptorSettings(
        minimum_signal_to_background_db=20.0,
        reject_clipped_conditions=False,
    ),
)
print(
    pp_description.structure_descriptor,
    pp_description.temporal_evolution_descriptor,
    pp_description.line_identity_descriptor,
    pp_description.confidence_descriptor,
)

dynamic_descriptions = describe_dynamic_response_regimes(comparison)
for sequence in dynamic_descriptions.descriptor_sequences:
    print(sequence.dimension, sequence.descriptors)
```

Response-regime descriptors use only already computed condition summaries and
dynamic-comparison results. They evaluate explicit weighted criteria for
independent dimensions: spectral structure, temporal evolution, operational
line identity and evidence quality. Examples of structural descriptors include
`discrete_line_dominated`, `mixed_line_and_continuum`, `dense_spectrum` and
`broadband_dominated`; temporal descriptors include `broadband_to_tonal`,
`tonal_to_broadband`, densification, sparsification and stable character.

Each descriptor exposes criterion-level observations, thresholds, operators,
weights, support/opposition scores, unavailable metrics, conflicts and
limitations such as clipping, low SNR, missing metrics, high repeat variability
or resolution-limited density. These labels are operational summaries, not
proof of linearity, nonlinearity, chaos, a physical regime transition or modal
identity.

## Spectral peaks and terminology

A spectral peak is a mathematical observation in a spectrum; it is not, by
itself, a physical modal classification. `bin_frequency_hz` is the FFT-grid
location, while `refined_frequency_hz` is an optional operational three-point
interpolation. Neither is a formal uncertainty estimate.

`Spectrum.bin_spacing_hz` is the FFT grid spacing, `sample_rate / n_fft`.
The legacy `frequency_resolution_hz` property remains an alias. Bin spacing is
not spectral resolution: the effective ability to separate nearby components
also depends on the window's main lobe and the analyzed time interval. Zero
padding creates a denser interpolated grid but does not create information or
improve physical separability.

## Requirements

- Python 3.11 or newer;
- runtime and development dependencies listed in `requirements.txt`.

## Development installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Structure

- `belllab.recording`: generalized recording model;
- `belllab.comparison`: generalized experiment model;
- `belllab.instruments`: instrument-family domain specializations;
- `belllab.io`, `temporal`, `spectrum` and `modal`: scientific modules;
- `belllab.plotting` and `report`: presentation interfaces;
- `belllab.config` and `utils`: shared configuration and utilities.

`BellRecording` and `BellComparison` remain available as temporary compatibility
aliases for `Recording` and `Experiment`, respectively.

## License

Distributed under the MIT License. See [LICENSE](LICENSE).
