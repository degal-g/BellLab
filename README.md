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

Adjacent dynamic conditions can be associated at the individual-candidate
level under a separate conservative contract:

```python
from belllab import (
    CrossConditionCandidateAssociationSettings,
    associate_candidates_across_adjacent_conditions,
)

result = associate_candidates_across_adjacent_conditions(
    pp_recording_candidates,
    p_recording_candidates,
    CrossConditionCandidateAssociationSettings(
        maximum_absolute_frequency_difference_hz=2.0,
    ),
)
for match in result.matches:
    print(
        match.lower_candidate_ref.representative_frequency_hz,
        match.higher_candidate_ref.representative_frequency_hz,
        match.frequency_change_classification,
        match.association_diagnostic.total_cost,
    )
```

Only adjacent nominal pairs are accepted: `pp -> p`, `p -> mf`, `mf -> f` and
`f -> ff`. Unmatched lower-condition candidates are reported as disappearing
candidates, unmatched higher-condition candidates as emerging candidates, and
possible split/merge events remain diagnostics only. A match is an operational
candidate correspondence, not a preserved physical mode and not a conversion to
`ModalMode`.

Adjacent association results can then be chained deterministically across a
contiguous nominal sequence without recomputing any local association cost:

```python
from belllab import build_cross_condition_candidate_chains

chains = build_cross_condition_candidate_chains(
    (pp_to_p, p_to_mf, mf_to_f, f_to_ff),
)
for chain in chains.chains:
    print(
        chain.chain_id,
        [node.dynamic_label for node in chain.nodes],
        chain.frequency_trajectory_hz,
        chain.maximum_association_cost,
        chain.contains_ambiguous_match,
    )
```

Accepted adjacent matches are reused as directed edges. Partial chains,
singleton candidates, emerging starts, disappearing ends and possible
split/merge contexts are preserved for audit. No direct `pp -> ff` association
is created, no gap is closed by frequency proximity, and no chain is promoted to
`ModalMode`.

Candidate chains can be evaluated as operational modal hypotheses by explicit
criteria, still without promoting them to physical modes:

```python
from belllab import ModalHypothesisSettings, build_modal_hypotheses

hypotheses = build_modal_hypotheses(
    chains,
    ModalHypothesisSettings(
        require_complete_chain=True,
        maximum_step_absolute_frequency_change_hz=2.0,
        require_decay_evidence=False,
    ),
)
for hypothesis in hypotheses.hypotheses:
    print(
        hypothesis.hypothesis_id,
        hypothesis.source_chain_id,
        hypothesis.status,
        hypothesis.score.normalized_score,
        hypothesis.frequency_evidence.maximum_step_change_hz,
    )
```

A `ModalHypothesis` means only that one operational candidate chain satisfied
the active, configurable and auditable criteria for being treated as a
hypothesis of the same modal component across the requested conditions. Status
is explicit (`accepted`, `accepted_with_reservations`, `inconclusive`,
`rejected`, `insufficient_evidence` or `invalid_input`), reasons are separated
into support, reservations, rejection and missing evidence, and coverage,
frequency continuity, association quality, tracking quality, tau consistency,
pre-impact evidence and possible split/merge context remain separate evidence
objects. A high score cannot override mandatory gates. An accepted operational
modal hypothesis is not a proved physical mode, not a `ModalMode`, not proof of
physical identity, not proof of linearity or nonlinearity, and does not resolve
split or merge.

Operational parameter estimates can be derived from already-built modal
hypotheses without reopening audio, recomputing spectra, rebuilding tracks,
creating non-adjacent matches or changing the hypotheses:

```python
from belllab import (
    ModalParameterEstimationSettings,
    ParameterLocationMethod,
    ParameterUncertaintyMethod,
    estimate_modal_parameters,
)

parameter_result = estimate_modal_parameters(
    hypotheses,
    ModalParameterEstimationSettings(
        frequency_location_method=ParameterLocationMethod.MEDIAN,
        tau_location_method=ParameterLocationMethod.GEOMETRIC_MEAN,
        frequency_uncertainty_method=ParameterUncertaintyMethod.CONSERVATIVE,
        bootstrap_random_seed=0,
    ),
)
for estimate in parameter_result.estimates:
    print(
        estimate.status,
        estimate.frequency_estimate.representative_frequency_hz,
        estimate.frequency_trajectory.total_signed_change_hz,
        estimate.decay_estimate.representative_tau_s,
        estimate.decay_rate_estimate.amplitude_decay_rate_per_s,
        estimate.provenance.settings_fingerprint,
    )
```

`ModalParameterEstimate` is an auditable quantitative summary of values already
available in a `ModalHypothesis`. It may report representative frequency,
frequency trajectory, drift summaries, tau, decay rate, operational
uncertainty, coverage, reservations and provenance. These outputs preserve the
distinctions `modal hypothesis != proved physical mode`,
`representative frequency != exact modal frequency`, `estimated tau !=
invariant physical constant` and `condition variation != proof of
nonlinearity`. Missing values remain `None`; no Q factor, bandwidth,
oscillator fit, split/merge resolution, gap closure or `ModalMode` promotion is
performed by this layer.

Operational Q-factor and bandwidth estimates are a separate layer over
`ModalParameterEstimate`. They reuse representative frequency, representative
tau, operational uncertainties, provenance and optional already-calculated
spectral widths or spectra:

```python
from belllab import (
    ModalBandwidthSource,
    ModalQFactorEstimationSettings,
    estimate_modal_q_factors,
)

q_result = estimate_modal_q_factors(
    parameter_result,
    ModalQFactorEstimationSettings(
        bootstrap_random_seed=0,
        combine_consistent_methods="geometric_mean",
    ),
    bandwidth_sources={
        parameter_result.estimates[0].estimate_id: ModalBandwidthSource(
            spectrum_id="bell-pp-global-spectrum",
            center_frequency_hz=1000.0,
            frequency_axis_hz=(990.0, 995.0, 1000.0, 1005.0, 1010.0),
            magnitude_values=(0.2, 0.707945784, 1.0, 0.707945784, 0.2),
            peak_frequencies_hz=(1000.0,),
            frequency_resolution_hz=1.0,
        ),
    },
)
for estimate in q_result.estimates:
    print(
        estimate.status,
        estimate.decay_q_estimate.q_decay if estimate.decay_q_estimate else None,
        estimate.bandwidth_estimate.bandwidth_hz if estimate.bandwidth_estimate else None,
        estimate.bandwidth_q_estimate.q_bandwidth if estimate.bandwidth_q_estimate else None,
        estimate.representative_q,
    )
```

The decay convention is `A(t)=A0 exp(-t/tau)` and the operational weak-damping
summary is `Q_decay = pi * f * tau`. The bandwidth summary uses an explicitly
declared convention, by default full amplitude width at -3 dB, and
`Q_bandwidth = f_center / bandwidth`. Existing `SpectralPeak.width_hz` and
`GlobalSpectralPeakMetric.width_hz` can also be reused, preserving their
half-prominence definitions. Agreement between methods is not physical
validation; disagreement is not proof of error, coupling or nonlinearity. The
layer does not fit oscillators, split overlapping peaks, close gaps, create
new associations or promote estimates to `ModalMode`.

Operational evidence compatible with possible energy redistribution can be
evaluated between already computed envelope series. The layer accepts existing
`Envelope` or `SpectralTrack` amplitude time series, or explicit time/amplitude
tuples, and reports trends, delayed growth, recovery, anticorrelation, lag,
pair-energy proxy stability, alternating dominance and possible beating context:

```python
from belllab import (
    ModalEnergyExchangeSettings,
    ModalEnergyProxy,
    evaluate_modal_energy_exchange_pair,
    prepare_modal_envelope_series,
)

times_s = tuple(index * 0.05 for index in range(21))
component_a = tuple(1.0 - 0.6 * time for time in times_s)
component_b = tuple((1.0 - amplitude**2) ** 0.5 for amplitude in component_a)
settings = ModalEnergyExchangeSettings(
    normalize_envelopes=False,
    energy_proxy=ModalEnergyProxy.AMPLITUDE_SQUARED,
    significance_method="disabled",
)

source_a = prepare_modal_envelope_series(
    times_s=times_s,
    amplitudes=component_a,
    source_id="component-a",
    settings=settings,
)
source_b = prepare_modal_envelope_series(
    times_s=times_s,
    amplitudes=component_b,
    source_id="component-b",
    settings=settings,
)
evidence = evaluate_modal_energy_exchange_pair(source_a, source_b, settings)
print(
    evidence.status,
    evidence.trend_evidence.opposed_trends,
    evidence.correlation_evidence.zero_lag_correlation,
    evidence.pair_energy_evidence.pair_energy_relative_range,
)
```

For the example above the pair-energy proxy is exactly constant before
normalization, the envelope trends are opposed and the correlation is strongly
negative. This is still only operational evidence compatible with possible
redistribution between components. It is not proof of physical energy transfer,
causality, modal coupling, split, merge, nonlinearity or physical modal
identity. The implementation does not open WAV files, recompute FFT/STFT,
rerun tracking, close gaps, create non-adjacent associations, fit coupled
oscillators or promote any estimate to `ModalMode`.

Controlled synthetic validation can generate known synthetic scenarios, run
the public BellLab stages, and compare known construction parameters with the
recovered operational results:

```python
from belllab import (
    generate_synthetic_validation_scenario,
    validate_synthetic_scenario,
)

scenario = generate_synthetic_validation_scenario("single_ideal")
result = validate_synthetic_scenario(scenario)
print(
    result.status,
    result.frequency_validations[0].estimated_frequency_hz,
    result.decay_validations[0].estimated_tau_s,
    result.q_validations[0].representative_q,
    result.energy_exchange_validation.supported_pairs,
)
```

The reference scenario contains a 500 Hz component with `tau = 2.0 s`, so the
known compatible Q convention is `Q = pi * f * tau = 3141.59`. The validation
layer records ground truth before running the pipeline, preserves missing
estimates as `None`, reports pipeline errors per stage, and can run deterministic
campaigns or Monte Carlo trials with explicit seeds. Synthetic recovery is only
controlled operational validation. It is not proof of validity on real
recordings, not a calibration of thresholds from the same results, not a
correction of tracking with ground truth, and not evidence of physical modal
identity, causality, split, merge, linearity or nonlinearity.

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

## Real experiment pipeline

```python
from belllab import (
    ExperimentDefinition,
    ExperimentPipelineSettings,
    ExperimentRecordingDefinition,
    analyze_experiment,
    summarize_experiment_analysis,
)

experiment = ExperimentDefinition(
    name="Bell impact series",
    specimen_id="bell-001",
    dynamic_labels=("pp", "p", "mf", "f", "ff"),
    recordings=(
        ExperimentRecordingDefinition("data/pp.wav", "pp", recording_id="pp_take_1"),
        ExperimentRecordingDefinition("data/p.wav", "p", recording_id="p_take_1"),
        ExperimentRecordingDefinition("data/mf.wav", "mf", recording_id="mf_take_1"),
        ExperimentRecordingDefinition("data/f.wav", "f", recording_id="f_take_1"),
        ExperimentRecordingDefinition("data/ff.wav", "ff", recording_id="ff_take_1"),
    ),
)

result = analyze_experiment(experiment, ExperimentPipelineSettings())
summary = summarize_experiment_analysis(result)
```

`analyze_experiment` is an orchestrator for real WAV files and metadata. It
coordinates the existing loaders and scientific layers: temporal analysis,
global spectrum, STFT, tracking, pre-impact evidence, excitation
characterization, modal candidates, within-condition association, adjacent
cross-condition association, chains, modal hypotheses, modal parameters, Q and
bandwidth estimation, and operational evidence of possible energy
redistribution.

The pipeline preserves every stage result, skipped stage, blocked dependency,
structured failure, file fingerprint, settings fingerprint and selected
replicate. Multiple takes are analyzed separately; reference-take selection is
explicit and auditable, and raw waveforms are not averaged by default. Channel
selection is explicit and deterministic; there is no silent downmixing or
resampling.

A completed pipeline is not a physical validation by itself. It does not prove
modal identity, linearity, nonlinearity, causality, physical energy transfer, or
split/merge resolution. Missing conditions remain explicit gaps, and candidate
associations are limited to nominally adjacent dynamic labels.

## Reproducible results export

```python
from belllab import (
    ResultsExportSettings,
    analyze_experiment,
    export_experiment_results,
    validate_experiment_export,
)

analysis = analyze_experiment(experiment, ExperimentPipelineSettings())
export = export_experiment_results(
    analysis,
    ResultsExportSettings(output_directory="belllab-export"),
)
validation = validate_experiment_export(export)
```

The export layer serializes already computed BellLab results into deterministic
JSON, normalized CSV tables, LaTeX fragments, Markdown summaries and a
provenance manifest with SHA-256 checksums. It preserves status values,
uncertainties, missing values, rejected or inconclusive results, diagnostics,
configuration fingerprints, file fingerprints and BellLab version metadata.

Exporting does not rerun the scientific pipeline, does not reopen audio for
analysis, does not replace missing values with zero, and does not turn an
operational modal hypothesis, Q estimate or possible energy-redistribution
evidence into a physical conclusion. Overwrite behavior, missing-value display,
non-finite-value handling and atomic writes are explicit settings.

## Reproducible scientific visualizations

```python
from belllab import (
    ScientificFigureType,
    ScientificVisualizationSettings,
    create_experiment_visualizations,
)

figures = create_experiment_visualizations(
    analysis,
    ScientificVisualizationSettings(
        output_directory="belllab-figures",
        formats=("png", "svg"),
        figure_types=(
            ScientificFigureType.WAVEFORM,
            ScientificFigureType.GLOBAL_SPECTRUM,
            ScientificFigureType.MODAL_HYPOTHESES,
            ScientificFigureType.MODAL_PARAMETERS,
            ScientificFigureType.MODAL_Q_FACTORS,
        ),
    ),
)
```

The visualization layer renders already computed BellLab results with
Matplotlib's non-interactive backend. It can create deterministic figures for
waveforms, envelopes, decay estimates, spectra, peaks, spectrograms, tracks,
candidates, associations, chains, modal hypotheses, parameter trajectories, Q,
bandwidth, dynamic-condition comparison, operational evidence of possible
energy redistribution, synthetic validation and experiment summaries.

Figures preserve source IDs, statuses, uncertainties, gaps, missing values,
reservations, invalid results and provenance. They do not recalculate FFT,
STFT, tracking, candidates, associations, modal parameters, Q or energy
evidence. A visually convincing figure is not treated as additional scientific
evidence: modal hypotheses remain hypotheses, trajectories do not prove
nonlinearity, and visual anticorrelation is not confirmed physical energy
transfer.

## Reproducible scientific report

```python
from belllab import ScientificReportSettings, create_scientific_report

report = create_scientific_report(
    analysis,
    export,
    figures,
    ScientificReportSettings(
        output_directory="belllab-report",
        title="BellLab scientific report",
    ),
)
```

The report layer organizes already computed analysis results, exported tables
and generated figures into deterministic Markdown, LaTeX, a provenance
manifest, cross references, limitations and optional PDF compilation when a
local LaTeX tool is available. It validates compatible analysis IDs,
experiment IDs and checksums before rendering when configured to do so.

Report generation does not rerun the scientific pipeline, does not regenerate
figures, does not rebuild tables, and does not replace missing values with
zero. Its automatic text is factual and conservative: a compiled report is not
a scientific conclusion by itself, included figures are not additional
evidence, modal hypotheses are not proven physical modes, trajectories do not
prove nonlinearity, and operational energy-redistribution evidence is not
confirmed physical transfer.

## Command-line interface

BellLab also exposes a stable public CLI as a thin adapter over the same public
Python APIs:

```bash
belllab version
belllab analyze --config experiment.toml
belllab export --analysis result.json --json --csv --output-dir results
belllab visualize --analysis result.json --all --output-dir figures
belllab report --analysis result.json --markdown --latex --output-dir report
belllab validate-synthetic --all-scenarios
```

The same interface is available without an installed console script:

```bash
python3 -m belllab --help
python3 -m belllab analyze --recording pp=audio/pp.wav --dry-run
```

The CLI supports JSON and TOML configuration, quick `LABEL=PATH` recording
definitions, effective-configuration printing, dry runs, result bundles,
inspection, structured JSON output, quiet mode, explicit overwrite policies and
documented exit codes. Exit code `0` means completed, `1` completed with
reservations, `2` invalid usage/configuration, `3` invalid input, `4`
insufficient evidence, `5` partial execution, `6` stage failure, `7`
unexpected internal error, `8` artifact validation failure and `9` report
compilation failure.

Command completion is not implemented in this round. A command finishing with
exit code `0` is not a physical proof, and results with reservations,
insufficiencies or failures are not reduced silently to success.

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
