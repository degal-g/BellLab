"""Testes quantitativos para picos STFT e trajetórias espectrais."""

from __future__ import annotations

from types import MappingProxyType
from dataclasses import replace

import numpy as np
import pytest

from belllab import (
    AnalysisSettings,
    FramePeakDetectionSettings,
    FramePeaks,
    PeakDetectionSettings,
    SpectralTrackingSettings,
    STFTSettings,
    TimeFrequencyPeakResults,
    TimeFrequencySpectrum,
    analyze_stft,
    detect_stft_peaks,
    track_spectral_peaks,
    characterize_spectral_track,
)
from belllab.synthetic import (
    damped_exponential,
    ideal_impulse,
    linear_chirp,
    pure_sine,
    sine_sum,
)
from belllab.types import (
    SpectralPeak,
    SpectralTrackCharacterization,
    TrackAmplitudeFit,
    TrackAssignmentDiagnostic,
)
from belllab.tracking import _compute_assignment_margins, _frequency_distance


def _pipeline(signal, *, peak_prominence: float = 0.05, **tracking: object):
    """Executa STFT, picos por quadro e associação com parâmetros explícitos."""
    stft = analyze_stft(
        signal,
        settings=STFTSettings(
            window_length=128,
            hop_length=32,
            remove_mean=False,
        ),
    ).time_frequency
    frames = detect_stft_peaks(
        stft,
        FramePeakDetectionSettings(
            peak_settings=PeakDetectionSettings(min_prominence=peak_prominence),
        ),
    )
    return frames, track_spectral_peaks(
        frames,
        SpectralTrackingSettings(frequency_tolerance=0.30, **tracking),
    )


def test_stationary_and_two_sines_create_distinct_tracks() -> None:
    """Componentes estacionárias distintas permanecem em trajetórias separadas."""
    frames, result = _pipeline(
        sine_sum(
            (48.0, 96.0), duration_s=2.0, sample_rate=512,
            amplitudes=(0.8, 0.3),
        )
    )
    assert frames.total_peak_count == 2 * frames.processed_frame_count
    assert result.track_count == 2
    assert [track.mean_frequency_hz for track in result.tracks] == pytest.approx(
        [48.0, 96.0], abs=0.02
    )
    assert all(track.gap_count == 0 for track in result.tracks)
    assert all(track.observation_count == frames.processed_frame_count for track in result.tracks)


@pytest.mark.parametrize(
    ("start", "end", "direction"),
    [(30.0, 110.0, 1), (110.0, 30.0, -1)],
)
def test_linear_chirps_have_deterministic_frequency_trajectories(
    start: float,
    end: float,
    direction: int,
) -> None:
    """Chirps seguem a direção esperada com erro operacional limitado."""
    signal = linear_chirp(start, end, duration_s=2.0, sample_rate=512)
    frames, result = _pipeline(signal)
    assert result.track_count == 1
    track = result.tracks[0]
    frequencies = np.asarray(
        [value if value is not None else raw for value, raw in zip(
            track.refined_frequencies_hz, track.bin_frequencies_hz, strict=True
        )]
    )
    expected = start + ((end - start) * np.asarray(track.times_s) / 2.0)
    assert np.mean(np.abs(frequencies - expected)) < 3.0
    assert direction * track.frequency_drift_hz > 0
    repeated = track_spectral_peaks(frames, SpectralTrackingSettings(frequency_tolerance=0.30))
    assert repeated.tracks == result.tracks


def test_damped_signal_has_decreasing_amplitude_without_decay_fit() -> None:
    """A trajetória descreve tendência de amplitude sem estimar amortecimento."""
    _, result = _pipeline(damped_exponential(64.0, 3.0, duration_s=2.0, sample_rate=512))
    track = result.tracks[0]
    assert track.frequency_std_hz < 0.1
    assert track.initial_amplitude > track.final_amplitude
    assert "not_a_physical_mode" in track.diagnostics


def test_silence_and_impulse_do_not_create_accepted_tracks() -> None:
    """Silêncio é válido; picos breves de impulso falham o comprimento mínimo."""
    silence_frames, silence = _pipeline(pure_sine(32, duration_s=1, sample_rate=512, amplitude=0))
    assert silence_frames.total_peak_count == 0
    assert silence.track_count == 0

    _, impulse = _pipeline(
        ideal_impulse(duration_s=1, sample_rate=512, sample_index=100),
        peak_prominence=0.001,
        min_track_length=10,
    )
    assert impulse.track_count == 0
    assert impulse.rejected_tracks


def _manual_results(frame_frequencies: tuple[tuple[float, ...], ...]) -> TimeFrequencyPeakResults:
    """Cria observações controladas para testar lacunas e associações."""
    frame_count = len(frame_frequencies)
    frequencies = tuple(float(index) for index in range(128))
    tf = TimeFrequencySpectrum(
        times_s=tuple(float(index) for index in range(frame_count)),
        frequencies_hz=frequencies,
        values=tuple(tuple(0.0 for _ in range(frame_count)) for _ in frequencies),
        magnitude_unit="normalized amplitude (peak)",
        sample_rate_hz=256,
        window_length=64,
        fft_size=64,
        hop_length=1,
        bin_spacing_hz=1.0,
        frame_spacing_s=1.0,
        window_name="hann",
        coherent_gain=0.5,
        channel_policy="select",
        channel_index=0,
        interval_start_s=0.0,
        interval_end_s=float(frame_count),
        padding_policy="discard_incomplete",
    )
    frames = []
    for frame_index, values in enumerate(frame_frequencies):
        peaks = tuple(
            SpectralPeak(
                bin_index=int(round(frequency * 10)),
                bin_frequency_hz=frequency,
                refined_frequency_hz=frequency,
                bin_amplitude=1.0,
                amplitude_unit=tf.magnitude_unit,
            )
            for frequency in values
        )
        frames.append(
            FramePeaks(
                frame_index=frame_index,
                time_s=float(frame_index),
                peaks=peaks,
                candidate_count=len(peaks),
                accepted_count=len(peaks),
            )
        )
    return TimeFrequencyPeakResults(
        time_frequency=tf,
        frames=tuple(frames),
        settings=FramePeakDetectionSettings(),
        processed_frame_count=frame_count,
        total_peak_count=sum(len(values) for values in frame_frequencies),
        frames_without_peaks=sum(not values for values in frame_frequencies),
    )


def _manual_amplitude_results(unit: str) -> TimeFrequencyPeakResults:
    """Two-frame assignment case whose frequency and amplitude preferences conflict."""
    result = _manual_results(((100.0, 102.0), (101.0, 101.1)))
    first, second = result.frames
    first_peaks = tuple(
        replace(peak, bin_amplitude=amplitude, amplitude_unit=unit)
        for peak, amplitude in zip(first.peaks, (1.0, 10.0), strict=True)
    )
    second_peaks = tuple(
        replace(peak, bin_amplitude=amplitude, amplitude_unit=unit)
        for peak, amplitude in zip(second.peaks, (10.0, 1.0), strict=True)
    )
    frames = (
        replace(first, peaks=first_peaks),
        replace(second, peaks=second_peaks),
    )
    return replace(result, frames=frames)


def _single_peak_results(
    frequencies: tuple[float, ...],
    amplitudes: tuple[float, ...] | None = None,
    unit: str = "normalized amplitude (peak)",
) -> TimeFrequencyPeakResults:
    """Build a controlled real-matching trajectory with optional amplitudes."""
    result = _manual_results(tuple((frequency,) for frequency in frequencies))
    if amplitudes is None:
        return result
    frames = tuple(
        replace(
            frame,
            peaks=(
                replace(
                    frame.peaks[0],
                    bin_amplitude=amplitude,
                    amplitude_unit=unit,
                ),
            ),
        )
        for frame, amplitude in zip(result.frames, amplitudes, strict=True)
    )
    time_frequency = replace(result.time_frequency, magnitude_unit=unit)
    return replace(result, time_frequency=time_frequency, frames=frames)


def _assert_public_cost_diagnostic(
    diagnostic: TrackAssignmentDiagnostic,
    *,
    selected_cost: float,
    frequency_distance: float,
    frequency_component: float,
    amplitude_distance: float | None = None,
    amplitude_component: float = 0.0,
) -> None:
    """Check every public field that describes an accepted assignment."""
    assert diagnostic.frame_index == 1
    assert diagnostic.track_id == 0
    assert diagnostic.peak_index == 0
    assert diagnostic.selected_cost == pytest.approx(selected_cost)
    assert diagnostic.frequency_distance == pytest.approx(frequency_distance)
    assert diagnostic.frequency_distance_unit == "hz"
    assert diagnostic.frequency_cost_component == pytest.approx(frequency_component)
    if amplitude_distance is None:
        assert diagnostic.amplitude_distance is None
    else:
        assert diagnostic.amplitude_distance == pytest.approx(amplitude_distance)
    assert diagnostic.amplitude_cost_component == pytest.approx(amplitude_component)
    assert diagnostic.selected_cost == pytest.approx(
        diagnostic.frequency_cost_component + diagnostic.amplitude_cost_component
    )
    assert diagnostic.assignment_margin is None
    assert diagnostic.ambiguous is False


@pytest.mark.parametrize("unit", ["normalized amplitude (peak)", "dBFS amplitude (ref=1.0)"])
def test_amplitude_weight_changes_real_assignment(unit: str) -> None:
    """A dominant amplitude term changes Hungarian matching in both scales."""
    source = _manual_amplitude_results(unit)
    base = dict(frequency_tolerance=3.0, frequency_distance_unit="hz", min_track_length=1)
    frequency_only = track_spectral_peaks(source, SpectralTrackingSettings(**base, amplitude_weight=0.0))
    amplitude_dominant = track_spectral_peaks(source, SpectralTrackingSettings(**base, amplitude_weight=10.0, maximum_association_cost=20.0))
    pairs_zero = tuple((item.track_id, item.peak_index) for item in frequency_only.assignment_diagnostics)
    pairs_weighted = tuple((item.track_id, item.peak_index) for item in amplitude_dominant.assignment_diagnostics)
    assert pairs_zero != pairs_weighted
    assert all(item.amplitude_cost_component == 0.0 for item in frequency_only.assignment_diagnostics)
    assert all(item.selected_cost == pytest.approx(item.frequency_cost_component + item.amplitude_cost_component) for item in amplitude_dominant.assignment_diagnostics)


@pytest.mark.parametrize("candidates, expected_margin", [
    ((100.0, 101.0), 0.5),
    ((100.0, 100.18), 0.09),
    ((100.0, 100.06), 0.03),
])
def test_real_row_only_assignment_margins(candidates, expected_margin) -> None:
    """One active track exposes only a row margin through real matching."""
    result = track_spectral_peaks(
        _manual_results(((100.0,), candidates)),
        SpectralTrackingSettings(frequency_tolerance=2.0, frequency_distance_unit="hz", min_track_length=1, ambiguity_margin=0.1),
    )
    diagnostic = result.assignment_diagnostics[0]
    assert diagnostic.row_assignment_margin == pytest.approx(expected_margin)
    assert diagnostic.column_assignment_margin is None
    assert diagnostic.assignment_margin == diagnostic.row_assignment_margin
    assert diagnostic.selected_cost == pytest.approx(diagnostic.frequency_cost_component + diagnostic.amplitude_cost_component)
    assert diagnostic.ambiguous is (expected_margin <= 0.1)


@pytest.mark.parametrize("sources, expected_margin", [
    ((100.0, 101.0), 0.5),
    ((100.0, 100.18), 0.09),
    ((100.0, 100.06), 0.03),
])
def test_real_column_only_assignment_margins(sources, expected_margin) -> None:
    """Two tracks competing for one peak expose only a column margin."""
    result = track_spectral_peaks(
        _manual_results((sources, (100.0,))),
        SpectralTrackingSettings(frequency_tolerance=2.0, frequency_distance_unit="hz", min_track_length=1, ambiguity_margin=0.1),
    )
    diagnostic = result.assignment_diagnostics[0]
    assert diagnostic.row_assignment_margin is None
    assert diagnostic.column_assignment_margin == pytest.approx(expected_margin)
    assert diagnostic.assignment_margin == diagnostic.column_assignment_margin
    assert diagnostic.ambiguous is (expected_margin <= 0.1)


def test_real_bidirectional_margins_and_deterministic_tie() -> None:
    """A 2x2 tie is deterministic, ambiguous, and preserves one-to-one use."""
    source = _manual_results(((100.0, 101.0), (100.0, 101.0)))
    settings = SpectralTrackingSettings(frequency_tolerance=2.0, frequency_distance_unit="hz", min_track_length=1, ambiguity_margin=0.5)
    first = track_spectral_peaks(source, settings)
    second = track_spectral_peaks(source, settings)
    assert first.assignment_diagnostics == second.assignment_diagnostics
    assert len({item.peak_index for item in first.assignment_diagnostics}) == 2
    for item in first.assignment_diagnostics:
        assert item.row_assignment_margin is not None
        assert item.column_assignment_margin is not None
        assert item.assignment_margin == min(item.row_assignment_margin, item.column_assignment_margin)
        assert item.selected_cost == pytest.approx(item.frequency_cost_component + item.amplitude_cost_component)


def test_real_zero_row_margin_and_single_alternative() -> None:
    """Equidistant distinct peaks tie in one row; one candidate has no margin."""
    settings = SpectralTrackingSettings(frequency_tolerance=2.0, frequency_distance_unit="hz", min_track_length=1, ambiguity_margin=0.1)
    tied = track_spectral_peaks(_manual_results(((100.0,), (99.0, 101.0))), settings)
    repeated = track_spectral_peaks(_manual_results(((100.0,), (99.0, 101.0))), settings)
    diagnostic = tied.assignment_diagnostics[0]
    assert diagnostic.row_assignment_margin == 0.0
    assert diagnostic.column_assignment_margin is None
    assert diagnostic.assignment_margin == 0.0 and diagnostic.ambiguous
    assert tied.assignment_diagnostics == repeated.assignment_diagnostics
    only = track_spectral_peaks(_manual_results(((100.0,), (100.0,))), settings).assignment_diagnostics[0]
    assert only.row_assignment_margin is None and only.column_assignment_margin is None
    assert only.assignment_margin is None and not only.ambiguous
    assert only.amplitude_cost_component == 0.0


def test_real_zero_column_margin_and_bidirectional_threshold_cases() -> None:
    """Column ties and 2x2 margins follow the inclusive ambiguity threshold."""
    settings = SpectralTrackingSettings(frequency_tolerance=2.0, frequency_distance_unit="hz", min_track_length=1, ambiguity_margin=0.1)
    column = track_spectral_peaks(_manual_results(((99.0, 101.0), (100.0,))), settings).assignment_diagnostics[0]
    assert column.row_assignment_margin is None
    assert column.column_assignment_margin == 0.0
    assert column.assignment_margin == 0.0 and column.ambiguous
    for candidates, ambiguous in (((100.0, 101.18), False), ((100.0, 101.2), False), ((100.0, 101.4), False)):
        result = track_spectral_peaks(_manual_results(((100.0, 101.0), candidates)), settings)
        for diagnostic in result.assignment_diagnostics:
            assert diagnostic.row_assignment_margin is not None
            assert diagnostic.column_assignment_margin is not None
            assert diagnostic.assignment_margin == min(diagnostic.row_assignment_margin, diagnostic.column_assignment_margin)
            assert diagnostic.ambiguous is ambiguous
            assert diagnostic.selected_cost == pytest.approx(diagnostic.frequency_cost_component + diagnostic.amplitude_cost_component)


@pytest.mark.parametrize(
    "alternative, expected, ambiguous",
    [(0.05, 0.05, True), (0.1, 0.1, True), (0.2, 0.2, False)],
    ids=["margin_0_05", "margin_0_10", "margin_0_20"],
)
def test_controlled_assignment_margin_threshold(alternative, expected, ambiguous) -> None:
    """Margens locais têm valores exatos sem depender do assignment Húngaro."""
    matrix = np.array([[0.0, alternative], [1.0, 2.0]])
    row, column, operational = _compute_assignment_margins(matrix, 0, 0)
    assert row == pytest.approx(expected)
    assert column == pytest.approx(1.0)
    assert operational == pytest.approx(expected)
    assert (operational <= 0.1) is ambiguous


def test_controlled_row_column_none_and_inadmissible_costs() -> None:
    """Células infinitas ou inadmissíveis não contam como alternativas."""
    matrix = np.array([[0.0, np.inf], [np.inf, 5.0]])
    assert _compute_assignment_margins(matrix, 0, 0) == (None, None, None)

    row_only = np.array([[0.0, 0.2], [np.inf, np.inf]])
    row_margin, column_margin, operational = _compute_assignment_margins(row_only, 0, 0)
    assert row_margin == pytest.approx(0.2)
    assert column_margin is None
    assert operational == pytest.approx(0.2)

    column_only = np.array([[0.0, np.inf], [0.2, np.inf]])
    row_margin, column_margin, operational = _compute_assignment_margins(column_only, 0, 0)
    assert row_margin is None
    assert column_margin == pytest.approx(0.2)
    assert operational == pytest.approx(0.2)

    finite_but_inadmissible = np.array([[0.0, 100.0], [100.0, 5.0]])
    assert _compute_assignment_margins(finite_but_inadmissible, 0, 0, invalid_cost=10.0) == (None, None, None)


def test_real_matching_diagnostic_uses_controlled_margin_helper() -> None:
    """O diagnóstico público recebe as margens locais do auxiliar interno.

    O Húngaro escolhe globalmente; a margem é calculada depois para a célula
    escolhida. Por isso os valores exatos são cobertos com matriz controlada e
    aqui verificamos somente a conexão com o matching real.
    """
    result = track_spectral_peaks(
        _manual_results(((100.0,), (100.0, 101.0))),
        SpectralTrackingSettings(
            frequency_tolerance=2.0,
            frequency_distance_unit="hz",
            min_track_length=1,
            ambiguity_margin=0.1,
        ),
    )
    diagnostic = result.assignment_diagnostics[0]
    # Para 100 Hz -> (100, 101) Hz com tolerância de 2 Hz: custos 0 e 0,5.
    expected = _compute_assignment_margins(np.array([[0.0, 0.5]]), 0, 0)

    assert diagnostic.frame_index == 1
    assert diagnostic.track_id == 0
    assert diagnostic.peak_index == 0
    assert diagnostic.selected_cost == pytest.approx(0.0)
    assert diagnostic.row_assignment_margin == pytest.approx(expected[0])
    assert diagnostic.column_assignment_margin is expected[1]
    assert diagnostic.assignment_margin == pytest.approx(expected[2])
    assert diagnostic.ambiguous is False
    assert diagnostic.frequency_distance == pytest.approx(0.0)
    assert diagnostic.frequency_distance_unit == "hz"
    assert diagnostic.amplitude_distance is None
    assert diagnostic.frequency_cost_component == pytest.approx(0.0)
    assert diagnostic.amplitude_cost_component == pytest.approx(0.0)
    assert diagnostic.selected_cost == pytest.approx(
        diagnostic.frequency_cost_component + diagnostic.amplitude_cost_component
    )


def test_short_and_long_gaps_have_explicit_birth_and_end_behavior() -> None:
    """Lacunas só sobrevivem até o limite configurado, sem interpolação."""
    short_gap = track_spectral_peaks(
        _manual_results(((50.0,), (), (50.0,), (50.0,))),
        SpectralTrackingSettings(frequency_tolerance=2.0, frequency_distance_unit="hz", max_gap_frames=1),
    )
    assert short_gap.track_count == 1
    assert short_gap.tracks[0].gap_count == 1
    assert short_gap.tracks[0].largest_gap_frames == 1

    long_gap = track_spectral_peaks(
        _manual_results(((50.0,), (), (), (50.0,))),
        SpectralTrackingSettings(frequency_tolerance=2.0, frequency_distance_unit="hz", max_gap_frames=1, min_track_length=1),
    )
    assert long_gap.track_count == 2
    assert all(track.observation_count == 1 for track in long_gap.tracks)


def test_crossing_is_deterministic_but_not_modal_identity() -> None:
    """Cruzamentos permanecem um-para-um e expõem a limitação instantânea."""
    crossing = _manual_results(((40.0, 60.0), (45.0, 55.0), (50.0, 50.1), (55.0, 45.0)))
    settings = SpectralTrackingSettings(
        frequency_tolerance=12.0,
        frequency_distance_unit="hz",
        min_track_length=1,
    )
    first = track_spectral_peaks(crossing, settings)
    second = track_spectral_peaks(crossing, settings)
    assert first.tracks == second.tracks
    references = [reference for track in first.tracks for reference in track.peak_references]
    assert len(references) == len(set(references))
    assert all(len(track.frame_indices) == len(set(track.frame_indices)) for track in first.tracks)


def test_frame_peak_selection_and_invalid_contracts_fail_clearly() -> None:
    """Intervalos, configurações e invariantes inválidos são rejeitados cedo."""
    stft = analyze_stft(
        pure_sine(32, duration_s=1, sample_rate=256),
        settings=STFTSettings(window_length=64, hop_length=32),
    ).time_frequency
    with pytest.raises(ValueError, match="start_frame"):
        detect_stft_peaks(stft, FramePeakDetectionSettings(start_frame=100))
    with pytest.raises(ValueError):
        FramePeakDetectionSettings(silence_policy="discard")
    for kwargs in (
        {"frequency_tolerance": -1.0},
        {"frequency_distance_unit": "bins"},
        {"max_gap_frames": -1},
        {"min_track_length": 0},
        {"association_method": "greedy"},
    ):
        with pytest.raises(ValueError):
            SpectralTrackingSettings(**kwargs)
    peak = SpectralPeak(1, 1.0, 1.0, "normalized amplitude (peak)")
    with pytest.raises(ValueError, match="duplicated"):
        FramePeaks(0, 0.0, (peak, peak), 2, 2)


def test_analysis_settings_and_moderate_frame_count_remain_deterministic() -> None:
    """Configuração agregada e centenas de quadros preservam contratos simples."""
    manual = _manual_results(tuple((40.0, 80.0, 120.0) for _ in range(300)))
    result = track_spectral_peaks(
        manual,
        AnalysisSettings(
            tracking=SpectralTrackingSettings(
                frequency_tolerance=0.05,
                frequency_distance_unit="relative",
            )
        ),
    )
    assert result.track_count == 3
    assert all(track.observation_count == 300 for track in result.tracks)

    source = pure_sine(32, duration_s=1, sample_rate=256)
    stft = analyze_stft(source, STFTSettings(window_length=64, hop_length=32)).time_frequency
    settings = AnalysisSettings(
        frame_peaks=FramePeakDetectionSettings(
            peak_settings=PeakDetectionSettings(min_prominence=0.05)
        )
    )
    assert detect_stft_peaks(stft, settings).settings is settings.frame_peaks


def test_gap_semantics_and_final_frame_count_are_explicit() -> None:
    """Lacunas contam intervalos e quadros ausentes separadamente."""
    results = _manual_results(((50.0,), (50.0,), (), (), (), (50.0,), (50.0,), (), (), (50.0,), (90.0,), (), ()))
    tracked = track_spectral_peaks(
        results,
        SpectralTrackingSettings(
            frequency_tolerance=2.0, frequency_distance_unit="hz",
            max_gap_frames=3, min_track_length=1,
        ),
    )
    main = tracked.tracks[0]
    assert (main.gap_count, main.total_missing_frames, main.largest_gap_frames) == (2, 5, 3)
    assert tracked.tracks_reaching_final_frame == 0
    assert tracked.active_track_count == 0


def test_amplitude_units_and_track_characterization_are_operational() -> None:
    """Unidades incompatíveis falham; ajuste linear recupera tau rastreado."""
    tracked = track_spectral_peaks(
        _manual_results(tuple((60.0,) for _ in range(6))),
        SpectralTrackingSettings(frequency_tolerance=1.0, frequency_distance_unit="hz"),
    )
    track = tracked.tracks[0]
    characterization = characterize_spectral_track(track)
    assert characterization.frequency_slope_hz_per_s == pytest.approx(0.0, abs=1e-12)
    assert characterization.decay_tau_s is None

    bad_peak = SpectralPeak(2, 2.0, -10.0, "dBFS amplitude (ref=1.0)")
    bad_frame = FramePeaks(1, 1.0, (bad_peak,), 1, 1)
    with pytest.raises(ValueError, match="amplitude unit"):
        TimeFrequencyPeakResults(
            time_frequency=tracked.frame_peaks.time_frequency,
            frames=(tracked.frame_peaks.frames[0], bad_frame),
            settings=FramePeakDetectionSettings(),
            processed_frame_count=2,
            total_peak_count=2,
            frames_without_peaks=0,
        )


@pytest.mark.parametrize("tau", [0.2, 1.0, 3.0])
def test_operational_linear_decay_recovers_known_tau(tau: float) -> None:
    """Log-linear amplitude fitting recovers the declared synthetic tau."""
    base = track_spectral_peaks(
        _manual_results(tuple((60.0,) for _ in range(8))),
        SpectralTrackingSettings(frequency_tolerance=1.0, frequency_distance_unit="hz"),
    ).tracks[0]
    amplitudes = tuple(float(np.exp(-time / tau)) for time in base.times_s)
    characterized = characterize_spectral_track(
        replace(base, amplitudes=amplitudes, amplitude_unit="linear_amplitude")
    )
    assert characterized.decay_tau_s == pytest.approx(tau, rel=1e-10)
    assert characterized.decay_method == "log_linear_amplitude"


def test_dbfs_decay_uses_amplitude_formula_not_value_sign() -> None:
    """dBFS stays dBFS even at zero/positive values and recovers tau exactly."""
    base = track_spectral_peaks(
        _manual_results(tuple((60.0,) for _ in range(6))),
        SpectralTrackingSettings(frequency_tolerance=1.0, frequency_distance_unit="hz"),
    ).tracks[0]
    slope = -20.0 / np.log(10.0)
    levels = tuple(float(3.0 + slope * time) for time in base.times_s)
    characterized = characterize_spectral_track(
        replace(base, amplitudes=levels, amplitude_unit="dbfs_amplitude")
    )
    assert characterized.decay_method == "linear_dbfs_amplitude"
    assert characterized.decay_tau_s == pytest.approx(1.0, rel=1e-10)
    assert characterized.decay_slope == pytest.approx(slope, rel=1e-12)


@pytest.mark.parametrize(
    ("left", "right", "unit", "expected"),
    [
        (100.0, 101.0, "hz", 1.0),
        (100.0, 110.0, "relative", 10.0 / 110.0),
        (440.0, 880.0, "cents", 1200.0),
        (440.0, 440.0 * 2 ** (1 / 12), "cents", 100.0),
    ],
)
def test_frequency_distances_are_quantitative(left, right, unit, expected) -> None:
    """Hz, relative symmetric and cents conventions have known values."""
    assert _frequency_distance(left, right, unit) == pytest.approx(expected)
    assert _frequency_distance(right, left, unit) == pytest.approx(expected)


def test_relative_zero_and_cents_nonpositive_are_safe() -> None:
    """Zero never triggers division by zero and invalid cents are not admitted."""
    assert _frequency_distance(0.0, 0.0, "relative") == 0.0
    assert np.isinf(_frequency_distance(0.0, 10.0, "cents"))


def test_public_assignment_diagnostic_decomposes_cost() -> None:
    """Accepted associations expose finite components without a cost matrix."""
    _, result = _pipeline(pure_sine(64, duration_s=2, sample_rate=512))
    diagnostic = result.assignment_diagnostics[0]
    assert diagnostic.selected_cost == pytest.approx(
        diagnostic.frequency_cost_component + diagnostic.amplitude_cost_component
    )
    assert diagnostic.frequency_distance_unit == "relative"


@pytest.mark.parametrize("margin", [0.0, 0.2, None])
def test_assignment_diagnostic_public_margin_contract(margin) -> None:
    """Public diagnostics represent finite margins or unavailable alternatives."""
    diagnostic = TrackAssignmentDiagnostic(
        1, 2, 0, 0.4, margin, 0.3, min(value for value in (margin, 0.3) if value is not None),
        False, False, 0.04, "relative", None, 0.4, 0.0,
    )
    assert diagnostic.assignment_margin is not None
    with pytest.raises(ValueError):
        TrackAssignmentDiagnostic(0, 0, 0, float("nan"), None, None, None, False, False, 0.0, "hz", None, 0.0, 0.0)


def test_track_amplitude_fit_rejects_contradictory_counts_and_tau() -> None:
    """Structured fits reject contradictory states before characterization."""
    with pytest.raises(ValueError, match="discarded"):
        TrackAmplitudeFit(True, False, None, "linear_amplitude", None, None, None, None, None, None, None, 2, 1, 1, 0, None, None)
    with pytest.raises(ValueError, match="agree"):
        TrackAmplitudeFit(True, False, None, "linear_amplitude", None, None, None, 1.0, None, None, None, 2, 2, 2, 0, None, None)


@pytest.mark.parametrize("kwargs", [
    {"maximum_association_cost": 0.0},
    {"maximum_association_cost": float("inf")},
    {"near_threshold_ratio": -0.1},
    {"near_threshold_ratio": 1.1},
])
def test_assignment_cost_settings_reject_invalid_limits(kwargs) -> None:
    """Maximum cost and near-threshold ratio are explicit runtime contracts."""
    with pytest.raises(ValueError):
        SpectralTrackingSettings(**kwargs)


@pytest.mark.parametrize(
    ("label", "frequency", "maximum", "accepted", "expected_cost"),
    [
        ("below_maximum", 102.0, 0.5, True, 0.4),
        ("exactly_at_maximum", 102.5, 0.5, True, 0.5),
        ("above_maximum", 103.0, 0.5, False, 0.6),
        ("cost_above_one_accepted", 108.0, 2.0, True, 1.6),
        ("cost_below_one_rejected", 103.0, 0.5, False, 0.6),
    ],
)
def test_maximum_association_cost_real_matching_boundaries(
    label: str,
    frequency: float,
    maximum: float,
    accepted: bool,
    expected_cost: float,
) -> None:
    """Total-cost gate is inclusive and honors configured maxima other than one."""
    del label
    source = _single_peak_results((100.0, frequency, 100.0))
    settings = SpectralTrackingSettings(
        frequency_tolerance=10.0,
        frequency_distance_unit="hz",
        frequency_weight=2.0,
        maximum_association_cost=maximum,
        min_track_length=1,
    )
    result = track_spectral_peaks(source, settings)
    first_frame_diagnostics = tuple(
        item for item in result.assignment_diagnostics if item.frame_index == 1
    )
    if accepted:
        assert len(first_frame_diagnostics) == 1
        diagnostic = first_frame_diagnostics[0]
        _assert_public_cost_diagnostic(
            diagnostic,
            selected_cost=expected_cost,
            frequency_distance=frequency - 100.0,
            frequency_component=expected_cost,
        )
        assert diagnostic.near_threshold is (
            expected_cost >= settings.near_threshold_ratio * maximum
        )
        assert result.tracks[0].frame_indices == (0, 1, 2)
    else:
        assert first_frame_diagnostics == ()
        original = next(track for track in result.tracks if track.track_id == 0)
        rejected_peak_track = next(track for track in result.tracks if track.track_id == 1)
        assert original.frame_indices == (0, 2)
        assert original.gap_count == 1
        assert rejected_peak_track.peak_references == ((1, 0),)
        assert all(
            not (item.frame_index == 1 and item.track_id == 0)
            for item in result.assignment_diagnostics
        )


@pytest.mark.parametrize(
    ("unit", "amplitudes", "expected_amplitude_distance", "expected_amplitude_cost", "maximum"),
    [
        ("normalized amplitude (peak)", (1.0, 0.5), 0.5, 1.0, 2.0),
        ("dBFS amplitude (ref=1.0)", (-10.0, -4.0), 0.3, 0.6, 1.0),
    ],
    ids=["linear_amplitude", "dbfs_amplitude"],
)
def test_maximum_cost_uses_frequency_and_amplitude_components(
    unit: str,
    amplitudes: tuple[float, float],
    expected_amplitude_distance: float,
    expected_amplitude_cost: float,
    maximum: float,
) -> None:
    """Acceptance uses total weighted cost in linear-amplitude and dBFS scales."""
    result = track_spectral_peaks(
        _single_peak_results((100.0, 102.0), amplitudes, unit),
        SpectralTrackingSettings(
            frequency_tolerance=10.0,
            frequency_distance_unit="hz",
            frequency_weight=2.0,
            amplitude_weight=2.0,
            maximum_association_cost=maximum,
            min_track_length=1,
        ),
    )
    diagnostic = result.assignment_diagnostics[0]
    expected_total = 0.4 + expected_amplitude_cost
    _assert_public_cost_diagnostic(
        diagnostic,
        selected_cost=expected_total,
        frequency_distance=2.0,
        frequency_component=0.4,
        amplitude_distance=expected_amplitude_distance,
        amplitude_component=expected_amplitude_cost,
    )
    assert diagnostic.near_threshold is (
        expected_total >= 0.9 * maximum
    )


def test_total_amplitude_cost_can_reject_frequency_admissible_pair() -> None:
    """The frequency gate admits the pair, but the total-cost gate rejects it."""
    source = _single_peak_results(
        (100.0, 102.0), (1.0, 0.5), "normalized amplitude (peak)"
    )
    settings = SpectralTrackingSettings(
        frequency_tolerance=10.0,
        frequency_distance_unit="hz",
        frequency_weight=2.0,
        amplitude_weight=2.0,
        maximum_association_cost=1.2,
        min_track_length=1,
    )
    assert 2.0 <= settings.frequency_tolerance
    # Frequency component 0.4 + amplitude component 1.0 = total 1.4 > 1.2.
    result = track_spectral_peaks(source, settings)
    assert result.assignment_diagnostics == ()
    assert tuple(track.track_id for track in result.tracks) == (0, 1)
    assert all(track.observation_count == 1 for track in result.tracks)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (0.0, "finite and positive"),
        (-0.1, "finite and positive"),
        (float("nan"), "finite and positive"),
        (float("inf"), "finite and positive"),
        (-float("inf"), "finite and positive"),
    ],
    ids=["zero", "negative", "nan", "positive_inf", "negative_inf"],
)
def test_maximum_association_cost_requires_finite_positive_value(
    value: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SpectralTrackingSettings(maximum_association_cost=value)


@pytest.mark.parametrize(
    ("frequency", "expected_cost", "near"),
    [
        (104.0, 0.8, False),
        (105.0, 1.0, True),
        (106.0, 1.2, True),
    ],
    ids=["below_near_threshold", "exactly_at_near_threshold", "above_near_threshold"],
)
def test_near_threshold_ratio_real_matching_boundaries(
    frequency: float, expected_cost: float, near: bool
) -> None:
    """Near-threshold comparison is inclusive and based on configured total cost."""
    settings = SpectralTrackingSettings(
        frequency_tolerance=10.0,
        frequency_distance_unit="hz",
        frequency_weight=2.0,
        maximum_association_cost=2.0,
        near_threshold_ratio=0.5,
        min_track_length=1,
    )
    result = track_spectral_peaks(_single_peak_results((100.0, frequency)), settings)
    diagnostic = result.assignment_diagnostics[0]
    _assert_public_cost_diagnostic(
        diagnostic,
        selected_cost=expected_cost,
        frequency_distance=frequency - 100.0,
        frequency_component=expected_cost,
    )
    assert settings.near_threshold_ratio * settings.maximum_association_cost == pytest.approx(1.0)
    assert diagnostic.near_threshold is near


@pytest.mark.parametrize(
    ("ratio", "frequency", "expected_cost", "near"),
    [
        (0.0, 100.0, 0.0, True),
        (1.0, 109.0, 1.8, False),
        (1.0, 110.0, 2.0, True),
    ],
    ids=["ratio_zero", "ratio_one_below_maximum", "ratio_one_at_maximum"],
)
def test_near_threshold_ratio_extremes(
    ratio: float, frequency: float, expected_cost: float, near: bool
) -> None:
    settings = SpectralTrackingSettings(
        frequency_tolerance=10.0,
        frequency_distance_unit="hz",
        frequency_weight=2.0,
        maximum_association_cost=2.0,
        near_threshold_ratio=ratio,
        min_track_length=1,
    )
    diagnostic = track_spectral_peaks(
        _single_peak_results((100.0, frequency)), settings
    ).assignment_diagnostics[0]
    assert diagnostic.selected_cost == pytest.approx(expected_cost)
    assert diagnostic.near_threshold is near


@pytest.mark.parametrize(
    "value",
    [-0.1, 1.1, float("nan"), float("inf"), -float("inf")],
    ids=["negative", "above_one", "nan", "positive_inf", "negative_inf"],
)
def test_near_threshold_ratio_requires_finite_unit_interval(value: float) -> None:
    with pytest.raises(ValueError, match=r"finite and in \[0, 1\]"):
        SpectralTrackingSettings(near_threshold_ratio=value)


def test_ambiguous_assignment_can_be_far_from_cost_threshold() -> None:
    """A small alternative margin does not imply a large absolute cost."""
    result = track_spectral_peaks(
        _manual_results(((100.0,), (101.0, 101.5))),
        SpectralTrackingSettings(
            frequency_tolerance=10.0,
            frequency_distance_unit="hz",
            maximum_association_cost=2.0,
            near_threshold_ratio=0.9,
            ambiguity_margin=0.1,
            min_track_length=1,
        ),
    )
    diagnostic = result.assignment_diagnostics[0]
    assert diagnostic.selected_cost == pytest.approx(0.1)
    assert diagnostic.assignment_margin == pytest.approx(0.05)
    assert diagnostic.ambiguous is True
    assert diagnostic.near_threshold is False
    repeated = track_spectral_peaks(result.frame_peaks, result.settings)
    assert repeated.tracks == result.tracks
    assert repeated.assignment_diagnostics == result.assignment_diagnostics


def test_near_threshold_assignment_can_be_unambiguous() -> None:
    """A large absolute cost does not require any competing alternative."""
    result = track_spectral_peaks(
        _single_peak_results((100.0, 109.0)),
        SpectralTrackingSettings(
            frequency_tolerance=10.0,
            frequency_distance_unit="hz",
            frequency_weight=2.0,
            maximum_association_cost=2.0,
            near_threshold_ratio=0.9,
            ambiguity_margin=0.1,
            min_track_length=1,
        ),
    )
    diagnostic = result.assignment_diagnostics[0]
    assert diagnostic.selected_cost == pytest.approx(1.8)
    assert diagnostic.assignment_margin is None
    assert diagnostic.ambiguous is False
    assert diagnostic.near_threshold is True
    repeated = track_spectral_peaks(result.frame_peaks, result.settings)
    assert repeated.tracks == result.tracks
    assert repeated.assignment_diagnostics == result.assignment_diagnostics


@pytest.mark.parametrize(
    ("frequencies", "settings"),
    [
        (
            (100.0, 102.5),
            SpectralTrackingSettings(
                frequency_tolerance=10.0, frequency_distance_unit="hz",
                frequency_weight=2.0, maximum_association_cost=0.5, min_track_length=1,
            ),
        ),
        (
            (100.0, 105.0),
            SpectralTrackingSettings(
                frequency_tolerance=10.0, frequency_distance_unit="hz",
                frequency_weight=2.0, maximum_association_cost=2.0,
                near_threshold_ratio=0.5, min_track_length=1,
            ),
        ),
        (
            (100.0, 106.0),
            SpectralTrackingSettings(
                frequency_tolerance=10.0, frequency_distance_unit="hz",
                frequency_weight=2.0, maximum_association_cost=2.0,
                near_threshold_ratio=0.5, min_track_length=1,
            ),
        ),
    ],
    ids=["exact_maximum", "exact_near_threshold", "above_near_threshold"],
)
def test_association_cost_threshold_scenarios_are_reproducible(
    frequencies: tuple[float, ...], settings: SpectralTrackingSettings
) -> None:
    source = _single_peak_results(frequencies)
    first = track_spectral_peaks(source, settings)
    second = track_spectral_peaks(source, settings)
    assert first.tracks == second.tracks
    assert first.rejected_tracks == second.rejected_tracks
    assert first.assignment_diagnostics == second.assignment_diagnostics


@pytest.mark.parametrize("unit, values", [
    ("linear_amplitude", (1.0, 1.0, 1.0, 1.0)),
    ("dbfs_amplitude", (-6.0, -6.0, -6.0, -6.0)),
])
def test_constant_amplitude_has_fit_but_no_operational_tau(unit, values) -> None:
    """A constant series is fit successfully but must not manufacture decay."""
    base = track_spectral_peaks(
        _manual_results(tuple((60.0,) for _ in values)),
        SpectralTrackingSettings(frequency_tolerance=1.0, frequency_distance_unit="hz"),
    ).tracks[0]
    result = characterize_spectral_track(replace(base, amplitudes=values, amplitude_unit=unit))
    fit = result.amplitude_fit
    assert fit.success and not fit.decay_detected and fit.tau_s is None
    assert fit.r_squared is None
    assert result.decay_tau_s is fit.tau_s


def test_increasing_and_nonfinite_amplitudes_are_structured() -> None:
    """Growing data has no tau; nonfinite samples are discarded before fitting."""
    base = track_spectral_peaks(
        _manual_results(tuple((60.0,) for _ in range(5))),
        SpectralTrackingSettings(frequency_tolerance=1.0, frequency_distance_unit="hz"),
    ).tracks[0]
    growing = characterize_spectral_track(
        replace(base, amplitudes=(0.1, 0.2, 0.4, 0.8, 1.0), amplitude_unit="linear_amplitude")
    )
    assert growing.amplitude_fit.success and growing.amplitude_fit.tau_s is None
    filtered = characterize_spectral_track(
        replace(base, amplitudes=(1.0, np.nan, np.inf, 0.5, 0.25), amplitude_unit="linear_amplitude")
    )
    assert filtered.amplitude_fit.finite_point_count == 3
    assert filtered.amplitude_fit.discarded_point_count == 2
    assert np.isfinite(filtered.amplitude_fit.rmse)


def test_linear_nonpositive_and_nonfinite_counts_are_auditable() -> None:
    """Zero/negative values never enter log fitting and retain separate counts."""
    base = track_spectral_peaks(
        _manual_results(tuple((60.0,) for _ in range(6))),
        SpectralTrackingSettings(frequency_tolerance=1.0, frequency_distance_unit="hz"),
    ).tracks[0]
    result = characterize_spectral_track(
        replace(base, amplitudes=(1.0, 0.0, -1.0, np.nan, 0.5, 0.25), amplitude_unit="linear_amplitude")
    )
    fit = result.amplitude_fit
    assert (fit.available_point_count, fit.finite_point_count, fit.used_point_count) == (6, 5, 3)
    assert (fit.nonfinite_discarded_point_count, fit.nonpositive_discarded_point_count) == (1, 2)
    assert fit.discarded_point_count == 3
    assert np.isfinite(result.amplitude_mean) and np.isfinite(result.amplitude_min)


def test_track_amplitude_fit_requires_compatible_success_contract() -> None:
    """Successful linear fits require their declared method, domain and units."""
    kwargs = dict(success=True, decay_detected=True, method="log_linear_amplitude_decay", amplitude_unit="linear_amplitude", fit_domain="natural_log_amplitude", slope=-1.0, intercept=0.0, tau_s=1.0, r_squared=1.0, rmse=0.0, rmse_unit="ln(amplitude)", available_point_count=2, finite_point_count=2, used_point_count=2, discarded_point_count=0, start_time_s=0.0, end_time_s=1.0, slope_unit="1/s")
    assert TrackAmplitudeFit(**kwargs).tau_s == 1.0
    with pytest.raises(ValueError, match="incompatible"):
        TrackAmplitudeFit(**{**kwargs, "slope_unit": "dB/s"})
    with pytest.raises(ValueError, match="negative slope"):
        TrackAmplitudeFit(**{**kwargs, "slope": 1.0})


def test_track_amplitude_fit_failure_policy_and_legacy_aliases() -> None:
    """Failed fits retain no fake regression and legacy aliases read canonical fit."""
    failed = TrackAmplitudeFit(False, False, None, "linear_amplitude", None, None, None, None, None, None, None, 1, 1, 1, 0, None, None, "insufficient_points")
    assert failed.failure_reason == "insufficient_points"


@pytest.mark.parametrize("tau", [float("nan"), float("inf"), -float("inf"), 0.0, -1.0], ids=["nan", "positive_inf", "negative_inf", "zero", "negative"])
def test_track_amplitude_fit_rejects_invalid_tau(tau) -> None:
    """Tau is always finite and strictly positive when supplied."""
    with pytest.raises(ValueError, match="tau_s"):
        TrackAmplitudeFit(True, True, "log_linear_amplitude_decay", "linear_amplitude", "natural_log_amplitude", -1.0, 0.0, tau, 1.0, 0.0, "ln(amplitude)", 2, 2, 2, 0, 0.0, 1.0, slope_unit="1/s")


@pytest.mark.parametrize("field,value", [("slope", float("nan")), ("intercept", float("inf")), ("r_squared", float("nan")), ("rmse", float("inf")), ("start_time_s", float("nan")), ("end_time_s", float("inf"))])
def test_track_amplitude_fit_rejects_nonfinite_numeric_fields(field, value) -> None:
    """No numerical regression field accepts NaN or infinity sentinels."""
    kwargs = dict(success=True, decay_detected=True, method="log_linear_amplitude_decay", amplitude_unit="linear_amplitude", fit_domain="natural_log_amplitude", slope=-1.0, intercept=0.0, tau_s=1.0, r_squared=1.0, rmse=0.0, rmse_unit="ln(amplitude)", available_point_count=2, finite_point_count=2, used_point_count=2, discarded_point_count=0, start_time_s=0.0, end_time_s=1.0, slope_unit="1/s")
    kwargs[field] = value
    with pytest.raises(ValueError):
        TrackAmplitudeFit(**kwargs)


@pytest.mark.parametrize("diagnostics", [("",), (" ",), ("repeat", "repeat")])
def test_track_amplitude_fit_rejects_invalid_diagnostics(diagnostics) -> None:
    """Diagnostic collections are immutable, textual, nonempty and unique."""
    with pytest.raises(ValueError, match="diagnostics"):
        TrackAmplitudeFit(False, False, None, "linear_amplitude", None, None, None, None, None, None, None, 0, 0, 0, 0, None, None, "insufficient_points", diagnostics)


@pytest.mark.parametrize("value", [0.95, 0.0], ids=["positive", "zero"])
def test_failed_fit_rejects_r_squared(value) -> None:
    """A failed regression cannot carry an R-squared result."""
    with pytest.raises(ValueError, match="failed TrackAmplitudeFit"):
        TrackAmplitudeFit(False, False, None, "linear_amplitude", None, None, None, None, value, None, None, 0, 0, 0, 0, None, None, "insufficient_points")


def test_legacy_aliases_read_only_from_canonical_fit() -> None:
    """All legacy decay aliases are direct projections of amplitude_fit."""
    base = track_spectral_peaks(_manual_results(((60.0,), (60.0,), (60.0,))), SpectralTrackingSettings(frequency_tolerance=1.0, frequency_distance_unit="hz")).tracks[0]
    result = characterize_spectral_track(replace(base, amplitudes=(1.0, np.exp(-1), np.exp(-2))))
    fit = result.amplitude_fit
    assert result.decay_method == "log_linear_amplitude"
    assert result.decay_tau_s == fit.tau_s
    assert result.decay_slope == fit.slope
    assert result.decay_r_squared == fit.r_squared
    assert result.decay_points_used == fit.used_point_count
    assert result.decay_points_discarded == fit.discarded_point_count


def test_legacy_decay_aliases_for_failed_amplitude_fit() -> None:
    """Legacy aliases expose only the canonical failed amplitude-fit fields."""
    track = track_spectral_peaks(
        _manual_results(((60.0,),)),
        SpectralTrackingSettings(
            frequency_tolerance=1.0,
            frequency_distance_unit="hz",
            min_track_length=1,
        ),
    ).tracks[0]
    characterization = characterize_spectral_track(track)

    assert characterization.decay_method is None
    assert characterization.decay_tau_s is None
    assert characterization.decay_slope is None
    assert characterization.decay_r_squared is None
    assert characterization.decay_points_used == characterization.amplitude_fit.used_point_count
    assert characterization.decay_points_discarded == characterization.amplitude_fit.discarded_point_count
