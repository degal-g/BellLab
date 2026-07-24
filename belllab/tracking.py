"""Picos por quadro e trajetórias espectrais, sem interpretação modal.

Este módulo opera sobre uma STFT já calculada. A detecção por quadro reutiliza
o detector estacionário do BellLab; o tracking associa observações de forma
determinística, um-para-um, e produz trajetórias matemáticas de picos. Nenhuma
trajetória é classificada como modo físico nesta camada.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np
from scipy.optimize import linear_sum_assignment

from belllab.config import (
    AnalysisSettings,
    FramePeakDetectionSettings,
    SpectralTrackingSettings,
)
from belllab.results import SpectralTrackingResults, TimeFrequencyPeakResults
from belllab.spectrum import detect_spectral_peaks
from belllab.types import (
    FramePeaks,
    SpectralPeak,
    SpectralTrack,
    SpectralTrackCharacterization,
    TrackAmplitudeFit,
    TrackAssignmentDiagnostic,
    TrackFrequencyFit,
    Spectrum,
    TimeFrequencySpectrum,
)


def characterize_spectral_track(track: SpectralTrack) -> SpectralTrackCharacterization:
    """Descreve frequência e amplitude observadas sem recalcular o tracking.

    Frequências refinadas válidas são preferidas. Para amplitude linear positiva
    ajusta ``log(A)`` versus tempo; para dBFS ajusta nível versus tempo. O
    resultado é operacional, não amortecimento modal validado.
    """
    times = np.asarray(track.times_s, dtype=float)
    selected_frequencies: list[float] = []
    sources: list[str] = []
    for refined, raw in zip(
        track.refined_frequencies_hz, track.bin_frequencies_hz, strict=True
    ):
        if refined is not None and np.isfinite(refined) and refined > 0:
            selected_frequencies.append(float(refined))
            sources.append("interpolated")
        else:
            selected_frequencies.append(float(raw))
            sources.append("bin")
    frequencies = np.asarray(selected_frequencies, dtype=float)
    frequency_source = sources[0] if len(set(sources)) == 1 else "mixed"
    amplitudes = np.asarray(track.amplitudes, dtype=float)
    diagnostics: list[str] = ["not_a_physical_mode"]
    if frequency_source == "mixed":
        diagnostics.append("mixed_frequency_source")

    finite_frequency = np.isfinite(times) & np.isfinite(frequencies)
    finite_frequencies = frequencies[finite_frequency]
    finite_frequency_times = times[finite_frequency]
    frequency_discarded = int((~finite_frequency).sum())
    if frequency_discarded:
        diagnostics.append("nonfinite_frequency_values_discarded")
    frequency_fit, frequency_fit_diagnostics = _fit_frequency(
        finite_frequency_times,
        finite_frequencies,
        available_point_count=len(frequencies),
    )
    diagnostics.extend(
        item for item in frequency_fit_diagnostics if item not in diagnostics
    )
    if finite_frequencies.size:
        frequency_initial = float(finite_frequencies[0])
        frequency_final = float(finite_frequencies[-1])
        frequency_mean = float(np.mean(finite_frequencies))
        frequency_median = float(np.median(finite_frequencies))
        frequency_min = float(np.min(finite_frequencies))
        frequency_max = float(np.max(finite_frequencies))
        frequency_std = float(np.std(finite_frequencies))
        frequency_drift = frequency_final - frequency_initial
        frequency_peak_to_peak = frequency_max - frequency_min
        if frequency_median > 0:
            relative_stability = frequency_std / frequency_median
        else:
            relative_stability = None
            diagnostics.append("zero_frequency_median")
    else:
        frequency_initial = frequency_final = frequency_mean = frequency_median = None
        frequency_min = frequency_max = frequency_std = frequency_drift = None
        frequency_peak_to_peak = relative_stability = None

    finite_amplitude = np.isfinite(amplitudes) & np.isfinite(times)
    finite_amplitudes = amplitudes[finite_amplitude]
    finite_amplitude_times = times[finite_amplitude]
    amplitude_discarded = int((~finite_amplitude).sum())
    if amplitude_discarded:
        diagnostics.append("nonfinite_amplitude_values_discarded")
    if finite_amplitudes.size:
        amplitude_initial = float(finite_amplitudes[0])
        amplitude_final = float(finite_amplitudes[-1])
        amplitude_mean = float(np.mean(finite_amplitudes))
        amplitude_median = float(np.median(finite_amplitudes))
        amplitude_min = float(np.min(finite_amplitudes))
        amplitude_max = float(np.max(finite_amplitudes))
        amplitude_std = float(np.std(finite_amplitudes))
        amplitude_peak_to_peak = amplitude_max - amplitude_min
    else:
        amplitude_initial = amplitude_final = amplitude_mean = amplitude_median = None
        amplitude_min = amplitude_max = amplitude_std = amplitude_peak_to_peak = None
    if finite_amplitudes.size >= 2 and np.ptp(finite_amplitude_times) > 0:
        amplitude_coefficients = np.polyfit(
            finite_amplitude_times, finite_amplitudes, 1
        )
        amplitude_slope = float(amplitude_coefficients[0])
        differences = np.diff(finite_amplitudes)
        constant = np.isclose(differences, 0.0, rtol=0.0, atol=1e-12)
        difference_count = len(differences)
        increase_fraction = float(np.sum(differences > 1e-12) / difference_count)
        decrease_fraction = float(np.sum(differences < -1e-12) / difference_count)
        constant_fraction = float(np.sum(constant) / difference_count)
        if constant_fraction == 1.0:
            diagnostics.append("constant_amplitude")
    else:
        amplitude_slope = None
        increase_fraction = decrease_fraction = constant_fraction = None
        diagnostics.append("insufficient_amplitude_points")

    amplitude_fit, amplitude_fit_diagnostics = _fit_amplitude(
        times, amplitudes, track.amplitude_unit
    )
    diagnostics.extend(
        item for item in amplitude_fit_diagnostics if item not in diagnostics
    )
    frame_span = track.last_frame - track.first_frame + 1
    return SpectralTrackCharacterization(
        track_id=track.track_id,
        frequency_source=frequency_source,
        frequency_initial_hz=frequency_initial,
        frequency_final_hz=frequency_final,
        frequency_mean_hz=frequency_mean,
        frequency_median_hz=frequency_median,
        frequency_min_hz=frequency_min,
        frequency_max_hz=frequency_max,
        frequency_std_hz=frequency_std,
        frequency_total_drift_hz=frequency_drift,
        frequency_peak_to_peak_hz=frequency_peak_to_peak,
        relative_frequency_stability=relative_stability,
        frequency_available_point_count=len(frequencies),
        frequency_finite_point_count=int(finite_frequency.sum()),
        frequency_discarded_point_count=frequency_discarded,
        frequency_fit=frequency_fit,
        amplitude_initial=amplitude_initial,
        amplitude_final=amplitude_final,
        amplitude_mean=amplitude_mean,
        amplitude_median=amplitude_median,
        amplitude_min=amplitude_min,
        amplitude_max=amplitude_max,
        amplitude_std=amplitude_std,
        amplitude_peak_to_peak=amplitude_peak_to_peak,
        amplitude_slope_per_s=amplitude_slope,
        amplitude_increase_fraction=increase_fraction,
        amplitude_decrease_fraction=decrease_fraction,
        amplitude_constant_fraction=constant_fraction,
        amplitude_available_point_count=len(amplitudes),
        amplitude_finite_point_count=int(finite_amplitude.sum()),
        amplitude_discarded_point_count=amplitude_discarded,
        amplitude_unit=track.amplitude_unit,
        amplitude_fit=amplitude_fit,
        first_frame=track.first_frame,
        last_frame=track.last_frame,
        start_time_s=float(track.times_s[0]),
        end_time_s=float(track.times_s[-1]),
        observed_duration_s=track.duration_s,
        observation_count=track.observation_count,
        frame_span_count=frame_span,
        coverage_fraction=track.observation_count / frame_span,
        gap_count=track.gap_count,
        total_missing_frames=track.total_missing_frames,
        largest_gap_frames=track.largest_gap_frames,
        reached_analysis_final_frame=(
            track.last_frame == track.analysis_final_frame
            if track.analysis_final_frame is not None
            else None
        ),
        diagnostics=tuple(diagnostics),
    )


def _fit_frequency(
    times: np.ndarray,
    frequencies: np.ndarray,
    *,
    available_point_count: int,
) -> tuple[TrackFrequencyFit, tuple[str, ...]]:
    """Ajusta deriva linear somente sobre pares finitos já selecionados."""
    finite_count = len(frequencies)
    discarded = available_point_count - finite_count
    diagnostics: list[str] = []
    if discarded:
        diagnostics.append("nonfinite_frequency_values_discarded")
    if finite_count < 2:
        diagnostics.append("insufficient_frequency_points")
        return TrackFrequencyFit(
            False, None, None, None, None, None,
            available_point_count, finite_count, finite_count, discarded,
            None, None, "insufficient_frequency_points", tuple(diagnostics),
        ), tuple(diagnostics)
    if np.ptp(times) == 0:
        diagnostics.append("non_distinct_frequency_times")
        return TrackFrequencyFit(
            False, None, None, None, None, None,
            available_point_count, finite_count, finite_count, discarded,
            None, None, "non_distinct_frequency_times", tuple(diagnostics),
        ), tuple(diagnostics)
    coefficients = np.polyfit(times, frequencies, 1)
    fitted = np.polyval(coefficients, times)
    residual_sum = float(np.sum((frequencies - fitted) ** 2))
    total_sum = float(np.sum((frequencies - np.mean(frequencies)) ** 2))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else None
    slope = float(coefficients[0])
    if np.ptp(frequencies) <= 1e-12:
        diagnostics.append("constant_frequency")
    fit = TrackFrequencyFit(
        success=True,
        method="linear_frequency_drift",
        slope_hz_per_s=slope,
        intercept_hz=float(coefficients[1]),
        r_squared=r_squared,
        rmse_hz=float(np.sqrt(np.mean((frequencies - fitted) ** 2))),
        available_point_count=available_point_count,
        finite_point_count=finite_count,
        used_point_count=finite_count,
        discarded_point_count=discarded,
        start_time_s=float(times[0]),
        end_time_s=float(times[-1]),
        diagnostics=tuple(diagnostics),
    )
    return fit, tuple(diagnostics)


def _fit_amplitude(
    times: np.ndarray,
    amplitudes: np.ndarray,
    unit: str,
) -> tuple[TrackAmplitudeFit, tuple[str, ...]]:
    """Preserva o ajuste operacional de amplitude como contrato canônico."""
    finite_pairs = np.isfinite(amplitudes) & np.isfinite(times)
    valid = finite_pairs.copy()
    nonfinite_discarded = int((~finite_pairs).sum())
    nonpositive_discarded = 0
    diagnostics: list[str] = []
    if nonfinite_discarded:
        diagnostics.append("nonfinite_amplitude_values_discarded")
    if unit == "linear_amplitude":
        nonpositive_discarded = int((finite_pairs & (amplitudes <= 0)).sum())
        valid &= amplitudes > 0
        transformed = np.log(amplitudes[valid])
    else:
        transformed = amplitudes[valid]
    success = int(valid.sum()) >= 2 and np.ptp(times[valid]) > 0
    decay_tau = None
    if success:
        coefficients = np.polyfit(times[valid], transformed, 1)
        fitted = np.polyval(coefficients, times[valid])
        residual = float(np.sum((transformed - fitted) ** 2))
        total = float(np.sum((transformed - np.mean(transformed)) ** 2))
        decay_slope = float(coefficients[0])
        decay_r_squared = 1.0 - residual / total if total > 0 else None
        if unit == "linear_amplitude" and decay_slope < -1e-12:
            decay_tau = -1.0 / decay_slope
        elif unit == "dbfs_amplitude" and decay_slope < -1e-12:
            decay_tau = -20.0 / (decay_slope * np.log(10.0))
        elif abs(decay_slope) <= 1e-12:
            diagnostics.append("constant_amplitude")
        else:
            diagnostics.append("amplitude_increasing")
    else:
        coefficients = fitted = None
        decay_slope = decay_r_squared = None
        diagnostics.append("amplitude_decay_fit_unavailable")
    fit = TrackAmplitudeFit(
        success=success,
        decay_detected=decay_tau is not None,
        method=("log_linear_amplitude_decay" if unit == "linear_amplitude" else "linear_dbfs_decay") if success else None,
        amplitude_unit=unit,
        fit_domain=("natural_log_amplitude" if unit == "linear_amplitude" else "dbfs") if success else None,
        slope=decay_slope,
        intercept=float(coefficients[1]) if success else None,
        tau_s=decay_tau,
        r_squared=decay_r_squared,
        rmse=float(np.sqrt(np.mean((transformed - fitted) ** 2))) if success else None,
        rmse_unit=("ln(amplitude)" if unit == "linear_amplitude" else "dB") if success else None,
        available_point_count=len(amplitudes),
        finite_point_count=int(finite_pairs.sum()),
        used_point_count=int(valid.sum()),
        discarded_point_count=nonfinite_discarded + nonpositive_discarded,
        start_time_s=float(times[valid][0]) if success else None,
        end_time_s=float(times[valid][-1]) if success else None,
        failure_reason=None if success else "insufficient_points",
        diagnostics=tuple(diagnostics),
        nonfinite_discarded_point_count=nonfinite_discarded,
        nonpositive_discarded_point_count=nonpositive_discarded,
        slope_unit=("1/s" if unit == "linear_amplitude" else "dB/s") if success else None,
    )
    return fit, tuple(diagnostics)


def detect_stft_peaks(
    time_frequency: TimeFrequencySpectrum,
    settings: AnalysisSettings | FramePeakDetectionSettings | None = None,
) -> TimeFrequencyPeakResults:
    """Detecta picos independentemente em cada quadro de uma STFT existente.

    A função não recalcula a STFT. Cada coluna ``values[:, frame_index]`` é
    apresentada a :func:`belllab.spectrum.detect_spectral_peaks` como um
    ``Spectrum`` compatível, preservando proeminência, largura, interpolação e
    piso local. ``end_frame`` é exclusivo.
    """
    if settings is None:
        cfg = FramePeakDetectionSettings()
    elif isinstance(settings, AnalysisSettings):
        cfg = settings.frame_peaks
    elif isinstance(settings, FramePeakDetectionSettings):
        cfg = settings
    else:
        raise TypeError(
            "settings must be AnalysisSettings, FramePeakDetectionSettings, or None."
        )
    frame_count = len(time_frequency.times_s)
    _validate_time_frequency(time_frequency)
    start = cfg.start_frame or 0
    end = cfg.end_frame if cfg.end_frame is not None else frame_count
    if start >= frame_count:
        raise ValueError("start_frame must refer to an existing STFT frame.")
    if end > frame_count:
        raise ValueError("end_frame must not exceed the STFT frame count.")
    if end <= start:
        raise ValueError("selected frame interval must not be empty.")

    peak_cfg = cfg.peak_settings
    if cfg.max_peaks_per_frame is not None:
        peak_cfg = replace(peak_cfg, max_peaks=cfg.max_peaks_per_frame)

    frames: list[FramePeaks] = []
    diagnostics: list[str] = []
    for frame_index in range(start, end):
        values = tuple(row[frame_index] for row in time_frequency.values)
        finite = np.asarray([value for value in values if np.isfinite(value)])
        frame_maximum = float(np.max(finite)) if finite.size else None
        floor = float(np.median(finite)) if finite.size else None
        frame_diagnostics: list[str] = []
        silent = not finite.size or np.all(finite == 0.0)
        below_threshold = (
            cfg.min_frame_amplitude is not None
            and frame_maximum is not None
            and frame_maximum < cfg.min_frame_amplitude
        )
        if silent:
            frame_diagnostics.append("silent_frame")
        if below_threshold:
            frame_diagnostics.append("frame_below_amplitude_threshold")
        if (silent or below_threshold) and cfg.silence_policy == "skip":
            frames.append(
                FramePeaks(
                    frame_index=frame_index,
                    time_s=time_frequency.times_s[frame_index],
                    peaks=(),
                    candidate_count=0,
                    accepted_count=0,
                    frame_maximum=frame_maximum,
                    spectral_floor=floor,
                    diagnostics=(
                        tuple(frame_diagnostics)
                        if cfg.store_frame_diagnostics
                        else ()
                    ),
                    parameters=MappingProxyType({"peak_settings": peak_cfg}),
                )
            )
            continue
        spectrum = _frame_spectrum(time_frequency, frame_index, values)
        detected = detect_spectral_peaks(spectrum, peak_cfg)
        frame_diagnostics.extend(detected.diagnostics)
        frame_diagnostics.extend(detected.warnings)
        frames.append(
            FramePeaks(
                frame_index=frame_index,
                time_s=time_frequency.times_s[frame_index],
                peaks=detected.peaks,
                candidate_count=detected.candidate_count,
                accepted_count=detected.accepted_count,
                frame_maximum=frame_maximum,
                spectral_floor=floor,
                diagnostics=(
                    tuple(frame_diagnostics) if cfg.store_frame_diagnostics else ()
                ),
                parameters=MappingProxyType({"peak_settings": peak_cfg}),
            )
        )

    total_peaks = sum(frame.accepted_count for frame in frames)
    without_peaks = sum(frame.accepted_count == 0 for frame in frames)
    if without_peaks:
        diagnostics.append(f"frames_without_peaks={without_peaks}")
    return TimeFrequencyPeakResults(
        time_frequency=time_frequency,
        frames=tuple(frames),
        settings=cfg,
        processed_frame_count=len(frames),
        total_peak_count=total_peaks,
        frames_without_peaks=without_peaks,
        diagnostics=tuple(diagnostics),
    )


def track_spectral_peaks(
    frame_peaks: TimeFrequencyPeakResults,
    settings: AnalysisSettings | SpectralTrackingSettings | None = None,
) -> SpectralTrackingResults:
    """Associa picos por quadro em trajetórias espectrais determinísticas.

    Associações válidas são resolvidas pelo algoritmo Húngaro, com custo
    normalizado por tolerância frequencial e componente de amplitude opcional.
    Cada pico e cada trajetória podem participar no máximo de uma associação
    por quadro. Não há interpolação de lacunas nem interpretação física.
    """
    if settings is None:
        cfg = SpectralTrackingSettings()
    elif isinstance(settings, AnalysisSettings):
        cfg = settings.tracking
    elif isinstance(settings, SpectralTrackingSettings):
        cfg = settings
    else:
        raise TypeError(
            "settings must be AnalysisSettings, SpectralTrackingSettings, or None."
        )
    _validate_frame_results(frame_peaks)
    active: list[_ActiveTrack] = []
    completed: list[_ActiveTrack] = []
    next_id = 0
    ambiguous_count = 0
    near_threshold_count = 0
    margins: list[float] = []
    assignment_diagnostics: list[TrackAssignmentDiagnostic] = []

    for frame in frame_peaks.frames:
        still_active: list[_ActiveTrack] = []
        for track in active:
            if frame.frame_index - track.last_frame - 1 > cfg.max_gap_frames:
                completed.append(track)
            else:
                still_active.append(track)
        active = still_active
        assignments, ambiguity = _associate(active, frame, cfg)
        ambiguous_count += ambiguity[0]
        near_threshold_count += ambiguity[1]
        margins.extend(ambiguity[2])
        matched_peaks = {peak_index for _, peak_index, _ in assignments}
        for track_index, peak_index, cost in assignments:
            active[track_index].append(frame, peak_index, frame.peaks[peak_index], cost)
        assignment_diagnostics.extend(ambiguity[3])
        for peak_index, peak in enumerate(frame.peaks):
            if peak_index not in matched_peaks:
                active.append(_ActiveTrack.from_peak(next_id, frame, peak_index, peak))
                next_id += 1

    completed.extend(active)
    analysis_final_frame = frame_peaks.frames[-1].frame_index
    materialized = tuple(
        _materialize(track, cfg, analysis_final_frame) for track in completed
    )
    accepted = tuple(
        track
        for track in materialized
        if track.observation_count >= cfg.min_track_length
    )
    rejected = tuple(
        track
        for track in materialized
        if track.observation_count < cfg.min_track_length
    )
    diagnostics = [
        "association_method=hungarian",
        f"tracks_created={len(materialized)}",
    ]
    if rejected:
        diagnostics.append(f"tracks_rejected_by_min_length={len(rejected)}")
    return SpectralTrackingResults(
        frame_peaks=frame_peaks,
        tracks=accepted,
        rejected_tracks=rejected,
        settings=cfg,
        track_count=len(accepted),
        tracks_reaching_final_frame=sum(
            track.last_frame == frame_peaks.frames[-1].frame_index
            for track in active
        ),
        ambiguous_assignment_count=ambiguous_count,
        near_threshold_assignment_count=near_threshold_count,
        assignment_margin_min=min(margins) if margins else None,
        assignment_diagnostics=tuple(assignment_diagnostics),
        diagnostics=tuple(diagnostics),
    )


def _frame_spectrum(
    source: TimeFrequencySpectrum,
    frame_index: int,
    values: tuple[float, ...],
) -> Spectrum:
    """Cria visão leve e compatível de um quadro para o detector existente."""
    return Spectrum(
        frequencies_hz=source.frequencies_hz,
        magnitudes=values,
        magnitude_unit=source.magnitude_unit,
        window_name=source.window_name,
        fft_size=source.fft_size,
        overlap=None,
        timestamp=source.times_s[frame_index],
        sample_rate_hz=source.sample_rate_hz,
        original_size=source.window_length,
        bin_spacing_hz=source.bin_spacing_hz,
        channel_policy=source.channel_policy,
        channel_index=source.channel_index,
        normalization="coherent_gain_amplitude",
        interval_start_s=source.interval_start_s,
        interval_end_s=source.interval_end_s,
        remove_mean=source.parameters.get("detrend_method") == "frame_mean",
        parameters=MappingProxyType({"stft_frame_index": frame_index}),
    )


def _validate_time_frequency(time_frequency: TimeFrequencySpectrum) -> None:
    """Valida forma e orientação mínimas antes de detectar picos por quadro."""
    if not time_frequency.times_s:
        raise ValueError("time_frequency must contain at least one frame.")
    if len(time_frequency.values) != len(time_frequency.frequencies_hz):
        raise ValueError("STFT values must have one row per frequency bin.")
    frame_count = len(time_frequency.times_s)
    if any(len(row) != frame_count for row in time_frequency.values):
        raise ValueError("each STFT frequency row must have one value per frame.")


def _validate_frame_results(results: TimeFrequencyPeakResults) -> None:
    """Garante que o tracking recebe quadros ordenados e não duplicados."""
    frames = results.frames
    if results.processed_frame_count != len(frames):
        raise ValueError("processed_frame_count must match the number of frames.")
    indices = tuple(frame.frame_index for frame in frames)
    times = tuple(frame.time_s for frame in frames)
    if any(later <= earlier for earlier, later in zip(indices, indices[1:])):
        raise ValueError("frame indices must be strictly increasing and unique.")
    if any(later <= earlier for earlier, later in zip(times, times[1:])):
        raise ValueError("frame times must be strictly increasing.")


@dataclass
class _ActiveTrack:
    """Estado interno mutável, convertido em ``SpectralTrack`` ao final."""

    track_id: int
    observations: list[tuple[FramePeaks, int, SpectralPeak]]
    costs: list[float]

    @classmethod
    def from_peak(
        cls,
        track_id: int,
        frame: FramePeaks,
        peak_index: int,
        peak: SpectralPeak,
    ) -> _ActiveTrack:
        """Inicia uma trajetória no primeiro pico ainda não associado."""
        return cls(track_id, [(frame, peak_index, peak)], [])

    @property
    def last_frame(self) -> int:
        """Índice do último quadro realmente observado."""
        return self.observations[-1][0].frame_index

    def append(
        self,
        frame: FramePeaks,
        peak_index: int,
        peak: SpectralPeak,
        cost: float,
    ) -> None:
        """Acrescenta uma única observação associada de forma um-para-um."""
        self.observations.append((frame, peak_index, peak))
        self.costs.append(cost)


def _associate(
    active: list[_ActiveTrack],
    frame: FramePeaks,
    settings: SpectralTrackingSettings,
) -> tuple[list[tuple[int, int, float]], tuple[int, int, list[float], list[TrackAssignmentDiagnostic]]]:
    """Resolve associações válidas de um quadro por custo Húngaro."""
    if not active or not frame.peaks:
        return [], (0, 0, [], [])
    invalid_cost = 1e12
    costs = np.full((len(active), len(frame.peaks)), invalid_cost, dtype=np.float64)
    for track_index, track in enumerate(active):
        previous = track.observations[-1][2]
        for peak_index, peak in enumerate(frame.peaks):
            cost = _association_cost(previous, peak, settings)
            if cost is not None:
                costs[track_index, peak_index] = cost
    rows, columns = linear_sum_assignment(costs)
    assignments = [
        (int(row), int(column), float(costs[row, column]))
        for row, column in zip(rows, columns, strict=True)
        if costs[row, column] <= settings.maximum_association_cost
    ]
    ambiguous = 0
    near_threshold = 0
    margins: list[float] = []
    diagnostics: list[TrackAssignmentDiagnostic] = []
    for row, column, cost in assignments:
        row_margin, column_margin, operational = _compute_assignment_margins(
            costs, row, column, invalid_cost
        )
        available = [margin for margin in (row_margin, column_margin) if margin is not None]
        if available:
            margin = min(available)
            margins.append(margin)
            if margin <= settings.ambiguity_margin:
                ambiguous += 1
        if cost >= settings.near_threshold_ratio * settings.maximum_association_cost:
            near_threshold += 1
        previous = active[row].observations[-1][2]
        current = frame.peaks[column]
        distance = _frequency_distance(_association_frequency(previous, settings), _association_frequency(current, settings), settings.frequency_distance_unit)
        amplitude_distance = _amplitude_distance(previous, current) if settings.amplitude_weight else None
        frequency_cost = settings.frequency_weight * (distance / settings.frequency_tolerance)
        amplitude_cost = settings.amplitude_weight * amplitude_distance if amplitude_distance is not None else 0.0
        near = cost >= settings.near_threshold_ratio * settings.maximum_association_cost
        diagnostics.append(TrackAssignmentDiagnostic(frame.frame_index, active[row].track_id, column, cost, row_margin, column_margin, operational, operational is not None and operational <= settings.ambiguity_margin, near, distance, settings.frequency_distance_unit, amplitude_distance, frequency_cost, amplitude_cost))
    return assignments, (ambiguous, near_threshold, margins, diagnostics)


def _compute_assignment_margins(
    costs: np.ndarray,
    row: int,
    column: int,
    invalid_cost: float = np.inf,
) -> tuple[float | None, float | None, float | None]:
    """Calcula margens locais após uma associação global já escolhida.

    Células não finitas ou com custo maior ou igual a ``invalid_cost`` não são
    alternativas admissíveis. O auxiliar não executa assignment.
    """
    selected = costs[row, column]
    row_alternatives = costs[row].copy()
    row_alternatives[column] = np.inf
    row_valid = row_alternatives[np.isfinite(row_alternatives) & (row_alternatives < invalid_cost)]
    column_alternatives = costs[:, column].copy()
    column_alternatives[row] = np.inf
    column_valid = column_alternatives[np.isfinite(column_alternatives) & (column_alternatives < invalid_cost)]
    row_margin = float(np.min(row_valid) - selected) if row_valid.size else None
    column_margin = float(np.min(column_valid) - selected) if column_valid.size else None
    available = [value for value in (row_margin, column_margin) if value is not None]
    return row_margin, column_margin, min(available) if available else None


def _association_cost(
    previous: SpectralPeak,
    current: SpectralPeak,
    settings: SpectralTrackingSettings,
) -> float | None:
    """Calcula custo sem misturar unidades frequenciais incompatíveis."""
    previous_frequency = _association_frequency(previous, settings)
    current_frequency = _association_frequency(current, settings)
    distance = _frequency_distance(
        previous_frequency,
        current_frequency,
        settings.frequency_distance_unit,
    )
    if distance > settings.frequency_tolerance:
        return None
    cost = settings.frequency_weight * (distance / settings.frequency_tolerance)
    if settings.amplitude_weight:
        cost += settings.amplitude_weight * _amplitude_distance(previous, current)
    return cost


def _association_frequency(
    peak: SpectralPeak,
    settings: SpectralTrackingSettings,
) -> float:
    """Usa frequência refinada válida quando configurada; senão usa o bin."""
    refined = peak.refined_frequency_hz
    if settings.use_refined_frequency and refined is not None and refined > 0:
        return refined
    return peak.bin_frequency_hz


def _frequency_distance(left: float, right: float, unit: str) -> float:
    """Expressa distância em uma única unidade explícita de cada vez."""
    if unit == "hz":
        return abs(right - left)
    if unit == "relative":
        if left == 0 and right == 0:
            return 0.0
        if left <= 0 or right <= 0:
            return float("inf")
        return abs(right - left) / max(abs(left), abs(right))
    if left <= 0 or right <= 0:
        return float("inf")
    return abs(1200.0 * np.log2(right / left))


def _amplitude_distance(left: SpectralPeak, right: SpectralPeak) -> float:
    """Normaliza diferença de amplitude conforme a escala declarada."""
    if "dbfs" in left.amplitude_unit.lower():
        return abs(right.bin_amplitude - left.bin_amplitude) / 20.0
    reference = max(abs(left.bin_amplitude), abs(right.bin_amplitude), 1e-12)
    return abs(right.bin_amplitude - left.bin_amplitude) / reference


def _materialize(
    track: _ActiveTrack,
    settings: SpectralTrackingSettings,
    analysis_final_frame: int,
) -> SpectralTrack:
    """Converte estado interno em contrato imutável com métricas operacionais."""
    frames, peak_indices, peaks = zip(*track.observations, strict=True)
    frequencies = np.asarray(
        [_association_frequency(peak, settings) for peak in peaks], dtype=np.float64
    )
    amplitudes = np.asarray([peak.bin_amplitude for peak in peaks], dtype=np.float64)
    frame_indices = tuple(frame.frame_index for frame in frames)
    times = tuple(frame.time_s for frame in frames)
    gaps = tuple(
        current - previous - 1
        for previous, current in zip(frame_indices, frame_indices[1:])
    )
    duration = times[-1] - times[0]
    drift = float(frequencies[-1] - frequencies[0])
    return SpectralTrack(
        track_id=track.track_id,
        frame_indices=frame_indices,
        times_s=times,
        bin_frequencies_hz=tuple(peak.bin_frequency_hz for peak in peaks),
        refined_frequencies_hz=tuple(peak.refined_frequency_hz for peak in peaks),
        amplitudes=tuple(float(value) for value in amplitudes),
        amplitude_unit=_canonical_track_unit(peaks[0].amplitude_unit),
        prominences=tuple(peak.prominence for peak in peaks),
        widths_hz=tuple(peak.width_hz for peak in peaks),
        local_snr_db=tuple(peak.local_snr_db for peak in peaks),
        peak_references=tuple(
            (frame.frame_index, peak_index)
            for frame, peak_index in zip(frames, peak_indices, strict=True)
        ),
        first_frame=frame_indices[0],
        last_frame=frame_indices[-1],
        duration_s=duration,
        observation_count=len(peaks),
        gap_count=sum(gap > 0 for gap in gaps),
        total_missing_frames=sum(gaps),
        largest_gap_frames=max(gaps, default=0),
        mean_frequency_hz=float(np.mean(frequencies)),
        median_frequency_hz=float(np.median(frequencies)),
        frequency_std_hz=float(np.std(frequencies)),
        initial_frequency_hz=float(frequencies[0]),
        final_frequency_hz=float(frequencies[-1]),
        frequency_drift_hz=drift,
        mean_drift_hz_per_s=drift / duration if duration > 0 else None,
        max_amplitude=float(np.max(amplitudes)),
        initial_amplitude=float(amplitudes[0]),
        final_amplitude=float(amplitudes[-1]),
        median_amplitude=float(np.median(amplitudes)),
        mean_association_cost=float(np.mean(track.costs)) if track.costs else None,
        max_association_cost=float(np.max(track.costs)) if track.costs else None,
        diagnostics=("frequency=refined_or_bin", "not_a_physical_mode"),
        analysis_final_frame=analysis_final_frame,
    )


def _canonical_track_unit(unit: str) -> str:
    """Converte aliases conhecidos, sem inferir unidade por valores."""
    normalized = unit.strip().lower()
    if "dbfs" in normalized and "amplitude" in normalized:
        return "dbfs_amplitude"
    if normalized in {"normalized amplitude (peak)", "linear_amplitude"}:
        return "linear_amplitude"
    raise ValueError(f"unsupported spectral peak amplitude unit: {unit}")
