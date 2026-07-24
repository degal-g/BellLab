"""Caracterização descritiva do espectro global de uma gravação.

As métricas deste módulo descrevem uma distribuição espectral. Elas não são
diagnóstico de ruído, caos, não linearidade, identidade modal ou regime físico.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite, log2
from typing import Literal

import numpy as np
from scipy.signal import peak_widths

from belllab.config import PeakDetectionSettings, SpectrumAnalysisSettings
from belllab.recording import Recording
from belllab.spectrum import analyze_spectrum, detect_spectral_peaks
from belllab.types import Signal, Spectrum


def _text(value: str, name: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be empty.")


def _diagnostics(values: tuple[str, ...]) -> None:
    if any(not value.strip() for value in values) or len(values) != len(set(values)):
        raise ValueError("diagnostics must contain unique non-empty strings.")


@dataclass(frozen=True, slots=True)
class SpectralBand:
    """Faixa configurável, com início inclusivo e fim exclusivo."""

    label: str
    frequency_start_hz: float
    frequency_end_hz: float

    def __post_init__(self) -> None:
        _text(self.label, "label")
        if not all(isfinite(x) for x in (self.frequency_start_hz, self.frequency_end_hz)):
            raise ValueError("band limits must be finite.")
        if self.frequency_start_hz < 0 or self.frequency_end_hz <= self.frequency_start_hz:
            raise ValueError("band limits must be non-negative and ordered.")


@dataclass(frozen=True, slots=True)
class GlobalSpectralCharacterizationSettings:
    """Parâmetros explícitos da caracterização espectral global."""

    start_time_s: float | None = None
    end_time_s: float | None = None
    frequency_min_hz: float = 0.0
    frequency_max_hz: float | None = None
    detrend_policy: Literal["none", "mean"] = "mean"
    window_name: Literal["rectangular", "hann"] = "hann"
    fft_size: int | None = None
    spectral_input_domain: Literal[
        "auto", "linear_amplitude", "dbfs_amplitude", "linear_power"
    ] = "auto"
    power_reference: float = 1.0
    minimum_bin_count: int = 3
    minimum_positive_bins_for_flatness: int = 2
    peak_min_power: float | None = None
    peak_min_prominence: float | None = None
    peak_distance_bins: int | None = None
    peak_min_width_bins: float | None = None
    peak_max_width_bins: float | None = None
    tonal_neighborhood_width_factor: float = 1.0
    rolloff_fractions: tuple[float, ...] = (0.05, 0.50, 0.85, 0.90, 0.95)
    occupied_lower_fraction: float = 0.05
    occupied_upper_fraction: float = 0.95
    bands: tuple[SpectralBand, ...] = ()
    numerical_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        numeric = (
            self.frequency_min_hz,
            self.power_reference,
            self.tonal_neighborhood_width_factor,
            self.occupied_lower_fraction,
            self.occupied_upper_fraction,
            self.numerical_tolerance,
        )
        if not all(isfinite(value) for value in numeric):
            raise ValueError("settings numeric values must be finite.")
        if self.frequency_min_hz < 0:
            raise ValueError("frequency_min_hz must not be negative.")
        if self.frequency_max_hz is not None and (
            not isfinite(self.frequency_max_hz)
            or self.frequency_max_hz <= self.frequency_min_hz
        ):
            raise ValueError("frequency_max_hz must be finite and above minimum.")
        if self.start_time_s is not None and (
            not isfinite(self.start_time_s) or self.start_time_s < 0
        ):
            raise ValueError("start_time_s must be finite and non-negative.")
        if self.end_time_s is not None and (
            not isfinite(self.end_time_s) or self.end_time_s < 0
        ):
            raise ValueError("end_time_s must be finite and non-negative.")
        if (
            self.start_time_s is not None
            and self.end_time_s is not None
            and self.end_time_s <= self.start_time_s
        ):
            raise ValueError("end_time_s must be above start_time_s.")
        if self.detrend_policy not in {"none", "mean"}:
            raise ValueError("detrend_policy must be 'none' or 'mean'.")
        if self.window_name not in {"rectangular", "hann"}:
            raise ValueError("unsupported window_name.")
        if self.spectral_input_domain not in {
            "auto", "linear_amplitude", "dbfs_amplitude", "linear_power"
        }:
            raise ValueError("unsupported spectral_input_domain.")
        if self.fft_size is not None and self.fft_size <= 0:
            raise ValueError("fft_size must be positive.")
        if self.minimum_bin_count <= 0 or self.minimum_positive_bins_for_flatness <= 0:
            raise ValueError("minimum bin counts must be positive.")
        if self.power_reference <= 0:
            raise ValueError("power_reference must be positive.")
        if self.tonal_neighborhood_width_factor < 0 or self.numerical_tolerance < 0:
            raise ValueError("width factor and tolerance must not be negative.")
        if not 0 < self.occupied_lower_fraction < self.occupied_upper_fraction < 1:
            raise ValueError("occupied fractions must be strictly ordered in (0, 1).")
        if not self.rolloff_fractions or any(
            not isfinite(x) or not 0 < x < 1 for x in self.rolloff_fractions
        ):
            raise ValueError("rolloff fractions must be finite and in (0, 1).")
        if tuple(sorted(set(self.rolloff_fractions))) != self.rolloff_fractions:
            raise ValueError("rolloff fractions must be unique and ordered.")
        optional_nonnegative = (
            self.peak_min_power,
            self.peak_min_prominence,
            self.peak_min_width_bins,
            self.peak_max_width_bins,
        )
        if any(x is not None and (not isfinite(x) or x < 0) for x in optional_nonnegative):
            raise ValueError("peak thresholds and widths must be finite and non-negative.")
        if self.peak_distance_bins is not None and self.peak_distance_bins <= 0:
            raise ValueError("peak_distance_bins must be positive.")
        if (
            self.peak_min_width_bins is not None
            and self.peak_max_width_bins is not None
            and self.peak_min_width_bins > self.peak_max_width_bins
        ):
            raise ValueError("minimum peak width must not exceed maximum.")
        labels = [band.label for band in self.bands]
        if len(labels) != len(set(labels)):
            raise ValueError("band labels must be unique.")
        ordered = sorted(self.bands, key=lambda band: band.frequency_start_hz)
        if tuple(ordered) != self.bands:
            raise ValueError("bands must be ordered.")
        if any(a.frequency_end_hz > b.frequency_start_hz for a, b in zip(ordered, ordered[1:])):
            raise ValueError("bands must not overlap.")


@dataclass(frozen=True, slots=True)
class GlobalSpectralPeakMetric:
    """Métrica de pico matemático; não representa candidato modal."""

    peak_index: int
    bin_frequency_hz: float
    refined_frequency_hz: float | None
    representative_frequency_hz: float
    power: float
    relative_power: float
    prominence: float
    width_bins: float
    width_hz: float
    left_frequency_hz: float
    right_frequency_hz: float
    isolation_index: float | None
    overlap_classification: Literal["isolated", "partially_overlapped", "overlapped", "indeterminate"]
    isolated: bool
    overlapping: bool
    resolution_limited: bool
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        values = (
            self.bin_frequency_hz, self.representative_frequency_hz, self.power,
            self.relative_power, self.prominence, self.width_bins, self.width_hz,
            self.left_frequency_hz, self.right_frequency_hz,
        )
        if self.peak_index < 0 or not all(isfinite(x) for x in values):
            raise ValueError("peak values must be finite and index non-negative.")
        if self.power < 0 or self.prominence < 0 or self.width_bins < 0 or self.width_hz < 0:
            raise ValueError("peak power, prominence and width must not be negative.")
        if not 0 <= self.relative_power <= 1:
            raise ValueError("relative_power must be in [0, 1].")
        if not self.left_frequency_hz <= self.representative_frequency_hz <= self.right_frequency_hz:
            raise ValueError("representative frequency must lie inside peak boundaries.")
        if self.isolated and self.overlapping:
            raise ValueError("a peak cannot be isolated and overlapping.")
        if self.isolation_index is not None and (
            not isfinite(self.isolation_index) or self.isolation_index < 0
        ):
            raise ValueError("isolation_index must be finite and non-negative.")
        _diagnostics(self.diagnostics)


@dataclass(frozen=True, slots=True)
class SpectralBandEnergy:
    """Energia espectral agregada; fronteira final é exclusiva."""

    label: str
    frequency_start_hz: float
    frequency_end_hz: float
    energy: float
    energy_fraction: float
    rms_equivalent: float
    bin_count: int
    peak_count: int
    peak_density_per_hz: float
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.label, "label")
        values = (
            self.frequency_start_hz, self.frequency_end_hz, self.energy,
            self.energy_fraction, self.rms_equivalent, self.peak_density_per_hz,
        )
        if not all(isfinite(x) for x in values) or self.frequency_start_hz < 0:
            raise ValueError("band metrics must be finite and non-negative.")
        if self.frequency_end_hz <= self.frequency_start_hz or self.energy < 0:
            raise ValueError("band bounds and energy are invalid.")
        if not 0 <= self.energy_fraction <= 1 or self.bin_count < 0 or self.peak_count < 0:
            raise ValueError("band fractions and counts are invalid.")
        _diagnostics(self.diagnostics)


@dataclass(frozen=True, slots=True)
class GlobalSpectralCharacterization:
    """Resultado imutável da caracterização global."""

    recording_id: str
    analysis_start_time_s: float
    analysis_end_time_s: float
    sample_rate_hz: int
    fft_size: int
    bin_spacing_hz: float
    frequency_resolution_hz: float
    frequency_min_hz: float
    frequency_max_hz: float
    frequency_bin_count: int
    finite_bin_count: int
    positive_bin_count: int
    zero_bin_count: int
    discarded_bin_count: int
    original_spectral_domain: str
    canonical_spectral_domain: str
    spectral_normalization: str
    total_spectral_energy: float
    spectral_centroid_hz: float | None
    spectral_variance_hz2: float | None
    spectral_spread_hz: float | None
    spectral_skewness: float | None
    spectral_kurtosis: float | None
    rolloff_frequencies_hz: tuple[tuple[float, float | None], ...]
    spectral_flatness: float | None
    spectral_entropy: float | None
    spectral_crest_factor: float | None
    peak_count: int
    significant_peak_count: int
    peak_density_per_hz: float
    peak_density_per_octave: float | None
    mean_peak_spacing_hz: float | None
    median_peak_spacing_hz: float | None
    minimum_peak_spacing_hz: float | None
    maximum_peak_spacing_hz: float | None
    peak_spacing_standard_deviation_hz: float | None
    tonal_energy: float
    tonal_energy_fraction: float | None
    residual_energy: float
    residual_energy_fraction: float | None
    occupied_frequency_lower_hz: float | None
    occupied_frequency_upper_hz: float | None
    occupied_bandwidth_hz: float | None
    occupied_frequency_fraction: float | None
    band_energy_metrics: tuple[SpectralBandEnergy, ...]
    peak_metrics: tuple[GlobalSpectralPeakMetric, ...]
    settings: GlobalSpectralCharacterizationSettings
    valid: bool
    failure_reason: str | None
    diagnostics: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.recording_id, "recording_id")
        if self.analysis_end_time_s <= self.analysis_start_time_s:
            raise ValueError("analysis times must be ordered.")
        if self.sample_rate_hz <= 0 or self.fft_size <= 0:
            raise ValueError("sample rate and FFT size must be positive.")
        if self.bin_spacing_hz <= 0 or self.frequency_resolution_hz <= 0:
            raise ValueError("frequency scales must be positive.")
        if self.frequency_max_hz < self.frequency_min_hz:
            raise ValueError("frequency limits must be ordered.")
        if not (0 <= self.finite_bin_count <= self.frequency_bin_count):
            raise ValueError("finite bin count is incoherent.")
        if self.positive_bin_count + self.zero_bin_count != self.finite_bin_count:
            raise ValueError("positive and zero bin counts are incoherent.")
        if self.discarded_bin_count < 0 or self.total_spectral_energy < 0:
            raise ValueError("discarded count and energy must not be negative.")
        for value in (
            self.spectral_flatness, self.spectral_entropy,
            self.tonal_energy_fraction, self.residual_energy_fraction,
            self.occupied_frequency_fraction,
        ):
            if value is not None and (not isfinite(value) or not 0 <= value <= 1):
                raise ValueError("spectral fractions must be finite and in [0, 1].")
        if self.spectral_crest_factor is not None and self.spectral_crest_factor < 1:
            raise ValueError("spectral crest factor must be at least one.")
        if self.tonal_energy_fraction is not None and not isclose(
            self.tonal_energy_fraction + (self.residual_energy_fraction or 0.0),
            1.0, abs_tol=self.settings.numerical_tolerance,
        ):
            raise ValueError("tonal and residual fractions must sum to one.")
        if self.valid != (self.failure_reason is None):
            raise ValueError("valid and failure_reason are incoherent.")
        _diagnostics(self.diagnostics)

    def rolloff(self, fraction: float) -> float | None:
        """Retorna o primeiro bin cuja energia acumulada atinge a fração."""
        return dict(self.rolloff_frequencies_hz).get(fraction)

    @property
    def spectral_rolloff_50_hz(self) -> float | None:
        return self.rolloff(0.50)

    @property
    def spectral_rolloff_85_hz(self) -> float | None:
        return self.rolloff(0.85)

    @property
    def spectral_rolloff_90_hz(self) -> float | None:
        return self.rolloff(0.90)

    @property
    def spectral_rolloff_95_hz(self) -> float | None:
        return self.rolloff(0.95)


@dataclass(frozen=True, slots=True)
class SpectralComparabilityResult:
    """Diagnóstico somente; não compara nem normaliza valores medidos."""

    comparable: bool
    incompatibilities: tuple[str, ...]

    def __post_init__(self) -> None:
        _diagnostics(self.incompatibilities)
        if self.comparable == bool(self.incompatibilities):
            raise ValueError("comparability state is incoherent.")


def _linear_power(spectrum: Spectrum, settings: GlobalSpectralCharacterizationSettings) -> tuple[np.ndarray, str]:
    values = np.asarray(spectrum.magnitudes, dtype=np.float64)
    unit = spectrum.magnitude_unit.lower()
    declared = settings.spectral_input_domain
    if declared == "linear_power":
        if np.any(np.isfinite(values) & (values < 0)):
            raise ValueError("linear power must not contain negative values.")
        return values.copy(), "linear_power"
    db_input = declared == "dbfs_amplitude" or (declared == "auto" and "dbfs" in unit)
    linear_input = declared == "linear_amplitude" or (declared == "auto" and "dbfs" not in unit)
    if db_input:
        if settings.power_reference <= 0:
            raise ValueError("a positive dBFS power reference is required.")
        amplitude = np.zeros_like(values)
        finite = np.isfinite(values)
        amplitude[finite] = np.power(10.0, values[finite] / 20.0)
        return settings.power_reference * amplitude**2, "dbfs_amplitude"
    if linear_input:
        return values**2, "linear_amplitude"
    raise ValueError("spectral input domain is not recoverable.")


def characterize_global_spectrum(
    spectrum: Spectrum,
    settings: GlobalSpectralCharacterizationSettings | None = None,
    *,
    recording_id: str = "spectrum",
) -> GlobalSpectralCharacterization:
    """Caracteriza uma distribuição espectral usando potência linear canônica."""
    cfg = settings or GlobalSpectralCharacterizationSettings()
    _text(recording_id, "recording_id")
    frequencies = np.asarray(spectrum.frequencies_hz, dtype=np.float64)
    power_all, original_domain = _linear_power(spectrum, cfg)
    if np.any(np.isfinite(power_all) & (power_all < 0)):
        raise ValueError("canonical power must not contain negative values.")
    upper = cfg.frequency_max_hz
    if upper is None:
        upper = float(frequencies[-1])
    mask = (frequencies >= cfg.frequency_min_hz) & (frequencies <= upper)
    selected_f = frequencies[mask]
    selected_p = power_all[mask]
    if selected_f.size < cfg.minimum_bin_count:
        raise ValueError("analyzed range contains fewer than minimum_bin_count bins.")
    finite = np.isfinite(selected_p)
    discarded = int(selected_p.size - np.count_nonzero(finite))
    f = selected_f[finite]
    p = selected_p[finite]
    if np.any(p < 0):
        raise ValueError("power must not contain negative values.")
    total = float(np.sum(p))
    positive = p > 0
    positive_count = int(np.count_nonzero(positive))
    zero_count = int(p.size - positive_count)
    diagnostics: list[str] = [
        "metricas_espectrais_globais_nao_sao_diagnostico_fisico",
        "bin_spacing_different_from_true_spectral_resolution",
        "canonical_domain:linear_power",
        f"original_domain:{original_domain}",
        "rolloff:first_bin_inclusive_no_interpolation",
        "flatness:positive_bins_only_no_epsilon",
        "entropy:zero_terms_omitted",
        "peak_width:half_prominence_in_canonical_power",
        "residual_energy_is_not_automatically_noise",
    ]
    if discarded:
        diagnostics.append(f"nonfinite_bins_discarded:{discarded}")
    if zero_count:
        diagnostics.append(f"zero_power_bins:{zero_count}")
    centroid = variance = spread = skewness = kurtosis = None
    rolloffs: list[tuple[float, float | None]] = []
    flatness = entropy = crest = None
    if total > 0 and p.size:
        weights = p / total
        centroid = float(np.sum(f * weights))
        variance = float(np.sum(((f - centroid) ** 2) * weights))
        spread = float(np.sqrt(max(variance, 0.0)))
        if variance > cfg.numerical_tolerance:
            standardized = (f - centroid) / spread
            skewness = float(np.sum(weights * standardized**3))
            kurtosis = float(np.sum(weights * standardized**4))
        else:
            diagnostics.append("spectral_skewness_kurtosis_undefined_zero_variance")
        cumulative = np.cumsum(p)
        for fraction in cfg.rolloff_fractions:
            index = min(int(np.searchsorted(cumulative, fraction * total, side="left")), p.size - 1)
            rolloffs.append((fraction, float(f[index])))
        if positive_count >= cfg.minimum_positive_bins_for_flatness:
            positive_power = p[positive]
            flatness = float(np.exp(np.mean(np.log(positive_power))) / np.mean(positive_power))
            flatness = min(1.0, max(0.0, flatness))
        else:
            diagnostics.append("spectral_flatness_insufficient_positive_bins")
        if positive_count == 1:
            entropy = 0.0
        elif positive_count > 1:
            probabilities = p[positive] / total
            entropy = float(-np.sum(probabilities * np.log(probabilities)) / np.log(positive_count))
            entropy = min(1.0, max(0.0, entropy))
        else:
            diagnostics.append("spectral_entropy_no_positive_bins")
        crest = float(np.max(p) / np.mean(p))
    else:
        rolloffs = [(fraction, None) for fraction in cfg.rolloff_fractions]
        diagnostics.extend(("zero_total_spectral_energy", "distribution_metrics_undefined"))

    peak_spectrum = Spectrum(
        frequencies_hz=tuple(float(x) for x in f),
        magnitudes=tuple(float(x) for x in p),
        magnitude_unit="linear power",
        window_name=spectrum.window_name,
        fft_size=spectrum.fft_size,
        sample_rate_hz=spectrum.sample_rate_hz,
        original_size=spectrum.original_size,
        bin_spacing_hz=spectrum.bin_spacing_hz,
        normalization="canonical_linear_power",
        interval_start_s=spectrum.interval_start_s,
        interval_end_s=spectrum.interval_end_s,
        remove_mean=spectrum.remove_mean,
    )
    peak_result = detect_spectral_peaks(
        peak_spectrum,
        PeakDetectionSettings(
            min_amplitude=cfg.peak_min_power,
            min_prominence=cfg.peak_min_prominence,
            distance_bins=cfg.peak_distance_bins,
            min_width_bins=cfg.peak_min_width_bins,
            max_width_bins=cfg.peak_max_width_bins,
            interpolate=True,
            sort_by="frequency",
        ),
    )
    base_peaks = peak_result.peaks
    spacing = spectrum.bin_spacing_hz or float(np.median(np.diff(f)))
    physical_resolution = 1.0 / (
        (spectrum.interval_end_s or 0.0) - (spectrum.interval_start_s or 0.0)
    ) if (
        spectrum.interval_start_s is not None
        and spectrum.interval_end_s is not None
        and spectrum.interval_end_s > spectrum.interval_start_s
    ) else (
        float(spectrum.sample_rate_hz / spectrum.original_size)
        if spectrum.sample_rate_hz and spectrum.original_size else spacing
    )
    local_indices = np.asarray([peak.bin_index for peak in base_peaks], dtype=int)
    widths, _, left_ips, right_ips = (
        peak_widths(p, local_indices, rel_height=0.5)
        if local_indices.size else (np.array([]),) * 4
    )
    provisional: list[dict[str, object]] = []
    for peak, width_bins, left_ip, right_ip in zip(base_peaks, widths, left_ips, right_ips, strict=True):
        left_hz = float(np.interp(left_ip, np.arange(f.size), f))
        right_hz = float(np.interp(right_ip, np.arange(f.size), f))
        representative = peak.refined_frequency_hz or peak.bin_frequency_hz
        representative = min(right_hz, max(left_hz, representative))
        provisional.append({
            "peak": peak, "width_bins": float(width_bins), "left": left_hz,
            "right": right_hz, "representative": representative,
        })
    peak_metrics: list[GlobalSpectralPeakMetric] = []
    for index, item in enumerate(provisional):
        peak = item["peak"]
        assert hasattr(peak, "bin_index")
        width_hz = float(item["right"]) - float(item["left"])
        neighbor_distances = [
            abs(float(item["representative"]) - float(other["representative"]))
            for j, other in enumerate(provisional) if j != index
        ]
        nearest = min(neighbor_distances) if neighbor_distances else None
        isolation = nearest / width_hz if nearest is not None and width_hz > 0 else None
        overlaps = [
            min(float(item["right"]), float(other["right"]))
            - max(float(item["left"]), float(other["left"]))
            for j, other in enumerate(provisional) if j != index
        ]
        positive_overlap = any(value > cfg.numerical_tolerance for value in overlaps)
        touching = any(abs(value) <= cfg.numerical_tolerance for value in overlaps)
        classification = (
            "overlapped" if positive_overlap
            else "partially_overlapped" if touching
            else "isolated" if provisional and len(provisional) > 1
            else "indeterminate"
        )
        peak_metrics.append(GlobalSpectralPeakMetric(
            peak_index=peak.bin_index,
            bin_frequency_hz=peak.bin_frequency_hz,
            refined_frequency_hz=peak.refined_frequency_hz,
            representative_frequency_hz=float(item["representative"]),
            power=float(p[peak.bin_index]),
            relative_power=float(p[peak.bin_index] / total) if total else 0.0,
            prominence=float(peak.prominence or 0.0),
            width_bins=float(item["width_bins"]),
            width_hz=width_hz,
            left_frequency_hz=float(item["left"]),
            right_frequency_hz=float(item["right"]),
            isolation_index=isolation,
            overlap_classification=classification,
            isolated=classification == "isolated",
            overlapping=classification == "overlapped",
            resolution_limited=width_hz <= physical_resolution + cfg.numerical_tolerance,
            diagnostics=("width_not_a_formal_uncertainty",),
        ))
    representative_frequencies = [peak.representative_frequency_hz for peak in peak_metrics]
    differences = np.diff(representative_frequencies)
    if differences.size:
        mean_spacing = float(np.mean(differences))
        median_spacing = float(np.median(differences))
        min_spacing = float(np.min(differences))
        max_spacing = float(np.max(differences))
        std_spacing = float(np.std(differences))
    else:
        mean_spacing = median_spacing = min_spacing = max_spacing = std_spacing = None
        diagnostics.append("peak_spacing_requires_at_least_two_significant_peaks")

    tonal_mask = np.zeros(p.size, dtype=bool)
    for item in provisional:
        center = float(item["representative"])
        half_width = 0.5 * (float(item["right"]) - float(item["left"])) * cfg.tonal_neighborhood_width_factor
        tonal_mask |= (f >= center - half_width) & (f <= center + half_width)
    tonal_energy = float(np.sum(p[tonal_mask]))
    residual_energy = max(0.0, total - tonal_energy)
    tonal_fraction = tonal_energy / total if total else None
    residual_fraction = residual_energy / total if total else None
    rolloff_map = dict(rolloffs)
    occupied_lower = rolloff_map.get(cfg.occupied_lower_fraction)
    occupied_upper = rolloff_map.get(cfg.occupied_upper_fraction)
    if total and (occupied_lower is None or occupied_upper is None):
        cumulative = np.cumsum(p)
        occupied_lower = float(f[np.searchsorted(cumulative, cfg.occupied_lower_fraction * total)])
        occupied_upper = float(f[np.searchsorted(cumulative, cfg.occupied_upper_fraction * total)])
    occupied_width = (
        occupied_upper - occupied_lower
        if occupied_lower is not None and occupied_upper is not None else None
    )
    analyzed_width = float(f[-1] - f[0])
    occupied_fraction = (
        occupied_width / analyzed_width if occupied_width is not None and analyzed_width > 0 else None
    )
    band_metrics: list[SpectralBandEnergy] = []
    for band in cfg.bands:
        band_mask = (f >= band.frequency_start_hz) & (f < band.frequency_end_hz)
        band_energy = float(np.sum(p[band_mask]))
        band_peak_count = sum(
            band.frequency_start_hz <= peak.representative_frequency_hz < band.frequency_end_hz
            for peak in peak_metrics
        )
        width = band.frequency_end_hz - band.frequency_start_hz
        band_metrics.append(SpectralBandEnergy(
            label=band.label,
            frequency_start_hz=band.frequency_start_hz,
            frequency_end_hz=band.frequency_end_hz,
            energy=band_energy,
            energy_fraction=band_energy / total if total else 0.0,
            rms_equivalent=float(np.sqrt(band_energy)),
            bin_count=int(np.count_nonzero(band_mask)),
            peak_count=band_peak_count,
            peak_density_per_hz=band_peak_count / width,
            diagnostics=("frequency_start_inclusive_end_exclusive",),
        ))
    density_hz = len(peak_metrics) / analyzed_width if analyzed_width > 0 else 0.0
    density_octave = None
    if f[0] > 0 and f[-1] > f[0]:
        density_octave = len(peak_metrics) / log2(f[-1] / f[0])
    else:
        diagnostics.append("peak_density_per_octave_unavailable_nonpositive_lower_bound")
    start = spectrum.interval_start_s or 0.0
    if spectrum.interval_end_s is not None:
        end = spectrum.interval_end_s
    elif spectrum.original_size and spectrum.sample_rate_hz:
        end = start + spectrum.original_size / spectrum.sample_rate_hz
    else:
        end = start + 1.0 / physical_resolution
    return GlobalSpectralCharacterization(
        recording_id=recording_id,
        analysis_start_time_s=start,
        analysis_end_time_s=end,
        sample_rate_hz=spectrum.sample_rate_hz or int(round(spacing * (spectrum.fft_size or 1))),
        fft_size=spectrum.fft_size or max(1, 2 * (len(spectrum.frequencies_hz) - 1)),
        bin_spacing_hz=spacing,
        frequency_resolution_hz=physical_resolution,
        frequency_min_hz=float(f[0]),
        frequency_max_hz=float(f[-1]),
        frequency_bin_count=int(selected_f.size),
        finite_bin_count=int(p.size),
        positive_bin_count=positive_count,
        zero_bin_count=zero_count,
        discarded_bin_count=discarded,
        original_spectral_domain=original_domain,
        canonical_spectral_domain="linear_power",
        spectral_normalization=f"{spectrum.normalization or 'unspecified'};power=amplitude_squared",
        total_spectral_energy=total,
        spectral_centroid_hz=centroid,
        spectral_variance_hz2=variance,
        spectral_spread_hz=spread,
        spectral_skewness=skewness,
        spectral_kurtosis=kurtosis,
        rolloff_frequencies_hz=tuple(rolloffs),
        spectral_flatness=flatness,
        spectral_entropy=entropy,
        spectral_crest_factor=crest,
        peak_count=peak_result.candidate_count,
        significant_peak_count=len(peak_metrics),
        peak_density_per_hz=density_hz,
        peak_density_per_octave=density_octave,
        mean_peak_spacing_hz=mean_spacing,
        median_peak_spacing_hz=median_spacing,
        minimum_peak_spacing_hz=min_spacing,
        maximum_peak_spacing_hz=max_spacing,
        peak_spacing_standard_deviation_hz=std_spacing,
        tonal_energy=tonal_energy,
        tonal_energy_fraction=tonal_fraction,
        residual_energy=residual_energy,
        residual_energy_fraction=residual_fraction,
        occupied_frequency_lower_hz=occupied_lower,
        occupied_frequency_upper_hz=occupied_upper,
        occupied_bandwidth_hz=occupied_width,
        occupied_frequency_fraction=occupied_fraction,
        band_energy_metrics=tuple(band_metrics),
        peak_metrics=tuple(peak_metrics),
        settings=cfg,
        valid=total > 0,
        failure_reason=None if total > 0 else "zero_total_spectral_energy",
        diagnostics=tuple(dict.fromkeys(diagnostics)),
    )


def characterize_signal_spectrum(
    signal: Signal,
    settings: GlobalSpectralCharacterizationSettings | None = None,
    *,
    recording_id: str = "signal",
) -> GlobalSpectralCharacterization:
    """Calcula a FFT explicitamente e caracteriza seu espectro global."""
    cfg = settings or GlobalSpectralCharacterizationSettings()
    spectrum = analyze_spectrum(
        signal,
        SpectrumAnalysisSettings(
            start_time_s=cfg.start_time_s,
            end_time_s=cfg.end_time_s,
            remove_mean=cfg.detrend_policy == "mean",
            window_name=cfg.window_name,
            n_fft=cfg.fft_size,
            scale="linear_amplitude",
        ),
    ).spectrum
    return characterize_global_spectrum(spectrum, cfg, recording_id=recording_id)


def characterize_recording_spectrum(
    recording: Recording,
    settings: GlobalSpectralCharacterizationSettings | None = None,
    *,
    recording_id: str | None = None,
) -> GlobalSpectralCharacterization:
    """Adaptador sem I/O para uma gravação que já contém um ``Signal``."""
    return characterize_signal_spectrum(
        recording.signal,
        settings,
        recording_id=recording_id or recording.bell_id,
    )


def evaluate_spectral_characterization_comparability(
    first: GlobalSpectralCharacterization,
    second: GlobalSpectralCharacterization,
) -> SpectralComparabilityResult:
    """Expõe incompatibilidades; não efetua comparação de métricas."""
    issues: list[str] = []
    checks = (
        ("sample_rate", first.sample_rate_hz, second.sample_rate_hz),
        ("frequency_range", (first.frequency_min_hz, first.frequency_max_hz), (second.frequency_min_hz, second.frequency_max_hz)),
        ("analysis_duration", first.analysis_end_time_s - first.analysis_start_time_s, second.analysis_end_time_s - second.analysis_start_time_s),
        ("fft_size", first.fft_size, second.fft_size),
        ("window", first.settings.window_name, second.settings.window_name),
        ("detrending", first.settings.detrend_policy, second.settings.detrend_policy),
        ("physical_resolution", first.frequency_resolution_hz, second.frequency_resolution_hz),
        ("peak_criteria", (
            first.settings.peak_min_power, first.settings.peak_min_prominence,
            first.settings.peak_distance_bins, first.settings.peak_min_width_bins,
            first.settings.peak_max_width_bins,
        ), (
            second.settings.peak_min_power, second.settings.peak_min_prominence,
            second.settings.peak_distance_bins, second.settings.peak_min_width_bins,
            second.settings.peak_max_width_bins,
        )),
        ("spectral_domain", (first.original_spectral_domain, first.canonical_spectral_domain), (second.original_spectral_domain, second.canonical_spectral_domain)),
        ("spectral_normalization", first.spectral_normalization, second.spectral_normalization),
    )
    for label, left, right in checks:
        if left != right:
            issues.append(f"incompatible_{label}")
    return SpectralComparabilityResult(not issues, tuple(issues))
