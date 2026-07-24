"""Testes dos contratos de dados científicos do BellLab."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from belllab import Envelope, ModalMode, RecordingMetrics, Signal, Spectrum


def test_signal_accepts_source_provenance() -> None:
    """Um sinal deve poder registrar a proveniência de seu arquivo de origem."""
    loaded_at = datetime.now(timezone.utc)
    signal = Signal(
        samples=((0.0,),),
        sample_rate=48_000,
        time=(0.0,),
        duration=1 / 48_000,
        channels=1,
        unit="normalized",
        path=Path("data/bell.wav"),
        filename="bell.wav",
        sha256="a" * 64,
        loaded_at=loaded_at,
    )

    assert signal.path == Path("data/bell.wav")
    assert signal.filename == "bell.wav"
    assert signal.sha256 == "a" * 64
    assert signal.loaded_at is loaded_at


def test_envelope_exposes_method_unit_and_parameters() -> None:
    """Um envelope deve declarar a proveniência de sua futura construção."""
    envelope = Envelope(
        times_s=(0.0, 0.1),
        amplitudes=(1.0, 0.5),
        method="future-method",
        unit="normalized",
        parameters={"window_size": 128},
    )

    assert envelope.method == "future-method"
    assert envelope.unit == "normalized"
    assert envelope.parameters["window_size"] == 128


def test_spectrum_exposes_analysis_context() -> None:
    """Um espectro deve registrar o contexto de sua futura geração."""
    spectrum = Spectrum(
        frequencies_hz=(100.0,),
        magnitudes=(0.5,),
        magnitude_unit="dBFS",
        window_name="hann",
        fft_size=4_096,
        overlap=0.5,
        timestamp=1.25,
    )

    assert spectrum.window_name == "hann"
    assert spectrum.fft_size == 4_096
    assert spectrum.overlap == 0.5
    assert spectrum.timestamp == 1.25


def test_metrics_accept_extended_measurement_fields() -> None:
    """As métricas devem acomodar medidas adicionais sem dicionários."""
    metrics = RecordingMetrics(
        duration_s=1.0,
        sample_rate_hz=48_000,
        channel_count=1,
        sample_count=48_000,
        peak_dbfs=-1.0,
        rms_dbfs=-12.0,
        crest_factor_db=11.0,
        clipping_fraction=0.001,
        clipping_sample_count=48,
    )

    assert metrics.peak_dbfs == -1.0
    assert metrics.clipping_sample_count == 48


def test_modal_mode_is_the_public_modal_type() -> None:
    """A nomenclatura modal pública deve usar ModalMode."""
    mode = ModalMode(name="hum", frequency_hz=100.0)

    assert mode.name == "hum"


def test_signal_rejects_inconsistent_core_dimensions() -> None:
    """Signal deve impedir canais, tempo e duração dimensionalmente incoerentes."""
    with pytest.raises(ValueError, match="channels"):
        Signal(
            samples=((0.0,),),
            sample_rate=48_000,
            time=(0.0,),
            duration=1 / 48_000,
            channels=2,
            unit="normalized",
        )
    with pytest.raises(ValueError, match="duration"):
        Signal(
            samples=((0.0,),),
            sample_rate=48_000,
            time=(0.0,),
            duration=0.0,
            channels=1,
            unit="normalized",
        )


def test_envelope_and_spectrum_validate_axes() -> None:
    """Séries científicas devem rejeitar eixos incompatíveis ou ambíguos."""
    with pytest.raises(ValueError, match="same length"):
        Envelope(times_s=(0.0,), amplitudes=(1.0, 2.0))
    with pytest.raises(ValueError, match="ordered"):
        Spectrum(
            frequencies_hz=(20.0, 10.0),
            magnitudes=(1.0, 1.0),
            magnitude_unit="linear",
        )
    with pytest.raises(ValueError, match="overlap"):
        Spectrum(
            frequencies_hz=(0.0,),
            magnitudes=(1.0,),
            magnitude_unit="linear",
            overlap=1.0,
        )
