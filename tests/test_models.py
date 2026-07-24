"""Testes do contrato básico dos modelos de domínio."""

from pathlib import Path

import pytest

from belllab import (
    BellComparison,
    BellRecording,
    Experiment,
    ModalResults,
    ModalMode,
    RecordingMetrics,
    Signal,
    Recording,
    Spectrum,
    SpectrumResults,
    TemporalResults,
)


def build_signal() -> Signal:
    """Cria um sinal mínimo para testar os modelos sem processá-lo."""
    return Signal(((0.0, 0.1),), 48_000, (0.0, 1 / 48_000), 2 / 48_000, 1, "normalized")


def test_recording_accepts_descriptive_metadata() -> None:
    """Uma gravação deve manter os metadados fornecidos pelo usuário."""
    recording = BellRecording(
        path=Path("data/bell.wav"),
        bell_id="bell-001",
        signal=build_signal(),
        metadata={"location": "tower"},
    )

    assert recording.bell_id == "bell-001"
    assert recording.metadata["location"] == "tower"


def test_recording_accepts_typed_future_results() -> None:
    """Uma gravação deve guardar contratos tipados de análises futuras."""
    metrics = RecordingMetrics(1.5, 48_000, 1, 72_000)
    spectrum = Spectrum((100.0, 200.0), (0.2, 0.1), "linear")
    recording = BellRecording(
        path=Path("data/bell.wav"),
        bell_id="bell-001",
        signal=build_signal(),
        metrics=metrics,
        temporal_results=TemporalResults(),
        spectrum_results=SpectrumResults(spectrum=spectrum),
        modal_results=ModalResults(modes=(ModalMode("hum", 100.0),)),
    )

    assert recording.metrics is metrics
    assert recording.spectrum_results is not None
    assert recording.modal_results is not None


def test_recording_rejects_empty_bell_id() -> None:
    """O identificador do sino é obrigatório."""
    with pytest.raises(ValueError, match="bell_id"):
        BellRecording(path=Path("data/bell.wav"), bell_id=" ", signal=build_signal())


def test_comparison_requires_two_distinct_recordings() -> None:
    """A mesma instância não deve ocupar os dois lados da comparação."""
    recording = BellRecording(
        path=Path("data/bell.wav"), bell_id="bell-001", signal=build_signal()
    )

    with pytest.raises(ValueError, match="distinct"):
        BellComparison(reference=recording, candidate=recording)


def test_generalized_models_preserve_legacy_aliases() -> None:
    """Os nomes históricos devem apontar para os novos modelos públicos."""
    assert BellRecording is Recording
    assert BellComparison is Experiment


def test_recording_accepts_generic_instrument_identifier() -> None:
    """O identificador genérico deve coexistir com o campo legado bell_id."""
    recording = Recording(
        path=Path("data/object.wav"),
        bell_id="legacy-001",
        instrument_id="gong-001",
        signal=build_signal(),
    )

    assert recording.bell_id == "legacy-001"
    assert recording.instrument_id == "gong-001"
