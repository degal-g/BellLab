"""Modelo de domínio para uma gravação acústica de um idiofone percutido.

Este módulo define a representação descritiva de uma gravação WAV e de seus
metadados. O carregamento e a análise do sinal pertencem, respectivamente, aos
módulos :mod:`belllab.io` e aos módulos analíticos; não são realizados aqui.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from belllab.results import (
    ModalResults,
    ProcessingContext,
    SpectrumResults,
    TemporalResults,
)
from belllab.types import RecordingMetrics, Signal


@dataclass(slots=True)
class Recording:
    """Representa uma gravação de idiofone e seus metadados científicos.

    A instância não lê o arquivo WAV nem mantém amostras de áudio nesta etapa.
    Ela estabelece um contrato de dados que poderá ser enriquecido com
    informações de aquisição e resultados tipados das análises temporal,
    espectral e modal.

    Args:
        path: Caminho para o arquivo WAV de origem da gravação.
        bell_id: Identificador estável histórico do objeto registrado.
            O nome é preservado temporariamente por compatibilidade; novas
            especializações de instrumento devem documentar sua identificação
            no campo ``metadata`` até uma RFC específica definir sua evolução.
        instrument_id: Identificador genérico opcional do instrumento ou objeto
            sonoro. Quando ausente, ``bell_id`` continua sendo o identificador
            compatível da gravação.
        signal: Sinal WAV já carregado em memória.
        label: Nome legível da gravação, quando disponível.
        metadata: Metadados complementares, como local, data de aquisição,
            microfone ou posição de medição.
        metrics: Métricas descritivas preenchidas pelo futuro carregador WAV.
        temporal_results: Resultados da futura análise temporal.
        spectrum_results: Resultados da futura análise espectral.
        modal_results: Resultados da futura análise modal.
        processing_context: Contexto que associa sinal, configurações e os
            resultados disponíveis para um ciclo de processamento.

    Raises:
        ValueError: Se ``bell_id`` estiver vazio ou contiver apenas espaços.
    """

    path: Path
    bell_id: str
    signal: Signal
    label: str | None = None
    instrument_id: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)
    metrics: RecordingMetrics | None = None
    temporal_results: TemporalResults | None = None
    spectrum_results: SpectrumResults | None = None
    modal_results: ModalResults | None = None
    processing_context: ProcessingContext | None = None

    def __post_init__(self) -> None:
        """Valida invariantes descritivas sem acessar o sistema de arquivos."""
        if not self.bell_id.strip():
            raise ValueError("bell_id must not be empty.")
        if self.instrument_id is not None and not self.instrument_id.strip():
            raise ValueError("instrument_id must not be empty when provided.")


# Alias de compatibilidade temporária. Será descontinuado em uma futura versão
# maior após a migração pública para Recording.
BellRecording = Recording
