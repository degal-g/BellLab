"""Interfaces para futuras análises modais de idiofones percutidos.

O módulo abrigará identificação e organização de parciais ou modos vibracionais
quando os métodos científicos correspondentes forem definidos.
"""

from __future__ import annotations

from belllab.config import AnalysisSettings
from belllab.results import ModalResults
from belllab.types import Signal


def analyze_modes(signal: Signal, settings: AnalysisSettings) -> ModalResults:
    """Reserva a interface de identificação modal de uma gravação.

    Args:
        signal: Sinal carregado a analisar.
        settings: Parâmetros tipados que orientarão a análise.

    Raises:
        NotImplementedError: Sempre, nesta etapa arquitetural.
    """
    raise NotImplementedError("Modal analysis is not implemented yet.")
