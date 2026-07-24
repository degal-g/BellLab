"""Modelo de domínio para experimentos entre gravações de idiofones.

O módulo organiza a intenção de comparar duas gravações. Métricas acústicas,
alinhamento temporal e cálculos de similaridade serão implementados em etapas
posteriores e não fazem parte deste esqueleto.
"""

from __future__ import annotations

from dataclasses import dataclass

from belllab.recording import Recording


@dataclass(slots=True)
class Experiment:
    """Descreve a comparação binária atual entre duas gravações.

    Args:
        reference: Gravação tomada como referência da comparação.
        candidate: Gravação a ser comparada à referência.
        label: Identificador opcional para relatórios ou comparações.

    Raises:
        ValueError: Se referência e candidata forem a mesma instância.
    
    Este contrato ainda não representa uma campanha ou um experimento com
    múltiplas condições; essas extensões permanecem fora do escopo atual.
    """

    reference: Recording
    candidate: Recording
    label: str | None = None

    def __post_init__(self) -> None:
        """Protege contra uma comparação sem dois objetos distintos."""
        if self.reference is self.candidate:
            raise ValueError("reference and candidate must be distinct objects.")


# Alias de compatibilidade temporária. Será descontinuado em uma futura versão
# maior após a migração pública para Experiment.
BellComparison = Experiment
