"""Interfaces para futuros relatórios científicos do BellLab.

O módulo reunirá a geração de relatórios reprodutíveis a partir de gravações e
resultados de análise, sem acoplar essa apresentação aos algoritmos.
"""

from __future__ import annotations

from pathlib import Path

from belllab.recording import Recording


def build_report(recording: Recording, output_path: Path) -> None:
    """Reserva a interface de criação de relatório para uma gravação.

    Args:
        recording: Gravação que fornecerá contexto e resultados futuros.
        output_path: Caminho destinado ao relatório gerado.

    Raises:
        NotImplementedError: Sempre, nesta etapa arquitetural.
    """
    raise NotImplementedError("Report generation is not implemented yet.")
