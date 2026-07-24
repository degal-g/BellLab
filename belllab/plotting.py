"""Interfaces para futuras visualizações científicas.

Este módulo será responsável por figuras temporais, espectrais e comparativas,
isolando dependências gráficas do restante do modelo de domínio.
"""

from __future__ import annotations

from belllab.recording import Recording


def plot_recording(recording: Recording) -> None:
    """Reserva a interface para visualizar dados de uma gravação.

    Args:
        recording: Gravação cujos resultados serão visualizados no futuro.

    Raises:
        NotImplementedError: Sempre, enquanto não houver visualizações.
    """
    raise NotImplementedError("Plotting is not implemented yet.")
