"""estimativa de confluência da cultura.

confluência = (área ocupada por células / área total da imagem) * 100.
usa a união das máscaras de instância (foreground), evitando dupla
contagem já que instâncias não se sobrepõem.
"""

from __future__ import annotations

import numpy as np


def confluence(cell_labels: np.ndarray) -> float:
    """percentual de área da imagem coberto por células."""
    total_px = int(cell_labels.size)
    if total_px == 0:
        return 0.0
    covered_px = int(np.count_nonzero(cell_labels))
    return 100.0 * covered_px / total_px
