"""segmentação subcelular e relação núcleo/citoplasma (N/C).

associa cada núcleo à célula que mais o contém, deriva a área do
citoplasma (célula menos núcleo) e calcula a razão N/C. trata o caso
de célula sem núcleo detectado atribuindo N/C = NaN.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class SubcellularRecord:
    """métricas de área por célula individualizada."""

    cell_id: int
    cell_area: float
    nucleus_area: float
    cytoplasm_area: float
    # razão núcleo/citoplasma; nan quando não há núcleo/citoplasma
    nc_ratio: float
    # nº de núcleos atribuídos (fragmentação sugere apoptose)
    n_fragments: int


@dataclass
class SubcellularResult:
    """resultado agregado da associação subcelular."""

    records: list[SubcellularRecord]
    # mapa de rótulos: pixel = cell_id onde há núcleo atribuído
    nucleus_by_cell: np.ndarray


def associate(
    cell_labels: np.ndarray,
    nucleus_labels: np.ndarray,
    min_overlap: float,
    pixel_area: float,
) -> SubcellularResult:
    """associa núcleos a células e calcula áreas e razão N/C.

    args:
        cell_labels: mapa de rótulos de instância das células.
        nucleus_labels: mapa de rótulos de instância dos núcleos.
        min_overlap: fração mínima do núcleo dentro da célula.
        pixel_area: área de um pixel na unidade configurada.
    """
    nucleus_by_cell = np.zeros_like(cell_labels, dtype=np.int32)
    cell_ids = np.unique(cell_labels)
    cell_ids = cell_ids[cell_ids > 0]

    nucleus_px: dict[int, int] = dict.fromkeys(cell_ids.tolist(), 0)
    fragments: dict[int, int] = dict.fromkeys(cell_ids.tolist(), 0)

    nuc_ids = np.unique(nucleus_labels)
    nuc_ids = nuc_ids[nuc_ids > 0]
    for nuc_id in nuc_ids.tolist():
        nuc_mask = nucleus_labels == nuc_id
        under = cell_labels[nuc_mask]
        under = under[under > 0]
        if under.size == 0:
            continue
        values, counts = np.unique(under, return_counts=True)
        best = int(values[int(np.argmax(counts))])
        overlap_frac = int(counts.max()) / int(nuc_mask.sum())
        if overlap_frac < min_overlap:
            continue
        # usa apenas a interseção núcleo ∩ célula para consistência
        intersection = nuc_mask & (cell_labels == best)
        nucleus_by_cell[intersection] = best
        nucleus_px[best] += int(intersection.sum())
        fragments[best] += 1

    records: list[SubcellularRecord] = []
    for cell_id in cell_ids.tolist():
        cell_px = int((cell_labels == cell_id).sum())
        nuc = nucleus_px[cell_id]
        cyto_px = max(cell_px - nuc, 0)
        # nan quando não há núcleo ou citoplasma para razão válida
        nc_ratio = math.nan
        if nuc and cyto_px:
            nc_ratio = nuc / cyto_px
        records.append(
            SubcellularRecord(
                cell_id=cell_id,
                cell_area=cell_px * pixel_area,
                nucleus_area=nuc * pixel_area,
                cytoplasm_area=cyto_px * pixel_area,
                nc_ratio=nc_ratio,
                n_fragments=fragments[cell_id],
            )
        )

    return SubcellularResult(records=records, nucleus_by_cell=nucleus_by_cell)
