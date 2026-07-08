"""features morfológicas e classificação heurística de estado.

sem dados rotulados, a classificação usa regras sobre features do
skimage.regionprops (área, solidez, circularidade, razão N/C,
fragmentação e intensidade nuclear). limiares vêm da config e são
calibráveis; a heurística é um ponto de partida, não um classificador
treinado.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from skimage.measure import regionprops

from cellpipe.config import CellState, MorphologyThresholds
from cellpipe.subcellular import SubcellularRecord


@dataclass
class CellAnalysis:
    """registro final por célula, pronto para tabela e visualização."""

    cell_id: int
    cell_area: float
    nucleus_area: float
    cytoplasm_area: float
    nc_ratio: float
    state: CellState
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]


@dataclass
class _Geometry:
    """features geométricas cruas por célula (em pixels)."""

    area_px: float
    solidity: float
    circularity: float
    nuclear_intensity: float
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]


def _circularity(area: float, perimeter: float) -> float:
    """circularidade isoperimétrica em [0, 1]; 1 = círculo."""
    if perimeter <= 0:
        return 0.0
    return float(min(4.0 * math.pi * area / (perimeter**2), 1.0))


def _extract_geometry(
    cell_labels: np.ndarray, nucleus_intensity: np.ndarray
) -> dict[int, _Geometry]:
    """extrai geometria por célula via regionprops."""
    geometry: dict[int, _Geometry] = {}
    props = regionprops(cell_labels, intensity_image=nucleus_intensity)
    for prop in props:
        geometry[int(prop.label)] = _Geometry(
            area_px=float(prop.area),
            solidity=float(prop.solidity),
            circularity=_circularity(prop.area, prop.perimeter),
            nuclear_intensity=float(prop.intensity_mean),
            centroid=(float(prop.centroid[0]), float(prop.centroid[1])),
            bbox=tuple(int(v) for v in prop.bbox),
        )
    return geometry


def _intensity_zscores(
    geometry: dict[int, _Geometry],
) -> dict[int, float]:
    """z-score da intensidade nuclear média entre as células."""
    values = np.array(
        [g.nuclear_intensity for g in geometry.values()], dtype=float
    )
    if values.size == 0:
        return {}
    mean, std = float(values.mean()), float(values.std())
    if std <= 0:
        return dict.fromkeys(geometry, 0.0)
    return {
        cid: (g.nuclear_intensity - mean) / std for cid, g in geometry.items()
    }


def _classify(
    geom: _Geometry,
    sub: SubcellularRecord,
    area_ratio: float,
    intensity_z: float,
    thr: MorphologyThresholds,
) -> CellState:
    """aplica as regras heurísticas de estado morfológico."""
    nc = sub.nc_ratio if not math.isnan(sub.nc_ratio) else 0.0

    # apoptose: célula pequena/irregular ou núcleo fragmentado
    is_small = area_ratio < thr.apoptosis_area_ratio_max
    is_irregular = geom.solidity < thr.apoptosis_solidity_max
    is_fragmented = sub.n_fragments >= 2
    if (is_small and is_irregular) or (
        is_fragmented and nc >= thr.apoptosis_nc_ratio_min
    ):
        return CellState.APOPTOSIS

    # mitose: arredondada, sólida, núcleo condensado e brilhante
    is_round = geom.circularity >= thr.mitosis_circularity_min
    is_solid = geom.solidity >= thr.mitosis_solidity_min
    is_condensed = nc >= thr.mitosis_nc_ratio_min
    is_bright = intensity_z >= thr.mitosis_nuclear_intensity_z_min
    if is_round and is_solid and (is_condensed or is_bright):
        return CellState.MITOSIS

    return CellState.INTERPHASE


def analyse(
    cell_labels: np.ndarray,
    nucleus_intensity: np.ndarray,
    subcellular: list[SubcellularRecord],
    thresholds: MorphologyThresholds,
) -> list[CellAnalysis]:
    """combina áreas subcelulares, geometria e classificação."""
    geometry = _extract_geometry(cell_labels, nucleus_intensity)
    zscores = _intensity_zscores(geometry)
    areas = np.array([g.area_px for g in geometry.values()], dtype=float)
    median_area = float(np.median(areas)) if areas.size else 1.0
    median_area = median_area or 1.0

    results: list[CellAnalysis] = []
    for sub in subcellular:
        geom = geometry.get(sub.cell_id)
        if geom is None:
            continue
        area_ratio = geom.area_px / median_area
        state = _classify(
            geom,
            sub,
            area_ratio,
            zscores.get(sub.cell_id, 0.0),
            thresholds,
        )
        results.append(
            CellAnalysis(
                cell_id=sub.cell_id,
                cell_area=sub.cell_area,
                nucleus_area=sub.nucleus_area,
                cytoplasm_area=sub.cytoplasm_area,
                nc_ratio=sub.nc_ratio,
                state=state,
                centroid=geom.centroid,
                bbox=geom.bbox,
            )
        )
    return results
