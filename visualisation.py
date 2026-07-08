"""sobreposição visual das segmentações e estados morfológicos.

desenha contornos de célula coloridos por estado, contorno do núcleo,
id e (opcionalmente) bounding box sobre a imagem rgb.
"""

from __future__ import annotations

import cv2
import numpy as np

from cellpipe.config import STATE_COLOURS, CellState
from cellpipe.morphology import CellAnalysis


def _draw_contour(
    canvas: np.ndarray,
    mask: np.ndarray,
    colour: tuple[int, int, int],
    thickness: int,
) -> None:
    """desenha o contorno de uma máscara binária no canvas."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(canvas, contours, -1, colour, thickness)


def draw_overlay(
    display_rgb: np.ndarray,
    cell_labels: np.ndarray,
    nucleus_by_cell: np.ndarray,
    analyses: list[CellAnalysis],
    draw_bbox: bool = False,
) -> np.ndarray:
    """gera a imagem rotulada com contornos, ids e estados."""
    canvas = display_rgb.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)

    for item in analyses:
        colour = STATE_COLOURS.get(
            item.state, STATE_COLOURS[CellState.UNKNOWN]
        )
        cell_mask = cell_labels == item.cell_id
        _draw_contour(canvas, cell_mask, colour, thickness=2)

        nuc_mask = nucleus_by_cell == item.cell_id
        if nuc_mask.any():
            _draw_contour(canvas, nuc_mask, colour, thickness=1)

        if draw_bbox:
            min_r, min_c, max_r, max_c = item.bbox
            cv2.rectangle(canvas, (min_c, min_r), (max_c, max_r), colour, 1)

        y, x = item.centroid
        cv2.putText(
            canvas,
            str(item.cell_id),
            (int(x), int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            colour,
            1,
            cv2.LINE_AA,
        )
    return canvas


def legend_entries() -> list[tuple[str, tuple[int, int, int]]]:
    """rótulos e cores para a legenda de estados no relatório."""
    return [
        (state.value, colour)
        for state, colour in STATE_COLOURS.items()
        if state is not CellState.UNKNOWN
    ]
