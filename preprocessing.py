"""pré-processamento e normalização robusta da entrada mista.

detecta automaticamente se a imagem é cinza ou rgb de cor verdadeira,
escolhe canais plausíveis para núcleo e citoplasma e normaliza por
percentil (1-99) como recomendado pelo cellpose.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# tolerância para considerar canais rgb como cinza replicado
_GRAY_CHANNEL_TOLERANCE = 6.0


@dataclass
class PreprocessedImage:
    """resultado do pré-processamento pronto para segmentação."""

    # rgb uint8 para visualização/sobreposição
    display_rgb: np.ndarray
    # imagem de intensidade normalizada [0, 1] para segmentar citoplasma
    cell_input: np.ndarray
    # imagem normalizada [0, 1] para segmentar núcleo
    nucleus_input: np.ndarray
    # intensidade nuclear crua (uint8) para features morfológicas
    nucleus_intensity: np.ndarray
    is_grayscale: bool
    # confiança do canal nuclear (menor quando inferido de cinza)
    nucleus_confidence: float


def is_grayscale(image: np.ndarray) -> bool:
    """detecta cinza (2d) ou rgb com canais quase idênticos."""
    if image.ndim == 2:
        return True
    if image.ndim == 3 and image.shape[-1] == 1:
        return True
    if image.ndim == 3 and image.shape[-1] == 3:
        b, g, r = (image[..., i].astype(np.float32) for i in range(3))
        max_diff = max(
            float(np.abs(b - g).mean()),
            float(np.abs(g - r).mean()),
        )
        return max_diff < _GRAY_CHANNEL_TOLERANCE
    return False


def to_intensity(image: np.ndarray) -> np.ndarray:
    """converte para intensidade 2d uint8."""
    if image.ndim == 2:
        return image
    if image.shape[-1] == 1:
        return image[..., 0]
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def normalise_percentile(
    image: np.ndarray, low: float = 1.0, high: float = 99.0
) -> np.ndarray:
    """normaliza para [0, 1] recortando nos percentis informados."""
    data = image.astype(np.float32)
    lo, hi = np.percentile(data, (low, high))
    if hi <= lo:
        return np.zeros_like(data, dtype=np.float32)
    return np.clip((data - lo) / (hi - lo), 0.0, 1.0)


def _to_display_rgb(image: np.ndarray) -> np.ndarray:
    """garante rgb uint8 de 3 canais para visualização."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[-1] == 1:
        return cv2.cvtColor(image[..., 0], cv2.COLOR_GRAY2RGB)
    return image


def preprocess(image: np.ndarray) -> PreprocessedImage:
    """padroniza a entrada, lidando com cinza vs rgb automaticamente.

    para rgb de cor verdadeira, usa o canal azul como candidato nuclear
    (convenção dapi/haematoxilina) e o verde como citoplasma. para cinza,
    o mesmo sinal alimenta ambos com confiança nuclear reduzida.
    """
    display_rgb = _to_display_rgb(image)
    grayscale = is_grayscale(image)

    if grayscale:
        intensity = to_intensity(image)
        cell_input = normalise_percentile(intensity)
        # sem canal nuclear dedicado -> mesmo sinal, confiança baixa
        nucleus_intensity = intensity
        nucleus_input = cell_input
        nucleus_confidence = 0.4
    else:
        # rgb verdadeiro: azul ~ núcleo, verde ~ citoplasma
        nucleus_intensity = image[..., 2]
        cytoplasm_channel = image[..., 1]
        cell_input = normalise_percentile(cytoplasm_channel)
        nucleus_input = normalise_percentile(nucleus_intensity)
        nucleus_confidence = 0.8

    return PreprocessedImage(
        display_rgb=display_rgb,
        cell_input=cell_input,
        nucleus_input=nucleus_input,
        nucleus_intensity=nucleus_intensity,
        is_grayscale=grayscale,
        nucleus_confidence=nucleus_confidence,
    )
