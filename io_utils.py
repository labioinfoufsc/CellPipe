"""entrada/saída de imagens e escrita atômica de arquivos.

lida com a base mista: jpg/jpeg/png (via pillow) e tif/tiff (via
tifffile, suportando 16-bit e multipágina/multicanal). normaliza a
profundidade de bits e remove canal alfa.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

# extensões aceitas na varredura de lote
SUPPORTED_EXTENSIONS: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".heic",
    ".heif"
)


def _drop_alpha(image: np.ndarray) -> np.ndarray:
    """remove canal alfa de arrays rgba/la, se presente."""
    if image.ndim == 3 and image.shape[-1] in (2, 4):
        return image[..., :-1]
    return image


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """normaliza qualquer profundidade de bits para uint8 [0, 255].

    usa min-max por imagem; imagens constantes viram zeros.
    """
    if image.dtype == np.uint8:
        return image
    data = image.astype(np.float64)
    lo, hi = float(data.min()), float(data.max())
    if hi <= lo:
        return np.zeros_like(data, dtype=np.uint8)
    scaled = (data - lo) / (hi - lo)
    return (scaled * 255.0).round().astype(np.uint8)


def _read_tiff(path: Path) -> np.ndarray:
    """lê tiff; achata multipágina pegando o primeiro plano 2d/3d."""
    data = tifffile.imread(str(path))
    # multipágina/z-stack: reduz para o primeiro plano espacial
    while data.ndim > 3:
        data = data[0]
    # heurística: se 3d com primeira dim pequena, é (c, y, x) -> (y, x, c)
    if data.ndim == 3 and data.shape[0] <= 4 and data.shape[-1] > 4:
        data = np.moveaxis(data, 0, -1)
    return data


def read_image(path: str | Path) -> np.ndarray:
    """lê uma imagem de qualquer formato suportado.

    retorna array uint8 em (y, x) para cinza ou (y, x, 3) para rgb,
    sem canal alfa.

    raises:
        FileNotFoundError: caminho inexistente.
        ValueError: extensão não suportada ou arquivo ilegível.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"imagem não encontrada: {path}")
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"extensão não suportada: {ext}")

    if ext in (".tif", ".tiff"):
        image = _read_tiff(path)
    else:
        with Image.open(path) as handle:
            image = np.asarray(handle)

    if image.size == 0:
        raise ValueError(f"imagem vazia ou ilegível: {path}")

    image = _drop_alpha(image)
    return _to_uint8(image)


def iter_image_paths(directory: str | Path) -> Iterator[Path]:
    """itera caminhos de imagens suportadas num diretório (ordenado)."""
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"diretório inválido: {directory}")
    for candidate in sorted(directory.iterdir()):
        if candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield candidate


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    """escreve bytes de forma atômica (tmp + os.replace).

    cria o diretório de saída se necessário. evita arquivos parciais
    em caso de falha durante a escrita.

    raises:
        OSError: falha de permissão ou de sistema de arquivos.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), suffix=path.suffix + ".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except OSError:
        # limpa o temporário para não deixar lixo
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    return path
