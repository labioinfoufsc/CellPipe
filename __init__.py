from __future__ import annotations

from typing import TYPE_CHECKING

from cellpipe.config import PipelineConfig

if TYPE_CHECKING:
    from cellpipe.pipeline import CellPipeline, ImageResult

__all__ = ["CellPipeline", "ImageResult", "PipelineConfig"]
__version__ = "0.1.0"

_LAZY = {"CellPipeline": "pipeline", "ImageResult": "pipeline"}


def __getattr__(name: str) -> object:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module 'cellpipe' has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f"cellpipe.{module_name}")
    return getattr(module, name)
