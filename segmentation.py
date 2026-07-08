"""segmentação de instância com cellpose (célula e núcleo).

usa fluxos de gradiente do cellpose, robustos a células muito próximas
("grudadinhas"), em vez de thresholding. carrega os modelos de forma
preguiçosa e reaproveita entre imagens do lote.
"""

from __future__ import annotations

import numpy as np

from cellpipe.config import SegmentationConfig


def _gpu_available(prefer_gpu: bool) -> bool:
    """detecta gpu de forma segura; recai para cpu se indisponível."""
    if not prefer_gpu:
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


class Segmenter:
    """encapsula os modelos cellpose de célula e núcleo."""

    def __init__(self, config: SegmentationConfig) -> None:
        self.config = config
        self.gpu = _gpu_available(config.use_gpu)
        
        # importa aqui para não exigir cellpose em quem só usa i/o
        from cellpose import models, io  # <--- ADICIONE O 'io' AQUI
        
        # Ativa o log interno para mostrar o progresso na tela
        io.logger_setup() 

        self._cell_model = models.CellposeModel(
            gpu=self.gpu, model_type=config.cell_model
        )
        self._nucleus_model = models.CellposeModel(
            gpu=self.gpu, model_type=config.nucleus_model
        )

    def _run(
        self,
        model: object,
        image: np.ndarray,
        diameter: float | None,
    ) -> np.ndarray:
        """executa um modelo e devolve o mapa de rótulos de instância."""
        masks, _flows, _styles = model.eval(
            image,
            diameter=diameter,
            flow_threshold=0.0,  # <--- TRUQUE 1: Desliga o controle de qualidade (pula o flow_error)
            cellprob_threshold=self.config.cellprob_threshold,
            resample=False,      # <--- TRUQUE 2: Mantém o upscaling desligado
            batch_size=1,        # <--- TRUQUE 3: Força a rede a processar o mínimo possível por vez
        )
        return masks.astype(np.int32)

    def segment_cells(self, cell_input: np.ndarray) -> np.ndarray:
        """segmenta células/citoplasma inteiro."""
        return self._run(
            self._cell_model, cell_input, self.config.cell_diameter
        )

    def segment_nuclei(self, nucleus_input: np.ndarray) -> np.ndarray:
        """segmenta núcleos."""
        return self._run(
            self._nucleus_model,
            nucleus_input,
            self.config.nucleus_diameter,
        )
