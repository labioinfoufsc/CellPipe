"""orquestração do pipeline de análise (imagem única e lote).

encadeia pré-processamento, segmentação, análise subcelular,
morfologia, confluência, visualização e reporting. em lote, falhas por
imagem são registradas e não interrompem o processamento (skip-and-
continue).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import cv2

from cellpipe import confluence as confluence_mod
from cellpipe import morphology, subcellular
from cellpipe.config import PipelineConfig
from cellpipe.io_utils import (
    atomic_write_bytes,
    iter_image_paths,
    read_image,
)
from cellpipe.morphology import CellAnalysis
from cellpipe.preprocessing import preprocess
from cellpipe.reporting import (
    ImageReportData,
    build_batch_summary,
    build_image_report,
)
from cellpipe.segmentation import Segmenter
from cellpipe.visualisation import draw_overlay

logger = logging.getLogger(__name__)


@dataclass
class ImageResult:
    """resultado do processamento de uma imagem."""

    name: str
    total_cells: int
    confluence: float
    dataframe: pl.DataFrame
    analyses: list[CellAnalysis]
    pdf_path: Path | None = None
    csv_path: Path | None = None


@dataclass
class BatchResult:
    """resultado agregado de um lote."""

    results: list[ImageResult] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    summary_path: Path | None = None


class CellPipeline:
    """pipeline configurável e reutilizável entre imagens."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self._segmenter: Segmenter | None = None

    @property
    def segmenter(self) -> Segmenter:
        """carrega os modelos cellpose sob demanda (uma vez)."""
        if self._segmenter is None:
            self._segmenter = Segmenter(self.config.segmentation)
        return self._segmenter

    def _to_dataframe(self, analyses: list[CellAnalysis]) -> pl.DataFrame:
        """monta o dataframe polars com as colunas do relatório."""
        unit = self.config.area_unit
        return pl.DataFrame(
            {
                "ID da Célula": [a.cell_id for a in analyses],
                f"Área Célula ({unit})": [a.cell_area for a in analyses],
                f"Área Núcleo ({unit})": [a.nucleus_area for a in analyses],
                f"Área Citoplasma ({unit})": [
                    a.cytoplasm_area for a in analyses
                ],
                "N/C Ratio": [a.nc_ratio for a in analyses],
                "Estado": [a.state.value for a in analyses],
            }
        )

    def _state_counts(self, analyses: list[CellAnalysis]) -> dict[str, int]:
        """conta células por estado morfológico."""
        counts: dict[str, int] = {}
        for item in analyses:
            counts[item.state.value] = counts.get(item.state.value, 0) + 1
        return counts

    def analyse_image(
        self, image: np.ndarray, name: str
    ) -> tuple[ImageResult, ImageReportData]:
        """executa toda a análise sem escrever arquivos."""
        
        scale = self.config.scale_factor
        if scale != 1.0:
            h, w = image.shape[:2]
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # ----------------------------------------------------

        pre = preprocess(image)
        cell_labels = self.segmenter.segment_cells(pre.cell_input)

        pre = preprocess(image)
        cell_labels = self.segmenter.segment_cells(pre.cell_input)
        
        # ---> LINHA QUE ESTAVA FALTANDO <---
        nucleus_labels = self.segmenter.segment_nuclei(pre.nucleus_input) 

        sub = subcellular.associate(
            cell_labels,
            nucleus_labels,
            self.config.min_nucleus_overlap,
            self.config.pixel_area,
        )
        analyses = morphology.analyse(
            cell_labels,
            pre.nucleus_intensity,
            sub.records,
            self.config.morphology,
        )
        conf = confluence_mod.confluence(cell_labels)
        overlay = draw_overlay(
            pre.display_rgb,
            cell_labels,
            sub.nucleus_by_cell,
            analyses,
        )
        dataframe = self._to_dataframe(analyses)
        counts = self._state_counts(analyses)

        result = ImageResult(
            name=name,
            total_cells=len(analyses),
            confluence=conf,
            dataframe=dataframe,
            analyses=analyses,
        )
        report_data = ImageReportData(
            name=name,
            overlay=overlay,
            analyses=analyses,
            total_cells=len(analyses),
            confluence=conf,
            state_counts=counts,
            is_grayscale=pre.is_grayscale,
            nucleus_confidence=pre.nucleus_confidence,
        )
        return result, report_data

    def process_image(
        self, image_path: str | Path, output_dir: str | Path
    ) -> ImageResult:
        """processa uma imagem e salva pdf e csv de forma atômica."""
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        image = read_image(image_path)
        result, report_data = self.analyse_image(image, image_path.name)

        pdf_bytes = build_image_report(report_data, self.config)
        pdf_path = output_dir / f"{image_path.stem}_relatorio.pdf"
        result.pdf_path = atomic_write_bytes(pdf_path, pdf_bytes)

        csv_bytes = result.dataframe.write_csv().encode("utf-8")
        csv_path = output_dir / f"{image_path.stem}_metricas.csv"
        result.csv_path = atomic_write_bytes(csv_path, csv_bytes)

        logger.info(
            "processada %s: %d células, %.1f%% confluência",
            image_path.name,
            result.total_cells,
            result.confluence,
        )
        return result

    def process_directory(
        self, input_dir: str | Path, output_dir: str | Path
    ) -> BatchResult:
        """processa um diretório em lote com skip-and-continue."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        batch = BatchResult()
        report_data_list: list[ImageReportData] = []

        for path in iter_image_paths(input_dir):
            try:
                image = read_image(path)
                result, report_data = self.analyse_image(image, path.name)
                pdf_bytes = build_image_report(report_data, self.config)
                result.pdf_path = atomic_write_bytes(
                    output_dir / f"{path.stem}_relatorio.pdf", pdf_bytes
                )
                csv_bytes = result.dataframe.write_csv().encode("utf-8")
                result.csv_path = atomic_write_bytes(
                    output_dir / f"{path.stem}_metricas.csv", csv_bytes
                )
                batch.results.append(result)
                report_data_list.append(report_data)
            except Exception as exc:  # isola falhas por imagem
                logger.exception("falha ao processar %s", path.name)
                batch.errors[path.name] = str(exc)

        if report_data_list:
            summary = build_batch_summary(report_data_list, self.config)
            batch.summary_path = atomic_write_bytes(
                output_dir / "resumo_lote.pdf", summary
            )
        return batch
