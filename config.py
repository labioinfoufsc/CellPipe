"""configuração central do pipeline.

reúne todos os parâmetros ajustáveis num único lugar para respeitar
o princípio DRY e facilitar a calibração da heurística morfológica.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class CellState(str, Enum):
    """estados morfológicos classificáveis por célula."""

    INTERPHASE = "Interfase"
    MITOSIS = "Mitose"
    APOPTOSIS = "Apoptose"
    UNKNOWN = "Indeterminado"


# cores rgb por estado, seguras para daltonismo (paleta Okabe-Ito)
STATE_COLOURS: dict[CellState, tuple[int, int, int]] = {
    CellState.INTERPHASE: (0, 158, 115),  # verde-azulado
    CellState.MITOSIS: (230, 159, 0),  # laranja
    CellState.APOPTOSIS: (204, 121, 167),  # rosa
    CellState.UNKNOWN: (128, 128, 128),  # cinza
}


@dataclass
class SegmentationConfig:
    """parâmetros dos modelos Cellpose."""

    # modelo para célula/citoplasma inteiro e para núcleo
    cell_model: str = "cyto3"
    nucleus_model: str = "nuclei"
    # diâmetro médio esperado em px; none deixa o cellpose estimar
    cell_diameter: float | None = None
    nucleus_diameter: float | None = None
    # limiares internos do cellpose
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    # usa gpu quando disponível (auto-detectado em runtime)
    use_gpu: bool = True


@dataclass
class MorphologyThresholds:
    """limiares da heurística de classificação morfológica.

    valores são pontos de partida e devem ser calibrados por dataset.
    áreas relativas são normalizadas pela mediana da imagem.
    """

    # apoptose: célula pequena, irregular e núcleo condensado/fragmentado
    apoptosis_area_ratio_max: float = 0.5
    apoptosis_solidity_max: float = 0.85
    apoptosis_nc_ratio_min: float = 0.6
    # mitose: célula arredondada, núcleo condensado e brilhante
    mitosis_circularity_min: float = 0.8
    mitosis_nc_ratio_min: float = 0.5
    mitosis_solidity_min: float = 0.9
    # normaliza intensidade nuclear (z-score) para detectar condensação
    mitosis_nuclear_intensity_z_min: float = 0.5


@dataclass
class ReportConfig:
    """parâmetros do relatório PDF."""

    title: str = "Relatório de Análise de Cultura de Células"
    author: str = "cellpipe"
    # miniatura máxima da imagem no pdf (largura em pontos)
    image_max_width_pt: float = 480.0
    # nº de casas decimais para razões
    ratio_decimals: int = 3


@dataclass
class PipelineConfig:
    """configuração agregada do pipeline."""

    segmentation: SegmentationConfig = field(
        default_factory=SegmentationConfig
    )
    morphology: MorphologyThresholds = field(
        default_factory=MorphologyThresholds
    )
    report: ReportConfig = field(default_factory=ReportConfig)
    # calibração espacial opcional; none => áreas em px²
    pixel_size_um: float | None = None
    # sobreposição mínima núcleo/célula para associação (fração do núcleo)
    min_nucleus_overlap: float = 0.3
    # exclui células que tocam a borda da imagem
    exclude_border: bool = False
    scale_factor: float = 1.0

    @property
    def area_unit(self) -> str:
        """rótulo de unidade de área conforme calibração."""
        return "µm²" if self.pixel_size_um else "px²"

    @property
    def pixel_area(self) -> float:
        """área de um pixel na unidade configurada, corrigida pela escala."""
        base_area = self.pixel_size_um**2 if self.pixel_size_um else 1.0
        # Se encolhemos a imagem (ex: 0.5), cada pixel novo representa uma área 4x maior da original
        return base_area / (self.scale_factor ** 2)
