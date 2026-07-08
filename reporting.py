"""módulo de reporting isolado: geração de PDF com ReportLab.

usa o framework Platypus, cujo LongTable faz paginação automática e
repete o cabeçalho em cada página (repeatRows=1). o pdf é montado em
memória e devolvido como bytes, deixando a escrita atômica para o
chamador (io_utils.atomic_write_bytes).
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    LongTable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from cellpipe.config import PipelineConfig
from cellpipe.morphology import CellAnalysis
from cellpipe.visualisation import legend_entries

_TABLE_HEADER = (
    "ID da Célula",
    "Área Célula",
    "Área Núcleo",
    "Área Citoplasma",
    "N/C Ratio",
)


@dataclass
class ImageReportData:
    """dados de uma imagem para compor o relatório."""

    name: str
    overlay: np.ndarray
    analyses: list[CellAnalysis]
    total_cells: int
    confluence: float
    state_counts: dict[str, int] = field(default_factory=dict)
    is_grayscale: bool = False
    nucleus_confidence: float = 0.0


def _rgb_to_flowable(image: np.ndarray, max_width_pt: float) -> Image:
    """converte array rgb em flowable Image mantendo proporção."""
    buffer = io.BytesIO()
    PILImage.fromarray(image.astype(np.uint8)).save(buffer, format="PNG")
    buffer.seek(0)
    height, width = image.shape[:2]
    scale = min(1.0, max_width_pt / float(width))
    return Image(buffer, width=width * scale, height=height * scale)


def _fmt_area(value: float) -> str:
    """formata área com separador de milhar e sem casas decimais."""
    if math.isnan(value):
        return "—"
    return f"{value:,.0f}".replace(",", " ")


def _fmt_ratio(value: float, decimals: int) -> str:
    """formata razão N/C; nan vira travessão."""
    if math.isnan(value):
        return "—"
    return f"{value:.{decimals}f}"


def _build_table(
    analyses: list[CellAnalysis], area_unit: str, decimals: int
) -> LongTable:
    """monta a LongTable de métricas com paginação automática."""
    header = [
        _TABLE_HEADER[0],
        f"{_TABLE_HEADER[1]} ({area_unit})",
        f"{_TABLE_HEADER[2]} ({area_unit})",
        f"{_TABLE_HEADER[3]} ({area_unit})",
        _TABLE_HEADER[4],
    ]
    rows: list[list[str]] = [header]
    for item in sorted(analyses, key=lambda a: a.cell_id):
        rows.append(
            [
                str(item.cell_id),
                _fmt_area(item.cell_area),
                _fmt_area(item.nucleus_area),
                _fmt_area(item.cytoplasm_area),
                _fmt_ratio(item.nc_ratio, decimals),
            ]
        )

    # larguras fixas evitam overflow horizontal em qualquer página
    col_widths = [30 * mm, 35 * mm, 35 * mm, 38 * mm, 25 * mm]
    table = LongTable(rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b3a67")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#eef1f7")],
                ),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _summary_block(data: ImageReportData, styles: object) -> Table:
    """bloco em destaque com total de células e confluência."""
    body = styles["Normal"]
    highlight = (
        f"<b>Total de células:</b> {data.total_cells} &nbsp;&nbsp; "
        f"<b>Confluência:</b> {data.confluence:.1f}%"
    )
    counts = "  ".join(
        f"{name}: {count}" for name, count in data.state_counts.items()
    )
    cells = [
        [Paragraph(highlight, body)],
        [Paragraph(f"<b>Estados:</b> {counts}", body)],
    ]
    table = Table(cells, colWidths=[170 * mm])
    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor("#eef1f7"),
                ),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#2b3a67")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def _legend_paragraph(styles: object) -> Paragraph:
    """legenda de cores por estado morfológico."""
    parts = []
    for name, (r, g, b) in legend_entries():
        swatch = f'<font color="#{r:02x}{g:02x}{b:02x}">■</font>'
        parts.append(f"{swatch} {name}")
    return Paragraph("Legenda: " + "  ".join(parts), styles["Normal"])


def build_image_report(data: ImageReportData, config: PipelineConfig) -> bytes:
    """monta o relatório pdf de uma imagem e devolve os bytes.

    a paginação (imagem, resumo e tabela longa) é resolvida pelo
    Platypus automaticamente.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        title=config.report.title,
        author=config.report.author,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
    )
    styles = getSampleStyleSheet()
    story: list[object] = []

    story.append(Paragraph(config.report.title, styles["Title"]))
    story.append(Paragraph(f"Imagem: {data.name}", styles["Heading2"]))
    modality = "escala de cinza" if data.is_grayscale else "rgb"
    story.append(
        Paragraph(
            f"Modalidade detectada: {modality} · confiança nuclear: "
            f"{data.nucleus_confidence:.0%}",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 6))
    story.append(_summary_block(data, styles))
    story.append(Spacer(1, 8))
    story.append(_legend_paragraph(styles))
    story.append(Spacer(1, 8))

    story.append(
        _rgb_to_flowable(data.overlay, config.report.image_max_width_pt)
    )
    story.append(Spacer(1, 10))
    story.append(
        Paragraph("Métricas quantitativas por célula", styles["Heading3"])
    )
    story.append(
        _build_table(
            data.analyses,
            config.area_unit,
            config.report.ratio_decimals,
        )
    )

    doc.build(story)
    return buffer.getvalue()


def build_batch_summary(
    reports: list[ImageReportData], config: PipelineConfig
) -> bytes:
    """relatório-resumo de lote com uma linha por imagem."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, title="Resumo do lote")
    styles = getSampleStyleSheet()
    story: list[object] = [
        Paragraph("Resumo do lote", styles["Title"]),
        Spacer(1, 8),
    ]
    header = ["Imagem", "Células", "Confluência (%)"]
    rows = [header]
    for item in reports:
        rows.append(
            [item.name, str(item.total_cells), f"{item.confluence:.1f}"]
        )
    table = LongTable(
        rows, colWidths=[110 * mm, 30 * mm, 35 * mm], repeatRows=1
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b3a67")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ]
        )
    )
    story.append(table)
    doc.build(story)
    return buffer.getvalue()
