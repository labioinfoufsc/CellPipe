"""interface de linha de comando (typer).

processa uma imagem ou um diretório inteiro, gerando relatórios pdf e
csv. trata erros de i/o de forma amigável e retorna código de saída
não-zero em falha.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from cellpipe.config import PipelineConfig, SegmentationConfig
from cellpipe.pipeline import CellPipeline

app = typer.Typer(
    add_completion=False,
    help="pipeline de análise de imagens de cultura de células.",
)


def _build_config(
    pixel_size_um: float | None,
    use_gpu: bool,
    exclude_border: bool,
    cell_diameter: float | None,
    nucleus_diameter: float | None,
    scale: float,
) -> PipelineConfig:
    """monta a configuração a partir das opções da cli."""
    # Se o usuário informou um diâmetro de célula em pixels, ajustamos para a nova escala
    if cell_diameter:
        cell_diameter *= scale
    if nucleus_diameter:
        nucleus_diameter *= scale

    segmentation = SegmentationConfig(
        use_gpu=use_gpu,
        cell_diameter=cell_diameter,
        nucleus_diameter=nucleus_diameter,
    )
    return PipelineConfig(
        segmentation=segmentation,
        pixel_size_um=pixel_size_um,
        exclude_border=exclude_border,
        scale_factor=scale,
    )

@app.command()
def run(
    input_path: Annotated[
        Path,
        typer.Argument(help="imagem ou diretório de entrada."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="diretório de saída."),
    ] = Path("resultados"),
    pixel_size_um: Annotated[
        float | None,
        typer.Option(help="tamanho do pixel em µm (calibração)."),
    ] = None,
    use_gpu: Annotated[
        bool, typer.Option(help="usar gpu se disponível.")
    ] = True,
    exclude_border: Annotated[
        bool, typer.Option(help="excluir células na borda.")
    ] = False,
    cell_diameter: Annotated[
        float | None,
        typer.Option(help="diâmetro médio da célula em px."),
    ] = None,
    nucleus_diameter: Annotated[
        float | None,
        typer.Option(help="diâmetro médio do núcleo em px."),
    ] = None,
    scale: Annotated[
        float, 
        typer.Option("--scale", "-s", help="fator de redução da imagem (ex: 0.5 para 50%).")
    ] = 1.0,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="log detalhado.")
    ] = False,
) -> None:
    """processa imagem única ou diretório em lote."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    config = _build_config(
        pixel_size_um,
        use_gpu,
        exclude_border,
        cell_diameter,
        nucleus_diameter,
        scale,
    )
    pipeline = CellPipeline(config)

    if not input_path.exists():
        typer.secho(
            f"entrada inexistente: {input_path}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        if input_path.is_dir():
            batch = pipeline.process_directory(input_path, output_dir)
            typer.secho(
                f"lote concluído: {len(batch.results)} imagens, "
                f"{len(batch.errors)} falhas.",
                fg=typer.colors.GREEN,
            )
            for name, message in batch.errors.items():
                typer.secho(
                    f"  falha em {name}: {message}",
                    fg=typer.colors.YELLOW,
                    err=True,
                )
        else:
            result = pipeline.process_image(input_path, output_dir)
            typer.secho(
                f"ok: {result.total_cells} células, "
                f"{result.confluence:.1f}% confluência -> "
                f"{result.pdf_path}",
                fg=typer.colors.GREEN,
            )
    except (OSError, ValueError) as exc:
        typer.secho(f"erro: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
