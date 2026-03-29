"""ExaForge command-line interface.

Provides the ``exaforge`` console entry point with subcommands:

* ``run``     — full inference pipeline
* ``launch``  — launch Aegis endpoints only
* ``status``  — show endpoint health
* ``merge``   — merge shard JSONL files into one
* ``preprocess`` — batch files into JSONL shards or ZIP archives
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from exaforge.config import ExaForgeConfig

app = typer.Typer(
    name="exaforge",
    help="Distributed LLM inference on Aurora.",
    add_completion=False,
)
console = Console()


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ------------------------------------------------------------------
# run
# ------------------------------------------------------------------

@app.command()
def run(
    config: Path = typer.Option(
        ..., "--config", "-c", help="Path to YAML config file"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    limit: int = typer.Option(
        0, "--limit", "-n", help="Process at most N items (0 = no limit). Overrides max_items in config."
    ),
) -> None:
    """Run a full inference pipeline."""
    _setup_logging(verbose)
    cfg = ExaForgeConfig.from_yaml(config)
    if limit > 0:
        cfg.max_items = limit

    from exaforge.aegis_bridge import get_endpoint_pool
    from exaforge.monitor import Monitor
    from exaforge.orchestrator import Orchestrator

    pool = get_endpoint_pool(cfg.aegis)

    monitor = Monitor(
        cfg.monitor,
        total=0,
        endpoint_urls=[ep.url for ep in pool.endpoints],
    )

    orch = Orchestrator(
        cfg, pool, on_progress=monitor.on_progress
    )

    summary = asyncio.run(orch.run())
    monitor.close()

    console.print()
    console.print("[bold green]Run complete[/bold green]")
    _print_summary(summary)


# ------------------------------------------------------------------
# launch
# ------------------------------------------------------------------

@app.command()
def launch(
    config: Path = typer.Option(
        ..., "--config", "-c", help="Path to YAML config file"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Launch Aegis vLLM endpoints only (no inference)."""
    _setup_logging(verbose)
    cfg = ExaForgeConfig.from_yaml(config)

    from exaforge.aegis_bridge import launch_aegis

    endpoints = launch_aegis(cfg.aegis)

    console.print(f"[bold green]{len(endpoints)} endpoint(s) ready[/bold green]")
    for ep in endpoints:
        console.print(f"  {ep}")


# ------------------------------------------------------------------
# status
# ------------------------------------------------------------------

@app.command()
def status(
    config: Path = typer.Option(
        ..., "--config", "-c", help="Path to YAML config file"
    ),
) -> None:
    """Show endpoint health and checkpoint progress."""
    _setup_logging()
    cfg = ExaForgeConfig.from_yaml(config)

    from exaforge.endpoints import EndpointPool

    try:
        pool = EndpointPool.from_file(cfg.aegis.endpoints_file)
    except FileNotFoundError:
        console.print(
            f"[red]Endpoints file not found: {cfg.aegis.endpoints_file}[/red]"
        )
        raise typer.Exit(1)

    health = asyncio.run(pool.check_health())

    table = Table(title="Endpoint Health")
    table.add_column("URL", style="cyan")
    table.add_column("Healthy", style="green")
    for url, ok in health.items():
        table.add_row(url, "yes" if ok else "[red]NO[/red]")
    console.print(table)

    # Checkpoint status
    from exaforge.checkpoint import CheckpointManager

    ckpt = CheckpointManager(cfg.checkpoint)
    console.print(
        f"\nCheckpoint: {ckpt.completed_count} / {ckpt.total_items} completed"
    )


# ------------------------------------------------------------------
# merge
# ------------------------------------------------------------------

@app.command()
def merge(
    output_dir: Path = typer.Argument(
        ..., help="Directory containing shard JSONL files"
    ),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Merged output file path"
    ),
    pattern: str = typer.Option(
        "*.jsonl", "--pattern", "-p", help="Glob pattern for shard files"
    ),
) -> None:
    """Merge shard JSONL files into a single file."""
    if not output_dir.is_dir():
        console.print(f"[red]Directory not found: {output_dir}[/red]")
        raise typer.Exit(1)

    shards = sorted(output_dir.glob(pattern))
    if not shards:
        console.print("[yellow]No matching files found[/yellow]")
        raise typer.Exit(0)

    if output_file is None:
        output_file = output_dir / "merged.jsonl"

    total_lines = 0
    with open(output_file, "w", encoding="utf-8") as out:
        for shard in shards:
            for line in shard.read_text().strip().splitlines():
                line = line.strip()
                if line:
                    out.write(line + "\n")
                    total_lines += 1

    console.print(
        f"[green]Merged {len(shards)} shard(s), "
        f"{total_lines} records -> {output_file}[/green]"
    )


# ------------------------------------------------------------------
# preprocess
# ------------------------------------------------------------------

@app.command()
def preprocess(
    input_dir: Path = typer.Option(
        ..., "--input-dir", "-i", help="Directory of source text files"
    ),
    output_dir: Path = typer.Option(
        ..., "--output-dir", "-o", help="Where to write preprocessed shards"
    ),
    fmt: str = typer.Option(
        "jsonl",
        "--format",
        "-f",
        help="Output format: 'jsonl' or 'zip'",
    ),
    batch_size: int = typer.Option(
        1000,
        "--batch-size",
        "-b",
        help="Number of items per output shard",
    ),
    glob_patterns: Optional[List[str]] = typer.Option(
        None,
        "--glob",
        "-g",
        help="Glob pattern(s) for source files (repeatable). Default: '*.txt'",
    ),
    base_name: str = typer.Option(
        "batch", "--base-name", "-n", help="Prefix for shard filenames"
    ),
    deduplicate: bool = typer.Option(
        False,
        "--deduplicate",
        "-d",
        help="Deduplicate files",
        show_default=True,
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Batch text files into JSONL shards or ZIP archives for fast I/O.

    Converts a directory of many small files (e.g. 58k .mmd papers on
    Lustre) into a handful of larger shards that are dramatically faster
    to read at runtime.

    Examples::

        # Pack 58k .mmd files into JSONL shards of 2000 each
        exaforge preprocess -i /data/papers -o /data/preprocessed -g '*.mmd' -K 2000

        # Pack into ZIP archives for staging to /tmp
        exaforge preprocess -i /data/papers -o /data/preprocessed -g '*.mmd' -f zip -K 2000
    """
    _setup_logging(verbose)

    if glob_patterns is None:
        glob_patterns = ["*.txt"]

    from exaforge.preprocess import preprocess_to_jsonl, preprocess_to_zip

    if fmt == "jsonl":
        total = preprocess_to_jsonl(
            input_dir=input_dir,
            output_dir=output_dir,
            glob_patterns=glob_patterns,
            batch_size=batch_size,
            base_name=base_name,
            deduplicate=deduplicate,
        )
    elif fmt == "zip":
        total = preprocess_to_zip(
            input_dir=input_dir,
            output_dir=output_dir,
            glob_patterns=glob_patterns,
            batch_size=batch_size,
            base_name=base_name,
            deduplicate=deduplicate,
        )
    else:
        console.print(f"[red]Unknown format: {fmt!r}. Use 'jsonl' or 'zip'.[/red]")
        raise typer.Exit(1)

    console.print(
        f"[bold green]Preprocessed {total} file(s) into "
        f"{fmt.upper()} shards in {output_dir}[/bold green]"
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _print_summary(summary: dict) -> None:
    table = Table(title="Run Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    for key, value in summary.items():
        table.add_row(key, str(value))
    console.print(table)


def main() -> None:
    """Entry point for the ``exaforge`` console script."""
    app()
