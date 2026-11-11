"""`fraudlab data ...` commands: create the demo data and inspect any dataset."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from fraudlab.datasets.config import ConfigError, load_config
from fraudlab.datasets.demo import DEMO_PATH, content_hash, write_demo
from fraudlab.datasets.leakage import check_leakage
from fraudlab.datasets.load import LoadedDataset, load_dataset
from fraudlab.datasets.payload import case_id, field_descriptions, load_question, payload_hash, render_case
from fraudlab.datasets.split import DatasetSplit, split_dataset
from fraudlab.datasets.synthetic import DEFAULT_ROWS, DEFAULT_SEED, GENERATOR_VERSION

data_app = typer.Typer(help="Create the demo data and inspect datasets.", no_args_is_help=True)

DatasetOpt = typer.Option(..., "--dataset", "-d", help="Dataset config, e.g. configs/datasets/ulb.yaml")


def _load(path: Path) -> tuple[LoadedDataset, DatasetSplit]:
    try:
        ds = load_dataset(load_config(path))
        split = split_dataset(ds)
        check_leakage(ds, split)
    except ConfigError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    return ds, split


@data_app.command("generate-demo")
def generate_demo(
    rows: int = typer.Option(DEFAULT_ROWS, help="Number of transactions"),
    seed: int = typer.Option(DEFAULT_SEED, help="Random seed"),
    out: Path = typer.Option(DEMO_PATH, help="Output Parquet file"),
) -> None:
    """Create the synthetic demo dataset (fictional transactions, ~3% fraud)."""
    df = write_demo(out, rows=rows, seed=seed)
    typer.echo(f"Wrote {len(df):,} rows ({df['is_fraud'].mean():.2%} fraud) to {out}")
    typer.echo(f"generator v{GENERATOR_VERSION}, seed {seed}, content hash {content_hash(df)}")


@data_app.command()
def inspect(dataset: Path = DatasetOpt) -> None:
    """Load a dataset, split it, run the leak check and print a summary."""
    ds, split = _load(dataset)
    spec = ds.config.dataset
    typer.echo(f"dataset: {spec.name}  ({len(ds.frame):,} rows)")
    id_name = spec.id_column or "(row number)"
    typer.echo(f"label: {spec.label_column}   id: {id_name}   time: {spec.time_column}")
    allowed = "ALLOWED" if spec.external_processing_allowed else "BLOCKED (local only)"
    typer.echo(f"sending rows to AI providers: {allowed}")
    if spec.license_note:
        typer.echo(f"license: {spec.license_note}")
    shown = ", ".join(ds.features[:15]) + (" ..." if len(ds.features) > 15 else "")
    typer.echo(f"features ({len(ds.features)}): {shown}")
    typer.echo(split.summary())
    typer.echo("leak check: passed")
    typer.echo("fingerprints: " + "  ".join(f"{k}={v}" for k, v in split.hashes.items()))


@data_app.command("show-case")
def show_case(
    dataset: Path = DatasetOpt,
    number: int = typer.Option(1, "--number", "-n", help="Which holdout case to show (1 = first)"),
) -> None:
    """Print exactly what a model would receive for one holdout transaction."""
    ds, split = _load(dataset)
    if not 1 <= number <= split.holdout.rows:
        typer.echo(f"Error: --number must be between 1 and {split.holdout.rows}", err=True)
        raise typer.Exit(code=1)
    position = int(split.holdout.index[number - 1])
    fields = render_case(ds, position)
    question = load_question()
    typer.echo(f"case {case_id(ds, position)}  (holdout #{number}; the ID is not sent to models)")
    typer.echo(f"question ({question.version}): {question.question}")
    typer.echo("fields sent to every model:")
    typer.echo(json.dumps(fields, indent=2, ensure_ascii=False))
    typer.echo(f"payload fingerprint: {payload_hash(fields)}")
    descriptions = field_descriptions(ds)
    if descriptions:
        typer.echo(f"({len(descriptions)} field descriptions are sent once with the question)")
