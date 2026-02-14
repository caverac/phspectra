"""CLI entry point for train-gui."""

from __future__ import annotations

from pathlib import Path

import click

_DEFAULT_DATA_DIR = "/tmp/phspectra/compare-docker"
_DEFAULT_TRAINING_SET = str(Path(__file__).resolve().parent.parent.parent / "data" / "training_set.json")


@click.command()
@click.option(
    "--data-dir",
    default=_DEFAULT_DATA_DIR,
    show_default=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory with benchmarks compare output.",
)
@click.option(
    "--start-index",
    default=0,
    show_default=True,
    help="Pixel index to start at.",
)
@click.option(
    "--training-set",
    default=_DEFAULT_TRAINING_SET,
    show_default=True,
    type=click.Path(dir_okay=False),
    help="Path to the training set JSON file.",
)
@click.option(
    "--survey",
    default="GRS",
    show_default=True,
    help="Survey name stored in JSON entries.",
)
def main(data_dir: str, start_index: int, training_set: str, survey: str) -> None:
    """Interactive GUI for curating a Gaussian component training set."""
    from train_gui._loader import load_comparison_data  # pylint: disable=import-outside-toplevel
    from train_gui._state import TrainingSet  # pylint: disable=import-outside-toplevel
    from train_gui._viewer import SpectrumViewer  # pylint: disable=import-outside-toplevel

    click.echo(f"Loading data from {data_dir} ...")
    data = load_comparison_data(data_dir)
    click.echo(f"Loaded {len(data)} spectra.")

    ts = TrainingSet(training_set, survey=survey)
    click.echo(f"Training set: {training_set} ({len(ts.curated_pixels)} curated pixels)")

    click.echo("\nControls:")
    click.echo("  a-z          -- toggle component (see labels on plot)")
    click.echo("  Left/Right   -- previous/next pixel")
    click.echo("  s            -- save training set")
    click.echo("  c            -- clear current pixel")
    click.echo("  q            -- save and quit\n")

    viewer = SpectrumViewer(data, ts, start_index=start_index)
    viewer.show()
