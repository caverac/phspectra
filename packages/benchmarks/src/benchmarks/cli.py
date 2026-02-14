"""Click CLI group entry point for benchmarks."""

from __future__ import annotations

import click
from benchmarks.commands.compare import compare
from benchmarks.commands.compare_plot import compare_plot
from benchmarks.commands.download import download
from benchmarks.commands.inspect_pixel import inspect_pixel
from benchmarks.commands.ncomp_rms_plot import ncomp_rms_plot
from benchmarks.commands.performance import performance_plot
from benchmarks.commands.persistence_plot import persistence_plot
from benchmarks.commands.pipeline import pipeline
from benchmarks.commands.survey_map import survey_map
from benchmarks.commands.synthetic import synthetic
from benchmarks.commands.train_beta import train_beta


@click.group()
def main() -> None:
    """Benchmark suite for phspectra vs GaussPy+."""


main.add_command(download)
main.add_command(compare)
main.add_command(compare_plot)
main.add_command(train_beta)
main.add_command(inspect_pixel)
main.add_command(performance_plot)
main.add_command(synthetic)
main.add_command(persistence_plot)
main.add_command(ncomp_rms_plot)
main.add_command(survey_map)
main.add_command(pipeline)
