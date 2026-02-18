"""Click CLI group entry point for benchmarks."""

from __future__ import annotations

import click
from benchmarks.commands.compare_plot import compare_plot
from benchmarks.commands.download import download
from benchmarks.commands.generate_logo import generate_logo
from benchmarks.commands.grs_map_plot import grs_map_plot
from benchmarks.commands.inspect_pixel import inspect_pixel
from benchmarks.commands.ncomp_rms_plot import ncomp_rms_plot
from benchmarks.commands.performance import performance_plot
from benchmarks.commands.persistence_plot import persistence_plot
from benchmarks.commands.pipeline import pipeline
from benchmarks.commands.pipeline_grs import pipeline_grs
from benchmarks.commands.pre_compute import pre_compute
from benchmarks.commands.survey_map import survey_map
from benchmarks.commands.train_beta import train_beta
from benchmarks.commands.train_synthetic import train_synthetic


@click.group()
def main() -> None:
    """Benchmark suite for phspectra vs GaussPy+."""


main.add_command(download)
main.add_command(generate_logo)
main.add_command(pre_compute)
main.add_command(compare_plot)
main.add_command(train_beta)
main.add_command(train_synthetic)
main.add_command(inspect_pixel)
main.add_command(performance_plot)
main.add_command(persistence_plot)
main.add_command(ncomp_rms_plot)
main.add_command(survey_map)
main.add_command(pipeline)
main.add_command(pipeline_grs)
main.add_command(grs_map_plot)
