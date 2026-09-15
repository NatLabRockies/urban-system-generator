"""``usg`` command-line interface. Importing this package (and ``--help``) must not import torch."""

from __future__ import annotations

import logging

import click

from usg.__about__ import __version__
from usg.cli import commands
from usg.cli._common import CONTEXT_SETTINGS
from usg.cli.models import models


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(__version__, prog_name="urban-system-generator")
def cli():
    """urban-system-generator: infer missing residential building characteristics for URBANopt.

    \b
    Typical use:
      usg workflow -i buildings.json -o uo_buildstock_mapping.csv
      usg workflow -i buildings.json -o uo_buildstock_mapping.csv --model tabddpm
      usg complete -i incomplete.csv -o completed.csv
      usg models list

    Inference commands write confidence_report.json/.html unless --no-confidence-report.

    Exit codes: 0 ok, 1 error, 3 model selection/download, 4 bundle verification, 5 invalid input.
    """
    logging.basicConfig(level=logging.ERROR)


for _command in (
    commands.geojson2csv,
    commands.predict_single,
    commands.complete,
    commands.confidence,
    commands.process,
    commands.workflow,
    commands.info,
    models,
):
    cli.add_command(_command)


def main() -> None:
    cli(prog_name="usg")


__all__ = ["cli", "main"]
