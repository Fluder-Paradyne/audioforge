"""Command-line interface for AudioForge."""

from __future__ import annotations

from typing import Annotated

import typer

from audioforge import __version__

app = typer.Typer(
    name="audioforge",
    help="FictionReaper Markdown → single-voice audiobook (local pipeline).",
    invoke_without_command=True,
    add_completion=False,
)


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            help="Show version and exit.",
        ),
    ] = False,
) -> None:
    """AudioForge CLI entrypoint.

    Subcommands for build/prepare/synthesize/package land in later tasks.
    """
    if version:
        typer.echo(f"audioforge {__version__}")
        raise typer.Exit()


def run() -> None:
    """Console-script friendly entry that invokes the Typer app."""
    app()
