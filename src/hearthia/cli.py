"""hearth — the Hearthia command line."""

import typer

from hearthia import __version__

app = typer.Typer(name="hearth", help="Hearthia — control plane for local models.")


@app.callback()
def main() -> None:
    """Hearthia — the self-tending fire for local models."""


@app.command()
def version() -> None:
    """Print the Hearthia version."""
    typer.echo(f"Hearthia {__version__}")
