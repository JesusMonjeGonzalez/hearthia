from typer.testing import CliRunner

from hearthia import __version__
from hearthia.cli import app

runner = CliRunner()


def test_version_command():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output
