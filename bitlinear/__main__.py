import click

from .commands.compile import _compile
from .commands.extract import _extract
from .commands.generate import _generate
from .commands.optimize import _optimize

cli = click.CommandCollection(sources=[
    _compile,
    _extract,
    _generate,
    _optimize,
])

if __name__ == "__main__":
    cli()
