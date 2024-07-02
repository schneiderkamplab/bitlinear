import click

from .commands.optimize import _optimize

cli = click.CommandCollection(sources=[
    _optimize,
])

if __name__ == "__main__":
    cli()
