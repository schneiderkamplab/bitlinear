import click
import os
import random
import torch

from ..utils import (
    INFO,
    install_signal_handler,
    get_verbosity,
    log,
    save_bitlinear,
    set_verbosity,
)

@click.group()
def _generate():
    pass
@_generate.command()
@click.argument("n", type=int)
@click.argument("m", type=int)
@click.option("--output-path", type=click.Path(exists=False), default="random")
@click.option("--seed", type=int, default=None)
@click.option("--verbosity", default=get_verbosity(), help=f"Verbosity of the output (default: {get_verbosity()})")
def generate(n, m, seed, output_path, verbosity):
    #install_signal_handler()
    do_generate(n, m, seed, output_path, verbosity)

def do_generate(
        n,
        m,
        seed = None,
        output_path="random",
        verbosity=get_verbosity(),
    ):
    set_verbosity(verbosity)
    os.makedirs(output_path, exist_ok=True)
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    log(INFO, f"Generating random ternary matrix of {n} columns and {m} rows")
    a = torch.randint(-1, 1, (m, n))
    log(INFO, f"Saving random ternary matrix to {output_path}/random-{n}-{m}.bitlinear")
    save_bitlinear(
        f"{output_path}/random-{n}-{m}.bitlinear",
        a.tolist(),
        f"random ternary matrix of {n} columns and {m} rows",
    )
