import click
import random
import torch

from ..utils import (
    DETAIL,
    INFO,
    install_signal_handler,
    load_slp,
    get_verbosity,
    log,
    set_verbosity,
)

def show(n, program, a, language="py"):
    if language == "py":
        return show_python(n, program, a)
    if language == "c":
        return show_c(n, program, a)

def show_python(n, program, a):
    def variable(x):
        if x == 0:
            return "0"
        if x <= n:
            return f"x_{x-1}"
        return f"t_{x-n}"
    lines = []
    variables = ", ".join(f"x_{i}" for i in range(n))
    lines.append(f"def f({variables}):")
    for (x, y), z in program:
        if x < 0:
            lines.append(f"  {variable(z)} = {variable(y)} - {variable(-x)}")
        elif y < 0:
            lines.append(f"  {variable(z)} = {variable(x)} - {variable(-y)}")
        else:
            lines.append(f"  {variable(z)} = {variable(x)} + {variable(y)}")
    lines.append(f"  return {', '.join(variable(x) for x in a)}")
    return "\n".join(lines)

def show_c(n, program, a):
    def variable(x):
        if x == 0:
            return "0"
        if x <= n:
            return f"x[{x-1}]"
        return f"t_{x-n}"
    lines = []
    variables = "int *x, int *y"
    lines.append(f"void f({variables}) {{")
    for (x, y), z in program:
        if x < 0:
            lines.append(f"  int {variable(z)} = {variable(y)} - {variable(-x)};")
        elif y < 0:
            lines.append(f"  int {variable(z)} = {variable(x)} - {variable(-y)};")
        else:
            lines.append(f"  int {variable(z)} = {variable(x)} + {variable(y)};")
    for i, x in enumerate(a):
        lines.append(f"  y[{i}] = {variable(x)};")
    lines.append("}")
    return "\n".join(lines)

@click.group()
def _compile():
    pass
@_compile.command()
@click.argument("programs", type=click.Path(exists=True), nargs=-1)
@click.option("--language", type=click.Choice(["py", "c"]), default="py")
@click.option("--verbosity", default=get_verbosity(), help=f"Verbosity of the output (default: {get_verbosity()})")
def compile(programs, language, verbosity):
    #install_signal_handler()
    do_compile(programs, language, verbosity)

def do_compile(
        programs,
        language="py",
        verbosity=get_verbosity(),
    ):
    set_verbosity(verbosity)
    for program_name in programs:
        log(INFO, "Compiling program", program_name)
        n, program, a = load_slp(program_name)
        result = show(n, program, a, language=language)
        log(DETAIL, f"Result:", result)
        result_name = program_name.rsplit(".", maxsplit=1)[0] + "." + language
        log(INFO, "Saving result to", result_name)
        with open(result_name, "wt") as f:
            f.write(result)