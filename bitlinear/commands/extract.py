import click
import os
import safetensors
import torch

from ..bitlinear import AbsMedian, round_clamp, scale
from ..utils import (
    INFO,
    install_signal_handler,
    get_verbosity,
    log,
    save_bitlinear,
    set_verbosity,
)

@click.group()
def _extract():
    pass
@_extract.command()
@click.argument("model-files", type=click.Path(exists=True), nargs=-1)
@click.option("--include", type=str, default=[], help="Include only layers with this substring in their name", multiple=True)
@click.option("--exclude", type=str, default=[], help="Exclude layers with this substring in their name", multiple=True)
@click.option("--output-path", type=click.Path(exists=False), default="output")
@click.option("--verbosity", default=get_verbosity(), help=f"Verbosity of the output (default: {get_verbosity()})")
def extract(model_files, include, exclude, output_path, verbosity):
    #install_signal_handler()
    do_extract(model_files, include, exclude, output_path, verbosity)

def do_extract(
        model_files,
        include=[],
        exclude=[],
        output_path="output",
        verbosity=get_verbosity(),
    ):
    set_verbosity(verbosity)
    os.makedirs(output_path, exist_ok=True)
    for model_file in model_files:
        log(INFO, "Extracting matrices from", model_file)
        td = safetensors.safe_open("model.safetensors", "pt")
        for k in td.keys():
            if (not include or any(x in k for x in include)) and not any(x in k for x in exclude):
                w = td.get_tensor(k)
                if w.dim() != 2:
                    log(INFO, f"Skipping {k} with shape {w.shape}")
                    continue
                log(INFO, "Quantizing", k)
                w_scale = scale(w, (-1, 1), AbsMedian(), False, 1e-5)
                w_quant = round_clamp(w * w_scale, (-1, 1)).to(torch.int8)
                log(INFO, f"Saving {k} to {output_path}/{k}.bitlinear")
                save_bitlinear(f"{output_path}/{k}.bitlinear", w_quant.tolist(), f"{k} quantized to 1.58-bit using AbsMedian quantization")
