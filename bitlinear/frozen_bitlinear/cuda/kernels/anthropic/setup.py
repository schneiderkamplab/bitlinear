from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. The extension requires CUDA.")

setup(
    name='naive',
    ext_modules=[
        CUDAExtension(
            name='naive',
            sources=['files/naive.cu']
        ),
        CUDAExtension(
            name='bitlinear_naive',
            sources=['files/bitlinear_naive.cu']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
