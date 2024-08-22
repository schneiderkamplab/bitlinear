from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. The extension requires CUDA.")

setup(
    name='pack_weights',
    ext_modules=[
        CUDAExtension(
            name='pack_weights',
            sources=['pack_weights.cu']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
