from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. The extension requires CUDA.")

setup(
    name='warptiling',
    ext_modules=[
        CUDAExtension(
            name='warptiling',
            sources=['runner.cu']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
