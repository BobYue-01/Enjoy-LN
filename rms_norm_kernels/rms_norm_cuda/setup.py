from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

project_root = os.path.abspath(os.path.dirname(__file__))

setup(
    name='rms_norm_cuda',
    ext_modules=[
        CUDAExtension(
            name='rms_norm_cuda',
            sources=[
                'csrc/rms_norm_cuda.cpp',
                'csrc/rms_norm_cuda_kernel.cu'
            ],
            include_dirs=[
                os.path.join(project_root, "csrc/include")
            ]
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
