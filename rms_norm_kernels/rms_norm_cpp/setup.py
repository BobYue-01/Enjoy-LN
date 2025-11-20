from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

project_root = os.path.abspath(os.path.dirname(__file__))

setup(
    name='rms_norm_cpp',
    ext_modules=[
        CppExtension(
            name='rms_norm_cpp',
            sources=[
                'csrc/rms_norm.cpp',
                'csrc/rms_norm_kernel.cpp'
            ],
            include_dirs=[
                os.path.join(project_root, "csrc/include")
            ]
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
