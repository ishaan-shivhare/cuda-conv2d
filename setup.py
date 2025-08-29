from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import pybind11
import os

setup(
    name='my_cuda_conv',
    ext_modules=[
        CUDAExtension(
            name='my_cuda_conv',
            sources=[
                'conv2d_bind.cpp',
                # 'conv2d_naive.cu',
                'conv2d_tma_3d.cu'
            ],
            include_dirs = [
                pybind11.get_include(),  # This adds the correct include path
            ],
            extra_compile_args = {
                'nvcc': [
                    '-gencode=arch=compute_90a,code=sm_90a',
                    '-lineinfo'
                ]
            },
            extra_link_args = [
                '-lcuda'
            ]
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
