from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_cuda_conv',
    ext_modules=[
        CUDAExtension(
            name='my_cuda_conv',
            sources=[
                'conv2d_bind.cpp',
                'conv2d_naive.cu',
            ],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
