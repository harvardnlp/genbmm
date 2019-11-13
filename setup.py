from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='struct_lib',
    ext_modules=[
        CUDAExtension('struct_lib', [
            'matmul_cuda.cpp',
            'matmul_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
