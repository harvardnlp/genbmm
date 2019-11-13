from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='genmatmul',
    version="0.1",
    author="Alexander Rush",
    author_email="arush@cornell.edu",
    packages=["genmatmul"],
    ext_modules=[
        CUDAExtension('struct_lib', [
            'matmul_cuda.cpp',
            'matmul_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
