from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='genbmm',
    version="0.1",
    author="Alexander Rush",
    author_email="arush@cornell.edu",
    packages=["genbmm"],
    ext_modules=[
        CUDAExtension('_genbmm', [
            'matmul_cuda.cpp',
            'matmul_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
