from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bev_nms_cuda',
    ext_modules=[
        CUDAExtension('bev_nms_cuda', [
            'bev_nms.cpp',
            'bev_nms_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })