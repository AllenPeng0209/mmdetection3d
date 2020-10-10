from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension,CppExtension

setup(
    name='iou_cuda',
    ext_modules=[
        CUDAExtension('iou_cuda', [
            'src/iou_cuda.cpp',
            'src/iou_cuda_kernel.cu',
        ]),
    ],

    cmdclass={
        'build_ext': BuildExtension
    })

setup(
    name='bev_nms_cuda',
    ext_modules=[
        CUDAExtension('bev_nms_cuda', [
            'src/bev_nms.cpp',
            'src/bev_nms_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

setup(
    name='rotate_iou_cpp',
    ext_modules=[
        CppExtension('rotate_iou_cpp', ['src/rotate_iou.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
})
