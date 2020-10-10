from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='rotate_iou_cpp',
    ext_modules=[
        CppExtension('rotate_iou_cpp', ['rotate_iou.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
})