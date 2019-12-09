from setuptools import setup
from torch.utils import cpp_extension

setup(name='custom_group_norm',
      ext_modules=[cpp_extension.CppExtension('custom_group_norm', ['custom_group_norm.cpp'],
                                              include_dirs = ['<PATH_TO_EIGEN_HEADER>'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
