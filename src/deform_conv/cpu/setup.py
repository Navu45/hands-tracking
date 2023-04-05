from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='conv',
      ext_modules=[cpp_extension.CppExtension('conv', ['conv.cpp'], extra_compile_args=['-g'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})