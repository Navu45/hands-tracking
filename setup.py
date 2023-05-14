import fnmatch
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as build_py_orig

excluded = ['feature_extractor/test_inference.py',
            'feature_extractor/test.py',
            'feature_pyramid/test_inference.py']


class build_py(build_py_orig):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [
            (pkg, mod, file)
            for (pkg, mod, file) in modules
            if not any(fnmatch.fnmatchcase(file, pat=pattern) for pattern in excluded)
        ]


setup(
    name='hand_detector',
    version='1.0.0',
    author='Alexey Yamoncheryaev',
    description='MogaNet + SepAttn Transformer Efficient-RepGFPN',
    packages=find_packages(),
    cmdclass={'build_py': build_py}
)
