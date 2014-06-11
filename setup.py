import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_module = Extension(
    '_cython',
    ['_cython.pyx'],
    include_dirs=[np.get_include()],
    extra_compile_args=['/openmp'],
)

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [ext_module],
)