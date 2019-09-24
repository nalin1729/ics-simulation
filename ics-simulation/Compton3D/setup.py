from distutils.core import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("compton3D.pyx")
)