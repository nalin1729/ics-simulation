# -------------------------------------------------------------------------------
from Cython.Distutils import build_ext
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("execute.pyx")
)