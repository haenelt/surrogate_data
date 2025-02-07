# import Cython.Compiler.Options
import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

ext = Extension(
    "variogram",
    ["surrogate_data/variogram.pyx"],
    include_dirs=[np.get_include()],
    libraries=["c"],
)
setup(
    ext_modules=cythonize([ext], annotate=True),
    script_args=["build_ext", "--inplace", "--build-lib", "surrogate_data"],
)
