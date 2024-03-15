from distutils.core import setup
from Cython.Build import cythonize
import numpy, cv2, os

setup(ext_modules=cythonize("unpatchify_mask.pyx"), include_dirs=[numpy.get_include()])
