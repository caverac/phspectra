"""Build configuration for phspectra C extension."""

import numpy as np
from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "phspectra._gaussfit",
            sources=["src/phspectra/_gaussfit.c"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3", "-std=c99"],
        ),
    ],
)
