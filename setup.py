from setuptools import Extension, setup

from Cython.Build import cythonize

import numpy as np

setup(
        name = 'Posits4TorcHA',
        author = 'Gabriel Vitor Klaumann Gubert',
        author_email = 'gvkg97@gmail.com',
        maintainer = 'Gabriel Vitor Klaumann Gubert',
        maintainer_email = 'gvkg97@gmail.com',
        ext_modules = cythonize(
            [
                Extension(
                    name = 'Posits4TorcHA', 
                    sources = [
                        'src/Posits4TorcHA.py'
                    ], 
                    include_dirs = [
                        np.get_include()
                    ],
                    extra_compile_args = ['-fopenmp'],
                    extra_link_args = ['-fopenmp']
                )   
            ]
        ),
    )   
