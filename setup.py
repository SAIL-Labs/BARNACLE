#!/usr/bin/env python

from setuptools import find_packages, setup

project_name = "barnacle"

setup(
    name=project_name,
    version="2",
    packages=find_packages(),
    author='Marc-Antoine Martinod',
    author_email='ma.martinod@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Professional Astronomers',
        'Topic :: High Angular Resolution Astronomy :: Interferometry :: \
            Nulling Interferometry',
        'Programming Language :: Python :: 3.8'
    ],
    install_requires=["astropy", "cupy", "functools", "h5py", "matplotlib", "numba",
                      "numpy", "scipy"]
)
