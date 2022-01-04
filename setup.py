#!/usr/bin/env python3

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

import numpy


with open('README.rst') as file:
    long_description = file.read()


extensions = [
    Extension(
        "cochlea3.zilany2009._pycat",
        [
            "cochlea3/zilany2009/_pycat.pyx",
            "cochlea3/zilany2009/catmodel.c",
            "cochlea3/zilany2009/complex.c"
        ]
    ),
    Extension(
        "cochlea3.holmberg2007._traveling_waves",
        [
            "cochlea3/holmberg2007/_traveling_waves.pyx",
        ]
    ),
    Extension(
        "cochlea3.zilany2014.c_wrapper",
        [
            "cochlea3/zilany2014/c_wrapper.pyx",
            "cochlea3/zilany2014/model_IHC.c",
            "cochlea3/zilany2014/model_Synapse.c",
            "cochlea3/zilany2014/complex.c"
        ]
    ),
]


setup(
    name="cochlea3",
    version="1",
    author="Marek Rudnicki",
    author_email="mrkrd@posteo.de",

    description="Inner ear models for Python 3",
    license="GPLv3+",
    url="https://github.com/mrkrd/cochlea3",
    download_url="https://github.com/mrkrd/cochlea3/tarball/master",

    packages=find_packages(),
    package_data={
        "cochlea3.asr": ["*.csv"]
    },
    include_dirs=[numpy.get_include()],
    ext_modules=cythonize(extensions),
    long_description=long_description,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Cython",
        "Programming Language :: C",
    ],

    platforms=["Linux", "Windows", "FreeBSD", "OSX"],
    install_requires=["numpy", "scipy"],
)
