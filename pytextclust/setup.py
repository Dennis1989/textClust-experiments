import setuptools
from subprocess import CalledProcessError

from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import sys

__version__ = "0.0.1"

with open("README.md", "r") as fh:
    long_description = fh.read()


ext_modules = [
    Pybind11Extension("fastskip",
        ["yskip/yskip.cpp"],
        # Example: passing in the version to the compiled code
        define_macros = [('VERSION_INFO', __version__)],
        ),
]


setuptools.setup(
    name="textClustPy",
    version="0.0.1",
    author="Blinded",
    author_email="Blinded",
    description="A python implementation of textclust",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="blinded",
    packages=setuptools.find_packages(),
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=['nltk', 'pandas', 'gensim', 'numpy','sklearn', "tweepy==3.10.0", "elasticsearch", "jsonpickle", "pyemd", "pybind11"],
)