#!/usr/bin/env python
import sys

from setuptools import find_packages, setup

try:
    from setuptools_rust import RustExtension
except ImportError:
    import subprocess

    errno = subprocess.call([sys.executable, "-m", "pip", "install", "setuptools-rust"])
    if errno:
        print("Please install setuptools-rust package")
        raise SystemExit(errno)
    else:
        from setuptools_rust import RustExtension

setup_requires = ["setuptools-rust>=0.10.1", "wheel", "setuptools_scm"]

setup(
    name="datacube_compute",
    setup_requires=setup_requires,
    author="Open Data Cube",
    author_email="",
    maintainer="Open Data Cube",
    description="Miscellaneous Algorithmic helper methods",
    long_description="",
    license="Apache License 2.0",
    tests_require=["pytest"],
    install_requires=["numpy", "xarray"],
    packages=find_packages(),
    zip_safe=False,
    rust_extensions=[RustExtension("datacube_compute.backend")],
)
