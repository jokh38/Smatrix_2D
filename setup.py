"""Setup script for smatrix_2d package."""

from setuptools import setup, find_packages

setup(
    name='smatrix_2d',
    version='7.2',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.9.0',
        'matplotlib>=3.3.0',
    ],
)
