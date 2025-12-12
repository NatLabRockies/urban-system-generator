"""
Setup script for Urban System Generator
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Urban System Generator - ML-based building attribute completion"

setup(
    name="urban-system-generator",
    version="0.1.0",
    author="Rawad El Kontar",
    author_email="rawad.elkontar@nrel.gov",
    maintainer="Rawad El Kontar",
    maintainer_email="rawad.elkontar@nrel.gov",
    description="Machine learning tool for completing building attributes in urban energy models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NREL/urban-system-generator",
    packages=find_packages(exclude=["tests", "examples", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "tensorflow>=2.10.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    include_package_data=True,
    package_data={
        "usg": ["*.py"],
    },
)