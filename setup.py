#!/usr/bin/env python3
"""
Setup script for RingClosingMM - Ring Closure Optimizer.

Installation:
    pip install -e .                    # Development install
    pip install .                       # Regular install
    
Usage after installation:
    rc-optimizer -i test.int -r bonds.txt -c rcp.txt
    python -m src -i test.int -r bonds.txt -c rcp.txt
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding='utf-8') if readme_file.exists() else ""

# Read requirements if exists
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    requirements = [line.strip() for line in requirements_file.read_text().splitlines() 
                   if line.strip() and not line.startswith('#')]
else:
    # Fallback: specify requirements directly
    requirements = [
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'openmm>=7.7.0',
    ]

setup(
    name="ringclosingmm",
    version="1.0.0",
    description="Ring closure optimization using hybrid genetic algorithm with OpenMM UFFvdW force field",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    author="Marco Foscato",
    author_email="marco.foscato@uib.no",  # Update with actual email
    
    url="https://github.com/your-username/RingClosingMM",  # Update with actual URL
    
    # Package configuration
    packages=find_packages(include=['src', 'src.*']),
    package_dir={'': '.'},
    
    # Include package data
    # Data files are included via MANIFEST.in (recursive-include data *.xml)
    # and will be available in the source directory structure
    include_package_data=True,
    
    # Also install data files to a known location for easy access
    data_files=[
        ('share/ringclosingmm/data', ['data/RCP_UFFvdW.xml']),
    ],
    
    # Dependencies
    install_requires=requirements,
    
    python_requires='>=3.8',
    
    # Entry points for command-line tools
    entry_points={
        'console_scripts': [
            'rc-optimizer=src.__main__:main',
        ],
    },
    
    # Classifiers for PyPI
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
    
    # Keywords for PyPI
    keywords='conformational-search torsional-optimization potential-energy-smoothing genetic-algorithm openmm chemistry',
    
    # Project URLs
    project_urls={
        'Bug Reports': 'https://github.com/your-username/RingClosingMM/issues',
        'Source': 'https://github.com/your-username/RingClosingMM',
        'Documentation': 'https://github.com/your-username/RingClosingMM/blob/main/README.md',
    },
)

