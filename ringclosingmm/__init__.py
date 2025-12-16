"""
RingClosingMM 

This package provides tools for molecular ring closing optimization using:
- Ring Closing Potential (RCP) for handling ring formation
- UFF van der Waals parameters for non-bonded interactions
- Internal coordinates (Z-matrix) for efficient torsional optimization
"""

try:
    from ._version import version as __version__
except ImportError:
    # Fallback for when _version.py doesn't exist (e.g., installed from sdist)
    __version__ = "0.0.0.dev0"

__author__ = "Marco Foscato"

# Import main classes for easier access
from .RingClosureOptimizer import RingClosureOptimizer
from .MolecularSystem import MolecularSystem
from . import IOTools
from .IOTools import read_int_file, read_sdf_file, write_zmatrix_file, write_xyz_file, save_structure_to_file
from .ZMatrix import ZMatrix

# Provide convenient aliases
RCOptimizer = RingClosureOptimizer

__all__ = [
    'RingClosureOptimizer',
    'RCOptimizer',
    'MolecularSystem',
    'IOTools',
    'read_int_file',
    'read_sdf_file',
    'write_zmatrix_file',
    'write_xyz_file',
    'save_structure_to_file',
    'ZMatrix',
]

