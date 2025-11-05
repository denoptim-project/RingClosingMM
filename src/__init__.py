"""
RingClosingMM - Ring Closure Optimizer

This package provides tools for molecular ring closure optimization using:
- Ring Closing Potential (RCP) for handling ring formation
- UFF van der Waals parameters for non-bonded interactions
- Internal coordinates (Z-matrix) for efficient torsional optimization
- Hybrid Genetic Algorithm for exploration of torsional space with post-processing local refinement
"""

__version__ = "1.0.0"
__author__ = "Marco Foscato"

# Import main classes for easier access
from .RingClosureOptimizer import RingClosureOptimizer
from .MolecularSystem import MolecularSystem
from .CoordinateConverter import CoordinateConverter

__all__ = [
    'RingClosureOptimizer',
    'MolecularSystem',
    'CoordinateConverter',
]

