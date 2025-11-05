#!/usr/bin/env python3
"""
Unit tests for RingClosingForceField module.

Tests geometry calculation functions.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from RingClosingForceField import (
    getDistance,
    getAngle,
    getUnitVector
)


class TestRingClosingForceFieldGeometry(unittest.TestCase):
    """Test geometry calculation functions."""
    
    def test_getDistance(self):
        """Test distance calculation."""
        from openmm import unit
        
        p1 = np.array([0.0, 0.0, 0.0]) * unit.angstroms
        p2 = np.array([1.0, 0.0, 0.0]) * unit.angstroms
        
        distance = getDistance(p1, p2)
        
        self.assertAlmostEqual(distance, 1.0, places=5)
    
    def test_getDistance_diagonal(self):
        """Test distance calculation for diagonal."""
        from openmm import unit
        
        p1 = np.array([0.0, 0.0, 0.0]) * unit.angstroms
        p2 = np.array([1.0, 1.0, 1.0]) * unit.angstroms
        
        distance = getDistance(p1, p2)
        expected = np.sqrt(3.0)
        
        self.assertAlmostEqual(distance, expected, places=5)
    
    def test_getAngle_right_angle(self):
        """Test angle calculation for right angle."""
        from openmm import unit
        import math
        
        # For 90-degree angle: p1 at vertex, p0 and p2 perpendicular
        # Vectors: p0-p1 should be perpendicular to p2-p1
        p0 = np.array([-1.0, 0.0, 0.0]) * unit.angstroms  # Point on -x axis
        p1 = np.array([0.0, 0.0, 0.0]) * unit.angstroms   # Vertex at origin
        p2 = np.array([0.0, 1.0, 0.0]) * unit.angstroms   # Point on +y axis
        
        angle = getAngle(p0, p1, p2)
        
        # Right angle in radians = π/2 (returns radians, not degrees)
        # Vectors: p0-p1 = [-1,0,0], p2-p1 = [0,1,0] -> perpendicular
        self.assertAlmostEqual(angle, math.pi / 2, places=3)
    
    def test_getAngle_straight_line(self):
        """Test angle calculation for straight line."""
        from openmm import unit
        import math
        
        p0 = np.array([0.0, 0.0, 0.0]) * unit.angstroms
        p1 = np.array([1.0, 0.0, 0.0]) * unit.angstroms
        p2 = np.array([2.0, 0.0, 0.0]) * unit.angstroms
        
        angle = getAngle(p0, p1, p2)
        
        # Straight line angle in radians = π (returns radians, not degrees)
        self.assertAlmostEqual(angle, math.pi, places=3)
    
    def test_getUnitVector(self):
        """Test unit vector calculation."""
        from openmm import unit
        
        vector = np.array([3.0, 4.0, 0.0]) * unit.angstroms
        unit_vec = getUnitVector(vector)
        
        # Check magnitude is approximately 1
        magnitude = np.linalg.norm([unit_vec._value[0], unit_vec._value[1], unit_vec._value[2]])
        self.assertAlmostEqual(magnitude, 1.0, places=5)
    
    def test_getUnitVector_direction(self):
        """Test unit vector preserves direction."""
        from openmm import unit
        
        vector = np.array([1.0, 0.0, 0.0]) * unit.angstroms
        unit_vec = getUnitVector(vector)
        
        # Should point in same direction
        self.assertAlmostEqual(unit_vec._value[0], 1.0, places=5)
        self.assertAlmostEqual(unit_vec._value[1], 0.0, places=5)
        self.assertAlmostEqual(unit_vec._value[2], 0.0, places=5)


def run_tests(verbosity=2):
    """Run all tests with specified verbosity."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRingClosingForceFieldGeometry))
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests(verbosity=2)
    sys.exit(0 if success else 1)

