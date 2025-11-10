#!/usr/bin/env python3
"""
Unit tests for coordinate conversion cycle: INT → XYZ → INT → XYZ

Tests the stability and consistency of conversions between Z-matrix (internal)
and Cartesian coordinates.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from CoordinateConverter import (
    zmatrix_to_cartesian,
    cartesian_to_zmatrix,
    apply_torsions,
    extract_torsions,
    _calc_distance,
    _calc_angle,
    _calc_dihedral
)


class TestGeometryCalculations(unittest.TestCase):
    """Test basic geometry calculation functions."""
    
    def test_calc_distance(self):
        """Test distance calculation."""
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(_calc_distance(p1, p2), 1.0, places=6)
        
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([3.0, 4.0, 0.0])
        self.assertAlmostEqual(_calc_distance(p1, p2), 5.0, places=6)
    
    def test_calc_angle(self):
        """Test angle calculation."""
        # 90 degree angle
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(_calc_angle(p1, p2, p3), 90.0, places=4)
        
        # 180 degree angle (linear)
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([-1.0, 0.0, 0.0])
        self.assertAlmostEqual(_calc_angle(p1, p2, p3), 180.0, places=4)
        
        # 60 degree angle
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([0.5, np.sqrt(3)/2, 0.0])
        self.assertAlmostEqual(_calc_angle(p1, p2, p3), 60.0, places=4)
    
    def test_calc_dihedral(self):
        """Test dihedral angle calculation."""
        # 0 degree dihedral (all in same plane)
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([2.0, 0.0, 0.0])
        p4 = np.array([3.0, 0.0, 0.0])
        dihedral = _calc_dihedral(p1, p2, p3, p4)
        self.assertAlmostEqual(dihedral, 0.0, places=4)

        # 90 degree dihedral (perpendicular plane)
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([0.0, 1.0, 0.0])
        p4 = np.array([0.0, 1.0, 1.0])
        dihedral = _calc_dihedral(p1, p2, p3, p4)
        self.assertAlmostEqual(dihedral, 90.0, places=4)
        dihedral = _calc_dihedral(p4, p3, p2, p1)
        self.assertAlmostEqual(dihedral, 90.0, places=4)
        dihedral = _calc_dihedral(p4, p1, p2, p3)
        self.assertAlmostEqual(dihedral, -45.0, places=4)

        # a degree dihedral 
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([0.0, 1.0, 0.0])
        p4 = np.array([1.0, 1.0, 1.0])
        dihedral = _calc_dihedral(p4, p3, p2, p1)
        self.assertAlmostEqual(dihedral, 125.26, places=2)
        dihedral = _calc_dihedral(p4, p1, p2, p3)
        self.assertAlmostEqual(dihedral, -45.0, places=4)

        # flat 
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([0.0, 1.0, 0.0])
        p4 = np.array([1.0, 1.0, 0.0])
        dihedral = _calc_dihedral(p4, p3, p2, p1)
        self.assertAlmostEqual(abs(dihedral), 180.0, places=2)
        dihedral = _calc_dihedral(p4, p1, p2, p3)
        self.assertAlmostEqual(abs(dihedral), 0.0, places=4)

        # abother degree dihedral 
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([0.0, 1.0, 0.0])
        p4 = np.array([1.0, 1.0, -1.0])
        dihedral = _calc_dihedral(p4, p3, p2, p1)
        self.assertAlmostEqual(dihedral, -125.26, places=2)
        dihedral = _calc_dihedral(p4, p1, p2, p3)
        self.assertAlmostEqual(dihedral, 45.0, places=4)



class TestSimpleZMatrix(unittest.TestCase):
    """Test with simple Z-matrix structures."""
    
    def setUp(self):
        """Create simple test Z-matrices."""
        # Linear molecule: H-H-H (0-based indices)
        self.linear_zmatrix = [
            {'id': 0, 'element': 'H', 'atomic_num': 1},
            {'id': 1, 'element': 'H', 'atomic_num': 1, 'bond_ref': 0, 'bond_length': 1.0},
            {'id': 2, 'element': 'H', 'atomic_num': 1, 'bond_ref': 1, 'bond_length': 1.0,
             'angle_ref': 0, 'angle': 180.0},
        ]
        
        # Bent molecule: H-O-H (water-like) (0-based indices)
        self.bent_zmatrix = [
            {'id': 0, 'element': 'O', 'atomic_num': 8},
            {'id': 1, 'element': 'H', 'atomic_num': 1, 'bond_ref': 0, 'bond_length': 0.96},
            {'id': 2, 'element': 'H', 'atomic_num': 1, 'bond_ref': 0, 'bond_length': 0.96,
             'angle_ref': 1, 'angle': 104.5},
        ]
        
        # Tetrahedral: C with 4 H (methane-like) (0-based indices)
        self.tetrahedral_zmatrix = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'H', 'atomic_num': 1, 'bond_ref': 0, 'bond_length': 1.09},
            {'id': 2, 'element': 'H', 'atomic_num': 1, 'bond_ref': 0, 'bond_length': 1.09,
             'angle_ref': 1, 'angle': 109.47},
            {'id': 3, 'element': 'H', 'atomic_num': 1, 'bond_ref': 0, 'bond_length': 1.09,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 2, 'dihedral': 120.0, 'chirality': 0},
            {'id': 4, 'element': 'H', 'atomic_num': 1, 'bond_ref': 0, 'bond_length': 1.09,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 2, 'dihedral': -120.0, 'chirality': 0},
        ]
    
    def test_linear_conversion(self):
        """Test that linear molecules raise ValueError."""
        # Linear geometries (180° angle) should raise ValueError
        with self.assertRaises(ValueError) as cm:
            zmatrix_to_cartesian(self.linear_zmatrix)
        
        # Check error message mentions linearity
        self.assertIn("linear", str(cm.exception).lower())
    
    def test_bent_conversion(self):
        """Test conversion of bent molecule."""
        coords = zmatrix_to_cartesian(self.bent_zmatrix)
        
        # Check number of atoms
        self.assertEqual(len(coords), 3)
        
        # Check distances
        d12 = _calc_distance(coords[0], coords[1])
        d13 = _calc_distance(coords[0], coords[2])
        self.assertAlmostEqual(d12, 0.96, places=5)
        self.assertAlmostEqual(d13, 0.96, places=5)
        
        # Check angle
        angle = _calc_angle(coords[1], coords[0], coords[2])
        self.assertAlmostEqual(angle, 104.5, places=3)
    
    def test_tetrahedral_conversion(self):
        """Test conversion of tetrahedral molecule."""
        coords = zmatrix_to_cartesian(self.tetrahedral_zmatrix)
        
        # Check number of atoms
        self.assertEqual(len(coords), 5)
        
        # Check all C-H distances
        for i in range(1, 5):
            d = _calc_distance(coords[0], coords[i])
            self.assertAlmostEqual(d, 1.09, places=4)
        
        # Check H-C-H angles (should all be ~109.47 degrees)
        angle_12_13 = _calc_angle(coords[1], coords[0], coords[2])
        self.assertAlmostEqual(angle_12_13, 109.47, places=2)


class TestConversionCycle(unittest.TestCase):
    """Test INT → XYZ → INT → XYZ conversion cycle."""
    
    def setUp(self):
        """Create test Z-matrices with various features."""
        # Simple 4-atom chain with dihedral (0-based indices)
        self.simple_chain = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 110.0},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 110.0, 'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0},
        ]
        
        # Mixed structure with both chiral and non-chiral atoms (0-based indices)
        self.mixed_chirality = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 110.0},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 110.0, 'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0},
            {'id': 4, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 110.0, 'dihedral_ref': 2, 'dihedral': 109.5, 'chirality': 1},
        ]
    
    def test_single_cycle(self):
        """Test one complete conversion cycle: INT → XYZ → INT."""
        # Convert to Cartesian
        coords1 = zmatrix_to_cartesian(self.simple_chain)
        
        # Extract back to Z-matrix
        zmatrix2 = cartesian_to_zmatrix(coords1, self.simple_chain)
        
        # Check bond lengths are preserved
        for i in range(1, len(self.simple_chain)):
            original = self.simple_chain[i]['bond_length']
            extracted = zmatrix2[i]['bond_length']
            self.assertAlmostEqual(original, extracted, places=5,
                                 msg=f"Bond length mismatch at atom {i}")
        
        # Check angles are preserved
        for i in range(2, len(self.simple_chain)):
            original = self.simple_chain[i]['angle']
            extracted = zmatrix2[i]['angle']
            self.assertAlmostEqual(original, extracted, places=3,
                                 msg=f"Angle mismatch at atom {i}")
        
        # Check dihedrals are preserved (for non-chiral atoms)
        for i in range(3, len(self.simple_chain)):
            if self.simple_chain[i].get('chirality', 0) == 0:
                original = self.simple_chain[i]['dihedral']
                extracted = zmatrix2[i]['dihedral']
                self.assertAlmostEqual(original, extracted, places=3,
                                     msg=f"Dihedral mismatch at atom {i}")
        
        # Check chirality flags are preserved
        for i in range(3, len(self.simple_chain)):
            original_chirality = self.simple_chain[i].get('chirality', 0)
            extracted_chirality = zmatrix2[i].get('chirality', 0)
            self.assertEqual(original_chirality, extracted_chirality,
                           msg=f"Chirality flag mismatch at atom {i}: "
                               f"original={original_chirality}, extracted={extracted_chirality}")
    
    def test_double_cycle(self):
        """Test two complete conversion cycles: INT → XYZ → INT → XYZ → INT."""
        # First cycle
        coords1 = zmatrix_to_cartesian(self.simple_chain)
        zmatrix2 = cartesian_to_zmatrix(coords1, self.simple_chain)
        
        # Second cycle
        coords2 = zmatrix_to_cartesian(zmatrix2)
        zmatrix3 = cartesian_to_zmatrix(coords2, zmatrix2)
        
        # Check that second cycle produces identical results
        for i in range(len(coords1)):
            np.testing.assert_array_almost_equal(
                coords1[i], coords2[i], decimal=5,
                err_msg=f"Cartesian coordinates differ at atom {i} after second cycle"
            )
        
        # Check internal coordinates
        for i in range(1, len(self.simple_chain)):
            self.assertAlmostEqual(
                zmatrix2[i]['bond_length'],
                zmatrix3[i]['bond_length'],
                places=5,
                msg=f"Bond length differs at atom {i} after second cycle"
            )
        
        for i in range(2, len(self.simple_chain)):
            self.assertAlmostEqual(
                zmatrix2[i]['angle'],
                zmatrix3[i]['angle'],
                places=3,
                msg=f"Angle differs at atom {i} after second cycle"
            )
        
        # Check chirality flags remain consistent
        for i in range(3, len(self.simple_chain)):
            self.assertEqual(
                zmatrix2[i].get('chirality', 0),
                zmatrix3[i].get('chirality', 0),
                msg=f"Chirality flag differs at atom {i} after second cycle"
            )
    
    def test_cartesian_stability(self):
        """Test that Cartesian coordinates remain stable through cycles."""
        coords1 = zmatrix_to_cartesian(self.simple_chain)
        
        # Do 5 cycles
        coords_current = coords1
        for cycle in range(5):
            zmatrix_temp = cartesian_to_zmatrix(coords_current, self.simple_chain)
            coords_current = zmatrix_to_cartesian(zmatrix_temp)
            
            # Check stability
            for i in range(len(coords1)):
                np.testing.assert_array_almost_equal(
                    coords1[i], coords_current[i], decimal=4,
                    err_msg=f"Coordinates drifted at atom {i} after {cycle+1} cycles"
                )
    
    def test_mixed_chirality_cycle(self):
        """Test conversion cycle with mixed chiral and non-chiral atoms."""
        # Convert to Cartesian
        coords1 = zmatrix_to_cartesian(self.mixed_chirality)
        
        # Extract back to Z-matrix
        zmatrix2 = cartesian_to_zmatrix(coords1, self.mixed_chirality)
        
        # Verify all chirality flags
        for i in range(3, len(self.mixed_chirality)):
            original_chirality = self.mixed_chirality[i].get('chirality', 0)
            extracted_chirality = zmatrix2[i].get('chirality', 0)
            self.assertEqual(original_chirality, extracted_chirality,
                           msg=f"Chirality flag mismatch at atom {i}: "
                               f"original={original_chirality}, extracted={extracted_chirality}")
        
        # Specifically check the chiral atom (index 4)
        self.assertEqual(zmatrix2[4]['chirality'], 1,
                        msg="Chiral atom should have chirality=1")
        
        # Check the non-chiral atom (index 3)
        self.assertEqual(zmatrix2[3]['chirality'], 0,
                        msg="Non-chiral atom should have chirality=0")
        
        # Verify coordinates are stable
        coords2 = zmatrix_to_cartesian(zmatrix2)
        np.testing.assert_array_almost_equal(
            coords1, coords2, decimal=4,
            err_msg="Coordinates changed after cycle with mixed chirality"
        )


class TestChiralityHandling(unittest.TestCase):
    """Test Z-matrix with chirality flags."""
    
    def setUp(self):
        """Create test Z-matrix with chiral center."""
        # Simple branched structure with chirality (0-based indices)
        self.chiral_zmatrix = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 110.0},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 110.0, 'dihedral_ref': 2, 'dihedral': 120.0, 'chirality': 1},
        ]
    
    def test_chirality_preservation(self):
        """Test that chirality sign is preserved through conversion cycle."""
        # Convert to Cartesian
        coords1 = zmatrix_to_cartesian(self.chiral_zmatrix)
        
        # Extract back to Z-matrix
        zmatrix2 = cartesian_to_zmatrix(coords1, self.chiral_zmatrix)
        
        # Check chirality flag is preserved
        self.assertEqual(
            self.chiral_zmatrix[3]['chirality'],
            zmatrix2[3]['chirality'],
            msg="Chirality sign not preserved"
        )
        
        # Check bond angle (stored in dihedral field for chiral atoms)
        self.assertAlmostEqual(
            self.chiral_zmatrix[3]['dihedral'],
            zmatrix2[3]['dihedral'],
            places=3,
            msg="Bond angle not preserved for chiral atom"
        )
    
    def test_chirality_cycle_stability(self):
        """Test chirality through multiple cycles."""
        coords1 = zmatrix_to_cartesian(self.chiral_zmatrix)
        
        for cycle in range(3):
            zmatrix_temp = cartesian_to_zmatrix(coords1, self.chiral_zmatrix)
            coords_temp = zmatrix_to_cartesian(zmatrix_temp)
            
            # Coordinates should remain stable
            np.testing.assert_array_almost_equal(
                coords1, coords_temp, decimal=4,
                err_msg=f"Coordinates changed after cycle {cycle+1}"
            )
            
            # Chirality should remain the same
            self.assertEqual(
                self.chiral_zmatrix[3]['chirality'],
                zmatrix_temp[3]['chirality'],
                msg=f"Chirality changed after cycle {cycle+1}"
            )


class TestCartesianToZMatrix(unittest.TestCase):
    """Test cartesian_to_zmatrix function in isolation."""
    
    def test_simple_bond_extraction(self):
        """Test extraction of simple bond length."""
        # Create known Cartesian coordinates
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0]  # 1.5 Å along X axis
        ])
        
        # Template Z-matrix (0-based indices)
        zmatrix = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 0.0}
        ]
        
        # Extract Z-matrix
        result = cartesian_to_zmatrix(coords, zmatrix)
        
        # Check bond length was correctly calculated
        self.assertAlmostEqual(result[1]['bond_length'], 1.5, places=6)
    
    def test_angle_extraction(self):
        """Test extraction of bond angle."""
        # Create known Cartesian coordinates forming 90° angle
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0]
        ])
        
        # Template Z-matrix (0-based indices)
        zmatrix = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 0.0},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 0.0,
             'angle_ref': 0, 'angle': 0.0}
        ]
        
        # Extract Z-matrix
        result = cartesian_to_zmatrix(coords, zmatrix)
        
        # Check angle was correctly calculated
        self.assertAlmostEqual(result[2]['angle'], 90.0, places=4)
        self.assertAlmostEqual(result[2]['bond_length'], 1.0, places=6)
    
    def test_dihedral_extraction(self):
        """Test extraction of dihedral angle."""
        # Create known Cartesian coordinates with specific dihedral
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.5, 1.0, 0.0],
            [2.5, 1.0, 1.0]
        ])
        
        # Template Z-matrix
        zmatrix = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 0.0},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 0.0,
             'angle_ref': 0, 'angle': 0.0},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 0.0,
             'angle_ref': 1, 'angle': 0.0, 'dihedral_ref': 0, 'dihedral': 0.0, 'chirality': 0}
        ]
        
        # Extract Z-matrix
        result = cartesian_to_zmatrix(coords, zmatrix)
        
        # Calculate expected dihedral manually
        expected_dihedral = _calc_dihedral(coords[3], coords[2], coords[1], coords[0])
        
        # Check dihedral was correctly calculated
        self.assertAlmostEqual(result[3]['dihedral'], expected_dihedral, places=4)
    
    def test_linear_molecule_extraction(self):
        """Test extraction from linear molecule."""
        # Linear chain along Z axis
        coords = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0]
        ])
        
        zmatrix = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 0.0},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 0.0,
             'angle_ref': 0, 'angle': 0.0}
        ]
        
        result = cartesian_to_zmatrix(coords, zmatrix)
        
        # Check bond lengths
        self.assertAlmostEqual(result[1]['bond_length'], 1.0, places=6)
        self.assertAlmostEqual(result[2]['bond_length'], 1.0, places=6)
        
        # Check angle (should be ~180° for linear)
        self.assertAlmostEqual(result[2]['angle'], 180.0, places=3)
    
    def test_chiral_center_extraction(self):
        """Test extraction of chirality information."""
        # Create a chiral center
        zmatrix = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 110.0},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 110.0, 'dihedral_ref': 2, 'dihedral': 120.0, 'chirality': 1}
        ]
        
        # Convert to Cartesian with known chirality
        coords = zmatrix_to_cartesian(zmatrix)
        
        # Extract back
        result = cartesian_to_zmatrix(coords, zmatrix)
        
        # Chirality sign should be preserved
        self.assertEqual(result[3]['chirality'], 1)
        
        # Bond angle should be preserved
        self.assertAlmostEqual(result[3]['dihedral'], 120.0, places=3)
    
    def test_negative_chirality_extraction(self):
        """Test extraction of negative chirality."""
        zmatrix = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 110.0},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 110.0, 'dihedral_ref': 2, 'dihedral': 120.0, 'chirality': -1}
        ]
        
        coords = zmatrix_to_cartesian(zmatrix)
        result = cartesian_to_zmatrix(coords, zmatrix)
        
        # Negative chirality should be preserved
        self.assertEqual(result[3]['chirality'], -1)
    
    def test_reference_atoms_preserved(self):
        """Test that reference atom indices are preserved."""
        zmatrix = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 110.0},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 110.0, 'dihedral_ref': 2, 'dihedral': 60.0, 'chirality': 0}
        ]
        
        coords = zmatrix_to_cartesian(zmatrix)
        result = cartesian_to_zmatrix(coords, zmatrix)
        
        # Check all reference atoms are preserved (0-based indices)
        self.assertEqual(result[1]['bond_ref'], 0)
        self.assertEqual(result[2]['bond_ref'], 0)
        self.assertEqual(result[2]['angle_ref'], 1)
        self.assertEqual(result[3]['bond_ref'], 1)
        self.assertEqual(result[3]['angle_ref'], 0)
        self.assertEqual(result[3]['dihedral_ref'], 2)
    
    def test_edge_case_very_small_angle(self):
        """Test extraction with very small angle."""
        # Create near-linear arrangement
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.01, 0.0]  # Nearly linear
        ])
        
        zmatrix = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 0.0},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 0.0,
             'angle_ref': 0, 'angle': 0.0}
        ]
        
        result = cartesian_to_zmatrix(coords, zmatrix)
        
        # Angle should be close to 180°
        self.assertGreater(result[2]['angle'], 179.0)
        self.assertLess(result[2]['angle'], 181.0)
    
    def test_multiple_atoms_extraction(self):
        """Test extraction with larger molecule."""
        # Create a chain of 6 atoms (0-based indices)
        zmatrix = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 110.0},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 110.0, 'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0},
            {'id': 4, 'element': 'C', 'atomic_num': 6, 'bond_ref': 3, 'bond_length': 1.54,
             'angle_ref': 2, 'angle': 110.0, 'dihedral_ref': 1, 'dihedral': -60.0, 'chirality': 0},
            {'id': 5, 'element': 'C', 'atomic_num': 6, 'bond_ref': 4, 'bond_length': 1.54,
             'angle_ref': 3, 'angle': 110.0, 'dihedral_ref': 2, 'dihedral': 180.0, 'chirality': 0}
        ]
        
        coords = zmatrix_to_cartesian(zmatrix)
        result = cartesian_to_zmatrix(coords, zmatrix)
        
        # Check all internal coordinates are preserved
        for i in range(1, len(zmatrix)):
            self.assertAlmostEqual(
                result[i]['bond_length'], 
                zmatrix[i]['bond_length'], 
                places=5,
                msg=f"Bond length mismatch at atom {i}"
            )
        
        for i in range(2, len(zmatrix)):
            self.assertAlmostEqual(
                result[i]['angle'], 
                zmatrix[i]['angle'], 
                places=3,
                msg=f"Angle mismatch at atom {i}"
            )


class TestTorsionExtraction(unittest.TestCase):
    """Test torsion angle extraction."""
    
    def setUp(self):
        """Create test Z-matrix."""
        self.zmatrix = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 110.0},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 110.0, 'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0},
            {'id': 4, 'element': 'C', 'atomic_num': 6, 'bond_ref': 3, 'bond_length': 1.54,
             'angle_ref': 2, 'angle': 110.0, 'dihedral_ref': 1, 'dihedral': -60.0, 'chirality': 0},
        ]
        
        self.rotatable_indices = [3, 4]  # Atoms 4 and 5 (0-based: 3, 4)
    
    def test_extract_torsions(self):
        """Test extraction of specific torsion angles."""
        coords = zmatrix_to_cartesian(self.zmatrix)
        torsions = extract_torsions(coords, self.zmatrix, self.rotatable_indices)
        
        # Check we got the right number of torsions
        self.assertEqual(len(torsions), 2)
        
        # Check torsion values
        self.assertAlmostEqual(torsions[0], 60.0, places=3)
        self.assertAlmostEqual(torsions[1], -60.0, places=3)
    
    def test_apply_and_extract_torsions(self):
        """Test applying and extracting torsions."""
        # Apply new torsion values
        new_torsions = np.array([90.0, -90.0])
        zmatrix_modified = apply_torsions(
            self.zmatrix, self.rotatable_indices, new_torsions
        )
        
        # Convert to Cartesian
        coords = zmatrix_to_cartesian(zmatrix_modified)
        
        # Extract torsions
        extracted_torsions = extract_torsions(
            coords, zmatrix_modified, self.rotatable_indices
        )
        
        # Check they match
        np.testing.assert_array_almost_equal(
            new_torsions, extracted_torsions, decimal=3
        )


def run_tests(verbosity=2):
    """Run all tests with specified verbosity."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestGeometryCalculations))
    suite.addTests(loader.loadTestsFromTestCase(TestSimpleZMatrix))
    suite.addTests(loader.loadTestsFromTestCase(TestConversionCycle))
    suite.addTests(loader.loadTestsFromTestCase(TestChiralityHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestCartesianToZMatrix))
    suite.addTests(loader.loadTestsFromTestCase(TestTorsionExtraction))
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests(verbosity=2)
    sys.exit(0 if success else 1)

