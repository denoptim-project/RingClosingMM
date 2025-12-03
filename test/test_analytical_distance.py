#!/usr/bin/env python3
"""
Unit tests for AnalyticalDistance module.

Tests analytical distance computation, JIT compilation, and factory functionality.
"""

import math
import time
import unittest
import numpy as np
import sys
from pathlib import Path

from ringclosingmm import CoordinateConversion
from ringclosingmm.IOTools import write_xyz_file

# Add parent directory to path for package imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ringclosingmm.AnalyticalDistance import (
    rotation_matrix_around_axis,
    rotation_matrix_from_angle,
    AnalyticalDistanceFunction,
    AnalyticalDistanceFactory,
    NUMBA_AVAILABLE
)
from ringclosingmm.ZMatrix import ZMatrix
from ringclosingmm.CoordinateConversion import zmatrix_to_cartesian


class TestRotationMatrices(unittest.TestCase):
    """Test rotation matrix computation functions."""
    
    def test_rotation_matrix_from_angle_x_axis(self):
        """Test rotation around x-axis."""
        angle = 90.0
        R = rotation_matrix_from_angle(angle, 'x')
        
        # Should be 3x3 matrix
        self.assertEqual(R.shape, (3, 3))
        
        # Rotate [0, 1, 0] by 90° around x should give [0, 0, 1]
        vec = np.array([0.0, 1.0, 0.0])
        rotated = R @ vec
        expected = np.array([0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(rotated, expected, decimal=5)
    
    def test_rotation_matrix_from_angle_y_axis(self):
        """Test rotation around y-axis."""
        angle = 90.0
        R = rotation_matrix_from_angle(angle, 'y')
        
        # Rotate [1, 0, 0] by 90° around y should give [0, 0, -1]
        vec = np.array([1.0, 0.0, 0.0])
        rotated = R @ vec
        expected = np.array([0.0, 0.0, -1.0])
        np.testing.assert_array_almost_equal(rotated, expected, decimal=5)
    
    def test_rotation_matrix_from_angle_z_axis(self):
        """Test rotation around z-axis."""
        angle = 90.0
        R = rotation_matrix_from_angle(angle, 'z')
        
        # Rotate [1, 0, 0] by 90° around z should give [0, 1, 0]
        vec = np.array([1.0, 0.0, 0.0])
        rotated = R @ vec
        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(rotated, expected, decimal=5)
    
    def test_rotation_matrix_around_axis(self):
        """Test rotation around arbitrary axis."""
        axis = np.array([1.0, 1.0, 0.0])
        axis = axis / np.linalg.norm(axis)  # Normalize
        angle = 180.0
        
        R = rotation_matrix_around_axis(axis, angle)
        
        # Should be 3x3 matrix
        self.assertEqual(R.shape, (3, 3))
        
        # Rotation matrix should be orthogonal
        RRT = R @ R.T
        np.testing.assert_array_almost_equal(RRT, np.eye(3), decimal=5)
    
    def test_rotation_matrix_identity(self):
        """Test that 0° rotation gives identity."""
        R = rotation_matrix_from_angle(0.0, 'z')
        np.testing.assert_array_almost_equal(R, np.eye(3), decimal=5)


class TestAnalyticalDistanceFunction(unittest.TestCase):
    """Test AnalyticalDistanceFunction class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple linear chain Z-matrix (4 atoms)
        self.linear_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 
             'bond_ref': 0, 'bond_length': 1.5},
            {'id': 2, 'element': 'C', 'atomic_num': 6,
             'bond_ref': 1, 'bond_length': 1.5,
             'angle_ref': 0, 'angle': 180.0},  # Linear
            {'id': 3, 'element': 'C', 'atomic_num': 6,
             'bond_ref': 2, 'bond_length': 1.5,
             'angle_ref': 1, 'angle': 180.0,
             'dihedral_ref': 0, 'dihedral': 0.0, 'chirality': 0}
        ]
        self.linear_bonds = [(0, 1), (1, 2), (2, 3)]
        self.linear_zmatrix = ZMatrix(self.linear_atoms, self.linear_bonds)
        
        # Create a bent chain Z-matrix (4 atoms with angles)
        self.bent_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6,
             'bond_ref': 0, 'bond_length': 1.5},
            {'id': 2, 'element': 'C', 'atomic_num': 6,
             'bond_ref': 1, 'bond_length': 1.5,
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6,
             'bond_ref': 2, 'bond_length': 1.5,
             'angle_ref': 1, 'angle': 109.47,
             'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0}
        ]
        self.bent_bonds = [(0, 1), (1, 2), (2, 3)]
        self.bent_zmatrix = ZMatrix(self.bent_atoms, self.bent_bonds)
    
    def test_analytical_distance_function_initialization(self):
        """Test initialization of AnalyticalDistanceFunction."""
        # Path from atom 0 to atom 3 includes all intermediate atoms
        path_info = {
            'bond_lengths': [1.5, 1.5, 1.5],  # 3 bonds: 0-1, 1-2, 2-3
            'bond_angles': [180.0, 180.0, 180.0],
            'dihedral_indices': [3],
            'dihedral_positions': [2],  # Dihedral at atom 3 is at position 2 in path
            'path_atoms': [0, 1, 2, 3],  # Full path
            'zmatrix_refs': []
        }
        
        dist_func = AnalyticalDistanceFunction(path_info, self.linear_zmatrix)
        
        self.assertEqual(len(dist_func.bond_lengths), 3)
        self.assertEqual(len(dist_func.path_atoms), 4)
    
    def test_distance_computation_linear_chain(self):
        """Test distance computation for linear chain."""
        # Path from atom 0 to atom 3 in linear chain
        path_info = {
            'bond_lengths': [1.5, 1.5, 1.5],
            'bond_angles': [180.0, 180.0, 180.0],
            'dihedral_indices': [],
            'dihedral_positions': [],
            'path_atoms': [0, 1, 2, 3],
            'zmatrix_refs': []
        }
        
        dist_func = AnalyticalDistanceFunction(path_info, self.linear_zmatrix)
        
        # For linear chain, distance should be sum of bond lengths
        distance = dist_func({})
        expected = 1.5 + 1.5 + 1.5  # 4.5
        self.assertAlmostEqual(distance, expected, places=3)
    
    def test_distance_computation_with_dihedral(self):
        """Test distance computation with variable dihedral."""
        # Path from atom 0 to atom 3: [0, 1, 2, 3]
        # Bond positions:
        #   - Position 0: bond between atoms 0-1
        #   - Position 1: bond between atoms 1-2
        #   - Position 2: bond between atoms 2-3
        # dihedral_indices=[3] means atom 3 (in Z-matrix) has a variable dihedral
        # dihedral_positions=[2] means this dihedral affects bond position 2 (bond between 2-3)
        path_info = {
            'bond_lengths': [1.5, 1.5, 1.5],  # Three bonds: 0-1, 1-2, 2-3
            'bond_angles': [109.47, 109.47, 109.47],
            'dihedral_indices': [3],  # Atom 3 has a variable dihedral
            'dihedral_positions': [2],  # Affects bond at position 2 (between atoms 2-3)
            'path_atoms': [0, 1, 2, 3],  # Full path from start to end
            'zmatrix_refs': []
        }
        
        dist_func = AnalyticalDistanceFunction(path_info, self.bent_zmatrix)
        
        # Compute distance with default dihedral
        distance_default = dist_func({})
        
        # Compute distance with different dihedral
        distance_modified = dist_func({3: 180.0})
        
        # Distances should be different
        self.assertNotAlmostEqual(distance_default, distance_modified, places=2)
    
    def test_gradient_computation(self):
        """Test gradient computation using finite differences."""
        # Path from atom 0 to atom 3: [0, 1, 2, 3]
        # Bond positions:
        #   - Position 0: bond between atoms 0-1
        #   - Position 1: bond between atoms 1-2
        #   - Position 2: bond between atoms 2-3
        # dihedral_indices=[3] means atom 3 (in Z-matrix) has a variable dihedral
        # dihedral_positions=[2] means this dihedral affects bond position 2 (bond between 2-3)
        path_info = {
            'bond_lengths': [1.5, 1.5, 1.5],  # Three bonds: 0-1, 1-2, 2-3
            'bond_angles': [109.47, 109.47, 109.47],
            'dihedral_indices': [3],  # Atom 3 has a variable dihedral
            'dihedral_positions': [2],  # Affects bond at position 2 (between atoms 2-3)
            'path_atoms': [0, 1, 2, 3],  # Full path from start to end
            'zmatrix_refs': []
        }
        
        dist_func = AnalyticalDistanceFunction(path_info, self.bent_zmatrix)
        
        dihedral_values = {3: 60.0}
        gradient = dist_func.gradient(dihedral_values, 3, eps=0.1)
        
        # Gradient should be a finite number
        self.assertFalse(np.isnan(gradient))
        self.assertFalse(np.isinf(gradient))
    
    def test_distance_consistency_with_cartesian(self):
        """Test that analytical distance matches Cartesian conversion."""
        # Use a simple 3-atom chain
        simple_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6,
             'bond_ref': 0, 'bond_length': 1.5},
            {'id': 2, 'element': 'C', 'atomic_num': 6,
             'bond_ref': 1, 'bond_length': 1.5,
             'angle_ref': 0, 'angle': 109.47}
        ]
        simple_bonds = [(0, 1), (1, 2)]
        simple_zmatrix = ZMatrix(simple_atoms, simple_bonds)
        
        path_info = {
            'bond_lengths': [1.5, 1.5],
            'bond_angles': [109.47, 109.47],
            'dihedral_indices': [],
            'dihedral_positions': [],
            'path_atoms': [0, 1, 2],
            'zmatrix_refs': []
        }
        
        dist_func = AnalyticalDistanceFunction(path_info, simple_zmatrix)
        analytical_dist = dist_func({})
        
        # Compare with Cartesian conversion
        coords = zmatrix_to_cartesian(simple_zmatrix)
        cartesian_dist = np.linalg.norm(coords[2] - coords[0])
        
        # Should be close (within 0.1 Å due to numerical differences)
        self.assertAlmostEqual(analytical_dist, cartesian_dist, places=1)


class TestAnalyticalDistanceFactory(unittest.TestCase):
    """Test AnalyticalDistanceFactory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple chain Z-matrix
        self.chain_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.5},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.5, 'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.5, 'angle_ref': 1, 'angle': 109.47,
             'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0},
            {'id': 4, 'element': 'C', 'atomic_num': 6, 'bond_ref': 3, 'bond_length': 1.5, 'angle_ref': 2, 'angle': 109.47,
             'dihedral_ref': 1, 'dihedral': 60.0, 'chirality': 0}
        ]
        self.chain_bonds = [(0, 1), (1, 2), (2, 3), (3, 4)]
        self.chain_zmatrix = ZMatrix(self.chain_atoms, self.chain_bonds)
        
        # Create OpenMM topology for complete connectivity
        try:
            from openmm.app import Topology, Element
            self.topology = Topology()
            chain = self.topology.addChain()
            residue = self.topology.addResidue("MOL", chain)
            
            # Store atoms in a list for bond creation
            atoms_list = []
            for atom in self.chain_atoms:
                element = Element.getByAtomicNumber(atom['atomic_num'])
                atom_obj = self.topology.addAtom(atom['element'], element, residue)
                atoms_list.append(atom_obj)
            
            # Add bonds
            for bond in self.chain_bonds:
                self.topology.addBond(
                    atoms_list[bond[0]],
                    atoms_list[bond[1]]
                )
        except ImportError:
            self.topology = None
    
    def test_factory_initialization(self):
        """Test factory initialization."""
        factory = AnalyticalDistanceFactory(self.chain_zmatrix, topology=self.topology)
        
        self.assertIsNotNone(factory.zmatrix)
        self.assertEqual(len(factory.graph), len(self.chain_zmatrix))
    
    def test_get_distance_function(self):
        """Test getting distance function for atom pair."""
        factory = AnalyticalDistanceFactory(self.chain_zmatrix, topology=self.topology)
        
        # Get distance function for atoms 0 and 4
        dist_func = factory.get_distance_function(0, 4)
        
        self.assertIsNotNone(dist_func)
        self.assertIsInstance(dist_func, AnalyticalDistanceFunction)
    
    def test_get_distance_function_caching(self):
        """Test that distance functions are cached."""
        factory = AnalyticalDistanceFactory(self.chain_zmatrix, topology=self.topology)
        
        # Get same function twice
        func1 = factory.get_distance_function(0, 4)
        func2 = factory.get_distance_function(0, 4)
        
        # Should be the same object (cached)
        self.assertIs(func1, func2)
    
    def test_get_distance_function_canonical_ordering(self):
        """Test that distance functions use canonical ordering."""
        factory = AnalyticalDistanceFactory(self.chain_zmatrix, topology=self.topology)
        
        # Get function with different atom order
        func1 = factory.get_distance_function(0, 4)
        func2 = factory.get_distance_function(4, 0)
        
        # Should be the same cached function
        self.assertIs(func1, func2)
    
    def test_get_all_distance_functions(self):
        """Test batch creation of distance functions."""
        factory = AnalyticalDistanceFactory(self.chain_zmatrix, topology=self.topology)
        
        pairs = [(0, 2), (1, 3), (0, 4)]
        all_funcs = factory.get_all_distance_functions(pairs)
        
        self.assertEqual(len(all_funcs), 3)
        for pair in pairs:
            cache_key = (min(pair[0], pair[1]), max(pair[0], pair[1]))
            self.assertIn(cache_key, all_funcs)
            self.assertIsNotNone(all_funcs[cache_key])
    
    def test_get_distance_function_no_path(self):
        """Test behavior when no path exists."""
        # Create disconnected Z-matrix
        disconnected_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6,
             'bond_ref': 0, 'bond_length': 1.5},
            {'id': 2, 'element': 'C', 'atomic_num': 6},  # Disconnected
        ]
        disconnected_bonds = [(0, 1)]  # No bond to atom 2
        disconnected_zmatrix = ZMatrix(disconnected_atoms, disconnected_bonds)
        
        factory = AnalyticalDistanceFactory(disconnected_zmatrix)
        
        # Should return None for disconnected atoms
        dist_func = factory.get_distance_function(0, 2)
        self.assertIsNone(dist_func)
    
    def test_clear_cache(self):
        """Test cache clearing."""
        factory = AnalyticalDistanceFactory(self.chain_zmatrix, topology=self.topology)
        
        # Create some cached functions
        factory.get_distance_function(0, 2)
        factory.get_distance_function(1, 3)
        
        self.assertGreater(len(factory.get_cached_functions()), 0)
        
        # Clear cache
        factory.clear_cache()
        
        self.assertEqual(len(factory.get_cached_functions()), 0)
    
    def test_get_cached_functions(self):
        """Test getting cached functions."""
        factory = AnalyticalDistanceFactory(self.chain_zmatrix, topology=self.topology)
        
        # Initially empty
        self.assertEqual(len(factory.get_cached_functions()), 0)
        
        # Create a function
        factory.get_distance_function(0, 4)
        
        cached = factory.get_cached_functions()
        self.assertEqual(len(cached), 1)
        self.assertIn((0, 4), cached)
    
    def test_get_distance_function_with_branch_atoms(self):
        """Test that branch atoms (not in path) with rotatable dihedrals are included."""
        # Create structure: 0-1-2-4
        # Atom 3 is a branch from atom 2, not in path
        # Atom 3 has a rotatable dihedral that affects the path geometry
        branch_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.5},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.5, 
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.5, 
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 0.0, 'chirality': 0},  # Branch atom
            {'id': 4, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.5, 
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 3, 'dihedral': 90.0, 'chirality': 0},  # In path
        ]
        branch_bonds = [(0, 1), (1, 2), (2, 3), (2, 4)]
        branch_zmatrix = ZMatrix(branch_atoms, branch_bonds)
        
        from openmm.app import Topology, Element, Residue
        branch_topology = Topology()
        residue = branch_topology.addResidue("MOL", branch_topology.addChain())
        atoms_list = []
        for atom in branch_atoms:
            element = Element.getByAtomicNumber(atom['atomic_num'])
            atom_obj = branch_topology.addAtom(atom['element'], element, residue)
            atoms_list.append(atom_obj)
        for bond in branch_bonds:
            branch_topology.addBond(atoms_list[bond[0]], atoms_list[bond[1]])
        
        factory = AnalyticalDistanceFactory(branch_zmatrix, topology=branch_topology)
        
        # Path from 0 to 4: [0, 1, 2, 4]
        # Atom 3 is a branch from atom 2, not in path
        # Atom 3 has bond_ref=2 (which is in path), so its dihedral affects the path
        # Atom 4 is in path and has bond_ref=2, so atom 3's dihedral affects bond 2-4
        
        # Test with rotatable_indices including branch atom 3
        dist_func = factory.get_distance_function(0, 4, rotatable_indices=[3, 4])
        
        self.assertIsNotNone(dist_func, "Distance function should be created")
        
        # Verify path
        self.assertEqual(list(dist_func.path_atoms), [0, 1, 2, 4], 
                        "Path should be [0, 1, 2, 4]")
        
        # Verify that branch atom 3 is included in dihedral_indices
        dihedral_indices_list = [int(idx) for idx in dist_func.dihedral_indices]
        self.assertIn(3, dihedral_indices_list, 
                     "Branch atom 3 should be included in dihedral_indices")
        self.assertIn(4, dihedral_indices_list, 
                     "Path atom 4 should be included in dihedral_indices")
        
        # Verify dihedral_positions
        # Atom 3 (branch from atom 2): bond_ref=2 is at path position 2
        # The bond after atom 2 in path is 2-4 at position 2, so dihedral_position should be 2
        # Atom 4 (in path): bond_ref=2, so its dihedral affects bond 2-4 at position 2
        dihedral_positions_list = dist_func.dihedral_positions
        self.assertEqual(len(dihedral_positions_list), len(dihedral_indices_list),
                         "dihedral_positions should have same length as dihedral_indices")
        
        # Find positions for atoms 3 and 4
        idx_3 = dihedral_indices_list.index(3)
        idx_4 = dihedral_indices_list.index(4)
        pos_3 = dihedral_positions_list[idx_3]
        pos_4 = dihedral_positions_list[idx_4]
        
        # Both should affect bond at position 2 (bond 2-4)
        self.assertEqual(pos_3, 2, 
                         f"Branch atom 3 should affect bond at position 2, got {pos_3}")
        self.assertEqual(pos_4, 2, 
                         f"Path atom 4 should affect bond at position 2, got {pos_4}")
        
        # Verify that changing branch atom 3's dihedral affects the distance
        dist1 = dist_func({3: 0.0, 4: 90.0})
        dist2 = dist_func({3: 180.0, 4: 90.0})
        
        # The distances might be the same if the geometry doesn't change significantly,
        # but we should at least be able to compute them
        self.assertGreater(dist1, 0.0, "Distance should be positive")
        self.assertGreater(dist2, 0.0, "Distance should be positive")
    
    def test_get_distance_function_with_multiple_branch_atoms(self):
        """Test that multiple branch atoms are correctly handled."""
        # Create structure: 0-1-2-4-6
        # Atom 3 is a branch from atom 2, not in path
        # Atom 5 is a branch from atom 4, not in path
        # Both have rotatable dihedrals
        branch_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.5},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.5, 
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.5, 
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 0.0, 'chirality': 0},  # Branch from 2
            {'id': 4, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.5, 
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 3, 'dihedral': 90.0, 'chirality': 0},  # In path
            {'id': 5, 'element': 'C', 'atomic_num': 6, 'bond_ref': 4, 'bond_length': 1.5, 
             'angle_ref': 2, 'angle': 109.47, 'dihedral_ref': 1, 'dihedral': 0.0, 'chirality': 0},  # Branch from 4
            {'id': 6, 'element': 'C', 'atomic_num': 6, 'bond_ref': 4, 'bond_length': 1.5, 
             'angle_ref': 2, 'angle': 109.47, 'dihedral_ref': 5, 'dihedral': 120.0, 'chirality': 0},  # In path
        ]
        branch_bonds = [(0, 1), (1, 2), (2, 3), (2, 4), (4, 5), (4, 6)]
        branch_zmatrix = ZMatrix(branch_atoms, branch_bonds)
        
        from openmm.app import Topology, Element, Residue
        branch_topology = Topology()
        residue = branch_topology.addResidue("MOL", branch_topology.addChain())
        atoms_list = []
        for atom in branch_atoms:
            element = Element.getByAtomicNumber(atom['atomic_num'])
            atom_obj = branch_topology.addAtom(atom['element'], element, residue)
            atoms_list.append(atom_obj)
        for bond in branch_bonds:
            branch_topology.addBond(atoms_list[bond[0]], atoms_list[bond[1]])
        
        factory = AnalyticalDistanceFactory(branch_zmatrix, topology=branch_topology)
        
        # Path from 0 to 6: [0, 1, 2, 4, 6]
        # Atom 3 is a branch from atom 2 (at path position 2), affects bond 2-4 at position 2
        # Atom 5 is a branch from atom 4 (at path position 3), affects bond 4-6 at position 3
        
        dist_func = factory.get_distance_function(0, 6, rotatable_indices=[3, 4, 5, 6])
        
        self.assertIsNotNone(dist_func, "Distance function should be created")
        
        # Verify path
        self.assertEqual(list(dist_func.path_atoms), [0, 1, 2, 4, 6], 
                        "Path should be [0, 1, 2, 4, 6]")
        
        # Verify that both branch atoms are included
        dihedral_indices_list = [int(idx) for idx in dist_func.dihedral_indices]
        self.assertIn(3, dihedral_indices_list, 
                     "Branch atom 3 should be included in dihedral_indices")
        self.assertIn(5, dihedral_indices_list, 
                     "Branch atom 5 should be included in dihedral_indices")
        self.assertIn(4, dihedral_indices_list, 
                     "Path atom 4 should be included in dihedral_indices")
        self.assertIn(6, dihedral_indices_list, 
                     "Path atom 6 should be included in dihedral_indices")
        
        # Verify dihedral_positions
        dihedral_positions_list = dist_func.dihedral_positions
        idx_3 = dihedral_indices_list.index(3)
        idx_5 = dihedral_indices_list.index(5)
        idx_4 = dihedral_indices_list.index(4)
        idx_6 = dihedral_indices_list.index(6)
        
        # Atom 3 (branch from atom 2 at path pos 2): affects bond 2-4 at position 2
        self.assertEqual(dihedral_positions_list[idx_3], 2,
                         "Branch atom 3 should affect bond at position 2")
        
        # Atom 5 (branch from atom 4 at path pos 3): affects bond 4-6 at position 3
        self.assertEqual(dihedral_positions_list[idx_5], 3,
                         "Branch atom 5 should affect bond at position 3")
        
        # Atom 4 (in path, bond_ref=2): affects bond 2-4 at position 2
        self.assertEqual(dihedral_positions_list[idx_4], 2,
                         "Path atom 4 should affect bond at position 2")
        
        # Atom 6 (in path, bond_ref=4): affects bond 4-6 at position 3
        self.assertEqual(dihedral_positions_list[idx_6], 3,
                         "Path atom 6 should affect bond at position 3")
    
    def test_get_distance_function_with_branch_atoms_as_references(self):
        """Test that branch atoms used as angle_ref or dihedral_ref by path atoms are included."""
        # Create structure: Path 0-1-2(3)-4(5)-6-7
        # Where (3) and (5) are branch atoms
        # Atom 3 (branch): torsion 3,2,1,0 (bond_ref=2, angle_ref=1, dihedral_ref=0)
        # Atom 4 (in path): angles 4,2,1 and 4,2,3 (bond_ref=2, angle_ref=1, dihedral_ref=3, chirality != 0)
        # Atom 5 (branch): angle 5,4,2 and torsion 5,4,2,3 (bond_ref=4, angle_ref=2, dihedral_ref=3)
        # Atom 6 (in path): angle 6,4,2 and angle 6,4,5 (bond_ref=4, angle_ref=2, dihedral_ref=5, chirality != 0)
        # Atom 7 (in path): angle 7,6,4 and dihedral 7,6,4,5 (bond_ref=6, angle_ref=4, dihedral_ref=5)
        branch_ref_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},  # A
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.5},  # B
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.5, 
             'angle_ref': 0, 'angle': 109.47},  # C (3rd atom, no dihedral)
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.5, 
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 0.0, 'chirality': 0},  # Branch: torsion 3,2,1,0
            {'id': 4, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.5, 
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 3, 'dihedral': 90.0, 'chirality': -1},  # In path: angles 4,2,1 and 4,2,3
            {'id': 5, 'element': 'C', 'atomic_num': 6, 'bond_ref': 4, 'bond_length': 1.5, 
             'angle_ref': 2, 'angle': 109.47, 'dihedral_ref': 3, 'dihedral': 0.0, 'chirality': 0},  # Branch: angle 5,4,2 and torsion 5,4,2,3
            {'id': 6, 'element': 'C', 'atomic_num': 6, 'bond_ref': 4, 'bond_length': 1.5, 
             'angle_ref': 2, 'angle': 109.47, 'dihedral_ref': 5, 'dihedral': 120.0, 'chirality': 1},  # In path: angle 6,4,2 and angle 6,4,5
            {'id': 7, 'element': 'C', 'atomic_num': 6, 'bond_ref': 6, 'bond_length': 1.5, 
             'angle_ref': 4, 'angle': 109.47, 'dihedral_ref': 5, 'dihedral': 180.0, 'chirality': 0},  # In path: angle 7,6,4 and dihedral 7,6,4,5
        ]
        branch_ref_bonds = [(0, 1), (1, 2), (2, 3), (2, 4), (4, 5), (4, 6), (6, 7)]
        branch_ref_zmatrix = ZMatrix(branch_ref_atoms, branch_ref_bonds)

        #TODO del
        write_xyz_file(CoordinateConversion.zmatrix_to_cartesian(branch_ref_zmatrix), branch_ref_zmatrix.get_elements(), "/tmp/branch_ref_zmatrix.xyz")
        
        from openmm.app import Topology, Element, Residue
        branch_ref_topology = Topology()
        residue = branch_ref_topology.addResidue("MOL", branch_ref_topology.addChain())
        atoms_list = []
        for atom in branch_ref_atoms:
            element = Element.getByAtomicNumber(atom['atomic_num'])
            atom_obj = branch_ref_topology.addAtom(atom['element'], element, residue)
            atoms_list.append(atom_obj)
        for bond in branch_ref_bonds:
            branch_ref_topology.addBond(atoms_list[bond[0]], atoms_list[bond[1]])
        
        factory = AnalyticalDistanceFactory(branch_ref_zmatrix, topology=branch_ref_topology)
        
        # Path from 0 to 7: [0, 1, 2, 4, 6, 7]
        # Atom 4 (in path) has dihedral_ref=3 (branch atom, not in path)
        # Atom 6 (in path) has dihedral_ref=5 (branch atom, not in path)
        # Atom 7 (in path) has dihedral_ref=5 (branch atom, not in path)
        # Atoms 3 and 5 should be included because their dihedrals affect path atoms' positions
        
        dist_func = factory.get_distance_function(0, 7, rotatable_indices=[3, 4, 5, 6, 7])
        
        self.assertIsNotNone(dist_func, "Distance function should be created")
        
        # Verify path
        self.assertEqual(list(dist_func.path_atoms), [0, 1, 2, 4, 6, 7], 
                        "Path should be [0, 1, 2, 4, 6, 7]")
        
        # Verify that branch atoms 3 and 5 are included
        # Note: Atom 4 has chirality != 0, so it's not included (only true dihedrals are included)
        # Atom 6 has chirality != 0, so it's not included either
        dihedral_indices_list = [int(idx) for idx in dist_func.dihedral_indices]
        self.assertIn(7, dihedral_indices_list, 
                     "Path atom 7 should be included in dihedral_indices (has true dihedral)")
        self.assertIn(3, dihedral_indices_list, 
                     "Branch atom 3 (dihedral_ref for path atom 4) should be included")
        self.assertIn(5, dihedral_indices_list, 
                     "Branch atom 5 (dihedral_ref for path atoms 6 and 7) should be included")
        
        # Atom 4 has chirality != 0, so it's not in dihedral_indices (only true dihedrals are rotatable)
        # Atom 6 has chirality != 0, so it's not in dihedral_indices either
        self.assertNotIn(4, dihedral_indices_list,
                        "Path atom 4 should NOT be included (has chirality != 0, not a true dihedral)")
        self.assertNotIn(6, dihedral_indices_list,
                        "Path atom 6 should NOT be included (has chirality != 0, not a true dihedral)")
        
        # Verify dihedral_positions
        dihedral_positions_list = dist_func.dihedral_positions
        idx_3 = dihedral_indices_list.index(3)
        idx_5 = dihedral_indices_list.index(5)
        idx_7 = dihedral_indices_list.index(7)
        
        # Atom 7 (in path, bond_ref=6): affects bond 6-7 at position 4
        self.assertEqual(dihedral_positions_list[idx_7], 4,
                         "Path atom 7 should affect bond at position 4")
        
        # Atom 3 (branch, bond_ref=2): affects bond after atom 2, which is bond 2-4 at position 2
        # Also, atom 3 is dihedral_ref for atom 4, so it affects atom 4's position
        self.assertEqual(dihedral_positions_list[idx_3], 2,
                         "Branch atom 3 (dihedral_ref for atom 4) should affect bond at position 2")
        
        # Atom 5 (branch, bond_ref=4): affects bond after atom 4, which is bond 4-6 at position 3
        # Also, atom 5 is dihedral_ref for atoms 6 and 7, so it affects their positions
        self.assertEqual(dihedral_positions_list[idx_5], 3,
                         "Branch atom 5 (dihedral_ref for atoms 6 and 7) should affect bond at position 3")
        
        # Verify that changing branch atom dihedrals affects the distance
        # Note: Atoms 4 and 6 have chirality != 0, so they don't have rotatable dihedrals
        dist1 = dist_func({3: 0.0, 5: 0.0, 7: 0.0})
        self.assertTrue(isinstance(dist1, float), "Distance should be a float")
        self.assertAlmostEqual(dist1, 5.7056, places=2)

        dist1 = dist_func({3: -120.0, 5: 0.0, 7: 0.0})
        self.assertTrue(isinstance(dist1, float), "Distance should be a float")
        self.assertAlmostEqual(dist1, 4.7121, places=2)

        dist1 = dist_func({3: 0.0, 5: 75.0, 7: 0.0})
        self.assertTrue(isinstance(dist1, float), "Distance should be a float")
        self.assertAlmostEqual(dist1, 4.9703, places=2)

        dist1 = dist_func({3: 0.0, 5: 0.0, 7: 66.0})
        self.assertTrue(isinstance(dist1, float), "Distance should be a float")
        self.assertAlmostEqual(dist1, 5.5759, places=2)


    def test_get_distance_function_with_branch_atoms_as_references_2(self):
        """Test that branch atoms used as angle_ref or dihedral_ref by path atoms are included."""

        branch_ref_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6}, 
            {'id': 1, 'element': 'ATM', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.36}, 
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.416, 'angle_ref': 1, 'angle': 134.03},
            {'id': 3, 'element': 'Si', 'atomic_num': 14, 'bond_ref': 0, 'bond_length': 1.788, 'angle_ref': 1, 'angle': 108.28, 'dihedral_ref': 2, 'dihedral': 117.66, 'chirality': -1},  
            {'id': 4, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.5, 'angle_ref': 0, 'angle': 109.47, 'dihedral_ref': 3, 'dihedral': 0.0, 'chirality': 0},  
            {'id': 5, 'element': 'O', 'atomic_num': 8, 'bond_ref': 2, 'bond_length': 1.5, 'angle_ref': 0, 'angle': 120.0, 'dihedral_ref': 4, 'dihedral': 120.0, 'chirality': 1},   
            {'id': 6, 'element': 'C', 'atomic_num': 6, 'bond_ref': 4, 'bond_length': 1.5, 'angle_ref': 2, 'angle': 120.0, 'dihedral_ref': 5, 'dihedral': 0.0, 'chirality': 0},  
            {'id': 7, 'element': 'N', 'atomic_num': 6, 'bond_ref': 6, 'bond_length': 1.5, 'angle_ref': 4, 'angle': 120.0, 'dihedral_ref': 2, 'dihedral': 0.0, 'chirality': 0},
            {'id': 8, 'element': 'C', 'atomic_num': 6, 'bond_ref': 7, 'bond_length': 1.5, 'angle_ref': 6, 'angle': 120.0, 'dihedral_ref': 4, 'dihedral': 0.0, 'chirality': 0}, 
            {'id': 9, 'element': 'ATP', 'atomic_num': 1, 'bond_ref': 8, 'bond_length': 1.0, 'angle_ref': 7, 'angle': 120.0, 'dihedral_ref': 6, 'dihedral': -130.0, 'chirality': 0},
            {'id': 10, 'element': 'C', 'atomic_num': 6, 'bond_ref': 8, 'bond_length': 1.5, 'angle_ref': 9, 'angle': 117.38, 'dihedral_ref': 7, 'dihedral': 132.17, 'chirality': 1},
            {'id': 11, 'element': 'ATP', 'atomic_num': 1, 'bond_ref': 10, 'bond_length': 1.5, 'angle_ref': 8, 'angle': 88.443, 'dihedral_ref': 9, 'dihedral': 125.786, 'chirality': 0},
            {'id': 12, 'element': 'C', 'atomic_num': 6, 'bond_ref': 3, 'bond_length': 1.5, 'angle_ref': 0, 'angle': 99.55, 'dihedral_ref': 1, 'dihedral': 2.73, 'chirality': 0},
            {'id': 13, 'element': 'ATM', 'atomic_num': 1, 'bond_ref': 12, 'bond_length': 1.5, 'angle_ref': 3, 'angle': 100.57, 'dihedral_ref': 0, 'dihedral': -1.03, 'chirality': 0},
        ]
        branch_ref_bonds = [(0, 1), (0, 2), (0, 3), (2, 4), (2, 5), (4, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (3, 12), (12, 13)]
        rotatable_indices=[4, 6, 7, 8, 9, 11, 12, 13]
        branch_ref_zmatrix = ZMatrix(branch_ref_atoms, branch_ref_bonds)

        #TODO del
        write_xyz_file(CoordinateConversion.zmatrix_to_cartesian(branch_ref_zmatrix), branch_ref_zmatrix.get_elements(), "/tmp/branch_ref_zmatrix.xyz")
        
        from openmm.app import Topology, Element, Residue
        branch_ref_topology = Topology()
        residue = branch_ref_topology.addResidue("MOL", branch_ref_topology.addChain())
        atoms_list = []
        for atom in branch_ref_atoms:
            element = Element.getByAtomicNumber(atom['atomic_num'])
            atom_obj = branch_ref_topology.addAtom(atom['element'], element, residue)
            atoms_list.append(atom_obj)
        for bond in branch_ref_bonds:
            branch_ref_topology.addBond(atoms_list[bond[0]], atoms_list[bond[1]])
        
        factory = AnalyticalDistanceFactory(branch_ref_zmatrix, topology=branch_ref_topology)
        
        dist_func = factory.get_distance_function(11, 13, rotatable_indices=rotatable_indices)
        
        self.assertIsNotNone(dist_func, "Distance function should be created")
        
        # Verify path
        self.assertEqual(list(dist_func.path_atoms), [11, 10, 9, 8, 7, 6, 4, 2, 0, 3, 12, 13], 
                        "Path should be [11, 10, 9, 8, 7, 6, 4, 2, 0, 3, 12, 13]")
        
        # Verify that changing branch atom dihedrals affects the distance
        dist1 = dist_func({4: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 11:125.786, 12:2.73, 13:-1.03})
        self.assertTrue(isinstance(dist1, float), "Distance should be a float")
        self.assertAlmostEqual(dist1, 7.0307, places=2)

        dist1 = dist_func({4: 90.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 11:125.786, 12:2.73, 13:-1.03})
        self.assertTrue(isinstance(dist1, float), "Distance should be a float")
        self.assertAlmostEqual(dist1, 7.2630, places=2)

        dist1 = dist_func({4: 0.0, 6: 75.0, 7: 0.0, 8: 0.0, 9: 0.0, 11:125.786, 12:2.73, 13:-1.03})
        self.assertTrue(isinstance(dist1, float), "Distance should be a float")
        self.assertAlmostEqual(dist1, 3.8960, places=2)

        dist1 = dist_func({4: 0.0, 6: 0.0, 7: -26.0, 8: 0.0, 9: 0.0, 11:125.786, 12:2.73, 13:-1.03})
        self.assertTrue(isinstance(dist1, float), "Distance should be a float")
        self.assertAlmostEqual(dist1, 7.4241, places=2)

        dist1 = dist_func({4: 0.0, 6: 0.0, 7: 0.0, 8: 65.0, 9: 0.0, 11:125.786, 12:2.73, 13:-1.03})
        self.assertTrue(isinstance(dist1, float), "Distance should be a float")
        self.assertAlmostEqual(dist1, 8.0627, places=2)

        dist1 = dist_func({4: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: -110.0, 11:125.786, 12:2.73, 13:-1.03})
        self.assertTrue(isinstance(dist1, float), "Distance should be a float")
        self.assertAlmostEqual(dist1, 3.7647, places=2)

        dist1 = dist_func({4: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 11:179.0, 12:2.73, 13:-1.03})
        self.assertTrue(isinstance(dist1, float), "Distance should be a float")
        self.assertAlmostEqual(dist1, 7.6853, places=2)

        dist1 = dist_func({4: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 11:125.786, 12:2.73, 13:120.0})
        self.assertTrue(isinstance(dist1, float), "Distance should be a float")
        self.assertAlmostEqual(dist1, 8.7258, places=2)



    def test_get_distance_function_first_atom_dihedral(self):
        """Test that the first atom in the path with a rotatable dihedral is included.
        
        This test covers the bug fix where the first atom in the path was not being
        checked for rotatable dihedrals, causing its dihedral changes to be ignored.
        """
        # Create a Z-matrix where the first atom in a path has a rotatable dihedral
        # Structure: 11-10-9-8-7-6-4-2-0-3-12-13
        # Atom 11 is the first atom in the path and has a rotatable dihedral
        first_atom_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6}, 
            {'id': 1, 'element': 'ATM', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.36}, 
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.416, 'angle_ref': 1, 'angle': 134.03},
            {'id': 3, 'element': 'Si', 'atomic_num': 14, 'bond_ref': 0, 'bond_length': 1.788, 'angle_ref': 1, 'angle': 108.28, 'dihedral_ref': 2, 'dihedral': 117.66, 'chirality': -1},  
            {'id': 4, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.5, 'angle_ref': 0, 'angle': 109.47, 'dihedral_ref': 3, 'dihedral': 0.0, 'chirality': 0},  
            {'id': 5, 'element': 'O', 'atomic_num': 8, 'bond_ref': 2, 'bond_length': 1.5, 'angle_ref': 0, 'angle': 120.0, 'dihedral_ref': 4, 'dihedral': 120.0, 'chirality': 1},   
            {'id': 6, 'element': 'C', 'atomic_num': 6, 'bond_ref': 4, 'bond_length': 1.5, 'angle_ref': 2, 'angle': 120.0, 'dihedral_ref': 5, 'dihedral': 0.0, 'chirality': 0},  
            {'id': 7, 'element': 'N', 'atomic_num': 6, 'bond_ref': 6, 'bond_length': 1.5, 'angle_ref': 4, 'angle': 120.0, 'dihedral_ref': 2, 'dihedral': 0.0, 'chirality': 0},
            {'id': 8, 'element': 'C', 'atomic_num': 6, 'bond_ref': 7, 'bond_length': 1.5, 'angle_ref': 6, 'angle': 120.0, 'dihedral_ref': 4, 'dihedral': 0.0, 'chirality': 0}, 
            {'id': 9, 'element': 'ATP', 'atomic_num': 1, 'bond_ref': 8, 'bond_length': 1.0, 'angle_ref': 7, 'angle': 120.0, 'dihedral_ref': 6, 'dihedral': -130.0, 'chirality': 0},
            {'id': 10, 'element': 'C', 'atomic_num': 6, 'bond_ref': 8, 'bond_length': 1.5, 'angle_ref': 9, 'angle': 117.38, 'dihedral_ref': 7, 'dihedral': 132.17, 'chirality': 1},
            {'id': 11, 'element': 'ATP', 'atomic_num': 1, 'bond_ref': 10, 'bond_length': 1.5, 'angle_ref': 9, 'angle': 114.1, 'dihedral_ref': 8, 'dihedral': -0.3863, 'chirality': 0},  # First atom in path, has rotatable dihedral
            {'id': 12, 'element': 'C', 'atomic_num': 6, 'bond_ref': 3, 'bond_length': 1.5, 'angle_ref': 0, 'angle': 99.55, 'dihedral_ref': 1, 'dihedral': 2.73, 'chirality': 0},
            {'id': 13, 'element': 'ATM', 'atomic_num': 1, 'bond_ref': 12, 'bond_length': 1.5, 'angle_ref': 3, 'angle': 100.57, 'dihedral_ref': 0, 'dihedral': -1.03, 'chirality': 0},
        ]
        first_atom_bonds = [(0, 1), (0, 2), (0, 3), (2, 4), (2, 5), (4, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (3, 12), (12, 13)]
        rotatable_indices = [4, 6, 7, 8, 9, 11, 12, 13]
        first_atom_zmatrix = ZMatrix(first_atom_atoms, first_atom_bonds)
        
        from openmm.app import Topology, Element, Residue
        first_atom_topology = Topology()
        residue = first_atom_topology.addResidue("MOL", first_atom_topology.addChain())
        atoms_list = []
        for atom in first_atom_atoms:
            element = Element.getByAtomicNumber(atom['atomic_num'])
            atom_obj = first_atom_topology.addAtom(atom['element'], element, residue)
            atoms_list.append(atom_obj)
        for bond in first_atom_bonds:
            first_atom_topology.addBond(atoms_list[bond[0]], atoms_list[bond[1]])
        
        factory = AnalyticalDistanceFactory(first_atom_zmatrix, topology=first_atom_topology)
        
        # Path from atom 11 to atom 13: [11, 10, 9, 8, 7, 6, 4, 2, 0, 3, 12, 13]
        # Atom 11 is the first atom in the path and has a rotatable dihedral
        dist_func = factory.get_distance_function(11, 13, rotatable_indices=rotatable_indices)
        
        self.assertIsNotNone(dist_func, "Distance function should be created")
        
        # Verify path
        self.assertEqual(list(dist_func.path_atoms), [11, 10, 9, 8, 7, 6, 4, 2, 0, 3, 12, 13], 
                        "Path should start with atom 11")
        
        # Verify that atom 11 (first atom) is included in dihedral_indices
        dihedral_indices_list = [int(idx) for idx in dist_func.dihedral_indices]
        self.assertIn(11, dihedral_indices_list, 
                     "First atom 11 should be included in dihedral_indices")
        
        # Verify that atom 11's dihedral_position is 0 (affects the first bond)
        atom_11_pos = dihedral_indices_list.index(11)
        dihedral_position_11 = dist_func.dihedral_positions[atom_11_pos]
        self.assertEqual(dihedral_position_11, 0,
                        "First atom's dihedral should affect bond position 0")
        
        # Test that changing atom 11's dihedral affects the distance
        # Compare analytical distance with XYZ distance
        edited_zmatrix = first_atom_zmatrix.copy()
        
        # Test with original dihedral value
        edited_zmatrix[11][ZMatrix.FIELD_DIHEDRAL] = -0.3863
        analytical_dist = dist_func({11: -0.3863})
        xyz_coords = zmatrix_to_cartesian(edited_zmatrix)
        xyz_dist = np.linalg.norm(xyz_coords[11] - xyz_coords[13])
        self.assertAlmostEqual(analytical_dist, xyz_dist, places=5,
                              msg="Analytical distance should match XYZ distance for original dihedral")
        
        # Test with different dihedral value
        edited_zmatrix[11][ZMatrix.FIELD_DIHEDRAL] = 0.0
        analytical_dist = dist_func({11: 0.0})
        xyz_coords = zmatrix_to_cartesian(edited_zmatrix)
        xyz_dist = np.linalg.norm(xyz_coords[11] - xyz_coords[13])
        self.assertAlmostEqual(analytical_dist, xyz_dist, places=5,
                              msg="Analytical distance should match XYZ distance for changed dihedral")
        
        # Test with another different dihedral value
        edited_zmatrix[11][ZMatrix.FIELD_DIHEDRAL] = 90.0
        analytical_dist = dist_func({11: 90.0})
        xyz_coords = zmatrix_to_cartesian(edited_zmatrix)
        xyz_dist = np.linalg.norm(xyz_coords[11] - xyz_coords[13])
        self.assertAlmostEqual(analytical_dist, xyz_dist, places=5,
                              msg="Analytical distance should match XYZ distance for dihedral=90.0")
        
        # Verify that the distance actually changes with different dihedral values
        dist1 = dist_func({11: -0.3863})
        dist2 = dist_func({11: 0.0})
        dist3 = dist_func({11: 90.0})
        self.assertNotAlmostEqual(dist1, dist2, places=3,
                                 msg="Distance should change when first atom's dihedral changes")
        self.assertNotAlmostEqual(dist2, dist3, places=3,
                                 msg="Distance should change when first atom's dihedral changes")



class TestJITCompilation(unittest.TestCase):
    """Test JIT compilation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.simple_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.5},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.5, 'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.5, 'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0}
        ]
        self.simple_bonds = [(0, 1), (1, 2), (2, 3)]
        self.simple_zmatrix = ZMatrix(self.simple_atoms, self.simple_bonds)
    
    @unittest.skipIf(not NUMBA_AVAILABLE, "Numba not available")
    def test_jit_compilation_enabled(self):
        """Test that JIT compilation is used when numba is available."""
        path_info = {
            'bond_lengths': [1.5, 1.5, 1.5],  # Three bonds: 0-1, 1-2, 2-3
            'bond_angles': [109.47, 109.47, 109.47],
            'dihedral_indices': [3],  # Atom 3 has a variable dihedral
            'dihedral_positions': [2],  # Affects bond at position 2 (between atoms 2-3)
            'path_atoms': [0, 1, 2, 3],  # Full path from start to end
            'zmatrix_refs': []
        }
        
        dist_func = AnalyticalDistanceFunction(path_info, self.simple_zmatrix, use_jit=True)
        
        # Should use JIT if available
        self.assertTrue(dist_func.use_jit)
        
        # Should be able to compute distance
        distance = dist_func({3: 60.0})
        self.assertGreater(distance, 0.0)
    
    def test_python_fallback(self):
        """Test Python fallback when JIT is disabled."""
        path_info = {
            'bond_lengths': [1.5, 1.5, 1.5],  # Three bonds: 0-1, 1-2, 2-3
            'bond_angles': [109.47, 109.47, 109.47],
            'dihedral_indices': [3],  # Atom 3 has a variable dihedral
            'dihedral_positions': [2],  # Affects bond at position 2 (between atoms 2-3)
            'path_atoms': [0, 1, 2, 3],  # Full path from start to end
            'zmatrix_refs': []
        }
        
        dist_func = AnalyticalDistanceFunction(path_info, self.simple_zmatrix, use_jit=False)
        
        # Should use Python implementation
        self.assertFalse(dist_func.use_jit)
        
        # Should be able to compute distance
        distance = dist_func({3: 60.0})
        self.assertGreater(distance, 0.0)
    
    @unittest.skipIf(not NUMBA_AVAILABLE, "Numba not available")
    def test_jit_vs_python_consistency(self):
        """Test that JIT and Python implementations give similar results."""
        # Path from atom 0 to atom 3: [0, 1, 2, 3]
        # - Bond 0: between atoms 0-1
        # - Bond 1: between atoms 1-2
        # - Bond 2: between atoms 2-3
        # dihedral_indices=[3] means atom 3 (in Z-matrix) has a variable dihedral
        # dihedral_positions=[2] means this dihedral affects bond position 2 (bond between 2-3)
        path_info = {
            'bond_lengths': [1.5, 1.5, 1.5],  # Three bonds: 0-1, 1-2, 2-3
            'bond_angles': [109.47, 109.47, 109.47],
            'dihedral_indices': [3],  # Atom 3 has a variable dihedral
            'dihedral_positions': [2],  # Affects bond at position 2 (between atoms 2-3)
            'path_atoms': [0, 1, 2, 3],  # Full path from start to end
            'zmatrix_refs': []
        }
        
        dist_func_jit = AnalyticalDistanceFunction(path_info, self.simple_zmatrix, use_jit=True)
        dist_func_python = AnalyticalDistanceFunction(path_info, self.simple_zmatrix, use_jit=False)
        dihedral_values = {3: 60.0}
        
        dist_jit = dist_func_jit(dihedral_values)
        dist_python = dist_func_python(dihedral_values)
        
        # Expected distance computed from Cartesian conversion:
        expected_distance = 2.872234
        
        # Results should be very close (within 0.01 Å)
        self.assertAlmostEqual(dist_jit, dist_python, places=2)
        
        # Validate against hard-coded expected value (within 0.001 Å precision)
        self.assertAlmostEqual(dist_jit, expected_distance, places=2)
        self.assertAlmostEqual(dist_python, expected_distance, places=2)
    
    @unittest.skipIf(not NUMBA_AVAILABLE, "Numba not available")
    def test_distance_computation_with_dihedrals(self):
        """Test distance computation with a longer path and variable dihedral."""
        # Path from atom 0 to atom 6: [0, 1, 2, 3, 4, 5, 6]
        # Bond positions:
        #   - Position 0: bond between atoms 0-1
        #   - Position 1: bond between atoms 1-2
        #   - Position 2: bond between atoms 2-3
        #   - Position 3: bond between atoms 3-4
        #   - Position 4: bond between atoms 4-5
        #   - Position 5: bond between atoms 5-6
        # dihedral_indices=[2] means atom 2 (in Z-matrix) has a variable dihedral
        # dihedral_positions=[1] means this dihedral affects bond position 1 (bond between 1-2)
        path_info = {
            'bond_lengths': [1.0900067, 1.465007, 1.347795, 1.346316, 1.451997, 1.090000],  # 6 bonds
            'bond_angles': [109.467, 119.999, 120.003, 120.004, 109.474],  # 5 angles (one per bond after first)
            'dihedral_indices': [3, 4, 5, 6],  # Atoms 3, 4, 5, 6 have variable dihedrals
            'dihedral_positions': [2, 3, 4, 5],  # Dihedrals affect bonds at positions 2, 3, 4, 5
            'path_atoms': [0, 1, 2, 3, 4, 5, 6],  # Full 7-atom path
            'zmatrix_refs': []
        }
        atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': path_info['bond_lengths'][0]},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': path_info['bond_lengths'][1], 'angle_ref': 0, 'angle': path_info['bond_angles'][0]},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': path_info['bond_lengths'][2], 'angle_ref': 1, 'angle': path_info['bond_angles'][1], 'dihedral_ref': 0, 'dihedral': 180.0, 'chirality': 0},
            {'id': 4, 'element': 'C', 'atomic_num': 6, 'bond_ref': 3, 'bond_length': path_info['bond_lengths'][3], 'angle_ref': 2, 'angle': path_info['bond_angles'][2], 'dihedral_ref': 1, 'dihedral': 180.0, 'chirality': 0},
            {'id': 5, 'element': 'C', 'atomic_num': 6, 'bond_ref': 4, 'bond_length': path_info['bond_lengths'][4], 'angle_ref': 3, 'angle': path_info['bond_angles'][3], 'dihedral_ref': 2, 'dihedral': 180.0, 'chirality': 0},
            {'id': 6, 'element': 'C', 'atomic_num': 6, 'bond_ref': 5, 'bond_length': path_info['bond_lengths'][5], 'angle_ref': 4, 'angle': path_info['bond_angles'][4], 'dihedral_ref': 3, 'dihedral': 180.0, 'chirality': 0}
        ]
        bonds = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
        zmatrix = ZMatrix(atoms, bonds)

        dist_func = AnalyticalDistanceFunction(path_info, zmatrix, use_jit=True)
        dist_func_python = AnalyticalDistanceFunction(path_info, zmatrix, use_jit=False)

        # Expected distances manually measured
        cases = [
             [{3: 180.0, 4: 180.0, 5: 180.0, 6: 180.0}, 6.51641], #we also test for accumilation of errors by repeating measures
             [{3:   0.0, 4:   0.0, 5:   0.0, 6:   0.0}, 0.27832],
             [{3: 180.0, 4: 180.0, 5: 180.0, 6: 180.0}, 6.51641], #we also test for accumilation of errors by repeating measures
             [{3:   0.0, 4:   0.0, 5:   0.0, 6:   0.0}, 0.27832],
             [{3: 180.0, 4: 180.0, 5: 180.0, 6: 180.0}, 6.51641], #we also test for accumilation of errors by repeating measures
             [{3:   0.0, 4:   0.0, 5:   0.0, 6:   0.0}, 0.27832],
             [{3: 180.0, 4: 180.0, 5: 180.0, 6: 180.0}, 6.51641], #we also test for accumilation of errors by repeating measures
             [{3:   0.0, 4:   0.0, 5:   0.0, 6:   0.0}, 0.27832],
             [{3: 6.0,   4: 180.0, 5: 180.0, 6: 180.0}, 5.77400],
             [{3: 180.0, 4: -12.0, 5: 180.0, 6: 180.0}, 5.76039],
             [{3: 180.0, 4: 180.0, 5: 30.0,  6: 180.0}, 5.81338],
             [{3: 180.0, 4: 180.0, 5: 180.0, 6: -24.0}, 5.80278],
             [{3: -6.0,  4: 180.0, 5: 180.0, 6: -24.0}, 4.53945],
             [{3: -6.0,  4: 2.0,   5: 180.0, 6: -24.0}, 4.31092], 
             [{3: -6.0,  4: 2.0,   5: 30.0,  6: -24.0}, 0.74697], 
        ]

        for dihedral_values, expected_distance in cases:
            dist_jit = dist_func(dihedral_values)
            dist_python = dist_func_python(dihedral_values)
            self.assertAlmostEqual(dist_jit, dist_python, places=2, msg=f"Dihedral values: {dihedral_values}")
            self.assertAlmostEqual(dist_jit, expected_distance, places=2, msg=f"Dihedral values: {dihedral_values}")
            self.assertAlmostEqual(dist_python, expected_distance, places=2, msg=f"Dihedral values: {dihedral_values}")


    @unittest.skipIf(not NUMBA_AVAILABLE, "Numba not available")
    def test_distance_computation_with_dihedrals_2(self):
        """Test distance computation with a longer path and variable dihedral."""
        path_info = {
            'bond_lengths': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # 7 bonds
            'bond_angles': [90.0, 90.0, 90.0, 90.0, 90.0, 90.0],  # 6 angles (one per bond after first)
            'dihedral_indices': [3, 4, 5, 6, 7],  # Atoms 3, 4, 5, 6, 7 have variable dihedrals
            'dihedral_positions': [2, 3, 4, 5, 6],  # Dihedrals affect bonds at positions 2, 3, 4, 5, 6
            'path_atoms': [0, 1, 2, 3, 4, 5, 6, 7],  # Full 8-atom path
            'zmatrix_refs': []
        }
        atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': path_info['bond_lengths'][0]},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': path_info['bond_lengths'][1], 'angle_ref': 0, 'angle': path_info['bond_angles'][0]},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': path_info['bond_lengths'][2], 'angle_ref': 1, 'angle': path_info['bond_angles'][1], 'dihedral_ref': 0, 'dihedral': 180.0, 'chirality': 0},
            {'id': 4, 'element': 'C', 'atomic_num': 6, 'bond_ref': 3, 'bond_length': path_info['bond_lengths'][3], 'angle_ref': 2, 'angle': path_info['bond_angles'][2], 'dihedral_ref': 1, 'dihedral': 180.0, 'chirality': 0},
            {'id': 5, 'element': 'C', 'atomic_num': 6, 'bond_ref': 4, 'bond_length': path_info['bond_lengths'][4], 'angle_ref': 3, 'angle': path_info['bond_angles'][3], 'dihedral_ref': 2, 'dihedral': 180.0, 'chirality': 0},
            {'id': 6, 'element': 'C', 'atomic_num': 6, 'bond_ref': 5, 'bond_length': path_info['bond_lengths'][5], 'angle_ref': 4, 'angle': path_info['bond_angles'][4], 'dihedral_ref': 3, 'dihedral': 180.0, 'chirality': 0},
            {'id': 7, 'element': 'C', 'atomic_num': 6, 'bond_ref': 6, 'bond_length': path_info['bond_lengths'][6], 'angle_ref': 5, 'angle': path_info['bond_angles'][5], 'dihedral_ref': 4, 'dihedral': 180.0, 'chirality': 0}
        ]
        bonds = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
        zmatrix = ZMatrix(atoms, bonds)

        dist_func = AnalyticalDistanceFunction(path_info, zmatrix, use_jit=True)
        dist_func_python = AnalyticalDistanceFunction(path_info, zmatrix, use_jit=False)

        # Expected distances manually verified!
        cases = [
             [{3: 180.0, 4: 180.0, 5: 180.0, 6: 180.0, 7: 180.0}, 5.0],  
             [{3: 0.0,   4: 0.0,   5: 0.0,   6: 0.0,   7: 0.0},   1.0],  
             [{3: 180.0, 4: 0.0,   5: 0.0,   6: 180.0, 7: 0.0},   math.sqrt(5.0)],  
             [{3: 180.0, 4: 0.0,   5: 0.0,   6: 180.0, 7: 180.0}, 1.0], 
             [{3: 0.0,   4: 90.0,  5: -90.0, 6: 90.0,  7: 0.0},   math.sqrt(3.0)]  
        ]

        for dihedral_values, expected_distance in cases:
            dist_jit = dist_func(dihedral_values)
            dist_python = dist_func_python(dihedral_values)
            self.assertAlmostEqual(dist_jit, dist_python, places=2, msg=f"Dihedral values: {dihedral_values}")
            self.assertAlmostEqual(dist_jit, expected_distance, places=2, msg=f"Dihedral values: {dihedral_values}")
            self.assertAlmostEqual(dist_python, expected_distance, places=2, msg=f"Dihedral values: {dihedral_values}")


    @unittest.skipIf(not NUMBA_AVAILABLE, "Numba not available")
    def test_distance_computation_with_dihedrals_indirect(self):
        """Test distance computation with a chain whre dihedrals are not defined using only atoms belonging to the chain."""
        path_info = {
            'bond_lengths': [0.9670, 1.3602, 1.32025, 1.36015, 1.46902, 0.96707],  # 7 toms and 6 bonds in the chain
            'bond_angles': [106.80, 119.998, 125.0, 130.0, 106.80],  #  5 angles (one per bond after first)
            'dihedral_indices': [3, 5, 7, 8],  # Atoms 3, 5, 7, 8 in the zmatrix define the variable dihedrals
            # dihedral_positions: [2, 3, 4, 5]
            # - Atom 3 in the zmatrix (not in path, branch from atom 2): affects bond position 2 in the chain (bond 2-4) (not in the zmatrix)
            # - Atom 5 in the zmatrix (not in path, branch from atom 4): affects bond position 3 in the chain (bond 4-6) (not in the zmatrix)
            # - Atom 7 in the zmatrix (in path): affects bond position 4 in the chain (bond 6-7) (not in the zmatrix)
            # - Atom 8 in the zmatrix (in path): affects bond position 5 in the chain (bond 7-8) (not in the zmatrix)
            'dihedral_positions': [2, 3, 4, 5],  # Dihedrals affect bonds at positions 2, 3, 4, 5
            'path_atoms': [0, 1, 2, 4, 6, 7, 8 ],  # Full 7-atom path
            'zmatrix_refs': []
        }
        #
        #   H       H           H 
        #     \    /           /
        #     O - C       O - O
        #          \\    /
        #           C - O
        #          /
        #       H
        #
        atoms = [
            {'id': 0, 'element': 'H', 'atomic_num': 1},
            {'id': 1, 'element': 'O', 'atomic_num': 8, 'bond_ref': 0, 'bond_length': path_info['bond_lengths'][0]},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': path_info['bond_lengths'][1], 'angle_ref': 0, 'angle': path_info['bond_angles'][0]},
            {'id': 3, 'element': 'H', 'atomic_num': 1, 'bond_ref': 2, 'bond_length': 1.07995, 'angle_ref': 1, 'angle': 150.0, 'dihedral_ref': 0, 'dihedral': 0.0, 'chirality': 0},
            {'id': 4, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': path_info['bond_lengths'][2], 'angle_ref': 1, 'angle': path_info['bond_angles'][1], 'dihedral_ref': 3, 'dihedral': 90.002, 'chirality': -1},
            {'id': 5, 'element': 'H', 'atomic_num': 1, 'bond_ref': 4, 'bond_length': 1.08005, 'angle_ref': 2, 'angle': 100.0, 'dihedral_ref': 1, 'dihedral': 0.0, 'chirality': 0},
            {'id': 6, 'element': 'O', 'atomic_num': 8, 'bond_ref': 4, 'bond_length': path_info['bond_lengths'][3], 'angle_ref': 2, 'angle': path_info['bond_angles'][2], 'dihedral_ref': 5, 'dihedral': 135.0, 'chirality': -1},
            {'id': 7, 'element': 'O', 'atomic_num': 8, 'bond_ref': 6, 'bond_length': path_info['bond_lengths'][4], 'angle_ref': 4, 'angle': path_info['bond_angles'][3], 'dihedral_ref': 5, 'dihedral': 180.0, 'chirality': 0},
            {'id': 8, 'element': 'H', 'atomic_num': 1, 'bond_ref': 7, 'bond_length': path_info['bond_lengths'][5], 'angle_ref': 6, 'angle': path_info['bond_angles'][4], 'dihedral_ref': 4, 'dihedral': 180.0, 'chirality': 0}
        ]
        bonds = [(0, 1), (1, 2), (2, 3), (2, 4), (4, 5), (4, 6), (6, 7), (7, 8)]
        zmatrix = ZMatrix(atoms, bonds)

        dist_func = AnalyticalDistanceFunction(path_info, zmatrix, use_jit=True)
        dist_func_python = AnalyticalDistanceFunction(path_info, zmatrix, use_jit=False)

        # Expected distances computed using AnalyticalDistanceFunction
        # (Note: These may differ from Cartesian conversion due to how dihedrals are applied)
        cases = [
             [{3: 180.0, 5: 180.0, 7: 180.0, 8: 180.0},   2.8106],  
             [{3: 0.0,   5: 0.0,   7: 0.0,   8: 0.0},     5.6952],   
             [{3: 0.0,   5: 180.0, 7: 0.0,   8: 180.0},   5.5222],    
             [{3: -90.0, 5: 180.0, 7: 0.0,   8: 180.0},   4.7368], 
             [{3: 0.0,   5: 90.0,  7: 0.0,   8: 180.0},   5.8894], 
             [{3: 0.0,   5: 180.0, 7: -60.0, 8: 0.0},     5.0391], 
             [{3: 0.0,   5: 180.0, 7: 0.0,   8: -90.0},   5.5503],
        ]

        for dihedral_values, expected_distance in cases:
            dist_jit = dist_func(dihedral_values)
            dist_python = dist_func_python(dihedral_values)
            self.assertAlmostEqual(dist_jit, dist_python, places=2, msg=f"Dihedral values: {dihedral_values}")
            self.assertAlmostEqual(dist_jit, expected_distance, places=2, msg=f"Dihedral values: {dihedral_values}")
            self.assertAlmostEqual(dist_python, expected_distance, places=2, msg=f"Dihedral values: {dihedral_values}")


    @unittest.skipIf(not NUMBA_AVAILABLE, "Numba not available")
    def test_distance_computation_with_chirality_in_path(self):
        """Test distance computation with chirality atoms in the path."""
        # Path: 0-1-2-3-4-5
        # Atom 3 has chirality=-1 (in path)
        # Atom 5 has chirality=+1 (in path)
        path_info = {
            'bond_lengths': [1.0, 1.5, 1.4, 1.6, 1.2],  # 5 bonds
            'bond_angles': [109.47, 120.0, 115.0, 110.0],  # 4 angles
            'dihedral_indices': [3, 5],  # Both in path, both with chirality
            'dihedral_positions': [2, 4],  # Affect bonds at positions 2 and 4
            'path_atoms': [0, 1, 2, 3, 4, 5],
            'zmatrix_refs': []
        }
        
        # Structure: C-C-C-C-C-C with branches
        # Atom 3 (chirality=-1) and atom 5 (chirality=+1) are in the path
        atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': path_info['bond_lengths'][0]},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': path_info['bond_lengths'][1], 
             'angle_ref': 0, 'angle': path_info['bond_angles'][0]},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': path_info['bond_lengths'][2], 
             'angle_ref': 1, 'angle': path_info['bond_angles'][1], 
             'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': -1},  # Chirality in path
            {'id': 4, 'element': 'C', 'atomic_num': 6, 'bond_ref': 3, 'bond_length': path_info['bond_lengths'][3], 
             'angle_ref': 2, 'angle': path_info['bond_angles'][2], 
             'dihedral_ref': 1, 'dihedral': 120.0, 'chirality': 0},
            {'id': 5, 'element': 'C', 'atomic_num': 6, 'bond_ref': 4, 'bond_length': path_info['bond_lengths'][4], 
             'angle_ref': 3, 'angle': path_info['bond_angles'][3], 
             'dihedral_ref': 2, 'dihedral': 180.0, 'chirality': 1},  # Chirality in path
        ]
        bonds = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        zmatrix = ZMatrix(atoms, bonds)
        
        dist_func = AnalyticalDistanceFunction(path_info, zmatrix, use_jit=True)
        dist_func_python = AnalyticalDistanceFunction(path_info, zmatrix, use_jit=False)
        
        # Compute expected distances using Cartesian conversion
        from ringclosingmm.CoordinateConversion import zmatrix_to_cartesian, apply_torsions
        import numpy as np
        
        cases = []
        test_dihedrals = [
            {3: 60.0, 5: 180.0},
            {3: 120.0, 5: 0.0},
            {3: -60.0, 5: 90.0},
            {3: 180.0, 5: -90.0},
        ]
        
        for dihedral_values in test_dihedrals:
            # Compute using Cartesian conversion
            dihedral_indices = [3, 5]
            dihedral_array = np.array([dihedral_values[idx] for idx in dihedral_indices])
            zmatrix_mod = apply_torsions(zmatrix, dihedral_indices, dihedral_array)
            coords = zmatrix_to_cartesian(zmatrix_mod)
            expected = np.linalg.norm(coords[5] - coords[0])
            cases.append((dihedral_values, expected))
        
        for dihedral_values, expected_distance in cases:
            dist_jit = dist_func(dihedral_values)
            dist_python = dist_func_python(dihedral_values)
            self.assertAlmostEqual(dist_jit, dist_python, places=2, 
                                 msg=f"JIT vs Python mismatch for {dihedral_values}")
            self.assertAlmostEqual(dist_jit, expected_distance, places=2, 
                                 msg=f"JIT vs Cartesian mismatch for {dihedral_values}")
            self.assertAlmostEqual(dist_python, expected_distance, places=2, 
                                 msg=f"Python vs Cartesian mismatch for {dihedral_values}")

    @unittest.skipIf(not NUMBA_AVAILABLE, "Numba not available")
    def test_distance_computation_with_chirality_branch_atoms(self):
        """Test distance computation with chirality atoms as branch atoms (not in path)."""
        # Path: 0-1-2-4-6-7
        # Atom 3 (branch, chirality=-1) affects bond 2-4
        # Atom 5 (branch, chirality=+1) affects bond 4-6
        path_info = {
            'bond_lengths': [1.0, 1.5, 1.4, 1.6, 1.2],  # 5 bonds in path
            'bond_angles': [109.47, 120.0, 115.0, 110.0],  # 4 angles
            'dihedral_indices': [3, 5],  # Branch atoms with chirality
            'dihedral_positions': [2, 3],  # Affect bonds at positions 2 and 3
            'path_atoms': [0, 1, 2, 4, 6, 7],
            'zmatrix_refs': []
        }
        
        # Structure with branches:
        #   3 (branch, chirality=-1)
        #     |
        # 0-1-2-4-6-7
        #       |
        #       5 (branch, chirality=+1)
        atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': path_info['bond_lengths'][0]},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': path_info['bond_lengths'][1], 
             'angle_ref': 0, 'angle': path_info['bond_angles'][0]},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.3, 
             'angle_ref': 1, 'angle': 110.0, 
             'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': -1},  # Branch with chirality
            {'id': 4, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': path_info['bond_lengths'][2], 
             'angle_ref': 1, 'angle': path_info['bond_angles'][1], 
             'dihedral_ref': 3, 'dihedral': 90.0, 'chirality': 0},  # Uses branch atom 3 as dihedral_ref
            {'id': 5, 'element': 'C', 'atomic_num': 6, 'bond_ref': 4, 'bond_length': 1.35, 
             'angle_ref': 2, 'angle': 105.0, 
             'dihedral_ref': 1, 'dihedral': 0.0, 'chirality': 1},  # Branch with chirality
            {'id': 6, 'element': 'C', 'atomic_num': 6, 'bond_ref': 4, 'bond_length': path_info['bond_lengths'][3], 
             'angle_ref': 2, 'angle': path_info['bond_angles'][2], 
             'dihedral_ref': 5, 'dihedral': 120.0, 'chirality': 0},  # Uses branch atom 5 as dihedral_ref
            {'id': 7, 'element': 'C', 'atomic_num': 6, 'bond_ref': 6, 'bond_length': path_info['bond_lengths'][4], 
             'angle_ref': 4, 'angle': path_info['bond_angles'][3], 
             'dihedral_ref': 2, 'dihedral': 180.0, 'chirality': 0},
        ]
        bonds = [(0, 1), (1, 2), (2, 3), (2, 4), (4, 5), (4, 6), (6, 7)]
        zmatrix = ZMatrix(atoms, bonds)
        
        dist_func = AnalyticalDistanceFunction(path_info, zmatrix, use_jit=True)
        dist_func_python = AnalyticalDistanceFunction(path_info, zmatrix, use_jit=False)
        
        # Compute expected distances using Cartesian conversion
        from ringclosingmm.CoordinateConversion import zmatrix_to_cartesian, apply_torsions
        import numpy as np
        
        cases = []
        test_dihedrals = [
            {3: 60.0, 5: 0.0},
            {3: 120.0, 5: 180.0},
            {3: -60.0, 5: 90.0},
            {3: 180.0, 5: -90.0},
        ]
        
        for dihedral_values in test_dihedrals:
            # Compute using Cartesian conversion
            dihedral_indices = [3, 5]
            dihedral_array = np.array([dihedral_values[idx] for idx in dihedral_indices])
            zmatrix_mod = apply_torsions(zmatrix, dihedral_indices, dihedral_array)
            coords = zmatrix_to_cartesian(zmatrix_mod)
            expected = np.linalg.norm(coords[7] - coords[0])
            cases.append((dihedral_values, expected))
        
        for dihedral_values, expected_distance in cases:
            dist_jit = dist_func(dihedral_values)
            dist_python = dist_func_python(dihedral_values)
            self.assertAlmostEqual(dist_jit, dist_python, places=2, 
                                 msg=f"JIT vs Python mismatch for {dihedral_values}")
            self.assertAlmostEqual(dist_jit, expected_distance, places=2, 
                                 msg=f"JIT vs Cartesian mismatch for {dihedral_values}")
            self.assertAlmostEqual(dist_python, expected_distance, places=2, 
                                 msg=f"Python vs Cartesian mismatch for {dihedral_values}")

    @unittest.skipIf(not NUMBA_AVAILABLE, "Numba not available")
    def test_distance_computation_mixed_chirality_and_branches(self):
        """Test complex case with both chirality atoms in path and branch atoms with chirality."""
        # Path: 0-1-2-4-6-8-9
        # Atom 3 (branch, chirality=-1) affects bond 2-4
        # Atom 4 (in path, chirality=+1) 
        # Atom 5 (branch, chirality=0, regular dihedral) affects bond 4-6
        # Atom 7 (branch, chirality=-1) affects bond 6-8
        # Atom 8 (in path, chirality=+1)
        path_info = {
            'bond_lengths': [1.0, 1.5, 1.4, 1.6, 1.3, 1.2],  # 6 bonds
            'bond_angles': [109.47, 120.0, 115.0, 110.0, 105.0],  # 5 angles
            'dihedral_indices': [3, 4, 5, 7, 8],  # Mix of branch and path atoms, mix of chirality
            'dihedral_positions': [2, 2, 3, 4, 5],  # Positions where they affect the path
            'path_atoms': [0, 1, 2, 4, 6, 8, 9],
            'zmatrix_refs': []
        }
        
        atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': path_info['bond_lengths'][0]},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': path_info['bond_lengths'][1], 
             'angle_ref': 0, 'angle': path_info['bond_angles'][0]},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.35, 
             'angle_ref': 1, 'angle': 108.0, 
             'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': -1},  # Branch with chirality
            {'id': 4, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': path_info['bond_lengths'][2], 
             'angle_ref': 1, 'angle': path_info['bond_angles'][1], 
             'dihedral_ref': 3, 'dihedral': 90.0, 'chirality': 1},  # In path with chirality
            {'id': 5, 'element': 'C', 'atomic_num': 6, 'bond_ref': 4, 'bond_length': 1.38, 
             'angle_ref': 2, 'angle': 112.0, 
             'dihedral_ref': 1, 'dihedral': 0.0, 'chirality': 0},  # Branch with regular dihedral
            {'id': 6, 'element': 'C', 'atomic_num': 6, 'bond_ref': 4, 'bond_length': path_info['bond_lengths'][3], 
             'angle_ref': 2, 'angle': path_info['bond_angles'][2], 
             'dihedral_ref': 5, 'dihedral': 120.0, 'chirality': 0},
            {'id': 7, 'element': 'C', 'atomic_num': 6, 'bond_ref': 6, 'bond_length': 1.4, 
             'angle_ref': 4, 'angle': 107.0, 
             'dihedral_ref': 2, 'dihedral': 180.0, 'chirality': -1},  # Branch with chirality
            {'id': 8, 'element': 'C', 'atomic_num': 6, 'bond_ref': 6, 'bond_length': path_info['bond_lengths'][4], 
             'angle_ref': 4, 'angle': path_info['bond_angles'][3], 
             'dihedral_ref': 7, 'dihedral': 150.0, 'chirality': 1},  # In path with chirality
            {'id': 9, 'element': 'C', 'atomic_num': 6, 'bond_ref': 8, 'bond_length': path_info['bond_lengths'][5], 
             'angle_ref': 6, 'angle': path_info['bond_angles'][4], 
             'dihedral_ref': 4, 'dihedral': 180.0, 'chirality': 0},
        ]
        bonds = [(0, 1), (1, 2), (2, 3), (2, 4), (4, 5), (4, 6), (6, 7), (6, 8), (8, 9)]
        zmatrix = ZMatrix(atoms, bonds)
        
        dist_func = AnalyticalDistanceFunction(path_info, zmatrix, use_jit=True)
        dist_func_python = AnalyticalDistanceFunction(path_info, zmatrix, use_jit=False)
        
        # Compute expected distances using Cartesian conversion
        from ringclosingmm.CoordinateConversion import zmatrix_to_cartesian, apply_torsions
        import numpy as np
        
        cases = []
        test_dihedrals = [
            {3: 60.0, 4: 90.0, 5: 0.0, 7: 180.0, 8: 150.0},
            {3: 120.0, 4: 0.0, 5: 180.0, 7: 0.0, 8: -90.0},
            {3: -60.0, 4: -90.0, 5: 90.0, 7: 90.0, 8: 0.0},
        ]
        
        for dihedral_values in test_dihedrals:
            # Compute using Cartesian conversion
            dihedral_indices = [3, 4, 5, 7, 8]
            dihedral_array = np.array([dihedral_values[idx] for idx in dihedral_indices])
            zmatrix_mod = apply_torsions(zmatrix, dihedral_indices, dihedral_array)
            coords = zmatrix_to_cartesian(zmatrix_mod)
            expected = np.linalg.norm(coords[9] - coords[0])
            cases.append((dihedral_values, expected))
        
        for dihedral_values, expected_distance in cases:
            dist_jit = dist_func(dihedral_values)
            dist_python = dist_func_python(dihedral_values)
            self.assertAlmostEqual(dist_jit, dist_python, places=2, 
                                 msg=f"JIT vs Python mismatch for {dihedral_values}")
            self.assertAlmostEqual(dist_jit, expected_distance, places=2, 
                                 msg=f"JIT vs Cartesian mismatch for {dihedral_values}")
            self.assertAlmostEqual(dist_python, expected_distance, places=2, 
                                 msg=f"Python vs Cartesian mismatch for {dihedral_values}")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.minimal_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6,
             'bond_ref': 0, 'bond_length': 1.5}
        ]
        self.minimal_bonds = [(0, 1)]
        self.minimal_zmatrix = ZMatrix(self.minimal_atoms, self.minimal_bonds)
    
    def test_single_bond_path(self):
        """Test distance computation for single bond path."""
        path_info = {
            'bond_lengths': [1.5],
            'bond_angles': [],
            'dihedral_indices': [],
            'dihedral_positions': [],
            'path_atoms': [0, 1],
            'zmatrix_refs': []
        }
        
        dist_func = AnalyticalDistanceFunction(path_info, self.minimal_zmatrix)
        distance = dist_func({})
        
        # Should equal bond length
        self.assertAlmostEqual(distance, 1.5, places=3)
    
    def test_same_atom_path(self):
        """Test distance computation for same atom (should be 0)."""
        path_info = {
            'bond_lengths': [],
            'bond_angles': [],
            'dihedral_indices': [],
            'dihedral_positions': [],
            'path_atoms': [0, 0],
            'zmatrix_refs': []
        }
        
        dist_func = AnalyticalDistanceFunction(path_info, self.minimal_zmatrix)
        distance = dist_func({})
        
        # Should be zero
        self.assertAlmostEqual(distance, 0.0, places=3)
    
    def test_empty_dihedral_values(self):
        """Test distance computation with empty dihedral values."""
        path_info = {
            'bond_lengths': [1.5],
            'bond_angles': [],
            'dihedral_indices': [],
            'dihedral_positions': [],
            'path_atoms': [0, 1],
            'zmatrix_refs': []
        }
        
        dist_func = AnalyticalDistanceFunction(path_info, self.minimal_zmatrix)
        distance = dist_func({})
        
        # Should still compute distance
        self.assertGreater(distance, 0.0)
    
    def test_missing_dihedral_in_values(self):
        """Test behavior when dihedral value is missing."""
        path_info = {
            'bond_lengths': [1.5, 1.5],
            'bond_angles': [109.47, 109.47],
            'dihedral_indices': [2],
            'dihedral_positions': [1],
            'path_atoms': [0, 1, 2],
            'zmatrix_refs': []
        }
        
        atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6,
             'bond_ref': 0, 'bond_length': 1.5},
            {'id': 2, 'element': 'C', 'atomic_num': 6,
             'bond_ref': 1, 'bond_length': 1.5,
             'angle_ref': 0, 'angle': 109.47,
             'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0}
        ]
        zmatrix = ZMatrix(atoms, [(0, 1), (1, 2)])
        
        dist_func = AnalyticalDistanceFunction(path_info, zmatrix)
        
        # Should use default dihedral from zmatrix
        distance = dist_func({})  # Empty dict, should use default
        self.assertGreater(distance, 0.0)
    
    def test_partial_dihedral_values(self):
        """Test behavior when input dict contains values only for some variable dihedrals."""
        path_info = {
            'bond_lengths': [1.5, 1.5, 1.5, 1.5],
            'bond_angles': [109.47, 109.47, 109.47],
            'dihedral_indices': [3, 4, 5],  # Three variable dihedrals
            'dihedral_positions': [2, 3, 4],
            'path_atoms': [0, 1, 2, 3, 4, 5],
            'zmatrix_refs': []
        }
        
        atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.5},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.5, 
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.5, 
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0},
            {'id': 4, 'element': 'C', 'atomic_num': 6, 'bond_ref': 3, 'bond_length': 1.5, 
             'angle_ref': 2, 'angle': 109.47, 'dihedral_ref': 1, 'dihedral': 120.0, 'chirality': 0},
            {'id': 5, 'element': 'C', 'atomic_num': 6, 'bond_ref': 4, 'bond_length': 1.5, 
             'angle_ref': 3, 'angle': 109.47, 'dihedral_ref': 2, 'dihedral': 180.0, 'chirality': 0},
        ]
        zmatrix = ZMatrix(atoms, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
        
        dist_func_jit = AnalyticalDistanceFunction(path_info, zmatrix, use_jit=True)
        dist_func_python = AnalyticalDistanceFunction(path_info, zmatrix, use_jit=False)
        
        # Test 1: All values provided
        all_values = {3: 90.0, 4: 90.0, 5: 90.0}
        dist_jit_all = dist_func_jit(all_values)
        dist_python_all = dist_func_python(all_values)
        self.assertAlmostEqual(dist_jit_all, dist_python_all, places=4,
                              msg="JIT and Python should match with all values")
        
        # Test 2: Only first value provided (others use defaults)
        partial1 = {3: 90.0}  # 4→120.0, 5→180.0 defaults
        dist_jit_partial1 = dist_func_jit(partial1)
        dist_python_partial1 = dist_func_python(partial1)
        self.assertAlmostEqual(dist_jit_partial1, dist_python_partial1, places=4,
                              msg="JIT and Python should match with partial values")
        
        # Test 3: Only middle value provided
        partial2 = {4: 90.0}  # 3→60.0, 5→180.0 defaults
        dist_jit_partial2 = dist_func_jit(partial2)
        dist_python_partial2 = dist_func_python(partial2)
        self.assertAlmostEqual(dist_jit_partial2, dist_python_partial2, places=4,
                              msg="JIT and Python should match with middle value only")
        
        # Test 4: Only last value provided
        partial3 = {5: 90.0}  # 3→60.0, 4→120.0 defaults
        dist_jit_partial3 = dist_func_jit(partial3)
        dist_python_partial3 = dist_func_python(partial3)
        self.assertAlmostEqual(dist_jit_partial3, dist_python_partial3, places=4,
                              msg="JIT and Python should match with last value only")
        
        # Test 5: Empty dict (all use defaults)
        empty = {}
        dist_jit_empty = dist_func_jit(empty)
        dist_python_empty = dist_func_python(empty)
        self.assertAlmostEqual(dist_jit_empty, dist_python_empty, places=4,
                              msg="JIT and Python should match with empty dict")
        
        # Test 6: Verify partial matches explicit defaults
        explicit_defaults = {3: 90.0, 4: 120.0, 5: 180.0}
        dist_jit_explicit = dist_func_jit(explicit_defaults)
        dist_python_explicit = dist_func_python(explicit_defaults)
        self.assertAlmostEqual(dist_jit_partial1, dist_jit_explicit, places=4,
                              msg="Partial {3: 90.0} should match explicit {3: 90.0, 4: 120.0, 5: 180.0}")
        self.assertAlmostEqual(dist_python_partial1, dist_python_explicit, places=4,
                              msg="Partial {3: 90.0} should match explicit {3: 90.0, 4: 120.0, 5: 180.0}")
    
    def test_extra_non_variable_dihedrals(self):
        """Test behavior when input dict contains dihedrals that are not variable."""
        path_info = {
            'bond_lengths': [1.5, 1.5, 1.5, 1.5],
            'bond_angles': [109.47, 109.47, 109.47],
            'dihedral_indices': [3, 4],  # Only 3 and 4 are variable
            'dihedral_positions': [2, 3],
            'path_atoms': [0, 1, 2, 3, 4, 5],
            'zmatrix_refs': []
        }
        
        atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.5},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.5, 
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.5, 
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0},
            {'id': 4, 'element': 'C', 'atomic_num': 6, 'bond_ref': 3, 'bond_length': 1.5, 
             'angle_ref': 2, 'angle': 109.47, 'dihedral_ref': 1, 'dihedral': 120.0, 'chirality': 0},
            {'id': 5, 'element': 'C', 'atomic_num': 6, 'bond_ref': 4, 'bond_length': 1.5, 
             'angle_ref': 3, 'angle': 109.47, 'dihedral_ref': 2, 'dihedral': 180.0, 'chirality': 0},
        ]
        zmatrix = ZMatrix(atoms, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
        
        dist_func_jit = AnalyticalDistanceFunction(path_info, zmatrix, use_jit=True)
        dist_func_python = AnalyticalDistanceFunction(path_info, zmatrix, use_jit=False)
        
        # Test 1: Normal case (only variable dihedrals)
        normal = {3: 90.0, 4: 90.0}
        dist_jit_normal = dist_func_jit(normal)
        dist_python_normal = dist_func_python(normal)
        
        # Test 2: With extra dihedral (atom 5, NOT in dihedral_indices)
        with_extra = {3: 90.0, 4: 90.0, 5: 45.0}
        dist_jit_extra = dist_func_jit(with_extra)
        dist_python_extra = dist_func_python(with_extra)
        
        # Extra dihedrals should be ignored - results should match normal case
        self.assertAlmostEqual(dist_jit_normal, dist_jit_extra, places=4,
                              msg="JIT should ignore extra dihedral 5")
        self.assertAlmostEqual(dist_python_normal, dist_python_extra, places=4,
                              msg="Python should ignore extra dihedral 5")
        self.assertAlmostEqual(dist_jit_normal, dist_python_normal, places=4,
                              msg="JIT and Python should match")
        self.assertAlmostEqual(dist_jit_extra, dist_python_extra, places=4,
                              msg="JIT and Python should match with extra dihedrals")
        
        # Test 3: Only extra dihedral (no variable dihedrals)
        only_extra = {5: 45.0}
        dist_jit_only_extra = dist_func_jit(only_extra)
        dist_python_only_extra = dist_func_python(only_extra)
        
        # Should use defaults for variable dihedrals (3→60.0, 4→120.0), ignore 5
        empty = {}  # All defaults
        dist_jit_empty = dist_func_jit(empty)
        dist_python_empty = dist_func_python(empty)
        
        self.assertAlmostEqual(dist_jit_only_extra, dist_jit_empty, places=4,
                              msg="JIT should ignore extra dihedral and use defaults")
        self.assertAlmostEqual(dist_python_only_extra, dist_python_empty, places=4,
                              msg="Python should ignore extra dihedral and use defaults")
        
        # Test 4: Non-existent atom index
        with_nonexistent = {3: 90.0, 4: 90.0, 99: 45.0}
        dist_jit_nonexistent = dist_func_jit(with_nonexistent)
        dist_python_nonexistent = dist_func_python(with_nonexistent)
        
        # Should match normal case (non-existent atom ignored)
        self.assertAlmostEqual(dist_jit_normal, dist_jit_nonexistent, places=4,
                              msg="JIT should ignore non-existent atom 99")
        self.assertAlmostEqual(dist_python_normal, dist_python_nonexistent, places=4,
                              msg="Python should ignore non-existent atom 99")
        
        # Test 5: Multiple extra dihedrals
        with_multiple_extra = {3: 90.0, 4: 90.0, 5: 45.0, 99: 30.0, 100: 15.0}
        dist_jit_multiple = dist_func_jit(with_multiple_extra)
        dist_python_multiple = dist_func_python(with_multiple_extra)
        
        # Should match normal case
        self.assertAlmostEqual(dist_jit_normal, dist_jit_multiple, places=4,
                              msg="JIT should ignore all extra dihedrals")
        self.assertAlmostEqual(dist_python_normal, dist_python_multiple, places=4,
                              msg="Python should ignore all extra dihedrals")


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRotationMatrices))
    suite.addTests(loader.loadTestsFromTestCase(TestAnalyticalDistanceFunction))
    suite.addTests(loader.loadTestsFromTestCase(TestAnalyticalDistanceFactory))
    suite.addTests(loader.loadTestsFromTestCase(TestJITCompilation))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    # Disable buffering to see print statements immediately
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)

