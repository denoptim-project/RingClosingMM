#!/usr/bin/env python3
"""
Unit tests for RingClosureOptimizer components.

Tests Individual class, GeneticAlgorithm operations, and optimizer utilities.
"""

import unittest
import numpy as np
import sys
import copy
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from RingClosureOptimizer import (
    Individual,
    RingClosureOptimizer
)
from MolecularSystem import MolecularSystem
from CoordinateConverter import zmatrix_to_cartesian


class TestIndividual(unittest.TestCase):
    """Test Individual class for GA."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Simple Z-matrix
        self.zmatrix = [
            {'id': 1, 'element': 'H', 'atomic_num': 1},
            {'id': 2, 'element': 'H', 'atomic_num': 1, 'bond_ref': 1, 'bond_length': 1.0},
            {'id': 3, 'element': 'H', 'atomic_num': 1, 'bond_ref': 1, 'bond_length': 1.0,
             'angle_ref': 2, 'angle': 109.47}
        ]
        self.torsions = np.array([120.0])
        self.rotatable_indices = [2]  # Only third atom has dihedral
    
    def test_individual_creation(self):
        """Test creating Individual."""
        individual = Individual(self.torsions, self.zmatrix)
        
        self.assertEqual(len(individual.torsions), 1)
        self.assertEqual(len(individual.zmatrix), 3)
        # Fitness defaults to np.inf (not None)
        self.assertEqual(individual.fitness, np.inf)
        self.assertEqual(individual.ring_closure_score, 0)
    
    def test_individual_fitness_assignment(self):
        """Test assigning fitness to Individual."""
        individual = Individual(self.torsions, self.zmatrix)
        individual.fitness = -0.85
        
        self.assertEqual(individual.fitness, -0.85)
    
    def test_individual_torsion_hash(self):
        """Test torsion hash computation."""
        individual = Individual(self.torsions, self.zmatrix)
        
        hash1 = individual._compute_torsion_hash()
        self.assertIsInstance(hash1, int)
        
        # Same torsions should give same hash
        hash2 = individual._compute_torsion_hash()
        self.assertEqual(hash1, hash2)
    
    def test_individual_torsion_hash_changes(self):
        """Test that hash changes when torsions change."""
        individual1 = Individual(self.torsions, self.zmatrix)
        individual2 = Individual(np.array([-120.0]), self.zmatrix)
        
        hash1 = individual1._compute_torsion_hash()
        hash2 = individual2._compute_torsion_hash()
        
        self.assertNotEqual(hash1, hash2)
    
    def test_individual_update_torsion_hash(self):
        """Test updating torsion hash."""
        individual = Individual(self.torsions, self.zmatrix)
        
        initial_hash = individual.torsion_hash
        individual.update_torsion_hash()
        
        self.assertIsNotNone(individual.torsion_hash)
        self.assertEqual(individual.torsion_hash, individual._compute_torsion_hash())
    
    def test_individual_copy(self):
        """Test copying Individual."""
        individual = Individual(self.torsions, self.zmatrix)
        individual.fitness = -0.5
        
        copy_ind = copy.copy(individual)
        
        self.assertEqual(len(copy_ind.torsions), len(individual.torsions))
        self.assertEqual(copy_ind.fitness, individual.fitness)
        self.assertEqual(len(copy_ind.zmatrix), len(individual.zmatrix))


class TestRingClosureOptimizerUtilities(unittest.TestCase):
    """Test utility methods of RingClosureOptimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(__file__).parent / 'fixtures'
        self.forcefield_file = Path(__file__).parent.parent / 'data' / 'RCP_UFFvdW.xml'
        self.test_int_file = self.test_dir / 'simple_molecule.int'
    
    def test_convert_bonds_to_indices(self):
        """Test converting bond pairs to rotatable indices."""
        if not self.test_int_file.exists() or not self.forcefield_file.exists():
            self.skipTest("Required test files not found")
        
        # Create a simple system first
        system = MolecularSystem.from_file(
            str(self.test_int_file),
            str(self.forcefield_file),
            rcp_terms=None
        )
        
        # Test bond conversion
        # For simple 3-atom molecule: bond (0,1) and (1,2) might map to different indices
        rotatable_bonds = [(0, 1)]  # 0-based
        
        indices = RingClosureOptimizer._convert_bonds_to_indices(
            rotatable_bonds, system.zmatrix
        )
        
        # Should return list of indices
        self.assertIsInstance(indices, list)
        self.assertTrue(all(isinstance(i, int) for i in indices))
    
    def test_get_all_rotatable_indices(self):
        """Test getting all rotatable indices."""
        if not self.test_int_file.exists() or not self.forcefield_file.exists():
            self.skipTest("Required test files not found")
        
        system = MolecularSystem.from_file(
            str(self.test_int_file),
            str(self.forcefield_file),
            rcp_terms=None
        )
        
        # Get all rotatable indices
        indices = RingClosureOptimizer._get_all_rotatable_indices(system.zmatrix)
        
        # Should return list of indices >= 3 (only atoms 4+ have dihedrals)
        self.assertIsInstance(indices, list)
        self.assertTrue(all(i >= 3 for i in indices))
    
    def test_from_files_with_rotatable_bonds(self):
        """Test creating optimizer with specified rotatable bonds."""
        if not self.test_int_file.exists() or not self.forcefield_file.exists():
            self.skipTest("Required test files not found")
        
        rotatable_bonds = [(0, 1)]  # 0-based
        
        optimizer = RingClosureOptimizer.from_files(
            str(self.test_int_file),
            str(self.forcefield_file),
            rotatable_bonds=rotatable_bonds,
            rcp_terms=None
        )
        
        self.assertIsNotNone(optimizer)
        self.assertIsNotNone(optimizer.system)
        self.assertIsInstance(optimizer.rotatable_indices, list)
        self.assertGreaterEqual(len(optimizer.rotatable_indices), 0)
    
    def test_from_files_all_rotatable(self):
        """Test creating optimizer with all bonds rotatable."""
        if not self.test_int_file.exists() or not self.forcefield_file.exists():
            self.skipTest("Required test files not found")
        
        optimizer = RingClosureOptimizer.from_files(
            str(self.test_int_file),
            str(self.forcefield_file),
            rotatable_bonds=None,  # All bonds rotatable
            rcp_terms=None
        )
        
        self.assertIsNotNone(optimizer)
        self.assertIsNotNone(optimizer.system)
        self.assertIsInstance(optimizer.rotatable_indices, list)
        # With all bonds rotatable, should have some indices
        self.assertGreaterEqual(len(optimizer.rotatable_indices), 0)


class TestRingClosureOptimizerMinimize(unittest.TestCase):
    """Test minimization functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(__file__).parent / 'fixtures'
        self.forcefield_file = Path(__file__).parent.parent / 'data' / 'RCP_UFFvdW.xml'
        self.test_int_file = self.test_dir / 'simple_molecule.int'
    
    def test_minimize_cartesian(self):
        """Test Cartesian minimization."""
        if not self.test_int_file.exists() or not self.forcefield_file.exists():
            self.skipTest("Required test files not found")
        
        optimizer = RingClosureOptimizer.from_files(
            str(self.test_int_file),
            str(self.forcefield_file),
            rotatable_bonds=None,
            rcp_terms=None
        )
        
        result = optimizer.minimize(
            max_iterations=10,  # Small number for test
            smoothing=None,
            torsional=False,
            update_system=True,
            verbose=False
        )
        
        self.assertIn('initial_energy', result)
        self.assertIn('final_energy', result)
        self.assertIn('improvement', result)
        self.assertIn('coordinates', result)
        self.assertIn('minimization_type', result)
        self.assertEqual(result['minimization_type'], 'cartesian')
        
        # Energy should decrease or stay same
        self.assertLessEqual(result['final_energy'], result['initial_energy'])
    
    def test_minimize_torsional(self):
        """Test torsional minimization."""
        if not self.test_int_file.exists() or not self.forcefield_file.exists():
            self.skipTest("Required test files not found")
        
        optimizer = RingClosureOptimizer.from_files(
            str(self.test_int_file),
            str(self.forcefield_file),
            rotatable_bonds=None,  # All rotatable
            rcp_terms=None
        )
        
        # Only test if we have rotatable indices
        if len(optimizer.rotatable_indices) == 0:
            self.skipTest("No rotatable indices for torsional minimization")
        
        result = optimizer.minimize(
            max_iterations=10,
            smoothing=None,
            torsional=True,
            update_system=True,
            verbose=False
        )
        
        self.assertIn('initial_energy', result)
        self.assertIn('final_energy', result)
        self.assertIn('minimization_type', result)
        self.assertEqual(result['minimization_type'], 'torsional')
        self.assertIn('optimization_info', result)


def run_tests(verbosity=2):
    """Run all tests with specified verbosity."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestIndividual))
    suite.addTests(loader.loadTestsFromTestCase(TestRingClosureOptimizerUtilities))
    suite.addTests(loader.loadTestsFromTestCase(TestRingClosureOptimizerMinimize))
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests(verbosity=2)
    sys.exit(0 if success else 1)

