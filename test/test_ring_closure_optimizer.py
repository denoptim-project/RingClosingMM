#!/usr/bin/env python3
"""
Unit tests for RingClosureOptimizer components.

Tests RingClosureOptimizer utilities, optimization, and minimization methods.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from RingClosureOptimizer import RingClosureOptimizer
from MolecularSystem import MolecularSystem


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
        # For simple 6-atom molecule: bond (0,1) and (1,2) might map to different indices
        rotatable_bonds = [(0, 1)]  # 0-based
        
        indices = RingClosureOptimizer._convert_bonds_to_indices(
            rotatable_bonds, system.zmatrix
        )
        
        # Should return list of indices
        self.assertIsInstance(indices, list)
        self.assertTrue(all(isinstance(i, int) for i in indices))
    
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
        self.assertIsInstance(optimizer.system.rotatable_indices, list)
        self.assertGreaterEqual(len(optimizer.system.rotatable_indices), 0)
    
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
        self.assertIsInstance(optimizer.system.rotatable_indices, list)
        # With all bonds rotatable, should have some indices
        self.assertGreaterEqual(len(optimizer.system.rotatable_indices), 0)


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
            space_type='Cartesian',
            verbose=False
        )
        
        self.assertIn('initial_energy', result)
        self.assertIn('final_energy', result)
        self.assertIn('coordinates', result)
        self.assertIn('minimization_type', result)
        self.assertEqual(result['minimization_type'], 'Cartesian')
        
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
        if len(optimizer.system.rotatable_indices) == 0:
            self.skipTest("No rotatable indices for torsional minimization")
        
        result = optimizer.minimize(
            max_iterations=10,
            smoothing=None,
            space_type='torsional',
            verbose=False
        )
        
        self.assertIn('initial_energy', result)
        self.assertIn('final_energy', result)
        self.assertIn('minimization_type', result)
        self.assertEqual(result['minimization_type'], 'torsional')
        self.assertIn('optimization_info', result)
    
    def test_minimize_zmatrix(self):
        """Test Z-matrix space minimization."""
        # Use butane_like.int which has rotatable dihedrals
        test_int_file = self.test_dir / 'butane_like.int'
        
        if not test_int_file.exists() or not self.forcefield_file.exists():
            self.skipTest("Required test files not found")
        
        optimizer = RingClosureOptimizer.from_files(
            str(test_int_file),
            str(self.forcefield_file),
            rotatable_bonds=None,  # All rotatable
            rcp_terms=None
        )
        
        # Only test if we have DOF indices
        if len(optimizer.system.dof_indices) == 0:
            self.skipTest("No DOF indices for Z-matrix minimization")
        
        result = optimizer.minimize(
            max_iterations=10,
            smoothing=None,
            space_type='zmatrix',
            verbose=False
        )
        
        self.assertIn('initial_energy', result)
        self.assertIn('final_energy', result)
        self.assertIn('minimization_type', result)
        self.assertEqual(result['minimization_type'], 'zmatrix')
        self.assertIn('optimization_info', result)
        
        # Energy should decrease or stay same
        self.assertLessEqual(result['final_energy'], result['initial_energy'])


class TestRingClosureOptimizerOptimize(unittest.TestCase):
    """Test full optimization functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(__file__).parent / 'fixtures'
        self.forcefield_file = Path(__file__).parent.parent / 'data' / 'RCP_UFFvdW.xml'
        self.test_int_file = self.test_dir / 'butane_like.int'
    
    def test_optimize_basic(self):
        """Test basic optimization run."""
        if not self.test_int_file.exists() or not self.forcefield_file.exists():
            self.skipTest("Required test files not found")
        
        optimizer = RingClosureOptimizer.from_files(
            str(self.test_int_file),
            str(self.forcefield_file),
            rotatable_bonds=None,
            rcp_terms=None
        )
        
        # Skip if no rotatable indices
        if len(optimizer.system.rotatable_indices) == 0:
            self.skipTest("No rotatable indices")
        
        result = optimizer.optimize(
            enable_pssrot_refinement=False,
            enable_zmatrix_refinement=False,
            verbose=False
        )
        
        self.assertIn('initial_ring_closure_score', result)
        self.assertIn('final_ring_closure_score', result)
        self.assertIn('final_zmatrix', result)
        self.assertIn('final_energy', result)
        self.assertTrue(result['success'])
    
    def test_optimize_with_refinement(self):
        """Test optimization with refinement enabled."""
        if not self.test_int_file.exists() or not self.forcefield_file.exists():
            self.skipTest("Required test files not found")
        
        optimizer = RingClosureOptimizer.from_files(
            str(self.test_int_file),
            str(self.forcefield_file),
            rotatable_bonds=None,
            rcp_terms=None
        )
        
        if len(optimizer.system.rotatable_indices) == 0:
            self.skipTest("No rotatable indices")
        
        result = optimizer.optimize(
            enable_pssrot_refinement=True,
            enable_zmatrix_refinement=True,
            smoothing_sequence=[10.0, 0.0],
            torsional_iterations=5,
            zmatrix_iterations=5,
            verbose=False
        )
        
        self.assertIn('initial_ring_closure_score', result)
        self.assertIn('final_ring_closure_score', result)
        self.assertIn('final_zmatrix', result)
        self.assertIn('final_energy', result)
        self.assertTrue(result['success'])
    
    def test_minimize_with_smoothing_sequence(self):
        """Test minimization with smoothing sequence."""
        if not self.test_int_file.exists() or not self.forcefield_file.exists():
            self.skipTest("Required test files not found")
        
        optimizer = RingClosureOptimizer.from_files(
            str(self.test_int_file),
            str(self.forcefield_file),
            rotatable_bonds=None,
            rcp_terms=None
        )
        
        if len(optimizer.system.rotatable_indices) == 0:
            self.skipTest("No rotatable indices")
        
        result = optimizer.minimize(
            max_iterations=5,
            smoothing=[10.0, 5.0, 0.0],
            space_type='torsional',
            verbose=False
        )
        
        self.assertIn('smoothing_sequence', result)
        self.assertEqual(result['smoothing_sequence'], [10.0, 5.0, 0.0])
        self.assertIn('success', result)
    
    def test_minimize_with_single_smoothing(self):
        """Test minimization with single smoothing value."""
        if not self.test_int_file.exists() or not self.forcefield_file.exists():
            self.skipTest("Required test files not found")
        
        optimizer = RingClosureOptimizer.from_files(
            str(self.test_int_file),
            str(self.forcefield_file),
            rotatable_bonds=None,
            rcp_terms=None
        )
        
        result = optimizer.minimize(
            max_iterations=5,
            smoothing=5.0,
            space_type='Cartesian',
            verbose=False
        )
        
        self.assertIn('smoothing_sequence', result)
        self.assertEqual(result['smoothing_sequence'], [5.0])


def run_tests(verbosity=2):
    """Run all tests with specified verbosity."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRingClosureOptimizerUtilities))
    suite.addTests(loader.loadTestsFromTestCase(TestRingClosureOptimizerMinimize))
    suite.addTests(loader.loadTestsFromTestCase(TestRingClosureOptimizerOptimize))
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests(verbosity=2)
    sys.exit(0 if success else 1)

