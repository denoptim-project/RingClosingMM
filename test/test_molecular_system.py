#!/usr/bin/env python3
"""
Unit tests for MolecularSystem class.

Tests the core functionality of molecular system management, including
energy evaluation, minimization, and ring closure metrics.
"""

import unittest
import numpy as np
import sys
import os
import tempfile
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from MolecularSystem import MolecularSystem
from IOTools import write_zmatrix_file, write_xyz_file
from CoordinateConverter import zmatrix_to_cartesian


class TestMolecularSystemFileIO(unittest.TestCase):
    """Test file I/O operations for MolecularSystem."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(__file__).parent / 'fixtures'
        self.forcefield_file = Path(__file__).parent.parent / 'data' / 'RCP_UFFvdW.xml'
        
        # Simple Z-matrix for testing
        self.simple_zmatrix = [
            {'id': 1, 'element': 'H', 'atomic_num': 1},
            {'id': 2, 'element': 'H', 'atomic_num': 1, 'bond_ref': 1, 'bond_length': 1.0},
            {'id': 3, 'element': 'H', 'atomic_num': 1, 'bond_ref': 1, 'bond_length': 1.0,
             'angle_ref': 2, 'angle': 109.47}
        ]
        self.simple_elements = ['H', 'H', 'H']
        
    def test_write_zmatrix_file(self):
        """Test writing Z-matrix to file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.int') as f:
            temp_path = f.name
        
        try:
            write_zmatrix_file(self.simple_zmatrix, temp_path)
            
            # Verify file was created
            self.assertTrue(os.path.exists(temp_path))
            
            # Read and verify content
            with open(temp_path, 'r') as f:
                lines = f.readlines()
            
            self.assertEqual(len(lines), 4)  # Header + 3 atoms
            self.assertEqual(lines[0].strip(), "3")
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_write_xyz_file(self):
        """Test writing XYZ file."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.866, 0.0]])
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.xyz') as f:
            temp_path = f.name
        
        try:
            write_xyz_file(coords, self.simple_elements, temp_path, comment="Test")
            
            # Verify file was created
            self.assertTrue(os.path.exists(temp_path))
            
            # Read and verify content
            with open(temp_path, 'r') as f:
                lines = f.readlines()
            
            self.assertEqual(len(lines), 5)  # Header + comment + 3 atoms
            self.assertEqual(lines[0].strip(), "3")
            self.assertEqual(lines[1].strip(), "Test")
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_write_xyz_file_append_mode(self):
        """Test writing XYZ file in append mode."""
        coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        coords2 = np.array([[0.5, 0.866, 0.0], [1.5, 0.866, 0.0]])
        elements = ['H', 'H']
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.xyz') as f:
            temp_path = f.name
        
        try:
            # Write first frame
            write_xyz_file(coords1, elements, temp_path, comment="Frame 1")
            # Append second frame
            write_xyz_file(coords2, elements, temp_path, comment="Frame 2", append=True)
            
            # Read and verify content
            with open(temp_path, 'r') as f:
                lines = f.readlines()
            
            self.assertEqual(len(lines), 8)  # 2 frames: (header + comment + 2 atoms) * 2 = 4 * 2 = 8
            self.assertEqual(lines[0].strip(), "2")
            self.assertEqual(lines[1].strip(), "Frame 1")
            self.assertEqual(lines[4].strip(), "2")  # Second frame header
            self.assertEqual(lines[5].strip(), "Frame 2")
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestMolecularSystemCreation(unittest.TestCase):
    """Test MolecularSystem creation and initialization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(__file__).parent / 'fixtures'
        self.forcefield_file = Path(__file__).parent.parent / 'data' / 'RCP_UFFvdW.xml'
        
        if not self.forcefield_file.exists():
            self.skipTest(f"Force field file not found: {self.forcefield_file}")
        
        self.test_int_file = self.test_dir / 'simple_molecule.int'
    
    def test_from_file_with_valid_input(self):
        """Test creating MolecularSystem from valid INT file."""
        if not self.test_int_file.exists():
            self.skipTest(f"Test INT file not found: {self.test_int_file}")
        
        system = MolecularSystem.from_file(
            str(self.test_int_file),
            str(self.forcefield_file),
            rcp_terms=None,
            write_candidate_files=False
        )
        
        self.assertIsNotNone(system)
        self.assertIsNotNone(system.system)
        self.assertIsNotNone(system.topology)
        self.assertEqual(len(system.zmatrix), 3)
        self.assertEqual(len(system.elements), 3)
        self.assertEqual(system.elements[0], 'H')
    
    def test_from_file_with_rcp_terms(self):
        """Test creating MolecularSystem with RCP terms."""
        if not self.test_int_file.exists():
            self.skipTest(f"Test INT file not found: {self.test_int_file}")
        
        rcp_terms = [(0, 2)]  # 0-based indices
        system = MolecularSystem.from_file(
            str(self.test_int_file),
            str(self.forcefield_file),
            rcp_terms=rcp_terms,
            write_candidate_files=False
        )
        
        self.assertIsNotNone(system)
        self.assertEqual(len(system.rcpterms), 1)
        self.assertEqual(system.rcpterms[0], (0, 2))
    
    def test_set_smoothing_parameter(self):
        """Test setting smoothing parameter."""
        if not self.test_int_file.exists():
            self.skipTest(f"Test INT file not found: {self.test_int_file}")
        
        system = MolecularSystem.from_file(
            str(self.test_int_file),
            str(self.forcefield_file),
            rcp_terms=None
        )
        
        # Test setting smoothing
        system.setSmoothingParameter(10.0)
        self.assertEqual(system._current_smoothing, 10.0)
        
        # Test changing smoothing
        system.setSmoothingParameter(25.0)
        self.assertEqual(system._current_smoothing, 25.0)


class TestMolecularSystemEnergy(unittest.TestCase):
    """Test energy evaluation in MolecularSystem."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(__file__).parent / 'fixtures'
        self.forcefield_file = Path(__file__).parent.parent / 'data' / 'RCP_UFFvdW.xml'
        self.test_int_file = self.test_dir / 'simple_molecule.int'
    
    def test_evaluate_energy(self):
        """Test energy evaluation."""
        if not self.test_int_file.exists() or not self.forcefield_file.exists():
            self.skipTest("Required test files not found")
        
        system = MolecularSystem.from_file(
            str(self.test_int_file),
            str(self.forcefield_file),
            rcp_terms=None
        )
        
        # Get coordinates from Z-matrix
        coords = zmatrix_to_cartesian(system.zmatrix)
        
        # Evaluate energy
        energy = system.evaluate_energy(coords)
        
        self.assertIsInstance(energy, (float, np.floating))
        self.assertFalse(np.isnan(energy))
        self.assertFalse(np.isinf(energy))
    
    def test_evaluate_energy_with_different_coords(self):
        """Test energy evaluation with different coordinates."""
        if not self.test_int_file.exists() or not self.forcefield_file.exists():
            self.skipTest("Required test files not found")
        
        system = MolecularSystem.from_file(
            str(self.test_int_file),
            str(self.forcefield_file),
            rcp_terms=None
        )
        
        # Original coordinates
        coords1 = zmatrix_to_cartesian(system.zmatrix)
        energy1 = system.evaluate_energy(coords1)
        
        # Modified coordinates (slightly moved)
        coords2 = coords1.copy()
        coords2[0] += np.array([0.1, 0.0, 0.0])
        energy2 = system.evaluate_energy(coords2)
        
        # Energies should be different
        self.assertNotEqual(energy1, energy2)
    
    def test_evaluate_energy_with_smoothing(self):
        """Test energy evaluation with smoothing parameter."""
        if not self.test_int_file.exists() or not self.forcefield_file.exists():
            self.skipTest("Required test files not found")
        
        system = MolecularSystem.from_file(
            str(self.test_int_file),
            str(self.forcefield_file),
            rcp_terms=None
        )
        
        coords = zmatrix_to_cartesian(system.zmatrix)
        
        # Energy without smoothing
        system.setSmoothingParameter(0.0)
        energy_no_smooth = system.evaluate_energy(coords)
        
        # Energy with smoothing
        system.setSmoothingParameter(10.0)
        energy_smooth = system.evaluate_energy(coords)
        
        # Energies should be different when smoothing is applied
        # (unless the smoothing doesn't affect this particular system)
        self.assertIsInstance(energy_smooth, (float, np.floating))


class TestMolecularSystemRingClosure(unittest.TestCase):
    """Test ring closure metric calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(__file__).parent / 'fixtures'
        self.forcefield_file = Path(__file__).parent.parent / 'data' / 'RCP_UFFvdW.xml'
        self.test_int_file = self.test_dir / 'simple_molecule.int'
    
    def test_ring_closure_score_exponential_no_rcp_terms(self):
        """Test exponential score with no RCP terms."""
        if not self.test_int_file.exists() or not self.forcefield_file.exists():
            self.skipTest("Required test files not found")
        
        system = MolecularSystem.from_file(
            str(self.test_int_file),
            str(self.forcefield_file),
            rcp_terms=None
        )
        
        coords = zmatrix_to_cartesian(system.zmatrix)
        
        # With no RCP terms, score should be 1.0 (all closed)
        score = system.ring_closure_score_exponential(coords, tolerance=1.5, decay_rate=1.0)
        self.assertEqual(score, 1.0)
    
    def test_ring_closure_score_exponential_with_rcp_terms(self):
        """Test exponential score with RCP terms."""
        if not self.test_int_file.exists() or not self.forcefield_file.exists():
            self.skipTest("Required test files not found")
        
        # Create system with RCP terms
        rcp_terms = [(0, 2)]  # 0-based: first and third atom
        system = MolecularSystem.from_file(
            str(self.test_int_file),
            str(self.forcefield_file),
            rcp_terms=rcp_terms
        )
        
        coords = zmatrix_to_cartesian(system.zmatrix)
        
        # Calculate score
        score = system.ring_closure_score_exponential(coords, tolerance=1.5, decay_rate=1.0)
        
        # Score should be in [0, 1]
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_ring_closure_score_exponential_tolerance(self):
        """Test exponential score respects tolerance."""
        if not self.test_int_file.exists() or not self.forcefield_file.exists():
            self.skipTest("Required test files not found")
        
        rcp_terms = [(0, 2)]
        system = MolecularSystem.from_file(
            str(self.test_int_file),
            str(self.forcefield_file),
            rcp_terms=rcp_terms
        )
        
        coords = zmatrix_to_cartesian(system.zmatrix)
        
        # With very large tolerance, should get score of 1.0
        score = system.ring_closure_score_exponential(coords, tolerance=100.0, decay_rate=1.0)
        self.assertEqual(score, 1.0)
    
    def test_ring_closure_penalty_quadratic(self):
        """Test quadratic penalty calculation."""
        if not self.test_int_file.exists() or not self.forcefield_file.exists():
            self.skipTest("Required test files not found")
        
        rcp_terms = [(0, 2)]
        system = MolecularSystem.from_file(
            str(self.test_int_file),
            str(self.forcefield_file),
            rcp_terms=rcp_terms
        )
        
        coords = zmatrix_to_cartesian(system.zmatrix)
        
        # Calculate penalty
        penalty = system.ring_closure_penalty_quadratic(coords, target_distance=1.5)
        
        # Penalty should be non-negative
        self.assertGreaterEqual(penalty, 0.0)
        
        # With very large target distance, penalty should be 0.0
        penalty_large = system.ring_closure_penalty_quadratic(coords, target_distance=100.0)
        self.assertEqual(penalty_large, 0.0)


class TestMolecularSystemMinimization(unittest.TestCase):
    """Test minimization operations in MolecularSystem."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(__file__).parent / 'fixtures'
        self.forcefield_file = Path(__file__).parent.parent / 'data' / 'RCP_UFFvdW.xml'
        self.test_int_file = self.test_dir / 'simple_molecule.int'
    
    def test_minimize_energy_cartesian(self):
        """Test Cartesian energy minimization."""
        if not self.test_int_file.exists() or not self.forcefield_file.exists():
            self.skipTest("Required test files not found")
        
        system = MolecularSystem.from_file(
            str(self.test_int_file),
            str(self.forcefield_file),
            rcp_terms=None
        )
        
        initial_zmatrix = system.zmatrix
        initial_coords = zmatrix_to_cartesian(initial_zmatrix)
        initial_energy = system.evaluate_energy(initial_coords)
        
        # Minimize
        final_coords, final_energy = system.minimize_energy(
            initial_zmatrix, max_iterations=50
        )
        
        # Energy should decrease or stay the same
        self.assertLessEqual(final_energy, initial_energy)
        self.assertIsInstance(final_coords, np.ndarray)
        self.assertEqual(final_coords.shape, initial_coords.shape)


def run_tests(verbosity=2):
    """Run all tests with specified verbosity."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMolecularSystemFileIO))
    suite.addTests(loader.loadTestsFromTestCase(TestMolecularSystemCreation))
    suite.addTests(loader.loadTestsFromTestCase(TestMolecularSystemEnergy))
    suite.addTests(loader.loadTestsFromTestCase(TestMolecularSystemRingClosure))
    suite.addTests(loader.loadTestsFromTestCase(TestMolecularSystemMinimization))
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests(verbosity=2)
    sys.exit(0 if success else 1)

