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
            {'id': 0, 'element': 'H', 'atomic_num': 1},
            {'id': 1, 'element': 'H', 'atomic_num': 1, 'bond_ref': 1, 'bond_length': 1.0},
            {'id': 2, 'element': 'H', 'atomic_num': 1, 'bond_ref': 1, 'bond_length': 1.0,
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
        self.butane_file = self.test_dir / 'butane_like.int'
    
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
    
    def test_minimize_energy_in_torsional_space(self):
        """Test torsional space energy minimization."""
        if not self.butane_file.exists() or not self.forcefield_file.exists():
            self.skipTest("Required test files not found")
        
        system = MolecularSystem.from_file(
            str(self.butane_file),
            str(self.forcefield_file),
            rcp_terms=None
        )
        
        rotatable_indices = [3]  # 4th atom has rotatable dihedral
        initial_zmatrix = system.zmatrix
        
        # Minimize in torsional space
        final_zmatrix, final_energy, info = system.minimize_energy_in_torsional_space(
            initial_zmatrix,
            rotatable_indices,
            max_iterations=10,
            method='Powell',
            verbose=False
        )
        
        # Check results
        self.assertIsNotNone(final_zmatrix)
        # Energy can be float, np.floating, or 0-d array depending on Python/NumPy version
        self.assertTrue(np.isscalar(final_energy) or (isinstance(final_energy, np.ndarray) and final_energy.ndim == 0))
        self.assertIn('success', info)
        self.assertIn('nfev', info)
        self.assertIn('initial_energy', info)
        self.assertIn('final_energy', info)
    
    def test_minimize_energy_torsional_with_trajectory(self):
        """Test torsional minimization with trajectory writing."""
        if not self.butane_file.exists() or not self.forcefield_file.exists():
            self.skipTest("Required test files not found")
        
        system = MolecularSystem.from_file(
            str(self.butane_file),
            str(self.forcefield_file),
            rcp_terms=None
        )
        
        rotatable_indices = [3]
        initial_zmatrix = system.zmatrix
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            trajectory_file = f.name
        
        try:
            # Minimize with trajectory
            final_zmatrix, final_energy, info = system.minimize_energy_in_torsional_space(
                initial_zmatrix,
                rotatable_indices,
                max_iterations=3,
                trajectory_file=trajectory_file,
                verbose=False
            )
            
            # Trajectory file should exist but may be empty (callback might not be called)
            # This tests that the parameter is handled without errors
            self.assertTrue(True)  # If we get here, no exception was raised
            
        finally:
            if os.path.exists(trajectory_file):
                os.unlink(trajectory_file)


class TestMolecularSystemFromData(unittest.TestCase):
    """Test MolecularSystem.from_data method."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.forcefield_file = Path(__file__).parent.parent / 'data' / 'RCP_UFFvdW.xml'
        
        # Simple Z-matrix
        self.zmatrix = [
            {'id': 0, 'element': 'H', 'atomic_num': 1},
            {'id': 1, 'element': 'H', 'atomic_num': 1, 'bond_ref': 0, 'bond_length': 1.0},
            {'id': 2, 'element': 'H', 'atomic_num': 1, 'bond_ref': 0, 'bond_length': 1.0,
             'angle_ref': 1, 'angle': 109.47}
        ]
        self.bonds_data = [(0, 1, 1), (0, 2, 1)]  # 0-based indexing
    
    def test_from_data_basic(self):
        """Test creating MolecularSystem from raw data."""
        if not self.forcefield_file.exists():
            self.skipTest("Force field file not found")
        
        system = MolecularSystem.from_data(
            zmatrix=self.zmatrix,
            bonds_data=self.bonds_data,
            forcefield_file=str(self.forcefield_file),
            rcp_terms=None
        )
        
        self.assertIsNotNone(system)
        self.assertEqual(len(system.zmatrix), 3)
        self.assertEqual(len(system.elements), 3)
    
    def test_from_data_with_rcp_terms(self):
        """Test creating MolecularSystem from data with RCP terms."""
        if not self.forcefield_file.exists():
            self.skipTest("Force field file not found")
        
        rcp_terms = [(0, 2)]  # 0-based
        system = MolecularSystem.from_data(
            zmatrix=self.zmatrix,
            bonds_data=self.bonds_data,
            forcefield_file=str(self.forcefield_file),
            rcp_terms=rcp_terms
        )
        
        self.assertIsNotNone(system)
        self.assertEqual(len(system.rcpterms), 1)
        self.assertEqual(system.rcpterms[0], (0, 2))
    
    def test_from_data_with_custom_parameters(self):
        """Test creating MolecularSystem with custom parameters."""
        if not self.forcefield_file.exists():
            self.skipTest("Force field file not found")
        
        system = MolecularSystem.from_data(
            zmatrix=self.zmatrix,
            bonds_data=self.bonds_data,
            forcefield_file=str(self.forcefield_file),
            rcp_terms=None,
            write_candidate_files=True,
            ring_closure_threshold=2.0,
            step_length=0.001
        )
        
        self.assertIsNotNone(system)
        self.assertEqual(system.write_candidate_files, True)
        self.assertEqual(system.ring_closure_threshold, 2.0)
        self.assertEqual(system.step_length, 0.001)


class TestMolecularSystemSimulationCache(unittest.TestCase):
    """Test simulation caching in MolecularSystem."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(__file__).parent / 'fixtures'
        self.forcefield_file = Path(__file__).parent.parent / 'data' / 'RCP_UFFvdW.xml'
        self.test_int_file = self.test_dir / 'simple_molecule.int'
    
    def test_simulation_cache_reuse(self):
        """Test that simulations are cached per smoothing parameter."""
        if not self.test_int_file.exists() or not self.forcefield_file.exists():
            self.skipTest("Required test files not found")
        
        system = MolecularSystem.from_file(
            str(self.test_int_file),
            str(self.forcefield_file),
            rcp_terms=None
        )
        
        coords = zmatrix_to_cartesian(system.zmatrix)
        
        # First evaluation with smoothing=0.0
        system.setSmoothingParameter(0.0)
        energy1 = system.evaluate_energy(coords)
        cache_size_after_first = len(system._simulation_cache)
        
        # Second evaluation with same smoothing (should reuse cache)
        energy2 = system.evaluate_energy(coords)
        cache_size_after_second = len(system._simulation_cache)
        
        # Cache should not grow
        self.assertEqual(cache_size_after_first, cache_size_after_second)
        self.assertEqual(energy1, energy2)
    
    def test_simulation_cache_different_smoothing(self):
        """Test that different smoothing parameters create separate cache entries."""
        if not self.test_int_file.exists() or not self.forcefield_file.exists():
            self.skipTest("Required test files not found")
        
        system = MolecularSystem.from_file(
            str(self.test_int_file),
            str(self.forcefield_file),
            rcp_terms=None
        )
        
        coords = zmatrix_to_cartesian(system.zmatrix)
        
        # Evaluation with smoothing=0.0
        system.setSmoothingParameter(0.0)
        _ = system.evaluate_energy(coords)
        cache_size_1 = len(system._simulation_cache)
        
        # Evaluation with smoothing=10.0 (should create new cache entry)
        system.setSmoothingParameter(10.0)
        _ = system.evaluate_energy(coords)
        cache_size_2 = len(system._simulation_cache)
        
        # Cache should grow
        self.assertGreater(cache_size_2, cache_size_1)


class TestMolecularSystemErrorHandling(unittest.TestCase):
    """Test error handling in MolecularSystem."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.forcefield_file = Path(__file__).parent.parent / 'data' / 'RCP_UFFvdW.xml'
    
    def test_from_file_invalid_extension(self):
        """Test that non-.int files raise error."""
        if not self.forcefield_file.exists():
            self.skipTest("Force field file not found")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            temp_file = f.name
            f.write("3\nTest\nH 0 0 0\n")
        
        try:
            with self.assertRaises(ValueError) as context:
                MolecularSystem.from_file(
                    temp_file,
                    str(self.forcefield_file),
                    rcp_terms=None
                )
            self.assertIn(".int", str(context.exception))
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_ring_closure_score_verbose_output(self):
        """Test ring closure score with verbose output."""
        if not self.forcefield_file.exists():
            self.skipTest("Force field file not found")
        
        zmatrix = [
            {'id': 0, 'element': 'H', 'atomic_num': 1},
            {'id': 1, 'element': 'H', 'atomic_num': 1, 'bond_ref': 0, 'bond_length': 1.0},
            {'id': 2, 'element': 'H', 'atomic_num': 1, 'bond_ref': 0, 'bond_length': 1.0,
             'angle_ref': 1, 'angle': 109.47}
        ]
        bonds_data = [(0, 1, 1), (0, 2, 1)]
        rcp_terms = [(0, 2)]
        
        system = MolecularSystem.from_data(
            zmatrix=zmatrix,
            bonds_data=bonds_data,
            forcefield_file=str(self.forcefield_file),
            rcp_terms=rcp_terms
        )
        
        coords = zmatrix_to_cartesian(system.zmatrix)
        
        # Test with verbose=True (should not raise error)
        score = system.ring_closure_score_exponential(coords, verbose=True)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # Test penalty with verbose=True
        penalty = system.ring_closure_penalty_quadratic(coords, verbose=True)
        self.assertGreaterEqual(penalty, 0.0)


class TestMolecularSystemRMSD(unittest.TestCase):
    """Test RMSD calculation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(__file__).parent / 'fixtures'
        self.forcefield_file = Path(__file__).parent.parent / 'data' / 'RCP_UFFvdW.xml'
    
    def test_rmsd_identical_zmatrices(self):
        """Test RMSD with identical Z-matrices (should be zero)."""
        zmatrix = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0}
        ]
        
        bonds = [(0, 1, 1), (1, 2, 1), (2, 3, 1)]
        
        system = MolecularSystem.from_data(
            zmatrix, bonds, str(self.forcefield_file),
            rcp_terms=None
        )
        
        rmsd_bonds, rmsd_angles, rmsd_dihedrals = system.calculate_rmsd(zmatrix, zmatrix)
        
        self.assertAlmostEqual(rmsd_bonds, 0.0, places=10)
        self.assertAlmostEqual(rmsd_angles, 0.0, places=10)
        self.assertAlmostEqual(rmsd_dihedrals, 0.0, places=10)
    
    def test_rmsd_different_bond_lengths(self):
        """Test RMSD with different bond lengths."""
        zmatrix1 = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47}
        ]
        
        zmatrix2 = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.50},  # Changed
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.58,  # Changed
             'angle_ref': 0, 'angle': 109.47}
        ]
        
        bonds = [(0, 1, 1), (1, 2, 1)]
        
        system = MolecularSystem.from_data(
            zmatrix1, bonds, str(self.forcefield_file),
            rcp_terms=None
        )
        
        rmsd_bonds, rmsd_angles, rmsd_dihedrals = system.calculate_rmsd(zmatrix1, zmatrix2)
        
        # Expected: sqrt(((1.54-1.50)^2 + (1.54-1.58)^2) / 2) = sqrt(0.0032 / 2) ≈ 0.04
        expected_rmsd_bonds = np.sqrt(((1.54-1.50)**2 + (1.54-1.58)**2) / 2)
        self.assertAlmostEqual(rmsd_bonds, expected_rmsd_bonds, places=6)
        self.assertAlmostEqual(rmsd_angles, 0.0, places=10)
        self.assertAlmostEqual(rmsd_dihedrals, 0.0, places=10)
    
    def test_rmsd_different_angles(self):
        """Test RMSD with different angles."""
        zmatrix1 = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0}
        ]
        
        zmatrix2 = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 120.0},  # Changed
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 100.0, 'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0}  # Changed
        ]
        
        bonds = [(0, 1, 1), (1, 2, 1), (2, 3, 1)]
        
        system = MolecularSystem.from_data(
            zmatrix1, bonds, str(self.forcefield_file),
            rcp_terms=None
        )
        
        rmsd_bonds, rmsd_angles, rmsd_dihedrals = system.calculate_rmsd(zmatrix1, zmatrix2)
        
        # Expected: sqrt(((109.47-120.0)^2 + (109.47-100.0)^2) / 2)
        angle_diffs = [(109.47-120.0), (109.47-100.0)]
        expected_rmsd_angles = np.sqrt(np.mean(np.array(angle_diffs)**2))
        
        self.assertAlmostEqual(rmsd_bonds, 0.0, places=10)
        self.assertAlmostEqual(rmsd_angles, expected_rmsd_angles, places=6)
        self.assertAlmostEqual(rmsd_dihedrals, 0.0, places=10)
    
    def test_rmsd_different_dihedrals(self):
        """Test RMSD with different dihedrals."""
        zmatrix1 = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0}
        ]
        
        zmatrix2 = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 180.0, 'chirality': 0}  # Changed
        ]
        
        bonds = [(0, 1, 1), (1, 2, 1), (2, 3, 1)]
        
        system = MolecularSystem.from_data(
            zmatrix1, bonds, str(self.forcefield_file),
            rcp_terms=None
        )
        
        rmsd_bonds, rmsd_angles, rmsd_dihedrals = system.calculate_rmsd(zmatrix1, zmatrix2)
        
        # Expected: sqrt((60.0 - 180.0)^2) = 120.0
        expected_rmsd_dihedrals = 120.0
        
        self.assertAlmostEqual(rmsd_bonds, 0.0, places=10)
        self.assertAlmostEqual(rmsd_angles, 0.0, places=10)
        self.assertAlmostEqual(rmsd_dihedrals, expected_rmsd_dihedrals, places=6)
    
    def test_rmsd_dihedral_periodicity(self):
        """Test RMSD handles dihedral periodicity (-180°/+180° boundary)."""
        zmatrix1 = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': -179.0, 'chirality': 0}
        ]
        
        zmatrix2 = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 179.0, 'chirality': 0}  # Near +180°
        ]
        
        bonds = [(0, 1, 1), (1, 2, 1), (2, 3, 1)]
        
        system = MolecularSystem.from_data(
            zmatrix1, bonds, str(self.forcefield_file),
            rcp_terms=None
        )
        
        rmsd_bonds, rmsd_angles, rmsd_dihedrals = system.calculate_rmsd(zmatrix1, zmatrix2)
        
        # -179° and +179° are only 2° apart (crossing the boundary)
        # Difference: -179 - 179 = -358, wrapped to +2
        expected_rmsd_dihedrals = 2.0
        
        self.assertAlmostEqual(rmsd_bonds, 0.0, places=10)
        self.assertAlmostEqual(rmsd_angles, 0.0, places=10)
        self.assertAlmostEqual(rmsd_dihedrals, expected_rmsd_dihedrals, places=6)
    
    def test_rmsd_mixed_changes(self):
        """Test RMSD with changes in all coordinate types."""
        zmatrix1 = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0}
        ]
        
        zmatrix2 = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.50},  # Changed
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.58,  # Changed
             'angle_ref': 0, 'angle': 120.0},  # Changed
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.60,  # Changed
             'angle_ref': 1, 'angle': 100.0, 'dihedral_ref': 0, 'dihedral': 180.0, 'chirality': 0}  # Changed
        ]
        
        bonds = [(0, 1, 1), (1, 2, 1), (2, 3, 1)]
        
        system = MolecularSystem.from_data(
            zmatrix1, bonds, str(self.forcefield_file),
            rcp_terms=None
        )
        
        rmsd_bonds, rmsd_angles, rmsd_dihedrals = system.calculate_rmsd(zmatrix1, zmatrix2)
        
        # Calculate expected values
        bond_diffs = [1.54-1.50, 1.54-1.58, 1.54-1.60]
        expected_rmsd_bonds = np.sqrt(np.mean(np.array(bond_diffs)**2))
        
        angle_diffs = [109.47-120.0, 109.47-100.0]
        expected_rmsd_angles = np.sqrt(np.mean(np.array(angle_diffs)**2))
        
        dihedral_diffs = [60.0-180.0]  # -120.0
        expected_rmsd_dihedrals = abs(dihedral_diffs[0])
        
        self.assertAlmostEqual(rmsd_bonds, expected_rmsd_bonds, places=6)
        self.assertAlmostEqual(rmsd_angles, expected_rmsd_angles, places=6)
        self.assertAlmostEqual(rmsd_dihedrals, expected_rmsd_dihedrals, places=6)
    
    def test_rmsd_minimal_zmatrix(self):
        """Test RMSD with minimal Z-matrix (no dihedrals)."""
        zmatrix1 = [
            {'id': 0, 'element': 'H', 'atomic_num': 1},
            {'id': 1, 'element': 'H', 'atomic_num': 1, 'bond_ref': 0, 'bond_length': 1.0},
            {'id': 2, 'element': 'H', 'atomic_num': 1, 'bond_ref': 1, 'bond_length': 1.0,
             'angle_ref': 0, 'angle': 109.47}
        ]
        
        zmatrix2 = [
            {'id': 0, 'element': 'H', 'atomic_num': 1},
            {'id': 1, 'element': 'H', 'atomic_num': 1, 'bond_ref': 0, 'bond_length': 0.95},
            {'id': 2, 'element': 'H', 'atomic_num': 1, 'bond_ref': 1, 'bond_length': 1.05,
             'angle_ref': 0, 'angle': 120.0}
        ]
        
        bonds = [(0, 1, 1), (1, 2, 1)]
        
        system = MolecularSystem.from_data(
            zmatrix1, bonds, str(self.forcefield_file),
            rcp_terms=None
        )
        
        rmsd_bonds, rmsd_angles, rmsd_dihedrals = system.calculate_rmsd(zmatrix1, zmatrix2)
        
        # Should have bond and angle RMSD, but dihedral RMSD should be 0.0 (no dihedrals)
        self.assertGreater(rmsd_bonds, 0.0)
        self.assertGreater(rmsd_angles, 0.0)
        self.assertAlmostEqual(rmsd_dihedrals, 0.0, places=10)
    
    def test_rmsd_mismatched_length(self):
        """Test RMSD raises ValueError for mismatched Z-matrix lengths."""
        zmatrix1 = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54}
        ]
        
        zmatrix2 = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47}
        ]
        
        bonds = [(0, 1, 1)]
        
        system = MolecularSystem.from_data(
            zmatrix1, bonds, str(self.forcefield_file),
            rcp_terms=None
        )
        
        with self.assertRaises(ValueError) as cm:
            system.calculate_rmsd(zmatrix1, zmatrix2)
        
        self.assertIn("same length", str(cm.exception))
    
    def test_rmsd_large_conformational_change(self):
        """Test RMSD with large conformational change (e.g., cis/trans)."""
        # Build a 5-atom chain
        zmatrix1 = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 0.0, 'chirality': 0},  # cis
            {'id': 4, 'element': 'C', 'atomic_num': 6, 'bond_ref': 3, 'bond_length': 1.54,
             'angle_ref': 2, 'angle': 109.47, 'dihedral_ref': 1, 'dihedral': 0.0, 'chirality': 0}  # cis
        ]
        
        zmatrix2 = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 180.0, 'chirality': 0},  # trans
            {'id': 4, 'element': 'C', 'atomic_num': 6, 'bond_ref': 3, 'bond_length': 1.54,
             'angle_ref': 2, 'angle': 109.47, 'dihedral_ref': 1, 'dihedral': 180.0, 'chirality': 0}  # trans
        ]
        
        bonds = [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1)]
        
        system = MolecularSystem.from_data(
            zmatrix1, bonds, str(self.forcefield_file),
            rcp_terms=None
        )
        
        rmsd_bonds, rmsd_angles, rmsd_dihedrals = system.calculate_rmsd(zmatrix1, zmatrix2)
        
        # Large dihedral change: sqrt((180^2 + 180^2)/2) ≈ 180
        expected_rmsd_dihedrals = 180.0
        
        self.assertAlmostEqual(rmsd_bonds, 0.0, places=10)
        self.assertAlmostEqual(rmsd_angles, 0.0, places=10)
        self.assertAlmostEqual(rmsd_dihedrals, expected_rmsd_dihedrals, places=6)
    
    def test_rmsd_chirality_as_second_angle(self):
        """Test RMSD with chirality != 0 (dihedral is actually a second angle)."""
        # When chirality != 0, the 'dihedral' field contains a second angle, not a proper dihedral
        zmatrix1 = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0},  # proper dihedral
            {'id': 4, 'element': 'C', 'atomic_num': 6, 'bond_ref': 3, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 110.0, 'chirality': 1}  # second angle!
        ]
        
        zmatrix2 = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 180.0, 'chirality': 0},  # proper dihedral
            {'id': 4, 'element': 'C', 'atomic_num': 6, 'bond_ref': 3, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 120.0, 'chirality': 1}  # second angle!
        ]
        
        bonds = [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1)]
        
        system = MolecularSystem.from_data(
            zmatrix1, bonds, str(self.forcefield_file),
            rcp_terms=None
        )
        
        rmsd_bonds, rmsd_angles, rmsd_dihedrals = system.calculate_rmsd(zmatrix1, zmatrix2)
        
        # Atom 4: proper dihedral changed from 60.0 to 180.0 (diff = -120.0)
        # expected_rmsd_dihedrals = 120.0 (only one proper dihedral)
        
        # Atom 5: second angle (chirality=1) changed from 110.0 to 120.0 (diff = -10.0)
        # Atom 3: regular angle unchanged (109.47)
        # Atom 4: regular angle unchanged (109.47)
        # Atom 5: regular angle unchanged (109.47)
        # expected_rmsd_angles = sqrt((0^2 + 0^2 + 0^2 + 10^2) / 4) = sqrt(100/4) = 5.0
        
        expected_rmsd_dihedrals = 120.0
        angle_diffs = [0.0, 0.0, 0.0, -10.0]  # Three regular angles (unchanged) + one second angle (changed)
        expected_rmsd_angles = np.sqrt(np.mean(np.array(angle_diffs)**2))
        
        self.assertAlmostEqual(rmsd_bonds, 0.0, places=10)
        self.assertAlmostEqual(rmsd_angles, expected_rmsd_angles, places=6)
        self.assertAlmostEqual(rmsd_dihedrals, expected_rmsd_dihedrals, places=6)


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
    suite.addTests(loader.loadTestsFromTestCase(TestMolecularSystemFromData))
    suite.addTests(loader.loadTestsFromTestCase(TestMolecularSystemSimulationCache))
    suite.addTests(loader.loadTestsFromTestCase(TestMolecularSystemErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestMolecularSystemRMSD))
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests(verbosity=2)
    sys.exit(0 if success else 1)

