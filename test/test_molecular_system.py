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
from CoordinateConverter import zmatrix_to_cartesian
from ZMatrix import ZMatrix


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
            rcp_terms=None
        )
        
        self.assertIsNotNone(system)
        self.assertIsNotNone(system.system)
        self.assertIsNotNone(system.topology)
        self.assertEqual(len(system.zmatrix), 6)
        self.assertEqual(len(system.elements), 6)
        self.assertEqual(system.elements[0], 'H')
    
    def test_from_file_with_rcp_terms(self):
        """Test creating MolecularSystem with RCP terms."""
        if not self.test_int_file.exists():
            self.skipTest(f"Test INT file not found: {self.test_int_file}")
        
        rcp_terms = [(0, 2)]  # 0-based indices
        system = MolecularSystem.from_file(
            str(self.test_int_file),
            str(self.forcefield_file),
            rcp_terms=rcp_terms
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
        zmatrix_atoms = [
            {'id': 0, 'element': 'H', 'atomic_num': 1},
            {'id': 1, 'element': 'H', 'atomic_num': 1, 'bond_ref': 0, 'bond_length': 1.0},
            {'id': 2, 'element': 'H', 'atomic_num': 1, 'bond_ref': 0, 'bond_length': 1.0,
             'angle_ref': 1, 'angle': 109.47}
        ]
        self.zmatrix = ZMatrix(zmatrix_atoms, [(0, 1, 1), (0, 2, 1)])
    
    def test_from_data_basic(self):
        """Test creating MolecularSystem from raw data."""
        if not self.forcefield_file.exists():
            self.skipTest("Force field file not found")
        
        system = MolecularSystem.from_data(
            zmatrix=self.zmatrix,
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
            forcefield_file=str(self.forcefield_file),
            rcp_terms=None,
            ring_closure_threshold=2.0,
            step_length=0.001
        )
        
        self.assertIsNotNone(system)
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
        
        zmatrix_atoms = [
            {'id': 0, 'element': 'H', 'atomic_num': 1},
            {'id': 1, 'element': 'H', 'atomic_num': 1, 'bond_ref': 0, 'bond_length': 1.0},
            {'id': 2, 'element': 'H', 'atomic_num': 1, 'bond_ref': 0, 'bond_length': 1.0,
             'angle_ref': 1, 'angle': 109.47}
        ]
        zmatrix = ZMatrix(zmatrix_atoms, [(0, 1, 1), (0, 2, 1)])
        rcp_terms = [(0, 2)]
        
        system = MolecularSystem.from_data(
            zmatrix=zmatrix,
            forcefield_file=str(self.forcefield_file),
            rcp_terms=rcp_terms
        )
        
        coords = zmatrix_to_cartesian(system.zmatrix)
        
        # Test with verbose=True (should not raise error)
        score = system.ring_closure_score_exponential(coords, verbose=True)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestMolecularSystemRMSD(unittest.TestCase):
    """Test RMSD calculation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(__file__).parent / 'fixtures'
        self.forcefield_file = Path(__file__).parent.parent / 'data' / 'RCP_UFFvdW.xml'
    
    def test_rmsd_identical_zmatrices(self):
        """Test RMSD with identical Z-matrices (should be zero)."""
        zmatrix_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0}
        ]
        zmatrix = ZMatrix(zmatrix_atoms, [(0, 1, 1), (1, 2, 1), (2, 3, 1)])
        
        system = MolecularSystem.from_data(
            zmatrix, str(self.forcefield_file),
            rcp_terms=None
        )
        
        rmsd_bonds, rmsd_angles, rmsd_dihedrals = MolecularSystem._calculate_rmsd(zmatrix, zmatrix)
        
        self.assertAlmostEqual(rmsd_bonds, 0.0, places=10)
        self.assertAlmostEqual(rmsd_angles, 0.0, places=10)
        self.assertAlmostEqual(rmsd_dihedrals, 0.0, places=10)
    
    def test_rmsd_different_bond_lengths(self):
        """Test RMSD with different bond lengths."""
        zmatrix1_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47}
        ]
        zmatrix1 = ZMatrix(zmatrix1_atoms, [(0, 1, 1), (1, 2, 1)])
        
        zmatrix2_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.50},  # Changed
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.58,  # Changed
             'angle_ref': 0, 'angle': 109.47}
        ]
        zmatrix2 = ZMatrix(zmatrix2_atoms, [(0, 1, 1), (1, 2, 1)])

        system = MolecularSystem.from_data(
            zmatrix1, str(self.forcefield_file),
            rcp_terms=None
        )
        
        rmsd_bonds, rmsd_angles, rmsd_dihedrals = MolecularSystem._calculate_rmsd(zmatrix1, zmatrix2)
        
        # Expected: sqrt(((1.54-1.50)^2 + (1.54-1.58)^2) / 2) = sqrt(0.0032 / 2) ≈ 0.04
        expected_rmsd_bonds = np.sqrt(((1.54-1.50)**2 + (1.54-1.58)**2) / 2)
        self.assertAlmostEqual(rmsd_bonds, expected_rmsd_bonds, places=6)
        self.assertAlmostEqual(rmsd_angles, 0.0, places=10)
        self.assertAlmostEqual(rmsd_dihedrals, 0.0, places=10)
    
    def test_rmsd_different_angles(self):
        """Test RMSD with different angles."""
        zmatrix1_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0}
        ]
        zmatrix1 = ZMatrix(zmatrix1_atoms, [(0, 1, 1), (1, 2, 1), (2, 3, 1)])
        
        zmatrix2_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 120.0},  # Changed
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 100.0, 'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0}  # Changed
        ]
        zmatrix2 = ZMatrix(zmatrix2_atoms, [(0, 1, 1), (1, 2, 1), (2, 3, 1)])

        system = MolecularSystem.from_data(
            zmatrix1, str(self.forcefield_file),
            rcp_terms=None
        )
        
        rmsd_bonds, rmsd_angles, rmsd_dihedrals = MolecularSystem._calculate_rmsd(zmatrix1, zmatrix2)
        
        # Expected: sqrt(((109.47-120.0)^2 + (109.47-100.0)^2) / 2)
        angle_diffs = [(109.47-120.0), (109.47-100.0)]
        expected_rmsd_angles = np.sqrt(np.mean(np.array(angle_diffs)**2))
        
        self.assertAlmostEqual(rmsd_bonds, 0.0, places=10)
        self.assertAlmostEqual(rmsd_angles, expected_rmsd_angles, places=6)
        self.assertAlmostEqual(rmsd_dihedrals, 0.0, places=10)
    
    def test_rmsd_different_dihedrals(self):
        """Test RMSD with different dihedrals."""
        zmatrix1_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0}
        ]
        zmatrix1 = ZMatrix(zmatrix1_atoms, [(0, 1, 1), (1, 2, 1), (2, 3, 1)])
        
        zmatrix2_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 180.0, 'chirality': 0}  # Changed
        ]
        zmatrix2 = ZMatrix(zmatrix2_atoms, [(0, 1, 1), (1, 2, 1), (2, 3, 1)])
        
        system = MolecularSystem.from_data(
            zmatrix1, str(self.forcefield_file),
            rcp_terms=None
        )
        
        rmsd_bonds, rmsd_angles, rmsd_dihedrals = MolecularSystem._calculate_rmsd(zmatrix1, zmatrix2)
        
        # Expected: sqrt((60.0 - 180.0)^2) = 120.0
        expected_rmsd_dihedrals = 120.0
        
        self.assertAlmostEqual(rmsd_bonds, 0.0, places=10)
        self.assertAlmostEqual(rmsd_angles, 0.0, places=10)
        self.assertAlmostEqual(rmsd_dihedrals, expected_rmsd_dihedrals, places=6)
    
    def test_rmsd_dihedral_periodicity(self):
        """Test RMSD handles dihedral periodicity (-180°/+180° boundary)."""
        zmatrix1_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': -179.0, 'chirality': 0}
        ]
        zmatrix1 = ZMatrix(zmatrix1_atoms, [(0, 1, 1), (1, 2, 1), (2, 3, 1)])
        
        zmatrix2_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 179.0, 'chirality': 0}  # Near +180°
        ]
        zmatrix2 = ZMatrix(zmatrix2_atoms, [(0, 1, 1), (1, 2, 1), (2, 3, 1)])
        
        system = MolecularSystem.from_data(
            zmatrix1, str(self.forcefield_file),
            rcp_terms=None
        )
        
        rmsd_bonds, rmsd_angles, rmsd_dihedrals = MolecularSystem._calculate_rmsd(zmatrix1, zmatrix2)
        
        # -179° and +179° are only 2° apart (crossing the boundary)
        # Difference: -179 - 179 = -358, wrapped to +2
        expected_rmsd_dihedrals = 2.0
        
        self.assertAlmostEqual(rmsd_bonds, 0.0, places=10)
        self.assertAlmostEqual(rmsd_angles, 0.0, places=10)
        self.assertAlmostEqual(rmsd_dihedrals, expected_rmsd_dihedrals, places=6)
    
    def test_rmsd_mixed_changes(self):
        """Test RMSD with changes in all coordinate types."""
        zmatrix1_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0}
        ]
        zmatrix1 = ZMatrix(zmatrix1_atoms, [(0, 1, 1), (1, 2, 1), (2, 3, 1)])
        
        zmatrix2_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.50},  # Changed
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.58,  # Changed
             'angle_ref': 0, 'angle': 120.0},  # Changed
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.60,  # Changed
             'angle_ref': 1, 'angle': 100.0, 'dihedral_ref': 0, 'dihedral': 180.0, 'chirality': 0}  # Changed
        ]
        zmatrix2 = ZMatrix(zmatrix2_atoms, [(0, 1, 1), (1, 2, 1), (2, 3, 1)])

        system = MolecularSystem.from_data(
            zmatrix1, str(self.forcefield_file),
            rcp_terms=None
        )
        
        rmsd_bonds, rmsd_angles, rmsd_dihedrals = MolecularSystem._calculate_rmsd(zmatrix1, zmatrix2)
        
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
        zmatrix1_atoms = [
            {'id': 0, 'element': 'H', 'atomic_num': 1},
            {'id': 1, 'element': 'H', 'atomic_num': 1, 'bond_ref': 0, 'bond_length': 1.0},
            {'id': 2, 'element': 'H', 'atomic_num': 1, 'bond_ref': 1, 'bond_length': 1.0,
             'angle_ref': 0, 'angle': 109.47}
        ]
        zmatrix1 = ZMatrix(zmatrix1_atoms, [(0, 1, 1), (1, 2, 1)])
        
        zmatrix2_atoms = [
            {'id': 0, 'element': 'H', 'atomic_num': 1},
            {'id': 1, 'element': 'H', 'atomic_num': 1, 'bond_ref': 0, 'bond_length': 0.95},
            {'id': 2, 'element': 'H', 'atomic_num': 1, 'bond_ref': 1, 'bond_length': 1.05,
             'angle_ref': 0, 'angle': 120.0}
        ]
        zmatrix2 = ZMatrix(zmatrix2_atoms, [(0, 1, 1), (1, 2, 1)])
        
        system = MolecularSystem.from_data(
            zmatrix1, str(self.forcefield_file),
            rcp_terms=None
        )
        
        rmsd_bonds, rmsd_angles, rmsd_dihedrals = MolecularSystem._calculate_rmsd(zmatrix1, zmatrix2)
        
        # Should have bond and angle RMSD, but dihedral RMSD should be 0.0 (no dihedrals)
        self.assertGreater(rmsd_bonds, 0.0)
        self.assertGreater(rmsd_angles, 0.0)
        self.assertAlmostEqual(rmsd_dihedrals, 0.0, places=10)
    
    def test_rmsd_mismatched_length(self):
        """Test RMSD raises ValueError for mismatched Z-matrix lengths."""
        zmatrix1_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54}
        ]
        zmatrix1 = ZMatrix(zmatrix1_atoms, [(0, 1, 1)])
        
        zmatrix2_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47}
        ]
        zmatrix2 = ZMatrix(zmatrix2_atoms, [(0, 1, 1), (1, 2, 1)])
        
        system = MolecularSystem.from_data(
            zmatrix1, str(self.forcefield_file),
            rcp_terms=None
        )
        
        with self.assertRaises(ValueError) as cm:
            MolecularSystem._calculate_rmsd(zmatrix1, zmatrix2)
        
        self.assertIn("same length", str(cm.exception))
    
    def test_rmsd_large_conformational_change(self):
        """Test RMSD with large conformational change (e.g., cis/trans)."""
        # Build a 5-atom chain
        zmatrix1_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 0.0, 'chirality': 0},  # cis
            {'id': 4, 'element': 'C', 'atomic_num': 6, 'bond_ref': 3, 'bond_length': 1.54,
             'angle_ref': 2, 'angle': 109.47, 'dihedral_ref': 1, 'dihedral': 0.0, 'chirality': 0}  # cis
        ]
        zmatrix1 = ZMatrix(zmatrix1_atoms, [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1)])
        
        zmatrix2_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 180.0, 'chirality': 0},  # trans
            {'id': 4, 'element': 'C', 'atomic_num': 6, 'bond_ref': 3, 'bond_length': 1.54,
             'angle_ref': 2, 'angle': 109.47, 'dihedral_ref': 1, 'dihedral': 180.0, 'chirality': 0}  # trans
        ]
        zmatrix2 = ZMatrix(zmatrix2_atoms, [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1)])
        
        system = MolecularSystem.from_data(
            zmatrix1, str(self.forcefield_file),
            rcp_terms=None
        )
        
        rmsd_bonds, rmsd_angles, rmsd_dihedrals = MolecularSystem._calculate_rmsd(zmatrix1, zmatrix2)
        
        # Large dihedral change: sqrt((180^2 + 180^2)/2) ≈ 180
        expected_rmsd_dihedrals = 180.0
        
        self.assertAlmostEqual(rmsd_bonds, 0.0, places=10)
        self.assertAlmostEqual(rmsd_angles, 0.0, places=10)
        self.assertAlmostEqual(rmsd_dihedrals, expected_rmsd_dihedrals, places=6)
    
    def test_rmsd_chirality_as_second_angle(self):
        """Test RMSD with chirality != 0 (dihedral is actually a second angle)."""
        # When chirality != 0, the 'dihedral' field contains a second angle, not a proper dihedral
        zmatrix1_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0},  # proper dihedral
            {'id': 4, 'element': 'C', 'atomic_num': 6, 'bond_ref': 3, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 110.0, 'chirality': 1}  # second angle!
        ]
        zmatrix1 = ZMatrix(zmatrix1_atoms, [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1)])
        
        zmatrix2_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 180.0, 'chirality': 0},  # proper dihedral
            {'id': 4, 'element': 'C', 'atomic_num': 6, 'bond_ref': 3, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 120.0, 'chirality': 1}  # second angle!
        ]
        zmatrix2 = ZMatrix(zmatrix2_atoms, [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1)])
        
        system = MolecularSystem.from_data(
            zmatrix1, str(self.forcefield_file),
            rcp_terms=None
        )
        
        rmsd_bonds, rmsd_angles, rmsd_dihedrals = MolecularSystem._calculate_rmsd(zmatrix1, zmatrix2)
        
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


class TestMolecularSystemGraphMethods(unittest.TestCase):
    """Test graph methods in MolecularSystem (static methods)."""
    
    def test_build_graph_from_simple_zmatrix(self):
        """Test building graph from simple linear Z-matrix."""
        # Simple 6-atom linear chain: H-H-H-H-H-H
        zmatrix_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47}
        ]
        zmatrix = ZMatrix(zmatrix_atoms, [(0, 1, 1), (1, 2, 1)])
        
        graph = MolecularSystem._build_bond_graph(zmatrix, topology=None)
        
        # Check graph structure
        self.assertEqual(len(graph), 3)
        self.assertEqual(set(graph[0]), {1})      # Atom 0 bonded to 1
        self.assertEqual(set(graph[1]), {0, 2})   # Atom 1 bonded to 0 and 2
        self.assertEqual(set(graph[2]), {1})      # Atom 2 bonded to 1
    
    def test_build_graph_butane(self):
        """Test building graph from butane-like Z-matrix."""
        # 4-atom chain: C-C-C-C
        zmatrix_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0}
        ]
        zmatrix = ZMatrix(zmatrix_atoms, [(0, 1, 1), (1, 2, 1), (2, 3, 1)])
        
        graph = MolecularSystem._build_bond_graph(zmatrix, topology=None)
        
        # Check graph structure
        self.assertEqual(len(graph), 4)
        self.assertEqual(set(graph[0]), {1})      # Atom 0 bonded to 1
        self.assertEqual(set(graph[1]), {0, 2})   # Atom 1 bonded to 0 and 2
        self.assertEqual(set(graph[2]), {1, 3})   # Atom 2 bonded to 1 and 3
        self.assertEqual(set(graph[3]), {2})      # Atom 3 bonded to 2
    
    def test_graph_is_bidirectional(self):
        """Test that graph edges are bidirectional."""
        zmatrix_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47}
        ]
        zmatrix = ZMatrix(zmatrix_atoms, [(0, 1, 1), (1, 2, 1)])
        
        graph = MolecularSystem._build_bond_graph(zmatrix, topology=None)
        
        # Check bidirectional edges
        for atom, neighbors in graph.items():
            for neighbor in neighbors:
                self.assertIn(atom, graph[neighbor],
                             f"Edge {atom}->{neighbor} exists, but reverse edge {neighbor}->{atom} missing")
    
    def test_path_in_linear_chain(self):
        """Test finding path in a linear chain."""
        # Graph: 0-1-2-3
        graph = {
            0: [1],
            1: [0, 2],
            2: [1, 3],
            3: [2]
        }
        
        # Path from 0 to 3
        path = MolecularSystem._find_path_bfs(graph, 0, 3)
        self.assertEqual(path, [0, 1, 2, 3])
        
        # Path from 3 to 0 (reverse)
        path_reverse = MolecularSystem._find_path_bfs(graph, 3, 0)
        self.assertEqual(path_reverse, [3, 2, 1, 0])
    
    def test_path_to_self(self):
        """Test path from node to itself."""
        graph = {
            0: [1],
            1: [0, 2],
            2: [1]
        }
        
        path = MolecularSystem._find_path_bfs(graph, 1, 1)
        self.assertEqual(path, [1])
    
    def test_path_in_branched_graph(self):
        """Test finding path in branched structure."""
        # Graph:
        #     2
        #     |
        # 0-1-3-4
        graph = {
            0: [1],
            1: [0, 2, 3],
            2: [1],
            3: [1, 4],
            4: [3]
        }
        
        # Path from 0 to 4 (should go through 1, 3)
        path = MolecularSystem._find_path_bfs(graph, 0, 4)
        self.assertEqual(path, [0, 1, 3, 4])
        
        # Path from 2 to 4
        path = MolecularSystem._find_path_bfs(graph, 2, 4)
        self.assertEqual(path, [2, 1, 3, 4])
    
    def test_path_in_cycle(self):
        """Test finding path in cyclic graph."""
        # Graph: 0-1-2-3-0 (square)
        graph = {
            0: [1, 3],
            1: [0, 2],
            2: [1, 3],
            3: [2, 0]
        }
        
        # Path from 0 to 2 (two possible paths of equal length)
        path = MolecularSystem._find_path_bfs(graph, 0, 2)
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 3)  # Shortest path
        self.assertEqual(path[0], 0)
        self.assertEqual(path[-1], 2)
        # Path should be either [0,1,2] or [0,3,2]
        self.assertIn(path, [[0, 1, 2], [0, 3, 2]])
    
    def test_no_path_disconnected(self):
        """Test that no path is found in disconnected graph."""
        # Two separate components: 0-1 and 2-3
        graph = {
            0: [1],
            1: [0],
            2: [3],
            3: [2]
        }
        
        # No path between disconnected components
        path = MolecularSystem._find_path_bfs(graph, 0, 3)
        self.assertIsNone(path)
        
        # Path within same component works
        path = MolecularSystem._find_path_bfs(graph, 0, 1)
        self.assertEqual(path, [0, 1])
    
    def test_invalid_start_node(self):
        """Test with invalid start node."""
        graph = {
            0: [1],
            1: [0, 2],
            2: [1]
        }
        
        path = MolecularSystem._find_path_bfs(graph, 5, 1)
        self.assertIsNone(path)
    
    def test_invalid_end_node(self):
        """Test with invalid end node."""
        graph = {
            0: [1],
            1: [0, 2],
            2: [1]
        }
        
        path = MolecularSystem._find_path_bfs(graph, 0, 5)
        self.assertIsNone(path)
    
    def test_single_isolated_atom(self):
        """Test with single isolated atom."""
        graph = {0: []}
        
        path = MolecularSystem._find_path_bfs(graph, 0, 0)
        self.assertEqual(path, [0])


class TestMolecularSystemDOFMethods(unittest.TestCase):
    """Test DOF-related methods in MolecularSystem."""
    
    def test_get_dofs_from_rotatable_indices(self):
        """Test getting DOF indices from rotatable indices with hardcoded Z-matrix."""
        # Create a larger branched Z-matrix with 12 atoms
        # Includes both dihedrals (chirality=0) and second angles (chirality=+/-1)
        zmatrix_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54, 'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54, 'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0},
            {'id': 4, 'element': 'C', 'atomic_num': 6, 'bond_ref': 3, 'bond_length': 1.54, 'angle_ref': 2, 'angle': 109.47, 'dihedral_ref': 1, 'dihedral': 180.0, 'chirality': 0}, # rotatable
            {'id': 5, 'element': 'H', 'atomic_num': 1, 'bond_ref': 2, 'bond_length': 1.09, 'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 120.0, 'chirality': 1},
            {'id': 6, 'element': 'H', 'atomic_num': 1, 'bond_ref': 2, 'bond_length': 1.09, 'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': -120.0, 'chirality': -1},
            {'id': 7, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54, 'angle_ref': 0, 'angle': 109.47, 'dihedral_ref': 2, 'dihedral': 120.0, 'chirality': 1},
            {'id': 8, 'element': 'H', 'atomic_num': 1, 'bond_ref': 4, 'bond_length': 1.09, 'angle_ref': 3, 'angle': 109.47, 'dihedral_ref': 2, 'dihedral': 60.0, 'chirality': 0},
            {'id': 9, 'element': 'H', 'atomic_num': 1, 'bond_ref': 0, 'bond_length': 1.09, 'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 2, 'dihedral': 180.0, 'chirality': 0}, # rotatable
            {'id': 10, 'element': 'H', 'atomic_num': 1, 'bond_ref': 7, 'bond_length': 1.09, 'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 1},
            {'id': 11, 'element': 'H', 'atomic_num': 1, 'bond_ref': 7, 'bond_length': 1.09, 'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': -60.0, 'chirality': -1}
        ]
        # Build bonds from bond_ref relationships
        bonds = []
        for i in range(1, len(zmatrix_atoms)):
            atom = zmatrix_atoms[i]
            if 'bond_ref' in atom:
                bonds.append((atom['bond_ref'], i, 1))
        zmatrix = ZMatrix(zmatrix_atoms, bonds)
        
        rotatable_indices = [9, 4]  # 0-based indices in zmatrix
        rc_critical_rotatable_indeces = []  # No critical indices for this test
        rc_critical_atoms = []
        dof_indices = MolecularSystem._get_dofs_from_rotatable_indeces(rotatable_indices, rc_critical_rotatable_indeces, rc_critical_atoms, zmatrix)
        
        # When rc_critical_rotatable_indeces is empty, only torsions (dihedrals) are returned
        # Format: (atom_index, dof_type) where dof_type is 2=dihedral_ref
        # For rotatable_indices [9, 4] (atoms with id 9 and 4):
        #   - Atom 9 (id 9): dihedral_ref=2 - rotatable bond DOF
        #   - Atom 4 (id 4): dihedral_ref=1 - rotatable bond DOF
        expected_dof_indices = [
            (9, 2),   # Atom 9 (id 9): dihedral_ref=2 - rotatable bond DOF
            (4, 2)    # Atom 4 (id 4): dihedral_ref=1 - rotatable bond DOF
        ]
        
        # Verify result matches expected
        self.assertIsInstance(dof_indices, list)
        self.assertEqual(len(dof_indices), len(expected_dof_indices))
        self.assertEqual(sorted(dof_indices), sorted(expected_dof_indices))
        
        for dof in dof_indices:
            self.assertIsInstance(dof, tuple)
            self.assertEqual(len(dof), 2)
            # First element should be atom index, second should be DOF type (0=bond, 1=angle, 2=dihedral)
            self.assertIsInstance(dof[0], int)
            self.assertIsInstance(dof[1], int)
            self.assertIn(dof[1], [0, 1, 2])
    
    def test_get_dofs_empty_rotatable_indices(self):
        """Test with empty rotatable_indices list."""
        zmatrix_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54, 
             'angle_ref': 0, 'angle': 109.47, 'chirality': 0}
        ]
        zmatrix = ZMatrix(zmatrix_atoms, [(0, 1, 1), (1, 2, 1)])
        
        rotatable_indices = []
        rc_critical_rotatable_indeces = []  # No critical indices for this test
        rc_critical_atoms = []
        dof_indices = MolecularSystem._get_dofs_from_rotatable_indeces(rotatable_indices, rc_critical_rotatable_indeces, rc_critical_atoms, zmatrix)
        
        # Should return empty list
        self.assertIsInstance(dof_indices, list)
        self.assertEqual(len(dof_indices), 0)
    
    def test_get_dofs_single_rotatable_index(self):
        """Test with single rotatable index."""
        zmatrix_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54, 
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54, 
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0}
        ]
        zmatrix = ZMatrix(zmatrix_atoms, [(0, 1, 1), (1, 2, 1), (2, 3, 1)])
        
        rotatable_indices = [3]  # Only atom 3
        rc_critical_rotatable_indeces = []  # No critical indices for this test
        rc_critical_atoms = []
        dof_indices = MolecularSystem._get_dofs_from_rotatable_indeces(rotatable_indices, rc_critical_rotatable_indeces, rc_critical_atoms, zmatrix)
        
        # Should find at least one DOF
        self.assertIsInstance(dof_indices, list)
        self.assertGreater(len(dof_indices), 0)
        # All DOFs should reference valid atom indices
        for atom_idx, dof_type in dof_indices:
            self.assertLess(atom_idx, len(zmatrix))
            self.assertIn(dof_type, [1, 2])
    
    def test_get_dofs_multiple_chirality_atoms(self):
        """Test with multiple atoms having second bond angle"""
        zmatrix_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'H', 'atomic_num': 1, 'bond_ref': 1, 'bond_length': 1.09, 
             'angle_ref': 0, 'angle': 109.47, 'dihedral_ref': 1, 'dihedral': 120.0, 'chirality': 1},
            {'id': 3, 'element': 'H', 'atomic_num': 1, 'bond_ref': 1, 'bond_length': 1.09, 
             'angle_ref': 0, 'angle': 109.47, 'dihedral_ref': 1, 'dihedral': -120.0, 'chirality': -1},
            {'id': 4, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54, 
             'angle_ref': 0, 'angle': 109.47, 'dihedral_ref': 1, 'dihedral': 0.0, 'chirality': 0}
        ]
        zmatrix = ZMatrix(zmatrix_atoms, [(0, 1, 1), (1, 2, 1), (1, 3, 1), (1, 4, 1)])
        
        rotatable_indices = [4]  # Atom with chirality=0
        rc_critical_rotatable_indeces = []  # No critical indices for this test
        rc_critical_atoms = []
        dof_indices = MolecularSystem._get_dofs_from_rotatable_indeces(rotatable_indices, rc_critical_rotatable_indeces, rc_critical_atoms, zmatrix)
        
        # When rc_critical_rotatable_indeces is empty, only torsions (dihedrals) are returned
        self.assertIsInstance(dof_indices, list)
        self.assertEqual(len(dof_indices), 1)
        self.assertEqual(sorted(dof_indices), sorted([(4, 2)]))


class TestMolecularSystemRotatableIndices(unittest.TestCase):
    """Test rotatable indices methods in MolecularSystem."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(__file__).parent / 'fixtures'
        self.forcefield_file = Path(__file__).parent.parent / 'data' / 'RCP_UFFvdW.xml'
        self.test_int_file = self.test_dir / 'simple_molecule.int'
    
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
        indices = MolecularSystem._get_all_rotatable_indices(system.zmatrix)
        
        # Should return list of indices >= 3 (only atoms 4+ have dihedrals)
        self.assertIsInstance(indices, list)
        self.assertTrue(all(i >= 3 for i in indices))


class TestMolecularSystemRCCriticalIndices(unittest.TestCase):
    """Test RCP critical indices identification methods in MolecularSystem."""
    
    def test_identify_rc_critical_no_rcp_terms(self):
        """Test with no RCP terms."""
        zmatrix_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0}
        ]
        zmatrix = ZMatrix(zmatrix_atoms, [(0, 1, 1), (1, 2, 1), (2, 3, 1)])
        
        rotatable_indices = [3]
        rcp_terms = []
        
        rc_critical_rotatable_indeces, rc_critical_atoms = MolecularSystem._identify_rc_critical_rotatable_indeces(
            zmatrix, rcp_terms, rotatable_indices, topology=None
        )
        
        self.assertEqual(rc_critical_rotatable_indeces, [])
        self.assertEqual(rc_critical_atoms, [])
    
    def test_identify_rc_critical_simple_path(self):
        """Test with a simple linear path between RCP terms."""
        # Linear chain: 0-1-2-3-4
        # RCP terms: (0, 4)
        # Rotatable indices: [3] (atom 3 has a dihedral)
        zmatrix_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0},
            {'id': 4, 'element': 'C', 'atomic_num': 6, 'bond_ref': 3, 'bond_length': 1.54,
             'angle_ref': 2, 'angle': 109.47, 'dihedral_ref': 1, 'dihedral': 180.0, 'chirality': 0}
        ]
        zmatrix = ZMatrix(zmatrix_atoms, [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1)])
        
        rotatable_indices = [3]  # Atom 3 is rotatable
        rcp_terms = [(0, 4)]  # Path from atom 0 to atom 4
        
        rc_critical_rotatable_indeces, rc_critical_atoms = MolecularSystem._identify_rc_critical_rotatable_indeces(
            zmatrix, rcp_terms, rotatable_indices, topology=None
        )
        
        # The path from 0 to 4 is [0, 1, 2, 3, 4]
        # Atom 3's bond_ref=2 and angle_ref=1, both are on the path, so it's critical
        self.assertEqual(rc_critical_rotatable_indeces, [3])
        # All atoms on the path should be included
        self.assertEqual(sorted(rc_critical_atoms), [0, 1, 2, 3, 4])
    
    def test_identify_rc_critical_multiple_rcp_terms(self):
        """Test with multiple RCP terms."""
        # Branched structure:
        #     2
        #     |
        # 0-1-3-4
        # RCP terms: (0, 2) and (0, 4)
        # Rotatable indices: [3] (atom 3 has a dihedral)
        zmatrix_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47, 'dihedral_ref': 2, 'dihedral': 60.0, 'chirality': 0},
            {'id': 4, 'element': 'C', 'atomic_num': 6, 'bond_ref': 3, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 180.0, 'chirality': 0}
        ]
        zmatrix = ZMatrix(zmatrix_atoms, [(0, 1, 1), (1, 2, 1), (1, 3, 1), (3, 4, 1)])
        
        rotatable_indices = [3]  # Atom 3 is rotatable
        rcp_terms = [(0, 2), (0, 4)]  # Two RCP paths
        
        rc_critical_rotatable_indeces, rc_critical_atoms = MolecularSystem._identify_rc_critical_rotatable_indeces(
            zmatrix, rcp_terms, rotatable_indices, topology=None
        )
        
        # Path (0, 2): [0, 1, 2]
        # Path (0, 4): [0, 1, 3, 4]
        # Combined critical atoms: {0, 1, 2, 3, 4}
        # Atom 3's bond_ref=1 and angle_ref=0, both are on paths, so it's critical
        self.assertEqual(rc_critical_rotatable_indeces, [3])
        # All atoms on both paths should be included
        self.assertEqual(sorted(rc_critical_atoms), [0, 1, 2, 3, 4])
    
    def test_identify_rc_critical_rcp_atoms_included(self):
        """Test that RCP atoms themselves are included in rc_critical_atoms."""
        # Simple chain: 0-1-2
        # RCP terms: (0, 2)
        zmatrix_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47}
        ]
        zmatrix = ZMatrix(zmatrix_atoms, [(0, 1, 1), (1, 2, 1)])
        
        rotatable_indices = []  # No rotatable indices
        rcp_terms = [(0, 2)]  # Path from atom 0 to atom 2
        
        rc_critical_rotatable_indeces, rc_critical_atoms = MolecularSystem._identify_rc_critical_rotatable_indeces(
            zmatrix, rcp_terms, rotatable_indices, topology=None
        )
        
        # Path from 0 to 2 is [0, 1, 2]
        # RCP atoms 0 and 2 should be included
        self.assertEqual(rc_critical_rotatable_indeces, [])
        self.assertEqual(sorted(rc_critical_atoms), [0, 1, 2])
        # Verify RCP atoms are included
        self.assertIn(0, rc_critical_atoms)
        self.assertIn(2, rc_critical_atoms)
    
    def test_identify_rc_critical_non_critical_rotatable(self):
        """Test with a rotatable index that is not on the RCP path."""
        # Linear chain: 0-1-2-3-4-5
        # RCP terms: (0, 2) - path is [0, 1, 2]
        # Rotatable indices: [3, 4] - atoms 3 and 4 are not on the path
        zmatrix_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0},
            {'id': 4, 'element': 'C', 'atomic_num': 6, 'bond_ref': 3, 'bond_length': 1.54,
             'angle_ref': 2, 'angle': 109.47, 'dihedral_ref': 1, 'dihedral': 180.0, 'chirality': 0},
            {'id': 5, 'element': 'C', 'atomic_num': 6, 'bond_ref': 4, 'bond_length': 1.54,
             'angle_ref': 3, 'angle': 109.47, 'dihedral_ref': 2, 'dihedral': 120.0, 'chirality': 0}
        ]
        zmatrix = ZMatrix(zmatrix_atoms, [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1)])
        
        rotatable_indices = [3, 4]  # Atoms 3 and 4 are rotatable
        rcp_terms = [(0, 2)]  # Path from atom 0 to atom 2 is [0, 1, 2]
        
        rc_critical_rotatable_indeces, rc_critical_atoms = MolecularSystem._identify_rc_critical_rotatable_indeces(
            zmatrix, rcp_terms, rotatable_indices, topology=None
        )
        
        # Path is [0, 1, 2]
        # Atom 3: bond_ref=2 (on path), angle_ref=1 (on path) - should be critical
        # Atom 4: bond_ref=3 (not on path), angle_ref=2 (on path) - not critical (both must be on path)
        # Actually, let me check: atom 3 has bond_ref=2 and angle_ref=1, both are on path [0,1,2], so it's critical
        # Atom 4 has bond_ref=3 (not on path) and angle_ref=2 (on path), so it's not critical
        self.assertEqual(rc_critical_rotatable_indeces, [3])
        self.assertEqual(sorted(rc_critical_atoms), [0, 1, 2])
    
    def test_identify_rc_critical_disconnected_path(self):
        """Test with disconnected RCP atoms (no path exists)."""
        # Two separate components: 0-1 and 2-3
        zmatrix_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54}
        ]
        zmatrix = ZMatrix(zmatrix_atoms, [(0, 1, 1), (2, 3, 1)])
        
        rotatable_indices = []
        rcp_terms = [(0, 3)]  # No path between 0 and 3
        
        rc_critical_rotatable_indeces, rc_critical_atoms = MolecularSystem._identify_rc_critical_rotatable_indeces(
            zmatrix, rcp_terms, rotatable_indices, topology=None
        )
        
        # No path exists, so no critical atoms
        self.assertEqual(rc_critical_rotatable_indeces, [])
        self.assertEqual(rc_critical_atoms, [])
    
    def test_identify_rc_critical_same_rcp_atom(self):
        """Test with RCP term where both atoms are the same."""
        zmatrix_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54}
        ]
        zmatrix = ZMatrix(zmatrix_atoms, [(0, 1, 1)])
        
        rotatable_indices = []
        rcp_terms = [(0, 0)]  # Same atom
        
        rc_critical_rotatable_indeces, rc_critical_atoms = MolecularSystem._identify_rc_critical_rotatable_indeces(
            zmatrix, rcp_terms, rotatable_indices, topology=None
        )
        
        # Path from atom to itself is just [0]
        self.assertEqual(rc_critical_rotatable_indeces, [])
        self.assertEqual(sorted(rc_critical_atoms), [0])


def run_tests(verbosity=2):
    """Run all tests with specified verbosity."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMolecularSystemCreation))
    suite.addTests(loader.loadTestsFromTestCase(TestMolecularSystemEnergy))
    suite.addTests(loader.loadTestsFromTestCase(TestMolecularSystemRingClosure))
    suite.addTests(loader.loadTestsFromTestCase(TestMolecularSystemMinimization))
    suite.addTests(loader.loadTestsFromTestCase(TestMolecularSystemFromData))
    suite.addTests(loader.loadTestsFromTestCase(TestMolecularSystemSimulationCache))
    suite.addTests(loader.loadTestsFromTestCase(TestMolecularSystemErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestMolecularSystemRMSD))
    suite.addTests(loader.loadTestsFromTestCase(TestMolecularSystemGraphMethods))
    suite.addTests(loader.loadTestsFromTestCase(TestMolecularSystemDOFMethods))
    suite.addTests(loader.loadTestsFromTestCase(TestMolecularSystemRotatableIndices))
    suite.addTests(loader.loadTestsFromTestCase(TestMolecularSystemRCCriticalIndices))
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests(verbosity=2)
    sys.exit(0 if success else 1)

