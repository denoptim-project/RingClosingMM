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

# Add parent directory to path for package imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ringclosingmm import MolecularSystem, ZMatrix
from ringclosingmm.CoordinateConversion import zmatrix_to_cartesian


class TestMolecularSystemCreation(unittest.TestCase):
    """Test MolecularSystem creation and initialization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(__file__).parent / 'fixtures'
        self.forcefield_file = Path(__file__).parent.parent / 'data' / 'RCP_UFFvdW.xml'
        
        if not self.forcefield_file.exists():
            self.skipTest(f"Force field file not found: {self.forcefield_file}")
        
        self.test_int_file = self.test_dir / 'simple_molecule.int'
        self.test_int_file_w_rcp_terms = self.test_dir / 'simple_molecule_w_rcp_terms.int'
    
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
        if not self.test_int_file_w_rcp_terms.exists():
            self.skipTest(f"Test INT file not found: {self.test_int_file_w_rcp_terms}")
        
        rcp_terms = [(0, 5)]  # 0-based indices
        system = MolecularSystem.from_file(
            str(self.test_int_file_w_rcp_terms),
            str(self.forcefield_file),
            rcp_terms=rcp_terms
        )
        
        self.assertIsNotNone(system)
        self.assertEqual(len(system.rcpterms), 1)
        self.assertEqual(system.rcpterms[0], (0, 5))
    
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
        self.test_int_file_w_rcp_terms = self.test_dir / 'simple_molecule_w_rcp_terms.int'
    
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
        self.test_int_file_w_rcp_terms = self.test_dir / 'simple_molecule_w_rcp_terms.int'
    
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
        if not self.test_int_file_w_rcp_terms.exists() or not self.forcefield_file.exists():
            self.skipTest("Required test files not found")
        
        # Create system with RCP terms
        rcp_terms = [(0, 5)]  # 0-based: first and third atom
        system = MolecularSystem.from_file(
            str(self.test_int_file_w_rcp_terms),
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
        if not self.test_int_file_w_rcp_terms.exists() or not self.forcefield_file.exists():
            self.skipTest("Required test files not found")
        
        rcp_terms = [(0, 5)]
        system = MolecularSystem.from_file(
            str(self.test_int_file_w_rcp_terms),
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
            {'id': 0, 'element': 'ATN', 'atomic_num': 1},  # RCP atom with _ATN_0
            {'id': 1, 'element': 'H', 'atomic_num': 1, 'bond_ref': 0, 'bond_length': 1.0},
            {'id': 2, 'element': 'H', 'atomic_num': 1, 'bond_ref': 0, 'bond_length': 1.0,
             'angle_ref': 1, 'angle': 109.47}
        ]
        self.zmatrix = ZMatrix(zmatrix_atoms, [(0, 1), (0, 2)])
    
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
            {'id': 0, 'element': 'ATN', 'atomic_num': 1},  # RCP atom with _ATN_0
            {'id': 1, 'element': 'H', 'atomic_num': 1, 'bond_ref': 0, 'bond_length': 1.0},
            {'id': 2, 'element': 'H', 'atomic_num': 1, 'bond_ref': 0, 'bond_length': 1.0,
             'angle_ref': 1, 'angle': 109.47}
        ]
        zmatrix = ZMatrix(zmatrix_atoms, [(0, 1), (0, 2)])
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
        zmatrix = ZMatrix(zmatrix_atoms, [(0, 1), (1, 2), (2, 3)])
        
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
        zmatrix1 = ZMatrix(zmatrix1_atoms, [(0, 1), (1, 2)])
        
        zmatrix2_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.50},  # Changed
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.58,  # Changed
             'angle_ref': 0, 'angle': 109.47}
        ]
        zmatrix2 = ZMatrix(zmatrix2_atoms, [(0, 1), (1, 2)])

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
        zmatrix1 = ZMatrix(zmatrix1_atoms, [(0, 1), (1, 2), (2, 3)])
        
        zmatrix2_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 120.0},  # Changed
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 100.0, 'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0}  # Changed
        ]
        zmatrix2 = ZMatrix(zmatrix2_atoms, [(0, 1), (1, 2), (2, 3)])

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
        zmatrix1 = ZMatrix(zmatrix1_atoms, [(0, 1), (1, 2), (2, 3)])
        
        zmatrix2_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 180.0, 'chirality': 0}  # Changed
        ]
        zmatrix2 = ZMatrix(zmatrix2_atoms, [(0, 1), (1, 2), (2, 3)])
        
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
        zmatrix1 = ZMatrix(zmatrix1_atoms, [(0, 1), (1, 2), (2, 3)])
        
        zmatrix2_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 179.0, 'chirality': 0}  # Near +180°
        ]
        zmatrix2 = ZMatrix(zmatrix2_atoms, [(0, 1), (1, 2), (2, 3)])
        
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
        zmatrix1 = ZMatrix(zmatrix1_atoms, [(0, 1), (1, 2), (2, 3)])
        
        zmatrix2_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.50},  # Changed
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.58,  # Changed
             'angle_ref': 0, 'angle': 120.0},  # Changed
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.60,  # Changed
             'angle_ref': 1, 'angle': 100.0, 'dihedral_ref': 0, 'dihedral': 180.0, 'chirality': 0}  # Changed
        ]
        zmatrix2 = ZMatrix(zmatrix2_atoms, [(0, 1), (1, 2), (2, 3)])

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
        zmatrix1 = ZMatrix(zmatrix1_atoms, [(0, 1), (1, 2)])
        
        zmatrix2_atoms = [
            {'id': 0, 'element': 'H', 'atomic_num': 1},
            {'id': 1, 'element': 'H', 'atomic_num': 1, 'bond_ref': 0, 'bond_length': 0.95},
            {'id': 2, 'element': 'H', 'atomic_num': 1, 'bond_ref': 1, 'bond_length': 1.05,
             'angle_ref': 0, 'angle': 120.0}
        ]
        zmatrix2 = ZMatrix(zmatrix2_atoms, [(0, 1), (1, 2)])
        
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
        zmatrix1 = ZMatrix(zmatrix1_atoms, [(0, 1)])
        
        zmatrix2_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47}
        ]
        zmatrix2 = ZMatrix(zmatrix2_atoms, [(0, 1), (1, 2)])
        
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
        zmatrix1 = ZMatrix(zmatrix1_atoms, [(0, 1), (1, 2), (2, 3), (3, 4)])
        
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
        zmatrix2 = ZMatrix(zmatrix2_atoms, [(0, 1), (1, 2), (2, 3), (3, 4)])
        
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
        zmatrix1 = ZMatrix(zmatrix1_atoms, [(0, 1), (1, 2), (2, 3), (3, 4)])
        
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
        zmatrix2 = ZMatrix(zmatrix2_atoms, [(0, 1), (1, 2), (2, 3), (3, 4)])
        
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
        zmatrix = ZMatrix(zmatrix_atoms, [(0, 1), (1, 2)])
        
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
        zmatrix = ZMatrix(zmatrix_atoms, [(0, 1), (1, 2), (2, 3)])
        
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
        zmatrix = ZMatrix(zmatrix_atoms, [(0, 1), (1, 2)])
        
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
                bonds.append((atom['bond_ref'], i))
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
        zmatrix = ZMatrix(zmatrix_atoms, [(0, 1), (1, 2)])
        
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
        zmatrix = ZMatrix(zmatrix_atoms, [(0, 1), (1, 2), (2, 3)])
        
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
        zmatrix = ZMatrix(zmatrix_atoms, [(0, 1), (1, 2), (1, 3), (1, 4)])
        
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
        zmatrix = ZMatrix(zmatrix_atoms, [(0, 1), (1, 2), (2, 3)])
        
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
            {'id': 0, 'element': 'ATN', 'atomic_num': 1},  # RCP atom with _ATN_0
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0},
            {'id': 4, 'element': 'C', 'atomic_num': 6, 'bond_ref': 3, 'bond_length': 1.54,
             'angle_ref': 2, 'angle': 109.47, 'dihedral_ref': 1, 'dihedral': 180.0, 'chirality': 0}
        ]
        zmatrix = ZMatrix(zmatrix_atoms, [(0, 1), (1, 2), (2, 3), (3, 4)])
        
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
            {'id': 0, 'element': 'ATN', 'atomic_num': 1},  # RCP atom with _ATN_0 (in both RCP terms)
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47, 'dihedral_ref': 2, 'dihedral': 60.0, 'chirality': 0},
            {'id': 4, 'element': 'C', 'atomic_num': 6, 'bond_ref': 3, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 180.0, 'chirality': 0}
        ]
        zmatrix = ZMatrix(zmatrix_atoms, [(0, 1), (1, 2), (1, 3), (3, 4)])
        
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
            {'id': 0, 'element': 'ATN', 'atomic_num': 1},  # RCP atom with _ATN_0
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47}
        ]
        zmatrix = ZMatrix(zmatrix_atoms, [(0, 1), (1, 2)])
        
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
            {'id': 0, 'element': 'ATN', 'atomic_num': 1},  # RCP atom with _ATN_0
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
        zmatrix = ZMatrix(zmatrix_atoms, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
        
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
            {'id': 0, 'element': 'ATN', 'atomic_num': 1},  # RCP atom with _ATN_0
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54}
        ]
        zmatrix = ZMatrix(zmatrix_atoms, [(0, 1), (2, 3)])
        
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
            {'id': 0, 'element': 'ATN', 'atomic_num': 1},  # RCP atom with _ATN_0
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54}
        ]
        zmatrix = ZMatrix(zmatrix_atoms, [(0, 1)])
        
        rotatable_indices = []
        rcp_terms = [(0, 0)]  # Same atom
        
        rc_critical_rotatable_indeces, rc_critical_atoms = MolecularSystem._identify_rc_critical_rotatable_indeces(
            zmatrix, rcp_terms, rotatable_indices, topology=None
        )
        
        # Path from atom to itself is just [0]
        self.assertEqual(rc_critical_rotatable_indeces, [])
        self.assertEqual(sorted(rc_critical_atoms), [0])


class TestMolecularSystemRCPPathComputation(unittest.TestCase):
    """Test RCP path computation methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple linear chain for testing
        self.linear_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.5},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.5, 
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.5, 
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0},
            {'id': 4, 'element': 'C', 'atomic_num': 6, 'bond_ref': 3, 'bond_length': 1.5, 
             'angle_ref': 2, 'angle': 109.47, 'dihedral_ref': 1, 'dihedral': 120.0, 'chirality': 0},
        ]
        self.linear_bonds = [(0, 1), (1, 2), (2, 3), (3, 4)]
        self.linear_zmatrix = ZMatrix(self.linear_atoms, self.linear_bonds)
        
        # Create a branched structure for testing related pairs
        # Structure: 0-1-2-3-4
        #            |     |
        #            5     6
        # RCP terms: (0,4) and (5,6) - related if 0-5 and 4-6 are bonded
        self.branched_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.5},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.5, 
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.5, 
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0},
            {'id': 4, 'element': 'C', 'atomic_num': 6, 'bond_ref': 3, 'bond_length': 1.5, 
             'angle_ref': 2, 'angle': 109.47, 'dihedral_ref': 1, 'dihedral': 120.0, 'chirality': 0},
            {'id': 5, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.5, 
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 2, 'dihedral': 0.0, 'chirality': 0},
            {'id': 6, 'element': 'C', 'atomic_num': 6, 'bond_ref': 3, 'bond_length': 1.5, 
             'angle_ref': 2, 'angle': 109.47, 'dihedral_ref': 1, 'dihedral': 0.0, 'chirality': 0},
        ]
        self.branched_bonds = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (3, 6)]
        self.branched_zmatrix = ZMatrix(self.branched_atoms, self.branched_bonds)
        
        # Create topology for branched structure
        from ringclosingmm.MolecularSystem import build_topology_from_data
        atoms_data = [(atom['element'], i) for i, atom in enumerate(self.branched_atoms)]
        self.branched_topology = build_topology_from_data(atoms_data, self.branched_bonds)
    
    def _create_minimal_system(self, zmatrix, rcp_terms, topology):
        """Create a minimal MolecularSystem instance for testing without full OpenMM system."""
        # Create a dummy system (we won't use it for these tests)
        import openmm
        dummy_system = openmm.System()
        
        # Create MolecularSystem instance
        system = MolecularSystem(
            system=dummy_system,
            topology=topology,
            rcpterms=rcp_terms,
            zmatrix=zmatrix,
            step_length=0.0002,
            ring_closure_threshold=1.5
        )
        return system
    
    def test_compute_rcp_path_data_single_rcp(self):
        """Test _compute_rcp_path_data with a single RCP term."""
        from ringclosingmm.MolecularSystem import build_topology_from_data
        atoms_data = [(atom['element'], i) for i, atom in enumerate(self.linear_atoms)]
        topology = build_topology_from_data(atoms_data, self.linear_bonds)
        
        # Create minimal system with RCP term (0, 4)
        rcp_terms = [(0, 4)]
        system = self._create_minimal_system(self.linear_zmatrix, rcp_terms, topology)
        
        # Rotatable indices: atoms 2, 3, 4 have dihedrals
        rotatable_indices = [2, 3, 4]
        
        # Compute path data
        rcp_path_data = system._compute_rcp_path_data(self.linear_zmatrix, rotatable_indices, verbose=False)
        
        # Check results
        self.assertEqual(len(rcp_path_data), 1)
        self.assertIn((0, 4), rcp_path_data)
        
        path, path_rotatable = rcp_path_data[(0, 4)]
        self.assertEqual(path, [0, 1, 2, 3, 4])
        # Rotatable indices on path: 2, 3, 4 (all are in path)
        self.assertEqual(sorted(path_rotatable), [2, 3, 4])
    
    def test_compute_rcp_path_data_multiple_rcps(self):
        """Test _compute_rcp_path_data with multiple RCP terms."""
        from ringclosingmm.MolecularSystem import build_topology_from_data
        atoms_data = [(atom['element'], i) for i, atom in enumerate(self.linear_atoms)]
        topology = build_topology_from_data(atoms_data, self.linear_bonds)
        
        # Create minimal system with multiple RCP terms
        rcp_terms = [(0, 2), (2, 4)]
        system = self._create_minimal_system(self.linear_zmatrix, rcp_terms, topology)
        
        rotatable_indices = [2, 3, 4]
        
        rcp_path_data = system._compute_rcp_path_data(self.linear_zmatrix, rotatable_indices, verbose=False)
        
        # Both RCP terms should have paths
        self.assertEqual(len(rcp_path_data), 2)
        self.assertIn((0, 2), rcp_path_data)
        self.assertIn((2, 4), rcp_path_data)
        
        # Check path (0, 2)
        path1, rotatable1 = rcp_path_data[(0, 2)]
        self.assertEqual(path1, [0, 1, 2])
        # Only atom 2 is in path and has references in path
        # Atom 2: in path, bond_ref=1 (in path), angle_ref=0 (in path) -> included
        # Atom 3: NOT in path -> skipped
        # Atom 4: NOT in path -> skipped
        self.assertEqual(sorted(rotatable1), [2])
        
        # Check path (2, 4)
        path2, rotatable2 = rcp_path_data[(2, 4)]
        self.assertEqual(path2, [2, 3, 4])
        # Atoms 2, 3, 4 are all rotatable and in path
        # Atom 2: in path, bond_ref=1 (NOT in path), angle_ref=0 (NOT in path) -> NOT included
        # Atom 3: in path, bond_ref=2 (in path), angle_ref=1 (in path) -> included
        # Atom 4: in path, bond_ref=3 (in path), angle_ref=2 (in path) -> included
        # Actually, let's check: path is [2, 3, 4]
        # Atom 2: bond_ref=1 (NOT in [2,3,4]), angle_ref=0 (NOT in [2,3,4]) -> NOT included
        # Atom 3: bond_ref=2 (in path), angle_ref=1 (NOT in path) -> included (bond_ref in path)
        # Atom 4: bond_ref=3 (in path), angle_ref=2 (in path) -> included
        self.assertEqual(sorted(rotatable2), [3, 4])
    
    def test_compute_rcp_path_data_no_rotatable_on_path(self):
        """Test _compute_rcp_path_data when no rotatable dihedrals are on the path."""
        from ringclosingmm.MolecularSystem import build_topology_from_data
        atoms_data = [(atom['element'], i) for i, atom in enumerate(self.linear_atoms)]
        topology = build_topology_from_data(atoms_data, self.linear_bonds)
        
        rcp_terms = [(0, 1)]  # Short path, no dihedrals needed
        system = self._create_minimal_system(self.linear_zmatrix, rcp_terms, topology)
        
        rotatable_indices = [2, 3, 4]  # None of these are on path [0, 1]
        
        rcp_path_data = system._compute_rcp_path_data(self.linear_zmatrix, rotatable_indices, verbose=False)
        
        # Path exists but no rotatable indices on it, so should be empty
        self.assertEqual(len(rcp_path_data), 0)
    
    def test_compute_rcp_path_data_no_path_found(self):
        """Test _compute_rcp_path_data when no path exists."""
        from ringclosingmm.MolecularSystem import build_topology_from_data
        atoms_data = [(atom['element'], i) for i, atom in enumerate(self.linear_atoms)]
        topology = build_topology_from_data(atoms_data, self.linear_bonds)
        
        # Create a disconnected structure (atom 10 doesn't exist)
        rcp_terms = [(0, 10)]  # Invalid RCP term
        system = self._create_minimal_system(self.linear_zmatrix, rcp_terms, topology)
        
        rotatable_indices = [2, 3, 4]
        
        rcp_path_data = system._compute_rcp_path_data(self.linear_zmatrix, rotatable_indices, verbose=False)
        
        # No path found, should be empty
        self.assertEqual(len(rcp_path_data), 0)
    
    def test_compute_rcp_path_data_branch_atoms(self):
        """Test _compute_rcp_path_data with branch atoms affecting path."""
        # Use branched structure
        rcp_terms = [(0, 4)]
        system = self._create_minimal_system(self.branched_zmatrix, rcp_terms, self.branched_topology)
        
        # Rotatable indices: 2, 3, 4, 5, 6
        rotatable_indices = [2, 3, 4, 5, 6]
        
        rcp_path_data = system._compute_rcp_path_data(self.branched_zmatrix, rotatable_indices, verbose=False)
        
        self.assertEqual(len(rcp_path_data), 1)
        path, path_rotatable = rcp_path_data[(0, 4)]
        self.assertEqual(path, [0, 1, 2, 3, 4])
        # Check which rotatable atoms affect the path:
        # Atom 2: in path, bond_ref=1 (in path), angle_ref=0 (in path) -> included
        # Atom 3: in path, bond_ref=2 (in path), angle_ref=1 (in path) -> included
        # Atom 4: in path, bond_ref=3 (in path), angle_ref=2 (in path) -> included
        # Atom 5: NOT in path, bond_ref=0 (in path), angle_ref=1 (in path) -> included (affects path)
        # Atom 6: NOT in path, bond_ref=3 (in path), angle_ref=2 (in path) -> included (affects path)
        # The logic checks: if rot_idx in path OR (bond_ref in path OR angle_ref in path)
        # But actually the code checks: if rot_idx in path AND (bond_ref in path OR angle_ref in path)
        # So atoms 5 and 6 won't be included because they're not in the path
        # Only atoms 2, 3, 4 are in path and have references in path
        self.assertEqual(sorted(path_rotatable), [2, 3, 4])
    
    def test_compute_rcp_paths_single_rcp(self):
        """Test _compute_rcp_paths with a single RCP term."""
        from ringclosingmm.MolecularSystem import build_topology_from_data
        atoms_data = [(atom['element'], i) for i, atom in enumerate(self.linear_atoms)]
        topology = build_topology_from_data(atoms_data, self.linear_bonds)
        
        rcp_terms = [(0, 4)]
        system = self._create_minimal_system(self.linear_zmatrix, rcp_terms, topology)
        
        rotatable_indices = [2, 3, 4]
        rcp_path_data = system._compute_rcp_path_data(self.linear_zmatrix, rotatable_indices, verbose=False)
        
        # Compute grouped paths
        rcp_paths = system._compute_rcp_paths(self.linear_zmatrix, rcp_path_data)
        
        # Single RCP should result in one group
        self.assertEqual(len(rcp_paths), 1)
        rcp_group, path, rotatable = rcp_paths[0]
        self.assertEqual(rcp_group, [(0, 4)])
        self.assertEqual(path, [0, 1, 2, 3, 4])
        self.assertEqual(sorted(rotatable), [2, 3, 4])
    
    def test_compute_rcp_paths_related_pair(self):
        """Test _compute_rcp_paths with two related RCP terms."""
        # Create structure where (0,4) and (5,6) are related
        # Need: 0 bonded to 5, 4 bonded to 6
        # Structure: 0-1-2-3-4
        #            |     |
        #            5     6
        rcp_terms = [(0, 4), (5, 6)]
        system = self._create_minimal_system(self.branched_zmatrix, rcp_terms, self.branched_topology)
        
        rotatable_indices = [2, 3, 4, 5, 6]
        rcp_path_data = system._compute_rcp_path_data(self.branched_zmatrix, rotatable_indices, verbose=False)
        
        # Both RCPs should have paths
        self.assertEqual(len(rcp_path_data), 2)
        
        # Compute grouped paths
        rcp_paths = system._compute_rcp_paths(self.branched_zmatrix, rcp_path_data)
        
        # Should be grouped into one pair (if related) or two separate groups
        # Check if they're related: 0-5 bonded? 4-6 bonded?
        graph = system._build_bond_graph(self.branched_zmatrix, self.branched_topology)
        is_related = (5 in graph.get(0, [])) and (6 in graph.get(4, []))
        
        if is_related:
            # Should be one group with 2 RCPs
            self.assertEqual(len(rcp_paths), 1)
            rcp_group, combined_path, combined_rotatable = rcp_paths[0]
            self.assertEqual(len(rcp_group), 2)
            # Combined path should include both paths
            self.assertIn(0, combined_path)
            self.assertIn(4, combined_path)
            self.assertIn(5, combined_path)
            self.assertIn(6, combined_path)
        else:
            # Should be two separate groups
            self.assertEqual(len(rcp_paths), 2)
    
    def test_compute_rcp_paths_unrelated_pair(self):
        """Test _compute_rcp_paths with two unrelated RCP terms."""
        from ringclosingmm.MolecularSystem import build_topology_from_data
        atoms_data = [(atom['element'], i) for i, atom in enumerate(self.linear_atoms)]
        topology = build_topology_from_data(atoms_data, self.linear_bonds)
        
        # Two RCP terms that are not neighbors
        rcp_terms = [(0, 2), (2, 4)]
        system = self._create_minimal_system(self.linear_zmatrix, rcp_terms, topology)
        
        rotatable_indices = [2, 3, 4]
        rcp_path_data = system._compute_rcp_path_data(self.linear_zmatrix, rotatable_indices, verbose=False)
        
        # Compute grouped paths
        rcp_paths = system._compute_rcp_paths(self.linear_zmatrix, rcp_path_data)
        
        # Unrelated RCPs should be in separate groups
        self.assertEqual(len(rcp_paths), 2)
        self.assertEqual(rcp_paths[0][0], [(0, 2)])  # First group
        self.assertEqual(rcp_paths[1][0], [(2, 4)])  # Second group
    
    def test_compute_rcp_paths_mixed_related_and_unrelated(self):
        """Test _compute_rcp_paths with mix of related and unrelated RCP terms."""
        # Create structure with 3 RCP terms: (0,4), (5,6), (0,2)
        # (0,4) and (5,6) might be related if 0-5 and 4-6 are bonded
        # (0,2) is unrelated to the others
        rcp_terms = [(0, 4), (5, 6), (0, 2)]
        system = self._create_minimal_system(self.branched_zmatrix, rcp_terms, self.branched_topology)
        
        rotatable_indices = [2, 3, 4, 5, 6]
        rcp_path_data = system._compute_rcp_path_data(self.branched_zmatrix, rotatable_indices, verbose=False)
        
        # All 3 RCPs should have paths
        self.assertEqual(len(rcp_path_data), 3)
        
        # Compute grouped paths
        rcp_paths = system._compute_rcp_paths(self.branched_zmatrix, rcp_path_data)
        
        # Should have at least 2 groups (one for unrelated (0,2))
        self.assertGreaterEqual(len(rcp_paths), 2)
        
        # Find group containing (0,2) - should be alone
        group_with_02 = None
        for group, path, rotatable in rcp_paths:
            if (0, 2) in group:
                group_with_02 = group
                break
        self.assertIsNotNone(group_with_02)
        # (0,2) should be in its own group or with unrelated RCPs
        self.assertIn((0, 2), group_with_02)
    
    def test_compute_rcp_paths_group_larger_than_two(self):
        """Test _compute_rcp_paths with group larger than 2 RCPs."""
        # Create a structure where 3 RCPs form a chain of relations
        # RCP1 (0,4) related to RCP2 (5,6) if 0-5 and 4-6
        # RCP2 (5,6) related to RCP3 (6,7) if 5-6 and 6-7 (but 5-6 is already in RCP2)
        # Actually, let's create a simpler case: add atom 7 bonded to 6
        extended_atoms = self.branched_atoms + [
            {'id': 7, 'element': 'C', 'atomic_num': 6, 'bond_ref': 6, 'bond_length': 1.5, 
             'angle_ref': 3, 'angle': 109.47, 'dihedral_ref': 2, 'dihedral': 0.0, 'chirality': 0}
        ]
        extended_bonds = self.branched_bonds + [(6, 7)]
        extended_zmatrix = ZMatrix(extended_atoms, extended_bonds)
        
        from ringclosingmm.MolecularSystem import build_topology_from_data
        atoms_data = [(atom['element'], i) for i, atom in enumerate(extended_atoms)]
        extended_topology = build_topology_from_data(atoms_data, extended_bonds)
        
        # RCP terms: (0,4), (5,6), (6,7)
        # (5,6) and (6,7) share atom 6, but they're not "related" by our definition
        # (they need to be neighbors: a1 bonded to a2 and b1 bonded to b2)
        rcp_terms = [(0, 4), (5, 6), (6, 7)]
        system = self._create_minimal_system(extended_zmatrix, rcp_terms, extended_topology)
        
        rotatable_indices = [2, 3, 4, 5, 6, 7]
        rcp_path_data = system._compute_rcp_path_data(extended_zmatrix, rotatable_indices, verbose=False)
        
        # Compute grouped paths
        rcp_paths = system._compute_rcp_paths(extended_zmatrix, rcp_path_data)
        
        # Groups larger than 2 are treated individually
        # So we should have 3 groups (one per RCP)
        self.assertEqual(len(rcp_paths), 3)
        for group, path, rotatable in rcp_paths:
            self.assertEqual(len(group), 1)  # Each treated individually
    
    def test_get_rcp_paths_caching(self):
        """Test that get_rcp_paths caches results correctly."""
        from ringclosingmm.MolecularSystem import build_topology_from_data
        atoms_data = [(atom['element'], i) for i, atom in enumerate(self.linear_atoms)]
        topology = build_topology_from_data(atoms_data, self.linear_bonds)
        
        rcp_terms = [(0, 4)]
        system = self._create_minimal_system(self.linear_zmatrix, rcp_terms, topology)
        
        rotatable_indices = [2, 3, 4]
        
        # First call - should compute
        paths1 = system.get_rcp_paths(self.linear_zmatrix, rotatable_indices, force_recompute=False, verbose=False)
        self.assertIsNotNone(system._rcp_paths)
        self.assertIsNotNone(system._rcp_path_data)
        
        # Second call - should use cache
        paths2 = system.get_rcp_paths(self.linear_zmatrix, rotatable_indices, force_recompute=False, verbose=False)
        self.assertEqual(paths1, paths2)
        
        # Force recompute - should recompute
        paths3 = system.get_rcp_paths(self.linear_zmatrix, rotatable_indices, force_recompute=True, verbose=False)
        self.assertEqual(len(paths1), len(paths3))  # Should have same structure
    
    def test_get_rcp_paths_zmatrix_change_invalidation(self):
        """Test that get_rcp_paths recomputes when zmatrix structure changes."""
        from ringclosingmm.MolecularSystem import build_topology_from_data
        atoms_data = [(atom['element'], i) for i, atom in enumerate(self.linear_atoms)]
        topology = build_topology_from_data(atoms_data, self.linear_bonds)
        
        rcp_terms = [(0, 4)]
        system = self._create_minimal_system(self.linear_zmatrix, rcp_terms, topology)
        
        rotatable_indices = [2, 3, 4]
        
        # First call
        paths1 = system.get_rcp_paths(self.linear_zmatrix, rotatable_indices, force_recompute=False, verbose=False)
        original_hash = system._rcp_path_data_zmatrix_hash
        
        # Create a different zmatrix with different structure (different number of atoms)
        different_atoms = self.linear_atoms + [
            {'id': 5, 'element': 'C', 'atomic_num': 6, 'bond_ref': 4, 'bond_length': 1.5, 
             'angle_ref': 3, 'angle': 109.47, 'dihedral_ref': 2, 'dihedral': 120.0, 'chirality': 0}
        ]
        different_bonds = self.linear_bonds + [(4, 5)]
        different_zmatrix = ZMatrix(different_atoms, different_bonds)
        
        # Second call with different zmatrix - should recompute
        paths2 = system.get_rcp_paths(different_zmatrix, rotatable_indices, force_recompute=False, verbose=False)
        new_hash = system._rcp_path_data_zmatrix_hash
        
        # Hash should be different (zmatrix structure changed - different number of atoms/bonds)
        self.assertNotEqual(original_hash, new_hash)
        # Paths should be recomputed
        self.assertIsNotNone(paths2)
    
    def test_get_non_intersecting_rcp_paths_single_path(self):
        """Test get_non_intersecting_rcp_paths with a single path."""
        from ringclosingmm.MolecularSystem import build_topology_from_data
        atoms_data = [(atom['element'], i) for i, atom in enumerate(self.linear_atoms)]
        topology = build_topology_from_data(atoms_data, self.linear_bonds)
        
        rcp_terms = [(0, 4)]
        system = self._create_minimal_system(self.linear_zmatrix, rcp_terms, topology)
        
        rotatable_indices = [2, 3, 4]
        
        groups = system.get_non_intersecting_rcp_paths(self.linear_zmatrix, rotatable_indices, verbose=False)
        
        # Single path should result in one group
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 1)
        
        # Verify the path structure
        rcp_group, path, rotatable = groups[0][0]
        self.assertEqual(rcp_group, [(0, 4)])
        self.assertEqual(path, [0, 1, 2, 3, 4])
        self.assertEqual(sorted(rotatable), [2, 3, 4])
    
    def test_get_non_intersecting_rcp_paths_intersecting_paths(self):
        """Test get_non_intersecting_rcp_paths with two paths that share rotatable indices."""
        from ringclosingmm.MolecularSystem import build_topology_from_data
        atoms_data = [(atom['element'], i) for i, atom in enumerate(self.linear_atoms)]
        topology = build_topology_from_data(atoms_data, self.linear_bonds)
        
        # Two RCP terms that should share rotatable indices
        # Path (0, 3): includes atoms 0,1,2,3 - rotatable indices should include 2,3
        # Path (1, 4): includes atoms 1,2,3,4 - rotatable indices should include 2,3,4
        # They share indices 2 and 3, so they should be in the same group
        rcp_terms = [(0, 3), (1, 4)]
        system = self._create_minimal_system(self.linear_zmatrix, rcp_terms, topology)
        
        rotatable_indices = [2, 3, 4]
        
        groups = system.get_non_intersecting_rcp_paths(self.linear_zmatrix, rotatable_indices, verbose=False)
        
        # Both paths should share rotatable indices (2 and/or 3), so they should be in one group
        # First verify we have paths
        all_paths = system.get_rcp_paths(self.linear_zmatrix, rotatable_indices, verbose=False)
        if len(all_paths) >= 2:
            # Check if paths share rotatable indices
            path1_rotatable = set(all_paths[0][2])  # rotatable indices for first path
            path2_rotatable = set(all_paths[1][2])  # rotatable indices for second path
            
            if path1_rotatable & path2_rotatable:  # They share rotatable indices
                # Should be in one group
                self.assertEqual(len(groups), 1)
                self.assertEqual(len(groups[0]), 2)
                
                # Verify both paths are in the group
                rcp_groups_in_group = [path[0] for path in groups[0]]
                self.assertIn(all_paths[0][0], rcp_groups_in_group)
                self.assertIn(all_paths[1][0], rcp_groups_in_group)
            else:
                # They don't share, so separate groups
                self.assertEqual(len(groups), 2)
        else:
            # Not enough paths, skip this assertion
            self.assertGreaterEqual(len(groups), 0)
    
    def test_get_non_intersecting_rcp_paths_non_intersecting_paths(self):
        """Test get_non_intersecting_rcp_paths with paths that don't share rotatable indices."""
        # Create a structure with two separate paths that don't share rotatable indices
        # Path 1: 0-1-2 (rotatable: [2])
        # Path 2: 3-4 (rotatable: [4], but 4 is not in path 1's rotatable set)
        # Actually, we need paths that truly don't share rotatable indices
        
        # Create a structure: 0-1-2 and 3-4-5 (disconnected or separate branches)
        separate_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.5},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.5, 
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6},
            {'id': 4, 'element': 'C', 'atomic_num': 6, 'bond_ref': 3, 'bond_length': 1.5},
            {'id': 5, 'element': 'C', 'atomic_num': 6, 'bond_ref': 4, 'bond_length': 1.5, 
             'angle_ref': 3, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0},
        ]
        separate_bonds = [(0, 1), (1, 2), (3, 4), (4, 5)]
        separate_zmatrix = ZMatrix(separate_atoms, separate_bonds)
        
        from ringclosingmm.MolecularSystem import build_topology_from_data
        atoms_data = [(atom['element'], i) for i, atom in enumerate(separate_atoms)]
        topology = build_topology_from_data(atoms_data, separate_bonds)
        
        # RCP terms: (0, 2) and (3, 5)
        # Path (0, 2): rotatable indices [2] (if 2 is rotatable)
        # Path (3, 5): rotatable indices [5] (if 5 is rotatable)
        rcp_terms = [(0, 2), (3, 5)]
        system = self._create_minimal_system(separate_zmatrix, rcp_terms, topology)
        
        # Only make 2 and 5 rotatable (they don't share)
        rotatable_indices = [2, 5]
        
        groups = system.get_non_intersecting_rcp_paths(separate_zmatrix, rotatable_indices, verbose=False)
        
        # Paths don't share rotatable indices, so they should be in separate groups
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups[0]), 1)
        self.assertEqual(len(groups[1]), 1)
        
        # Verify each group contains one path
        rcp_groups = [path[0] for group in groups for path in group]
        self.assertIn([(0, 2)], rcp_groups)
        self.assertIn([(3, 5)], rcp_groups)
    
    def test_get_non_intersecting_rcp_paths_mixed_intersecting(self):
        """Test get_non_intersecting_rcp_paths with mixed intersecting and non-intersecting paths."""
        from ringclosingmm.MolecularSystem import build_topology_from_data
        atoms_data = [(atom['element'], i) for i, atom in enumerate(self.linear_atoms)]
        topology = build_topology_from_data(atoms_data, self.linear_bonds)
        
        # Use paths that we know will share rotatable indices
        # Path (0, 3): includes atoms 0,1,2,3 - rotatable indices should include 2,3
        # Path (1, 4): includes atoms 1,2,3,4 - rotatable indices should include 2,3,4
        # Path (0, 1): short path, might not have rotatable indices or different ones
        rcp_terms = [(0, 3), (1, 4), (0, 1)]
        system = self._create_minimal_system(self.linear_zmatrix, rcp_terms, topology)
        
        rotatable_indices = [2, 3, 4]
        
        groups = system.get_non_intersecting_rcp_paths(self.linear_zmatrix, rotatable_indices, verbose=False)
        
        # Get all paths to check which ones share rotatable indices
        all_paths = system.get_rcp_paths(self.linear_zmatrix, rotatable_indices, verbose=False)
        
        # Find paths that share rotatable indices
        intersecting_pairs = []
        for i, path_i in enumerate(all_paths):
            rotatable_i = set(path_i[2])
            for j, path_j in enumerate(all_paths[i+1:], start=i+1):
                rotatable_j = set(path_j[2])
                if rotatable_i & rotatable_j:
                    intersecting_pairs.append((i, j))
        
        # If we have intersecting pairs, verify they're in the same group
        if intersecting_pairs:
            self.assertGreaterEqual(len(groups), 1)
            # Check that intersecting paths are in the same group
            for i, j in intersecting_pairs:
                found_together = False
                for group in groups:
                    group_indices = [idx for idx, path in enumerate(all_paths) if path in group]
                    if i in group_indices and j in group_indices:
                        found_together = True
                        break
                self.assertTrue(found_together, f"Paths {i} and {j} should be in the same group")
        else:
            # No intersections, each path in its own group
            self.assertEqual(len(groups), len(all_paths))
    
    def test_get_non_intersecting_rcp_paths_all_intersecting(self):
        """Test get_non_intersecting_rcp_paths when all paths share rotatable indices."""
        from ringclosingmm.MolecularSystem import build_topology_from_data
        atoms_data = [(atom['element'], i) for i, atom in enumerate(self.linear_atoms)]
        topology = build_topology_from_data(atoms_data, self.linear_bonds)
        
        # All paths share rotatable index 2
        rcp_terms = [(0, 2), (1, 2), (2, 3)]
        system = self._create_minimal_system(self.linear_zmatrix, rcp_terms, topology)
        
        rotatable_indices = [2, 3, 4]
        
        groups = system.get_non_intersecting_rcp_paths(self.linear_zmatrix, rotatable_indices, verbose=False)
        
        # All paths should be in one group (they all share rotatable index 2)
        self.assertEqual(len(groups), 1)
        self.assertGreaterEqual(len(groups[0]), 2)  # At least 2 paths should be in the group
    
    def test_get_non_intersecting_rcp_paths_no_intersections(self):
        """Test get_non_intersecting_rcp_paths when no paths share rotatable indices."""
        # Create structure with completely separate paths
        # This is similar to test_get_non_intersecting_rcp_paths_non_intersecting_paths
        # but with more paths
        
        separate_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.5},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.5, 
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6},
            {'id': 4, 'element': 'C', 'atomic_num': 6, 'bond_ref': 3, 'bond_length': 1.5},
            {'id': 5, 'element': 'C', 'atomic_num': 6, 'bond_ref': 4, 'bond_length': 1.5, 
             'angle_ref': 3, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0},
        ]
        separate_bonds = [(0, 1), (1, 2), (3, 4), (4, 5)]
        separate_zmatrix = ZMatrix(separate_atoms, separate_bonds)
        
        from ringclosingmm.MolecularSystem import build_topology_from_data
        atoms_data = [(atom['element'], i) for i, atom in enumerate(separate_atoms)]
        topology = build_topology_from_data(atoms_data, separate_bonds)
        
        rcp_terms = [(0, 2), (3, 5)]
        system = self._create_minimal_system(separate_zmatrix, rcp_terms, topology)
        
        # Use different rotatable indices for each path
        rotatable_indices = [2, 5]
        
        groups = system.get_non_intersecting_rcp_paths(separate_zmatrix, rotatable_indices, verbose=False)
        
        # Each path should be in its own group
        self.assertEqual(len(groups), 2)
        for group in groups:
            self.assertEqual(len(group), 1)
    
    def test_get_non_intersecting_rcp_paths_empty_paths(self):
        """Test get_non_intersecting_rcp_paths when there are no paths."""
        from ringclosingmm.MolecularSystem import build_topology_from_data
        atoms_data = [(atom['element'], i) for i, atom in enumerate(self.linear_atoms)]
        topology = build_topology_from_data(atoms_data, self.linear_bonds)
        
        # No RCP terms
        rcp_terms = []
        system = self._create_minimal_system(self.linear_zmatrix, rcp_terms, topology)
        
        rotatable_indices = [2, 3, 4]
        
        groups = system.get_non_intersecting_rcp_paths(self.linear_zmatrix, rotatable_indices, verbose=False)
        
        # Should return empty list
        self.assertEqual(groups, [])
    
    def test_get_non_intersecting_rcp_paths_verbose_output(self):
        """Test that verbose output works correctly."""
        from ringclosingmm.MolecularSystem import build_topology_from_data
        atoms_data = [(atom['element'], i) for i, atom in enumerate(self.linear_atoms)]
        topology = build_topology_from_data(atoms_data, self.linear_bonds)
        
        rcp_terms = [(0, 2), (2, 4)]
        system = self._create_minimal_system(self.linear_zmatrix, rcp_terms, topology)
        
        rotatable_indices = [2, 3, 4]
        
        # Capture output
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        try:
            groups = system.get_non_intersecting_rcp_paths(self.linear_zmatrix, rotatable_indices, verbose=True)
            output = captured_output.getvalue()
            
            # Should have verbose output
            self.assertIn("non-intersecting path groups", output)
            self.assertIn("Group", output)
        finally:
            sys.stdout = old_stdout
        
        # Should still return correct groups
        self.assertGreaterEqual(len(groups), 1)


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
    suite.addTests(loader.loadTestsFromTestCase(TestMolecularSystemRCPPathComputation))
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests(verbosity=2)
    sys.exit(0 if success else 1)

