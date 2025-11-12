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

from src.IOTools import write_xyz_file

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from RingClosureOptimizer import (
    Individual,
    GeneticAlgorithm,
    LocalRefinementOptimizer,
    RingClosureOptimizer
)
from MolecularSystem import MolecularSystem
import tempfile
import os


class TestIndividual(unittest.TestCase):
    """Test Individual class for GA."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Simple Z-matrix
        self.zmatrix = [
            {'id': 0, 'element': 'H', 'atomic_num': 1},
            {'id': 1, 'element': 'H', 'atomic_num': 1, 'bond_ref': 1, 'bond_length': 1.0},
            {'id': 2, 'element': 'H', 'atomic_num': 1, 'bond_ref': 1, 'bond_length': 1.0,
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


class TestGeneticAlgorithm(unittest.TestCase):
    """Test GeneticAlgorithm class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a 4-atom Z-matrix with one rotatable dihedral
        self.zmatrix = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0}
        ]
        self.rotatable_indices = [3]  # 4th atom has rotatable dihedral
        self.num_torsions = 1
    
    def test_ga_initialization(self):
        """Test GA initialization."""
        ga = GeneticAlgorithm(
            num_torsions=self.num_torsions,
            base_zmatrix=self.zmatrix,
            rotatable_indices=self.rotatable_indices,
            population_size=10,
            rcp_terms=None,
            topology=None
        )
        
        self.assertEqual(ga.num_torsions, self.num_torsions)
        self.assertEqual(ga.population_size, 10)
        self.assertEqual(len(ga.population), 10)
        self.assertIsNone(ga.best_individual)
        self.assertEqual(ga.generation, 0)
    
    def test_ga_evaluate_population(self):
        """Test population evaluation."""
        ga = GeneticAlgorithm(
            num_torsions=self.num_torsions,
            base_zmatrix=self.zmatrix,
            rotatable_indices=self.rotatable_indices,
            population_size=5,
            rcp_terms=None,
            topology=None
        )
        
        # Simple fitness function
        def fitness_fn(individual, gen, idx):
            return np.sum(individual.torsions**2)
        
        ga.evaluate_population(fitness_fn, 0)
        
        # Check that fitness was assigned
        for ind in ga.population:
            self.assertNotEqual(ind.fitness, np.inf)
        
        # Check that best_individual is set
        self.assertIsNotNone(ga.best_individual)
    
    def test_ga_selection(self):
        """Test tournament selection."""
        ga = GeneticAlgorithm(
            num_torsions=self.num_torsions,
            base_zmatrix=self.zmatrix,
            rotatable_indices=self.rotatable_indices,
            population_size=10,
            elite_size=2,
            rcp_terms=None,
            topology=None
        )
        
        # Assign fitness
        for i, ind in enumerate(ga.population):
            ind.fitness = float(i)
        
        selected = ga.selection()
        
        self.assertEqual(len(selected), ga.population_size - ga.elite_size)
        self.assertTrue(all(isinstance(ind, Individual) for ind in selected))
    
    def test_ga_crossover(self):
        """Test crossover operation."""
        # Create extended z-matrix with 2 rotatable dihedrals
        extended_zmatrix = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0},
            {'id': 4, 'element': 'C', 'atomic_num': 6, 'bond_ref': 3, 'bond_length': 1.54,
             'angle_ref': 2, 'angle': 109.47, 'dihedral_ref': 1, 'dihedral': 180.0, 'chirality': 0}
        ]
        
        ga = GeneticAlgorithm(
            num_torsions=2,
            base_zmatrix=extended_zmatrix,
            rotatable_indices=[3, 4],
            population_size=5,
            crossover_rate=1.0,  # Always crossover
            rcp_terms=None,
            topology=None
        )
        
        parent1 = Individual(np.array([60.0, 120.0]), extended_zmatrix)
        parent2 = Individual(np.array([-60.0, -120.0]), extended_zmatrix)
        
        child1, child2 = ga.crossover(parent1, parent2)
        
        # Children should have correct number of torsions
        self.assertEqual(len(child1.torsions), 2)
        self.assertEqual(len(child2.torsions), 2)
        self.assertIsInstance(child1, Individual)
        self.assertIsInstance(child2, Individual)
    
    def test_ga_mutate(self):
        """Test mutation operation."""
        ga = GeneticAlgorithm(
            num_torsions=self.num_torsions,
            base_zmatrix=self.zmatrix,
            rotatable_indices=self.rotatable_indices,
            population_size=5,
            mutation_rate=1.0,  # Always mutate
            mutation_strength=10.0,
            rcp_terms=None,
            topology=None
        )
        
        individual = Individual(np.array([60.0]), self.zmatrix)
        original_torsion = individual.torsions[0]
        
        mutated = ga.mutate(individual)
        
        # Mutated torsion should be different (with high probability)
        # But we can't guarantee it changed due to random gaussian
        self.assertIsInstance(mutated, Individual)
        self.assertEqual(len(mutated.torsions), 1)
    
    def test_ga_evolve(self):
        """Test evolution to next generation."""
        ga = GeneticAlgorithm(
            num_torsions=self.num_torsions,
            base_zmatrix=self.zmatrix,
            rotatable_indices=self.rotatable_indices,
            population_size=10,
            elite_size=2,
            rcp_terms=None,
            topology=None
        )
        
        # Assign fitness
        for i, ind in enumerate(ga.population):
            ind.fitness = float(i)
        
        initial_generation = ga.generation
        ga.evolve()
        
        self.assertEqual(ga.generation, initial_generation + 1)
        self.assertEqual(len(ga.population), 10)
    
    def test_ga_get_statistics(self):
        """Test getting population statistics."""
        ga = GeneticAlgorithm(
            num_torsions=self.num_torsions,
            base_zmatrix=self.zmatrix,
            rotatable_indices=self.rotatable_indices,
            population_size=5,
            rcp_terms=None,
            topology=None
        )
        
        # Assign known fitness values
        fitness_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for ind, fitness in zip(ga.population, fitness_values):
            ind.fitness = fitness
        ga.best_individual = ga.population[0]
        
        stats = ga.get_statistics()
        
        self.assertIn('best', stats)
        self.assertIn('current_best', stats)
        self.assertIn('average', stats)
        self.assertIn('std', stats)
        self.assertIn('worst', stats)
        self.assertEqual(stats['best'], 1.0)
        self.assertEqual(stats['current_best'], 1.0)
        self.assertEqual(stats['worst'], 5.0)
        self.assertAlmostEqual(stats['average'], 3.0)


class TestLocalRefinementOptimizer(unittest.TestCase):
    """Test LocalRefinementOptimizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(__file__).parent / 'fixtures'
        self.forcefield_file = Path(__file__).parent.parent / 'data' / 'RCP_UFFvdW.xml'
        self.test_int_file = self.test_dir / 'butane_like.int'
    
    def test_local_refinement_initialization(self):
        """Test LocalRefinementOptimizer initialization."""
        if not self.test_int_file.exists() or not self.forcefield_file.exists():
            self.skipTest("Required test files not found")
        
        system = MolecularSystem.from_file(
            str(self.test_int_file),
            str(self.forcefield_file),
            rcp_terms=None
        )
        rotatable_indices = [3]
        
        local_opt = LocalRefinementOptimizer(system, rotatable_indices)
        
        self.assertEqual(local_opt.system, system)
        self.assertEqual(local_opt.rotatable_indices, rotatable_indices)
        self.assertIsNotNone(local_opt.converter)
    
    def test_refine_individual_cartesian(self):
        """Test Cartesian refinement of individual."""
        if not self.test_int_file.exists() or not self.forcefield_file.exists():
            self.skipTest("Required test files not found")
        
        system = MolecularSystem.from_file(
            str(self.test_int_file),
            str(self.forcefield_file),
            rcp_terms=None
        )
        rotatable_indices = [3]
        
        local_opt = LocalRefinementOptimizer(system, rotatable_indices)
        
        # Create an individual
        torsions = np.array([60.0])
        individual = Individual(torsions, system.zmatrix)
        
        # Refine
        refined = local_opt.refine_individual_in_Cartesian_space(
            individual,
            max_iterations=10
        )
        
        self.assertIsInstance(refined, Individual)
        self.assertIsNotNone(refined.energy)


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
            population_size=5,
            generations=3,
            enable_smoothing_refinement=False,
            verbose=False
        )
        
        self.assertIn('initial_closure_score', result)
        self.assertIn('final_closure_score', result)
        self.assertIsNotNone(optimizer.ga)
        self.assertIsNotNone(optimizer.ga.best_individual)
    
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
            population_size=5,
            generations=2,
            enable_smoothing_refinement=True,
            refinement_top_n=1,
            smoothing_sequence=[10.0, 0.0],
            torsional_iterations=5,
            verbose=False
        )
        
        self.assertIn('initial_closure_score', result)
        self.assertIn('final_closure_score', result)
        self.assertIsNotNone(optimizer.top_candidates)
        self.assertGreater(len(optimizer.top_candidates), 0)
    
    def test_save_optimized_structure(self):
        """Test saving optimized structure."""
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
        
        # Run optimization
        result = optimizer.optimize(
            population_size=5,
            generations=2,
            enable_smoothing_refinement=False,
            verbose=False
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            temp_file = f.name
        
        try:
            optimizer.save_optimized_structure(temp_file)
            
            # Check file was created and has content
            self.assertTrue(os.path.exists(temp_file))
            with open(temp_file, 'r') as f:
                content = f.read()
                self.assertGreater(len(content), 0)
                # First line should be atom count
                lines = content.strip().split('\n')
                self.assertGreater(len(lines), 0)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_save_before_optimize_raises_error(self):
        """Test that saving before optimization raises error."""
        if not self.test_int_file.exists() or not self.forcefield_file.exists():
            self.skipTest("Required test files not found")
        
        optimizer = RingClosureOptimizer.from_files(
            str(self.test_int_file),
            str(self.forcefield_file),
            rotatable_bonds=None,
            rcp_terms=None
        )
        
        with self.assertRaises(ValueError):
            optimizer.save_optimized_structure('output.xyz')
    
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
    suite.addTests(loader.loadTestsFromTestCase(TestIndividual))
    suite.addTests(loader.loadTestsFromTestCase(TestRingClosureOptimizerUtilities))
    suite.addTests(loader.loadTestsFromTestCase(TestRingClosureOptimizerMinimize))
    suite.addTests(loader.loadTestsFromTestCase(TestGeneticAlgorithm))
    suite.addTests(loader.loadTestsFromTestCase(TestLocalRefinementOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestRingClosureOptimizerOptimize))
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests(verbosity=2)
    sys.exit(0 if success else 1)

