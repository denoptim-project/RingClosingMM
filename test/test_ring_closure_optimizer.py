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
from CoordinateConverter import zmatrix_to_cartesian
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
            torsional_space=False,
            update_system=True,
            verbose=False
        )
        
        self.assertIn('initial_energy', result)
        self.assertIn('final_energy', result)
        self.assertIn('improvement', result)
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
        if len(optimizer.rotatable_indices) == 0:
            self.skipTest("No rotatable indices for torsional minimization")
        
        result = optimizer.minimize(
            max_iterations=10,
            smoothing=None,
            torsional_space=True,
            update_system=True,
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
        if len(optimizer.dof_indices) == 0:
            self.skipTest("No DOF indices for Z-matrix minimization")
        
        result = optimizer.minimize(
            max_iterations=10,
            smoothing=None,
            torsional_space=False,
            zmat_space=True,
            update_system=True,
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


class TestGeneticAlgorithmGraphMethods(unittest.TestCase):
    """Test graph methods in GeneticAlgorithm (static methods)."""
    
    def test_build_graph_from_simple_zmatrix(self):
        """Test building graph from simple linear Z-matrix."""
        # Simple 3-atom linear chain: C-C-C
        zmatrix = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47}
        ]
        
        graph = GeneticAlgorithm._build_bond_graph(zmatrix, topology=None)
        
        # Check graph structure
        self.assertEqual(len(graph), 3)
        self.assertEqual(set(graph[0]), {1})      # Atom 0 bonded to 1
        self.assertEqual(set(graph[1]), {0, 2})   # Atom 1 bonded to 0 and 2
        self.assertEqual(set(graph[2]), {1})      # Atom 2 bonded to 1
    
    def test_build_graph_butane(self):
        """Test building graph from butane-like Z-matrix."""
        # 4-atom chain: C-C-C-C
        zmatrix = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0}
        ]
        
        graph = GeneticAlgorithm._build_bond_graph(zmatrix, topology=None)
        
        # Check graph structure
        self.assertEqual(len(graph), 4)
        self.assertEqual(set(graph[0]), {1})      # Atom 0 bonded to 1
        self.assertEqual(set(graph[1]), {0, 2})   # Atom 1 bonded to 0 and 2
        self.assertEqual(set(graph[2]), {1, 3})   # Atom 2 bonded to 1 and 3
        self.assertEqual(set(graph[3]), {2})      # Atom 3 bonded to 2
    
    def test_graph_is_bidirectional(self):
        """Test that graph edges are bidirectional."""
        zmatrix = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47}
        ]
        
        graph = GeneticAlgorithm._build_bond_graph(zmatrix, topology=None)
        
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
        path = GeneticAlgorithm._find_path_bfs(graph, 0, 3)
        self.assertEqual(path, [0, 1, 2, 3])
        
        # Path from 3 to 0 (reverse)
        path_reverse = GeneticAlgorithm._find_path_bfs(graph, 3, 0)
        self.assertEqual(path_reverse, [3, 2, 1, 0])
    
    def test_path_to_self(self):
        """Test path from node to itself."""
        graph = {
            0: [1],
            1: [0, 2],
            2: [1]
        }
        
        path = GeneticAlgorithm._find_path_bfs(graph, 1, 1)
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
        path = GeneticAlgorithm._find_path_bfs(graph, 0, 4)
        self.assertEqual(path, [0, 1, 3, 4])
        
        # Path from 2 to 4
        path = GeneticAlgorithm._find_path_bfs(graph, 2, 4)
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
        path = GeneticAlgorithm._find_path_bfs(graph, 0, 2)
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
        path = GeneticAlgorithm._find_path_bfs(graph, 0, 3)
        self.assertIsNone(path)
        
        # Path within same component works
        path = GeneticAlgorithm._find_path_bfs(graph, 0, 1)
        self.assertEqual(path, [0, 1])
    
    def test_invalid_start_node(self):
        """Test with invalid start node."""
        graph = {
            0: [1],
            1: [0, 2],
            2: [1]
        }
        
        path = GeneticAlgorithm._find_path_bfs(graph, 5, 1)
        self.assertIsNone(path)
    
    def test_invalid_end_node(self):
        """Test with invalid end node."""
        graph = {
            0: [1],
            1: [0, 2],
            2: [1]
        }
        
        path = GeneticAlgorithm._find_path_bfs(graph, 0, 5)
        self.assertIsNone(path)
    
    def test_single_isolated_atom(self):
        """Test with single isolated atom."""
        graph = {0: []}
        
        path = GeneticAlgorithm._find_path_bfs(graph, 0, 0)
        self.assertEqual(path, [0])


class TestRingClosureOptimizerDOFMethods(unittest.TestCase):
    """Test DOF-related methods in RingClosureOptimizer."""
    
    def test_get_dofs_from_rotatable_indices(self):
        """Test getting DOF indices from rotatable indices with hardcoded Z-matrix."""
        # Create a larger branched Z-matrix with 12 atoms
        # Includes both dihedrals (chirality=0) and second angles (chirality=+/-1)
        zmatrix = [
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
        
        rotatable_indices = [9, 4]  # 0-based indices in zmatrix
        dof_indices = RingClosureOptimizer._get_dofs_from_rotatable_indeces(rotatable_indices, zmatrix)
        
        # Define expected DOF indices based on Z-matrix structure
        # Format: (atom_index, dof_type) where dof_type is 1=angle_ref, 2=dihedral_ref
        # For rotatable_indices [9, 4] (atoms with id 9 and 4):
        #   - Atom 9 (id 9): bond_ref=0, angle_ref=1
        #   - Atom 4 (id 4): bond_ref=3, angle_ref=2
        expected_dof_indices = [
            (2, 1),   # Atom 2 (id 2): bond_ref=1, angle_ref=0 matches atom 9's refs (reversed)
            (3, 1),   # Atom 3 (id 3): bond_ref=2, angle_ref=1 matches atom 9's angle_ref
            (4, 1),   # Atom 4 (id 4): bond_ref=3, angle_ref=2 - angle DOF
            (4, 2),   # Atom 4 (id 4): dihedral_ref=1 - rotatable bond DOF 
            (7, 1),   # Atom 7 (id 7): bond_ref=1, angle_ref=0 matches atom 9's refs (chirality=1)
            (9, 1),   # Atom 9 (id 9): bond_ref=0, angle_ref=1 - angle DOF
            (9, 2)    # Atom 9 (id 9): dihedral_ref=2 - rotatable bond DOF
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
        zmatrix = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54, 
             'angle_ref': 0, 'angle': 109.47, 'chirality': 0}
        ]
        
        rotatable_indices = []
        dof_indices = RingClosureOptimizer._get_dofs_from_rotatable_indeces(rotatable_indices, zmatrix)
        
        # Should return empty list
        self.assertIsInstance(dof_indices, list)
        self.assertEqual(len(dof_indices), 0)
    
    def test_get_dofs_single_rotatable_index(self):
        """Test with single rotatable index."""
        zmatrix = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54, 
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54, 
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0}
        ]
        
        rotatable_indices = [3]  # Only atom 3
        dof_indices = RingClosureOptimizer._get_dofs_from_rotatable_indeces(rotatable_indices, zmatrix)
        
        # Should find at least one DOF
        self.assertIsInstance(dof_indices, list)
        self.assertGreater(len(dof_indices), 0)
        # All DOFs should reference valid atom indices
        for atom_idx, dof_type in dof_indices:
            self.assertLess(atom_idx, len(zmatrix))
            self.assertIn(dof_type, [1, 2])
    
    def test_get_dofs_multiple_chirality_atoms(self):
        """Test with multiple atoms having second bond angle"""
        zmatrix = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'H', 'atomic_num': 1, 'bond_ref': 1, 'bond_length': 1.09, 
             'angle_ref': 0, 'angle': 109.47, 'dihedral_ref': 1, 'dihedral': 120.0, 'chirality': 1},
            {'id': 3, 'element': 'H', 'atomic_num': 1, 'bond_ref': 1, 'bond_length': 1.09, 
             'angle_ref': 0, 'angle': 109.47, 'dihedral_ref': 1, 'dihedral': -120.0, 'chirality': -1},
            {'id': 4, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54, 
             'angle_ref': 0, 'angle': 109.47, 'dihedral_ref': 1, 'dihedral': 0.0, 'chirality': 0}
        ]
        
        rotatable_indices = [4]  # Atom with chirality=0
        dof_indices = RingClosureOptimizer._get_dofs_from_rotatable_indeces(rotatable_indices, zmatrix)
        
        self.assertIsInstance(dof_indices, list)
        self.assertEqual(len(dof_indices), 4)
        self.assertEqual(sorted(dof_indices), sorted([(2, 1), (3, 1), (4, 1), (4, 2)]))


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
        if len(optimizer.rotatable_indices) == 0:
            self.skipTest("No rotatable indices")
        
        result = optimizer.optimize(
            population_size=5,
            generations=3,
            enable_smoothing_refinement=False,
            enable_cartesian_refinement=False,
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
        
        if len(optimizer.rotatable_indices) == 0:
            self.skipTest("No rotatable indices")
        
        result = optimizer.optimize(
            population_size=5,
            generations=2,
            enable_smoothing_refinement=True,
            enable_cartesian_refinement=True,
            refinement_top_n=1,
            smoothing_sequence=[10.0, 0.0],
            torsional_iterations=5,
            cartesian_iterations=5,
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
        
        if len(optimizer.rotatable_indices) == 0:
            self.skipTest("No rotatable indices")
        
        # Run optimization
        result = optimizer.optimize(
            population_size=5,
            generations=2,
            enable_smoothing_refinement=False,
            enable_cartesian_refinement=False,
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
        
        if len(optimizer.rotatable_indices) == 0:
            self.skipTest("No rotatable indices")
        
        result = optimizer.minimize(
            max_iterations=5,
            smoothing=[10.0, 5.0, 0.0],
            torsional_space=True,
            update_system=True,
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
            torsional_space=False,
            update_system=False,
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
    suite.addTests(loader.loadTestsFromTestCase(TestGeneticAlgorithmGraphMethods))
    suite.addTests(loader.loadTestsFromTestCase(TestRingClosureOptimizerDOFMethods))
    suite.addTests(loader.loadTestsFromTestCase(TestLocalRefinementOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestRingClosureOptimizerOptimize))
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests(verbosity=2)
    sys.exit(0 if success else 1)

