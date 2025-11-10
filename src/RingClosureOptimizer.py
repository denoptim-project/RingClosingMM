#!/usr/bin/env python3
"""
Hybrid Genetic Algorithm for Ring Closure Optimization

This module provides the main framework for optimizing molecular conformation  
to achieve ring closure. This is done by combining a genetic algorithm for 
exploration of the torsional space and post-processing local refinement to improve the best candidates.

Classes:
    CoordinateConverter: Handles Z-matrix/Cartesian coordinate transformations
    Individual: Represents a single candidate solution with torsions and Z-matrix
    GeneticAlgorithm: Implements genetic algorithm operations (global search of the torsional space)
    LocalRefinementOptimizer: Coordinates post-processing local refinement
    RingClosureOptimizer: Main optimizer coordinating all components
    
Example:
    optimizer = RingClosureOptimizer.from_files(
        structure_file='molecule.int',
        forcefield_file='force_field.xml',
        rotatable_bonds=[(1, 2), (5, 31)],
        rcp_terms=[(7, 39), (77, 35)]
    )
    
    result = optimizer.optimize(
        population_size=30,
        generations=50,
        enable_smoothing_refinement=True
    )
"""

import time
import numpy as np
from typing import List, Tuple, Dict, Optional, Any, Union
from pathlib import Path
import copy

from src import IOTools

try:
    # Relative imports for package use
    from .CoordinateConverter import (
        zmatrix_to_cartesian,
        CoordinateConverter
    )
    from .MolecularSystem import MolecularSystem
    from .IOTools import save_structure_to_file
except ImportError:
    # Absolute imports for direct script use
    from CoordinateConverter import (
        zmatrix_to_cartesian,
        CoordinateConverter
    )
    from MolecularSystem import MolecularSystem
    from IOTools import save_structure_to_file

# =============================================================================
# Individual Representation
# =============================================================================

class Individual:
    """
    Represents a single individual in the genetic algorithm population.
    
    Each individual contains both the torsional angles and the associated
    Z-matrix (which may have been refined through local minimization).
    
    Attributes
    ----------
    torsions : np.ndarray
        Array of torsional angles (degrees)
    zmatrix : List[Dict]
        Z-matrix representation with current internal coordinates
    fitness : float
        Fitness (energy) of this individual
    ring_closure_score : float
        Ring closure score of the individual
    cartesian_refined : bool
        Whether this individual has undergone Cartesian refinement
    torsional_refined : bool
        Whether this individual has undergone torsional refinement
    torsion_hash : int
        Hash of torsion angles to detect changes
    """
    
    def __init__(self, torsions: np.ndarray, zmatrix: List[Dict], fitness: float = np.inf, energy : float = None):
        """
        Initialize an individual.
        
        Parameters
        ----------
        torsions : np.ndarray
            Torsional angles
        zmatrix : List[Dict]
            Z-matrix representation
        fitness : float
            Fitness value (energy)
        energy : float
            Energy of the individual
        """
        self.torsions = torsions.copy()
        self.zmatrix = copy.deepcopy(zmatrix)
        self.fitness = fitness
        self.energy = energy
        self.ring_closure_score = 0.0  # Initialized to 0, set by fitness function
        self.cartesian_refined = False
        self.torsional_refined = False
        self.torsion_hash = self._compute_torsion_hash()
    
    def _compute_torsion_hash(self) -> int:
        """Compute hash of torsion angles to detect changes."""
        # Round to 2 decimal places to avoid floating point noise
        rounded_torsions = np.round(self.torsions, 2)
        return hash(rounded_torsions.tobytes())
    
    def update_torsion_hash(self):
        """Update the torsion hash after refinement (keeps refinement flags as-is)."""
        self.torsion_hash = self._compute_torsion_hash()
    
    def copy(self) -> 'Individual':
        """Create a deep copy of this individual."""
        new_ind = Individual(self.torsions, self.zmatrix, self.fitness, self.energy)
        new_ind.ring_closure_score = self.ring_closure_score
        new_ind.cartesian_refined = self.cartesian_refined
        new_ind.torsional_refined = self.torsional_refined
        new_ind.torsion_hash = self.torsion_hash
        return new_ind


# =============================================================================
# Genetic Algorithm
# =============================================================================

class GeneticAlgorithm:
    """
    Implements genetic algorithm operations for torsional optimization.
    
    This class manages the population of Individuals, performs selection,
    crossover, and mutation operations, and tracks the best solution found.
    
    Each individual contains both torsional angles and a Z-matrix, allowing
    independent zmatrix corrections (from refinement) to compete.
    
    Attributes
    ----------
    population : List[Individual]
        Current population of individuals
    best_individual : Individual
        Best solution found so far
    generation : int
        Current generation number
    """
    
    def __init__(self, num_torsions: int, base_zmatrix: List[Dict], 
                 rotatable_indices: List[int],
                 population_size: int = 30,
                 mutation_rate: float = 0.15, mutation_strength: float = 10.0,
                 crossover_rate: float = 0.7, elite_size: int = 3,
                 torsion_range: Tuple[float, float] = (-180.0, 180.0),
                 rcp_terms: Optional[List[Tuple[int, int]]] = None,
                 systematic_sampling_divisions: int = 6,
                 topology = None):
        """
        Initialize genetic algorithm.
        
        Parameters
        ----------
        num_torsions : int
            Number of torsional angles to optimize
        base_zmatrix : List[Dict]
            Base Z-matrix structure to use for initialization
        rotatable_indices : List[int]
            Indices of rotatable atoms in Z-matrix
        population_size : int
            Size of population
        mutation_rate : float
            Probability of mutation per gene
        mutation_strength : float
            Standard deviation of Gaussian mutation (degrees)
        crossover_rate : float
            Probability of crossover
        elite_size : int
            Number of elite individuals to preserve
        torsion_range : Tuple[float, float]
            Min and max torsion angles (degrees)
        rcp_terms : Optional[List[Tuple[int, int]]]
            Ring closure pair terms (0-based atom indices)
        systematic_sampling_divisions : int
            Number of divisions for systematic sampling of critical torsions
        topology : Topology, optional
            OpenMM topology with complete bond information
        """
        self.num_torsions = num_torsions
        self.base_zmatrix = base_zmatrix
        self.rotatable_indices = rotatable_indices
        self.rc_critical_rotatable_indeces = None
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.torsion_range = torsion_range
        self.rcp_terms = rcp_terms
        self.systematic_sampling_divisions = systematic_sampling_divisions
        self.topology = topology
        
        self.converter = CoordinateConverter()
        self.population = self._initialize_population()
        self.best_individual = None
        self.generation = 0
    
    @staticmethod
    def _build_bond_graph(base_zmatrix: List[Dict], topology=None) -> Dict[int, List[int]]:
        """
        Build bond connectivity graph from molecular topology or Z-matrix.
        
        Uses the complete molecular topology (including all bonds) if available,
        otherwise falls back to Z-matrix bond_ref (which only includes tree structure).
        
        Parameters
        ----------
        base_zmatrix : List[Dict]
            Z-matrix representation of the molecule
        topology : openmm.app.Topology, optional
            OpenMM topology object containing bond information
        
        Returns
        -------
        Dict[int, List[int]]
            Adjacency list representation (0-based indices)
        """
        num_atoms = len(base_zmatrix)
        graph = {i: [] for i in range(num_atoms)}
        
        if topology is not None:
            # Use complete topology (includes all bonds, including RCP bonds)
            for bond in topology.bonds():
                atom1_idx = bond.atom1.index
                atom2_idx = bond.atom2.index
                if atom1_idx < num_atoms and atom2_idx < num_atoms:
                    graph[atom1_idx].append(atom2_idx)
                    graph[atom2_idx].append(atom1_idx)
        else:
            # Fall back to Z-matrix (only tree structure bonds)
            for i, atom in enumerate(base_zmatrix):
                if i == 0:
                    continue
                bond_ref = atom['bond_ref']
                graph[i].append(bond_ref)
                graph[bond_ref].append(i)
        
        return graph
    
    @staticmethod
    def _find_path_bfs(graph: Dict[int, List[int]], start: int, end: int) -> Optional[List[int]]:
        """
        Find shortest path between two atoms using BFS.
        
        Parameters
        ----------
        graph : Dict[int, List[int]]
            Bond connectivity graph (adjacency list)
        start : int
            Start atom index (0-based)
        end : int
            End atom index (0-based)
        
        Returns
        -------
        Optional[List[int]]
            Path as list of atom indices, or None if no path exists
        """
        from collections import deque
        
        # Validate indices
        if start not in graph:
            return None
        if end not in graph:
            return None
        
        if start == end:
            return [start]
        
        # BFS using deque for O(1) popleft (more efficient than list.pop(0))
        visited = {start}
        queue = deque([(start, [start])])
        
        while queue:
            node, path = queue.popleft()
            
            # Safety check (should not happen if graph is well-formed)
            if node not in graph:
                continue
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    
                    if neighbor == end:
                        return new_path
                    
                    queue.append((neighbor, new_path))
        
        return None
    
    def _identify_rc_critical_rotatable_indeces(self) -> List[int]:
        """
        Identify rotatable torsions on paths between RCP atoms.
        
        Returns
        -------
        List[int]
            Indices (into rotatable_indices) of critical torsions
        """
        if not self.rcp_terms:
            return []
        
        graph = self._build_bond_graph(self.base_zmatrix, self.topology)
        num_atoms = len(self.base_zmatrix)
        critical_atoms = set()
        
        # Find all atoms on paths between RCP pairs
        # RCP terms are already 0-based
        for atom1, atom2 in self.rcp_terms:
            # Validate RCP atom indices (0-based, so range is [0, num_atoms-1])
            if atom1 < 0 or atom1 >= num_atoms:
                print(f"Warning: RCP atom1 index {atom1} is out of range [0, {num_atoms-1}], skipping")
                continue
            if atom2 < 0 or atom2 >= num_atoms:
                print(f"Warning: RCP atom2 index {atom2} is out of range [0, {num_atoms-1}], skipping")
                continue
            
            path = self._find_path_bfs(graph, atom1, atom2)
            if path:
                critical_atoms.update(path)
            else:
                print(f"Warning: No path found between RCP atoms {atom1} and {atom2} (0-based)")
        
        # Identify which rotatable torsions involve critical atoms
        # A torsion is critical if ANY of its 4 defining atoms are on a critical path
        critical_torsion_indices = []
        for i in range(len(self.rotatable_indices)):
            rot_idx = self.rotatable_indices[i]
            atom = self.base_zmatrix[rot_idx]
            atoms_in_rotatable_bond = [] 
            if atom.get('bond_ref'):
                atoms_in_rotatable_bond.append(atom['bond_ref'])
            if atom.get('angle_ref'):
                atoms_in_rotatable_bond.append(atom['angle_ref'])
            
            # Check if both atoms are on a critical path
            if all(atom_idx in critical_atoms for atom_idx in atoms_in_rotatable_bond):
                critical_torsion_indices.append(self.rotatable_indices[i])
        
        return critical_torsion_indices
    
    def _generate_systematic_samples(self) -> List[np.ndarray]:
        """
        Generate systematic samples for critical torsions.
        
        Returns
        -------
        List[np.ndarray]
            List of torsion arrays
        """

        critical_indices = [ i for i,idx in enumerate(self.rotatable_indices) if idx in self.rc_critical_rotatable_indeces]

        if not self.rc_critical_rotatable_indeces:
            return []
        
        # Generate discrete values for systematic sampling
        divisions = self.systematic_sampling_divisions
        discrete_values = np.linspace(
            self.torsion_range[0],
            self.torsion_range[1],
            divisions,
            endpoint=False
        )
        
        # Calculate total combinations
        num_critical = len(self.rc_critical_rotatable_indeces)
        total_combinations = divisions ** num_critical
        
        # Limit combinations if too many
        if total_combinations > self.population_size:
            # Use random sampling of combinations instead
            samples = []
            for _ in range(self.population_size):
                # Start with random torsions
                torsions = np.random.uniform(
                    self.torsion_range[0],
                    self.torsion_range[1],
                    self.num_torsions
                )
                # Set critical torsions to discrete values
                for idx in critical_indices:
                    torsions[idx] = np.random.choice(discrete_values)
                samples.append(torsions)
            return samples
        
        # Generate all combinations
        samples = []
        from itertools import product
        
        for combo in product(discrete_values, repeat=num_critical):
            if len(samples) >= self.population_size:
                break
            
            # Start with random torsions
            torsions = np.random.uniform(
                self.torsion_range[0],
                self.torsion_range[1],
                self.num_torsions
            )
            
            # Set critical torsions to systematic values
            for i, idx in enumerate(critical_indices):
                torsions[idx] = combo[i]
            
            samples.append(torsions)
        
        return samples
    
    def _initialize_population(self) -> List[Individual]:
        """
        Initialize population with systematic sampling for critical torsions.
        
        If RCP terms are provided, identifies torsions on paths between RCP atoms
        and systematically samples these critical torsions. Remaining population
        slots are filled with random individuals.
        """
        population = []
        
        # Identify critical torsions and generate systematic samples
        self.rc_critical_rotatable_indeces = self._identify_rc_critical_rotatable_indeces()
        
        if self.rc_critical_rotatable_indeces:
            # Generate systematic samples (up to population_size)
            systematic_samples = self._generate_systematic_samples()
            
            print(f"  Systematic initialization: {len(self.rc_critical_rotatable_indeces)} critical torsions "
                  f"(on RCP paths), generating {len(systematic_samples)} systematic samples")
            
            # Create individuals from systematic samples
            for i, torsions in enumerate(systematic_samples):
                zmatrix = self.converter.apply_torsions(
                    self.base_zmatrix,
                    self.rotatable_indices,
                    torsions
                )
                #elements = [atom['element'] for atom in zmatrix]
                #write_xyz_file(zmatrix_to_cartesian(zmatrix), elements, f"systematic_sample_{i}.xyz")
                individual = Individual(torsions, zmatrix)
                population.append(individual)
        
        # Fill remaining slots with random individuals
        while len(population) < self.population_size:
            torsions = np.random.uniform(
                self.torsion_range[0],
                self.torsion_range[1],
                self.num_torsions
            )
            zmatrix = self.converter.apply_torsions(
                self.base_zmatrix,
                self.rotatable_indices,
                torsions
            )
            individual = Individual(torsions, zmatrix)
            population.append(individual)
        
        return population
    
    def evaluate_population(self, fitness_function, gen: int) -> None:
        """
        Evaluate fitness for entire population.
        
        Parameters
        ----------
        fitness_function : callable
            Function that takes Individual and returns fitness (energy)
        gen : int
            Generation number
        """
        for i, individual in enumerate(self.population):
            individual.fitness = fitness_function(individual, gen, i)
        
        # Update best solution
        best_in_generation = min(self.population, key=lambda ind: ind.fitness)
        if self.best_individual is None or best_in_generation.fitness < self.best_individual.fitness:
            self.best_individual = best_in_generation.copy()
    
    def selection(self) -> List[Individual]:
        """
        Tournament selection.
        
        Returns
        -------
        List[Individual]
            Selected individuals
        """
        selected = []
        tournament_size = 3
        num_to_select = self.population_size - self.elite_size
        
        for _ in range(num_to_select):
            contestants_idx = np.random.choice(
                len(self.population),
                min(tournament_size, len(self.population)),
                replace=False
            )
            contestants = [self.population[i] for i in contestants_idx]
            winner = min(contestants, key=lambda ind: ind.fitness)
            selected.append(winner.copy())
        
        return selected
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Single-point crossover on torsions only.
        
        Operates on torsional angles, then creates new individuals with updated zmatrix.
        
        Parameters
        ----------
        parent1, parent2 : Individual
            Parent individuals
            
        Returns
        -------
        Tuple[Individual, Individual]
            Two offspring individuals
        """
        if np.random.random() > self.crossover_rate or self.num_torsions == 1:
            return parent1.copy(), parent2.copy()
        
        point = np.random.randint(1, self.num_torsions)
        child1_torsions = np.concatenate([parent1.torsions[:point], parent2.torsions[point:]])
        child2_torsions = np.concatenate([parent2.torsions[:point], parent1.torsions[point:]])
        
        # Apply torsions to create zmatrix for offspring
        # Use the parent's zmatrix as base (inheriting any refinements)
        child1_zmatrix = self.converter.apply_torsions(
            parent1.zmatrix, self.rotatable_indices, child1_torsions
        )
        child2_zmatrix = self.converter.apply_torsions(
            parent2.zmatrix, self.rotatable_indices, child2_torsions
        )
        
        return Individual(child1_torsions, child1_zmatrix), Individual(child2_torsions, child2_zmatrix)
    
    def mutate(self, individual: Individual) -> Individual:
        """
        Gaussian mutation on torsions only.
        
        Mutates multiple torsion angles: each torsion has a probability 
        `mutation_rate` of being mutated. If no torsions are selected,
        at least one random torsion is mutated to ensure variation.
        
        Parameters
        ----------
        individual : Individual
            Individual to mutate
            
        Returns
        -------
        Individual
            Mutated individual with updated zmatrix
        """
        new_torsions = individual.torsions.copy()
        
        # Probabilistic selection: each torsion has mutation_rate chance
        mask = np.random.random(self.num_torsions) < self.mutation_rate
        
        # Ensure at least one torsion is mutated
        if not np.any(mask):
            random_idx = np.random.randint(0, self.num_torsions)
            mask[random_idx] = True
        
        # Apply Gaussian mutations to selected torsions
        mutations = np.random.normal(0, self.mutation_strength, self.num_torsions)
        new_torsions[mask] += mutations[mask]
        new_torsions = np.clip(new_torsions, self.torsion_range[0], self.torsion_range[1])
        
        # Apply mutated torsions to individual's zmatrix
        new_zmatrix = self.converter.apply_torsions(
            individual.zmatrix, self.rotatable_indices, new_torsions
        )
        
        return Individual(new_torsions, new_zmatrix)
    
    def evolve(self) -> None:
        """Create next generation through selection, crossover, and mutation."""
        # Preserve elite individuals (best performers)
        sorted_pop = sorted(self.population, key=lambda ind: ind.fitness)
        new_population = [ind.copy() for ind in sorted_pop[:self.elite_size]]
        
        # Generate offspring through selection, crossover, and mutation
        selected = self.selection()
        for i in range(0, len(selected) - 1, 2):
            child1, child2 = self.crossover(selected[i], selected[i + 1])
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        # Fill any remaining slots with mutated elite individuals
        while len(new_population) < self.population_size:
            idx = np.random.randint(0, self.elite_size)
            new_individual = self.mutate(new_population[idx].copy())
            new_population.append(new_individual)
        
        self.population = new_population[:self.population_size]
        self.generation += 1
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get current population statistics.
        
        Returns
        -------
        Dict[str, float]
            Statistics including best, average, std, etc.
        """
        fitness_values = [ind.fitness for ind in self.population]
        return {
            'best': self.best_individual.fitness if self.best_individual else np.inf,
            'current_best': min(fitness_values) if fitness_values else np.inf,
            'average': np.mean(fitness_values) if fitness_values else np.inf,
            'std': np.std(fitness_values) if fitness_values else 0.0,
            'worst': max(fitness_values) if fitness_values else np.inf
        }


# =============================================================================
# Local Refinement Optimizer (Post-Processing)
# =============================================================================

class LocalRefinementOptimizer:
    """
    Handles post-processing local refinement.
    """
    
    def __init__(self, molecular_system: MolecularSystem, rotatable_indices: List[int]):
        """
        Initialize local refinement optimizer.
        
        Parameters
        ----------
        molecular_system : MolecularSystem
            Molecular system for energy evaluation
        rotatable_indices : List[int]
            Indices of rotatable atoms
        """
        self.system = molecular_system
        self.rotatable_indices = rotatable_indices
        self.converter = CoordinateConverter()
    
    def refine_individual_in_torsional_space_with_smoothing(self, individual: Individual,
                                         smoothing_sequence: List[float] = [50.0, 25.0, 10.0, 5.0, 2.5, 1.0, 0.0],
                                         torsional_iterations: int = 50,
                                         rotatable_indices: List[int] = None) -> Individual:
        """
        Refine an individual using smoothing algorithm with torsional optimization only.
        
        Performs a sequence of torsional optimizations with decreasing smoothing values.
        
        Parameters
        ----------
        individual : Individual
            Individual to refine
        smoothing_sequence : List[float]
            Sequence of smoothing values (default: [50.0, 25.0, 10.0, 5.0, 2.5, 1.0, 0.0])
        torsional_iterations : int
            Maximum iterations for each torsional optimization step
        rotatable_indices : List[int]
            Indices of rotatable atoms
        Returns
        -------
        Individual
            Refined individual with updated zmatrix and fitness
        """
        try:
            current_zmatrix = individual.zmatrix

            # Torsional optimization with decreasing smoothing
            for smoothing in smoothing_sequence:
                self.system.setSmoothingParameter(smoothing)

                refined_zmatrix, refined_energy, info = self.system.minimize_energy_in_torsional_space(
                    current_zmatrix,
                    rotatable_indices,
                    max_iterations=torsional_iterations,
                    method='Powell',
                    verbose=False
                )
                
                if info['success']:
                    current_zmatrix = refined_zmatrix
            
            # Extract refined torsions
            refined_torsions = np.array([current_zmatrix[idx]['dihedral']
                                        for idx in self.rotatable_indices])
            
            return Individual(refined_torsions, current_zmatrix, energy=refined_energy)
        
        except Exception as e:
            # Return original individual on failure
            print(f"Warning: Smoothing refinement failed!")
            return individual.copy()

    def refine_individual_in_zmatrix_space_with_smoothing(self, individual: Individual,
                                          smoothing_sequence: List[float] = [50.0, 25.0, 10.0, 5.0, 2.5, 1.0, 0.0],
                                          max_iterations: int = 500,
                                          dof_indices: List[int] = None,
                                          dof_bounds: Optional[List[Tuple[float, float]]] = [[0.02, 15.0, 10.0], [0.02, 5.0, 5.0]]) -> Individual:
        """
        Refine an individual using Z-matrix space minimization.
        
        Parameters
        ----------
        individual : Individual
            Individual to refine    
        smoothing_sequence : List[float]
            Sequence of smoothing values (default: [50.0, 25.0, 10.0, 5.0, 2.5, 1.0, 0.0])
        max_iterations : int
            Maximum minimization iterations
        dof_bounds : Optional[List[Tuple[float, float]]]
            Bounds for the degrees of freedom. Example: [(0.0, 1.0), (0.0, 1.0)] means 
            the first atom's first degree of freedom (distance) is between 0.0 and 1.0, 
            and the second atom's third degree of freedom (torsion) is between 0.0 and 1.0.
            None means no bounds.
        Returns
        -------
        Individual
            Refined individual
        """
        try:
            current_zmatrix = individual.zmatrix

            for smoothing in smoothing_sequence:
                self.system.setSmoothingParameter(smoothing)

                minimized_zmatrix, minimized_energy, info = self.system.minimize_energy_in_zmatrix_space(
                    current_zmatrix,
                    dof_indices = dof_indices,
                    dof_bounds = dof_bounds,
                    max_iterations=max_iterations
                )

                if info['success']:
                    current_zmatrix = minimized_zmatrix

            # Extract refined torsions
            refined_torsions = np.array([current_zmatrix[idx]['dihedral']
                                        for idx in self.rotatable_indices])

            return Individual(refined_torsions, current_zmatrix, energy=minimized_energy)
        
        except Exception as e:
            # Return original individual on failure
            print(e)
            print(f"Warning: Z-matrix refinement failed!")
            return individual.copy()
            

    def refine_individual_in_Cartesian_space(self, individual: Individual,
                                              max_iterations: int = 500) -> Individual:
        """
        Refine an individual using Cartesian space minimization.
        
        Parameters
        ----------
        individual : Individual
            Individual to refine
        max_iterations : int
            Maximum minimization iterations
        
        Returns
        -------
        Individual
            Refined individual
        """
        try:
            current_zmatrix = individual.zmatrix
            
            minimized_coords, minimized_energy = self.system.minimize_energy(
                current_zmatrix,
                max_iterations=max_iterations
            )
            
            # Extract refined zmatrix and torsions from minimized coordinates
            refined_zmatrix = self.converter.extract_zmatrix(
                minimized_coords,
                current_zmatrix
            )
            refined_torsions = self.converter.extract_torsions(
                minimized_coords,
                current_zmatrix,
                self.rotatable_indices
            )
            
            # Create refined individual
            return Individual(refined_torsions, refined_zmatrix, energy=minimized_energy)
        
        except Exception as e:
            # Return original individual on failure
            print(e)
            print(f"Warning: Cartesian refinement failed!")
            return individual.copy()
   

# =============================================================================
# Main Ring Closure Optimizer
# =============================================================================

class RingClosureOptimizer:
    """
    Main coordinator for ring closure optimization using hybrid genetic algorithm.
    
    This class brings together all components: molecular system, genetic algorithm,
    and post-processing local refinement to perform comprehensive ring closure optimization.
    
    The algorithm operates in sequential phases:
    1. Genetic Algorithm (GA) for global exploration until convergence
    2. Optional: Smoothing-based torsional refinement on top candidates
    3. Optional: Cartesian space minimization for final polish
    
    Example
    -------
    >>> optimizer = RingClosureOptimizer.from_files(
    ...     structure_file='molecule.int',
    ...     forcefield_file='ff.xml',
    ...     rotatable_bonds=[(1, 2), (5, 31)]
    ... )
    >>> result = optimizer.optimize(generations=50, enable_smoothing_refinement=True)
    >>> print(f"Final ring closure score: {result['final_closure_score']:.4f}")
    """
    
    def __init__(self, molecular_system: MolecularSystem,
                 rotatable_indices: List[int],
                 write_candidate_files: bool = False):
        """
        Initialize ring closure optimizer.
        
        Parameters
        ----------
        molecular_system : MolecularSystem
            Molecular system to optimize
        rotatable_indices : List[int]
            Indices of rotatable atoms in Z-matrix
        write_candidate_files : bool
            Write candidate files (int and xyz)
        """
        self.system = molecular_system
        self.rotatable_indices = rotatable_indices
        self.rc_critical_rotatable_indeces = self._identify_rc_critical_rotatable_indeces()
        self.dof_indices = self._get_dofs_from_rotatable_indeces(self.rotatable_indices, self.rc_critical_rotatable_indeces, self.system.zmatrix)
        self.converter = CoordinateConverter()
        self.ga = None
        self.local_refinement = None
        self.write_candidate_files = write_candidate_files
        self.top_candidates = None
    
    
    @classmethod
    def from_files(cls, structure_file: str, forcefield_file: str,
                   rotatable_bonds: Optional[List[Tuple[int, int]]] = None,
                   rcp_terms: Optional[List[Tuple[int, int]]] = None,
                   write_candidate_files: bool = False,
                   ring_closure_threshold: float = 1.5) -> 'RingClosureOptimizer':
        """
        Create optimizer from input files.
        
        Parameters
        ----------
        structure_file : str
            Path to structure file (.int)
        forcefield_file : str
            Path to force field XML
        rotatable_bonds : Optional[List[Tuple[int, int]]]
            List of rotatable bond pairs as (atom1, atom2) in 0-based indices.
            Each pair identifies atoms that define a rotatable bond.
            If None, all bonds will be considered rotatable.
        rcp_terms : Optional[List[Tuple[int, int]]]
            List of RCP term pairs as (atom1, atom2) in 0-based indices.
            Pairs of atoms that should form ring-closing bonds.
        write_candidate_files : bool
            Write candidate files (int and xyz)
        ring_closure_threshold : float
            Distance threshold (Angstroms) for considering a ring nearly closed
        
        Returns
        -------
        RingClosureOptimizer
            Initialized optimizer
        """
        # Create molecular system
        system = MolecularSystem.from_file(
            structure_file,
            forcefield_file,
            rcp_terms=rcp_terms,
            write_candidate_files=write_candidate_files,
            ring_closure_threshold=ring_closure_threshold
        )
        
        # Convert rotatable bonds to rotatable indices
        if rotatable_bonds is None:
            # All bonds are rotatable: get all dihedrals with chirality == 0
            rotatable_indices = cls._get_all_rotatable_indices(system.zmatrix)
        else:
            rotatable_indices = cls._convert_bonds_to_indices(rotatable_bonds, system.zmatrix)
        
        return cls(system, rotatable_indices, write_candidate_files)
    

    def _identify_rc_critical_rotatable_indeces(self) -> List[int]:
        """
        Identify rotatable torsions on paths between RCP atoms.
        
        Returns
        -------
        List[int]
            Indices (into rotatable_indices) of critical torsions
        """
        rcp_terms = self.system.rcpterms
        topology = self.system.topology
        if not rcp_terms:
            return []
        
        graph = GeneticAlgorithm._build_bond_graph(self.system.zmatrix, topology)
        num_atoms = len(self.system.zmatrix)
        critical_atoms = set()
        
        # Find all atoms on paths between RCP pairs
        # RCP terms are already 0-based
        for atom1, atom2 in rcp_terms:
            # Validate RCP atom indices (0-based, so range is [0, num_atoms-1])
            if atom1 < 0 or atom1 >= num_atoms:
                print(f"Warning: RCP atom1 index {atom1} is out of range [0, {num_atoms-1}], skipping")
                continue
            if atom2 < 0 or atom2 >= num_atoms:
                print(f"Warning: RCP atom2 index {atom2} is out of range [0, {num_atoms-1}], skipping")
                continue
            
            path = GeneticAlgorithm._find_path_bfs(graph, atom1, atom2)
            if path:
                critical_atoms.update(path)
            else:
                print(f"Warning: No path found between RCP atoms {atom1} and {atom2} (0-based)")
        
        # Identify which rotatable torsions involve critical atoms
        # A torsion is critical if ANY of its 4 defining atoms are on a critical path
        critical_torsion_indices = []
        for i in range(len(self.rotatable_indices)):
            rot_idx = self.rotatable_indices[i]
            atom = self.system.zmatrix[rot_idx]
            atoms_in_rotatable_bond = [] 
            if atom.get('bond_ref'):
                atoms_in_rotatable_bond.append(atom['bond_ref'])
            if atom.get('angle_ref'):
                atoms_in_rotatable_bond.append(atom['angle_ref'])
            
            # Check if both atoms are on a critical path
            if all(atom_idx in critical_atoms for atom_idx in atoms_in_rotatable_bond):
                critical_torsion_indices.append(self.rotatable_indices[i])
        
        return critical_torsion_indices


    @staticmethod
    def _get_dofs_from_rotatable_indeces(rotatable_indices: List[int], rc_critical_rotatable_indeces: List[int], zmatrix: List[Dict]) -> List[int]:
        """
        Get indexes of degrees of freedon on Z-matrix from rotatable indices.
        
        Parameters
        ----------
        rotatable_indices : List[int]
            Indices of rotatable atoms in Z-matrix
        zmatrix : List[Dict]
            Z-matrix representation
        
        Returns
        -------
        List[Tuple[int, int]]
            Indices of DOFs in Z-matrix. The first index is the atom index, 
            the second index is the degree of freedom index. Example: [(0, 0), (1, 2)] means 
            the first atom's first degree of freedom (distance) and the second atom's third 
            degree of freedom (torsion).
        """
        dof_indices = []
        dof_names = ['id', 'bond_ref', 'angle_ref', 'dihedral_ref']

        all_atoms_in_rot_bonds = []
        for idx in rc_critical_rotatable_indeces:
            zatom = zmatrix[idx]
            rb_bond_ref = zatom.get(dof_names[1])
            rb_angle_ref = zatom.get(dof_names[2])
            if not rb_bond_ref in all_atoms_in_rot_bonds:
                all_atoms_in_rot_bonds.append(rb_bond_ref)
            if not rb_angle_ref in all_atoms_in_rot_bonds:
                all_atoms_in_rot_bonds.append(rb_angle_ref)

        for idx in rotatable_indices:
            zatom = zmatrix[idx]
            # these two indexes identify the roatable bond
            rb_bond_ref = zatom.get(dof_names[1])
            rb_angle_ref = zatom.get(dof_names[2])
            if rb_bond_ref is None or rb_angle_ref is None:
                continue  # Skip if references don't exist

            if True:
                # Identify bond angles that act on the rotatable bond
                for idx2, zatom2 in enumerate(zmatrix):
                    zatom2_id = zatom2.get(dof_names[0])
                    zatom2_bond_ref = zatom2.get(dof_names[1])
                    zatom2_angle_ref = zatom2.get(dof_names[2])
                    zatom2_dihedral_ref = zatom2.get(dof_names[3])

                    if zatom2_bond_ref is not None and zatom2_angle_ref is not None:
                        if zatom2_id in all_atoms_in_rot_bonds and zatom2_bond_ref in all_atoms_in_rot_bonds and zatom2_angle_ref in all_atoms_in_rot_bonds:  
                            if (idx2, 1) not in dof_indices:
                                dof_indices.append((idx2, 1))
                        if zatom2_id in all_atoms_in_rot_bonds and zatom2_bond_ref in all_atoms_in_rot_bonds and zatom2_dihedral_ref in all_atoms_in_rot_bonds and zatom2.get('chirality', 0) != 0:  
                            if (idx2, 2) not in dof_indices:
                                dof_indices.append((idx2, 2))
            else:
                # Identify bond angles that act on the rotatable bond
                for idx2, zatom2 in enumerate(zmatrix):
                    zatom2_id = zatom2.get(dof_names[0])
                    zatom2_bond_ref = zatom2.get(dof_names[1])
                    zatom2_angle_ref = zatom2.get(dof_names[2])
                    zatom2_dihedral_ref = zatom2.get(dof_names[3])

                    # if either of the atoms defining the rotatable bond is the first atom and the other is the center of the angle
                    if zatom2_bond_ref is not None and zatom2_angle_ref is not None:
                        if (zatom2_id == rb_bond_ref and zatom2_bond_ref == rb_angle_ref) or \
                        (zatom2_id == rb_angle_ref and zatom2_bond_ref == rb_bond_ref):
                            if (idx2, 1) not in dof_indices:
                                dof_indices.append((idx2, 1))
                            if zatom2.get('chirality', 0) != 0:
                                if (idx2, 2) not in dof_indices:
                                    dof_indices.append((idx2, 2))
                    
                    # if the center of the first angle is either of the atoms defining the rotatable bond
                    if zatom2_bond_ref is not None and zatom2_angle_ref is not None:
                        if (zatom2_bond_ref == rb_bond_ref and zatom2_angle_ref == rb_angle_ref) or \
                        (zatom2_angle_ref == rb_bond_ref and zatom2_bond_ref == rb_angle_ref):
                            if (idx2, 1) not in dof_indices:
                                dof_indices.append((idx2, 1))
                    
                    # if the center of the second angle is either of the atoms defining the rotatable bond
                    if zatom2.get('chirality', 0) != 0 and zatom2_bond_ref is not None and zatom2_dihedral_ref is not None:
                        if (zatom2_bond_ref == rb_bond_ref and zatom2_dihedral_ref == rb_angle_ref) or \
                        (zatom2_dihedral_ref == rb_bond_ref and zatom2_bond_ref == rb_angle_ref):
                            if (idx2, 2) not in dof_indices:
                                dof_indices.append((idx2, 2))

        # Add the torsions
        for idx in rotatable_indices:
            if (idx, 2) not in dof_indices:
                dof_indices.append((idx, 2))

        return dof_indices  

    @staticmethod
    def _get_all_rotatable_indices(zmatrix: List[Dict]) -> List[int]:
        """
        Get all rotatable Z-matrix indices (all dihedrals with chirality == 0).
        
        When rotatable_bonds is not specified, all dihedrals that are not
        chirality-constrained are considered rotatable.
        
        Parameters
        ----------
        zmatrix : List[Dict]
            Z-matrix representation
            
        Returns
        -------
        List[int]
            0-based indices of all rotatable atoms in Z-matrix
        """
        rotatable_indices = []
        for i in range(3, len(zmatrix)):  # Only atoms 4+ have dihedrals
            atom = zmatrix[i]
            if atom.get('chirality', 0) == 0:  # Only true dihedrals (not chirality-constrained)
                rotatable_indices.append(i)
        
        return rotatable_indices
    
    @staticmethod
    def _convert_bonds_to_indices(rotatable_bonds: List[Tuple[int, int]], 
                                   zmatrix: List[Dict]) -> List[int]:
        """
        Convert rotatable bond pairs to rotatable Z-matrix indices.
        
        Parameters
        ----------
        rotatable_bonds : List[Tuple[int, int]]
            List of (atom1, atom2) pairs
        zmatrix : List[Dict]
            Z-matrix representation
            
        Returns
        -------
        List[int]
            Indices of rotatable atoms in Z-matrix
        """
        bonded_pairs = set(rotatable_bonds)
        
        rotatable_indices = []
        for i in range(3, len(zmatrix)):  # Only atoms 4+ have dihedrals
            atom = zmatrix[i]
            if atom.get('chirality', 0) == 0:  # Only true dihedrals
                bond_ref = atom['bond_ref']
                angle_ref = atom['angle_ref']
                # Check both orderings (bond_ref, angle_ref) and (angle_ref, bond_ref)
                if ((bond_ref, angle_ref) in bonded_pairs or 
                    (angle_ref, bond_ref) in bonded_pairs):
                    rotatable_indices.append(i)
        
        return rotatable_indices
    

    def _create_fitness_function(self, ring_closure_tolerance: float = 0.1,
                                   ring_closure_decay_rate: float = 0.5):
        """
        Create fitness function for GA evaluation.
        
        The fitness function uses exponential ring closure score (range [0,1], higher is better).
        To maintain compatibility with GA (which minimizes fitness), we return the negative score.
        
        Parameters
        ----------
        ring_closure_tolerance : float
            Distance threshold for perfect ring closure score
        ring_closure_decay_rate : float
            Exponential decay rate for ring closure score
        """
        def fitness_fn(individual: Individual, gen: int, i: int) -> float:
            coords = zmatrix_to_cartesian(individual.zmatrix)
            
            # Use exponential ring closure score (range [0, 1], higher is better)
            closure_score = self.system.ring_closure_score_exponential(
                coords,
                tolerance=ring_closure_tolerance,
                decay_rate=ring_closure_decay_rate,
                verbose=False
            )
            
            # Store for monitoring (fraction format for compatibility)
            individual.ring_closure_score = closure_score
            
            # Return negative score so GA minimizes (lower fitness = better closure)
            # This way, perfect closure (score=1.0) gives fitness=-1.0 (best)
            # Poor closure (score→0.0) gives fitness→0.0 (worst)
            individual.fitness = -closure_score
            
            return individual.fitness
        
        return fitness_fn
    

    def optimize(self, population_size: int = 30, 
                generations: int = 50,
                mutation_rate: float = 0.15, 
                mutation_strength: float = 10.0,
                crossover_rate: float = 0.7, 
                elite_size: int = 5,
                torsion_range: Tuple[float, float] = (-180.0, 180.0),
                ring_closure_tolerance: float = 0.1,
                ring_closure_decay_rate: float = 0.5, 
                convergence_interval: int = 5,
                protect_rc_critical_rotatable_bonds: bool = True,
                enable_smoothing_refinement: bool = True,
                enable_zmatrix_refinement: bool = True,
                enable_cartesian_refinement: bool = True,
                refinement_top_n: int = 1,
                smoothing_sequence: List[float] = None,
                torsional_iterations: int = 50,
                zmatrix_iterations: int = 50,
                cartesian_iterations: int = 500,
                refinement_convergence: float = 0.01,
                systematic_sampling_divisions: int = 6,
                verbose: bool = True, 
                print_interval: int = 5) -> Dict[str, Any]:
        """
        Run torsional optimization using exponential ring closure score.
        
        The algorithm runs in 4 sequential phases:
        1. GA runs until convergence (based on ring closure score)
        2. Optional: Smoothing refinement on top N candidates (torsional optimization)
        3. Optional: Z-matrix refinement on top N candidates (optimizes all DOFs in Z-matrix space)
        4. Optional: Cartesian refinement on top N candidates (full geometry optimization)
        
        The GA fitness is based purely on the exponential ring closure score (not energy).
        
        Parameters
        ----------
        population_size : int
            GA population size (default: 30)
        generations : int
            Maximum number of generations (default: 50)
        mutation_rate : float
            Mutation probability per gene (default: 0.15)
        mutation_strength : float
            Mutation strength in degrees (default: 10.0)
        crossover_rate : float
            Crossover probability (default: 0.7)
        elite_size : int
            Number of elite individuals to preserve (default: 5)
        torsion_range : Tuple[float, float]
            Min/max torsion angles in degrees (default: (-180.0, 180.0))
        ring_closure_tolerance : float
            Distance threshold (Å) for perfect ring closure score (default: 0.1)
        ring_closure_decay_rate : float
            Exponential decay rate for ring closure score (default: 0.5)
        convergence_interval : int
            Number of consecutive generations with small improvement to declare convergence (default: 5)
        enable_smoothing_refinement : bool
            Enable smoothing-based torsional refinement after GA (default: True)
        enable_zmatrix_refinement : bool
            Enable Z-matrix space refinement after smoothing (default: True)
        enable_cartesian_refinement : bool
            Enable Cartesian space refinement after Z-matrix refinement (default: True)
        refinement_top_n : int
            Number of top candidates to refine (default: 1)
        smoothing_sequence : List[float], optional
            Sequence of smoothing values for torsional refinement (default: [50.0, 25.0, 10.0, 5.0, 2.5, 1.0, 0.0])
        torsional_iterations : int
            Iterations per torsional optimization step (default: 50)
        zmatrix_iterations : int
            Iterations for Z-matrix space minimization (default: 50)
        cartesian_iterations : int
            Iterations for Cartesian minimization (default: 500)
        refinement_convergence : float
            Fitness improvement threshold for GA convergence (default: 0.01)
        systematic_sampling_divisions : int
            Number of discrete values for systematic sampling of critical torsions (default: 6)
        verbose : bool
            Print progress information (default: True)
        print_interval : int
            Print statistics every N generations (default: 5)
            
        Returns
        -------
        Dict[str, Any]
            Optimization results including:
            - 'initial_closure_score': Initial ring closure score
            - 'final_closure_score': Final ring closure score  
            - 'best_individual': Best Individual object
            - 'generations': Number of generations run
            - 'refinement_stats': Refinement statistics (if enabled)
        """
        # Set default smoothing sequence if not provided
        if smoothing_sequence is None:
            smoothing_sequence = [50.0, 25.0, 10.0, 5.0, 2.5, 1.0, 0.0]
        
        # Get RCP terms and topology from system
        rcp_terms = self.system.rcpterms
        topology = self.system.topology
        
        # Initialize GA with base zmatrix
        self.ga = GeneticAlgorithm(
            num_torsions=len(self.rotatable_indices),
            base_zmatrix=self.system.zmatrix,
            rotatable_indices=self.rotatable_indices,
            population_size=population_size,
            mutation_rate=mutation_rate,
            mutation_strength=mutation_strength,
            crossover_rate=crossover_rate,
            elite_size=elite_size,
            torsion_range=torsion_range,
            rcp_terms=rcp_terms,
            systematic_sampling_divisions=systematic_sampling_divisions,
            topology=topology
        )
        
        # Initialize local refinement optimizer if needed
        if enable_smoothing_refinement or enable_zmatrix_refinement or enable_cartesian_refinement:
            self.local_refinement = LocalRefinementOptimizer(
                self.system,
                self.rotatable_indices
            )
        
        # Evaluate initial ring closure score
        coords = zmatrix_to_cartesian(self.system.zmatrix)
        initial_closure_score = self.system.ring_closure_score_exponential(coords)
        initial_energy = self.system.evaluate_energy(coords)
        
        if verbose:
            print(f"Initial ring closure score: {initial_closure_score:.4f}")
            print(f"Initial energy: {initial_energy:.2f} kcal/mol")
            print(f"Rotatable dihedrals: {len(self.rotatable_indices)}")
            print(f"Fitness function: exponential ring closure score (tolerance={ring_closure_tolerance:.2f} Å, decay_rate={ring_closure_decay_rate:.2f})")
            if enable_smoothing_refinement:
                print(f"Smoothing refinement: enabled on top {refinement_top_n} candidates")
                print(f"  Smoothing sequence: {smoothing_sequence}")
                print(f"  Torsional iterations: {torsional_iterations}")
            if enable_cartesian_refinement:
                print(f"Cartesian refinement: enabled on top {refinement_top_n} candidates")
                print(f"  Cartesian iterations: {cartesian_iterations}")
        
        # Create fitness function
        fitness_fn = self._create_fitness_function(ring_closure_tolerance, ring_closure_decay_rate)
        
        # Genetic Algorithm 
        prev_best = np.inf
        refinement_stats = {'refined': 0, 'improved': 0}
        num_conv_gen = 0
        
        # Run GA
        print(f"\nGA exploration of torsional space with {len(self.rotatable_indices)} torsions...")

        for gen in range(generations):
            # Evaluate population
            self.ga.evaluate_population(fitness_fn, gen)
            
            # Get current best fitness
            current_best = self.ga.best_individual.fitness if self.ga.best_individual else np.inf
            
            # Count number of consecutive generations with improvement less than threshold
            if gen > 0 and abs(prev_best - current_best) < refinement_convergence:
                num_conv_gen += 1
            else:
                num_conv_gen = 0

            # Check if convergence interval has been reached
            converged = False if num_conv_gen < convergence_interval else True;
            if converged:
                if verbose:
                    print(f"Converged at generation {gen}")
                break
            
            # Print progress
            if verbose and (gen % print_interval == 0 or gen == generations - 1):
                stats = self.ga.get_statistics()
                print(f"Gen {gen:3d}: Best fitness = {stats['best']:8.4f}, "
                      f"Avg = {stats['average']:8.4f} - Conv: {num_conv_gen}/{convergence_interval}")
                
            prev_best = current_best
            
            # Evolve
            if gen < generations - 1:
                self.ga.evolve()
        
        # Store final generation number
        final_gen = gen

        # Show detailed RCP analysis after GA convergence
        if verbose:
            print("\nRCP analysis of best individual after GA convergence:")
            best_coords = zmatrix_to_cartesian(self.ga.best_individual.zmatrix)
            self.system.ring_closure_score_exponential(best_coords, verbose=True)
        
        # Take top N candidates from GA population
        sorted_indices = sorted(range(len(self.ga.population)),
                               key=lambda i: self.ga.population[i].fitness)
        top_indices = sorted_indices[:refinement_top_n]
        top_candidates = [self.ga.population[i] for i in top_indices]

        # Select which torsions to refine: do we protect the critical torsions?
        selected_rotatable_indices = self.rotatable_indices
        if protect_rc_critical_rotatable_bonds:
            selected_rotatable_indices = [idx for idx in self.rotatable_indices if idx not in self.ga.rc_critical_rotatable_indeces]
        
        non_rc_dof_indexes = []
        for idx in selected_rotatable_indices:
            if (idx, 2) not in non_rc_dof_indexes:
                non_rc_dof_indexes.append((idx, 2))

        # Refinement in torsional space with optional smoothing of the potential energy
        print(f"\nApplying torsional refinement ({len(selected_rotatable_indices)} torsions) with potential energy smoothing to top candidates...")

        if len(selected_rotatable_indices) == 0:
            print("Warning: No torsions to refine. Skipping torsional refinement.")
        else:
            pss_ref_init = time.time()
            for i, individual in enumerate(top_candidates):
                initial_energy = individual.energy
                if initial_energy is None:
                    initial_energy = self.system.evaluate_energy(zmatrix_to_cartesian(individual.zmatrix))
                    individual.energy = initial_energy

                refined_individual = self.local_refinement.refine_individual_in_torsional_space_with_smoothing(
                    individual, 
                    rotatable_indices=selected_rotatable_indices, 
                    smoothing_sequence=smoothing_sequence, 
                    torsional_iterations=torsional_iterations)

                top_candidates[i] = refined_individual
                refined_energy = refined_individual.energy
                energy_improvement = refined_energy - initial_energy
                print(f"  Torsional refinement {i+1}/{len(top_candidates)}: Energy improvement = {energy_improvement:.2f} kcal/mol (from {initial_energy:.2f} to {refined_energy:.2f} kcal/mol)")

            pss_ref_time = time.time() - pss_ref_init
            print(f"  Torsional refinement time: {pss_ref_time:.2f} seconds")

        print(f"\nApplying Z-matrix refinement ({len(self.dof_indices)} DOFs) to top candidates...")

        if len(selected_rotatable_indices) == 0:
            print("Warning: No torsions to refine. Skipping Z-matrix refinement.")
        else:
            zms_ref_init = time.time()
            for i, individual in enumerate(top_candidates):
                initial_energy = individual.energy
                if initial_energy is None:
                    initial_energy = self.system.evaluate_energy(zmatrix_to_cartesian(individual.zmatrix))
                    individual.energy = initial_energy
                refined_individual = self.local_refinement.refine_individual_in_zmatrix_space_with_smoothing(individual,
                    smoothing_sequence = [0.0],
                    dof_indices = self.dof_indices,
                    max_iterations=zmatrix_iterations)
                top_candidates[i] = refined_individual
                refined_energy = refined_individual.energy
                energy_improvement = refined_energy - initial_energy
                print(f"  Z-matrix refinement {i+1}/{len(top_candidates)}: Energy improvement = {energy_improvement:.2f} kcal/mol (from {initial_energy:.2f} to {refined_energy:.2f} kcal/mol)")
            zms_ref_time = time.time() - zms_ref_init
            print(f"  Z-matrix refinement time: {zms_ref_time:.2f} seconds")

        # Compile results
        final_closure_scores = np.array([self.system.ring_closure_score_exponential(
            zmatrix_to_cartesian(individual.zmatrix)
        ) for individual in top_candidates])

        results = {
            'initial_closure_score': initial_closure_score,
            'final_closure_score': final_closure_scores,
            'refined_individuals': [individual.zmatrix for individual in top_candidates],
            'refined_energies': [individual.energy for individual in top_candidates]
        }

        self.top_candidates = top_candidates
        
        return results
    

    def save_optimized_structure(self, filepath: str) -> None:
        """
        Save optimized structure to file.
        
        Saves the best individual's zmatrix, possibly refined, converted to Cartesian coordinates according the the givne file extension.
        
        Parameters
        ----------
        filepath : str
            Output file path
        """
        if self.top_candidates is None:
            raise ValueError("No optimization has been performed yet")
        
        IOTools.save_structure_to_file(filepath, self.top_candidates[0].zmatrix, self.top_candidates[0].energy)
        
        
    def minimize(self, max_iterations: int = 500,
                 smoothing: Optional[Union[float, List[float]]] = None,
                 torsional_space: bool = False,
                 zmat_space: bool = False,
                 update_system: bool = True,
                 verbose: bool = True) -> Dict[str, Any]:
        """
        Perform energy minimization on the current structure.
        
        This method minimizes the energy either in Cartesian space (default) or
        torsional space. It operates on a single structure (the current Z-matrix
        in the molecular system) without requiring a genetic algorithm population.
        
        Supports smoothing sequences: if smoothing is a list, performs a sequence
        of minimizations with decreasing smoothing values (similar to
        refine_candidate_with_smoothing).
        
        Parameters
        ----------
        max_iterations : int
            Maximum number of minimization iterations per step (default: 500)
        smoothing : Optional[Union[float, List[float]]]
            Smoothing parameter(s). Can be:
            - None (default): no smoothing (uses smoothing=0.0)
            - float: single smoothing value
            - List[float]: sequence of smoothing values to apply in decreasing order
              Example: [50.0, 25.0, 10.0, 5.0, 2.5, 1.0, 0.0]
        torsional_space : bool
            If True, perform minimization in torsional space only (optimize dihedrals).
            If False, perform minimization in Cartesian space (default: False)
        zmat_space : bool
            If True, perform minimization in Z-matrix space only (optimize bond lengths, angles, and dihedrals).
            If False, perform minimization in Cartesian space (default: False)
        update_system : bool
            If True, update self.system.zmatrix with minimized structure (default: True)
        verbose : bool
            Print minimization progress (default: True)
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'initial_energy': Energy before minimization (kcal/mol)
            - 'final_energy': Energy after minimization (kcal/mol)
            - 'coordinates': Minimized Cartesian coordinates (numpy array)
            - 'zmatrix': Minimized Z-matrix (list of dicts)
            - 'torsions': Extracted torsional angles (numpy array)
            - 'ring_closure_score': Ring closure score after minimization
            - 'improvement': Energy improvement (kcal/mol)
            - 'minimization_type': 'cartesian' or 'torsional'
            - 'optimization_info': Additional optimization info (for torsional minimization)
            - 'smoothing_sequence': List of smoothing values used
        
        Example
        -------
        >>> optimizer = RingClosureOptimizer.from_files(...)
        >>> # Cartesian minimization (default, no smoothing)
        >>> result = optimizer.minimize(max_iterations=1000)
        >>> # Torsional minimization with smoothing sequence
        >>> result = optimizer.minimize(torsional=True, 
        ...                              smoothing=[50.0, 25.0, 10.0, 0.0],
        ...                              max_iterations=100)
        >>> # Single smoothing value
        >>> result = optimizer.minimize(smoothing=10.0, max_iterations=500)
        """
        # Get initial Z-matrix and convert to Cartesian for energy evaluation
        initial_zmatrix = self.system.zmatrix
        initial_coords = zmatrix_to_cartesian(initial_zmatrix)
        
        # Determine smoothing sequence to use
        if smoothing is None:
            # Default: no smoothing (0.0)
            smoothing_values = [0.0]
        elif isinstance(smoothing, (list, tuple)):
            # Sequence of smoothing values
            smoothing_values = list(smoothing)
        else:
            # Single smoothing value
            smoothing_values = [smoothing]
        
        # Evaluate initial energy at first smoothing value
        self.system.setSmoothingParameter(smoothing_values[0])
        initial_energy = self.system.evaluate_energy(initial_coords)

        #write_xyz_file(initial_coords, self.system.elements, "initial_coords.xyz")
        
        # Extract initial torsions
        initial_torsions = np.array([initial_zmatrix[idx]['dihedral']
                                    for idx in self.rotatable_indices])
        
        if verbose:
            space_type = "torsional" if torsional_space else "zmatrix" if zmat_space else "Cartesian"
            if len(smoothing_values) > 1:
                print(f"Initial energy: {initial_energy:.4f} kcal/mol")
                print(f"Minimizing in {space_type} space with smoothing sequence: {smoothing_values}")
            else:
                print(f"Initial energy: {initial_energy:.4f} kcal/mol")
                print(f"Minimizing in {space_type} space with {max_iterations} max iterations (smoothing={smoothing_values[0]:.2f})...")
        
        try:
            current_zmatrix = initial_zmatrix
            all_opt_info = []
            
            # Perform sequence of minimizations with decreasing smoothing
            for step, smoothing_val in enumerate(smoothing_values):
                self.system.setSmoothingParameter(smoothing_val)
                
                if verbose and len(smoothing_values) > 1:
                    print(f"Step {step+1}/{len(smoothing_values)}: smoothing={smoothing_val:.2f}")
                
                if torsional_space:
                    # Perform torsional minimization
                    refined_zmatrix, step_energy, opt_info = self.system.minimize_energy_in_torsional_space(
                        current_zmatrix,
                        self.rotatable_indices,
                        max_iterations=max_iterations,
                        method='Powell',
                        verbose=verbose and len(smoothing_values) == 1  # Only verbose for single step
                    )
                    
                    if opt_info.get('success'):
                        current_zmatrix = refined_zmatrix
                        all_opt_info.append(opt_info)
                    else:
                        if verbose:
                            print(f"  Warning: Minimization step failed, continuing with previous structure")
                        
                elif zmat_space:
                    # Perform Z-matrix minimization
                    init_time = time.time()
                    refined_zmatrix, step_energy, opt_info = self.system.minimize_energy_in_zmatrix_space(
                        current_zmatrix,
                        self.dof_indices,
                        dof_bounds = [[0.02, 10.0, 10.0],[0.02, 5.0, 5.0]],
                        max_iterations=max_iterations,
                        method='L-BFGS-B',
                        verbose=verbose and len(smoothing_values) == 1  # Only verbose for single step
                    )
                    time_taken = time.time() - init_time
                    print(f"  Time taken: {time_taken:.2f} seconds")

                    if opt_info.get('success'):
                        current_zmatrix = refined_zmatrix
                        all_opt_info.append(opt_info)
                    else:
                        if verbose:
                            print(f"  Warning: Minimization step failed, continuing with previous structure")
                        
                else:
                    # Work in Cartesian space, even though results are in Z-matrix space
                    minimized_coords, step_energy = self.system.minimize_energy(
                        current_zmatrix,
                        max_iterations=max_iterations
                    )
                    
                    # Extract refined Z-matrix from minimized coordinates
                    refined_zmatrix = self.converter.extract_zmatrix(
                        minimized_coords,
                        current_zmatrix
                    )
                    current_zmatrix = refined_zmatrix
                    opt_info = {}
            
            # Final state after all smoothing steps
            minimized_zmatrix = current_zmatrix
            minimized_coords = zmatrix_to_cartesian(minimized_zmatrix)
            
            # Evaluate final energy
            self.system.setSmoothingParameter(0.0)  # Final evaluation at no smoothing
            minimized_energy = self.system.evaluate_energy(minimized_coords)
            
            # Extract refined torsions
            refined_torsions = np.array([minimized_zmatrix[idx]['dihedral']
                                        for idx in self.rotatable_indices])
            
            # Combine optimization info (for torsional minimization)
            if torsional_space and all_opt_info:
                combined_opt_info = {
                    'nfev': sum(info.get('nfev', 0) for info in all_opt_info),
                    'nit': sum(info.get('nit', 0) for info in all_opt_info),
                    'steps': len(all_opt_info),
                    'step_info': all_opt_info
                }
            elif zmat_space and all_opt_info:
                combined_opt_info = {
                    'nfev': sum(info.get('nfev', 0) for info in all_opt_info),
                    'nit': sum(info.get('nit', 0) for info in all_opt_info),
                    'steps': len(all_opt_info),
                    'step_info': all_opt_info
                }
            else:
                combined_opt_info = {}
            
            # Calculate ring closure score
            ring_closure_score = self.system.ring_closure_score_exponential(
                minimized_coords,
                verbose=False
            )
            
            # Update system Z-matrix if requested
            if update_system:
                self.system.zmatrix = minimized_zmatrix
            
            improvement = initial_energy - minimized_energy
            
            if verbose:
                print(f"Final energy:   {minimized_energy:.4f} kcal/mol")
                print(f"Improvement:    {improvement:.4f} kcal/mol")
                print(f"Ring closure:   {ring_closure_score:.4f}")
                rmsd_bond_lengths, rmsd_angles, rmsd_dihedrals = self.system.calculate_rmsd(initial_zmatrix, minimized_zmatrix)
                print(f"RMSD bond lengths: {rmsd_bond_lengths:.4f} Å")
                print(f"RMSD angles: {rmsd_angles:.4f} deg")
                print(f"RMSD dihedrals: {rmsd_dihedrals:.4f} deg")

            return {
                'initial_energy': initial_energy,
                'final_energy': minimized_energy,
                'coordinates': minimized_coords,
                'zmatrix': minimized_zmatrix,
                'torsions': refined_torsions,
                'ring_closure_score': ring_closure_score,
                'rmsd_bond_lengths': rmsd_bond_lengths,
                'rmsd_angles': rmsd_angles,
                'rmsd_dihedrals': rmsd_dihedrals,
                'improvement': improvement,
                'minimization_type': 'torsional' if torsional_space else 'zmatrix' if zmat_space else 'Cartesian',
                'optimization_info': combined_opt_info,
                'smoothing_sequence': smoothing_values,
                'success': True
            }
            
        except Exception as e:
            if verbose:
                print(f"Error during minimization: {e}")
            return {
                'initial_energy': initial_energy,
                'final_energy': initial_energy,
                'coordinates': initial_coords,
                'zmatrix': initial_zmatrix,
                'torsions': initial_torsions,
                'ring_closure_score': self.system.ring_closure_score_exponential(
                    initial_coords, verbose=False),
                'rmsd_bond_lengths': 0.0,
                'rmsd_angles': 0.0,
                'rmsd_dihedrals': 0.0,
                'improvement': 0.0,
                'minimization_type': 'torsional' if torsional_space else 'zmatrix' if zmat_space else 'Cartesian',
                'optimization_info': {},
                'smoothing_sequence': smoothing_values,
                'success': False,
                'error': str(e)
            }

