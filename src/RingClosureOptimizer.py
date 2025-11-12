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
    
    
    def _identify_rc_critical_rotatable_indeces(self) -> List[int]:
        """
        Identify rotatable torsions on paths between RCP atoms.
        
        Returns
        -------
        List[int]
            Indices (into rotatable_indices) of critical torsions
        """
        rc_critical_rotatable_indeces, _ = MolecularSystem._identify_rc_critical_rotatable_indeces(
            self.base_zmatrix,
            self.rcp_terms,
            self.rotatable_indices,
            self.topology
        )
        return rc_critical_rotatable_indeces
    
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
        # Set rotatable indices on the system (this computes rc_critical_rotatable_indeces and dof_indices)
        self.system.set_rotatable_indices(rotatable_indices)
        self.converter = CoordinateConverter()
        self.ga = None
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
            rotatable_indices = MolecularSystem._get_all_rotatable_indices(system.zmatrix)
        else:
            rotatable_indices = cls._convert_bonds_to_indices(rotatable_bonds, system.zmatrix)
        
        return cls(system, rotatable_indices, write_candidate_files)
    

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
                refinement_top_n: int = 1,
                smoothing_sequence: List[float] = None,
                torsional_iterations: int = 50,
                zmatrix_iterations: int = 50,
                refinement_convergence: float = 0.01,
                systematic_sampling_divisions: int = 6,
                verbose: bool = True, 
                print_interval: int = 5) -> Dict[str, Any]:
        """
        Run torsional optimization using exponential ring closure score.
        
        The algorithm runs in 3 sequential phases:
        1. GA runs until convergence (based on ring closure score)
        2. Optional: Smoothing refinement on top N candidates (torsional optimization)
        3. Optional: Z-matrix refinement on top N candidates (optimizes all DOFs in Z-matrix space)
        
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
        refinement_top_n : int
            Number of top candidates to refine (default: 1)
        smoothing_sequence : List[float], optional
            Sequence of smoothing values for torsional refinement (default: [50.0, 25.0, 10.0, 5.0, 2.5, 1.0, 0.0])
        torsional_iterations : int
            Iterations per torsional optimization step (default: 50)
        zmatrix_iterations : int
            Iterations for Z-matrix space minimization (default: 50)
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
            - 'refined_individuals': List of refined Z-matrices for top candidates
            - 'refined_energies': List of refined energies for top candidates
        """
        # Set default smoothing sequence if not provided
        if smoothing_sequence is None:
            smoothing_sequence = [50.0, 25.0, 10.0, 5.0, 2.5, 1.0, 0.0]
        
        # Get RCP terms and topology from system
        rcp_terms = self.system.rcpterms
        topology = self.system.topology
        
        # Initialize GA with base zmatrix
        self.ga = GeneticAlgorithm(
            num_torsions=len(self.system.rotatable_indices),
            base_zmatrix=self.system.zmatrix,
            rotatable_indices=self.system.rotatable_indices,
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
        
        # Evaluate initial ring closure score
        coords = zmatrix_to_cartesian(self.system.zmatrix)
        initial_zmatrix = self.system.zmatrix
        initial_coords = coords
        initial_closure_score = self.system.ring_closure_score_exponential(coords)
        initial_energy = self.system.evaluate_energy(coords)
        
        if verbose:
            print(f"Initial ring closure score: {initial_closure_score:.4f}")
            print(f"Initial energy: {initial_energy:.2f} kcal/mol")
            print(f"Rotatable dihedrals: {len(self.system.rotatable_indices)}")
            print(f"Fitness function: exponential ring closure score (tolerance={ring_closure_tolerance:.2f} Å, decay_rate={ring_closure_decay_rate:.2f})")
            if enable_smoothing_refinement:
                print(f"Smoothing refinement: enabled on top {refinement_top_n} candidates")
                print(f"  Smoothing sequence: {smoothing_sequence}")
                print(f"  Torsional iterations: {torsional_iterations}")
            if enable_zmatrix_refinement:
                print(f"Z-matrix refinement: enabled on top {refinement_top_n} candidates")
                print(f"  Z-matrix iterations: {zmatrix_iterations}")
        
        # Create fitness function
        fitness_fn = self._create_fitness_function(ring_closure_tolerance, ring_closure_decay_rate)
        
        # Genetic Algorithm 
        prev_best = np.inf
        num_conv_gen = 0
        
        # Run GA
        print(f"\nGA exploration of torsional space with {len(self.system.rotatable_indices)} torsions...")

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
        selected_rotatable_indices = self.system.rotatable_indices
        if protect_rc_critical_rotatable_bonds:
            selected_rotatable_indices = [idx for idx in self.system.rotatable_indices if idx not in self.system.rc_critical_rotatable_indeces]
        
        non_rc_dof_indexes = []
        for idx in selected_rotatable_indices:
            if (idx, 2) not in non_rc_dof_indexes:
                non_rc_dof_indexes.append((idx, 2))

        # Refinement in torsional space with optional smoothing of the potential energy
        print(f"\nApplying torsional refinement ({len(selected_rotatable_indices)} torsions) with potential energy smoothing to top candidates...")

        if enable_smoothing_refinement:
            if len(selected_rotatable_indices) == 0:
                print("Warning: No torsions to refine. Skipping torsional refinement.")
            else:
                pss_ref_init = time.time()
                for i, individual in enumerate(top_candidates):
                    initial_energy = individual.energy
                    if initial_energy is None:
                        initial_energy = self.system.evaluate_energy(zmatrix_to_cartesian(individual.zmatrix))
                        individual.energy = initial_energy

                    # Refine individual in torsional space with smoothing
                    current_zmatrix = individual.zmatrix
                    for smoothing in smoothing_sequence:
                        self.system.setSmoothingParameter(smoothing)
                        refined_zmatrix, refined_energy, info = self.system.minimize_energy_in_torsional_space(
                            current_zmatrix,
                            selected_rotatable_indices,
                            max_iterations=torsional_iterations,
                            verbose=False
                        )
                        if info['success']:
                            current_zmatrix = refined_zmatrix
                    
                    # Extract refined torsions
                    refined_torsions = np.array([current_zmatrix[idx]['dihedral']
                                                for idx in self.system.rotatable_indices])
                    refined_individual = Individual(refined_torsions, current_zmatrix, energy=refined_energy)
                    top_candidates[i] = refined_individual
                    refined_energy = refined_individual.energy
                    energy_improvement = refined_energy - initial_energy
                    print(f"  Torsional refinement {i+1}/{len(top_candidates)}: Energy improvement = {energy_improvement:.2f} kcal/mol (from {initial_energy:.2f} to {refined_energy:.2f} kcal/mol)")

                pss_ref_time = time.time() - pss_ref_init
                print(f"  Torsional refinement time: {pss_ref_time:.2f} seconds")

        if enable_zmatrix_refinement:
            print(f"\nApplying Z-matrix refinement ({len(self.system.dof_indices)} DOFs) to top candidates...")

            if len(self.system.dof_indices) == 0:
                print("Warning: No torsions to refine. Skipping Z-matrix refinement.")
            else:
                zms_ref_init = time.time()
                for i, individual in enumerate(top_candidates):
                    initial_energy = individual.energy
                    if initial_energy is None:
                        initial_energy = self.system.evaluate_energy(zmatrix_to_cartesian(individual.zmatrix))
                        individual.energy = initial_energy
                    # Refine individual in Z-matrix space with smoothing
                    current_zmatrix = individual.zmatrix
                    for smoothing in [0.0]:
                        self.system.setSmoothingParameter(smoothing)
                        refined_zmatrix, refined_energy, info = self.system.minimize_energy_in_zmatrix_space(
                            current_zmatrix,
                            dof_indices=self.system.dof_indices,
                            max_iterations=zmatrix_iterations,
                            verbose=False
                        )
                        if info['success']:
                            current_zmatrix = refined_zmatrix
                    
                    # Extract refined torsions
                    refined_torsions = np.array([current_zmatrix[idx]['dihedral']
                                                for idx in self.system.rotatable_indices])
                    refined_individual = Individual(refined_torsions, current_zmatrix, energy=refined_energy)
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


        print("\nBest individual:")
        rmsd_bond_lengths, rmsd_angles, rmsd_dihedrals = MolecularSystem._calculate_rmsd(initial_zmatrix, top_candidates[0].zmatrix)
        print(f"RMSD bond lengths: {rmsd_bond_lengths:.4f} Å")
        print(f"RMSD angles: {rmsd_angles:.4f} deg")
        print(f"RMSD dihedrals: {rmsd_dihedrals:.4f} deg")

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
                 space_type: str = "Cartesian",
                 zmatrix_dof_bounds_per_type: Optional[List[Tuple[float, float, float]]] = [[10.0, 180.0, 180.0]],
                 gradient_tolerance: float = 0.01,
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
        space_type : str
            Type of space to minimize in: 'torsional', 'zmatrix', or 'Cartesian' (default: 'Cartesian')
        zmatrix_dof_bounds_per_type : Optional[List[Tuple[float, float, float]]]
            Bounds for types of degrees of freedom in Z-matrix space. Example: [0.02, 5.0, 10.0] means 
            that bond lengths are bound to change to up to 0.02 Å of the current value, angles and 5.0 degrees, and torsions by 10.0 degrees. Multiple tuples can be provided to request any stepwise application of bounds. Example: [[0.02, 5.0, 10.0], [0.01, 3.0, 8.0]] means will make the minimization run with [0.02, 5.0, 10.0] for the first step and [0.01, 3.0, 8.0] for the second step. Default is [(10.0, 180.0, 180.0)].
        gradient_tolerance : float
            Gradient tolerance for minimization (default: 0.01)
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
            - 'ring_closure_score': Ring closure score after minimization
            - 'minimization_type': 'cartesian' or 'torsional' or 'zmatrix'
            - 'optimization_info': Additional optimization info (for torsional minimization)
            - 'smoothing_sequence': List of smoothing values used
            - 'success': True if minimization was successful, False otherwise
            - 'error': Error message if minimization failed
        
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
        ring_closure_score = self.system.ring_closure_score_exponential(
                initial_coords,
                verbose=False
            )
        
        if verbose:
            print(f"Initial energy: {initial_energy:.4f} kcal/mol")
            print(f"Initial ring closure score: {ring_closure_score:.4f}")
            if len(smoothing_values) > 1:
                print(f"Minimizing in {space_type} space with smoothing sequence: {smoothing_values}")
            else:
                print(f"Minimizing in {space_type} space with {max_iterations} max iterations (smoothing={smoothing_values[0]:.2f})...")
        
        try:
            current_zmatrix = initial_zmatrix
            all_opt_info = []
            
            # Perform sequence of minimizations with smoothed potential energy
            for step, smoothing_val in enumerate(smoothing_values):
                self.system.setSmoothingParameter(smoothing_val)
                
                if verbose and len(smoothing_values) > 1:
                    print(f"Step {step+1}/{len(smoothing_values)}: smoothing={smoothing_val:.2f}")
                
                if space_type == 'torsional':
                    # Perform torsional minimization
                    refined_zmatrix, step_energy, opt_info = self.system.minimize_energy_in_torsional_space(
                        current_zmatrix,
                        self.system.rotatable_indices,
                        max_iterations=max_iterations,
                        verbose=verbose
                    )
                    
                    if opt_info.get('success'):
                        current_zmatrix = refined_zmatrix
                        all_opt_info.append(opt_info)
                    else:
                        if verbose:
                            print(f"  Warning: Minimization step failed, continuing with previous structure")
                        
                elif space_type == 'zmatrix':
                    # Perform Z-matrix minimization
                    init_time = time.time()
                    refined_zmatrix, step_energy, opt_info = self.system.minimize_energy_in_zmatrix_space(
                        current_zmatrix,
                        self.system.dof_indices,
                        dof_bounds_per_type = zmatrix_dof_bounds_per_type,
                        max_iterations=max_iterations,
                        gradient_tolerance=gradient_tolerance,
                        verbose=verbose
                    )
                    time_taken = time.time() - init_time
                    print(f"\nTime taken for Z-matrix minimization: {time_taken:.2f} seconds")

                    if opt_info.get('success'):
                        current_zmatrix = refined_zmatrix
                        all_opt_info.append(opt_info)
                    else:
                        if verbose:
                            print(f"  Warning: Minimization step failed, continuing with previous structure")
                        
                elif space_type == 'Cartesian':
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
                
                else:
                    raise ValueError(f"Invalid space type: {space_type}")
            
            # Final state after all smoothing steps
            minimized_zmatrix = current_zmatrix
            minimized_coords = zmatrix_to_cartesian(minimized_zmatrix)
            
            # Evaluate final energy
            self.system.setSmoothingParameter(0.0)  # Final evaluation at no smoothing
            minimized_energy = self.system.evaluate_energy(minimized_coords)
            
            # Calculate ring closure score
            ring_closure_score = self.system.ring_closure_score_exponential(
                minimized_coords,
                verbose=False
            )
            
            self.system.zmatrix = minimized_zmatrix
            
            if verbose:
                print(f"\nFinal energy:   {minimized_energy:.4f} kcal/mol")
                print(f"Improvement:    {initial_energy - minimized_energy:.4f} kcal/mol")
                print(f"Ring closure:   {ring_closure_score:.4f}")
                rmsd_bond_lengths, rmsd_angles, rmsd_dihedrals = MolecularSystem._calculate_rmsd(initial_zmatrix, minimized_zmatrix)
                print(f"RMSD bond lengths: {rmsd_bond_lengths:.4f} Å")
                print(f"RMSD angles: {rmsd_angles:.4f} deg")
                print(f"RMSD dihedrals: {rmsd_dihedrals:.4f} deg")

            return {
                'initial_energy': initial_energy,
                'final_energy': minimized_energy,
                'coordinates': minimized_coords,
                'zmatrix': minimized_zmatrix,
                'ring_closure_score': ring_closure_score,
                'rmsd_bond_lengths': rmsd_bond_lengths,
                'rmsd_angles': rmsd_angles,
                'rmsd_dihedrals': rmsd_dihedrals,
                'minimization_type': space_type,
                'optimization_info': all_opt_info,
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
                'ring_closure_score': self.system.ring_closure_score_exponential(
                    initial_coords, verbose=False),
                'rmsd_bond_lengths': 0.0,
                'rmsd_angles': 0.0,
                'rmsd_dihedrals': 0.0,
                'minimization_type': space_type,
                'optimization_info': {},
                'smoothing_sequence': smoothing_values,
                'success': False,
                'error': str(e)
            }

