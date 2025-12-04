#!/usr/bin/env python3

import time
from typing import List, Tuple, Dict, Optional, Any, Union

from ringclosingmm.IOTools import write_xyz_file, write_zmatrix_file

from .CoordinateConversion import (
    zmatrix_to_cartesian,
    cartesian_to_zmatrix
)
from .MolecularSystem import MolecularSystem
from .ZMatrix import ZMatrix

class RingClosureOptimizer:
    """
    Tools for optimization of ring closure conformations of 3D molecular fragments.

    This module provides the main framework for optimizing molecular conformation  
    to achieve ring closure. 
        
    Example:
        optimizer = RingClosureOptimizer.from_files(
            structure_file='molecule.int',
            forcefield_file='force_field.xml',
            rotatable_bonds=[(1, 2), (5, 31)],
            rcp_terms=[(7, 39), (77, 35)]
        )
        
        result = optimizer.optimize(...)
    """
    
    def __init__(self, molecular_system: MolecularSystem,
                 rotatable_indices: List[int]):
        """
        Initialize ring closure optimizer.
        
        Parameters
        ----------
        molecular_system : MolecularSystem
            Molecular system to optimize
        rotatable_indices : List[int]
            Indices of rotatable atoms in Z-matrix
        """
        self.system = molecular_system
        # Set rotatable indices on the system (this computes rc_critical_rotatable_indeces and dof_indices)
        self.system.set_rotatable_indices(rotatable_indices)
    
    @classmethod
    def from_files(cls, structure_file: str, forcefield_file: str,
                   rotatable_bonds: Optional[List[Tuple[int, int]]] = None,
                   rcp_terms: Optional[List[Tuple[int, int]]] = None,
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
            ring_closure_threshold=ring_closure_threshold
        )
        
        # Convert rotatable bonds to rotatable indices
        if rotatable_bonds is None:
            # All bonds are rotatable: get all dihedrals with chirality == 0
            rotatable_indices = MolecularSystem._get_all_rotatable_indices(system.zmatrix)
        else:
            rotatable_indices = cls._convert_bonds_to_indices(rotatable_bonds, system.zmatrix)
        
        return cls(system, rotatable_indices)
    

    @staticmethod
    def _convert_bonds_to_indices(rotatable_bonds: List[Tuple[int, int]], 
                                   zmatrix: ZMatrix) -> List[int]:
        """
        Convert rotatable bond pairs to rotatable Z-matrix indices.
        
        Parameters
        ----------
        rotatable_bonds : List[Tuple[int, int]]
            List of (atom1, atom2) pairs
        zmatrix : ZMatrix
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
            if atom.get(ZMatrix.FIELD_CHIRALITY, 0) == 0:  # Only true dihedrals
                bond_ref = atom[ZMatrix.FIELD_BOND_REF]
                angle_ref = atom[ZMatrix.FIELD_ANGLE_REF]
                # Check both orderings (bond_ref, angle_ref) and (angle_ref, bond_ref)
                if ((bond_ref, angle_ref) in bonded_pairs or 
                    (angle_ref, bond_ref) in bonded_pairs):
                    rotatable_indices.append(i)
        
        return rotatable_indices
    

    def print_force_field(self, max_sample_size: Optional[int] = None, 
                         atom_indices: Optional[List[int]] = None) -> None:
        """
        Print all force field contributions in an organized list.
        
        This method iterates through all forces in the OpenMM system and prints
        detailed information about each force type, including the number of terms,
        parameters, and other relevant information.
        
        Parameters
        ----------
        max_sample_size : Optional[int], default=None
            Maximum number of sample parameters to print for each force type.
            If None, all parameters are printed. If specified, only the first
            max_sample_size parameters are shown, with a message indicating
            how many more exist.
        atom_indices : Optional[List[int]], default=None
            List of atom indices (0-based) to filter forces. If specified, only
            forces involving any of these atoms will be shown. If None, all
            forces are shown.
        """
        import openmm as mm
        
        system = self.system.system
        forces = system.getForces()
        
        # Convert atom_indices to set for faster lookup
        atom_set = set(atom_indices) if atom_indices is not None else None
        
        print("=" * 80)
        print("Force Field Contributions")
        if atom_indices is not None:
            # Convert to 1-based for display
            atom_indices_1based = [idx + 1 for idx in atom_indices]
            print(f"Filtered to atoms: {atom_indices_1based} (1-based)")
        print("=" * 80)
        print(f"Total number of forces: {len(forces)}\n")
        
        for i, force in enumerate(forces):
            force_name = force.getName() if hasattr(force, 'getName') and callable(force.getName) else force.__class__.__name__
            force_type = force.__class__.__name__
            
            print(f"{i+1}. {force_type}")
            if force_name and force_name != force_type:
                print(f"   Name: {force_name}")
            
            # Handle different force types
            if isinstance(force, mm.HarmonicBondForce):
                num_bonds = force.getNumBonds()
                # Filter bonds involving specified atoms
                matching_bonds = []
                for j in range(num_bonds):
                    p1, p2, length, k = force.getBondParameters(j)
                    if atom_set is None or p1 in atom_set or p2 in atom_set:
                        matching_bonds.append((j, p1, p2, length, k))
                
                print(f"   Number of bonds: {num_bonds} (matching filter: {len(matching_bonds)})")
                if len(matching_bonds) > 0:
                    sample_size = len(matching_bonds) if max_sample_size is None else min(max_sample_size, len(matching_bonds))
                    print(f"   Bonds (showing {sample_size} of {len(matching_bonds)}):")
                    for idx, (orig_idx, p1, p2, length, k) in enumerate(matching_bonds[:sample_size]):
                        print(f"     Bond {orig_idx+1}: particles {p1}-{p2}, length={length}, k={k}")
                    if max_sample_size is not None and len(matching_bonds) > max_sample_size:
                        print(f"     ... and {len(matching_bonds) - max_sample_size} more bonds")
            
            elif isinstance(force, mm.HarmonicAngleForce):
                num_angles = force.getNumAngles()
                # Filter angles involving specified atoms
                matching_angles = []
                for j in range(num_angles):
                    p1, p2, p3, angle, k = force.getAngleParameters(j)
                    if atom_set is None or p1 in atom_set or p2 in atom_set or p3 in atom_set:
                        matching_angles.append((j, p1, p2, p3, angle, k))
                
                print(f"   Number of angles: {num_angles} (matching filter: {len(matching_angles)})")
                if len(matching_angles) > 0:
                    sample_size = len(matching_angles) if max_sample_size is None else min(max_sample_size, len(matching_angles))
                    print(f"   Angles (showing {sample_size} of {len(matching_angles)}):")
                    for idx, (orig_idx, p1, p2, p3, angle, k) in enumerate(matching_angles[:sample_size]):
                        print(f"     Angle {orig_idx+1}: particles {p1}-{p2}-{p3}, angle={angle}, k={k}")
                    if max_sample_size is not None and len(matching_angles) > max_sample_size:
                        print(f"     ... and {len(matching_angles) - max_sample_size} more angles")
            
            elif isinstance(force, mm.PeriodicTorsionForce):
                num_torsions = force.getNumTorsions()
                # Filter torsions involving specified atoms
                matching_torsions = []
                for j in range(num_torsions):
                    p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(j)
                    if atom_set is None or p1 in atom_set or p2 in atom_set or p3 in atom_set or p4 in atom_set:
                        matching_torsions.append((j, p1, p2, p3, p4, periodicity, phase, k))
                
                print(f"   Number of torsions: {num_torsions} (matching filter: {len(matching_torsions)})")
                if len(matching_torsions) > 0:
                    sample_size = len(matching_torsions) if max_sample_size is None else min(max_sample_size, len(matching_torsions))
                    print(f"   Torsions (showing {sample_size} of {len(matching_torsions)}):")
                    for idx, (orig_idx, p1, p2, p3, p4, periodicity, phase, k) in enumerate(matching_torsions[:sample_size]):
                        print(f"     Torsion {orig_idx+1}: particles {p1}-{p2}-{p3}-{p4}, "
                              f"periodicity={periodicity}, phase={phase}, k={k}")
                    if max_sample_size is not None and len(matching_torsions) > max_sample_size:
                        print(f"     ... and {len(matching_torsions) - max_sample_size} more torsions")
            
            elif isinstance(force, mm.NonbondedForce):
                num_particles = force.getNumParticles()
                num_exceptions = force.getNumExceptions()
                # Filter particles matching specified atoms
                matching_particles = []
                for j in range(num_particles):
                    if atom_set is None or j in atom_set:
                        charge, sigma, epsilon = force.getParticleParameters(j)
                        matching_particles.append((j, charge, sigma, epsilon))
                
                print(f"   Number of particles: {num_particles} (matching filter: {len(matching_particles)})")
                print(f"   Number of exceptions: {num_exceptions}")
                print(f"   Nonbonded method: {force.getNonbondedMethod()}")
                print(f"   Cutoff distance: {force.getCutoffDistance()}")
                if len(matching_particles) > 0:
                    sample_size = len(matching_particles) if max_sample_size is None else min(max_sample_size, len(matching_particles))
                    print(f"   Particle parameters (showing {sample_size} of {len(matching_particles)}):")
                    for idx, (particle_idx, charge, sigma, epsilon) in enumerate(matching_particles[:sample_size]):
                        print(f"     Particle {particle_idx}: charge={charge}, sigma={sigma}, epsilon={epsilon}")
                    if max_sample_size is not None and len(matching_particles) > max_sample_size:
                        print(f"     ... and {len(matching_particles) - max_sample_size} more particles")
            
            elif isinstance(force, mm.CustomNonbondedForce):
                num_particles = force.getNumParticles()
                num_exclusions = force.getNumExclusions()
                energy_function = force.getEnergyFunction()
                num_global_params = force.getNumGlobalParameters()
                num_per_particle_params = force.getNumPerParticleParameters()
                # Count matching particles
                matching_count = sum(1 for j in range(num_particles) if atom_set is None or j in atom_set)
                print(f"   Number of particles: {num_particles} (matching filter: {matching_count})")
                print(f"   Number of exclusions: {num_exclusions}")
                print(f"   Energy function: {energy_function}")
                print(f"   Global parameters ({num_global_params}):")
                for j in range(num_global_params):
                    param_name = force.getGlobalParameterName(j)
                    param_value = force.getGlobalParameterDefaultValue(j)
                    print(f"     {param_name} = {param_value}")
                if num_per_particle_params > 0:
                    print(f"   Per-particle parameters ({num_per_particle_params}):")
                    for j in range(num_per_particle_params):
                        param_name = force.getPerParticleParameterName(j)
                        print(f"     {param_name}")
            
            elif isinstance(force, mm.CustomCompoundBondForce):
                num_bonds = force.getNumBonds()
                energy_function = force.getEnergyFunction()
                num_global_params = force.getNumGlobalParameters()
                num_per_bond_params = force.getNumPerBondParameters()
                print(f"   Number of bonds: {num_bonds}")
                print(f"   Energy function: {energy_function}")
                print(f"   Global parameters ({num_global_params}):")
                for j in range(num_global_params):
                    param_name = force.getGlobalParameterName(j)
                    param_value = force.getGlobalParameterDefaultValue(j)
                    print(f"     {param_name} = {param_value}")
                if num_per_bond_params > 0:
                    print(f"   Per-bond parameters ({num_per_bond_params}):")
                    for j in range(num_per_bond_params):
                        param_name = force.getPerBondParameterName(j)
                        print(f"     {param_name}")
                # Filter bonds involving specified atoms
                matching_bonds = []
                for j in range(num_bonds):
                    particles, params = force.getBondParameters(j)
                    if atom_set is None or any(p in atom_set for p in particles):
                        matching_bonds.append((j, particles, params))
                
                if len(matching_bonds) > 0:
                    sample_size = len(matching_bonds) if max_sample_size is None else min(max_sample_size, len(matching_bonds))
                    print(f"   Bonds (showing {sample_size} of {len(matching_bonds)} matching, total: {num_bonds}):")
                    for idx, (orig_idx, particles, params) in enumerate(matching_bonds[:sample_size]):
                        print(f"     Bond {orig_idx+1}: particles {particles}, params={params}")
                    if max_sample_size is not None and len(matching_bonds) > max_sample_size:
                        print(f"     ... and {len(matching_bonds) - max_sample_size} more bonds")
            
            elif isinstance(force, mm.CustomBondForce):
                num_bonds = force.getNumBonds()
                energy_function = force.getEnergyFunction()
                num_global_params = force.getNumGlobalParameters()
                num_per_bond_params = force.getNumPerBondParameters()
                print(f"   Number of bonds: {num_bonds}")
                print(f"   Energy function: {energy_function}")
                print(f"   Global parameters ({num_global_params}):")
                for j in range(num_global_params):
                    param_name = force.getGlobalParameterName(j)
                    param_value = force.getGlobalParameterDefaultValue(j)
                    print(f"     {param_name} = {param_value}")
                if num_per_bond_params > 0:
                    print(f"   Per-bond parameters ({num_per_bond_params}):")
                    for j in range(num_per_bond_params):
                        param_name = force.getPerBondParameterName(j)
                        print(f"     {param_name}")
                # Filter bonds involving specified atoms
                matching_bonds = []
                for j in range(num_bonds):
                    p1, p2, params = force.getBondParameters(j)
                    if atom_set is None or p1 in atom_set or p2 in atom_set:
                        matching_bonds.append((j, p1, p2, params))
                
                if len(matching_bonds) > 0:
                    sample_size = len(matching_bonds) if max_sample_size is None else min(max_sample_size, len(matching_bonds))
                    print(f"   Bonds (showing {sample_size} of {len(matching_bonds)} matching, total: {num_bonds}):")
                    for idx, (orig_idx, p1, p2, params) in enumerate(matching_bonds[:sample_size]):
                        print(f"     Bond {orig_idx+1}: particles {p1}-{p2}, params={params}")
                    if max_sample_size is not None and len(matching_bonds) > max_sample_size:
                        print(f"     ... and {len(matching_bonds) - max_sample_size} more bonds")
            
            elif isinstance(force, mm.CustomAngleForce):
                num_angles = force.getNumAngles()
                energy_function = force.getEnergyFunction()
                num_global_params = force.getNumGlobalParameters()
                print(f"   Number of angles: {num_angles}")
                print(f"   Energy function: {energy_function}")
                print(f"   Global parameters ({num_global_params}):")
                for j in range(num_global_params):
                    param_name = force.getGlobalParameterName(j)
                    param_value = force.getGlobalParameterDefaultValue(j)
                    print(f"     {param_name} = {param_value}")
                # Filter angles involving specified atoms
                matching_angles = []
                for j in range(num_angles):
                    p1, p2, p3, params = force.getAngleParameters(j)
                    if atom_set is None or p1 in atom_set or p2 in atom_set or p3 in atom_set:
                        matching_angles.append((j, p1, p2, p3, params))
                
                if len(matching_angles) > 0:
                    sample_size = len(matching_angles) if max_sample_size is None else min(max_sample_size, len(matching_angles))
                    print(f"   Angles (showing {sample_size} of {len(matching_angles)} matching, total: {num_angles}):")
                    for idx, (orig_idx, p1, p2, p3, params) in enumerate(matching_angles[:sample_size]):
                        print(f"     Angle {orig_idx+1}: particles {p1}-{p2}-{p3}, params={params}")
                    if max_sample_size is not None and len(matching_angles) > max_sample_size:
                        print(f"     ... and {len(matching_angles) - max_sample_size} more angles")
            
            else:
                # Generic force - try to get basic info
                print(f"   Force type: {force_type}")
                if hasattr(force, 'getNumBonds'):
                    try:
                        num_bonds = force.getNumBonds()
                        print(f"   Number of bonds: {num_bonds}")
                        if num_bonds > 0 and max_sample_size is not None:
                            sample_size = min(max_sample_size, num_bonds)
                            print(f"   Sample bonds (showing {sample_size} of {num_bonds})")
                    except:
                        pass
                if hasattr(force, 'getNumAngles'):
                    try:
                        num_angles = force.getNumAngles()
                        print(f"   Number of angles: {num_angles}")
                        if num_angles > 0 and max_sample_size is not None:
                            sample_size = min(max_sample_size, num_angles)
                            print(f"   Sample angles (showing {sample_size} of {num_angles})")
                    except:
                        pass
                if hasattr(force, 'getNumGlobalParameters'):
                    try:
                        num_global = force.getNumGlobalParameters()
                        if num_global > 0:
                            print(f"   Global parameters ({num_global}):")
                            for j in range(num_global):
                                param_name = force.getGlobalParameterName(j)
                                param_value = force.getGlobalParameterDefaultValue(j)
                                print(f"     {param_name} = {param_value}")
                    except:
                        pass
            
            print()  # Empty line between forces
        
        print("=" * 80)


    def optimize(self, 
                ring_closure_tolerance: float = 0.001,
                ring_closure_decay_rate: float = 1.0, 
                enable_pssrot_refinement: bool = True,
                enable_zmatrix_refinement: bool = True,
                smoothing_sequence: List[float] = None,
                torsional_iterations: int = 50,
                zmatrix_iterations: int = 50,
                gradient_tolerance: float = 0.01,
                trajectory_file: Optional[str] = None,
                verbose: bool = False) -> Dict[str, Any]:
        """
        Run torsional optimization using a divide and conquer strategy:
        1) first use differential evolution to maximize ring closure score.
        2) then use smoothing-based torsional refinement to reduce strain energy by changing torsions that are not critical for ring closure.
        3) finally use Z-matrix space refinement to reduce strain energy by changing all degrees of freedom along ring closing path.
        
        Parameters
        ----------
        ring_closure_tolerance : float
            Distance threshold (Å) for perfect ring closure score (default: 0.1)
        ring_closure_decay_rate : float
            Exponential decay rate for ring closure score (default: 0.5)
        enable_pssrot_refinement : bool
            Enable smoothing-based torsional refinement after differential evolution (default: True)
        enable_zmatrix_refinement : bool
            Enable Z-matrix space refinement after smoothing (default: True)
        smoothing_sequence : List[float], optional
            Sequence of smoothing values for torsional refinement (default: [50.0, 15.0, 3.0, 1.0, 0.0])
        torsional_iterations : int
            Iterations per torsional optimization step (default: 50)
        zmatrix_iterations : int
            Iterations for Z-matrix space minimization (default: 50)
        trajectory_file : Optional[str]
            If provided, write optimization trajectory to this XYZ file (append mode).
            Writes coordinates for each refinement step in dedicated files. The given string is used as basename for trajectory files of each refinement step.
        verbose : bool
            Print progress information (default: True)
            
        Returns
        -------
        Dict[str, Any]
            Optimization results including:
            - 'initial_energy': Energy before optimization (kcal/mol)
            - 'final_energy': Energy after optimization (kcal/mol)
            - 'initial_ring_closure_score': Initial ring closure score
            - 'final_ring_closure_score': Final ring closure score
            - 'final_coords': Final Cartesian coordinates (numpy array)
            - 'final_zmatrix': Final Z-matrix (ZMatrix instance)
            - 'rmsd_bond_lengths': RMSD of bond lengths from initial structure
            - 'rmsd_angles': RMSD of angles from initial structure
            - 'rmsd_dihedrals': RMSD of dihedrals from initial structure
            - 'success': True if optimization was successful
        """
        # Set default smoothing sequence if not provided
        if smoothing_sequence is None:
            smoothing_sequence = [50.0, 15.0, 3.0, 1.0, 0.0]

        initial_zmatrix = self.system.zmatrix
        initial_coords = zmatrix_to_cartesian(initial_zmatrix)
        initial_ring_closure_score = self.system.ring_closure_score_exponential(initial_coords)
        initial_energy = self.system.evaluate_energy(initial_coords)

        trajectory_file_rc_opt = None
        if trajectory_file:
            trajectory_file_rc_opt = trajectory_file.replace('.xyz', '_rc_opt.xyz')

        non_intersecting_rcp_paths = self.system.get_non_intersecting_rcp_paths(initial_zmatrix, self.system.rc_critical_rotatable_indeces)
       
        print(f"\nRing-closing optimization ({len(self.system.rc_critical_rotatable_indeces)} RC critical torsions - {len(non_intersecting_rcp_paths)} intersecting RCP paths)...")

        rc_opt_time = time.time()
        if len(non_intersecting_rcp_paths) < 2:
            ring_closed_zmatrix, final_score, info = self.system.maximize_ring_closure_in_torsional_space_fabrik(
                zmatrix=self.system.zmatrix,
                rotatable_indices=self.system.rc_critical_rotatable_indeces,
                max_iterations=50,
                ring_closure_tolerance=ring_closure_tolerance,
                ring_closure_decay_rate=ring_closure_decay_rate,
                trajectory_file=trajectory_file_rc_opt,
                verbose=verbose)
        else:
            ring_closed_zmatrix, final_score, info = self.system.maximize_ring_closure_in_torsional_space(
                zmatrix=self.system.zmatrix,
                rotatable_indices=self.system.rc_critical_rotatable_indeces,
                max_iterations=500,
                ring_closure_tolerance=ring_closure_tolerance,
                ring_closure_decay_rate=ring_closure_decay_rate,
                trajectory_file=trajectory_file_rc_opt,
                verbose=verbose)
        rc_opt_time = time.time() - rc_opt_time
        print(f"  Time: {rc_opt_time:.2f} seconds")

        best_coords = zmatrix_to_cartesian(ring_closed_zmatrix)
        best_zmatrix = ring_closed_zmatrix
        best_energy = self.system.evaluate_energy(best_coords)
        # Since the calculation of the score in the previous step may change, calcualte the score independently from diff evo settings
        best_rc_score = self.system.ring_closure_score_exponential(best_coords)

        print(f"  Ring closure score change = {best_rc_score - initial_ring_closure_score:.4f} (from {initial_ring_closure_score:.4f} to {best_rc_score:.4f})")

        # Select which torsions to refine: do we protect the critical torsions?
        non_rc_critical_rotatable_indices = [idx for idx in self.system.rotatable_indices if idx not in self.system.rc_critical_rotatable_indeces]

        if enable_pssrot_refinement:
            if len(non_rc_critical_rotatable_indices) == 0:
                print("\nWarning: No torsions to refine. Skipping torsional refinement.")
            else:
                print(f"\nApplying torsional refinement ({len(non_rc_critical_rotatable_indices)} torsions) with potential energy smoothing...")

                current_zmatrix = best_zmatrix
                current_energy = best_energy
                pss_ref_init = time.time()
                trajectory_file_torsional = None
                if trajectory_file:
                    trajectory_file_torsional = trajectory_file.replace('.xyz', '_torsional.xyz')
                for smoothing in smoothing_sequence:
                    self.system.setSmoothingParameter(smoothing)
                    refined_zmatrix, refined_energy, info = self.system.minimize_energy_in_torsional_space(
                        current_zmatrix,
                        non_rc_critical_rotatable_indices,
                        max_iterations=torsional_iterations,
                        trajectory_file=trajectory_file_torsional,
                        verbose=verbose
                    )
                    if info['success']:
                        current_zmatrix = refined_zmatrix
                        current_energy = refined_energy

                pss_ref_time = time.time() - pss_ref_init
                current_rc_score = self.system.ring_closure_score_exponential(zmatrix_to_cartesian(current_zmatrix))
                print(f"  Time: {pss_ref_time:.2f} seconds")
                print(f"  Energy change = {best_energy - current_energy:.2f} kcal/mol (from {best_energy:.2f} to {current_energy:.2f} kcal/mol)")
                print(f"  Ring closure score change = {current_rc_score - best_rc_score:.4f} (from {best_rc_score:.4f} to {current_rc_score:.4f})")

                best_zmatrix = current_zmatrix
                best_energy = current_energy
                best_rc_score = current_rc_score

        if enable_zmatrix_refinement:
            if len(self.system.dof_indices) == 0:
                print("\nWarning: No DOFs to refine. Skipping Z-matrix refinement.")
            else:
                print(f"\nApplying Z-matrix refinement ({len(self.system.dof_indices)} DOFs) to top candidates...")

                current_zmatrix = best_zmatrix
                current_energy = best_energy
                zms_ref_init = time.time()
                trajectory_file_zmatrix = None
                if trajectory_file:
                    trajectory_file_zmatrix = trajectory_file.replace('.xyz', '_zmat.xyz')
                self.system.setSmoothingParameter(0.0)
                refined_zmatrix, refined_energy, info = self.system.minimize_energy_in_zmatrix_space(
                    current_zmatrix,
                    dof_indices=self.system.dof_indices,
                    dof_bounds_per_type=[0.05, 7.0, 15.0],
                    max_step_length_per_type=[0.1, 1.0, 2.0],
                    max_iterations=zmatrix_iterations,
                    gradient_tolerance=gradient_tolerance,
                    trajectory_file=trajectory_file_zmatrix,
                    verbose=verbose
                )
                #TODO: deal with abnormal termination that have improved the system but did not converge successfully.
                if info['success']:
                    current_zmatrix = refined_zmatrix
                    current_energy = refined_energy
                else:
                    if (refined_energy < current_energy):
                        print(f"  Warning: Z-matrix refinement failed, but energy improved, continuing with previous structure")
                        current_zmatrix = refined_zmatrix
                        current_energy = refined_energy
                    else:
                        print(f"  Warning: Z-matrix refinement failed, and energy did not improve, continuing with previous structure")

                zms_ref_time = time.time() - zms_ref_init
                current_rc_score = self.system.ring_closure_score_exponential(zmatrix_to_cartesian(current_zmatrix))
                print(f"  Time: {zms_ref_time:.2f} seconds")
                print(f"  Energy change = {best_energy - current_energy:.2f} kcal/mol (from {best_energy:.2f} to {current_energy:.2f} kcal/mol)")
                print(f"  Ring closure score change = {current_rc_score - best_rc_score:.4f} (from {best_rc_score:.4f} to {current_rc_score:.4f})")
                
                best_zmatrix = current_zmatrix
                best_energy = current_energy
                best_rc_score = current_rc_score

        print("\nGeometric changes from initial Z-matrix:")
        rmsd_bond_lengths, rmsd_angles, rmsd_dihedrals = MolecularSystem._calculate_rmsd(initial_zmatrix, best_zmatrix)
        print(f"RMSD bond lengths: {rmsd_bond_lengths:.4f} Å")
        print(f"RMSD angles: {rmsd_angles:.4f} deg")
        print(f"RMSD dihedrals: {rmsd_dihedrals:.4f} deg")

        return {
            'initial_energy': initial_energy,
            'initial_ring_closure_score': initial_ring_closure_score,
            'final_energy': best_energy,
            'final_ring_closure_score': best_rc_score,
            'final_coords': best_coords,
            'final_zmatrix': best_zmatrix,
            'rmsd_bond_lengths': rmsd_bond_lengths,
            'rmsd_angles': rmsd_angles,
            'rmsd_dihedrals': rmsd_dihedrals,
            'success': True
        }

        
    def minimize(self, max_iterations: int = 500,
                 smoothing: Optional[Union[float, List[float]]] = None,
                 space_type: str = "Cartesian",
                 zmatrix_dof_bounds_per_type: Optional[List[Tuple[float, float, float]]] = [[10.0, 180.0, 180.0]],
                 gradient_tolerance: float = 0.01,
                 trajectory_file: Optional[str] = None,
                 verbose: bool = True) -> Dict[str, Any]:
        """
        Perform energy minimization on the current structure.
        
        This method provides a wrapper for minimizing the energy in Cartesian space, or in Z-matrix space, or in torsional space.
        It operates on a single structure (the current Z-matrix in the molecular system).
        
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
        trajectory_file : Optional[str]
            If provided, write optimization trajectory to this XYZ file (append mode).
            Writes coordinates for the torsional and Z-matrix refinements in dedicated files.
        verbose : bool
            Print minimization progress (default: True)
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'initial_energy': Energy before minimization (kcal/mol)
            - 'final_energy': Energy after minimization (kcal/mol)
            - 'initial_ring_closure_score': Ring closure score before minimization
            - 'final_ring_closure_score': Ring closure score after minimization
            - 'final_coords': Minimized Cartesian coordinates (numpy array)
            - 'final_zmatrix': Minimized Z-matrix (ZMatrix instance)
            - 'rmsd_bond_lengths': RMSD of bond lengths after minimization
            - 'rmsd_angles': RMSD of angles after minimization
            - 'rmsd_dihedrals': RMSD of dihedrals after minimization
            - 'success': True if minimization was successful, False otherwise
        
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

        #TODOdel
        write_xyz_file(initial_coords, initial_zmatrix.get_elements(), "/tmp/server_initial.xyz")
        write_zmatrix_file(initial_zmatrix, "/tmp/server_initial.int")

        
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
        initial_ring_closure_score = self.system.ring_closure_score_exponential(
                initial_coords,
                verbose=False
            )
        
        if verbose:
            print(f"Initial energy: {initial_energy:.4f} kcal/mol")
            print(f"Initial ring closure score: {initial_ring_closure_score:.4f}")
            if len(smoothing_values) > 1:
                print(f"Minimizing in {space_type} space with smoothing sequence: {smoothing_values}")
            else:
                print(f"Minimizing in {space_type} space with {max_iterations} max iterations (smoothing={smoothing_values[0]:.2f})...")
        
        current_zmatrix = initial_zmatrix
        current_energy = initial_energy
        all_opt_info = []
        
        # Perform sequence of minimizations with smoothed potential energy
        for step, smoothing_val in enumerate(smoothing_values):
            self.system.setSmoothingParameter(smoothing_val)
            
            if verbose and len(smoothing_values) > 1:
                print(f"Step {step+1}/{len(smoothing_values)}: smoothing={smoothing_val:.2f}")
            
            opt_info = {'success': False, 'message': 'Minimization step failed'}
            if space_type == 'torsional':
                # Perform torsional minimization
                refined_zmatrix, step_energy, opt_info = self.system.minimize_energy_in_torsional_space(
                    current_zmatrix,
                    self.system.rotatable_indices,
                    max_iterations=max_iterations,
                    trajectory_file=trajectory_file,
                    verbose=verbose
                )
                
                if opt_info.get('success'):
                    current_zmatrix = refined_zmatrix
                    current_energy = step_energy
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
                    trajectory_file=trajectory_file,
                    verbose=verbose
                )
                time_taken = time.time() - init_time
                print(f"\nTime taken for Z-matrix minimization: {time_taken:.2f} seconds")

                if opt_info.get('success'):
                    current_zmatrix = refined_zmatrix
                    current_energy = step_energy
                else:
                    if verbose:
                        print(f"  Warning: Minimization step failed, continuing with previous structure")
                    
            elif space_type == 'Cartesian':
                # Work in Cartesian space, even though results are in Z-matrix space
                try:
                    minimized_coords, step_energy = self.system.minimize_energy(
                        current_zmatrix,
                        max_iterations=max_iterations
                    )
                    
                    # Extract refined Z-matrix from minimized coordinates
                    refined_zmatrix = cartesian_to_zmatrix(
                        minimized_coords,
                        current_zmatrix
                    )
                    current_zmatrix = refined_zmatrix
                    current_energy = step_energy
                    opt_info = {'success': True}
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Minimization step failed: {e}, continuing with previous structure")
            else:
                raise ValueError(f"Invalid space type: {space_type}")
        
            all_opt_info.append(opt_info)   

        # Final state after all smoothing steps
        minimized_zmatrix = current_zmatrix
        minimized_energy = current_energy
        minimized_coords = zmatrix_to_cartesian(minimized_zmatrix)
        
        # Calculate ring closure score
        final_ring_closure_score = self.system.ring_closure_score_exponential(
            minimized_coords,
            verbose=False
        )
        
        #TODOdel
        write_xyz_file(minimized_coords, minimized_zmatrix.get_elements(), "/tmp/server_final.xyz")

        self.system.zmatrix = minimized_zmatrix

        rmsd_bond_lengths, rmsd_angles, rmsd_dihedrals = MolecularSystem._calculate_rmsd(initial_zmatrix, minimized_zmatrix)
        
        if verbose:
            print(f"\nFinal energy:   {minimized_energy:.4f} kcal/mol")
            print(f"Improvement:    {initial_energy - minimized_energy:.4f} kcal/mol")
            print(f"Ring closure:   {final_ring_closure_score:.4f}")
            print(f"RMSD bond lengths: {rmsd_bond_lengths:.4f} Å")
            print(f"RMSD angles: {rmsd_angles:.4f} deg")
            print(f"RMSD dihedrals: {rmsd_dihedrals:.4f} deg")

        return {
            'initial_energy': initial_energy,
            'initial_ring_closure_score': initial_ring_closure_score,
            'final_energy': minimized_energy,
            'final_ring_closure_score': final_ring_closure_score,
            'final_coords': minimized_coords,
            'final_zmatrix': minimized_zmatrix,
            'rmsd_bond_lengths': rmsd_bond_lengths,
            'rmsd_angles': rmsd_angles,
            'rmsd_dihedrals': rmsd_dihedrals,
            'success': all_opt_info[-1].get('success', False)
        }
