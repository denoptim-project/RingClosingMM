#!/usr/bin/env python3
"""
Molecular System Management

This module provides the MolecularSystem class for creating and managing OpenMM systems and
molecular manipulation (e.g., identification of rotatable bonds and related internal coordinates) and molecualr modeling, including energy evaluation and minimization.

Classes:
    MolecularSystem: Manages OpenMM system, topology, and provides energy evaluation methods
"""

import traceback
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
import copy

import openmm.unit as unit
from openmm.app import Element, Simulation, Topology
from ringclosingmm.AnalyticalDistance import AnalyticalDistanceFactory

# Package imports
from .IOTools import read_int_file, read_sdf_file, write_xyz_file
from .ZMatrix import ZMatrix
from .RingClosingForceField import (
    create_simulation_from_system,
    create_system,
    setGlobalParameterToAllForces
)
from .CoordinateConversion import (
    zmatrix_to_cartesian,
    _calc_distance
)

from scipy.optimize import OptimizeResult, differential_evolution, minimize


def build_topology_from_data(atoms_data: List[Tuple[str, int]], 
                             bonds_data: List[Tuple[int, int]]) -> Topology:
    """
    Build OpenMM Topology from parsed atom and bond data.
    
    Parameters
    ----------
    atoms_data : list of tuples
        List of (element_symbol, atom_index) tuples
    bonds_data : list of tuples
        List of (atom1_idx, atom2_idx) tuples
    
    Returns
    -------
    topo : openmm.app.Topology
        The topology object with atoms and bonds
    """
    # Create the base Topology
    topo = Topology()
    c0 = topo.addChain()
    r0 = topo.addResidue('res0', c0)
    atoms = []
    
    # Add atoms to topology
    for element, idx in atoms_data:
        try:
            # Try to get the element (handles case-insensitive lookup)
            elem = Element.getBySymbol(element)
            atom = topo.addAtom(element + str(idx), elem, r0)
        except KeyError:
            # Handle pseudo-atoms (ATP, ATM, ATN, etc.)
            atom = topo.addAtom("_" + element + "_" + str(idx), Element.getBySymbol('H'), r0)
        atoms.append(atom)
    
    # Add bonds to topology
    for atom1_idx, atom2_idx in bonds_data:
        topo.addBond(atoms[atom1_idx], atoms[atom2_idx])
    
    return topo
    

# =============================================================================
# Molecular System Management
# =============================================================================

class MolecularSystem:
    """
    Manages OpenMM system and molecular structure.
    
    This class encapsulates the OpenMM system (force field definition) and provides
    methods for energy evaluation and coordinate manipulation. Simulation objects
    are cached per smoothing parameter for efficiency, avoiding expensive recreation
    on every energy evaluation.
    
    Attributes
    ----------
    system : openmm.System
        OpenMM system object (force field definition)
    topology : openmm.app.Topology
        OpenMM topology object
    rcpterms : List[Tuple[int, int]]
        Ring closure pair terms
    zmatrix : ZMatrix
        Z-matrix (internal coordinates) representation
    step_length : float
        Integration step length for simulations
    """
    
    def __init__(self, system, topology, rcpterms: List[Tuple[int, int]], 
                 zmatrix: ZMatrix,
                 step_length: float = 0.0002,
                 ring_closure_threshold: float = 1.5):
        """
        Initialize molecular system.
        
        Parameters
        ----------
        system : openmm.System
            OpenMM system object
        topology : openmm.app.Topology
            OpenMM topology object
        rcpterms : List[Tuple[int, int]]
            Ring closure pair terms
        zmatrix : ZMatrix
            Z-matrix representation
        step_length : float
            Integration step length
        ring_closure_threshold : float
            Distance threshold (Angstroms) for considering a ring nearly closed
        """
        self.system = system
        self.topology = topology
        self.rcpterms = rcpterms if rcpterms else []
        self.zmatrix = zmatrix
        self.step_length = step_length
        self.ring_closure_threshold = ring_closure_threshold
        
        # Rotatable indices and related fields (computed when rotatable_indices is set)
        self.rotatable_indices: Optional[List[int]] = None
        self.rc_critical_atoms: Optional[List[int]] = None
        self.rc_critical_rotatable_indeces: Optional[List[int]] = None

        # Degrees of freedom indices are computed automatically when rotatable_indices is set, but can be
        # set by a specific call to set_dof_indices
        self.dof_indices: Optional[List[Tuple[int, int]]] = None
        
        # Cache simulations by smoothing parameter to avoid recreating them
        self._simulation_cache = {}
        self._current_smoothing = 0.0
        
        # RCP path data and groups (computed on demand, cached)
        self._rcp_path_data: Optional[Dict[Tuple[int, int], Tuple[List[int], List[int]]]] = None
        self._rcp_paths: Optional[List[Tuple[List[Tuple[int, int]], List[int], List[int]]]] = None
        self._rcp_path_data_zmatrix_hash: Optional[int] = None  # Hash to detect zmatrix changes
    
    @property
    def elements(self) -> List[str]:
        """Get element symbols from Z-matrix."""
        return self.zmatrix.get_elements()
    
    def set_rotatable_indices(self, rotatable_indices: List[int]):
        """
        Set rotatable indices and compute dependent fields.
        
        This method sets the rotatable_indices and automatically computes:
        - rc_critical_rotatable_indeces: Critical rotatable indices on RCP paths
        - dof_indices: Degrees of freedom indices derived from rotatable indices
        
        Parameters
        ----------
        rotatable_indices : List[int]
            Indices of rotatable atoms in Z-matrix (0-based)
        """
        self.rotatable_indices = rotatable_indices
        
        # Compute critical rotatable indices
        if self.rotatable_indices:
            self.rc_critical_rotatable_indeces, self.rc_critical_atoms = self._identify_rc_critical_rotatable_indeces(
                self.zmatrix, self.rcpterms, self.rotatable_indices, self.topology)
        else:
            self.rc_critical_rotatable_indeces = []
            self.rc_critical_atoms = []
        
        # Compute DOF indices
        if self.rotatable_indices:
            self.dof_indices = self._get_dofs_from_rotatable_indeces(
                self.rotatable_indices,
                self.rc_critical_rotatable_indeces,
                self.rc_critical_atoms,
                self.zmatrix
            )
        else:
            self.dof_indices = []
        
        # Invalidate RCP path cache when rotatable indices change
        self._rcp_path_data = None
        self._rcp_paths = None
        self._rcp_path_data_zmatrix_hash = None

    def set_dof_indices(self, dof_indices: List[Tuple[int, int]]):
        """
        Set degrees of freedom indices.
        
        Parameters
        ----------
        dof_indices : List[Tuple[int, int]]
            Degrees of freedom indices (0-based). The first index is the atom index, 
            the second index is the degree of freedom index. Example: [(0, 0), (1, 2)] means 
            the first atom's first degree of freedom (distance) and the second atom's third 
            degree of freedom (torsion).
        """
        self.dof_indices = dof_indices

    @classmethod
    def from_file(cls, structure_file: str, forcefield_file: str,
                  rcp_terms: Optional[List[Tuple[int, int]]] = None,
                  ring_closure_threshold: float = 1.5,
                  step_length: float = 0.0002) -> 'MolecularSystem':
        """
        Create molecular system from structure file.
        
        The structure file uses 1-based indexing. All Z-matrix reference indices
        are automatically converted to 0-based (internal representation).
        
        Parameters
        ----------
        structure_file : str
            Path to structure file (.int)
        forcefield_file : str
            Path to force field XML file
        rcp_terms : Optional[List[Tuple[int, int]]]
            RCP terms (optional). All indices must be in 0-based indexing.
        ring_closure_threshold : float
            Distance threshold (Angstroms) for considering a ring nearly closed
        step_length : float
            Integration step length
        
        Returns
        -------
        MolecularSystem
            Initialized molecular system with Z-matrix in 0-based indexing
        """
        # Read structure data
        if structure_file.endswith('.int'):
            zmatrix = read_int_file(structure_file)
        elif structure_file.endswith('.sdf'):
            zmatrix = read_sdf_file(structure_file)
        else:
            raise ValueError("Only .int and .sdf files are supported as input files.")
        
        if not isinstance(zmatrix, ZMatrix):
            raise TypeError(f"Expected ZMatrix instance, got {type(zmatrix)}")
        
        return cls.from_data(zmatrix, forcefield_file, rcp_terms, ring_closure_threshold, step_length)
    
    @classmethod
    def from_data(cls, zmatrix: ZMatrix, 
                     forcefield_file: str,
                     rcp_terms: Optional[List[Tuple[int, int]]] = None,
                     ring_closure_threshold: float = 1.5,
                     step_length: float = 0.0002) -> 'MolecularSystem':
        """
        Create molecular system from raw data.
        
        Parameters
        ----------
        zmatrix : ZMatrix
            Z-matrix representation. All reference indices (bond_ref, angle_ref, 
            dihedral_ref) must be in 0-based indexing.
        forcefield_file : str
            Path to force field XML file
        rcp_terms : Optional[List[Tuple[int, int]]]
            RCP terms (optional). All indices must be in 0-based indexing.
        ring_closure_threshold : float
            Distance threshold (Angstroms) for considering a ring nearly closed
        step_length : float
            Integration step length
            
        Returns
        -------
        MolecularSystem
            Initialized molecular system
        """
        # Convert Z-matrix to Cartesian and convert to nm
        coords = zmatrix_to_cartesian(zmatrix) * 0.1 * unit.nanometer  # Angstroms to nm
        
        # Build topology from Z-matrix data
        atoms_data = [(zmatrix[i][ZMatrix.FIELD_ELEMENT], i) for i in range(len(zmatrix))]

        bonds_data = zmatrix.bonds

        # Build topology from data
        topology = build_topology_from_data(atoms_data, bonds_data)
        
        # Create OpenMM system (force field definition)
        system = create_system(
            topology,
            rcp_terms if rcp_terms else [],
            forcefield_file,
            positions=coords,
            smoothing=0.0,
            scalingNonBonded=1.0,
            scalingRCP=1.0
        )
        
        return cls(system, topology, rcp_terms or [], zmatrix, step_length, ring_closure_threshold)


    def setSmoothingParameter(self, smoothing: float):
        """
        Set smoothing parameter for all forces.
        
        Parameters
        ----------
        smoothing : float
            Smoothing parameter
        """
        self._current_smoothing = smoothing
        setGlobalParameterToAllForces(self.system, 'smoothing', smoothing)
    
    
    def _get_or_create_simulation(self, positions) -> Simulation:
        """
        Get cached simulation for current smoothing parameter or create a new one.
        
        Parameters
        ----------
        positions : positions with units
            Initial positions for the simulation
        
        Returns
        -------
        Simulation
            OpenMM simulation object (cached or newly created)
        """
        smoothing_key = self._current_smoothing
        
        if smoothing_key not in self._simulation_cache:
            # Create new simulation for this smoothing parameter
            simulation = create_simulation_from_system(self.topology, self.system, positions, self.step_length)
            self._simulation_cache[smoothing_key] = simulation
        else:
            # Use cached simulation and update positions
            simulation = self._simulation_cache[smoothing_key]
            simulation.context.setPositions(positions)
        
        return simulation
        
    
    def evaluate_energy(self, coords: np.ndarray) -> float:
        """
        Evaluate energy for given Cartesian coordinates.
        
        Uses a cached Simulation for the current smoothing parameter (creating it
        if necessary on first use), then updates positions and evaluates energy.
        
        Parameters
        ----------
        coords : np.ndarray
            Cartesian coordinates to evaluate (Angstroms)

        Returns
        -------
        float
            Energy in kcal/mol
        """
        try:
            # Convert to OpenMM units (nm)
            positions = coords * 0.1 * unit.nanometer
            
            # Get or create cached simulation and update positions
            simulation = self._get_or_create_simulation(positions)
            
            # Evaluate energy
            state = simulation.context.getState(getEnergy=True)
            energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
            
            return energy
        except Exception as e:
            print(f"Error evaluating energy: {e}")
            # Return high penalty for invalid geometries
            return 1e6


    def minimize_energy(self, zmatrix: ZMatrix, max_iterations: int = 100) -> Tuple[np.ndarray, float]:
        """
        Perform energy minimization with OpenMM engine in Cartesian space and return optimized coordinates.
        
        Uses a cached Simulation for the current smoothing parameter (creating it
        if necessary on first use), then updates positions and performs minimization.
        
        Parameters
        ----------
        zmatrix : ZMatrix
            Starting Z-matrix
        max_iterations : int
            Maximum minimization iterations
            
        Returns
        -------
        Tuple[np.ndarray, float]
            Minimized Cartesian coordinates (Angstroms) and energy (kcal/mol)
        """
        try:
            # Convert to Cartesian and set positions
            coords = zmatrix_to_cartesian(zmatrix)
            positions = coords * 0.1 * unit.nanometer
            
            # Get or create cached simulation and update positions
            simulation = self._get_or_create_simulation(positions)
            
            # Minimize
            simulation.minimizeEnergy(maxIterations=max_iterations, tolerance=0.1)
            
            # Get minimized state
            state = simulation.context.getState(getPositions=True, getEnergy=True)
            minimized_positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
            minimized_energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
            
            return minimized_positions, minimized_energy
        except Exception:
            # Return original with high penalty
            coords = zmatrix_to_cartesian(zmatrix)
            return coords, 1e6


    def minimize_energy_in_zmatrix_space(self, zmatrix: ZMatrix, 
                                            dof_indices: List[Tuple[int,int]],
                                            dof_bounds_per_type: Optional[List[Tuple[float, float, float]]] = [[10.0, 180.0, 180.0]],
                                            max_iterations: int = 100,
                                            gradient_tolerance: float = 0.01,
                                            verbose: bool = False,
                                            trajectory_file: Optional[str] = None) -> Tuple[ZMatrix, float, Dict[str, Any]]:
        """
        Perform energy minimization in Z-matrix space only.
        
        Optimizes only the degrees of freedom (dof)while keeping the other degrees of freedom fixed.
        Uses scipy.optimize.minimize with numerical gradients.
        
        Parameters
        ----------
        zmatrix : ZMatrix
            Starting Z-matrix
        dof_indices : List[int]
            Indices of degrees of freedom in Z-matrix. The first index is the atom index, 
            the second index is the degree of freedom index. Example: [(0, 0), (1, 2)] means 
            the first atom's first degree of freedom (distance) and the second atom's third 
            degree of freedom (torsion).
        dof_bounds_per_type : List[Tuple[float, float, float]]
            Bounds for types of degrees of freedom. Example: [(0.1, 10.0, 20.0)] means 
            the distances can be changed by up to 0.1, angles can be changed by up to 10.0, and torsions can be changed by up to 20.0. Multiple tuples can be provided to request any stepwise application of bounds. Example: [(0.1, 10.0, 20.0), (0.01, 1.0, 2.0)] means will make the minimization run with [0.1, 10.0, 20.0] for the first step and [0.01, 1.0, 2.0] for the second step.
            Default is [(10.0, 180.0, 180.0)].
        max_iterations : int
            Maximum minimization iterations
        gradient_tolerance : float
            Gradient tolerance for minimization
        verbose : bool
            Print minimization progress
        trajectory_file : Optional[str]
            If provided, write optimization trajectory to this XYZ file (append mode).
            Writes coordinates after each optimizer iteration (not every function evaluation).
            
        Returns
        -------
            Tuple[ZMatrix, float, Dict[str, Any]]
            Optimized Z-matrix, optimized energy (kcal/mol), and optimization info dict
        """

        # Step size for numerical gradient calculation
        eps = 0.1

        # Use DOF names from ZMatrix class
        dof_names = ZMatrix.DOF_NAMES

        # Record the initial values of the dofs
        initial_dofs = np.array([zmatrix.get_dof(idx[0], idx[1]) for idx in dof_indices])
        
        # Clear trajectory file if provided
        if trajectory_file:
            Path(trajectory_file).unlink(missing_ok=True)
        
        # Iteration counter for trajectory
        iteration_counter = [0]

        def get_updated_zmatrix(zmatrix: ZMatrix, dofs: np.ndarray) -> ZMatrix:
            """Get updated Z-matrix with given dofs."""
            updated_zmatrix = zmatrix.copy()
            for j, idx in enumerate(dof_indices):
                zmat_idx = idx[0]
                dof_idx = idx[1]
                updated_zmatrix.update_dof(zmat_idx, dof_idx, dofs[j])
            return updated_zmatrix
        
        def objective(dofs: np.ndarray) -> float:
            """Objective function: evaluate energy for given dofs."""
            try:
                # Update zmatrix with new dofs
                updated_zmatrix = get_updated_zmatrix(zmatrix, dofs)
                
                # Convert to Cartesian and evaluate energy
                coords = zmatrix_to_cartesian(updated_zmatrix)
                energy = self.evaluate_energy(coords)

                return energy
            except Exception as e:
                print(f"  Warning: Failed to evaluate energy at iteration {iteration_counter[0]}: {e}")
                return 1e6
        
        def callback(dofs: np.ndarray):
            """Callback function called after each scipy.optimize.minimize iteration."""
            if trajectory_file:
                try:
                    # Update zmatrix with current dofs values
                    current_zmatrix = get_updated_zmatrix(zmatrix, dofs)
                    
                    # Convert to Cartesian
                    coords = zmatrix_to_cartesian(current_zmatrix)
                    elements = current_zmatrix.get_elements()
                    
                    # Evaluate energy for comment
                    energy = objective(dofs)
                    
                    # Write to trajectory (append mode)
                    print(f"Iteration {iteration_counter[0]}, E={energy:.2f} kcal/mol")
                    write_xyz_file(
                        coords, 
                        elements, 
                        trajectory_file,
                        comment=f"Iteration {iteration_counter[0]}, E={energy:.2f} kcal/mol",
                        append=True
                    )
                    iteration_counter[0] += 1
                except Exception as e:
                    # Don't fail minimization if trajectory writing fails
                    if verbose:
                        print(f"  Warning: Failed to write trajectory at iteration {iteration_counter[0]}: {e}")
        
        try:           
            if dof_bounds_per_type is None:
                dof_bound_coeffs_sequence = [[10.0, 180.0, 180.0]]
            else:
                # Check if dof_bounds is already a sequence of coefficient sets
                # by checking if the first element is itself a sequence (list/tuple)
                if len(dof_bounds_per_type) > 0 and isinstance(dof_bounds_per_type[0], (list, tuple)):
                    # Already a sequence of coefficient sets
                    dof_bound_coeffs_sequence = dof_bounds_per_type
                else:
                    # Single coefficient set, wrap it in a list
                    dof_bound_coeffs_sequence = [dof_bounds_per_type]

            initial_energy = objective(initial_dofs)

            current_dofs = initial_dofs.copy()
            current_energy = initial_energy
            current_gradient = None
            current_gradient_norm_check = None
            current_max_gradient_check = None

            for i in range(len(dof_bound_coeffs_sequence)):
                dof_bound_coeffs = dof_bound_coeffs_sequence[i]
                current_dof_bounds = []
                for j,idx in enumerate(dof_indices):
                    zmat_idx = idx[0]
                    dof_idx = idx[1]
                    dof_current_value = current_dofs[j]
                    dof_type = dof_names[dof_idx]
                    if dof_type == 'bond_length':
                        deviation = dof_bound_coeffs[0]
                        current_dof_bounds.append((max(dof_current_value-deviation, eps+0.005), dof_current_value+deviation))
                    elif (dof_type == 'angle') or (dof_type == 'dihedral' and zmatrix[zmat_idx].get('chirality', 0) != 0):
                        deviation = dof_bound_coeffs[1]
                        if deviation < 179.0:
                            current_dof_bounds.append((dof_current_value-deviation, dof_current_value+deviation))
                        else:
                            current_dof_bounds.append((-180.0, 180.0))
                    elif dof_type == 'dihedral' and zmatrix[zmat_idx].get('chirality', 0) == 0:
                        deviation = dof_bound_coeffs[2]
                        if deviation < 179.0:
                            current_dof_bounds.append((dof_current_value-deviation, dof_current_value+deviation))
                        else:
                            current_dof_bounds.append((-180.0, 180.0))
                    else:
                        raise ValueError(f"Unknown degree of freedom type: {dof_type}")

                if verbose and i == 0:
                    print(f"\n  Initial DOFs:")
                    self.report_dof_info(zmatrix, dof_indices, current_dofs, current_dof_bounds, current_gradient) 

                result = minimize(
                    objective,
                    current_dofs,
                    method='L-BFGS-B',
                    # This triggers calcualtion of numerica lgradient, which is controlled by the 'eps' option below
                    jac=None,  
                    bounds=current_dof_bounds,
                    callback=callback if trajectory_file else None,
                    options={
                        'maxiter': max_iterations,
                        'ftol': 0,  # 0 disables the function tolerance
                        'gtol': gradient_tolerance, 
                        'disp': True,
                        'eps': eps  # step size for numerical gradient calculation
                    }
                )
                
                # Update step info
                current_dofs = result.x.copy()
                current_energy = objective(result.x)
                current_gradient = result.jac
                current_gradient_norm_check = np.linalg.norm(current_gradient)
                current_max_gradient_check = np.max(np.abs(current_gradient))
                
                if verbose:
                    print(f"\n  Iteration {i+1}: nfev={result.nfev} nit={result.nit} E={current_energy:.4f} kcal/mol Gnorm={current_gradient_norm_check:.4f} Gmax={current_max_gradient_check:.4f} message={result.message}")
                    self.report_dof_info(zmatrix, dof_indices, current_dofs, current_dof_bounds, current_gradient)
                
            # Create optimized zmatrix
            optimized_zmatrix = get_updated_zmatrix(zmatrix, result.x)
            
            # Prepare info dict
            info = {
                'success': result.success,
                'message': result.message,
                'nfev': result.nfev,
                'nit': result.nit,
                'initial_energy': initial_energy,
                'final_energy': current_energy,
                'improvement': initial_energy - current_energy,
                'initial_dofs': initial_dofs.copy(),
                'final_dofs': current_dofs.copy()
            }
            
            return optimized_zmatrix, current_energy, info
            
        except Exception as e:
            if verbose:
                print(f"  ZMatrix minimization failed: {e}")
            # Return original zmatrix with high penalty
            info = {
                'success': False,
                'message': str(e),
                'nfev': 0,
                'nit': 0,
                'initial_energy': initial_energy if 'initial_energy' in locals() else 1e6,
                'final_energy': 1e6,
                'improvement': 0.0,
                'initial_dofs': initial_dofs.copy() if 'initial_dofs' in locals() else np.array([]),
                'final_dofs': initial_dofs.copy() if 'initial_dofs' in locals() else np.array([])
            }
            return zmatrix.copy(), 1e6, info


    def report_dof_info(self, zmatrix: ZMatrix, dof_indices: List[Tuple[int,int]], dofs: np.ndarray, dof_bounds: List[Tuple[float, float]], gradient: Optional[np.ndarray]):
        """Report information about the degrees of freedom on screen."""
        # Format gradient norm and max gradient with NA handling    
        dof_names = ZMatrix.DOF_NAMES
        
        for j, dof_index in enumerate(dof_indices):
            zmat_idx, dof_type = dof_index
            atom = zmatrix[zmat_idx]
            atom_id = atom[ZMatrix.FIELD_ID] + 1
            
            # Build atom reference list based on DOF type
            ref_atoms = [atom.get(ZMatrix.FIELD_BOND_REF, -1) + 1]
            if dof_type >= 1:
                ref_atoms.append(atom.get(ZMatrix.FIELD_ANGLE_REF, -1) + 1)
            if dof_type == 2:
                ref_atoms.append(atom.get(ZMatrix.FIELD_DIHEDRAL_REF, -1) + 1)
            
            # Format atom references
            ref_str = ' '.join(f"{ref:3d}" for ref in ref_atoms)
            
            # Format with fixed column widths
            dof_name = dof_names[dof_type]
            dof_value = dofs[j]
            bound_low, bound_high = dof_bounds[j]
            
            # Format gradient value with NA handling (reserve space for sign)
            if gradient is not None and j < len(gradient):
                grad_value_str = f"{gradient[j]: 11.6f}"
            else:
                grad_value_str = "         NA"
            
            print(f"  DOF[{j:2d}] {dof_name:12s} [{atom_id:3d} {ref_str:>12s}]"
                      f"={dof_value: 9.4f}  bounds=[{bound_low: 9.4f}, {bound_high: 9.4f}]  grad={grad_value_str}")
            

    def minimize_energy_in_torsional_space(self, zmatrix: ZMatrix, 
                                            rotatable_indices: List[int],
                                            max_iterations: int = 100,
                                            method: str = 'Powell', 
                                            verbose: bool = False,
                                            trajectory_file: Optional[str] = None) -> Tuple[ZMatrix, float, Dict[str, Any]]:
        """
        Perform energy minimization in torsional space only.
        
        Optimizes only the dihedral angles while keeping bond lengths and angles fixed.
        Uses scipy.optimize.minimize with numerical gradients.
        
        Parameters
        ----------
        zmatrix : ZMatrix
            Starting Z-matrix
        rotatable_indices : List[int]
            Indices of rotatable atoms in Z-matrix (atoms whose dihedrals can be optimized)
        max_iterations : int
            Maximum optimization iterations
        method : str
            Scipy optimization method (default: 'Powell')
        verbose : bool
            Print optimization progress
        trajectory_file : Optional[str]
            If provided, write optimization trajectory to this XYZ file (append mode).
            Writes coordinates after each optimizer iteration (not every function evaluation).
            
        Returns
        -------
            Tuple[ZMatrix, float, Dict[str, Any]]
            Optimized Z-matrix, optimized energy (kcal/mol), and optimization info dict
        """
        # Extract initial torsion angles
        initial_torsions = np.array([zmatrix[idx][ZMatrix.FIELD_DIHEDRAL] for idx in rotatable_indices])
        
        # Clear trajectory file if provided
        if trajectory_file:
            Path(trajectory_file).unlink(missing_ok=True)
        
        # Iteration counter for trajectory
        iteration_counter = [0]
        
        def objective(torsions: np.ndarray) -> float:
            """Objective function: evaluate energy for given torsions."""
            try:
                # Update zmatrix with new torsion values
                updated_zmatrix = zmatrix.copy()
                for j, idx in enumerate(rotatable_indices):
                    updated_zmatrix[idx][ZMatrix.FIELD_DIHEDRAL] = torsions[j]
                
                # Convert to Cartesian and evaluate energy
                coords = zmatrix_to_cartesian(updated_zmatrix)
                energy = self.evaluate_energy(coords)
                
                return energy
            except Exception as e:
                # Return high penalty on failure
                return 1e6
        
        def callback(xk: np.ndarray):
            """Callback function called after each optimizer iteration."""
            if trajectory_file:
                try:
                    # Update zmatrix with current torsion values
                    current_zmatrix = zmatrix.copy()
                    for j, idx in enumerate(rotatable_indices):
                        current_zmatrix[idx][ZMatrix.FIELD_DIHEDRAL] = xk[j]
                    
                    # Convert to Cartesian
                    coords = zmatrix_to_cartesian(current_zmatrix)
                    elements = current_zmatrix.get_elements()
                    
                    # Evaluate energy for comment
                    energy = objective(xk)
                    
                    # Write to trajectory (append mode)
                    print(f"Iteration {iteration_counter[0]}, E={energy:.2f} kcal/mol")
                    write_xyz_file(
                        coords, 
                        elements, 
                        trajectory_file,
                        comment=f"Iteration {iteration_counter[0]}, E={energy:.2f} kcal/mol",
                        append=True
                    )
                    iteration_counter[0] += 1
                except Exception as e:
                    # Don't fail optimization if trajectory writing fails
                    if verbose:
                        print(f"  Warning: Failed to write trajectory at iteration {iteration_counter[0]}: {e}")
        
        try:
            # Initial energy
            initial_energy = objective(initial_torsions)
            
            if verbose:
                print(f"  Initial torsions: {initial_torsions}")
                print(f"  Initial energy: {initial_energy:.4f} kcal/mol")
                if trajectory_file:
                    print(f"  Writing trajectory to: {trajectory_file}")
            
            # Bounds: allow torsions to vary within [-180, 180]
            bounds = [(-180.0, 180.0) for _ in range(len(initial_torsions))]
            
            # Optimize torsions
            result = minimize(
                objective,
                initial_torsions,
                method=method,
                bounds=bounds,
                callback=callback if trajectory_file else None,
                options={
                    'maxiter': max_iterations,
                    'ftol': 0.05,    # Function tolerance (kcal/mol)
                    'xtol': 0.1,    # Tolerance for changes in the variables (degrees)
                    'disp': False
                }
            )
            
            if verbose:
                print(f"  Optimization: nfev={result.nfev}, nit={result.nit}, "
                      f"success={result.success}")
                print(f"  Final energy: {result.fun:.4f} kcal/mol")
                print(f"  Improvement: {initial_energy - result.fun:.4f} kcal/mol")
            
            # Create optimized zmatrix
            optimized_zmatrix = zmatrix.copy()
            for j, idx in enumerate(rotatable_indices):
                optimized_zmatrix[idx][ZMatrix.FIELD_DIHEDRAL] = result.x[j]
            
            # Prepare info dict
            info = {
                'success': result.success,
                'message': result.message,
                'nfev': result.nfev,
                'nit': result.nit,
                'initial_energy': initial_energy,
                'final_energy': result.fun,
                'improvement': initial_energy - result.fun,
                'initial_torsions': initial_torsions.copy(),
                'final_torsions': result.x.copy()
            }
            
            return optimized_zmatrix, result.fun, info
            
        except Exception as e:
            print(e)
            if verbose:
                print(f"  Torsional minimization failed: {e}")
            # Return original zmatrix with high penalty
            info = {
                'success': False,
                'message': str(e),
                'nfev': 0,
                'nit': 0,
                'initial_energy': initial_energy if 'initial_energy' in locals() else 1e6,
                'final_energy': 1e6,
                'improvement': 0.0,
                'initial_torsions': initial_torsions.copy() if 'initial_torsions' in locals() else np.array([]),
                'final_torsions': initial_torsions.copy() if 'initial_torsions' in locals() else np.array([])
            }
            return zmatrix.copy(), 1e6, info


    def maximize_ring_closure_in_torsional_space(self, zmatrix: ZMatrix, 
                                                  rotatable_indices: List[int],
                                                  max_iterations: int = 100,
                                                  ring_closure_tolerance: float = 0.2,
                                                  ring_closure_decay_rate: float = 0.5,
                                                  trajectory_file: Optional[str] = None,
                                                  verbose: bool = False) -> Tuple[ZMatrix, float, Dict[str, Any]]:
        """
        Maximize ring closure in torsional space by dual annealing.
        
        Parameters
        ----------
        zmatrix : ZMatrix
            Starting Z-matrix
        rotatable_indices : List[int]
            Indices of rotatable atoms in Z-matrix (atoms whose dihedrals can be optimized)
        max_iterations : int
            Maximum optimization iterations
        ring_closure_tolerance : float
            Ring closure tolerance (Å) for computing ring closure score
        ring_closure_decay_rate : float
            Ring closure decay rate for computing ring closure score
        trajectory_file : Optional[str]
            If provided, write optimization trajectory to this XYZ file (append mode).
            Writes coordinates after each optimizer iteration (not every function evaluation).
        verbose : bool
            Print optimization progress
        """

        # We work on a reduced version of the ZMatrix, only containing the rotatable indices
        reduced_zmatrix = zmatrix.copy()
        #TOD

        # Clear trajectory file if provided
        if trajectory_file:
            Path(trajectory_file).unlink(missing_ok=True)

        # Iteration counter for trajectory
        iteration_counter = [0]

        popsize = min(15 * len(rotatable_indices), 100)  # Cap at 100
        
        def objective(torsions: np.ndarray) -> float:
            """Objective function: evaluate ring closure score for given torsions."""
            try:
                # Update zmatrix with new torsion values
                updated_zmatrix = zmatrix.copy()
                for j, idx in enumerate(rotatable_indices):
                    updated_zmatrix[idx][ZMatrix.FIELD_DIHEDRAL] = torsions[j]
                coords = zmatrix_to_cartesian(updated_zmatrix)
                # Negated because we want to maximize ring closure score
                return -self.ring_closure_score_exponential(coords, ring_closure_tolerance, ring_closure_decay_rate)
            except Exception as e:
                print(f'Error in objective function: {e.message if hasattr(e, "message") else str(e)}')
                return 0.0

        def callback(intermediate_result: OptimizeResult):
            """Callback function called after each optimizer iteration to verify reaching of theoretical best solution (i.e., f(x) = -1.0)."""
            if trajectory_file:
                try:
                    # Update zmatrix with current torsion values
                    current_zmatrix = zmatrix.copy()
                    for j, idx in enumerate(rotatable_indices):
                        current_zmatrix[idx][ZMatrix.FIELD_DIHEDRAL] = intermediate_result.x[j]
                    coords = zmatrix_to_cartesian(current_zmatrix)
                    elements = current_zmatrix.get_elements()
                    
                    # Write to trajectory (append mode)
                    print(f"Iteration {iteration_counter[0]}, Ring closure score: {-intermediate_result.fun:.2f}")
                    write_xyz_file(
                        coords, 
                        elements, 
                        trajectory_file,
                        comment=f"Iteration {iteration_counter[0]}, Ring closure score: {-intermediate_result.fun:.2f}",
                        append=True
                    )
                    iteration_counter[0] += 1
                except Exception as e:
                    # Don't fail optimization if trajectory writing fails
                    if verbose:
                        print(f"  Warning: Failed to write trajectory at iteration {iteration_counter[0]}: {e}")
            # scipy works on minimization, hence we use a negated ring closing score.
            return (intermediate_result.fun + 1.0) < 0.02 

        initial_torsions = np.array([zmatrix[idx][ZMatrix.FIELD_DIHEDRAL] for idx in rotatable_indices])
        bounds = [(-180.0, 180.0) for _ in range(len(initial_torsions))]
        initial_score = -objective(initial_torsions)

        info = {
            'initial_score': initial_score,
            'final_score': None,
            'improvement': None
        }

        try:
            result = differential_evolution(
                objective,
                bounds,
                args=(),
                popsize=popsize,
                #workers=-1, # Use all available workers
                callback=callback,
                atol=0.05,
                maxiter=max_iterations,
                polish=False,
                disp=verbose
            )
            final_score = -result.fun

            # Create optimized zmatrix
            optimized_zmatrix = zmatrix.copy()
            for j, idx in enumerate(rotatable_indices):
                optimized_zmatrix[idx][ZMatrix.FIELD_DIHEDRAL] = result.x[j]
            info['final_score'] = final_score
            info['improvement'] = final_score - initial_score
            return optimized_zmatrix, final_score, info

        except Exception as e:
            print(e)
            info['success'] = False
            info['message'] = str(e)
            info['final_score'] = initial_score
            info['improvement'] = 0.0
            return zmatrix.copy(), -initial_score, info


    def _compute_rcp_path_data(self, zmatrix: ZMatrix, rotatable_indices: List[int], verbose: bool = False) -> Dict[Tuple[int, int], Tuple[List[int], List[int]]]:
        """
        Compute paths and rotatable indices for all RCP terms.
        
        Parameters
        ----------
        zmatrix : ZMatrix
            Z-matrix representation
        rotatable_indices : List[int]
            Indices of rotatable atoms in Z-matrix
        verbose : bool
            Print warnings if paths not found
            
        Returns
        -------
        Dict[Tuple[int, int], Tuple[List[int], List[int]]]
            Dictionary mapping RCP terms to (path, path_rotatable_indices)
        """
        # Build bond graph to find paths
        graph = self._build_bond_graph(zmatrix, self.topology)
        
        # Find paths for all RCP terms
        rcp_path_data = {}
        for rca1, rca2 in self.rcpterms:
            path = self._find_path_bfs(graph, rca1, rca2)
            if path is None:
                if verbose:
                    print(f"Warning: No path found between RCA atoms {rca1} and {rca2}")
                continue
            
            # Find rotatable dihedrals on this path
            path_rotatable_indices = []
            
            # First pass: check rotatable indices that are directly in the path
            for rot_idx in rotatable_indices:
                if rot_idx in path:
                    # Check if this dihedral affects atoms on the path
                    atom = zmatrix[rot_idx]
                    bond_ref = atom.get(ZMatrix.FIELD_BOND_REF)
                    angle_ref = atom.get(ZMatrix.FIELD_ANGLE_REF)
                    # If any reference atom is on path, this dihedral affects the path
                    if angle_ref is not None and (bond_ref in path or angle_ref in path):
                        path_rotatable_indices.append(rot_idx)
            
            # Second pass: check for chirality cases
            # For atoms in the path with chirality != 0, their position depends on
            # both the angle_ref atom (C) and the dihedral_ref atom (D).
            # Both C and D define angles that determine the position of the chirality atom.
            # If either C or D has a rotatable dihedral, it affects the path.
            for path_atom_idx in path:
                path_atom = zmatrix[path_atom_idx]
                chirality = path_atom.get(ZMatrix.FIELD_CHIRALITY, 0)
                
                if chirality != 0:
                    # This atom is defined by two angles rather than a dihedral:
                    # - First angle: centered at bond_ref (B), defined by angle_ref (C)
                    # - Second angle: centered at bond_ref (B), defined by dihedral_ref (D)
                    # The position of this atom depends on the positions of both C and D
                    angle_ref = path_atom.get(ZMatrix.FIELD_ANGLE_REF)
                    dihedral_ref = path_atom.get(ZMatrix.FIELD_DIHEDRAL_REF)
                    
                    # Check if angle_ref (C) has a rotatable dihedral
                    if angle_ref is not None:
                        if angle_ref in rotatable_indices and angle_ref not in path_rotatable_indices:
                            # The rotatable dihedral of angle_ref affects the position of path_atom
                            path_rotatable_indices.append(angle_ref)
                    
                    # Check if dihedral_ref (D) has a rotatable dihedral
                    if dihedral_ref is not None:
                        if dihedral_ref in rotatable_indices and dihedral_ref not in path_rotatable_indices:
                            # The rotatable dihedral of dihedral_ref affects the position of path_atom
                            path_rotatable_indices.append(dihedral_ref)
            
            if path_rotatable_indices:
                rcp_path_data[(rca1, rca2)] = (path, path_rotatable_indices)
        
        return rcp_path_data
    
    def _compute_rcp_paths(self, zmatrix: ZMatrix, rcp_path_data: Dict[Tuple[int, int], Tuple[List[int], List[int]]]) -> List[Tuple[List[Tuple[int, int]], List[int], List[int]]]:
        """
        Group RCP terms into related pairs and create combined path info.
        
        Two RCP terms (a,b) and (c,d) are related if:
        - a is bonded to c and b is bonded to d, OR
        - a is bonded to d and b is bonded to c
        
        Parameters
        ----------
        zmatrix : ZMatrix
            Z-matrix representation
        rcp_path_data : Dict[Tuple[int, int], Tuple[List[int], List[int]]]
            Dictionary mapping RCP terms to (path, path_rotatable_indices)
            
        Returns
        -------
        List[Tuple[List[Tuple[int, int]], List[int], List[int]]]
            List of (rcp_group, combined_path, combined_rotatable_indices)
        """
        # Build bond graph for grouping
        graph = self._build_bond_graph(zmatrix, self.topology)
        
        def are_rcp_terms_related(rcp1: Tuple[int, int], rcp2: Tuple[int, int], graph: Dict[int, List[int]]) -> bool:
            """Check if two RCP terms are related (neighbors)."""
            a1, b1 = rcp1
            a2, b2 = rcp2
            # Check if a1 is bonded to a2 and b1 is bonded to b2
            if (a2 in graph.get(a1, [])) and (b2 in graph.get(b1, [])):
                return True
            # Check if a1 is bonded to b2 and b1 is bonded to a2
            if (b2 in graph.get(a1, [])) and (a2 in graph.get(b1, [])):
                return True
            return False
        
        # Group RCP terms into related pairs
        rcp_groups = []
        remaining_rcps = set(rcp_path_data.keys())
        
        while remaining_rcps:
            # Start a new group with the first remaining RCP
            current_group = [remaining_rcps.pop()]
            
            # Find all RCPs related to any RCP in the current group
            found_new = True
            while found_new:
                found_new = False
                for rcp in list(remaining_rcps):
                    for group_rcp in current_group:
                        if are_rcp_terms_related(rcp, group_rcp, graph):
                            current_group.append(rcp)
                            remaining_rcps.remove(rcp)
                            found_new = True
                            break
            
            rcp_groups.append(current_group)
        
        # For each group, create combined path info
        # If group has 2 RCPs, combine them; otherwise treat as single
        rcp_paths = []
        for group in rcp_groups:
            if len(group) == 2:
                # Related pair: combine paths and rotatable indices
                rcp1, rcp2 = group
                path1, rotatable1 = rcp_path_data[rcp1]
                path2, rotatable2 = rcp_path_data[rcp2]
                
                # Union of paths (remove duplicates, preserve order)
                combined_path = list(dict.fromkeys(path1 + path2))

                # Union of rotatable indices keeping order of longest list
                if len(rotatable1) > len(rotatable2):
                    longest_list = rotatable1
                    shortest_list = rotatable2
                else:
                    longest_list = rotatable2
                    shortest_list = rotatable1
                combined_rotatable = longest_list
                for rotatable in shortest_list:
                    if rotatable not in combined_rotatable:
                        combined_rotatable.append(rotatable)
                rcp_paths.append((group, combined_path, combined_rotatable))
            else:
                # Single RCP or larger group: treat individually
                for rcp in group:
                    path, rotatable = rcp_path_data[rcp]
                    rcp_paths.append(([rcp], path, rotatable))
        
        return rcp_paths
    
    def get_rcp_paths(self, zmatrix: ZMatrix, rotatable_indices: List[int], force_recompute: bool = False, verbose: bool = False) -> List[Tuple[List[Tuple[int, int]], List[int], List[int]]]:
        """
        Get RCP paths and groups, computing and caching if necessary.
        
        Parameters
        ----------
        zmatrix : ZMatrix
            Z-matrix representation
        rotatable_indices : List[int]
            Indices of rotatable atoms in Z-matrix
        force_recompute : bool
            If True, recompute even if cached
        verbose : bool
            Print warnings if paths not found
            
        Returns
        -------
        List[Tuple[List[Tuple[int, int]], List[int], List[int]]]
            List of (rcp_group, combined_path, combined_rotatable_indices)
        """
        # Check if we need to recompute (zmatrix changed or force_recompute)
        # Create a hashable representation of zmatrix structure (bonds and atom count)
        zmatrix_repr = tuple(sorted(zmatrix.bonds)) + (len(zmatrix),)
        zmatrix_hash = hash(zmatrix_repr)
        if force_recompute or self._rcp_paths is None or self._rcp_path_data_zmatrix_hash != zmatrix_hash:
            # Compute path data
            self._rcp_path_data = self._compute_rcp_path_data(zmatrix, rotatable_indices, verbose)
            # Compute grouped paths
            self._rcp_paths = self._compute_rcp_paths(zmatrix, self._rcp_path_data)
            self._rcp_path_data_zmatrix_hash = zmatrix_hash
        
        return self._rcp_paths
    
    
    def get_non_intersecting_rcp_paths(self, zmatrix: ZMatrix, rotatable_indices: List[int], 
                                   force_recompute: bool = False, verbose: bool = False) -> List[List[Tuple[List[Tuple[int, int]], List[int], List[int]]]]:
        """
        Group RCP paths into non-intersecting sets.
        
        Two paths intersect if they share at least one rotatable dihedral (torsion).
        Paths that don't intersect can be manipulated independently, reducing the
        complexity of the ring-closing problem.
        
        Parameters
        ----------
        zmatrix : ZMatrix
            Z-matrix representation
        rotatable_indices : List[int]
            Indices of rotatable atoms in Z-matrix
        force_recompute : bool
            If True, recompute paths even if cached
        verbose : bool
            Print information about grouping
        
        Returns
        -------
        List[List[Tuple[List[Tuple[int, int]], List[int], List[int]]]]
            List of groups, where each group is a list of intersecting paths.
            Each path is a tuple of (rcp_group, combined_path, combined_rotatable_indices).
            Paths within a group share at least one rotatable index.
            Paths in different groups do not share any rotatable indices.
        
        Examples
        --------
        If paths A and B share rotatable index 5, and paths C and D share rotatable index 7,
        but A/B don't share any with C/D, then:
        - Group 1: [path_A, path_B]
        - Group 2: [path_C, path_D]
        """
        # Get all RCP paths
        all_paths = self.get_rcp_paths(zmatrix, rotatable_indices, force_recompute, verbose)
        
        if not all_paths:
            return []
        
        # Build intersection graph: paths are nodes, edges connect paths that share rotatable indices
        n_paths = len(all_paths)
        intersection_graph = {i: [] for i in range(n_paths)}
        
        for i in range(n_paths):
            rcp_group_i, path_i, rotatable_i = all_paths[i]
            rotatable_set_i = set(rotatable_i)
            
            for j in range(i + 1, n_paths):
                rcp_group_j, path_j, rotatable_j = all_paths[j]
                rotatable_set_j = set(rotatable_j)
                
                # Check if paths share any rotatable indices
                if rotatable_set_i & rotatable_set_j:  # Intersection is non-empty
                    intersection_graph[i].append(j)
                    intersection_graph[j].append(i)
        
        # Find connected components (groups of intersecting paths)
        visited = set()
        groups = []
        
        def dfs(node: int, current_group: List[int]):
            """Depth-first search to find all connected paths."""
            visited.add(node)
            current_group.append(node)
            for neighbor in intersection_graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, current_group)
        
        for i in range(n_paths):
            if i not in visited:
                current_group = []
                dfs(i, current_group)
                # Convert indices to actual path tuples
                group_paths = [all_paths[idx] for idx in current_group]
                groups.append(group_paths)
        
        if verbose:
            print(f"Found {len(groups)} non-intersecting path groups:")
            for group_idx, group in enumerate(groups):
                rcp_terms_in_group = []
                rotatable_in_group = set()
                for rcp_group, path, rotatable in group:
                    rcp_terms_in_group.extend(rcp_group)
                    rotatable_in_group.update(rotatable)
                print(f"  Group {group_idx + 1}: {len(group)} path(s), "
                      f"RCP terms: {rcp_terms_in_group}, "
                      f"Rotatable indices: {sorted(rotatable_in_group)}")
        
        return groups


    def maximize_ring_closure_in_torsional_space_fabrik(self, zmatrix: ZMatrix, 
                                                         rotatable_indices: List[int],
                                                         max_iterations: int = 50,
                                                         max_num_converged: int = 2,
                                                         convergence_tolerance: float = 0.005,
                                                         ring_closure_tolerance: float = 0.001,
                                                         ring_closure_decay_rate: float = 1.0,
                                                         trajectory_file: Optional[str] = None,
                                                         verbose: bool = False) -> Tuple[ZMatrix, float, Dict[str, Any]]:
        """
        Maximize ring closure in torsional space using FABRIK (Forward And Backward Reaching Inverse Kinematics).
        
        FABRIK is an iterative IK solver that works by:
        1. Forward pass: Adjust dihedrals along path from RCP atom 1 toward RCP atom 2
        2. Backward pass: Adjust dihedrals along path from RCP atom 2 back toward RCP atom 1
        3. Iterate until convergence
        
        This method is typically 10-100x faster than differential evolution for ring closure.
        
        Parameters
        ----------
        zmatrix : ZMatrix
            Starting Z-matrix
        rotatable_indices : List[int]
            Indices of rotatable atoms in Z-matrix (atoms whose dihedrals can be optimized)
        max_iterations : int
            Maximum optimization iterations (default: 50, typically converges in 5-20)
        max_num_converged : int
            Maximum number of consecutive iterations with no improvement to consider converged
        ring_closure_tolerance : float
            Ring closure tolerance (Å) for computing ring closure score
        ring_closure_decay_rate : float
            Ring closure decay rate for computing ring closure score
        trajectory_file : Optional[str]
            If provided, write optimization trajectory to this XYZ file (append mode).
            Writes coordinates after each iteration.
        verbose : bool
            Print optimization progress
            
        Returns
        -------
        Tuple[ZMatrix, float, Dict[str, Any]]
            Optimized Z-matrix, final ring closure score, and info dict
        """
        if not self.rcpterms:
            # No RCP terms, return original
            initial_score = self.ring_closure_score_exponential(
                zmatrix_to_cartesian(zmatrix), ring_closure_tolerance, ring_closure_decay_rate)
            return zmatrix.copy(), initial_score, {
                'initial_score': initial_score,
                'final_score': initial_score,
                'improvement': 0.0,
                'iterations': 0
            }
        
        # Clear trajectory file if provided
        if trajectory_file:
            Path(trajectory_file).unlink(missing_ok=True)
        
        # Get RCP paths (computed and cached if needed)
        rcp_paths = self.get_rcp_paths(zmatrix, rotatable_indices, force_recompute=False, verbose=verbose)
        
        if not rcp_paths:
            if verbose:
                print("Warning: No valid paths found for RCP terms")
            initial_score = self.ring_closure_score_exponential(
                zmatrix_to_cartesian(zmatrix), ring_closure_tolerance, ring_closure_decay_rate)
            return zmatrix.copy(), initial_score, {
                'initial_score': initial_score,
                'final_score': initial_score,
                'improvement': 0.0,
                'iterations': 0
            }
        
        # Initialize
        distance_function_factory = AnalyticalDistanceFactory(zmatrix, self.topology)
        distance_functions = distance_function_factory.get_all_distance_functions(self.rcpterms)
        current_zmatrix = zmatrix.copy()
        
        rcp_distances = {}
        for rcp_term in self.rcpterms:
            rcp1, rcp2 = rcp_term
            # Use canonical ordering for dictionary lookup
            cache_key = (min(rcp1, rcp2), max(rcp1, rcp2))
            distance_function = distance_functions[cache_key]
            rcp_distances[rcp_term] = distance_function({})
        initial_score = self.ring_closure_score_exponential_from_distances(rcp_distances, ring_closure_tolerance, ring_closure_decay_rate)
        
        # Step size for dihedral adjustments (degrees)
        step_size = 60.0        # Start with larger steps
        step_size_factor = 0.30 # reduction factor for step size
        min_step_size = 1.0     # minimum step size
        
        info = {
            'initial_score': initial_score,
            'final_score': None,
            'improvement': None,
            'iterations': 0
        }

        current_score = initial_score
        best_score = initial_score
        prev_score = initial_score
        num_converged = 0
        final_coords = None  # Initialize to None, will be set if trajectory_file is provided or at end
        try:
            for iteration in range(max_iterations):
                # Check if already closed (all RCPs in all groups)
                if current_score > 0.99:
                    break
                
                scores_per_group = {}
                group_index = -1
                for rcp_group, path, path_rotatable_indices in rcp_paths:
                    group_index += 1

                    # Compute combined distance: sum of distances for all RCPs in group
                    current_dihedrals = {idx: current_zmatrix[idx][ZMatrix.FIELD_DIHEDRAL] for idx in path_rotatable_indices}
                    
                    # Forward pass: along the combined path
                    path_rotatable_ordered = path_rotatable_indices
                    # Backward pass: reverse order
                    path_rotatable_ordered_reverse = [id for id in reversed(path_rotatable_indices)]

                    for direction in [path_rotatable_ordered, path_rotatable_ordered_reverse]:
                        for rot_idx in direction:
                            current_dihedral = current_dihedrals[rot_idx]

                            rcp_distances = {}
                            for rcp_term in rcp_group:
                                rcp1, rcp2 = rcp_term
                                # Use canonical ordering for dictionary lookup
                                cache_key = (min(rcp1, rcp2), max(rcp1, rcp2))
                                distance_function = distance_functions[cache_key]
                                rcp_distances[rcp_term] = distance_function(current_dihedrals)
                            best_loc_score = self.ring_closure_score_exponential_from_distances(rcp_distances, ring_closure_tolerance, ring_closure_decay_rate)
                            
                            # Try adjusting dihedral to reduce combined distance
                            # Use finite differences to estimate gradient
                            test_dihedral_values = [
                                current_dihedral - step_size,
                                current_dihedral + step_size
                            ]
                            
                            best_loc_dihedral = current_dihedral
                            
                            for test_dihedral_val in test_dihedral_values:
                                # Wrap to [-180, 180]
                                wrapped_dihedral = test_dihedral_val
                                while wrapped_dihedral > 180.0:
                                    wrapped_dihedral -= 360.0
                                while wrapped_dihedral < -180.0:
                                    wrapped_dihedral += 360.0
                                
                                # Test this dihedral: compute sum of distances for all RCPs in group
                                test_dihedrals = current_dihedrals.copy()
                                test_dihedrals[rot_idx] = wrapped_dihedral
                                rcp_distances = {}
                                for rcp_term in rcp_group:
                                    rcp1, rcp2 = rcp_term
                                    # Use canonical ordering for dictionary lookup
                                    cache_key = (min(rcp1, rcp2), max(rcp1, rcp2))
                                    distance_function = distance_functions[cache_key]
                                    rcp_distances[rcp_term] = distance_function(test_dihedrals)
                                test_score = self.ring_closure_score_exponential_from_distances(rcp_distances, ring_closure_tolerance, ring_closure_decay_rate)
                                accepted = False
                                if test_score > best_loc_score:
                                    best_loc_score = test_score
                                    best_loc_dihedral = wrapped_dihedral
                                    accepted = True
                                #print(f"  Step {step_size:.1f} - Test dihedral: {rot_idx} from {current_dihedral:.3f} to {wrapped_dihedral:.3f} -> score: {test_score:.4f} ({'accepted' if accepted else 'rejected'})")
                            
                            current_dihedrals[rot_idx] = best_loc_dihedral
                            scores_per_group[group_index] = best_loc_score
                            
                            if best_loc_score > 0.99:
                                break

                    # Update final values in zmatrix
                    for idx in path_rotatable_indices:
                        current_zmatrix[idx][ZMatrix.FIELD_DIHEDRAL] = current_dihedrals[idx]

                    rcp_distances = {}
                    for rcp_term in self.rcpterms:
                        rcp1, rcp2 = rcp_term
                        # Use canonical ordering for dictionary lookup
                        cache_key = (min(rcp1, rcp2), max(rcp1, rcp2))
                        distance_function = distance_functions[cache_key]
                        rcp_distances[rcp_term] = distance_function(current_dihedrals)
                    current_score = self.ring_closure_score_exponential_from_distances(rcp_distances, ring_closure_tolerance, ring_closure_decay_rate)

                    if current_score > 0.99:
                        break
                
                # Check convergence: all RCP pairs closed
                all_closed = True
                score_str = ""
                for group_index in scores_per_group:
                    score_str += f" Group {rcp_paths[group_index][0]}: {scores_per_group[group_index]:.4f}"
                    if scores_per_group[group_index] < 0.99:
                        all_closed = False
                if current_score < 0.99:
                    all_closed = False
                
                # check convergence: no improvement
                if abs(prev_score - current_score) < convergence_tolerance:
                    num_converged += 1
                else:
                    num_converged = 0

                if num_converged > max_num_converged:
                    break

                prev_score = current_score

                # Write trajectory if requested
                if trajectory_file:
                    final_coords = zmatrix_to_cartesian(current_zmatrix)
                    try:
                        final_score = self.ring_closure_score_exponential(
                            final_coords, ring_closure_tolerance, ring_closure_decay_rate)
                        elements = current_zmatrix.get_elements()
                        write_xyz_file(
                            final_coords,
                            elements,
                            trajectory_file,
                            comment=f"Iteration {iteration}, Ring closure score: {final_score:.4f}",
                            append=True
                        )
                        if verbose:
                            print(f"Iteration {iteration}, Ring closure score: {final_score:.4f}")
                    except Exception as e:
                        if verbose:
                            print(f"  Warning: Failed to write trajectory at iteration {iteration}: {e}")
                
                #print(f"Iteration {iteration}: RC score({ring_closure_tolerance:.3f}, {ring_closure_decay_rate:.3f}) {final_score:.4f} - {score_str}")
                #print(f'Dihedrals: {current_dihedrals}')

                # Reduce step size for fineer tuning in next iteratiom
                step_size = max(step_size * step_size_factor, min_step_size)

                if all_closed:
                    break
                
            
            # Prepare final results, if not done already
            if final_coords is None:
                final_coords = zmatrix_to_cartesian(current_zmatrix)
                final_score = self.ring_closure_score_exponential(
                    final_coords, ring_closure_tolerance, ring_closure_decay_rate)
            
            info['final_score'] = final_score
            info['improvement'] = final_score - initial_score
            info['iterations'] = iteration + 1
            
            return current_zmatrix, final_score, info
            
        except Exception as e:
            print(traceback.format_exc())
            print(f"FABRIK optimization failed: {e}")
            final_coords = zmatrix_to_cartesian(zmatrix)
            final_score = self.ring_closure_score_exponential(
                final_coords, ring_closure_tolerance, ring_closure_decay_rate)
            info['success'] = False
            info['message'] = str(e)
            info['final_score'] = final_score
            info['improvement'] = final_score - initial_score
            info['iterations'] = iteration if 'iteration' in locals() else 0
            return zmatrix.copy(), final_score, info


    def ring_closure_score_exponential(self, coords: np.ndarray, 
                                        tolerance: float = 0.001,
                                        decay_rate: float = 1.0,
                                        verbose: bool = False) -> float:
        return self.ring_closure_score_exponential_from_coords(coords, self.rcpterms, tolerance, decay_rate, verbose)


    @staticmethod
    def ring_closure_score_exponential_from_coords(coords: np.ndarray, 
                                        rcp_terms: List[Tuple[int, int]],
                                        tolerance: float = 0.001,
                                        decay_rate: float = 1.0,
                                        verbose: bool = False) -> float:
        if not rcp_terms:
            return 1.0  # No constraints means "perfectly satisfied"

        rcp_distances = {} 
        for rcp_term in rcp_terms:
            distance = _calc_distance(coords[rcp_term[0]], coords[rcp_term[1]])
            rcp_distances[rcp_term] = distance
        
        return MolecularSystem.ring_closure_score_exponential_from_distances(rcp_distances, tolerance, decay_rate, verbose)
    

    @staticmethod
    def ring_closure_score_exponential_from_distances(rcp_distances: Dict[Tuple[int, int], float], 
                                        tolerance: float = 0.001,
                                        decay_rate: float = 1.0,
                                        verbose: bool = False) -> float:
        """
        Calculate normalized ring closure score using exponential decay.
        
        This provides a smooth, continuous metric in [0, 1] where:
        - 1.0 indicates all rings are closed (distances <= tolerance)
        - 0.0 indicates rings are very far from closing
        - Values between 0 and 1 indicate partial closure with exponential decay
        
        The score for each RCP term is:
        - If distance <= tolerance: score = 1.0 (perfectly closed)
        - If distance > tolerance: score = exp(-decay_rate * (distance - tolerance))
        
        The overall score is the average across all RCP terms.
        
        Parameters
        ----------
        rcp_distances : Dict[Tuple[int, int], float]
            Dictionary of RCP distances (Angstroms)
        tolerance : float
            Distance threshold below which rings are considered closed (default: 1.54 Å for C-C single bond)
        decay_rate : float
            Exponential decay rate parameter (default: 1.0). Higher values = faster decay.
            With decay_rate=1.0:
            - At tolerance + 1 Å: score ≈ 0.37
            - At tolerance + 2 Å: score ≈ 0.14
            - At tolerance + 3 Å: score ≈ 0.05
        verbose : bool
            If True, print detailed score breakdown for each RCP term
        
        Returns
        -------
        float
            Average exponential closure score across all RCP terms, in range [0, 1].
            Returns 1.0 if no RCP terms are defined.
        
        Notes
        -----
        This metric is the opposite of a penalty - higher is better. It's suitable
        for use as a fitness component where you want to maximize ring closure.
        The exponential form provides smooth gradients for optimization.
        
        Examples
        --------
        If RCP distances are [1.4, 2.0, 5.0] Å with tolerance=1.54, decay_rate=1.0:
        - RCP 1: 1.4 <= 1.54 → score = 1.000 (closed)
        - RCP 2: exp(-1.0 * (2.0-1.54)) = exp(-0.46) ≈ 0.631
        - RCP 3: exp(-1.0 * (5.0-1.54)) = exp(-3.46) ≈ 0.031
        Average score = (1.000 + 0.631 + 0.031) / 3 ≈ 0.554
        """
        if not rcp_distances:
            return 1.0  # No constraints means "perfectly satisfied"
        
        total_score = 0.0
        
        if verbose:
            print(f"  RCP Exponential Score Analysis (tolerance: {tolerance:.2f} Å, decay_rate: {decay_rate:.2f}):")
        
        for rcp_term,distance in rcp_distances.items():
            if distance <= tolerance:
                # Within tolerance - perfect score
                score = 1.0
            else:
                # Beyond tolerance - exponential decay
                excess = distance - tolerance
                score = np.exp(-decay_rate * excess)
            
            total_score += score
            
            if verbose:
                if distance <= tolerance:
                    status = "✓ CLOSED"
                elif score > 0.5:
                    status = "⚠ NEAR"
                else:
                    status = "✗ FAR"
                    
                print(f"    RCP {rcp_term[0]:3d}-{rcp_term[1]:3d}: "
                      f"dist={distance:6.3f} Å, score={score:6.4f}  {status}")
        
        # Return average score across all RCP terms
        avg_score = total_score / len(rcp_distances)
        
        if verbose:
            print(f"  Average score: {avg_score:.4f} (range: [0.0, 1.0])")
        
        return avg_score


    @staticmethod
    def _calculate_rmsd(zmatrix1: ZMatrix, zmatrix2: ZMatrix) -> Tuple[float, float, float]:
        """
        Calculate RMSD between two Z-matrices for bond lengths, angles, and dihedrals.
        
        RMSD (Root Mean Square Deviation) quantifies the structural difference between
        two conformations by computing the RMS of differences in internal coordinates.
        
        Parameters
        ----------
        zmatrix1 : ZMatrix
            First Z-matrix (e.g., initial structure)
        zmatrix2 : ZMatrix
            Second Z-matrix (e.g., optimized structure)
        
        Returns
        -------
        Tuple[float, float, float]
            RMSD for bond lengths (Å), angles (degrees), and dihedrals (degrees)
        
        Notes
        -----
        - Atoms without a particular coordinate type (e.g., first atom has no bond) are skipped
        - When chirality != 0, the 'dihedral' field contains a second angle (not a proper dihedral)
          and is treated as an angle for RMSD calculation (no periodicity handling)
        - Proper dihedrals (chirality == 0) use angular difference accounting for periodicity 
          at the -180°/+180° boundary
        - Empty lists for any coordinate type result in RMSD of 0.0
        
        Raises
        ------
        ValueError
            If Z-matrices have different lengths
        
        Examples
        --------
        >>> rmsd_bonds, rmsd_angles, rmsd_dihedrals = MolecularSystem._calculate_rmsd(initial_zmat, final_zmat)
        >>> print(f"Bond RMSD: {rmsd_bonds:.4f} Å")
        >>> print(f"Angle RMSD: {rmsd_angles:.4f}°")
        >>> print(f"Dihedral RMSD: {rmsd_dihedrals:.4f}°")
        """
        if len(zmatrix1) != len(zmatrix2):
            raise ValueError(f"Z-matrices must have same length: {len(zmatrix1)} vs {len(zmatrix2)}")
        
        bond_diffs = []
        angle_diffs = []
        dihedral_diffs = []
        
        for atom1, atom2 in zip(zmatrix1, zmatrix2):
            # Bond lengths
            if ZMatrix.FIELD_BOND_LENGTH in atom1 and ZMatrix.FIELD_BOND_LENGTH in atom2:
                bond_diffs.append(atom1[ZMatrix.FIELD_BOND_LENGTH] - atom2[ZMatrix.FIELD_BOND_LENGTH])
            
            # Angles
            if ZMatrix.FIELD_ANGLE in atom1 and ZMatrix.FIELD_ANGLE in atom2:
                angle_diffs.append(atom1[ZMatrix.FIELD_ANGLE] - atom2[ZMatrix.FIELD_ANGLE])
            
            # Dihedrals or second angles (depending on chirality)
            if ZMatrix.FIELD_DIHEDRAL in atom1 and ZMatrix.FIELD_DIHEDRAL in atom2:
                chirality1 = atom1.get(ZMatrix.FIELD_CHIRALITY, 0)
                chirality2 = atom2.get(ZMatrix.FIELD_CHIRALITY, 0)
                
                # If chirality != 0, the 'dihedral' is actually a second angle
                if chirality1 != 0 or chirality2 != 0:
                    # Treat as angle (no periodicity handling)
                    angle_diffs.append(atom1[ZMatrix.FIELD_DIHEDRAL] - atom2[ZMatrix.FIELD_DIHEDRAL])
                else:
                    # Treat as proper dihedral (handle periodicity: -180/+180 boundary)
                    diff = atom1[ZMatrix.FIELD_DIHEDRAL] - atom2[ZMatrix.FIELD_DIHEDRAL]
                    # Wrap to [-180, 180] range
                    while diff > 180.0:
                        diff -= 360.0
                    while diff < -180.0:
                        diff += 360.0
                    dihedral_diffs.append(diff)
        
        # Calculate RMSD for each coordinate type
        rmsd_bonds = np.sqrt(np.mean(np.array(bond_diffs)**2)) if bond_diffs else 0.0
        rmsd_angles = np.sqrt(np.mean(np.array(angle_diffs)**2)) if angle_diffs else 0.0
        rmsd_dihedrals = np.sqrt(np.mean(np.array(dihedral_diffs)**2)) if dihedral_diffs else 0.0
        
        return rmsd_bonds, rmsd_angles, rmsd_dihedrals

    @staticmethod
    def _build_bond_graph(zmatrix: ZMatrix, topology=None) -> Dict[int, List[int]]:
        """
        Build bond connectivity graph from molecular topology or Z-matrix.
        
        Uses the complete molecular topology (including all bonds) if available,
        otherwise falls back to Z-matrix bond_ref (which only includes tree structure).
        
        Parameters
        ----------
        zmatrix : ZMatrix
            Z-matrix representation of the molecule
        topology : openmm.app.Topology, optional
            OpenMM topology object containing bond information
        
        Returns
        -------
        Dict[int, List[int]]
            Adjacency list representation (0-based indices)
        """
        num_atoms = len(zmatrix)
        graph = {i: [] for i in range(num_atoms)}
        
        if topology is not None:
            # Use complete topology (includes all bonds, including RCP bonds)
            for bond in topology.bonds():
                graph[bond.atom1.index].append(bond.atom2.index)
                graph[bond.atom2.index].append(bond.atom1.index)
        else:
            # Fall back to Z-matrix (only tree structure bonds)
            for i in range(num_atoms):
                if i == 0:
                    continue
                atom = zmatrix[i]
                bond_ref = atom.get(ZMatrix.FIELD_BOND_REF)
                if bond_ref is not None:
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
    
    @staticmethod
    def _identify_rc_critical_rotatable_indeces(zmatrix: ZMatrix, 
                                                 rcp_terms: List[Tuple[int, int]], 
                                                 rotatable_indices: List[int],
                                                 topology=None) -> Tuple[List[int], List[int]]:
        """
        Identify rotatable torsions on paths between RCP atoms and all atoms on the RCP paths.
        
        Parameters
        ----------
        zmatrix : ZMatrix
            Z-matrix representation
        rcp_terms : List[Tuple[int, int]]
            Ring closure pairs (0-based atom indices)
        rotatable_indices : List[int]
            List of rotatable atom indices in Z-matrix
        topology : openmm.app.Topology, optional
            OpenMM topology object
        
        Returns
        -------
        Tuple[List[int], List[int]]
            A tuple containing:
            - List of critical rotatable indices (indices into rotatable_indices)
            - List of all atoms on paths between RCP terms (including RCP atoms themselves)
        """
        if not rcp_terms:
            return [], []
        
        graph = MolecularSystem._build_bond_graph(zmatrix, topology)
        num_atoms = len(zmatrix)
        rc_critical_atoms = set()
        
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
            
            path = MolecularSystem._find_path_bfs(graph, atom1, atom2)
            if path:
                rc_critical_atoms.update(path)
            else:
                print(f"Warning: No path found between RCP atoms {atom1} and {atom2} (0-based)")
        
        # Identify which rotatable torsions involve critical atoms
        # A torsion is critical if ANY of its 4 defining atoms are on a critical path
        rc_critical_rotatable_indeces = []
        for rot_idx in rotatable_indices:
            atom = zmatrix[rot_idx]
            atoms_in_rotatable_bond = [] 
            if atom.get(ZMatrix.FIELD_BOND_REF) is not None:
                atoms_in_rotatable_bond.append(atom[ZMatrix.FIELD_BOND_REF])
            if atom.get(ZMatrix.FIELD_ANGLE_REF) is not None:
                atoms_in_rotatable_bond.append(atom[ZMatrix.FIELD_ANGLE_REF])
            
            # Check if both atoms are on a critical path
            if all(atom_idx in rc_critical_atoms for atom_idx in atoms_in_rotatable_bond):
                rc_critical_rotatable_indeces.append(rot_idx)
        
        return rc_critical_rotatable_indeces, sorted(list(rc_critical_atoms))
    

    @staticmethod
    def _get_dofs_from_rotatable_indeces(rotatable_indices: List[int], 
                                         rc_critical_rotatable_indeces: List[int], 
                                         rc_critical_atoms: List[int],
                                         zmatrix: ZMatrix) -> List[Tuple[int, int]]:
        """
        Get indexes of degrees of freedom on Z-matrix from rotatable bonds and atoms that are on paths between RCP terms.
        
        Parameters
        ----------
        rotatable_indices : List[int]
            Indices of rotatable atoms in Z-matrix
        rc_critical_rotatable_indeces : List[int]
            Indices of critical rotatable atoms (from RCP paths)
        rc_critical_atoms : List[int]
            Indices of all atoms on paths between RCP terms (including RCP atoms themselves)
        zmatrix : ZMatrix
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

        all_atoms_in_rc_critical_rot_bonds = []
        for idx in rc_critical_rotatable_indeces:
            zatom = zmatrix[idx]
            rb_bond_ref = zatom.get(dof_names[1])
            rb_angle_ref = zatom.get(dof_names[2])
            if rb_bond_ref is not None and rb_bond_ref not in all_atoms_in_rc_critical_rot_bonds:
                all_atoms_in_rc_critical_rot_bonds.append(rb_bond_ref)
            if rb_angle_ref is not None and rb_angle_ref not in all_atoms_in_rc_critical_rot_bonds:
                all_atoms_in_rc_critical_rot_bonds.append(rb_angle_ref)

        # Check assumption
        if [idx for idx in all_atoms_in_rc_critical_rot_bonds if idx not in rc_critical_atoms]:
            raise ValueError("Assumption violated: All atoms in rc_critical_rot_bonds must be in rc_critical_atoms")

        # Add any angle involving only rc-critical atoms
        for idx2, zatom2 in enumerate(zmatrix):
            zatom2_id = zatom2.get(dof_names[0])
            zatom2_bond_ref = zatom2.get(dof_names[1])
            zatom2_angle_ref = zatom2.get(dof_names[2])
            zatom2_dihedral_ref = zatom2.get(dof_names[3])

            if zatom2_bond_ref is not None and zatom2_angle_ref is not None:
                if zatom2_id in rc_critical_atoms and zatom2_bond_ref in rc_critical_atoms and zatom2_angle_ref in rc_critical_atoms:  
                    if (idx2, 1) not in dof_indices:
                        dof_indices.append((idx2, 1))
                if zatom2_id in rc_critical_atoms and zatom2_bond_ref in rc_critical_atoms and zatom2_dihedral_ref in rc_critical_atoms and zatom2.get('chirality', 0) != 0:  
                    if (idx2, 2) not in dof_indices:
                        dof_indices.append((idx2, 2))

        # Add the torsions
        for idx in rotatable_indices:
            if (idx, 2) not in dof_indices:
                dof_indices.append((idx, 2))

        return dof_indices
    
    @staticmethod
    def _get_all_rotatable_indices(zmatrix: ZMatrix) -> List[int]:
        """
        Get all rotatable Z-matrix indices (all dihedrals with chirality == 0).
        
        When rotatable_bonds is not specified, all dihedrals that are not
        chirality-constrained are considered rotatable.
        
        Parameters
        ----------
        zmatrix : ZMatrix
            Z-matrix representation
            
        Returns
        -------
        List[int]
            0-based indices of all rotatable atoms in Z-matrix
        """
        return zmatrix.get_rotatable_indices()
