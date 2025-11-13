#!/usr/bin/env python3
"""
Molecular System Management

This module provides the MolecularSystem class for creating and managing OpenMM systems and
molecular manipulation (e.g., identification of rotatable bonds and related internal coordinates) and molecualr modeling, including energy evaluation and minimization.

Classes:
    MolecularSystem: Manages OpenMM system, topology, and provides energy evaluation methods
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
import copy

import openmm.unit as unit
from openmm.app import Element, Simulation, Topology

# Dual import handling for package and direct script use
try:
    from .IOTools import read_int_file, write_xyz_file
except ImportError:
    from IOTools import read_int_file, write_xyz_file

try:
    # Relative imports for package use
    from .RingClosingForceField import (
        create_simulation_from_system,
        create_system,
        setGlobalParameterToAllForces
    )
    from .CoordinateConverter import (
        zmatrix_to_cartesian,
        _calc_distance
    )
except ImportError:
    # Absolute imports for direct script use
    from RingClosingForceField import (
        create_simulation_from_system,
        create_system,
        setGlobalParameterToAllForces
    )
    from CoordinateConverter import (
        zmatrix_to_cartesian,
        _calc_distance
    )

from scipy.optimize import OptimizeResult, differential_evolution, minimize


def build_topology_from_data(atoms_data, bonds_data):
    """
    Build OpenMM Topology from parsed atom and bond data.
    
    Parameters
    ----------
    atoms_data : list of tuples
        List of (element_symbol, atom_index) tuples
    bonds_data : list of tuples
        List of (atom1_idx, atom2_idx, bond_type) tuples
    
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
    for atom1_idx, atom2_idx, bond_type in bonds_data:
        topo.addBond(atoms[atom1_idx], atoms[atom2_idx], bond_type)
    
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
    zmatrix : List[Dict]
        Z-matrix (internal coordinates) representation
    step_length : float
        Integration step length for simulations
    """
    
    def __init__(self, system, topology, rcpterms: List[Tuple[int, int]], 
                 zmatrix: List[Dict],
                 step_length: float = 0.0002,
                 write_candidate_files: bool = False,
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
        zmatrix : List[Dict]
            Z-matrix representation
        step_length : float
            Integration step length
        write_candidate_files : bool
            Write candidate files (int and xyz)
        ring_closure_threshold : float
            Distance threshold (Angstroms) for considering a ring nearly closed
        """
        self.system = system
        self.topology = topology
        self.rcpterms = rcpterms if rcpterms else []
        self.zmatrix = zmatrix
        self.step_length = step_length
        self.write_candidate_files = write_candidate_files
        self.candidate_files_prefix = "candidate"
        self.ring_closure_threshold = ring_closure_threshold
        
        # Rotatable indices and related fields (computed when rotatable_indices is set)
        self.rotatable_indices: Optional[List[int]] = None
        self.rc_critical_atoms: Optional[List[int]] = None
        self.rc_critical_rotatable_indeces: Optional[List[int]] = None
        self.dof_indices: Optional[List[Tuple[int, int]]] = None
        
        # Cache simulations by smoothing parameter to avoid recreating them
        self._simulation_cache = {}
        self._current_smoothing = 0.0
    
    @property
    def elements(self) -> List[str]:
        """Get element symbols from Z-matrix."""
        return [atom['element'] for atom in self.zmatrix]
    
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

    @classmethod
    def from_file(cls, structure_file: str, forcefield_file: str,
                  rcp_terms: Optional[List[Tuple[int, int]]] = None,
                  write_candidate_files: bool = False,
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
        write_candidate_files : bool
            Write candidate files (int and xyz)
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
        if not structure_file.endswith('.int'):
            raise ValueError("Only .int files containing Z-matrix data are supported as input files.")
        
        data = read_int_file(structure_file)
        
        # Get Z-matrix
        if 'zmatrix' in data and data['zmatrix']:
            zmatrix = data['zmatrix']
        else:
            raise ValueError("Z-matrix not found in file. Use .int format for Z-matrix files.")

        bonds_data = data['bonds']
        
        return cls.from_data(zmatrix, bonds_data, forcefield_file, rcp_terms, write_candidate_files, ring_closure_threshold, step_length)
    
    @classmethod
    def from_data(cls, zmatrix: List[Dict], 
                     bonds_data: List[Tuple[int, int, int]], 
                     forcefield_file: str,
                     rcp_terms: Optional[List[Tuple[int, int]]] = None,
                     write_candidate_files: bool = False,
                     ring_closure_threshold: float = 1.5,
                     step_length: float = 0.0002) -> 'MolecularSystem':
        """
        Create molecular system from raw data.
        
        Parameters
        ----------
        zmatrix : List[Dict]
            Z-matrix representation. All reference indices (bond_ref, angle_ref, 
            dihedral_ref) must be in 0-based indexing.
        bonds_data : List[Tuple[int, int, int]]
            List of atom indices that are bonded to each other and the bond type.
            All indices must be in 0-based indexing.
        forcefield_file : str
            Path to force field XML file
        rcp_terms : Optional[List[Tuple[int, int]]]
            RCP terms (optional). All indices must be in 0-based indexing.
        write_candidate_files : bool
            Write candidate files (int and xyz)
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
        atoms_data = [(atom['element'], i) for i, atom in enumerate(zmatrix)]

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
        
        return cls(system, topology, rcp_terms or [], zmatrix, step_length, 
                  write_candidate_files, ring_closure_threshold)


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


    def minimize_energy(self, zmatrix: List[Dict], max_iterations: int = 100) -> Tuple[np.ndarray, float]:
        """
        Perform energy minimization with OpenMM engine in Cartesian space and return optimized coordinates.
        
        Uses a cached Simulation for the current smoothing parameter (creating it
        if necessary on first use), then updates positions and performs minimization.
        
        Parameters
        ----------
        zmatrix : List[Dict]
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


    def minimize_energy_in_zmatrix_space(self, zmatrix: List[Dict], 
                                            dof_indices: List[Tuple[int,int]],
                                            dof_bounds_per_type: Optional[List[Tuple[float, float, float]]] = [[10.0, 180.0, 180.0]],
                                            max_iterations: int = 100,
                                            gradient_tolerance: float = 0.01,
                                            verbose: bool = False,
                                            trajectory_file: Optional[str] = None) -> Tuple[List[Dict], float, Dict[str, Any]]:
        """
        Perform energy minimization in Z-matrix space only.
        
        Optimizes only the degrees of freedom (dof)while keeping the other degrees of freedom fixed.
        Uses scipy.optimize.minimize with numerical gradients.
        
        Parameters
        ----------
        zmatrix : List[Dict]
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
        Tuple[List[Dict], float, Dict[str, Any]]
            Optimized Z-matrix, optimized energy (kcal/mol), and optimization info dict
        """
        # Convention in Zmatrix: should come from the ZMatrix class, not here
        # WARNING: the 'dihedral' can actially be a second angle!
        dof_names = ['bond_length', 'angle', 'dihedral']

        # Record the initial values of the dofs
        initial_dofs = np.array([zmatrix[idx[0]][dof_names[idx[1]]] for idx in dof_indices])
        
        # Clear trajectory file if provided
        if trajectory_file:
            Path(trajectory_file).unlink(missing_ok=True)
        
        # Iteration counter for trajectory
        iteration_counter = [0]

        def get_updated_zmatrix(zmatrix: List[Dict], dofs: np.ndarray) -> List[Dict]:
            """Get updated Z-matrix with given dofs."""
            updated_zmatrix = copy.deepcopy(zmatrix)
            for j, idx in enumerate(dof_indices):
                zmat_idx = idx[0]
                dof_idx = idx[1]
                updated_zmatrix[zmat_idx][dof_names[dof_idx]] = dofs[j]
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
                    elements = [current_zmatrix[idx]['element'] for idx in range(len(current_zmatrix))]
                    
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
                        current_dof_bounds.append((dof_current_value-deviation, dof_current_value+deviation))
                    elif (dof_type == 'angle') or (dof_type == 'dihedral' and zmatrix[zmat_idx]['chirality'] != 0):
                        deviation = dof_bound_coeffs[1]
                        current_dof_bounds.append((dof_current_value-deviation, dof_current_value+deviation))
                    elif dof_type == 'dihedral' and zmatrix[zmat_idx]['chirality'] == 0:
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

                # Provide jac (gradient function) so scipy uses our numerical gradient
                # with appropriate step size (h=0.1) instead of machine epsilon
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
                        'eps': 0.1  # step size for numerical gradient calculation
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
                'initial_dofs': initial_dofs.copy(),
                'final_dofs': initial_dofs.copy()
            }
            return zmatrix, 1e6, info


    def report_dof_info(self, zmatrix: List[Dict], dof_indices: List[Tuple[int,int]], dofs: np.ndarray, dof_bounds: List[Tuple[float, float]], gradient: Optional[np.ndarray]):
        """Report information about the degrees of freedom on screen."""
        # Format gradient norm and max gradient with NA handling    
        
        dof_names = ['bond_length', 'angle', 'dihedral']
        
        for j, dof_index in enumerate(dof_indices):
            zmat_idx, dof_type = dof_index
            atom = zmatrix[zmat_idx]
            atom_id = atom['id'] + 1
            
            # Build atom reference list based on DOF type
            ref_atoms = [atom.get('bond_ref', -1) + 1]
            if dof_type >= 1:
                ref_atoms.append(atom.get('angle_ref', -1) + 1)
            if dof_type == 2:
                ref_atoms.append(atom.get('dihedral_ref', -1) + 1)
            
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
            

    def minimize_energy_in_torsional_space(self, zmatrix: List[Dict], 
                                            rotatable_indices: List[int],
                                            max_iterations: int = 100,
                                            method: str = 'Powell', 
                                            verbose: bool = False,
                                            trajectory_file: Optional[str] = None) -> Tuple[List[Dict], float, Dict[str, Any]]:
        """
        Perform energy minimization in torsional space only.
        
        Optimizes only the dihedral angles while keeping bond lengths and angles fixed.
        Uses scipy.optimize.minimize with numerical gradients.
        
        Parameters
        ----------
        zmatrix : List[Dict]
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
        Tuple[List[Dict], float, Dict[str, Any]]
            Optimized Z-matrix, optimized energy (kcal/mol), and optimization info dict
        """
        # Extract initial torsion angles
        initial_torsions = np.array([zmatrix[idx]['dihedral'] for idx in rotatable_indices])
        
        # Clear trajectory file if provided
        if trajectory_file:
            Path(trajectory_file).unlink(missing_ok=True)
        
        # Iteration counter for trajectory
        iteration_counter = [0]
        
        def objective(torsions: np.ndarray) -> float:
            """Objective function: evaluate energy for given torsions."""
            try:
                # Update zmatrix with new torsion values
                updated_zmatrix = copy.deepcopy(zmatrix)
                for j, idx in enumerate(rotatable_indices):
                    updated_zmatrix[idx]['dihedral'] = torsions[j]
                
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
                    current_zmatrix = copy.deepcopy(zmatrix)
                    for j, idx in enumerate(rotatable_indices):
                        current_zmatrix[idx]['dihedral'] = xk[j]
                    
                    # Convert to Cartesian
                    coords = zmatrix_to_cartesian(current_zmatrix)
                    elements = [current_zmatrix[idx]['element'] for idx in range(len(current_zmatrix))]
                    
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
            optimized_zmatrix = copy.deepcopy(zmatrix)
            for j, idx in enumerate(rotatable_indices):
                optimized_zmatrix[idx]['dihedral'] = result.x[j]
            
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
                'initial_torsions': initial_torsions.copy(),
                'final_torsions': initial_torsions.copy()
            }
            return zmatrix, 1e6, info


    def maximize_ring_closure_in_torsional_space(self, zmatrix: List[Dict], 
                                                  rotatable_indices: List[int],
                                                  max_iterations: int = 100,
                                                  ring_closure_tolerance: float = 0.2,
                                                  ring_closure_decay_rate: float = 0.5,
                                                  verbose: bool = False) -> Tuple[List[Dict], float, Dict[str, Any]]:
        """
        Maximize ring closure in torsional space by dual annealing.
        
        Parameters
        ----------
        zmatrix : List[Dict]
            Starting Z-matrix
        rotatable_indices : List[int]
            Indices of rotatable atoms in Z-matrix (atoms whose dihedrals can be optimized)
        max_iterations : int
            Maximum optimization iterations
        verbose : bool
            Print optimization progress
        """

        def objective(torsions: np.ndarray) -> float:
            """Objective function: evaluate ring closure score for given torsions."""
            try:
                # Update zmatrix with new torsion values
                updated_zmatrix = copy.deepcopy(zmatrix)
                for j, idx in enumerate(rotatable_indices):
                    updated_zmatrix[idx]['dihedral'] = torsions[j]
                coords = zmatrix_to_cartesian(updated_zmatrix)
                # Negated because we want to maximize ring closure score
                return -self.ring_closure_score_exponential(coords, ring_closure_tolerance, ring_closure_decay_rate)
            except Exception as e:
                print(f'Error in objective function: {e.message if hasattr(e, "message") else str(e)}')
                return 0.0

        def callback(intermediate_result: OptimizeResult):
            """Callback function called after each optimizer iteration to verify reaching of theoretical best solution (i.e., f(x) = -1.0)."""
            # scipy works on minimization, hence we use a negated ring closing score.
            return (intermediate_result.fun + 1.0) < 0.02 

        initial_torsions = np.array([zmatrix[idx]['dihedral'] for idx in rotatable_indices])
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
                callback=callback,
                atol=0.05,
                maxiter=250,
                polish=False,
                disp=verbose
            )
            final_score = -result.fun

            # Create optimized zmatrix
            optimized_zmatrix = copy.deepcopy(zmatrix)
            for j, idx in enumerate(rotatable_indices):
                optimized_zmatrix[idx]['dihedral'] = result.x[j]
            info['final_score'] = final_score
            info['improvement'] = final_score - initial_score
            return optimized_zmatrix, final_score, info

        except Exception as e:
            print(e)
            info['success'] = False
            info['message'] = str(e)
            info['final_score'] = initial_score
            info['improvement'] = 0.0
            return zmatrix, -initial_score, info

    def ring_closure_score_exponential(self, coords: np.ndarray, 
                                        tolerance: float = 0.001,
                                        decay_rate: float = 1.0,
                                        verbose: bool = False) -> float:
        return self._ring_closure_score_exponential(coords, self.rcpterms, tolerance, decay_rate, verbose)
    

    @staticmethod
    def _ring_closure_score_exponential(coords: np.ndarray, 
                                        rcp_terms: List[Tuple[int, int]],
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
        coords : np.ndarray
            Cartesian coordinates (Angstroms)
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
        if not rcp_terms:
            return 1.0  # No constraints means "perfectly satisfied"
        
        total_score = 0.0
        
        if verbose:
            print(f"  RCP Exponential Score Analysis (tolerance: {tolerance:.2f} Å, decay_rate: {decay_rate:.2f}):")
        
        for rcp_term in rcp_terms:
            distance = _calc_distance(coords[rcp_term[0]], coords[rcp_term[1]])
            
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
        avg_score = total_score / len(rcp_terms)
        
        if verbose:
            print(f"  Average score: {avg_score:.4f} (range: [0.0, 1.0])")
        
        return avg_score


    @staticmethod
    def _calculate_rmsd(zmatrix1: List[Dict], zmatrix2: List[Dict]) -> Tuple[float, float, float]:
        """
        Calculate RMSD between two Z-matrices for bond lengths, angles, and dihedrals.
        
        RMSD (Root Mean Square Deviation) quantifies the structural difference between
        two conformations by computing the RMS of differences in internal coordinates.
        
        Parameters
        ----------
        zmatrix1 : List[Dict]
            First Z-matrix (e.g., initial structure)
        zmatrix2 : List[Dict]
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
            if 'bond_length' in atom1 and 'bond_length' in atom2:
                bond_diffs.append(atom1['bond_length'] - atom2['bond_length'])
            
            # Angles
            if 'angle' in atom1 and 'angle' in atom2:
                angle_diffs.append(atom1['angle'] - atom2['angle'])
            
            # Dihedrals or second angles (depending on chirality)
            if 'dihedral' in atom1 and 'dihedral' in atom2:
                chirality1 = atom1.get('chirality', 0)
                chirality2 = atom2.get('chirality', 0)
                
                # If chirality != 0, the 'dihedral' is actually a second angle
                if chirality1 != 0 or chirality2 != 0:
                    # Treat as angle (no periodicity handling)
                    angle_diffs.append(atom1['dihedral'] - atom2['dihedral'])
                else:
                    # Treat as proper dihedral (handle periodicity: -180/+180 boundary)
                    diff = atom1['dihedral'] - atom2['dihedral']
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
    def _build_bond_graph(zmatrix: List[Dict], topology=None) -> Dict[int, List[int]]:
        """
        Build bond connectivity graph from molecular topology or Z-matrix.
        
        Uses the complete molecular topology (including all bonds) if available,
        otherwise falls back to Z-matrix bond_ref (which only includes tree structure).
        
        Parameters
        ----------
        zmatrix : List[Dict]
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
            for i, atom in enumerate(zmatrix):
                if i == 0:
                    continue
                bond_ref = atom.get('bond_ref')
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
    def _identify_rc_critical_rotatable_indeces(zmatrix: List[Dict], 
                                                 rcp_terms: List[Tuple[int, int]], 
                                                 rotatable_indices: List[int],
                                                 topology=None) -> Tuple[List[int], List[int]]:
        """
        Identify rotatable torsions on paths between RCP atoms and all atoms on the RCP paths.
        
        Parameters
        ----------
        zmatrix : List[Dict]
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
            if atom.get('bond_ref') is not None:
                atoms_in_rotatable_bond.append(atom['bond_ref'])
            if atom.get('angle_ref') is not None:
                atoms_in_rotatable_bond.append(atom['angle_ref'])
            
            # Check if both atoms are on a critical path
            if all(atom_idx in rc_critical_atoms for atom_idx in atoms_in_rotatable_bond):
                rc_critical_rotatable_indeces.append(rot_idx)
        
        return rc_critical_rotatable_indeces, sorted(list(rc_critical_atoms))
    

    @staticmethod
    def _get_dofs_from_rotatable_indeces(rotatable_indices: List[int], 
                                         rc_critical_rotatable_indeces: List[int], 
                                         rc_critical_atoms: List[int],
                                         zmatrix: List[Dict]) -> List[Tuple[int, int]]:
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
