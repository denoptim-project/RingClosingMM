#!/usr/bin/env python3
"""
Molecular System Management

This module provides the MolecularSystem class for managing OpenMM systems and
molecular structures, including energy evaluation and minimization.

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

from scipy.optimize import minimize


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
        
        # Cache simulations by smoothing parameter to avoid recreating them
        self._simulation_cache = {}
        self._current_smoothing = 0.0
    
    @property
    def elements(self) -> List[str]:
        """Get element symbols from Z-matrix."""
        return [atom['element'] for atom in self.zmatrix]

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
    def from_data(cls, zmatrix: List[Dict], bonds_data: List[Tuple[int, int, int]], 
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
        Perform energy minimization and return optimized coordinates.
        
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
                                            dof_bounds: Optional[List[Tuple[float, float]]] = None,
                                            max_iterations: int = 100,
                                            method: str = 'L-BFGS-B', #'Powell', 
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
        dof_bounds : List[Tuple[float, float]]
            Bounds for the degrees of freedom. Example: [(0.0, 1.0), (0.0, 1.0)] means 
            the first atom's first degree of freedom (distance) is between 0.0 and 1.0, 
            and the second atom's third degree of freedom (torsion) is between 0.0 and 1.0.
            None means no bounds.
            
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
                # Return high penalty on failure
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
                    # Don't fail optimization if trajectory writing fails
                    if verbose:
                        print(f"  Warning: Failed to write trajectory at iteration {iteration_counter[0]}: {e}")
        
        try:
            # Initial energy
            initial_energy = objective(initial_dofs)
            
            if verbose:
                print(f"  Initial dofs: {initial_dofs}")
                print(f"  Initial energy: {initial_energy:.4f} kcal/mol")
                if trajectory_file:
                    print(f"  Writing trajectory to: {trajectory_file}")
            
            if dof_bounds is None:
                dof_bound_coeffs_sequence = [[0.02, 20.0, 180.0], [0.02, 10.0, 10.0], [0.001, 5.0, 5.0]]
            else:
                # Check if dof_bounds is already a sequence of coefficient sets
                # by checking if the first element is itself a sequence (list/tuple)
                if len(dof_bounds) > 0 and isinstance(dof_bounds[0], (list, tuple)):
                    # Already a sequence of coefficient sets
                    dof_bound_coeffs_sequence = dof_bounds
                else:
                    # Single coefficient set, wrap it in a list
                    dof_bound_coeffs_sequence = [dof_bounds]

            current_dofs = initial_dofs.copy()
            for i in range(len(dof_bound_coeffs_sequence)):
                dof_bound_coeffs = dof_bound_coeffs_sequence[i]
                dof_bounds = []
                for j,idx in enumerate(dof_indices):
                    zmat_idx = idx[0]
                    dof_idx = idx[1]
                    dof_current_value = current_dofs[j]
                    dof_type = dof_names[dof_idx]
                    if dof_type == 'bond_length':
                        deviation = dof_bound_coeffs[0]
                        dof_bounds.append((dof_current_value-deviation, dof_current_value+deviation))
                    elif (dof_type == 'angle') or (dof_type == 'dihedral' and zmatrix[zmat_idx]['chirality'] != 0):
                        deviation = dof_bound_coeffs[1]
                        dof_bounds.append((dof_current_value-deviation, dof_current_value+deviation))
                    elif dof_type == 'dihedral' and zmatrix[zmat_idx]['chirality'] == 0:
                        deviation = dof_bound_coeffs[2]
                        if deviation < 179.0:
                            dof_bounds.append((dof_current_value-deviation, dof_current_value+deviation))
                        else:
                            dof_bounds.append((-180.0, 180.0))
                    else:
                        raise ValueError(f"Unknown degree of freedom type: {dof_type}")

                if verbose:
                    print(f"Bounds for iteration {i+1}:")
                    for j, dof_index in enumerate(dof_indices):
                        dof_zmat_line = zmatrix[dof_index[0]]
                        txt = ""
                        if (dof_index[1] == 0):
                            txt = f"  DOF {dof_index}: {dof_zmat_line['id']+1} {dof_zmat_line['bond_ref']+1} bounds: {dof_bounds[j][0]:.4f}, {dof_bounds[j][1]:.4f}"
                        elif (dof_index[1] == 1):
                            txt = f"  DOF {dof_index}: {dof_zmat_line['id']+1} {dof_zmat_line['bond_ref']+1} {dof_zmat_line['angle_ref']+1} bounds: {dof_bounds[j][0]:.4f}, {dof_bounds[j][1]:.4f}"
                        elif (dof_index[1] == 2 and zmatrix[dof_index[0]]['chirality'] != 0):
                            txt = f"  DOF {dof_index}: {dof_zmat_line['id']+1} {dof_zmat_line['bond_ref']+1} {dof_zmat_line['dihedral_ref']+1} ({dof_zmat_line['chirality']}) bounds: {dof_bounds[j][0]:.4f}, {dof_bounds[j][1]:.4f}"
                        elif (dof_index[1] == 2 and zmatrix[dof_index[0]]['chirality'] == 0):
                            txt = f"  DOF {dof_index}: {dof_zmat_line['id']+1} {dof_zmat_line['bond_ref']+1} {dof_zmat_line['angle_ref']+1} {dof_zmat_line['dihedral_ref']+1} ({dof_zmat_line['chirality']}) bounds: {dof_bounds[j][0]:.4f}, {dof_bounds[j][1]:.4f}"
                        print(txt)
            
                # Optimize dofs
                result = minimize(
                    objective,
                    current_dofs,
                    method=method,
                    bounds=dof_bounds,
                    # Only to print trajectory
                    callback=callback if trajectory_file else None,
                    options={
                        'maxiter': max_iterations,
                        'ftol': 0.05,    # Function tolerance (kcal/mol)
                        'xtol': 0.01,    # Tolerance for changes in the variable
                        'disp': verbose
                    }
                )
                current_dofs = result.x.copy()
            
            if verbose:
                print(f"  Final DOFs: {result.x}")
                print(f"  Changed DOFs: {result.x - initial_dofs}")
                print(f"  Optimization: nfev={result.nfev}, nit={result.nit}, "
                      f"success={result.success}")
                print(f"  Final energy: {result.fun:.4f} kcal/mol")
                print(f"  Improvement: {initial_energy - result.fun:.4f} kcal/mol")
            
            # Create optimized zmatrix
            optimized_zmatrix = get_updated_zmatrix(zmatrix, result.x)
            
            # Prepare info dict
            info = {
                'success': result.success,
                'message': result.message,
                'nfev': result.nfev,
                'nit': result.nit,
                'initial_energy': initial_energy,
                'final_energy': result.fun,
                'improvement': initial_energy - result.fun,
                'initial_dofs': initial_dofs.copy(),
                'final_torsions': result.x.copy()
            }
            
            return optimized_zmatrix, result.fun, info
            
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
                # Only to print trajectory
                # callback=callback if trajectory_file else None,
                options={
                    'maxiter': max_iterations,
                    'ftol': 0.05,    # Function tolerance (kcal/mol)
                    'xtol': 0.01,    # Tolerance for changes in the variables (degrees)
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
    

    def ring_closure_penalty_quadratic(self, coords: np.ndarray, 
                                        target_distance: float = 1.54,
                                        verbose: bool = False) -> float:
        """
        Calculate continuous ring closure penalty using squared excess distances.
        
        This provides a smooth, continuous metric suitable for optimization,
        unlike the discrete count from count_near_closed_rings(). The penalty
        is zero when all RCP distances are at or below the target distance,
        and increases quadratically with distance above the target.
        
        Parameters
        ----------
        coords : np.ndarray
            Cartesian coordinates (Angstroms)
        target_distance : float
            Target bond distance for closed rings (default: 1.54 Å for C-C single bond)
        verbose : bool
            If True, print detailed penalty breakdown for each RCP term
        
        Returns
        -------
        float
            Sum of squared excess distances (Ų). Zero indicates all rings are closed
            or nearly closed (within target_distance).
        
        Notes
        -----
        The quadratic penalty penalizes large distances more heavily than small ones,
        providing better gradient information for optimization algorithms.
        
        Examples
        --------
        If RCP distances are [1.5, 2.5, 5.0] Å with target=1.54:
        - RCP 1: (1.5-1.54)² = 0.0016 (essentially closed, small penalty)
        - RCP 2: (2.5-1.54)² = 0.9216 (moderate penalty)
        - RCP 3: (5.0-1.54)² = 11.9716 (large penalty for distant atoms)
        Total penalty = 12.8948 Ų
        """
        if not self.rcpterms:
            return 0.0
        
        total_penalty = 0.0
        
        if verbose:
            print(f"\n  RCP Penalty Analysis (target distance: {target_distance:.2f} Å):")
        
        for rcp_term in self.rcpterms:
            distance = _calc_distance(coords[rcp_term[0]], coords[rcp_term[1]])
            excess = max(0.0, distance - target_distance)
            penalty = excess ** 2
            total_penalty += penalty
            
            if verbose:
                status = "✓ OK" if excess < 0.1 else "⚠ OPEN"
                print(f"    RCP {rcp_term[0]:3d}-{rcp_term[1]:3d}: "
                      f"dist={distance:6.3f} Å, excess={excess:6.3f} Å, "
                      f"penalty={penalty:8.4f} Ų  {status}")
        
        if verbose:
            print(f"  Total penalty: {total_penalty:.4f} Ų\n")
        
        return total_penalty


    def ring_closure_score_exponential(self, coords: np.ndarray, 
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
        if not self.rcpterms:
            return 1.0  # No constraints means "perfectly satisfied"
        
        total_score = 0.0
        
        if verbose:
            print(f"  RCP Exponential Score Analysis (tolerance: {tolerance:.2f} Å, decay_rate: {decay_rate:.2f}):")
        
        for rcp_term in self.rcpterms:
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
        avg_score = total_score / len(self.rcpterms)
        
        if verbose:
            print(f"  Average score: {avg_score:.4f} (range: [0.0, 1.0])")
        
        return avg_score


    def calculate_rmsd(self, zmatrix1: List[Dict], zmatrix2: List[Dict]) -> Tuple[float, float, float]:
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
        >>> rmsd_bonds, rmsd_angles, rmsd_dihedrals = system.calculate_rmsd(initial_zmat, final_zmat)
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
