#!/usr/bin/env python3
"""
Analytical Distance Computation

This module provides analytical computation of interatomic distances along paths
in Z-matrix space, avoiding expensive full zmatrix-to-cartesian conversions.

The key insight is that for a serial chain between two atoms, the end-to-end
distance can be computed analytically using forward kinematics (robotic arm math).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Set
from collections import defaultdict

# Try to import numba for JIT compilation (optional)
try:
    from numba import jit, types
    from numba.typed import Dict as NumbaDict
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorator if numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from .ZMatrix import ZMatrix


def rotation_matrix_around_axis(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Compute rotation matrix for rotation around an axis.
    
    Uses Rodrigues' rotation formula.
    
    Parameters
    ----------
    axis : np.ndarray
        Unit vector representing rotation axis
    angle_deg : float
        Rotation angle in degrees
        
    Returns
    -------
    np.ndarray
        3x3 rotation matrix
    """
    angle_rad = np.deg2rad(angle_deg)
    
    # Normalize axis
    axis = axis / np.linalg.norm(axis)
    
    # Rodrigues' rotation formula
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Cross product matrix
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    # Rotation matrix: R = I + sin(θ)K + (1-cos(θ))K²
    R = (np.eye(3) + 
         sin_a * K + 
         (1 - cos_a) * np.dot(K, K))
    
    return R


@jit(nopython=True, cache=True)
def _compute_atom_position_numba(
    atom_idx: int,
    bond_ref: int,
    angle_ref: int,
    dihedral_ref: int,
    bond_length: float,
    angle_deg: float,
    dihedral_deg: float,
    chirality: int,
    coords: np.ndarray,
    dihedral_override: float = -999.0  # Use -999 as "not set" flag
) -> np.ndarray:
    """
    JIT-compiled function to compute atom position using Z-matrix geometry.
    
    Parameters
    ----------
    atom_idx : int
        Index of atom to compute (0-based)
    bond_ref, angle_ref, dihedral_ref : int
        Reference atom indices (-1 if not applicable)
    bond_length, angle_deg, dihedral_deg : float
        Z-matrix parameters
    chirality : int
        Chirality flag (0 = dihedral, non-zero = chirality)
    coords : np.ndarray
        Array of all coordinates (will be modified in place for atom_idx)
    dihedral_override : float
        Override dihedral value if != -999.0
        
    Returns
    -------
    np.ndarray
        3D position of atom
    """
    # Use override if provided
    if dihedral_override > -900.0:  # Check if set
        dihedral_deg = dihedral_override
    
    # Get reference coordinates
    if bond_ref >= 0:
        coords_ia = coords[bond_ref]
    else:
        coords_ia = np.array([0.0, 0.0, 0.0])
    
    if angle_ref >= 0:
        coords_ib = coords[angle_ref]
    else:
        coords_ib = np.array([0.0, 0.0, 1.0])
    
    if dihedral_ref >= 0:
        coords_ic = coords[dihedral_ref]
    else:
        coords_ic = np.array([1.0, 0.0, 0.0])
    
    # Convert to radians
    angle_rad = np.deg2rad(angle_deg)
    dihedral_rad = np.deg2rad(-dihedral_deg)  # Negate for XZ plane convention
    
    sin1 = np.sin(angle_rad)
    cos1 = np.cos(angle_rad)
    sin2 = np.sin(dihedral_rad)
    cos2 = np.cos(dihedral_rad)
    
    if chirality == 0:
        # Dihedral case
        xab = coords_ia - coords_ib
        rab = np.linalg.norm(xab)
        if rab > 1e-8:
            xab = xab / rab
        else:
            xab = np.array([1.0, 0.0, 0.0])
        
        xbc = coords_ib - coords_ic
        rbc = np.linalg.norm(xbc)
        if rbc > 1e-8:
            xbc = xbc / rbc
        else:
            xbc = np.array([0.0, 1.0, 0.0])
        
        # Cross products
        xt = np.cross(xab, xbc)
        cosine = np.dot(xab, xbc)
        sine = np.sqrt(max(1.0 - cosine * cosine, 1e-8))
        if sine > 1e-8:
            xt = xt / sine
        else:
            xt = np.array([0.0, 0.0, 1.0])
        
        xu = np.cross(xab, xt)
        
        result = coords_ia + bond_length * (
            xu * sin1 * cos2 + 
            xt * sin1 * sin2 - 
            xab * cos1
        )
    else:
        # Chirality case (same as zmatrix_to_cartesian)
        xba = coords_ib - coords_ia
        rba = np.linalg.norm(xba)
        if rba > 1e-8:
            xba = xba / rba
        else:
            xba = np.array([1.0, 0.0, 0.0])
        
        xac = coords_ia - coords_ic
        rac = np.linalg.norm(xac)
        if rac > 1e-8:
            xac = xac / rac
        else:
            xac = np.array([0.0, 1.0, 0.0])
        
        # Check for linearity
        cosine = np.dot(xba, xac)
        if abs(abs(cosine) - 1.0) < 1e-6:
            # Linear case - use simplified formula
            result = coords_ia + bond_length * (xba * cos1 + xac * sin1)
        else:
            # Cross product order matters for handedness in XZ plane convention
            xt = np.cross(xac, xba)
            sine2 = max(1.0 - cosine * cosine, 1e-8)
            
            a = (-cos2 - cosine * cos1) / sine2
            b = (cos1 + cosine * cos2) / sine2
            c = (1.0 + a * cos2 - b * cos1) / sine2
            
            eps = 1e-8
            if c > eps:
                c = chirality * np.sqrt(c)
            elif c < -eps:
                c_denom = np.linalg.norm(a * xac + b * xba)
                if c_denom > 1e-8:
                    a = a / c_denom
                    b = b / c_denom
                c = 0.0
            else:
                c = 0.0
            
            result = coords_ia + bond_length * (a * xac + b * xba + c * xt)
    
    return result


@jit(nopython=True, cache=True)
def _analytical_distance_numba(
    path_atoms: np.ndarray,
    all_atom_indices: np.ndarray,  # All atoms needed (path + references)
    bond_refs: np.ndarray,
    angle_refs: np.ndarray,
    dihedral_refs: np.ndarray,
    bond_lengths: np.ndarray,
    bond_angles: np.ndarray,
    default_dihedrals: np.ndarray,
    chiralities: np.ndarray,
    dihedral_indices: np.ndarray,
    dihedral_values: np.ndarray,
    atom_to_array_idx: np.ndarray,  # Mapping from atom idx to position in arrays
    max_atoms: int
) -> float:
    """
    JIT-compiled analytical distance computation.
    
    Computes positions for all atoms in dependency order, then returns distance
    between start and end atoms of the path.
    
    Parameters
    ----------
    path_atoms : np.ndarray[int]
        Atom indices along the path (start and end)
    all_atom_indices : np.ndarray[int]
        All atom indices needed (path + all reference atoms), in dependency order
    bond_refs, angle_refs, dihedral_refs : np.ndarray[int]
        Reference atom indices for each atom (-1 if not applicable)
    bond_lengths, bond_angles, default_dihedrals : np.ndarray[float]
        Z-matrix parameters for each atom
    chiralities : np.ndarray[int]
        Chirality flags
    dihedral_indices : np.ndarray[int]
        Which atoms have variable dihedrals (Z-matrix indices)
    dihedral_values : np.ndarray[float]
        Dihedral values (ordered by dihedral_indices)
    atom_to_array_idx : np.ndarray[int]
        Mapping from atom index to position in parameter arrays (-1 if not in arrays)
    max_atoms : int
        Maximum atom index (for array allocation)
        
    Returns
    -------
    float
        Distance between start and end atoms
    """
    # Allocate coordinate array
    coords = np.zeros((max_atoms + 1, 3))
    
    # Create mapping from dihedral index to value
    dihedral_map = np.full(max_atoms + 1, -999.0)  # -999 = not set
    for i in range(len(dihedral_indices)):
        if dihedral_indices[i] < len(dihedral_map):
            dihedral_map[dihedral_indices[i]] = dihedral_values[i]
    
    # Compute positions for all atoms in dependency order
    for atom_idx in all_atom_indices:
        array_idx = atom_to_array_idx[atom_idx]
        
        # Special handling for first 3 atoms (same as zmatrix_to_cartesian)
        if atom_idx == 0:
            # Atom 0: always at origin
            coords[atom_idx] = np.array([0.0, 0.0, 0.0])
            continue
        
        if atom_idx == 1:
            # Atom 1: along +Z axis
            if array_idx >= 0:
                bond_length = bond_lengths[array_idx]
            else:
                bond_length = 1.5  # Default
            coords[atom_idx] = np.array([0.0, 0.0, bond_length])
            continue
        
        if atom_idx == 2:
            # Atom 2: place in XZ plane (same logic as zmatrix_to_cartesian)
            if array_idx >= 0:
                ref_bond = bond_refs[array_idx]
                ref_angle = angle_refs[array_idx]
                bond_length = bond_lengths[array_idx]
                angle_deg = bond_angles[array_idx]
            else:
                # Fallback (shouldn't happen)
                coords[atom_idx] = np.array([1.0, 0.0, 0.0])
                continue
            
            # Get reference coordinates
            if ref_bond >= 0:
                coords_ref_bond = coords[ref_bond]
            else:
                coords_ref_bond = np.array([0.0, 0.0, 0.0])
            
            if ref_angle >= 0:
                coords_ref_angle = coords[ref_angle]
            else:
                coords_ref_angle = np.array([0.0, 0.0, 1.0])
            
            # Calculate reference vector from ref_bond to ref_angle
            v_ref = coords_ref_angle - coords_ref_bond
            r_ref = np.linalg.norm(v_ref)
            
            if r_ref > 1e-8:
                v_ref_norm = v_ref / r_ref
            else:
                # Degenerate case - use default
                coords[atom_idx] = np.array([1.0, 0.0, 0.0])
                continue
            
            # Rotate reference vector by the bond angle (around Y axis) to get atom 2 direction
            angle_rad = np.deg2rad(angle_deg)
            cos_angle = np.cos(angle_rad)
            sin_angle = np.sin(angle_rad)
            
            v_atom2_x = v_ref_norm[0] * cos_angle + v_ref_norm[2] * sin_angle
            v_atom2_z = -v_ref_norm[0] * sin_angle + v_ref_norm[2] * cos_angle
            
            coords[atom_idx] = coords_ref_bond + bond_length * np.array([v_atom2_x, 0.0, v_atom2_z])
            continue
        
        if array_idx < 0:
            # Atom not in our parameter arrays (shouldn't happen for atoms 3+)
            continue
        
        # Atoms 3+: use general Z-matrix formula
        # Get dihedral override if this atom has a variable dihedral
        dihedral_override = dihedral_map[atom_idx]
        
        # Compute position
        coords[atom_idx] = _compute_atom_position_numba(
            atom_idx=atom_idx,
            bond_ref=bond_refs[array_idx],
            angle_ref=angle_refs[array_idx],
            dihedral_ref=dihedral_refs[array_idx],
            bond_length=bond_lengths[array_idx],
            angle_deg=bond_angles[array_idx],
            dihedral_deg=default_dihedrals[array_idx],
            chirality=chiralities[array_idx],
            coords=coords,
            dihedral_override=dihedral_override
        )
    
    # Compute distance between start and end
    if len(path_atoms) >= 2:
        start_coords = coords[path_atoms[0]]
        end_coords = coords[path_atoms[-1]]
        diff = end_coords - start_coords
        return np.linalg.norm(diff)
    else:
        return 0.0


def rotation_matrix_from_angle(angle_deg: float, axis_type: str = 'y') -> np.ndarray:
    """
    Compute rotation matrix for rotation around a coordinate axis.
    
    Parameters
    ----------
    angle_deg : float
        Rotation angle in degrees
    axis_type : str
        'x', 'y', or 'z'
        
    Returns
    -------
    np.ndarray
        3x3 rotation matrix
    """
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    if axis_type == 'x':
        return np.array([
            [1, 0, 0],
            [0, cos_a, -sin_a],
            [0, sin_a, cos_a]
        ])
    elif axis_type == 'y':
        return np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])
    else:  # 'z'
        return np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])


class AnalyticalDistanceFunction:
    """
    Pre-compiled analytical distance function for a specific atom pair.
    
    This class encapsulates the analytical computation of distance between two
    atoms along a path in Z-matrix space, using forward kinematics.
    
    The implementation uses a hybrid approach: it computes positions incrementally
    along the path using Z-matrix geometry, avoiding full zmatrix-to-cartesian conversion.

    The function takes a dictionary of dihedral values and returns the distance between the two atoms.
    The dictionary keys are the Z-matrix atom indices that have variable dihedrals.
    The dictionary values are the dihedral values in degrees.
    The input dictionary may not contain all the dihedrals, if so, the values of the zmatrix given upon construction are used. If these are also not set, the default value of 0.0 is used.
    The input dictionary may contain dihedral values that are not in the path, if so, they are ignored.

    The function uses the JIT compilation to speed up the computation.
    If the JIT compilation is not available, the function uses the Python implementation.

    The function is thread-safe.
    """
    
    def __init__(self, path_info: Dict, zmatrix: ZMatrix, use_jit: bool = True):
        """
        Initialize analytical distance function from path information.
        
        Parameters
        ----------
        path_info : Dict
            Path information containing:
            - 'bond_lengths': List[float] - bond lengths along path (one per bond)
            - 'bond_angles': List[float] - bond angles along path (one per bond)
            - 'dihedral_indices': List[int] - Z-matrix atom indices that have variable dihedrals.
              These are the atoms whose dihedral angles can be changed to affect the distance.
              Example: If path_atoms=[0,1,2,3] and atom 2 has a variable dihedral, then
              dihedral_indices=[2].
            - 'dihedral_positions': List[int] - Bond positions in the path where each dihedral
              affects the geometry. Each value corresponds to the bond position (0-based) in the
              path that is influenced by changing the corresponding dihedral. The length must
              match dihedral_indices.
              
              For a path [a0, a1, a2, ..., an], bond positions are:
                - Position 0: bond between atoms a0-a1
                - Position 1: bond between atoms a1-a2
                - Position i: bond between atoms path[i]-path[i+1]
              
              Rules for determining dihedral_positions:
              1. If the dihedral atom is IN the path at position j:
                 - The dihedral affects the bond it defines (between the dihedral atom and its
                   bond_ref atom)
                 - Find this bond in the path and use its position
                 - Example: path=[0,1,2,3], atom 2 has dihedral affecting bond 1-2,
                   then dihedral_positions=[1]
              
              2. If the dihedral atom is NOT in the path (branch atom):
                 - The dihedral affects the geometry around its bond_ref atom (which IS in path)
                 - Use the position of the bond AFTER the bond_ref atom in the path
                 - Example: path=[0,1,2,4], atom 3 (branch from atom 2) has dihedral,
                   atom 2 is at path position 2, next bond is 2-4 at position 2,
                   then dihedral_positions=[2]
              
              Example: For path_atoms=[0,1,2,3]:
                - If atom 2 (in path) has a dihedral affecting bond 1-2, then dihedral_positions=[1]
                - If atom 3 (branch from atom 2, not in path) has a dihedral, and atom 2 is at
                  path position 2, then dihedral_positions=[2] (bond 2-3 in path)
            - 'path_atoms': List[int] - Complete sequence of atom indices along the path from
              start to end atom (e.g., [0, 1, 2, 3] for a 4-atom path)
            - 'zmatrix_refs': List[Dict] - Z-matrix reference information for each atom
        zmatrix : ZMatrix
            Z-matrix object for accessing atom information
        use_jit : bool
            Whether to use JIT compilation (if numba is available)
            
        Notes
        -----
        The relationship between dihedral_indices and dihedral_positions:
        - dihedral_indices[i] is the Z-matrix atom index that has a variable dihedral
        - dihedral_positions[i] is the bond position (0-based) in the path where this
          dihedral affects the geometry
        - For a path [a0, a1, a2, ..., an], bond position i corresponds to the bond
          between atoms path[i] and path[i+1]
        
        Important: When a dihedral atom is not in the path (branch atom), its dihedral
        position is determined by finding the bond in the path that comes AFTER its
        bond_ref atom. This is because changing the branch dihedral affects the geometry
        around the path atom, which in turn affects subsequent bonds in the path.
        """
        self.bond_lengths = np.array(path_info['bond_lengths'], dtype=np.float64)
        self.bond_angles = np.array(path_info['bond_angles'], dtype=np.float64)
        self.dihedral_indices = np.array(path_info['dihedral_indices'], dtype=np.int64)
        self.dihedral_positions = path_info['dihedral_positions']
        self.path_atoms = np.array(path_info['path_atoms'], dtype=np.int64)
        self.zmatrix = zmatrix
        self.zmatrix_refs = path_info.get('zmatrix_refs', [])
        
        # Determine if we can use JIT
        self.use_jit = use_jit and NUMBA_AVAILABLE
        
        # Pre-compute numpy arrays for JIT compilation
        self._prepare_jit_data()
        
        # Pre-allocate working arrays (reused for efficiency)
        self.working_coords = {}  # Cache for computed coordinates along path
        
        # Create mapping from dihedral index to position in path
        self.dihedral_to_path_pos = {
            int(dih_idx): pos for dih_idx, pos in 
            zip(self.dihedral_indices, self.dihedral_positions)
        }
    
    def _prepare_jit_data(self):
        """Pre-compute numpy arrays needed for JIT compilation."""
        # Collect all atoms needed: path atoms + all their reference atoms
        atoms_needed = set()
        for atom_idx in self.path_atoms:
            atoms_needed.add(int(atom_idx))
            # Add all reference atoms recursively
            self._collect_reference_atoms(int(atom_idx), atoms_needed)
        
        # Topologically sort atoms (parents before children)
        all_atoms_ordered = self._topological_sort(list(atoms_needed))
        
        # Create mapping from atom index to position in arrays
        max_atom_idx = max(atoms_needed) if atoms_needed else 0
        atom_to_array_idx = np.full(max_atom_idx + 1, -1, dtype=np.int64)
        for i, atom_idx in enumerate(all_atoms_ordered):
            atom_to_array_idx[atom_idx] = i
        
        n_atoms = len(all_atoms_ordered)
        
        # Extract reference indices and default values for all atoms
        bond_refs = np.full(n_atoms, -1, dtype=np.int64)
        angle_refs = np.full(n_atoms, -1, dtype=np.int64)
        dihedral_refs = np.full(n_atoms, -1, dtype=np.int64)
        default_dihedrals = np.zeros(n_atoms, dtype=np.float64)
        chiralities = np.zeros(n_atoms, dtype=np.int64)
        bond_lengths_all = np.zeros(n_atoms, dtype=np.float64)
        bond_angles_all = np.zeros(n_atoms, dtype=np.float64)
        
        # Extract data from zmatrix for each atom
        for i, atom_idx in enumerate(all_atoms_ordered):
            atom = self.zmatrix[atom_idx]
            
            bond_ref = atom.get(ZMatrix.FIELD_BOND_REF)
            angle_ref = atom.get(ZMatrix.FIELD_ANGLE_REF)
            dihedral_ref = atom.get(ZMatrix.FIELD_DIHEDRAL_REF)
            
            bond_refs[i] = int(bond_ref) if bond_ref is not None else -1
            angle_refs[i] = int(angle_ref) if angle_ref is not None else -1
            dihedral_refs[i] = int(dihedral_ref) if dihedral_ref is not None else -1
            
            bond_lengths_all[i] = float(atom.get(ZMatrix.FIELD_BOND_LENGTH, 1.5))
            bond_angles_all[i] = float(atom.get(ZMatrix.FIELD_ANGLE, 109.47))
            default_dihedrals[i] = float(atom.get(ZMatrix.FIELD_DIHEDRAL, 0.0))
            chiralities[i] = int(atom.get(ZMatrix.FIELD_CHIRALITY, 0))
        
        # Store for JIT function
        self.jit_all_atoms = np.array(all_atoms_ordered, dtype=np.int64)
        self.jit_bond_refs = bond_refs
        self.jit_angle_refs = angle_refs
        self.jit_dihedral_refs = dihedral_refs
        self.jit_bond_lengths_all = bond_lengths_all
        self.jit_bond_angles_all = bond_angles_all
        self.jit_default_dihedrals = default_dihedrals
        self.jit_chiralities = chiralities
        self.jit_atom_to_array_idx = atom_to_array_idx
        self.max_atom_idx = max_atom_idx
    
    def _collect_reference_atoms(self, atom_idx: int, collected: set):
        """Recursively collect all reference atoms."""
        if atom_idx < 0 or atom_idx >= len(self.zmatrix):
            return
        
        atom = self.zmatrix[atom_idx]
        bond_ref = atom.get(ZMatrix.FIELD_BOND_REF)
        angle_ref = atom.get(ZMatrix.FIELD_ANGLE_REF)
        dihedral_ref = atom.get(ZMatrix.FIELD_DIHEDRAL_REF)
        
        for ref_idx in [bond_ref, angle_ref, dihedral_ref]:
            if ref_idx is not None and ref_idx >= 0 and ref_idx not in collected:
                collected.add(ref_idx)
                self._collect_reference_atoms(ref_idx, collected)
    
    def _topological_sort(self, atoms: List[int]) -> List[int]:
        """Topologically sort atoms so parents (reference atoms) come before children."""
        # Build dependency graph: atom depends on its reference atoms
        dependencies = {}
        for atom_idx in atoms:
            atom = self.zmatrix[atom_idx]
            deps = []
            for ref_field in [ZMatrix.FIELD_BOND_REF, ZMatrix.FIELD_ANGLE_REF, ZMatrix.FIELD_DIHEDRAL_REF]:
                ref_idx = atom.get(ref_field)
                if ref_idx is not None and ref_idx in atoms:
                    deps.append(ref_idx)
            dependencies[atom_idx] = deps
        
        # Topological sort using Kahn's algorithm
        # in_degree[atom] = number of dependencies (reference atoms) that must be computed first
        in_degree = {}
        for atom_idx in atoms:
            in_degree[atom_idx] = len(dependencies[atom_idx])
        
        # Start with atoms that have no dependencies (all references already computed or not needed)
        queue = [atom for atom in atoms if in_degree[atom] == 0]
        result = []
        
        while queue:
            atom = queue.pop(0)
            result.append(atom)
            # Find atoms that depend on this one (have it as a reference)
            for other_atom in atoms:
                if atom in dependencies[other_atom]:
                    in_degree[other_atom] -= 1
                    if in_degree[other_atom] == 0:
                        queue.append(other_atom)
        
        # If we have cycles or missing dependencies, just return original order
        if len(result) != len(atoms):
            return atoms
        
        return result
    
    def _compute_atom_position(self, atom_idx: int, dihedral_values: Dict[int, float],
                               coords_cache: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Compute position of an atom using Z-matrix geometry.
        
        This follows the same conventions as zmatrix_to_cartesian:
        - Atom 0 at origin
        - Atom 1 along +Z axis
        - Atom 2 in XZ plane
        - Atoms 3+ using general Z-matrix formula
        
        Parameters
        ----------
        atom_idx : int
            Atom index (0-based)
        dihedral_values : Dict[int, float]
            Current dihedral values (degrees)
        coords_cache : Dict[int, np.ndarray]
            Cache of already-computed coordinates
        
        Returns
        -------
        np.ndarray
            3D position of atom
        """
        if atom_idx in coords_cache:
            return coords_cache[atom_idx]
        
        atom = self.zmatrix[atom_idx]
        
        # Special handling for first 3 atoms (same as zmatrix_to_cartesian)
        if atom_idx == 0:
            # Atom 0: always at origin
            coords = np.array([0.0, 0.0, 0.0])
            coords_cache[atom_idx] = coords
            return coords
        
        if atom_idx == 1:
            # Atom 1: along +Z axis
            bond_length = atom.get(ZMatrix.FIELD_BOND_LENGTH, 1.5)
            coords = np.array([0.0, 0.0, bond_length])
            coords_cache[atom_idx] = coords
            return coords
        
        if atom_idx == 2:
            # Atom 2: place in XZ plane (same logic as zmatrix_to_cartesian)
            ref_bond = atom.get(ZMatrix.FIELD_BOND_REF)
            ref_angle = atom.get(ZMatrix.FIELD_ANGLE_REF)
            bond_length = atom.get(ZMatrix.FIELD_BOND_LENGTH, 1.5)
            angle_deg = atom.get(ZMatrix.FIELD_ANGLE, 109.47)
            
            # Get reference coordinates
            if ref_bond is not None:
                coords_ref_bond = self._compute_atom_position(ref_bond, dihedral_values, coords_cache)
            else:
                coords_ref_bond = np.array([0.0, 0.0, 0.0])
            
            if ref_angle is not None:
                coords_ref_angle = self._compute_atom_position(ref_angle, dihedral_values, coords_cache)
            else:
                coords_ref_angle = np.array([0.0, 0.0, 1.0])
            
            # Calculate reference vector from ref_bond to ref_angle
            v_ref = coords_ref_angle - coords_ref_bond
            r_ref = np.linalg.norm(v_ref)
            
            if r_ref > 1e-8:
                v_ref_norm = v_ref / r_ref
            else:
                raise ValueError(
                    f"Degenerate reference geometry: The distance between atoms {ref_bond} "
                    f"and {ref_angle} is too small (≈{r_ref:.2e} Å) for atom 2 placement."
                )
            
            # Rotate reference vector by the bond angle (around Y axis) to get atom 2 direction
            angle_rad = np.deg2rad(angle_deg)
            cos_angle = np.cos(angle_rad)
            sin_angle = np.sin(angle_rad)
            
            v_atom2_x = v_ref_norm[0] * cos_angle + v_ref_norm[2] * sin_angle
            v_atom2_z = -v_ref_norm[0] * sin_angle + v_ref_norm[2] * cos_angle
            
            coords = coords_ref_bond + bond_length * np.array([v_atom2_x, 0.0, v_atom2_z])
            coords_cache[atom_idx] = coords
            return coords
        
        # Atoms 3+: use general Z-matrix formula
        # Get reference atoms
        ia = atom.get(ZMatrix.FIELD_BOND_REF)
        ib = atom.get(ZMatrix.FIELD_ANGLE_REF)
        ic = atom.get(ZMatrix.FIELD_DIHEDRAL_REF)
        
        bond_length = atom.get(ZMatrix.FIELD_BOND_LENGTH, 1.5)
        angle_deg = atom.get(ZMatrix.FIELD_ANGLE, 109.47)
        dihedral_deg = atom.get(ZMatrix.FIELD_DIHEDRAL, 0.0)
        chirality = atom.get(ZMatrix.FIELD_CHIRALITY, 0)
        
        # Override dihedral if it's in the variable dihedrals
        if atom_idx in dihedral_values:
            dihedral_deg = dihedral_values[atom_idx]
        
        # Get reference coordinates (compute recursively if needed)
        if ia is not None:
            coords_ia = self._compute_atom_position(ia, dihedral_values, coords_cache)
        else:
            raise ValueError(f"Atom {atom_idx} must have a bond reference")
        
        if ib is not None:
            coords_ib = self._compute_atom_position(ib, dihedral_values, coords_cache)
        else:
            raise ValueError(f"Atom {atom_idx} must have an angle reference")
        
        if ic is not None:
            coords_ic = self._compute_atom_position(ic, dihedral_values, coords_cache)
        else:
            raise ValueError(f"Atom {atom_idx} must have a dihedral reference")
        
        # Compute position using Z-matrix geometry (similar to zmatrix_to_cartesian)
        angle_rad = np.deg2rad(angle_deg)
        dihedral_rad = np.deg2rad(-dihedral_deg)  # Negate for XZ plane convention
        
        sin1 = np.sin(angle_rad)
        cos1 = np.cos(angle_rad)
        sin2 = np.sin(dihedral_rad)
        cos2 = np.cos(dihedral_rad)
        
        if chirality == 0:
            # Dihedral case
            xab = coords_ia - coords_ib
            rab = np.linalg.norm(xab)
            if rab > 1e-8:
                xab = xab / rab
            else:
                xab = np.array([1.0, 0.0, 0.0])
            
            xbc = coords_ib - coords_ic
            rbc = np.linalg.norm(xbc)
            if rbc > 1e-8:
                xbc = xbc / rbc
            else:
                xbc = np.array([0.0, 1.0, 0.0])
            
            # Cross products
            xt = np.cross(xab, xbc)
            sine = np.sqrt(max(1.0 - np.dot(xab, xbc)**2, 1e-8))
            if sine > 1e-8:
                xt = xt / sine
            else:
                xt = np.array([0.0, 0.0, 1.0])
            
            xu = np.cross(xab, xt)
            
            coords = coords_ia + bond_length * (
                xu * sin1 * cos2 + 
                xt * sin1 * sin2 - 
                xab * cos1
            )
        else:
            # Chirality case (same as zmatrix_to_cartesian)
            xba = coords_ib - coords_ia
            rba = np.linalg.norm(xba)
            if rba > 1e-8:
                xba = xba / rba
            else:
                xba = np.array([1.0, 0.0, 0.0])
            
            xac = coords_ia - coords_ic
            rac = np.linalg.norm(xac)
            if rac > 1e-8:
                xac = xac / rac
            else:
                xac = np.array([0.0, 1.0, 0.0])
            
            # Check for linearity
            cosine = np.dot(xba, xac)
            if abs(abs(cosine) - 1.0) < 1e-6:
                angle_deg = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
                raise ValueError(
                    f"Linearity detected: Cannot compute chirality angle for atom {atom_idx}. "
                    f"The geometry around atoms {ib}-{ia}-{ic} is linear (angle ≈ {angle_deg:.2f}°)."
                )
            
            # Cross product order matters for handedness in XZ plane convention
            xt = np.cross(xac, xba)
            sine2 = max(1.0 - cosine**2, 1e-8)
            
            a = (-cos2 - cosine * cos1) / sine2
            b = (cos1 + cosine * cos2) / sine2
            c = (1.0 + a * cos2 - b * cos1) / sine2
            
            eps = 1e-8
            if c > eps:
                c = chirality * np.sqrt(c)
            elif c < -eps:
                c_denom = np.linalg.norm(a * xac + b * xba)
                if c_denom > 1e-8:
                    a = a / c_denom
                    b = b / c_denom
                c = 0.0
            else:
                c = 0.0
            
            coords = coords_ia + bond_length * (a * xac + b * xba + c * xt)
        
        coords_cache[atom_idx] = coords
        return coords
    
    def __call__(self, dihedral_values: Dict[int, float]) -> float:
        """
        Compute distance for given dihedral values.
        
        Parameters
        ----------
        dihedral_values : Dict[int, float]
            Dictionary mapping Z-matrix dihedral indices to their values (degrees)
            Only dihedrals on the path need to be provided.
        
        Returns
        -------
        float
            Distance between the two atoms (Angstroms)
        """
        if self.use_jit:
            return self._compute_distance_jit(dihedral_values)
        else:
            return self._compute_distance_python(dihedral_values)
    
    def _compute_distance_jit(self, dihedral_values: Dict[int, float]) -> float:
        """Compute distance using JIT-compiled function."""
        # Convert dihedral values dict to numpy array (ordered by dihedral_indices)
        dihedral_array = np.zeros(len(self.dihedral_indices), dtype=np.float64)
        for i, dih_idx in enumerate(self.dihedral_indices):
            if int(dih_idx) in dihedral_values:
                dihedral_array[i] = float(dihedral_values[int(dih_idx)])
            else:
                # Use default from zmatrix
                atom = self.zmatrix[int(dih_idx)]
                dihedral_array[i] = float(atom.get(ZMatrix.FIELD_DIHEDRAL, 0.0))
        
        # Call JIT-compiled function
        return _analytical_distance_numba(
            path_atoms=self.path_atoms,
            all_atom_indices=self.jit_all_atoms,
            bond_refs=self.jit_bond_refs,
            angle_refs=self.jit_angle_refs,
            dihedral_refs=self.jit_dihedral_refs,
            bond_lengths=self.jit_bond_lengths_all,
            bond_angles=self.jit_bond_angles_all,
            default_dihedrals=self.jit_default_dihedrals,
            chiralities=self.jit_chiralities,
            dihedral_indices=self.dihedral_indices,
            dihedral_values=dihedral_array,
            atom_to_array_idx=self.jit_atom_to_array_idx,
            max_atoms=self.max_atom_idx
        )
    
    def _compute_distance_python(self, dihedral_values: Dict[int, float]) -> float:
        """Compute distance using Python implementation (fallback)."""
        # Filter dihedral_values to only include variable dihedrals (for consistency with JIT)
        filtered_dihedral_values = {}
        for dih_idx in self.dihedral_indices:
            dih_idx_int = int(dih_idx)
            if dih_idx_int in dihedral_values:
                filtered_dihedral_values[dih_idx_int] = float(dihedral_values[dih_idx_int])
            else:
                # Use default from zmatrix
                atom = self.zmatrix[dih_idx_int]
                filtered_dihedral_values[dih_idx_int] = float(atom.get(ZMatrix.FIELD_DIHEDRAL, 0.0))
        
        # Clear coordinate cache
        coords_cache = {}
        
        # Compute positions of start and end atoms
        start_atom = int(self.path_atoms[0])
        end_atom = int(self.path_atoms[-1])
        
        coords_start = self._compute_atom_position(start_atom, filtered_dihedral_values, coords_cache)
        coords_end = self._compute_atom_position(end_atom, filtered_dihedral_values, coords_cache)
        
        # Return distance
        return np.linalg.norm(coords_end - coords_start)
    
    def gradient(self, dihedral_values: Dict[int, float], 
                 dihedral_idx: int, 
                 eps: float = 1.0) -> float:
        """
        Compute gradient of distance w.r.t. a specific dihedral using finite differences.
        
        Parameters
        ----------
        dihedral_values : Dict[int, float]
            Current dihedral values
        dihedral_idx : int
            Index of dihedral to compute gradient for
        eps : float
            Step size for finite differences (degrees)
        
        Returns
        -------
        float
            ∂distance/∂dihedral (Angstroms/degree)
        """
        if dihedral_idx not in dihedral_values:
            return 0.0
        
        # Forward difference
        dihedral_values_plus = dihedral_values.copy()
        dihedral_values_plus[dihedral_idx] = dihedral_values[dihedral_idx] + eps
        dist_plus = self(dihedral_values_plus)
        
        # Backward difference
        dihedral_values_minus = dihedral_values.copy()
        dihedral_values_minus[dihedral_idx] = dihedral_values[dihedral_idx] - eps
        dist_minus = self(dihedral_values_minus)
        
        # Central difference
        gradient = (dist_plus - dist_minus) / (2 * eps)
        
        return gradient


class AnalyticalDistanceFactory:
    """
    Factory for creating and managing analytical distance functions.
    
    This class can create analytical distance functions for any pair of atoms
    in a Z-matrix, managing them as reusable objects.
    """
    
    def __init__(self, zmatrix: ZMatrix, topology=None):
        """
        Initialize factory with Z-matrix.
        
        Parameters
        ----------
        zmatrix : ZMatrix
            Z-matrix representation of the molecule
        topology : openmm.app.Topology, optional
            OpenMM topology for complete bond connectivity
        """
        self.zmatrix = zmatrix
        self.topology = topology
        
        # Cache of distance functions: (atom1, atom2) -> AnalyticalDistanceFunction
        self._distance_function_cache: Dict[Tuple[int, int], AnalyticalDistanceFunction] = {}
        
        # Build bond graph for path finding
        self.graph = self._build_bond_graph()
    
    def _build_bond_graph(self) -> Dict[int, List[int]]:
        """
        Build bond connectivity graph.
        
        Returns
        -------
        Dict[int, List[int]]
            Adjacency list representation (0-based indices)
        """
        num_atoms = len(self.zmatrix)
        graph = {i: [] for i in range(num_atoms)}
        
        if self.topology is not None:
            # Use complete topology (includes all bonds)
            for bond in self.topology.bonds():
                graph[bond.atom1.index].append(bond.atom2.index)
                graph[bond.atom2.index].append(bond.atom1.index)
        else:
            # Fall back to Z-matrix (only tree structure bonds)
            for i in range(num_atoms):
                if i == 0:
                    continue
                atom = self.zmatrix[i]
                bond_ref = atom.get(ZMatrix.FIELD_BOND_REF)
                if bond_ref is not None:
                    graph[i].append(bond_ref)
                    graph[bond_ref].append(i)
        
        return graph
    
    def _find_path_bfs(self, start: int, end: int) -> Optional[List[int]]:
        """
        Find shortest path between two atoms using BFS.
        
        Parameters
        ----------
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
        
        if start not in self.graph or end not in self.graph:
            return None
        
        if start == end:
            return [start]
        
        visited = {start}
        queue = deque([(start, [start])])
        
        while queue:
            node, path = queue.popleft()
            
            if node not in self.graph:
                continue
            
            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    
                    if neighbor == end:
                        return new_path
                    
                    queue.append((neighbor, new_path))
        
        return None
    
    def _extract_path_info(self, path: List[int], 
                          rotatable_indices: Optional[List[int]] = None) -> Dict:
        """
        Extract path information needed for analytical distance computation.
        
        Parameters
        ----------
        path : List[int]
            Path of atom indices (0-based)
        rotatable_indices : Optional[List[int]]
            List of rotatable dihedral indices. If None, all dihedrals are considered.
        
        Returns
        -------
        Dict
            Path information containing:
            - 'bond_lengths': bond lengths along path
            - 'bond_angles': bond angles along path
            - 'dihedral_indices': Z-matrix indices of dihedrals on path
            - 'dihedral_positions': position in path for each dihedral
            - 'path_atoms': the path itself
            - 'zmatrix_refs': Z-matrix reference information for each atom
        """
        if len(path) < 2:
            raise ValueError("Path must contain at least 2 atoms")
        
        bond_lengths = []
        bond_angles = []
        dihedral_indices = []
        dihedral_positions = []
        zmatrix_refs = []
        
        # Extract information for each bond along the path
        for i in range(len(path) - 1):
            atom_idx = path[i + 1]  # Atom at position i+1 in path
            atom = self.zmatrix[atom_idx]
            
            # Bond length
            if ZMatrix.FIELD_BOND_LENGTH in atom:
                bond_lengths.append(atom[ZMatrix.FIELD_BOND_LENGTH])
            else:
                # Fallback: compute from previous atom if needed
                bond_lengths.append(1.5)  # Default
            
            # Bond angle (if available)
            if ZMatrix.FIELD_ANGLE in atom:
                bond_angles.append(atom[ZMatrix.FIELD_ANGLE])
            else:
                bond_angles.append(109.47)  # Default tetrahedral
            
            # Dihedral (if this atom has a dihedral and it's rotatable)
            # Note: For path [a0, a1, a2, ...], when processing bond at position i,
            # we're looking at atom path[i+1]. If this atom has a variable dihedral,
            # it affects the bond at position i (between path[i] and path[i+1]).
            if ZMatrix.FIELD_DIHEDRAL in atom:
                chirality = atom.get(ZMatrix.FIELD_CHIRALITY, 0)
                # Only consider true dihedrals (chirality == 0) as rotatable
                if chirality == 0:
                    if rotatable_indices is None or atom_idx in rotatable_indices:
                        dihedral_indices.append(atom_idx)  # Z-matrix atom index
                        dihedral_positions.append(i)  # Bond position in path (0-based)
            
            # Store Z-matrix reference information
            zmatrix_refs.append({
                'bond_ref': atom.get(ZMatrix.FIELD_BOND_REF),
                'angle_ref': atom.get(ZMatrix.FIELD_ANGLE_REF),
                'dihedral_ref': atom.get(ZMatrix.FIELD_DIHEDRAL_REF),
                'chirality': atom.get(ZMatrix.FIELD_CHIRALITY, 0)
            })
        
        # Second pass: Check for branch atoms (not in path) with rotatable dihedrals
        # that affect the path geometry. A branch atom affects the path if its
        # bond_ref is in the path.
        path_set = set(path)
        for atom_idx in range(len(self.zmatrix)):
            # Skip atoms already in path (handled in first pass)
            if atom_idx in path_set:
                continue
            
            atom = self.zmatrix[atom_idx]
            bond_ref = atom.get(ZMatrix.FIELD_BOND_REF)
            
            # Check if this is a branch atom (not in path) whose bond_ref is in path
            if bond_ref is not None and bond_ref in path_set:
                # Check if this atom has a rotatable dihedral
                if ZMatrix.FIELD_DIHEDRAL in atom:
                    chirality = atom.get(ZMatrix.FIELD_CHIRALITY, 0)
                    # Only consider true dihedrals (chirality == 0) as rotatable
                    if chirality == 0:
                        if rotatable_indices is None or atom_idx in rotatable_indices:
                            # Find the position of bond_ref in the path
                            bond_ref_pos = path.index(bond_ref)
                            # The branch dihedral affects the bond AFTER bond_ref in the path
                            # If bond_ref is the last atom in path, use the last bond position
                            if bond_ref_pos < len(path) - 1:
                                dihedral_position = bond_ref_pos
                            else:
                                # bond_ref is the last atom, use the last bond position
                                dihedral_position = len(path) - 2
                            
                            dihedral_indices.append(atom_idx)  # Z-matrix atom index
                            dihedral_positions.append(dihedral_position)  # Bond position in path (0-based)
        
        return {
            'bond_lengths': bond_lengths,
            'bond_angles': bond_angles,
            'dihedral_indices': dihedral_indices,
            'dihedral_positions': dihedral_positions,
            'path_atoms': path,
            'zmatrix_refs': zmatrix_refs
        }
    
    def get_distance_function(self, atom1: int, atom2: int,
                              rotatable_indices: Optional[List[int]] = None,
                              force_recompute: bool = False) -> Optional[AnalyticalDistanceFunction]:
        """
        Get or create analytical distance function for a pair of atoms.
        
        Parameters
        ----------
        atom1 : int
            First atom index (0-based)
        atom2 : int
            Second atom index (0-based)
        rotatable_indices : Optional[List[int]]
            List of rotatable dihedral indices. If None, all dihedrals are considered.
        force_recompute : bool
            If True, recompute even if cached
        
        Returns
        -------
        Optional[AnalyticalDistanceFunction]
            Analytical distance function, or None if no path exists
        """
        # Use canonical ordering for cache key
        cache_key = (min(atom1, atom2), max(atom1, atom2))
        
        # Check cache
        if not force_recompute and cache_key in self._distance_function_cache:
            return self._distance_function_cache[cache_key]
        
        # Find path between atoms
        path = self._find_path_bfs(atom1, atom2)
        if path is None:
            return None
        
        # Extract path information
        try:
            path_info = self._extract_path_info(path, rotatable_indices)
        except Exception as e:
            print(f"Warning: Failed to extract path info for atoms {atom1}-{atom2}: {e}")
            return None
        
        # Create analytical distance function
        try:
            distance_func = AnalyticalDistanceFunction(path_info, self.zmatrix)
            self._distance_function_cache[cache_key] = distance_func
            return distance_func
        except Exception as e:
            print(f"Warning: Failed to create distance function for atoms {atom1}-{atom2}: {e}")
            return None
    
    def get_all_distance_functions(self, atom_pairs: List[Tuple[int, int]],
                                   rotatable_indices: Optional[List[int]] = None) -> Dict[Tuple[int, int], Optional[AnalyticalDistanceFunction]]:
        """
        Get or create analytical distance functions for multiple atom pairs.
        
        Parameters
        ----------
        atom_pairs : List[Tuple[int, int]]
            List of (atom1, atom2) pairs (0-based indices)
        rotatable_indices : Optional[List[int]]
            List of rotatable dihedral indices
        
        Returns
        -------
        Dict[Tuple[int, int], Optional[AnalyticalDistanceFunction]]
            Dictionary mapping atom pairs to their distance functions
            (None if no path exists for a pair)
        """
        result = {}
        for atom1, atom2 in atom_pairs:
            cache_key = (min(atom1, atom2), max(atom1, atom2))
            if cache_key not in result:
                result[cache_key] = self.get_distance_function(
                    atom1, atom2, rotatable_indices
                )
        return result
    
    def clear_cache(self):
        """Clear the cache of distance functions."""
        self._distance_function_cache.clear()
    
    def get_cached_functions(self) -> Dict[Tuple[int, int], AnalyticalDistanceFunction]:
        """
        Get all currently cached distance functions.
        
        Returns
        -------
        Dict[Tuple[int, int], AnalyticalDistanceFunction]
            Dictionary of cached distance functions
        """
        return self._distance_function_cache.copy()

