#!/usr/bin/env python3
"""
Coordinate Conversion Utilities

This module provides utilities for converting between internal coordinates (Z-matrix)
and Cartesian coordinates. 
The format of the Z-matrix follows the conventions of the TINKER program.

Conventions:
    - Atom 1 at origin
    - Atom 2 along +Z axis
    - Atom 3 in XZ plane
    - Atom 4+: use TINKER conventions
    - Chirality: 0 for true dihedral, +1/-1 for chiral angle
    - Dihedral: dihedral angle in degrees (or bond angle for chirality)
    - Bond: bond length in Angstroms
    - Angle: bond angle in degrees

Main Functions:
    zmatrix_to_cartesian: Convert Z-matrix to Cartesian coordinates
    cartesian_to_zmatrix: Convert Cartesian coordinates to Z-matrix
    apply_torsions: Modify torsional angles in Z-matrix
    extract_torsions: Extract specific torsional angles from Cartesian coordinates
    generate_zmatrix: Generate Z-matrix from Cartesian coordinates and bond connectivity
    
Helper Functions:
    compute_chirality_sign: Determine chirality sign from Cartesian coordinates
    _calc_distance: Calculate distance between two points
    _calc_angle: Calculate angle between three points
    _calc_dihedral: Calculate dihedral angle between four points (atan2 method)
    _get_atomic_number: Get atomic number from element symbol
    _get_first_ref_atom_id: Get first reference atom for Z-matrix construction
    _get_second_ref_atom_id: Get second reference atom for Z-matrix construction
    _get_third_ref_atom_id: Get third reference atom and determine chirality
    _count_predefined_neighbours: Count predefined neighbors of a reference atom
    _is_torsion_used: Check if a torsion is already used in Z-matrix
    
"""

import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
import copy

from openmm.app import Element

from .ZMatrix import ZMatrix


# =============================================================================
# Geometry Calculation Helpers
# =============================================================================

def _calc_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Calculate distance between two points.
    
    Parameters
    ----------
    p1, p2 : np.ndarray
        3D points
        
    Returns
    -------
    float
        Distance in same units as input
    """
    return np.linalg.norm(p2 - p1)


def _calc_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Calculate angle at p2 between p1-p2-p3.
    
    Parameters
    ----------
    p1, p2, p3 : np.ndarray
        3D points
        
    Returns
    -------
    float
        Angle in degrees
    """
    v1 = p1 - p2
    v2 = p3 - p2
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    # Handle degenerate cases (zero-length vectors)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    
    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def _calc_dihedral(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    """
    Calculate dihedral angle for atoms p1-p2-p3-p4. Sign convention: positive when p1-p2-p3-p4 is a clockwise rotation.
    
    Parameters
    ----------
    p1, p2, p3, p4 : np.ndarray
        3D points defining the dihedral
        
    Returns
    -------
    float
        Dihedral angle in degrees
    """
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    
    norm_n1 = np.linalg.norm(n1)
    norm_n2 = np.linalg.norm(n2)
    norm_b2 = np.linalg.norm(b2)
    
    # Check for degenerate cases (collinear atoms or zero-length bonds)
    if norm_n1 > 1e-8 and norm_n2 > 1e-8 and norm_b2 > 1e-8:
        n1 = n1 / norm_n1
        n2 = n2 / norm_n2
        m1 = np.cross(n1, b2 / norm_b2)
        x = np.dot(n1, n2)
        y = np.dot(m1, n2)
        return -np.degrees(np.arctan2(y, x))
    return 0.0


def zmatrix_to_cartesian(zmatrix: ZMatrix) -> np.ndarray:
    """
    Convert Z-matrix to Cartesian coordinates.
    
    Conventions for internal coordinate conversion:
    - Atom 1 at origin
    - Atom 2 along +Z axis
    - Atom 3 in XZ plane
    
    This function detects and raises errors when linear geometries emerge during
    the conversion process. Linear geometries (angles ≈ 0° or ≈ 180°) make
    dihedral angles undefined, which causes the conversion to fail.
    
    Parameters
    ----------
    zmatrix : ZMatrix
        Z-matrix representation with keys (all indices must be 0-based):
        - 'bond_ref': reference atom for bond (0-based index)
        - 'bond_length': bond length in Angstroms
        - 'angle_ref': reference atom for angle (0-based index)
        - 'angle': bond angle in degrees (must not be 0° or 180°)
        - 'dihedral_ref': reference atom for dihedral (0-based index)
        - 'dihedral': dihedral angle in degrees (or bond angle for chirality)
        - 'chirality': 0 for true dihedral, +1/-1 for chiral angle
        
        Note: This function expects 0-based indexing. Input conversion from
        1-based to 0-based should be done before calling this function.
    
    Returns
    -------
    np.ndarray
        Nx3 array of Cartesian coordinates in Angstroms
    
    Raises
    ------
    ValueError
        If linear geometries are detected (angles ≈ 0° or ≈ 180°) that make
        dihedral or chirality angles undefined.
    """
    # Use list interface for iteration
    atoms_list = zmatrix.to_list()
    num_atoms = len(atoms_list)
    coords = np.zeros((num_atoms, 3))
    
    # Atom 1: place at origin
    coords[0] = [0.0, 0.0, 0.0]
    
    if num_atoms < 2:
        return coords
    
    # Atom 2: place along +Z axis
    coords[1] = [0.0, 0.0, atoms_list[1][ZMatrix.FIELD_BOND_LENGTH]]
    
    if num_atoms < 3:
        return coords
    
    # Atom 3: place in XZ plane
    atom3 = atoms_list[2]
    ref_bond = atom3[ZMatrix.FIELD_BOND_REF]  # Already 0-based internally
    ref_angle = atom3[ZMatrix.FIELD_ANGLE_REF]  # Already 0-based internally
    
    bond_length = atom3[ZMatrix.FIELD_BOND_LENGTH]
    angle_deg = atom3[ZMatrix.FIELD_ANGLE]
    angle_rad = np.deg2rad(angle_deg)
    
    # Check for linearity in Z-matrix definition (angle = 0° or 180°)
    if abs(angle_deg) < 1e-6 or abs(abs(angle_deg) - 180.0) < 1e-6:
        raise ValueError(
            f"Linearity detected: Atom 3 (0-based: 2) has a linear geometry. "
            f"The bond angle is {angle_deg:.2f}° (should be between 0° and 180°, exclusive). "
            f"Linear geometries (0° or 180°) are not allowed in Z-matrix definitions. "
            f"Consider adjusting the Z-matrix definition to avoid linear arrangements."
        )
    
    # Place atom 3 in XZ plane
    # The angle is measured at ref_bond between ref_angle and atom 3
    # Calculate reference vector from ref_bond to ref_angle
    v_ref = coords[ref_angle] - coords[ref_bond]
    r_ref = np.linalg.norm(v_ref)
    
    if r_ref > 1e-8:
        v_ref_norm = v_ref / r_ref
    else:
        raise ValueError(
            f"Degenerate reference geometry: The distance between atoms {ref_bond+1} (0-based: {ref_bond}) "
            f"and {ref_angle+1} (0-based: {ref_angle}) is too small (≈{r_ref:.2e} Å) for atom 3 placement. "
            f"Consider adjusting the Z-matrix definition."
        )
    
    # Rotate reference vector by the bond angle (around Y axis) to get atom 3 direction
    # Rotation matrix around Y: [cos θ, 0, sin θ; 0, 1, 0; -sin θ, 0, cos θ]
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    
    v_atom3_x = v_ref_norm[0] * cos_angle + v_ref_norm[2] * sin_angle
    v_atom3_z = -v_ref_norm[0] * sin_angle + v_ref_norm[2] * cos_angle
    
    coords[2] = coords[ref_bond] + bond_length * np.array([v_atom3_x, 0.0, v_atom3_z])
    
    # Atoms 4+
    for i in range(3, num_atoms):
        atom = atoms_list[i]
        ia = atom[ZMatrix.FIELD_BOND_REF]  # Already 0-based internally
        ib = atom[ZMatrix.FIELD_ANGLE_REF]  # Already 0-based internally
        ic = atom[ZMatrix.FIELD_DIHEDRAL_REF]  # Already 0-based internally
        
        bond_length = atom[ZMatrix.FIELD_BOND_LENGTH]
        angle_rad = np.deg2rad(atom[ZMatrix.FIELD_ANGLE])
        # Negate dihedral for XZ plane convention (opposite sign from XY plane)
        dihedral_rad = np.deg2rad(-atom[ZMatrix.FIELD_DIHEDRAL])
        chiral = atom.get(ZMatrix.FIELD_CHIRALITY, 0)
        
        sin1 = np.sin(angle_rad)
        cos1 = np.cos(angle_rad)
        sin2 = np.sin(dihedral_rad)
        cos2 = np.cos(dihedral_rad)
        
        if chiral == 0:
            # Dihedral case
            xab = coords[ia] - coords[ib]
            rab = np.linalg.norm(xab)
            xab = xab / rab if rab > 1e-8 else xab
            
            xbc = coords[ib] - coords[ic]
            rbc = np.linalg.norm(xbc)
            xbc = xbc / rbc if rbc > 1e-8 else xbc
            
            # Check for linearity: if vectors are parallel (angle ~0° or ~180°), dihedral is undefined
            cosine = np.dot(xab, xbc)
            # cosine close to 1.0 means angle ~0°, cosine close to -1.0 means angle ~180°
            if abs(abs(cosine) - 1.0) < 1e-6:
                angle_deg = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
                raise ValueError(
                    f"Linearity detected: Cannot compute dihedral angle for atom {i+1} (0-based: {i}). "
                    f"The geometry around atoms {ib+1}-{ia+1}-{ic+1} (0-based: {ib}-{ia}-{ic}) "
                    f"is linear (angle ≈ {angle_deg:.2f}°). Dihedral angles are undefined for linear geometries. "
                    f"Consider adjusting the Z-matrix definition or the current geometry to avoid linear arrangements."
                )
            
            # Keep original cross product for xt
            xt = np.cross(xab, xbc)
            sine = np.sqrt(max(1.0 - cosine**2, 1e-8))
            xt = xt / sine if sine > 1e-8 else xt
            
            # Flip xu cross product order for XZ plane convention
            xu = np.cross(xab, xt)
            
            coords[i] = coords[ia] + bond_length * (
                xu * sin1 * cos2 + 
                xt * sin1 * sin2 - 
                xab * cos1
            )
        else:
            # Chirality case (bond angle instead of dihedral)
            xba = coords[ib] - coords[ia]
            rba = np.linalg.norm(xba)
            xba = xba / rba if rba > 1e-8 else xba
            
            xac = coords[ia] - coords[ic]
            rac = np.linalg.norm(xac)
            xac = xac / rac if rac > 1e-8 else xac
            
            # Check for linearity: if vectors are parallel (angle ~0° or ~180°), chirality is undefined
            cosine = np.dot(xba, xac)
            # cosine close to 1.0 means angle ~0°, cosine close to -1.0 means angle ~180°
            if abs(abs(cosine) - 1.0) < 1e-6:
                angle_deg = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
                raise ValueError(
                    f"Linearity detected: Cannot compute chirality angle for atom {i+1} (0-based: {i}). "
                    f"The geometry around atoms {ib+1}-{ia+1}-{ic+1} (0-based: {ib}-{ia}-{ic}) "
                    f"is linear (angle ≈ {angle_deg:.2f}°). Chirality angles are undefined for linear geometries. "
                    f"Consider adjusting the Z-matrix definition or the current geometry to avoid linear arrangements."
                )
            
            # Cross product order matters for handedness in XZ plane convention
            xt = np.cross(xac, xba)
            sine2 = max(1.0 - cosine**2, 1e-8)
            
            a = (-cos2 - cosine * cos1) / sine2
            b = (cos1 + cosine * cos2) / sine2
            c = (1.0 + a * cos2 - b * cos1) / sine2
            
            eps = 1e-8
            if c > eps:
                c = chiral * np.sqrt(c)
            elif c < -eps:
                c_denom = np.linalg.norm(a * xac + b * xba)
                a = a / c_denom if c_denom > 1e-8 else a
                b = b / c_denom if c_denom > 1e-8 else b
                c = 0.0
            else:
                c = 0.0
            
            coords[i] = coords[ia] + bond_length * (a * xac + b * xba + c * xt)
    
    return coords


def compute_chirality_sign(coords: np.ndarray, atom_idx: int, 
                           bond_ref_idx: int, angle_ref_idx: int, dihedral_ref_idx: int) -> int:
    """
    Compute the chirality sign for the secondsry angle needed to define the position of the 
    first atom index given those of the second, third, and forth.
    
    Parameters
    ----------
    coords : np.ndarray
        Nx3 array of all Cartesian coordinates in Angstroms
    atom_idx : int
        Index of the atom whose chirality we're determining
    bond_ref_idx : int
        Index of the bond reference atom
    angle_ref_idx : int
        Index of the angle reference atom
    dihedral_ref_idx : int
        Index of the dihedral reference atom
        
    Returns
    -------
    int
        Chirality sign: +1 or -1
    """

    chirality = 1
    dihedral = _calc_dihedral(
        coords[atom_idx],
        coords[bond_ref_idx],
        coords[angle_ref_idx],
        coords[dihedral_ref_idx]
    )
    if dihedral > 0:
        chirality=-1

    return chirality

def apply_torsions(base_zmatrix: ZMatrix, rotatable_indices: List[int],
                  torsion_values: np.ndarray) -> ZMatrix:
    """
    Apply torsional angle changes to Z-matrix.
    
    Parameters
    ----------
    base_zmatrix : ZMatrix
        Base Z-matrix structure
    rotatable_indices : List[int]
        Indices of rotatable atoms
    torsion_values : np.ndarray
        New dihedral values in degrees
        
    Returns
    -------
    ZMatrix
        Modified Z-matrix
    """
    new_zmatrix = base_zmatrix.copy()
    for idx, new_dihedral in zip(rotatable_indices, torsion_values):
        new_zmatrix[idx][ZMatrix.FIELD_DIHEDRAL] = new_dihedral
    return new_zmatrix


def extract_torsions(coords: np.ndarray, zmatrix: ZMatrix,
                    rotatable_indices: List[int]) -> np.ndarray:
    """
    Extract dihedral angles from Cartesian coordinates.
    
    Parameters
    ----------
    coords : np.ndarray
        Nx3 Cartesian coordinates in Angstroms
    zmatrix : ZMatrix
        Z-matrix with reference atoms (0-based indices)
    rotatable_indices : List[int]
        Indices of rotatable atoms (0-based)
        
    Returns
    -------
    np.ndarray
        Extracted dihedral angles in degrees
    """
    torsions = []
    for idx in rotatable_indices:
        atom = zmatrix[idx]
        ia = atom[ZMatrix.FIELD_BOND_REF]
        ib = atom[ZMatrix.FIELD_ANGLE_REF]
        ic = atom[ZMatrix.FIELD_DIHEDRAL_REF]
        i = idx
        
        dihedral = _calc_dihedral(coords[ic], coords[ib], coords[ia], coords[i])
        torsions.append(dihedral)
    
    return np.array(torsions)


def cartesian_to_zmatrix(coords: np.ndarray, zmatrix: ZMatrix) -> ZMatrix:
    """
    Convert Cartesian coordinates to Z-matrix (internal coordinates).
    
    Recalculates internal coordinates (bond lengths, angles, dihedrals) from
    Cartesian coordinates while preserving the original reference atom definitions.
    Properly handles chirality.
    
    Parameters
    ----------
    coords : np.ndarray
        Nx3 Cartesian coordinates in Angstroms
    zmatrix : ZMatrix
        Z-matrix template with reference atoms (defines connectivity)
        
    Returns
    -------
    ZMatrix
        Updated Z-matrix with recalculated internal coordinates
    """
    new_zmatrix = zmatrix.copy()
    
    # Update internal coordinates for all atoms (except first one)
    for i in range(1, len(new_zmatrix)):
        atom = new_zmatrix[i]
        
        # Recalculate bond length
        bond_ref_idx = atom[ZMatrix.FIELD_BOND_REF]  # Already 0-based internally
        atom[ZMatrix.FIELD_BOND_LENGTH] = _calc_distance(coords[bond_ref_idx], coords[i])
        
        # Recalculate angle (if atom 2 or later)
        if i >= 2:
            angle_ref_idx = atom[ZMatrix.FIELD_ANGLE_REF]  # Already 0-based internally
            atom[ZMatrix.FIELD_ANGLE] = _calc_angle(coords[angle_ref_idx], coords[bond_ref_idx], coords[i])
        
        # Recalculate dihedral/chirality angle (if atom 3 or later)
        if i >= 3:
            dihedral_ref_idx = atom[ZMatrix.FIELD_DIHEDRAL_REF]  # Already 0-based internally
            chirality = atom.get(ZMatrix.FIELD_CHIRALITY, 0)
            
            if chirality == 0:
                # True dihedral angle: i → bond_ref → angle_ref → dihedral_ref
                atom[ZMatrix.FIELD_DIHEDRAL] = _calc_dihedral(
                    coords[i],
                    coords[bond_ref_idx],
                    coords[angle_ref_idx],
                    coords[dihedral_ref_idx]
                )
            else:
                # Chirality case: store bond angle at bond_ref between i and dihedral_ref
                bond_angle = _calc_angle(
                    coords[i],
                    coords[bond_ref_idx],
                    coords[dihedral_ref_idx]
                )
                atom[ZMatrix.FIELD_DIHEDRAL] = bond_angle
                
                # Compute chirality sign using the centralized function
                atom[ZMatrix.FIELD_CHIRALITY] = compute_chirality_sign(
                    coords=coords,
                    atom_idx=i,
                    bond_ref_idx=bond_ref_idx,
                    angle_ref_idx=angle_ref_idx,
                    dihedral_ref_idx=dihedral_ref_idx
                )
        # Update the ZMatrix with modified atom (atom dict is already updated by reference)
        # No need to reassign since we're modifying the dict in place
    
    return new_zmatrix


# =============================================================================
# Z-Matrix Generation from Cartesian Coordinates
# =============================================================================

def _get_atomic_number(element_symbol: str) -> int:
    """
    Get atomic number from element symbol.
    
    Parameters
    ----------
    element_symbol : str
        Element symbol (e.g., 'H', 'C', 'N')
        
    Returns
    -------
    int
        Atomic number
    """
    try:
        elem = Element.getBySymbol(element_symbol)
        return elem.atomic_number
    except KeyError:
        # For unknown elements, default to 1 (H)
        return 1


def _get_first_ref_atom_id(atom_idx: int, graph: Dict[int, List[int]], 
                           coords: Optional[np.ndarray] = None) -> int:
    """
    Get first reference atom ID for Z-matrix construction.
    
    Finds a connected atom with index < atom_idx (previously defined atom).
    If no bonded neighbor exists, uses the closest previously processed atom by distance.
    
    Parameters
    ----------
    atom_idx : int
        Current atom index (0-based)
    graph : Dict[int, List[int]]
        Bond connectivity graph (adjacency list, 0-based indices)
    coords : Optional[np.ndarray]
        Nx3 array of Cartesian coordinates (used if no bonded neighbor found)
        
    Returns
    -------
    int
        Reference atom index (0-based)
    """
    candidates = []
    if atom_idx in graph:
        for nbr in graph[atom_idx]:
            if nbr < atom_idx:
                candidates.append(nbr)
    
    if candidates:
        # Sort to get consistent ordering (smallest index first)
        candidates.sort()
        return candidates[0]
    
    # No bonded neighbor found - use closest previously processed atom by distance
    if coords is not None:
        min_dist = float('inf')
        closest_idx = -1
        for i in range(atom_idx):
            dist = _calc_distance(coords[atom_idx], coords[i])
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        if closest_idx >= 0:
            return closest_idx
    
    raise ValueError(f"No reference atom found for atom {atom_idx}")


def _get_second_ref_atom_id(atom_idx: int, first_ref: int, graph: Dict[int, List[int]]) -> int:
    """
    Get second reference atom ID for Z-matrix construction.
    
    Finds a connected atom to first_ref with index < atom_idx and != atom_idx.
    
    Parameters
    ----------
    atom_idx : int
        Current atom index (0-based)
    first_ref : int
        First reference atom index (0-based)
    graph : Dict[int, List[int]]
        Bond connectivity graph (adjacency list, 0-based indices)
        
    Returns
    -------
    int
        Second reference atom index (0-based)
    """
    candidates = []
    if first_ref in graph:
        for nbr in graph[first_ref]:
            if nbr < atom_idx and nbr != atom_idx:
                candidates.append(nbr)
    
    if not candidates:
        raise ValueError(f"No second reference atom found for atom {atom_idx} (first_ref={first_ref})")
    
    # Sort to get consistent ordering (smallest index first)
    candidates.sort()
    return candidates[0]


def _count_predefined_neighbours(atom_idx: int, ref_atom: int, graph: Dict[int, List[int]]) -> int:
    """
    Count neighbors of ref_atom that have index < atom_idx.
    
    Parameters
    ----------
    atom_idx : int
        Current atom index (0-based)
    ref_atom : int
        Reference atom index (0-based)
    graph : Dict[int, List[int]]
        Bond connectivity graph (adjacency list, 0-based indices)
        
    Returns
    -------
    int
        Number of predefined neighbors
    """
    count = 0
    if ref_atom in graph:
        for nbr in graph[ref_atom]:
            if nbr < atom_idx:
                count += 1
    return count


def _is_torsion_used(second_ref: int, third_ref: int, zmatrix_atoms: List[Dict]) -> bool:
    """
    Check if a torsion (second_ref, third_ref) is already used in the Z-matrix.
    
    Parameters
    ----------
    second_ref : int
        Second reference atom index (0-based)
    third_ref : int
        Third reference atom index (0-based)
    zmatrix_atoms : List[Dict]
        List of Z-matrix atoms already constructed
        
    Returns
    -------
    bool
        True if torsion is already used
    """
    for atom in zmatrix_atoms:
        if (atom.get(ZMatrix.FIELD_BOND_REF) == second_ref and 
            atom.get(ZMatrix.FIELD_ANGLE_REF) == third_ref):
            return True
        if (atom.get(ZMatrix.FIELD_BOND_REF) == third_ref and 
            atom.get(ZMatrix.FIELD_ANGLE_REF) == second_ref):
            return True
    return False


def _get_third_ref_atom_id(atom_idx: int, first_ref: int, second_ref: int, 
                           graph: Dict[int, List[int]], coords: np.ndarray,
                           zmatrix_atoms: List[Dict]) -> Tuple[int, int]:
    """
    Get third reference atom ID for Z-matrix construction.
    
    Determines if we need a dihedral (chirality=0) or second angle (chirality=1/-1).
    
    Parameters
    ----------
    atom_idx : int
        Current atom index (0-based)
    first_ref : int
        First reference atom index (0-based)
    second_ref : int
        Second reference atom index (0-based)
    graph : Dict[int, List[int]]
        Bond connectivity graph (adjacency list, 0-based indices)
    coords : np.ndarray
        Nx3 array of Cartesian coordinates
    zmatrix_atoms : List[Dict]
        List of Z-matrix atoms already constructed
        
    Returns
    -------
    Tuple[int, int]
        (third_ref_index, chirality_flag) where:
        - third_ref_index: Third reference atom index (0-based)
        - chirality_flag: 0 for dihedral, 1/-1 for second angle
    """
    # Check if we need a second angle (chirality case)
    use_second_angle = (_is_torsion_used(second_ref, first_ref, zmatrix_atoms) or
                       _count_predefined_neighbours(atom_idx, second_ref, graph) == 1)
    
    candidates = []
    
    if use_second_angle:
        # Look for neighbors of first_ref (not atom_idx, not second_ref)
        chirality_flag = 1  # Will be adjusted based on sign
        if first_ref in graph:
            for nbr in graph[first_ref]:
                if nbr < atom_idx and nbr != atom_idx and nbr != second_ref:
                    # Check angle between nbr-first_ref-second_ref to avoid collinearity
                    angle = _calc_angle(coords[nbr], coords[first_ref], coords[second_ref])
                    if angle > 1.0:  # Not collinear
                        candidates.append(nbr)
    else:
        # Look for neighbors of second_ref (not atom_idx, not first_ref)
        chirality_flag = 0
        if second_ref in graph:
            for nbr in graph[second_ref]:
                if nbr < atom_idx and nbr != atom_idx and nbr != first_ref:
                    candidates.append(nbr)
    
    if not candidates:
        raise ValueError(f"Unable to make internal coordinates for atom {atom_idx}. "
                        f"Consider using dummy atoms.")
    
    # Sort to get consistent ordering (smallest index first)
    candidates.sort()
    third_ref = candidates[0]
    
    # If using second angle, determine chirality sign
    if use_second_angle:
        # Calculate dihedral to determine sign
        dihedral = _calc_dihedral(
            coords[atom_idx],
            coords[first_ref],
            coords[second_ref],
            coords[third_ref]
        )
        if dihedral > 0.0:
            chirality_flag = -1
        else:
            chirality_flag = 1
    
    return third_ref, chirality_flag


def generate_zmatrix(atoms: List[Dict[str, np.ndarray]], bonds: List[Tuple[int, int]]) -> ZMatrix:
    """
    Generate Z-matrix from Cartesian coordinates and bond connectivity.
    
    This function converts Cartesian coordinates to Z-matrix format following the given bond connectivity. The atom type map uses atomic numbers: each element
    symbol maps to its atomic number as the atom type identifier.
    
    Parameters
    ----------
    atoms : List[Dict[str, np.ndarray]]
        List of atom dictionaries, each containing:
        - 'element': element symbol (str)
        - 'coords': 3D coordinates (np.ndarray)
    bonds : List[Tuple[int, int]]
        List of bonds as (atom1_idx, atom2_idx) tuples (0-based indices)
        
    Returns
    -------
    ZMatrix
        Z-matrix representation with 0-based indices
        
    Raises
    ------
    ValueError
        If conversion fails (e.g., disconnected atoms, invalid geometry)
    """
    num_atoms = len(atoms)
    if num_atoms < 1:
        raise ValueError("At least one atom is required")
    
    # Extract coordinates and build atom type map
    coords = []
    atom_type_map = {}  # element -> atomic number
    
    for atom in atoms:
        coords.append(atom['coords'])
        element = atom['element']
        if element not in atom_type_map:
            atom_type_map[element] = _get_atomic_number(element)
    
    coords = np.array(coords)
    
    # Build bond connectivity graph (adjacency list, 0-based)
    graph = defaultdict(list)
    
    for atom1, atom2 in bonds:
        graph[atom1].append(atom2)
        graph[atom2].append(atom1)
    
    # Convert to Z-matrix
    zmatrix_atoms = []
    
    for i in range(num_atoms):
        atom_data = {
            ZMatrix.FIELD_ID: i,
            ZMatrix.FIELD_ELEMENT: atoms[i]['element'],
            ZMatrix.FIELD_ATOMIC_NUM: atom_type_map[atoms[i]['element']]
        }
        
        i2 = 0
        i3 = 0
        i4 = 0
        chirality = 0
        bond_length = 0.0
        angle = 0.0
        dihedral = 0.0
        
        # Define bond length (for atoms 1+)
        if i > 0:
            i2 = _get_first_ref_atom_id(i, graph, coords)
            bond_length = _calc_distance(coords[i], coords[i2])
            atom_data[ZMatrix.FIELD_BOND_REF] = i2
            atom_data[ZMatrix.FIELD_BOND_LENGTH] = bond_length
        
        # Define bond angle (for atoms 2+)
        if i > 1:
            i3 = _get_second_ref_atom_id(i, i2, graph)
            angle = _calc_angle(coords[i], coords[i2], coords[i3])
            atom_data[ZMatrix.FIELD_ANGLE_REF] = i3
            atom_data[ZMatrix.FIELD_ANGLE] = angle
        
        # Define dihedral or second angle (for atoms 3+)
        if i > 2:
            i4, chirality = _get_third_ref_atom_id(i, i2, i3, graph, coords, zmatrix_atoms)
            atom_data[ZMatrix.FIELD_DIHEDRAL_REF] = i4
            
            if chirality != 0:
                # Second angle case
                dihedral = _calc_angle(coords[i], coords[i2], coords[i4])
                atom_data[ZMatrix.FIELD_DIHEDRAL] = dihedral
                atom_data[ZMatrix.FIELD_CHIRALITY] = chirality
            else:
                # True dihedral case
                dihedral = _calc_dihedral(coords[i], coords[i2], coords[i3], coords[i4])
                atom_data[ZMatrix.FIELD_DIHEDRAL] = dihedral
                atom_data[ZMatrix.FIELD_CHIRALITY] = 0
        
        zmatrix_atoms.append(atom_data)
    
    # Create ZMatrix instance with all bonds (preserve complete connectivity)
    zmatrix_obj = ZMatrix(zmatrix_atoms, bonds)
    
    return zmatrix_obj

