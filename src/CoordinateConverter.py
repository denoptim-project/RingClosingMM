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
    
Helper Functions:
    compute_chirality_sign: Determine chirality sign from Cartesian coordinates
    _calc_distance: Calculate distance between two points
    _calc_angle: Calculate angle between three points
    _calc_dihedral: Calculate dihedral angle between four points (atan2 method)
    
Backward Compatibility:
    CoordinateConverter: Wrapper class for backward compatibility (deprecated)
"""

import numpy as np
from typing import List, Dict
import copy

# Dual import handling for package and direct script use
try:
    from .ZMatrix import ZMatrix
except ImportError:
    from ZMatrix import ZMatrix


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
    coords[1] = [0.0, 0.0, atoms_list[1]['bond_length']]
    
    if num_atoms < 3:
        return coords
    
    # Atom 3: place in XZ plane
    atom3 = atoms_list[2]
    ref_bond = atom3['bond_ref']  # Already 0-based internally
    ref_angle = atom3['angle_ref']  # Already 0-based internally
    
    bond_length = atom3['bond_length']
    angle_deg = atom3['angle']
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
        ia = atom['bond_ref']  # Already 0-based internally
        ib = atom['angle_ref']  # Already 0-based internally
        ic = atom['dihedral_ref']  # Already 0-based internally
        
        bond_length = atom['bond_length']
        angle_rad = np.deg2rad(atom['angle'])
        # Negate dihedral for XZ plane convention (opposite sign from XY plane)
        dihedral_rad = np.deg2rad(-atom['dihedral'])
        chiral = atom.get('chirality', 0)
        
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
                    f"The geometry around atoms {ib+1} (0-based: {ib}), {ia+1} (0-based: {ia}), and {ic+1} (0-based: {ic}) "
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
                    f"The geometry around atoms {ib+1} (0-based: {ib}), {ia+1} (0-based: {ia}), and {ic+1} (0-based: {ic}) "
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
        new_zmatrix[idx]['dihedral'] = new_dihedral
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
        ia = atom['bond_ref']
        ib = atom['angle_ref']
        ic = atom['dihedral_ref']
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
        bond_ref_idx = atom['bond_ref']  # Already 0-based internally
        atom['bond_length'] = _calc_distance(coords[bond_ref_idx], coords[i])
        
        # Recalculate angle (if atom 2 or later)
        if i >= 2:
            angle_ref_idx = atom['angle_ref']  # Already 0-based internally
            atom['angle'] = _calc_angle(coords[angle_ref_idx], coords[bond_ref_idx], coords[i])
        
        # Recalculate dihedral/chirality angle (if atom 3 or later)
        if i >= 3:
            dihedral_ref_idx = atom['dihedral_ref']  # Already 0-based internally
            chirality = atom.get('chirality', 0)
            
            if chirality == 0:
                # True dihedral angle: i → bond_ref → angle_ref → dihedral_ref
                atom['dihedral'] = _calc_dihedral(
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
                atom['dihedral'] = bond_angle
                
                # Compute chirality sign using the centralized function
                atom['chirality'] = compute_chirality_sign(
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
# Backward Compatibility Class
# =============================================================================

class CoordinateConverter:
    """
    Backward compatibility wrapper for coordinate conversion functions.
    
    This class provides the same interface as before, but now delegates to
    module-level functions. All methods are static.
    
    .. deprecated::
        Use module-level functions directly instead:
        - `zmatrix_to_cartesian()`
        - `cartesian_to_zmatrix()`
        - `apply_torsions()`
        - `extract_torsions()`
    """
    
    @staticmethod
    def apply_torsions(base_zmatrix: ZMatrix, rotatable_indices: List[int],
                      torsion_values: np.ndarray) -> ZMatrix:
        """Apply torsions (delegates to module function)."""
        return apply_torsions(base_zmatrix, rotatable_indices, torsion_values)
    
    @staticmethod
    def extract_torsions(coords: np.ndarray, zmatrix: ZMatrix,
                        rotatable_indices: List[int]) -> np.ndarray:
        """Extract torsions (delegates to module function)."""
        return extract_torsions(coords, zmatrix, rotatable_indices)
    
    @staticmethod
    def extract_zmatrix(coords: np.ndarray, zmatrix: ZMatrix) -> ZMatrix:
        """Extract Z-matrix (delegates to cartesian_to_zmatrix)."""
        return cartesian_to_zmatrix(coords, zmatrix)

