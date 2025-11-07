#!/usr/bin/env python3
"""
I/O Tools for Molecular System Files

This module provides functions for reading and writing molecular structure files,
including INT (Z-matrix) and XYZ formats.
"""

import numpy as np
from typing import List, Dict

# Dual import handling for package and direct script use
try:
    from .CoordinateConverter import zmatrix_to_cartesian
except ImportError:
    from CoordinateConverter import zmatrix_to_cartesian

def read_int_file(pathname):
    """
    Read and parse INT file data in Z-matrix (internal coordinates) format.
    
    The INT file format uses 1-based indexing for atom references:
    Line 1: number of atoms
    Subsequent lines: atom_id element atomic_num [ref_bond bond_length] [ref_angle angle] [ref_dihedral dihedral chirality]
    
    This function reads the Z-matrix, converts all reference indices from 
    1-based (file format) to 0-based (internal representation), converts 
    to Cartesian coordinates, and extracts bond connectivity.
    
    Returns
    -------
    dict with keys:
        'atoms': list of tuples (element_symbol, atom_index) - 0-based indices
        'positions': list of [x, y, z] in Angstroms (converted from Z-matrix)
        'bonds': list of tuples (atom1_idx, atom2_idx, bond_type=1) - 0-based indices
        'rcpterms': empty list
        'rotatableBonds': empty list
        'zmatrix': list of dict (Z-matrix representation) - all reference indices are 0-based
    """
    with open(pathname, 'r') as f:
        lines = f.readlines()
    
    # Read number of atoms
    num_atoms = int(lines[0].strip().split()[0])
    
    # Parse Z-matrix
    zmatrix = []
    atoms = []
    bonds = []
    
    for i in range(1, num_atoms + 1):
        parts = lines[i].split()
        
        atom_data = {
            'id': int(parts[0]),
            'element': parts[1],
            'atomic_num': int(parts[2])
        }
        
        atoms.append((parts[1], i - 1))
        
        # Parse internal coordinates based on number of fields
        # Input file uses 1-based indexing, convert to 0-based for internal use
        if len(parts) > 3:
            # Has bond reference (1-based in file, convert to 0-based)
            bond_ref_1based = int(parts[3])
            atom_data['bond_ref'] = bond_ref_1based - 1  # Convert to 0-based
            atom_data['bond_length'] = float(parts[4])
            # Add bond (already 0-based: i-1 and bond_ref-1)
            bonds.append((i - 1, bond_ref_1based - 1, 1))
        
        if len(parts) > 5:
            # Has angle reference (1-based in file, convert to 0-based)
            angle_ref_1based = int(parts[5])
            atom_data['angle_ref'] = angle_ref_1based - 1  # Convert to 0-based
            atom_data['angle'] = float(parts[6])
        
        if len(parts) > 7:
            # Has dihedral reference (1-based in file, convert to 0-based)
            dihedral_ref_1based = int(parts[7])
            atom_data['dihedral_ref'] = dihedral_ref_1based - 1  # Convert to 0-based
            atom_data['dihedral'] = float(parts[8])
            atom_data['chirality'] = int(parts[9])
        
        zmatrix.append(atom_data)

    # Skip one line if at all present
    i += 1
    
    # After the Z-matrix we have the corrections to the connectivity:
    # first, the bonds to be added
    go_on = True
    while go_on:
        i += 1
        if i >= len(lines):
            go_on = False
            break
        line = lines[i]
        if not line.strip():
            go_on = False
        else:
            parts = line.split()
            bonds.append((int(parts[0]) - 1, int(parts[1]) - 1, 1))
    
    # then, those to be removed
    go_on = True
    while go_on:
        i += 1
        if i >= len(lines):
            go_on = False
            break
        line = lines[i]
        if not line.strip():
            go_on = False
        else:
            parts = line.split()
            atom1 = int(parts[0]) - 1
            atom2 = int(parts[1]) - 1
            # Remove bond (check both directions)
            bonds = [(a1, a2, bt) for a1, a2, bt in bonds 
                    if not ((a1 == atom1 and a2 == atom2) or (a1 == atom2 and a2 == atom1))]

    # Convert Z-matrix to Cartesian coordinates
    positions = zmatrix_to_cartesian(zmatrix)
    
    return {
        'atoms': atoms,
        'positions': positions.tolist(),
        'bonds': bonds,
        'rcpterms': [],
        'rotatableBonds': [],
        'zmatrix': zmatrix
    }


def write_zmatrix_file(zmatrix: List[Dict], filepath: str) -> None:
    """
    Write Z-matrix to INT file format.
    
    The INT file format uses 1-based indexing for atom references.
    This function converts from 0-based (internal representation) to 
    1-based (file format) when writing.
    
    Parameters
    ----------
    zmatrix : List[Dict]
        Z-matrix representation. All reference indices (bond_ref, angle_ref, 
        dihedral_ref) are expected to be in 0-based indexing (internal format).
    filepath : str
        Output file path
    """
    with open(filepath, 'w') as f:
        f.write(f"{len(zmatrix)}\n")
        for i, atom in enumerate(zmatrix):
            if i == 0:
                f.write(f"{atom['id']:6d}  {atom['element']:<5s}{atom['atomic_num']:4d}\n")
            elif i == 1:
                # Convert 0-based to 1-based for file format
                bond_ref_1based = atom['bond_ref'] + 1
                f.write(f"{atom['id']:6d}  {atom['element']:<5s}{atom['atomic_num']:4d}"
                       f"{bond_ref_1based:6d}{atom['bond_length']:12.6f}\n")
            elif i == 2:
                # Convert 0-based to 1-based for file format
                bond_ref_1based = atom['bond_ref'] + 1
                angle_ref_1based = atom['angle_ref'] + 1
                f.write(f"{atom['id']:6d}  {atom['element']:<5s}{atom['atomic_num']:4d}"
                       f"{bond_ref_1based:6d}{atom['bond_length']:12.6f}"
                       f"{angle_ref_1based:6d}{atom['angle']:12.6f}\n")
            else:
                # Convert 0-based to 1-based for file format
                bond_ref_1based = atom['bond_ref'] + 1
                angle_ref_1based = atom['angle_ref'] + 1
                dihedral_ref_1based = atom['dihedral_ref'] + 1
                f.write(f"{atom['id']:6d}  {atom['element']:<5s}{atom['atomic_num']:4d}"
                       f"{bond_ref_1based:6d}{atom['bond_length']:12.6f}"
                       f"{angle_ref_1based:6d}{atom['angle']:12.6f}"
                       f"{dihedral_ref_1based:6d}{atom['dihedral']:12.6f}"
                       f"{atom['chirality']:6d}\n")


def write_xyz_file(coords: np.ndarray, elements: List[str], filepath: str, comment: str = "", append: bool = False) -> None:
    """
    Write XYZ file.
    
    Parameters
    ----------
    coords : np.ndarray
        Coordinates of the atoms (Nx3 array) in Angstroms
    elements : List[str]
        Elements of the atoms
    filepath : str
        Output file path
    comment : str
        Comment line for XYZ file
    append : bool
        If True, append to existing file instead of overwriting (default: False)
    """
    mode = 'a' if append else 'w'
    with open(filepath, mode) as f:
        f.write(f"{len(elements)}\n")
        f.write(f"{comment}\n")
        for element, coord in zip(elements, coords):
            correctedElement = element.replace('Du', 'He').replace('ATP', 'Ne').replace('ATM', 'Ar')
            f.write(f"{correctedElement:<4s} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}\n")
