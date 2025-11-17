#!/usr/bin/env python3
"""
I/O Tools for Molecular System Files

This module provides functions for reading and writing molecular structure files,
including INT (Z-matrix) and XYZ formats.
"""

from os.path import basename
import numpy as np
from typing import List, Dict

from .CoordinateConversion import zmatrix_to_cartesian, generate_zmatrix
from .ZMatrix import ZMatrix

def read_int_file(pathname: str) -> ZMatrix:
    """
    Read and parse INT file data in Z-matrix (internal coordinates) format.
    
    The INT file format uses 1-based indexing for atom references:
    Line 1: number of atoms
    Subsequent lines: atom_id element atomic_num [ref_bond bond_length] [ref_angle angle] [ref_dihedral dihedral chirality]
    
    This function reads the Z-matrix, converts all reference indices from 
    1-based (file format) to 0-based (internal representation).
    
    Returns
    -------
    ZMatrix instance
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
            ZMatrix.FIELD_ID: int(parts[0]) - 1,
            ZMatrix.FIELD_ELEMENT: parts[1],
            ZMatrix.FIELD_ATOMIC_NUM: int(parts[2])
        }
        
        atoms.append((parts[1], i - 1))
        
        # Parse internal coordinates based on number of fields
        # Input file uses 1-based indexing, convert to 0-based for internal use
        if len(parts) > 3:
            # Has bond reference (1-based in file, convert to 0-based)
            bond_ref_1based = int(parts[3])
            atom_data[ZMatrix.FIELD_BOND_REF] = bond_ref_1based - 1  # Convert to 0-based
            atom_data[ZMatrix.FIELD_BOND_LENGTH] = float(parts[4])
            # Add bond (already 0-based: i-1 and bond_ref-1)
            bonds.append((i - 1, bond_ref_1based - 1, 1))
        
        if len(parts) > 5:
            # Has angle reference (1-based in file, convert to 0-based)
            angle_ref_1based = int(parts[5])
            atom_data[ZMatrix.FIELD_ANGLE_REF] = angle_ref_1based - 1  # Convert to 0-based
            atom_data[ZMatrix.FIELD_ANGLE] = float(parts[6])
        
        if len(parts) > 7:
            # Has dihedral reference (1-based in file, convert to 0-based)
            dihedral_ref_1based = int(parts[7])
            atom_data[ZMatrix.FIELD_DIHEDRAL_REF] = dihedral_ref_1based - 1  # Convert to 0-based
            atom_data[ZMatrix.FIELD_DIHEDRAL] = float(parts[8])
            atom_data[ZMatrix.FIELD_CHIRALITY] = int(parts[9])
        
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

    # Create ZMatrix instance
    zmatrix_obj = ZMatrix(zmatrix, bonds)
    
    return zmatrix_obj


def write_zmatrix_file(zmatrix: ZMatrix, filepath: str) -> None:
    """
    Write Z-matrix to INT file format.
    
    The INT file format uses 1-based indexing for atom references.
    This function converts from 0-based (internal representation) to 
    1-based (file format) when writing.
    
    Parameters
    ----------
    zmatrix : ZMatrix
        Z-matrix representation. All reference indices (bond_ref, angle_ref, 
        dihedral_ref) are expected to be in 0-based indexing (internal format).
    filepath : str
        Output file path
    """
    atoms_list = zmatrix.to_list()

    bonds_implicit_in_zmatrix = []
    for atom in zmatrix.atoms:
        if atom.get(ZMatrix.FIELD_BOND_REF) is not None:
            atm1 = atom[ZMatrix.FIELD_ID]
            atm2 = atom[ZMatrix.FIELD_BOND_REF]
            if atm1 < atm2:
                bonds_implicit_in_zmatrix.append((atm1, atm2, 1))
            else:
                bonds_implicit_in_zmatrix.append((atm2, atm1, 1))

    bonds_declared_sorted = [(min(bond[0], bond[1]), max(bond[0], bond[1]), bond[2]) for bond in zmatrix.bonds]

    bonds_to_add = []
    for bond in zmatrix.bonds:
        atm1 = bond[0]
        atm2 = bond[1]
        sorted_bond = (min(atm1, atm2), max(atm1, atm2), 1)
        if sorted_bond not in bonds_implicit_in_zmatrix:
            bonds_to_add.append(sorted_bond)
    
    bonds_to_remove = []
    for bond in bonds_implicit_in_zmatrix:
        atm1 = bond[0]
        atm2 = bond[1]
        sorted_bond = (min(atm1, atm2), max(atm1, atm2), 1)
        if sorted_bond not in bonds_declared_sorted:
            bonds_to_remove.append(sorted_bond)

    with open(filepath, 'w') as f:
        f.write(f"{len(atoms_list)}\n")
        for i, atom in enumerate(atoms_list):
            id_1based = atom[ZMatrix.FIELD_ID] + 1
            if i == 0:
                f.write(f"{id_1based:6d}  {atom[ZMatrix.FIELD_ELEMENT]:<5s}{atom[ZMatrix.FIELD_ATOMIC_NUM]:4d}\n")
            elif i == 1:
                # Convert 0-based to 1-based for file format
                bond_ref_1based = atom[ZMatrix.FIELD_BOND_REF] + 1
                f.write(f"{id_1based:6d}  {atom[ZMatrix.FIELD_ELEMENT]:<5s}{atom[ZMatrix.FIELD_ATOMIC_NUM]:4d}"
                       f"{bond_ref_1based:6d}{atom[ZMatrix.FIELD_BOND_LENGTH]:12.6f}\n")
            elif i == 2:
                # Convert 0-based to 1-based for file format
                bond_ref_1based = atom[ZMatrix.FIELD_BOND_REF] + 1
                angle_ref_1based = atom[ZMatrix.FIELD_ANGLE_REF] + 1
                f.write(f"{id_1based:6d}  {atom[ZMatrix.FIELD_ELEMENT]:<5s}{atom[ZMatrix.FIELD_ATOMIC_NUM]:4d}"
                       f"{bond_ref_1based:6d}{atom[ZMatrix.FIELD_BOND_LENGTH]:12.6f}"
                       f"{angle_ref_1based:6d}{atom[ZMatrix.FIELD_ANGLE]:12.6f}\n")
            else:
                # Convert 0-based to 1-based for file format
                bond_ref_1based = atom[ZMatrix.FIELD_BOND_REF] + 1
                angle_ref_1based = atom[ZMatrix.FIELD_ANGLE_REF] + 1
                dihedral_ref_1based = atom[ZMatrix.FIELD_DIHEDRAL_REF] + 1
                f.write(f"{id_1based:6d}  {atom[ZMatrix.FIELD_ELEMENT]:<5s}{atom[ZMatrix.FIELD_ATOMIC_NUM]:4d}"
                       f"{bond_ref_1based:6d}{atom[ZMatrix.FIELD_BOND_LENGTH]:12.6f}"
                       f"{angle_ref_1based:6d}{atom[ZMatrix.FIELD_ANGLE]:12.6f}"
                       f"{dihedral_ref_1based:6d}{atom[ZMatrix.FIELD_DIHEDRAL]:12.6f}"
                       f"{atom[ZMatrix.FIELD_CHIRALITY]:6d}\n")

        f.write("\n")
        for bond in bonds_to_add:
            f.write(f"{bond[0]+1:6d} {bond[1]+1:6d}\n")
        f.write("\n")
        for bond in bonds_to_remove:
            f.write(f"{bond[0]+1:6d} {bond[1]+1:6d}\n")


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
            correctedElement = element.replace('Du', '*').replace('ATP', '*').replace('ATM', '*')
            f.write(f"{correctedElement:<4s} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}\n")


def write_sdf_file(zmatrix: ZMatrix, filepath: str) -> None:
    """
    Write SDF file from Z-matrix.
    
    Converts Z-matrix to Cartesian coordinates and writes SDF format.
    Includes all bonds: those implicit in Z-matrix structure plus explicit bonds.
    
    Parameters
    ----------
    zmatrix : ZMatrix
        Z-matrix to save
    filepath : str
        Output file path
    """
    # Convert Z-matrix to Cartesian coordinates
    coords = zmatrix_to_cartesian(zmatrix)
    elements = zmatrix.get_elements()
    
    # Collect all bonds: implicit from Z-matrix structure + explicit bonds
    all_bonds = []
    bonds_set = set()
    
    # Add implicit bonds from Z-matrix structure (bond_ref relationships)
    for i in range(1, len(zmatrix)):
        atom = zmatrix[i]
        if ZMatrix.FIELD_BOND_REF in atom:
            bond_ref = atom[ZMatrix.FIELD_BOND_REF]
            bond_key = (min(i, bond_ref), max(i, bond_ref))
            if bond_key not in bonds_set:
                all_bonds.append((i, bond_ref, 1))  # Default bond type 1
                bonds_set.add(bond_key)
    
    # Add explicit bonds from zmatrix.bonds
    for bond in zmatrix.bonds:
        bond_key = (min(bond[0], bond[1]), max(bond[0], bond[1]))
        if bond_key not in bonds_set:
            all_bonds.append(bond)
            bonds_set.add(bond_key)
    
    with open(filepath, 'w') as f:
        f.write("from Z-Matrix\n")
        f.write(" RingclosingMM\n")
        f.write("\n")
        f.write(f"{len(zmatrix):3d}{len(all_bonds):3d}  0  0  0  0  0  0  0  0  0  0999 V2000\n")
        
        # Write atoms with coordinates
        for i, (element, coord) in enumerate(zip(elements, coords)):
            f.write(f"{coord[0]:10.4f}{coord[1]:10.4f}{coord[2]:10.4f} {element:3s} 0  0  0  0  0  0  0  0  0  0  0  0\n")
        
        # Write bonds (1-based indices in SDF format)
        for bond in all_bonds:
            f.write(f"{bond[0]+1:3d}{bond[1]+1:3d}{bond[2]:3d}  0  0  0  0\n")
        
        f.write("M  END\n")
        f.write("$$$$\n")


def save_structure_to_file(filepath: str, zmatrix: ZMatrix, energy: float) -> None:
    """
    Save structure to file.
    
    Parameters
    ----------
    filepath : str
        Output file path
    zmatrix : ZMatrix
        Z-matrix to save
    energy : float
        Energy of the structure (default: None)
    """
    filename_xyz = None;
    filename_int = None;
    filename_sdf = None;
    if filepath.endswith('.xyz'):
        filename_xyz = filepath;
    elif filepath.endswith('.int'):
        filename_int = filepath;
    elif filepath.endswith('.sdf'):
        filename_sdf = filepath;
    else:
        if '.' in basename(filepath):
            raise ValueError(f"Unsupported file extension: {filepath}")
        else:
            filename_xyz = filepath + '.xyz';
            filename_int = filepath + '.int';
            filename_sdf = filepath + '.sdf';

    if filename_xyz is not None:
        optimized_coords = zmatrix_to_cartesian(zmatrix)
        elements = zmatrix.get_elements()
        comment = f"E={energy:.2f} kcal/mol" if energy is not None else ""
        write_xyz_file(optimized_coords, elements, filename_xyz, comment=comment)

    if filename_int is not None:
        write_zmatrix_file(zmatrix, filename_int)

    if filename_sdf is not None:
        write_sdf_file(zmatrix, filename_sdf)


def read_sdf_file(pathname: str) -> ZMatrix:
    """
    Read and parse SDF file, converting to Z-matrix (internal coordinates).
    
    This function reads an SDF file containing Cartesian coordinates and bond
    connectivity, then converts it to return a Z-matrix representation.
    
    Parameters
    ----------
    pathname : str
        Path to SDF file
        
    Returns
    -------
    ZMatrix
        Z-matrix representation with 0-based indices
        
    Raises
    ------
    ValueError
        If the SDF file format is invalid or conversion fails
    """
    with open(pathname, 'r') as f:
        lines = [line.rstrip('\n\r') for line in f.readlines()]
    
    if len(lines) < 4:
        raise ValueError(f"Invalid SDF file: too few lines in {pathname}")
    
    # Parse header
    # Line 0: molecule name (ignored)
    # Line 1: program info (ignored)
    # Line 2: blank (ignored)
    # Line 3: counts line
    counts_line = lines[3]
    num_atoms = int(counts_line[0:3].strip())
    num_bonds = int(counts_line[3:6].strip())
    
    if num_atoms < 1:
        raise ValueError(f"Invalid number of atoms in SDF file: {num_atoms}")
    
    # Parse atoms (lines 4 to 4+num_atoms-1)
    atoms_data = []
    
    for i in range(num_atoms):
        line = lines[4 + i]
        x = float(line[0:10].strip())
        y = float(line[10:20].strip())
        z = float(line[20:30].strip())
        element = line[30:33].strip()
        
        atoms_data.append({
            'element': element,
            'coords': np.array([x, y, z])
        })
    
    # Parse bonds (lines after atoms until "M  END")
    bonds = []
    bond_start_line = 4 + num_atoms
    for i in range(num_bonds):
        if bond_start_line + i >= len(lines):
            break
        line = lines[bond_start_line + i]
        if line.strip() == "M  END":
            break
        
        # SDF bond format: "iii jjj ttt" (1-based indices)
        atom1_1based = int(line[0:3].strip())
        atom2_1based = int(line[3:6].strip())
        bond_type = int(line[6:9].strip())
        
        # Convert to 0-based
        bonds.append((atom1_1based - 1, atom2_1based - 1, bond_type))
    
    # Convert to Z-matrix using CoordinateConversion
    zmatrix_obj = generate_zmatrix(atoms_data, bonds)
    
    return zmatrix_obj