#!/usr/bin/env python3
"""
Z-Matrix Data Structure

This module provides the ZMatrix class for encapsulating Z-matrix (internal coordinate)
data structures along with bond connectivity information.

The Z-matrix format is similar to the Tinker format:
    - Atom 1 at origin
    - Atom 2 along +Z axis
    - Atom 3 in XZ plane
    - Atom 4+: may use dihedral angleor second bond angle
    - Chirality: distinguishes between true dihedral and second bond angle: 0 for true dihedral, +1/-1 for second bond angle
    - Dihedral: dihedral angle in degrees (or bond angle for chirality)
    - Bond: bond length in Angstroms
    - Angle: bond angle in degrees
    - All indices are 0-based internally
    - ctopological connections (i.e., bonds) may differ from the distances specified in the Z-matrix, so bonds are also specified as (atom1_idx, atom2_idx, bond_type) tuples (0-based indices)

Classes:
    ZMatrix: Encapsulates Z-matrix atoms and bond connectivity
"""

import copy
from typing import List, Dict, Tuple, Optional, Any


class ZMatrix:
    """
    Encapsulates Z-matrix data structure with bond connectivity.
    
    This class wraps the List[Dict] representation of Z-matrix atoms and includes
    bond connectivity as a first-class attribute. All internal indices are 0-based.
    
    Attributes
    ----------
    atoms : List[Dict]
        List of atom dictionaries containing Z-matrix data:
        - 'id': atom index (0-based)
        - 'element': element symbol
        - 'atomic_num': atomic number
        - 'bond_ref': reference atom for bond (0-based, optional)
        - 'bond_length': bond length in Angstroms (optional)
        - 'angle_ref': reference atom for angle (0-based, optional)
        - 'angle': bond angle in degrees (optional)
        - 'dihedral_ref': reference atom for dihedral (0-based, optional)
        - 'dihedral': dihedral angle in degrees (optional)
        - 'chirality': 0 for true dihedral, +1/-1 for chiral angle (optional)
    bonds : List[Tuple[int, int, int]]
        List of bonds as (atom1_idx, atom2_idx, bond_type) tuples (0-based indices)
    
    Examples
    --------
    >>> zmat = ZMatrix(atoms=[...], bonds=[...])
    >>> atom = zmat[0]  # Get first atom
    >>> dihedral = zmat[5]['dihedral']  # Get dihedral of 6th atom
    >>> zmat[5]['dihedral'] = 120.0  # Update dihedral
    >>> len(zmat)  # Number of atoms
    >>> zmat.to_list()  # Convert to List[Dict] for backward compatibility
    """
    
    # DOF names mapping: 0=bond_length, 1=angle, 2=dihedral
    DOF_NAMES = ['bond_length', 'angle', 'dihedral']
    
    def __init__(self, atoms: List[Dict], bonds: List[Tuple[int, int, int]]):
        """
        Initialize Z-matrix from atoms and bonds.
        
        Parameters
        ----------
        atoms : List[Dict]
            List of atom dictionaries with Z-matrix data (0-based indices)
        bonds : List[Tuple[int, int, int]]
            List of bonds as (atom1_idx, atom2_idx, bond_type) tuples (0-based indices)
        
        Raises
        ------
        ValueError
            If atoms or bonds contain invalid data
        """
        self._atoms = copy.deepcopy(atoms)
        self._bonds = copy.deepcopy(bonds)
        self._validate()
    
    def _validate(self) -> None:
        """Validate Z-matrix data integrity."""
        if not isinstance(self._atoms, list):
            raise ValueError("atoms must be a list")
        if not isinstance(self._bonds, list):
            raise ValueError("bonds must be a list")
        
        # Validate atom indices are 0-based and sequential
        for i, atom in enumerate(self._atoms):
            if not isinstance(atom, dict):
                raise ValueError(f"Atom {i} must be a dictionary")
            if 'id' in atom and atom['id'] != i:
                raise ValueError(f"Atom {i} has inconsistent id: {atom['id']} (expected {i})")
            # Ensure id is set correctly
            atom['id'] = i
            
            # Validate reference indices are 0-based and in range
            for ref_key in ['bond_ref', 'angle_ref', 'dihedral_ref']:
                if ref_key in atom:
                    ref_idx = atom[ref_key]
                    if not isinstance(ref_idx, int):
                        raise ValueError(f"Atom {i} {ref_key} must be an integer")
                    if ref_idx < 0 or ref_idx >= len(self._atoms):
                        raise ValueError(f"Atom {i} {ref_key} index {ref_idx} out of range [0, {len(self._atoms)-1}]")
        
        # Validate bond indices are 0-based and in range
        for bond_idx, (atom1, atom2, bond_type) in enumerate(self._bonds):
            if not isinstance(atom1, int) or not isinstance(atom2, int):
                raise ValueError(f"Bond {bond_idx} atom indices must be integers")
            if atom1 < 0 or atom1 >= len(self._atoms):
                raise ValueError(f"Bond {bond_idx} atom1 index {atom1} out of range [0, {len(self._atoms)-1}]")
            if atom2 < 0 or atom2 >= len(self._atoms):
                raise ValueError(f"Bond {bond_idx} atom2 index {atom2} out of range [0, {len(self._atoms)-1}]")
    
    def __len__(self) -> int:
        """Return number of atoms."""
        return len(self._atoms)
    
    def __getitem__(self, index: int) -> Dict:
        """
        Get atom by index.
        
        Parameters
        ----------
        index : int
            Atom index (0-based)
        
        Returns
        -------
        Dict
            Atom dictionary (returns reference, not copy)
        """
        return self._atoms[index]
    
    def __setitem__(self, index: int, value: Dict):
        """
        Set atom by index.
        
        Parameters
        ----------
        index : int
            Atom index (0-based)
        value : Dict
            Atom dictionary
        """
        if not isinstance(value, dict):
            raise ValueError("Value must be a dictionary")
        self._atoms[index] = copy.deepcopy(value)
        self._atoms[index]['id'] = index  # Ensure id is correct
        self._validate()
    
    def __iter__(self):
        """Iterate over atoms."""
        return iter(self._atoms)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ZMatrix(n_atoms={len(self._atoms)}, n_bonds={len(self._bonds)})"
    
    @property
    def atoms(self) -> List[Dict]:
        """Get list of atoms (returns copy)."""
        return copy.deepcopy(self._atoms)
    
    @property
    def bonds(self) -> List[Tuple[int, int, int]]:
        """Get list of bonds (returns copy)."""
        return copy.deepcopy(self._bonds)
    
    def get_atom(self, index: int) -> Dict:
        """
        Get atom by index (returns copy).
        
        Parameters
        ----------
        index : int
            Atom index (0-based)
        
        Returns
        -------
        Dict
            Copy of atom dictionary
        """
        if index < 0 or index >= len(self._atoms):
            raise IndexError(f"Atom index {index} out of range [0, {len(self._atoms)-1}]")
        return copy.deepcopy(self._atoms[index])
    
    def get_bonds(self) -> List[Tuple[int, int, int]]:
        """
        Get bonds (returns copy).
        
        Returns
        -------
        List[Tuple[int, int, int]]
            Copy of bonds list
        """
        return copy.deepcopy(self._bonds)
    
    def update_dof(self, atom_idx: int, dof_type: int, value: float):
        """
        Update a degree of freedom value.
        
        Parameters
        ----------
        atom_idx : int
            Atom index (0-based)
        dof_type : int
            Degree of freedom type: 0=bond_length, 1=angle, 2=dihedral
        value : float
            New value for the degree of freedom
        
        Raises
        ------
        IndexError
            If atom_idx is out of range
        ValueError
            If dof_type is invalid
        """
        if atom_idx < 0 or atom_idx >= len(self._atoms):
            raise IndexError(f"Atom index {atom_idx} out of range [0, {len(self._atoms)-1}]")
        if dof_type < 0 or dof_type >= len(self.DOF_NAMES):
            raise ValueError(f"DOF type {dof_type} must be in [0, {len(self.DOF_NAMES)-1}]")
        
        dof_name = self.DOF_NAMES[dof_type]
        if dof_name not in self._atoms[atom_idx]:
            raise ValueError(f"Atom {atom_idx} does not have {dof_name}")
        
        self._atoms[atom_idx][dof_name] = value
    
    def get_dof(self, atom_idx: int, dof_type: int) -> float:
        """
        Get a degree of freedom value.
        
        Parameters
        ----------
        atom_idx : int
            Atom index (0-based)
        dof_type : int
            Degree of freedom type: 0=bond_length, 1=angle, 2=dihedral
        
        Returns
        -------
        float
            Value of the degree of freedom
        
        Raises
        ------
        IndexError
            If atom_idx is out of range
        ValueError
            If dof_type is invalid or DOF doesn't exist
        """
        if atom_idx < 0 or atom_idx >= len(self._atoms):
            raise IndexError(f"Atom index {atom_idx} out of range [0, {len(self._atoms)-1}]")
        if dof_type < 0 or dof_type >= len(self.DOF_NAMES):
            raise ValueError(f"DOF type {dof_type} must be in [0, {len(self.DOF_NAMES)-1}]")
        
        dof_name = self.DOF_NAMES[dof_type]
        if dof_name not in self._atoms[atom_idx]:
            raise ValueError(f"Atom {atom_idx} does not have {dof_name}")
        
        return self._atoms[atom_idx][dof_name]
    
    def copy(self) -> 'ZMatrix':
        """
        Create a deep copy of the Z-matrix.
        
        Returns
        -------
        ZMatrix
            Deep copy of this Z-matrix
        """
        return ZMatrix(self._atoms, self._bonds)
    
    def to_list(self) -> List[Dict]:
        """
        Convert to List[Dict] format for backward compatibility.
        
        Returns
        -------
        List[Dict]
            List of atom dictionaries (deep copy)
        """
        return copy.deepcopy(self._atoms)
    
    @classmethod
    def from_list(cls, atoms: List[Dict], bonds: List[Tuple[int, int, int]]) -> 'ZMatrix':
        """
        Create ZMatrix from List[Dict] format.
        
        Parameters
        ----------
        atoms : List[Dict]
            List of atom dictionaries
        bonds : List[Tuple[int, int, int]]
            List of bonds
        
        Returns
        -------
        ZMatrix
            New ZMatrix instance
        """
        return cls(atoms, bonds)
    
    def get_elements(self) -> List[str]:
        """
        Get list of element symbols.
        
        Returns
        -------
        List[str]
            List of element symbols for all atoms
        """
        return [atom['element'] for atom in self._atoms]
    
    def get_rotatable_indices(self) -> List[int]:
        """
        Get all rotatable atom indices (atoms with dihedrals where chirality == 0).
        
        Only atoms 4+ (index 3+) have dihedrals. Atoms with chirality != 0
        are not considered rotatable.
        
        Returns
        -------
        List[int]
            0-based indices of rotatable atoms
        """
        rotatable_indices = []
        for i in range(3, len(self._atoms)):  # Only atoms 4+ have dihedrals
            atom = self._atoms[i]
            if atom.get('chirality', 0) == 0:  # Only true dihedrals (not chirality-constrained)
                rotatable_indices.append(i)
        return rotatable_indices

