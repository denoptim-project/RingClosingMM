#!/usr/bin/env python3
"""
Unit tests for IOTools module.

Tests file I/O operations for reading and writing molecular structure files.
"""

import unittest
import numpy as np
import sys
import os
import tempfile
from pathlib import Path

# Add parent directory to path for package imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ringclosingmm import IOTools, ZMatrix
from ringclosingmm.IOTools import read_int_file, write_zmatrix_file, write_xyz_file, read_sdf_file, write_sdf_file


class TestIOToolsFileIO(unittest.TestCase):
    """Test file I/O operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(__file__).parent / 'fixtures'
        self.test_int_file = self.test_dir / 'simple_molecule.int'
    
    def test_read_int_file_valid_file(self):
        """Test reading valid INT file."""
        if not self.test_int_file.exists():
            self.skipTest(f"Test INT file not found: {self.test_int_file}")
        
        data = read_int_file(str(self.test_int_file))
        self.assertIsInstance(data, ZMatrix)
        self.assertEqual(len(data.atoms), 6)
        self.assertEqual(len(data.bonds), 6)
    
    def test_read_int_file_zmatrix_structure(self):
        """Test that read INT file data has correct Z-matrix structure."""
        if not self.test_int_file.exists():
            self.skipTest(f"Test INT file not found: {self.test_int_file}")
        
        data = read_int_file(str(self.test_int_file))
        zmatrix = data.atoms
        
        # First atom should have id, element, atomic_num
        self.assertIn('id', data[0])
        self.assertIn('element', data[0])
        self.assertIn('atomic_num', data[0])
        
        # Second atom should have bond_ref and bond_length
        if len(zmatrix) > 1:
            self.assertIn('bond_ref', zmatrix[1])
            self.assertIn('bond_length', zmatrix[1])
    
    def test_read_int_file_bonds(self):
        """Test that bonds are extracted correctly."""
        if not self.test_int_file.exists():
            self.skipTest(f"Test INT file not found: {self.test_int_file}")
        
        zmatrix = read_int_file(str(self.test_int_file))
        
        self.assertIsInstance(zmatrix.bonds, list)
        self.assertEqual(len(zmatrix.bonds), 6)
        # Each bond should be a tuple of (atom1_idx, atom2_idx, bond_type)
        if len(zmatrix.bonds) > 0:
            for bond in zmatrix.bonds:
                self.assertEqual(len(bond), 3)
                self.assertIsInstance(bond[0], int)
                self.assertIsInstance(bond[1], int)
                self.assertIsInstance(bond[2], int)
        # Check that the bonds are correct
        expected_bonds = [(0, 1, 1), (1, 2, 1), (2, 4, 1), (2, 5, 1), (4, 5, 1), (3, 4, 1)]
        actual_bonds_sorted = [(bond[0], bond[1], bond[2]) if bond[0] < bond[1] else (bond[1], bond[0], bond[2]) for bond in zmatrix.bonds]
        for bond in actual_bonds_sorted:
            self.assertIn(bond, expected_bonds)
    
    def test_write_zmatrix_file(self):
        """Test writing Z-matrix to file."""
        zmatrix_atoms = [
            {'id': 0, 'element': 'H', 'atomic_num': 1},
            {'id': 1, 'element': 'H', 'atomic_num': 1, 'bond_ref': 0, 'bond_length': 1.0},
            {'id': 2, 'element': 'H', 'atomic_num': 1, 'bond_ref': 0, 'bond_length': 1.0,
             'angle_ref': 1, 'angle': 109.47}
        ]
        zmatrix = ZMatrix(zmatrix_atoms, [(0, 1, 1), (0, 2, 1)])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.int', delete=False) as f:
            temp_path = f.name
        
        try:
            write_zmatrix_file(zmatrix, temp_path)
            
            # Verify file was created and has correct content
            self.assertTrue(os.path.exists(temp_path))
            
            with open(temp_path, 'r') as f:
                lines = f.readlines()
            
            # First line should be number of atoms
            self.assertEqual(int(lines[0].strip()), 3)
            
            # Verify structure
            self.assertIn('H', lines[1])
            self.assertIn('H', lines[2])
            self.assertIn('H', lines[3])
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_write_xyz_file(self):
        """Test writing XYZ file."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0]
        ])
        elements = ['H', 'H', 'H']
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            temp_path = f.name
        
        try:
            write_xyz_file(coords, elements, temp_path, comment="Test molecule")
            
            # Verify file was created
            self.assertTrue(os.path.exists(temp_path))
            
            with open(temp_path, 'r') as f:
                lines = f.readlines()
            
            # First line: number of atoms
            self.assertEqual(int(lines[0].strip()), 3)
            # Second line: comment
            self.assertEqual(lines[1].strip(), "Test molecule")
            # Following lines: atom coordinates
            self.assertEqual(len(lines), 5)  # 3 atoms + 2 header lines
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_write_xyz_file_append_mode(self):
        """Test writing XYZ file in append mode."""
        coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        coords2 = np.array([[0.0, 0.0, 0.5], [1.0, 0.0, 0.5]])
        elements = ['H', 'H']
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            temp_path = f.name
        
        try:
            # Write first frame
            write_xyz_file(coords1, elements, temp_path, comment="Frame 1")
            # Append second frame
            write_xyz_file(coords2, elements, temp_path, comment="Frame 2", append=True)
            
            with open(temp_path, 'r') as f:
                lines = f.readlines()
            
            # Should have 2 frames (each frame has 4 lines: count, comment, 2 atoms)
            self.assertEqual(len(lines), 8)
            
            # Verify first frame
            self.assertEqual(int(lines[0].strip()), 2)
            self.assertEqual(lines[1].strip(), "Frame 1")
            
            # Verify second frame
            self.assertEqual(int(lines[4].strip()), 2)
            self.assertEqual(lines[5].strip(), "Frame 2")
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_read_write_roundtrip(self):
        """Test that reading and writing INT file preserves data."""
        if not self.test_int_file.exists():
            self.skipTest(f"Test INT file not found: {self.test_int_file}")
        
        # Read original file
        original_zmatrix = read_int_file(str(self.test_int_file))
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.int', delete=False) as f:
            temp_path = f.name
        
        try:
            write_zmatrix_file(original_zmatrix, temp_path)
            
            # Read back
            new_zmatrix = read_int_file(temp_path)
            
            # Compare structure
            self.assertEqual(len(original_zmatrix.atoms), len(new_zmatrix.atoms))
            self.assertEqual(len(original_zmatrix.bonds), len(new_zmatrix.bonds))
            
            for orig_atom, new_atom in zip(original_zmatrix.atoms, new_zmatrix.atoms):
                self.assertEqual(orig_atom['id'], new_atom['id'])
                self.assertEqual(orig_atom['element'], new_atom['element'])
                self.assertEqual(orig_atom['atomic_num'], new_atom['atomic_num'])
                
                if 'bond_ref' in orig_atom:
                    self.assertIn('bond_ref', new_atom)
                    self.assertEqual(orig_atom['bond_ref'], new_atom['bond_ref'])
                    self.assertAlmostEqual(orig_atom['bond_length'], new_atom['bond_length'], places=5)
                
                if 'angle_ref' in orig_atom:
                    self.assertIn('angle_ref', new_atom)
                    self.assertEqual(orig_atom['angle_ref'], new_atom['angle_ref'])
                    self.assertAlmostEqual(orig_atom['angle'], new_atom['angle'], places=5)
                
                if 'dihedral_ref' in orig_atom:
                    self.assertIn('dihedral_ref', new_atom)
                    self.assertEqual(orig_atom['dihedral_ref'], new_atom['dihedral_ref'])
                    self.assertAlmostEqual(orig_atom['dihedral'], new_atom['dihedral'], places=5)
                    self.assertEqual(orig_atom.get('chirality', 0), new_atom.get('chirality', 0))

            # Compare bonds
            original_bonds_sorted = [(bond[0], bond[1], bond[2]) if bond[0] < bond[1] else (bond[1], bond[0], bond[2]) for bond in original_zmatrix.bonds]
            new_bonds_sorted = [(bond[0], bond[1], bond[2]) if bond[0] < bond[1] else (bond[1], bond[0], bond[2]) for bond in new_zmatrix.bonds]

            for bond in new_bonds_sorted:
                self.assertIn(bond, original_bonds_sorted)
            for bond in original_bonds_sorted:
                self.assertIn(bond, new_bonds_sorted)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestIOToolsBondModifications(unittest.TestCase):
    """Test bond addition and removal in INT files."""
    
    def test_read_int_file_with_bond_additions(self):
        """Test reading INT file with bonds to add."""
        int_content = """4
     1  C   6
     2  C   6     1  1.540000
     3  C   6     2  1.540000     1 109.470000
     4  C   6     3  1.540000     2 109.470000     1  60.000000  0

1 4
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.int', delete=False) as f:
            temp_path = f.name
            f.write(int_content)
        
        try:
            data = read_int_file(temp_path)
            
            # Check that bond (1,4) was added (0-based: 0,3)
            bonds = data.bonds
            bond_set = {(min(a, b), max(a, b)) for a, b, _ in bonds}
            self.assertIn((0, 3), bond_set)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_read_int_file_with_bond_removals(self):
        """Test reading INT file with bonds to remove."""
        int_content = """4
     1  C   6
     2  C   6     1  1.540000
     3  C   6     2  1.540000     1 109.470000
     4  C   6     3  1.540000     2 109.470000     1  60.000000  0


2 3
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.int', delete=False) as f:
            temp_path = f.name
            f.write(int_content)
        
        try:
            data = read_int_file(temp_path)
            
            # Check that bond (2,3) was removed (0-based: 1,2)
            bonds = data.bonds
            bond_set = {(min(a, b), max(a, b)) for a, b, _ in bonds}
            self.assertNotIn((1, 2), bond_set)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_read_int_file_with_both_modifications(self):
        """Test reading INT file with both bond additions and removals."""
        int_content = """4
     1  C   6
     2  C   6     1  1.540000
     3  C   6     2  1.540000     1 109.470000
     4  C   6     3  1.540000     2 109.470000     1  60.000000  0

1 4

2 3
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.int', delete=False) as f:
            temp_path = f.name
            f.write(int_content)
        
        try:
            data = read_int_file(temp_path)
            
            bonds = data.bonds
            bond_set = {(min(a, b), max(a, b)) for a, b, _ in bonds}
            
            # Check additions and removals
            self.assertIn((0, 3), bond_set)   # Bond added
            self.assertNotIn((1, 2), bond_set)  # Bond removed
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestIOToolsComplexZMatrices(unittest.TestCase):
    """Test handling of complex Z-matrix structures."""
    
    def test_write_zmatrix_with_dihedrals(self):
        """Test writing Z-matrix with dihedral angles."""
        zmatrix_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0}
        ]
        zmatrix = ZMatrix(zmatrix_atoms, [(0, 1, 1), (1, 2, 1), (2, 3, 1)])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.int', delete=False) as f:
            temp_path = f.name
        
        try:
            write_zmatrix_file(zmatrix, temp_path)
            
            with open(temp_path, 'r') as f:
                lines = f.readlines()
            
            # Verify structure
            self.assertEqual(int(lines[0].strip()), 4)
            
            # Fourth atom should have dihedral
            parts = lines[4].split()
            self.assertEqual(len(parts), 10)  # id, element, atomic_num, bond_ref, bond_length, 
                                               # angle_ref, angle, dihedral_ref, dihedral, chirality
            self.assertAlmostEqual(float(parts[8]), 60.0, places=4)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_write_zmatrix_formatting(self):
        """Test that Z-matrix file has correct formatting."""
        zmatrix_atoms = [
            {'id': 0, 'element': 'H', 'atomic_num': 1},
            {'id': 1, 'element': 'H', 'atomic_num': 1, 'bond_ref': 0, 'bond_length': 0.74},
        ]
        zmatrix = ZMatrix(zmatrix_atoms, [(0, 1, 1)])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.int', delete=False) as f:
            temp_path = f.name
        
        try:
            write_zmatrix_file(zmatrix, temp_path)
            
            with open(temp_path, 'r') as f:
                lines = f.readlines()
            
            # Check formatting
            self.assertTrue(lines[1].strip().startswith('1'))
            self.assertTrue(lines[2].strip().startswith('2'))
            
            # Check bond reference is 1-based in file
            parts = lines[2].split()
            self.assertEqual(int(parts[3]), 1)  # bond_ref should be 1 (1-based)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestIOToolsEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def test_write_xyz_empty_comment(self):
        """Test writing XYZ file with empty comment."""
        coords = np.array([[0.0, 0.0, 0.0]])
        elements = ['H']
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            temp_path = f.name
        
        try:
            write_xyz_file(coords, elements, temp_path, comment="")
            
            with open(temp_path, 'r') as f:
                lines = f.readlines()
            
            # Comment line should be empty (just newline)
            self.assertEqual(lines[1].strip(), "")
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_write_xyz_single_atom(self):
        """Test writing XYZ file with single atom."""
        coords = np.array([[1.5, 2.5, 3.5]])
        elements = ['C']
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            temp_path = f.name
        
        try:
            write_xyz_file(coords, elements, temp_path, comment="Single atom")
            
            with open(temp_path, 'r') as f:
                lines = f.readlines()
            
            self.assertEqual(int(lines[0].strip()), 1)
            self.assertEqual(len(lines), 3)  # count + comment + 1 atom
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_read_int_file_minimal(self):
        """Test reading minimal INT file (single atom)."""
        int_content = """1
     1  H   1
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.int', delete=False) as f:
            temp_path = f.name
            f.write(int_content)
        
        try:
            data = read_int_file(temp_path)
            
            self.assertEqual(len(data.atoms), 1)
            self.assertEqual(len(data.atoms), 1)
            self.assertEqual(data.atoms[0]['element'], 'H')
            self.assertEqual(len(data.bonds), 0)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestIOToolsSDF(unittest.TestCase):
    """Test SDF file reading operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(__file__).parent / 'fixtures'
        self.test_sdf_file = self.test_dir / 'mol2zmat.sdf'
    
    def test_read_sdf_file_valid_file(self):
        """Test reading valid SDF file."""
        if not self.test_sdf_file.exists():
            self.skipTest(f"Test SDF file not found: {self.test_sdf_file}")
        
        zmat = read_sdf_file(str(self.test_sdf_file))
        
        self.assertIsInstance(zmat, ZMatrix)
        self.assertEqual(len(zmat), 6)
        # Should have some bonds (those not used in Z-matrix construction)
        self.assertEqual(len(zmat.bonds), 7)
    
    def test_read_sdf_file_atom_data(self):
        """Test that SDF file data is correctly parsed into Z-matrix."""
        if not self.test_sdf_file.exists():
            self.skipTest(f"Test SDF file not found: {self.test_sdf_file}")
        
        zmat = read_sdf_file(str(self.test_sdf_file))
        
        # Check first atom
        self.assertEqual(zmat[0][ZMatrix.FIELD_ELEMENT], 'H')
        self.assertEqual(zmat[0][ZMatrix.FIELD_ATOMIC_NUM], 1)
        self.assertEqual(zmat[5][ZMatrix.FIELD_ELEMENT], 'N')
        self.assertEqual(zmat[5][ZMatrix.FIELD_ATOMIC_NUM], 7)
        
        # Check that all atoms have required fields
        for i in range(len(zmat)):
            self.assertIn(ZMatrix.FIELD_ID, zmat[i])
            self.assertIn(ZMatrix.FIELD_ELEMENT, zmat[i])
            self.assertIn(ZMatrix.FIELD_ATOMIC_NUM, zmat[i])
            self.assertEqual(zmat[i][ZMatrix.FIELD_ID], i)
    
    def test_read_sdf_file_zmatrix_structure(self):
        """Test that Z-matrix structure is correctly generated."""
        if not self.test_sdf_file.exists():
            self.skipTest(f"Test SDF file not found: {self.test_sdf_file}")
        
        zmat = read_sdf_file(str(self.test_sdf_file))
        
        # Atom 0 should have no internal coordinates
        self.assertNotIn(ZMatrix.FIELD_BOND_REF, zmat[0])
        
        # Atom 1 should have bond_ref
        self.assertIn(ZMatrix.FIELD_BOND_REF, zmat[1])
        self.assertIn(ZMatrix.FIELD_BOND_LENGTH, zmat[1])
        self.assertLess(zmat[1][ZMatrix.FIELD_BOND_REF], 1)  # Reference to previous atom
        
        # Atom 2 should have bond_ref and angle_ref
        if len(zmat) > 2:
            self.assertIn(ZMatrix.FIELD_BOND_REF, zmat[2])
            self.assertIn(ZMatrix.FIELD_ANGLE_REF, zmat[2])
            self.assertIn(ZMatrix.FIELD_ANGLE, zmat[2])
        
        # Atom 3+ should have dihedral or chirality
        if len(zmat) > 3:
            self.assertIn(ZMatrix.FIELD_DIHEDRAL_REF, zmat[3])
            self.assertIn(ZMatrix.FIELD_DIHEDRAL, zmat[3])
            self.assertIn(ZMatrix.FIELD_CHIRALITY, zmat[3])
    
    def test_read_sdf_file_reference_indices(self):
        """Test that reference indices are 0-based."""
        if not self.test_sdf_file.exists():
            self.skipTest(f"Test SDF file not found: {self.test_sdf_file}")
        
        zmat = read_sdf_file(str(self.test_sdf_file))
        
        for i in range(1, len(zmat)):
            atom = zmat[i]
            if ZMatrix.FIELD_BOND_REF in atom:
                self.assertGreaterEqual(atom[ZMatrix.FIELD_BOND_REF], 0)
                self.assertLess(atom[ZMatrix.FIELD_BOND_REF], i)
            if ZMatrix.FIELD_ANGLE_REF in atom:
                self.assertGreaterEqual(atom[ZMatrix.FIELD_ANGLE_REF], 0)
                self.assertLess(atom[ZMatrix.FIELD_ANGLE_REF], i)
            if ZMatrix.FIELD_DIHEDRAL_REF in atom:
                self.assertGreaterEqual(atom[ZMatrix.FIELD_DIHEDRAL_REF], 0)
                self.assertLess(atom[ZMatrix.FIELD_DIHEDRAL_REF], i)
    
    def test_read_sdf_file_invalid_file(self):
        """Test that invalid SDF file raises error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False) as f:
            temp_path = f.name
            f.write("invalid\ncontent\n")
        
        try:
            with self.assertRaises(ValueError):
                read_sdf_file(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_read_sdf_file_empty_file(self):
        """Test that empty SDF file raises error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False) as f:
            temp_path = f.name
            f.write("")
        
        try:
            with self.assertRaises(ValueError):
                read_sdf_file(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_read_sdf_file_minimal(self):
        """Test reading minimal SDF file (single atom)."""
        sdf_content = """mol
 OpenBabel
   
  1  0  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
M  END
$$$$
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False) as f:
            temp_path = f.name
            f.write(sdf_content)
        
        try:
            zmat = read_sdf_file(temp_path)
            
            self.assertEqual(len(zmat), 1)
            self.assertEqual(zmat[0][ZMatrix.FIELD_ELEMENT], 'H')
            self.assertEqual(zmat[0][ZMatrix.FIELD_ATOMIC_NUM], 1)
            self.assertEqual(len(zmat.bonds), 0)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_read_sdf_file_two_atoms(self):
        """Test reading SDF file with two atoms."""
        sdf_content = """mol
 OpenBabel
   
  2  1  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.0000    1.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
M  END
$$$$
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False) as f:
            temp_path = f.name
            f.write(sdf_content)
        
        try:
            zmat = read_sdf_file(temp_path)
            
            self.assertEqual(len(zmat), 2)
            self.assertEqual(zmat[0][ZMatrix.FIELD_ELEMENT], 'H')
            self.assertEqual(zmat[1][ZMatrix.FIELD_ELEMENT], 'H')
            self.assertIn(ZMatrix.FIELD_BOND_REF, zmat[1])
            self.assertAlmostEqual(zmat[1][ZMatrix.FIELD_BOND_LENGTH], 1.0, places=6)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_read_sdf_file_conversion_roundtrip(self):
        """Test that SDF -> Z-matrix -> Cartesian conversion works."""
        if not self.test_sdf_file.exists():
            self.skipTest(f"Test SDF file not found: {self.test_sdf_file}")
        
        from ringclosingmm.CoordinateConversion import zmatrix_to_cartesian
        
        zmat = read_sdf_file(str(self.test_sdf_file))
        
        # Convert back to Cartesian
        coords = zmatrix_to_cartesian(zmat)
        
        # Should have same number of atoms
        self.assertEqual(len(coords), len(zmat))
        self.assertEqual(coords.shape[1], 3)  # 3D coordinates
    
    def test_read_write_sdf_file_roundtrip(self):
        """Test reading SDF file, writing it back, and comparing with original."""
        if not self.test_sdf_file.exists():
            self.skipTest(f"Test SDF file not found: {self.test_sdf_file}")
        
        import tempfile
        
        # Read original SDF file
        zmat = read_sdf_file(str(self.test_sdf_file))
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False) as f:
            temp_path = f.name
        
        try:
            write_sdf_file(zmat, temp_path)
            
            # Read both files and compare
            with open(str(self.test_sdf_file), 'r') as f:
                original_lines = [line.rstrip('\n\r') for line in f.readlines()]
            
            with open(temp_path, 'r') as f:
                written_lines = [line.rstrip('\n\r') for line in f.readlines()]
            
            # Find counts line (should be line 3, 0-indexed, but find it by looking for V2000)
            original_counts = None
            written_counts = None
            for line in original_lines:
                if 'V2000' in line:
                    original_counts = line.strip()
                    break
            for line in written_lines:
                if 'V2000' in line:
                    written_counts = line.strip()
                    break
            
            self.assertIsNotNone(original_counts, "Could not find counts line in original file")
            self.assertIsNotNone(written_counts, "Could not find counts line in written file")
            
            # Extract atom and bond counts (format: "  N  M ...")
            # Split by whitespace and take first two numbers
            orig_parts = original_counts.split()
            writ_parts = written_counts.split()
            
            orig_num_atoms = int(orig_parts[0])
            orig_num_bonds = int(orig_parts[1])
            writ_num_atoms = int(writ_parts[0])
            writ_num_bonds = int(writ_parts[1])
            
            # Check counts match
            self.assertEqual(writ_num_atoms, orig_num_atoms, 
                           f"Atom count mismatch: written={writ_num_atoms}, original={orig_num_atoms}")
            # Bond count may differ slightly due to Z-matrix tree structure, but should be close
            self.assertGreaterEqual(writ_num_bonds, orig_num_bonds - 1,
                                  f"Bond count too low: written={writ_num_bonds}, original={orig_num_bonds}")
            
            # Find the line index where atoms start (after counts line with V2000)
            orig_atom_start = None
            writ_atom_start = None
            for i, line in enumerate(original_lines):
                if 'V2000' in line:
                    orig_atom_start = i + 1
                    break
            for i, line in enumerate(written_lines):
                if 'V2000' in line:
                    writ_atom_start = i + 1
                    break
            
            self.assertIsNotNone(orig_atom_start, "Could not find atom start in original file")
            self.assertIsNotNone(writ_atom_start, "Could not find atom start in written file")
            
            # Parse and compare atom coordinates
            original_atoms = []
            written_atoms = []
            
            for i in range(orig_num_atoms):
                orig_line = original_lines[orig_atom_start + i]
                orig_x = float(orig_line[0:10].strip())
                orig_y = float(orig_line[10:20].strip())
                orig_z = float(orig_line[20:30].strip())
                orig_element = orig_line[30:33].strip()
                original_atoms.append((orig_element, orig_x, orig_y, orig_z))
            
            for i in range(writ_num_atoms):
                writ_line = written_lines[writ_atom_start + i]
                writ_x = float(writ_line[0:10].strip())
                writ_y = float(writ_line[10:20].strip())
                writ_z = float(writ_line[20:30].strip())
                writ_element = writ_line[30:33].strip()
                written_atoms.append((writ_element, writ_x, writ_y, writ_z))
            
            # Match atoms by element and relative distances (Z-matrix conversion may rotate)
            # Calculate distances from first atom (origin) for each atom
            import math
            
            def calc_distance(x1, y1, z1, x2, y2, z2):
                return math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
            
            # Get first atom coordinates (should be at origin or close to it)
            orig_first = original_atoms[0]
            writ_first = written_atoms[0]
            
            # Calculate distances from first atom for all atoms
            orig_distances = []
            for orig_el, orig_x, orig_y, orig_z in original_atoms:
                dist = calc_distance(orig_first[1], orig_first[2], orig_first[3], orig_x, orig_y, orig_z)
                orig_distances.append((orig_el, dist))
            
            writ_distances = []
            for writ_el, writ_x, writ_y, writ_z in written_atoms:
                dist = calc_distance(writ_first[1], writ_first[2], writ_first[3], writ_x, writ_y, writ_z)
                writ_distances.append((writ_el, dist))
            
            # Match atoms by element and distance from first atom
            matched_original = set()
            matched_written = set()
            max_dist_diff = 0.0
            
            for orig_idx, (orig_el, orig_dist) in enumerate(orig_distances):
                best_match = None
                best_diff = float('inf')
                
                for writ_idx, (writ_el, writ_dist) in enumerate(writ_distances):
                    if writ_idx in matched_written:
                        continue
                    # Check if elements match
                    if writ_el != orig_el:
                        continue
                    # Compare distances from first atom
                    dist_diff = abs(orig_dist - writ_dist)
                    if dist_diff < best_diff and dist_diff < 0.1:  # Allow up to 0.1 Å difference
                        best_diff = dist_diff
                        best_match = (writ_idx, writ_el, writ_dist)
                
                if best_match is None:
                    self.fail(f"Could not find matching atom for original atom {orig_idx} ({orig_el} at distance {orig_dist:.4f} from first atom)")
                
                writ_idx, writ_el, writ_dist = best_match
                matched_original.add(orig_idx)
                matched_written.add(writ_idx)
                max_dist_diff = max(max_dist_diff, best_diff)
            
            # All original atoms should be matched
            self.assertEqual(len(matched_original), orig_num_atoms,
                           f"Not all original atoms were matched: {len(matched_original)}/{orig_num_atoms}")
            
            # Distance differences should be small
            self.assertLess(max_dist_diff, 0.1,
                          f"Maximum distance difference too large: {max_dist_diff:.6f} Å")
            
            # Parse and compare bonds
            original_bonds = []
            written_bonds = []
            
            # Original bonds (lines after atoms until "M  END")
            orig_bond_start = orig_atom_start + orig_num_atoms
            for i in range(orig_num_bonds):
                if orig_bond_start + i >= len(original_lines) or original_lines[orig_bond_start + i].strip() == "M  END":
                    break
                line = original_lines[orig_bond_start + i]
                atom1 = int(line[0:3].strip())
                atom2 = int(line[3:6].strip())
                bond_type = int(line[6:9].strip())
                original_bonds.append((min(atom1, atom2), max(atom1, atom2), bond_type))
            
            # Written bonds
            writ_bond_start = writ_atom_start + writ_num_atoms
            for i in range(writ_num_bonds):
                if writ_bond_start + i >= len(written_lines) or written_lines[writ_bond_start + i].strip() == "M  END":
                    break
                line = written_lines[writ_bond_start + i]
                atom1 = int(line[0:3].strip())
                atom2 = int(line[3:6].strip())
                bond_type = int(line[6:9].strip())
                written_bonds.append((min(atom1, atom2), max(atom1, atom2), bond_type))
            
            # Convert to sets for comparison (ignore bond type for now)
            original_bond_pairs = {(a1, a2) for a1, a2, _ in original_bonds}
            written_bond_pairs = {(a1, a2) for a1, a2, _ in written_bonds}
            
            # All original bonds should be present in written file
            # (written file may have additional bonds from Z-matrix structure)
            missing_bonds = original_bond_pairs - written_bond_pairs
            self.assertEqual(len(missing_bonds), 0,
                           f"Missing bonds in written file: {missing_bonds}")
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


def run_tests(verbosity=2):
    """Run all tests with specified verbosity."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestIOToolsFileIO))
    suite.addTests(loader.loadTestsFromTestCase(TestIOToolsBondModifications))
    suite.addTests(loader.loadTestsFromTestCase(TestIOToolsComplexZMatrices))
    suite.addTests(loader.loadTestsFromTestCase(TestIOToolsEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestIOToolsSDF))
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests(verbosity=2)
    sys.exit(0 if success else 1)

