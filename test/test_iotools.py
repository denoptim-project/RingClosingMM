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
from ringclosingmm.IOTools import read_int_file, write_zmatrix_file, write_xyz_file


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


def run_tests(verbosity=2):
    """Run all tests with specified verbosity."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestIOToolsFileIO))
    suite.addTests(loader.loadTestsFromTestCase(TestIOToolsBondModifications))
    suite.addTests(loader.loadTestsFromTestCase(TestIOToolsComplexZMatrices))
    suite.addTests(loader.loadTestsFromTestCase(TestIOToolsEdgeCases))
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests(verbosity=2)
    sys.exit(0 if success else 1)

