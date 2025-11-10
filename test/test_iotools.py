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

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from IOTools import read_int_file, write_zmatrix_file, write_xyz_file
from CoordinateConverter import zmatrix_to_cartesian


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
        
        self.assertIn('atoms', data)
        self.assertIn('zmatrix', data)
        self.assertIn('positions', data)
        self.assertIn('bonds', data)
        self.assertEqual(len(data['atoms']), 3)
        self.assertEqual(len(data['zmatrix']), 3)
    
    def test_read_int_file_zmatrix_structure(self):
        """Test that read INT file data has correct Z-matrix structure."""
        if not self.test_int_file.exists():
            self.skipTest(f"Test INT file not found: {self.test_int_file}")
        
        data = read_int_file(str(self.test_int_file))
        zmatrix = data['zmatrix']
        
        # First atom should have id, element, atomic_num
        self.assertIn('id', zmatrix[0])
        self.assertIn('element', zmatrix[0])
        self.assertIn('atomic_num', zmatrix[0])
        
        # Second atom should have bond_ref and bond_length
        if len(zmatrix) > 1:
            self.assertIn('bond_ref', zmatrix[1])
            self.assertIn('bond_length', zmatrix[1])
    
    def test_read_int_file_bonds(self):
        """Test that bonds are extracted correctly."""
        if not self.test_int_file.exists():
            self.skipTest(f"Test INT file not found: {self.test_int_file}")
        
        data = read_int_file(str(self.test_int_file))
        
        self.assertIn('bonds', data)
        self.assertIsInstance(data['bonds'], list)
        # Each bond should be a tuple of (atom1_idx, atom2_idx, bond_type)
        if len(data['bonds']) > 0:
            bond = data['bonds'][0]
            self.assertEqual(len(bond), 3)
            self.assertIsInstance(bond[0], int)
            self.assertIsInstance(bond[1], int)
            self.assertIsInstance(bond[2], int)
    
    def test_write_zmatrix_file(self):
        """Test writing Z-matrix to file."""
        zmatrix = [
            {'id': 0, 'element': 'H', 'atomic_num': 1},
            {'id': 1, 'element': 'H', 'atomic_num': 1, 'bond_ref': 0, 'bond_length': 1.0},
            {'id': 2, 'element': 'H', 'atomic_num': 1, 'bond_ref': 0, 'bond_length': 1.0,
             'angle_ref': 1, 'angle': 109.47}
        ]
        
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
        original_data = read_int_file(str(self.test_int_file))
        original_zmatrix = original_data['zmatrix']
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.int', delete=False) as f:
            temp_path = f.name
        
        try:
            write_zmatrix_file(original_zmatrix, temp_path)
            
            # Read back
            new_data = read_int_file(temp_path)
            new_zmatrix = new_data['zmatrix']
            
            # Compare structure
            self.assertEqual(len(original_zmatrix), len(new_zmatrix))
            
            for orig_atom, new_atom in zip(original_zmatrix, new_zmatrix):
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
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestIOToolsElementReplacement(unittest.TestCase):
    """Test element replacement in XYZ file writing."""
    
    def test_write_xyz_with_dummy_atoms(self):
        """Test that Du elements are replaced with He in XYZ files."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        elements = ['Du', 'C']
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            temp_path = f.name
        
        try:
            write_xyz_file(coords, elements, temp_path)
            
            with open(temp_path, 'r') as f:
                lines = f.readlines()
            
            # Check that Du was replaced with He
            self.assertIn('He', lines[2])
            self.assertNotIn('Du', lines[2])
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_write_xyz_with_atp_atoms(self):
        """Test that ATP elements are replaced with Ne in XYZ files."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        elements = ['ATP', 'H']
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            temp_path = f.name
        
        try:
            write_xyz_file(coords, elements, temp_path)
            
            with open(temp_path, 'r') as f:
                lines = f.readlines()
            
            # Check that ATP was replaced with Ne
            self.assertIn('Ne', lines[2])
            self.assertNotIn('ATP', lines[2])
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_write_xyz_with_atm_atoms(self):
        """Test that ATM elements are replaced with Ar in XYZ files."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        elements = ['ATM', 'H']
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            temp_path = f.name
        
        try:
            write_xyz_file(coords, elements, temp_path)
            
            with open(temp_path, 'r') as f:
                lines = f.readlines()
            
            # Check that ATM was replaced with Ar
            self.assertIn('Ar', lines[2])
            self.assertNotIn('ATM', lines[2])
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_write_xyz_with_all_special_elements(self):
        """Test replacement of all special element types."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        elements = ['Du', 'ATP', 'ATM', 'C']
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            temp_path = f.name
        
        try:
            write_xyz_file(coords, elements, temp_path)
            
            with open(temp_path, 'r') as f:
                lines = f.readlines()
            
            # Check replacements
            self.assertIn('He', lines[2])   # Du -> He
            self.assertIn('Ne', lines[3])   # ATP -> Ne
            self.assertIn('Ar', lines[4])   # ATM -> Ar
            self.assertIn('C', lines[5])    # C stays C
            
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
            bonds = data['bonds']
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
            bonds = data['bonds']
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
            
            bonds = data['bonds']
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
        zmatrix = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.54},
            {'id': 2, 'element': 'C', 'atomic_num': 6, 'bond_ref': 1, 'bond_length': 1.54,
             'angle_ref': 0, 'angle': 109.47},
            {'id': 3, 'element': 'C', 'atomic_num': 6, 'bond_ref': 2, 'bond_length': 1.54,
             'angle_ref': 1, 'angle': 109.47, 'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0}
        ]
        
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
        zmatrix = [
            {'id': 0, 'element': 'H', 'atomic_num': 1},
            {'id': 1, 'element': 'H', 'atomic_num': 1, 'bond_ref': 0, 'bond_length': 0.74},
        ]
        
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
            
            self.assertEqual(len(data['atoms']), 1)
            self.assertEqual(len(data['zmatrix']), 1)
            self.assertEqual(data['zmatrix'][0]['element'], 'H')
            self.assertEqual(len(data['bonds']), 0)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


def run_tests(verbosity=2):
    """Run all tests with specified verbosity."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestIOToolsFileIO))
    suite.addTests(loader.loadTestsFromTestCase(TestIOToolsElementReplacement))
    suite.addTests(loader.loadTestsFromTestCase(TestIOToolsBondModifications))
    suite.addTests(loader.loadTestsFromTestCase(TestIOToolsComplexZMatrices))
    suite.addTests(loader.loadTestsFromTestCase(TestIOToolsEdgeCases))
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests(verbosity=2)
    sys.exit(0 if success else 1)

