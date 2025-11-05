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
            {'id': 1, 'element': 'H', 'atomic_num': 1},
            {'id': 2, 'element': 'H', 'atomic_num': 1, 'bond_ref': 1, 'bond_length': 1.0},
            {'id': 3, 'element': 'H', 'atomic_num': 1, 'bond_ref': 1, 'bond_length': 1.0,
             'angle_ref': 2, 'angle': 109.47}
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


def run_tests(verbosity=2):
    """Run all tests with specified verbosity."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestIOToolsFileIO))
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests(verbosity=2)
    sys.exit(0 if success else 1)

