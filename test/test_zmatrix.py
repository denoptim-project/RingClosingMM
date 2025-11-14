#!/usr/bin/env python3
"""
Unit tests for ZMatrix class.

Tests the Z-matrix data structure encapsulation, including initialization,
validation, list-like interface, and all methods.
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path for package imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ringclosingmm import ZMatrix


class TestZMatrixInitialization(unittest.TestCase):
    """Test ZMatrix initialization and validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Simple valid Z-matrix
        self.valid_atoms = [
            {'id': 0, 'element': 'C', 'atomic_num': 6},
            {'id': 1, 'element': 'C', 'atomic_num': 6, 
             'bond_ref': 0, 'bond_length': 1.5},
            {'id': 2, 'element': 'C', 'atomic_num': 6,
             'bond_ref': 1, 'bond_length': 1.5,
             'angle_ref': 0, 'angle': 109.5},
            {'id': 3, 'element': 'H', 'atomic_num': 1,
             'bond_ref': 2, 'bond_length': 1.1,
             'angle_ref': 1, 'angle': 109.5,
             'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0}
        ]
        self.valid_bonds = [
            (0, 1, 1),
            (1, 2, 1),
            (2, 3, 1),
        ]
    
    def test_valid_initialization(self):
        """Test initialization with valid data."""
        zmat = ZMatrix(self.valid_atoms, self.valid_bonds)
        self.assertEqual(len(zmat), 4)
        self.assertEqual(len(zmat.bonds), 3)
    
    def test_initialization_sets_id(self):
        """Test that initialization sets correct id values."""
        atoms_no_id = [
            {'element': 'C', 'atomic_num': 6},
            {'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.5}
        ]
        zmat = ZMatrix(atoms_no_id, [(0, 1, 1)])
        self.assertEqual(zmat[0]['id'], 0)
        self.assertEqual(zmat[1]['id'], 1)
    
    def test_initialization_raises_error_for_inconsistent_id(self):
        """Test that initialization raises error for inconsistent id values."""
        atoms_wrong_id = [
            {'id': 5, 'element': 'C', 'atomic_num': 6},  # Wrong id
            {'id': 10, 'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.5}
        ]
        with self.assertRaises(ValueError) as cm:
            ZMatrix(atoms_wrong_id, [(0, 1, 1)])
        self.assertIn("inconsistent id", str(cm.exception))
    
    def test_empty_zmatrix(self):
        """Test initialization with empty Z-matrix."""
        zmat = ZMatrix([], [])
        self.assertEqual(len(zmat), 0)
        self.assertEqual(len(zmat.bonds), 0)
    
    def test_single_atom(self):
        """Test initialization with single atom."""
        atoms = [{'element': 'H', 'atomic_num': 1}]
        zmat = ZMatrix(atoms, [])
        self.assertEqual(len(zmat), 1)
        self.assertEqual(zmat[0]['element'], 'H')
    
    def test_atoms_not_list(self):
        """Test that atoms must be a list."""
        with self.assertRaises(ValueError) as cm:
            ZMatrix("not a list", self.valid_bonds)
        self.assertIn("atoms must be a list", str(cm.exception))
    
    def test_bonds_not_list(self):
        """Test that bonds must be a list."""
        with self.assertRaises(ValueError) as cm:
            ZMatrix(self.valid_atoms, "not a list")
        self.assertIn("bonds must be a list", str(cm.exception))
    
    def test_atom_not_dict(self):
        """Test that atoms must be dictionaries."""
        atoms = [{'element': 'C'}, "not a dict"]
        with self.assertRaises(ValueError) as cm:
            ZMatrix(atoms, [])
        self.assertIn("must be a dictionary", str(cm.exception))
    
    def test_bond_ref_out_of_range(self):
        """Test that bond_ref must be in valid range."""
        atoms = [
            {'element': 'C'},
            {'element': 'C', 'bond_ref': 10, 'bond_length': 1.5}  # Out of range
        ]
        with self.assertRaises(ValueError) as cm:
            ZMatrix(atoms, [(0, 1, 1)])
        self.assertIn("out of range", str(cm.exception))
    
    def test_angle_ref_out_of_range(self):
        """Test that angle_ref must be in valid range."""
        atoms = [
            {'element': 'C'},
            {'element': 'C', 'bond_ref': 0, 'bond_length': 1.5},
            {'element': 'C', 'bond_ref': 1, 'bond_length': 1.5,
             'angle_ref': 10, 'angle': 109.5}  # Out of range
        ]
        with self.assertRaises(ValueError) as cm:
            ZMatrix(atoms, [(0, 1, 1), (1, 2, 1)])
        self.assertIn("out of range", str(cm.exception))
    
    def test_bond_index_out_of_range(self):
        """Test that bond atom indices must be in valid range."""
        atoms = [{'element': 'C'}, {'element': 'C'}]
        bonds = [(0, 10, 1)]  # Out of range
        with self.assertRaises(ValueError) as cm:
            ZMatrix(atoms, bonds)
        self.assertIn("out of range", str(cm.exception))
    
    def test_bond_ref_not_integer(self):
        """Test that bond_ref must be an integer."""
        atoms = [
            {'element': 'C'},
            {'element': 'C', 'bond_ref': 'not an int', 'bond_length': 1.5}
        ]
        with self.assertRaises(ValueError) as cm:
            ZMatrix(atoms, [(0, 1, 1)])
        self.assertIn("must be an integer", str(cm.exception))


class TestZMatrixListInterface(unittest.TestCase):
    """Test list-like interface of ZMatrix."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.atoms = [
            {'element': 'C', 'atomic_num': 6},
            {'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.5},
            {'element': 'H', 'atomic_num': 1, 'bond_ref': 1, 'bond_length': 1.1}
        ]
        self.bonds = [(0, 1, 1), (1, 2, 1)]
        self.zmat = ZMatrix(self.atoms, self.bonds)
    
    def test_len(self):
        """Test __len__ method."""
        self.assertEqual(len(self.zmat), 3)
    
    def test_getitem_valid_index(self):
        """Test __getitem__ with valid index."""
        atom = self.zmat[0]
        self.assertEqual(atom['element'], 'C')
        self.assertEqual(atom['atomic_num'], 6)
    
    def test_getitem_returns_reference(self):
        """Test that __getitem__ returns a reference, not a copy."""
        atom = self.zmat[1]
        atom['bond_length'] = 2.0
        # Should modify the original
        self.assertEqual(self.zmat[1]['bond_length'], 2.0)
    
    def test_getitem_invalid_index(self):
        """Test __getitem__ with invalid index."""
        with self.assertRaises(IndexError):
            _ = self.zmat[10]
        # Note: negative indices work in Python (they wrap around), so -1 would access the last element
        # We test with an index that's definitely out of range
        with self.assertRaises(IndexError):
            _ = self.zmat[100]
    
    def test_setitem_valid(self):
        """Test __setitem__ with valid data."""
        new_atom = {'element': 'N', 'atomic_num': 7, 'bond_ref': 0, 'bond_length': 1.4}
        self.zmat[2] = new_atom
        self.assertEqual(self.zmat[2]['element'], 'N')
        self.assertEqual(self.zmat[2]['id'], 2)  # id should be set correctly
    
    def test_setitem_not_dict(self):
        """Test __setitem__ with non-dict value."""
        with self.assertRaises(ValueError) as cm:
            self.zmat[0] = "not a dict"
        self.assertIn("must be a dictionary", str(cm.exception))
    
    def test_setitem_validates_references(self):
        """Test that __setitem__ validates references."""
        invalid_atom = {'element': 'C', 'bond_ref': 10, 'bond_length': 1.5}
        with self.assertRaises(ValueError):
            self.zmat[1] = invalid_atom
    
    def test_iter(self):
        """Test __iter__ method."""
        elements = [atom['element'] for atom in self.zmat]
        self.assertEqual(elements, ['C', 'C', 'H'])
    
    def test_repr(self):
        """Test __repr__ method."""
        repr_str = repr(self.zmat)
        self.assertIn("ZMatrix", repr_str)
        self.assertIn("n_atoms=3", repr_str)
        self.assertIn("n_bonds=2", repr_str)


class TestZMatrixProperties(unittest.TestCase):
    """Test ZMatrix properties."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.atoms = [
            {'element': 'C', 'atomic_num': 6},
            {'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.5}
        ]
        self.bonds = [(0, 1, 1)]
        self.zmat = ZMatrix(self.atoms, self.bonds)
    
    def test_atoms_property_returns_copy(self):
        """Test that atoms property returns a copy."""
        atoms_copy = self.zmat.atoms
        atoms_copy[0]['element'] = 'N'
        # Original should not be modified
        self.assertEqual(self.zmat[0]['element'], 'C')
    
    def test_bonds_property_returns_copy(self):
        """Test that bonds property returns a copy."""
        bonds_copy = self.zmat.bonds
        bonds_copy.append((1, 2, 1))
        # Original should not be modified
        self.assertEqual(len(self.zmat.bonds), 1)


class TestZMatrixMethods(unittest.TestCase):
    """Test ZMatrix methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.atoms = [
            {'element': 'C', 'atomic_num': 6},
            {'element': 'C', 'atomic_num': 6, 'bond_ref': 0, 'bond_length': 1.5},
            {'element': 'C', 'atomic_num': 6,
             'bond_ref': 1, 'bond_length': 1.5,
             'angle_ref': 0, 'angle': 109.5},
            {'element': 'H', 'atomic_num': 1,
             'bond_ref': 2, 'bond_length': 1.1,
             'angle_ref': 1, 'angle': 109.5,
             'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 0}
        ]
        self.bonds = [(0, 1, 1), (1, 2, 1), (2, 3, 1)]
        self.zmat = ZMatrix(self.atoms, self.bonds)
    
    def test_get_atom_valid(self):
        """Test get_atom with valid index."""
        atom = self.zmat.get_atom(1)
        self.assertEqual(atom['element'], 'C')
        self.assertEqual(atom['bond_length'], 1.5)
    
    def test_get_atom_returns_copy(self):
        """Test that get_atom returns a copy."""
        atom = self.zmat.get_atom(1)
        atom['bond_length'] = 2.0
        # Original should not be modified
        self.assertEqual(self.zmat[1]['bond_length'], 1.5)
    
    def test_get_atom_invalid_index(self):
        """Test get_atom with invalid index."""
        with self.assertRaises(IndexError) as cm:
            self.zmat.get_atom(10)
        self.assertIn("out of range", str(cm.exception))
    
    def test_get_bonds(self):
        """Test get_bonds method."""
        bonds = self.zmat.get_bonds()
        self.assertEqual(len(bonds), 3)
        self.assertEqual(bonds[0], (0, 1, 1))
    
    def test_get_bonds_returns_copy(self):
        """Test that get_bonds returns a copy."""
        bonds = self.zmat.get_bonds()
        bonds.append((3, 4, 1))
        # Original should not be modified
        self.assertEqual(len(self.zmat.get_bonds()), 3)
    
    def test_update_dof_bond_length(self):
        """Test update_dof for bond_length."""
        self.zmat.update_dof(1, 0, 2.0)  # 0 = bond_length
        self.assertEqual(self.zmat[1]['bond_length'], 2.0)
    
    def test_update_dof_angle(self):
        """Test update_dof for angle."""
        self.zmat.update_dof(2, 1, 120.0)  # 1 = angle
        self.assertEqual(self.zmat[2]['angle'], 120.0)
    
    def test_update_dof_dihedral(self):
        """Test update_dof for dihedral."""
        self.zmat.update_dof(3, 2, 90.0)  # 2 = dihedral
        self.assertEqual(self.zmat[3]['dihedral'], 90.0)
    
    def test_update_dof_invalid_atom_index(self):
        """Test update_dof with invalid atom index."""
        with self.assertRaises(IndexError) as cm:
            self.zmat.update_dof(10, 0, 2.0)
        self.assertIn("out of range", str(cm.exception))
    
    def test_update_dof_invalid_dof_type(self):
        """Test update_dof with invalid DOF type."""
        with self.assertRaises(ValueError) as cm:
            self.zmat.update_dof(1, 10, 2.0)
        self.assertIn("must be in", str(cm.exception))
    
    def test_update_dof_missing_dof(self):
        """Test update_dof when DOF doesn't exist."""
        with self.assertRaises(ValueError) as cm:
            self.zmat.update_dof(0, 0, 2.0)  # Atom 0 has no bond_length
        self.assertIn("does not have", str(cm.exception))
    
    def test_get_dof_bond_length(self):
        """Test get_dof for bond_length."""
        value = self.zmat.get_dof(1, 0)  # 0 = bond_length
        self.assertEqual(value, 1.5)
    
    def test_get_dof_angle(self):
        """Test get_dof for angle."""
        value = self.zmat.get_dof(2, 1)  # 1 = angle
        self.assertEqual(value, 109.5)
    
    def test_get_dof_dihedral(self):
        """Test get_dof for dihedral."""
        value = self.zmat.get_dof(3, 2)  # 2 = dihedral
        self.assertEqual(value, 60.0)
    
    def test_get_dof_invalid_atom_index(self):
        """Test get_dof with invalid atom index."""
        with self.assertRaises(IndexError) as cm:
            self.zmat.get_dof(10, 0)
        self.assertIn("out of range", str(cm.exception))
    
    def test_get_dof_invalid_dof_type(self):
        """Test get_dof with invalid DOF type."""
        with self.assertRaises(ValueError) as cm:
            self.zmat.get_dof(1, 10)
        self.assertIn("must be in", str(cm.exception))
    
    def test_get_dof_missing_dof(self):
        """Test get_dof when DOF doesn't exist."""
        with self.assertRaises(ValueError) as cm:
            self.zmat.get_dof(0, 0)  # Atom 0 has no bond_length
        self.assertIn("does not have", str(cm.exception))
    
    def test_copy(self):
        """Test copy method."""
        zmat_copy = self.zmat.copy()
        self.assertEqual(len(zmat_copy), len(self.zmat))
        self.assertEqual(zmat_copy[1]['bond_length'], self.zmat[1]['bond_length'])
    
    def test_copy_is_independent(self):
        """Test that copy is independent of original."""
        zmat_copy = self.zmat.copy()
        zmat_copy[1]['bond_length'] = 2.0
        # Original should not be modified
        self.assertEqual(self.zmat[1]['bond_length'], 1.5)
    
    def test_to_list(self):
        """Test to_list method."""
        atoms_list = self.zmat.to_list()
        self.assertEqual(len(atoms_list), 4)
        self.assertEqual(atoms_list[0]['element'], 'C')
    
    def test_to_list_returns_copy(self):
        """Test that to_list returns a copy."""
        atoms_list = self.zmat.to_list()
        atoms_list[0]['element'] = 'N'
        # Original should not be modified
        self.assertEqual(self.zmat[0]['element'], 'C')
    
    def test_from_list(self):
        """Test from_list class method."""
        zmat_new = ZMatrix.from_list(self.atoms, self.bonds)
        self.assertEqual(len(zmat_new), 4)
        self.assertEqual(zmat_new[1]['bond_length'], 1.5)
    
    def test_get_elements(self):
        """Test get_elements method."""
        elements = self.zmat.get_elements()
        self.assertEqual(elements, ['C', 'C', 'C', 'H'])
    
    def test_get_rotatable_indices(self):
        """Test get_rotatable_indices method."""
        rotatable = self.zmat.get_rotatable_indices()
        # Only atom 3 (index 3) has a dihedral with chirality == 0
        self.assertEqual(rotatable, [3])
    
    def test_get_rotatable_indices_excludes_chirality(self):
        """Test that get_rotatable_indices excludes atoms with chirality != 0."""
        atoms_with_chirality = [
            {'element': 'C'},
            {'element': 'C', 'bond_ref': 0, 'bond_length': 1.5},
            {'element': 'C', 'bond_ref': 1, 'bond_length': 1.5,
             'angle_ref': 0, 'angle': 109.5},
            {'element': 'H', 'bond_ref': 2, 'bond_length': 1.1,
             'angle_ref': 1, 'angle': 109.5,
             'dihedral_ref': 0, 'dihedral': 60.0, 'chirality': 1}  # Not rotatable
        ]
        zmat = ZMatrix(atoms_with_chirality, [(0, 1, 1), (1, 2, 1), (2, 3, 1)])
        rotatable = zmat.get_rotatable_indices()
        # Atom 3 has chirality != 0, so it's not rotatable
        self.assertEqual(rotatable, [])
    
    def test_get_rotatable_indices_only_atoms_4_plus(self):
        """Test that only atoms 4+ (index 3+) are considered for rotatability."""
        # Atoms 0, 1, 2 don't have dihedrals, so they're not rotatable
        rotatable = self.zmat.get_rotatable_indices()
        self.assertNotIn(0, rotatable)
        self.assertNotIn(1, rotatable)
        self.assertNotIn(2, rotatable)


class TestZMatrixEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios."""
    
    def test_minimal_zmatrix(self):
        """Test minimal Z-matrix with just one atom."""
        atoms = [{'element': 'H', 'atomic_num': 1}]
        zmat = ZMatrix(atoms, [])
        self.assertEqual(len(zmat), 1)
        self.assertEqual(zmat.get_rotatable_indices(), [])
    
    def test_atom_without_optional_fields(self):
        """Test Z-matrix with atoms missing optional fields."""
        atoms = [
            {'element': 'C'},
            {'element': 'C', 'bond_ref': 0, 'bond_length': 1.5},
            {'element': 'C', 'bond_ref': 1, 'bond_length': 1.5}  # No angle
        ]
        zmat = ZMatrix(atoms, [(0, 1, 1), (1, 2, 1)])
        self.assertEqual(len(zmat), 3)
        # Should not raise error when accessing missing field
        self.assertNotIn('angle', zmat[2])
    
    def test_multiple_bonds_same_atoms(self):
        """Test Z-matrix with multiple bonds between same atoms."""
        atoms = [
            {'element': 'C'},
            {'element': 'C', 'bond_ref': 0, 'bond_length': 1.5}
        ]
        bonds = [(0, 1, 1), (1, 0, 1)]  # Same bond twice
        zmat = ZMatrix(atoms, bonds)
        self.assertEqual(len(zmat.bonds), 2)
    
    def test_dof_names_constant(self):
        """Test that DOF_NAMES constant is accessible."""
        self.assertEqual(ZMatrix.DOF_NAMES, ['bond_length', 'angle', 'dihedral'])
    
    def test_dict_like_access(self):
        """Test dict-like access to atom fields."""
        atoms = [
            {'element': 'C'},
            {'element': 'C', 'bond_ref': 0, 'bond_length': 1.5}
        ]
        zmat = ZMatrix(atoms, [(0, 1, 1)])
        # Should be able to access fields like a dict
        self.assertEqual(zmat[1]['bond_length'], 1.5)
        # Should be able to modify fields
        zmat[1]['bond_length'] = 2.0
        self.assertEqual(zmat[1]['bond_length'], 2.0)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestZMatrixInitialization))
    suite.addTests(loader.loadTestsFromTestCase(TestZMatrixListInterface))
    suite.addTests(loader.loadTestsFromTestCase(TestZMatrixProperties))
    suite.addTests(loader.loadTestsFromTestCase(TestZMatrixMethods))
    suite.addTests(loader.loadTestsFromTestCase(TestZMatrixEdgeCases))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)

