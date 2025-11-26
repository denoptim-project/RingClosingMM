#!/usr/bin/env python3
"""
Unit tests for RingClosingForceField module.

Tests geometry calculation functions, TopologicalInfo, and topological relation functions.
"""

import unittest
import numpy as np
import sys
from pathlib import Path
from collections import defaultdict

# Add parent directory to path for package imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ringclosingmm.RingClosingForceField import (
    getDistance,
    getAngle,
    getUnitVector,
    TopologicalInfo,
    _build_neighbors_from_bonds,
    _compute_and_store_topological_info,
    _compute_topological_info_from_topology,
    _get_atoms_in_1_2_relation,
    _get_atoms_in_1_3_relation,
    _get_atoms_in_1_4_relation,
    _get_atoms_at_distances
)


class TestRingClosingForceFieldGeometry(unittest.TestCase):
    """Test geometry calculation functions."""
    
    def test_getDistance(self):
        """Test distance calculation."""
        from openmm import unit
        
        p1 = np.array([0.0, 0.0, 0.0]) * unit.angstroms
        p2 = np.array([1.0, 0.0, 0.0]) * unit.angstroms
        
        distance = getDistance(p1, p2)
        
        self.assertAlmostEqual(distance, 1.0, places=5)
    
    def test_getDistance_diagonal(self):
        """Test distance calculation for diagonal."""
        from openmm import unit
        
        p1 = np.array([0.0, 0.0, 0.0]) * unit.angstroms
        p2 = np.array([1.0, 1.0, 1.0]) * unit.angstroms
        
        distance = getDistance(p1, p2)
        expected = np.sqrt(3.0)
        
        self.assertAlmostEqual(distance, expected, places=5)
    
    def test_getAngle_right_angle(self):
        """Test angle calculation for right angle."""
        from openmm import unit
        import math
        
        # For 90-degree angle: p1 at vertex, p0 and p2 perpendicular
        # Vectors: p0-p1 should be perpendicular to p2-p1
        p0 = np.array([-1.0, 0.0, 0.0]) * unit.angstroms  # Point on -x axis
        p1 = np.array([0.0, 0.0, 0.0]) * unit.angstroms   # Vertex at origin
        p2 = np.array([0.0, 1.0, 0.0]) * unit.angstroms   # Point on +y axis
        
        angle = getAngle(p0, p1, p2)
        
        # Right angle in radians = π/2 (returns radians, not degrees)
        # Vectors: p0-p1 = [-1,0,0], p2-p1 = [0,1,0] -> perpendicular
        self.assertAlmostEqual(angle, math.pi / 2, places=3)
    
    def test_getAngle_straight_line(self):
        """Test angle calculation for straight line."""
        from openmm import unit
        import math
        
        p0 = np.array([0.0, 0.0, 0.0]) * unit.angstroms
        p1 = np.array([1.0, 0.0, 0.0]) * unit.angstroms
        p2 = np.array([2.0, 0.0, 0.0]) * unit.angstroms
        
        angle = getAngle(p0, p1, p2)
        
        # Straight line angle in radians = π (returns radians, not degrees)
        self.assertAlmostEqual(angle, math.pi, places=3)
    
    def test_getUnitVector(self):
        """Test unit vector calculation."""
        from openmm import unit
        
        vector = np.array([3.0, 4.0, 0.0]) * unit.angstroms
        unit_vec = getUnitVector(vector)
        
        # Check magnitude is approximately 1
        magnitude = np.linalg.norm([unit_vec._value[0], unit_vec._value[1], unit_vec._value[2]])
        self.assertAlmostEqual(magnitude, 1.0, places=5)
    
    def test_getUnitVector_direction(self):
        """Test unit vector preserves direction."""
        from openmm import unit
        
        vector = np.array([1.0, 0.0, 0.0]) * unit.angstroms
        unit_vec = getUnitVector(vector)
        
        # Should point in same direction
        self.assertAlmostEqual(unit_vec._value[0], 1.0, places=5)
        self.assertAlmostEqual(unit_vec._value[1], 0.0, places=5)
        self.assertAlmostEqual(unit_vec._value[2], 0.0, places=5)


class TestTopologicalInfo(unittest.TestCase):
    """Test TopologicalInfo class."""
    
    def test_init(self):
        """Test TopologicalInfo initialization."""
        topo_info = TopologicalInfo()
        self.assertEqual(len(topo_info.rcp_relations), 0)
        self.assertIsInstance(topo_info.rcp_relations, dict)
    
    def test_add_rcp_relations(self):
        """Test adding relations for an RCP pair."""
        topo_info = TopologicalInfo()
        relation12 = {(0, 1), (1, 2)}
        relation13 = {(0, 2), (1, 3)}
        relation14 = {(0, 3)}
        
        topo_info.add_rcp_relations(5, 10, relation12, relation13, relation14)
        
        # Check that relations are stored with canonical key
        key = (5, 10)  # min, max
        self.assertIn(key, topo_info.rcp_relations)
        self.assertEqual(topo_info.rcp_relations[key]['relation12'], relation12)
        self.assertEqual(topo_info.rcp_relations[key]['relation13'], relation13)
        self.assertEqual(topo_info.rcp_relations[key]['relation14'], relation14)
    
    def test_add_rcp_relations_order_independent(self):
        """Test that RCP pair order doesn't matter."""
        topo_info = TopologicalInfo()
        relation12 = {(0, 1)}
        relation13 = {(0, 2)}
        relation14 = {(0, 3)}
        
        # Add with order (10, 5)
        topo_info.add_rcp_relations(10, 5, relation12, relation13, relation14)
        
        # Should be stored with canonical key (5, 10)
        key = (5, 10)
        self.assertIn(key, topo_info.rcp_relations)
    
    def test_get_relations_for_rcp(self):
        """Test getting relations for an RCP pair."""
        topo_info = TopologicalInfo()
        relation12 = {(0, 1), (1, 2)}
        relation13 = {(0, 2)}
        relation14 = {(0, 3)}
        
        topo_info.add_rcp_relations(5, 10, relation12, relation13, relation14)
        
        # Get relations
        relations = topo_info.get_relations_for_rcp(5, 10)
        self.assertEqual(relations['relation12'], relation12)
        self.assertEqual(relations['relation13'], relation13)
        self.assertEqual(relations['relation14'], relation14)
    
    def test_get_relations_for_rcp_order_independent(self):
        """Test that getting relations works regardless of order."""
        topo_info = TopologicalInfo()
        relation12 = {(0, 1)}
        topo_info.add_rcp_relations(5, 10, relation12, set(), set())
        
        # Get with reversed order
        relations1 = topo_info.get_relations_for_rcp(5, 10)
        relations2 = topo_info.get_relations_for_rcp(10, 5)
        
        self.assertEqual(relations1, relations2)
    
    def test_get_relations_for_rcp_nonexistent(self):
        """Test getting relations for non-existent RCP pair returns empty sets."""
        topo_info = TopologicalInfo()
        relations = topo_info.get_relations_for_rcp(5, 10)
        
        self.assertEqual(relations['relation12'], set())
        self.assertEqual(relations['relation13'], set())
        self.assertEqual(relations['relation14'], set())


class TestBuildNeighborsFromBonds(unittest.TestCase):
    """Test _build_neighbors_from_bonds function."""
    
    def test_build_neighbors_direct_indices(self):
        """Test building neighbors from bonds with direct indices."""
        # Mock bond objects with direct atom indices
        class MockBond:
            def __init__(self, atom1, atom2):
                self.atom1 = atom1
                self.atom2 = atom2
        
        bonds = [
            MockBond(0, 1),
            MockBond(1, 2),
            MockBond(2, 3)
        ]
        
        neighbors = _build_neighbors_from_bonds(bonds)
        
        self.assertEqual(neighbors[0], {1})
        self.assertEqual(neighbors[1], {0, 2})
        self.assertEqual(neighbors[2], {1, 3})
        self.assertEqual(neighbors[3], {2})
    
    def test_build_neighbors_with_index_attribute(self):
        """Test building neighbors from bonds with .index attribute."""
        # Mock bond objects with .index attribute (like topo.bonds())
        class MockAtom:
            def __init__(self, index):
                self.index = index
        
        class MockBond:
            def __init__(self, atom1_idx, atom2_idx):
                self.atom1 = MockAtom(atom1_idx)
                self.atom2 = MockAtom(atom2_idx)
        
        bonds = [
            MockBond(0, 1),
            MockBond(1, 2),
            MockBond(2, 3)
        ]
        
        neighbors = _build_neighbors_from_bonds(bonds)
        
        self.assertEqual(neighbors[0], {1})
        self.assertEqual(neighbors[1], {0, 2})
        self.assertEqual(neighbors[2], {1, 3})
        self.assertEqual(neighbors[3], {2})


class TestTopologicalRelations(unittest.TestCase):
    """Test topological relation functions."""
    
    def setUp(self):
        """Set up a simple molecular graph for testing."""
        # Create a simple chain: 0-1-2-3-4-5-6-7
        # RCP pair: 0-7 (ring closure)
        self.neighbors = defaultdict(set)
        self.neighbors[0] = {1}
        self.neighbors[1] = {0, 2}
        self.neighbors[2] = {1, 3}
        self.neighbors[3] = {2, 4}
        self.neighbors[4] = {3, 5}
        self.neighbors[5] = {4, 6}
        self.neighbors[6] = {5, 7}
        self.neighbors[7] = {6}
    
    def test_get_atoms_at_distances(self):
        """Test getting atoms at different distances."""
        atoms_a, atoms_b, atoms_c, atoms_d = _get_atoms_at_distances(0, self.neighbors)
        
        self.assertEqual(atoms_a, {0})  # Distance 0: self
        self.assertEqual(atoms_b, {1})  # Distance 1: direct neighbor
        self.assertEqual(atoms_c, {2})  # Distance 2: neighbor of neighbor
        self.assertEqual(atoms_d, {3})  # Distance 3: neighbor of distance 2
    
    def test_get_atoms_in_1_2_relation(self):
        """Test 1-2 relation calculation."""
        relations = _get_atoms_in_1_2_relation(0, 7, self.neighbors)
        
        # Check that expected pairs are present
        self.assertIn((0, 1), relations)  # Direct neighbor of 0
        self.assertIn((6, 7), relations)  # Direct neighbor of 7
        self.assertIn((0, 6), relations)  # Cross relation
        self.assertIn((1, 7), relations)  # Cross relation
    
    def test_get_atoms_in_1_3_relation(self):
        """Test 1-3 relation calculation."""
        relations = _get_atoms_in_1_3_relation(0, 7, self.neighbors)
        
        # Check that expected pairs are present
        self.assertIn((0, 2), relations)  # Distance 2 from 0
        self.assertIn((5, 7), relations)  # Distance 2 from 7
        self.assertIn((0, 5), relations)  # Cross relation
        self.assertIn((1, 6), relations)  # Cross relation
        self.assertIn((2, 7), relations)  # Cross relation
    
    def test_get_atoms_in_1_4_relation(self):
        """Test 1-4 relation calculation."""
        relations = _get_atoms_in_1_4_relation(0, 7, self.neighbors)
        
        # Check that expected pairs are present
        self.assertIn((0, 4), relations)  # Distance 3 from 0
        self.assertIn((3, 7), relations)  # Distance 3 from 7
        self.assertIn((0, 4), relations)  # Cross relation
        self.assertIn((2, 6), relations)  # Cross relation
        self.assertIn((1, 5), relations)  # Cross relation
    
    def test_relations_canonical_order(self):
        """Test that all relations return pairs in canonical (min, max) order."""
        relations12 = _get_atoms_in_1_2_relation(0, 4, self.neighbors)
        relations13 = _get_atoms_in_1_3_relation(0, 4, self.neighbors)
        relations14 = _get_atoms_in_1_4_relation(0, 4, self.neighbors)
        
        for pair in relations12 | relations13 | relations14:
            self.assertLess(pair[0], pair[1], f"Pair {pair} not in canonical order")


class TestComputeTopologicalInfo(unittest.TestCase):
    """Test topological info computation functions."""
    
    def test_compute_and_store_topological_info_caching(self):
        """Test that topological info is cached in args."""
        # Mock data object
        class MockBond:
            def __init__(self, atom1, atom2):
                self.atom1 = atom1
                self.atom2 = atom2
        
        class MockData:
            def __init__(self):
                self.bonds = [
                    MockBond(0, 1),
                    MockBond(1, 2),
                    MockBond(2, 3),
                    MockBond(3, 4)
                ]
        
        data = MockData()
        args = {'rcpterms': [[0, 4]], 'positions': [None] * 5}
        
        # First call should compute
        topo_info1 = _compute_and_store_topological_info(data, args)
        
        # Second call should return cached version
        topo_info2 = _compute_and_store_topological_info(data, args)
        
        self.assertIs(topo_info1, topo_info2)
        self.assertIn('topological_info', args)
        self.assertIs(args['topological_info'], topo_info1)
    
    def test_compute_and_store_topological_info_empty_rcpterms(self):
        """Test with empty RCP terms."""
        class MockData:
            def __init__(self):
                self.bonds = []
        
        data = MockData()
        args = {'rcpterms': [], 'positions': []}
        
        topo_info = _compute_and_store_topological_info(data, args)
        
        self.assertIsInstance(topo_info, TopologicalInfo)
        self.assertEqual(len(topo_info.rcp_relations), 0)
    
    def test_compute_topological_info_from_topology(self):
        """Test computing topological info from topology object."""
        from openmm.app import Topology
        from openmm.app.element import Element
        
        # Create a simple topology
        topo = Topology()
        chain = topo.addChain()
        residue = topo.addResidue("RES", chain)
        
        # Add atoms
        atoms = []
        for i in range(5):
            elem = Element.getByAtomicNumber(6)  # Carbon
            atom = topo.addAtom(f"C{i}", elem, residue)
            atoms.append(atom)
        
        # Add bonds: 0-1-2-3-4
        topo.addBond(atoms[0], atoms[1])
        topo.addBond(atoms[1], atoms[2])
        topo.addBond(atoms[2], atoms[3])
        topo.addBond(atoms[3], atoms[4])
        
        rcpterms = [(0, 4)]
        
        topo_info = _compute_topological_info_from_topology(topo, rcpterms)
        
        self.assertIsInstance(topo_info, TopologicalInfo)
        relations = topo_info.get_relations_for_rcp(0, 4)
        
        # Should have some relations
        self.assertGreater(len(relations['relation12']), 0)
        self.assertGreater(len(relations['relation13']), 0)
        self.assertGreater(len(relations['relation14']), 0)


def run_tests(verbosity=2):
    """Run all tests with specified verbosity."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRingClosingForceFieldGeometry))
    suite.addTests(loader.loadTestsFromTestCase(TestTopologicalInfo))
    suite.addTests(loader.loadTestsFromTestCase(TestBuildNeighborsFromBonds))
    suite.addTests(loader.loadTestsFromTestCase(TestTopologicalRelations))
    suite.addTests(loader.loadTestsFromTestCase(TestComputeTopologicalInfo))
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests(verbosity=2)
    sys.exit(0 if success else 1)

