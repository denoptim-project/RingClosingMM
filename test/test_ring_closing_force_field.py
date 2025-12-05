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
        # Check that crossing and noncrossing relations are initialized with keys 1-4
        self.assertEqual(set(topo_info.crossing_relations.keys()), {1, 2, 3, 4})
        self.assertEqual(set(topo_info.noncrossing_relations.keys()), {1, 2, 3, 4})
        # Check that all are empty sets initially
        for relation_type in [1, 2, 3, 4]:
            self.assertEqual(topo_info.crossing_relations[relation_type], set())
            self.assertEqual(topo_info.noncrossing_relations[relation_type], set())
    
    def test_add_crossing_relations(self):
        """Test adding crossing relations."""
        topo_info = TopologicalInfo()
        relations = {(0, 1), (1, 2), (2, 3)}
        
        topo_info.add_crossing_relations(relations, 2)  # Add as type 2 (1-2 relations)
        
        # Check that relations are stored
        self.assertEqual(topo_info.crossing_relations[2], relations)
        # Check that other types are still empty
        self.assertEqual(topo_info.crossing_relations[1], set())
        self.assertEqual(topo_info.crossing_relations[3], set())
        self.assertEqual(topo_info.crossing_relations[4], set())
    
    def test_add_noncrossing_relations(self):
        """Test adding non-crossing relations."""
        topo_info = TopologicalInfo()
        relations = {(0, 1), (1, 2)}
        
        topo_info.add_noncrossing_relations(relations, 3)  # Add as type 3 (1-3 relations)
        
        # Check that relations are stored
        self.assertEqual(topo_info.noncrossing_relations[3], relations)
        # Check that other types are still empty
        self.assertEqual(topo_info.noncrossing_relations[1], set())
        self.assertEqual(topo_info.noncrossing_relations[2], set())
        self.assertEqual(topo_info.noncrossing_relations[4], set())
    
    def test_add_relations_canonical_order(self):
        """Test that relations are stored in canonical (min, max) order."""
        topo_info = TopologicalInfo()
        relations = {(1, 0), (2, 1), (3, 2)}  # Not in canonical order
        
        topo_info.add_crossing_relations(relations, 2)
        
        # Check that all pairs are in canonical order
        for pair in topo_info.crossing_relations[2]:
            self.assertLess(pair[0], pair[1], f"Pair {pair} not in canonical order")
        # Check that canonical pairs are present
        self.assertIn((0, 1), topo_info.crossing_relations[2])
        self.assertIn((1, 2), topo_info.crossing_relations[2])
        self.assertIn((2, 3), topo_info.crossing_relations[2])
    
    def test_add_relations_tighter_relation_check(self):
        """Test that pairs already in tighter relations are not added to looser relations."""
        topo_info = TopologicalInfo()
        pair = (0, 1)
        
        # Add pair as type 2 
        topo_info.add_crossing_relations({pair}, 2)
        self.assertIn(pair, topo_info.crossing_relations[2])
        
        # Try to add same pair as type 3  - should not be added
        topo_info.add_crossing_relations({pair}, 3)
        self.assertNotIn(pair, topo_info.crossing_relations[3])
    
    def test_iter_crossing_relations(self):
        """Test iterating over crossing relations."""
        topo_info = TopologicalInfo()
        topo_info.add_crossing_relations({(0, 1)}, 2)
        topo_info.add_crossing_relations({(1, 2)}, 3)
        topo_info.add_crossing_relations({(2, 3)}, 4)
        
        # Iterate over types 2-4
        pairs = list(topo_info.iter_crossing_relations(2, 4))
        self.assertEqual(len(pairs), 3)
        self.assertIn((0, 1), pairs)
        self.assertIn((1, 2), pairs)
        self.assertIn((2, 3), pairs)
        
        # Iterate over types 3-4
        pairs = list(topo_info.iter_crossing_relations(3, 4))
        self.assertEqual(len(pairs), 2)
        self.assertIn((1, 2), pairs)
        self.assertIn((2, 3), pairs)
        self.assertNotIn((0, 1), pairs)
    
    def test_iter_noncrossing_relations(self):
        """Test iterating over non-crossing relations."""
        topo_info = TopologicalInfo()
        topo_info.add_noncrossing_relations({(0, 1)}, 2)
        topo_info.add_noncrossing_relations({(1, 2)}, 3)
        topo_info.add_noncrossing_relations({(2, 3)}, 4)
        
        # Iterate over types 2-4
        pairs = list(topo_info.iter_noncrossing_relations(2, 4))
        self.assertEqual(len(pairs), 3)
        self.assertIn((0, 1), pairs)
        self.assertIn((1, 2), pairs)
        self.assertIn((2, 3), pairs)
        
        # Iterate over types 2-3
        pairs = list(topo_info.iter_noncrossing_relations(2, 3))
        self.assertEqual(len(pairs), 2)
        self.assertIn((0, 1), pairs)
        self.assertIn((1, 2), pairs)
        self.assertNotIn((2, 3), pairs)
    
    def test_iter_relations_empty(self):
        """Test iterating over empty relations."""
        topo_info = TopologicalInfo()
        
        # Should return empty iterator
        pairs = list(topo_info.iter_crossing_relations(2, 4))
        self.assertEqual(len(pairs), 0)
        
        pairs = list(topo_info.iter_noncrossing_relations(1, 4))
        self.assertEqual(len(pairs), 0)


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
        # RCP pair: 0-6 (typical ring closure: does not consider the very last atoms!)
        self.neighbors = defaultdict(set)
        self.neighbors[0] = {1}
        self.neighbors[1] = {0, 2}
        self.neighbors[2] = {1, 3}
        self.neighbors[3] = {2, 4}
        self.neighbors[4] = {3, 5}
        self.neighbors[5] = {4, 6}
        self.neighbors[6] = {5, 7}
        self.neighbors[7] = {6}

        self.neighbors_with_branch = defaultdict(set)
        self.neighbors_with_branch[0] = {1, 14}
        self.neighbors_with_branch[1] = {0, 2, 8}
        self.neighbors_with_branch[2] = {1, 3, 9}
        self.neighbors_with_branch[3] = {2, 4, 10}
        self.neighbors_with_branch[4] = {3, 5, 11}
        self.neighbors_with_branch[5] = {4, 6, 12}
        self.neighbors_with_branch[6] = {5, 7, 13}
        self.neighbors_with_branch[7] = {6}
        self.neighbors_with_branch[8] = {1} 
        self.neighbors_with_branch[9] = {2}
        self.neighbors_with_branch[10] = {3}
        self.neighbors_with_branch[11] = {4}
        self.neighbors_with_branch[12] = {5}
        self.neighbors_with_branch[13] = {6}
        self.neighbors_with_branch[14] = {0}

    def test_get_atoms_at_distances(self):
        """Test getting atoms at different distances."""
        atoms_a, atoms_b, atoms_c, atoms_d = _get_atoms_at_distances(0, self.neighbors)
        
        self.assertEqual(atoms_a, {0})  # Distance 0: self
        self.assertEqual(atoms_b, {1})  # Distance 1: direct neighbor
        self.assertEqual(atoms_c, {2})  # Distance 2: neighbor of neighbor
        self.assertEqual(atoms_d, {3})  # Distance 3: neighbor of distance 2

    def test_get_atoms_in_1_2_relation(self):
        """Test 1-2 relation calculation"""
        noncrossing, crossing = _get_atoms_in_1_2_relation(0, 7, self.neighbors)
        relations_all = noncrossing | crossing
        
        # Check that expected pairs are present
        self.assertEqual(len(relations_all), 4)
        self.assertIn((0, 1), noncrossing) 
        self.assertIn((6, 7), noncrossing)
        self.assertIn((0, 6), crossing)
        self.assertIn((1, 7), crossing)

    def test_get_atoms_in_1_2_relation_with_nonterminal_atom(self):
        """Test 1-2 relation calculation with nonterminal atom."""
        noncrossing, crossing = _get_atoms_in_1_2_relation(0, 6, self.neighbors)
        relations_all = noncrossing | crossing
        
        # Check that expected pairs are present
        self.assertEqual(len(relations_all), 6)
        self.assertIn((0, 1), noncrossing) 
        self.assertIn((5, 6), noncrossing) 
        self.assertIn((6, 7), noncrossing) 
        self.assertIn((0, 5), crossing)
        self.assertIn((0, 7), crossing)
        self.assertIn((1, 6), crossing)

    def test_get_atoms_in_1_2_relation_with_branch(self):
        """Test 1-2 relation calculation with branch."""
        noncrossing, crossing = _get_atoms_in_1_2_relation(0, 6, self.neighbors_with_branch)
        relations_all = noncrossing | crossing
        
        # Non-crossing: p1 with its neighbors, p2 with its neighbors
        # p1=0 has neighbors {1, 14}, p2=6 has neighbors {5, 7, 13}
        self.assertIn((0, 1), noncrossing) 
        self.assertIn((0, 14), noncrossing) 
        self.assertIn((5, 6), noncrossing) 
        self.assertIn((6, 7), noncrossing) 
        self.assertIn((6, 13), noncrossing) 
        
        # Crossing: mixing atoms from p1 side with atoms from p2 side
        # atoms_a1={0} with atoms_b2={5,7,13}, atoms_a2={6} with atoms_b1={1,14}
        self.assertIn((0, 5), crossing)
        self.assertIn((0, 7), crossing)
        self.assertIn((0, 13), crossing)
        self.assertIn((1, 6), crossing)
        self.assertIn((6, 14), crossing)
        
        # Check total count
        self.assertEqual(len(noncrossing), 5)
        self.assertEqual(len(crossing), 5)
        self.assertEqual(len(relations_all), 10)
    
    def test_get_atoms_in_1_3_relation(self):
        """Test 1-3 relation calculation."""
        noncrossing, crossing = _get_atoms_in_1_3_relation(0, 7, self.neighbors)
        relations_all = noncrossing | crossing
        
        # Check that expected pairs are present
        self.assertEqual(len(relations_all), 5)
        self.assertIn((0, 2), noncrossing)  
        self.assertIn((5, 7), noncrossing)  
        self.assertIn((0, 5), crossing) 
        self.assertIn((1, 6), crossing)  
        self.assertIn((2, 7), crossing)
    
    def test_get_atoms_in_1_3_relation_with_nonterminal_atom(self):
        """Test 1-3 relation calculation with nonterminal atom."""
        noncrossing, crossing = _get_atoms_in_1_3_relation(0, 6, self.neighbors)
        relations_all = noncrossing | crossing
        
        # Check that expected pairs are present
        self.assertEqual(len(relations_all), 6)
        self.assertIn((0, 2), noncrossing)  
        self.assertIn((4, 6), noncrossing)  
        self.assertIn((0, 4), crossing) 
        self.assertIn((1, 5), crossing) 
        self.assertIn((1, 7), crossing)  
        self.assertIn((2, 6), crossing)
    
    def test_get_atoms_in_1_3_relation_with_branch(self):
        """Test 1-3 relation calculation with branch."""
        noncrossing, crossing = _get_atoms_in_1_3_relation(0, 6, self.neighbors_with_branch)
        relations_all = noncrossing | crossing
        
        # Non-crossing: p1 with atoms_c1 (distance 2), p2 with atoms_c2 (distance 2)
        # p1=0: atoms_c1 includes {2, 8} (neighbors of {1,14} excluding 0)
        # p2=6: atoms_c2 includes {4, 12} (neighbors of {5,7,13} excluding 6)
        self.assertIn((0, 2), noncrossing)  
        self.assertIn((0, 8), noncrossing)  
        self.assertIn((4, 6), noncrossing)  
        self.assertIn((6, 12), noncrossing)  
        
        # Crossing: mixing atoms from different sides
        # atoms_a1={0} with atoms_c2={4,12}
        self.assertIn((0, 4), crossing) 
        self.assertIn((0, 12), crossing)
        # atoms_b1={1,14} with atoms_b2={5,7,13}
        self.assertIn((1, 5), crossing) 
        self.assertIn((1, 7), crossing)  
        self.assertIn((1, 13), crossing)
        self.assertIn((5, 14), crossing)
        self.assertIn((7, 14), crossing)
        self.assertIn((13, 14), crossing)
        # atoms_a2={6} with atoms_c1={2,8}
        self.assertIn((2, 6), crossing)
        self.assertIn((6, 8), crossing)
        
        # Check counts (some pairs may be duplicates when combined)
        self.assertGreaterEqual(len(noncrossing), 4)
        self.assertGreaterEqual(len(crossing), 9) 
    
    def test_get_atoms_in_1_4_relation(self):
        """Test 1-4 relation calculation."""
        noncrossing, crossing = _get_atoms_in_1_4_relation(0, 7, self.neighbors)
        relations_all = noncrossing | crossing
        
        # Check that expected pairs are present
        self.assertEqual(len(relations_all), 6)
        self.assertIn((0, 3), noncrossing)  
        self.assertIn((4, 7), noncrossing) 
        self.assertIn((0, 4), crossing)
        self.assertIn((3, 7), crossing)
        self.assertIn((2, 6), crossing)
        self.assertIn((1, 5), crossing)
    
    def test_get_atoms_in_1_4_relation_with_nonterminal_atom(self):
        """Test 1-4 relation calculation with nonterminal atom."""
        noncrossing, crossing = _get_atoms_in_1_4_relation(0, 6, self.neighbors)
        relations_all = noncrossing | crossing
        
        # Check that expected pairs are present
        self.assertEqual(len(noncrossing), 2)
        self.assertEqual(len(crossing), 5)
        self.assertEqual(len(relations_all), 5) #because of duplicates
        self.assertIn((0, 3), noncrossing)  
        self.assertIn((3, 6), noncrossing) 
        self.assertIn((0, 3), crossing)
        self.assertIn((3, 6), crossing)
        self.assertIn((2, 5), crossing)
        self.assertIn((2, 7), crossing)
        self.assertIn((1, 4), crossing)
    
    def test_get_atoms_in_1_4_relation_with_branch(self):
        """Test 1-4 relation calculation with branch."""
        noncrossing, crossing = _get_atoms_in_1_4_relation(0, 6, self.neighbors_with_branch)
        relations_all = noncrossing | crossing
        
        # Non-crossing: p1 with atoms_d1 (distance 3), p2 with atoms_d2 (distance 3)
        # p1=0: atoms_d1 includes atoms at distance 3 from 0
        # p2=6: atoms_d2 includes atoms at distance 3 from 6
        self.assertIn((0, 3), noncrossing)  
        self.assertIn((0, 9), noncrossing)  
        self.assertIn((3, 6), noncrossing)  
        self.assertIn((6, 11), noncrossing)  
        
        # Crossing: mixing atoms from different sides
        # atoms_a1={0} with atoms_d2
        self.assertIn((0, 3), crossing)  # Note: (0,3) is both noncrossing and crossing
        self.assertIn((0, 11), crossing)
        # atoms_b1={1,14} with atoms_c2={4,12}
        self.assertIn((1, 4), crossing)
        self.assertIn((1, 12), crossing)
        self.assertIn((4, 14), crossing)
        self.assertIn((12, 14), crossing)
        # atoms_a2={6} with atoms_d1={3,9}
        self.assertIn((3, 6), crossing)  # Note: (3,6) is both noncrossing and crossing
        self.assertIn((6, 9), crossing)
        # atoms_b2={5,7,13} with atoms_c1={2,8}
        self.assertIn((2, 5), crossing)
        self.assertIn((2, 7), crossing)
        self.assertIn((2, 13), crossing)
        self.assertIn((5, 8), crossing)
        self.assertIn((7, 8), crossing)
        self.assertIn((8, 13), crossing)
        
        # Check counts (some pairs appear in both noncrossing and crossing)
        self.assertGreaterEqual(len(noncrossing), 4)
        self.assertGreaterEqual(len(crossing), 14)
    
    def test_relations_canonical_order(self):
        """Test that all relations return pairs in canonical (min, max) order."""
        noncrossing12, crossing12 = _get_atoms_in_1_2_relation(0, 4, self.neighbors)
        noncrossing13, crossing13 = _get_atoms_in_1_3_relation(0, 4, self.neighbors)
        noncrossing14, crossing14 = _get_atoms_in_1_4_relation(0, 4, self.neighbors)
        
        all_relations = noncrossing12 | crossing12 | noncrossing13 | crossing13 | noncrossing14 | crossing14
        for pair in all_relations:
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
        # Check that all relation types are empty
        for relation_type in [1, 2, 3, 4]:
            self.assertEqual(len(topo_info.crossing_relations[relation_type]), 0)
            self.assertEqual(len(topo_info.noncrossing_relations[relation_type]), 0)
    
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
        
        # Should have some relations (check using iterators)
        crossing_pairs = list(topo_info.iter_crossing_relations(2, 4))
        noncrossing_pairs = list(topo_info.iter_noncrossing_relations(2, 4))
        
        self.assertGreater(len(crossing_pairs), 0, "Should have crossing relations")
        self.assertGreater(len(noncrossing_pairs), 0, "Should have non-crossing relations")
        
        # Check that relations are stored in the correct type dictionaries
        # Type 2 = 1-2 relations, Type 3 = 1-3 relations, Type 4 = 1-4 relations
        total_crossing = (len(topo_info.crossing_relations[2]) + 
                         len(topo_info.crossing_relations[3]) + 
                         len(topo_info.crossing_relations[4]))
        total_noncrossing = (len(topo_info.noncrossing_relations[2]) + 
                            len(topo_info.noncrossing_relations[3]) + 
                            len(topo_info.noncrossing_relations[4]))
        
        self.assertGreater(total_crossing, 0)
        self.assertGreater(total_noncrossing, 0)


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

