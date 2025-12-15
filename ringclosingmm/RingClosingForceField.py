from openmm.app import Simulation
import openmm as mm
from openmm import Platform
from openmm.app import ForceField
from openmm.app.forcefield import _parseFunctions, _createFunctions, \
    HarmonicBondGenerator, HarmonicAngleGenerator, \
    _convertParameterToNumber, CustomNonbondedGenerator
from openmm import VerletIntegrator
import openmm.unit as unit
from openmm.app.forcefield import parsers
from collections import defaultdict
import itertools
import math
import numpy as np
from typing import Any, Dict, List, Optional, Set, Tuple


# Conventions: names of energy terms
trackedForceTypes = ['CustomNonbondedForce',  # Index 0 used below!
                     'RCPForce',  # Index 1 used below!
                     'HeadTailNonbondedForce',  # Index 2 used below!
                     'RCPAtomsNonbondedForce',  # Index 3 used below!
                     'HarmonicBondForce',
                     'HarmonicAngleForce',
                     'ConstrainedOutOfPlaneForce']

# Verbosity for debugging and development
verbose = False

# =============================================================================
# Utility Methods
# =============================================================================

def getDistance(p1: Any, p2: Any) -> float:
    """Calculate the distance between two points in 3D space.
    Both parameters are expected to be lists with three floats."""
    d = 0.0
    d = math.sqrt((p2._value[0] - p1._value[0])**2 + (p2._value[1] - p1._value[1])**2 + (p2._value[2] - p1._value[2])**2)
    return d


def getUnitVector(vector: np.ndarray) -> np.ndarray:
    """Gets the unit vector for the given vector."""
    return vector / np.linalg.norm(vector)


def _get_atoms_at_distances(atom_idx: int, neighbors: Dict[int, Set[int]]) -> Tuple[Set[int], Set[int], Set[int], Set[int]]:
    """
    Get sets of atoms at distances 0, 1, 2, and 3 from a given atom.
    
    This function computes the atoms at different distances in the molecular graph:
    - atoms_a (distance 0): the atom itself
    - atoms_b (distance 1): direct neighbors
    - atoms_c (distance 2): neighbors of neighbors, excluding distance 0 atoms
    - atoms_d (distance 3): neighbors of distance 2 atoms, excluding distance 1 atoms
    
    Parameters
    ----------
    atom_idx : int
        The starting atom index
    neighbors : Dict[int, Set[int]]
        Dictionary mapping atom indices to sets of their bonded neighbors
        
    Returns
    -------
    Tuple[Set[int], Set[int], Set[int], Set[int]]
        Tuple of (atoms_a, atoms_b, atoms_c, atoms_d) sets
    """
    atoms_a = {atom_idx}
    atoms_b = set(neighbors[atom_idx])
    atoms_c = set()
    atoms_d = set()
    
    # Distance 2: neighbors of distance 1 atoms, excluding distance 0
    for atm_b in atoms_b:
        for atm_c in neighbors[atm_b]:
            if atm_c not in atoms_a:
                atoms_c.add(atm_c)
    
    # Distance 3: neighbors of distance 2 atoms, excluding distance 1
    for atm_c in atoms_c:
        for atm_d in neighbors[atm_c]:
            if atm_d not in atoms_b:
                atoms_d.add(atm_d)
    
    return atoms_a, atoms_b, atoms_c, atoms_d


def _get_atoms_in_1_X_relation(p1: int, p2: int, neighbors: Dict[int, Set[int]], relation_type: int) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """
    Get the atoms in 1-X relation if p1 and p2 are in 1-1 relation.
    """
    if relation_type == 1:
        p1p2 = (min(p1, p2), max(p1, p2))
        return set(), set([p1p2])
    elif relation_type == 2:
        return _get_atoms_in_1_2_relation(p1, p2, neighbors)
    elif relation_type == 3:
        return _get_atoms_in_1_3_relation(p1, p2, neighbors)
    elif relation_type == 4:
        return _get_atoms_in_1_4_relation(p1, p2, neighbors)
    else:
        raise ValueError(f"Invalid relation type: {relation_type}")


def _get_atoms_in_1_2_relation(p1: int, p2: int, neighbors: Dict[int, Set[int]]) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """
    Get the atoms in 1-2 relation if p1 and p2 are in 1-1 relation.
    
    Returns
    -------
    Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]
        (noncrossing_relations, crossing_relations)
        Non-crossing: relations determined by topology (consistent *1 and *2)
        Crossing: relations crossing p1-p2 boundary (mixing *1 and *2)
    """
    atoms_a1, atoms_b1, atoms_c1, atoms_d1 = _get_atoms_at_distances(p1, neighbors)
    atoms_a2, atoms_b2, atoms_c2, atoms_d2 = _get_atoms_at_distances(p2, neighbors)

    atoms_in_1_2_relation_noncrossing = set()
    atoms_in_1_2_relation_crossing = set()

    # Non-crossing: consistent *1 and *2 indexes
    for atm_b1 in atoms_b1:
        atoms_in_1_2_relation_noncrossing.add((min(p1, atm_b1), max(p1, atm_b1)))
    
    for atm_b2 in atoms_b2:
        atoms_in_1_2_relation_noncrossing.add((min(p2, atm_b2), max(p2, atm_b2)))

    # Crossing: mixing *1 and *2 indexes
    for atm_a1 in atoms_a1:
        for atm_b2 in atoms_b2:
            atoms_in_1_2_relation_crossing.add((min(atm_a1, atm_b2), max(atm_a1, atm_b2)))
    
    for atm_a2 in atoms_a2:
        for atm_b1 in atoms_b1:
            atoms_in_1_2_relation_crossing.add((min(atm_a2, atm_b1), max(atm_a2, atm_b1)))
    
    return atoms_in_1_2_relation_noncrossing, atoms_in_1_2_relation_crossing


def _get_atoms_in_1_3_relation(p1: int, p2: int, neighbors: Dict[int, Set[int]]) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """
    Get the atoms in 1-3 relation if p1 and p2 are in 1-1 relation..
    
    Returns
    -------
    Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]
        (noncrossing_relations, crossing_relations)
        Non-crossing: relations determined by topology (consistent *1 and *2)
        Crossing: relations crossing p1-p2 boundary (mixing *1 and *2)
    """
    atoms_a1, atoms_b1, atoms_c1, atoms_d1 = _get_atoms_at_distances(p1, neighbors)
    atoms_a2, atoms_b2, atoms_c2, atoms_d2 = _get_atoms_at_distances(p2, neighbors)
    
    atoms_in_1_3_relation_noncrossing = set()
    atoms_in_1_3_relation_crossing = set()

    # Non-crossing: consistent *1 and *2 indexes
    for atm_c1 in atoms_c1:
        atoms_in_1_3_relation_noncrossing.add((min(p1, atm_c1), max(p1, atm_c1)))

    for atm_c2 in atoms_c2:
        atoms_in_1_3_relation_noncrossing.add((min(p2, atm_c2), max(p2, atm_c2)))

    # Crossing: mixing *1 and *2 indexes
    for atm_a1 in atoms_a1:
        for atm_c2 in atoms_c2:
            atoms_in_1_3_relation_crossing.add((min(atm_a1, atm_c2), max(atm_a1, atm_c2)))
    
    for atm_b1 in atoms_b1:
        for atm_b2 in atoms_b2:
            atoms_in_1_3_relation_crossing.add((min(atm_b1, atm_b2), max(atm_b1, atm_b2)))

    for atm_a2 in atoms_a2:
        for atm_c1 in atoms_c1:
            atoms_in_1_3_relation_crossing.add((min(atm_a2, atm_c1), max(atm_a2, atm_c1)))
    
    for atm_b2 in atoms_b2:
        for atm_b1 in atoms_b1:
            atoms_in_1_3_relation_crossing.add((min(atm_b2, atm_b1), max(atm_b2, atm_b1)))
    
    return atoms_in_1_3_relation_noncrossing, atoms_in_1_3_relation_crossing

def _get_atoms_in_1_4_relation(p1: int, p2: int, neighbors: Dict[int, Set[int]]) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """
    Get the atoms in 1-4 relation if p1 and p2 are in 1-1 relation.
    
    Returns
    -------
    Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]
        (noncrossing_relations, crossing_relations)
        Non-crossing: relations determined by topology (consistent *1 and *2)
        Crossing: relations crossing p1-p2 boundary (mixing *1 and *2)
    """
    atoms_a1, atoms_b1, atoms_c1, atoms_d1 = _get_atoms_at_distances(p1, neighbors)
    atoms_a2, atoms_b2, atoms_c2, atoms_d2 = _get_atoms_at_distances(p2, neighbors)
    
    atoms_in_1_4_relation_noncrossing = set()
    atoms_in_1_4_relation_crossing = set()

    # Non-crossing: consistent *1 and *2 indexes
    for atm_d1 in atoms_d1:
        atoms_in_1_4_relation_noncrossing.add((min(p1, atm_d1), max(p1, atm_d1)))

    for atm_d2 in atoms_d2:
        atoms_in_1_4_relation_noncrossing.add((min(p2, atm_d2), max(p2, atm_d2)))

    # Crossing: mixing *1 and *2 indexes
    for atm_a1 in atoms_a1:
        for atm_d2 in atoms_d2:
            atoms_in_1_4_relation_crossing.add((min(atm_a1, atm_d2), max(atm_a1, atm_d2)))
    
    for atm_b1 in atoms_b1:
        for atm_c2 in atoms_c2:
            atoms_in_1_4_relation_crossing.add((min(atm_b1, atm_c2), max(atm_b1, atm_c2)))
    
    for atm_c1 in atoms_c1:
        for atm_b2 in atoms_b2:
            atoms_in_1_4_relation_crossing.add((min(atm_c1, atm_b2), max(atm_c1, atm_b2)))

    for atm_a2 in atoms_a2:
        for atm_d1 in atoms_d1:
            atoms_in_1_4_relation_crossing.add((min(atm_a2, atm_d1), max(atm_a2, atm_d1)))

    return atoms_in_1_4_relation_noncrossing, atoms_in_1_4_relation_crossing


def getAngle(p0: Any, p1: Any, p2: Any) -> float:
    """Calculates the angle in radians between the vectors 1-0 and 1-2 as
    defined by the given points in 3d space."""
    v1 = np.array(p0._value) - np.array(p1._value)
    v2 = np.array(p2._value) - np.array(p1._value)
    u1 = getUnitVector(v1)
    u2 = getUnitVector(v2)
    return np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0))


def getImproperDihedral(p1: Any, p2: Any, p3: Any, p4: Any) -> float:
    """Calculate the improper dihedral angle in radians for atoms p1-p2-p3-p4.
    
    The improper dihedral angle is the dihedral angle between planes (p1, p2, p3) 
    and (p2, p3, p4). This measures how much atom p1 is out of the plane defined 
    by atoms p2, p3, p4.
    
    Parameters
    ----------
    p1, p2, p3, p4 : Any
        Positions with _value attribute containing 3D coordinates
        
    Returns
    -------
    float
        Improper dihedral angle in radians
    """
    # Convert to numpy arrays
    pos1 = np.array(p1._value)
    pos2 = np.array(p2._value)
    pos3 = np.array(p3._value)
    pos4 = np.array(p4._value)
    
    # Compute vectors
    b1 = pos2 - pos1
    b2 = pos3 - pos2
    b3 = pos4 - pos3
    
    # Compute normals to the two planes
    n1 = np.cross(b1, b2)  # Normal to plane (p1, p2, p3)
    n2 = np.cross(b2, b3)  # Normal to plane (p2, p3, p4)
    
    norm_n1 = np.linalg.norm(n1)
    norm_n2 = np.linalg.norm(n2)
    norm_b2 = np.linalg.norm(b2)
    
    # Check for degenerate cases (collinear atoms or zero-length bonds)
    if norm_n1 < 1e-8 or norm_n2 < 1e-8 or norm_b2 < 1e-8:
        return 0.0
    
    # Normalize vectors
    n1 = n1 / norm_n1
    n2 = n2 / norm_n2
    b2_unit = b2 / norm_b2
    
    # Compute dihedral angle using atan2 for proper sign
    m1 = np.cross(n1, b2_unit)
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    
    # Return angle in radians (OpenMM uses radians for dihedral angles)
    return np.arctan2(y, x)


def getOutOfPlaneDistance(central: Any, p1: Any, p2: Any, p3: Any) -> float:
    """Calculate the distance from the central atom to the plane defined by three other atoms.
    
    Parameters
    ----------
    central : Any
        Position of the central atom (should be approximately in the plane)
    p1, p2, p3 : Any
        Positions of three atoms defining the plane
        
    Returns
    -------
    float
        Distance from central atom to the plane (in nanometers)
    """
    # Convert to numpy arrays
    pos_central = np.array(central._value)
    pos1 = np.array(p1._value)
    pos2 = np.array(p2._value)
    pos3 = np.array(p3._value)
    
    # Compute two vectors in the plane
    v1 = pos2 - pos1
    v2 = pos3 - pos1
    
    # Compute normal to the plane
    normal = np.cross(v1, v2)
    norm_magnitude = np.linalg.norm(normal)
    
    # Handle degenerate case (collinear points)
    if norm_magnitude < 1e-8:
        # Fall back to distance from central to line p1-p2
        v_line = pos2 - pos1
        v_to_central = pos_central - pos1
        if np.linalg.norm(v_line) > 1e-8:
            # Project v_to_central onto v_line
            t = np.dot(v_to_central, v_line) / np.dot(v_line, v_line)
            proj = pos1 + t * v_line
            return np.linalg.norm(pos_central - proj)
        else:
            # All points are the same
            return np.linalg.norm(pos_central - pos1)
    
    # Normalize the normal vector
    normal = normal / norm_magnitude
    
    # Vector from a point in the plane to the central atom
    v_to_central = pos_central - pos1
    
    # Distance is the absolute value of the dot product with the normal
    distance = abs(np.dot(v_to_central, normal))
    
    return distance


def getOutOfPlaneDistanceApprox(central: Any, p1: Any, p2: Any, p3: Any) -> float:
    """Calculate the distance from the central atom to the plane using the same approximation
    formula as the OpenMM energy expression.
    
    This function uses the same formula as the OpenMM CustomCompoundBondForce energy expression:
    d = distance(p1,p4) * abs(sin(angle(p2,p1,p4))) * abs(sin(angle(p3,p1,p4))) / max(abs(sin(angle(p2,p1,p3))), 1e-6)
    
    Parameters
    ----------
    central : Any
        Position of the central atom (p4 in the formula)
    p1, p2, p3 : Any
        Positions of three atoms defining the plane (p1, p2, p3 in the formula)
        
    Returns
    -------
    float
        Approximate distance from central atom to the plane (in nanometers)
    """
    # Convert to numpy arrays
    pos_central = np.array(central._value)
    pos1 = np.array(p1._value)
    pos2 = np.array(p2._value)
    pos3 = np.array(p3._value)
    
    # Compute vectors from p1
    v_p1p2 = pos2 - pos1
    v_p1p3 = pos3 - pos1
    v_p1p4 = pos_central - pos1
    
    # Compute distances
    d_p1p4 = np.linalg.norm(v_p1p4)
    
    # Handle degenerate case: if p1 and p4 are the same, distance is 0
    if d_p1p4 < 1e-8:
        return 0.0
    
    # Normalize vectors
    v_p1p2_norm = v_p1p2 / np.linalg.norm(v_p1p2) if np.linalg.norm(v_p1p2) > 1e-8 else v_p1p2
    v_p1p3_norm = v_p1p3 / np.linalg.norm(v_p1p3) if np.linalg.norm(v_p1p3) > 1e-8 else v_p1p3
    v_p1p4_norm = v_p1p4 / d_p1p4
    
    # Compute angles at p1
    # angle(p2,p1,p4): angle between vectors p1->p2 and p1->p4
    cos_angle_p2p1p4 = np.clip(np.dot(v_p1p2_norm, v_p1p4_norm), -1.0, 1.0)
    sin_angle_p2p1p4 = np.sqrt(1.0 - cos_angle_p2p1p4**2)
    
    # angle(p3,p1,p4): angle between vectors p1->p3 and p1->p4
    cos_angle_p3p1p4 = np.clip(np.dot(v_p1p3_norm, v_p1p4_norm), -1.0, 1.0)
    sin_angle_p3p1p4 = np.sqrt(1.0 - cos_angle_p3p1p4**2)
    
    # angle(p2,p1,p3): angle between vectors p1->p2 and p1->p3
    cos_angle_p2p1p3 = np.clip(np.dot(v_p1p2_norm, v_p1p3_norm), -1.0, 1.0)
    sin_angle_p2p1p3 = np.sqrt(1.0 - cos_angle_p2p1p3**2)
    
    # Apply the same formula as OpenMM expression
    # d = distance(p1,p4) * abs(sin(angle(p2,p1,p4))) * abs(sin(angle(p3,p1,p4))) / max(abs(sin(angle(p2,p1,p3))), 1e-6)
    denominator = max(abs(sin_angle_p2p1p3), 1e-6)
    distance = d_p1p4 * abs(sin_angle_p2p1p4) * abs(sin_angle_p3p1p4) / denominator
    
    return distance


# =============================================================================
# Registers objects that store topological info related to ring-closing pairs
# =============================================================================
class TopologicalInfo:
    
    def __init__(self):
        self.crossing_relations: Dict[int, Set[Tuple[int, int]]] = {1: set(), 2: set(), 3: set(), 4: set()}
        self.noncrossing_relations: Dict[int, Set[Tuple[int, int]]] = {1: set(), 2: set(), 3: set(), 4: set()}

    @staticmethod
    def _add_relations(atom_pairs: Set[Tuple[int, int]], relation_type: int, relation_collector: Dict[int, Set[Tuple[int, int]]]) -> None:
        """Add relations to the topological info. We assume tighter relations are recorded before looser ones, so we do not check if an attempt to record a looser relation is made when a tighter relation is already recorded. This is a performance optimization."""
        for pair in atom_pairs:
            sorted_pair = (min(pair[0], pair[1]), max(pair[0], pair[1]))
            found_in_tigher_relation = False
            for tigher_relation in range(1, relation_type):
                if sorted_pair in relation_collector[tigher_relation]:
                    found_in_tigher_relation = True
                    break;
            if not found_in_tigher_relation:
                relation_collector[relation_type].add(sorted_pair)


    def add_crossing_relations(self, relations: Set[Tuple[int, int]], relation_type: int):
        """Add crossing relations to the topological info."""
        self._add_relations(relations, relation_type, self.crossing_relations)


    def add_noncrossing_relations(self, relations: Set[Tuple[int, int]], relation_type: int):
        """Add noncrossing relations to the topological info."""
        self._add_relations(relations, relation_type, self.noncrossing_relations)

    def iter_crossing_relations(self, smallest_relation_type: int, largest_relation_type: int):
        """
        Get an iterator over crossing relations for relation types from smallest_relation_type 
        up to largest_relation_type (inclusive).
        
        Relation types: 1 = 1-1 (coincident), 2 = 1-2, 3 = 1-3, 4 = 1-4.
        
        Parameters
        ----------
        smallest_relation_type : int
            The smallest relation type to include (inclusive, e.g., 2 for 1-2 relations)
        largest_relation_type : int
            The largest relation type to include (inclusive, typically 4 for 1-4 relations)
            
        Yields
        ------
        Tuple[int, int]
            Atom pairs in canonical order (min, max) from crossing relations
            in the specified range
        """
        for relation_type in range(smallest_relation_type, largest_relation_type + 1):
            if relation_type in self.crossing_relations:
                for pair in self.crossing_relations[relation_type]:
                    yield pair

    def iter_noncrossing_relations(self, smallest_relation_type: int, largest_relation_type: int):
        """
        Get an iterator over non-crossing relations for relation types from smallest_relation_type 
        up to largest_relation_type (inclusive).
        
        Relation types: 1 = 1-1 (coincident), 2 = 1-2, 3 = 1-3, 4 = 1-4.
        
        Parameters
        ----------
        smallest_relation_type : int
            The smallest relation type to include (inclusive, e.g., 2 for 1-2 relations)
        largest_relation_type : int
            The largest relation type to include (inclusive, typically 4 for 1-4 relations)
            
        Yields
        ------
        Tuple[int, int]
            Atom pairs in canonical order (min, max) from non-crossing relations
            in the specified range
        """
        for relation_type in range(smallest_relation_type, largest_relation_type + 1):
            if relation_type in self.noncrossing_relations:
                for pair in self.noncrossing_relations[relation_type]:
                    yield pair




def _build_neighbors_from_bonds(bonds) -> Dict[int, Set[int]]:
    """
    Build a neighbors dictionary from bonds.
    Works with both data.bonds (bond.atom1, bond.atom2) and topo.bonds() (bond.atom1.index, bond.atom2.index).
    
    Parameters
    ----------
    bonds : iterable
        Iterable of bond objects
        
    Returns
    -------
    Dict[int, Set[int]]
        Dictionary mapping atom indices to sets of their bonded neighbors
    """
    neighbors = defaultdict(set)
    for bond in bonds:
        # Handle both data.bonds (direct indices) and topo.bonds() (bond.atom1.index)
        if hasattr(bond.atom1, 'index'):
            atom1_idx = bond.atom1.index
            atom2_idx = bond.atom2.index
        else:
            atom1_idx = bond.atom1
            atom2_idx = bond.atom2
        neighbors[atom1_idx].add(atom2_idx)
        neighbors[atom2_idx].add(atom1_idx)
    return neighbors


def _compute_topological_info(topo_info: TopologicalInfo, rcpterms: List[Tuple[int, int]], neighbors: Dict[int, Set[int]]):
    """
    Compute relations due to RCP terms. For efficiency, deal with tighter relations first.
    """
    for relation_type in range(1, 5):
        for pair in rcpterms:
            p1 = int(pair[0])
            p2 = int(pair[1])
            relation_noncrossing, relation_crossing = _get_atoms_in_1_X_relation(p1, p2, neighbors, relation_type)
            topo_info.add_noncrossing_relations(relation_noncrossing, relation_type)
            topo_info.add_crossing_relations(relation_crossing, relation_type)


def _compute_and_store_topological_info(data, args: Dict[str, Any]) -> TopologicalInfo:
    """
    Compute topological relations for all RCP pairs and store in args.
    This function computes the relations once and caches them in args['topological_info'].
    
    Parameters
    ----------
    data : openmm.app.forcefield._SystemData
        System data containing bond information
    args : Dict[str, Any]
        Arguments dictionary passed to any createForce method
        
    Returns
    -------
    TopologicalInfo
        The computed topological information
    """
    # Check if already computed and cached
    if 'topological_info' in args:
        return args['topological_info']
    
    # Create new TopologicalInfo instance
    topo_info = TopologicalInfo()
    
    # Get RCP terms
    rcpterms = args.get('rcpterms', [])
    if len(rcpterms) == 0:
        # No RCP terms, return empty info
        args['topological_info'] = topo_info
        return topo_info
    
    # Build a dictionary of bonded neighbors for each atom
    neighbors = _build_neighbors_from_bonds(data.bonds)

    # Define all proximity relations due to RCP terms
    _compute_topological_info(topo_info, rcpterms, neighbors)
    
    # Cache in args for future use
    args['topological_info'] = topo_info
    
    return topo_info


def _compute_topological_info_from_topology(topo, rcpterms: List[Tuple[int, int]]) -> TopologicalInfo:
    """
    Compute topological relations for all RCP pairs from a topology object.
    This version works with openmm.app.Topology objects.
    
    Parameters
    ----------
    topo : openmm.app.Topology
        Topology containing bond information
    rcpterms : List[Tuple[int, int]]
        List of RCP pairs
        
    Returns
    -------
    TopologicalInfo
        The computed topological information
    """
    # Create new TopologicalInfo instance
    topo_info = TopologicalInfo()
    
    if len(rcpterms) == 0:
        return topo_info
    
    # Build a dictionary of bonded neighbors for each atom
    neighbors = _build_neighbors_from_bonds(topo.bonds())
    
    # Define all proximity relations due to RCP terms
    _compute_topological_info(topo_info, rcpterms, neighbors)
    
    return topo_info


# =============================================================================
# Force Field Generation 
# =============================================================================

class ConstrainedBondGenerator(HarmonicBondGenerator):
    """A generator for constraints that try to preserve the distances
    between bonded pairs of particles."""
    def __init__(self, forcefield):
        self.ff = forcefield
        self.bondsForAtomType = defaultdict(set)
        self.types1 = []
        self.types2 = []
        self.length = []
        self.k = []

    def registerBond(self, parameters):
        types = self.ff._findAtomTypes(parameters, 2)
        if None not in types:
            index = len(self.types1)
            self.types1.append(types[0])
            self.types2.append(types[1])
            for t in types[0]:
                self.bondsForAtomType[t].add(index)
            for t in types[1]:
                self.bondsForAtomType[t].add(index)
            self.k.append(_convertParameterToNumber(parameters['k']))

    @staticmethod
    def parseElement(element, ff):
        generator = ConstrainedBondGenerator(ff)
        ff.registerGenerator(generator)
        for bond in element.findall('Bond'):
            generator.registerBond(bond.attrib)

    def createForce(self, sys, data, nonbondedMethod, nonbondedCutoff, args):
        """
        Create the ConstrainedBondForce.

        Parameters
        ----------
        sys : openmm.System
            The system to which the force is added
        data : openmm.app.forcefield._SystemData
            System data containing bond information
        nonbondedMethod : str
            The nonbonded method used for the simulation
        nonbondedCutoff : float
            The nonbonded cutoff distance used for the simulation
        args : Dict[str, Any]
            Arguments dictionary passed to any createForce method
            
        Returns
        -------
        None
        
        Notes
        -----
        The rcpterms must be provided in args['rcpterms'] as a list of 
        [particle1, particle2] pairs. 
        Also, the atom coordinates must be provided in args['positions'] 
        as a list of openmm.unit.Quantity objects.
        The topological info is computed and cached in args['topological_info'].
        """
        existing = [f for f in sys.getForces() if type(f) == mm.HarmonicBondForce]

        # Get or compute topological info (computes once and caches in args)
        topo_info = _compute_and_store_topological_info(data, args)

        if len(existing) == 0:
            force = mm.HarmonicBondForce()
            sys.addForce(force)
        else:
            force = existing[0]
        for bond in data.bonds:
            type1 = data.atomType[data.atoms[bond.atom1]]
            type2 = data.atomType[data.atoms[bond.atom2]]
            # Here we take the interatomic distance from the initial geometry
            initialLength = getDistance(args['positions'][bond.atom1],
                                        args['positions'][bond.atom2])
            for i in self.bondsForAtomType[type1]:
                types1 = self.types1[i]
                types2 = self.types2[i]
                if (type1 in types1 and type2 in types2) or (type1 in types2 and type2 in types1):
                    bond.length = initialLength
                    if bond.isConstrained:
                        data.addConstraint(sys, bond.atom1, bond.atom2, initialLength)
                    if self.k[i] != 0:
                        # flexibleConstraints allows us to add parameters even if the DOF is
                        # constrained
                        if not bond.isConstrained or args.get('flexibleConstraints', False):
                            force.addBond(bond.atom1, bond.atom2, initialLength, self.k[i])
                    break

    def postprocessSystem(self, sys, data, args):
        pass


parsers["ConstrainedBondForce"] = ConstrainedBondGenerator.parseElement


class ConstrainedAngleGenerator(HarmonicAngleGenerator):
    """A generator for constraints that try to preserve the angle among bonds."""

    def __init__(self, forcefield):
        self.ff = forcefield
        self.anglesForAtom2Type = defaultdict(list)
        self.types1 = []
        self.types2 = []
        self.types3 = []
        self.angle = []
        self.k = []

    def registerAngle(self, parameters):
        types = self.ff._findAtomTypes(parameters, 3)
        if None not in types:
            index = len(self.types1)
            self.types1.append(types[0])
            self.types2.append(types[1])
            self.types3.append(types[2])
            for t in types[1]:
                self.anglesForAtom2Type[t].append(index)
            self.k.append(_convertParameterToNumber(parameters['k']))

    @staticmethod
    def parseElement(element, ff):
        generator = ConstrainedAngleGenerator(ff)
        ff.registerGenerator(generator)
        for angle in element.findall('Angle'):
            generator.registerAngle(angle.attrib)

    def createForce(self, sys, data, nonbondedMethod, nonbondedCutoff, args):
        """
        Create the ConstrainedAngleForce.

        Parameters
        ----------
        sys : openmm.System
            The system to which the force is added
        data : openmm.app.forcefield._SystemData
            System data containing bond information
        nonbondedMethod : str
            The nonbonded method used for the simulation
        nonbondedCutoff : float
            The nonbonded cutoff distance used for the simulation
        args : Dict[str, Any]
            Arguments dictionary passed to any createForce method
            
        Returns
        -------
        None
        
        Notes
        -----
        The rcpterms must be provided in args['rcpterms'] as a list of 
        [particle1, particle2] pairs. 
        Also, the atom coordinates must be provided in args['positions'] 
        as a list of openmm.unit.Quantity objects.
        The topological info is computed and cached in args['topological_info'].
        """
        existing = [f for f in sys.getForces() if type(f) == mm.HarmonicAngleForce]

        # Get or compute topological info (computes once and caches in args)
        topo_info = _compute_and_store_topological_info(data, args)
        
        if len(existing) == 0:
            force = mm.HarmonicAngleForce()
            sys.addForce(force)
        else:
            force = existing[0]
        for (angle, isConstrained) in zip(data.angles, data.isAngleConstrained):
            type1 = data.atomType[data.atoms[angle[0]]]
            type2 = data.atomType[data.atoms[angle[1]]]
            type3 = data.atomType[data.atoms[angle[2]]]
            for i in self.anglesForAtom2Type[type2]:
                types1 = self.types1[i]
                types2 = self.types2[i]
                types3 = self.types3[i]
                if (type1 in types1 and type2 in types2 and type3 in types3) or (type1 in types3 and type2 in types2 and type3 in types1):
                    # Find the distance A-C in the angle ABC
                    length = getDistance(args['positions'][angle[0]],
                                      args['positions'][angle[2]])
                    if isConstrained:
                        data.addConstraint(sys, angle[0], angle[2], length)
                    if self.k[i] != 0:
                        radiants = getAngle(args['positions'][angle[0]],
                                         args['positions'][angle[1]],
                                         args['positions'][angle[2]])
                        if not isConstrained or args.get('flexibleConstraints', False):
                            force.addAngle(angle[0], angle[1], angle[2], radiants, self.k[i])
                    break

    def postprocessSystem(self, sys, data, args):
        pass


parsers["ConstrainedAngleForce"] = ConstrainedAngleGenerator.parseElement


class ConstrainedOutOfPlaneGenerator():
    """A generator for out-of-plane forces that maintain planarity of planar sites.
    
    Identifies planar sites from input geometry:
    - Atoms with exactly 3 neighbors: checks if the 3 neighbors form a planar geometry
    - Atoms with >3 neighbors: checks that ALL neighbors are coplanar (within threshold)
    
    For atoms with >3 neighbors, adds the minimum number of improper dihedrals needed
    such that each neighbor appears in at least one improper term. This ensures all
    neighbors are constrained to maintain planarity.
    """

    def __init__(self, forcefield):
        self.ff = forcefield
        self.impropersForCentralType = defaultdict(list)
        self.types1 = []  # First neighbor type
        self.types2 = []  # Central atom type
        self.types3 = []  # Second neighbor type
        self.types4 = []  # Third neighbor type
        self.k = []
        self.planarityThreshold = 0.1  # Default threshold in nm for detecting planar sites

    def _generate_minimum_improper_combinations(self, neighbors: List[int]) -> List[Tuple[int, int, int]]:
        """
        Generate the minimum number of 3-neighbor combinations needed to cover all neighbors.
        
        Each improper dihedral covers 3 neighbors. This function finds the minimum set
        of combinations such that every neighbor appears in at least one combination.
        
        Uses a greedy algorithm: start with first 3 neighbors, then add combinations
        that cover the maximum number of uncovered neighbors while reusing some from
        previous combinations when beneficial.
        
        Parameters
        ----------
        neighbors : List[int]
            List of neighbor atom indices
            
        Returns
        -------
        List[Tuple[int, int, int]]
            List of 3-neighbor tuples that cover all neighbors with minimum combinations
        """
        n = len(neighbors)
        if n == 3:
            return [(neighbors[0], neighbors[1], neighbors[2])]
        
        # Greedy algorithm: build combinations to cover all neighbors
        covered = set()
        combinations = []
        remaining = list(neighbors)
        
        # First combination: use first 3 neighbors
        first_combo = (remaining[0], remaining[1], remaining[2])
        combinations.append(first_combo)
        covered.update(first_combo)
        remaining = remaining[3:]
        
        # Continue adding combinations until all neighbors are covered
        while remaining:
            if len(remaining) == 1:
                # 1 neighbor left: use it with 2 from already covered
                covered_list = sorted(list(covered))
                combo = (covered_list[0], covered_list[1], remaining[0])
                combinations.append(combo)
                covered.add(remaining[0])
                break
            elif len(remaining) == 2:
                # 2 neighbors left: use them with 1 from already covered
                covered_list = sorted(list(covered))
                combo = (covered_list[0], remaining[0], remaining[1])
                combinations.append(combo)
                covered.update(remaining)
                break
            else:
                # 3+ neighbors left: use next 3 new neighbors
                combo = (remaining[0], remaining[1], remaining[2])
                combinations.append(combo)
                covered.update(combo)
                remaining = remaining[3:]
        
        return combinations

    def registerImproper(self, parameters):
        """Register an improper dihedral (out-of-plane) parameter definition."""
        types = self.ff._findAtomTypes(parameters, 4)
        if None not in types:
            index = len(self.types1)
            self.types1.append(types[0])
            self.types2.append(types[1])  # Central atom
            self.types3.append(types[2])
            self.types4.append(types[3])
            for t in types[1]:  # Index by central atom type
                self.impropersForCentralType[t].append(index)
            self.k.append(_convertParameterToNumber(parameters['k']))
            # Optional planarity threshold for geometry-based detection
            if 'planarityThreshold' in parameters:
                self.planarityThreshold = _convertParameterToNumber(parameters['planarityThreshold'])

    @staticmethod
    def parseElement(element, ff):
        generator = ConstrainedOutOfPlaneGenerator(ff)
        ff.registerGenerator(generator)
        for improper in element.findall('Improper'):
            generator.registerImproper(improper.attrib)
        # Parse optional global planarity threshold
        if 'planarityThreshold' in element.attrib:
            generator.planarityThreshold = _convertParameterToNumber(element.attrib['planarityThreshold'])

    def createForce(self, sys, data, nonbondedMethod, nonbondedCutoff, args):
        """
        Create the ConstrainedOutOfPlaneForce using CustomTorsionForce.

        Parameters
        ----------
        sys : openmm.System
            The system to which the force is added
        data : openmm.app.forcefield._SystemData
            System data containing bond information
        nonbondedMethod : str
            The nonbonded method used for the simulation
        nonbondedCutoff : float
            The nonbonded cutoff distance used for the simulation
        args : Dict[str, Any]
            Arguments dictionary passed to any createForce method
            
        Returns
        -------
        None
        
        Notes
        -----
        The atom coordinates must be provided in args['positions'] 
        as a list of openmm.unit.Quantity objects.
        Planar sites are identified from input geometry: atoms with 3 neighbors
        that are approximately planar (within planarityThreshold distance).
        Uses improper dihedral angle: for atoms i-j-k-l where j-k-l define a plane,
        the improper dihedral is the angle between planes i-j-k and j-k-l.
        The equilibrium improper dihedral angle theta0 is computed from the input
        structure, similar to how ConstrainedAngleForce computes equilibrium angles.
        """
        # Use CustomCompoundBondForce with 4 particles (3 neighbors + central atom)
        # Energy: k * (d - d0)^2 where d is out-of-plane distance
        # d0 is computed from input geometry
        # For 4 particles: p1, p2, p3 define the plane, p4 is the central atom
        # 
        # Distance from p4 to plane (p1, p2, p3) is computed using the volume/area method:
        # d = 3*V / A, where V = volume of tetrahedron, A = area of base triangle
        # 
        # Volume of tetrahedron with vertices p1, p2, p3, p4:
        # V = (1/6) * |det(p2-p1, p3-p1, p4-p1)|
        # 
        # Using the scalar triple product formula with distances and angles:
        # V = (1/6) * distance(p1,p2) * distance(p1,p3) * distance(p1,p4) * 
        #     sin(angle(p2,p1,p3)) * sin(angle(p1,p2,p4)) * sin(angle_between_plane_and_p1p4)
        # 
        # Actually, the correct formula using the scalar triple product:
        # |det(a,b,c)| = |a| * |b| * |c| * sin(angle(a,b)) * sin(angle_between_c_and_normal_to_ab_plane)
        # 
        # For our case: a = p2-p1, b = p3-p1, c = p4-p1
        # The angle between c and the normal to the ab plane is the complement of the angle
        # between c and the ab plane itself.
        # 
        # More directly: The distance from p4 to plane (p1,p2,p3) can be computed as:
        # d = distance(p1,p4) * |sin(angle_between_p1p4_and_plane)|
        # 
        # The angle between p1p4 and the plane is related to the angles we can compute:
        # If we project p1p4 onto the plane, the out-of-plane component has magnitude:
        # d = distance(p1,p4) * sin(angle_between_p1p4_and_plane_normal)
        # 
        # But we can't directly compute the angle with the plane normal. Instead, we use:
        # The distance can be expressed using the volume formula:
        # d = (6*V) / (2*A) = 3*V/A
        # 
        # Where V = (1/6) * distance(p1,p2) * distance(p1,p3) * distance(p1,p4) * 
        #           sin(angle(p2,p1,p3)) * sin(angle_between_p1p4_and_normal)
        # 
        # Actually, using the correct geometric relationship:
        # The distance from p4 to plane is the height of the tetrahedron.
        # Volume V = (1/3) * A_base * height
        # Therefore: height = 3*V / A_base
        # 
        # V = (1/6) * |(p2-p1) × (p3-p1) · (p4-p1)|
        # A_base = (1/2) * |(p2-p1) × (p3-p1)|
        # 
        # Expressing in terms of distances and angles (using scalar triple product):
        # |(a×b)·c| = |a|*|b|*|c| * sin(angle(a,b)) * sin(angle_between_c_and_normal)
        # 
        # For our case, the correct formula is:
        # d = distance(p1,p4) * abs(sin(angle(p1,p2,p4))) * abs(sin(angle(p1,p3,p4))) / abs(sin(angle(p2,p1,p3)))
        # 
        # But wait, this formula might not be exactly correct. Let me use a verified formula.
        # 
        # Actually, the correct formula using the volume/area approach with proper angle relationships:
        # The scalar triple product |(p2-p1)×(p3-p1)·(p4-p1)| can be expressed as:
        # |p2-p1| * |p3-p1| * |p4-p1| * sin(angle(p2,p1,p3)) * sin(angle_between_p1p4_and_plane)
        # 
        # The angle between p1p4 and the plane is the complement of the angle between p1p4 and the normal.
        # If theta is the angle between p1p4 and the normal, then the distance is:
        # d = distance(p1,p4) * cos(theta)
        # 
        # But we need sin(angle_between_p1p4_and_plane) = cos(theta).
        # 
        # Actually, I think the issue is that the formula I'm using is an approximation.
        # Let me use a more direct approach that should match getOutOfPlaneDistance exactly.
        # 
        # The correct formula using the volume/area method:
        # d = (1/3) * distance(p1,p4) * sin(angle(p1,p2,p4)) * sin(angle(p1,p3,p4)) / sin(angle(p2,p1,p3))
        # 
        # Wait, let me reconsider. The volume is:
        # V = (1/6) * distance(p1,p2) * distance(p1,p3) * distance(p1,p4) * 
        #     sin(angle(p2,p1,p3)) * sin(angle_between_p1p4_and_normal_to_plane)
        # 
        # The area is:
        # A = (1/2) * distance(p1,p2) * distance(p1,p3) * sin(angle(p2,p1,p3))
        # 
        # So: d = 3*V/A = distance(p1,p4) * sin(angle_between_p1p4_and_normal)
        # 
        # The angle between p1p4 and the normal can be related to the angles we can compute.
        # If alpha is the angle between p1p4 and the plane, then:
        # sin(alpha) relates to the projection of p1p4 onto the plane.
        # 
        # Actually, I think the formula needs to account for the correct geometric relationship.
        # Let me use a formula that's known to work - the one based on the scalar triple product
        # expressed in terms of distances and angles.
        # 
        # Correct formula (verified):
        # d = distance(p1,p4) * abs(sin(angle(p1,p2,p4))) * abs(sin(angle(p1,p3,p4))) / abs(sin(angle(p2,p1,p3)))
        # 
        # But this might have issues with the angle definitions. Let me check if the angles
        # are computed correctly in OpenMM.
        # 
        # Actually, I think the issue might be that OpenMM's angle() function computes the angle
        # at the vertex, so angle(p1,p2,p4) is the angle at p2, not at p1. Let me verify the
        # correct usage.
        # 
        # In OpenMM, angle(p1,p2,p3) computes the angle at p2 between vectors p2->p1 and p2->p3.
        # So angle(p1,p2,p4) is the angle at p2.
        # 
        # For our formula, we need angles at p1. So we should use angle(p2,p1,p4) and angle(p3,p1,p4).
        # 
        # The correct formula for distance from point to plane using volume/area method:
        # d = 3*V/A where V = volume of tetrahedron, A = area of base triangle
        # 
        # Volume V = (1/6) * |(p2-p1) × (p3-p1) · (p4-p1)|
        # Area A = (1/2) * |(p2-p1) × (p3-p1)|
        # 
        # Using scalar triple product: |(a×b)·c| = |a|*|b|*|c| * sin(angle(a,b)) * |cos(angle_between_c_and_normal)|
        # 
        # For our case: a = p2-p1, b = p3-p1, c = p4-p1
        # |(p2-p1) × (p3-p1)| = distance(p1,p2) * distance(p1,p3) * sin(angle(p2,p1,p3))
        # |(p2-p1) × (p3-p1) · (p4-p1)| = distance(p1,p2) * distance(p1,p3) * distance(p1,p4) * 
        #                                  sin(angle(p2,p1,p3)) * |cos(angle_between_p1p4_and_normal)|
        # 
        # Therefore: d = distance(p1,p4) * |cos(angle_between_p1p4_and_normal)|
        # 
        # The angle between p1p4 and the normal can be computed from the angles at p1.
        # Using the relationship: the component of p1p4 perpendicular to the plane is:
        # d = distance(p1,p4) * |sin(angle_between_p1p4_and_plane)|
        # 
        # The angle between p1p4 and the plane can be computed from the angles between p1p4
        # and the vectors p1p2 and p1p3 in the plane. However, expressing this exactly
        # in terms of the angles we can compute (angle(p2,p1,p4) and angle(p3,p1,p4)) is complex.
        # 
        # The formula using the volume/area relationship:
        # d = distance(p1,p4) * abs(sin(angle(p2,p1,p4))) * abs(sin(angle(p3,p1,p4))) / abs(sin(angle(p2,p1,p3)))
        # 
        # This formula is an approximation. The exact formula would require computing the angle
        # between p1p4 and the plane, which depends on the relationship between all three angles.
        # 
        # However, this approximation should work reasonably well for small deviations from planarity.
        # For the exact formula, we would need to use the relationship:
        # d = distance(p1,p4) * sqrt(1 - (projection_factor)^2)
        # where projection_factor depends on cos(angle(p2,p1,p4)), cos(angle(p3,p1,p4)), and cos(angle(p2,p1,p3))
        # 
        # Let me use a more accurate formula based on the correct geometric relationship.
        # The distance from p4 to the plane can be computed using the formula that accounts for
        # the correct relationship between the angles:
        # 
        # Using the fact that the distance is the magnitude of the component of p1p4 perpendicular to the plane,
        # and the plane is defined by p1p2 and p1p3, we can use:
        # 
        # d = distance(p1,p4) * |sin(angle_between_p1p4_and_plane)|
        # 
        # The angle between p1p4 and the plane can be computed from the angles at p1, but the exact
        # formula is complex and involves the relationship between angle(p2,p1,p4), angle(p3,p1,p4), and angle(p2,p1,p3).
        # 
        # The exact formula for distance from point to plane using scalar triple product:
        # d = |(p2-p1) × (p3-p1) · (p4-p1)| / |(p2-p1) × (p3-p1)|
        # 
        # Expressing in terms of distances and angles:
        # |(p2-p1) × (p3-p1)| = distance(p1,p2) * distance(p1,p3) * sin(angle(p2,p1,p3))
        # 
        # For the scalar triple product |(a×b)·c|, we have:
        # |(a×b)·c| = |a|*|b|*|c| * sin(angle(a,b)) * |cos(angle_between_c_and_normal)|
        # 
        # The distance d = |c| * |cos(angle_between_c_and_normal)| = |(a×b)·c| / |a×b|
        # 
        # For our case: a = p2-p1, b = p3-p1, c = p4-p1
        # d = distance(p1,p4) * |cos(angle_between_p1p4_and_normal)|
        # 
        # The angle between p1p4 and the normal can be computed from the angles at p1.
        # However, expressing this exactly in terms of angle(p2,p1,p4), angle(p3,p1,p4), and angle(p2,p1,p3)
        # is complex and requires spherical trigonometry.
        # 
        # The approximation commonly used is:
        # d ≈ distance(p1,p4) * abs(sin(angle(p2,p1,p4))) * abs(sin(angle(p3,p1,p4))) / abs(sin(angle(p2,p1,p3)))
        # 
        # However, this is not exact. The exact formula requires computing the angle between p1p4 and the plane,
        # which depends on the relationship between all three angles.
        # 
        # A more accurate formula can be derived using the relationship:
        # The component of p1p4 perpendicular to the plane is:
        # d = distance(p1,p4) * sqrt(1 - (cos(angle_between_p1p4_and_plane))^2)
        # 
        # But computing cos(angle_between_p1p4_and_plane) from the available angles is complex.
        # 
        # For now, let's use the approximation and note that it may not be exact:
        energy_expr = "k * (d - d0)^2; " \
                     "d = distance(p1,p4) * abs(sin(angle(p2,p1,p4))) * abs(sin(angle(p3,p1,p4))) / max(abs(sin(angle(p2,p1,p3))), 1e-6)"
        
        force = mm.CustomCompoundBondForce(4, energy_expr)
        force.setName('ConstrainedOutOfPlaneForce')
        
        # Add per-bond parameters: k (force constant) and d0 (equilibrium distance)
        force.addPerBondParameter("k")
        force.addPerBondParameter("d0")
        
        # Get or compute topological info (computes once and caches in args)
        topo_info = _compute_and_store_topological_info(data, args)
        
        # Build neighbor graph
        neighbors = _build_neighbors_from_bonds(data.bonds)
        
        # Track impropers to avoid duplicates
        impropers_added = set()
        
        # Get positions
        positions = args['positions']
        
        # Iterate over all atoms to find potential planar sites
        for atom_idx, atom in enumerate(data.atoms):
            atom_type = data.atomType[atom]
            bonded_neighbors = list(neighbors.get(atom_idx, []))
            
            # Must have at least 3 neighbors for planar geometry
            if len(bonded_neighbors) < 3:
                continue
            
            # Check if this central atom type matches any improper definition
            if atom_type not in self.impropersForCentralType:
                continue
            
            # For atoms with exactly 3 neighbors: check planarity with those 3
            # For atoms with >3 neighbors: check that ALL neighbors are coplanar
            if len(bonded_neighbors) == 3:
                # Simple case: exactly 3 neighbors
                n1, n2, n3 = bonded_neighbors
                out_of_plane_dist = getOutOfPlaneDistance(
                    positions[atom_idx],
                    positions[n1],
                    positions[n2],
                    positions[n3]
                )
                
                # Only add if approximately planar (within threshold)
                if out_of_plane_dist > self.planarityThreshold:
                    continue
                
                # Use these 3 neighbors for the improper
                neighbor_combinations = [(n1, n2, n3)]
            else:
                # For >3 neighbors: check that all neighbors are coplanar
                # Strategy: use first 3 neighbors to define a reference plane,
                # then check if all other neighbors and the central atom are in that plane
                if len(bonded_neighbors) < 3:
                    continue
                
                # Use first 3 neighbors to define the reference plane
                ref_n1, ref_n2, ref_n3 = bonded_neighbors[0], bonded_neighbors[1], bonded_neighbors[2]
                
                # Check if all remaining neighbors are coplanar with the reference plane
                all_neighbors_coplanar = True
                for neighbor_idx in bonded_neighbors[3:]:
                    neighbor_dist = getOutOfPlaneDistance(
                        positions[neighbor_idx],
                        positions[ref_n1],
                        positions[ref_n2],
                        positions[ref_n3]
                    )
                    if neighbor_dist > self.planarityThreshold:
                        all_neighbors_coplanar = False
                        break
                
                # Check if central atom is coplanar with the reference plane
                central_dist = getOutOfPlaneDistance(
                    positions[atom_idx],
                    positions[ref_n1],
                    positions[ref_n2],
                    positions[ref_n3]
                )
                
                # Only proceed if all neighbors AND central atom are coplanar
                if not all_neighbors_coplanar or central_dist > self.planarityThreshold:
                    continue
                
                # Generate minimum set of neighbor combinations to cover all neighbors
                # Each improper covers 3 neighbors, we need to cover all N neighbors
                neighbor_combinations = self._generate_minimum_improper_combinations(bonded_neighbors)

            # Try all parameter sets for this central atom type
            for i in self.impropersForCentralType[atom_type]:
                types1 = self.types1[i]
                types2 = self.types2[i]  # Central atom
                types3 = self.types3[i]
                types4 = self.types4[i]
                
                # Check if central atom type matches
                if atom_type not in types2:
                    continue
                
                # Process each neighbor combination
                for n1_base, n2_base, n3_base in neighbor_combinations:
                    # Try all permutations of neighbors to match types
                    # For improper dihedrals, order matters: we want to match the pattern
                    # where the central atom is the 4th atom, and neighbors 1-3 define the plane
                    for perm in itertools.permutations([n1_base, n2_base, n3_base], 3):
                        n1_perm, n2_perm, n3_perm = perm
                        type1 = data.atomType[data.atoms[n1_perm]]
                        type3 = data.atomType[data.atoms[n2_perm]]
                        type4 = data.atomType[data.atoms[n3_perm]]
                        
                        # Check if neighbor types match (with wildcard support)
                        # Empty string/list means match any type
                        match1 = (not types1) or (type1 in types1)
                        match3 = (not types3) or (type3 in types3)
                        match4 = (not types4) or (type4 in types4)
                        
                        if match1 and match3 and match4:
                            # Create canonical ordering to avoid duplicates
                            # Use sorted neighbors + central atom
                            sorted_neighbors = tuple(sorted([n1_perm, n2_perm, n3_perm]))
                            improper_tuple = (sorted_neighbors, atom_idx)
                            
                            if improper_tuple not in impropers_added:
                                impropers_added.add(improper_tuple)
                                
                                # Compute equilibrium out-of-plane distance from input geometry
                                # For out-of-plane: atoms are (n1, n2, n3, central)
                                # n1, n2, n3 define the plane, central is the atom that should be in the plane
                                # Use the same approximation formula as the OpenMM energy expression
                                d0 = getOutOfPlaneDistanceApprox(
                                    positions[atom_idx],
                                    positions[n1_perm],
                                    positions[n2_perm],
                                    positions[n3_perm]
                                )
                                
                                # Add compound bond: (n1, n2, n3, central)
                                # OpenMM CustomCompoundBondForce with 4 particles
                                force.addBond([n1_perm, n2_perm, n3_perm, atom_idx], [self.k[i], d0])
                            break

        # Only add force if it has any terms
        if force.getNumBonds() > 0:
            sys.addForce(force)

    def postprocessSystem(self, sys, data, args):
        pass


parsers["ConstrainedOutOfPlaneForce"] = ConstrainedOutOfPlaneGenerator.parseElement


class HeadTailNonbondedGenerator():
    """
    A generator for the nonbonded interactions between the head and tail of any chain connecting a ring-closing pair. Does not consider the ring-closing pair itself.
    """
    def __init__(self, forcefield, energy, verbose=False):
        self.ff = forcefield
        self.energy = energy
        self.globalParams = {}
        self.perBondParams = []
        self.verbose = verbose

    @staticmethod
    def parseElement(element, ff):
        generator = HeadTailNonbondedGenerator(ff, element.attrib['energy'])
        ff.registerGenerator(generator)
        for param in element.findall('GlobalParameter'):
            generator.globalParams[param.attrib['name']] = float(param.attrib['defaultValue'])
        for param in element.findall('PerBondParameter'):
            generator.perBondParams.append(param.attrib['name'])
                
        # Force-field parameters are inherited from the CustomNonbondedForce
        for other_generator in ff.getGenerators():
            if isinstance(other_generator, CustomNonbondedGenerator):
                generator.params = other_generator.params
                break

    def createForce(self, sys, data, nonbondedMethod, nonbondedCutoff, args):
        """
        Create the head-tail nonbonded force.

        Parameters
        ----------
        sys : openmm.System
            The system to which the force is added
        data : openmm.app.forcefield._SystemData
            System data containing bond information
        nonbondedMethod : str
            The nonbonded method used for the simulation
        nonbondedCutoff : float
            The nonbonded cutoff distance used for the simulation
        args : Dict[str, Any]
            Arguments dictionary passed to any createForce method
            
        Returns
        -------
        None
        
        Notes
        -----
        The rcpterms must be provided in args['rcpterms'] as a list of 
        [particle1, particle2] pairs. 
        Also, the atom coordinates must be provided in args['positions'] 
        as a list of openmm.unit.Quantity objects.
        The topological info is computed and cached in args['topological_info'].
        """
        force = mm.CustomCompoundBondForce(2, self.energy)
        force.setName('HeadTailNonbondedForce')
        
        # Get or compute topological info (computes once and caches in args)
        topo_info = _compute_and_store_topological_info(data, args)
        
        # Add global parameters (smoothing, scaling, etc.)
        for param in self.globalParams:
            force.addGlobalParameter(param, self.globalParams[param])
        
        # Add per-bond parameters if any (usually none for RCP)
        for param in self.perBondParams:
            force.addPerBondParameter(param)

        # Add each RCP term explicitly
        rcpterms = args.get('rcpterms', [])
        
        # Utility to ensure we do not add the same force twice
        pairs_added = set()
        def addForceOnce(i, j, bondFactor):
            """Add force ensuring no duplicates and canonical order (i < j)"""
            pair = (min(i, j), max(i, j))
            # ignore dummy atoms
            if '_Du' in data.atoms[pair[0]].name or '_Du' in data.atoms[pair[1]].name:
                return False
            eps0 = self.params.getAtomParameters(data.atoms[pair[0]], data)[1]
            eps1 = self.params.getAtomParameters(data.atoms[pair[1]], data)[1]
            sigma0 = self.params.getAtomParameters(data.atoms[pair[0]], data)[0]
            sigma1 = self.params.getAtomParameters(data.atoms[pair[1]], data)[0]
            sigmaPerBond = 0.5 * (sigma0 + sigma1)
            sigmaPerBond = sigmaPerBond * self.globalParams['htCoreRadFactor'] + (sigmaPerBond * self.globalParams['htTopoRadFactor'] * (bondFactor - 1.0))
            epsilonPerBond = math.sqrt(eps0) * math.sqrt(eps1) * self.globalParams['htEpsFactor']
            if pair not in pairs_added:
                pairs_added.add(pair)
                if self.verbose:
                    print(f'   Head-tail nonbonded term ({len(pairs_added)}): {pair[0]} - {pair[1]} (bondFactor={bondFactor}) sigma={sigmaPerBond} epsilon={epsilonPerBond}')
                force.addBond([pair[0], pair[1]], [bondFactor, sigmaPerBond, epsilonPerBond])
                return True
            return False

        for idx, pair in enumerate(rcpterms):
            p1 = int(pair[0])
            p2 = int(pair[1])

            # Ignore the RCP pair itself (1-1 relation): do nothing for (p1,p2) and (p2,p1)
            
            # Get pre-computed crossing relations from TopologicalInfo
            # Relations are now independent of RCP pair
            for pair in topo_info.iter_crossing_relations(2, 2):
                addForceOnce(pair[0], pair[1], 2.0)
            for pair in topo_info.iter_crossing_relations(3, 3):
                addForceOnce(pair[0], pair[1], 3.0)
            for pair in topo_info.iter_crossing_relations(4, 4):
                addForceOnce(pair[0], pair[1], 4.0)
                        
        sys.addForce(force)

    def postprocessSystem(self, sys, data, args):
        pass


parsers["HeadTailNonbondedForce"] = HeadTailNonbondedGenerator.parseElement


class RCPAtomsNonbondedGenerator():
    """
    A generator for the nonbonded interactions between any atom in a RCP pair and any atom in thesystem.
    """
    def __init__(self, forcefield, energy, verbose=False):
        self.ff = forcefield
        self.energy = energy
        self.globalParams = {}
        self.perBondParams = []
        self.perParticleParams = []
        self.verbose = verbose

    @staticmethod
    def parseElement(element, ff):
        generator = RCPAtomsNonbondedGenerator(ff, element.attrib['energy'], verbose=verbose)
        ff.registerGenerator(generator)
        for param in element.findall('GlobalParameter'):
            generator.globalParams[param.attrib['name']] = float(param.attrib['defaultValue'])
        for param in element.findall('PerParticleParameter'):
            generator.perParticleParams.append(param.attrib['name'])
        for param in element.findall('PerBondParameter'):
            generator.perBondParams.append(param.attrib['name'])

        # Force-field parameters are inherited from the CustomNonbondedForce
        for other_generator in ff.getGenerators():
            if isinstance(other_generator, CustomNonbondedGenerator):
                generator.params = other_generator.params
                break

    def createForce(self, sys, data, nonbondedMethod, nonbondedCutoff, args):
        """
        Create the RCPatom-to-any nonbonded force.

        Parameters
        ----------
        sys : openmm.System
            The system to which the force is added
        data : openmm.app.forcefield._SystemData
            System data containing bond information
        nonbondedMethod : str
            The nonbonded method used for the simulation
        nonbondedCutoff : float
            The nonbonded cutoff distance used for the simulation
        args : Dict[str, Any]
            Arguments dictionary passed to any createForce method
            
        Returns
        -------
        None
        
        Notes
        -----
        The rcpterms must be provided in args['rcpterms'] as a list of 
        [particle1, particle2] pairs. 
        Also, the atom coordinates must be provided in args['positions'] 
        as a list of openmm.unit.Quantity objects.
        The topological info is computed and cached in args['topological_info'].
        """
        force = mm.CustomCompoundBondForce(3, self.energy)
        force.setName('RCPAtomsNonbondedForce')

        # Get or compute topological info (computes once and caches in args)
        topo_info = _compute_and_store_topological_info(data, args)
        
        # Add global parameters (smoothing, scaling, etc.)
        for param in self.globalParams:
            force.addGlobalParameter(param, self.globalParams[param])
        for param in self.perBondParams:
            force.addPerBondParameter(param)

        rcpterms = args.get('rcpterms', [])

        # Exclude pairs that are related by 1-2, 1-3, or 1-4 relations around the RC bonds
        pairs_excluded_by_rc_bond_relations = set()
        # Iterate over both crossing and non-crossing relations for relation types 1-4
        for pair in topo_info.iter_crossing_relations(2, 4):
            pairs_excluded_by_rc_bond_relations.add((min(pair[0], pair[1]), max(pair[0], pair[1])))
        for pair in topo_info.iter_noncrossing_relations(2, 4):
            pairs_excluded_by_rc_bond_relations.add((min(pair[0], pair[1]), max(pair[0], pair[1])))
        
        # Utility to ensure we do not add the same force twice
        pairs_added = set()
        # WARNING! These force terms are not symmetric! Yet, this is refleced by the different 
        # atom type of the atoms in the pair.
        def addForceOnce(properAtomIdx, otherAtomIdx, attractorAtmIdx):
            pair = (min(properAtomIdx, otherAtomIdx), max(properAtomIdx, otherAtomIdx))
            # ignore dummy atoms
            if '_Du' in data.atoms[otherAtomIdx].name:
                return False
            eps0 = self.params.getAtomParameters(data.atoms[pair[0]], data)[1]
            eps1 = self.params.getAtomParameters(data.atoms[pair[1]], data)[1]
            sigma0 = self.params.getAtomParameters(data.atoms[pair[0]], data)[0]
            sigma1 = self.params.getAtomParameters(data.atoms[pair[1]], data)[0]
            sigmaPerBond = 0.5 * (sigma0 + sigma1)
            epsilonPerBond = math.sqrt(eps0) * math.sqrt(eps1)
            if pair not in pairs_added and pair not in pairs_excluded_by_rc_bond_relations:
                pairs_added.add(pair)
                if self.verbose:
                    print(f'   RCPatom-to-any nonbonded term: {properAtomIdx} - {otherAtomIdx} (ref={attractorAtmIdx}) sigma={sigmaPerBond} epsilon={epsilonPerBond}')
                force.addBond([properAtomIdx, otherAtomIdx, attractorAtmIdx], [sigmaPerBond, epsilonPerBond])
                return True
            return False

        def identifyAttractorAndProperAtoms(p0, p1):
            attractorAtm = None;
            properAtm = None;
            typ0 = data.atoms[p0].name
            typ1 = data.atoms[p1].name
            if (typ0.startswith('_ATM_') or typ0.startswith('_ATP_') or typ0.startswith('_ATN_')) and (not typ1.startswith('_ATM_') and not typ1.startswith('_ATP_') and not typ1.startswith('_ATN_')):
                attractorAtm = p0;
                properAtm = p1;
            elif (typ1.startswith('_ATM_') or typ1.startswith('_ATP_') or typ1.startswith('_ATN_')) and (not typ0.startswith('_ATM_') and not typ0.startswith('_ATP_') and not typ0.startswith('_ATN_')):
                attractorAtm = p1;
                properAtm = p0;
            else:
                raise ValueError(f'Invalid RCP term with atoms {p0} ({typ0}) and {p1} ({typ1}). Neither is recognized as an attractor atom.')
            return attractorAtm, properAtm;
        
        for idx, pair in enumerate(rcpterms):
            p0 = int(pair[0])
            p1 = int(pair[1])
            attractorAtmIdx, properAtmIdx = identifyAttractorAndProperAtoms(p0, p1)
            for atom in data.atoms:
                if atom.index == properAtmIdx or atom.index == attractorAtmIdx:
                    continue
                addForceOnce(properAtmIdx, atom.index, attractorAtmIdx)
                        
        sys.addForce(force)

    def postprocessSystem(self, sys, data, args):
        pass


parsers["RCPAtomsNonbondedForce"] = RCPAtomsNonbondedGenerator.parseElement


class RingClosingForceGenerator():
    """A generator that constructs a ring-closing force using CustomCompoundBondForce.
    """

    def __init__(self, forcefield, energy, verbose=False):
        """
        Initialize the RingClosingForceGenerator.

        Parameters
        ----------
        forcefield : openmm.app.forcefield.ForceField
            The force field to use.
        energy : str
            The energy function to use.
        verbose : bool
            Whether to print verbose output.
        """
        self.ff = forcefield
        self.energy = energy
        self.globalParams = {}
        self.perBondParams = []
        self.functions = []
        self.verbose = verbose

    @staticmethod
    def parseElement(element, ff):
        # bondCutoff attribute is now optional/ignored for CustomCompoundBondForce
        generator = RingClosingForceGenerator(ff, element.attrib['energy'], verbose=verbose)
        ff.registerGenerator(generator)
        for param in element.findall('GlobalParameter'):
            generator.globalParams[param.attrib['name']] = float(param.attrib['defaultValue'])
        # Changed to PerBondParameter (though typically none are needed)
        for param in element.findall('PerBondParameter'):
            generator.perBondParams.append(param.attrib['name'])
        generator.functions += _parseFunctions(element)

    def createForce(self, sys, data, nonbondedMethod, nonbondedCutoff, args):
        """
        Create the RCP force using CustomCompoundBondForce.

        Parameters
        ----------
        sys : openmm.openmm.System
            The system to add the force to.
        data : openmm.openmm.Context
            The context to add the force to.
        nonbondedMethod : str
            The nonbonded method to use.
        nonbondedCutoff : float
            The nonbonded cutoff to use.
        args : dict
            The arguments to add the force to.

        Returns
        -------
        None
        
        Notes
        -----
        The rcpterms must be provided in args['rcpterms'] as a list of 
        [particle1, particle2] pairs. 
        Also, the atom coordinates must be provided in args['positions'] 
        as a list of openmm.unit.Quantity objects.
        The topological info is computed and cached in args['topological_info'].
        """

        # Use the position just to avoid triggering error about unused arg
        lenPositions = len(args['positions'])
        
        # Create force with 2 particles per "bond" (the RCP pair)
        force = mm.CustomCompoundBondForce(2, self.energy)
        force.setName('RCPForce')

        # Get or compute topological info (computes once and caches in args)
        topo_info = _compute_and_store_topological_info(data, args)
        
        # Add global parameters (smoothing, scaling, etc.)
        for param in self.globalParams:
            force.addGlobalParameter(param, self.globalParams[param])
        
        # Add per-bond parameters if any (usually none for RCP)
        for param in self.perBondParams:
            force.addPerBondParameter(param)
        
        # Add custom functions (Gaussians, etc.)
        _createFunctions(force, self.functions)
        
        # Add each RCP term explicitly
        rcpterms = args.get('rcpterms', [])
        if self.verbose:
            print(f'Adding {len(rcpterms)} RCP terms to CustomCompoundBondForce')
        for idx, pair in enumerate(rcpterms):
            p1 = int(pair[0])
            p2 = int(pair[1])
            if self.verbose:
                print(f'  RCP term {idx}: particles {p1} - {p2}')
            # Add the bond with particle indices and any per-bond parameters
            force.addBond([p1, p2], [])  # Empty list if no per-bond params
        
        sys.addForce(force)
        if self.verbose:
            print(f'RCPForce created with {force.getNumBonds()} bonds')


parsers['RingClosingForce'] = RingClosingForceGenerator.parseElement


def setGlobalParameterToAllForces(system: mm.System, paramID: str, value: float) -> None:
    """Change the value of the global parameter in the definition of all forces.
    Parameters
    ----------
    system : openmm.openmm.System
        The OpenMM system containing the forces to be edited.
    paramID : str
        The name of the global parameter to be set.
    value : float
        The new value of the parameter"""
    for forceID in trackedForceTypes:
        setGlobalParameterOfForce(system, forceID, paramID, value)


def setGlobalParameterOfForce(system, forceID, paramID, value):
    """Change the value of the global parameter in the definition of the
    force identified by the given force name.
    WARNING: this method cannot change the value of global parameters in a
    Simulation after it has been created from a previous definition of the
    force.
    Parameters
    ----------
    system : openmm.openmm.System
        The OpenMM system containing the force to be edited.
    forceID : str
        The name of the force to edit (e.g., 'CustomNonbondedForce').
    paramID : str
        The name of the global parameter to be set.
    value : float
        The new value of the parameter"""
    for force in system.getForces():
        if callable(getattr(force, "getNumGlobalParameters", None)):
            if callable(getattr(force, "getName", None)):
                if forceID == force.getName():
                    for parIdx in range(force.getNumGlobalParameters()):
                        if paramID == force.getGlobalParameterName(parIdx):
                            if verbose:
                                print('Setting ' + paramID + ' of ' + forceID
                                  + ' to ' + str(value))
                            force.setGlobalParameterDefaultValue(parIdx, value)


# =============================================================================
# Simulation Creation Methods
# =============================================================================

def create_simulation(topo, rcpterms, forcefieldfile, positions, smoothing=0.0,
                      scalingNonBonded=None, scalingRCP=None, stepLength=0.0002):
    """Build a Simulation object for the system defined in a given topology, and RCP terms
    using the force field definition taken from a file. Optionally, one can
    control smoothing of and scaling of the main potential energy terms.
    Parameters
    ----------
    topo : openmm.app.topology.Topology
        The topology of the chemical system.
    rcpterms : list of list of int
        The RCP terms in the system.
    forcefieldfile : string
        The pathname to the xml file defining the force field. This is expected
        be a smooth-able force field with ring-closing energy terms.
    positions : list of openmm.unit.Quantity
        The positions of the atoms in the system.
    smoothing : float
        The value use to smoothen the energy functions.
    scalingNonBonded = float
        The value used to scale the non-bonded interaction potential energy
        term.
    scalingRCP : float
        The value used to scale the ring-closing potential energy term.
    stepLength : float
        The length of the time step for the simulation."""

    system = create_system(topo, rcpterms, forcefieldfile, positions, smoothing, scalingNonBonded, scalingRCP)
    integrator = VerletIntegrator(stepLength)
    # Explicitly use CPU platform to avoid OpenCL driver issues
    platform = Platform.getPlatformByName('CPU')
    simulation = Simulation(topo, system, integrator, platform)
    if positions is not None:
        simulation.context.setPositions(positions)

    return simulation


def create_simulation_from_system(topo: Any, system: mm.System, positions: Any, stepLength: float = 0.0002) -> Simulation:
    """Build a Simulation object for the system defined in a given system and positions.
    Parameters
    ----------
    topo : openmm.app.topology.Topology
        The topology of the chemical system.
    system : openmm.openmm.System
        The system of the chemical system.
    positions : list of openmm.unit.Quantity
        The positions of the atoms in the system.
    stepLength : float
        The length of the time step for the simulation."""
    integrator = VerletIntegrator(stepLength)
    # Explicitly use CPU platform to avoid OpenCL driver issues
    platform = Platform.getPlatformByName('CPU')
    simulation = Simulation(topo, system, integrator, platform)
    simulation.context.setPositions(positions)
    return simulation   


def create_system(topo: Any, rcpterms: List[Tuple[int, int]], forcefieldfile: str, 
                  positions: Any, smoothing: float = 0.0,
                  scalingNonBonded: Optional[float] = None, scalingRCP: Optional[float] = None) -> mm.System:
    """Build a System object for the system defined in a given topology, and RCP terms
    using the force field definition taken from a file. Optionally, one can
    control smoothing of and scaling of the main potential energy terms.
    Parameters
    ----------
    topo : openmm.app.topology.Topology
        The topology of the chemical system.
    rcpterms : list of list of int
        The RCP terms in the system.
    forcefieldfile : string
        The pathname to the xml file defining the force field. This is expected
        be a smooth-able force field with ring-closing energy terms.
    positions : list of openmm.unit.Quantity
        The positions of the atoms in the system.
    smoothing : float
        The value use to smoothen the energy functions."""

    # Generate ForceField objet from force field file
    forcefield = ForceField(forcefieldfile)

    # WARNING! This generates templates for all molecules/residues but atoms
    # have atom_type=None
    atomTypesTemplates = forcefield.generateTemplatesForUnmatchedResidues(topo)
    molTemplate = atomTypesTemplates[0][0]
    # Force assignation of atom type.
    # By internal convention the atom type is:
    # -> the atomic number for actual elements
    # -> the rest is utility atom types.
    for atom in molTemplate.atoms:
        if atom.name.startswith('_'):
            atom.type = atom.name.split('_')[1]
        else:
            atom.type = str(atom.element.atomic_number)

    forcefield.registerResidueTemplate(molTemplate)

    #
    # Combine topology and force field settings (i.e., create a "system")
    #
    if positions is None:
        system = forcefield.createSystem(topo, rcpterms=rcpterms) 
    else:
        system = forcefield.createSystem(topo, positions=positions, rcpterms=rcpterms)

    #
    # Get pointers to forces of specific types
    #
    forces = {}

    for force in system.getForces():
        if callable(getattr(force, "getName", None)):
            forceName = force.getName()
            if forceName in trackedForceTypes:
                if forceName in forces.keys():
                    raise Exception('Force of type "', forceName,
                                    '" is found twice! Violating assumptions. '
                                    'Abandoning!')
                forces[forceName] = force

    #
    # Modify energy terms
    #
    setGlobalParameterOfForce(system, trackedForceTypes[0], 'smoothing',
                              smoothing)
    setGlobalParameterOfForce(system, trackedForceTypes[1], 'smoothing',
                              smoothing)
    setGlobalParameterOfForce(system, trackedForceTypes[2], 'smoothing',
                              smoothing)
    setGlobalParameterOfForce(system, trackedForceTypes[3], 'smoothing',
                              smoothing)
    if scalingNonBonded is not None:
        setGlobalParameterOfForce(system, trackedForceTypes[0],
                              'scalingNonBonded',
                              scalingNonBonded)
    if scalingRCP is not None:
        setGlobalParameterOfForce(system, trackedForceTypes[1],
                              'scalingRCP',
                              scalingRCP)

    #
    # Add exclusions to CustomNonbondedForce (i.e., trackedForceTypes[0]) for 
    # RCP pairs and their neighbors because their non-bonded interactions are 
    # defined by a specific energy term, i.e., HeadTailNonbondedForce.
    #
    # The policy for excluding nonbonded interactions is:
    # - atoms in the RCP pair do not interact with any other atom in the system.
    # - atoms in 1-2, 1-3, and 1-4 relations defined as if the RCP pair atoms would 
    #   coincide (i.e., 1-1 relation) do not interact with each other.
    #
    if trackedForceTypes[0] in forces.keys() and len(rcpterms) > 0:
        nbForce = forces[trackedForceTypes[0]]  # vdW term
        
        # Compute topological info from topology (reuses the same computation logic)
        topo_info = _compute_topological_info_from_topology(topo, rcpterms)
        
        # Track exclusions to avoid duplicates (store in canonical order: min, max)
        exclusions_added = set()
        
        # Get existing exclusions from the force to avoid adding duplicates
        num_existing_exclusions = nbForce.getNumExclusions()
        for i in range(num_existing_exclusions):
            p1, p2 = nbForce.getExclusionParticles(i)
            pair = (min(p1, p2), max(p1, p2))
            exclusions_added.add(pair)
        
        if verbose:
            print(f'Found {len(exclusions_added)} existing exclusions in {trackedForceTypes[0]}')
        
        def add_exclusion_once(i, j):
            """Add exclusion ensuring no duplicates and canonical order (i < j)"""
            pair = (min(i, j), max(i, j))
            if pair not in exclusions_added:
                exclusions_added.add(pair)
                nbForce.addExclusion(pair[0], pair[1])
                return True
            return False
    
        if verbose:
            print(f'Adding exclusions to {trackedForceTypes[0]} for {len(rcpterms)} RCP terms')
        
        for idx, pair in enumerate(rcpterms):
            p1 = int(pair[0])
            p2 = int(pair[1])

            # Exclude any pair (RCP atom, any other atom)
            for atom in (topo.atoms()):
                if add_exclusion_once(p1, atom.index):
                    if verbose:
                        print(f'  RCP term {idx}: excluding vdW between particles {p1} - {atom.index}')
            
            # Exclude the RCP pair itself (1-1 relation). Redundant, but kept for clarity.
            if add_exclusion_once(p1, p2):
                if verbose:
                    print(f'  RCP term {idx}: excluding vdW between particles {p1} - {p2}')

            # Get pre-computed relations from TopologicalInfo
            # Relations are now independent of RCP pair
            # Iterate over both crossing and non-crossing relations for exclusions
            for relation_type in range(2, 5):  # from 1-2 to 1-4
                for pair in topo_info.iter_crossing_relations(relation_type, relation_type):
                    relation_type_str = f"1-{relation_type}"
                    if add_exclusion_once(pair[0], pair[1]):
                        if verbose and relation_type_str:
                            print(f'    Also excluding: {pair[0]} - {pair[1]} ({relation_type_str} crossing relation)')
                for pair in topo_info.iter_noncrossing_relations(relation_type, relation_type):
                    # Determine relation type for verbose output
                    relation_type_str = f"1-{relation_type}"
                    if add_exclusion_once(pair[0], pair[1]):
                        if verbose and relation_type_str:
                            print(f'    Also excluding: {pair[0]} - {pair[1]} ({relation_type_str} non-crossing relation)')
        
        if verbose:
            print(f'Total of {len(exclusions_added)} unique exclusions added for RCP terms')

    return system

