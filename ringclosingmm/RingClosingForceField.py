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
import math
import numpy as np
from typing import Any, Dict, List, Optional, Set, Tuple


# Conventions: names of energy terms
trackedForceTypes = ['CustomNonbondedForce',  # Index 0 used below!
                     'RCPForce',  # Index 1 used below!
                     'HeadTailNonbondedForce',  # Index 2 used below!
                     'RCPAtomsNonbondedForce',  # Index 3 used below!
                     'HarmonicBondForce',
                     'HarmonicAngleForce']

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

