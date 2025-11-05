#
# This is code made to understand if we can get overlap between the two RCAs
#

import subprocess
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).resolve().parent.parent.parent / 'src'
sys.path.insert(0, str(src_path))

from RingClosingForceField import (
    create_simulation
)

import openmm.unit as unit
import openmm
import matplotlib.pyplot as plt
from openmm.app import Topology, Element
import numpy as np

forcefieldfile = '../../data/RCP_UFFvdW.xml'
initDistance = 0.00 # in nm
initVdwDistance = 0.25 # in nm
step = 0.0025 # in nm
nsteps = 200
distances = []
energies = []

# Create the base Topology for the vdW term
topo = Topology()
c0 = topo.addChain()
r0 = topo.addResidue('res0', c0)
atoms = []
atom = topo.addAtom('_ATM_0', Element.getBySymbol('H'), r0)
atoms.append(atom)
atom = topo.addAtom('C1', Element.getBySymbol('C'), r0)
atoms.append(atom)
atom = topo.addAtom('H2', Element.getBySymbol('H'), r0)
atoms.append(atom)
atom = topo.addAtom('C3', Element.getBySymbol('C'), r0)
atoms.append(atom)
atom = topo.addAtom('C4', Element.getBySymbol('C'), r0)
atoms.append(atom)
topo.addBond(atoms[1], atoms[2], 1)
topo.addBond(atoms[1], atoms[3], 1)
topo.addBond(atoms[0], atoms[4], 1)

#        C4
#        |
#      ATM_0
#        |
#        |
#        C1 - H2
#        |  
#        C3

# Create positions as numpy array with units (nanometers)
positions = np.array([[0, 0, 0], 
                      [0, 0, 0],
                      [0.1, 0, 0],
                      [0, 0.1, 0],
                      [0.1, 0, 0]]) * unit.nanometer

rcpterms = [[0, 1]]
simulation = create_simulation(topo, rcpterms, forcefieldfile, positions)

for i in range(nsteps):
    distance = i * step + initDistance
    # Update positions array (keep units)
    positions_updated = positions + np.array([[0, 0, distance], 
                                   [0, 0, 0],
                                   [0, 0, 0],
                                   [0, 0, 0],
                                   [0, 0, distance]]) * unit.nanometer
    simulation.context.setPositions(positions_updated)
    e = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    distances.append(distance)
    energies.append(e)
    print(f'Step {i} of {nsteps} Distance: {distance:.4f}, Energy: {e}')

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(distances, energies, 'b-', linewidth=2)
plt.xlabel('Distance (nm)', fontsize=12)
plt.ylabel('Energy (kJ/mol)', fontsize=12)
plt.legend(['OpenMM'], fontsize=12)
plt.title('RCP Energy Profile', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print('Done')
