#
# This is code made to understand if we can get overlap between the two RCAs
#

# Package imports (assumes package is installed)
from ringclosingmm.RingClosingForceField import create_simulation

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
atom = topo.addAtom('C0', Element.getBySymbol('C'), r0)
atoms.append(atom)
atom = topo.addAtom('C1', Element.getBySymbol('C'), r0)
atoms.append(atom)
atom = topo.addAtom('_ATM_2', Element.getBySymbol('H'), r0)
atoms.append(atom)
atom = topo.addAtom('C3', Element.getBySymbol('C'), r0)
atoms.append(atom)
atom = topo.addAtom('H4', Element.getBySymbol('H'), r0)
atoms.append(atom)
atom = topo.addAtom('C5', Element.getBySymbol('C'), r0)
atoms.append(atom)
atom = topo.addAtom('H6', Element.getBySymbol('H'), r0)
atoms.append(atom)
atom = topo.addAtom('C7', Element.getBySymbol('C'), r0)
atoms.append(atom)
atom = topo.addAtom('C8', Element.getBySymbol('C'), r0)
atoms.append(atom)
atom = topo.addAtom('C9', Element.getBySymbol('C'), r0)
atoms.append(atom)
atom = topo.addAtom('H10', Element.getBySymbol('H'), r0)
atoms.append(atom)
atom = topo.addAtom('H11', Element.getBySymbol('H'), r0)
atoms.append(atom)
topo.addBond(atoms[0], atoms[1], 1)
topo.addBond(atoms[1], atoms[2], 1)
topo.addBond(atoms[6], atoms[1], 1)
topo.addBond(atoms[3], atoms[4], 1)
topo.addBond(atoms[3], atoms[5], 1)
topo.addBond(atoms[5], atoms[7], 1)
topo.addBond(atoms[8], atoms[0], 1)
topo.addBond(atoms[7], atoms[9], 1)
topo.addBond(atoms[0], atoms[10], 1)
topo.addBond(atoms[9], atoms[11], 1)

#
#        C8
#        |
#        C_0 - H10
#        |
#        C_1 - H6
#        |
#      ATM_2
#        .
#        .
#        C3 - H4
#        |  
#        C5
#        |
#        C7
#        |
#        C9 - H11
#

# Create positions as numpy array with units (nanometers)
positions = np.array([[-0.15, 0, 0],   # C0    
                      [0.0, 0, 0],     # C1
                      [0.15, 0, 0],    # ATM_2
                      [0.15, 0, 0],    # C3
                      [0.15, 0.1, 0],  # H4
                      [0.30, 0, 0],    # C5
                      [0, 0.1, 0],       # H6
                      [0.45, 0, 0],    # C7
                      [-0.30, 0, 0],    # C8
                      [0.60, 0, 0],    # C9
                      [-0.15, -0.1, 0],    # H10
                      [0.60, -0.1, 0],    # H11
                      ]) * unit.nanometer  

rcpterms = [[3, 2]]
simulation = create_simulation(topo, rcpterms, forcefieldfile, positions)

for i in range(nsteps):
    distance = i * step + initDistance
    # Update positions array (keep units)
    positions_updated = positions + np.array([
        [0, 0, 0],    # C0
        [0, 0, 0],    # C1
        [0, 0, 0],    # ATM_2
        [0, 0, distance],    # C3
        [0, 0, distance],    # H4
        [0, 0, distance],    # C5   
        [0, 0, 0],           # H6
        [0, 0, distance],    # C7
        [0, 0, 0],           # C8
        [0, 0, distance],    # C9
        [0, 0, 0],           # H10
        [0, 0, distance],    # H11
    ]) * unit.nanometer
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
