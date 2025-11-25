#
# This is code made to understand if we can get overlap between the two RCAs
#

# Package imports (assumes package is installed)
from ringclosingmm.RingClosingForceField import create_simulation

import openmm.unit as unit
import matplotlib.pyplot as plt
from openmm.app import Topology, Element

initDistance = -1.0 # in nm
finalDistance = 1.0 # in nm
nsteps = 1000
step = (finalDistance - initDistance) / nsteps
distances = []
energiesCustom = []
energiesCustomS1 = []
energiesCustomS2 = []
energiesCustomS3 = []

# Create the base Topology for the vdW term
topo = Topology()
c0 = topo.addChain()
r0 = topo.addResidue('res0', c0)
topo.addAtom('_ATN_0', Element.getBySymbol('H'), r0)
topo.addAtom('N', Element.getBySymbol('N'), r0)
topo.addAtom('Cl', Element.getBySymbol('Cl'), r0)
topo.addAtom('C', Element.getBySymbol('C'), r0)
positions = [[0, 0, 0], 
             [0, 0.1, 0],
             [0, 0.5, 0],
             [0, 0, 0],
             ]
rcpterms = [[0, 3]]

smoothingValues = [0.0, 0.05, 0.1, 0.5]

simulationCustom = create_simulation(topo, rcpterms, "components_both.xml", positions, smoothing=smoothingValues[0])
simulationCustomS1 = create_simulation(topo, rcpterms, "components_dist.xml", positions, smoothing=smoothingValues[0])
simulationCustomS2 = create_simulation(topo, rcpterms, "components_sigma.xml", positions, smoothing=smoothingValues[0])

for i in range(nsteps):
    distance = i * step + initDistance
    positions[3] = [0, distance, 0]
    simulationCustom.context.setPositions(positions)
    eCustom = simulationCustom.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    simulationCustomS1.context.setPositions(positions)
    eCustomS1 = simulationCustomS1.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    simulationCustomS2.context.setPositions(positions)
    eCustomS2 = simulationCustomS2.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    distances.append(distance)
    energiesCustom.append(eCustom)
    energiesCustomS1.append(eCustomS1)
    energiesCustomS2.append(eCustomS2)

    #print(f'Step {i} of {nsteps} Distance: {distance:.4f}, Energy Custom: {eCustom}, Energy Custom S1: {eCustomS1}, Energy Custom S2: {eCustomS2}')


energiesCustom = [e - min(energiesCustom) for e in energiesCustom]
energiesCustomS1 = [e - min(energiesCustomS1) for e in energiesCustomS1]
energiesCustomS2 = [e - min(energiesCustomS2) for e in energiesCustomS2]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(distances, energiesCustom, 'b-', linewidth=2)
plt.plot(distances, energiesCustomS1, 'y-.', linewidth=2)
plt.plot(distances, energiesCustomS2, 'm-.', linewidth=2)
plt.xlabel('Distance (nm)', fontsize=12)
plt.ylabel('Energy (kJ/mol)', fontsize=12)
plt.legend(['Both', 'Distance', 'Sigma'], fontsize=12)
plt.grid(True, alpha=0.3)
#plt.ylim(-5.0, 10000.0)
plt.xlim(initDistance, finalDistance)
plt.tight_layout()
#plt.savefig('energy_profile.png', dpi=300)
#print('Plot saved as energy_profile.png')
plt.show()

print('Done')
