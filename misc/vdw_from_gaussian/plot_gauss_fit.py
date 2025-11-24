#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# Initial guess for the parameters
amplNbA = 35500.0
widthNbA = 60.0
amplNbB = -3.0
widthNbB = 5.3
fourth_weight = 13.0

# Constants related to smoothing
smoothing = 0.0
ta = 1.0 + 4.0 * smoothing
tb = np.sqrt(ta**3)

# Homonuclear parameters
sigmaC = 0.3851  # nm
sigmaH = 0.2886  # nm
epsilonC= 0.1050  # kcal/mol
epsilonH = 0.044  # kcal/mol

# Combined parameters for specific interactions
epsCC = np.sqrt(epsilonC) * np.sqrt(epsilonC)
sigmaCC = 0.5*(sigmaC+sigmaC)

epsCH = np.sqrt(epsilonC) * np.sqrt(epsilonH)
sigmaCH = 0.5*(sigmaC+sigmaH)

epsHH = np.sqrt(epsilonH) * np.sqrt(epsilonH)
sigmaHH = 0.5*(sigmaH+sigmaH)

def gau(r, amplNbA, widthNbA, amplNbB, widthNbB, fourth_weight, sigma, eps):
    """
    Compute vdW term
    """
    sigma_dep = sigma * (1 + fourth_weight*sigma*sigma*sigma)
    aa = amplNbA * eps
    ba = widthNbA / sigma_dep
    ab = amplNbB * eps
    bb = widthNbB / sigma_dep
    r_squared = r**2
    gA = (aa / tb) * np.exp(-ba * r_squared / ta)
    gB = (ab / tb) * np.exp(-bb * r_squared / ta)
    return (gA + gB)


def lj(r, sigma, eps):
    """
    Computes the energy according to the 6-12 potential function
    """
    return eps * ((sigma / r)**12 - 2.0 *(sigma / r)**6)


# Prepare data for curve_fit: stack all r values and corresponding LJ target values
n_points = 1000
r_all = np.linspace(0.1, 1.0, n_points)

# Compute function values
lj_CC_values = lj(r_all, sigmaCC, epsCC)
lj_CH_values = lj(r_all, sigmaCH, epsCH)
lj_HH_values = lj(r_all, sigmaHH, epsHH)
gau_CC_values = gau(r_all, amplNbA, widthNbA, amplNbB, widthNbB, fourth_weight, sigmaCC, epsCC)
gau_CH_values = gau(r_all, amplNbA, widthNbA, amplNbB, widthNbB, fourth_weight, sigmaCH, epsCH)
gau_HH_values = gau(r_all, amplNbA, widthNbA, amplNbB, widthNbB, fourth_weight, sigmaHH, epsHH)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(r_all, lj_CC_values, 'y-', linewidth=2, label='lj_CC(r)')
plt.plot(r_all, lj_CH_values, 'r-', linewidth=2, label='lj_CH(r)')
plt.plot(r_all, lj_HH_values, 'k-', linewidth=2, label='lj_HH(r)')
plt.plot(r_all, gau_CC_values, 'b-.', linewidth=2, label='gau_CC(r)')
plt.plot(r_all, gau_CH_values, 'g-.', linewidth=2, label='gau_CH(r)')
plt.plot(r_all, gau_HH_values, 'm-.', linewidth=2, label='gau_HH(r)')
plt.xlabel('Distance r (nm)', fontsize=12)
plt.ylabel('Energy (kcal/mol)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.ylim(-0.3, 1.0)
plt.xlim(0.2, 0.5)

plt.show()

print('\nDone')

