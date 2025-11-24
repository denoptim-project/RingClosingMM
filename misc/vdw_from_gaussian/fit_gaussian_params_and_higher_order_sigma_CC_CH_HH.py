#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning

# Initial guess for the parameters
amplNbAInitial = 14000.0
widthNbAInitial = 34.0
amplNbBInitial = -5.25
widthNbBInitial = 4.0
fourth_weightInitial = 10.0

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
    aa = amplNbA * eps
    ba = widthNbA / (sigma*(1+fourth_weight*sigma*sigma*sigma))
    ab = amplNbB * eps
    bb = widthNbB / (sigma*(1+fourth_weight*sigma*sigma*sigma))
    r_squared = r**2
    gA = (aa / tb) * np.exp(-ba * r_squared / ta)
    gB = (ab / tb) * np.exp(-bb * r_squared / ta)
    return (gA + gB)


def lj(r, sigma, eps):
    """
    Computes the energy according to the 6-12 potential function
    """
    return eps * ((sigma / r)**12 - 2.0 *(sigma / r)**6)

# Range of values considered
n_points = 1000

# Prepare data for curve_fit: stack all r values and corresponding LJ target values
r_CC = np.linspace(sigmaCC-0.1, sigmaCC+0.3, n_points)
r_CH = np.linspace(sigmaCH-0.1, sigmaCH+0.3, n_points)
r_HH = np.linspace(sigmaHH-0.1, sigmaHH+0.3, n_points)
r_all = np.concatenate([r_CC, r_CH, r_HH])

# Target values (LJ energies) for each interaction type
lj_CC_target = lj(r_CC, sigmaCC, epsCC)
lj_CH_target = lj(r_CH, sigmaCH, epsCH)
lj_HH_target = lj(r_HH, sigmaHH, epsHH)
lj_all = np.concatenate([lj_CC_target, lj_CH_target, lj_HH_target])

def gau_fit_function(r_data, amplNbA, widthNbA, amplNbB, widthNbB, fourth_weight):
    """
    Wrapper function for curve_fit that handles multiple interaction types.
    The r_data is expected to be stacked: [r_CC, r_CH, r_HH]
    """
    n = len(r_data) // 3  # Each interaction type has n_points values
    r_CC = r_data[:n]
    r_CH = r_data[n:2*n]
    r_HH = r_data[2*n:]
    
    gau_CC = gau(r_CC, amplNbA, widthNbA, amplNbB, widthNbB, fourth_weight, sigmaCC, epsCC)
    gau_CH = gau(r_CH, amplNbA, widthNbA, amplNbB, widthNbB, fourth_weight, sigmaCH, epsCH)
    gau_HH = gau(r_HH, amplNbA, widthNbA, amplNbB, widthNbB, fourth_weight, sigmaHH, epsHH)
    
    return np.concatenate([gau_CC, gau_CH, gau_HH])

# Use curve_fit for least squares fitting
# Initialize with initial guesses in case fit fails
popt = [amplNbAInitial, widthNbAInitial, amplNbBInitial, widthNbBInitial, fourth_weightInitial]
pcov = None
fit_success = False

try:
    popt, pcov = curve_fit(
        gau_fit_function, 
        r_all, 
        lj_all,
        maxfev=10000,
        gtol=0.0001,
        bounds=((0.0, 0.0, -20.0, 0.0, 0.0), (np.inf, np.inf, -1.0, 10.0, 20.0)),
        p0=[amplNbAInitial, widthNbAInitial, amplNbBInitial, widthNbBInitial, fourth_weightInitial]
    )
    fit_success = True
except (RuntimeError, ValueError, OptimizeWarning) as e:
    print(f"Warning: curve_fit failed or hit maxfev limit")
    print(f"Error: {e}")
    print(f"Using initial parameter guesses instead")
    # popt already contains initial guesses
except Exception as e:
    print(f"Unexpected error: {e}")
    print(f"Error type: {type(e)}")
    print(f"Using initial parameter guesses instead")

amplNbA = popt[0]
widthNbA = popt[1]
amplNbB = popt[2]
widthNbB = popt[3]
fourth_weight = popt[4]

print(f"Optimized parameters: amplNbA = {amplNbA}, widthNbA = {widthNbA}, amplNbB = {amplNbB}, widthNbB = {widthNbB}, fourth_weight = {fourth_weight}")
if fit_success and pcov is not None:
    print(f"Parameter uncertainties (std dev):")
    param_errors = np.sqrt(np.diag(pcov))
    print(f"  amplNbA: ±{param_errors[0]:.6f}")
    print(f"  widthNbA: ±{param_errors[1]:.6f}")
    print(f"  amplNbB: ±{param_errors[2]:.6f}")
    print(f"  widthNbB: ±{param_errors[3]:.6f}")
    print(f"  fourth_weight: ±{param_errors[4]:.6f}")
else:
    print("Note: Parameter uncertainties not available (fit did not converge)")


# Compute function values
lj_CC_values = lj(r_CC, sigmaCC, epsCC)
lj_CH_values = lj(r_CH, sigmaCH, epsCH)
lj_HH_values = lj(r_HH, sigmaHH, epsHH)
gau_CC_values = gau(r_CC, amplNbA, widthNbA, amplNbB, widthNbB, fourth_weight, sigmaCC, epsCC)
gau_CH_values = gau(r_CH, amplNbA, widthNbA, amplNbB, widthNbB, fourth_weight, sigmaCH, epsCH)
gau_HH_values = gau(r_HH, amplNbA, widthNbA, amplNbB, widthNbB, fourth_weight, sigmaHH, epsHH)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(r_CC, lj_CC_values, 'r-', linewidth=2, label='lj_CC(r)')
plt.plot(r_CH, lj_CH_values, 'r-', linewidth=2, label='lj_CH(r)')
plt.plot(r_HH, lj_HH_values, 'r-', linewidth=2, label='lj_HH(r)')
plt.plot(r_CC, gau_CC_values, 'g-.', linewidth=2, label='gau_CC(r)')
plt.plot(r_CH, gau_CH_values, 'g-.', linewidth=2, label='gau_CH(r)')
plt.plot(r_HH, gau_HH_values, 'g-.', linewidth=2, label='gau_HH(r)')
plt.xlabel('Distance r (nm)', fontsize=12)
plt.ylabel('Energy (kcal/mol)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.ylim(-1, 3.0)

plt.show()

print('\nDone')

