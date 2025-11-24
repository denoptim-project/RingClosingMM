#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning

# Initial guess for the parameters
amplNbAInitial = 900.0
widthNbAInitial = 20.0
amplNbBInitial = -6.0
widthNbBInitial = 4.0

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

square_weightInitial = 0.0
fourth_weightInitial = 10.0

def gau(r, amplNbA, widthNbA, amplNbB, widthNbB, square_weight, fourth_weight, sigma, eps):
    """
    Compute vdW term
    """
    sigma_dep = sigma * (1 + square_weight*sigma + fourth_weight*sigma*sigma*sigma)
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

# Range of values considered
n_points = 1000

choser_eps = epsCH
choser_sigma = sigmaCH

chosen_r = np.linspace(choser_sigma-0.1, choser_sigma+0.5, n_points)
chosen_lj_target = lj(chosen_r, choser_sigma, choser_eps)

def gau_fit_function(r_data, amplNbA, widthNbA, amplNbB, widthNbB, square_weight, fourth_weight):
    """
    Wrapper function for curve_fit that handles CH interaction only.
    """
    return gau(r_data, amplNbA, widthNbA, amplNbB, widthNbB, square_weight, fourth_weight, choser_sigma, choser_eps)

# Use curve_fit for least squares fitting
# Initialize with initial guesses in case fit fails
popt = [amplNbAInitial, widthNbAInitial, amplNbBInitial, widthNbBInitial, square_weightInitial, fourth_weightInitial]
pcov = None
fit_success = False

try:
    popt, pcov = curve_fit(
        gau_fit_function, 
        chosen_r, 
        chosen_lj_target,
        maxfev=10000,
        gtol=0.00001,
        bounds=((0.0, 0.0, -20.0, 0.0, 0.0, 0.0), (np.inf, np.inf, -1.0, 10.0, 20.0, 20.0)),
        p0=[amplNbAInitial, widthNbAInitial, amplNbBInitial, widthNbBInitial, square_weightInitial, fourth_weightInitial]
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
square_weight = popt[4]
fourth_weight = popt[5]

print(f"Optimized parameters: \namplNbA = {amplNbA}\nwidthNbA = {widthNbA}\namplNbB = {amplNbB}\nwidthNbB = {widthNbB}\nsquare_weight = {square_weight}\nfourth_weight = {fourth_weight}")
if fit_success and pcov is not None:
    print(f"Parameter uncertainties (std dev):")
    param_errors = np.sqrt(np.diag(pcov))
    print(f"  amplNbA: ±{param_errors[0]:.6f}")
    print(f"  widthNbA: ±{param_errors[1]:.6f}")
    print(f"  amplNbB: ±{param_errors[2]:.6f}")
    print(f"  widthNbB: ±{param_errors[3]:.6f}")
    print(f"  square_weight: ±{param_errors[4]:.6f}")
    print(f"  fourth_weight: ±{param_errors[5]:.6f}")
else:
    print("Note: Parameter uncertainties not available (fit did not converge)")

plotted_r = np.linspace(0.1, 0.7, n_points)

lj_CC_values = lj(plotted_r, sigmaCC, epsCC)
lj_CH_values = lj(plotted_r, sigmaCH, epsCH)
lj_HH_values = lj(plotted_r, sigmaHH, epsHH)
gau_CC_values = gau(plotted_r, amplNbA, widthNbA, amplNbB, widthNbB, square_weight, fourth_weight, sigmaCC, epsCC)
gau_CH_values = gau(plotted_r, amplNbA, widthNbA, amplNbB, widthNbB, square_weight, fourth_weight, sigmaCH, epsCH)
gau_HH_values = gau(plotted_r, amplNbA, widthNbA, amplNbB, widthNbB, square_weight, fourth_weight, sigmaHH, epsHH)

plt.figure(figsize=(10, 6))
plt.plot(plotted_r, lj_CC_values, 'y-', linewidth=2, label='lj_CC(r)')
plt.plot(plotted_r, lj_CH_values, 'r-', linewidth=2, label='lj_CH(r)')
plt.plot(plotted_r, lj_HH_values, 'r-', linewidth=2, label='lj_HH(r)')
plt.plot(plotted_r, gau_CC_values, 'b-.', linewidth=2, label='gau_CC(r)')
plt.plot(plotted_r, gau_CH_values, 'g-.', linewidth=2, label='gau_CH(r)')
plt.plot(plotted_r, gau_HH_values, 'g-.', linewidth=2, label='gau_HH(r)')
plt.xlabel('Distance r (nm)', fontsize=12)
plt.ylabel('Energy (kcal/mol)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.ylim(-0.2, 0.5)
plt.xlim(0.2, 0.5)

plt.show()

print('\nDone')

