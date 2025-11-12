#!/usr/bin/env python3
"""
Plot ring closure score as a function of distance.

This script visualizes how the exponential ring closure score changes with
the distance between the two atoms in an RCP term.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src directory to Python path
src_path = Path(__file__).resolve().parent.parent.parent / 'src'
sys.path.insert(0, str(src_path))

from MolecularSystem import MolecularSystem


def calculate_score_for_distance(distance: float, tolerance: float, decay_rate: float) -> float:
    """
    Calculate ring closure score for a given distance using MolecularSystem._ring_closure_score_exponential.
    
    Parameters
    ----------
    distance : float
        Distance between RCP term atoms (Angstroms)
    tolerance : float
        Distance threshold below which rings are considered closed
    decay_rate : float
        Exponential decay rate parameter
    
    Returns
    -------
    float
        Ring closure score in range [0, 1]
    """
    # Create dummy coordinates for two atoms at the specified distance
    # Atom 0 at origin, atom 1 at (distance, 0, 0)
    coords = np.array([
        [0.0, 0.0, 0.0],  # Atom 0
        [distance, 0.0, 0.0]  # Atom 1
    ])
    
    # RCP term connecting the two atoms
    rcp_terms = [(0, 1)]
    
    # Use the actual method from MolecularSystem
    score = MolecularSystem._ring_closure_score_exponential(
        coords=coords,
        rcp_terms=rcp_terms,
        tolerance=tolerance,
        decay_rate=decay_rate,
        verbose=False
    )
    
    return score


def plot_ring_closure_score(tolerance: float = 1.0,
                           decay_rates: list = [1.0],
                           max_distance: float = 10.0,
                           num_points: int = 1000,
                           output_file: str = None):
    """
    Plot ring closure score vs distance for one or more decay rates.
    
    Parameters
    ----------
    tolerance : float
        Distance threshold below which rings are considered closed (Angstroms)
    decay_rates : list of float
        Exponential decay rate parameter(s). Multiple values will be plotted as separate lines.
    max_distance : float
        Maximum distance to plot (Angstroms)
    num_points : int
        Number of points to plot
    output_file : str, optional
        Output file path. If None, display interactively.
    """
    # Generate distance values
    distances = np.linspace(0.0, max_distance, num_points)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colors for multiple lines
    colors = plt.cm.tab10(np.linspace(0, 1, len(decay_rates)))
    
    # Plot each decay rate as a separate line
    for i, decay_rate in enumerate(decay_rates):
        # Calculate scores for this decay rate
        scores = [calculate_score_for_distance(d, tolerance, decay_rate) for d in distances]
        
        # Plot the line
        ax.plot(distances, scores, '-', linewidth=2, color=colors[i],
               label=f'decay_rate={decay_rate:.2f}')
    
    # Add vertical line at tolerance
    ax.axvline(x=tolerance, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Tolerance ({tolerance:.2f} Å)')
    
    # Add horizontal line at score=1.0
    ax.axhline(y=1.0, color='g', linestyle='--', linewidth=1, alpha=0.5)
    
    # Mark some key points for the first decay rate (if available)
    if decay_rates:
        first_decay_rate = decay_rates[0]
        # Point at tolerance
        ax.plot(tolerance, 1.0, 'ro', markersize=8, label='Perfect closure')
        
        # Points at tolerance + 1, 2, 3 Å (using first decay rate)
        for offset in [1.0, 2.0, 3.0]:
            d = tolerance + offset
            if d <= max_distance:
                score = calculate_score_for_distance(d, tolerance, first_decay_rate)
                ax.plot(d, score, 'go', markersize=6, alpha=0.6)
                ax.annotate(f'{score:.3f}', xy=(d, score), xytext=(5, 5),
                           textcoords='offset points', fontsize=9, alpha=0.7)
    
    ax.set_xlabel('Distance between RCP term atoms (Å)', fontsize=12)
    ax.set_ylabel('Ring Closure Score', fontsize=12)
    ax.set_title('Ring Closure Score vs Distance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(0, max_distance)
    ax.set_ylim(0, 1.1)
    
    # Add text box with parameters
    decay_rates_str = ', '.join([f'{dr:.2f}' for dr in decay_rates])
    textstr = f'Tolerance: {tolerance:.2f} Å\nDecay rate(s): {decay_rates_str}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Plot ring closure score as a function of distance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default plot (single decay rate)
  python plot_ring_closure_score.py
  
  # Single decay rate
  python plot_ring_closure_score.py --tolerance 1.54 --decay-rate 0.5 --max-distance 15.0
  
  # Multiple decay rates (each as a separate line)
  python plot_ring_closure_score.py --decay-rate 0.5 1.0 2.0 3.0
  
  # Save to file
  python plot_ring_closure_score.py --output ring_closure_score.png
        """
    )
    
    parser.add_argument(
        '--tolerance',
        type=float,
        default=1.5,
        help='Distance threshold below which rings are considered closed (default: 1.5 Å)'
    )
    
    parser.add_argument(
        '--decay-rate',
        type=float,
        nargs='+',
        default=[1.0],
        help='Exponential decay rate parameter(s). Can specify multiple values to plot multiple lines. (default: 1.0)'
    )
    
    parser.add_argument(
        '--max-distance',
        type=float,
        default=10.0,
        help='Maximum distance to plot (default: 10.0 Å)'
    )
    
    parser.add_argument(
        '--num-points',
        type=int,
        default=1000,
        help='Number of points to plot (default: 1000)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path. If not specified, display interactively.'
    )
    
    args = parser.parse_args()
    
    plot_ring_closure_score(
        tolerance=args.tolerance,
        decay_rates=args.decay_rate,
        max_distance=args.max_distance,
        num_points=args.num_points,
        output_file=args.output
    )


if __name__ == '__main__':
    main()

