#!/usr/bin/env python3
"""
Example: Using RingClosureOptimizer Python API

This script demonstrates how to use the RingClosureOptimizer module
programmatically for custom workflows.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
src_path = Path(__file__).resolve().parent.parent / 'src'
sys.path.insert(0, str(src_path))

from RingClosureOptimizer import RingClosureOptimizer


def example_1_simple_usage():
    """Example 1: Simple high-level API usage."""
    print("=" * 70)
    print("Example 1: Simple Usage")
    print("=" * 70)
    
    # Create optimizer from files
    # Note: rotatable_bonds and rcp_terms are now direct lists of tuples (0-based indices)
    # Example: rotatable_bonds=[(0, 1), (4, 30)] means atoms 1-2 and 5-31 form rotatable bonds
    optimizer = RingClosureOptimizer.from_files(
        structure_file='../test/cyclic_bond_formation/test.int',
        forcefield_file='../data/RCP_UFFvdW.xml',
        rotatable_bonds=[(0, 1), (4, 30), (30, 31), (31, 34), (34, 39), (34, 40), (31, 35), (31, 36)],  # 0-based
        rcp_terms=[(6, 38), (76, 34)]  # 0-based
    )
    
    # Run optimization
    result = optimizer.optimize(
        population_size=20,
        generations=10,
        enable_smoothing_refinement=True,
        enable_cartesian_refinement=True,
        verbose=True
    )
    
    # Print results
    print(f"\n✅ Optimization complete!")
    print(f"   Initial closure score: {result['initial_closure_score']:.4f}")
    final_scores = result['final_closure_score']
    if isinstance(final_scores, (list, np.ndarray)):
        best_score = max(final_scores)
    else:
        best_score = final_scores
    print(f"   Final closure score:   {best_score:.4f}")
    print(f"   Improvement:           {best_score - result['initial_closure_score']:+.4f}")
    
    # Energy is stored per individual
    if optimizer.top_candidates and optimizer.top_candidates[0].energy is not None:
        print(f"   Final energy:          {optimizer.top_candidates[0].energy:.2f} kcal/mol")
    
    # Save structure
    optimizer.save_optimized_structure('optimized_simple.xyz')
    print(f"   Saved to: optimized_simple.xyz")


def example_2_custom_parameters():
    """Example 2: Custom GA parameters."""
    print("\n" + "=" * 70)
    print("Example 2: Custom Parameters")
    print("=" * 70)
    
    optimizer = RingClosureOptimizer.from_files(
        structure_file='../test/cyclic_bond_formation/test.int',
        forcefield_file='../data/RCP_UFFvdW.xml',
        rotatable_bonds=[(0, 1), (4, 30), (30, 31), (31, 34), (34, 39), (34, 40), (31, 35), (31, 36)],  # 0-based
        rcp_terms=[(6, 38), (76, 34)]  # 0-based
    )
    
    # Aggressive optimization settings
    result = optimizer.optimize(
        population_size=50,
        generations=20,
        mutation_rate=0.2,
        mutation_strength=15.0,
        crossover_rate=0.8,
        elite_size=10,
        enable_smoothing_refinement=True,
        enable_cartesian_refinement=True,
        refinement_top_n=5,
        verbose=False  # Quiet mode
    )
    
    print(f"✅ Aggressive optimization complete!")
    final_scores = result['final_closure_score']
    if isinstance(final_scores, (list, np.ndarray)):
        best_score = max(final_scores)
    else:
        best_score = final_scores
    print(f"   Final closure score: {best_score:.4f}")
    if optimizer.top_candidates and optimizer.top_candidates[0].energy is not None:
        print(f"   Final energy: {optimizer.top_candidates[0].energy:.2f} kcal/mol")


def example_3_pure_ga():
    """Example 3: Pure GA (no local refinement)."""
    print("\n" + "=" * 70)
    print("Example 3: Pure GA (No Refinement)")
    print("=" * 70)
    
    optimizer = RingClosureOptimizer.from_files(
        structure_file='../test/cyclic_bond_formation/test.int',
        forcefield_file='../data/RCP_UFFvdW.xml',
        rotatable_bonds=[(0, 1), (4, 30), (30, 31), (31, 34), (34, 39), (34, 40), (31, 35), (31, 36)],  # 0-based
        rcp_terms=[(6, 38), (76, 34)]  # 0-based
    )
    
    # Pure genetic algorithm (no local refinement)
    result = optimizer.optimize(
        population_size=30,
        generations=20,
        enable_smoothing_refinement=False,  # Disable refinement
        enable_cartesian_refinement=False,
        verbose=False
    )
    
    print(f"✅ Pure GA optimization complete!")
    final_scores = result['final_closure_score']
    if isinstance(final_scores, (list, np.ndarray)):
        best_score = max(final_scores)
    else:
        best_score = final_scores
    print(f"   Final closure score: {best_score:.4f}")
    print(f"   No refinement was used (pure GA)")


def example_4_access_components():
    """Example 4: Access individual components."""
    print("\n" + "=" * 70)
    print("Example 4: Accessing Components")
    print("=" * 70)
    
    optimizer = RingClosureOptimizer.from_files(
        structure_file='../test/cyclic_bond_formation/test.int',
        forcefield_file='../data/RCP_UFFvdW.xml',
        rotatable_bonds=[(0, 1), (4, 30), (30, 31), (31, 34), (34, 39), (34, 40), (31, 35), (31, 36)],  # 0-based
        rcp_terms=[(6, 38), (76, 34)]  # 0-based
    )
    
    # Access molecular system
    print(f"Molecular system:")
    print(f"  Atoms: {len(optimizer.system.elements)}")
    print(f"  Z-matrix size: {len(optimizer.system.zmatrix)}")
    print(f"  Rotatable dihedrals: {len(optimizer.rotatable_indices)}")
    
    # Show rotatable atoms
    print(f"\nRotatable atoms:")
    for i, idx in enumerate(optimizer.rotatable_indices[:5]):  # Show first 5
        atom = optimizer.system.zmatrix[idx]
        element = optimizer.system.elements[idx]
        print(f"  {i+1}. Atom {idx+1} ({element}) - "
              f"bond {atom['bond_ref']}-{atom['angle_ref']}")
    
    if len(optimizer.rotatable_indices) > 5:
        print(f"  ... and {len(optimizer.rotatable_indices) - 5} more")


def main():
    """Run all examples."""
    print("\n")
    print("*" * 70)
    print("* RingClosureOptimizer Python API Examples")
    print("*" * 70)
    print()
    print("These examples demonstrate different ways to use the")
    print("RingClosureOptimizer module programmatically.")
    print()
    
    try:
        # Run examples
        example_1_simple_usage()
        example_2_custom_parameters()
        example_3_pure_ga()
        example_4_access_components()
        
        print("\n" + "=" * 70)
        print("✅ All examples completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

