#!/usr/bin/env python3
"""
Example: Using RingClosureOptimizer Python API

This script demonstrates how to use the RingClosureOptimizer module
programmatically for custom workflows.

Prerequisites:
    The package must be installed (via conda: `conda install ringclosingmm`,
    or in editable mode for development within a conda environment).
"""

import sys
import numpy as np

from ringclosingmm import RingClosureOptimizer, IOTools


def usage_example():
    """Example: Simple high-level API usage."""
    print("=" * 70)
    print("Example: Simple Usage")
    print("=" * 70)
    
    # Create optimizer from files
    # Note: rotatable_bonds and rcp_terms are now direct lists of tuples (0-based indices)
    # Example: rotatable_bonds=[(0, 1), (4, 30)] means atoms 1-2 and 5-31 form rotatable bonds
    optimizer = RingClosureOptimizer.from_files(
        structure_file='test.int',
        forcefield_file='../../data/RCP_UFFvdW.xml',
        rotatable_bonds=[(0, 1), (4, 30), (30, 31), (31, 34), (34, 39), (34, 40), (31, 35), (31, 36)],  # 0-based
        rcp_terms=[(6, 38), (76, 34)]  # 0-based
    )
    
    # Run optimization
    result = optimizer.optimize(
        enable_pssrot_refinement=True,
        enable_zmatrix_refinement=True,
        verbose=True
    )
    
    # Print results
    print(f"\n✅ Optimization complete!")
    print(f"   Initial ring closure score: {result['initial_ring_closure_score']:.4f}")
    print(f"   Final ring closure score:   {result['final_ring_closure_score']:.4f}")
    print(f"   Improvement:                {result['final_ring_closure_score'] - result['initial_ring_closure_score']:+.4f}")
    print(f"   Initial energy:             {result['initial_energy']:.2f} kcal/mol")
    print(f"   Final energy:               {result['final_energy']:.2f} kcal/mol")
    print(f"   Energy improvement:        {result['initial_energy'] - result['final_energy']:+.2f} kcal/mol")
    
    # Save structure
    IOTools.save_structure_to_file('optimized.xyz', result['final_zmatrix'], result['final_energy'])
    print(f"   Saved to: optimized.xyz")


def access_components_example():
    """Example: Access individual components."""
    print("\n" + "=" * 70)
    print("Example: Accessing Components")
    print("=" * 70)
    
    optimizer = RingClosureOptimizer.from_files(
        structure_file='test.int',
        forcefield_file='../../data/RCP_UFFvdW.xml',
        rotatable_bonds=[(0, 1), (4, 30), (30, 31), (31, 34), (34, 39), (34, 40), (31, 35), (31, 36)],  # 0-based
        rcp_terms=[(6, 38), (76, 34)]  # 0-based
    )
    
    # Access molecular system
    print(f"Molecular system:")
    print(f"  Atoms: {len(optimizer.system.elements)}")
    print(f"  Z-matrix size: {len(optimizer.system.zmatrix)}")
    print(f"  Rotatable dihedrals: {len(optimizer.system.rotatable_indices)}")
    print(f"  RC-critical rotatable dihedrals: {len(optimizer.system.rc_critical_rotatable_indeces)}")
    print(f"  DOF indices: {len(optimizer.system.dof_indices)}")
    
    # Show rotatable atoms
    print(f"\nRotatable atoms (first 5):")
    for i, idx in enumerate(optimizer.system.rotatable_indices[:5]):  # Show first 5
        atom = optimizer.system.zmatrix[idx]
        element = optimizer.system.elements[idx]
        is_critical = idx in optimizer.system.rc_critical_rotatable_indeces
        critical_marker = " [RC-critical]" if is_critical else ""
        print(f"  {i+1}. Atom {idx} ({element}) - "
              f"dihedral ref: {atom.get('dihedral_ref', 'N/A')}{critical_marker}")
    
    if len(optimizer.system.rotatable_indices) > 5:
        print(f"  ... and {len(optimizer.system.rotatable_indices) - 5} more")


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
        usage_example()
        access_components_example()
        
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

