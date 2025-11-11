#!/usr/bin/env python3
"""
Script to compute energy and gradient for a structure file.

This script loads a structure file (complex_result.int) and computes the energy
and gradient similar to how minimize_energy_in_zmatrix_space does it.

Usage:
    python compute_energy_gradient.py [structure_file] [forcefield_file] [--rot-bonds ...] [--rcp-terms ...]
"""

import argparse
import sys
import copy
from pathlib import Path
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.IOTools import write_xyz_file
from src.MolecularSystem import MolecularSystem
from src.RingClosureOptimizer import RingClosureOptimizer
from src.CoordinateConverter import zmatrix_to_cartesian

# Default forcefield path
DEFAULT_FORCEFIELD = str(Path(__file__).parent / 'data' / 'RCP_UFFvdW.xml')


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Compute energy and gradient for a structure file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('structure_file',
                       type=str,
                       nargs='?',
                       default='examples/strained_ring_closure/complex_result.int',
                       help='Input structure file (.int format)')
    
    parser.add_argument('-f', '--forcefield',
                       type=str,
                       default=DEFAULT_FORCEFIELD,
                       help='Force field XML file')
    
    parser.add_argument('-r', '--rot-bonds',
                       nargs='+',
                       type=int,
                       help='Rotatable bonds as space-separated pairs of atom indices (1-based). '
                            'Each pair is two integers: atom1 atom2 atom3 atom4 ... '
                            'Example: --rot-bonds 1 2 5 31. '
                            'If not provided, all bonds will be considered rotatable.')
    
    parser.add_argument('-c', '--rcp-terms',
                       nargs='+',
                       type=int,
                       help='RCP terms as space-separated pairs of atom indices (1-based, optional). '
                            'Each pair is two integers: atom1 atom2 atom3 atom4 ... '
                            'Example: --rcp-terms 7 39 77 35')
    
    parser.add_argument('-v', '--verbose',
                       action='store_true',
                       help='Print detailed information')
    
    args = parser.parse_args()
    
    # Validate rotatable bonds if provided (must be even number for pairs)
    if args.rot_bonds and len(args.rot_bonds) % 2 != 0:
        parser.error("Rotatable bonds must be specified as pairs. "
                     f"Got {len(args.rot_bonds)} values, need even number.")
    
    # Validate RCP terms if provided (must be even number for pairs)
    if args.rcp_terms and len(args.rcp_terms) % 2 != 0:
        parser.error("RCP terms must be specified as pairs. "
                     f"Got {len(args.rcp_terms)} values, need even number.")
    
    return args


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Check if structure file exists
    structure_path = Path(args.structure_file)
    if not structure_path.exists():
        print(f"Error: Structure file not found: {args.structure_file}", file=sys.stderr)
        return 1
    
    # Check if forcefield file exists
    forcefield_path = Path(args.forcefield)
    if not forcefield_path.exists():
        print(f"Error: Force field file not found: {args.forcefield}", file=sys.stderr)
        return 1
    
    print("=" * 70)
    print("Energy and Gradient Computation")
    print("=" * 70)
    print(f"\nStructure file: {args.structure_file}")
    print(f"Force field: {args.forcefield}")
    
    try:
        # Parse rotatable bonds from list of integers to list of tuples (0-based)
        rotatable_bonds = None
        if args.rot_bonds:
            rotatable_bonds = [(args.rot_bonds[i] - 1, args.rot_bonds[i+1] - 1) 
                              for i in range(0, len(args.rot_bonds), 2)]
            print(f"Rotatable bonds: {rotatable_bonds}")
        else:
            print("Rotatable bonds: ALL (not specified)")
        
        # Parse RCP terms from list of integers to list of tuples (0-based, if provided)
        rcp_terms = None
        if args.rcp_terms:
            rcp_terms = [(args.rcp_terms[i] - 1, args.rcp_terms[i+1] - 1) 
                         for i in range(0, len(args.rcp_terms), 2)]
            print(f"RCP terms: {rcp_terms}")
        
        # Create optimizer to get DOF indices (similar to how minimize_energy_in_zmatrix_space works)
        print("\nInitializing molecular system...")
        optimizer = RingClosureOptimizer.from_files(
            structure_file=str(structure_path),
            forcefield_file=str(forcefield_path),
            rotatable_bonds=rotatable_bonds,
            rcp_terms=rcp_terms,
            write_candidate_files=False
        )
        
        print(f"  Atoms: {len(optimizer.system.elements)}")
        print(f"  Z-matrix size: {len(optimizer.system.zmatrix)}")
        print(f"  Rotatable dihedrals: {len(optimizer.rotatable_indices)}")
        print(f"  DOF indices: {len(optimizer.dof_indices)}")
        
        # Get the zmatrix and dof_indices
        zmatrix = optimizer.system.zmatrix
        dof_indices = optimizer.dof_indices
        
        if len(dof_indices) == 0:
            print("\nWarning: No DOF indices found. Cannot compute gradient.")
            print("This might happen if no rotatable bonds are specified or found.")
            return 1
        
        # Compute energy and gradient (similar to minimize_energy_in_zmatrix_space)
        print("\nComputing energy and gradient...")
        print("-" * 70)
        
        # Convert Z-matrix to Cartesian coordinates
        coords = zmatrix_to_cartesian(zmatrix)
        
        # Evaluate energy
        energy = optimizer.system.evaluate_energy(coords)
        
        # Evaluate gradient
        gradient = optimizer.system.evaluate_dofs_gradient(zmatrix, dof_indices)
        
        # Compute gradient statistics
        gradient_norm = np.linalg.norm(gradient)
        max_gradient = np.max(np.abs(gradient))
        min_gradient = np.min(np.abs(gradient))
        mean_gradient = np.mean(np.abs(gradient))
        
        # Print results
        print(f"\nResults:")
        print(f"  Energy: {energy:.6f} kcal/mol")
        print(f"\n  Gradient Statistics:")
        print(f"    Gradient norm: {gradient_norm:.6f}")
        print(f"    Max gradient:  {max_gradient:.6f}")
        print(f"    Min gradient:  {min_gradient:.6f}")
        print(f"    Mean gradient: {mean_gradient:.6f}")
        
        if args.verbose:
            print(f"\n  Detailed Gradient Information:")
            print(f"    Number of DOFs: {len(dof_indices)}")
            print(f"\n    DOF gradients:")
            
            dof_names = ['bond_length', 'angle', 'dihedral']
            for j, dof_index in enumerate(dof_indices):
                zmat_idx = dof_index[0]
                dof_type = dof_index[1]
                atom = zmatrix[zmat_idx]
                grad_value = gradient[j]
                
                # Get DOF value from zmatrix
                dof_value = atom.get(dof_names[dof_type], None)
                
                # Format DOF description and value
                if dof_type == 0:
                    dof_desc = f"Bond length: atom {atom['id']+1} - {atom.get('bond_ref', 'N/A')+1 if atom.get('bond_ref') is not None else 'N/A'}"
                    dof_value_str = f"{dof_value:.4f} Å" if dof_value is not None else "N/A"
                elif dof_type == 1:
                    dof_desc = f"Angle: atom {atom['id']+1} - {atom.get('bond_ref', 'N/A')+1 if atom.get('bond_ref') is not None else 'N/A'} - {atom.get('angle_ref', 'N/A')+1 if atom.get('angle_ref') is not None else 'N/A'}"
                    dof_value_str = f"{dof_value:.4f}°" if dof_value is not None else "N/A"
                elif dof_type == 2:
                    if atom.get('chirality', 0) != 0:
                        dof_desc = f"Second angle (chirality={atom.get('chirality', 0)}): atom {atom['id']+1}"
                        dof_value_str = f"{dof_value:.4f}°" if dof_value is not None else "N/A"
                    else:
                        dof_desc = f"Dihedral: atom {atom['id']+1} - {atom.get('bond_ref', 'N/A')+1 if atom.get('bond_ref') is not None else 'N/A'} - {atom.get('angle_ref', 'N/A')+1 if atom.get('angle_ref') is not None else 'N/A'} - {atom.get('dihedral_ref', 'N/A')+1 if atom.get('dihedral_ref') is not None else 'N/A'}"
                        dof_value_str = f"{dof_value:.4f}°" if dof_value is not None else "N/A"
                
                print(f"      DOF {j+1:3d} ({dof_index[0]:3d}, {dof_type}): value={dof_value_str:>10s}, gradient={grad_value:10.6f}  [{dof_desc}]")
        
        # Steepest descent optimization loop
        step_size = -1.0
        gradient_tolerance = 0.001
        max_iterations = 1000
        
        print(f"\nStarting steepest descent optimization...")
        print(f"  Step size: {step_size}")
        print(f"  Gradient tolerance: {gradient_tolerance}")
        print(f"  Max iterations: {max_iterations}")
        print("-" * 70)
        
        # Initialize for loop
        current_zmatrix = copy.deepcopy(zmatrix)
        current_energy = energy
        current_gradient = gradient.copy()
        current_gradient_norm = gradient_norm
        dof_names = ['bond_length', 'angle', 'dihedral']
        current_dofs = np.array([current_zmatrix[idx[0]][dof_names[idx[1]]] for idx in dof_indices])
        initial_dofs = current_dofs.copy()  # Store initial DOF values for comparison
        
        iteration = 0
        converged = False
        
        while current_gradient_norm >= gradient_tolerance and iteration < max_iterations:
            iteration += 1
            
            # Steepest descent: move in direction opposite to gradient (for minimization)
            new_dofs = current_dofs - current_gradient * step_size
            
            # Update zmatrix with new DOF values
            updated_zmatrix = copy.deepcopy(current_zmatrix)
            for j, dof_index in enumerate(dof_indices):
                zmat_idx = dof_index[0]
                dof_idx = dof_index[1]
                updated_zmatrix[zmat_idx][dof_names[dof_idx]] = new_dofs[j]
            
            # Recalculate energy and gradient with updated zmatrix
            updated_coords = zmatrix_to_cartesian(updated_zmatrix)
            updated_energy = optimizer.system.evaluate_energy(updated_coords)
            updated_gradient = optimizer.system.evaluate_dofs_gradient(updated_zmatrix, dof_indices)
            updated_gradient_norm = np.linalg.norm(updated_gradient)
            
            # Print iteration progress
            energy_change = updated_energy - current_energy
            gradient_change = updated_gradient_norm - current_gradient_norm
            print(f"\nIter {iteration:4d}: Energy={updated_energy:12.6f} kcal/mol (Δ={energy_change:+10.6f}), "
                  f"Gradient norm={updated_gradient_norm:.6f} (Δ={gradient_change:+.6f})")
            write_xyz_file(updated_coords, optimizer.system.elements, "trajectory.xyz", comment=f"Iter {iteration:4d}: Energy={updated_energy:12.6f} kcal/mol (Δ={energy_change:+10.6f}), Gradient norm={updated_gradient_norm:.6f} (Δ={gradient_change:+.6f})", append=True)
            if args.verbose:
                print(f"  DOF gradients:")
                
                for j, dof_index in enumerate(dof_indices):
                    zmat_idx = dof_index[0]
                    dof_type = dof_index[1]
                    atom = updated_zmatrix[zmat_idx]
                    grad_value = updated_gradient[j]
                    dof_value = new_dofs[j]
                    dof_change = new_dofs[j] - current_dofs[j]  # Change from previous iteration
                    
                    # Format DOF description and value
                    if dof_type == 0:
                        dof_desc = f"Bond length: atom {atom['id']+1} - {atom.get('bond_ref', 'N/A')+1 if atom.get('bond_ref') is not None else 'N/A'}"
                        dof_value_str = f"{dof_value:.4f} Å"
                        dof_change_str = f"{dof_change:+.4f} Å"
                    elif dof_type == 1:
                        dof_desc = f"Angle: atom {atom['id']+1} - {atom.get('bond_ref', 'N/A')+1 if atom.get('bond_ref') is not None else 'N/A'} - {atom.get('angle_ref', 'N/A')+1 if atom.get('angle_ref') is not None else 'N/A'}"
                        dof_value_str = f"{dof_value:.4f}°"
                        dof_change_str = f"{dof_change:+.4f}°"
                    elif dof_type == 2:
                        if atom.get('chirality', 0) != 0:
                            dof_desc = f"Second angle (chirality={atom.get('chirality', 0)}): atom {atom['id']+1}"
                            dof_value_str = f"{dof_value:.4f}°"
                            dof_change_str = f"{dof_change:+.4f}°"
                        else:
                            dof_desc = f"Dihedral: atom {atom['id']+1} - {atom.get('bond_ref', 'N/A')+1 if atom.get('bond_ref') is not None else 'N/A'} - {atom.get('angle_ref', 'N/A')+1 if atom.get('angle_ref') is not None else 'N/A'} - {atom.get('dihedral_ref', 'N/A')+1 if atom.get('dihedral_ref') is not None else 'N/A'}"
                            dof_value_str = f"{dof_value:.4f}°"
                            dof_change_str = f"{dof_change:+.4f}°"
                    
                    print(f"    DOF {j+1:3d} ({dof_index[0]:3d}, {dof_type}): value={dof_value_str:>10s} (Δ={dof_change_str:>10s}), gradient={grad_value:10.6f}  [{dof_desc}]")
            
            # Update for next iteration
            current_zmatrix = updated_zmatrix
            current_energy = updated_energy
            current_gradient = updated_gradient
            current_gradient_norm = updated_gradient_norm
            current_dofs = new_dofs
            
            # Check convergence
            if current_gradient_norm < gradient_tolerance:
                converged = True
                break
        
        # Print final results
        print("\n" + "=" * 70)
        if converged:
            print(f"Optimization converged after {iteration} iterations")
        else:
            print(f"Optimization stopped after {iteration} iterations (max iterations reached)")
        
        print(f"\nFinal Results:")
        print(f"  Energy: {current_energy:.6f} kcal/mol (initial: {energy:.6f} kcal/mol, change: {current_energy - energy:+.6f} kcal/mol)")
        
        # Compute final gradient statistics
        final_max_gradient = np.max(np.abs(current_gradient))
        final_min_gradient = np.min(np.abs(current_gradient))
        final_mean_gradient = np.mean(np.abs(current_gradient))
        
        print(f"\n  Final Gradient Statistics:")
        print(f"    Gradient norm: {current_gradient_norm:.6f} (initial: {gradient_norm:.6f}, change: {current_gradient_norm - gradient_norm:+.6f})")
        print(f"    Max gradient:  {final_max_gradient:.6f} (initial: {max_gradient:.6f}, change: {final_max_gradient - max_gradient:+.6f})")
        print(f"    Min gradient:  {final_min_gradient:.6f} (initial: {min_gradient:.6f}, change: {final_min_gradient - min_gradient:+.6f})")
        print(f"    Mean gradient: {final_mean_gradient:.6f} (initial: {mean_gradient:.6f}, change: {final_mean_gradient - mean_gradient:+.6f})")
        
        if args.verbose:
            print(f"\n  Detailed Final Gradient Information:")
            print(f"    Number of DOFs: {len(dof_indices)}")
            print(f"\n    DOF gradients:")
            
            # Get initial DOF values for comparison
            initial_dofs = np.array([zmatrix[idx[0]][dof_names[idx[1]]] for idx in dof_indices])
            
            for j, dof_index in enumerate(dof_indices):
                zmat_idx = dof_index[0]
                dof_type = dof_index[1]
                atom = current_zmatrix[zmat_idx]
                grad_value = current_gradient[j]
                dof_value = current_dofs[j]
                dof_change = current_dofs[j] - initial_dofs[j]
                
                # Format DOF description and value
                if dof_type == 0:
                    dof_desc = f"Bond length: atom {atom['id']+1} - {atom.get('bond_ref', 'N/A')+1 if atom.get('bond_ref') is not None else 'N/A'}"
                    dof_value_str = f"{dof_value:.4f} Å"
                    dof_change_str = f"{dof_change:+.4f} Å"
                elif dof_type == 1:
                    dof_desc = f"Angle: atom {atom['id']+1} - {atom.get('bond_ref', 'N/A')+1 if atom.get('bond_ref') is not None else 'N/A'} - {atom.get('angle_ref', 'N/A')+1 if atom.get('angle_ref') is not None else 'N/A'}"
                    dof_value_str = f"{dof_value:.4f}°"
                    dof_change_str = f"{dof_change:+.4f}°"
                elif dof_type == 2:
                    if atom.get('chirality', 0) != 0:
                        dof_desc = f"Second angle (chirality={atom.get('chirality', 0)}): atom {atom['id']+1}"
                        dof_value_str = f"{dof_value:.4f}°"
                        dof_change_str = f"{dof_change:+.4f}°"
                    else:
                        dof_desc = f"Dihedral: atom {atom['id']+1} - {atom.get('bond_ref', 'N/A')+1 if atom.get('bond_ref') is not None else 'N/A'} - {atom.get('angle_ref', 'N/A')+1 if atom.get('angle_ref') is not None else 'N/A'} - {atom.get('dihedral_ref', 'N/A')+1 if atom.get('dihedral_ref') is not None else 'N/A'}"
                        dof_value_str = f"{dof_value:.4f}°"
                        dof_change_str = f"{dof_change:+.4f}°"
                
                print(f"      DOF {j+1:3d} ({dof_index[0]:3d}, {dof_type}): value={dof_value_str:>10s} (Δ={dof_change_str:>10s}), gradient={grad_value:10.6f}  [{dof_desc}]")
        
        print("\n" + "=" * 70)
        return 0
    
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

