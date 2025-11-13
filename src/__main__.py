#!/usr/bin/env python3
"""
Command-line interface for Ring Closure Optimization.

This script provides a user-friendly interface to the RingClosureOptimizer module
for optimizing molecular ring closing conformations using a divide and conquer strategy for
exploration of the torsional space with subsequent local refinement.

Example usage:
    # After installation: pip install -e .
    rc-optimizer -i molecule.int -r bonds.txt -c rcp.txt -o output.xyz
    
    # As a Python module
    python -m src -i molecule.int -r bonds.txt -c rcp.txt -o output.xyz
    
    # Direct script execution
    python src/__main__.py -i molecule.int -r bonds.txt -c rcp.txt -o output.xyz
"""

import argparse
import sys
import numpy as np
from pathlib import Path

from src import IOTools

# Custom formatter that removes "(default: False)" from boolean flags
class CustomHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _get_help_string(self, action):
        help_str = super()._get_help_string(action)
        # Remove "(default: False)" from help text
        if help_str and "(default: False)" in help_str:
            help_str = help_str.replace(" (default: False)", "")
        return help_str

# Handle both module and script execution
try:
    from .RingClosureOptimizer import RingClosureOptimizer
    from .RCOServer import start, stop, status
except ImportError:
    from RingClosureOptimizer import RingClosureOptimizer
    from RCOServer import start, stop, status


# Default configuration values
DEFAULT_RING_CLOSURE_TOLERANCE = 0.1  # Angstroms (C-C bond)
DEFAULT_RING_CLOSURE_DECAY_RATE = 0.5  # Exponential decay rate
DEFAULT_TORSIONAL_ITERATIONS = 50
DEFAULT_ZMATRIX_ITERATIONS = 50


# Default forcefield path (relative to this script's location)
DEFAULT_FORCEFIELD = str(Path(__file__).parent.parent / 'data' / 'RCP_UFFvdW.xml')


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Ring Closure Optimization using Hybrid Genetic Algorithm with OpenMM',
        formatter_class=CustomHelpFormatter,
        epilog='For more information, see RingClosureOptimizer module documentation.'
    )
    
    # Required arguments (conditionally required - not needed for server operations)
    required = parser.add_argument_group('Required Arguments')
    required.add_argument('-i', '--input',
                         help='Input structure file (.int format with Z-matrix). '
                              'Required unless using server operations (--server-start, --server-stop, --server-status)')
    required.add_argument('-r', '--rot-bonds', required=False, nargs='+', type=int,
                         help='Rotatable bonds as space-separated pairs of atom indices (1-based). '
                              'Each pair is two integers: atom1 atom2 atom3 atom4 ... '
                              'Example: --rot-bonds 1 2 5 31 31 32. '
                              'If not provided, all bonds will be considered rotatable.')
    
    # Optional input/output
    io_group = parser.add_argument_group('Input/Output Files')
    io_group.add_argument('-c', '--rcp-terms', nargs='+', type=int,
                         help='RCP terms as space-separated pairs of atom indices (1-based, optional). '
                              'Each pair is two integers: atom1 atom2 atom3 atom4 ... '
                              'Example: --rcp-terms 7 39 77 35')
    io_group.add_argument('-f', '--forcefield',
                         default=DEFAULT_FORCEFIELD,
                         help='Force field XML file')
    io_group.add_argument('-o', '--output',
                         default='optimized_structure.xyz',
                         help='Output optimized structure file')
    
    # Ring closure parameters
    ring_closure_group = parser.add_argument_group('Ring Closure Parameters')
    ring_closure_group.add_argument('--ring-closure-tolerance', type=float,
                          default=DEFAULT_RING_CLOSURE_TOLERANCE,
                          help='Distance threshold (Å) for perfect ring closure score')
    ring_closure_group.add_argument('--ring-closure-decay-rate', type=float,
                          default=DEFAULT_RING_CLOSURE_DECAY_RATE,
                          help='Exponential decay rate for ring closure score')

    # Refinement parameters
    mem_group = parser.add_argument_group('Refinement Parameters')
    mem_group.add_argument('--no-pssrot-refinement', action='store_true',
                          help='Disable smoothing-based torsional refinement')
    mem_group.add_argument('--no-zmatrix-refinement', action='store_true',
                          help='Disable Z-matrix space refinement')
    mem_group.add_argument('--torsional-iterations', type=int,
                          default=DEFAULT_TORSIONAL_ITERATIONS,
                          help='Iterations per torsional optimization step')
    mem_group.add_argument('--zmatrix-iterations', type=int,
                          default=DEFAULT_ZMATRIX_ITERATIONS,
                          help='Iterations for Z-matrix space minimization')
    
    # Minimization option 
    min_group = parser.add_argument_group('Local Energy Minimization')
    min_group.add_argument('--minimize', action='store_true',
                          help='Perform single-structure minimization instead of GA optimization')
    min_group.add_argument('--space-type', type=str, default='zmatrix',
                          choices=['torsional', 'zmatrix', 'Cartesian'],
                          help='Define the space type for minimization (default: zmatrix)')
    min_group.add_argument('--smoothing', type=float, nargs='+',
                          help='Smoothing parameter(s) for minimization. '
                               'Single value or sequence (e.g., 10.0 or 50.0 25.0 10.0 0.0). '
                               'No smoothing by default')
    min_group.add_argument('--max-iterations', type=int,
                          default=500,
                          help='Maximum iterations for minimization')
    min_group.add_argument('--gradient-tolerance', type=float,
                          default=0.01,
                          help='Gradient tolerance convergecy critrion for minimization')
    min_group.add_argument('--zmatrix-dof-bounds', nargs='+', type=float,
                          help='Bounds for the three types of degrees of freedom in Z-matrix space. Example: 0.02 5.0 10.0 means that bond lengths are bound to change by up to 0.02 Å from the current value, angles and 5.0 degrees, and torsions by 10.0 degrees from the current value. Multiple triplets can be provided to request any stepwise application of bounds. Example: 0.02 5.0 10.0 0.01 3.0 8.0 means will make the minimization run with [0.02, 5.0, 10.0] for the first step and [0.01, 3.0, 8.0] for the second step. Default is 0.02 20.0 180.0.')
   
   # Server management options
    server_group = parser.add_argument_group('Server Management')
    server_group.add_argument('--server-start', action='store_true',
                             help='Start the socket server (--host and --port optional, '
                                  'defaults to localhost:0 for auto-select)')
    server_group.add_argument('--server-stop', action='store_true',
                              help='Stop the running server (requires --host and --port)')
    server_group.add_argument('--server-status', action='store_true',
                              help='Check server status (requires --host and --port)')
    server_group.add_argument('--host', type=str, default='localhost',
                              help='Server hostname (default: localhost)')
    server_group.add_argument('--port', type=int,
                              help='Server port number (required for --server-stop and --server-status, '
                                   'optional for --server-start: 0 = auto-select)')
    
    # Other options
    parser.add_argument('-v', '--verbose', action='store_true',
                        default=False,
                        help='Print verbose progress output')
    args = parser.parse_args()
    
    # Validate rotatable bonds if provided (must be even number for pairs)
    if args.rot_bonds and len(args.rot_bonds) % 2 != 0:
        parser.error("Rotatable bonds must be specified as pairs. "
                     f"Got {len(args.rot_bonds)} values, need even number.")
    
    # Validate RCP terms if provided (must be even number for pairs)
    if args.rcp_terms and len(args.rcp_terms) % 2 != 0:
        parser.error("RCP terms must be specified as pairs. "
                     f"Got {len(args.rcp_terms)} values, need even number.")
    
    # Note: --space-type can be used with --minimize to specify the minimization space
    
    # Validate server operations
    server_ops = [args.server_start, args.server_stop, args.server_status]
    if any(server_ops):
        if sum(server_ops) > 1:
            parser.error("Only one server operation can be specified at a time")
        
        # For server-start, port is optional (defaults to 0 for auto-select)
        # For stop/status, port is required
        if args.server_stop or args.server_status:
            if args.port is None:
                parser.error(f"--port is required for --server-stop and --server-status")
        
        # If starting server, port=0 is allowed (auto-select)
        if args.server_start and args.port is None:
            args.port = 0
    else:
        # If not a server operation, --input is required
        if not args.input:
            parser.error("--input is required unless using server operations (--server-start, --server-stop, --server-status)")
    
    return args


def main():
    """Main execution function."""
    args = parse_arguments()
    
    verbose = args.verbose
    
    # Handle server operations first (these don't require input files)
    if args.server_start:
        try:
            print("=" * 70)
            print("Starting Ring Closure Optimizer Server")
            print("=" * 70)
            print(f"\nHost: {args.host}")
            print(f"Port: {args.port} (auto-select if 0)")
        
            # Start server (blocks until server stops)
            # Use nohup or & for background execution: nohup rc-optimizer --server-start &
            print(f"\nServer starting...")
            
            try:
                host, port = start(host=args.host, port=args.port)
            except KeyboardInterrupt:
                print("\nServer stopped.")
                return 0
            except Exception as e:
                # If server fails to start, raise it
                raise
            
            return 0
        except Exception as e:
            print(f"\nError starting server: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1
    
    if args.server_stop:
        try:
            print(f"Stopping server at {args.host}:{args.port}...")
            stop((args.host, args.port))
            print("Server stopped successfully.")
            
            return 0
        except Exception as e:
            print(f"\nError stopping server: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1
    
    if args.server_status:
        try:
            status_info = status((args.host, args.port))
            
            if verbose:
                print("=" * 70)
                print("Server Status")
                print("=" * 70)
                print(f"Host: {status_info['host']}")
                print(f"Port: {status_info['port']}")
                print(f"Status: {'RUNNING' if status_info['running'] else 'NOT RUNNING'}")
                if status_info['error']:
                    print(f"Error: {status_info['error']}")
            else:
                # Quiet mode: just print status
                print("RUNNING" if status_info['running'] else "NOT RUNNING")
            
            return 0 if status_info['running'] else 1
        except Exception as e:
            print(f"\nError checking server status: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1
    
    # Parse Z-matrix DOF bounds from list of floats to list of tuples
    zmatrix_dof_bounds_per_type = None
    if args.zmatrix_dof_bounds:
        zmatrix_dof_bounds_per_type = [(args.zmatrix_dof_bounds[i], args.zmatrix_dof_bounds[i+1], args.zmatrix_dof_bounds[i+2]) for i in range(0, len(args.zmatrix_dof_bounds), 3)]

    # Parse rotatable bonds from list of integers to list of tuples (0-based)
    # Convert from 1-based (user input) to 0-based (internal representation)
    # If not provided, set to None (will be interpreted as "all bonds rotatable")
    rotatable_bonds = None
    if args.rot_bonds:
        rotatable_bonds = [(args.rot_bonds[i] - 1, args.rot_bonds[i+1] - 1) 
                          for i in range(0, len(args.rot_bonds), 2)]
    
    # Parse RCP terms from list of integers to list of tuples (0-based, if provided)
    # Convert from 1-based (user input) to 0-based (internal representation)
    rcp_terms = None
    if args.rcp_terms:
        rcp_terms = [(args.rcp_terms[i] - 1, args.rcp_terms[i+1] - 1) 
                     for i in range(0, len(args.rcp_terms), 2)]
    
    print("=" * 70)
    print("Ring Closure Optimizer")
    print("=" * 70)
    print(f"\nInput structure: {args.input}")
    if rotatable_bonds:
        print(f"Rotatable bonds: {rotatable_bonds}")
    else:
        print(f"Rotatable bonds: ALL (not specified)")
    if rcp_terms:
        print(f"RCP terms: {rcp_terms}")
    print(f"Force field: {args.forcefield}")
    print(f"Output: {args.output}")
        
    try:
        optimizer = RingClosureOptimizer.from_files(
            structure_file=args.input,
            forcefield_file=args.forcefield,
            rotatable_bonds=rotatable_bonds,
            rcp_terms=rcp_terms,
            ring_closure_threshold=args.ring_closure_tolerance
        )
        
        if verbose:
            print(f"  Z-matrix size: {len(optimizer.system.zmatrix)}")
            print(f"  Rotatable dihedrals: {len(optimizer.system.rotatable_indices)}")
            if args.rcp_terms:
                print(f"  Ring closure tolerance: {args.ring_closure_tolerance} Å")
                print(f"  Ring closure decay rate: {args.ring_closure_decay_rate}")
            print("-" * 70)
        
        # Run minimization or optimization
        if args.minimize:
            # Parse smoothing parameter
            smoothing = None
            if args.smoothing:
                if len(args.smoothing) == 1:
                    smoothing = args.smoothing[0]
                else:
                    smoothing = args.smoothing
            
            result = optimizer.minimize(
                    max_iterations=args.max_iterations,
                    smoothing=smoothing,
                    space_type=args.space_type,
                    zmatrix_dof_bounds_per_type=zmatrix_dof_bounds_per_type,
                    gradient_tolerance=args.gradient_tolerance,
                    verbose=verbose
                )

            IOTools.save_structure_to_file(args.output, result['zmatrix'], result['final_energy'])
    
        else:
            result = optimizer.optimize(
                ring_closure_tolerance=args.ring_closure_tolerance,
                ring_closure_decay_rate=args.ring_closure_decay_rate,
                enable_pssrot_refinement=not args.no_pssrot_refinement,
                enable_zmatrix_refinement=not args.no_zmatrix_refinement,
                smoothing_sequence=None,  # Use default
                torsional_iterations=args.torsional_iterations,
                zmatrix_iterations=args.zmatrix_iterations,
                verbose=verbose
            )
            
            print("\n" + "=" * 70)
            print("RESULTS")
            print("=" * 70)
            print(f"Initial ring closure score: {result['initial_closure_score']:.4f}")
            print(f"Initial energy: {result['initial_energy']:.4f} kcal/mol")
            print(f"Final ring closure score:   {result['final_closure_score']:.4f}")
            print(f"Final energy: {result['final_energy']:.4f} kcal/mol")
            
            IOTools.save_structure_to_file(args.output, result['final_zmatrix'], result['final_energy'])
    
        print(f"\nFinal structure saved to: {args.output}")
        print("=" * 70)
        
        return 0
    
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

