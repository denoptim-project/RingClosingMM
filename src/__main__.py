#!/usr/bin/env python3
"""
Command-line interface for Ring Closure Optimization.

This script provides a user-friendly interface to the RingClosureOptimizer module
for optimizing molecular ring closure using a genetic algorithm for
exploration of the torsional space with optional post-processing local refinement.

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
DEFAULT_POPULATION = 50
DEFAULT_GENERATIONS = 50
DEFAULT_CROSSOVER = 0.7
DEFAULT_MUTATION = 0.50
DEFAULT_MUTATION_STRENGTH = 30.0
DEFAULT_ELITE_SIZE = 5
DEFAULT_TORSION_MIN = -180.0
DEFAULT_TORSION_MAX = 180.0
DEFAULT_CONVERGENCE = 0.01
DEFAULT_PRINT_INTERVAL = 1
DEFAULT_CONVERGENCE_INTERVAL = 5
DEFAULT_SYSTEMATIC_SAMPLING_DIVISIONS = 4
#
# Ring closure score parameters
DEFAULT_RING_CLOSURE_TOLERANCE = 0.1  # Angstroms (C-C bond)
DEFAULT_RING_CLOSURE_DECAY_RATE = 0.5  # Exponential decay rate
#
# Smoothing refinement parameters  
DEFAULT_REFINEMENT_TOP_N = 1
DEFAULT_TORSIONAL_ITERATIONS = 50
DEFAULT_ZMATRIX_ITERATIONS = 50
DEFAULT_REFINEMENT_CONVERGENCE = 0.01
DEFAULT_SMOOTHING_SEQUENCE = [50.0, 25.0, 10.0, 5.0, 2.5, 1.0, 0.0]


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
    
    # Genetic Algorithm parameters
    ga_group = parser.add_argument_group('Genetic Algorithm Parameters')
    ga_group.add_argument('--population', type=int,
                         default=DEFAULT_POPULATION,
                         help='Population size')
    ga_group.add_argument('--generations', type=int,
                         default=DEFAULT_GENERATIONS,
                         help='Number of generations')
    ga_group.add_argument('--crossover', type=float,
                         default=DEFAULT_CROSSOVER,
                         help='Crossover rate (0.0-1.0)')
    ga_group.add_argument('--mutation', type=float,
                         default=DEFAULT_MUTATION,
                         help='Mutation rate (0.0-1.0)')
    ga_group.add_argument('--mutation-strength', type=float,
                         default=DEFAULT_MUTATION_STRENGTH,
                         help='Mutation strength in degrees')
    ga_group.add_argument('--elite-size', type=int,
                         default=DEFAULT_ELITE_SIZE,
                         help='Number of elite individuals preserved')
    ga_group.add_argument('--systematic-sampling-divisions', type=int,
                         default=DEFAULT_SYSTEMATIC_SAMPLING_DIVISIONS,
                         help='Number of discrete values for systematic sampling of critical torsions')
    
    # Optimization parameters
    opt_group = parser.add_argument_group('Optimization Parameters')
    opt_group.add_argument('--torsion-min', type=float,
                          default=DEFAULT_TORSION_MIN,
                          help='Minimum torsion angle (degrees)')
    opt_group.add_argument('--torsion-max', type=float,
                          default=DEFAULT_TORSION_MAX,
                          help='Maximum torsion angle (degrees)')
    opt_group.add_argument('--convergence', type=float,
                          default=DEFAULT_CONVERGENCE,
                          help='Fitness convergence threshold for early stopping')
    opt_group.add_argument('--convergence-interval', type=int,
                          default=DEFAULT_CONVERGENCE_INTERVAL,
                          help='Number of generations without improvement to wait for convergence')
    opt_group.add_argument('--ring-closure-tolerance', type=float,
                          default=DEFAULT_RING_CLOSURE_TOLERANCE,
                          help='Distance threshold (Å) for perfect ring closure score')
    opt_group.add_argument('--ring-closure-decay-rate', type=float,
                          default=DEFAULT_RING_CLOSURE_DECAY_RATE,
                          help='Exponential decay rate for ring closure score')

    # Refinement parameters
    mem_group = parser.add_argument_group('Refinement (Applied after GA Convergence)')
    mem_group.add_argument('--no-smoothing-refinement', action='store_true',
                          help='Disable smoothing-based torsional refinement')
    mem_group.add_argument('--no-zmatrix-refinement', action='store_true',
                          help='Disable Z-matrix space refinement')
    mem_group.add_argument('--refinement-top-n', type=int,
                          default=DEFAULT_REFINEMENT_TOP_N,
                          help='Number of top candidates to refine')
    mem_group.add_argument('--torsional-iterations', type=int,
                          default=DEFAULT_TORSIONAL_ITERATIONS,
                          help='Iterations per torsional optimization step')
    mem_group.add_argument('--zmatrix-iterations', type=int,
                          default=DEFAULT_ZMATRIX_ITERATIONS,
                          help='Iterations for Z-matrix space minimization')
    mem_group.add_argument('--refinement-convergence', type=float,
                          default=DEFAULT_REFINEMENT_CONVERGENCE,
                          help='Fitness improvement threshold for GA convergence')
    
    # Minimization option 
    min_group = parser.add_argument_group('Standalone EnergyMinimization')
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
    parser.add_argument('--print-interval', type=int,
                       default=DEFAULT_PRINT_INTERVAL,
                       help='Print progress every N generations')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress progress output')
    parser.add_argument('--write-candidate-files', action='store_true',
                       default=False,
                       help='Write candidate files (int and xyz)')
    args = parser.parse_args()
    
    # Validate arguments
    if args.crossover < 0 or args.crossover > 1:
        parser.error("Crossover rate must be between 0.0 and 1.0")
    if args.mutation < 0 or args.mutation > 1:
        parser.error("Mutation rate must be between 0.0 and 1.0")
    if args.population < 2:
        parser.error("Population size must be at least 2")
    if args.generations < 1:
        parser.error("Number of generations must be at least 1")
    if args.elite_size >= args.population:
        parser.error("Elite size must be smaller than population size")
    if args.torsion_min >= args.torsion_max:
        parser.error("Minimum torsion must be less than maximum torsion")
    
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
    
    verbose = not args.quiet
    
    # Handle server operations first (these don't require input files)
    if args.server_start:
        try:
            if verbose:
                print("=" * 70)
                print("Starting Ring Closure Optimizer Server")
                print("=" * 70)
                print(f"\nHost: {args.host}")
                print(f"Port: {args.port} (auto-select if 0)")
            
            # Start server (blocks until server stops)
            # Use nohup or & for background execution: nohup rc-optimizer --server-start &
            if verbose:
                print(f"\nServer starting...")
            
            try:
                host, port = start(host=args.host, port=args.port)
            except KeyboardInterrupt:
                if verbose:
                    print("\nServer stopped.")
                return 0
            except Exception as e:
                # If server fails to start, raise it
                raise
            
            return 0
        except Exception as e:
            print(f"\nError starting server: {e}", file=sys.stderr)
            if verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    if args.server_stop:
        try:
            if verbose:
                print(f"Stopping server at {args.host}:{args.port}...")
            
            stop((args.host, args.port))
            
            if verbose:
                print("Server stopped successfully.")
            
            return 0
        except Exception as e:
            print(f"\nError stopping server: {e}", file=sys.stderr)
            if verbose:
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
            if verbose:
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
    
    if verbose:
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
            write_candidate_files=args.write_candidate_files,
            ring_closure_threshold=args.ring_closure_tolerance
        )
        
        if verbose:
            print(f"  Atoms: {len(optimizer.system.elements)}")
            print(f"  Z-matrix size: {len(optimizer.system.zmatrix)}")
            print(f"  Rotatable dihedrals: {len(optimizer.system.rotatable_indices)}")
            if args.rcp_terms:
                print(f"  Ring closure tolerance: {args.ring_closure_tolerance} Å")
                print(f"  Ring closure decay rate: {args.ring_closure_decay_rate}")
            
            if args.minimize:
                # Minimization mode - show minimization parameters
                print(f"\nMinimization Mode")
                print(f"  Space: {args.space_type}")
                if args.smoothing:
                    if len(args.smoothing) > 1:
                        print(f"  Smoothing sequence: {args.smoothing}")
                    else:
                        print(f"  Smoothing: {args.smoothing[0]:.2f}")
                else:
                    print(f"  Smoothing: 0.0 (none)")
                print(f"  Max iterations: {args.max_iterations}")
                print(f"  Gradient tolerance: {args.gradient_tolerance}")
                if zmatrix_dof_bounds_per_type:
                    print(f"  Z-matrix DOF bounds per type: {zmatrix_dof_bounds_per_type}")
            else:
                # GA optimization mode - show GA parameters
                enable_smoothing = not args.no_smoothing_refinement
                enable_zmatrix = not args.no_zmatrix_refinement
                
                refinements = []
                if enable_smoothing:
                    refinements.append("Smoothing")
                if enable_zmatrix:
                    refinements.append("Z-matrix")
                algo_type = f"GA + {' + '.join(refinements)}" if refinements else "Pure GA"
                
                print(f"\nAlgorithm: {algo_type}")
                print(f"  Fitness: Exponential ring closure score")
                print(f"  Population: {args.population}")
                print(f"  Generations: {args.generations}")
                print(f"  Crossover: {args.crossover}")
                print(f"  Mutation: {args.mutation} (strength: {args.mutation_strength}°)")
                print(f"  Elite size: {args.elite_size}")
                print(f"  Torsion range: {args.torsion_min}° to {args.torsion_max}°")
                print(f"  Convergence threshold: {args.refinement_convergence}")
                if args.rcp_terms:
                    print(f"  Systematic sampling: {args.systematic_sampling_divisions} divisions for critical torsions")
                
                if enable_smoothing or enable_zmatrix:
                    print(f"\n  Refinement (after GA convergence):")
                    print(f"    Top N candidates: {args.refinement_top_n}")
                    if enable_smoothing:
                        print(f"    Smoothing: enabled ({args.torsional_iterations} iterations per step)")
                        print(f"      Sequence: {DEFAULT_SMOOTHING_SEQUENCE}")
                    if enable_zmatrix:
                        print(f"    Z-matrix: enabled ({args.zmatrix_iterations} iterations)")
            
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

            # Save optimized structure
            IOTools.save_structure_to_file(args.output, result['zmatrix'], result['final_energy'])
    
        else:
            # Run optimization (GA)
            result = optimizer.optimize(
                population_size=args.population,
                generations=args.generations,
                mutation_rate=args.mutation,
                mutation_strength=args.mutation_strength,
                crossover_rate=args.crossover,
                elite_size=args.elite_size,
                torsion_range=(args.torsion_min, args.torsion_max),
                ring_closure_tolerance=args.ring_closure_tolerance,
                ring_closure_decay_rate=args.ring_closure_decay_rate,
                convergence_interval=args.convergence_interval,
                enable_smoothing_refinement=not args.no_smoothing_refinement,
                enable_zmatrix_refinement=not args.no_zmatrix_refinement,
                refinement_top_n=args.refinement_top_n,
                smoothing_sequence=None,  # Use default
                torsional_iterations=args.torsional_iterations,
                zmatrix_iterations=args.zmatrix_iterations,
                refinement_convergence=args.refinement_convergence,
                systematic_sampling_divisions=args.systematic_sampling_divisions,
                print_interval=args.print_interval,
                verbose=verbose
            )
            
            # Print results
            if verbose:
                print("\n" + "=" * 70)
                print("RESULTS")
                print("=" * 70)
                print(f"Initial ring closure score: {result['initial_closure_score']:.4f}")
                
                # final_closure_score is now an array (one per top candidate)
                final_scores = result['final_closure_score']
                if isinstance(final_scores, (list, np.ndarray)):
                    if len(final_scores) == 1:
                        print(f"Final ring closure score:   {final_scores[0]:.4f}")
                    else:
                        print(f"Final ring closure scores (top {len(final_scores)}):")
                        for i, score in enumerate(final_scores):
                            print(f"  Candidate {i+1}: {score:.4f}")
                        print(f"Best score: {max(final_scores):.4f}")
                else:
                    print(f"Final ring closure score:   {final_scores:.4f}")
                
                improvement = max(final_scores) if isinstance(final_scores, (list, np.ndarray)) else final_scores
                improvement = improvement - result['initial_closure_score']
                print(f"Improvement:           {improvement:+.4f}")
                
                # Get best individual from top_candidates
                if optimizer.top_candidates and len(optimizer.top_candidates) > 0:
                    best_individual = optimizer.top_candidates[0]
                    print(f"\nOptimal dihedrals:")
                    for i, (idx, torsion) in enumerate(zip(optimizer.system.rotatable_indices, 
                                                           best_individual.torsions)):
                        atom = optimizer.system.zmatrix[idx]
                        element = optimizer.system.elements[idx]
                        print(f"  {i+1}. Atom {idx+1} ({element}): {torsion:6.1f}°")
                    
                    if best_individual.energy is not None:
                        print(f"\nFinal energy: {best_individual.energy:.4f} kcal/mol")
            
            # Save optimized structure
            IOTools.save_structure_to_file(args.output, optimizer.top_candidates[0].zmatrix, optimizer.top_candidates[0].energy)
    
        if verbose:
            structure_type = "Minimized" if args.minimize else "Optimized"
            print(f"\n{structure_type} structure saved to: {args.output}")
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

