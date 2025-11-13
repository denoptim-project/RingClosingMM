#!/usr/bin/env python3
"""
Socket server for Ring Closure Optimizer.

This module provides a socket server that accepts JSON-formatted requests
and returns optimized conformations using RingClosureOptimizer.

Usage:
    from RCOServer import start, stop
    
    host, port = start(host='localhost', port=0)
    # ... use server ...
    stop((host, port))
"""

import socket
import sys
import json
import socketserver
import time
import signal
from typing import Tuple, Any, Dict, Union, Optional
from pathlib import Path
import logging

# Note: Using pathlib for resource access, which works in both development and installed packages

# Import RingClosureOptimizer and dependencies
try:
    from .MolecularSystem import MolecularSystem
    from .RingClosureOptimizer import RingClosureOptimizer
    from .CoordinateConverter import zmatrix_to_cartesian
    from .ZMatrix import ZMatrix
except ImportError:
    from MolecularSystem import MolecularSystem
    from RingClosureOptimizer import RingClosureOptimizer
    from CoordinateConverter import zmatrix_to_cartesian
    from ZMatrix import ZMatrix

MY_NAME = "rc-optimizer-server"

# Server startup timeout
SERVER_START_MAX_TIME = 5  # seconds

# Default force field file name
DEFAULT_FORCEFIELD_FILE = "RCP_UFFvdW.xml"

# Global server instance (for shutdown from signal handlers)
_server_instance = None


def get_default_forcefield_path() -> Optional[str]:
    """Get the path to the default force field file.
    
    Tries multiple methods to locate the force field file:
    1. Relative to this file's directory (development mode)
    2. Relative to package installation directory (installed package)
    3. Relative to current working directory
    4. In site-packages data directory
    
    Returns
    -------
    Optional[str]
        Path to force field file, or None if not found
    """
    # Method 1: Try relative to this file's directory (development mode)
    # This works when running from source: src/RCOServer.py -> ../data/RCP_UFFvdW.xml
    try:
        src_dir = Path(__file__).parent.parent
        data_file = src_dir / 'data' / DEFAULT_FORCEFIELD_FILE
        if data_file.exists():
            return str(data_file.resolve())
    except Exception:
        pass
    
    # Method 2: Try relative to package installation (installed package)
    # When installed, data files should be in the package root
    try:
        # Try to find the package root
        import sys
        package_root = Path(sys.prefix) / 'share' / 'ringclosingmm' / 'data'
        data_file = package_root / DEFAULT_FORCEFIELD_FILE
        if data_file.exists():
            return str(data_file.resolve())
    except Exception:
        pass
    
    # Method 3: Try relative to current working directory
    try:
        cwd_file = Path.cwd() / 'data' / DEFAULT_FORCEFIELD_FILE
        if cwd_file.exists():
            return str(cwd_file.resolve())
    except Exception:
        pass
    
    # Method 4: Try to find in site-packages data directory
    try:
        import site
        for site_dir in site.getsitepackages():
            # Try various possible locations
            for subdir in ['', 'data', 'ringclosingmm/data']:
                if subdir:
                    data_file = Path(site_dir) / subdir / DEFAULT_FORCEFIELD_FILE
                else:
                    data_file = Path(site_dir) / DEFAULT_FORCEFIELD_FILE
                if data_file.exists():
                    return str(data_file.resolve())
    except Exception:
        pass
    
    return None

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
log_stream_handler = logging.StreamHandler(sys.stdout)
log_stream_handler.setLevel(logging.DEBUG)
log_formatter = logging.Formatter(
    "%(name)s %(asctime)s %(levelname)s %(message)s")
log_stream_handler.setFormatter(log_formatter)
logger.addHandler(log_stream_handler)


class ServerError(Exception):
    """Exception for server errors with JSON-formatted error message."""
    
    def __init__(self, message: str, error_code: str = "SERVER_ERROR"):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.json_error = {
            "STATUS": "ERROR",
            "ERROR_CODE": error_code,
            "ERROR_MESSAGE": f"#{MY_NAME}: {message}",
            "ERROR_DETAILS": {}
        }


def start(host: Union[str, Tuple[str, int]] = 'localhost', port: int = 0) -> Tuple[str, int]:
    """Start server in the main thread (blocks until server stops).
    
    The server always uses RingClosureOptimizer to process requests.
    This function blocks and runs the server until it is stopped.
    
    Parameters
    ----------
    host : str or Tuple[str, int]
        Hostname (default: 'localhost') or address tuple (host, port)
    port : int
        Port number (default: 0 = find available port).
        Ignored if host is a tuple.
        
    Returns
    -------
    Tuple[str, int]
        (host, port) where server was running (returns after server stops)
        
    Raises
    ------
    Exception
        If server fails to start
    """
    global _server_instance
    
    # Handle tuple address
    if isinstance(host, tuple):
        host, port = host[0], host[1]
    
    # Find available port if port=0
    if port == 0:
        logger.debug('Searching for an available port')
        sock = socket.socket()
        sock.bind((host, 0))
        port = sock.getsockname()[1]
        logger.debug(f'Port {port} is available')
        sock.close()
    
    print(f"Starting server at {host}:{port}")
    print(f"Use 'rc-optimizer --server-stop --host {host} --port {port}' to stop the server or press Ctrl+C in the terminal.")
    
    # Create and run server in main thread
    server = socketserver.ThreadingTCPServer(
        (host, port),
        OptimizationRequestHandler
    )
    
    # Store server instance globally for shutdown from signal handlers
    _server_instance = server
    
    try:
        # Run server (blocks until shutdown)
        # Use poll_interval to make shutdown more responsive
        # The default SIGINT handler raises KeyboardInterrupt which breaks out of serve_forever()
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        # Handle Ctrl+C - KeyboardInterrupt is raised by default SIGINT handler
        logger.debug("KeyboardInterrupt received, shutting down server...")
        print("\nShutting down server...")
        # Explicitly shutdown to ensure clean exit
        server.shutdown()
    finally:
        # Clean up - ensure server is fully stopped
        _server_instance = None
        if hasattr(server, 'server_close'):
            server.server_close()
        logger.debug(f"Server stopped at {host}:{port}")
    
    return host, port


class OptimizationRequestHandler(socketserver.StreamRequestHandler):
    """Request handler for Ring Closure Optimizer requests."""
    
    def handle(self):
        try:
            # Read message from socket (UTF-8)
            message = self.rfile.read().decode('utf8')
            logger.debug(f"Handling request: {message[:200]}...")  # Log first 200 chars
            
            # Check for shutdown message
            if 'shutdown' in message.lower():
                logger.debug("Shutting down server upon request")
                self.server.shutdown()
                return
            
            if message == '':
                logger.debug("Received empty request")
                return
            
            # Parse JSON request
            try:
                json_msg = json.loads(message)
            except json.decoder.JSONDecodeError as e:
                error_response = {
                    "STATUS": "ERROR",
                    "ERROR_CODE": "INVALID_JSON",
                    "ERROR_MESSAGE": f"Invalid JSON: {str(e)}",
                    "ERROR_DETAILS": {}
                }
                answer = json.dumps(error_response) + '\n'
                self.wfile.write(answer.encode('utf8'))
                return
            
        # Process request using RingClosureOptimizer
            try:
            # Call the handler function directly (it's in the same module)
                response = _handle_optimization_request(json_msg)
                logger.debug(f"Response STATUS: {response.get('STATUS', 'UNKNOWN')}")
            except Exception as e:
                logger.error(f"Error processing request: {e}", exc_info=True)
                response = {
                    "STATUS": "ERROR",
                    "ERROR_CODE": "PROCESSING_ERROR",
                    "ERROR_MESSAGE": str(e),
                    "ERROR_DETAILS": {}
                }
            
            # Send response
            answer = json.dumps(response)
            logger.debug(f"Sending response: {answer[:200]}...")  # Log first 200 chars
            answer += '\n'
            self.wfile.write(answer.encode('utf8'))
            
        except Exception as e:
            logger.error(f"Unexpected error in handler: {e}", exc_info=True)
            try:
                error_response = {
                    "STATUS": "ERROR",
                    "ERROR_CODE": "SERVER_ERROR",
                    "ERROR_MESSAGE": f"Unexpected server error: {str(e)}",
                    "ERROR_DETAILS": {}
                }
                answer = json.dumps(error_response) + '\n'
                self.wfile.write(answer.encode('utf8'))
            except Exception:
                pass  # Can't send response if connection is broken
    

def _handle_optimization_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """Handle optimization request using RingClosureOptimizer.
    
    Parameters
    ----------
    request : Dict[str, Any]
        JSON request containing zmatrix, bonds_data, and other optional 
        parameters: rotatable_bonds, rcp_terms, etc. All are expected to 
        use 1-based indexing.
        
    Returns
    -------
    Dict[str, Any]
        JSON response with STATUS, ENERGY, RCSCORE, coordinates, etc.
    """
    # Extract required fields
    zmatrix = request.get('zmatrix')
    if not zmatrix:
        raise ValueError("Missing required field: zmatrix")
    
    # Convert Z-matrix reference indices from 1-based (JSON input) to 0-based (internal)
    # The Z-matrix from JSON uses 1-based indexing for all reference atoms
    zmatrix_0based = []
    for atom in zmatrix:
        atom_0based = atom.copy() 
        # Convert reference indices from 1-based to 0-based
        atom_0based['id'] = atom_0based['id'] - 1
        if 'bond_ref' in atom_0based:
            atom_0based['bond_ref'] = atom_0based['bond_ref'] - 1
        if 'angle_ref' in atom_0based:
            atom_0based['angle_ref'] = atom_0based['angle_ref'] - 1
        if 'dihedral_ref' in atom_0based:
            atom_0based['dihedral_ref'] = atom_0based['dihedral_ref'] - 1
        zmatrix_0based.append(atom_0based)

    # List of atom indices that are bonded to each other and the bond type
    # This is needed because the Z-matrix does not imply bond definition
    bonds_data = request.get('bonds_data') # 1-based indexing
    if not bonds_data:
        raise ValueError("Missing required field: bonds_data")
    bonds_data = [(a-1, b-1, c) for a, b, c in bonds_data] # 0-based indexing

    # Convert to ZMatrix immediately - this is the only boundary where we convert List[Dict] to ZMatrix
    zmatrix = ZMatrix(zmatrix_0based, bonds_data)
    
    # Get forcefield file (use default if not provided)
    forcefield_file = request.get('forcefield_file')
    if not forcefield_file:
        # Try to get default force field
        forcefield_file = get_default_forcefield_path()
        if not forcefield_file:
            raise ValueError(
                "Missing required field: forcefield_file. "
                "Please provide forcefield_file in the request or ensure "
                "the default force field file is available."
            )
        logger.debug(f"Using default force field: {forcefield_file}")
    
    mode = request.get('mode', 'optimize')  # 'optimize' or 'minimize'

    # Input about defining bonds that can be rotated
    rotatable_bonds = request.get('rotatable_bonds')  # 1-based or None
    if rotatable_bonds is not None:
        rotatable_bonds = [(a-1, b-1) for a, b in rotatable_bonds] # 0-based indexing
    
    # Definition of ring-closing pairs,if any
    rcp_terms = request.get('rcp_terms', [])  # 1-based
    if rcp_terms:
        rcp_terms = [(a-1, b-1) for a, b in rcp_terms] # 0-based indexing
        
    # Create MolecularSystem from Z-matrix
    try:
        system = MolecularSystem.from_data(
            zmatrix=zmatrix,
            forcefield_file=forcefield_file,
            rcp_terms=rcp_terms if rcp_terms else None
        )
    except Exception as e:
        raise ValueError(f"Failed to create MolecularSystem: {str(e)}")
    
    # Determine rotatable indices
    if rotatable_bonds is None:
        # Get all rotatable indices (all non-chirality-constrained dihedrals)
        rotatable_indices = MolecularSystem._get_all_rotatable_indices(zmatrix)
    else:
        # Convert bond pairs to Z-matrix indices
        rotatable_indices = RingClosureOptimizer._convert_bonds_to_indices(rotatable_bonds, zmatrix)
    
    # Create RingClosureOptimizer directly
    try:
        optimizer = RingClosureOptimizer(
            molecular_system=system,
            rotatable_indices=rotatable_indices
        )
    except Exception as e:
        raise ValueError(f"Failed to create RingClosureOptimizer: {str(e)}")
    
    # Process based on mode
    if mode == 'optimize':
        # Extract optimization parameters
        result = optimizer.optimize(
            ring_closure_tolerance=request.get('ring_closure_tolerance', 0.1),
            ring_closure_decay_rate=request.get('ring_closure_decay_rate', 0.5),
            enable_pssrot_refinement=request.get('enable_pssrot_refinement', True),
            enable_zmatrix_refinement=request.get('enable_zmatrix_refinement', True),
            smoothing_sequence=request.get('smoothing_sequence'),
            torsional_iterations=request.get('torsional_iterations', 50),
            zmatrix_iterations=request.get('zmatrix_iterations', 50),
            gradient_tolerance=request.get('gradient_tolerance', 0.01),
            verbose=request.get('verbose', False)
        )
        
        # Get best result from optimize()
        best_zmatrix = result['final_zmatrix']
        best_coords = zmatrix_to_cartesian(best_zmatrix)
        final_energy = result['final_energy']
        final_rcscore = result['final_closure_score']
        if isinstance(final_rcscore, (list, tuple)):
            final_rcscore = max(final_rcscore) if final_rcscore else 0.0
        
        # Convert Z-matrix back to 1-based for JSON response
        zmatrix_1based = []
        # Convert ZMatrix to list if needed
        if isinstance(best_zmatrix, ZMatrix):
            zmatrix_list = best_zmatrix.to_list()
        else:
            zmatrix_list = best_zmatrix
        
        for atom in zmatrix_list:
            atom_1based = atom.copy()
            # Convert reference indices from 0-based to 1-based
            if 'bond_ref' in atom_1based:
                atom_1based['bond_ref'] = atom_1based['bond_ref'] + 1
            if 'angle_ref' in atom_1based:
                atom_1based['angle_ref'] = atom_1based['angle_ref'] + 1
            if 'dihedral_ref' in atom_1based:
                atom_1based['dihedral_ref'] = atom_1based['dihedral_ref'] + 1
            zmatrix_1based.append(atom_1based)
        
        response = {
            "STATUS": "SUCCESS",
            "ENERGY": float(final_energy) if final_energy is not None else None,
            "RCSCORE": float(final_rcscore),
            "Cartesian_coordinates": best_coords.tolist(),
            "zmatrix": zmatrix_1based,
            "METADATA": {
                "initial_closure_score": result.get('initial_closure_score', 0.0),
                "final_closure_score": final_rcscore,
                "generations": result.get('generations', 0)
            }
        }
        
    elif mode == 'minimize':
        # Extract minimization parameters
        smoothing = request.get('smoothing')
        if isinstance(smoothing, list):
            smoothing_sequence = smoothing
        elif smoothing is not None:
            smoothing_sequence = [smoothing]
        else:
            smoothing_sequence = None
        
        result = optimizer.minimize(
            max_iterations=request.get('max_iterations', 500),
            smoothing=smoothing_sequence,
            space_type=request.get('space_type', 'Cartesian'),
            gradient_tolerance=request.get('gradient_tolerance', 0.01),
            verbose=request.get('verbose', False)
        )
        
        # Get final energy and ring closure score
        final_energy = result['final_energy']
        final_rcscore = result['final_ring_closure_score']
        
        # Convert Z-matrix back to 1-based for JSON response
        zmatrix_1based = []
        zmatrix_internal = result['zmatrix']
        # Convert ZMatrix to list if needed
        if isinstance(zmatrix_internal, ZMatrix):
            zmatrix_list = zmatrix_internal.to_list()
        else:
            zmatrix_list = zmatrix_internal
        
        for atom in zmatrix_list:
            atom_1based = atom.copy()
            # Convert reference indices from 0-based to 1-based
            if 'bond_ref' in atom_1based:
                atom_1based['bond_ref'] = atom_1based['bond_ref'] + 1
            if 'angle_ref' in atom_1based:
                atom_1based['angle_ref'] = atom_1based['angle_ref'] + 1
            if 'dihedral_ref' in atom_1based:
                atom_1based['dihedral_ref'] = atom_1based['dihedral_ref'] + 1
            zmatrix_1based.append(atom_1based)
        
        # Compute Cartesian coordinates from the final Z-matrix to ensure consistency
        # Convert ZMatrix to list if needed for zmatrix_to_cartesian
        if isinstance(zmatrix_internal, ZMatrix):
            zmatrix_for_coords = zmatrix_internal
        else:
            zmatrix_for_coords = ZMatrix(zmatrix_internal, [])
        final_coords = zmatrix_to_cartesian(zmatrix_for_coords)
        
        response = {
            "STATUS": "SUCCESS",
            "ENERGY": float(final_energy),
            "RCSCORE": float(final_rcscore),
            "Cartesian_coordinates": final_coords.tolist(),
            "zmatrix": zmatrix_1based,
            "METADATA": {
                "initial_energy": result.get('initial_energy', 0.0),
                "final_energy": final_energy,
                "improvement": result.get('improvement', 0.0),
                "minimization_type": result.get('minimization_type', 'cartesian')
            }
        }
        
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'optimize' or 'minimize'")
    
    return response


def stop(address: Tuple[str, int]):
    """Send shutdown request to the server.
    
    Parameters
    ----------
    address : Tuple[str, int]
        (host, port) of server to stop
        
    Raises
    ------
    Exception
        If cannot communicate with server
    """
    try:
        socket_connection = socket.create_connection(address, timeout=2.0)
        socket_connection.send('shutdown'.encode('utf8'))
        socket_connection.close()
    except Exception as e:
        raise Exception(f'Could not communicate with socket server at {address}: {e}')


def status(address: Tuple[str, int]) -> Dict[str, Any]:
    """Check server status by attempting to connect.
    
    Parameters
    ----------
    address : Tuple[str, int]
        (host, port) of server to check
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with status information:
        - 'running': bool, whether server is responding
        - 'host': str, hostname
        - 'port': int, port number
        - 'error': str or None, error message if connection failed
        
    Note
    ----
    This function does not send a request, it only checks if the server
    is accepting connections.
    """
    host, port = address
    try:
        sock = socket.create_connection(address, timeout=2.0)
        sock.close()
        return {
            'running': True,
            'host': host,
            'port': port,
            'error': None
        }
    except Exception as e:
        return {
            'running': False,
            'host': host,
            'port': port,
            'error': str(e)
        }

