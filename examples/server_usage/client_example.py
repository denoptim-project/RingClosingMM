#!/usr/bin/env python3
"""
Example: Client for socket server.

This script demonstrates how to send requests to the ring closure optimizer server.
"""

import socket
import json
import sys

# Package should be installed, no need to modify sys.path
from ringclosingmm.CoordinateConversion import zmatrix_to_cartesian


def send_request(request: dict, host: str = 'localhost', port: int = 8080) -> dict:
    """Send request to server and return response.
    
    Parameters
    ----------
    request : dict
        JSON request dictionary
    host : str
        Server hostname
    port : int
        Server port
        
    Returns
    -------
    dict
        JSON response dictionary
    """
    # Connect to server
    sock = socket.create_connection((host, port), timeout=300.0)  # 5 minute timeout
    
    try:
        # Send request
        request_json = json.dumps(request)
        sock.sendall(request_json.encode('utf-8'))
        sock.shutdown(socket.SHUT_WR)  # Signal end of writing
        
        # Receive response
        response_data = b''
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            response_data += chunk
        
        # Parse response
        response = json.loads(response_data.decode('utf-8'))
        return response
        
    finally:
        sock.close()


def example_optimization_request():
    """Example: Send optimization request."""
    print("=" * 70)
    print("Example: Optimization Request")
    print("=" * 70)
    
    # Simple 3-atom Z-matrix for testing
    zmatrix = [
        {'id': 1, 'element': 'H', 'atomic_num': 1},
        {'id': 2, 'element': 'H', 'atomic_num': 1, 'bond_ref': 1, 'bond_length': 1.0},
        {'id': 3, 'element': 'H', 'atomic_num': 1, 'bond_ref': 1, 'bond_length': 1.0,
         'angle_ref': 2, 'angle': 109.47}
    ]
    
    request = {
        "zmatrix": zmatrix,
        # "forcefield_file": str(Path(__file__).parent.parent / 'data' / 'RCP_UFFvdW.xml'),  # Optional: server uses default
        "rotatable_bonds": None,  # All rotatable
        "rcp_terms": [(0, 2)],  # 0-based: first and third atom
        "mode": "optimize",
        "population_size": 20,
        "generations": 10,
        "verbose": False
    }
    
    print("\nSending request...")
    print(f"  Mode: {request['mode']}")
    print(f"  Atoms: {len(zmatrix)}")
    print(f"  RCP terms: {len(request['rcp_terms'])}")
    
    response = send_request(request, host='localhost', port=8080)
    
    print("\nResponse:")
    print(f"  STATUS: {response.get('STATUS')}")
    
    if response.get('STATUS') == 'SUCCESS':
        print(f"  Ring closure score: {response.get('RCSCORE', 'N/A'):.4f}")
        print(f"  Energy: {response.get('ENERGY', 'N/A'):.2f} kcal/mol")
        print(f"  Generations: {response.get('GENERATIONS', 'N/A')}")
        if 'coordinates' in response and response['coordinates']:
            print(f"  Coordinates: {len(response['coordinates'])} atoms")
    else:
        print(f"  ERROR: {response.get('ERROR_MESSAGE', 'Unknown error')}")
    
    return response


def example_minimization_request():
    """Example: Send minimization request."""
    print("\n" + "=" * 70)
    print("Example: Minimization Request")
    print("=" * 70)
    
    # Simple 3-atom Z-matrix
    zmatrix = [
        {'id': 1, 'element': 'H', 'atomic_num': 1},
        {'id': 2, 'element': 'H', 'atomic_num': 1, 'bond_ref': 1, 'bond_length': 1.0},
        {'id': 3, 'element': 'H', 'atomic_num': 1, 'bond_ref': 1, 'bond_length': 1.0,
         'angle_ref': 2, 'angle': 109.47}
    ]
    
    request = {
        "zmatrix": zmatrix,
        # "forcefield_file": str(Path(__file__).parent.parent / 'data' / 'RCP_UFFvdW.xml'),  # Optional: server uses default
        "rcp_terms": [(0, 2)],
        "mode": "minimize",
        "max_iterations": 100,
        "smoothing": None,
        "torsional": False
    }
    
    print("\nSending request...")
    print(f"  Mode: {request['mode']}")
    
    response = send_request(request, host='localhost', port=8080)
    
    print("\nResponse:")
    print(f"  STATUS: {response.get('STATUS')}")
    
    if response.get('STATUS') == 'SUCCESS':
        print(f"  Ring closure score: {response.get('RCSCORE', 'N/A'):.4f}")
        print(f"  Energy: {response.get('ENERGY', 'N/A'):.2f} kcal/mol")
        print(f"  Improvement: {response.get('IMPROVEMENT', 'N/A'):.2f} kcal/mol")
    
    return response


def main():
    """Run examples."""
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    else:
        port = 8080
    
    print("\nRing Closure Optimizer Client Example")
    print(f"Connecting to server at localhost:{port}\n")
    
    try:
        # Try optimization request
        example_optimization_request()
        
        # Try minimization request
        # example_minimization_request()
        
    except ConnectionRefusedError:
        print(f"\nError: Could not connect to server at localhost:{port}")
        print("Make sure the server is running:")
        print("  python examples/server_example.py")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

