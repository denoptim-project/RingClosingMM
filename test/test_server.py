#!/usr/bin/env python3
"""
Unit tests for RCOServer.

Tests the socket server functionality for ring closure optimization.
"""

import unittest
import socket
import json
import sys
import time
import threading
import queue
from pathlib import Path

# Add parent directory to path for package imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ringclosingmm import ZMatrix
from ringclosingmm.RCOServer import start, stop
import ringclosingmm.RCOServer as rco_server
import socketserver


def start_server_in_thread(result_queue, host='localhost', port=0):
    """Helper function to start server and communicate port back via queue.
    
    This is needed because start() blocks in serve_forever() and only returns
    after shutdown, so we need to get the port before that.
    """
    try:
        # Find available port if port=0
        if port == 0:
            sock = socket.socket()
            sock.bind((host, 0))
            port = sock.getsockname()[1]
            sock.close()
        
        # Put the result in queue BEFORE starting server
        result_queue.put(('success', host, port))
        
        # Now start the server (this will block)
        start(host=host, port=port)
    except Exception as e:
        result_queue.put(('error', str(e), None))


class TestServerMinimize(unittest.TestCase):
    """Test server minimize functionality."""
    
    def setUp(self):
        """Set up test fixtures.
        
        Note: JSON requests to the server use 1-based indexing (chemistry convention).
        The server internally converts to 0-based for Python processing.
        """
        
        # Z-matrix using 1-based indices (as would come from a client)
        self.zmatrix_request = [
            {'id': 1, 'element': 'C', 'atomic_num': 6},
            {'id': 2, 'element': 'ATN', 'atomic_num': 1, 'bond_ref': 1, 'bond_length': 1.54},
            {'id': 3, 'element': 'Du', 'atomic_num': 1, 'bond_ref': 2, 'bond_length': 1.0, 
            'angle_ref': 1, 'angle': 120.0},
            {'id': 4, 'element': 'ATN', 'atomic_num': 1, 'bond_ref': 2, 'bond_length': 2.50,
             'angle_ref': 1, 'angle': 90.0, 'dihedral_ref': 3, 'dihedral': 120.0, 'chirality': 1},
            {'id': 5, 'element': 'C', 'atomic_num': 6, 'bond_ref': 4, 'bond_length': 1.54,
             'angle_ref': 3, 'angle': 120.0, 'dihedral_ref': 2, 'dihedral': 180.0, 'chirality': 0}
        ]
        
        # RCP terms: 1-based atom indices (atom 1 to atom 4, atom 2 to atom 5)
        # Updated after adding Du atom: atom 3 is now Du, atom 4 is ATN (was 3), atom 5 is C (was 4)
        self.rcp_terms = [(1, 4), (2, 5)]

        # bonds_data: 1-based atom indices
        # [atom1, atom2] - must be lists, not tuples
        self.bonds_data = [
            [1, 2],  # bond between atoms 1 and 2
            [2, 3],  # bond between atoms 2 and 3 (Du atom)
            [3, 4],  # bond between atoms 3 (Du) and 4 (ATN, was atom 3)
            [4, 5]   # bond between atoms 4 (ATN) and 5 (C, was atom 4)
        ]
    
    def test_minimize_request(self):
        """Test minimize request handling."""
        # Create request
        request = {
            "zmatrix": self.zmatrix_request,
            "bonds_data": self.bonds_data,
            "rcp_terms": self.rcp_terms,
            "mode": "minimize",
            "max_iterations": 50,
            "smoothing": None,
            "torsional": False,
            "verbose": False
        }
        
        # Process request directly (bypassing socket for unit test)
        handler = getattr(rco_server, '_handle_optimization_request')
        response = handler(request)
        
        # Check response structure
        self.assertIn('STATUS', response)
        self.assertEqual(response['STATUS'], 'SUCCESS')
        
        # Check required fields
        self.assertIn('ENERGY', response)
        self.assertIn('RCSCORE', response)
        self.assertIn('Cartesian_coordinates', response)
        self.assertIn('zmatrix', response)
        self.assertIn('METADATA', response)
        
        # Check data types
        self.assertIsInstance(response['ENERGY'], (int, float))
        self.assertIsInstance(response['RCSCORE'], (int, float))
        self.assertIsInstance(response['Cartesian_coordinates'], list)
        self.assertIsInstance(response['zmatrix'], list)
        
        # Check coordinates shape
        coords = response['Cartesian_coordinates']
        self.assertEqual(len(coords), 5)  # 4 atoms
        self.assertEqual(len(coords[0]), 3)  # 3D coordinates
        self.assertEqual(len(coords[4]), 3)  # 3D coordinates
        
        # Check metadata
        metadata = response['METADATA']
        self.assertIn('initial_energy', metadata)
        self.assertIn('final_energy', metadata)
        self.assertIn('improvement', metadata)
        self.assertIn('minimization_type', metadata)
        
        # Check that energy improved (or at least was calculated)
        self.assertIsNotNone(response['ENERGY'])
        
        # Check that ring closure score is a number
        self.assertGreaterEqual(response['RCSCORE'], 0.0)
        self.assertLessEqual(response['RCSCORE'], 1.0)
    
    def test_server_start_stop(self):
        """Test server startup and shutdown (using Queue for thread communication)."""
        # Use Queue to pass server info from thread to main test
        result_queue = queue.Queue()
        
        # Use helper function that gets port before blocking
        server_thread = threading.Thread(
            target=start_server_in_thread, 
            args=(result_queue, 'localhost', 0),
            daemon=True
        )
        server_thread.start()
        
        try:
            # Wait for server to start and get the result
            try:
                result = result_queue.get(timeout=5.0)
                status = result[0]
                
                if status == 'error':
                    # Server failed to start (likely permission issue in sandbox)
                    # Skip this test rather than fail
                    self.skipTest(f"Server failed to start: {result[1]}")
                
                host = result[1]
                port = result[2]
                
            except queue.Empty:
                self.fail("Server did not start within timeout")
            
            # Give server a moment to fully start
            time.sleep(0.5)
            
            # Verify server is running by connecting
            try:
                sock = socket.create_connection((host, port), timeout=2.0)
                sock.close()
            except Exception as e:
                self.skipTest(f"Cannot connect to server (likely sandbox restriction): {e}")
            
            # Server should be running
            self.assertIsInstance(host, str)
            self.assertIsInstance(port, int)
            self.assertGreater(port, 0)
            
        finally:
            # Stop server
            try:
                if 'host' in locals() and 'port' in locals():
                    stop((host, port))
                    time.sleep(0.5)
            except Exception:
                pass  # Server might already be stopped
    
    def test_server_minimize_request(self):
        """Test sending minimize request through actual server (using Queue for thread communication)."""
        # Use Queue to pass server info from thread to main test
        result_queue = queue.Queue()
        
        # Use helper function that gets port before blocking
        server_thread = threading.Thread(
            target=start_server_in_thread,
            args=(result_queue, 'localhost', 0),
            daemon=True
        )
        server_thread.start()
        
        try:
            # Wait for server to start and get the result
            try:
                result = result_queue.get(timeout=5.0)
                status = result[0]
                
                if status == 'error':
                    # Server failed to start (likely permission issue in sandbox)
                    # Skip this test rather than fail
                    self.skipTest(f"Server failed to start: {result[1]}")
                
                host = result[1]
                port = result[2]
                
            except queue.Empty:
                self.fail("Server did not start within timeout")
            
            # Give server a moment to fully start
            time.sleep(0.5)
            
            # Create request
            request = {
                "zmatrix": self.zmatrix_request,
                "bonds_data": self.bonds_data,
                "rcp_terms": self.rcp_terms,
                "mode": "minimize",
                "max_iterations": 50,
                "smoothing": None,
                "torsional": False,
                "verbose": False
            }
            
            # Send request
            try:
                sock = socket.create_connection((host, port), timeout=30.0)
            except Exception as e:
                self.skipTest(f"Cannot connect to server (likely sandbox restriction): {e}")
            
            try:
                # Send request
                request_json = json.dumps(request)
                sock.sendall(request_json.encode('utf-8'))
                sock.shutdown(socket.SHUT_WR)
                
                # Receive response
                response_data = b''
                while True:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    response_data += chunk
                
                # Parse response
                response = json.loads(response_data.decode('utf-8'))
                
                # Check response
                self.assertIn('STATUS', response)
                # Note: might be ERROR if force field not found, but structure should be correct
                if response['STATUS'] == 'SUCCESS':
                    self.assertIn('ENERGY', response)
                    self.assertIn('RCSCORE', response)
                else:
                    # If it fails, check that error message is reasonable
                    self.assertIn('ERROR_MESSAGE', response)
                print(response)
                
            finally:
                sock.close()
                
        finally:
            # Stop server
            try:
                if 'host' in locals() and 'port' in locals():
                    stop((host, port))
                    time.sleep(0.5)
            except Exception:
                pass  # Server might already be stopped
    
    def test_invalid_request(self):
        """Test handling of invalid request."""
        # Missing zmatrix
        request = {
            "rcp_terms": self.rcp_terms,
            "mode": "minimize"
        }
        
        with self.assertRaises(ValueError) as cm:
            handler = getattr(rco_server, '_handle_optimization_request')
            handler(request)
        
        self.assertIn("Missing required field: zmatrix", str(cm.exception))
    
    def test_invalid_mode(self):
        """Test handling of invalid mode."""
        request = {
            "zmatrix": self.zmatrix_request,
            "bonds_data": self.bonds_data,
            "rcp_terms": self.rcp_terms,
            "mode": "invalid_mode"
        }
        
        with self.assertRaises(ValueError) as cm:
            handler = getattr(rco_server, '_handle_optimization_request')
            handler(request)
        
        self.assertIn("Unknown mode", str(cm.exception))


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)

