#!/usr/bin/env python3
"""
Example: Starting the socket server.

This script demonstrates how to start the ring closure optimizer server.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).resolve().parent.parent / 'src'
sys.path.insert(0, str(src_path))

from RCOServer import start, stop


def main():
    """Start the server and wait for termination."""
    print("=" * 70)
    print("Ring Closure Optimizer Server")
    print("=" * 70)
    
    # Start server
    host, port = start(host='localhost', port=0)
    
    print(f"\nServer started at {host}:{port}")
    print(f"Waiting for requests...")
    print(f"Send 'shutdown' message to stop the server\n")
    
    try:
        # Keep main thread alive
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        stop((host, port))
        print("Server stopped.")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

