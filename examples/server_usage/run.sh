#!/bin/bash
# Starts a server and sends a request using nc.
#
# Usage:
#   ./test_minimize_request.sh [host] [port]
#
# Example:
#   ./test_minimize_request.sh localhost 8080

HOST="${1:-localhost}"
INITIAL_PORT="${2:-59557}"

# Function to check if a port is in use
check_port_in_use() {
    local host=$1
    local port=$2
    # Try to connect to the port - if connection succeeds, port is in use
    # Use timeout to avoid hanging
    timeout 1 bash -c "echo >/dev/tcp/${host}/${port}" 2>/dev/null
    # Return 0 if port is in use (connection succeeded), 1 if available (connection failed)
    return $?
}

# Try to find an available port (up to 10 attempts)
PORT=$INITIAL_PORT
FOUND_PORT=false
for i in {1..10}; do
    if ! check_port_in_use "${HOST}" "${PORT}"; then
        # Port appears to be available (connection failed = nothing listening)
        FOUND_PORT=true
        break
    else
        # Port is in use (connection succeeded = something is listening)
        if [ $i -lt 10 ]; then
            echo "Port ${PORT} is in use, trying $((PORT + 1))..."
        fi
        PORT=$((PORT + 1))
    fi
done

if [ "$FOUND_PORT" = false ]; then
    echo "ERROR: Could not find an available port after 10 attempts (tried ${INITIAL_PORT} to $((INITIAL_PORT + 9)))"
    exit -1
fi

echo "Starting server on ${HOST}:${PORT}"

rc-optimizer --server-start --host "${HOST}" --port "${PORT}" > server.log 2>&1 &

sleep 3

if ! rc-optimizer --server-status --host "${HOST}" --port "${PORT}" | grep -q "RUNNING" ; then
    echo "ERROR: failed to start server on port ${PORT}"
    exit -1
else
    echo "Server started on ${HOST}:${PORT}!"
fi

# Send the request
cat request.json | nc "${HOST}" "${PORT}" > response.json

# Check outcome
if [ $? -ne 0 ]; then
    echo "ERROR: failed to submit request."
    exit -1
fi
if ! grep -q "\"STATUS\": \"SUCCESS\"" response.json ; then
    echo "ERROR: unsuccessful result (wrong status)."
    exit -1
fi
if ! grep -q "\"RCSCORE\": 0\.9" response.json ; then
    echo "ERROR: unsuccessful result (wrong RCSCORE)."
    exit -1
fi

# Stop the server
rc-optimizer --server-stop --host "${HOST}" --port "${PORT}" 

if [ $? -ne 0 ]; then
    echo "ERROR: failed to stop server"
    exit -1
fi
