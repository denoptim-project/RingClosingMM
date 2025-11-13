#!/bin/bash
# Starts a server and sends a request using nc.
#
# Usage:
#   ./test_minimize_request.sh [host] [port]
#
# Example:
#   ./test_minimize_request.sh localhost 8080

HOST="${1:-localhost}"
PORT="${2:-59557}"

rc-optimizer --server-start --host "${HOST}" --port "${PORT}" > server.log 2>&1 &

sleep 3

if ! rc-optimizer --server-status --host "${HOST}" --port "${PORT}" | grep -q "RUNNING" ; then
    echo "ERROR: failed to start server"
    exit -1
else
    echo "Server started!"
fi

# Send the request
cat request.json | nc "${HOST}" "${PORT}" > response.json

# Check outcome
if [ $? -ne 0 ]; then
    echo "ERROR: failed to submit request."
    exit -1
fi
if ! grep -q "\"STATUS\": \"SUCCESS\"" response.json ; then
    echo "ERROR: unsuccessful result."
    exit -1
fi
if ! grep -q "\"RCSCORE\": 0\.9" response.json ; then
    echo "ERROR: unsuccessful result."
    exit -1
fi

# Stop the server
rc-optimizer --server-stop --host "${HOST}" --port "${PORT}" 

if [ $? -ne 0 ]; then
    echo "ERROR: failed to stop server"
    exit -1
fi
