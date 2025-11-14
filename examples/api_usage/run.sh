#!/bin/bash

set -e  # Exit on error

rm -f optimized.xyz api_usage.log

echo "Running API usage examples..."

# Run the Python API example script
python api_usage_example.py > api_usage.log 2>&1

# Check if the script completed successfully
if [ $? -ne 0 ]; then
    echo "Error: API usage example script failed"
    exit 1
fi

# Check if expected output files were created
if [ ! -f optimized.xyz ]; then
    echo "Error: optimized.xyz not created"
    exit 1
fi

# Check if optimization completed successfully (look for success messages)
if ! grep -q "✅.*completed successfully" api_usage.log; then
    echo "Error: Examples did not complete successfully"
    exit 1
fi

# Check if all examples ran (look for all example headers)
if ! grep -q "Example: Simple Usage" api_usage.log || \
   ! grep -q "Example: Accessing Components" api_usage.log; then
    echo "Error: Not all examples were executed"
    exit 1
fi

echo "API usage examples completed successfully!"

