#!/bin/bash

set -e  # Exit on error

rm -f test_result.xyz optimization.log

echo "Running acyclic bond formation optimization..."
python ../../src/__main__.py -i test.int -r 1 2 2 6 8 9 9 10 10 11 -c 6 10 9 2 -o test_result.xyz > optimization.log 2>&1

# Check if output file was created
if [ ! -f test_result.xyz ]; then
    echo "Error: test_result.xyz not created"
    exit 1
fi

# Check if optimization completed successfully (look for final ring closure score)
if ! grep -q "Final ring closure score: .* 1\.0" optimization.log; then
    echo "Error: Optimization did not complete successfully"
    exit 1
fi

echo "Acyclic bond formation optimization completed successfully!"
