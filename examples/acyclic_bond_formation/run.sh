#!/bin/bash

set -e  # Exit on error

rm -f test_result.xyz optimization.log

echo "Running acyclic bond formation optimization..."
rc-optimizer -i test.int -c 5 14 6 13 --dof-indices 9 1 9 2 9 3 10 2 10 3 -o test_min.xyz --minimize --space-type Cartesian  > test_min.log 2>&1

# Check if output file was created
if [ ! -f test_min.xyz ]; then
    echo "Error: test_result.xyz not created"
    exit 1
fi

if ! grep -q "Final ring closure score: .* 0\.9" test_min.log; then
    echo "Error: minimization did not complete successfully"
    exit 1
fi

echo "Acyclic bond formation optimization completed successfully!"
