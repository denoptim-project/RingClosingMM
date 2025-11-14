#!/bin/bash

set -e  # Exit on error

rm -f test_result.xyz optimization.log

echo "Running polycyclic bond formation optimization..."
rc-optimizer -i test.int -r 1 3 3 5 5 7 11 13 17 6 -c 1 12 2 11 13 23 14 21 22 18 17 26 -o test_opt.xyz > test_opt.log 2>&1

# Check if output file was created
if [ ! -f test_opt.xyz ]; then
    echo "Error: test_result.xyz not created"
    exit 1
fi

# Check if optimization completed successfully (look for final ring closure score)
if ! grep -q "Final ring closure score: .* 0\.[8-9]" test_opt.log; then
    echo "Error: Optimization did not complete successfully"
    exit 1
fi

echo "Polycyclic bond formation optimization completed successfully!"
