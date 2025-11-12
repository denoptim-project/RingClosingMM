#!/bin/bash

set -e  # Exit on error

rm -f minimized_torsional.xyz torsional.log

python ../../src/__main__.py -i test.int -c 7 39 77 35 -r 1 2 5 31 31 32 32 35 35 40 35 41 32 36 32 37 --minimize --space-type torsional --smoothing 5.0 2.5 1.0 0.0 --max-iterations 100 -o minimized_torsional.xyz > torsional.log 2>&1

if [ ! -f minimized_torsional.xyz ]; then
    echo "Error: minimized_torsional.xyz not created"
    exit 1
fi

if ! tail -n 6 torsional.log | grep -q "Ring closure:.* 0\.9[0-9]" ; then
    echo "Error: Ring closure score not found or incorrect"
    exit 1
fi

smoothing_count=$(grep -c "Step.*smoothing" torsional.log || true)
if [ "$smoothing_count" -ne 4 ]; then
    echo "Error: Expected 4 smoothing steps, found $smoothing_count"
    exit 1
fi

echo "Torsional minimization completed successfully"

rm -f minimized_cartesian.xyz cartesian.log

python ../../src/__main__.py -i test.int -c 7 39 77 35 --minimize --space-type Cartesian --smoothing 5.0 2.5 1.0 0.0 --max-iterations 500 -o minimized_cartesian.xyz > cartesian.log 2>&1

if [ ! -f minimized_cartesian.xyz ]; then
    echo "Error: minimized_cartesian.xyz not created"
    exit 1
fi

if ! tail -n 6 cartesian.log | grep -q "Ring closure:.*[0-9]\+\.[0-9]\+" ; then
    echo "Error: Ring closure score not found or incorrect"
    exit 1
fi

smoothing_count=$(grep -c "Step.*smoothing" cartesian.log || true)
if [ "$smoothing_count" -ne 4 ]; then
    echo "Error: Expected 4 smoothing steps, found $smoothing_count"
    exit 1
fi

echo "Cartesian minimization completed successfully"
echo "All minimizations completed successfully!"

