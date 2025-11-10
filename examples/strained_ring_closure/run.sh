#!/bin/bash

set -e  # Exit on error

rm -f minimized_torsional.xyz torsional.log

python ../../src/__main__.py -i c4.int -c 1 11 2 14 --minimize-zmatrix -o c4_minimized_zmatrix.xyz > c4_zmatrix.log 2>&1

if [ ! -f c4_minimized_zmatrix.xyz ]; then
    echo "Error: c4_minimized_zmatrix.xyz not created"
    exit 1
fi

if ! tail -n 6 c4_zmatrix.log | grep -q "Ring closure:.*0\.9[0-9]" ; then
    echo "Error: Ring closure score not found or incorrect"
    exit 1
fi


