#!/bin/bash

set -e  # Exit on error

rm -f minimized_torsional.xyz torsional.log

function ensure_file_exists() {
    if [ ! -f $1 ]; then
        echo "Error: $1 not created"
        exit 1
    fi
}

python ../../src/__main__.py -i c4.int -c 1 11 2 14 --minimize --space-type zmatrix -o c4_min > c4_min.log 2>&1

ensure_file_exists c4_min.xyz
ensure_file_exists c4_min.int

if ! tail -n 6 c4_min.log | grep -q "Ring closure:.*0\.9[0-9]" ; then
    echo "Error: Ring closure score not found or incorrect"
    exit 1
fi

n=$(diff c4.int c4_min.int  | grep -cE "^[<>]")
n=$((n/2))
if [ $n != 5 ]; then
    echo "Error: Expected 5 lines changed, found $n"
    exit 1
fi


echo "Distorted c4 chain closure completed successfully"

python ../../src/__main__.py -i distorted_complex.int -r 1 2 5 31 31 32 32 35 35 40 35 41 32 36 32 37 -c 7 39 77 35 -o distorted_complex_min --minimize --space-type zmatrix > distorted_complex_min.log 2>&1

ensure_file_exists distorted_complex_min.xyz
ensure_file_exists distorted_complex_min.int

# Twice the number of lines changed
n=$(diff distorted_complex.int distorted_complex_min.int  | grep -cE "^[<>]")
n=$((n/2))
if [ $n != 8 ]; then
    echo "Error: Expected 8 lines changed, found $n"
    exit 1
fi

if ! tail -n 10 distorted_complex_min.log | grep -q "Ring closure:.*0\.9[0-9]" ; then
    echo "Error: Ring closure score not found or incorrect"
    exit 1
fi

echo "Distorted complex minimization completed successfully"


