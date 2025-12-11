#!/bin/bash

set -e  # Exit on error

rm -f minimized_torsional.xyz torsional.log

function ensure_file_exists() {
    if [ ! -f $1 ]; then
        echo "Error: $1 not created"
        exit 1
    fi
}

rc-optimizer --optimize -i c4.int -c 1 11 2 14 -o c4_min --zmatrix-max-change-angle-bend 30.0 --zmatrix-max-step-angle-bend 30.0 > c4_min.log 2>&1

ensure_file_exists c4_min.xyz
ensure_file_exists c4_min.int

if ! tail -n 6 c4_min.log | grep -q "Final ring closure score:.*0\.9" ; then
    echo "Error: Ring closure score not found or incorrect for c4"
    exit 1
fi

n=$(diff c4.int c4_min.int  | grep -cE "^[<>]")
n=$((n/2))
if [ $n != 7 ]; then
    echo "Error: Expected 7 lines changed, found $n"
    exit 1
fi

echo "Distorted c4 chain closure completed successfully"

rc-optimizer --optimize -i c5.int -c 1 6 2 7 -o c5_min --zmatrix-max-change-angle-bend 15 --zmatrix-max-step-angle-bend 15 > c5_min.log 2>&1

ensure_file_exists c5_min.xyz
ensure_file_exists c5_min.int

if ! tail -n 6 c5_min.log | grep -q "Final ring closure score:.*0\.9[0-9]" ; then
    echo "Error: Ring closure score not found or incorrect for c5"
    exit 1
fi

n=$(diff c5.int c5_min.int  | grep -cE "^[<>]")
n=$((n/2))
if [ $n != 5 ]; then
    echo "Error: Expected 5 lines changed, found $n"
    exit 1
fi

echo "Distorted c5 chain closure completed successfully"

rc-optimizer --optimize -i distorted_complex.int -r 1 2 5 31 31 32 32 35 35 40 35 41 32 36 32 37 -c 7 39 77 35 -o distorted_complex_min  --zmatrix-max-change-angle-bend 15.0 --zmatrix-max-step-angle-bend 15.0  > distorted_complex_min.log 2>&1

ensure_file_exists distorted_complex_min.xyz
ensure_file_exists distorted_complex_min.int

# Twice the number of lines changed
n=$(diff distorted_complex.int distorted_complex_min.int  | grep -cE "^[<>]")
n=$((n/2))
if [ $n != 10 ]; then
    echo "Error: Expected 10 lines changed, found $n"
    exit 1
fi

if ! tail -n 10 distorted_complex_min.log | grep -q "Final ring closure score:.*0\.9[0-9]" ; then
    echo "Error: Ring closure score not found or incorrect for complex"
    exit 1
fi

echo "Distorted complex minimization completed successfully"


