# Ring Closure Optimizer

[![Build and Test](https://github.com/your-username/RingClosingMM/actions/workflows/build_and_test_package.yml/badge.svg)](https://github.com/your-username/RingClosingMM/actions/workflows/build_and_test_package.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Overview

The Ring Closure Optimizer is a molecular modeling tool meant to identify conformations that allow to close molecular ring, i.e., modify the geometry to bring the two ends of a chain in a relative position suitable to define a bond between them. It builds on an [OpenMM](https://openmm.org/) molecular mechanics engine and [scipy](https://scipy.org/) optimizers.
The molecular mechanics is based on force field definition designed for potential energy smoothing algorithms and that includes a ring-closing (i.e., bond-forming) attractive interaction operating over specific atom pairs (i.e., the ring-closing potential terms, RCP terms). Moreover, the inter atomic repulsion terms involving such pairs of to-be-bonded atoms are excluded as if those atoms were bonded, thus allowing those atoms to come close enough to define a bond between them. Other force field components introduce forces meant to protect as much as possible the initial geometry, while allowing for the minimal geometrical adaptation (some bond bending, and substantial bond torsion) that might be needed to access the ring-closing conformation and distribute the resulting strain over the formed ring. To retain generality, the force field uses the Universal Force Field parameters for the non-bonded interactions and fixed force constant for bond length and angle protection. Hence, the force field is not parametrized to reproduce any expected output, but only to exclude atom clashes while searching for ring-closing conformations.
The conformational search implements a divide and conquer strategy:
1. identify the torsions along the ring-closing chain that allow to bring the to-be-bonded atoms in the most suitable relative position to define the new bond,
2. remove strain by adapting any torsional degree of freedom other than those determining the ring-closing conformations,
3. refine the ring-closing conformations by tuning torsions and bond angles involved in the newly formed ring.

## Features

- ✅ Geometry-protective force field: bond lengths and angles in the input geometry are taken as the equilibrium value coupled with strong force constants for stretching and bending.
- ✅ Potential energy smoothing for global optimization in Z-matrix and Cartesian space.
- ✅ Energy minimization in Z-matrix space with selection of the degrees of freedom to vary and of their bounds.
- ✅ Socket TCP server to provide low-latency interface with any client application requesting the bond-formation/ring-closure service.

## Installation

### User Mode

Install with `Conda`:
```
conda install rc-optimizer
```
or `pip`:
```
pip install rc-optimizer
```
Next, see below for a (quick start)[#quick-start] guide.

### Development Mode

```bash
# Clone the repository
git clone <repository-url> <folder-name>
cd <folder-name>

# Option 1: Using conda (recommended for OpenMM)
conda env create -f environment.yml
conda activate rco_devel
# Package is automatically installed in development mode

# Option 2: Using pip only
pip install -e .[dev]  # Installs with development dependencies
```

After installation, you can use `rc-optimizer` from anywhere!

**Note**: All package configuration is in `pyproject.toml`.


## Quick Start

### 1. Command Line

After installation you can run this to get help
```bash
rc-optimizer -h
```

Here, is an example of an actual command for optimising a ring-closing conformation:
```bash
rc-optimizer \
    -i zmatrix.int \
    -r 1 2 3 4 \
    -c 4 5 6 7 \
    -o optimized.xyz \
    --verbose
```

For standalone energy minimization instead of global optimization:
```bash
rc-optimizer \
    -i zmatrix.int \
    -c 4 5 6 7 \
    --minimize \
    --space-type zmatrix \
    --smoothing 50.0 25.0 10.0 0.0 \
    --gradient-tolerance 0.01 \
    -o minimized.xyz
```

Available space types for minimization:
- `torsional`: changes only dihedral angles.
- `zmatrix`: changes bond lengths, angles, and dihedrals, possibly manually selected.
- `Cartesian`: changes Cartesian coordinate.

### 2. Server

Start the server that will provide the ring-closing optimization service:
```bash
rc-optimizer --server-start
```
The server will keep running on the terminal until you stop it, and you can now send any request to the port indicated in the log from the above command from any application.
For example, here is the content of a JSON file defining the request:
```json
{
  "zmatrix": [
    {"id": 1, "element": "C", "atomic_num": 6},
    {"id": 2, "element": "ATN", "atomic_num": 1, "bond_ref": 1, "bond_length": 1.54},
    {"id": 3, "element": "ATN", "atomic_num": 1, "bond_ref": 2, "bond_length": 2.50, "angle_ref": 1, "angle": 90.0},
    {"id": 4, "element": "C", "atomic_num": 6, "bond_ref": 3, "bond_length": 1.54, "angle_ref": 2, "angle": 180.0, "dihedral_ref": 1, "dihedral": 0.0, "chirality": 0}
  ],
  "bonds_data": [[0, 1, 1], [2, 3, 1]],
  "rcp_terms": [[1, 3], [2, 4]],
  "mode": "minimize",
  "torsional": false
}
```
that can be set from a terminal as follows:
```bash
echo  <path_to_json_file> | nc localhost <port>
```


### 3. Python API

```python
from RingClosureOptimizer import RingClosureOptimizer

# Create optimizer
optimizer = RingClosureOptimizer.from_files(
    structure_file='molecule.int',
    forcefield_file='data/RCP_UFFvdW.xml',
    rotatable_bonds=[(1, 2), (5, 31)],
    rcp_terms=[(7, 39), (77, 35)]
)

# Run optimization
result = optimizer.optimize(
    ...
)

# Save results to file
optimizer.save_optimized_structure('optimized.xyz')

# Or further process the numerical results
print(f"Final energy: {result['final_energy']:.2f} kcal/mol")
```


## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## Acknowledgments

The Research Council of Norway for various kinds of funding.
