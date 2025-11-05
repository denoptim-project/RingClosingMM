# Ring Closure Optimizer

[![Build and Test](https://github.com/your-username/RingClosingMM/actions/workflows/build_and_test_package.yml/badge.svg)](https://github.com/your-username/RingClosingMM/actions/workflows/build_and_test_package.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Overview

The Ring Closure Optimizer is a molecular modeling tool meant to identify ring-closing conformations. It build on an (OpenMM)[https://openmm.org/] molecular modeling engine and a dedicated force field definition that favors formation of bonds where no topological bond is originally defined, by introducing ring-closing (i.e., bond-forming) interactions and excluding inter atomic repulsion terms according expected presence of the to-be-formed bonds. With this force field that allows and favors bond formation, a conformational search can identify the conformation that bring the to-be-bonded atoms in a relative position compatible with definition of a formal bond and with running of further geometrical refinements.

## Features

- ✅ Force Field protecting bond length and bond angles: the input geometry is taken as the equilibrium geometry and strong force constants act as protection of the initial geometry.
- ✅ Fast exploration of torsional space with a genetic algorithm operating on the conformations of rotatable bonds. 
- ✅ Potential energy smoothing applied to search for global minimum in torsional allows fast identification of ring-closing conformations.
- ✅ Final geometrical refinement in Cartesian space to adjust bond angles/lengths to the ring-closing conformation.
- ✅ Socket TCP server to provide low-latency interface with any client application.

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
Next, see below for a [quick start](#quick-start) guide.

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

**Note**: All package configuration is in `pyproject.toml` (modern Python standard).


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
    --generations 50 \
    --population 30
```

### 2. Server

Start the server that will provide the ring-closing optimiization service:
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
    population_size=30,
    generations=50,
    enable_smoothing_refinement=True,
    enable_cartesian_refinement=True
)

# Save results
optimizer.save_optimized_structure('optimized.xyz')

# Result contains closure scores (final_closure_score is an array for top candidates)
import numpy as np
final_scores = result['final_closure_score']
best_score = max(final_scores) if isinstance(final_scores, (list, np.ndarray)) else final_scores
print(f"Final closure score: {best_score:.4f}")

# Energy is stored in top_candidates after optimization
if optimizer.top_candidates and optimizer.top_candidates[0].energy:
    print(f"Final energy: {optimizer.top_candidates[0].energy:.2f} kcal/mol")
```


## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## Acknowledgments

The Research Council of Norway for various kinds of funding.
