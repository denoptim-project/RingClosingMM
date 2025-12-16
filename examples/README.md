# RingClosureOptimizer Python API Examples

This directory contains example demonstrating how to use the RingClosureOptimizer module programmatically.

All examples include a `run.sh` script that uses the `rc-optimizer` command. The package must be installed (via conda: `conda install ringclosingmm`, or in editable mode for development within a conda environment) for the `rc-optimizer` command to be available.

Each `run.sh` includes a minimal evaluation of the outcome meant to verify the expected behaviour. To run all the examples do the following:

```bash
cd examples
./run_all_examples.sh
```

> [!WARNING]
> 1-based indexing is expected for any input to the command line interface or the socket server interface. This allows to use input files generated with [Tinker](https://dasher.wustl.edu/tinker/).  

### Run from Python Interpreter

> [!WARNING]
> 0-based indexing is expected for any data living within the Python interpreter, while the content of the input files is expected to use 1-based indexing.

```python
# Once installed, simply use:
from ringclosingmm import RingClosureOptimizer

# Or for development (before installation):
# import sys
# sys.path.insert(0, '../ringclosingmm')
# from RingClosureOptimizer import RingClosureOptimizer

# rotatable_bonds and rcp_terms are now direct lists of tuples (0-based indices)
optimizer = RingClosureOptimizer.from_files(
    structure_file='../test/cyclic_bond_formation/test.int', # contains 1-based indexes
    forcefield_file='../data/RCP_UFFvdW.xml',
    rotatable_bonds=[(0, 1), (4, 30), (30, 31)],  # 0-based indexed atom pairs
    rcp_terms=[(6, 38), (76, 34)]  # 0-based indexed atom pairs
)

result = optimizer.optimize()

rc_scores = result['final_ring_closure_score']
print(f"Final ring closure score: {rc_scores:.4f}")
```
