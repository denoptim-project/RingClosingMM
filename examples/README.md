# RingClosureOptimizer Python API Examples

This directory contains example scripts demonstrating how to use the RingClosureOptimizer module programmatically.

## Available Examples

### api_usage_example.py

Comprehensive examples showing different usage patterns:

1. **Simple Usage** - Basic high-level API
2. **Custom Parameters** - Aggressive optimization with custom settings
3. **Pure GA** - Genetic algorithm without local refinement
4. **Component Access** - Accessing individual system components

## Running the Examples

### Prerequisites

Make sure you have:
- OpenMM installed
- The molecular system files in `test/cyclic_bond_formation/` or `test/acyclic_bond_formation/`
- Python 3.7+

### Run All Examples

```bash
cd examples
python api_usage_example.py
```

### Run from Python Interpreter

```python
import sys
sys.path.insert(0, '../src')

from RingClosureOptimizer import RingClosureOptimizer

# rotatable_bonds and rcp_terms are now direct lists of tuples (0-based indices)
optimizer = RingClosureOptimizer.from_files(
    structure_file='../test/cyclic_bond_formation/test.int',
    forcefield_file='../data/RCP_UFFvdW.xml',
    rotatable_bonds=[(0, 1), (4, 30), (30, 31)],  # 0-based atom pairs
    rcp_terms=[(6, 38), (76, 34)]  # 0-based atom pairs
)

result = optimizer.optimize(generations=10, population_size=20)
# Result now uses closure scores instead of energy
final_scores = result['final_closure_score']
if isinstance(final_scores, (list, np.ndarray)):
    best_score = max(final_scores)
else:
    best_score = final_scores
print(f"Final closure score: {best_score:.4f}")

# Energy is stored in top_candidates
if optimizer.top_candidates and optimizer.top_candidates[0].energy:
    print(f"Final energy: {optimizer.top_candidates[0].energy:.2f} kcal/mol")
```

## Example Patterns

### 1. Quick Optimization

For a quick optimization run:

```python
optimizer = RingClosureOptimizer.from_files(...)
result = optimizer.optimize(
    generations=30,
    population_size=20,
    enable_smoothing_refinement=True,  # Split into smoothing and Cartesian
    enable_cartesian_refinement=True
)
```

### 2. High-Quality Optimization

For thorough optimization:

```python
result = optimizer.optimize(
    population_size=50,
    generations=100,
    enable_smoothing_refinement=True,
    enable_cartesian_refinement=True,
    refinement_top_n=5,
    torsional_iterations=1000,  # Renamed from refinement_iterations
    cartesian_iterations=1000,
    refinement_convergence=0.001  # GA convergence threshold
)
```

### 3. Custom Workflow

Build your own optimization loop:

```python
from RingClosureOptimizer import (
    RingClosureOptimizer,
    Individual
)
from MolecularSystem import MolecularSystem
from CoordinateConverter import apply_torsions

# For advanced use, you can access internal components
optimizer = RingClosureOptimizer.from_files(...)

# Access the genetic algorithm after optimization starts
# The GA is initialized in optimize(), so for direct GA access:
system = MolecularSystem.from_file(...)
# Note: For direct GA usage, you'd need to manually create the GA instance
# This is typically not needed; use the high-level API instead
```

### 4. Batch Processing

Optimize multiple molecules:

```python
molecules = [
    'molecule1.int',
    'molecule2.int',
    'molecule3.int'
]

results = []
for mol_file in molecules:
    optimizer = RingClosureOptimizer.from_files(
        structure_file=mol_file,
        # ... other parameters
    )
        result = optimizer.optimize(verbose=False)
        results.append(result)
        final_scores = result['final_closure_score']
        best_score = max(final_scores) if isinstance(final_scores, (list, np.ndarray)) else final_scores
        energy_str = ""
        if optimizer.top_candidates and optimizer.top_candidates[0].energy:
            energy_str = f" (E={optimizer.top_candidates[0].energy:.2f} kcal/mol)"
        print(f"{mol_file}: closure score={best_score:.4f}{energy_str}")
```

### 5. Parameter Sweep

Test different parameter combinations:

```python
population_sizes = [20, 30, 50]
mutation_rates = [0.1, 0.15, 0.2]

best_result = None
best_energy = float('inf')

for pop_size in population_sizes:
    for mut_rate in mutation_rates:
        optimizer = RingClosureOptimizer.from_files(...)
        result = optimizer.optimize(
            population_size=pop_size,
            mutation_rate=mut_rate,
            generations=30,
            verbose=False
        )
        
        final_scores = result['final_closure_score']
        best_score = max(final_scores) if isinstance(final_scores, (list, np.ndarray)) else final_scores
        if best_score > best_score_val:  # Higher closure score is better
            best_score_val = best_score
            best_result = (pop_size, mut_rate)

print(f"Best: pop={best_result[0]}, mut={best_result[1]}, score={best_score_val:.4f}")
```

## Output Files

The examples will generate:
- `optimized_simple.xyz` - From example 1
- Various intermediate candidate files during optimization

## Tips

1. **Start Small**: Use small `generations` and `population_size` for testing
2. **Enable Refinement**: Usually gives better results
3. **Monitor Progress**: Set `verbose=True` to watch optimization
4. **Save Results**: Always call `save_optimized_structure()` to preserve results
5. **Batch Mode**: Set `verbose=False` when optimizing many molecules

## Common Use Cases

### Research: Parameter Studies

```python
# Test effect of refinement interval
for interval in [5, 10, 20]:
    result = optimizer.optimize(
        generations=50
    )
    final_scores = result['final_closure_score']
    best_score = max(final_scores) if isinstance(final_scores, (list, np.ndarray)) else final_scores
    print(f"Interval {interval}: closure score={best_score:.4f}")
```

### Production: Best Quality

```python
# High-quality optimization for publication
result = optimizer.optimize(
    population_size=100,
    generations=200,
    enable_smoothing_refinement=True,
    enable_cartesian_refinement=True,
    refinement_top_n=10,
    torsional_iterations=2000,
    cartesian_iterations=2000,
    refinement_convergence=0.0001
)
```

### Testing: Quick Validation

```python
# Quick test to verify setup
result = optimizer.optimize(
    population_size=10,
    generations=5,
    enable_smoothing_refinement=False,
    enable_cartesian_refinement=False
)
```

## Need Help?

- See `REFACTORING_GUIDE.md` for migration from old version
- Check docstrings in `RingClosureOptimizer.py` for detailed API docs
- Run `python ../src/__main__.py --help` or `rc-optimizer --help` for CLI options

## Contributing Examples

If you develop a useful pattern, consider adding it here!

1. Create a new example script
2. Document the use case
3. Add clear comments
4. Include expected output

