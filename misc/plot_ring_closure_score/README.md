# Plotting Ring Closure Score

This directory contains a script to visualize the ring closure score as a function of the distance between the two atoms in an RCP term.

### Prerequisites

The package must be installed (either via `pip install -e .` for development or `pip install ringclosingmm` from PyPI).

### Usage

```bash
python plot_ring_closure_score.py --tolerance 0.1 --decay-rate 1.0 0.5 0.25 --max-distance 15.0
```



