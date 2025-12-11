# Comparing Algorithmic Approaches for Maximum Flow Problem (under Edge Constraints)

**Course:** DS8001 - Design of Algorithms & Programming for Massive Data  
**Institution:** Toronto Metropolitan University  
**Term:** Fall 2025 

**Team Members:**
- Yosef Moustafa - Implementation & Experimentation
- Alex Eliseev - Analysis & Report Writing

## Overview

This project implements and compares multiple algorithmic approaches for solving the Maximum Flow Problem (MFP) and its constrained variants. We solve the standard MFP using three approaches: Edmonds-Karp algorithm, Greedy heuristic, and Linear Programming. Additionally, we explore two NP-hard constrained variants—**Edge Removal** (maximize edges removed while maintaining flow) and **Edge Selection** (maximize flow with limited edge budget) using Integer Programming, Simulated Annealing, and Genetic Algorithms.

## Project Structure
```
DS8001-MFP-Project/
├── src/                          # Source code
│   ├── graph.py                  # Flow network data structure
│   ├── edmonds_karp.py           # Edmonds-Karp algorithm (O(VE²))
│   ├── linear_programming.py     # Gurobi LP solver
│   ├── greedy.py                 # Greedy max flow heuristic
│   ├── test_generator.py         # Random network generator
│   ├── metaheuristics/           # Generic optimization frameworks
│   │   ├── simulated_annealing.py
│   │   └── genetic_algorithm.py
│   └── variants/                 # Constrained MFP variants
│       ├── edge_removal.py       # Edge removal problem (IP/SA/GA)
│       └── edge_selection.py     # Edge selection problem (IP/SA/GA)
├── tests/                        # Experimental framework
│   ├── run_experiments.py        # Main experiment runner
│   ├── experiment_config.py      # Configuration presets
│   ├── experiment_runners.py     # Test execution functions
│   ├── analysis.ipynb            # Results visualization (Jupyter)
│   └── results/                  # Experimental outputs (CSV files)
|
└── README.md                     # This file
```

## Requirements

### Python Version
tested on Python 3.11

### Required Libraries

**Core Implementation:**
- Standard library modules: `random`, `math`, `copy`, `collections`, `csv`, `time`, `os`
- `gurobipy` - Gurobi Optimizer (requires academic license)

**Analysis & Visualization (optional):**
- `pandas`, `matplotlib`, `seaborn`, `numpy`, `jupyter`

### Installation
```bash
# Install Gurobi
pip install gurobipy

# Optional: For running analysis notebooks
pip install pandas matplotlib seaborn numpy jupyter
```

### Gurobi License Setup

Gurobi requires a valid license. Academic licenses are free:

1. Register at [gurobi.com/academia](https://www.gurobi.com/academia/academic-program-and-licenses/)
2. Download license file (`gurobi.lic`)
3. Set environment variable:
```bash
   # Windows
   set GRB_LICENSE_FILE=C:\path\to\gurobi.lic
   
   # Linux/Mac
   export GRB_LICENSE_FILE=/path/to/gurobi.lic
```

## Usage

### Running Experiments

To run all experiments with default configuration:
```bash
cd tests
python run_experiments.py
```

### Configuration

Modify experiment settings in `tests/experiment_config.py`:
```python
# Choose network size configuration
NETWORK_SIZE = 'small'  # Options: 'quick', 'small', 'medium', 'large'

# Choose algorithm hyperparameters
SA_HYPERPARAMS = 'default'  # Options: 'default', 'aggressive', 'conservative'
GA_HYPERPARAMS = 'default'  # Options: 'default', 'large_pop', 'more_gens'
```

### Output

Results are saved as CSV files in `tests/results/`:
- `results_standard_maxflow.csv` - Standard max flow comparison
- `results_edge_removal.csv` - Edge removal experiments
- `results_edge_selection.csv` - Edge selection experiments

**Optional:** Run `tests/analysis.ipynb` for visualization and summary statistics.

## Implementation Details

### Core Algorithms

**Edmonds-Karp Algorithm** (`edmonds_karp.py`)
- Implements the classical O(VE²) max flow algorithm using BFS for path finding
- Based on the original Edmonds-Karp (1972) paper formulation

**Greedy Heuristic** (`greedy.py`)
- Modified Edmonds-Karp that prioritizes augmenting paths by residual capacity
- Sorts edges by (capacity - flow) in descending order during BFS exploration

**Linear Programming** (`linear_programming.py`)
- Formulated as LP problem in Gurobi:
  - **Objective:** Maximize Σ flow[source, v]
  - **Constraints:** Flow conservation (flow_in = flow_out) and capacity bounds
- Solves standard max flow to optimality

### Metaheuristic Framework

**Object-Oriented Design:**
- Generic framework classes in `metaheuristics/`:
  - `SimulatedAnnealing` - Problem-agnostic SA optimizer
  - `GeneticAlgorithm` - Population-based GA optimizer
- Problem-specific implementations in `variants/`:
  - `EdgeRemovalProblem` - Maximize edges removed while maintaining flow
  - `EdgeSelectionProblem` - Maximize flow with limited edge budget
- Each problem class implements required interface methods (fitness evaluation, neighbor generation, crossover, mutation)

**Key Techniques:**
- **Edge Removal:** Penalty-based fitness (fitness = edges_removed - flow_deficit) to handle constraint violations
- **Edge Selection:** Swap-based mutation to maintain exactly K edges selected

### Graph Representation

**Data Structure** (`graph.py`)
- Adjacency list representation with Edge objects
- Each edge stores: destination, capacity, current flow, and reverse edge pointer
- Follows standard textbook approach (e.g., Cormen et al., "Introduction to Algorithms")
- Supports efficient residual graph operations for flow algorithms

## Experimental Results

**Quick Summary:**
- **Standard Max Flow:** All algorithms find optimal; EK fastest on small instances
- **Edge Removal:** IP optimal, SA ~80% optimal, GA ~70% optimal
- **Edge Selection:** All algorithms excellent (SA 99.97%, GA 98% of optimal)

**Results Location:** `tests/results/` (CSV files)  

Comprehensive experimental analysis is available in the project report.
**Full Report:** [Link to be added upon completion]

## References

- Edmonds, J., & Karp, R. M. (1972). Theoretical improvements in algorithmic efficiency for network flow problems.
- Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by simulated annealing.
- Holland, J. H. (1975). Adaptation in Natural and Artificial Systems.
- Michalewicz, Z., & Schoenauer, M. (1996). Evolutionary algorithms for constrained parameter optimization problems.
- Cormen, T. H., et al. (2009). Introduction to Algorithms (3rd ed.). MIT Press.