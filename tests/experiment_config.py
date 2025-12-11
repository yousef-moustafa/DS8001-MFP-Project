"""
Experimental configurations for algorithm comparison.
"""

# Network configurations
NETWORK_CONFIGS = {
    'quick': {
        'sizes': [20],
        'edge_probs': [0.2, 0.5],
        'capacity_range': (1, 100),
        'num_trials': 2
    },
    'small': {
        'sizes': [20, 50],
        'edge_probs': [0.2, 0.5],
        'capacity_range': (1, 100),
        'num_trials': 2
    },
    'medium': {
        'sizes': [20, 50, 100],
        'edge_probs': [0.2, 0.5, 0.8],
        'capacity_range': (1, 100),
        'num_trials': 3
    },
    'large': {
        'sizes': [50, 100, 200],
        'edge_probs': [0.2, 0.5],
        'capacity_range': (1, 100),
        'num_trials': 5
    }
}

# Simulated Annealing hyperparameters
SA_CONFIGS = {
    'default': {
        'initial_temp': 1000,
        'cooling_rate': 0.95,
        'iterations_per_temp': 500
    },
    'aggressive': {
        'initial_temp': 2000,
        'cooling_rate': 0.90,
        'iterations_per_temp': 1000
    },
    'conservative': {
        'initial_temp': 500,
        'cooling_rate': 0.98,
        'iterations_per_temp': 300
    }
}

# Genetic Algorithm hyperparameters
GA_CONFIGS = {
    'default': {
        'population_size': 100,
        'num_generations': 200,
        'mutation_rate': 0.05,
        'crossover_rate': 0.8,
        'elite_ratio': 0.1
    },
    'large_pop': {
        'population_size': 200,
        'num_generations': 100,
        'mutation_rate': 0.05,
        'crossover_rate': 0.8,
        'elite_ratio': 0.1
    },
    'more_gens': {
        'population_size': 50,
        'num_generations': 400,
        'mutation_rate': 0.05,
        'crossover_rate': 0.8,
        'elite_ratio': 0.1
    }
}

# Edge selection budget fractions
BUDGET_FRACTIONS = [0.3, 0.5, 0.7]
EDGE_SELECTION_TRIALS = 1