"""
Run algorithm comparison experiments.

Change the config settings below to run different experiment configurations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from experiment_config import NETWORK_CONFIGS, SA_CONFIGS, GA_CONFIGS
from experiment_runners import (
    run_standard_maxflow_experiments,
    run_edge_removal_experiments,
    run_edge_selection_experiments,
    export_results
)

# CONFIGURATION - Change these to run different experiments

# Network size: 'quick' (fast), 'small', 'medium', or 'large'
NETWORK_SIZE = 'small'

# Algorithm hyperparameters
SA_HYPERPARAMS = 'default'  # Options: 'default', 'aggressive', 'conservative'
GA_HYPERPARAMS = 'default'  # Options: 'default', 'large_pop', 'more_gens'


def main():
    # Load configurations
    network_config = NETWORK_CONFIGS[NETWORK_SIZE]
    sa_params = SA_CONFIGS[SA_HYPERPARAMS]
    ga_params = GA_CONFIGS[GA_HYPERPARAMS]
    
    print(f"\nRunning experiments with:")
    print(f"  Networks: {network_config['sizes']} nodes, {network_config['num_trials']} trials each")
    print(f"  SA: temp={sa_params['initial_temp']}, cooling={sa_params['cooling_rate']}")
    print(f"  GA: pop={ga_params['population_size']}, gens={ga_params['num_generations']}")
    print()
    
    # Run standard max flow experiments
    results_standard = run_standard_maxflow_experiments(network_config)
    export_results(results_standard, 'results_standard_maxflow.csv')
    
    # Run edge removal experiments
    results_removal = run_edge_removal_experiments(network_config, sa_params, ga_params)
    export_results(results_removal, 'results_edge_removal.csv')
    
    # Run edge selection experiments
    results_selection = run_edge_selection_experiments(network_config, sa_params, ga_params)
    export_results(results_selection, 'results_edge_selection.csv')
    
    # Summary
    print("\n" + "=" * 60)
    print("Experiments complete!")
    print(f"  {len(results_standard)} standard max flow runs")
    print(f"  {len(results_removal)} edge removal runs")
    print(f"  {len(results_selection)} edge selection runs")
    print("\nResults saved to tests/results/")
    print("=" * 60)

if __name__ == "__main__":
    main()