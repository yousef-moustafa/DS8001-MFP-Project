"""
Experiment runner functions for algorithm comparison.
"""

import time
import csv
import os
from test_generator import generate_random_network
from edmonds_karp import edmonds_karp
from linear_programming import max_flow_lp
from greedy import greedy_max_flow
from variants.edge_removal import EdgeRemovalProblem
from variants.edge_selection import EdgeSelectionProblem
from experiment_config import BUDGET_FRACTIONS


def run_standard_maxflow_experiments(network_config):
    """Test Edmonds-Karp, LP, and Greedy on generated networks."""
    print("Testing Standard Max Flow Algorithms...")
    print("-" * 60)

    results = []  # Collect results to write to csv
    
    for n_nodes in network_config['sizes']:
        for edge_prob in network_config['edge_probs']:
            for trial in range(network_config['num_trials']):
                source, sink = 0, n_nodes - 1
                
                # Test Edmonds-Karp
                net1 = generate_random_network(n_nodes, edge_prob, network_config['capacity_range'][0], network_config['capacity_range'][1], seed=trial)
                start = time.time()
                ek_flow = edmonds_karp(net1, source, sink)
                ek_time = time.time() - start

                record_result(results, 'standard_max_flow', 'edmonds_karp', n_nodes, edge_prob, trial+1, ek_flow, ek_time)
                
                # Test LP
                net2 = generate_random_network(n_nodes, edge_prob, network_config['capacity_range'][0], network_config['capacity_range'][1], seed=trial)
                start = time.time()
                lp_flow = max_flow_lp(net2, source, sink)
                lp_time = time.time() - start

                record_result(results, 'standard_max_flow', 'linear_programming', n_nodes, edge_prob, trial+1, lp_flow, lp_time)
                
                # Test Greedy
                net3 = generate_random_network(n_nodes, edge_prob, network_config['capacity_range'][0], network_config['capacity_range'][1], seed=trial)
                start = time.time()
                greedy_flow = greedy_max_flow(net3, source, sink)
                greedy_time = time.time() - start

                record_result(results, 'standard_max_flow', 'greedy', n_nodes, edge_prob, trial+1, greedy_flow, greedy_time)
                
                print(f"n={n_nodes}, p={edge_prob}, trial={trial+1}")
                print(f"  EK: flow={ek_flow:.1f}, time={ek_time:.4f}s")
                print(f"  LP: flow={lp_flow:.1f}, time={lp_time:.4f}s")
                print(f"  Greedy: flow={greedy_flow:.1f}, time={greedy_time:.4f}s")
                print()

    return results

def run_edge_removal_experiments(network_config, sa_params, ga_params):
    """Test IP, SA, GA on edge removal problem."""
    print("\nTesting Edge Removal Algorithms...")
    print("-" * 60)
    
    results = []
    for n_nodes in network_config['sizes']:
        for edge_prob in network_config['edge_probs']:
            for trial in range(network_config['num_trials']):
                source, sink = 0, n_nodes - 1
                
                # Test IP
                net1 = generate_random_network(n_nodes, edge_prob, network_config['capacity_range'][0], network_config['capacity_range'][1], seed=trial)
                problem_ip = EdgeRemovalProblem(net1, source, sink)
                start = time.time()
                _, lp_num_removed = problem_ip.solve_with_ip()
                lp_time = time.time() - start
                
                record_result(results, 'edge_removal', 'IP', n_nodes, edge_prob, trial+1, problem_ip.original_max_flow, lp_time, num_removed=int(lp_num_removed))
                
                # Test SA
                net2 = generate_random_network(n_nodes, edge_prob, network_config['capacity_range'][0], network_config['capacity_range'][1], seed=trial)
                problem_sa = EdgeRemovalProblem(net2, source, sink)
                start = time.time()
                _, sa_num_removed = problem_sa.solve_with_sa(**sa_params)
                sa_time = time.time() - start
                
                record_result(results, 'edge_removal', 'SA', n_nodes, edge_prob, trial+1, problem_sa.original_max_flow, sa_time, num_removed=sa_num_removed)
                
                # Test GA
                net3 = generate_random_network(n_nodes, edge_prob, network_config['capacity_range'][0], network_config['capacity_range'][1], seed=trial)
                problem_ga = EdgeRemovalProblem(net3, source, sink)
                start = time.time()
                _, ga_num_removed = problem_ga.solve_with_ga(**ga_params)
                ga_time = time.time() - start
                
                record_result(results, 'edge_removal', 'GA', n_nodes, edge_prob, trial+1, problem_ga.original_max_flow, ga_time, num_removed=ga_num_removed)
                
                print(f"n={n_nodes}, p={edge_prob}, trial={trial+1}")
                print(f"  IP: removed={lp_num_removed}, time={lp_time:.4f}s")
                print(f"  SA: removed={sa_num_removed}, time={sa_time:.4f}s")
                print(f"  GA: removed={ga_num_removed}, time={ga_time:.4f}s")
                print()
    
    return results

def run_edge_selection_experiments(network_config, sa_params, ga_params):
    """Test IP, SA, GA on edge selection problem."""
    print("\nTesting Edge Selection Algorithms...")
    print("-" * 60)
    
    results = []
    
    for n_nodes in network_config['sizes']:
        for edge_prob in network_config['edge_probs']:
            for trial in range(network_config['num_trials']):
                source, sink = 0, n_nodes - 1
                
                # Generate base network to calculate total edges
                net_base = generate_random_network(n_nodes, edge_prob, network_config['capacity_range'][0], network_config['capacity_range'][1], seed=trial)
                total_edges = sum(1 for node in range(net_base.n) 
                                 for edge in net_base.graph[node] if edge.capacity > 0)
                
                for budget_frac in BUDGET_FRACTIONS:
                    budget = int(total_edges * budget_frac)
                    
                    # Test IP
                    net1 = generate_random_network(n_nodes, edge_prob, network_config['capacity_range'][0], network_config['capacity_range'][1], seed=trial)
                    problem_ip = EdgeSelectionProblem(net1, source, sink, budget)
                    start = time.time()
                    _, ip_flow = problem_ip.solve_with_ip()
                    ip_time = time.time() - start
                    record_result(results, 'edge_selection', 'IP', n_nodes, edge_prob, trial+1, 
                                 ip_flow, ip_time, budget=budget, budget_frac=budget_frac)
                    
                    # Test SA
                    net2 = generate_random_network(n_nodes, edge_prob, network_config['capacity_range'][0], network_config['capacity_range'][1], seed=trial)
                    problem_sa = EdgeSelectionProblem(net2, source, sink, budget)
                    start = time.time()
                    _, sa_flow = problem_sa.solve_with_sa(**sa_params)
                    sa_time = time.time() - start
                    record_result(results, 'edge_selection', 'SA', n_nodes, edge_prob, trial+1,
                                 sa_flow, sa_time, budget=budget, budget_frac=budget_frac)
                    
                    # Test GA
                    net3 = generate_random_network(n_nodes, edge_prob, network_config['capacity_range'][0], network_config['capacity_range'][1], seed=trial)
                    problem_ga = EdgeSelectionProblem(net3, source, sink, budget)
                    start = time.time()
                    _, ga_flow = problem_ga.solve_with_ga(**ga_params)
                    ga_time = time.time() - start
                    record_result(results, 'edge_selection', 'GA', n_nodes, edge_prob, trial+1,
                                 ga_flow, ga_time, budget=budget, budget_frac=budget_frac)
                    
                    print(f"n={n_nodes}, p={edge_prob}, trial={trial+1}, budget={budget} ({budget_frac*100:.0f}%)")
                    print(f"  IP: flow={ip_flow:.1f}, time={ip_time:.4f}s")
                    print(f"  SA: flow={sa_flow:.1f}, time={sa_time:.4f}s")
                    print(f"  GA: flow={ga_flow:.1f}, time={ga_time:.4f}s")
                    print()
    
    return results

def record_result(results, problem, algorithm, n_nodes, edge_prob, trial, flow, time_taken, **extra):
    """Helper to record a single result."""
    result = {
        'problem': problem,
        'algorithm': algorithm,
        'n_nodes': n_nodes,
        'edge_prob': edge_prob,
        'trial': trial,
        'flow': flow,
        'time': time_taken
    }
    result.update(extra)  # Add any extra fields
    results.append(result)

import csv

def export_results(results, filename):
    """Export results to CSV file."""
    if not results:
        return
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Full path to output file
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results exported to {filepath}")