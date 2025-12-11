"""
run_experiments.py - Algorithm comparison experiments
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
from test_generator import generate_random_network
from edmonds_karp import edmonds_karp
from linear_programming import max_flow_lp
from greedy import greedy_max_flow
from variants.edge_removal import EdgeRemovalProblem

# Configuration
NETWORK_SIZES = [20, 50, 100, 200]
EDGE_PROBABILITIES = [0.2, 0.5]
CAPACITY_RANGE = (1, 100)
NUM_TRIALS = 2

def test_standard_max_flow():
    """Test Edmonds-Karp, LP, and Greedy on generated networks."""
    print("Testing Standard Max Flow Algorithms...")
    print("-" * 60)

    results = []  # Collect results to write to csv
    
    for n_nodes in NETWORK_SIZES:
        for edge_prob in EDGE_PROBABILITIES:
            for trial in range(NUM_TRIALS):
                # Generate network
                network = generate_random_network(
                    n_nodes, edge_prob, 
                    CAPACITY_RANGE[0], CAPACITY_RANGE[1],
                    seed=trial
                )
                
                source, sink = 0, n_nodes - 1
                
                # Test Edmonds-Karp
                net1 = generate_random_network(n_nodes, edge_prob, CAPACITY_RANGE[0], CAPACITY_RANGE[1], seed=trial)
                start = time.time()
                ek_flow = edmonds_karp(net1, source, sink)
                ek_time = time.time() - start

                record_result(results, 'standard_max_flow', 'edmonds_karp', n_nodes, edge_prob, trial+1, ek_flow, ek_time)
                
                # Test LP
                net2 = generate_random_network(n_nodes, edge_prob, CAPACITY_RANGE[0], CAPACITY_RANGE[1], seed=trial)
                start = time.time()
                lp_flow = max_flow_lp(net2, source, sink)
                lp_time = time.time() - start

                record_result(results, 'standard_max_flow', 'linear_programming', n_nodes, edge_prob, trial+1, lp_flow, lp_time)
                
                # Test Greedy
                net3 = generate_random_network(n_nodes, edge_prob, CAPACITY_RANGE[0], CAPACITY_RANGE[1], seed=trial)
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

def test_edge_removal():
    """Test IP, SA, GA on edge removal problem."""
    print("\nTesting Edge Removal Algorithms...")
    print("-" * 60)
    
    results = []
    for n_nodes in NETWORK_SIZES:
        for edge_prob in EDGE_PROBABILITIES:
            for trial in range(NUM_TRIALS):
                source, sink = 0, n_nodes - 1
                
                # Test IP
                net1 = generate_random_network(n_nodes, edge_prob, CAPACITY_RANGE[0], CAPACITY_RANGE[1], seed=trial)
                problem_ip = EdgeRemovalProblem(net1, source, sink)
                start = time.time()
                _, lp_num_removed = problem_ip.solve_with_ip()
                lp_time = time.time() - start
                
                record_result(results, 'edge_removal', 'IP', n_nodes, edge_prob, trial+1, problem_ip.original_max_flow, lp_time, num_removed=lp_num_removed)
                
                # Test SA
                net2 = generate_random_network(n_nodes, edge_prob, CAPACITY_RANGE[0], CAPACITY_RANGE[1], seed=trial)
                problem_sa = EdgeRemovalProblem(net2, source, sink)
                start = time.time()
                _, sa_num_removed = problem_sa.solve_with_sa()
                sa_time = time.time() - start
                
                record_result(results, 'edge_removal', 'SA', n_nodes, edge_prob, trial+1, problem_sa.original_max_flow, sa_time, num_removed=sa_num_removed)
                
                # Test GA
                net3 = generate_random_network(n_nodes, edge_prob, CAPACITY_RANGE[0], CAPACITY_RANGE[1], seed=trial)
                problem_ga = EdgeRemovalProblem(net3, source, sink)
                start = time.time()
                _, ga_num_removed = problem_ga.solve_with_ga()
                ga_time = time.time() - start
                
                record_result(results, 'edge_removal', 'GA', n_nodes, edge_prob, trial+1, problem_ga.original_max_flow, ga_time, num_removed=ga_num_removed)
                
                print(f"n={n_nodes}, p={edge_prob}, trial={trial+1}")
                print(f"  IP: removed={lp_num_removed}, time={lp_time:.4f}s")
                print(f"  SA: removed={sa_num_removed}, time={sa_time:.4f}s")
                print(f"  GA: removed={ga_num_removed}, time={ga_time:.4f}s")
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
    
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results exported to {filename}")


if __name__ == "__main__":
    # Test standard max flow
    results_standard = test_standard_max_flow()
    export_results(results_standard, 'results_standard_maxflow.csv')
    
    # Test edge removal
    results_removal = test_edge_removal()
    export_results(results_removal, 'results_edge_removal.csv')