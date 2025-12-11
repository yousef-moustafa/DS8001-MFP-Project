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

# Configuration
NETWORK_SIZES = [20, 50, 100, 200]
EDGE_PROBABILITIES = [0.2, 0.5]
CAPACITY_RANGE = (1, 100)
NUM_TRIALS = 2

def test_standard_max_flow():
    """Test Edmonds-Karp, LP, and Greedy on generated networks."""
    print("Testing Standard Max Flow Algorithms...")
    print("-" * 60)
    
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
                
                # Test LP
                net2 = generate_random_network(n_nodes, edge_prob, CAPACITY_RANGE[0], CAPACITY_RANGE[1], seed=trial)
                start = time.time()
                lp_flow = max_flow_lp(net2, source, sink)
                lp_time = time.time() - start
                
                # Test Greedy
                net3 = generate_random_network(n_nodes, edge_prob, CAPACITY_RANGE[0], CAPACITY_RANGE[1], seed=trial)
                start = time.time()
                greedy_flow = greedy_max_flow(net3, source, sink)
                greedy_time = time.time() - start
                
                print(f"n={n_nodes}, p={edge_prob}, trial={trial+1}")
                print(f"  EK: flow={ek_flow:.1f}, time={ek_time:.4f}s")
                print(f"  LP: flow={lp_flow:.1f}, time={lp_time:.4f}s")
                print(f"  Greedy: flow={greedy_flow:.1f}, time={greedy_time:.4f}s")
                print()

if __name__ == "__main__":
    test_standard_max_flow()