"""
test_generator.py - Generate reproducable flow networks for testing

Creates flow networks with variable sizes, densites (either sparse or dense), and capacity ranges.
"""

import random
from graph import FlowNetwork

def generate_random_network(n_nodes, edge_probability, min_capacity, max_capacity, seed=None):
    """
    Generate a random flow network.
    
    Args:
        n_nodes: Number of nodes in the network
        edge_probability: Probability of edge existing between any two nodes (0.0 to 1.0)
        min_capacity: Minimum edge capacity
        max_capacity: Maximum edge capacity
        seed: Random seed for reproducibility
    
    Returns:
        FlowNetwork object with source=0, sink=n_nodes-1
    """
    random.seed(seed)
    network = FlowNetwork(n_nodes)
    source = 0
    sink = n_nodes - 1

    # First step is to create a backbone path (single guaranteed source-to-sink path)
    for i in range(source, sink):
        capacity = random.randint(min_capacity, max_capacity)
        network.add_edge(i, i + 1, capacity)

    # Then add additional random edges
    for u in range(n_nodes):
        for v in range(u + 2, n_nodes):
            prob = random.random()

            if prob < edge_probability:
                capacity = random.randint(min_capacity, max_capacity)
                network.add_edge(u, v, capacity)

    return network