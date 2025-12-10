"""
greedy.py - Greedy Heuristic for Maximum Flow

Uses modified BFS that prioritizes high-capacity edges.
"""

from graph import FlowNetwork
from edmonds_karp import bfs

def greedy_neighbor_sort(edges):
    """Sort edges by residual capacity (descending)."""
    return sorted(edges, key=lambda e: e.capacity - e.flow, reverse=True)

def greedy_max_flow(network, source, sink):
    """
    Compute maximum flow using greedy heuristic.
    
    Prioritizes augmenting paths with higher capacity edges.
    
    Args:
        network: FlowNetwork object
        source: source node index
        sink: sink node index
    
    Returns:
        Maximum flow value
    """
    max_flow = 0
    
    while True:
        # Find augmenting path using greedy neighbor ordering
        path = bfs(network, source, sink, neighbor_sort_fn=greedy_neighbor_sort)
        
        if path is None:
            break
        
        # Find bottleneck
        bottleneck = min(edge.capacity - edge.flow for edge in path)
        
        # Update flows
        for edge in path:
            edge.flow += bottleneck
            edge.reverse.capacity += bottleneck
        
        max_flow += bottleneck
    
    return max_flow