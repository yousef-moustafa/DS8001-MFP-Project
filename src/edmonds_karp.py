"""
edmonds_karp.py - Edmonds-Karp Algorithm for Maximum Flow

Implements the Edmonds-Karp algorithm using BFS to find augmenting paths.
Time complexity: O(VE^2)
"""

from collections import deque
from graph import FlowNetwork

def edmonds_karp(network, source, sink):
    """
    Compute maximum flow using Edmonds-Karp algorithm.
    
    Args:
        network: FlowNetwork object
        source: source node index
        sink: sink node index
    
    Returns:
        Maximum flow value
    """
    max_flow = 0
    
    # Keep finding augmenting paths until none exist
    while True:
        # Find augmenting path using BFS
        augmenting_path = bfs(network, source, sink)
        
        if augmenting_path == None:
            break
        
        # Find bottleneck (minimum residual capacity along path)
        bottleneck = min([edge.capacity - edge.flow for edge in augmenting_path])
        
        # Update flows along the path
        for edge in augmenting_path:
            edge.flow += bottleneck 
            edge.reverse.capacity += bottleneck
        
        max_flow += bottleneck
    
    return max_flow
    
def bfs(network, source, sink, neighbor_sort_fn=None):
    """Find shortest augmenting path using BFS."""
    parent = {}  # parent[node] = (parent_node, edge_used)
    visited = set([source])
    queue = deque([source])
    
    while queue:
        current = queue.popleft()
        
        # If we reached sink, reconstruct path
        if current == sink:
            # Reconstruct path by backtracking from sink to source
            path = []
            node = sink
            
            while node != source:
                parent_node, parent_edge = parent[node]
                path.append(parent_edge)
                node = parent_node
    
            path.reverse()  # We built it backwards, so reverse
            return path
        
        # Get neighbors
        neighbors = network.graph[current]
        
        # Apply optional sorting
        if neighbor_sort_fn is not None:
            neighbors = neighbor_sort_fn(neighbors)
        
        # Explore neighbors
        for edge in neighbors:
            # Check if edge has residual capacity AND dest not visited
            residual_capacity = edge.capacity - edge.flow
            
            if residual_capacity > 0 and edge.dest not in visited:
                visited.add(edge.dest)
                parent[edge.dest] = (current, edge)
                queue.append(edge.dest)
     
    return None  # No path found
