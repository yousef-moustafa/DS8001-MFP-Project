"""
graph.py - Flow Network Data Structure

This module implements the graph representation for maximum flow problems
using an adjacency list with Edge objects.
"""

class Edge:
    """Represents a directed edge in the flow network."""
    def __init__(self, dest, capacity):
        self.dest = dest              # destination node
        self.capacity = capacity  # max capacity
        self.flow = 0            # current flow
        self.reverse = None      # pointer to reverse edge

class FlowNetwork:
    """Flow network represented as adjacency list of Edge objects."""
    def __init__(self, n):
        self.n = n  # number of nodes
        self.graph = [[] for _ in range(n)]  # adjacency list

    def add_edge(self, src_node, dest_node, capacity):
        """Add directed edge with given capacity."""
        # Create forward edge
        forward_edge = Edge(dest_node, capacity)

        # Create backward edge with 0 capacity (for residual graph)
        backward_edge = Edge(src_node, 0)

        # Link them together
        forward_edge.reverse = backward_edge
        backward_edge.reverse = forward_edge

        # Add to adjacency lists
        self.graph[src_node].append(forward_edge)
        self.graph[dest_node].append(backward_edge)