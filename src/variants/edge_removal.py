"""
Edge Removal Problem

Given a flow network with maximum flow F, find the maximum number of edges
that can be removed while maintaining flow >= F.
"""

import math
import random
from graph import FlowNetwork
from edmonds_karp import edmonds_karp
from metaheuristics.genetic_algorithm import GeneticAlgorithm
from metaheuristics.simulated_annealing import SimulatedAnnealing
import copy

class EdgeRemovalProblem:
    def __init__(self, network, source, sink):
        self.network = network
        self.source = source
        self.sink = sink
        
        # Calculate original max flow ONCE
        self.original_max_flow = self._compute_original_flow()

        # Extract all edges into an array
        self.edges = self._extract_edges()
    
    # Helper Methods
    def _compute_original_flow(self):
        network_copy = copy.deepcopy(self.network)
        return edmonds_karp(network_copy, self.source, self.sink)
    
    def _extract_edges(self):
        edges = []
        for node in range(self.network.n):
            for edge in self.network.graph[node]:
                # Only extract forward edges
                if edge.capacity > 0:
                    # Store node because we need to know where the edge came from
                    edges.append((node, edge))
        return edges

    def _create_subgraph_from_solution(self, solution):
        subgraph = FlowNetwork(self.network.n)
        
        for i, keep in enumerate(solution):
            if keep == 1:
                from_node, edge = self.edges[i]  # Unpack tuple
                subgraph.add_edge(from_node, edge.dest, edge.capacity)

        return subgraph
    
    # Interface Methods for SA/GA
    
    def get_random_solution(self):
            return [random.choice([0, 1]) for _ in range(len(self.edges))]

    def evaluate_fitness(self, solution):
        subgraph = self._create_subgraph_from_solution(solution)

        # Compute max flow on the solution
        flow = edmonds_karp(subgraph, self.source, self.sink)

        if flow >= self.original_max_flow:
            # Return number of edges removed
            return solution.count(0)
        else:
            return 0
        
    def get_initial_solution(self):
        return [1] * len(self.edges)

    def get_neighbor(self, solution):
        # Make a copy and flip a random bit
        neighbor = solution[:]
        index = random.randint(0, len(solution) - 1)
        neighbor[index] ^= 1
        return neighbor
    
    def is_valid(self, solution):
        # Always True for now (we handle validity in fitness). Can be extended for cheap constraint checks
        return True

    def crossover(self, parent1, parent2):
        # Choose a random index in the chromosome as the crossover point then swap the segments of the parents after the crossover point to create a child
        crossover_point = random.randint(1, len(parent1) - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child
    
    def mutation(self, individual):
        mutated = individual[:]
        mutation_prob = 1.0 / len(mutated)  # Standard rate in GA research papers
        
        for i in range(len(mutated)):
            if random.random() < mutation_prob:
                mutated[i] ^= 1
        
        return mutated
    
    def solve_with_sa(self, **kwargs):
        """Solve using Simulated Annealing."""
        sa = SimulatedAnnealing(problem=self, **kwargs)
        return sa.optimize()

    def solve_with_ga(self, **kwargs):
        """Solve using Genetic Algorithm."""
        ga = GeneticAlgorithm(problem=self, **kwargs)
        return ga.optimize()

        

    


