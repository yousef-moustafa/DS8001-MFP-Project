"""
Edge Selection Problem

Given a flow network and edge budget K, select K edges to maximize achievable flow.
"""

"""
Edge Selection Problem

Given a flow network and edge budget K, select K edges to maximize achievable flow.
"""

import random
from graph import FlowNetwork
from edmonds_karp import edmonds_karp
from metaheuristics.genetic_algorithm import GeneticAlgorithm
from metaheuristics.simulated_annealing import SimulatedAnnealing
import copy

class EdgeSelectionProblem:
    def __init__(self, network, source, sink, budget):
        self.network = network
        self.source = source
        self.sink = sink
        self.budget = budget  # K number of edges we can select
        
        # Extract all edges into an array
        self.edges = self._extract_edges()

    # Helper Methods

    # Identical to Edge Removal version
    def _extract_edges(self):
        edges = []
        for node in range(self.network.n):
            for edge in self.network.graph[node]:
                # Only extract forward edges
                if edge.capacity > 0:
                    # Store node because we need to know where the edge came from
                    edges.append((node, edge))
        return edges
    
    # Identical to Edge Removal version
    def _create_subgraph_from_solution(self, solution):
        subgraph = FlowNetwork(self.network.n)
        
        for i, keep in enumerate(solution):
            if keep == 1:
                from_node, edge = self.edges[i]
                subgraph.add_edge(from_node, edge.dest, edge.capacity)
        
        return subgraph
    
    # Interface Methods for SA/GA

    def get_random_solution(self):
        # start with all zeros
        solution = [0] * len(self.edges)

        # Randomly sample k indices
        selected_indices = random.sample(range(len(self.edges)), self.budget)
        
        # Set them to 1
        for i in selected_indices:
            solution[i] = 1
        
        return solution

    def evaluate_fitness(self, solution):
        if solution.count(1) == self.budget:
            subgraph = self._create_subgraph_from_solution(solution)
            
            # Compute solution's flow
            flow = edmonds_karp(subgraph, self.source, self.sink)

            return flow
        else:
            return 0
        
    def get_initial_solution(self, strategy='random'):
        if strategy == 'random':
            return self.get_random_solution()
        # Deterministic: select first K edges
        elif strategy == 'first_k':
            solution = [0] * len(self.edges)
            for i in range(min(self.budget, len(self.edges))):
                solution[i] = 1
            return solution
        
    def get_neighbor(self, solution):
        # Make a copy and flip a random bit
        neighbor = solution[:]

        zero_index = random.choice([i for i, val in enumerate(solution) if val == 0])
        one_index = random.choice([i for i, val in enumerate(solution) if val == 1])

        # Swap them
        neighbor[zero_index], neighbor[one_index] = neighbor[one_index], neighbor[zero_index]

        return neighbor

    def is_valid(self, solution):
        return True if solution.count(1) == self.budget else False
    
    def crossover(self, parent1, parent2):
        # Single-point crossover
        crossover_point = random.randint(1, len(parent1) - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        
        # Repair to ensure child has exactly K edges selected
        current = child.count(1)

        if current > self.budget:
            # Too many selected therefore deselect some
            selected = [i for i, val in enumerate(child) if val == 1]
            to_remove = random.sample(selected, current - self.budget)
            for i in to_remove:
                child[i] = 0
        elif current < self.budget:
            # Too few selected therefore select more
            unselected = [i for i, val in enumerate(child) if val == 0]
            to_add = random.sample(unselected, self.budget - current)
            for i in to_add:
                child[i] = 1

        return child

    def mutation(self, individual):
        # Swap-based mutation maintaining exactly K edges (symmetric so no need for else)
        mutated = individual[:]
        mutation_prob = 1.0 / len(mutated)
        
        for i in range(len(mutated)):
            if random.random() < mutation_prob and mutated[i] == 1:
                zeros = [j for j, val in enumerate(mutated) if val == 0]
                if zeros:
                    swap_with = random.choice(zeros)
                    mutated[i], mutated[swap_with] = 0, 1
        
        return mutated

    def solve_with_sa(self, **kwargs):
        """Solve using Simulated Annealing."""
        sa = SimulatedAnnealing(problem=self, **kwargs)
        return sa.optimize()

    def solve_with_ga(self, **kwargs):
        """Solve using Genetic Algorithm."""
        ga = GeneticAlgorithm(problem=self, **kwargs)
        return ga.optimize()
