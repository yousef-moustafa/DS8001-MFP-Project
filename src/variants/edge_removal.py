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

    
    # Solve using Gurobi
    def solve_with_ip(self):
        """Solve edge removal using Integer Programming."""
        import gurobipy as gp
        from gurobipy import GRB
        
        model = gp.Model("EdgeRemoval")
        model.setParam('OutputFlag', 0)
        
        # Decision variables
        flow_vars = {}  # f[u,v] = amount of flow on edge (u,v)
        edge_vars = {}  # x[i] = 1 if keep edge i, 0 if remove edge i

        for i, (u, edge) in enumerate(self.edges):
            v = edge.dest
            capacity = edge.capacity
            
            # Flow variable (How much flow goes through this edge)
            flow_vars[(u, v)] = model.addVar(lb=0.0, ub=capacity, name=f"f_{u}_{v}")
            
            # Binary edge selection variable (Do we keep this edge)
            edge_vars[i] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}")

        # Objective: maximize number of removed edges = maximize sum(1 - x[i])
        model.setObjective(
            gp.quicksum(1 - edge_vars[i] for i in range(len(self.edges))),
            GRB.MAXIMIZE
        )

        # Constraint 1: if an edge is not selected (edge_vars[i] = 0) flow should be 0
        for i, (u, edge) in enumerate(self.edges):
            v = edge.dest
            model.addConstr(flow_vars[(u, v)] <= edge_vars[i] * edge.capacity, name=f"cap_{i}")

        # Constraint 2: Flow Conservation (flow in = flow out) for all nodes except source and sink
        for node in range(self.network.n):
            if node == self.source or node == self.sink:
                continue
            
            # Sum of all flows INTO node
            flow_in = gp.quicksum(
                flow_vars[(u, node)]
                for (u, edge) in self.edges
                if edge.dest == node
            )
            
            # Sum of all flows OUT OF node
            flow_out = gp.quicksum(
                flow_vars[(node, edge.dest)]
                for (u, edge) in self.edges
                if u == node
            )
            
            model.addConstr(flow_in == flow_out, name=f"conservation_{node}")
        
        # Constraint 3: Ensure total flow ≥ original max flow
        # Note: By flow conservation, total flow leaving source equals total flow entering sink, which is the definition of max flow in the network
        total_flow_out = gp.quicksum(
            flow_vars[(self.source, edge.dest)]
            for (u, edge) in self.edges
            if u == self.source
        )
        model.addConstr(total_flow_out >= self.original_max_flow, name="min_flow")

        model.optimize()

        if model.status == GRB.OPTIMAL:
            # extracts the solution value from a Gurobi variable and convert it to an int
            solution = [int(edge_vars[i].X) for i in range(len(self.edges))]
            num_removed = solution.count(0)
            return solution, num_removed
        else:
            return None, 0
    


