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

    # Solve using Gurobi
    def solve_with_ip(self):
        """Solve edge selection using Integer Programming."""
        import gurobipy as gp
        from gurobipy import GRB
        
        model = gp.Model("EdgeSelection")
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

        # Objective: maximize total flow leaving source (different from edge removal)
        total_flow_out = gp.quicksum(
            flow_vars[(self.source, edge.dest)]
            for (u, edge) in self.edges
            if u == self.source
        )
        model.setObjective(total_flow_out, GRB.MAXIMIZE)

        # Constraint 1: if an edge is not selected (edge_vars[i] = 0) flow should be 0
        # Same as edgeRemoval
        for i, (u, edge) in enumerate(self.edges):
            v = edge.dest
            model.addConstr(flow_vars[(u, v)] <= edge_vars[i] * edge.capacity, name=f"cap_{i}")

        # Constraint 2: Flow Conservation (flow in = flow out) for all nodes except source and sink
        # Same as edgeRemoval
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
        
        # Constraint 3: Budget constraint (k edges selected)
        # Different
        model.addConstr(
            gp.quicksum(edge_vars[i] for i in range(len(self.edges))) == self.budget,
            name="budget"
        )

        model.optimize()

        if model.status == GRB.OPTIMAL:
            # extracts the solution value from a Gurobi variable and convert it to an int
            solution = [int(edge_vars[i].X) for i in range(len(self.edges))]
            achieved_flow = total_flow_out.getValue()  # Get the flow value
            return solution, achieved_flow
        else:
            return None, 0