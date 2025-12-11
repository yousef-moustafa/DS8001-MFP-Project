import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from test_generator import generate_random_network
from variants.edge_removal import EdgeRemovalProblem

# Generate small test network
network = generate_random_network(10, 0.5, 1, 100, seed=42)
problem = EdgeRemovalProblem(network, 0, 9)

print(f"Total edges: {len(problem.edges)}")
print(f"Original max flow: {problem.original_max_flow}")
print()

# Test random solutions
print("Testing 10 random solutions:")
for i in range(10):
    solution = problem.get_random_solution()
    fitness = problem.evaluate_fitness(solution)
    print(f"  Solution {i+1}: {sum(solution)} edges kept, fitness={fitness}")

print()

# Test crossover
print("Testing crossover:")
parent1 = problem.get_random_solution()
parent2 = problem.get_random_solution()

print(f"Parent1: {sum(parent1)} edges kept, fitness={problem.evaluate_fitness(parent1)}")
print(f"Parent2: {sum(parent2)} edges kept, fitness={problem.evaluate_fitness(parent2)}")

child = problem.crossover(parent1, parent2)
print(f"Child: {sum(child)} edges kept, fitness={problem.evaluate_fitness(child)}")

print()

# Test mutation
print("Testing mutation:")
for i in range(5):
    individual = problem.get_random_solution()
    mutated = problem.mutation(individual)
    print(f"  Original: fitness={problem.evaluate_fitness(individual)}, Mutated: fitness={problem.evaluate_fitness(mutated)}")

# Test IP solver (optimal)
print("\nTesting IP (optimal solution):")
solution, num_removed = problem.solve_with_ip()
print(f"IP solution: {len(solution) - sum(solution)} edges removed out of {len(solution)}")
print(f"Edges kept: {sum(solution)}")
print(f"Fitness: {problem.evaluate_fitness(solution)}")
print(f"Percentage removable: {num_removed / len(solution) * 100:.1f}%")


print("\nRunning full GA:")
solution, fitness = problem.solve_with_ga(
    population_size=100,
    num_generations=200,
)
print(f"GA Result: {solution.count(0)} edges removed, fitness={fitness}")
print(f"IP Optimal: 9 edges removed")
print(f"GA Gap: {abs(solution.count(0) - 9)} edges from optimal")

print("\nVerifying GA solution:")
ga_solution_fitness = problem.evaluate_fitness(solution)
print(f"GA solution fitness (recalculated): {ga_solution_fitness}")
print(f"GA solution: {solution}")
print(f"Sum (edges kept): {sum(solution)}")
print(f"Edges removed: {solution.count(0)}")

# Check if it maintains flow
subgraph = problem._create_subgraph_from_solution(solution)
from edmonds_karp import edmonds_karp
flow = edmonds_karp(subgraph, 0, 9)
print(f"Actual flow with GA solution: {flow}")
print(f"Required flow: {problem.original_max_flow}")
print(f"Flow maintained? {flow >= problem.original_max_flow}")