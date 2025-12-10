"""
Generic Simulated Annealing framework for optimization problems.

Based on the algorithm by Kirkpatrick et al. (1983).
"""

import random
import math

class SimulatedAnnealing:
    def __init__(self, problem, initial_temp=100, cooling_rate=0.95, min_temp=1, iterations_per_temp=100):
        """
        Generic Simulated Annealing optimizer.
        
        Args:
            problem: Problem instance implementing required methods
            initial_temp: Starting temperature
            cooling_rate: Temperature reduction factor (0 < rate < 1)
            min_temp: Minimum temperature (stopping criterion)
            iterations_per_temp: Iterations at each temperature level
        """
        self.problem = problem
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.cooling_rate = cooling_rate
        self.iterations_per_temp  = iterations_per_temp

    def optimize(self, seed=None):
        if seed is not None:
            random.seed(seed)

        current = self.problem.get_initial_solution()
        current_fitness = self.problem.evaluate_fitness(current)
        
        best = current
        best_fitness = current_fitness
        
        temp = self.initial_temp
        
        while temp > self.min_temp:
            for _ in range(self.iterations_per_temp):
                neighbor = self.problem.get_neighbor(current)
                
                if not self.problem.is_valid(neighbor):
                    continue
                
                neighbor_fitness = self.problem.evaluate_fitness(neighbor)
                delta = neighbor_fitness - current_fitness
                
                # Metropolis acceptance criterion
                if delta > 0 or random() < math.exp(delta / temp):
                    current = neighbor
                    current_fitness = neighbor_fitness
                    
                    if current_fitness > best_fitness:
                        best = current
                        best_fitness = current_fitness
            
            temp *= self.cooling_rate
        
        return best, best_fitness