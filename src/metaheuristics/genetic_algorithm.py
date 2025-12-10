"""
Generic Genetic Algorithm framework for optimization problems.

Based on Holland's Genetic Algorithm (1975) with tournament selection and elitism strategy.
"""

import random

class GeneticAlgorithm:
    def __init__(self, problem, population_size=50, num_generations=100, 
                 elite_ratio=0.1, mutation_rate=0.05, crossover_rate=0.8):
        """
        Generic Genetic Algorithm optimizer.
        
        Args:
            problem: Problem instance implementing required methods
            population_size: Number of solutions in population
            num_generations: Number of evolutionary cycles
            elite_ratio: Fraction of top solutions to preserve (0.0 to 1.0)
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover 
        """
        self.problem = problem
        self.population_size = population_size
        self.num_generations = num_generations
        self.elite_ratio = elite_ratio
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
    
    def optimize(self, seed=None):
        if seed is not None:
            random.seed(seed)
        
        # Initialize population
        population = []
        for _ in range(self.population_size):
            individual = self.problem.get_random_solution()
            fitness = self.problem.evaluate_fitness(individual)
            population.append((individual, fitness))
        
        for generation in range(self.num_generations):
            # Sort by fitness (best first)
            population.sort(key=lambda x: x[1], reverse=True)
                        
            # Calculate how many elites to keep
            n_elites = max(1, int(self.elite_ratio * len(population)))
            elites = population[:n_elites]
            
            offspring = []
            while len(offspring) < self.population_size - n_elites:
                # Select two parents
                parent1 = self._select_parent(population)
                parent2 = self._select_parent(population)
                
                # Crossover
                child = self._crossover(parent1, parent2)
                
                # Mutate
                child = self._mutate(child)
                
                # Evaluate fitness
                fitness = self.problem.evaluate_fitness(child)
                
                # Add to offspring
                offspring.append((child, fitness))
            
            new_population = elites + offspring
            population = new_population

        population.sort(key=lambda x: x[1], reverse=True)
        best_individual, best_fitness = population[0]
        return best_individual, best_fitness
    
    def _select_parent(self, population, tournament_size=3):
        """
        Select one parent using tournament selection.
        """
        tournament = random.sample(population, tournament_size)
        winner = max(tournament, key=lambda x: x[1])  # individual with max fitness
        return winner[0]

    def _crossover(self, parent1, parent2):
        """
        Create offspring by crossing over two parents.
        """
        if random.random() < self.crossover_rate:
            return self.problem.crossover(parent1, parent2)
        else:
            # Return a copy of one of the parents
            return random.choice([parent1, parent2])
    def _mutate(self, individual):
        """
        Apply mutation to an individual.
        """
        if random.random() < self.mutation_rate:
            return self.problem.mutation(individual)
        else:
            return individual