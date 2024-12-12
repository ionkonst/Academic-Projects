"""
Multivariate RGA developed to test genetic algorithm performance on Heston stochastic volatility model's parameter calibration
"""
import numpy as np
import random as rd
import matplotlib.pyplot as plt

class RealValuedGeneticAlgorithm:
    def __init__(self, objective_function, bounds, population_size, generations, 
                 mutation_rate, crossover_rate, sigma, elitism=1, minmax_fit_percentage=0.1):
        self.f = objective_function
        self.bounds = bounds
        self.N = population_size
        self.generations = generations
        self.Mutation_prob = mutation_rate
        self.Cross_prob = crossover_rate
        self.sigma = sigma
        self.C_MINMAX_FIT_PERCENTAGE = minmax_fit_percentage
        self.bounds = bounds
        self.elitism = elitism
        self.population = self.init_population()
        self.fitness_cache = self.calculate_fitness(self.population)
        self.best_solution = None
        self.best_fitness = float('inf')

    def init_population(self):
        """Initialize a random population within the bounds."""
        return [[rd.uniform(lb, ub) for lb, ub in self.bounds] for _ in range(self.N)]
    
    def calculate_fitness(self, pop):
        """Calculate the fitness for all individuals in the population."""
        return {tuple(ind): self.f(ind) for ind in pop}

    def fitness(self, pop):
        """
        Retrieve or calculate fitness for the population.
        """
        # Find individuals not yet in the fitness cache
        missing_individuals = [ind for ind in pop if tuple(ind) not in self.fitness_cache]
    
        if missing_individuals:
            # Compute fitness for missing individuals and update the cache
            new_fitness = {tuple(ind): self.f(ind) for ind in missing_individuals}
            self.fitness_cache.update(new_fitness)
    
        # Return fitness for all individuals in the population
        return [self.fitness_cache[tuple(ind)] for ind in pop]

    def avg_fitness(self, pop):
        """Calculate the average fitness of the population."""
        return np.mean(self.fitness(pop))
    
    
    
    def min12_fitness(self, pop):
        """Find the 2 chromosomes with the minimum fitness in the population."""
        fit = self.fitness(pop)
        indexed_fit = list(enumerate(fit))
        sorted_fit = sorted(indexed_fit, key=lambda x: x[1])
        min1_value = sorted_fit[0][0]
        min2_value = sorted_fit[1][0]
        
        return [min1_value, min2_value]

    def selection(self, pop):
        """Select a chromosome for reproduction based on fitness."""
        fit = self.fitness(pop)
        max_fit, min_fit = max(fit), min(fit)
        c = self.C_MINMAX_FIT_PERCENTAGE * (max_fit - min_fit)
    
        # Compute inverted fitness
        inv_fit = max_fit + c - np.array(fit)
        inv_fit_sum = inv_fit.sum()
    
        if inv_fit_sum == 0:
            # Handle edge case where all fitness values are identical
            return rd.randint(0, self.N - 1)
    
        # Normalize to probabilities
        prob = inv_fit / inv_fit_sum
    
        # Compute cumulative probabilities
        cum_prob = np.cumsum(prob)
    
        # Select based on random value
        selector = rd.random()
        return np.searchsorted(cum_prob, selector)


    def blend(self, x1, x2):
        """Blend crossover between two chromosomes represented as lists."""
        if rd.random() < self.Cross_prob:
            b1, b2 = rd.random(), rd.random()
            y1 = [b1 * xi1 + (1 - b1) * xi2 for xi1, xi2 in zip(x1, x2)]
            y2 = [b2 * xi1 + (1 - b2) * xi2 for xi1, xi2 in zip(x1, x2)]
        else:
            y1, y2 = x1[:], x2[:]  # Return copies of the original lists
        return y1, y2


    def mutation(self, x):
        """Apply mutation to a chromosome represented as a list."""
        x = [
                x[i] + self.sigma[i] * rd.normalvariate(0, 1) if rd.random() < self.Mutation_prob else x[i]
                for i in range(len(x))
        ]
        # Ensure values stay within bounds
        x = [    
                rd.uniform(self.bounds[i][0], self.bounds[i][1]) if xi < self.bounds[i][0] 
                or xi > self.bounds[i][1] else xi for i, xi in enumerate(x)
        ]
        return x

    def evolve_one_generation(self, pop):
        """Perform selection, crossover, and mutation to evolve one generation."""
        next_gen = []
        selected = []
        if (self.elitism == 1):
            for i in range(self.N - 2):
                selected.append(pop[self.selection(pop)])
                if i % 2 == 1:
                    offspring1, offspring2 = self.blend(selected[i - 1], selected[i])
                    next_gen.append(self.mutation(offspring1))
                    next_gen.append(self.mutation(offspring2))
            next_gen.append(pop[self.min12_fitness(pop)[0]])
            next_gen.append(pop[self.min12_fitness(pop)[1]])
        else:
            for i in range(self.N):
                selected.append(pop[self.selection(pop)])
                if i % 2 == 1:
                    offspring1, offspring2 = self.blend(selected[i - 1], selected[i])
                    next_gen.append(self.mutation(offspring1))
                    next_gen.append(self.mutation(offspring2))
        return next_gen[:self.N]
    
    def run(self):
        """Run the genetic algorithm and return the best solution."""
        pop = self.population
        for gen in range(self.generations):
            if gen % 2 == 0:
                    print(f"gen = {gen}")
            pop = self.evolve_one_generation(pop)
            gen_fitness = self.fitness(pop)
            gen_best = min(gen_fitness)
            gen_best_point = np.argmin(gen_fitness)
            if (gen_best < self.best_fitness):
                self.best_fitness = gen_best
                self.best_solution = pop[gen_best_point]
        
        return self.best_solution, self.best_fitness
    
    def run_report(self):
        """Run the genetic algorithm and return the best solution + reporting"""
        pop = self.population
        for gen in range(self.generations):
            plot_points(pop, self.bounds)
            pop = self.evolve_one_generation(pop)
            gen_fitness = self.fitness(pop)
            gen_best = min(gen_fitness)
            gen_best_point = np.argmin(gen_fitness)
            #print(gen_best) #print for 2d func
            
            if (gen_best < self.best_fitness):
                self.best_fitness = gen_best
                self.best_solution = pop[gen_best_point]

            print(f"Generation {gen + 1} - Best Fitness: {self.best_fitness:.4f}, Avg Fitness: {self.avg_fitness(pop):.4f}")
            print(self.best_solution)
        
        return self.best_solution, self.best_fitness

def RGA1(func, bounds, seed=0):
    #GA parameters
    rd.seed(seed)
    population_size = 200 #even number
    generations = 50
    mutation_rate = 0.2
    crossover_rate = 0.6
    elitism = 1
    sigma = 1
    sigma = [sigma*(i[1]-i[0]) for i in bounds]
    
    RGA = RealValuedGeneticAlgorithm(objective_function=func, bounds=bounds, 
                                    population_size=population_size, generations=generations, 
                                    mutation_rate=mutation_rate, crossover_rate=crossover_rate, elitism=elitism, sigma=sigma)
    
    #Switch to run_report for a detailed run
    return RGA.run()

def plot_points(points, bounds):
    """
    Plot a list of points.

    Args:
    points (list of tuples): Each tuple represents a point (x, y).

    Returns:
    None
    """
    # Unzip the list of points into x and y coordinates
    x_coords, y_coords = zip(*points)

    # Create a scatter plot
    plt.scatter(x_coords, y_coords, color='blue', label='Points')
    
    # Set axis scales based on bounds
    margin = 0.1  # Add a 10% margin around the points
    plt.xlim(bounds[0][0] - margin * (bounds[0][1] - bounds[0][0]), bounds[0][1] + margin * (bounds[0][1] - bounds[0][0]))
    plt.ylim(bounds[1][0] - margin * (bounds[1][1] - bounds[1][0]), bounds[1][1] + margin * (bounds[1][1] - bounds[1][0]))

    # Add labels and title
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Plot of Points")
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.show()

"""
# Define a multi-variable objective function
def f(x):
    return 2 * np.sin(x[0]) + x[0] + 0.1 * (x[0] - 5.5)**2 + np.cos(x[1]) + 0.2 * (x[1] - 2)**2

b = [(-5, 10), (1, 5)]
a = RGA1(f,b,888)
print(a)
"""
