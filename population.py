import random

import numpy as np


class Population(object):
    """Represents a collection of candidate solutions in a problem space"""
    def __init__(self, size):
        self.size = size
        self.pop = [None] * size

        self.total_fitness = 0

    def add(self, individual, pos):
        """Add an individual to the population"""
        # Ensure no individuals exist at the position
        assert self.pop[pos] is None
        self.pop[pos] = individual

    def apply_one_step_crossover(self, single_point_crossover_point):
        """Apply single point crossover to the population"""
        # Ensure the population consists of an even number of individuals.
        # Crossover requires an even population.
        assert self.size % 2 == 0

        for i in range(self.size - 1):
            # Ensure the cross over point is less than the size of the
            # chromosome
            assert single_point_crossover_point < self.pop[i].chromosome_size
            assert single_point_crossover_point < self.pop[i+1].chromosome_size

            tail1 = self.pop[i].chromosome[single_point_crossover_point:]
            tail2 = self.pop[i + 1].chromosome[single_point_crossover_point:]
            self.pop[i].chromosome[single_point_crossover_point:] = tail2
            self.pop[i + 1].chromosome[single_point_crossover_point:] = tail1

    def apply_two_step_crossover(self, point1, point2):
        """Apply two step point crossover to the population"""
        # Ensure the population consists of an even number of individuals.
        # Crossover requires an even population.
        assert self.size % 2 == 0

        for i in np.arange(0, self.size, 2):
            # Ensure the max cross over point is less than the size of the
            # chromosome
            assert max([point1, point2]) < self.pop[i].chromosome_size
            assert max([point1, point2]) < self.pop[i + 1].chromosome_size

            # Extract the genes between the two points in each parent
            mid1 = self.pop[i].chromosome[point1:point2]
            mid2 = self.pop[i + 1].chromosome[point1:point2]

            # Swap the middle points
            self.pop[i].chromosome[point1:point2] = mid2
            self.pop[i + 1].chromosome[point1:point2] = mid1

    def apply_mutation(self, mutation_rate):
        """Apply bit wise mutation to the population"""
        for indiv in self.pop:
            indiv.mutate(mutation_rate)

    def generate_population(self, chromosome_size, indiv_class,
                            indiv_additional_args):
        """Generate a population. Each individual is generated as defined by
        the specific 'Individual' class"""
        for i in range(self.size):
            # Generate an individual with its core arguments and arguments
            # specific to the problem representation.
            self.pop[i] = indiv_class(chromosome_size, *indiv_additional_args)

    def evaluate_fitness(self):
        """Calculates and sets the fitness of the entire population"""
        # Reset total fitness
        self.total_fitness = 0

        for individual in self.pop:
            fitness = individual.calculate_fitness()
            individual.fitness = fitness
            self.total_fitness += fitness

    def get_fittest(self):
        """Returns the fittest individual"""
        fittest = self.pop[0]

        for i in self.pop:
            if i.fitness > fittest.fitness:
                fittest = i

        return fittest

    def get_random_individual(self):
        """Return a random individual from the population"""
        return self.pop[random.randint(0, self.size - 1)]

    def get_mean_fittest(self):
        """Get the mean fitness of the population"""
        return float(self.total_fitness) / float(self.size)


class Individual(object):
    """Represents and individual in the population. This acts as an abstract
    class."""
    def __init__(self, chromosome_size):
        self.chromosome_size = chromosome_size
        # A chromosome consists of binary values
        self.chromosome = self.generate()
        self.fitness = 0

    def __gt__(self, other):
        return self.fitness > other.fitness

    def calculate_fitness(self):
        """Calculates the fitness of the individual, which is specific to a
        problem"""
        raise NotImplementedError

    def generate(self):
        """Generates a chromosome specific to a problem"""
        raise NotImplementedError

    def mutate(self, mutation_rate):
        """Apply bit wise mutation to the population"""
        for i in range(self.chromosome_size):
            # Generate a random number between 0 and 1, if its less than the
            # mutation rate then flip the bit of the chromosome
            val = random.uniform(0.0, 1.0)

            if val <= float(mutation_rate):
                # Flip bit
                if self.chromosome[i] == 0:
                    self.chromosome[i] = 1

                else:
                    assert self.chromosome[i] == 1
                    self.chromosome[i] = 0
