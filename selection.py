import copy
import random

from GeneticAlgorithm.population import Population


def apply_roulette_wheel_selection(current_population):
    """Apply roulette wheel selection to the population"""
    # Initialize a new population
    offspring = Population(current_population.size)

    for i in range(current_population.size):
        # Pick a point between 0 and and the total amount of fitness in the
        # population.
        selection_point = random.randint(1, current_population.total_fitness)
        # Keep a running total of how many fitness points have elapsed
        running_total = 0
        j = 0

        # Iterate over each individual until the threshold is met
        while running_total < selection_point:
            running_total += current_population.pop[j].fitness
            j += 1

        offspring.add(copy.deepcopy(current_population.pop[j-1]), i)

    return offspring


def apply_tournament_selection(current_population, population_size, tournament_size):
    """Apply tournament selection to the population to generate offspring"""
    # NOTE: the same individual can be copied multiple times in the new
    # offspring
    offspring = Population(population_size)

    # Generate offspring
    for i in range(population_size):
        # Pick out some random individuals to act as parents
        parents = [current_population.get_random_individual()
                   for p in range(tournament_size)]

        # Store the parent with the best fitness. A copy operation is performed
        # on the fittest individual as it may be referenced multiple times in
        # the population. Changing one occurrence of the individual will change
        # the other occurrences to (unless copied).
        offspring.add(copy.deepcopy(max(parents)), i)

    return offspring
