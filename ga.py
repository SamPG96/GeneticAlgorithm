import random

import matplotlib.pyplot as plt

from GeneticAlgorithm.population import Population
from GeneticAlgorithm.selection import apply_tournament_selection, \
    apply_roulette_wheel_selection

# Bold colors supported by matplotlib
GRAPH_COLORS = [
    "brown",
    "grey",
    "steelblue",
    "purple",
    "black",
    "seagreen",
    "red",
    "darkblue",
]


def graph_ga_sessions_on_single_graph(graph_title, session_results,
                                      session_names, generations):
    """Graphs metric data from running multiple GA sessions onto a single
    graph"""
    rem_colours = GRAPH_COLORS

    for session_result, session_name in zip(session_results, session_names):
        color = rem_colours.pop()

        # Plot the highest fitness detected in each generation
        plt.plot(range(generations),
                 session_result["highest_fit_per_gen"],
                 label=session_name + " - highest fitness of population",
                 color=color)

        # This will plot the mean fitness of the population at each generation
        plt.plot(range(generations),
                 session_result["mean_fit_per_gen"],
                 label=session_name + " - mean fitness of population",
                 linestyle="--",
                 color=color)

    plt.title(graph_title)
    plt.gca().legend(loc=9,
                     bbox_to_anchor=(0.5, -0.075),
                     ncol=len(session_results))
    plt.xlabel("Generations")
    plt.ylabel('Fitness')
    plt.grid()

    plt.show()


def run_genetic_algorithm_multiple(runs=None,
                                   generations=None,
                                   population_size=None,
                                   chromosome_size=None,
                                   selection_method=None,
                                   tournament_size=None,
                                   single_point_crossover_point=None,
                                   two_point_crossover_points=None,
                                   mutation_rate=None,
                                   indiv_class=None,
                                   indiv_additional_args=[],
                                   supress_all_out=False,
                                   suppress_run_out=True):
    """Run the GA multiple times with a problem in order to get averaged
    results, as a GA is a stochastic algorithm."""
    # Stores statistics for running the GA multiple times
    all_runs_stats = {
        "runs": [None] * runs,
        "avg_run": {
            "total_fit_per_gen": [None] * generations,
            "highest_fit_per_gen":[None] * generations,
            "mean_fit_per_gen": [None] * generations
        },
        "summary": {}
    }

    # The following structures store specific data to help summarize all runs.

    # Fittest individual from each run
    fittest = [None] * runs
    # Mean fitness of the final population from each run
    mean_fitness_of_final_gens = []
    # Total fitness of the final population from each run
    total_fitness_of_final_gens = []

    # Repeatedly run the GA for a set number of 'runs'
    for i in range(runs):
        if supress_all_out is False:
            print("\n\nRUN: " + str(i))

        # Run the GA
        fittest[i], all_runs_stats["runs"][i] = run_genetic_algorithm_once(
            generations,
            population_size,
            chromosome_size,
            selection_method,
            tournament_size,
            mutation_rate,
            indiv_class,
            single_point_crossover_point=single_point_crossover_point,
            two_point_crossover_points=two_point_crossover_points,
            indiv_additional_args=indiv_additional_args,
            suppress_out=suppress_run_out)

        # Store the mean fitness of the population at the final generation
        mean_fitness_of_final_gens.append(all_runs_stats["runs"][i][
                                              "mean_fit_per_gen"][-1])

        # Store the total fitness of the population at the final generation
        total_fitness_of_final_gens.append(all_runs_stats["runs"][i][
                                               "total_fit_per_gen"][-1])

        if supress_all_out is False:
            print("\n\nHighest fitness of run: " + str(fittest[i].fitness))
            print("Mean fitness at end of run: " +
                  str(all_runs_stats["runs"][i]["mean_fit_per_gen"][-1]))
            print("Total fitness at end of run: " +
                  str(all_runs_stats["runs"][i]["total_fit_per_gen"][-1]))

    # Average the core metrics over each run for each generation
    for g in range(generations):
        total_total_fitness = 0
        total_highest_fit = 0
        total_mean_fit = 0

        # Total up each of the core metrics of the current generation
        for r in range(runs):
            total_total_fitness += all_runs_stats["runs"][r][
                "total_fit_per_gen"][g]
            total_highest_fit += all_runs_stats["runs"][r][
                "highest_fit_per_gen"][g]
            total_mean_fit += all_runs_stats["runs"][r][
                "mean_fit_per_gen"][g]

        # Average the totals over the number of runs
        all_runs_stats["avg_run"]["total_fit_per_gen"][g] = \
            total_total_fitness / runs

        all_runs_stats["avg_run"]["highest_fit_per_gen"][g] = \
            total_highest_fit / runs

        all_runs_stats["avg_run"]["mean_fit_per_gen"][g] = \
            total_mean_fit / runs

    # Store summary information
    all_runs_stats["summary"] = {
        "fittest_of_all_runs": max(fittest, key=lambda indiv: indiv.fitness),
        "avg_highest_fitness_of_final_gen": (sum(i.fitness for i in fittest) /
                                             runs),
        "avg_mean_fitness_of_final_gen": sum(mean_fitness_of_final_gens) / runs,
        "avg_total_fitness_of_final_gen": (sum(total_fitness_of_final_gens) /
                                           runs)
    }

    if supress_all_out is False:
        print("\n\n\nComplete!\n")
        print("Stats from completing " + str(runs) + " GA runs")
        print("=" * 32)
        print("Highest fitness: " +
              str(all_runs_stats["summary"]["fittest_of_all_runs"].fitness))
        print("Average highest fitness: " +
              str(all_runs_stats["summary"]["avg_highest_fitness_of_final_gen"]))
        print("Average mean fitness of all final generations: " +
              str(all_runs_stats["summary"]["avg_mean_fitness_of_final_gen"]))
        print("Average total fitness of all final generations: " +
              str(all_runs_stats["summary"]["avg_total_fitness_of_final_gen"]))

    return all_runs_stats["summary"]["fittest_of_all_runs"], all_runs_stats


def run_genetic_algorithm_once(generations,
                               population_size,
                               chromosome_size,
                               selection_method,
                               tournament_size,
                               mutation_rate,
                               indiv_class,
                               single_point_crossover_point=None,
                               two_point_crossover_points=None,
                               indiv_additional_args=[],
                               suppress_out=True):
    """
    Execute the genetic algorithm (GA). The GA requires the following core
    parameters:
        Crossover point         The point at which the tail of two individuals
                                are swapped.
        Gene size               The size of a chromosome for an individual.
        Generations             Number of generations to run GA for.
        Mutation rate           The probability a bit in a chromosome is flipped.
                                This is typically between between
                                1/population_size and 1/chromosome_length.
        Parent quantity         The number of individuals to pick at random
                                from the current population to act as parents.
        Population size         The sample population size.
    """
    run_stats = {
        "highest_fit_per_gen": [],
        "mean_fit_per_gen": [],
        "total_fit_per_gen": []
    }

    if population_size % 2 != 0:
        raise Exception("Population size is not an even number")

    population = Population(population_size)

    # Generate the initial population
    population.generate_population(chromosome_size,
                                   indiv_class,
                                   indiv_additional_args)

    # Initialize the fittest individual to be the first individual in the
    # randomly generated population.
    fittest_indiv_for_run = population.pop[0]

    # Evaluate fitness of initial population
    population.evaluate_fitness()

    # Store stats for generation 0
    run_stats["total_fit_per_gen"].append(
        population.total_fitness)
    run_stats["highest_fit_per_gen"].append(
        population.get_fittest().fitness)
    run_stats["mean_fit_per_gen"].append(
        population.get_mean_fittest())

    if suppress_out is False:
        print("Total fitness of initial population: " +
              str(population.total_fitness))

    # Run the GA for a set amount of generations
    for i in range(1, generations):
        if suppress_out is False:
            print("\nGeneration: " + str(i))

        if selection_method == "tournament":
            # Use tournament selection on the population to generate a new
            # population
            population = apply_tournament_selection(population,
                                                    population_size,
                                                    tournament_size)

        else:
            assert selection_method == "roulette"
            # Use roulette selection on the population to generate a new
            # population
            population = apply_roulette_wheel_selection(population)

        # Use the required selection method
        if single_point_crossover_point is not None:
            # Apply one step crossover to the population
            population.apply_one_step_crossover(single_point_crossover_point)

        else:
            # Apply two step crossover to the population
            population.apply_two_step_crossover(*two_point_crossover_points)

        # Apply bit wise mutation
        population.apply_mutation(mutation_rate)

        # Evaluate fitness after generation
        population.evaluate_fitness()

        # Store generation stats
        run_stats["total_fit_per_gen"].append(
            population.total_fitness)
        run_stats["highest_fit_per_gen"].append(
            population.get_fittest().fitness)
        run_stats["mean_fit_per_gen"].append(
            population.get_mean_fittest())

        # If their is a fitter individual in the current generation,
        # then store it. This should be the case most of the time.
        if population.get_fittest().fitness >= fittest_indiv_for_run.fitness:
            fittest_indiv_for_run = population.get_fittest()

        if suppress_out is False:
            print("Total population fitness after generation: " +
                  str(population.total_fitness))
            print("Highest fitness: " + str(population.get_fittest().fitness))
            print("Mean fitness: " + str(population.get_mean_fittest()))

    return fittest_indiv_for_run, run_stats
