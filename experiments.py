"""
Functions to aid experimenting with GA parameters.
"""

import csv
import time

import matplotlib.pyplot as plt

from GeneticAlgorithm.ga import run_genetic_algorithm_multiple


AVG_RUNS_DIR = "avg_runs/"
FIT_VAR_DIR = "fitness_variation/"
TEST_RESULTS_DIR = "tests/"


def graph_varied_setting_experiments(all_experiments_stats, graph_dir):
    """Generate a graph to represent the results of each experiment given for
    varied settings"""

    def plot_metric(x, y, label, colour):
        """Plot metric on the graph with a trend line"""
        plt.scatter(x, y)
        plt.plot(x,
                 y,
                 label=label,
                 color=colour,
                 linestyle="--")

    high_fit_colour = "blue"
    avg_high_fit_colour = "orange"
    mean_fitness_colour = "green"

    # Create a graph for each experiment
    for i, exp_stats in enumerate(all_experiments_stats):
        graph_file = (graph_dir + exp_stats["exp_title"] + time.strftime(
            " %d-%b-%Y_%H-%M-%S") + ".png")

        plt.figure(i)

        # Plot the highest fitness detected for each setting configuration
        plot_metric(exp_stats["varied_settings"],
                    exp_stats["highest_fit"],
                    "Highest fitness",
                    high_fit_colour)

        # This will plot the average fitness of all the fittest individuals
        # found from all runs for each setting configuration.
        plot_metric(exp_stats["varied_settings"],
                    exp_stats["avg_high_fit_last_gen"],
                    "Average highest fitness of last generation",
                    avg_high_fit_colour)

        # The GA returns the mean fitness of the entire population at the point
        # of the termination condition being met. The experiment runs the GA
        # multiple times for each setting configuration. The mean fitness's from
        # all runs for a particular setting configuration is averaged. Each
        # averaged result for each setting is plotted on the graph.
        plot_metric(exp_stats["varied_settings"],
                    exp_stats["avg_mean_fit_last_gen"],
                    "Mean fitness of last generation",
                    mean_fitness_colour)

        plt.title(exp_stats["exp_title"])
        plt.gca().legend()
        plt.xlabel(exp_stats["varied_setting_name"])
        plt.ylabel('Fitness')
        plt.grid()

        plt.savefig(graph_file)

    # Displaying graphs will block
    # plt.show()


def write_experiment_data_to_file(all_experiments_stats, results_dir):
    """Write data collected from experiment to file"""
    fit_var_headings = [
        "Varied setting",
        "Highest fitness",
        "Avg high fitness last generation",
        "Avg mean fitness last generation"
    ]

    core_headings_for_each_avg_run = [
        "Highest fitness",
        "Mean fitness"
    ]

    test_headings = [
        "Varied setting",
        "Training fitness",
        "Training pass rate",
        "Test fitness",
        "Test pass rate",
        "Overall"
    ]

    for exp_stats in all_experiments_stats:
        fit_var_data_file = (results_dir + FIT_VAR_DIR + exp_stats[
            "exp_title"] + time.strftime(" %d-%b-%Y_%H-%M-%S") + ".csv")

        avg_run_data_file = (results_dir + AVG_RUNS_DIR + exp_stats[
            "exp_title"] + time.strftime(" %d-%b-%Y_%H-%M-%S") + ".csv")

        test_results_file = (results_dir + TEST_RESULTS_DIR + exp_stats[
            "exp_title"] + time.strftime(" %d-%b-%Y_%H-%M-%S") + ".csv")

        # Write stats for fitness variation for each setting value
        with open(fit_var_data_file, 'w', newline='') as dfile:
            data = list(zip(*[exp_stats["varied_settings"],
                              exp_stats["highest_fit"],
                              exp_stats["avg_high_fit_last_gen"],
                              exp_stats["avg_mean_fit_last_gen"]]))

            rows = [fit_var_headings]
            rows.extend(data)

            writer = csv.writer(dfile)
            writer.writerows(rows)

        # Write stats for the average run for each setting variant to a file
        with open(avg_run_data_file, 'w', newline='') as dfile:
            headings = []
            data = []

            for si, varied_setting in enumerate(exp_stats["varied_settings"]):
                if type(varied_setting) is list:
                    # convert list to a readable string
                    setting_str = "-".join([str(s) for s in varied_setting])

                else:
                    setting_str = str(varied_setting)

                headings.extend([setting_str + " - " + h for h in
                                 core_headings_for_each_avg_run])

                data.append(exp_stats["avg_stats_per_gen"][si][
                                "highest_fit_per_gen"])
                data.append(exp_stats["avg_stats_per_gen"][si][
                                "mean_fit_per_gen"])

            rows = [headings]
            rows.extend(list(zip(*data)))
            writer = csv.writer(dfile)
            writer.writerows(rows)

        # Write testing stats if testing was done
        with open(test_results_file, 'w', newline='') as dfile:
            data = list(zip(*[exp_stats["varied_settings"],
                              exp_stats["test_results"]["train_fitness"],
                              exp_stats["test_results"]["train_pass_rate"],
                              exp_stats["test_results"]["test_fitness"],
                              exp_stats["test_results"]["test_pass_rate"],
                              exp_stats["test_results"]["overall_success"]]))

            rows = [test_headings]
            rows.extend(data)

            writer = csv.writer(dfile)
            writer.writerows(rows)


def run_varied_setting_experiment(exp_title, settings, varied_setting,
                                  test_func=None):
    """Executes the GA multiple times for each setting variance"""
    best_individual = None
    best_individual_settings = None

    # Store the stats of the experiment
    exp_stats = {
        "exp_title": exp_title,
        "varied_setting_name": varied_setting,
        "varied_settings": settings[varied_setting],
        "highest_fit": [],
        "avg_high_fit_last_gen": [],
        "avg_mean_fit_last_gen": [],
        "avg_stats_per_gen": [],
        "test_results": {
            "train_fitness": [],
            "train_pass_rate": [],
            "test_fitness": [],
            "test_pass_rate": [],
            "overall_success": []
        }
    }

    print("\n\n\n\nEXPERIMENT: " + exp_title)
    # Iterate over each variance of a setting and run the GA multiple times
    # for it.
    for i, curr_varied_setting_value in enumerate(settings[varied_setting]):
        print("\nTrying \'{setting}\' as {val}".format(
            setting=varied_setting,
            val=curr_varied_setting_value))

        # Make a copy of the settings and replace the varied settings value
        # with the current value of the experiment, as in the initial settings
        # this will be a list.
        current_exp_settings = dict(settings)
        current_exp_settings[varied_setting] = curr_varied_setting_value

        # Check if the crossover point is represented as a decimal. If it is
        # then the crossover point is given as a percentage of the genes size.
        if (current_exp_settings.get("single_point_crossover_point") is not None and
                current_exp_settings["single_point_crossover_point"] < 1):
            current_exp_settings["single_point_crossover_point"] = \
                int(current_exp_settings["single_point_crossover_point"] *
                    current_exp_settings["chromosome_size"])

        # Check if the mutation rate has been set as a percentage between
        # 1/population_size and 1/chromosome_length. Use the percentage to get
        # the actual mutation rate between the range.
        if current_exp_settings.get("mutation_rate_percent") is not None:
            diff = abs((1 / float(current_exp_settings["population_size"])) -
                       (1 / float(current_exp_settings["chromosome_size"])))
            lower_value = min([1 / current_exp_settings["population_size"],
                               1 / current_exp_settings["chromosome_size"]])
            mutation_rate = lower_value + (diff * current_exp_settings.get(
                "mutation_rate_percent"))
            current_exp_settings["mutation_rate"] = mutation_rate
            del current_exp_settings["mutation_rate_percent"]

        bst_exp_individual, stats = run_genetic_algorithm_multiple(
            supress_all_out=False,
            **current_exp_settings)

        # Update experiment stats with the data from running the GA with the
        # current settings.
        exp_stats["highest_fit"].append(bst_exp_individual.fitness)

        exp_stats["avg_high_fit_last_gen"].append(
            stats["summary"]["avg_highest_fitness_of_final_gen"])

        exp_stats["avg_mean_fit_last_gen"].append(
            stats["summary"]["avg_mean_fitness_of_final_gen"])

        # Average all stats from all runs on this experiment and store them
        exp_stats["avg_stats_per_gen"].append(stats["avg_run"])

        # If the experiment uncovered a fitter individual, overwrite previous.
        if (best_individual is None or bst_exp_individual.fitness >
                best_individual.fitness):
            best_individual = bst_exp_individual
            best_individual_settings = current_exp_settings

        print("\tHighest fitness: " +
              str(bst_exp_individual.fitness))
        print("\tAverage highest fitness at final generation: " +
              str(stats["summary"]["avg_highest_fitness_of_final_gen"]))
        print("\tAverage mean fitness at final generation: " +
              str(stats["summary"]["avg_mean_fitness_of_final_gen"]))
        print("\tAverage total fitness at final generation: " +
              str(stats["summary"]["avg_total_fitness_of_final_gen"]))

        # Test the best individual from the experiment against a data set.
        # Stats are stored from running the test.
        if test_func is not None:
            print("Testing best individual ...")

            success, train_fit, train_pass_rate, test_fit, test_pass_rate = \
                test_func(bst_exp_individual)

            print("Training fitness: " + str(train_fit))
            print("Training pass rate: " + str(train_pass_rate) + "%")
            print("Testing fitness: " + str(test_fit))
            print("Testing pass rate: " + str(test_pass_rate) + "%")
            print("Overall success rate: " + str(success) + "%")

            exp_stats["test_results"]["train_fitness"].append(train_fit)
            exp_stats["test_results"]["train_pass_rate"].append(train_pass_rate)
            exp_stats["test_results"]["test_fitness"].append(test_fit)
            exp_stats["test_results"]["test_pass_rate"].append(test_pass_rate)
            exp_stats["test_results"]["overall_success"].append(success)

    return best_individual, best_individual_settings, exp_stats
