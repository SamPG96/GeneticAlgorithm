"""
All functionality for rule base problem representations.
"""

import numpy as np
import random

from GeneticAlgorithm.population import Individual


def is_gene_condition(gene_position, condition_size, result_size):
    """Determines if the gene resides in the condition part of the chromosome"""
    # TODO: handle when result size is greater than 1
    assert result_size == 1

    return ((gene_position + 1) % (condition_size + result_size)) != 0


def print_rule_set(individual, condition_size, result_size):
    """Output the rule set of an individual"""
    print("\n\nRULE SET")
    for rule_num, i in enumerate(np.arange(0, individual.chromosome_size,
                                           condition_size + 1)):
        print(("RULE {num}:\n"
               "condition -> {cond}"
               "   result -> {result}").format(
            num=rule_num,
            cond=individual.chromosome[i: i + condition_size],
            result=individual.chromosome[i + condition_size:
                                         i + condition_size + result_size]))


def print_rule_set_with_tolerance(individual, condition_size, result_size):
    """Output the rule set of an individual which has tolerance in the
    condition part."""
    print("\n\nRULE SET")
    for rule_num, i in enumerate(np.arange(0, individual.chromosome_size,
                                           condition_size + 1)):
        cond_raw = individual.chromosome[i: i + condition_size]

        cond_str = ""
        for cond_i in range(len(cond_raw)):
            if cond_i % 2 == 0:
                cond_str += str(cond_raw[cond_i])

            else:
                cond_str += "(t:" + str(cond_raw[cond_i]) + ")"

        print(("RULE {num}:\n"
               "condition -> {cond}"
               "   result -> {result}").format(
            num=rule_num,
            cond=cond_str,
            result=individual.chromosome[i + condition_size:
                                         i + condition_size + result_size]))


def test_rule_set(ruleset_indv):
    """Tests how good a rule set is against some testing rules"""
    train_fitness = ruleset_indv.fitness
    max_train_fitness = (ruleset_indv.train_to_rule -
                         ruleset_indv.train_from_rule + 1)
    train_pass_rate = (train_fitness / max_train_fitness) * 100

    test_fitness = ruleset_indv.test_fitness()
    max_test_fitness = (ruleset_indv.test_to_rule -
                        ruleset_indv.test_from_rule + 1)
    test_pass_rate = (test_fitness / max_test_fitness) * 100

    total_success = (train_pass_rate + test_pass_rate) / 2

    return total_success, train_fitness, train_pass_rate, test_fitness, \
           test_pass_rate


def read_data_file(tfile, condition_format, start_at_rule, stop_at_rule):
    """A generator that yields every condition and result in a training file
    from a rule number to another rule number. By default this function will
    read the entire file."""
    with open(tfile, "r") as f:
        # Skip straight to the desired rule in the file.
        for l in range(start_at_rule - 1):
            next(f)

        rule_num = start_at_rule

        # Yield each condition and result until the required amount of rules
        # have been met OR the end of file has been reached
        while rule_num <= stop_at_rule:
            line = next(f)

            # The way in which a line in a data file is parsed is dependant
            # on whether the condition is built up of binary values or
            # floating point numbers.
            if condition_format == "binary":
                # Condition is a continuous string of 1s and 0s. It is
                # separated from the result by a single space.
                cond_raw, result_raw = line.split(" ")

                # Conditions are made up of int variables, so cast each
                # variable from strings to ints.
                cond = [int(c) for c in cond_raw if c != "\n"]
                result = [int(r) for r in result_raw if r != "\n"]

            elif condition_format == "float":
                # Each variable in the condition is separated by a space. The
                # result is always at the end of the line.
                params = line.split(" ")

                # Conditions are made up of float variables, so cast each
                # variable from strings to floats.
                cond = [float(c) for c in params[:len(params) - 1] if c != "\n"]
                result = [int(r) for r in params[-1] if r != "\n"]

            else:
                raise Exception("condition format net recognised: " +
                                condition_format)

            # Move to next line/rule for the next iteration
            rule_num += 1
            yield cond, result


def do_rules_match_exactly(rule1, rule2, wildcard_enabled):
    """Checks if two rules match. The check can account for wildcards"""
    for bit1, bit2 in zip(rule1, rule2):
        if wildcard_enabled is True and (bit1 == '#' or bit2 == '#'):
            # At least one gene is a wildcard, so the genes 'match'
            continue

        elif bit1 != bit2:
            # Rule does not match
            return False

    return True


def do_rules_match_with_tolerance(rule1_with_tolerance, rule2):
    """Checks if two rules match within a given amount of tolerance"""
    for i in range(len(rule2)):
        gene1 = rule1_with_tolerance[i]
        gene1_tolerance = rule1_with_tolerance[(i * 2) + 1]
        gene2 = rule2[i]

        upper_bound = gene1 + gene1_tolerance
        lower_bound = gene1 - gene1_tolerance

        if gene2 < lower_bound or gene2 > upper_bound:
            return False

    return True


class BinaryRuleSet(Individual):
    """A candidate solution (or individual) in the population for a rule set
    that uses binary conditionals."""
    def __init__(self, chromosome_size, condition_size, result_size,
                 training_file, train_from_rule, train_to_rule,
                 wildcard_enabled):
        self.condition_size = condition_size
        self.result_size = result_size
        self.training_file = training_file
        self.train_from_rule = train_from_rule
        self.train_to_rule = train_to_rule
        self.wildcard_enabled = wildcard_enabled

        super(BinaryRuleSet, self).__init__(chromosome_size)

    def generate(self):
        """Generate a chromosome of random binary values"""
        chromosome = [None] * self.chromosome_size

        for n in range(self.chromosome_size):
            # If the bit is part of a conditional then it can also be a
            # wildcard as well as a 0 or 1 (if wildcards are enabled.
            if self.wildcard_enabled is True and is_gene_condition(
                    n,
                    self.condition_size,
                    self.result_size) is True:
                opts = [0, 1, '#']

            else:
                opts = [0, 1]

            # Generate a random chromosome based on the options available
            chromosome[n] = opts[random.randint(0, len(opts) - 1)]

        return chromosome

    def calculate_fitness(self):
        """Fitness function that calculates the fitness of this individual.
        The more conditions and results in the individual that match the
        training set, the fitter the individual."""
        new_fitness = 0

        # Check each training condition and result
        for training_cond, training_result in read_data_file(
                self.training_file, "binary", self.train_from_rule,
                self.train_to_rule):
            gene_result = None

            # Find the current training condition in the rule set of this
            # individual
            for i in np.arange(0, self.chromosome_size,
                               self.condition_size + 1):
                gene_condition = self.chromosome[i: i + self.condition_size]

                if do_rules_match_exactly(gene_condition,
                                          training_cond,
                                          self.wildcard_enabled) is True:
                    # Condition found, store the genes result fot the condition
                    gene_result = self.chromosome[i + self.condition_size:
                                                  i + self.condition_size +
                                                  self.result_size]

                    # Ensure the result does not contain a wildcard
                    assert '#' not in gene_result
                    break

            # Increase fitness if the result in the chromosome is the same as
            # the result in the training set.
            if gene_result == training_result:
                new_fitness += 1

        return new_fitness

    def mutate(self, mutation_rate):
        """Apply bit wise mutation to the rules which allow the option of
        mutating to a wildcard"""
        # If wildcards are enabled then the gene has the chance to mutate to one
        if self.wildcard_enabled is True:
            valid_mutation_choices = [0, 1, '#']

        else:
            valid_mutation_choices = [0, 1]

        for i in range(self.chromosome_size):
            # Generate a random number between 0 and 1, if its less than the
            # mutation rate then the bit is mutated
            val = random.uniform(0.0, 1.0)

            if val <= float(mutation_rate):
                # The way the gene is mutated is dependant on if the gene is
                # part of a condition or result.
                if is_gene_condition(i, self.condition_size,
                                     self.result_size) is True:
                    # Generate the options the bit can mutate to, the bit MUST
                    # mutate to a different value
                    opts = [o for o in valid_mutation_choices
                            if o != self.chromosome[i]]

                    # Mutate to one of the options at random.
                    self.chromosome[i] = opts[random.randint(0, len(opts) - 1)]

                else:
                    # Flip bit
                    if self.chromosome[i] == 0:
                        self.chromosome[i] = 1

                    else:
                        assert self.chromosome[i] == 1
                        self.chromosome[i] = 0


class FloatRuleSet(Individual):
    """A candidate solution (or individual) in the population for a rule set
    that uses real conditionals values."""
    def __init__(self, chromosome_size, condition_size, result_size,
                 mutation_creep, training_file, train_from_rule, train_to_rule,
                 test_file, test_from_rule, test_to_rule):
        self.condition_size = condition_size
        self.result_size = result_size
        self.mutation_creep = mutation_creep

        self.training_file = training_file
        self.train_from_rule = train_from_rule
        self.train_to_rule = train_to_rule

        self.test_file = test_file
        self.test_from_rule = test_from_rule
        self.test_to_rule = test_to_rule

        super(FloatRuleSet, self).__init__(chromosome_size)

    def generate(self):
        """Generate a chromosome of random float values"""
        chromosome = [None] * self.chromosome_size

        for n in range(self.chromosome_size):
            # A gene that is part of a condition should have a float randomly
            # generated between 0.000001 and 0.999999. A gene that is part of a
            # result should have a random binary value set.
            if is_gene_condition(n,
                                 self.condition_size,
                                 self.result_size) is True:
                chromosome[n] = round(random.uniform(0.000001, 0.999999), 6)

            else:
                chromosome[n] = random.randint(0, 1)

        return chromosome

    def calculate_fitness(self):
        """Fitness function that calculates the fitness of this individual.
        The more conditions and results in the individual that match the
        training set with a tolerance, the fitter the individual."""
        return self.evaluate_rule_set_against_data(self.training_file,
                                                   self.train_from_rule,
                                                   self.train_to_rule)

    def test_fitness(self):
        """Tests the fitness of the individual against a test set of data"""
        return self.evaluate_rule_set_against_data(self.test_file,
                                                   self.test_from_rule,
                                                   self.test_to_rule)

    def mutate(self, mutation_rate):
        """Apply mutation to the rules. If mutation is to occur on a gene
        then a value is randomly generated between a range to determine how
        much to 'creep' around the current value of the gene"""
        for i in range(self.chromosome_size):
            # Generate a random number between 0 and 1, if its less than the
            # mutation rate then the bit is mutated
            val = random.uniform(0.0, 1.0)

            if val <= float(mutation_rate):
                # The way the gene is mutated is dependant on if the gene is
                # part of a condition or result.
                if is_gene_condition(i, self.condition_size, self.result_size) \
                        is True:
                    # Mutate the gene by +/- a random value in the mutation
                    # range
                    curr_value = self.chromosome[i]
                    new_value = random.uniform(curr_value - self.mutation_creep,
                                               curr_value + self.mutation_creep)

                    # Handle when value is out of bounds
                    if new_value < 0.0:
                        new_value = 0

                    elif new_value > 1.0:
                        new_value = 1

                else:
                    # Mutate binary result
                    if self.chromosome[i] == 1:
                        new_value = 0

                    else:
                        assert self.chromosome[i] == 0
                        new_value = 1

                self.chromosome[i] = new_value

    def evaluate_rule_set_against_data(self, data_file, eval_from_rule,
                                       eval_to_rule):
        """Evaluate the fitness of this rule set based on a section in a
        training file"""
        fitness = 0

        # Check each training condition and result
        for train_cond, train_result in read_data_file(data_file,
                                                       "float",
                                                       eval_from_rule,
                                                       eval_to_rule):
            gene_result = None

            # Find the current training condition in the rule set of this
            # individual
            for i in np.arange(0, self.chromosome_size,
                               self.condition_size + 1):
                gene_condition = self.chromosome[i: i + self.condition_size]

                if do_rules_match_with_tolerance(gene_condition,
                                                 train_cond) is True:
                    # Condition found, store the genes result fot the condition
                    gene_result = self.chromosome[i + self.condition_size:
                                                  i + self.condition_size +
                                                  self.result_size]

                    break

            # Increase fitness if the result in the chromosome is the same as
            # the result in the training set.
            if gene_result == train_result:
                fitness += 1

        return fitness
