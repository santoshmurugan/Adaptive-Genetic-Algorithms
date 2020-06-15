# -*- coding: utf-8 -*-
"""
    pyeasyga module

"""

import random
import copy
from operator import attrgetter
import numpy as np


class ModifiedGeneticAlgorithm(object):
    """Genetic Algorithm class.

    This is the main class that controls the functionality of the Genetic
    Algorithm.

    A simple example of usage:

    >>> # Select only two items from the list and maximise profit
    >>> from pyeasyga.pyeasyga import GeneticAlgorithm
    >>> input_data = [('pear', 50), ('apple', 35), ('banana', 40)]
    >>> easyga = GeneticAlgorithm(input_data)
    >>> def fitness (member, data):
    >>>     return sum([profit for (selected, (fruit, profit)) in
    >>>                 zip(member, data) if selected and
    >>>                 member.count(1) == 2])
    >>> easyga.fitness_function = fitness
    >>> easyga.run()
    >>> print easyga.best_individual()

    """

    def __init__(self,
                 seed_data,
                 population_size=50,
                 generations=100,
                 crossover_probability=0.8,
                 mutation_probability=0.2,
                 adaptive_mutation_factor = 1.3, # CHANGED
                 elitism=True,
                 maximise_fitness=True,
                 clipping_threshold = 0.9,
                 num_turbines = 2):
        """Instantiate the Genetic Algorithm.

        :param seed_data: input data to the Genetic Algorithm
        :type seed_data: list of objects
        :param int population_size: size of population
        :param int generations: number of generations to evolve
        :param float crossover_probability: probability of crossover operation
        :param float mutation_probability: probability of mutation operation

        """

        self.seed_data = seed_data
        self.num_turbines = num_turbines
        self.population_size = population_size
        self.generations = generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.adaptive_mutation_factor = adaptive_mutation_factor # CHANGED
        self.elitism = elitism
        self.maximise_fitness = maximise_fitness
        self.clipping_threshold = clipping_threshold # CHANGED

        self.current_generation = []

        def create_individual(seed_data):
            """Create a candidate solution representation.

            e.g. for a bit array representation:

            >>> return [random.randint(0, 1) for _ in xrange(len(data))]

            :param seed_data: input data to the Genetic Algorithm
            :type seed_data: list of objects
            :returns: candidate solution representation as a list

            """
            farm_layout = [1]*self.num_turbines + [0]*(len(seed_data)-self.num_turbines)
            random.shuffle(farm_layout)
            return farm_layout

        def crossover(parent_1, parent_2, parent_1_fitness, parent_2_fitness):
            """Crossover (mate) two parents to produce two children.

            :param parent_1: candidate solution representation (list)
            :param parent_2: candidate solution representation (list)
            :returns: tuple containing two children

            """
            NUMERICAL_STABILITY_FACTOR = 0.1 # need this to prevent divide by zero in case both parent's fitness is zero.

            # STEP 1: calculate probabilities of inheriting from either parent; and thus, prob of inheriting from dominant parent

            prob_inherit_from_parent_1 = float(parent_1_fitness) / (parent_1_fitness + parent_2_fitness + NUMERICAL_STABILITY_FACTOR)
            prob_inherit_from_parent_2 = float(parent_2_fitness) / (parent_1_fitness + parent_2_fitness + NUMERICAL_STABILITY_FACTOR)
            dominant_parent_prob = max(prob_inherit_from_parent_1, prob_inherit_from_parent_2)
            
            #STEP 2: select dominant and recessive parents.
            dominant_parent = parent_1
            recessive_parent = parent_2
            if prob_inherit_from_parent_1 < 0.5:
                dominant_parent = parent_2
                recessive_parent = parent_1

            # STEP 3: perform clipping if needed
            if dominant_parent_prob > self.clipping_threshold: 
                dominant_parent_prob = self.clipping_threshold

            # STEP 4: create children. child 1 gets preference on inheriting from dominant parent.
            child_1 = []
            child_2 = []
            for i in range(len(parent_1)):
                curr_allele_inherit_probability = np.random.rand()
                if(curr_allele_inherit_probability > 1.0 -dominant_parent_prob): # e.g. random > .1 gives 90% chance
                    child_1.append(dominant_parent[i])
                    child_2.append(recessive_parent[i])
                else:
                    child_1.append(recessive_parent[i])
                    child_2.append(dominant_parent[i])

            return child_1, child_2

        def mutate(individual):
            """Reverse the bit of a random index in an individual."""
            mutate_index = random.randrange(len(individual))
            individual[mutate_index] = (0, 1)[individual[mutate_index] == 0]

        def random_selection(population):
            """Select and return a random member of the population."""
            return random.choice(population)

        def tournament_selection(population):
            """Select a random number of individuals from the population and
            return the fittest member of them all.
            """
            if self.tournament_size == 0:
                self.tournament_size = 2
            members = random.sample(population, int(self.tournament_size))
            members.sort(
                key=attrgetter('fitness'), reverse=self.maximise_fitness)
            return members[0]

        self.fitness_function = None
        self.tournament_selection = tournament_selection
        self.tournament_size = self.population_size / 10
        self.random_selection = random_selection
        self.create_individual = create_individual
        self.crossover_function = crossover
        self.mutate_function = mutate
        self.selection_function = self.tournament_selection

    def create_initial_population(self):
        """Create members of the first population randomly.
        """
        initial_population = []
        for _ in range(self.population_size):
            genes = self.create_individual(self.seed_data)
            individual = Chromosome(genes)
            initial_population.append(individual)
        self.current_generation = initial_population

    def calculate_population_fitness(self):
        """Calculate the fitness of every member of the given population using
        the supplied fitness_function.
        """
        for individual in self.current_generation:
            individual.fitness = self.fitness_function(
                individual.genes, self.seed_data)

    def calculate_individual_fitness(self, individual): # CHANGED
        """Calculate the fitness of a single member of the given population using
        the supplied fitness_function.
        """
        current_individual_fitness = self.fitness_function(individual.genes, self.seed_data)
        individual.fitness = current_individual_fitness
        return current_individual_fitness


    def rank_population(self):
        """Sort the population by fitness according to the order defined by
        maximise_fitness.
        """
        self.current_generation.sort(
            key=attrgetter('fitness'), reverse=self.maximise_fitness)

    def create_new_population(self, generationNumber):
        """Create a new population using the genetic operators (selection,
        crossover, and mutation) supplied.
        """
        new_population = []
        elite = copy.deepcopy(self.current_generation[0])
        selection = self.selection_function

        self.mutation_probability *= self.adaptive_mutation_factor # CHANGED . adaptive mutation rate

        while len(new_population) < self.population_size:
            parent_1 = copy.deepcopy(selection(self.current_generation))
            parent_2 = copy.deepcopy(selection(self.current_generation))

            child_1, child_2 = parent_1, parent_2
            child_1.fitness, child_2.fitness = 0, 0

            can_crossover = random.random() < self.crossover_probability
            can_mutate = random.random() < self.mutation_probability

            if can_crossover:
                #child_1.genes, child_2.genes = self.crossover_function(parent_1.genes, parent_2.genes)
                parent_1_fitness = self.calculate_individual_fitness(parent_1) 
                parent_2_fitness = self.calculate_individual_fitness(parent_2)
                child_1.genes, child_2.genes = self.crossover_function(parent_1.genes, parent_2.genes, parent_1_fitness, parent_2_fitness) # CHANGED

            if can_mutate:
                self.mutate_function(child_1.genes)
                self.mutate_function(child_2.genes)

            new_population.append(child_1)
            if len(new_population) < self.population_size:
                new_population.append(child_2)

        if self.elitism:
            new_population[0] = elite

        self.current_generation = new_population

    def create_first_generation(self):
        """Create the first population, calculate the population's fitness and
        rank the population by fitness according to the order specified.
        """
        self.create_initial_population()
        self.calculate_population_fitness()
        self.rank_population()

    def create_next_generation(self, generationNumber):
        """Create subsequent populations, calculate the population fitness and
        rank the population by fitness in the order specified.
        """
        self.create_new_population(generationNumber)
        self.calculate_population_fitness()
        self.rank_population()

    def run(self):
        """Run (solve) the Genetic Algorithm."""
        best_fitnesses = []
        self.create_first_generation()
        best_fitnesses.append([0,self.best_individual()])

        for generationNumber in range(1, self.generations):
            #self.create_next_generation()
            best_fitnesses.append([generationNumber,self.best_individual()])
            self.create_next_generation(generationNumber)

        return best_fitnesses 

    def best_individual(self):
        """Return the individual with the best fitness in the current
        generation.
        """
        best = self.current_generation[0]
        return (best.fitness, best.genes)

    def last_generation(self):
        """Return members of the last generation as a generator function."""
        return ((member.fitness, member.genes) for member
                in self.current_generation)


class Chromosome(object):
    """ Chromosome class that encapsulates an individual's fitness and solution
    representation.
    """
    def __init__(self, genes):
        """Initialise the Chromosome."""
        self.genes = genes
        self.fitness = 0.00001

    def __repr__(self):
        """Return initialised Chromosome representation in human readable form.
        """
        return repr((self.fitness, self.genes))
