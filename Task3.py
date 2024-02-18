import random
import matplotlib.pyplot as plt
import numpy as np


def main():
    # newProblem = Problem(target, population, mutation, crossover_probability)
    newProblem = Problem("Joar_Breivik*588289", 50, 0.05, 0.8)
    newProblem.generation = 1000

    print(newProblem.run_genetic_algorithm())

    x = np.linspace(0,newProblem.generation,newProblem.generation+1)
    y = newProblem.best_fitness

    if (len(x)!=newProblem.generation-1):
        for i in range(len(x) - len(y)):
            y.append(newProblem.best_fitness_level)

    plt.title("Fitness Performance Graph, Solution Found: " + str(newProblem.target_lock) + "\n Target: " + newProblem.target_name
              + ", Pop: " + str(newProblem.pop) + ", Pm: " + str(newProblem.mutation)
             + ", Pc: " + str(newProblem.crossover_probability))
    plt.xlabel("Generations")
    plt.ylabel("Fitness Chromosome")

    plt.plot(x, y)
    plt.show()


def DNA():
    characters = " ABCDEFGHIJKLMNOPQRTSUVWXYZØÆÅabcdefghijklmnopqrstuvwxyzæøå#*_-^<>:;@.,123456789"
    char = random.choice(characters)
    return char


class Individual:
    def __init__(self, name_length):
        self.name = []
        self.weight = 1

        # Creates the name for the indivual
        for i in range(name_length):
            self.name.append(DNA())


class Problem:
    def __init__(self, target_name, pop, mutation, crossover_probability):

        self.mutation = mutation
        self.crossover_probability = crossover_probability
        self.target_name = target_name
        self.target_length = len(target_name)
        self.target_lock = False

        self.generation = 0
        self.generations_to_found = 0

        self.best_fitness = []
        self.best_fitness_level = 0

        self.weight_multiplier = 5
        self.best_current_fitness = 1
        self.best = ""

        self.pop = pop
        self.population = []

        for i in range(pop):
            self.population.append(Individual(self.target_length))
        self.fitness_test()

    def fitness_test(self):
        for i in range(len(self.population)):
            self.population[i] = self.weight_calculation(self.population[i])
        self.best_fitness.append(self.best_fitness_level)

    def weight_calculation(self, current_individual):
        current_name = ""
        current_fitness = 0

        for i in range(self.target_length):
            if current_individual.name[i] == self.target_name[i]:
                current_individual.weight += self.weight_calc_func(current_individual.weight)
                current_fitness += 1
            current_name += current_individual.name[i]

        if self.best_current_fitness <= current_individual.weight:
            self.best_current_fitness = current_individual.weight
            self.best_fitness_level = current_fitness/self.target_length
            self.best = current_name

        return current_individual

    def weight_calc_func(self, weigth_to_modify):
        return weigth_to_modify * self.weight_multiplier

    def mating(self):
        new_population = []
        for i in range(len(self.population) // 2):
            parent1, parent2 = self.weighted_random_selection()
            child1, child2 = self.single_point_crossover(parent1, parent2)
            new_population.append(child1)
            new_population.append(child2)

        for i in range(len(new_population)):
            self.population[i].name = new_population[i]
            self.population[i].weight = 1

    def weighted_random_selection(self):
        weights = []
        listOfNames = []

        for i in range(len(self.population)):
            weights.append(self.population[i].weight)
            listOfNames.append(self.population[i].name)

        # Randomly selects name, weighted towards name closest to the solution
        parent1 = random.choices(listOfNames, weights)[0]
        parent2 = random.choices(listOfNames, weights)[0]

        return parent1, parent2

    def single_point_crossover(self, parent1, parent2):
        child1 = []
        child2 = []

        mid_point = random.randint(0, self.target_length)
        size = len(self.target_name)
        if random.uniform(0, 1) <= self.crossover_probability:
            for x in range(mid_point):
                child1.append(parent1[x])
                child2.append(parent2[x])

            for x in range(size - mid_point):
                child1.append(parent2[x])
                child2.append(parent1[x])

            child1 = self.mutate_child(child1)
            child2 = self.mutate_child(child2)
        else:
            child1 = parent1
            child2 = parent2

        return child1, child2

    def mutate_child(self, child):
        for i in range(self.target_length):
            if random.uniform(0, 1) <= self.mutation:
                child[i] = DNA()
        return child

    def run_genetic_algorithm(self):
        for i in range(self.generation):
            self.mating()
            self.fitness_test()
            self.generations_to_found = i

            print(i)
            print(self.best)
            if self.best == self.target_name:
                self.target_lock = True
                return "Target name found, " + self.best + " With " + str(i) + " generations"
        return "Target not found"


main()
