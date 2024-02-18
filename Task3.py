import random

def main():
    # newProblem = Problem("VIJANDER_SINGH*200135",200,0.1)
    newProblem = Problem("Joar_Matias*588289", 500, 0.05, 1)
    newProblem.generation = 1000
    print(newProblem.iterate())

def DNA():
    characters = " ABCDEFGHIJKLMNOPQRTSUVWXYZØÆÅabcdefghijklmnopqrstuvwxyzæøå#*_-^<>:;123456789"
    char = random.choice(characters)
    return char
class Individual:
    def __init__(self,name_length):
        self.name = []
        self.weight = 1

        for i in range(name_length):
            self.name.append(DNA())

class Problem:
    def __init__(self, target_name, pop, mutation,crossover_probability):

        self.mutation = mutation
        self.crossover_probability = crossover_probability
        self.target_name = target_name
        self.target_length = len(target_name)

        self.generation = 0
        self.generations_to_found = 0

        self.weight_multiplier = 10
        self.best_fitness = 1
        self.best = ""

        self.population = []

        for i in range(pop):
            self.population.append(Individual(self.target_length))
        self.fitness_test()

    def fitness_test(self):
        for i in range(len(self.population)):
            self.population[i] = self.weight_calculation(self.population[i])

    def weight_calculation(self, current_individual):
        current_name = ""

        for i in range(self.target_length):
            if current_individual.name[i] == self.target_name[i]:
                current_individual.weight += self.weight_calc_func(current_individual.weight)
            current_name += current_individual.name[i]


        if self.best_fitness <= current_individual.weight:
            self.best_fitness = current_individual.weight
            self.best = current_name

        return current_individual

    def weight_calc_func(self, weigth_to_modify):
        return weigth_to_modify * self.weight_multiplier

    def mating(self):
        new_population = []
        for i in range(len(self.population)//2):
            parent1,parent2 = self.weighted_random_selection()
            child1,child2 = self.single_point_crossover(parent1,parent2)
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
        #print(weights)

        parent1 = random.choices(listOfNames, weights)[0]
        parent2 = random.choices(listOfNames, weights)[0]

        return parent1,parent2

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

    def mutate_child(self,child):
        for i in range(self.target_length):
            if random.uniform(0,1) <= self.mutation:
                child[i] = DNA()
        return child

    def iterate(self):
        for i in range(self.generation):
            self.mating()
            self.fitness_test()
            self.generations_to_found += 1

            print(self.generations_to_found)
            print(self.best)
            if self.best == self.target_name:
                return "Target name found, " + self.best + " With " + str(self.generations_to_found) + " generations"
        return "Target not found"

main()