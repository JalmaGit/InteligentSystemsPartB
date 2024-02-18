import random

def DNA():
    characters = "ABCDEFGHIJKLMNOPQRTSUVWXYZØÆÅabcdefghijklmnopqrstuvwxyzæøå*_"
    char = random.choice(characters)
    return char
class Individual:
    def __init__(self,name_length):
        self.name = []
        self.weight = 0

        for i in range(name_length):
            self.name.append(DNA())

class Problem:
    def __init__(self, target_name, pop, mutation):

        self.target_name = target_name
        self.generation = 0
        self.finished = False
        self.mutation = mutation
        self.best_fitness = 0
        self.best = ""

        self.population = []
        for i in range(pop):
            self.population.append(Individual(len(target_name)))

    def fitness_test(self):
        for j in range(len(self.population)):

            current_fitness = 0
            current_name = ""

            for k in range(len(self.population[j].name)):
                if self.population[j].name[k] == self.target_name[k]:
                    current_fitness += 1
                current_name += self.population[j].name[k]

            self.population[j].weight = current_fitness

            if self.best_fitness <= current_fitness:
                self.best_fitness = current_fitness
                self.best = current_name
            print(self.best_fitness)
            print(self.best)

    #Using Weighted Random Selection
    def mating_time(self):
        weights = []
        listOfNames = []
        for i in range(len(self.population)):
            weights.append(self.population[i].weight)
            listOfNames.append(i)

        print(listOfNames)
        print(weights)

        mating_partner_one = random.choices(listOfNames, weights, 1)
        print(mating_partner_one)


newProblem = Problem("VIJANDER_SINGH*200135",10,0.02)

newProblem.fitness_test()

newProblem.mating_time()

#print(newProblem.best)
#print(newProblem.best_fitness)

'''
for x in range(len("VIJANDER_SINGH*200135")):
    print(newProblem.population[0].name[x], end='')
print('')
'''