import numpy as np
import matplotlib.pyplot as plt

class Pop:
    def __init__(self,targetName,populationSize):

        populationSize = populationSize // 2 * 2
        self.population = Pop.createInitialPopulation(len(targetName), populationSize)

    def DNA():
        characters = " ABCDEFGHIJKLMNOPQRTSUVWXYZØÆÅabcdefghijklmnopqrstuvwxyzæøå#*_-^<>:;@.,123456789"
        char_array = np.array(list(characters))
        char = np.random.choice(char_array)
        return char
    
    def randomString(nameLength):
        randomIndividual = ''
        for _ in range(nameLength):
            randomIndividual += Pop.DNA()
        
        return randomIndividual
    
    def createInitialPopulation(nameLength, populationSize):
        return np.array([Pop.randomString(nameLength) for _ in range(populationSize)])
    
class GA:
    def __init__(self, targetName, mutationRate, crossoverProbability):
        self.targetName = targetName
        self.targetLength = len(targetName)
        self.mutatationRate = mutationRate
        self.targetFound = False
        self.crossoverProbability = crossoverProbability

    def fitnesFunction(self, candidate): 
        fitness = 1
        for expteted, actual in zip(self.targetName, candidate):
            if expteted == actual:
                fitness += 1

        return 1/fitness

    def doublePointCrossover(self,parent1, parent2): 

        if np.random.uniform(0, 1) <= self.crossoverProbability:
            crossoverPoints = np.sort(np.random.choice(range(1, len(parent1)), 2, replace=False))

            child1 = parent1[:crossoverPoints[0]] + parent2[crossoverPoints[0]:crossoverPoints[1]] + parent1[crossoverPoints[1]:]
            child2 = parent2[:crossoverPoints[0]] + parent1[crossoverPoints[0]:crossoverPoints[1]] + parent2[crossoverPoints[1]:]
        else:
            child1 = parent1
            child2 = parent2 

        child1 = self.mutateChild(child1)
        child2 = self.mutateChild(child2)

        return child1, child2
    
    def singlePointCrossover(self,parent1, parent2):

        if np.random.uniform(0, 1) <= self.crossoverProbability:
            crossoverPoint = np.random.randint(0, len(parent1))

            child1 = parent1[:crossoverPoint] + parent2[crossoverPoint:]
            child2 = parent2[:crossoverPoint] + parent1[crossoverPoint:]
        else:
            child1 = parent1
            child2 = parent2 

        child1 = self.mutateChild(child1)
        child2 = self.mutateChild(child2)

        return child1, child2

    def mutateChild(self, child):
        mutatedChild = ''
        for char in child:
            if np.random.uniform(0, 1) <= self.mutatationRate:
                mutatedChild += Pop.DNA()
            else:
                mutatedChild += char
        return mutatedChild

    def costBasedSelection(self, fitnessValues, population):

        ranks = np.argsort(fitnessValues)

        selectionProbabilities = (2 - np.linspace(0, 2, len(fitnessValues))) / len(fitnessValues)

        cumulativeProbabilities = np.cumsum(selectionProbabilities)
        selectedIndices = np.searchsorted(cumulativeProbabilities, np.random.rand(len(fitnessValues)))

        matingPartners = population[ranks[selectedIndices]]

        return matingPartners
    
    def getFitnessValues(self, population):
        return np.array([self.fitnesFunction(individual) for individual in population])
    
    def getNewPopulation(self, parents):
        parents = parents.reshape(-1,2)

        newPopulation = np.empty((len(parents) * 2,), dtype=parents.dtype)
        for i, (parent1, parent2) in enumerate(parents):
            child1, child2 = self.doublePointCrossover(parent1, parent2)

            newPopulation[i*2] = child1
            newPopulation[i*2 + 1] = child2

        return newPopulation
    
    def hasConverged(self, fitnessValues, population):
        self.targetFound =  population[np.argmin(fitnessValues)] == self.targetName

    def bestSoFar(self,fitnessValues, population):
        bestIndex = np.argmin(fitnessValues)
        print(population[bestIndex])
        return np.min(fitnessValues)

class Logger:
    def __init__(self) -> None:
        self.fitnessCostLog = np.empty(0)
        self.generationList = np.empty(0)

    def updatelog(self, generations, lowestFitnesCost):
        self.generationList = np.append(self.generationList, generations)
        self.fitnessCostLog = np.append(self.fitnessCostLog,lowestFitnesCost)


#Problem Definition
targetString = "Joar_Breivik*588289"
populationSize = 400
mutationRate = 0.05
crossoverProbability = 0.98
maximumNumberOfGenerations = 1000

pop = Pop(targetString, populationSize)
algorithm = GA(targetString,mutationRate,crossoverProbability)
logger = Logger()

population = pop.population

generations = 0
while True:
    fitnessValues =  algorithm.getFitnessValues(population)
    
    #Cheking
    algorithm.hasConverged(fitnessValues, population)
    lowestCost = algorithm.bestSoFar(fitnessValues, population)
    if algorithm.targetFound or generations == maximumNumberOfGenerations:
        break

    
    parents = algorithm.costBasedSelection(fitnessValues,population)
    newPopulation = algorithm.getNewPopulation(parents)
    population = newPopulation
    
    logger.updatelog(generations, lowestCost)

    generations += 1
    print(generations)

plt.title("Fitness Performance Graph, Solution Found: " + str(algorithm.targetFound) + "\n Target: " + targetString + ", Pop: " + str(populationSize) + ", Pm: " + str(mutationRate)+ ", Pc: " + str(crossoverProbability))
plt.xlabel("Generations")
plt.ylabel("Fitness Chromosome")

plt.plot(logger.generationList, logger.fitnessCostLog)
plt.show()