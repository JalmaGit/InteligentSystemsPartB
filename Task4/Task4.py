import numpy as np
import matplotlib.pyplot as plt

#This Code solves for unbound Knapsack Problem using Genetic Algorithm

def printPopInfo(population):
    for ind in population:
        print(f"New Individual with {np.size(ind.individual)} items")
        weight = 0
        volume = 0
        value = 0
        for item in ind.individual:
            weight += item.weight
            volume += item.volume
            value += item.value
        
        print(f"{weight=}, {volume=}, {value=}")

class Item:
    def __init__(self, weight, volume, value):
        self.weight = weight
        self.volume = volume
        self.value = value

class Indiv:
    def __init__(self, stock, weightCapacity, volumeCapacity, accountBalance):     
        self.individual = Indiv.createIndiviual(stock, weightCapacity, volumeCapacity, accountBalance)

    def createIndiviual(stock, weightCapacity, volumeCapacity, accountBalance):
        individual = np.empty(0)

        totalWeight = 0
        totalVolume = 0
        totalCost = 0
        numberOfItems = 0

        while True:
            itemToAdd = stock[np.random.randint(np.size(stock))]
            totalWeight += itemToAdd.weight
            totalVolume += itemToAdd.volume
            totalCost += itemToAdd.value
            
            if (totalWeight > weightCapacity) or (totalVolume > volumeCapacity) or (totalCost > accountBalance):
                totalWeight -= itemToAdd.weight
                totalVolume -= itemToAdd.volume
                totalCost -= itemToAdd.value
                break

            numberOfItems += 1

            individual = np.append(individual,itemToAdd)

        return individual

class Pop:
    def __init__(self, items, weightCapacity, volumeCapacity, accountBalance, populationSize):
        self.population = Pop.createInitialPopulation(items, weightCapacity, volumeCapacity, accountBalance, populationSize)

    def createInitialPopulation(items, weightCapacity, volumeCapacity, accountBalance, populationSize):
        stock = Pop.createStock(items, weightCapacity, volumeCapacity, accountBalance)

        population = np.empty(0)
        for i in range(populationSize):    
            individual = Indiv(stock, weightCapacity, volumeCapacity, accountBalance)
            population = np.append(population, individual)

        return population

    def createStock(items, weightCapacity, volumeCapacity, accountBalance):
        stock = np.empty(0)

        while True:  # Create Items
            randomWeight = np.random.randint(weightCapacity//5)
            randomVolume = np.random.randint(volumeCapacity//5)
            randomCost = np.random.randint(accountBalance//10)

            itemToAdd = Item(randomWeight, randomVolume, randomCost)
            stock = np.append(stock, itemToAdd)
        
            if np.size(stock) == items:
                break

        return stock

class GA:
    def __init__(self, mutationRate, crossoverProbability):
        self.mutationRate = mutationRate
        self.crossoverProbability = crossoverProbability

    def fitnesFunction(self, candidate): 
        fitness = 1
        return 1/fitness

    def nameOfCrossover(self,parent1, parent2): 

        child1 = 0
        child2 = 0
        return child1, child2
    
    def nameOfAnotherTypeCrossover(self,parent1, parent2):

        child1 = 0
        child2 = 0
        return child1, child2

    def mutateChild(self, child):
        mutatedChild = 0
        return mutatedChild

    def nameOfSelection(self, fitnessValues, population):
        matingPartners = 0
        return matingPartners
    
    def getFitnessValues(self, population):
        fitnesValuesArray = np.array(["Need To Do Calc"])
        return fitnesValuesArray
    
    def getNewPopulation(self, parents):
        newPopulation = 0
        return newPopulation

    def bestSoFar(self,fitnessValues, population):
        bestIndex = np.argmin(fitnessValues)
        print(population[bestIndex])
        return np.min(fitnessValues)


#Test Stuff

# Constrains
items = 5
maxWeightCapacity = 20 # Kilogram
maxVolumeCapacity = 15 # Liters
accountBalance = 100 # Dollars
populationSize = 6

pop = Pop(items, maxWeightCapacity, maxVolumeCapacity, accountBalance, populationSize)

printPopInfo(pop.population)