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
        self.weightCapacity = weightCapacity
        self.volumeCapacity = volumeCapacity
        self.accountBalance = accountBalance

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

    def uniformCrossover(self,parent1, parent2):
        
        sizeP1 = np.size(parent1.individual)
        sizeP2 = np.size(parent2.individual)

        extraChromosomes = np.empty(0)

        if sizeP1 < sizeP2:
            extraChromosomes = parent2.individual[-(sizeP2-sizeP1):]
            parent2.individual = parent2.individual[:-(sizeP2-sizeP1)]
        elif sizeP1 > sizeP2:
            extraChromosomes = parent1.individual[-(sizeP1-sizeP2):]
            parent1.individual = parent1.individual[:-(sizeP1-sizeP2)]
        else:
            pass

        child1 = parent1
        child2 = parent2

        child1.individual = np.empty(0)
        child1.individual = np.empty(1)

        for gen1, gen2 in zip(parent1.individual, parent2.individual):

            if self.crossoverProbability < np.random.uniform(0, 1):
                print("No Cross")
                child1.individual = np.append(child1, gen1)
                child2.individual = np.append(child2, gen2)
            else:
                print("Cross")
                child1.individual = np.append(child1, gen2)
                child2.individual = np.append(child2, gen1)

        for gen in extraChromosomes:
        
            if self.crossoverProbability < np.random.uniform(0, 1):
                child1.individual = np.append(child1, gen)
            else:
                child2.individual = np.append(child2, gen)

        #child1, child2 = self.checkChildLimits(child1, child2)
                
        return child1, child2
    
    def checkChildLimits(self, child1, child2):

        # [child1, child2]
        weight = []
        volume = []
        value = []
        for i, child in enumerate[child1, child2]:
            for item in child.individual:
                weight[i] += item.weight
                volume[i] += item.volume
                value[i] += item.value

        while (weight[0] > child1.weightCapacity) or (weight[1] > child2.weightCapacity):
            if (weight[0] > child1.weightCapacity):
                child1.individual = sorted(child1.individual, key=lambda x: x.weight)
                child2.individual = np.append(child2.individual, child1.individual[0])
                child1.individual = child1.individual[1:]
            
            if (weight[1] > child2.weightCapacity):
                child2.individual = sorted(child2.individual, key=lambda x: x.weight)
                child1.individual = np.append(child1.individual, child2.individual[0])
                child2.individual = child2.individual[1:]


        while (volume[0] > child1.volumeCapacity) or (weight[1] > child2.volumeCapacity):
            if (volume[0] > child1.volumeCapacity):
                child1.individual = sorted(child1.individual, key=lambda x: x.volume)
                child2.individual = np.append(child2.individual, child1.individual[0])
                child1.individual = child1.individual[1:]
            
            if (volume[1] > child2.volumeCapacity):
                child2.individual = sorted(child2.individual, key=lambda x: x.volume)
                child1.individual = np.append(child1.individual, child2.individual[0])
                child2.individual = child2.individual[1:]


        while (value[0] > child1.accountBalance) or (value[1] > child2.accountBalance):
            if (value[0] > child1.accountBalance):
                child1.individual = sorted(child1.individual, key=lambda x: x.value)
                child2.individual = np.append(child2.individual, child1.individual[0])
                child1.individual = child1.individual[1:]
            
            if (value[1] > child2.accountBalancey):
                child2.individual = sorted(child2.individual, key=lambda x: x.value)
                child1.individual = np.append(child1.individual, child2.individual[0])
                child2.individual = child2.individual[1:]

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

mutationRate = 0.2
crossOverProb = 0.98
ga = GA(mutationRate, crossOverProb)

child1, child2 = ga.uniformCrossover(pop.population[0], pop.population[1])

print(pop.population[0].individual)

#printPopInfo([child1, child2])