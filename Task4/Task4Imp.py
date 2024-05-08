import numpy as np
import matplotlib.pyplot as plt

#This Code solves for unbound Knapsack Problem using Genetic Algorithm

class Pop:
    def __init__(self, items, weightCapacity, volumeCapacity, maxItemWeight, maxItemVolume, maxItemPrice, populationSize):
        populationSize = populationSize // 2 * 2
        self.stock = Pop.createStock(items, maxItemWeight, maxItemVolume, maxItemPrice)
        self.population = Pop.createInitialPopulation(self.stock, weightCapacity, volumeCapacity, populationSize)
        self.weightCapacity = weightCapacity
        self.volumeCapacity = volumeCapacity
        self.maxItemPrice = maxItemPrice

    def createInitialPopulation(stock, weightCapacity, volumeCapacity, populationSize):

        population = {}

        for i in range(populationSize):
            population[i] = Pop.createIndividual(stock, weightCapacity, volumeCapacity)
            
        return population

    def createStock(items, maxItemWeight, maxItemVolume, maxItemPrice):
        
        #[weight, volume, value]
        stock = np.array([np.random.randint(1,maxItemWeight), np.random.randint(1,maxItemVolume),np.random.randint(1,maxItemPrice)])
        
        for i in range(items-1):  # Create Items
            randomWeight = np.random.randint(1, maxItemWeight)
            randomVolume = np.random.randint(1, maxItemVolume)
            randomValue = np.random.randint(1, maxItemPrice)

            itemToAdd = np.array([randomWeight, randomVolume, randomValue])
            stock = np.vstack([stock, itemToAdd])
        
        #print(stock)

        return stock
    
    def createIndividual(stock, weightCapacity, volumeCapacity):
        individual = np.array(stock[np.random.randint(stock.shape[0])])

        totalWeight = individual[0] #Weight
        totalVolume = individual[1] #Volume

        while True:
            itemToAdd = stock[np.random.randint(stock.shape[0])]
            totalWeight += itemToAdd[0] #Weight
            totalVolume += itemToAdd[1] #Volume
            
            if (totalWeight > weightCapacity) or (totalVolume > volumeCapacity):
                totalWeight -= itemToAdd[0] #Weight
                totalVolume -= itemToAdd[1] #Volume
                #print("Weight= ", totalWeight, "Volume= ", totalVolume, "Cost= ", totalCost)
                break

            individual = np.vstack([individual, itemToAdd])
            
        return individual

class GA:
    def __init__(self, mutationRate, crossoverProbability, weightCapacity, volumeCapacity, stock):
        self.mutationRate = mutationRate
        self.crossoverProbability = crossoverProbability
        self.weightCapacity = weightCapacity
        self.volumeCapacity = volumeCapacity
        self.stock = stock
        self.penalty = 0.001
        self.higestBackPackValue = 0

    def fitnesFunction(self, candidate):
        valuesFromCandidate = candidate[:, 2]

        totalValue = np.sum(valuesFromCandidate)

        fitness = totalValue
        
        return 100/fitness

    def uniformCrossover(self,parent1, parent2):

        child1 = np.array([0,0,0])
        child2 = np.array([0,0,0])

        if self.crossoverProbability >= np.random.uniform(0, 1):

            extraChromosomes = np.empty(0)        

            if len(parent1) < len(parent2):
                extraChromosomes = parent2[(len(parent1)-len(parent2)):]
                parent2 = parent2[:(len(parent1)-len(parent2))]
            elif len(parent2) < len(parent1):
                extraChromosomes = parent1[(len(parent2)-len(parent1)):]
                parent1 = parent1[:(len(parent2)-len(parent1))]

            for gen1, gen2 in zip(parent1, parent2):
                
                if np.random.uniform(0, 1) < 0.5: # Coin Toss
                    child1 = np.vstack([child1, gen1])
                    child2 = np.vstack([child2, gen2])
                else:
                    child1 = np.vstack([child1, gen2])
                    child2 = np.vstack([child2, gen1])

            child1 = np.delete(child1, (0), axis=0)
            child2 = np.delete(child2, (0), axis=0)

            for gen in extraChromosomes:
                if 0.5 < np.random.uniform(0, 1):
                    child1 = np.vstack([child1, gen])
                else:
                    child2 = np.vstack([child2, gen])

        else:
            child1 = parent1
            child2 = parent2

        child1 = self.mutateChild(child1)
        child2 = self.mutateChild(child2)

        child1 = self.checkChildLimits(child1)
        child2 = self.checkChildLimits(child2)

        return child1, child2
    
    def checkChildLimits(self, child):

        totalWeight = np.sum(child[:, 0])
        totalVolume = np.sum(child[:, 1])

        while totalWeight > self.weightCapacity: 
            minValueIndex = np.argmin(child[:, 2])

            child = np.delete(child, minValueIndex, axis=0)
            totalWeight = np.sum(child[:, 0])

        while totalVolume > self.volumeCapacity:
            minValueIndex = np.argmin(child[:, 2])

            child = np.delete(child, minValueIndex, axis=0)
            totalVolume = np.sum(child[:, 1])
                
        return child
    
    def mutateChild(self, child):
        mutatedChild = np.array([0,0,0])       
        for gen in child:
            if self.mutationRate >= np.random.uniform(0, 1):
                gen = self.stock[np.random.randint(0, len(self.stock))]
                mutatedChild = np.vstack([mutatedChild, gen])
            else:
                mutatedChild = np.vstack([mutatedChild, gen])
                
        mutatedChild = np.delete(mutatedChild, (0), axis=0)

        return mutatedChild

    def costBasedSelection(self, fitnessValues, population):

        ranks = np.argsort(fitnessValues)

        selectionProbabilities = (2 - np.linspace(0, 2, len(fitnessValues))) / len(fitnessValues)

        cumulativeProbabilities = np.cumsum(selectionProbabilities)
        selectedIndices = np.searchsorted(cumulativeProbabilities, np.random.rand(len(fitnessValues)))

        matingPartners = {}

        for i in range(len(population)):
            matingPartners[i] = population[ranks[selectedIndices[i]]]

        return matingPartners
    
    def getFitnessValues(self, population):
        fitnesValuesArray = np.array([self.fitnesFunction(population[individual]) for individual in population])
        return fitnesValuesArray
    
    def getNewPopulation(self, parents):

        newPopulation = {}
        for i in range(0,len(parents),2):
            child1, child2 = self.uniformCrossover(parents[i],parents[i+1])

            newPopulation[i] = child1
            newPopulation[i+1] = child2

        return newPopulation

    def bestSoFar(self,fitnessValues, population):
        bestIndex = np.argmin(fitnessValues)
        #print(population[bestIndex])

        totalWeight = np.sum(population[bestIndex][:, 0])
        totalVolume = np.sum(population[bestIndex][:, 1])
        totalValue = np.sum(population[bestIndex][:, 2])

        print("Weight= ", totalWeight, "Volume= ", totalVolume, "Cost= ", totalValue)
        self.higestBackPackValue = totalValue
        return np.min(fitnessValues)

class Logger:
    def __init__(self) -> None:
        self.fitnessCostLog = np.empty(0)
        self.generationList = np.empty(0)

    def updatelog(self, generations, lowestFitnesCost):
        self.generationList = np.append(self.generationList, generations)
        self.fitnessCostLog = np.append(self.fitnessCostLog,lowestFitnesCost)

# Constrains

##Item Constraints
items = 20
maxItemWeight = 20 # Kg
maxItemVolume = 20 # Liters
maxItemPrices = 100 # Dollars

##Backpack Constraints
weightCapacity = 50 # Kilogram
volumeCapacity = 60 # Liters

##Setting For GA
populationSize = 1000
mutationRate = 0.02
crossOverProb = 0.5

numberOfGenerations = 100

pop = Pop(items, weightCapacity, volumeCapacity, maxItemWeight, maxItemVolume, maxItemPrices, populationSize)
population = pop.population

algorithm = GA(mutationRate, crossOverProb, weightCapacity, volumeCapacity, pop.stock)
logger = Logger()

generations = 0
while True:
    fitnessValues = algorithm.getFitnessValues(population)

    lowestCost = algorithm.bestSoFar(fitnessValues,population)

    if generations > numberOfGenerations:
        break

    parents = algorithm.costBasedSelection(fitnessValues, population)
    newPopulation = algorithm.getNewPopulation(parents)
    population = newPopulation
    logger.updatelog(generations, lowestCost)

    generations += 1

highestBackPackValue = algorithm.higestBackPackValue

plt.title("Fitness Performance Graph, Highest Backpack Value: " + str(highestBackPackValue) + "\n Pop: " + str(populationSize) + ", Pm: " + str(mutationRate)+ ", Pc: " + str(crossOverProb))
plt.xlabel("Generations")
plt.ylabel("Fitness Chromosome")

plt.plot(logger.generationList, logger.fitnessCostLog)
plt.show()