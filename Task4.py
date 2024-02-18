import random

class item:
    def __init__(self,weight,value):
        self.weight = weight
        self.value = value

class Individual:
    def __init__(self):

class Knapsack:
    def __init__(self, max_weight, nr_items, pop):
        self.maximum_weight = max_weight
        self.current_weight = 0
        self.current_value = []
        self.items = []
        self.nr_items =  nr_items

        self.items.append(item(2, 1))
        self.items.append(item(5, 4))
        self.items.append(item(1, 6))
        self.items.append(item(8, 2))

        self.population = []
        '''
        for i in range(nr_items):
            cost = random.randint(1,30)
            weight = random.randint(1,30)
            self.items.append(item(weight,cost))
'''

newProblem = Knapsack(12,4)

for i in range(4):
    print(newProblem.items[i].value)
    print(newProblem.items[i].weight)

