import numpy as np
import random
import matplotlib as plt
import math

numNeurons = 10
numUpdates = 10

def MatrixCreate(rows, cols):
    matrix = np.zeros(shape=(rows, cols))
    return matrix
    
def MatrixRandomize(v):
    random_m = [[random.random() for y in range(len(v[x]))] for x in range(len(v))]
    return random_m
    
def Update (neuronValues, synapses, i):
    temp = 0
    sum = 0 
    for j in range(numNeurons):
        for k in range(numNeurons):
            temp = neuronValues[i-1][k] * synapses[j][k]
            sum = sum + temp
        if (sum < 0):
            sum = 0
        if (sum > 1):
            sum = 1
        neuronValues[i][j] = sum        
    return neuronValues

parent = MatrixCreate(numNeurons, numNeurons)
parent = MatrixRandomize(parent)

neuronValues = MatrixCreate(numUpdates, numNeurons)
neuronValues[0] = 0.5

desiredNeuronValues = np.zeros(10)

for i in range(1, numUpdates):
    neuronValues = Update (neuronValues, parent, i)

print (desiredNeuronValues)