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
    random_m = [[random.uniform(-1, 1) for y in range(len(v[x]))] for x in range(len(v))]
    return random_m
    
def Update (neuronValues, synapses, i):
    temp = 0
    sum = 0 
    for j in range(numUpdates):
        for k in range(numUpdates):
            temp = neuronValues[i-1][k] * synapses[j][k]
            sum = sum + temp
        if (sum < 0):
            sum = 0
        if (sum > 1):
            sum = 1
        neuronValues[i][j] = sum        
    return neuronValues
    
def FitnessParent(parent):
    neuronValues = MatrixCreate(numUpdates, numNeurons)
    neuronValues[0] = 0.5
    print neuronValues
    for i in range(1, numUpdates):
        Update (neuronValues, parent, i)
    print neuronValues
    #return fitness
    
def HillClimber(generation):
    parent = MatrixCreate(numNeurons,numNeurons) 
    parent = MatrixRandomize(parent) 
    print parent
    parentFitness = FitnessParent(parent) 
    #     for currentGeneration in range(0,5000):
    #          print currentGeneration, parentFitness 
    #          child = MatrixPerturb(parent,0.05)   
    #          childFitness = Fitness(child) 
    #          if ( childFitness > parentFitness ):
    #               parent = child 
    #               parentFitness = childFitness


HillClimber(1000)

#desiredNeuronValues = np.zeros(10)
#
#for i in range(1, numUpdates):
#    neuronValues = Update (neuronValues, parent, i)
#
#print (neuronValues)

