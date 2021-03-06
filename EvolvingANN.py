import numpy as np
import random
import matplotlib as plt
import copy
#import math

numNeurons = 10
numUpdates = 10

def VectorCreate(width):
    v = np.zeros((width), dtype='f')
    return v

def MeanDistance(v1,v2):
    d = ((v1 - v2) ** 2).mean(axis=None)
    return d
    
def MatrixCreate(rows, cols):
    matrix = np.zeros(shape=(rows, cols))
    return matrix
    
def MatrixRandomize(v):
    random_m = [[random.uniform(-1, 1) for y in range(len(v[x]))] for x in range(len(v))]
    return random_m
    
def MatrixPerturb(p, prob):
    c = copy.deepcopy(p)
    #print(p)
    #print(c)
    for x in range(len(c)):
        for y in range(len(c[x])):
            if prob > random.random():
                c[x][y] = random.random()
    return c
    
def Fitness(v):
    mean_val = np.mean(v)
    return mean_val
    
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
    
def FitnessParent(neuronValues, parent):
    neuronValues = MatrixCreate(numUpdates, numNeurons)
    neuronValues[0] = 0.5
    print neuronValues
    for i in range(1, numUpdates):
        Update (neuronValues, parent, i)
    print neuronValues
    
    actualNeuronValues = neuronValues[9,:]
    
    desiredNeuronValues = VectorCreate(10)
    for j in range(1,10,2):
        desiredNeuronValues[j]=1
    #print desiredNeuronValues
    
    d = MeanDistance(actualNeuronValues, desiredNeuronValues)
    fit = 1 - d
    print fit
    ### PLOT ###
    #plt.pyplot.imshow(neuronValues, cmap=plt.cm.gray, aspect='auto', interpolation='nearest')
    #plt.pyplot.show()
    
    return fit
    
    
def HillClimber(generation):
    parent = MatrixCreate(numNeurons,numNeurons) 
    parent = MatrixRandomize(parent) 
    #print parent
    
    neuronValues = MatrixCreate(numUpdates, numNeurons)
    neuronValues[0] = 0.5
    
    parentFitness = FitnessParent(neuronValues, parent) 
    
    neuronValues = MatrixCreate(numUpdates, numNeurons)
    neuronValues[0] = 0.5
    
    for i in range(1, numUpdates):
        Update (neuronValues, parent, i)
    #print neuronValues
    
    #plt.pyplot.xlabel('Neuron')
    #plt.pyplot.ylabel('Time Step')
    plt.pyplot.imshow(neuronValues, cmap=plt.cm.gray, aspect='auto', interpolation='nearest')
    plt.pyplot.show()
    
    for currentGeneration in range(0,5000):
         print currentGeneration, parentFitness
         child = MatrixPerturb(parent,0.05)   
         childFitness = Fitness(child)
         print childFitness 
         if ( childFitness > parentFitness ):
              parent = child 
              parentFitness = childFitness


HillClimber(1000)

#desiredNeuronValues = np.zeros(10)
#
#for i in range(1, numUpdates):
#    neuronValues = Update (neuronValues, parent, i)
#
#print (neuronValues)

