import numpy as np
import random
import matplotlib as plt
import copy

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
    
def FitnessParent(parent):
    neuronValues = MatrixCreate(numUpdates, numNeurons)
    neuronValues[0] = 0.5
    for i in range(1, numUpdates):
        Update (neuronValues, parent, i)
    #print "Neuron Values", neuronValues
    
    ### PLOT ###
    plt.pyplot.imshow(neuronValues, cmap=plt.cm.gray, aspect='auto', interpolation='nearest')
    plt.pyplot.show()
    
    actualNeuronValues = neuronValues[9,:]
    desiredNeuronValues = VectorCreate(10)
    for j in range(1,10,2):
        desiredNeuronValues[j] = 1
    print "Actual", actualNeuronValues
    print "Desired", desiredNeuronValues  
    return neuronValues

#The synaptic weights of the parent neural network
parent = MatrixCreate(numNeurons,numNeurons) 
parent = MatrixRandomize(parent)
print "Parent", parent 
parentFitness = FitnessParent(parent) 
#     for currentGeneration in range(0,5000):
#          print currentGeneration, parentFitness 
#          child = MatrixPerturb(parent,0.05) 
#          childFitness = Fitness(child) 
#          if ( childFitness > parentFitness ):
#               parent = child 
#               parentFitness = childFitness

