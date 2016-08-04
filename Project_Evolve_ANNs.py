import numpy as np
import random
import matplotlib as plt
import copy

numNeurons = 10
numUpdates = 10
numGenerations = 1000

def VectorCreate(width):
    v = np.zeros((width), dtype='f')
    return v
    
def VectorPlot(v):
    plt.pyplot.plot(v)
    plt.pyplot.show()

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
    
def MatrixPlot(neuronValues):
    plt.pyplot.imshow(neuronValues, cmap=plt.cm.gray, aspect='auto', interpolation='nearest')
    plt.pyplot.show()
    
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

neuronValues = MatrixCreate(numUpdates, numNeurons)  
neuronValues[0] = 0.5

def FitnessParent(parent):
    for i in range(1, numUpdates):
        Update (neuronValues, parent, i)
    #print "Neuron Values", neuronValues
    
    ### PLOT ###
    #MatrixPlot(neuronValues)
    
    actualNeuronValues = neuronValues[9,:]
    desiredNeuronValues = VectorCreate(10)
    for j in range(1,10,2):
        desiredNeuronValues[j] = 1
    #print "Actual", actualNeuronValues
    #print "Desired", desiredNeuronValues 
    
    #Compute Mean Distance
    d = MeanDistance(actualNeuronValues, desiredNeuronValues)
    f = 1 - d
    #print "Fitness = ", f
    return f
    
def Fitness2(parent):
    for i in range(1, numUpdates):
        Update (neuronValues, parent, i)
    #print "Neuron Values", neuronValues
    
    ### PLOT ###
    #MatrixPlot(neuronValues)
    
    actualNeuronValues = neuronValues[9,:]
    desiredNeuronValues = VectorCreate(10)
    for j in range(1,10,2):
        desiredNeuronValues[j] = 1
    #print "Actual", actualNeuronValues
    #print "Desired", desiredNeuronValues 
    
    #Compute Avg Difference
    diff = 0.0
    for i in range(1,9): 
        for j in range(0,9):
            diff=diff + abs(neuronValues[i, j] - neuronValues[i, j+1])
            diff=diff + abs(neuronValues[i+1, j] - neuronValues[i, j]) 
    diff=diff/(2*8*9)
    f = 1 - diff
    #print "Fitness = ", f
    return f

#The synaptic weights of the parent neural network
parent = MatrixCreate(numNeurons,numNeurons) 
parent = MatrixRandomize(parent)
#print "Parent", parent 
#parentFitness = FitnessParent(parent) 
parentFitness = Fitness2(parent)

fitnessVector = VectorCreate(numGenerations)

#neuronValues = MatrixCreate(numUpdates, numNeurons)
#neuronValues[0] = 0.5
#for i in range(1, numUpdates):
#        Update (neuronValues, parent, i)
#        

#MatrixPlot(neuronValues)

for currentGeneration in range(0,numGenerations):
    #print currentGeneration, parentFitness 
    fitnessVector[currentGeneration] = parentFitness
    child = MatrixPerturb(parent,0.05) 
    #childFitness = FitnessParent(child) 
    childFitness = Fitness2(parent)
    if ( childFitness > parentFitness ):
        parent = child 
        parentFitness = childFitness
        
for i in range(1, numUpdates):
        Update (neuronValues, parent, i)

#print "Neuron Values", neuronValues        
### PLOT ###
#MatrixPlot(neuronValues)
VectorPlot(fitnessVector)

#print "Parent = ", parent
#print "Child = ", child
print "Fitness Vector = ", fitnessVector

