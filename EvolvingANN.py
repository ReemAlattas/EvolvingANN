import numpy
import random
import matplotlib as plt
import math

numNeurons = 10

def MatrixCreate(rows, cols):
    matrix = numpy.zeros(shape=(rows, cols))
    return matrix
    
def MatrixRandomize(v):
    random_m = [[random.random() for y in range(len(v[x]))] for x in range(len(v))]
    return random_m

parent = MatrixCreate(numNeurons, numNeurons)
parent = MatrixRandomize(parent)

print (parent)