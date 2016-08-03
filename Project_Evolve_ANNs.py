import numpy as np
import random
import matplotlib as plt
import copy

def MatrixCreate(rows, cols):
    matrix = np.zeros(shape=(rows, cols))
    return matrix
    
def MatrixRandomize(v):
    random_m = [[random.uniform(-1, 1) for y in range(len(v[x]))] for x in range(len(v))]
    return random_m
    
parent = MatrixCreate(1,50) 
parent = MatrixRandomize(parent) 
#     parentFitness = FitnessParent(parent) 
#     for currentGeneration in range(0,5000):
#          print currentGeneration, parentFitness 
#          child = MatrixPerturb(parent,0.05) 
#          childFitness = Fitness(child) 
#          if ( childFitness > parentFitness ):
#               parent = child 
#               parentFitness = childFitness

