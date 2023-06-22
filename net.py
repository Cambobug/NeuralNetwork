import numpy as np
import math as m
import copy as c

class network():
    def __init__(self, learningRate, epochs, bias, inputLayerSize, numHiddenLayers, hiddenSizes):
        self.learningRate = learningRate
        self.epochs = epochs
        self.bias = bias
        self.inputLayerSize = inputLayerSize
        self.numHiddenLayers = numHiddenLayers
        self.hiddenSizes = hiddenSizes
        self.weightLayers = self.createInitialWeights(self.numHiddenLayers + 2, 784, self.hiddenSizes, 10)
        
    def createInitialWeights(self, numTotalLayers, inputLayerSize, hiddenLayerSizes, outputLayerSize):
    
        neuronLayersSizes = []
        weightLayers = []
        numWeightLayers = numTotalLayers - 1 
        
        #gets all of the layer sizes into one array in order of which they are used
        neuronLayersSizes.append(inputLayerSize)
        for i in range(1, len(hiddenLayerSizes) + 1):
            neuronLayersSizes.append(hiddenLayerSizes[i - 1])
        neuronLayersSizes.append(outputLayerSize)
        
        #creates inital weights using He Weight Initialization
        for i in range(0, numWeightLayers):
            numPrevNodes = neuronLayersSizes[i]
            lower, upper = -(1.0/m.sqrt(numPrevNodes)), (1.0/m.sqrt(numPrevNodes))

            randWeights = np.random.randn(neuronLayersSizes[i+1], neuronLayersSizes[i])
            scaledRandWeights = lower + randWeights * (upper - lower)
            weightLayers.append(c.deepcopy(scaledRandWeights)) 
            
        return weightLayers