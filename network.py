import numpy as np
import helpers as h
import net
import time as t

def expectedIndex(layerLabel): #returns a ten long array which has a 1 in index that corresponds to that images label
        expectedArray = np.zeros(10, dtype=float)
        expectedArray[int(layerLabel)] = 1.0
        
        return expectedArray
    
def createHiddenLayers(numHiddenLayers, hiddenSizes):
        hiddenLayers = []
        
        #creates x number of np.arrays and places them in the hiddenlayers list
        for i in range(0, numHiddenLayers):
            hiddenLayers.append(np.zeros(hiddenSizes[i], dtype=float))
            
        return hiddenLayers
    
def getResultFromOutput(outputLayer): #takes in a 10 long array and finds the highest value in the array and returns the corresponding index
    maxVal = -10000
    maxIndx = -1

    for i in range(0, 10):
        currValue = outputLayer[i]
        if(currValue > maxVal):
            maxVal = currValue
            maxIndx = i

    return maxIndx

def forwardPass(neuronLayers, neuronLayersSizes, weightLayers, bias):

    #layer traversal
    for layerNum in range(0, len(neuronLayers) - 1): #for each layer except last
        currLayer = neuronLayers[layerNum] #gets current working layer
        xs = []
        for nextLayerNode in range(0, neuronLayersSizes[layerNum + 1]): 
            x = np.dot(currLayer, weightLayers[layerNum][nextLayerNode]) + bias # gets the dot product of the layer and the weights
            xs.append(x)
            neuronLayers[layerNum + 1][nextLayerNode] = h.sigmoid(x) #applies sigmoidal activation function to x
        
    return neuronLayers[len(neuronLayers)-1]
    
def backwardsPass(expectedOut, neuronLayers, weightLayers, learningRate):
    
    #the actual output of the forward pass - expected output (bunch of 0s and a single 1)
    outputError =  (neuronLayers[len(neuronLayers)-1] - expectedOut)
    deltas = []
    
    #delta of output layer is error * actual output * (1-actual output)
    deltaOutput = outputError * neuronLayers[len(neuronLayers)-1] * (1 - neuronLayers[len(neuronLayers)-1])
    deltas.append(deltaOutput)
    
    #gets delta of hidden layers
    deltaHidden = 0
    for i in range(len(neuronLayers)-2, 0, -1): # moves backwards through the hidden layers starting from the hiddenlayer behind the output
        dot = np.dot(deltas[0], weightLayers[i]) # calculates the dot product of current weights by the prevous delta
        deltaHidden = dot * neuronLayers[i] * (1 - neuronLayers[i])
        deltas.insert(0, deltaHidden) 
        
    newWeights = []
    for i in range(0, len(weightLayers)):
        weightGradient = np.outer(deltas[i], neuronLayers[i]) # calculates the weight gradients using the previously calculated deltas
        newWeights.append(weightLayers[i] - (learningRate * weightGradient)) # multiplies the change in weights by the learning rate and subracts from the weights
        
    return newWeights
        
def training(network, trainingArray, trainingLabels, validationArray, validationLabels):
    trainingLosss = []
    trainingAccuracys = []
    validationLosss = []
    validationAccuracys= []
    
    # creates hidden layers and output layer
    hiddenlayers = createHiddenLayers(network.numHiddenLayers, network.hiddenSizes)
    outputLayer = np.zeros(10, dtype=float)

    neuronLayersSizes = []
    neuronLayers = []
    
    #places all layer sizes in a single array
    neuronLayersSizes.append(784)
    for i in range(1, len(network.hiddenSizes) + 1):
        neuronLayersSizes.append(network.hiddenSizes[i - 1])
    neuronLayersSizes.append(10)
    
    #places all layers in a single array
    neuronLayers.insert(0,None)
    for i in range(1, len(hiddenlayers) + 1):
        neuronLayers.append(hiddenlayers[i - 1])
    neuronLayers.append(outputLayer)
    
    progTime = t.time()
    for e in range(0, network.epochs): # runs the training set through the epochs
        eLoss = []
        correctPred = 0
            
        startTime = t.time()
        for i in range(0, len(trainingLabels)):
            
            #gets the expected output layer from the current images label
            expectedOut = expectedIndex(trainingLabels[i])
            #refreshed the layers for the next input
            neuronLayers[0] = trainingArray[i] #sets the new image as first layer
            neuronLayers[len(neuronLayers)-1] = np.zeros(10, dtype=float) #sets the last layer as all zeros
            
            #runs forward pass on the image, populating the layers with values
            outputLayer = forwardPass(neuronLayers, neuronLayersSizes, network.weightLayers, network.bias)
            #used for accuracy at end of epoch
            if(getResultFromOutput(outputLayer) == trainingLabels[i]):
                correctPred += 1
            #used for loss at end of epoch
            layerLoss = np.mean((expectedOut - outputLayer)**2)
            eLoss.append(layerLoss)
            #runs backward propagation on the layers to adjust weights
            network.weightLayers = backwardsPass(expectedOut, neuronLayers, network.weightLayers, network.learningRate)
            
        #per epoch stats
        print("Epoch: " + str(e+1) + " Accuracy: " + str(round(float(correctPred/len(trainingLabels)), 3)) + " Loss: " + str(round(float(sum(eLoss)/len(trainingLabels)), 3)))
        endTime = t.time()
        totalTime = endTime - startTime
        print("Time Taken: " + str(totalTime))
        
        #testing classification
        trainingLoss, trainingAccuracy = classify(network, trainingArray, trainingLabels)
        validationLoss, validationAccuracy = classify(network, validationArray, validationLabels)
        
        trainingLosss.append(trainingLoss)
        trainingAccuracys.append(trainingAccuracy)
        validationLosss.append(validationLoss)
        validationAccuracys.append(validationAccuracy)
       
    progEndTime = t.time()
    finalTime = progEndTime - progTime 
    print("============================= Total Time Taken: " + str(finalTime))
    return trainingLosss, trainingAccuracys, validationLosss, validationAccuracys

def classify(network, testData, testLabels):
    totalAccuracy = 0
    layerLoss = 0
    
    # creates hidden and output layers
    hiddenlayers = createHiddenLayers(network.numHiddenLayers, network.hiddenSizes)
    outputLayer = np.zeros(10, dtype=float)
    
    neuronLayersSizes = []
    neuronLayers = []
    
    #places all layer sizes in a single array
    neuronLayersSizes.append(784)
    for i in range(1, len(network.hiddenSizes) + 1):
        neuronLayersSizes.append(network.hiddenSizes[i - 1])
    neuronLayersSizes.append(10)
    
    #places all layers in a single array
    neuronLayers.insert(0,None)
    for i in range(1, len(hiddenlayers) + 1):
        neuronLayers.append(hiddenlayers[i - 1])
    neuronLayers.append(outputLayer)
    
    hiddenlayers = None
    outputLayer = None
    
    for i in range(0, len(testLabels)):
        #gets the expected output layer from the current images label
        expectedOut = expectedIndex(testLabels[i])
        #refreshed the layers for the next input
        neuronLayers[0] = testData[i]
        #runs forward pass on the given test data
        outputLayer = forwardPass(neuronLayers, neuronLayersSizes, network.weightLayers, network.bias)
        #gets the result of the forward pass
        if(getResultFromOutput(outputLayer) == testLabels[i]):
            totalAccuracy += 1
        #determines the loss using mean squared
        layerLoss = np.mean((expectedOut - outputLayer)**2)
        
    return layerLoss/len(testData), totalAccuracy/len(testData)
        

#main script
trainingArray, trainingLabels, testingArray, testingLabels, validationArray, validationLabel = h.readTrainingFile()

network = net.network(0.01, 100, 1, 784, 0, [])
trainingLosss, trainingAccuracys, validationLosss, validationAccuracys = training(network, trainingArray, trainingLabels, validationArray, validationLabel)
testLoss, testAccuracy = classify(network, testingArray, testingLabels)

print("Final Accuracy: " + str(round(testAccuracy, 3)) + " Final Loss: " + str(round(testLoss, 3)))

h.plot(trainingLosss, trainingAccuracys, validationLosss, validationAccuracys, testLoss, testAccuracy, "[784, 10]")