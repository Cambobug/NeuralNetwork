import numpy as np
import math as m
import copy as c
import pickle as pick
import os
import matplotlib
import matplotlib.pyplot as plt

#creates the training, testing, and validation sets from the csv files
def readTrainingFile():

    if os.path.isfile("trainingData.pkl") and os.path.isfile("trainingDataLabel.pkl") and os.path.isfile("testingData.pkl") and os.path.isfile("testingDataLabel.pkl") and os.path.isfile("validationData.pkl") and os.path.isfile("validationDataLabel.pkl"):

        trainFile = open("trainingData.pkl", "rb")
        newtrainingData = pick.load(trainFile)

        trainFileLabel = open("trainingDataLabel.pkl", "rb")
        newtrainingDataLabel = pick.load(trainFileLabel)
        
        testFile = open("testingData.pkl", "rb")
        testingArray = pick.load(testFile)
        
        testFileLabel = open("testingDataLabel.pkl", "rb")
        testingDataLabel = pick.load(testFileLabel)
        
        validFile = open("validationData.pkl", "rb")
        validArray = pick.load(validFile)

        validFileLabel = open("validationDataLabel.pkl", "rb")
        validDataLabel = pick.load(validFileLabel)
    else:

        file = open("mnist_train.csv", "r")
        file.readline()

        trainingData = np.empty((60000, 784))
        trainingDataLabel = np.empty((60000), dtype=int)

        for i in range(0, 60000):
            line = file.readline()
            line = line.strip()
            nums = line.split(",")
            trainingDataLabel[i] = nums[0]
            nums.pop(0) 
            nums = np.array(nums)
            for j in range(0, 784):
                trainingData[i][j] = c.deepcopy(int(nums[j])/255)
                
        testingArray = trainingData[36000:48000]
        validArray = trainingData[48000:]
        newtrainingData = trainingData[:36000]
        
        testingDataLabel = trainingDataLabel[36000:48000]
        validDataLabel = trainingDataLabel[48000:]
        newtrainingDataLabel = trainingDataLabel[:36000]

        trainFile = open("trainingData.pkl", "wb")
        pick.dump(newtrainingData, trainFile)

        trainFileLabel = open("trainingDataLabel.pkl", "wb")
        pick.dump(newtrainingDataLabel, trainFileLabel)
        
        testFile = open("testingData.pkl", "wb")
        pick.dump(testingArray, testFile)

        testFileLabel = open("testingDataLabel.pkl", "wb")
        pick.dump(testingDataLabel, testFileLabel)
        
        validFile = open("validationData.pkl", "wb")
        pick.dump(validArray, validFile)

        validFileLabel = open("validationDataLabel.pkl", "wb")
        pick.dump(validDataLabel, validFileLabel)

    return newtrainingData, newtrainingDataLabel, testingArray, testingDataLabel, validArray, validDataLabel

#sigmoidal activation function
def sigmoid(x):
    return 1/(1 + m.exp(-x))

#creates the plot graph for the final result
def plot(trainingLoss, trainingAccuracy, validationLoss, validationAccuracy, testingLoss, testingAccuracy, networkLayout):
    lossPlot = plt.subplot(1,2,1)
    accPlot = plt.subplot(1,2,2)
    lossPlot.set_title(str(networkLayout) + ": Test Accuracy = " + str(round(testingAccuracy,3)), loc="left")
    lossPlot.set_xlabel("Epoch")
    lossPlot.set_ylabel("Loss")
    lossPlot.set_xticks(np.arange(0, 101, 10))
    epochs = []
    for i in range(100):
        epochs.append(i)
    lossPlot.plot(epochs, trainingLoss)
    lossPlot.plot(epochs, validationLoss)
    lossPlot.legend(["Training", "Validation"], loc="upper left")
    
    accPlot.set_xlabel("Epoch")
    accPlot.set_ylabel("Accuracy")
    accPlot.set_yticks(np.arange(0, 1, 0.1))
    accPlot.set_xticks(np.arange(0, 101, 10))
    accPlot.plot(epochs, trainingAccuracy)
    accPlot.plot(epochs, validationAccuracy)
    accPlot.legend(["Training", "Validation"], loc="lower right")
    
    plt.subplots_adjust(wspace=0.3)
    plt.show()
    