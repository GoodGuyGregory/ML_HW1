import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time
import copy




class Perceptron_Optimizer:

    def backward(self, perceptronLayer, truthVector, input):
        # startTime = time.time()
        #  determine accuracy
        perceptronLayer.updateAccuracy(truthVector)

        # compute the error at the output:
        inputActivationsDiff = 1 - perceptronLayer.outputLogits
        truthVectorDiff = truthVector - perceptronLayer.outputLogits
        # based on the equation from the notes.
        deltaKError = perceptronLayer.outputLogits * inputActivationsDiff * truthVectorDiff  # 10, 1

        #  compute the error for the hidden layer
        hiddenError = 1.0 - np.append(perceptronLayer.hiddenLogits, 1)
        # outer will allow for the deltaKError [10x1] to be combined with the [Nx1]
        # to form a [20 x 10], [50 x 10]. [100 x 10]
        # print(perceptronLayer.hiddenLogits.shape)
        hiddenDelta = np.append(perceptronLayer.hiddenLogits, 1) * hiddenError * np.dot(perceptronLayer.hWeights, deltaKError)

        # update the hidden to output weights.
        # perceptronLayer.hWeights += perceptronLayer.eta * np.outer(deltaKError, np.append(perceptronLayer.hiddenLogits, 1)).T
        # This is the line that is implemented from the slides as specified but is causing problems
        hiddenWeightDelta = perceptronLayer.eta * np.outer(deltaKError, np.append(perceptronLayer.hiddenLogits, 1)).T
        perceptronLayer.hWeights += hiddenWeightDelta + (perceptronLayer.momentum * perceptronLayer.prevHiddenWeightDelta)

        # update the input to the hidden weights
        # perceptronLayer.weights += perceptronLayer.eta * np.outer(hiddenDelta[:-1], input).T
        # This is the line that is implemented from the slides as specified but is causing problems
        inputWeightDelta = perceptronLayer.eta * np.outer(hiddenDelta[:-1], input).T
        perceptronLayer.weights += inputWeightDelta + (perceptronLayer.momentum * perceptronLayer.prevInputWeightDelta)

        #      set the previous weights for later momentum calculations:
        perceptronLayer.prevHiddenWeightDelta = hiddenWeightDelta
        perceptronLayer.prevInputWeightDelta = inputWeightDelta
        # print(time.time() - startTime , "BackProp")


    #  produces on hot encoded sigmoid vector
    def convertTruth(self, Y):
        groundTruth = np.ones(10, dtype='float') * 0.1
        groundTruth[Y] = 0.9

        return groundTruth



class Perceptron_Layer:
    def __init__(self, n_inputs, n_neurons, h_neurons, learning_rate, momentum):
        self.eta = learning_rate
        self.momentum = momentum

        #  standard input weights
        self.weights = np.random.uniform(low=-0.05, high=0.05, size=(n_inputs + 1, h_neurons))
        # add a bias term inside each weight vector

        # hidden weights.
        self.hWeights = np.random.uniform(low=-0.05, high=0.05, size=(h_neurons + 1, n_neurons))
        # add a bias term inside each weight vector

        self.hiddenLogits = []

        self.outputLogits = []

        # initialize weights to 0 sized arrays similar to the hWeights and weights

        self.prevHiddenWeightDelta = np.zeros((h_neurons+1,n_neurons))
        self.prevInputWeightDelta = np.zeros((n_inputs+1,h_neurons))

        self.accuracy = 0.0
        self.accurateCount = 0
        self.incorrect = 0
        self.inputSize = 0

    def sigmoid(self, x):
        x *= -1
        return 1 / (1 + np.exp(x))

    #  append the bias.
    def forward(self, input):
        # startTime = time.time()
        #  convert the target to a vector representation

        # original [784, 1]
        # inputs [785x1] dot [785 x N] + [1xN]
        layerOneLogits = np.dot(input.T, self.weights)
        self.hiddenLogits = self.sigmoid(layerOneLogits)

        # print(self.hiddenLogits.shape)
        # self.hiddenLogits = np.append(self.hiddenLogits, 1)
        # hWeights [10 x 50] dot [1x50]
        # print(self.hWeights.shape)
        hiddenLayerLogits = np.dot(self.hWeights.T, np.append(self.hiddenLogits, 1).T)

        # activation function
        self.outputLogits = self.sigmoid(hiddenLayerLogits)


        self.inputSize += 1
        # print(time.time() - startTime, "Forward")



    def displayAccuracy(self):
        if self.inputSize == 0:
            print("No input data. Accuracy set to 0.")
            self.accuracy = 0.0
        else:
            self.accuracy = round((self.accurateCount / self.inputSize) * 100, 2)
            print("Accuracy: " + str(self.accuracy) + "%")

    def updateAccuracy(self, truthVector):
        if np.argmax(truthVector) == np.argmax(self.outputLogits):
            self.accurateCount += 1

    def calculateConfusionMatrix(self):
        cm = confusion_matrix(self.grounds, self.potentials)
        return cm

    def displayConfusionMatrix(self, cm):
        print("Confusion Matrix:")
        print(cm)

    def plotConfusionMatrix(self, cm, testType):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=np.arange(10), yticklabels=np.arange(10))
        plt.title("Confusion Matrix " + str(testType))
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()

    def clearAccuracy(self):
        self.accuracy = 0.0
        self.accurateCount = 0
        self.incorrect = 0
        self.inputSize = 0
        self.grounds = []
        self.potentials = []

    def plotInitialAccuracy(self, accuracies):
        plt.figure(figsize=(8, 6))
        plt.plot(np.arange(1, len(accuracies) + 1), accuracies, marker='o')
        plt.title("Accuracy Over Epochs")
        plt.xlabel("Batch")
        plt.ylabel("Accuracy (%)")
        plt.grid(True)
        plt.show()

    def plotAccuracyOverTestTrainingEpochs(self, trainAccuracies, testAccuracies, epochs):
        plt.figure(figsize=(8, 6))
        plt.plot(trainAccuracies, marker='o')
        plt.plot(testAccuracies, marker='o')
        plt.title("Accuracy Over Epochs Against Test Samples")
        plt.xlabel("Batch")
        plt.ylabel("Accuracy (%)")
        # plt.xticks(np.arange(0, epochs), fontsize=12)
        # # Set the number of intervals on the y-axis
        # plt.yticks(np.arange(0, 100, step=15))
        plt.grid(True)
        plt.show()

    def plotTestAccuracySingleEpoch(self, testAccuracies):
        plt.figure(figsize=(8, 6))
        plt.plot(np.arange(1, len(testAccuracies) + 1), testAccuracies, marker='o')
        plt.title("Accuracy Over Single Test Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.grid(True)
        plt.show()


def getTrainingLabels(dataFrame):
    groundTruthLables = dataFrame['label'].to_numpy()
    # drop the labels of the data and then divide each array
    return groundTruthLables


def getTrainingInputs(dataFrame):

    dataFrame['bias'] = 255
    inputs = dataFrame.drop("label", axis=1).to_numpy()
    # divide each value vector by 255...
    normalizedInputs = inputs / 255
    return normalizedInputs


def shuffleTrainData():
    mnist_data = pd.read_csv("mnist_train.csv")
    # shuffle the contents of the dataset for training
    #  the prevents the perceptrons from being trained on ordering of the input data.
    shuffledMnistData = mnist_data.sample(frac=1)
    return shuffledMnistData


def getTesingLabels(dataFrame):
    groundTruthTestLables = dataFrame['label'].to_numpy()
    # drop the labels of the data and then divide each array
    return groundTruthTestLables


def getTestingInputs(dataFrame):
    inputs = dataFrame.drop("label", axis=1).to_numpy()
    # divide each value vector by 255...
    normalizedTestInputs = inputs / 255
    return normalizedTestInputs


def shuffleTestData():
    mnist_data = pd.read_csv("mnist_test.csv")
    # shuffle the contents of the dataset for training
    #  the prevents the perceptrons from being trained on ordering of the input data.
    shuffledMnistTestData = mnist_data.sample(frac=1)
    return shuffledMnistTestData


# input 785 inputs within 10 perceptrons each input is a vector of 785 ie 28x28+1 (+1 for bias)

def main():
    # read in the MNIST data:

    normalizedInputs = shuffleTrainData()

    Y = getTrainingLabels(normalizedInputs)
    x = getTrainingInputs(normalizedInputs)

    epoch = 0
    # batch = 0
    #  add loop to control epoch count
    perceptronLayer = Perceptron_Layer(784, 10, 50, 0.1, 0.9)

    perceptronOptimizer = Perceptron_Optimizer()

    while epoch <= 50:
        print("epoch" + str(epoch))
        batch = 0
        normalizedInputs = shuffleTrainData()

        Y = getTrainingLabels(normalizedInputs)
        x = getTrainingInputs(normalizedInputs)
        while batch < len(normalizedInputs):
            # print("batch:" +  str(batch))


            #  forward pass through the connected input layer and connected layer.

            perceptronLayer.forward(x[batch])

            #  convert truth value
            groundTruth = perceptronOptimizer.convertTruth(Y[batch])

            #  backward pass will update the current neuron
            #  update it's weights based on error.
            perceptronOptimizer.backward(perceptronLayer, groundTruth, x[batch])

            batch += 1
            if batch % 5 == 0:
                perceptronLayer.displayAccuracy()
        epoch += 1
    print("done training")



main()