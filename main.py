import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import math
import time




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

        # confusion matrix variables:
        self.y_predictions = []
        self.y_truths = []

        self.accuracy = 0.0
        self.accurateCount = 0
        self.incorrect = 0
        self.inputSize = 0

    def sigmoid(self, x):
        x = np.array(x)
        x *= -1
        return 1 / (1 + np.exp(x))

    #  append the bias.
    def forward(self, input, confusion_matrix):
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

        if confusion_matrix:
            # used to store predictions for confusion matrix
            self.y_predictions.append(self.outputLogits)


        self.inputSize += 1
        # print(time.time() - startTime, "Forward")



    def displayAccuracy(self, modelType):
        if self.inputSize == 0:
            print("No input data. Accuracy set to 0.")
            self.accuracy = 0.0
        else:
            self.accuracy = round((self.accurateCount / self.inputSize) * 100, 2)
            print(modelType + " Accuracy: " + str(self.accuracy) + "%")


    def updateAccuracy(self, truthVector):
        if np.argmax(truthVector) == np.argmax(self.outputLogits):
            self.accurateCount += 1

    def calculateConfusionMatrix(self):
        # specified the argMax as the truths and predictions
        y_truths = np.concatenate(self.y_truths).reshape(-1, 10)
        y_predictions = np.concatenate(self.y_predictions).reshape(-1, 10)

        # Build the confusion matrix
        cm = confusion_matrix(y_truths.argmax(axis=1), y_predictions.argmax(axis=1))
        return cm

    def plotConfusionMatrix(self, cm, testType):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=np.arange(10), yticklabels=np.arange(10))
        plt.title("Confusion Matrix " + str(testType))
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.savefig('100_hidden_neuron_layer_momentum_0.0_confusion_matrix.png')
        plt.show()

    def clearAccuracy(self):
        self.accuracy = 0.0
        self.accurateCount = 0
        self.incorrect = 0
        self.inputSize = 0



    def plotAccuracyOverTestTrainingEpochs(self, trainAccuracies, testAccuracies, epochs, title):
        plt.figure(figsize=(8, 6))
        plt.plot(trainAccuracies, marker='o', label='Training Accuracy')
        plt.plot(testAccuracies, marker='o', label='Test Accuracy')
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.xticks(np.arange(0, epochs, step=5))  # Adjust the step according to your data
        plt.yticks(np.arange(80, 100, step=1))  # Adjust the step according to your data
        plt.ylim(80, 100)  # Set y-axis limits for better visualization
        plt.legend()  # Add legend to differentiate between training and test accuracies
        plt.grid(True)
        plt.savefig('100_hidden_neuron_layer_momentum_0_accuracy.png')
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

def splitTrainingData(splitRatio):
    #  opens the training data
    mnist_data = pd.read_csv("mnist_train.csv")
    # determines the split size
    trainingQty_split = len(mnist_data) * splitRatio
    #  determines number needed to evenly divide per output values 0-9
    needed_input_qty_per_output = int(math.floor(trainingQty_split / 10))

    #  concatenates the rand 0-9 for the input data based on label and sample size
    splitTrainingDF = pd.concat([
        mnist_data[mnist_data['label'] == i].sample(needed_input_qty_per_output)
        for i in range(10)
    ], ignore_index=True)
    # shuffles the data
    return splitTrainingDF.sample(frac=1)


def getTestingLabels(dataFrame):
    groundTruthTestLables = dataFrame['label'].to_numpy()
    # drop the labels of the data and then divide each array
    return groundTruthTestLables


def getTestingInputs(dataFrame):
    dataFrame['bias'] = 255
    inputs = dataFrame.drop("label", axis=1).to_numpy()
    # divide each value vector by 255...
    normalizedInputs = inputs / 255
    return normalizedInputs


def shuffleTestData():
    mnist_data = pd.read_csv("mnist_test.csv")
    # shuffle the contents of the dataset for training
    #  the prevents the perceptrons from being trained on ordering of the input data.
    shuffledMnistTestData = mnist_data.sample(frac=1)
    return shuffledMnistTestData


# input 785 inputs within 10 perceptrons each input is a vector of 785 ie 28x28+1 (+1 for bias)

def main():
    train_epoch = 0
    # batch = 0
    #  add loop to control epoch count
    perceptronLayerOne = Perceptron_Layer(784, 10, 100, 0.1, 0.9)

    perceptronOptimizer = Perceptron_Optimizer()

    # create lists to hold the training results for graphing.
    trainingDataAccuracyResults = []
    testingDataAccuracyResults = []

    while train_epoch < 50:
        print("Training and Test Epoch " + str(train_epoch) + ":")
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        train_batch = 0

        normalizedTrainingInputs = shuffleTrainData()

        # EXPERIMENT #3:
        # uncomment
        # sub splits and shuffles the training data
        # normalizedTrainingInputs = splitTrainingData(.25)

        Y_train = getTrainingLabels(normalizedTrainingInputs)
        x_train = getTrainingInputs(normalizedTrainingInputs)
        print("Training Epoch: ")
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        while train_batch < len(normalizedTrainingInputs):
            print("Training batch:" +  str(train_batch))
            #  forward pass through the connected input layer and connected layer.

            #  second argument for confusion matrix
            perceptronLayerOne.forward(x_train[train_batch], False)

            #  convert truth value
            train_groundTruth = perceptronOptimizer.convertTruth(Y_train[train_batch])

            #  backward pass will update the current neuron
            #  update it's weights based on error.
            perceptronOptimizer.backward(perceptronLayerOne, train_groundTruth, x_train[train_batch])

            # after each epoch calculate the accuracy of the training data with this model
            perceptronLayerOne.displayAccuracy("Training Model")
            train_batch += 1
        trainingDataAccuracyResults.append(perceptronLayerOne.accuracy)
        perceptronLayerOne.clearAccuracy()

        # Begins Testing Epoch

        test_batch = 0
        # run the trained model against the test data
        normalizedTestInputs = shuffleTestData()

        Y_test = getTestingLabels(normalizedTestInputs)
        x_test = getTestingInputs(normalizedTestInputs)
        print("Testing Epoch: ")
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        while test_batch < len(normalizedTestInputs):
            print("Testing batch:" + str(test_batch))
            #  forward pass through the connected input layer and connected layer.

            #  second argument for confusion matrix
            perceptronLayerOne.forward(x_test[test_batch], False)

            #  convert truth value
            test_groundTruth = perceptronOptimizer.convertTruth(Y_test[test_batch])

            #  backward pass will update the current neuron
            #  update it's weights based on error.
            perceptronOptimizer.backward(perceptronLayerOne, test_groundTruth, x_test[test_batch])

            # after each epoch calculate the accuracy of the training data with this model
            perceptronLayerOne.displayAccuracy("Test Model")
            test_batch += 1
        testingDataAccuracyResults.append(perceptronLayerOne.accuracy)
        perceptronLayerOne.clearAccuracy()

        train_epoch += 1
    print("done training")
    print("plotting training and test data results")
    perceptronLayerOne.plotAccuracyOverTestTrainingEpochs(trainingDataAccuracyResults, testingDataAccuracyResults, 50,"Accuracy Over Epochs Against Test with 100 Hidden Neurons 25% of the Training Data")

    #  Final Test with Full Trained Model (50) epochs
    final_test_batch_one = 0
    finalTestingDataAccuracyResults = []
    # run the trained model against the test data
    normalizedFinalTestInputs = shuffleTestData()

    Y_Finaltest = getTestingLabels(normalizedFinalTestInputs)
    x_Finaltest = getTestingInputs(normalizedFinalTestInputs)

    while final_test_batch_one < len(normalizedFinalTestInputs):
        # print("batch:" +  str(batch))
        #  forward pass through the connected input layer and connected layer.

        #  notice the truth to add to the perceptronLayer's confusion matrix
        # data structure
        perceptronLayerOne.forward(x_Finaltest[final_test_batch_one], True)

        #  convert truth value
        test_groundTruth = perceptronOptimizer.convertTruth(Y_Finaltest[final_test_batch_one])

        # appends the y_test_truth logits into the truths list for confusion matrix
        perceptronLayerOne.y_truths.append(test_groundTruth)

        #  backward pass will update the current neuron
        #  update it's weights based on error.
        perceptronOptimizer.backward(perceptronLayerOne, test_groundTruth, x_Finaltest[final_test_batch_one])

        # after each epoch calculate the accuracy of the training data with this model
        perceptronLayerOne.displayAccuracy("Full Trained Test Model")
        final_test_batch_one += 1
    finalTestingDataAccuracyResults.append(perceptronLayerOne.accuracy)
    perceptronLayerOne.clearAccuracy()

    perceptronLayerOne.plotConfusionMatrix(perceptronLayerOne.calculateConfusionMatrix(),"Trained Model 100 Hidden Neurons Provided 25% Training Examples")

#     perceptron_layer #2 with .5 of the data


#     Testing of the trained model for the confusion matrix results:

# Experiment #3:
# uncomment the lines below for the use of experiment #3

# Train two MLP with differing percentages of the data.

    # train_epoch_two = 0
    # # batch = 0
    # #  add loop to control epoch count
    # perceptronLayerTwo = Perceptron_Layer(784, 10, 100, 0.1, 0.9)
    #
    # perceptronOptimizer = Perceptron_Optimizer()
    #
    # # create lists to hold the training results for graphing.
    # trainingDataAccuracyResults = []
    # testingDataAccuracyResults = []
    #
    # while train_epoch_two < 50:
    #     print("Training and Test Epoch " + str(train_epoch_two) + ":")
    #     print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    #     train_batch = 0
    #
    #     # EXPERIMENT #3:
    #     normalizedTrainingInputs = splitTrainingData(.50)
    #
    #     Y_train = getTrainingLabels(normalizedTrainingInputs)
    #     x_train = getTrainingInputs(normalizedTrainingInputs)
    #     print("Training Epoch: ")
    #     print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    #     while train_batch < len(normalizedTrainingInputs):
    #         print("Training batch:" +  str(train_batch))
    #         #  forward pass through the connected input layer and connected layer.
    #
    #         #  second argument for confusion matrix
    #         perceptronLayerTwo.forward(x_train[train_batch], False)
    #
    #         #  convert truth value
    #         train_groundTruth = perceptronOptimizer.convertTruth(Y_train[train_batch])
    #
    #         #  backward pass will update the current neuron
    #         #  update it's weights based on error.
    #         perceptronOptimizer.backward(perceptronLayerTwo, train_groundTruth, x_train[train_batch])
    #
    #         # after each epoch calculate the accuracy of the training data with this model
    #         perceptronLayerTwo.displayAccuracy("Training Model")
    #         train_batch += 1
    #     trainingDataAccuracyResults.append(perceptronLayerTwo.accuracy)
    #     perceptronLayerTwo.clearAccuracy()
    #
    #     # Begins Testing Epoch
    #
    #     test_batch_two = 0
    #     # run the trained model against the test data
    #     normalizedTestInputs = shuffleTestData()
    #
    #     Y_test = getTestingLabels(normalizedTestInputs)
    #     x_test = getTestingInputs(normalizedTestInputs)
    #     print("Testing Epoch: ")
    #     print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    #     while test_batch_two < len(normalizedTestInputs):
    #         print("Testing batch:" + str(test_batch_two))
    #         #  forward pass through the connected input layer and connected layer.
    #
    #         #  second argument for confusion matrix
    #         perceptronLayerTwo.forward(x_test[test_batch_two], False)
    #
    #         #  convert truth value
    #         test_groundTruth = perceptronOptimizer.convertTruth(Y_test[test_batch_two])
    #
    #         #  backward pass will update the current neuron
    #         #  update it's weights based on error.
    #         perceptronOptimizer.backward(perceptronLayerTwo, test_groundTruth, x_test[test_batch_two])
    #
    #         # after each epoch calculate the accuracy of the training data with this model
    #         perceptronLayerTwo.displayAccuracy("Test Model")
    #         test_batch_two += 1
    #     testingDataAccuracyResults.append(perceptronLayerTwo.accuracy)
    #     perceptronLayerTwo.clearAccuracy()
    #
    #     train_epoch_two += 1
    # print("done training network two")
    # print("plotting training and test data results")
    # perceptronLayerTwo.plotAccuracyOverTestTrainingEpochs(trainingDataAccuracyResults, testingDataAccuracyResults, 50, "Accuracy Over Epochs Against Test with 100 Hidden Neurons 50% of the Training Data")
    #
    # #  Final Test with Full Trained Model (50) epochs
    # final_test_batch_two = 0
    # finalTestingDataAccuracyResults = []
    # # run the trained model against the test data
    # normalizedFinalTestInputs = shuffleTestData()
    #
    # Y_Finaltest = getTestingLabels(normalizedFinalTestInputs)
    # x_Finaltest = getTestingInputs(normalizedFinalTestInputs)
    #
    # while final_test_batch_two < len(normalizedFinalTestInputs):
    #     # print("batch:" +  str(batch))
    #     #  forward pass through the connected input layer and connected layer.
    #
    #     #  notice the truth to add to the perceptronLayer's confusion matrix
    #     # data structure
    #     perceptronLayerTwo.forward(x_Finaltest[final_test_batch_two], True)
    #
    #     #  convert truth value
    #     test_groundTruth = perceptronOptimizer.convertTruth(Y_Finaltest[final_test_batch_two])
    #
    #     # appends the y_test_truth logits into the truths list for confusion matrix
    #     perceptronLayerTwo.y_truths.append(test_groundTruth)
    #
    #     #  backward pass will update the current neuron
    #     #  update it's weights based on error.
    #     perceptronOptimizer.backward(perceptronLayerTwo, test_groundTruth, x_Finaltest[final_test_batch_two])
    #
    #     # after each epoch calculate the accuracy of the training data with this model
    #     perceptronLayerTwo.displayAccuracy("Full Trained Test Model")
    #     final_test_batch_two += 1
    # finalTestingDataAccuracyResults.append(perceptronLayerTwo.accuracy)
    # perceptronLayerTwo.clearAccuracy()
    #
    # perceptronLayerTwo.plotConfusionMatrix(perceptronLayerTwo.calculateConfusionMatrix(),"Trained Model 100 Hidden Neurons Provided 50% Training Examples")






main()