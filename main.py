from sympy.printing.pretty.pretty_symbology import line_width

import neuralNetwork1 as nn
from lorenzSystem import System
import matplotlib.pyplot as plt

training = True
epochs = 1000

precision = 100
sigma = 10
rho = 28
beta = 8/3

trainingSplitPoints = (0, 800)
testingSplitPoints = (801, 900)
validationSplitPoints = (901, 1000)

completeSystem = System(sigma, rho, beta, precision)
totalX, totalY, totalZ, totalTimes = completeSystem.values(validationSplitPoints[1])

trainingData = totalX[:(trainingSplitPoints[1]*precision)], totalY[:(trainingSplitPoints[1]*precision)], totalZ[:(trainingSplitPoints[1]*precision)], totalTimes[:(trainingSplitPoints[1]*precision)]
testingData = totalX[(testingSplitPoints[0]*precision):(testingSplitPoints[1]*precision)], totalY[(testingSplitPoints[0]*precision):(testingSplitPoints[1]*precision)], totalZ[(testingSplitPoints[0]*precision):(testingSplitPoints[1]*precision)], totalTimes[(testingSplitPoints[0]*precision):(testingSplitPoints[1]*precision)]
validationData = totalX[(validationSplitPoints[0]*precision):(validationSplitPoints[1]*precision)], totalY[(validationSplitPoints[0]*precision):(validationSplitPoints[1]*precision)], totalZ[(validationSplitPoints[0]*precision):(validationSplitPoints[1]*precision)], totalTimes[(validationSplitPoints[0]*precision):(validationSplitPoints[1]*precision)]

if __name__ == '__main__':

    if training:
        try:
            for t in range(epochs):
                print(f'Epoch {t}')
                #print(trainingData)
                nn.train(nn.model, trainingData)
        finally:
            nn.torch.save(nn.model.state_dict(), 'model.pth')

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    trainX, trainY, trainZ, traintimes = trainingData
    testX, testY, testZ, times = testingData
    recursivePredictions = nn.getRecrusivePrediction(testX, testY, testZ, (1000 * precision))
    fullDataPredictions = nn.getPredictionsFullData(testX, testY, testZ)
    ax.plot3D(*fullDataPredictions, linewidth=0.1, c='red')
    #ax.plot3D(testX, testY, testZ, linewidth=0.1, c='black')
    plt.show()