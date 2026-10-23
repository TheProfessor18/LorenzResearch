import neuralNetwork1 as nn
from lorenzSystem import System
import matplotlib.pyplot as plt
from PlotIndividualPredvsActual import plotIndividual
from plotErrorVsTime import plotError
import numpy as np

training = False
epochs = 200

predictRecursively = True

visualizeGroundTruth = True
visualize2d = True
visualizeError = True

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

    testX, testY, testZ, times = testingData

    if training:
        try:
            for t in range(epochs):
                print(f'Epoch {t}')
                nn.train(nn.model, trainingData)
        finally:
            nn.torch.save(nn.model.state_dict(), 'model.pth')

    if predictRecursively:
        predictions = nn.getRecrusivePrediction(testX[:100], testY[:100], testZ[:100], (99 * precision))
    else:
        predictions = nn.getPredictionsFullData(testX, testY, testZ)

    predX, predY, predZ = predictions

    #find early data that couldn't be predicted and remove it.
    pointsToDelete = np.where(np.isnan(predX))
    testX = np.delete(testX, pointsToDelete); testY = np.delete(testY, pointsToDelete); testZ = np.delete(testZ, pointsToDelete); times = np.delete(times, pointsToDelete)
    predX = np.delete(predX, pointsToDelete); predY = np.delete(predY, pointsToDelete); predZ = np.delete(predZ, pointsToDelete)

    if visualize2d:
        plotIndividual(testX, testY, testZ, predX, predY, predZ, times)

    if visualizeError:
        plotError(testX, testY, testZ, predX, predY, predZ, times)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(predX, predY, predZ, linewidth=0.1, c='red')
    if visualizeGroundTruth:
        ax.plot3D(testX, testY, testZ, linewidth=0.1, c='black')
    plt.show()

