import neuralNetwork1 as nn
from lorenzSystem import System
import matplotlib.pyplot as plt
from PlotIndividualPredvsActual import plotIndividual
from sklearn.metrics import mean_absolute_error

training = False
epochs = 200

predictRecursively = True
visualize2d = False

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
        predictions = nn.getRecrusivePrediction(testX[:100], testY[:100], testZ[:100], (99 * precision), True)
    else:
        predictions = nn.getPredictionsFullData(testX, testY, testZ)

    if visualize2d:
        plotIndividual(testX, testY, testZ, *predictions, times)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(*predictions, linewidth=0.1, c='red')
    ax.plot3D(testX, testY, testZ, linewidth=0.1, c='black')
    plt.show()

