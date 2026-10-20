from sklearn.metrics import mean_absolute_error
import numpy as np
from matplotlib import pyplot as plt

def plotError(actualX, actualY, actualZ, predictedX, predictedY, predictedZ, time):
    Xerror = abs(actualX - predictedX)
    Yerror = abs(actualY - predictedY)
    Zerror = abs(actualZ - predictedZ)

    fig, ax = plt.subplots(3, 1, sharex=True)

    # plot each variable
    ax[0].plot(time, Xerror, c='r', linewidth=0.4)
    ax[0].set_ylabel('X-Axis Error (MAE)')
    ax[1].plot(time, Yerror, c='b', linewidth=0.4)
    ax[1].set_ylabel('Y-Axis Error (MAE)')
    ax[2].plot(time, Zerror, c='g', linewidth=0.4)
    ax[2].set_ylabel('Z-Axis Error (MAE)')

    plt.xlabel('Time')
    ax[0].xaxis.grid()
    ax[1].xaxis.grid()
    ax[2].xaxis.grid()

    plt.show()