from matplotlib import pyplot as plt
import numpy as np

def plotError(actualX, actualY, actualZ, predictedX, predictedY, predictedZ, time, allTics = False):
    precision = int(1 / (time[1] - time[0]))

    Xerror = abs(actualX - predictedX)
    Yerror = abs(actualY - predictedY)
    Zerror = abs(actualZ - predictedZ)

    precision = time[1] - time[0]

    fig, ax = plt.subplots(3, 1, sharex=True)

    # plot each variable
    ax[0].plot(time, Xerror, c='r', linewidth=0.4)
    ax[0].set_ylabel('X-Axis Error (AE)')
    ax[1].plot(time, Yerror, c='b', linewidth=0.4)
    ax[1].set_ylabel('Y-Axis Error (AE)')
    ax[2].plot(time, Zerror, c='g', linewidth=0.4)
    ax[2].set_ylabel('Z-Axis Error (AE)')

    if allTics:
        ax[0].set_xticks(np.arange(time[0], time[-1], 1/precision))
        ax[0].tick_params("x", rotation=45, rotation_mode="xtick")
        ax[1].set_xticks(np.arange(time[0], time[-1], 1/precision))
        ax[1].tick_params("x", rotation=45, rotation_mode="xtick")
        ax[2].set_xticks(np.arange(time[0], time[-1], 1/precision))
        ax[2].tick_params("x", rotation=45, rotation_mode="xtick")


    plt.xlabel('Time')
    ax[0].xaxis.grid()
    ax[1].xaxis.grid()
    ax[2].xaxis.grid()

    plt.show()