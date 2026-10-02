from matplotlib import pyplot as plt
from lorenzSystem import System
import numpy as np
from sklearn.metrics import mean_absolute_error
import neuralNetwork1 as nn

#define precision and timespan
precision = 100
timespan = 100

#get actual values from the Lorenz System
system = System(10, 28, 8/3, precision)
values, times = system.values(timespan)
x_gt, y_gt, z_gt = values

#get predicted values
x_pred, y_pred, z_pred = nn.getPredictionsFullData(x_gt, y_gt, z_gt, times)

#create 3 rows and 1 column of plots
fig, ax = plt.subplots(3,1, sharex=True)

#plot each variable
ax[0].plot(times, x_gt, c='r', linewidth=0.4)
ax[0].plot(times, x_pred, c='black', linewidth=0.4)
ax[0].set_ylabel('X-Axis')
ax[1].plot(times, y_gt, c='b', linewidth=0.4)
ax[1].plot(times, y_pred, c='black', linewidth=0.4)
ax[1].set_ylabel('Y-Axis')
ax[2].plot(times, z_gt, c='g', linewidth=0.4)
ax[2].plot(times, z_pred, c='black', linewidth=0.4)
ax[2].set_ylabel('Z-Axis')

plt.xlabel('Time')
ax[0].xaxis.grid()
ax[1].xaxis.grid()
ax[2].xaxis.grid()

print(f'Mean absolute error of X: {mean_absolute_error(x_gt, x_pred)}')
print(f'Mean absolute error of Y: {mean_absolute_error(y_gt, y_pred)}')
print(f'Mean absolute error of Z: {mean_absolute_error(z_gt, z_pred)}')

plt.show()