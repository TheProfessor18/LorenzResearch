import matplotlib.pyplot as plt
from lorenzSystem import System
import neuralNetwork1 as nn
import numpy as np

fig = plt.figure()
ax = plt.axes(projection='3d')

system = System(10, 28, 8/3, 100)
#wholeNumberSystem = System(10, 28, 8/3, 10)

values, times = system.values(500)
#wholeValues, wholeTimes = wholeNumberSystem.values(500)
predictions = nn.getPredictionsFullData(*values)

ax.plot3D(*values, linewidth = 0.1, c='b')
#ax.scatter(*wholeValues, s = 0.1, c='r')

ax.plot3D(*predictions, linewidth = 0.1, c='black')

plt.show()

