import matplotlib.pyplot as plt
from lorenzSystem import System
import neuralNetwork1 as nn

fig = plt.figure()
ax = plt.axes(projection='3d')

system = System(10, 28, 8/3, 100)

values, times = system.values(500)
predictions = nn.getPredictionsFullData(*values, times)

ax.plot3D(*values, linewidth = 0.1, c='b')
ax.plot3D(*predictions, linewidth = 0.1, c='black')

plt.show()

