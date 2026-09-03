import matplotlib.pyplot as plt
from lorenzSystem import System

fig = plt.figure()
ax = plt.axes(projection='3d')

system = System(10, 28, 8/3)

ax.plot3D(system.coordinates(100))

plt.show()

