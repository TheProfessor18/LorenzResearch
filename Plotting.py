import matplotlib.pyplot as plt
from lorenzSystem import System

fig = plt.figure()
ax = plt.axes(projection='3d')

system = System(10, 28, 8/3)

values, times = system.values(500)

ax.plot3D(*values, linewidth = 0.1)

plt.show()

