
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from SplineTK.spline import Spline

t = (-2, -1, 0, 1, 2, 3, 4)
c = np.random.randint(low=-10, high=10, size=(2, 4))
x = np.linspace(0, 6, 3)
p = 2
f = Spline(p, t, c)
y = f(x)

fig = plt.figure()
axs = plt.gca()
axs.plot(*y)
axs.plot(*f.control_polygon, ls='dashed')
plt.show()