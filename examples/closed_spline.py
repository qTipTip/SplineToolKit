import matplotlib.pyplot as plt
import numpy as np

from SplineTK.spline import ClosedSpline

t = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
c = [
    [0, 1, 1, 0, 2, 3],
    [0, 0, 1, 1, 4, 1]
]
p = 3
f = ClosedSpline(p, t, c)
x = np.linspace(*f.domain_of_definition, num=100)
y = f(x)

fig = plt.figure()
axs = plt.gca()
axs.plot(*y)
axs.plot(*f.control_polygon, ls='dashed')
plt.show()
