
import matplotlib.pyplot as plt
import numpy as np
from SplineTK.spline import Spline, ClosedSpline

t = (0, 1, 2, 3, 4, 5, 6, 7)
c = [
    [0, 1, 1, 0],
    [0, 0, 1, 1]
]
x = np.linspace(3, 7, 100)
p = 3
f = ClosedSpline(p, t, c)
y = f(x)

print(f.t)
fig = plt.figure()
axs = plt.gca()
axs.plot(*y)
axs.plot(*f.control_polygon, ls='dashed')
print(f.control_polygon)
plt.show()