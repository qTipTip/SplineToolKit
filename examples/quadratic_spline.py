import matplotlib.pyplot as plt
import numpy as np

from SplineTK.lib import evaluate_spline_vectorized, knot_averages, index
from SplineTK.spline import Spline

t = (-2, -1, 0, 1, 2, 3, 4)
c = [-1, -1, 1, 1]
x = np.linspace(0, 2, 100)
p = 2
f = Spline(p, t, c)
y = f(x)
k = knot_averages(t, p)
plt.plot(x, y)
plt.plot(k, c)
plt.show()
