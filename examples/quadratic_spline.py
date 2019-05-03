import matplotlib.pyplot as plt
import numpy as np

from SplineTK.lib import evaluate_spline_vectorized

t = (0, 0, 0, 3, 5, 5, 5)
c = [-1, 2, 1, -3]
x = np.linspace(0, 8, 100)
p = 2

y = evaluate_spline_vectorized(x, t, c, p)

plt.plot(x, y)
plt.show()
