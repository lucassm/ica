"""Load Data."""

import numpy as np
import matplotlib.pyplot as plt

f = open('Gauss3.dat.txt')
x = list()
y = list()
for i, j in enumerate(f.readlines()):
    if i > 61:
        y.append(float(j[0:11]))
        x.append(float(j[12:21]))
x = np.array(x)
y = np.array(y)

plt.plot(x, y, 'ro')
plt.show()
