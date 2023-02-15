import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

t = np.linspace(0, 1, 4001)
x = np.sin(2 * np.pi * 30 * t) + np.sin(2 * np.pi * 60 * t)
y = signal.decimate(x, 4, ftype = 'fir')
fig, ax = plt.subplots(1, 2, figsize = (8, 6))
ax[0].stem(x[0:120])
ax[0].set_xlim(0, 120)
ax[0].set_ylim(-2, 2)
ax[1].stem(y[0:30])

plt.show()