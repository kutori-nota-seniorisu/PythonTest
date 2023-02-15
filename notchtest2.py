import numpy as np
from scipy import signal
import numpy.fft as fft
import matplotlib.pyplot as plt

f0 = 135
Ts = 0.001
Fs = 1 / Ts
L = 512
delta_f = Fs / L
t = np.array(range(0, L))
x = 1.2 * np.cos(2 * np.pi * 135 * t * Ts) + np.cos(2 * np.pi * 100 * t * Ts)
fft_x = fft.fft(x)
pows2_x = np.abs(fft_x / L)
pows1_x = pows2_x[0 : int(L / 2)]
pows1_x[1 : int(L / 2)] = 2 * pows1_x[1 : int(L / 2)]
freqs_x = np.arange(0, int(L / 2)) * delta_f

C1 = -2 * np.cos(2 * np.pi * f0 * Ts)
C2 = 0.95
a = np.array([1, C1 * C2, C2 * C2])
b = np.array([1, C1, 1])

y = signal.filtfilt(b, a, x)
fft_y = fft.fft(y)
pows2_y = np.abs(fft_y / L)
pows1_y = pows2_y[0 : int(L / 2)]
pows1_y[1 : int(L / 2)] = 2 * pows1_y[1 : int(L / 2)]
freqs_y = np.arange(0, int(L / 2)) * delta_f

fig, ax = plt.subplots(2, 1, figsize=(8, 6))
ax[0].plot(freqs_x, pows1_x, color='blue')
ax[1].plot(freqs_y, pows1_y, color='green')
plt.show()