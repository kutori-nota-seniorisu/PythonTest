# -*- coding: UTF-8 -*-
# 降采样的实现，调用 scipy 库中 signal 的函数 decimate 进行降采样，使用方法类似于 MATLAB 中的操作
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# 生成信号
t = np.linspace(0, 1, 4001)
x = np.sin(2 * np.pi * 30 * t) + np.sin(2 * np.pi * 60 * t)
# 降采样
y = signal.decimate(x, 4, ftype = 'fir')
# 绘图
fig, ax = plt.subplots(1, 2)
ax[0].stem(x[0:120])
ax[0].set_xlim(0, 120)
ax[0].set_ylim(-2, 2)
ax[1].stem(y[0:30])

plt.show()