# 使用 python 进行信号处理

在 python 中进行信号处理时，我们使用 scipy 这个包，在使用时，我们导入 scipy 包中的 signal 模块。

该模块中内置了许多信号处理的函数，且与 matlab 十分相似，具体函数及其使用方法可以查看 signal 模块的对应源文件。通过查看函数定义，可以了解函数输入输出参数的含义，以及官方给出的例程。

## 陷波滤波

```py
# notch_filter.py
# -*- coding: UTF-8 -*-
# 陷波滤波的实现，使用了两种方法
import numpy as np
from scipy import signal
import numpy.fft as fft
import matplotlib.pyplot as plt

# 生成原始信号并绘制幅频图
# 陷波滤波频率
f0 = 135
# 采样时长
Ts = 0.001
# 采样频率
Fs = 1 / Ts
# 采样点数
L = 512
# 频率分辨率
delta_f = Fs / L
# 生成时域信号
t = np.array(range(0, L))
x = 1.2 * np.cos(2 * np.pi * 135 * t * Ts) + np.cos(2 * np.pi * 100 * t * Ts)
# 傅里叶变换
fft_x = fft.fft(x)
# 双边谱的幅值恢复
pows2_x = np.abs(fft_x / L)
# 双边谱转单边谱
pows1_x = pows2_x[0 : int(L / 2)]
pows1_x[1 : int(L / 2)] = 2 * pows1_x[1 : int(L / 2)]
# 频率坐标
freqs_x = np.arange(0, int(L / 2)) * delta_f

# 陷波滤波一：利用归一化频率w0与品质因数Q生成传递函数系数
Q = 30.0
w0 = f0 / (Fs / 2)
# 设计陷波滤波器
b, a = signal.iirnotch(w0, Q)
# 进行滤波
y1 = signal.filtfilt(b, a, x)
fft_y1 = fft.fft(y1)
pows2_y1 = np.abs(fft_y1 / L)
pows1_y1 = pows2_y1[0 : int(L / 2)]
pows1_y1[1 : int(L / 2)] = 2 * pows1_y1[1 : int(L / 2)]
freqs_y1 = np.arange(0, int(L / 2)) * delta_f

# 陷波滤波二：自行设计滤波器参数
C1 = -2 * np.cos(2 * np.pi * f0 * Ts)
C2 = 0.95
a = np.array([1, C1 * C2, C2 * C2])
b = np.array([1, C1, 1])
# 进行滤波
y2 = signal.filtfilt(b, a, x)
fft_y2 = fft.fft(y2)
pows2_y2 = np.abs(fft_y2 / L)
pows1_y2 = pows2_y2[0 : int(L / 2)]
pows1_y2[1 : int(L / 2)] = 2 * pows1_y2[1 : int(L / 2)]
freqs_y2 = np.arange(0, int(L / 2)) * delta_f

# 绘图，从图上看第二种滤波的方法更好一点，效果更加显著
fig, ax = plt.subplots(3, 1)
ax[0].plot(freqs_x, pows1_x, color='blue')
ax[0].set_title("source frequency")
ax[1].plot(freqs_y1, pows1_y1, color='green')
ax[1].set_title("notch frequency 1")
ax[2].plot(freqs_y2, pows1_y2, color='red')
ax[2].set_title("notch frequency 2")
plt.show()
```

## 降采样

```py
# decimate.py
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
```
