# -*- coding: UTF-8 -*-
import numpy as np
import scipy.io as scio
from sklearn.cross_decomposition import CCA
from scipy import signal
from statistics import median

# 相关变量及参数设置
# 数组缓存区大小
BUFFSIZE = 18432
# 采样频率
sampleRate = 2048
# 频率序列
freqList = [9, 10, 11, 12, 13, 14, 15, 16, 17]

# downsampling, 50Hz notch filter, remove baseline, band-pass filter
# 参数：降采样
downSamplingNum = 8
downSampleRate = sampleRate / downSamplingNum
downBuffSize = int(BUFFSIZE / downSamplingNum)

# 参数：50Hz陷波滤波器
# 将要被移除的频率 (Hz)
f_notch = 50
Ts = 1 / downSampleRate
alpha = -2 * np.cos(2 * np.pi * f_notch * Ts)
beta = 0.95
# 构造滤波器
notch_b = [1, alpha, 1]
notch_a = [1, alpha * beta, beta**2]

# 参数：带通滤波器
# 通带阻带截止频率
Wp = [7 / (downSampleRate / 2), 70 / (downSampleRate / 2)]
Ws = [5 / (downSampleRate / 2), 80 / (downSampleRate / 2)]
# 通带最大衰减 [dB]
Rp = 3
# 阻带最小衰减 [dB]
Rs = 60
N, Wn = signal.cheb1ord(Wp, Ws, Rp, Rs)
# peak-to-peak ripple with R dB in the passband
bp_R = 0.5
B, A = signal.cheby1(N, bp_R, Wn, "bandpass")

# 生成参考信号
num_harms = 4
w_sincos = 0
num_freqs = len(freqList)
y_ref = np.zeros((num_freqs, 2 * num_harms, downBuffSize))
t = np.array([i / downSampleRate for i in range(1, downBuffSize + 1)])
# 对每个参考频率都生成参考波形
for freq_i in range(0, num_freqs):
	tmp = np.zeros((2 * num_harms, downBuffSize))
	# harm:harmonic wave 谐波
	for harm_i in range(0, num_harms):
		stim_freq = freqList[freq_i]
		# Frequencies other than the reference frequency
		d_sin = np.zeros((num_freqs, downBuffSize))
		d_cos = np.zeros((num_freqs, downBuffSize))
		for freq_j in range(0, num_freqs):
			if freq_j != freq_i:
				d_freq = freqList[freq_j]
				d_sin[freq_j, :] = np.sin(2 * np.pi * (harm_i + 1) * d_freq * t)
				d_cos[freq_j, :] = np.cos(2 * np.pi * (harm_i + 1) * d_freq * t)
		temp_d_sin = np.sum(d_sin, 0)
		temp_d_cos = np.sum(d_cos, 0)
		# superposition of the reference frequency with other frequencies
		tmp[2 * harm_i] = (np.sin(2 * np.pi * (harm_i + 1) *stim_freq * t) + w_sincos * temp_d_sin)
		tmp[2 * harm_i + 1] = (np.cos(2 * np.pi * (harm_i + 1) *stim_freq * t) + w_sincos * temp_d_cos)
	y_ref[freq_i] = tmp

# 标志相机启动与否的变量，为 false 时未启动，为 true 时启动
camera_on = False

# 从 mat 文件中读取数据
rawdata = scio.loadmat('E:/VSCode/eegdata.mat')
data = np.array(rawdata['eegdata'])
print("数组形状：", data.shape)
print("数组第一维：", data.shape[1])

# 存储结果
res = np.zeros((data.shape[3], data.shape[2]))

for exper_i in range(0, data.shape[3]):
	for target_i in range(0, data.shape[2]):
		# 把原数组降至二维
		data_used = data[:, :, target_i, exper_i]
		# data_used = data[:, :, 0, 0]
		# print("data_used形状：", data_used.shape)

		# 如果数组长度超过缓存长度，则进行处理
		if not (data_used.shape[1] < BUFFSIZE):
			# select the signal from the useful channels
			# channels used
			# ch_used = [30, 31, 32]
			# ch_used = [21, 25, 26, 27, 28, 29, 30, 31, 32]
			ch_used = [20, 24, 25, 26, 27, 28, 29, 30, 31]

			# data used
			data_chused = data_used[ch_used, :]
			# print("data_chused形状", data_chused.shape)

			# the number of channels usd
			channel_usedNum = len(ch_used)

			# 构造数组，存储处理的数据
			data_downSample = np.zeros((channel_usedNum, downBuffSize))
			data_50hz = np.zeros((channel_usedNum, downBuffSize))
			data_removeBaseline = np.zeros((channel_usedNum, downBuffSize))
			data_bandpass = np.zeros((channel_usedNum, downBuffSize))

			# data pre-processing
			for chan_th in range(0, channel_usedNum):
				# downsampling
				data_downSample[chan_th, :] = signal.decimate(data_chused[chan_th, :], downSamplingNum, ftype='fir')
				# 50Hz notch filter
				data_50hz[chan_th, :] = signal.filtfilt(notch_b, notch_a, data_downSample[chan_th, :])
				# remove baseline
				data_removeBaseline[chan_th, :] = data_50hz[chan_th,:] - median(data_50hz[chan_th, :])
				# bandpass filter
				data_bandpass[chan_th, :] = signal.filtfilt(B, A, data_removeBaseline[chan_th, :])

			# print("降采样后的数组形状：", data_downSample.shape)

			# Intercept a data segment
			num_smpls = 4 * downSampleRate
			ref_data = y_ref[:, :, int(0.5 * downSampleRate): int(0.5 * downSampleRate + num_smpls)]
			test_data = data_bandpass[:, int(0.5 * downSampleRate): int(0.5 * downSampleRate + num_smpls)].T
			# CCA
			num_class_cca = len(freqList)
			r_cca = np.zeros((num_class_cca))
			for class_i in range(0, num_class_cca):
				refdata_cca = ref_data[class_i].T
				cca = CCA(n_components=1)
				cca.fit(test_data, refdata_cca)
				U, V = cca.transform(test_data, refdata_cca)
				r_tmp_cca = np.corrcoef(U[:, 0], V[:, 0])[0, 1]
				r_cca[class_i] = r_tmp_cca
			index_class_cca = np.argmax(r_cca)
			result = freqList[index_class_cca]
			res[exper_i, target_i] = result

			# 根据分析结果发布指令
			if result == 20:
				# do something
				print("the frequency to start camera is", result)
				camera_on = True
			if camera_on == True:
				match result:
					case 9:
						print(9)
					case 10:
						print(10)
					case 11:
						print(11)
					case 12:
						print(12)
					case 13:
						print(13)
					case 14:
						print(14)
					case 15:
						print(15)
					case 16:
						print(16)
					case 17:
						print(17)
					case _:
						print("I am everything~")



print(res)
