# -*- coding: UTF-8 -*-
import numpy as np
import scipy.io as scio
from sklearn.cross_decomposition import CCA
from scipy import signal
import find
import basic_filterbank
import sincos_ref

# 使用的两个数据集
str1 = "E:/VSCode/eegdata_del3s_v7.mat"
str2 = "E:/VSCode/eegdata_del_later1.5s_v7.mat"
sss = [str1, str2]
# 使用的时间数组
ttt = [2, 2.5, 3, 3.5, 4]
# 使用的阈值数组
rrr = [0, 0.07, 0.1]
# 使用的权重数组
www = [0, -0.02]
# 使用的输出数组
ooo = [0, 1]

for s in sss:
	for r in rrr:
		for o in ooo:
			for w in www:
				for t in ttt:
					# 相关变量及参数设置
					# 采样频率
					sampleRate = 2048
					# 使用的数据长度，单位：s
					t_used = t
					# 数组缓存区大小
					BUFFSIZE = int(sampleRate * t_used)
					# 频率序列
					freqList = [9, 10, 11, 12, 13, 14, 15, 16, 17]
					# 每个数据包大小
					packetSize = 512
					# 分析间隔：0.5s
					anaInter = 0.5
					# 分类结果阈值
					r_threshold = r
					# 选用分析方法，method = 1:CCA，method = 2:FBCCA
					method = 2
					# 选择结果判断方法，isfind = 1:至少三次相同才输出结果，isfind = 0:直接输出结果
					isfind = o

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

					# 五个子带
					num_fbs = 2
					# 权重系数
					a_fbcca = 1.25
					b_fbcca = 0.25
					fb_coefs = np.array([(n + 1)**(-a_fbcca) + b_fbcca for n in range(0, num_fbs)])

					# 生成参考信号
					num_harms = 4
					# 权重给-0.02
					w_sincos = w
					y_ref = sincos_ref.sincosref(freqList, downSampleRate, downBuffSize, num_harms, w_sincos)

					# 标志相机启动与否的变量，为 false 时未启动，为 true 时启动
					camera_on = False

					# 从 mat 文件中读取数据，v7版本前的用scio读取，v7版本后的用h5py读取
					eegdata = np.array(scio.loadmat(s)['eegdata'])
					# print("数组形状：", eegdata.shape)
					# print("数组第二维：", eegdata.shape[1])
					packetNum = int(eegdata.shape[1] / packetSize)

					# 存储结果
					ana_count = int((eegdata.shape[1] - BUFFSIZE) / (2 * packetSize) + 1)
					# print('ana_count:', ana_count)
					# 存储所有的结果
					res = np.zeros((eegdata.shape[3], eegdata.shape[2], ana_count))

					# 计数：统计分析总次数
					analysis_count = 0
					# 计数：统计分析成功的次数
					analysis_accu_count = 0

					for exper_i in range(0, eegdata.shape[3]):
						for target_i in range(0, eegdata.shape[2]):
							# print("本次理论结果为:", freqList[target_i])
							# 把原数组降至二维
							data = eegdata[:, :, target_i, exper_i]
							# 用于存储最近四次的分析结果
							res_arr = np.zeros(4)
							# 用于分析的数据数组
							data_used = np.array([])

							# 每次读取一个 packet 的数据并拼接
							for packet_i in range(0, packetNum):
								packet = data[:, packet_i * packetSize : (packet_i + 1) * packetSize]
								if data_used.size == 0:
									data_used = packet
								else:
									data_used = np.hstack((data_used, packet))
									delta = data_used.shape[1] - BUFFSIZE
									if delta >= 0:
										if (delta / packetSize) % 2 == 0:
											data_used = data_used[:, -BUFFSIZE : ]

								if data_used.shape[1] == BUFFSIZE:
									# print("analysis start")
									ch_used = [20, 24, 25, 26, 27, 28, 29, 30, 31]
									# data used
									data_chused = data_used[ch_used, :]
									# the number of channels usd
									channel_usedNum = len(ch_used)

									## data pre-processing
									# downsampling
									data_downSample = signal.decimate(data_chused, downSamplingNum, ftype='fir')
									# 50Hz notch filter
									data_50hz = signal.filtfilt(notch_b, notch_a, data_downSample)
									# remove baseline
									data_removeBaseline = data_50hz - np.median(data_50hz, -1).reshape(channel_usedNum, 1)
									# bandpass filter
									data_bandpass = signal.filtfilt(B, A, data_removeBaseline)

									result = 0
									if method == 1:
										# CCA
										num_class_cca = len(freqList)
										# 用于存储数据与参考信号的相关系数
										r_cca = np.zeros((num_class_cca))
										for class_i in range(0, num_class_cca):
											refdata_cca = y_ref[class_i].T
											cca = CCA(n_components=1)
											cca.fit(data_bandpass.T, refdata_cca)
											U, V = cca.transform(data_bandpass.T, refdata_cca)
											r_cca[class_i] = np.corrcoef(U[:, 0], V[:, 0])[0, 1]
										# print("本次分析采用CCA")
										# print("CCA:", r_cca)

										# 获取相关系数排序
										index_r = np.argsort(r_cca)
										# 判断本次分类是否有效
										d = r_cca[index_r[-1]] - r_cca[index_r[-2]]
										# print("差值：", d)
										d_normal = d * 1.0 / r_cca[index_r[-1]]
										# print("归一化差值：", d_normal)
										if d_normal > r_threshold:
											# 获取相关系数最大的索引并查找对应频率
											# 将查找到的频率添加到结果数组中
											# 四次中三次相同则可确定
											i_r = np.argmax(r_cca)
											result = freqList[i_r]
											# print("本次分类有效，结果为", result)
										else:
											# print("本次分类无效")
											result = 0
											# 跳过下面所有环节
									elif method == 2:
										# FBCCA
										num_class_fbcca = len(freqList)
										# eigenvalue_r_fbcca:存储子带数据与各个参考信号的相关系数，num_fbs x num_class_fbcca的数组
										eigenvalue_r_fbcca = np.zeros((num_fbs, num_class_fbcca))

										# num_fbs:子带数量
										for fb_i in range(0, num_fbs):
											data_fbcca = basic_filterbank.filterbank(data_bandpass, downSampleRate, fb_i)
											# 子带数据与参考数据进行CCA分析
											for class_i in range(0, num_class_fbcca):
												refdata_fbcca = y_ref[class_i].T
												fbcca = CCA(n_components=1)
												fbcca.fit(data_fbcca.T, refdata_fbcca)
												U, V = fbcca.transform(data_fbcca.T, refdata_fbcca)
												eigenvalue_r_fbcca[fb_i, class_i] = np.corrcoef(U[:, 0], V[:, 0])[0, 1]
										# 计算加权后的相关系数
										r_fbcca = fb_coefs @ (eigenvalue_r_fbcca ** 2)
										# print("本次分析采用FBCCA")
										# print("FBCCA:", r_fbcca)

										# 获取相关系数排序
										index_r = np.argsort(r_fbcca)
										# 判断本次分类是否有效
										d = r_fbcca[index_r[-1]] - r_fbcca[index_r[-2]]
										# print("差值：", d)
										d_normal = d * 1.0 / r_fbcca[index_r[-1]]
										# print("归一化差值：", d_normal)
										if d_normal > r_threshold:
											# 获取相关系数最大的索引并查找对应频率
											# 将查找到的频率添加到结果数组中
											# 四次中三次相同则可确定
											i_r = np.argmax(r_fbcca)
											result = freqList[i_r]
											# print("本次分类有效，结果为", result)
										else:
											# print("本次分类无效")
											result = 0
											# 跳过下面所有环节
										
										# index_class_cca = np.argmax(r_fbcca)
										# result = freqList[index_class_cca]

									# 分析结束，分析总次数加一
									analysis_count = analysis_count + 1

									real_res = 0
									if isfind == 1:
										# 四次中三次相同
										res_arr = np.append(res_arr, result)[1:]
										# print("res array:", res_arr)
										real_res = int(find.find(res_arr))
										# if real_res == 0:
										# 	print("本次未分析出结果！！")
										# else:
										# 	print("分析成功！！")
										# 	print("输出结果为：", real_res)
									elif isfind == 0:
										# 直接输出结果
										real_res = result
									
									if real_res == freqList[target_i]:
										analysis_accu_count = analysis_accu_count + 1

									# buffNum = BUFFSIZE / packetSize - 1
									ana_i = int((packet_i - (BUFFSIZE / packetSize - 1)) / 2)
									res[exper_i, target_i, ana_i] = real_res

									# print("analysis finish\n")

						# print("第", exper_i + 1, "次实验结果为：", res[exper_i])
					# print("最终结果汇总:", res)
					
					with open('log.txt', 'a+') as f:
						f.write("本次使用的数据集:"+s+"\n")
						if method == 1:
							print("本次离线分析使用了CCA")
							f.write("本次离线分析使用了CCA\n")
						elif method == 2:
							print("本次离线分析使用了FBCCA")
							f.write("本次离线分析使用了FBCCA\n")
						if isfind == 1:
							print("本次采用的结果输出方式为:3/4")
							f.write("本次采用的结果输出方式为:3/4\n")
						elif isfind == 0:
							print("本次采用的结果输出方式为:直接输出")
							f.write("本次采用的结果输出方式为:直接输出\n")
						print("本次使用的数据长度为:", t_used, "s")
						print("本次使用的参考信号权重为:", w_sincos)
						print("本次使用的归一化阈值为:", r_threshold)
						print("本次共分析:", analysis_count, "次")
						print("其中分析正确的有:", analysis_accu_count, "次")
						accuracy_rate = (analysis_accu_count * 1.0) / (analysis_count * 1.0)
						print("正确率为:", accuracy_rate * 100, "%")

						f.write("本次使用的数据长度为:"+str(t_used)+"s\n")
						f.write("本次使用的参考信号权重为:"+str(w_sincos)+"\n")
						f.write("本次使用的归一化阈值为:"+str(r_threshold)+"\n")
						f.write("本次共分析:"+str(analysis_count)+"次\n")
						f.write("其中分析正确的有:"+str(analysis_accu_count)+"次\n")
						f.write("正确率为:"+str(accuracy_rate * 100)+"%\n\n")
