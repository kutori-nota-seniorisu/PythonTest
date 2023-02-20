# -*- coding: UTF-8 -*-
# mat 数组读取
import scipy.io as scio
import numpy as np

data = scio.loadmat('E:/VSCode/eegdata.mat')
print(type(data))

# 这个方法是读不出来数组内容的
pydata1 = np.array(data)

# 读取的时候一定要带上需要读取变量的名称
pydata2 = np.array(data['eegdata'])

print(pydata2)
print(pydata2[:, :, :, 0].shape)

