# -*- coding: UTF-8 -*-
import numpy as np
import scipy.io as scio
from scipy import signal

rawdata = scio.loadmat('./eegdata.mat')
data = np.array(rawdata['eegdata'])
data_used = data[20, :, 0, 0]
print(data_used)
data_downSample = signal.decimate(data_used, 8)
print(data_downSample)