# Python_BCI

这是关于用python进行脑电信号分析的相关内容。

## 2023.4.9更新

1. 实现了 FBCCA 算法，具体内容详见`onlineanalysis_2methods.py`文件中的`methon=2`部分
2. 实现了分文件编写，将滤波器组的实现封装成函数，具体内容详见`basic_filterbank.py` 文件
3. 用`numpy`中的`median`函数替代了`statistics`中的`median`函数，并用矩阵运算代替循环操作
