# 典型相关分析在 Python 中的使用

## 协方差与相关系数

在学习典型相关分析之前，我们先了解一下协方差与相关系数的概念。

### 协方差(Covariance)

在概率论与统计学中，协方差通常用于衡量两个变量之间的总体误差，方差则是协方差的一种特殊情况，即当两个变量是相同的情况。期望值分别为$EX$与$EY$的两个实随机变量$X$与$Y$之间的**协方差**$Cov(X,Y)$定义为：
$$
Cov(X,Y) = E|(X-EX)(Y-EY)|
$$
从直观上来看，协方差表示两个变量总体误差的期望。

### 协方差矩阵(covariance matrix)

从**协方差**可以很自然地推出**协方差矩阵**。设$\boldsymbol{X}=(X_1,X_2,\ldots,X_n)^T$为n维随机变量，称矩阵
$$
C = (c_{ij})_{n \times n}=
\left(
    \begin{matrix}
    c_{11} & c_{12} & \cdots & c_{1n} \\
    c_{21} & c_{22} & \cdots & c_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    c_{n1} & c_{n2} & \cdots & c_{nn}
    \end{matrix}
\right)
$$
为n维随机变量$\boldsymbol{X}$的协方差矩阵，其中
$$
c_{ij} = Cov(X_i, X_j), i,j = 1, 2, \ldots, n
$$
为$\boldsymbol{X}$的分量$X_i$和$X_j$的协方差（假设它们都存在）。

例如，二维随机变量$(X_1, X_2)$的协方差矩阵为
$$
C=
\left(
    \begin{matrix}
    c_{11} & c_{12} \\
    c_{21} & c_{22}
    \end{matrix}
\right)
$$
其中
$$
c_{11} = E(X_1 - EX_1)^2 \\
c_{12} = E(X_1 - EX_1)(X_2 - EX_2) \\
c_{21} = E(X_2 - EX_2)(X_1 - EX_1) \\
c_{22} = E(X_2 - EX_2)^2
$$

由于$c_{ij} = c_{ji}(i,j = 1, 2, \ldots, n)$，所以协方差矩阵为对称非负定矩阵。

进一步，我们可以得到**两个列向量随机变量**的协方差矩阵。设$\boldsymbol{X} = (X_1, X_2, \ldots, X_m)^T$为m维随机变量，$\boldsymbol{Y} = (Y_1, Y_2, \ldots, Y_n)^T$维n维随机变量，则这两个变量之间的协方差定义为$m \times n$矩阵，其具体形式为
$$
\Sigma = Cov(\boldsymbol{X}, \boldsymbol{Y}) = (\sigma_{ij})_{m \times n} =
\left(
    \begin{matrix}
    \sigma_{11} & \sigma_{12} & \cdots & \sigma_{1n} \\
    \sigma_{21} & \sigma_{22} & \cdots & \sigma_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    \sigma_{m1} & \sigma_{m2} & \cdots & \sigma_{mn}
    \end{matrix}
\right)
$$
其中
$$
\sigma_{ij} = Cov(X_i, Y_j),i = 1, 2, \ldots, m,j = 1, 2, \ldots, n
$$

两个列向量随机变量的协方差$Cov(\boldsymbol{X}, \boldsymbol{Y})$与$Cov(\boldsymbol{Y}, \boldsymbol{X})$互为**转置矩阵**。

### 相关系数(Correlation Coefficient)

我们通常说的**相关系数**，实际上叫做**皮尔逊相关系数(Pearson Correlation Coefficient)**,这是一种“**线性相关系数**”，它衡量了两个随机变量之间的**线性关系程度**。若两个随机变量有非线性函数关系时，用相关系数来衡量是不合理的。

设有两个随机变量$X$与$Y$，它们的协方差记为$Cov(X, Y)$，方差分别记为$DX,DY$，那么相关系数可以表示为：
$$
\rho_{XY} = \frac{Cov(X, Y)}{\sqrt{DX}\sqrt{DY}}
$$
也可以写作：
$$
\rho_{XY} = \frac{\sigma_{XY}}{\sigma_{X}\sigma_{Y}}
$$

从功能上来说，协方差足以刻画两个随机变量之间的相关关系，但是协方差是**带有单位**的，它的值和$X,Y$的数值有关，不能实现统一的度量，故我们将其**无量纲化**，也即**单位化**，以消除数值量级差异的影响，因而引入了皮尔逊相关系数，这样就消除了单位，使得计算的值介于-1与1之间，相互之间是可以比较的。

### 相关矩阵(Correlation Matrix)

**相关矩阵**又叫做**相关系数矩阵**，它是由协方差矩阵而来的，将协方差矩阵中的每一个元素换成其对应的相关系数即是相关系数矩阵。

设$\boldsymbol{X}=(X_1,X_2,\ldots,X_n)^T$为n维随机变量，称矩阵
$$
R = (\rho_{ij})_{n \times n} =
\left(
    \begin{matrix}
    \rho_{11} & \rho_{12} & \cdots & \rho_{1n} \\
    \rho_{21} & \rho_{22} & \cdots & \rho_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    \rho_{n1} & \rho_{n2} & \cdots & \rho_{nn}
    \end{matrix}
\right)
$$
为n维随机变量$\boldsymbol{X}$的相关矩阵，其中
$$
\rho_{ij} = \frac{Cov(X_i, X_j)}{\sqrt{DX_i}\sqrt{DX_j}},i,j = 1, 2, \ldots, n
$$

## 典型相关分析

下面这篇文章讲的很好

[典型关联分析(CCA)原理总结](https://www.cnblogs.com/pinard/p/6288716.html)

[典型关联分析（Canonical Correlation Analysis）](https://www.cnblogs.com/jerrylead/archive/2011/06/20/2085491.html)

## 典型相关分析的 Python 实现

在 Python 中，我们可以使用 sklearn 这个包进行典型相关分析。

首先，我们下载安装 sklearn 这个包。在 conda 虚拟环境中，输入`conda install scikit-learn'，并在出现proceed ([y]/n)?后，输入y并回车，等待安装完成。

接下来，我们来看一个例程。这个例程来自于CCA的定义。

```py
from sklearn.cross_decomposition import CCA
X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [3.,5.,4.]]
Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
cca = CCA(n_components=1)
cca.fit(X, Y)
X_c, Y_c = cca.transform(X, Y)
```

一共六行语句，我们来分别解释其含义。

第一条语句，导入 sklearn 包中的 CCA 模块。可以转到定义查看相关解释说明与例程。

第二与第三条语句分别生成两个随机向量。其中$\boldsymbol{X}$是一个三维随机向量，样本数量为4；$\boldsymbol{Y}$是一个二维随机向量，样本数量为4。需要注意的是，进行典型相关分析的两个随机向量，维度可以不一样，但是样本数量是一致的。

第四行语句，初始化一个 CCA 的对象，在初始化时若未给定`n_components`的值，则默认为2。设$\boldsymbol{X}$的维度为$m$，$\boldsymbol{Y}$的维度为$n$，则`n_components`的值最大为$min(m, n)$。`n_components`的值决定了 CCA 会寻找到第几相关系数。

第五行语句，将数据集输入 CCA 中。该步运行结束后，在模型中已经生成了对应的线性变换矩阵。

第六行语句，将线性变换矩阵运用到原始数据集中，即将原始矩阵投影到一维。返回的是一个矩阵，矩阵的列数即为`n_components`的值。

如果需要计算第一主成分对应的相关系数，则需要添加以下语句：

```py
import numpy as np
print(np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1])
```

若要计算第二主成分对应的相关系数，则需要在初始化 CCA 对象时将`n_components`的值改为2，输出语句改为`print(np.corrcoef(X_c[:, 1], Y_c[:, 1])[0, 1])`。其他情况以此类推。
