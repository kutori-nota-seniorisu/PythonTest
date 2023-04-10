# python 学习

## 使用 numpy 生成二维数组

```py
import numpy as np
# 生成 2x3 的数组，数组内每个元素都为零
test = np.zeros((2, 3))
```

## python 中的运算符

```py
a = 21
b = 10
c = 0
 
c = a + b
print ("1 - c 的值为：", c)
 
c = a - b
print ("2 - c 的值为：", c)
 
c = a * b
print ("3 - c 的值为：", c)
 
c = a / b
print ("4 - c 的值为：", c)
 
c = a % b
print ("5 - c 的值为：", c)
 
# 修改变量 a 、b 、c
a = 2
b = 3
c = a**b 
print ("6 - c 的值为：", c)
 
a = 10
b = 5
c = a//b 
print ("7 - c 的值为：", c)
```

以上实例输出结果为

```text
1 - c 的值为： 31
2 - c 的值为： 11
3 - c 的值为： 210
4 - c 的值为： 2.1
5 - c 的值为： 1
6 - c 的值为： 8
7 - c 的值为： 2
```

## vscode 与 anaconda 连接

参考文档：

[VScode连接Anaconda]('https://blog.csdn.net/m0_46388544/article/details/123300898')

按照文档中的顺序即可完成 vscode 与 anaconda 的连接。

### 问题1：打开 power shell 时报错“表达式或语句中包含意外的标记”

[Anaconda powershell prompt 表达式或语句中包含意外的标记](https://blog.csdn.net/weixin_43913261/article/details/121410998)

## vscode 与 miniconda

俗话说，1B内存一寸金，对于我这种追求极简主义的人来说，充分利用好空间，减少不必要的内存开销也是十分重要的（主要是没钱加固态www）。作为 conda 的发行版本，miniconda 具有比 anaconda 更小的内存占用，对于我这种又原又崩又写代码的秃子来说，真的是十分实用了。接下来就是 miniconda 的安装及使用。

参考文档：

[Miniconda安装教程](https://www.bilibili.com/read/cv10603513/)

[miniconda安装](https://www.jianshu.com/p/0511decff9f8)

[Miniconda安装及使用--小白上路](https://zhuanlan.zhihu.com/p/133494097)

[conda常用命令](https://blog.csdn.net/huangjiaaaaa/article/details/115704241)

[Miniconda安装 虚拟环境创建 与包管理](https://blog.csdn.net/yimenglin/article/details/107906643)

### miniconda 下载与安装

1. 百度搜索 conda 清华
2. 进入[Anaconda | 镜像站使用帮助 | 清华大学开源软件镜像站 | Tsinghua Open Source Mirror](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)
3. 找到"Miniconda 安装包可以到 [https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/](https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/) 下载。"
4. 下载所需版本并安装

### 添加环境变量

安装完成后，需要在环境变量里修改“系统变量”中 path 的值，分别添加以下三个路径：
D:\miniconda3
D:\miniconda3\Scripts
D:\miniconda3\Library\bin
寻找自己安装 miniconda 的位置并将对应路径修改即可。

### 安装 Python 版本以及工具包

（1） 添加 conda 的镜像服务器
因为 conda 下载文件要用到国外的服务器，速度比较慢，我们可以用清华的镜像服务器来搞定。

在cmd终端里输入以下两行命令：

```shell
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes
```

（2） 创建虚拟环境

在cmd终端输入`conda create -n ros_and_py python=3.6`

等待安装，需要下载一些其他工具，会显示proceed ([y]/n)?，输入y并回车。如果出现了“to activate this environment, use...” 的信息，说明安装成功。这样就创建了一个环境名称为 ros_and_py，Python 版本为 3.6 的虚拟环境了

（3） 安装 Python 包

在激活了虚拟环境，但是没有进入 Python 环境的情况下，可以使用`conda install package`来安装需要的包，如`conda install scipy`

### 问题一：在 vscode 中运行时会出现“无法加载文件，因为在此系统上禁止运行脚本”

参考文档：
[无法加载文件 ，因为在此系统上禁止运行脚本](https://blog.csdn.net/qq_34516746/article/details/123615008)

解决方法：搜索 Windows powershell 并以管理员身份打开，在命令行中输入`set-executionpolicy remotesigned`，然后输入y并回车，最后重启 vscode 即可。

## python for 循环

python for 循环可以遍历任何可以迭代的对象，如一个列表或者一个字符串。
for 循环的一般格式如下：

```py
for <variable> in <sequence>:
    <statements>
else:
    <statements>
```

打印列表中的每一个元素：

```py
sites = ["Baidu", "Google","Runoob","Taobao"]
for site in sites:
    print(site)
```

输出结果为：

```text
Baidu
Google
Runoob
Taobao
```

打印字符串中的每一个字符：

```py
word = 'runoob'
for letter in word:
    print(letter)
```

输出结果为：

```text
r
u
n
o
o
b
```

对于整数范围值，可以配合 range() 函数使用：

```py
# 从 1 到 5 的所有数字
for number in range(1, 6):
    print(number)
```

输出结果为：

```t
1
2
3
4
5
```

## 用 for 循环生成 list

[Python：list的列表生成式 for循环的升级版](https://blog.csdn.net/qq_28766729/article/details/82747292)

## 对 axis 的理解

[Numpy:对Axis的理解](https://zhuanlan.zhihu.com/p/31275071)

## 寻找最大最小值及其索引

[python寻找最大最小值并返回位置](https://blog.csdn.net/yitanjiong4414/article/details/88965668)

## python 读取 .mat 文件数据

```py
# readMat.py
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
```
