# Ubuntu下ROS与Windows下MATLAB的互联

参考文档：

[Windows和Ubuntu使用网线直连搭建局域网](https://www.bbsmax.com/A/xl56WyLodr/)

[Windows上的ROS和Ubuntu系统中的ROS通信(详细图文)](https://blog.csdn.net/weixin_41802388/article/details/115298878)

实现了 Windows 下 MATLAB 与 Ubuntu 下 ROS 的连接（Ubuntu 为主机，Windows 为从机）

配置要求：

- Windows: Win10
- Ubuntu: 18.04
- 一根网线

## 尝试 ping 通 Windows 与 Ubuntu

Ubuntu配置：system settings -> Network→wired -> 右下角Options -> IPv4 Settingd -> Method(manual) -> Add -> Address(192.168.1.2) -> Netmask(255.255.255.0) -> Gateway(192.168.1.1)

Windows下的配置：右键右下角的网络图标（或者右键网络→属性）→更改适配器设置→以太网→右键属性→TCP/IPv4→IP地址(192.168.1.3)→子网掩码(255.255.255.0)→默认网关(192.168.1.1)

***Ubuntu 与 Windows 的防火墙均需要关闭。***

注意事项：

1. 默认网关需要相同（一般为192.168.1.1）

2. 子网掩码

3. IP地址不能完全一样，最后一位应该是从 2——？ 都可以，具体上限我忘了，然后互相ping一下吧。

## ROS 与 MATLAB 连接

在 `.bashrc` 文件的最后添加以下两行

```text
export ROS_MASTER_URI=http://192.168.1.2:11311
export ROS_IP=192.168.1.2
```

在 MATLAB 里设置网络

```m
setenv('ROS_MASTER_URI','http://192.168.1.2:11311')
setenv('ROS_IP','192.168.1.3')
```

接下来，启动 roscore，在 MATLAB 里运行 `rosinit`，即可实现连接。
