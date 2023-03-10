# -*- coding: utf-8 -*-
# tcp/ip 的 python 实现，服务端
# 引入socket库
import socket
# 定义一个ip协议版本AF_INET(IPv4)，定义一个传输TCP协议(SOCK_STREAM)
sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 定义ip地址与端口号
ip_port = ('127.0.0.1', 8000)
# 绑定一个端口
sk.bind(ip_port)
# 监听一个端口，最大等待数为3
sk.listen(3)
# 接受客户端的数据，并返回两个参数。a为连接信息，b为客户端的ip地址与端口号
a, b = sk.accept()
print('a')
print(a)
print('b')
print(b)
while True:
    # 客户端发送的数据存储在data里，1024为最大接受的字节数
    data = a.recv(1024)
    print(data.decode('utf-8'))
    message = input("you can say:")
    a.send(message.encode('utf-8'))
    if message == ('bye'):
        break