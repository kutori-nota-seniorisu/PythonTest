# -*- coding: utf-8 -*-
# 导入socket库
import socket
# 定义一个ip协议版本AF_INET(IPv4)，定义一个传输TCP协议(SOCK_STREAM)
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 定义ip地址与端口号
ip_port = ('127.0.0.1', 8000)
# 进行连接
client.connect(ip_port)
while True:
    message = input('You can say:')
    client.send(message.encode('utf-8'))
    a = client.recv(1024)
    print(a.decode('utf-8'))
    if a.decode('utf-8') == 'bye':
        break