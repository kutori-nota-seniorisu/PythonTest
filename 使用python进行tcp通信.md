# 在python中实现tcp通信

在python中实现tcp通信比较简单，只需要导入socket库即可。

首先是代码

服务端

```python
# -*- coding: utf-8 -*-
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
```

客户端

```py
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
```

以上代码实现效果为：服务端程序接收控制台输入的内容，服务端控制台每输入一行，就向客户端发送，客户端接收到消息后，将消息打印到控制台；客户端程序也能接收控制台输入的内容，客户端控制台每输入一行，就向服务端发送，服务端接收到消息之后，将消息打印到控制台。服务端输入"bye"时，服务端与客户端断开连接，服务端与客户端的程序均退出。
