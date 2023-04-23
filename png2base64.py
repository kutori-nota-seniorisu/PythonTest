import base64
f = open('C:\\Users\\14432\\Downloads\\Compressed\\BCI_ROS-master\\qt1.png', 'rb')
ls_f = base64.b64encode(f.read())
f.close()
print(ls_f)
# imgdata = base64.b64decode(ls_f)
# file = open('qq.jpg', 'wb')
# file.write(imgdata)
# file.close()