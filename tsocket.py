import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(5)
try:
    sock.connect(("192.168.2.2", 5432))
    print("Socket连接成功")
except Exception as e:
    print("Socket连接失败:", e)
finally:
    sock.close()
