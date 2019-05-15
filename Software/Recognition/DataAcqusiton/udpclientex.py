import socket

UDP_IP_ADDRESS = "127.0.0.1"
UDP_PORT_NO = 6790
cs = socket.socket(socket.AF_INET , socket.SOCK_DGRAM)

process = str(1)+","+str(10)+","+str(100)

while True:
    ch = str(input())
    cs.sendto(ch, (UDP_IP_ADDRESS,UDP_PORT_NO))




