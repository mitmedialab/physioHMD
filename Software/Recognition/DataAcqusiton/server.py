import socket
import sys
import time
import serial
import threading
import numpy as np
from scipy import signal
from scipy.signal import lfilter , iirnotch , butter


win = 100
fs = 1000
whp = 30.0/fs
Q = 30

read = True 
startP = False
m1 = []
m2 = []
m3 = []
m4 = []
o1 = []
o2 = []
o3 = []
o4 = []
m1p = np.zeros(win)
m2p = np.zeros(win)
m3p = np.zeros(win)
m4p = np.zeros(win)
o1p = np.zeros(win)
o2p = np.zeros(win)
o3p = np.zeros(win)
o4p = np.zeros(win)

ser1 = serial.Serial('/dev/ttyACM0', 2000000, timeout=None, xonxoff=False, rtscts=False, dsrdtr=False)
b1, a1 = iirnotch(60*2/fs, Q)
b2, a2 = iirnotch(60*4/fs, Q)
b3, a3 = iirnotch(60*6/fs, Q)
b4, a4 = iirnotch(60*8/fs, Q)
b5, a5 = iirnotch(60*10/fs, Q)
b6, a6 = iirnotch(60*12/fs, Q)
bemg , aemg  = butter(3,whp, btype='high', analog=False)

strm = 0

def read_from_port(ser):
     global strm
     global m1 , m2 , m3 , m4 , o1 , o2 , o3 , o4
     while read:
         data = ser.readline()  
         if data:
            strSample = str(data).split(',')
            strm = strm + 1
            m1.append(int(strSample[3]))
            m2.append(int(strSample[4]))
            m3.append(int(strSample[9]))
            m4.append(int(strSample[10]))
            o1.append(int(strSample[5]))
            o2.append(int(strSample[6]))
            o3.append(int(strSample[7]))
            o4.append(int(strSample[8]))
            if strm == 2*win:
                m1 = m1[win:]
                m2 = m2[win:]
                m3 = m3[win:]
                m4 = m4[win:]
                o1 = o1[win:]
                o2 = o2[win:]
                o3 = o3[win:]
                o4 = o4[win:]
                strm = win
        

def process():
     if strm >= 100:
        remove_artifacts()
        return rms(m1p)
     else :
        return 0
     

def rms(x):
    return np.sqrt(x.dot(x)/x.size)
         
def remove_artifacts():
    global m1p,m2p,m3p,m4p,o1p,o2p,o3p,o4p
    m1p = remove_powerline(m1[:win])
    m1p = lfilter(bemg,aemg,m1p)
    m2p = remove_powerline(m2[:win])
    m2p = lfilter(bemg,aemg,m2p)
    m3p = remove_powerline(m3[:win])
    m3p = lfilter(bemg,aemg,m3p)
    m4p = remove_powerline(m4[:win])
    m4p = lfilter(bemg,aemg,m4p)
    o1p = remove_powerline(o1[:win])
    o2p = remove_powerline(o2[:win])
    o3p = remove_powerline(o3[:win])
    o4p = remove_powerline(o4[:win])

def remove_powerline(x):
     y = lfilter(b1, a1, x)
     y = lfilter(b2, a2, y)
     y = lfilter(b3, a3, y)
     y = lfilter(b4, a4, y)
     y = lfilter(b5, a5, y)
     y = lfilter(b6, a6, y)
     return y
     
thr1 = threading.Thread(target=read_from_port, args=(ser1,))
thr1.start()


HOST = ''	# Symbolic name, meaning all available interfaces
PORT = 8888	# Arbitrary non-privileged port

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print 'Socket created'

#Bind socket to local host and port
try:
	s.bind((HOST, PORT))
except socket.error as msg:
	print 'Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1]
	sys.exit()
	
print 'Socket bind complete'

#Start listening on socket
s.listen(10)
print 'Socket now listening'

#now keep talking with the client

conn, addr = s.accept()
print 'Connected with ' + addr[0] + ':' + str(addr[1])
while True:
            
            data = conn.recv(10)
            if data :
                print >>sys.stderr, 'received "%s"' % data
                if data=='1':
                    ser1.write(data)
                    startP = True
                if data=='2':
                    ser1.write(data)
                    startP = False   
                if data=='5':
                    read = False
                    ser1.close()
                    thr1.exit()
                    break
                
            
 
	
s.close()
print("Closed")
