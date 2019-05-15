import socket
import sys
import time
import serial
import numpy as np
from scipy import signal
from scipy.signal import lfilter , iirnotch , butter , filtfilt
import csv

def process():
     global strm,m1p,m2p,m3p,m4p,o1p,o2p,o3p,o4p,win,cnt
     if strm >= 100:
        normalize()
        remove_artifacts()
        samp = str(win)+","+str(int(rms(m1p)))+","+str(int(rms(m2p)))+","+str(int(rms(m3p)))+","+str(int(rms(m4p)))
        return samp
     else :
        return "Null"
        return
     

def rms(x):
    return np.sqrt(x.dot(x)/x.size)

def normalize():
    global m1 , m2 , m3 , m4
    m1[:win] = m1[:win] - np.mean(m1[:win])
    m2[:win] = m2[:win] - np.mean(m2[:win])
    m3[:win] = m3[:win] - np.mean(m3[:win])
    m4[:win] = m4[:win] - np.mean(m4[:win])
    
def remove_artifacts():
    global m1p,m2p,m3p,m4p,o1p,o2p,o3p,o4p,m1,m2,m3,m4,o1,o2,o3,o4,win
    m1p = remove_powerline(m1[:win])
    m1p = lfilter(bemg,aemg,m1p)
    m2p = remove_powerline(m2[:win])
    m2p = lfilter(bemg,aemg,m2p)
    m3p = remove_powerline(m3[:win])
    m3p = lfilter(bemg,aemg,m3p)
    m4p = remove_powerline(m4[:win])
    m4p = lfilter(bemg,aemg,m4p)
    

def remove_powerline(x):
     y = lfilter(b1, a1, x)
     y = lfilter(b2, a2, y)
     return y


UDP_IP_ADDRESS = "127.0.0.1"
UDP_PORT_NO = 6789
cs = socket.socket(socket.AF_INET , socket.SOCK_DGRAM)

ser = serial.Serial('COM12', 2000000, timeout=None, xonxoff=False, rtscts=False, dsrdtr=False)

win = 100
fs = 1000
whp = 40.0/fs
Q = 30

read = True 
strm = 0 
cnt = 0
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

b1, a1 = iirnotch(60*2/fs, Q)
b2, a2 = iirnotch(60*4/fs, Q)
bemg , aemg  = butter(3,whp, btype='high', analog=True)
ofile  = open('test.csv', "wb")
writer = csv.writer(ofile, delimiter=' ', quotechar='"', quoting=csv.QUOTE_NONE , escapechar='\n')

ser.write("1")
while read:
    data = ser.readline()  
    if data:
        writer.writerow([data])
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
          cs.sendto(process(), (UDP_IP_ADDRESS,UDP_PORT_NO))


