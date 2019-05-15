import socket
import sys
import time
import serial
import threading
import numpy as np
from scipy import signal
from scipy.signal import lfilter , iirnotch , butter , medfilt , filtfilt
import csv

writeToFile = True
read = True 
calibration = True
calibrated = False
e2c = 0 
thrf = 160
thrs = 100
def process():
     global strm,m1p,m2p,m3p,m4p,o1p,o2p,o3p,o4p,win,cnt,calibration,calibrated,m1b,m2b,m3b,m4b,s_emoji,e2c,emo_strings,thrf,thrs
     if strm >= win:
        normalize()
        remove_artifacts() 
        if calibration:
            calibrate(e2c)
            samp = "Calibrating"
        else:
            emoji = 0
            if calibrated:
                if abs(rms(m1p)-m1b)>thrf:
                    emoji = 2
                elif abs(rms(m2p)-m2b)>thrs:
                    emoji = 1
            else:
                m1b = m1b/(s_emoji)
                m2b = m2b/(s_emoji)
                m3b = m3b/(s_emoji)
                m4b = m4b/(s_emoji)
                calibrated = True
            samp = emo_strings[emoji]      
        print(samp)
        return samp
     else :
        return "Null"
        return

def calibrate(i):
    global m1p,m2p,m3p,m4p,m1b,m2b,m3b,m4b,s_emoji
    m1b += rms(m1p)
    m2b += rms(m2p)
    m3b += rms(m3p)
    m4b += rms(m4p)
    s_emoji = s_emoji + 1

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
    m1p = medfilt(m1[:win],5);
    m1p = remove_powerline(m1[:win])
    m1p = lfilter(bemg,aemg,m1p)
    m2p = medfilt(m2[:win], 5);
    m2p = remove_powerline(m2[:win])
    m2p = lfilter(bemg,aemg,m2p)
    m3p = medfilt(m3[:win], 5);
    m3p = remove_powerline(m3[:win])
    m3p = lfilter(bemg,aemg,m3p)
    m4p = medfilt(m4[:win], 5);
    m4p = remove_powerline(m4[:win])
    m4p = lfilter(bemg,aemg,m4p)
    

def remove_powerline(x):
     y = lfilter(b1, a1, x)
     y = lfilter(b2, a2, y)
     return y

def udp_receive():
    global calibration , e2c ,m1b,m2b,m3b,m4b,s_emoji,calibrated
    UDP_RX_IP_ADDRESS = "127.0.0.1"
    UDP_RX_PORT_NO = 6790
    ss = socket.socket(socket.AF_INET , socket.SOCK_DGRAM)
    ss.bind((UDP_RX_IP_ADDRESS,UDP_RX_PORT_NO))
    threadRun = True 
    while threadRun:
        data , addr = ss.recvfrom(1024)
        print "Message from Thread:" , data
        if data == '1':
            ser.write("1")
        elif data == '2':
            e2c = 1
        elif data == '3':
            e2c = 2
        elif data == '4':
            e2c = 3
        elif data == '5':
            e2c = 4
        elif data == '6':
            calibration = False
        elif data == '9':
            calibration = True
            calibrated = False
            ser.write("2")
            m1b = 0
            m2b = 0
            m3b = 0
            m4b = 0
            s_emoji = 0
        elif data == '0':
            ser.write("2")
            threadRun = False
            
            
ser = serial.Serial('/dev/ttyACM1', 2000000, timeout=None, xonxoff=False, rtscts=False, dsrdtr=False)

UDP_IP_ADDRESS = "127.0.0.1"
UDP_PORT_NO = 6789

thr1 = threading.Thread(target=udp_receive, args=())
thr1.start()


cs = socket.socket(socket.AF_INET , socket.SOCK_DGRAM)
win = 100
fs = 1000
whp = 20.0/fs
Q = 30
strm = 0 
cnt = 0

eda = []
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
m1b = 0
m2b = 0
m3b = 0
m4b = 0
s_emoji = 0

emo_strings = ["neutral","happy","angry","dollar","sad"]

b1, a1 = iirnotch(60*2/fs, Q)
b2, a2 = iirnotch(60*4/fs, Q)
bemg , aemg  = butter(3,whp, btype='high', analog=False)
if(writeToFile):
    ofile  = open('Gul_test_0522.csv', "wb")
    writer = csv.writer(ofile, delimiter=' ', quotechar='"', quoting=csv.QUOTE_NONE , escapechar='\n')

ser.write('1')

while read:
    data = ser.readline()
    if data:
        if(writeToFile):
            writer.writerow([data])
        strSample = str(data).split(',')
        strm = strm + 1
        eda.append(int(strSample[2]))
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


