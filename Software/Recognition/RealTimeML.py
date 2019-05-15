# encoding=utf-8
from __future__ import print_function
import scipy.io
import scipy.io as scio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# from builtins import print
from scipy import signal
from sklearn import preprocessing
from scipy.signal import lfilter , iirnotch , butter , medfilt , filtfilt, normalize
import serial
import threading
import csv




writeToFile = True
read = True
ser = serial.Serial('/dev/ttyACM0', 2000000, timeout=None, xonxoff=False, rtscts=False, dsrdtr=False)
t = 3
fs = 500
wind = fs*t
signal =[]
strm = 0
m1 = []
m2 = []
m3 = []
m4 = []

learning_rate = 0.001
# batch_size = 1
# display_step = 5

# Network Parameters
n_input = 2000
n_classes = 12
# dropout = 0.5 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, 4,500])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 40, 50, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])  ##200*11*2000*32
    print("conv1")
    print(conv1)
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)  ##200*6*1000*32
    print("maxpool2d")
    print(conv1)


    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])  ## 200*6*1000*64
    print("conv2")
    print(conv2)
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)  ##200*6*500*64
    print("maxpool2d")
    print(conv2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])   ##
    ##fc1 = tf.reshape(conv2, [-1,1600])
    print(weights['wd1'])
    print("fc1")
    print(fc1)
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    print(fc1)
    fc1 = tf.nn.relu(fc1)
    print(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)
    print(fc1)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    print(out)
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 16])),
    # 5x5 conv, 32 inputs, 64 outputs0
    'wc2': tf.Variable(tf.random_normal([3,3, 16, 32])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([4160, 4096])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([4096, n_classes]))
}
print("weights['wc1']")
print(weights['wc1'])

biases = {
    'bc1': tf.Variable(tf.random_normal([16])),
    'bc2': tf.Variable(tf.random_normal([32])),
    'bd1': tf.Variable(tf.random_normal([4096])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
print("biases[ 'bc1']")
print(biases[ 'bc1'])

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

pred_class_index=tf.argmax(pred, 1)
# # Evaluate model
correct_pred = tf.equal(pred_class_index, tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



def process(data):
    global pred,accuracy,init,x,y
    Sig = np.array(data,dtype=float)

    #Sig = np.transpose(Sig)
    # Sig=Sig[:,::1]
    # Sig=Sig[:,1:58000]
    b1, a1 = iirnotch(60*2/500, 30.0)
    b2, a2 = iirnotch(60*4/500, 30.0)
    b, a = butter(3, 0.12, 'highpass')

    Sig[0, :]=lfilter(b1, a1, Sig[0, :])
    Sig[1, :]=lfilter(b1, a1, Sig[1, :])
    Sig[2, :]=lfilter(b1, a1, Sig[2, :])
    Sig[3, :]=lfilter(b1, a1, Sig[3, :])
    Sig[0, :]=lfilter(b2, a2, Sig[0, :])
    Sig[1, :]=lfilter(b2, a2, Sig[1, :])
    Sig[2, :]=lfilter(b2, a2, Sig[2, :])
    Sig[3, :]=lfilter(b2, a2, Sig[3, :])
    # print(Sig.dtype)

    Sig[0, :]=lfilter(b, a, Sig[0, :])
    Sig[1, :]=lfilter(b, a, Sig[1, :])
    Sig[2, :]=lfilter(b, a, Sig[2, :])
    Sig[3, :]=lfilter(b, a, Sig[3, :])
    # plt.plot(Sig[0, :])
    # plt.show()
    print(np.shape(Sig))

    Sig=Sig[:,1200:]
    # Sig[0, :]= preprocessing.minmax_scale(Sig[0, :], feature_range=(0.1,0.9))
    # Sig[1, :]= preprocessing.minmax_scale(Sig[1, :], feature_range=(0.1,0.9))
    # Sig[2, :]= preprocessing.minmax_scale(Sig[2, :], feature_range=(0.1,0.9))
    # Sig[3, :]= preprocessing.minmax_scale(Sig[3, :], feature_range=(0.1,0.9))
    print(np.shape(Sig))

    Lengh=500;
    Win=250;
    Shape=Sig.shape
    SP=np.array([4,(np.floor(Shape[1]/Lengh))*Lengh])


    size=[int(SP[0]),int(SP[1])]
    TestSignal= np.zeros(size)
    # Sig[3, :]=signal.medfilt(Sig[3, :],[3])
    # Sig[4, :]=signal.medfilt(Sig[4, :],[3])
    # Sig[9, :]=signal.medfilt(Sig[9, :],[3])
    # Sig[10, :]=signal.medfilt(Sig[10, :],[3])

    TestSignal[0, :]= Sig[0, 0:size[1]]
    TestSignal[1, :]= Sig[1, 0:size[1]]
    TestSignal[2, :]= Sig[2, 0:size[1]]
    TestSignal[3, :]= Sig[3, 0:size[1]]

    print('TEST',np.shape(TestSignal))

    # TestSignal=TestSignalMat


    Index=np.arange(0,size[1]-Win,Win)
    # print(Index)


    TestDataCh511=np.zeros([int(size[1]/Win)-1,4,Lengh])
    # print('TestDataCh511',np.shape(TestDataCh511))

    for i in range(0,3):
        # print(i*Win)
        # print(i*Win+500)
        TestDataCh511[i,0,:]=  TestSignal[0,i*Win:i*Win+500]
        TestDataCh511[i,1,:]=  TestSignal[1,i*Win:i*Win+500]
        TestDataCh511[i,2,:]=  TestSignal[2,i*Win:i*Win+500]
        TestDataCh511[i,3,:]=  TestSignal[3,i*Win:i*Win+500]


        # tmp1 = TestDataCh511[i, 0, :]
        # tmp2 = TestDataCh511[i, 1, :]
        # tmp3 = TestDataCh511[i, 2, :]
        # tmp4 = TestDataCh511[i, 3, :]
        #
        # TestDataCh511[i, 0, :] = TestDataCh511[i, 0, :] - np.mean(tmp1, axis=0)
        # TestDataCh511[i, 1, :] = TestDataCh511[i, 1, :] - np.mean(tmp2, axis=0)
        # TestDataCh511[i, 2, :] = TestDataCh511[i, 2, :] - np.mean(tmp3, axis=0)
        # TestDataCh511[i, 3, :] = TestDataCh511[i, 3, :] - np.mean(tmp4, axis=0)

        # tmpC=TestDataCh511[i,:,:]
        # tmpC = np.reshape(tmpC, [1, 4, 500])
        # TestDataCh511_reshape[i, :, :]=tmpC

    TestLabelCh511=np.zeros([int(size[1]/Win)-1,12])

    # Initializing the variables
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # Launch the graph
    with tf.Session() as sess:
        # sess.run(init)
        # print('working')
        load_path = saver.restore(sess, "./Test_CNN_model_0504_4096_3_3_CH4_Gul12Exp_v1"
                                        ".ckpt")
        step = 1
        # Keep training until reach max iterations
        # Calculate batch loss and accuracy
        print(TestDataCh511.shape)
        pred_value, accuracy1 = sess.run([pred_class_index, accuracy],
                                        feed_dict={x: TestDataCh511, y: TestLabelCh511, keep_prob: 1.})
        print("pred_value:", pred_value)
        SZ = np.array(pred_value)

        for i in range(0, SZ.size, 1):
            if pred_value[i] == 1:
                print('Smile        at ', (i + 1) / 4 + 0.25, 's')
            elif pred_value[i] == 2:
                print('Sad          at ', (i + 1) / 4 + 0.25, 's')
            elif pred_value[i] == 3:
                print('Afraid       at ', (i + 1) / 4 + 0.25, 's')
            elif pred_value[i] == 4:
                print('Angry        at ', (i + 1) / 4 + 0.25, 's')
            elif pred_value[i] == 5:
                print('Flirt        at ', (i + 1) / 4 + 0.25, 's')
            elif pred_value[i] == 6:
                print('Cry          at ', (i + 1) / 4 + 0.25, 's')
            elif pred_value[i] == 7:
                print('Rage         at ', (i + 1) / 4 + 0.25, 's')
            elif pred_value[i] == 8:
                print('Sarcastic    at ', (i + 1) / 4 + 0.25, 's')
            elif pred_value[i] == 9:
                print('Shock        at ', (i + 1) / 4 + 0.25, 's')
            elif pred_value[i] == 10:
                print('Snarl        at ', (i + 1) / 4 + 0.25, 's')
            elif pred_value[i] == 11:
                print('Wink         at ', (i + 1) / 4 + 0.25, 's')
            elif pred_value[i] == 0:
                print('Natural      at ', (i + 1) / 4 + 0.25, 's')
            else:
                print('Error        at ', (i + 1) / 4 + 0.25, 's')


if(writeToFile):
    ofile  = open('tao_ml_01.csv', "wb")
    writer = csv.writer(ofile, delimiter=' ', quotechar='"', quoting=csv.QUOTE_NONE , escapechar='\n')

ser.write(bytes(b'1'))

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
        if strm == 2*wind:
            strm = wind + wind/2
            m1 = m1[wind/2:]
            m2 = m2[wind/2:]
            m3 = m3[wind/2:]
            m4 = m4[wind/2:]
            signal = [m1,m2,m3,m4]
            print(np.shape(signal))
            process(signal)

            #ser.write(bytes(b'2'))


