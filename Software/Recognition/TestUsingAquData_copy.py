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
from sklearn.decomposition import FastICA,PCA
from scipy.signal import lfilter , iirnotch , butter , medfilt , filtfilt, normalize
import mne



Sig = np.genfromtxt(open('/home/fluid/Downloads/Data0428/Gul_Sarcastic_042618_1.csv', "rb"), delimiter=",", skip_footer=100)
Sig = np.transpose(Sig)
# Sig=Sig[:,::1]
# Sig=Sig[:,1:30000]

############ denoising #################
b1, a1 = iirnotch(60*2/500, 30.0)
b2, a2 = iirnotch(60*4/500, 30.0)
b, a = signal.butter(3,0.06, 'highpass')
# bl, al = signal.butter(3,1.6, 'lowpass')


# print('@@@@@@@',b,'  ',a)
# plt.plot(Sig[10, 0:29000])
# plt.show()
#
# print(Sig.shape)
Sig[3, :]=lfilter(b1, a1, Sig[3, :])
Sig[4, :]=lfilter(b1, a1, Sig[4, :])
Sig[9, :]=lfilter(b1, a1, Sig[9, :])
Sig[10, :]=lfilter(b1, a1, Sig[10, :])
Sig[3, :]=lfilter(b2, a2, Sig[3, :])
Sig[4, :]=lfilter(b2, a2, Sig[4, :])
Sig[9, :]=lfilter(b2, a2, Sig[9, :])
Sig[10, :]=lfilter(b2, a2, Sig[10, :])
# # plt.plot(Sig[10, :102000])
# plt.show()
#
Sig[3, :]=lfilter(b, a, Sig[3, :])
Sig[4, :]=lfilter(b, a, Sig[4, :])
Sig[9, :]=lfilter(b, a, Sig[9, :])
Sig[10, :]=lfilter(b, a, Sig[10, :])

# Sig[3, :]=lfilter(bl, al, Sig[3, :])
# Sig[4, :]=lfilter(bl, al, Sig[4, :])
# Sig[9, :]=lfilter(bl, al, Sig[9, :])
# Sig[10, :]=lfilter(bl, al, Sig[10, :])
# Sig[3, :]=lfilter(bl, al, Sig[3, :])
# Sig[4, :]=lfilter(bl, al, Sig[4, :])
# Sig[9, :]=lfilter(bl, al, Sig[9, :])
# Sig[10, :]=lfilter(bl, al, Sig[10, :])

Sig=Sig[:,1000:]
# Sig=Sig[:,1000:100001]
# print(Sig.dtype)
# Sig[3, :]= preprocessing.minmax_scale(Sig[3, :], feature_range=(0.1,0.9))
# Sig[4, :]= preprocessing.minmax_scale(Sig[4, :], feature_range=(0.1,0.9))
# Sig[9, :]= preprocessing.minmax_scale(Sig[9, :], feature_range=(0.1,0.9))
# Sig[10, :]= preprocessing.minmax_scale(Sig[10, :], feature_range=(0.1,0.9))

####### ICA  #####
# # compute ICA
# np.random.seed(0)  # set seed for reproducible results
# n_samples = 2000
# time = np.linspace(0, 8, n_samples)
#
# s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
# s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
# s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: sawtooth signal
#
# S = np.c_[s1, s2, s3]
#
#
# X=Sig[4,1:60001]
# X=np.reshape(X,[-6,10000])
#
#
# # X=np.reshape(X,[-1,3])
# # plt.plot(X)
# # plt.show()
# # X=np.reshape(X,[1,-1])
# XX=np.zeros([3,60000])
#
#
# XX[0,:]=Sig[3,1:60001]
# XX[1,:]=Sig[4,1:60001]
# XX[2,:]=Sig[9,1:60001]
# XXX=XX.T
# # XX[3,:]=Sig[10,1:60001]
#
#
#
# ica = FastICA(n_components=3)
# S_ = ica.fit_transform(XXX)  # Get the estimated sources
# A_ = ica.mixing_  # Get estimated mixing matrix
# # assert np.allclose(XXX, np.dot(S_, A_.T) + ica.mean_)
#
#
# plt.subplot(6,1,1)
# plt.plot(S_[:,0].T)
# plt.subplot(6,1,2)
# plt.plot(S_[:,1])
# plt.subplot(6,1,3)
# plt.plot(S_[:,2])
#
# plt.subplot(6,1,4)
# plt.plot(XX[0,:])
# plt.subplot(6,1,5)
# plt.plot(XX[1,:])
# plt.subplot(6,1,6)
# plt.plot(XX[2,:])
#
# plt.show()
#
# # compute PCA
#
#
#
# # SS=mne.preprocessing.ICA.find_bads_ecg(XX)
#
# pca = PCA(n_components=3)
# H = pca.fit_transform(XX)  # estimate PCA sources
#
# plt.figure(figsize=(9, 6))
#
# models = [X, S, S_, H]
# names = ['Observations (mixed signal)',
#          'True Sources',
#          'ICA estimated sources',
#          'PCA estimated sources']
# colors = ['red', 'steelblue', 'orange']
#
# for ii, (model, name) in enumerate(zip(models, names), 1):
#     plt.subplot(4, 1, ii)
#     plt.title(name)
#     for sig, color in zip(model.T, colors):
#         plt.plot(sig, color=color)
#
# # plt.tight_layout()
# plt.show()

####### segment the data #############33
Lengh=500;
Win=250;
Shape=Sig.shape
SP=np.array([4,(np.floor(Shape[1]/Lengh))*Lengh])

size=[int(SP[0]),int(SP[1])]
TestSignal= np.zeros(size)
# Sig[3, :]=signal.medfilt(Sig[3, :],[7])
# Sig[4, :]=signal.medfilt(Sig[4, :],[7])
# Sig[9, :]=signal.medfilt(Sig[9, :],[11])
# Sig[10, :]=signal.medfilt(Sig[10, :],[11])

TestSignal[0, :]= Sig[3, 0:size[1]]
TestSignal[1, :]= Sig[4, 0:size[1]]
TestSignal[2, :]= Sig[9, 0:size[1]]
TestSignal[3, :]= Sig[10,0:size[1]]
# scio.savemat('/home/fluid/Downloads/Data0424/Tao_Anger_04024_4.mat',{'Anger4':TestSignal})

# TestSignal=TestSignalMat


Index=np.arange(0,size[1]-Win,Win)
# print(Index)


print('size[1]',size[1])
TestDataCh511=np.zeros([int(size[1]/Win-2),4,Lengh])
TestDataCh511_reshape=np.zeros([int(size[1]/Win-2),4,500])

for i in range(0,int(size[1]/Win-2),1):
    TestDataCh511[i,0,:]=  TestSignal[0,Index[i]:Index[i]+Lengh]
    TestDataCh511[i,1,:]=  TestSignal[1,Index[i]:Index[i]+Lengh]
    TestDataCh511[i,2,:]=  TestSignal[2,Index[i]:Index[i]+Lengh]
    TestDataCh511[i,3,:]=  TestSignal[3,Index[i]:Index[i]+Lengh]
    # tmp1=TestDataCh511[i,0,:]
    # tmp2=TestDataCh511[i,1,:]
    # tmp3=TestDataCh511[i,2,:]
    # tmp4=TestDataCh511[i,3,:]
    #
    # TestDataCh511[i, 0, :] = TestDataCh511[i,0,:]-np.mean(tmp1,axis=0)
    # TestDataCh511[i, 1, :] = TestDataCh511[i, 1, :] - np.mean(tmp2, axis=0)
    # TestDataCh511[i, 2, :] = TestDataCh511[i,2,:]-np.mean(tmp3,axis=0)
    # TestDataCh511[i, 3, :] = TestDataCh511[i, 3, :] - np.mean(tmp4, axis=0)

# scio.savemat('TestDataCh511.mat',{'TestDataCh511':TestDataCh511})
TestLabelCh511=np.zeros([int(size[1]/Win)-2,12])

# Parameters
learning_rate = 0.001
batch_size = 1
display_step = 5

# Network Parameters
n_input = 2000
n_classes = 12


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
    'wc1': tf.Variable(tf.random_normal([3,3, 1, 16])),
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

# Initializing the variables
init = tf.global_variables_initializer()

saver = tf.train.Saver()
# Launch the graph
with tf.Session() as sess:
    #sess.run(init)
    load_path = saver.restore(sess,"./Test_CNN_model_0501_4096_3_3_CH4_Gul12Exp_v6"
                                   ".ckpt")
    step = 1
    # Keep training until reach max iterations
    # Calculate batch loss and accuracy
    pred_value,accuracy = sess.run([pred_class_index,accuracy], feed_dict={x: TestDataCh511, y: TestLabelCh511,keep_prob: 1.})
    print("pred_value:",pred_value)
    # print("pred accuracy:",accuracy)

    SZ=np.array(pred_value)
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
        print('Irritaed     at ', (i + 1) / 4 + 0.25, 's')
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
plt.plot(Sig[3, :])
plt.show()
plt.plot(Sig[4, :])
plt.show()
plt.plot(Sig[9, :])
plt.show()
plt.plot(Sig[10, :])
plt.show()
# # # # #
# #

