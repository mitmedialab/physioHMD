# encoding=utf-8
from __future__ import print_function
import scipy.io
import scipy.io as scio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from builtins import print
from scipy import signal
from sklearn import preprocessing


## test data
# TestData = scipy.io.loadmat('TestSignalMat.mat')['TestSignal'].ravel()
# TestSignalMat = np.reshape(TestData, [2, 32000])
# TestLabel = scipy.io.loadmat('TestData0208_3.mat')['TestLabel'].ravel()
# TestLabel = np.reshape(TestLabel, [10, 2])
Sig = np.genfromtxt(open('/home/fluid/Downloads/test2.csv', "rb"), delimiter=",", skip_footer=4)
Sig = np.transpose(Sig)
Shape=Sig.shape
SP=np.array([2,(np.floor(Shape[1]/1000))*1000])
size=[int(SP[0]),int(SP[1])]
TestSignal= np.zeros(size)
Sig[4, :]=signal.medfilt(Sig[4, :],[3])
Sig[10, :]=signal.medfilt(Sig[10, :],[3])
TestSignal[0, :]= Sig[4, 1:size[1]+1]
TestSignal[1, :]= Sig[10,1:size[1]+1]
scio.savemat('TestSignal.mat',{'TestSignal':TestSignal})

# TestSignal=TestSignalMat


Index=np.arange(0,size[1]-500,500)
TestDataCh511=np.zeros([int(size[1]/500-2),2,1000])
for i in range(0,int(size[1]/500-2),1):
    TestDataCh511[i,0,:]=  TestSignal[0,Index[i]:Index[i]+1000]
    TestDataCh511[i,1,:]=  TestSignal[1,Index[i]:Index[i]+1000]
    tmpA=TestDataCh511[i,0,:]
    tmpB=TestDataCh511[i,1,:]
    TestDataCh511[i, 0, :] = TestDataCh511[i,0,:]-np.mean(tmpA,axis=0)
    TestDataCh511[i, 1, :] = TestDataCh511[i, 1, :] - np.mean(tmpB, axis=0)

    # print(i)
    # print(Index[i],Index[i]+2000-1)
scio.savemat('TestDataCh511.mat',{'TestDataCh511':TestDataCh511})
TestLabelCh511=np.zeros([int(size[1]/500)-2,2])
#signal= np.reshape(signal,[])

# Parameters
learning_rate = 0.001
batch_size = 1
display_step = 5

# Network Parameters
n_input = 2000
n_classes = 2
dropout = 0.6 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, 2,1000])
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
    x = tf.reshape(x, shape=[-1, 2, 1000, 1])

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
    'wc1': tf.Variable(tf.random_normal([1, 256, 1, 8])),
    # 5x5 conv, 32 inputs, 64 outputs0
    'wc2': tf.Variable(tf.random_normal([1, 256, 8, 16])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([4000, 512])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([512, n_classes]))
}
print("weights['wc1']")
print(weights['wc1'])

biases = {
    'bc1': tf.Variable(tf.random_normal([8])),
    'bc2': tf.Variable(tf.random_normal([16])),
    'bd1': tf.Variable(tf.random_normal([512])),
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
    load_path = saver.restore(sess,"./Test_CNN_model_0214_1"
                                   ".ckpt")
    step = 1
    # Keep training until reach max iterations
    # Calculate batch loss and accuracy
    pred_value,accuracy = sess.run([pred_class_index,accuracy], feed_dict={x: TestDataCh511, y: TestLabelCh511,keep_prob: 1.})
    print("pred_value:",pred_value)
    # print("pred accuracy:",accuracy)

    SZ=np.array(pred_value)
for i in range(0,SZ.size,1):
    if pred_value[i]==1:
        print('smile at ',(i+1)/2+0.5,'s')
plt.plot(TestSignal[1, :])
plt.show()
plt.plot(TestSignal[0, :])
plt.show()





