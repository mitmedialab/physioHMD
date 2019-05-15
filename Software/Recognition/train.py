
from __future__ import print_function

# from builtins import print

import numpy as np
import scipy.io
import tensorflow as tf
import scipy.io as scio
from timeit import default_timer as timer
import matplotlib.pyplot as plt

start = timer()
trainData = scipy.io.loadmat('Trainset0504_Gul_12Exp_8960_2579.mat')['TrainData'].ravel()
trainData = np.reshape(trainData, [8960, 2000])
LabelData= scipy.io.loadmat('Trainset0504_Gul_12Exp_8960_2579.mat')['TrainLabel'].ravel()
LabelData = np.reshape(LabelData,[8960, 12])

## test data
TestData = scipy.io.loadmat('Trainset0504_Gul_12Exp_8960_2579.mat')['TestData'].ravel()
TestData = np.reshape(TestData,  [2579, 2000])
TestLabel = scipy.io.loadmat('Trainset0504_Gul_12Exp_8960_2579.mat')['TestLabel'].ravel()
TestLabel = np.reshape(TestLabel, [2579, 12])

# Parameters
learning_rate = 0.001
training_iters = 90000
batch_size = 32
display_step = 2800

# Network Parameters14
n_input = 2000
n_classes = 12
dropout = 0.5




with tf.name_scope('input'):
    # tf Graph input
    x = tf.placeholder(tf.float32, [None, n_input],name='x-input')
    y = tf.placeholder(tf.float32, [None, n_classes],name='y-input')
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
    # Reshape input data
    x = tf.reshape(x, shape=[-1,40, 50, 1])
    print(x)
    x_summary = tf.summary.image("x", x)

    # Convolution Layer
    with tf.name_scope('Conv1'):
        conv1 = conv2d(x, weights['wc1'], biases['bc1'])  ##200*11*2000*32
        tf.summary.histogram("weightsWC1", weights['wc1'])
        tf.summary.histogram("conv1",conv1)

    print("conv1")
    print(conv1)
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)  ##200*6*1000*32
    print("maxpool2d")
    print(conv1)


    # Convolution Layer
    with tf.name_scope('Conv2'):
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])  ## 200*6*1000*64
        tf.summary.histogram("weightsWC2", weights['wc2'])
    print("conv2")
    print(conv2)
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)  ##200*6*500*64
    print("maxpool2d")
    print(conv2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    with tf.name_scope('FC'):
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])   ##
        tf.summary.histogram("weightsWD1", weights['wd1'])

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
    'wc2': tf.Variable(tf.random_normal([3, 3, 16, 32])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    #'wd1': tf.Variable(tf.random_normal([16000, 1024])),
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
def getMiniBatch(batch_size,data,label,i_batch):
    data_size=np.size(trainData,0)
    arr=np.arange(data_size)
    np.random.shuffle(arr)
    arr.astype(int)
    # print(type(arr))
    i_batch=i_batch
    starindex=i_batch*batch_size
    endindex=(i_batch+1)*batch_size
    starindex=int(starindex)
    endindex=int(endindex)

    index=arr[starindex:endindex]
    dataOutput=data[index,:]
    labelOutput=label[index,:]
    return dataOutput,labelOutput

pred = conv_net(x, weights, biases, keep_prob)
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
cost_summary=tf.summary.scalar("cost",cost)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
pred_class_index=tf.argmax(pred, 1)
# # Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
accuracy_summary=tf.summary.scalar("accuracy",accuracy)

# Initializing the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()
# Launch the graph
merged = tf.summary.merge_all()

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


# with tf.device("/GPU:0"):
#     sess= tf.Session(config=tf.ConfigProto(log_device_placement=True))

# write summary files
log_dir='/home/fluid/PycharmProjects/ExpressionsRecg/tensorboard'
print(log_dir)
# merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
test_writer = tf.summary.FileWriter(log_dir + '/test')


# with tf.device("/gpu:0"):
sess.run(init)
step = 1
# Keep training until reach max iterations
# merged = tf.summary.merge_all()
for i in range(training_iters):



    Lengh = data_size = np.size(trainData, 0)
    ibatch = i % (Lengh / batch_size)
    #
    # print('step  ',step)
    # print('i     ', i)
    # print('ibatch', ibatch)

    trainData_batch, LabelData_batch = getMiniBatch(batch_size, trainData, LabelData, ibatch)

    sess.run(optimizer, feed_dict={x: trainData_batch, y: LabelData_batch, keep_prob: dropout})

    if step % display_step == 0:
        print(step)
        ## Calculate batch loss and accuracy
        merged_op,loss, acc = sess.run([merged,cost, accuracy], feed_dict={x: trainData,
                                                                           y: LabelData,
                                                                           keep_prob: 1.})
        print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
              "{:.6f}".format(loss) + ", Training Accuracy= " + \
              "{:.5f}".format(acc))

        #  Do test
        TestMerge,TestPred,Testaccuracy = sess.run([merged,pred,accuracy], feed_dict={x: TestData, y: TestLabel, keep_prob: 1.})
        print("### Testing Accuracy:", Testaccuracy)
        train_writer.add_summary(merged_op, i)
        test_writer.add_summary(TestMerge, i)
    step += 1



print("Optimization Finished!")
print("save model")
save_path = saver.save(sess,"./Test_CNN_model_0509_4096_3_3_CH4_Gul12Exp_v1.ckpt")
print("save model:{0} Finished".format(save_path))

# Calculate accuracy
TrainTimer=timer()
pred_class_index,accuracy = sess.run([pred_class_index,accuracy], feed_dict={x: TestData, y: TestLabel, keep_prob: 1.})
print(pred_class_index)
True_class_index=tf.argmax(TestLabel, 1)
TestCon=tf.confusion_matrix(True_class_index,pred_class_index,num_classes=12)
print("TestCon:",'\n',sess.run(TestCon))
scio.savemat('TestCon.mat',{'TestCon':sess.run(TestCon)})
scio.savemat('TestPred.mat',{'TestPred':TestPred})
scio.savemat('True_class_index.mat',{'True_class_index':sess.run(True_class_index)})
print("Testing pred:", pred)
print("Testing Accuracy:", accuracy)

