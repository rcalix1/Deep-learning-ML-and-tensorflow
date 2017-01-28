##!/usr/bin/env python
## Deep Learning code
## Ricardo A. Calix, PNW, 2017
## Getting started with deep learning: Programming and methodologies using Python 
## By Ricardo Calix
## and notes from tensorflow.org and sklearn.org
## simple example of a convolutional neural network for mnist data
###################################################################

import tensorflow as tf
import numpy as np
from numpy import genfromtxt
from sklearn import datasets
from sklearn.cross_validation import train_test_split
import sklearn

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt


###########################################################
## set parameters

np.set_printoptions(threshold=np.inf) #print all values in numpy array

###########################################################
#parameters for the main loop

learning_rate = 0.001
n_epochs = 200000  ##27000  
batch_size = 128
display_step = 10


# Parameters for the network

n_input = 784 # MNIST has 784 features because each image has shape of 28*28
n_classes = 10 # MNIST (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units


###########################################################
## create csv files from mnist

def buildDataFromMnist(data_set):
    #iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data_set.data, 
                data_set.target, test_size=0.30, random_state=42)
    f=open('2.0_training_mnist.csv','w')  
    for i,j in enumerate(X_train):
        k=np.append(np.array(  y_train[i]), j   )
        f.write(",".join([str(s) for s in k]) + '\n')
    f.close()
    f=open('2.0_testing_mnist.csv','w')
    for i,j in enumerate(X_test):
        k=np.append(np.array( y_test[i]), j   )
        f.write(",".join([str(s) for s in k]) + '\n')
    f.close()


##################################################################

# load MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


###################################################################
## print stats 
precision_scores_list = []
accuracy_scores_list = []

def print_stats_metrics(y_test, y_pred):    
    print('Accuracy: %.2f' % accuracy_score(y_test,   y_pred) )
    #Accuracy: 0.84
    accuracy_scores_list.append(accuracy_score(y_test,   y_pred) )
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print "confusion matrix"
    print(confmat)
    print pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
    precision_scores_list.append(precision_score(y_true=y_test, y_pred=y_pred))
    print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
    print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
    print('F1-measure: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))

#####################################################################

def plot_metric_per_epoch():
    x_epochs = []
    y_epochs = [] 
    for i, val in enumerate(precision_scores_list):
        x_epochs.append(i)
        y_epochs.append(val)
    
    plt.scatter(x_epochs, y_epochs,s=50,c='lightgreen', marker='s', label='score')
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.title('Score per epoch')
    plt.legend()
    plt.grid()
    plt.show()

########################################################################

def conv2d(x, W, b, strides=1):
    # Conv2D function, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


##########################################################################

def maxpool2d(x, k=2):
    # MaxPool2D function
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


################################################################

def layer(input, weight_shape, bias_shape):
    W = tf.get_variable(tf.random_normal(weight_shape))
    b = tf.get_variable(tf.random_normal(bias_shape))
    mapping = tf.matmul(input, W)   
    result = tf.add( mapping ,  b )
    return result


################################################################

def conv_layer(input, weight_shape, bias_shape):
    W = tf.get_variable(tf.random_normal(weight_shape))
    b = tf.get_variable(tf.random_normal(bias_shape))
    conv = conv2d(x, W, b)
    # Max Pooling (down-sampling)
    conv_max = maxpool2d(conv, k=2)
    return conv_max

################################################################

def fully_connected_layer(conv_input, fc_weight_shape, fc_bias_shape, dropout):   
    new_shape = [-1, tf.Variable(tf.random_normal(fc_weight_shape)).get_shape().as_list()[0]]
    fc = tf.reshape(conv_input, new_shape)
    mapping = tf.matmul(fc, tf.Variable(tf.random_normal( fc_weight_shape))
    fc = tf.add(mapping, tf.Variable(tf.random_normal(fc_bias_shape)))
    fc = tf.nn.relu(fc)
    # Apply Dropout
    fc = tf.nn.dropout(fc, dropout)
    return fc


###########################################################
## define the architecture here

def inference_conv_net2(x, dropout):
    # Reshape input picture 
    # shape = [-1, size_image_x, size_image_y, 1 channel (e.g. grey scale)]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer 1, filter 5x5 conv, 1 input, 16 outputs
    # max pool will reduce image from 28*28 to 14*14
    with tf.variable_scope("conv1"):
        conv1 = conv_layer(x, [5, 5, 1, 16], [16] )
    
    # Convolution Layer 2, filter 5x5 conv, 16 inputs, 36 outputs
    # max pool will reduce image from 14*14 to 7*7
    with tf.variable_scope("conv2"):
        conv2 = conv_layer(conv1, [5, 5, 16, 36], [36] )
    
    # Fully connected layer, 7*7*36 inputs, 128 outputs
    # Reshape conv2 output to fit fully connected layer input
    with tf.variable_scope("fc1"):
        fc1 = fully_connected_layer(conv2, [7*7*36, 128], [128] , dropout)
    
    # Output, 128 inputs, 10 outputs (class prediction)
    with tf.variable_scope("output"):
        output = layer(fc1 ,[128, n_classes], [n_classes] )
    return output



###########################################################
## define the architecture here

def inference_conv_net(x, dropout):
    # Reshape input picture
    # shape = [-1, size_image_x, size_image_y, 1 channel (e.g. grey scale)]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer 1, 5x5 conv, 1 input, 32 outputs
    with tf.variable_scope("conv1"):
        conv1 = conv_layer(x, [5, 5, 1, 32], [32] )
    
    # Convolution Layer 2, 5x5 conv, 32 inputs, 64 outputs
    with tf.variable_scope("conv2"):
        conv2 = conv_layer(conv1, [5, 5, 32, 64], [64] )
    
    # Fully connected layer, 7*7*64 inputs, 1024 outputs
    # Reshape conv2 output to fit fully connected layer input
    with tf.variable_scope("fc1"):
        fc1 = fully_connected_layer(conv2, [7*7*64, 1024], [1024] , dropout)
    
    # Output, 1024 inputs, 10 outputs (class prediction)
    with tf.variable_scope("output"):
        output = layer(fc1 ,[1024, n_classes], [n_classes] )
    return output


###########################################################

def loss_deep_conv_net(output, y_tf):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(output, y_tf)
    loss = tf.reduce_mean(xentropy) 
    return loss


###########################################################

def training(cost):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(cost)
    return train_op


###########################################################
 

def evaluate(output, y_tf):
    correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y_tf,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy


###########################################################


x_tf = tf.placeholder(tf.float32, [None, n_input])
y_tf = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


###############################################################
         
#output = inference_conv_net2(x_tf, keep_prob) 
output = inference_conv_net(x_tf, keep_prob) 
cost = loss_deep_conv_net(output, y_tf)

train_op = training(cost) 
eval_op = evaluate(output, y_tf)


##################################################################
## for metrics

y_p_metrics = tf.argmax(output, 1)

##################################################################
# Initialize and run

#init = tf.global_variables_initializer() 

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


###########################################################################################
dropout2 = 1.0

step = 1
# Keep training until reach max iterations
while step * batch_size < n_epochs:
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    sess.run(train_op, feed_dict={x_tf: batch_x, y_tf: batch_y, keep_prob: dropout})

    loss, acc = sess.run([cost, eval_op], feed_dict={x_tf: batch_x,
                                                     y_tf: batch_y,
                                                     keep_prob: dropout2})
    

    result = sess.run(eval_op, feed_dict={x_tf: mnist.test.images[:256],
                                      y_tf: mnist.test.labels[:256],
                                      keep_prob: dropout2})

    result2, y_result_metrics = sess.run([eval_op, y_p_metrics], feed_dict={x_tf: mnist.test.images[:256],
                                                                            y_tf: mnist.test.labels[:256],
                                                                            keep_prob: dropout2})



    print "test1 {},{}".format(step,result)
        
    print "test2 {},{}".format(step,result2)
    y_true = np.argmax(mnist.test.labels[:256],1)
    print_stats_metrics(y_true, y_result_metrics)
    if step == 1000:
        plot_metric_per_epoch()
    step = step + 1


##########################################################################################

print "<<<<<<DONE>>>>>>"





