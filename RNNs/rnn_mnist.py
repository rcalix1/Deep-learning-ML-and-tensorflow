################################################################
# Getting started with deep learning 
# by Ricardo A. Calix
# simple recurrent neural network for mnist
# rnn

###############################################################

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

#############################################################

#from tensorflow.contrib import rnn

from tensorflow.python.ops import rnn, rnn_cell

###########################################################
## set parameters

import warnings
warnings.filterwarnings("ignore") 

np.set_printoptions(threshold=np.inf) #print all values in numpy array

###########################################################
#parameters for the main loop

learning_rate = 0.001
n_epochs = 100000  ##27000  
batch_size = 100


#to predict the class per 28x28 image, we now think of the image
#as a sequence of rows. Therefore, you have 28 rows of 28 pixels
#each

chunk_size = 28 # MNIST data input (img shape: 28*28)
n_chunks = 28 # chunks per image
rnn_size = 128 # size of rnn
n_classes = 10 # MNIST total classes (0-9 digits) ## B

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
    #precision_scores_list.append(precision_score(y_true=y_test, y_pred=y_pred))
    #print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
    #print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
    #print('F1-measure: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))

#####################################################################

def plot_metric_per_epoch():
    x_epochs = []
    y_epochs = [] 
    for i, val in enumerate(accuracy_scores_list):
        x_epochs.append(i)
        y_epochs.append(val)
    
    plt.scatter(x_epochs, y_epochs,s=50,c='lightgreen', marker='s', label='score')
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.title('Score per epoch')
    plt.legend()
    plt.grid()
    plt.show()


###########################################################

def RNN(x):
    W = tf.Variable(   tf.random_normal(   [rnn_size, n_classes]  ))
    b = tf.Variable(   tf.random_normal(   [n_classes]   ))  

    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, chunk_size] )
    x = tf.split(0, n_chunks, x)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    rnn_output =  tf.matmul(outputs[-1], W) + b
    return rnn_output

#########################################################################

def loss_deep_rnn(output, y_tf):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(output, y_tf)
    loss = tf.reduce_mean(xentropy) 
    return loss


###########################################################

def training(cost):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cost)
    return train_op

###########################################################

def evaluate(output, y_tf):
    correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y_tf,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy


###########################################################

x_tf = tf.placeholder("float", [None, n_chunks, chunk_size]  )
y_tf = tf.placeholder("float", [None, n_classes])

###############################################################
         
output = RNN(x_tf)
cost = loss_deep_rnn(output, y_tf)
train_op = training(cost)
eval_op = evaluate(output, y_tf)

##################################################################
## for metrics

y_p_metrics = tf.argmax(output, 1)

##################################################################
# Initialize and run

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

###########################################################################################

step = 1
while step * batch_size < n_epochs:
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    # batch_x has a vector of 784 which needs to be converted
    # to a sequence
    # Reshape data to get 28 chunks of 28 elements each
    batch_x = batch_x.reshape(  (batch_size, n_chunks, chunk_size)  )

    sess.run(train_op, feed_dict={x_tf: batch_x, y_tf: batch_y})

    loss, acc = sess.run([cost, eval_op], feed_dict={x_tf: batch_x,y_tf: batch_y })
    

    test_len = 256
    test_data = mnist.test.images[:test_len].reshape((-1, n_chunks,chunk_size))
    test_label = mnist.test.labels[:test_len]
    result_eval =  sess.run(   eval_op, feed_dict={x_tf: test_data, y_tf: test_label}   )

    print "result {},{}".format(step, result_eval)
    
    step = step + 1


#######################################################################################

print '<<<<<DONE>>>>>>'
