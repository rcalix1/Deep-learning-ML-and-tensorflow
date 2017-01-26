#!/usr/bin/env python
## Ricardo A. Calix, PNW, 2016
## Code put together from notes from:
## Getting started with deep learning By Ricardo Calix
## Programming and methodologies using Python 
## Fundamentals of deep learning (O'Reily) By N. Buduma
## and notes from tensorflow.org
##########################################################

import tensorflow as tf
import numpy as np
from numpy import genfromtxt

###########################################################
# Build Example Data is CSV format, but use Iris data
#from sklearn import datasets
#from sklearn.cross_validation import train_test_split
#import sklearn

###########################################################

#parameters
#learning_rate = 0.01
#training_epochs = 1000
#batch_size = 100
#display_step = 1

###########################################################

# Convert to one hot
def convertOneHot(data):
    y=np.array([int(i[0]) for i in data])
    y_onehot=[0]*len(y)
    for i,j in enumerate(y):
        y_onehot[i]=[0]*(y.max() + 1)
        y_onehot[i][j]=1
    return (y,y_onehot)

#############################################################

data = genfromtxt('cs-training.csv',delimiter=',')  # Training data
test_data = genfromtxt('cs-testing.csv',delimiter=',')  # Test data

#print 'train', data
#x = raw_input()
#print 'test', test_data
#x = raw_input()

############################################################

# creates train set just features with no classes
x_train=np.array([ i[1::] for i in data])
# classes vectors (setosa, virginica, versicolor)
y_train,y_train_onehot = convertOneHot(data)

# creates test set just features with no classes
x_test=np.array([ i[1::] for i in test_data])
# classes vectors (setosa, virginica, versicolor)
y_test,y_test_onehot = convertOneHot(test_data)

###########################################################
#features (A) and classes (B)
#  A number of features, 4 in this example
#  B = 3 species of Iris (setosa, virginica and versicolor)
A=data.shape[1]-1 # Number of features, Note first is y
B=len(y_train_onehot[0])


###########################################################
#this works
#x = tf.placeholder(tf.float32, name="x", shape=[None, 4])
#W = tf.Variable(tf.random_uniform([4, 3], -1, 1), name="W")
#b = tf.Variable(tf.zeros([3]), name="biases")
#output = tf.matmul(x, W) + b

#init_op = tf.initialize_all_variables()

#sess = tf.Session()
#sess.run(init_op)
#feed_dict = { x : x_train }
#result = sess.run(output, feed_dict=feed_dict)
#print result

###########################################################

def layer(input, weight_shape, bias_shape):
    weight_stddev = (2.0/weight_shape[0])**0.5
    w_init = tf.random_normal_initializer(stddev=weight_stddev)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable("W", weight_shape, initializer=w_init)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    return tf.nn.relu(tf.matmul(input, W) + b)

##########################################################
#defines network architecture
#deep neural network with 2 hidden layers

def inference_deep(x, A, B):
    with tf.variable_scope("hidden_1"):
        hidden_1 = layer(x, [A, 4],[4])
    with tf.variable_scope("hidden_2"):
        hidden_2 = layer(hidden_1, [4, 4],[4])
    with tf.variable_scope("output"):
        output = layer(hidden_2, [4, B], [B])
    return output

###########################################################

def loss_deep(output, y):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(output, y)
    loss = tf.reduce_mean(xentropy) 
    return loss


###########################################################
#defines the network architecture
#simple logistic regression

def inference(x, A, B):
    W = tf.Variable(tf.zeros([A,B]))
    b = tf.Variable(tf.zeros([B]))
    output = tf.nn.softmax(tf.matmul(x, W) + b)
    return output
   
###########################################################

def loss(output, y):
    dot_product = y * tf.log(output)
    xentropy = -tf.reduce_sum(dot_product, reduction_indices=1)#remove indices?
    loss = tf.reduce_mean(xentropy) #remove this line?
    return loss
    
###########################################################

def training(cost):
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train_op = optimizer.minimize(cost)
    return train_op

###########################################################
## add accuracy checking nodes

def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy

###########################################################

x = tf.placeholder("float", [None, A]) # Features
y = tf.placeholder("float", [None,B]) #correct label for x

#output = inference_deep(x, A, B) ## for deep NN with 2 hidden layers
#cost = loss_deep(output, y)

output = inference(x, A, B) ## for logistic regression
cost = loss(output, y)

train_op = training(cost)
eval_op = evaluate(output, y)

##################################################################
# Initialize and run
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

##################################################################

print("...")
# Run the training
for i in range(300):
    sess.run(train_op, feed_dict={x: x_train, y: y_train_onehot})
    result = sess.run(eval_op, feed_dict={x: x_test, y: y_test_onehot})
    print "Run {},{}".format(i,result)


##################################################################

print "<<<<<<DONE>>>>>>"
