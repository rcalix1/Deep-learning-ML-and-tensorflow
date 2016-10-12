#!/usr/bin/env python
## Deep Learning code
## Ricardo A. Calix, PNW, 2016
## Code put together from notes from the book
## Fundamentals of deep learning (O'Reily)
## Book author: Nikhil Buduma
## deep with 2 layers and mnist obtains 97% accuracy
##########################################################

import tensorflow as tf
import numpy as np
from numpy import genfromtxt
from sklearn import datasets
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
import sklearn
from sklearn.preprocessing import StandardScaler

###########################################################
## set parameters

np.set_printoptions(threshold=np.inf) #print all values in numpy array

###########################################################
#parameters

learning_rate = 0.01
n_epochs = 27000  #1000
batch_size = 100
#display_step = 1

## a smarter learning rate for gradient optimizer
#learningRate = tf.train.exponential_decay(learning_rate=0.0008,
#                                          global_step=1,
#                                          decay_steps=trainX.shape[0],
#                                          decay_rate=0.95,
#                                          staircase=True)

###########################################################
## create csv files from mnist

def buildDataFromMnist(data_set):
    #iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data_set.data, 
                data_set.target, test_size=0.33, random_state=42)
    f=open('cs-training_mnist.csv','w')  
    for i,j in enumerate(X_train):
        k=np.append(np.array(  y_train[i]), j   )
        f.write(",".join([str(s) for s in k]) + '\n')
    f.close()
    f=open('cs-testing_mnist.csv','w')
    for i,j in enumerate(X_test):
        k=np.append(np.array( y_test[i]), j   )
        f.write(",".join([str(s) for s in k]) + '\n')
    f.close()


###########################################################

# Convert to one hot data
def convertOneHot_data(data):
    y=np.array([int(i[0]) for i in data])
    y_onehot=[0]*len(y)
    for i,j in enumerate(y):
        y_onehot[i]=[0]*(y.max() + 1)
        y_onehot[i][j]=1
    return (y,y_onehot)


############################################################
## The tensorflow way to get mnist
## tensorflow models work with this dataset
## logistic regression achieves 0.92 % accuracy

#from tensorflow.examples.tutorials.mnist import input_data
#mnist_tf = input_data.read_data_sets("MNIST_data/", one_hot=True)

#xs_train = mnist_tf.train.images
#ys_train = mnist_tf.train.labels

#xs_validation = mnist_tf.validation.images
#ys_validation = mnist_tf.validation.labels

#xs_test = mnist_tf.test.images
#ys_test = mnist_tf.test.labels

############################################################
## create csv files

#custom_data_home="/home/purdueml/Desktop/tensorflow_rc/mnist_code/data"
#mnist = fetch_mldata('MNIST original', data_home=custom_data_home)

#iris = datasets.load_iris()
#buildDataFromMnist(mnist)

#print "done building datasets"
#rr = raw_input()

#############################################################
## load csv files

#mnist_train_data_gen = genfromtxt('cs-training_mnist.csv',delimiter=',') 
#mnist_test_data_gen = genfromtxt('cs-testing_mnist.csv',delimiter=',') 


############################################################
## load email data from csv

#trainX = np.genfromtxt("email_data/trainX.csv",delimiter="\t",dtype=None) 
#trainY = np.genfromtxt("email_data/trainY.csv",delimiter="\t",dtype=None)
#testX = np.genfromtxt("email_data/testX.csv",delimiter="\t",dtype=None)
#testY = np.genfromtxt("email_data/testY.csv",delimiter="\t",dtype=None)

############################################################

## load mnist data from csv

trainX = np.genfromtxt("data/trainX_mnist.csv",delimiter=",",dtype=None)
trainY = np.genfromtxt("data/trainY_mnist.csv",delimiter=",",dtype=None)
testX = np.genfromtxt("data/testX_mnist.csv",delimiter=",",dtype=None)
testY = np.genfromtxt("data/testY_mnist.csv",delimiter=",",dtype=None)


############################################################

#X_train = np.array([ i[1::] for i in mnist_train_data_gen])
#y_train, y_train_onehot = convertOneHot_data(mnist_train_data_gen) 

#X_test = np.array([ i[1::] for i in mnist_test_data_gen  ])
#y_test, y_test_onehot = convertOneHot_data(mnist_test_data_gen)

print "data has been loaded from csv"
############################################################
## feature scaling

sc = StandardScaler()
sc.fit(trainX)
trainX_std = sc.transform(trainX)
testX_std = sc.transform(testX)


###########################################################
# features (A) and classes (B)
#  A number of features, 784 in this example
#  B = number of classes, 10 numbers for mnist (0,1,2,3,4,5,6,7,8,9)


A = trainX.shape[1]   #num features
B = trainY.shape[1]   #num classes
samples_in_train = trainX.shape[0]
samples_in_test = testX.shape[0]
#A = len(X_train[0,:])  # Number of features
#B = len(y_train_onehot[0]) #num classes
print "num features", A
print "num classes", B
print "num samples train", samples_in_train
print "num samples test", samples_in_test
rr = raw_input()

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

def inference_deep(x_tf, A, B):
    with tf.variable_scope("hidden_1"):
        hidden_1 = layer(x_tf, [A, 256],[256])
    with tf.variable_scope("hidden_2"):
        hidden_2 = layer(hidden_1, [256, 256],[256])
    with tf.variable_scope("output"):
        output = layer(hidden_2, [256, B], [B])
    return output

###########################################################

def loss_deep(output, y_tf):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(output, y_tf)
    loss = tf.reduce_mean(xentropy) 
    return loss

###########################################################
#defines the network architecture
#simple logistic regression

def inference(x_tf, A, B):
    init = tf.constant_initializer(value=0)
    #W = tf.Variable(tf.zeros([A,B]))
    W = tf.get_variable("W", [A,B],initializer=init)
    #b = tf.Variable(tf.zeros([B]))
    b = tf.get_variable("b", [B], initializer=init)

    output = tf.nn.softmax(tf.matmul(x_tf, W) + b)
    return output

   
###########################################################
## logistic regression

def loss(output, y_tf):
    output2 = tf.clip_by_value(output,1e-10,1.0)
    dot_product = y_tf * tf.log(output2)
    xentropy = -tf.reduce_sum(dot_product, reduction_indices=[1])

    loss = tf.reduce_mean(xentropy) 
    return loss

###########################################################

def training(cost):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(cost)
    return train_op

###########################################################
## add accuracy checking nodes

def evaluate(output, y_tf):
    correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y_tf,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy


###########################################################

def email_inference(x_tf, A, B):
    W = tf.Variable(tf.random_normal([A, B],
                    mean=0,
                    stddev=(np.sqrt(6/numFeatures+
                                      numLabels+1)),
                    name="weights"))
    #W = tf.get_variable("W", [A,B],initializer=init)
    b = tf.Variable(tf.random_normal([1, B], 
                    mean=0,
                    stddev=(np.sqrt(6/numFeatures+
                                      numLabels+1)),
                    name="bias"))

    #b = tf.get_variable("b", [B], initializer=init)

    output = tf.nn.softmax(tf.matmul(x_tf, W) + b) #sigmoid instead of softmax?
    return output


###########################################################

x_tf = tf.placeholder("float", [None, A]) # Features
y_tf = tf.placeholder("float", [None,B]) #correct label for x

###############################################################

output = inference_deep(x_tf, A, B) ## for deep NN with 2 hidden layers
cost = loss_deep(output, y_tf)

#output = inference(x_tf, A, B) ## for logistic regression
#cost = loss(output, y_tf)

train_op = training(cost)
eval_op = evaluate(output, y_tf)


##################################################################
# Initialize and run

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

##################################################################
#batch size is 100

num_samples_train_set = trainX.shape[0] #len(X_train[:,0]) 
#num_samples_train_set = len(X_train[:,0])
num_batches = int(num_samples_train_set/batch_size)

##################################################################


print "starting training and testing"
print("...")
# Run the training
final_result = ""
for i in range(n_epochs):
    print "epoch %s out of %s" % (i, n_epochs)
    for batch_n in range(num_batches):
        sta = batch_n*batch_size
        end = sta + batch_size
        #sess.run(train_op, feed_dict={x_tf: xs_train[sta:end,:], 
        #                              y_tf: ys_train[sta:end,:]})
        #sess.run(train_op, feed_dict={x_tf: X_train[sta:end,:], 
        #                              y_tf: y_train_onehot[sta:end]})
        sess.run(train_op, feed_dict={x_tf: trainX_std[sta:end,:],
                                            y_tf: trainY[sta:end,:]}) 
        
    print "Accuracy score"
    #temp = sess.run(y_test_onehot)
    #result = sess.run(eval_op, feed_dict={x_tf: X_test, 
    #                                      y_tf: y_test_onehot})
    #result = sess.run(eval_op, feed_dict={x_tf: xs_test, y_tf: ys_test })
    result = sess.run(eval_op, feed_dict={x_tf: testX_std,
                                          y_tf: testY})
    print "Run {},{}".format(i,result)
    #print final_result
    #rr = raw_input()

#print final_result


##################################################################

print "<<<<<<DONE>>>>>>"
