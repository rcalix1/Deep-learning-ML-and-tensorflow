#!/usr/bin/env python
## Deep Learning code
## Ricardo A. Calix, PNW, 2016
## Getting started with deep learning: Programming and methodologies using Python 
## By Ricardo Calix
## Fundamentals of deep learning (O'Reily) By N. Buduma
## and notes from tensorflow.org and sklearn.org
## a deep neural net with 2 or 3 layers 
##########################################################

import tensorflow as tf
import numpy as np
from numpy import genfromtxt
from sklearn import datasets
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
import sklearn
from sklearn.preprocessing import StandardScaler

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


###########################################################

# Convert to one hot data
def convertOneHot_data2(data):
    y=np.array([int(i) for i in data])
    #print y[:20]
    rows = len(y)
    columns = y.max() + 1
    a = np.zeros(shape=(rows,columns))
    #print a[:20,:]
    print rows
    print columns
    #rr = raw_input()
    #y_onehot=[0]*len(y)
    for i,j in enumerate(y):
        #y_onehot[i]=np.array([0]*(y.max() + 1) )
        #y_onehot[i][j]=1
        a[i][j]=1
    return (a)

#############################################################
## manual 10 fold crossvalidation get sets

def select_fold_to_use_rc(X, y, k):
    K = 10
    num_samples = len(X[:,0])
    #print num_samples
    #rr = raw_input()
    #for i, x in enumerate(X):
    #    print "i", str(i)
    #    print "k", str(k)
    #    print " i % K != k"   
    #    print i % K
    #    print i % K != k 
    #    rr = raw_input()
    training_indices = [i for i, x in enumerate(X) if i % K != k]
    testing_indices = [i for i, x in enumerate(X) if i % K == k]
    #print training_indices
    #print testing_indices
    #rr = raw_input()
    X_train = X[training_indices]
    y_train = y[training_indices]
    X_test  = X[testing_indices]
    y_test  = y[testing_indices]		
    return X_train, X_test, y_train, y_test


#############################################################
## load csv files

#mnist_train_data_gen = genfromtxt('2.0_training_mnist.csv',delimiter=',') 
#mnist_test_data_gen = genfromtxt('2.0_testing_mnist.csv',delimiter=',') 

#print "done reading data from generated csv files"
############################################################

## load mnist data from csv
## these work

#trainX = np.genfromtxt("data/trainX_mnist.csv",delimiter=",",dtype=None)
#trainY = np.genfromtxt("data/trainY_mnist.csv",delimiter=",",dtype=None)
#testX = np.genfromtxt("data/testX_mnist.csv",delimiter=",",dtype=None)
#testY = np.genfromtxt("data/testY_mnist.csv",delimiter=",",dtype=None)

##################################################################

f_numpy = open("data/rc_12559_Training_19_Dataset.csv",'r')
Matrix_data = np.loadtxt(f_numpy, delimiter=",", skiprows=1)

##################################################################

#A = len(Matrix_data[0,:])
#print "num features,", A

#X=Matrix_data[:, [1,2,3,4,5,6]]
X = Matrix_data[:,:18] #[:,:149] #[:,:21]
y = Matrix_data[:, 19]

###########################################################################################
## manual 10 fold cross validation

#X_train_10fold, X_test_null, y_train_10fold, y_test_null = train_test_split(X, y, test_size=0.01, random_state=42)

#select each fold
#X_train, X_test, y_train, y_test = select_fold_to_use_rc(X_train_10fold, y_train_10fold, 1)
#X_train, X_test, y_train, y_test = select_fold_to_use_rc(X_train_10fold, y_train_10fold, 2)
#X_train, X_test, y_train, y_test = select_fold_to_use_rc(X_train_10fold, y_train_10fold, 3)
#X_train, X_test, y_train, y_test = select_fold_to_use_rc(X_train_10fold, y_train_10fold, 4)
#X_train, X_test, y_train, y_test = select_fold_to_use_rc(X_train_10fold, y_train_10fold, 5)
#X_train, X_test, y_train, y_test = select_fold_to_use_rc(X_train_10fold, y_train_10fold, 6)
#X_train, X_test, y_train, y_test = select_fold_to_use_rc(X_train_10fold, y_train_10fold, 7)
#X_train, X_test, y_train, y_test = select_fold_to_use_rc(X_train_10fold, y_train_10fold, 8)
#X_train, X_test, y_train, y_test = select_fold_to_use_rc(X_train_10fold, y_train_10fold, 9)
#X_train, X_test, y_train, y_test = select_fold_to_use_rc(X_train_10fold, y_train_10fold, 0)


############################################################################################
## % train test split

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)


############################################################################################
## loading a separate and independent test set

f_test = open("data/rc_3156_Test_19_features.csv",'r')
Matrix_test = np.loadtxt(f_test, delimiter=",", skiprows=1)

X_test = Matrix_test[:,:18] #[:,:149] #[:,:21]
y_test = Matrix_test[:, 19]


############################################################################################

print "starting to convert data to one hot encoding"


y_train_onehot = convertOneHot_data2(y_train) 

y_test_onehot = convertOneHot_data2(y_test) 


#print y_train_onehot[:20,:]
#rr = raw_input()

print "data has been loaded from csv"
############################################################
## feature scaling

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

###########################################################
# features (A) and classes (B)
#  A number of features, 784 in this example
#  B = number of classes, 10 numbers for mnist (0,1,2,3,4,5,6,7,8,9)


A = X_train_std.shape[1]   #num features
B = y_train_onehot.shape[1]   #num classes
samples_in_train = X_train_std.shape[0]
samples_in_test = X_test_std.shape[0]
#A = len(X_train[0,:])  # Number of features
#B = len(y_train_onehot[0]) #num classes
print "num features", A
print "num classes", B
print "num samples train", samples_in_train
print "num samples test", samples_in_test
print "press enter"
rr = raw_input()

###################################################
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


#####################################################################

def layer(input, weight_shape, bias_shape):
    weight_stddev = (2.0/weight_shape[0])**0.5
    w_init = tf.random_normal_initializer(stddev=weight_stddev)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable("W", weight_shape, initializer=w_init)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    return tf.nn.relu(tf.matmul(input, W) + b)

##########################################################
#defines network architecture
#deep neural network with 4 hidden layers

def inference_deep_4layers(x_tf, A, B):
    with tf.variable_scope("hidden_1"):
        hidden_1 = layer(x_tf, [A, 21],[21])
    with tf.variable_scope("hidden_2"):
        hidden_2 = layer(hidden_1, [21, 21],[21])
    with tf.variable_scope("hidden_3"):
        hidden_3 = layer(hidden_2, [21, 21],[21])
    with tf.variable_scope("hidden_4"):
        hidden_4 = layer(hidden_3, [21, 21],[21])
    with tf.variable_scope("output"):
        output = layer(hidden_4, [21, B], [B])
    return output

##########################################################
#defines network architecture
#deep neural network with 3 hidden layers

def inference_deep_3layers(x_tf, A, B):
    with tf.variable_scope("hidden_1"):
        hidden_1 = layer(x_tf, [A, 21],[21])
    with tf.variable_scope("hidden_2"):
        hidden_2 = layer(hidden_1, [21, 21],[21])
    with tf.variable_scope("hidden_3"):
        hidden_3 = layer(hidden_2, [21, 21],[21])
    with tf.variable_scope("output"):
        output = layer(hidden_3, [21, B], [B])
    return output


##########################################################
#defines network architecture
#deep neural network with 2 hidden layers

def inference_deep(x_tf, A, B):
    with tf.variable_scope("hidden_1"):
        hidden_1 = layer(x_tf, [A, 21],[21])
    with tf.variable_scope("hidden_2"):
        hidden_2 = layer(hidden_1, [21, 21],[21])
    with tf.variable_scope("output"):
        output = layer(hidden_2, [21, B], [B])
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

x_tf = tf.placeholder("float", [None, A]) # Features
y_tf = tf.placeholder("float", [None,B]) #correct label for x

###############################################################

#output = inference_deep_3layers(x_tf, A, B) ## for deep NN with 3 or 4 hidden layers
output = inference_deep(x_tf, A, B) ## for deep NN with 2 hidden layers
cost = loss_deep(output, y_tf)

#output = inference(x_tf, A, B) ## for logistic regression
#cost = loss(output, y_tf)

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

##################################################################
#batch size is 100

num_samples_train_set = X_train_std.shape[0] #len(X_train[:,0]) 
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
        sess.run(train_op, feed_dict={x_tf: X_train_std[sta:end,:],
                                            y_tf: y_train_onehot[sta:end,:]}) 
    
    print "-------------------------------------------------------------------------------"    
    print "Accuracy score"
    result, y_result_metrics = sess.run([eval_op, y_p_metrics], feed_dict={x_tf: X_test_std,
                                          y_tf: y_test_onehot})
    print "Run {},{}".format(i,result)
    #print final_result
    y_true = np.argmax(y_test_onehot,1)
    print_stats_metrics(y_true, y_result_metrics)
    if i == 1000:
        plot_metric_per_epoch()



##################################################################

print "<<<<<<DONE>>>>>>"
