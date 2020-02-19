## log reg, nn, deep neural net algorithm
## supervised learning classifier

#############################################################

import tensorflow as tf
import numpy as np
from numpy import genfromtxt
import sklearn
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt


#############################################################
## remove if not using a mac

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#############################################################

number_epochs = 10000
learning_rate = 0.01
batch_size = 1000

#############################################################

x_train = genfromtxt('mnist_train.csv', delimiter=',', usecols=(i for i in range(1,785)) , skip_header=1) 
y_train = genfromtxt('mnist_train.csv', delimiter=',', usecols=(0), skip_header=1)

x_test = genfromtxt('mnist_test.csv', delimiter=',', usecols=(i for i in range(1,785)) ,skip_header=1 )
y_test = genfromtxt('mnist_test.csv', delimiter=',', usecols=(0), skip_header=1)

############################################################
## normalizing

sc = StandardScaler()
sc.fit(x_train)

x_train_normalized = sc.transform(x_train)
x_test_normalized  = sc.transform(x_test)


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


############################################################
# one-hot encoding

depth = 10
#y_train_onehot = tf.one_hot(y_train, depth)
#y_test_onehot  = tf.one_hot(y_test, depth)

y_train_onehot = convertOneHot_data2(y_train)
y_test_onehot  = convertOneHot_data2(y_test)


#############################################################
# features (A) 

A = len(x_train[0])
print A # number of features

#############################################################
# classes (B)

B = 10 #len(sess.run(y_train_onehot[0]))
print "number of classes ", B

############################################################
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
    precision_scores_list.append(precision_score(y_true=y_test, y_pred=y_pred, average='weighted'))
    print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred, average='weighted'))
    print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred, average='weighted'))
    print('F1-measure: %.3f' % f1_score(y_true=y_test, y_pred=y_pred, average='weighted'))

##############################################################

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


############################################################

def layer(input, weight_shape, bias_shape):
    bias_init = tf.constant_initializer(value=0)
    weight_stddev = (2.0/weight_shape[0])**0.5
    w_init = tf.random_normal_initializer(stddev=weight_stddev)
    W = tf.get_variable("W", weight_shape, initializer=w_init)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    return tf.nn.relu(   tf.matmul(input, W) + b     )

############################################################
## a neural network

def inference_nn(x, A, B):
    with tf.variable_scope("hidden_1"):
        hidden_1 = layer(x, [A, 128], [128] )
    with tf.variable_scope("output"):
        output = layer(hidden_1, [128,B], [B])
    return output

############################################################
## deep neural network 3 hidden layers

def inference_deep_3layers(x, A, B):
    with tf.variable_scope("hidden_1"):
        hidden_1 = layer(x, [A,300], [300])
    with tf.variable_scope("hidden_2"):
        hidden_2 = layer(hidden_1, [300,200], [200])
    with tf.variable_scope("hidden_3"):
        hidden_3 = layer(hidden_2, [200,100], [100])
    with tf.variable_scope("output"):
        output = layer(hidden_3, [100, B], [B])
    return output


############################################################
## logistic regression

def inference(x, A, B):
    W = tf.Variable(  tf.zeros([A, B])   )
    b = tf.Variable(tf.zeros( [B]) ) 
    output = tf.nn.softmax(   tf.matmul(x, W) + b  )
    return output

############################################################

def loss_deep(output, y):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)
    loss = tf.reduce_mean(xentropy)
    return loss


############################################################

def loss(output, y):
    output = tf.clip_by_value(output, 1e-10, 1.0)
    dot_product = y * tf.log(output)
    xentropy = -tf.reduce_sum( dot_product  )
    loss = tf.reduce_mean(  xentropy    )
    return loss

#############################################################

def training(cost):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(cost)
    return train_op

#############################################################

def evaluate(output, y):
    correct_prediction = tf.equal( tf.argmax(output,1)  ,   tf.argmax(y,1)  )
    accuracy = tf.reduce_mean( tf.cast(correct_prediction, "float")   )
    return accuracy

#############################################################

x = tf.placeholder("float", [None, A])
y = tf.placeholder("float", [None, B])

#############################################################
## call the core functions

#output = inference(x, A, B)   ###log reg
#cost = loss(output, y)        ###log reg


#output = inference_nn(x, A, B)  ## nn
#cost = loss_deep(output, y)     ## nn

output = inference_deep_3layers(x, A, B)   ## deep nn
cost = loss_deep(output, y)                ## deep nn

train_op = training(cost)
eval_op = evaluate(output, y)

############################################################

y_pred_metrics = tf.argmax(output,1)

############################################################

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

###########################################################
## batch parameters

num_samples_train =  len(y_train)
print num_samples_train
num_batches = int(num_samples_train/batch_size)


############################################################
# MAIN_LOOP()

#y_test_temp = sess.run(y_test_onehot)
#y_train_temp = sess.run(y_train_onehot)

y_test_temp = y_test_onehot
y_train_temp = y_train_onehot


print "running..."
for i in range(number_epochs):
    for batch_n in range(num_batches):
        sta= batch_n*batch_size
        end= sta + batch_size
       
        sess.run( train_op , feed_dict={x: x_train_normalized[sta:end,:] , y: y_train_temp[sta:end, :]})

        print "accuracy ..."
        accuracy_value, y_pred = sess.run([eval_op, y_pred_metrics], feed_dict={x: x_test_normalized, y: y_test_temp})
        print "run {}, {}".format(i, accuracy_value)

        y_true = np.argmax(y_test_temp, 1)
        print y_true
        print y_pred
        print_stats_metrics(y_true, y_pred)
        print '#####################################################################################'

#############################################################

print "<<<<<<<<DONE>>>>>>>>>"
