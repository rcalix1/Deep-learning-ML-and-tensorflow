## logistic regression algorithm
## supervised learning classifier

#############################################################

import tensorflow as tf
import numpy as np
from numpy import genfromtxt
import sklearn
from sklearn.preprocessing import StandardScaler

#############################################################

number_epochs = 1000
learning_rate = 0.01
batch_size = 10

#############################################################

#training_data = genfromtxt('cs-training.csv', delimiter=',')
#testing_data  = genfromtxt('cs-testing.csv', delimiter=',')

#############################################################

x_train = genfromtxt('cs-training.csv', delimiter=',', usecols=(i for i in range(1,5)) ) 
y_train = genfromtxt('cs-training.csv', delimiter=',', usecols=(0))

x_test = genfromtxt('cs-testing.csv', delimiter=',', usecols=(i for i in range(1,5))  )
y_test = genfromtxt('cs-testing.csv', delimiter=',', usecols=(0))

############################################################
## normalizing

sc = StandardScaler()
sc.fit(x_train)

x_train_normalized = sc.transform(x_train)
x_test_normalized  = sc.transform(x_test)

############################################################
# one-hot encoding

depth = 3
y_train_onehot = tf.one_hot(y_train, depth)
y_test_onehot  = tf.one_hot(y_test, depth)

#############################################################
# features (A) 

A = len(x_train[0])
print A # number of features

#############################################################
# classes (B)

B = 3 #len(sess.run(y_train_onehot[0]))
print "number of classes ", B

############################################################

def inference(x, A, B):
    W = tf.Variable(  tf.zeros([A, B])   )
    b = tf.Variable(tf.zeros( [B]) ) 
    output = tf.nn.softmax(   tf.matmul(x, W) + b  )
    return output

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

output = inference(x, A, B)
cost = loss(output, y)
train_op = training(cost)
eval_op = evaluate(output, y)

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

print "running..."
for i in range(number_epochs):
    for batch_n in range(num_batches):
        sta= batch_n*batch_size
        end= sta + batch_size
        y_temp = sess.run(y_train_onehot)
        sess.run( train_op , feed_dict={x: x_train_normalized[sta:end,:] , y: y_temp[sta:end, :]})

        y_test_temp = sess.run(y_test_onehot)
        print "accuracy ..."
        accuracy_score = sess.run(eval_op, feed_dict={x: x_test_normalized, y: y_test_temp})
        print "run {}, {}".format(i, accuracy_score)
#############################################################

print "<<<<<<<<DONE>>>>>>>>>"
