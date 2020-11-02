## activation functions 

#############################################################

import numpy as np

import tensorflow as tf


#############################################################

X = np.array([1, 1.4, 2.5]) 

w = np.array([0.4, 0.3, 0.5])

#############################################################

def net_input(X, w):
    return np.dot(X, w)

############################################################

def logistic(z):
    return 1.0 / ( 1.0 + np.exp(-z) )

#############################################################

def logistic_activation(X, w):
    z = net_input(X, w)
    return logistic(z)

#############################################################

print('P(y=1|x) = %.3f' % logistic_activation(X, w)  )

#############################################################
## now multi-class

## W: shape=(,)
## dot(X, W) = [1 x 4] * [4 x 3] = [1 x 3]

W = np.array( [[ 1.1, 1.2, 0.8, 0.4],
               [ 0.2, 0.4, 1.0, 0.2],
               [ 0.6, 1.5, 1.2, 0.7]]  )


A = np.array( [[1, 0.1, 0.4, 0.6]])

z = np.dot(W, A[0])

y_probas = logistic(z)

print(  'Net input: \n', z   )

print(  'Output units: \n', y_probas)

#############################################################

## or the correct way to do this

print("***********************************************")
print("or the correct way with transpose ...")

W_transpose = np.transpose(W)

z_tra = np.dot(A, W_transpose)

y_probas_tra = logistic(z)

print(  'Net input: \n', z_tra   )

print(  'Output units: \n', y_probas_tra)

#############################################################

## pick the max val

y_class = np.argmax(z, axis=0)  ## selects index of max value
print(y_class)

y_class = np.argmax(z_tra[0], axis=0)  ## selects index of max value
print(y_class)

#############################################################

## the softmax
## softmax = (e**z) / sum(e**z)

def softmax(z):
    return np.exp(z) / np.sum(  np.exp(z)  )

#############################################################

y_probas_softmax = softmax(z)

print("softmax results")
print(y_probas_softmax)


print(np.sum(y_probas_softmax))

#############################################################

## now in tensorflow 

print(z)
z_tensor = tf.expand_dims(z, axis=0)
print(z_tensor) 

#############################################################
## softmax

z_tensor_softmax = tf.keras.activations.softmax(z_tensor)
print("softmax ")
print(z_tensor_softmax)

#############################################################
## tanh

z_tensor_tanh = tf.keras.activations.tanh(z_tensor)
print("tanh [-1, +1]")
print(z_tensor_tanh)

#############################################################
## sigmoid

z_tensor_sigmoid = tf.keras.activations.sigmoid(z_tensor)
print("sigmoid range [0, 1]")
print(z_tensor_sigmoid)

#############################################################
## RELU
## RELU improves on the vanishing gradients problem
## weights are not learned efficiently
## relu -->   theta(z) = max(0, z)
## RELU is still non-linear 

z_tensor_relu = tf.keras.activations.relu(z_tensor)
print("relu range [0, +infinity]")
print(z_tensor_relu)

############# other example

print("***********************************************")

other_z = np.array([1.76, -2.4, 1.46])
other_z_tensor = tf.expand_dims(other_z, axis=0)
print(other_z_tensor) 

z_tensor_relu = tf.keras.activations.relu(other_z_tensor)
print("relu range [0, +infinity]")
print(z_tensor_relu)


#############################################################

print("<<<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>")





