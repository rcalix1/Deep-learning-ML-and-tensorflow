## this is tensorflow 2.0

import tensorflow as tf
import numpy as np

###########################################################

np.set_printoptions(precision=2)  ## decimal precision

###########################################################

x = np.array([5, 4, 3, 2, 1], dtype=np.int32)
y = [1, 2, 3, 4, 5]  ## a list

x_tf = tf.convert_to_tensor(x)
y_tf = tf.convert_to_tensor(y)

print(x_tf)

ones_tf = tf.ones(  (2, 3)   )
print(ones_tf)


print("look at values of a tensor in tf 2.0")
print(     ones_tf.numpy()      )

###########################################################

print("cast data type ")
x_tf_64 = tf.cast(  x_tf,   tf.int64   )

print(x_tf_64.dtype)

###########################################################

print("transpose")
m = tf.random.uniform(shape=(5, 10))
m_transpose = tf.transpose(m)
print(m.shape, " transposes to ....", m_transpose.shape )

###########################################################

n = tf.ones(   (20,)   )
n_2d = tf.reshape(n, shape=(4, 5))    ## 4X5 = 10

print(n_2d.shape)

###########################################################

p = tf.zeros(   (1, 3, 1, 5 , 1)      )
p_squeezed = tf.squeeze(p, axis=(2, 4)  )   ## remove on these axis
print(p_squeezed.shape)

###########################################################

matrix1_tf = tf.random.uniform(  shape=(5, 2),  minval=-1.0,   maxval=1.0   )

matrix2_tf = tf.random.normal(   shape=(5, 2),  mean=0.0,  stddev=1.0  )

###########################################################

print("element wise multiplication ")

###########################################################

print("<<<<<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>>>>")

