## simple auto-encoder

import tensorflow as tf
import numpy as np
import math

###############################################################################
## data

input = np.array([[2.0, 1.0, 1.0, 2.0],
                  [-2.0, 1.0, -1.0, 2.0],
                  [0.0, 1.0, 0.0, 2.0],
                  [0.0, -1.0, 0.0, -2.0],
                  [0.0,-1.0, 0.0, -2.0]])

print input

noisy_input = input + 0.2 * np.random.random_sample((input.shape)) - 0.1

print noisy_input

output = input

#################################################################################
# normalizing

## scale to [0, 1]
scaled_input_1 = np.divide((noisy_input - noisy_input.min()),(noisy_input.max()-noisy_input.min()))
scaled_output_1 = np.divide( (output-output.min()), (output.max()-output.min()))

print scaled_input_1
print scaled_output_1

## scale to [-1, 1]
scaled_input_2 = (scaled_input_1*2)-1
scaled_output_2 = (scaled_output_1*2)-1

print scaled_input_2
print scaled_output_2

################################################################################
## data set to use

input_data = scaled_input_2
output_data = scaled_output_2

###############################################################################

print input_data.shape
print output_data.shape

n_sample, n_features = input_data.shape

###############################################################################

x = tf.placeholder("float", [None, n_features]) ## (None, 4)
y_ = tf.placeholder("float", [None, n_features]) ## (None, 4)

###############################################################################
## nn with 1 hidden layer, neural net 4x3x4

def inference(x):
    Wh = tf.Variable( tf.random_uniform(  (4,3), -1.0 / math.sqrt(n_features), 1.0 / math.sqrt(n_features)   )  )
    bh = tf.Variable(   tf.zeros([3])     )
    h1 = tf.nn.tanh( tf.matmul(x, Wh) +  bh   )

    #Wy = tf.transpose(Wh) #tied weights
    Wy = tf.Variable( tf.random_uniform( (3,4) ,  -1.0 / math.sqrt(n_features), 1.0 / math.sqrt(n_features)      )      )
    by = tf.Variable(  tf.zeros([4])        )
    y = tf.nn.tanh( tf.matmul(h1, Wy) + by    )

    return y, Wh, bh, h1

###############################################################################

def loss_lse(y, y_): 
    meansq = tf.reduce_mean(  tf.square(y_ - y)  )  ## lse
    return meansq

#############################################################################

def loss_cross_entropy(y, y_):
    #cross_entropy = -tf.reduce_sum(y_ * tf.log(y)) ##cross entropy
    #return cross_entropy
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    loss = tf.reduce_mean(xentropy)
    return loss


##############################################################################

def train(cost):
    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
    return train_step

################################################################################

def test():
    print "hello"

##############################################################################

y, Wh, bh, h1 = inference(x)

cost_lse = loss_lse(y, y_)
cost_xentropy = loss_cross_entropy(y, y_) ## alternative

train_op = train(cost_lse)

##############################################################################

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#############################################################################

n_epochs = 5000
batch_size = min(50, n_sample) ## (50, 5)

for i in range(n_epochs):
    sample = np.random.randint(n_sample, size=batch_size)
    batch_xs = input_data[sample,:]
    batch_ys =  output_data[sample,:]
    result = sess.run(train_op, feed_dict={x: batch_xs, y_:batch_ys})
    if i % 100 == 0:
        print i, sess.run(cost_xentropy, feed_dict={x:batch_xs,y_:batch_ys}), sess.run(cost_lse, feed_dict={x:batch_xs , y_:batch_ys})



print "output_data"
print output_data

print "predicted_output_data"
print sess.run(y, feed_dict={x: input_data})    

print "Wh"
print sess.run(Wh)

print "bh"
print sess.run(bh)

print "h1"
print sess.run(h1, feed_dict={x: input_data})


