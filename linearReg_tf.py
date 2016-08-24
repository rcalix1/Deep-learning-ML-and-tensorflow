import tensorflow as tf
import numpy as np

#######################################################

def inference(x):
    W = tf.Variable(tf.zeros([1,1]))
    b = tf.Variable(tf.zeros([1]))
    y = tf.matmul(x, W) + b
    return y

#######################################################

def loss(y, y_):
    cost = tf.reduce_sum(tf.pow((y_ - y),2))
    return cost

#######################################################

def training(cost):
    train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(cost)
    return train_step   

#######################################################

def evaluate(y, y_):
    #a = tf.argmax(y,1)
    #b = tf.argmax(y_,1)
    correct_prediction = y #tf.equal(y, y_)
    float_val = tf.cast(correct_prediction,tf.float32)
    prediction_as_float = tf.reduce_mean(float_val)
    return prediction_as_float

#######################################################
x = tf.placeholder(tf.float32, [None, 1])
y_ = tf.placeholder(tf.float32, [None, 1])

#W = tf.Variable(tf.zeros([1,1]))
#b = tf.Variable(tf.zeros([1]))
#y = tf.matmul(x, W) + b

y = inference(x)
cost = loss(y, y_ )
train_step = training(cost)
eval_op = evaluate(y, y_)

#cost = tf.reduce_sum(tf.pow((y_ - y),2))
#train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(cost)

###########################################
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

###########################################
steps = 100
for i in range(steps):
    xs = np.array([[i]])   #house size
    ys = np.array([[5*i]]) #house price
    
    feed = {x:xs, y_:ys}
    sess.run(train_step, feed_dict=feed)    
    
    print("After %d iteration: " % i)
    #print("W: %f" % sess.run(W))
    #print("b: %f" % sess.run(b))
    ##########################################

for i in range(100,200):
    xs_test = np.array([[i]])   #house size
    ys_test = np.array([[2*i]]) #house price
    feed_test = {x:xs_test, y_:ys_test}
    result = sess.run(eval_op, feed_dict=feed_test)
    #print sess.run(y)
    print "Run {},{}".format(i, result)
    x_input = raw_input()

