import tensorflow as tf


# Creates a graph.
with tf.device('/cpu:0'):
  a = tf.zeros(shape=[10000,1000], dtype=tf.float32) #constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.zeros(shape=[1000,1000], dtype=tf.float32) #constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
for i in range(200):
    print  i
    print sess.run(c)
