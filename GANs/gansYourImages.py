import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

###################################################################

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


###################################################################

X = tf.placeholder(tf.float32, shape=[None, 784])

D_W1 = tf.Variable(xavier_init([784, 128]))
D_b1 = tf.Variable(tf.zeros(shape=[128]))

D_W2 = tf.Variable(xavier_init([128, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]


Z = tf.placeholder(tf.float32, shape=[None, 100])

G_W1 = tf.Variable(xavier_init([100, 128]))
G_b1 = tf.Variable(tf.zeros(shape=[128]))

G_W2 = tf.Variable(xavier_init([128, 784]))
G_b2 = tf.Variable(tf.zeros(shape=[784]))

theta_G = [G_W1, G_W2, G_b1, G_b2]

############################################################################

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

############################################################################

def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

###############################################################################

G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

# D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# G_loss = -tf.reduce_mean(tf.log(D_fake))

# Alternative losses:
# -------------------
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

mb_size = 128
Z_dim = 100


############################################################################

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

############################################################################
## this will create your own data set (i.e. your_mnist)
## put your images in testA (I used pngs)

from PIL import Image

#import cv2
import glob
import numpy as np


train = []

files = glob.glob ("testA/*.png") # your image path

for myFile in files:
    image = np.array(Image.open(myFile).convert('LA'))
    image = image[...,:1] ## size is (28,28,2) and now (28,28,1) - removes png transparency
    #print(image.shape[1])
    #print(image.shape[0])
    new_image = image.reshape(image.shape[1]*image.shape[0])
    #print(new_image.shape)
    #input("???")
    train.append(new_image)
    

train = np.array(train,dtype='float32') #as mnist


# convert (number of images x height x width x number of channels) to (number of images x (height * width *3))
# for example (120 * 40 * 40 * 3)-> (120 * 4800)
#train = np.reshape(train,[train.shape[0],train.shape[1]*train.shape[2]*train.shape[3]])
#train = np.reshape(train,[train.shape[0],train.shape[1]*train.shape[2]])

print(train.shape)
print(train.shape[0])
print(train.shape[1])
#input("train size is")

# save numpy array as .npy formats
np.save('train',train)

your_mnist = train

###########################################################################
## normalization is very important

x=your_mnist
xmax, xmin = x.max(), x.min()
x = (x - xmin)/(xmax - xmin)

your_mnist = x


############################################################################

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0
i2 = 0
for it in range(1000000):
    if it % 1000 == 0:
        samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})

        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    ########################################################################
    ## if using built in mnist

    #X_mb, _ = mnist.train.next_batch(mb_size)

    ########################################################################
    ## if using your own data - this could be better

    index = i2*mb_size
    X_mb = your_mnist[index:index+mb_size,:]
    #print(X_mb.shape)
    #input("???")
    i2 = i2 + 1    
    if index >= your_mnist.shape[0]:
        i2 = 0
        

    #######################################################################


    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()


############################################################################


