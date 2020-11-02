## Functional API for XOR problem TF 2 - Keras

#############################################################

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#############################################################

## generate XOR data

tf.random.set_seed(1)
np.random.seed(1)

#############################################################

X = np.random.uniform(  low=-1, high=1, size=(200, 2)  )  ## 200 samples and 2 features
y = np.ones(  len(X)   )
#indices_that_meet_XOR_condition =   X[:, 0] * X[:, 1] < 0             ## x0 * x1 < 0
print("XOR indices for class 0")

y[    X[:, 0] * X[:, 1] < 0    ] = 0
print(  y  )


x_train = X[:100, :]
y_train = y[:100]

x_valid = X[100:, :]
y_valid = y[100:]

#############################################################

## plot data

fig = plt.figure(     figsize=(6, 6)      )
plt.plot(  
           X[y==0, 0], 
           X[y==0, 1],
           'o',
           alpha=0.75,
           markersize=10
        )
plt.plot(  
           X[y==1, 0], 
           X[y==1, 1],
           '<',
           alpha=0.75,
           markersize=10
        )
plt.xlabel(   r'$x_1$', size=15   )
plt.ylabel(   r'$x_2$', size=15   )
plt.show()




#############################################################
#############################################################
#############################################################
#############################################################

## architecture using functional API



tf.random.set_seed(1)

inputs = tf.keras.Input(shape=(2,))


h1 = tf.keras.layers.Dense(units=4,  activation='relu' )(inputs)                    ## w = [2, 4]  b=[4]
h2 = tf.keras.layers.Dense(units=4,  activation='relu' )(h1)                        ## w = [4, 4]  b=[4]
h3 = tf.keras.layers.Dense(units=4,  activation='relu' )(h2)                        ## w = [4, 4]  b=[4]


outputs = tf.keras.layers.Dense(units=1,  activation='sigmoid' )(h3)                 ## w = [4, 1]  b=[1]


model = tf.keras.Model(inputs=inputs, outputs=outputs)


model.summary()

################################################################

model.compile(
                optimizer=tf.keras.optimizers.SGD(),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[   tf.keras.metrics.BinaryAccuracy()    ]
             )


hist = model.fit(
                    x_train, 
                    y_train, 
                    validation_data = (x_valid, y_valid),
                    epochs=200,
                    batch_size=2,
                    verbose=0
                )


#############################################################

## install this library for visualization

## on conda linux terminal:
## conda install mlxtend -c conda-forge


from mlxtend.plotting import plot_decision_regions

history = hist.history

fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1, 3, 1)
plt.plot(   history['loss'], lw=4   )
plt.plot(   history['val_loss'], lw=4   )
plt.legend(  ['train loss', 'validation loss'], fontsize=15 )
ax.set_xlabel('Epochs', size=15)

ax = fig.add_subplot(  1, 3, 2  )
plt.plot(  history['binary_accuracy'], lw=4   )
plt.plot(  history['val_binary_accuracy'], lw=4   )
plt.legend(  ['train accuracy', 'validation accu'], fontsize=15 )
ax.set_xlabel('Epochs', size=15)

ax = fig.add_subplot(  1, 3, 3  )
plot_decision_regions(X=x_valid, y=y_valid.astype(np.integer), clf=model)
ax.set_xlabel(r'$x_1$', size=15)
ax.xaxis.set_label_coords(1, -0.025)
ax.set_ylabel(   r'$x_2$', size=15   )
ax.yaxis.set_label_coords(-0.025, 1)
plt.show()



#############################################################
#############################################################
#############################################################
#############################################################

## Now, implementing models based on the Keras Model Class approach
## this is useful for object oriented coding in Tensorflow

## this is also referred to as the famous "subclassing"
## a lot of new code is written this way such as Transformers

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.hidden_1 = tf.keras.layers.Dense(units=4, activation='relu')
        self.hidden_2 = tf.keras.layers.Dense(units=4, activation='relu')
        self.hidden_3 = tf.keras.layers.Dense(units=4, activation='relu')
        self.output_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')

    def call(self, inputs):
        h1 = self.hidden_1(inputs)
        h2 = self.hidden_2(h1)
        h3 = self.hidden_3(h2)
        return self.output_layer(h3)
 
#############################################################

tf.random.set_seed(1)

model = MyModel()
model.build(input_shape=(None, 2))

model.summary()

#############################################################

model.compile(
                optimizer=tf.keras.optimizers.SGD(),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[   tf.keras.metrics.BinaryAccuracy()    ]
             )


hist = model.fit(
                    x_train, 
                    y_train, 
                    validation_data = (x_valid, y_valid),
                    epochs=200,
                    batch_size=2,
                    verbose=0
                )


#############################################################

## install this library for visualization

## on conda linux terminal:
## conda install mlxtend -c conda-forge


from mlxtend.plotting import plot_decision_regions

history = hist.history

fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1, 3, 1)
plt.plot(   history['loss'], lw=4   )
plt.plot(   history['val_loss'], lw=4   )
plt.legend(  ['train loss', 'validation loss'], fontsize=15 )
ax.set_xlabel('Epochs', size=15)

ax = fig.add_subplot(  1, 3, 2  )
plt.plot(  history['binary_accuracy'], lw=4   )
plt.plot(  history['val_binary_accuracy'], lw=4   )
plt.legend(  ['train accuracy', 'validation accu'], fontsize=15 )
ax.set_xlabel('Epochs', size=15)

ax = fig.add_subplot(  1, 3, 3  )
plot_decision_regions(X=x_valid, y=y_valid.astype(np.integer), clf=model)
ax.set_xlabel(r'$x_1$', size=15)
ax.xaxis.set_label_coords(1, -0.025)
ax.set_ylabel(   r'$x_2$', size=15   )
ax.yaxis.set_label_coords(-0.025, 1)
plt.show()

#############################################################
#############################################################
#############################################################
#############################################################
#############################################################

## subclassing for custom layers
## example: change the standard w*x + b
## to:                          w*(x + e) + b
## where:          e -->> noise

class NoisyLinear(tf.keras.layers.Layer):

    def __init__(self, output_dim, noise_stddev=0.1, **kwargs):
        self.output_dim = output_dim
        self.noise_stddev = noise_stddev
        super(NoisyLinear, self).__init__(**kwargs)


    def build(self, input_shape):
        self.w = self.add_weight(   
                                    name='weights', 
                                    shape=(input_shape[1], self.output_dim),    
                                    initializer='random_normal',
                                    trainable=True
                                )

        self.b = self.add_weight(
                                    shape=(self.output_dim,),
                                    initializer='zeros',
                                    trainable=True
                                )
 

    def call(self, inputs, training=False):
        if training:
            batch = tf.shape(inputs)[0]
            dim   = tf.shape(inputs)[1]
            noise = tf.random.normal(   shape=(batch, dim), mean=0.0, stddev=self.noise_stddev   )
            noisy_inputs = tf.add(inputs, noise)
        else:
            noisy_inputs = inputs
        z = tf.matmul(noisy_inputs, self.w) + self.b
        return tf.keras.activations.relu(z)


    ## this serializes the object 
    def get_config(self):
        config = super(NoisyLinear, self).get_config()
        config.update(   {'output_dim': self.output_dim, 'noise_stddev': self.noise_stddev}    )
        return config


#############################################################


tf.random.set_seed(1)
noisy_layer = NoisyLinear(4)
noisy_layer.build(  input_shape=(None, 4)   )

x = tf.zeros(   shape=(1,4)   )
tf.print(  noisy_layer(x, training=True)   )


config = noisy_layer.get_config()
new_layer = NoisyLinear.from_config(config)
tf.print(   new_layer(x, training=True)   )


################################################################

tf.random.set_seed(1)

model = tf.keras.Sequential([
                               NoisyLinear(4, noise_stddev=0.1),
                               tf.keras.layers.Dense(units=4, activation='relu'),
                               tf.keras.layers.Dense(units=4, activation='relu'),
                               tf.keras.layers.Dense(units=1, activation='sigmoid')
                           ])
                  
model.build(input_shape=(None, 2))

model.summary()

################################################################

model.compile(
                optimizer=tf.keras.optimizers.SGD(),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[   tf.keras.metrics.BinaryAccuracy()    ]
             )


hist = model.fit(
                    x_train, 
                    y_train, 
                    validation_data = (x_valid, y_valid),
                    epochs=200,
                    batch_size=2,
                    verbose=0
                )

#############################################################


from mlxtend.plotting import plot_decision_regions

history = hist.history

fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1, 3, 1)
plt.plot(   history['loss'], lw=4   )
plt.plot(   history['val_loss'], lw=4   )
plt.legend(  ['train loss', 'validation loss'], fontsize=15 )
ax.set_xlabel('Epochs', size=15)

ax = fig.add_subplot(  1, 3, 2  )
plt.plot(  history['binary_accuracy'], lw=4   )
plt.plot(  history['val_binary_accuracy'], lw=4   )
plt.legend(  ['train accuracy', 'validation accu'], fontsize=15 )
ax.set_xlabel('Epochs', size=15)

ax = fig.add_subplot(  1, 3, 3  )
plot_decision_regions(X=x_valid, y=y_valid.astype(np.integer), clf=model)
ax.set_xlabel(r'$x_1$', size=15)
ax.xaxis.set_label_coords(1, -0.025)
ax.set_ylabel(   r'$x_2$', size=15   )
ax.yaxis.set_label_coords(-0.025, 1)
plt.show()

#############################################################

print("<<<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>>")







