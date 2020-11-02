## XOR problem TF 2 - Keras

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

## no hidden layers (i.e. logistic regression)


## w = [2, 1]  and bias [1]
model = tf.keras.Sequential()
model.add(
             tf.keras.layers.Dense(
                                      units=1,          # just one neuron in the output layer   ## w = 2x1
                                      input_shape=(2,),
                                      activation='sigmoid' 
                                  )
         )

model.summary()

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

## now add hidden layers to create non linear decision boundaries

tf.random.set_seed(1)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=4, input_shape=(2,), activation='relu' ))   ## w = [2, 4]  b=[4]
model.add(tf.keras.layers.Dense(units=4,  activation='relu' ))                    ## w = [4, 4]  b=[4]
model.add(tf.keras.layers.Dense(units=4,  activation='relu' ))                    ## w = [4, 4]  b=[4]
model.add(tf.keras.layers.Dense(units=1,  activation='sigmoid' ))                 ## w = [4, 1]  b=[1]

model.summary()


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

print("<<<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>>")



