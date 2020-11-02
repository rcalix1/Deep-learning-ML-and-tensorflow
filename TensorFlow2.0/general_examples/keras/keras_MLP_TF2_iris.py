## IRIS data and classification
## two layer multi layer perceptron

########################################################

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

########################################################

## run
## pip install tensorflow-datasets

import tensorflow_datasets as tfds

########################################################

iris, iris_info = tfds.load('iris', with_info=True)
print(iris_info)

########################################################

## shuffle and split the data

tf.random.set_seed(1)

ds_orig = iris['train']

## train and test should always be separate sets
ds_orig = ds_orig.shuffle(   150, reshuffle_each_iteration=False  )  ## false set here so train and test are not mixed  

ds_train_orig = ds_orig.take(100)
ds_test       = ds_orig.skip(100)

########################################################
## use map to convert dictionary to tuple

ds_train_orig = ds_train_orig.map(
                                    lambda x: (x['features'], x['label'])
                                 )

ds_test       = ds_test.map(
                                    lambda x: (x['features'], x['label'])
                           )

########################################################
## Keras API
## dense -> is the fully connected (fc) or linear layer [  f(w*x+b)  ] here f() is the activation function

iris_model = tf.keras.Sequential([
                                    tf.keras.layers.Dense(16, activation='sigmoid', name='fc1', input_shape=(4,)), 
                                    tf.keras.layers.Dense(3,  activation='softmax', name='fc2'                  )
                                ])

## number of parameters is (neurons_in + 1) * neurons_out
iris_model.summary()

########################################################

iris_model.compile(
                      optimizer='adam', 
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy']
                  )

########################################################

num_epochs = 120
training_size = 100
batch_size= 5
steps_per_epoch = np.ceil(       training_size/batch_size        )

########################################################

ds_train = ds_train_orig.shuffle(    buffer_size=training_size    )
ds_train = ds_train.repeat()
ds_train = ds_train.batch(batch_size=batch_size)
ds_train = ds_train.prefetch(   buffer_size=1000   ) 

history = iris_model.fit(   
                            ds_train,
                            epochs=num_epochs,
                            steps_per_epoch=steps_per_epoch,
                            verbose=0
                        )

########################################################

hist = history.history

fig = plt.figure(   figsize=(12, 5)   )
ax = fig.add_subplot(1, 2, 1)
ax.plot(   hist['loss'], lw=3    )
ax.set_title(  'Training loss', size=15   )
ax.set_xlabel(  'Epoch', size=15   )
ax.tick_params(  axis='both', which='major', labelsize=15  )
ax = fig.add_subplot(1, 2, 2)
ax.plot(     hist['accuracy'], lw=3       )
ax.set_title(  'Training accuracy', size=15         ) 
ax.set_xlabel(   'Epoch:', size=15   )
ax.tick_params(   axis='both', which='major', labelsize=15   )
plt.show()

########################################################

results = iris_model.evaluate(  ds_test.batch(50), verbose=0  )
print(  'Test loss: {:.4f}   Test accuracy: {:.4f}'.format(*results)   )

########################################################
## saving and reloading the trained model

iris_model.save( 
                   'iris-classifier.h5',
                   overwrite=True,
                   include_optimizer=True,
                   save_format='h5'

               )

########################################################
## Now reload the saved model

iris_model_new = tf.keras.models.load_model('iris-classifier.h5')

iris_model_new.summary()

########################################################

results2 = iris_model_new.evaluate(  ds_test.batch(33), verbose=0  )
print(  'Test loss: {:.4f}   Test accuracy: {:.4f}'.format(*results2)   )


########################################################

print("<<<<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>>")



