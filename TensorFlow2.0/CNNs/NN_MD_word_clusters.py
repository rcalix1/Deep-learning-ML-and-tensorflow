
## use conda prompt
## conda create -n tf_v2 tensorflow
## conda env list
## conda activate tf_v2
## pip install keras, sklearn


import tensorflow as tf
import numpy as np

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

##################################################################




data_covid     = np.load('data/data_structures_covid.npy', allow_pickle=True)
data_pulmonary = np.load('data/data_structures_pulmonary.npy', allow_pickle=True)

y_covid     = np.ones(   (data_covid.shape[0],     ), dtype=int)
y_pulmonary = np.zeros(  (data_pulmonary.shape[0], ), dtype=int)

print(data_covid.shape)
print(data_pulmonary.shape)

#print(data_covid[0, :, :])

X_data = np.concatenate((data_covid, data_pulmonary), axis=0 )
y_data = np.concatenate((y_covid, y_pulmonary), axis=0 )

print(X_data.shape)
print(y_data.shape)

#################################################################
# create random train/test split


indices = np.arange(X_data.shape[0])
num_training_instances = int(0.7 * X_data.shape[0])
np.random.shuffle(indices)

train_indices = indices[:num_training_instances]
test_indices = indices[num_training_instances:]


################################################################
##  (105727, 11, 128)

X_data = X_data.reshape(105727,11,128, 1)


################################################################

# split the actual data
X_train, X_test = X_data[train_indices, :, :, :], X_data[test_indices, :, :, :]
y_train, y_test = y_data[train_indices], y_data[test_indices]

################################################################
#one-hot encode target column

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

X_train = np.asarray(X_train).astype(np.float32)
X_test = np.asarray(X_test).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)

#################################################################


model = Sequential()#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(11,128,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#################################################################
## The number of epochs is the number of times the model will cycle through the data. 
## The more epochs we run, the more the model will improve, up to a certain point. 


model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)

#################################################################


y_pred = model.predict(X_test)

##################################################################

y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)


print(y_pred[:20])
print(y_test[:20])

# Print f1, precision, and recall scores
print(precision_score(y_test, y_pred , average="macro"))
print(recall_score(y_test, y_pred , average="macro"))
print(f1_score(y_test, y_pred , average="macro"))

#################################################################

# save numpy array as .npy formats
#np.save('train',train)


#################################################################
## normalization is very important

'''

x=your_mnist
xmax, xmin = x.max(), x.min()
x = (x - xmin)/(xmax - xmin)

your_mnist = x

'''

#################################################################

print("<<<<<<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>>>>>>>>")






