
## Tensorflow 2.0 ideas 
## keras in TF2
## tf1 to tf2 examples

##########################################################

import tensorflow as tf
import numpy as np
import os


##########################################################

## tf.compat()

## sess = tf.compat.v1.Session()

## sess = tf.compat.v1.Session(graph=g)

'''
## v1
with tf.compat.v1.Session(graph=g) as sess:
    sess.run(something)


'''

#########################################################

## TF v2 style



a = tf.constant(1, name='a')
b = tf.constant(2, name='b')
c = tf.constant(3, name='c')

z = 2 * (a - b) + c

tf.print("my result: ", z)


'''

tf.substract(a, b)
tf.multiply(2, r1)  ## elementwise
tf.add(a, b)

'''

#########################################################

## function decorators
## autograph - transforms python code into tensorflow graph code


@tf.function
def compute_z(a, b, c):
    r1 = tf.substract(a, b)
    r2 = tf.multiply(2, r1)
    z  = tf.add(r2, c) 
    return z


##########################################################
## define input_signature

@tf.function(input_signature = (
                                tf.TensorSpec(shape=[None], dtype=tf.int32),
                                tf.TensorSpec(shape=[None], dtype=tf.int32),
                                tf.TensorSpec(shape=[None], dtype=tf.int32),
                              ))
def compute_z2(a, b, c):
    r1 = tf.subtract(a, b)
    r2 = tf.multiply(2, r1)
    z  = tf.add(r2, c) 
    return z


#tf.print("result ", compute_z2(1, 2, 3) )   ## this gives error 
tf.print("result ", compute_z2([1], [2], [3]) )   ## works
tf.print("result ", compute_z2([3, 2], [1, 1], [5, 7]) )     ## works 

##########################################################

x = tf.Variable(initial_value=[1, 2, 3], name='var_x')
print(x)
## x.assign()
## x.assign_add()
## x.value()       ## returns the value in a tensor format 

## tf.Module

'''

class MyModule(tf.Module):
    def __init__(self):
        init = tf.keras.initializers.GlorotNormal()
        self.w1 = tf.Variable(   init(shape=(2, 3))    )
        self.w2 = tf.Variable(   init(shape=(1, 2))    )


m = MyModule()

'''

##########################################################
## cannot create tf.Variable inside the function 

weights = tf.Variable(tf.random.uniform( (3, 3)  ))

@tf.function
def compute_z(x):
    return tf.matmul(weights, x)


x = tf.constant( [ [1], [2], [3]  ], dtype=tf.float32  )
tf.print(compute_z(x))

##########################################################
## computing gradients
## automatic differentiation
## gradient tape

## tf.GradientTape

w = tf.Variable(1.0)
b = tf.Variable(0.5)

x = tf.convert_to_tensor(  [1.4]    )
y = tf.convert_to_tensor(  [2.1]    )
with tf.GradientTape() as tape:
    z = tf.add(  tf.multiply(w, x), b  )
    loss = tf.reduce_sum(  tf.square(  y - z   )  )


dloss_dw = tape.gradient( loss, w  )
tf.print("loss ", dloss_dw)

'''
optimizer = tf.keras.optimizers.SGD()
optimizer.apply_gradients([dloss_dw], [w])

tf.print(w)
'''

##########################################################
## Keras

model = tf.keras.Sequential()
model.add(  tf.keras.layers.Dense(units=16, activation='relu')   )
model.add(  tf.keras.layers.Dense(units=32, activation='relu')   )
model.build( input_shape=(None, 4)  )
model.summary()

##########################################################

t1 = tf.random.uniform(    shape=(5, 2),    minval=-1.0,    maxval=1.0    )
t2 = tf.random.normal(     shape=(5, 2),    mean=0.0,       stddev=1.0    )


## elementwise

t3 = tf.multiply(t1, t2).numpy()
print(t3)

##########################################################

t4 = tf.math.reduce_mean(t1, axis=0)
print(t4)

t5 = tf.linalg.matmul( t1, t2, transpose_b=True)
print(  t5.numpy()    )

##########################################################
## L2 normal

t1_norm = tf.norm(t1, ord=2, axis=1).numpy()
print(t1_norm)

t1_norm_alt = np.sqrt(  np.sum(  np.square(t1), axis=1   )  )
print(t1_norm_alt)

##########################################################
## TF 2.0 datasets module 
## 4 samples
## 3 features per sample

tf.random.set_seed(1)
t_x = tf.random.uniform(   [4, 3], dtype=tf.float32    )
t_y = tf.range(4)

ds_joint = tf.data.Dataset.from_tensor_slices(     (t_x, t_y)      )


for example in ds_joint:
    print(    ' x: ',    example[0].numpy(),    ' y: ',      example[1].numpy()    )


##########################################################
## transform the data

ds_trans = ds_joint.map(   lambda x, y: (x*2-1.0, y)     )

print("now data transformed ")

for example in ds_trans:
    print(   ' x: ',    example[0].numpy(),    ' y: ',    example[1].numpy()    )


##########################################################

tf.random.set_seed(1)
ds = ds_joint.shuffle(   buffer_size=len(t_x)  )

print("now batches shuffled ")

for example in ds:
    print('  X:', example[0].numpy(),
          '  y:', example[1].numpy()   )

##########################################################

ds = ds_joint.batch(batch_size=3, drop_remainder=False)

batch_x, batch_y = next(   iter(ds)    )

print(     'Batch-x:\n', batch_x.numpy()      )

print(     'Batch-y:\n', batch_y.numpy()      )


##########################################################

ds = ds_joint.batch(3).repeat(count=2)
for i, (batch_x, batch_y) in enumerate(ds):
    print(   i, batch_x.shape, batch_y.numpy()    )

##########################################################

import pathlib
imgdir_path = pathlib.Path('images/fruit')
file_list = sorted([    str(path) for path in imgdir_path.glob(  '*.jpg'  )  ])
print(file_list)

import matplotlib.pyplot as plt

fig = plt.figure(  figsize=(10, 5)  )
for i, file in enumerate(  file_list  ):
    img_raw = tf.io.read_file(  file   )
    img = tf.image.decode_image(   img_raw   )
    print('Image shape ', img.shape)
    ax = fig.add_subplot(5, 5, i+1)                 ## the grid size that will show the images 5x5 for 20 images
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(img)
    ax.set_title(    os.path.basename(file), size=15    )

plt.tight_layout()
plt.show()

##########################################################


labels = [     1 if 'apple' in os.path.basename(file) else 0 for file in file_list         ]

print(labels)

ds_files_labels = tf.data.Dataset.from_tensor_slices(  (file_list, labels)  )

for item in ds_files_labels:
    print(item[0].numpy(), item[1].numpy()  )

##########################################################
## resize the images and transform using .map() function 

def load_and_preprocess(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)  ## channels 3 for RGB
    image = tf.image.resize(image, [img_height, img_width])
    image = image / 255.0
    return image, label

##########################################################

img_width, img_height = 120, 80
ds_images_labels = ds_files_labels.map(    load_and_preprocess     )    ## maps to the previous function

fig = plt.figure(  figsize=(10, 6)   )  ## play with these values
for i, example in enumerate(    ds_images_labels    ):
    ax = fig.add_subplot(5, 5, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(  example[0]  )
    ax.set_title(       '{}'.format(example[1].numpy()), size=15        )

plt.tight_layout()
plt.show()

##########################################################
## run
## pip install tensorflow-datasets

import tensorflow_datasets as tfds
print( len(  tfds.list_builders()  )  )


print(  tfds.list_builders()[:5]   )

##########################################################
## view properties
## celeba = celebrity

celeba_bldr = tfds.builder('celeb_a')

print(celeba_bldr.info.features)

print(celeba_bldr.info.features['image'])

print(   celeba_bldr.info.features['attributes'].keys()   )

print(   celeba_bldr.info.citation    )

##########################################################
## download datasets 

## this does not download 

'''
celeba_bldr.download_and_prepare()

datasets = celeba_bldr.as_dataset(shuffle_files=False)
datasets.keys()
'''

##########################################################

mnist, mnist_info = tfds.load( 'mnist', with_info=True, shuffle_files=False   )
print(mnist_info)

print(   mnist.keys()    )

ds_train = mnist['train']
ds_train = ds_train.map(      lambda item:    (item['image'], item['label'])       )

ds_train = ds_train.batch(10)

batch = next(iter(   ds_train   ))

print(    batch[0].shape, batch[1]     )

##########################################################

fig = plt.figure(figsize=(15, 6))

for i, (image, label) in enumerate( zip(batch[0], batch[1]) ):
    ax = fig.add_subplot(2, 5, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(  image[:,:,0], cmap='gray_r'  )
    ax.set_title(   '{}'.format(label), size=15   )

plt.show()

##########################################################

print("<<<<<<<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>>>>>>>")







