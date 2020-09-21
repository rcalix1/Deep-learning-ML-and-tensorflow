##########################################################################
## extract datasets from Tensorflow 2 module called tensorflow_datasets
## and store them in a regular python dictionary
## this example uses portuguese to english data sets
##########################################################################

import tensorflow as tf
import tensorflow_datasets as tfds
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle

##########################################################################

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']

##########################################################################

encoding = 'utf-8'

train_count = 0
en_pt_dict_train = {}

for pt, en in train_examples:
    en_pt_dict_train[train_count] = {}
    en_pt_dict_train[train_count]['en'] = str(en.numpy(), encoding)
    en_pt_dict_train[train_count]['pt'] = str(pt.numpy(), encoding)
    train_count = train_count + 1
 
###########################################################################

en_pt_dict_val = {}
val_count   = 0

for pt, en in val_examples:
    en_pt_dict_val[val_count] = {}
    en_pt_dict_val[val_count]['en'] = str(en.numpy(), encoding)
    en_pt_dict_val[val_count]['pt'] = str(pt.numpy(), encoding)
    val_count = val_count + 1
    
###########################################################################

print(en_pt_dict_train)
print(en_pt_dict_val)

############################################################################

print("sizes")
print(train_count)
print(val_count)

###########################################################################
    
def load_dictionary(file_name):
    with open(file_name, 'rb') as handle:
        dict = pickle.loads(   handle.read()  )
    return dict

###########################################################################

def write_dictionary(file_name, dict):
    with open(file_name, 'wb') as handle:
        pickle.dump(dict, handle)

###########################################################################

write_dictionary("data/en_pt_train_dictionary.txt", en_pt_dict_train)
write_dictionary("data/en_pt_val_dictionary.txt",   en_pt_dict_val  )

###########################################################################


train      = load_dictionary("data/en_pt_train_dictionary.txt")
validation = load_dictionary("data/en_pt_val_dictionary.txt")

print(train)
print(validation)

###########################################################################

print("<<<<<<<<<<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")









