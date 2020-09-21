## create dict english spanish
####################################################

import sklearn
import tensorflow as tf
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import pickle

####################################################

en_file = "europarl-v7.es-en.en"
sp_file = "europarl-v7.es-en.es"

###################################################

en_sents = open(en_file).readlines()
sp_sents = open(sp_file).readlines()

###################################################

def get_tokens(sentence):
    tokens_list = []
    tokens = word_tokenize(sentence)
    for word in tokens:
        tokens_list.append(word)
    return tokens_list

###################################################

encoding = 'utf-8'

i = 0

train_count = 0
en_sp_dict_train = {}

test_count   = 0
en_sp_dict_test = {}


for line in en_sents:

    print("*********************************************")
    
    en_sent = en_sents[i]
    sp_sent = sp_sents[i]
    
    en_sent = en_sent.replace("\n", "")
    sp_sent = sp_sent.replace("\n", "")
    
    print(    en_sent   )
    print(    sp_sent   )
    
    if i < 1755000:
        en_sp_dict_train[train_count] = {}
        en_sp_dict_train[train_count]['en'] = str(en_sent)
        en_sp_dict_train[train_count]['sp'] = str(sp_sent)
        train_count = train_count + 1
    else:
        en_sp_dict_test[test_count] = {}
        en_sp_dict_test[test_count]['en'] = str(en_sent)
        en_sp_dict_test[test_count]['sp'] = str(sp_sent)
        test_count = test_count + 1
        
    i = i + 1
  

    
    '''
    en_length = len(get_tokens(en_sent))
    es_length = len(get_tokens(es_sent))
    
    if en_length <= 38 and es_length <= 38:
        sents_less_than_40 = sents_less_than_40 + 1
    '''

    
####################################################

print("number of all pairs ", i )

print("train count ", train_count)

print("test count ",  test_count)


#####################################################


print(en_sp_dict_test )

print(len(en_sp_dict_train))
print(len(en_sp_dict_test ))

######################################################
    
def load_dictionary(file_name):
    with open(file_name, 'rb') as handle:
        dict = pickle.loads(   handle.read()  )
    return dict

###########################################################################

def write_dictionary(file_name, dict):
    with open(file_name, 'wb') as handle:
        pickle.dump(dict, handle)

###########################################################################

write_dictionary("data/en_sp_train_dictionary.txt", en_sp_dict_train )
write_dictionary("data/en_sp_test_dictionary.txt",  en_sp_dict_test  )

###########################################################################


train_      = load_dictionary("data/en_sp_train_dictionary.txt")
test_       = load_dictionary("data/en_sp_test_dictionary.txt")

print(train_)
print(test_ )


######################################################

print("<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>")
