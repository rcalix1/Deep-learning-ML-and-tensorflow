################################################################
# Getting started with deep learning 
# by Ricardo A. Calix
# simple recurrent neural network for nlp
# rnn with LSTM cells

###############################################################


import tensorflow as tf
import numpy as np
from numpy import genfromtxt
from sklearn import datasets
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
import sklearn

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt

import random
import collections
import time

from tensorflow.contrib import rnn

#from tensorflow.python.ops import rnn, rnn_cell

###########################################################
## set parameters

import warnings
warnings.filterwarnings("ignore") 

np.set_printoptions(threshold=np.inf) #print all values in numpy array


#############################################################

learning_rate = 0.001
n_epochs = 50000   #100000  ##27000  
#batch_size = 100
display_step = 1000

vocab_size = 0  ## updated once data is read

n_input = 3
# number of units in RNN cell
n_hidden = 512


##########################################################################

training_file = 'belling_the_cat.txt'

##########################################################################

def read_data(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [word for i in range(len(content)) for word in content[i].split()]
    content = np.array(content)
    return content

#########################################################################

training_data = read_data(training_file)

##########################################################################

def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

#######################################################################

dictionary, reverse_dictionary = build_dataset(training_data)
#print(reverse_dictionary)
vocab_size = len(dictionary)

###################################################################
## print stats 
precision_scores_list = []
accuracy_scores_list = []

def print_stats_metrics(y_test, y_pred):    
    print('Accuracy: %.2f' % accuracy_score(y_test,   y_pred) )
    #Accuracy: 0.84
    accuracy_scores_list.append(accuracy_score(y_test,   y_pred) )
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print("confusion matrix")
    print(confmat)
    print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
    precision_scores_list.append(precision_score(y_true=y_test, y_pred=y_pred, average='weighted'))
    print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred, average='weighted'))
    print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred, average='weighted'))
    print('F1-measure: %.3f' % f1_score(y_true=y_test, y_pred=y_pred, average='weighted'))

#####################################################################

def plot_metric_per_epoch():
    x_epochs = []
    y_epochs = [] 
    for i, val in enumerate(accuracy_scores_list):
        x_epochs.append(i)
        y_epochs.append(val)
    
    plt.scatter(x_epochs, y_epochs,s=50,c='lightgreen', marker='s', label='score')
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.title('Score per epoch')
    plt.legend()
    plt.grid()
    plt.show()


###############################################################

def RNN_nlp(x):
    
    W = tf.Variable(   tf.random_normal(   [n_hidden, vocab_size]  ))
    b = tf.Variable(   tf.random_normal(   [vocab_size]   ))

    # reshape to [1, n_input], only 1 per batch
    x = tf.reshape(x, [-1, n_input])
    
    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x,n_input,1)
    
    rnn_cell = rnn.BasicLSTMCell(n_hidden)   ## 1 layer LSTM
    #rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])  ## 2 layer LSTM
    
    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    
    # there are n_input outputs but
    # we only want the last output
    rnn_output = tf.matmul(outputs[-1], W) + b
    return rnn_output


#########################################################################

def loss_deep_rnn(output, y_tf):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y_tf)
    loss = tf.reduce_mean(xentropy) 
    return loss


###########################################################

def training(cost):
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cost)
    return train_op

###########################################################

def evaluate(output, y_tf):
    correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y_tf,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy


###########################################################

x_tf = tf.placeholder("float", [None, n_input, 1])   ## the 1 could be vocab_size
y_tf = tf.placeholder("float", [None, vocab_size])

###############################################################
         
output = RNN_nlp(x_tf)
cost = loss_deep_rnn(output, y_tf)
train_op = training(cost)
eval_op = evaluate(output, y_tf)

##################################################################
## predicted word id

output_no_onehot  =   tf.argmax(output, 1)

##################################################################
# Initialize and run

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

###########################################################################################

step = 0

###########################################################################################
## use offset to slide the window accross the document
## every iteration offset is incremented - it is the index that
## tracks the current position in the document

offset = random.randint(0, n_input+1)  ## pick current index on story
end_offset = n_input + 1

while step < n_epochs:
    if offset > (  len(training_data) -  end_offset  ):    ## training_data is all the words in the story
        offset = random.randint(0, n_input + 1)    ## reset to beginning of document
    
    symbols_in_keys = [ [dictionary[ str(training_data[i])]] for i in range(offset, offset+n_input) ]
    symbols_in_keys = np.reshape(  np.array(symbols_in_keys), [-1, n_input, 1]   )
        
    symbols_out_onehot = np.zeros(  [vocab_size], dtype=float   )
    symbols_out_onehot[dictionary[ str( training_data[ offset+n_input ])]] = 1.0  ## e.g. vector[37] = 1.0
    symbols_out_onehot = np.reshape(   symbols_out_onehot,  [1,-1]   )
        
    _, acc, loss, pred_onehot, pred_word = sess.run([train_op, eval_op, cost, output, output_no_onehot],
                                                feed_dict={x_tf: symbols_in_keys, y_tf: symbols_out_onehot})
        

    test_sequence = [training_data[i] for i in range(offset, offset + n_input)]
    test_actual_after_sequence = training_data[offset + n_input]  ## the actual word after a sequence of 3
    test_pred_after_sequence = reverse_dictionary[int(pred_word)]
    print("%s ... %s - [%s] vs [%s]" % (step, test_sequence,test_actual_after_sequence,test_pred_after_sequence) )
    
    step = step + 1
    offset = offset + (n_input+1)

###############################################################################
## this is after training - this generates a short story using
## the trained rnn


while True:
    try: 
        prompt = "%s words: " % n_input
        print(prompt)
        sentence = raw_input()
        print(sentence)
        sentence = sentence.replace("\n", "")
        sentence = sentence.strip()
        words = sentence.split(' ')
        print(words)
        try:
            symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
            for i in range(64):
                keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
                predicted_word_int = sess.run(output_no_onehot, feed_dict={x_tf: keys})
                sentence = "%s %s" % (   sentence, reverse_dictionary[   int(predicted_word_int)   ]   )
                symbols_in_keys = symbols_in_keys[1:]
                symbols_in_keys.append(predicted_word_int)
            print(sentence)
        except:
            print("Words not in dictionary")
    except:
        print("error with your input")


#######################################################################################

print('<<<<<<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>>>>>>>')
