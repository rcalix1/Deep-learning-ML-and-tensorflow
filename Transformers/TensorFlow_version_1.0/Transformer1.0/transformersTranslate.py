##!/usr/bin/env python
## A very simple Transformer based Translator using the English to portuguese dataset
## implemented on the Tensorflow low level API version < 2.0     (~ TF 1.10)
## Followed the paper: Attention Is All You Need by Vaswani et al.
## To do: upgrade to Tensorflow 2.0
## Ricardo A. Calix, Ph.D.
## www.rcalix.com
## rcalix.ai@gmail.com
## Last update: August, 2020
## Deep Learning code - Transformers implementation on static computational graph
## Book:
## Deep Learning Algorithms: transformers, gans, autoencoders, cnns, and more
## By Ricardo Calix
######################################################################################

import sklearn
import tensorflow as tf
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from numpy import genfromtxt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
import pandas as pd
import pickle
import collections
#import matplotlib.pyplot as plt

#######################################################################################
## some settings

import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf) #print all values in numpy array


######################################################################################
## parameters


batch_size = 16

d_model = 512     ## hidden dimension of encoder/decoder
d_ff = 2048       ## hidden dimension of feedforward layer

dropout_rate = 0.3
smoothing = 0.1    ## label smoothing rate
learning_rate = 0.0001   ## 0.0003

MAX_LENGTH = 40

START_TOKEN  = "<sos>"
END_TOKEN    = "<eos>"
UNK_TOKEN    = "<unk>"

n_epochs = 2  ##20000

VOCAB_SIZE_EN = 0
VOCAB_SIZE_PT = 0

###########################################################################################
###########################################################################################
###########################################################################################
## data wrangling
###########################################################################################
###########################################################################################
###########################################################################################


def load_dictionary(file_name):
    with open(file_name, 'rb') as handle:
        dict = pickle.loads(   handle.read()  )
    return dict
    
##########################################################################
## Includes <eos> and <sos> tokens

def build_dataset(words):
    START_TOKEN  = "<sos>"
    END_TOKEN    = "<eos>"
    UNK_TOKEN    = "<unk>"
    count = collections.Counter(words).most_common(12000)
    dictionary = dict()
    for word, _ in count:
        ## add + 1 so that 0 is not used as an index to avoid padding conflict
        dictionary[word] = len(dictionary) + 1           ## + 1
    size_vocab = len(dictionary)
    dictionary[START_TOKEN] = size_vocab
    dictionary[END_TOKEN]   = size_vocab + 1
    dictionary[UNK_TOKEN]   = size_vocab + 2
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary
    
###############################################################################################

def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    sentence = sentence.strip()
    return sentence
    
###############################################################################################
## using regular tokenization. Consider byte pair encoding
## byte pair encoding example instead of 1 token, you do  "walk" and "ing", so 2 tokens
## byte-pair encoding is used to tokenize a language, which, like the WordPiece encoding
## breaks words up into tokens that are slightly larger than single characters but less than entire words


def get_tokens(sentence_list):
    tokens_list = []
    for sentence in sentence_list:
        tokens = word_tokenize(sentence)
        for word in tokens:
            tokens_list.append(word)
    tokens_list = np.array(tokens_list)
    return tokens_list
    
###############################################################################################

def encode(sentence, dictionary):
    ids_list = []
    tokens = word_tokenize(sentence)
    for word in tokens:
        if word in dictionary.keys():
            ids_list.append(  dictionary[word]  )
    return ids_list
    
###############################################################################################

def decode(list_ids, reverse_dictionary):
    words_list = []
    for id in list_ids:
        if id in reverse_dictionary.keys():
            words_list.append(  reverse_dictionary[id]  )
    return words_list

###############################################################################################
## this returns 2 lists of english and portuguese sentences that are aligned by index

def get_en_and_pt_sentences(train_dict):
    en_list, pt_list = [], []
    for key, val in train_dict.items():
        print(key)
        print(val)
        en_list.append(      val['en']        )
        pt_list.append(      val['pt']        )
    return en_list, pt_list

###############################################################################################
## Read in the data of english and portuguese sentences

train_dict   = load_dictionary("data/en_pt_train_dictionary.txt")
validation   = load_dictionary("data/en_pt_val_dictionary.txt")

###############################################################################################

english_sentence_list, portuguese_sentence_list = get_en_and_pt_sentences(train_dict)

###############################################################################################

print("")
print("")
print("creating the dictionaries takes a while ... ")

en_tokens = get_tokens(english_sentence_list)
pt_tokens = get_tokens(portuguese_sentence_list)

###############################################################################################
## when 2 languages, you have 2 separate tokenizers.

en_dictionary, en_reverse_dictionary = build_dataset(en_tokens)
pt_dictionary, pt_reverse_dictionary = build_dataset(pt_tokens)

VOCAB_SIZE_EN = len(en_dictionary)
VOCAB_SIZE_PT = len(pt_dictionary)

print("vocab size english ", VOCAB_SIZE_EN)
print("vocab size portuguese ", VOCAB_SIZE_PT)

###############################################################################################

english_sentence_ids_list    = []
portuguese_sentence_ids_list = []

for i in range(   len(english_sentence_list)    ):
    en_sentence = english_sentence_list[i]
    pt_sentence = portuguese_sentence_list[i]

    en_sentence_ids = encode(en_sentence, en_dictionary)
    pt_sentence_ids = encode(pt_sentence, pt_dictionary)
    
    en_sentence_ids = np.array(en_sentence_ids)
    pt_sentence_ids = np.array(pt_sentence_ids)
    
    en_START_TOKEN_id = en_dictionary['<sos>']
    en_END_TOKEN_id   = en_dictionary['<eos>']
    
    pt_START_TOKEN_id = pt_dictionary['<sos>']
    pt_END_TOKEN_id   = pt_dictionary['<eos>']
    
    en_sentence_ids = np.concatenate(   [   [en_START_TOKEN_id],  en_sentence_ids,  [en_END_TOKEN_id]   ]    )
    pt_sentence_ids = np.concatenate(   [   [pt_START_TOKEN_id],  pt_sentence_ids,  [pt_END_TOKEN_id]   ]    )

    if len(en_sentence_ids) <= MAX_LENGTH and len( pt_sentence_ids) <= MAX_LENGTH:
        english_sentence_ids_list.append(    en_sentence_ids)
        portuguese_sentence_ids_list.append( pt_sentence_ids)
        
###############################################################################################
## to do: do this without keras
## padding='post'    or    padding='pre'
## padding happens before shifting (i.e. [:, :-1] and [:, 1:])

## shifting will remove 1 from target so added 1 to target
## I think en and pt can have different max lengths but I just want
## everything to be sequence = 40

en_MAX_LENGTH = MAX_LENGTH
pt_MAX_LENGTH = MAX_LENGTH + 1

english_sentence_ids_list    = tf.keras.preprocessing.sequence.pad_sequences(
                                      english_sentence_ids_list, maxlen=en_MAX_LENGTH, padding='post')
                                      
portuguese_sentence_ids_list = tf.keras.preprocessing.sequence.pad_sequences(
                                      portuguese_sentence_ids_list, maxlen=pt_MAX_LENGTH, padding='post')
                                    

###############################################################################################
## to view data before or after padding
## to view only, comment out otherwise

'''

for i in range(   len(english_sentence_ids_list)    ):
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print(english_sentence_ids_list[i])
    print(portuguese_sentence_ids_list[i])
    ## input()

'''

##################################################################################################
## data looks like this with padding=post

'''

    en
    
    [12110   203     4  3947    29     2   168     2     4    27    68  4333
         8  3622  2943  1012     1 12111     0     0     0     0     0     0
         0     0     0     0     0     0     0     0     0     0     0     0
         0     0     0     0]
         
    pt
         
    [12210    13     4  3947    29     2     5    32    36    16  1145     4
        58    34  7905    58    25    28   354  2482     3    17    27    28
      4395     9  2886     7 12211     0     0     0     0     0     0     0
         0     0     0     0]
         
'''

##################################################################################################
## data looks like this without padding

'''

    en
    
    [12110    13     4  3947    29     2     5    32    36    16  1145     4
        58    34  7905    58    25    28   354  2482     3    17    27    28
      4395     9  2886     7 12111    ]
         
    pt
         
    [12210    62   585   132   202  4395 11969     3    43    18    27   107
      7042    15    10   814 11717     4  4053    89  2960     2   157   119
         1 12211     ]

'''


###########################################################################################
###########################################################################################
###########################################################################################
## Transformer
###########################################################################################
###########################################################################################
###########################################################################################

def label_smoothing(y):
    ## do this
    return y
   
#######################################################################################

def variable_dropout():
    ## variable droput
    return "need to do this"
    
#######################################################################################

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

    
#######################################################################################


def positional_encoding(embeddings, dropout):           ## embeddings is [N, 40, 512]

    ## limit = 5
    ## tf.range(5)  # [0, 1, 2, 3, 4]
    
    
    ## tf.newaxis changes i from [512,]   to [1, 512]
    #i_vec   =  tf.range(512, dtype=tf.float32)[tf.newaxis, :]
    i_vec   =  np.arange(512.0)[np.newaxis, :]
    
    ## tf.newaxis changes pos from [40,]   to [40, 1]
    ##pos = tf.range(40, dtype=tf.float32)[:, tf.newaxis]
    pos   = np.arange(40.0)[:, np.newaxis]
    
    ##angle_rates = 1 / tf.pow  (10000,     (2 * (i_vec // 2)) / tf.cast(512, tf.float32)      )
    angle_rates   = 1 / np.power(10000,     (2 * (i_vec // 2)) / 512.0      )
    
    ## multiply 2 vectors to get a matrix of size [40, 512]
    angle_rads = pos * angle_rates

    ###################################################################
    ## this assignment operation cannot be done in tensorflow
    ## have to do this in numpy because tensors are not assignable
    
    angle_rads[:, 0::2]    = np.sin(angle_rads[:, 0::2])       ## even index
    angle_rads[:, 1::2]    = np.cos(angle_rads[:, 1::2])       ## odd index

    ## angle_rads  [40, 512]
    ####################################################################
  
    angle_rads = angle_rads[np.newaxis, ...]   ## (1, 40, 512)   for broadcasting

    pos_encoding = angle_rads
    ## pos_encoding = tf.cast(angle_rads, tf.float32)
    
    ####################################################################
    ## embeddings          +           pos_encoding            ## broadcasting
    ## [N, 40, 512]        +           [1, 40, 412]    =   [N, 40, 512]
    ## for broadcasting
    
    ## tensorflow allows you to add numpy array to tensor
    emb_pos = embeddings + pos_encoding
    
    ## dropout is applied to the combination of pos_encoding and embeddings
    emb_pos = tf.nn.dropout(emb_pos, dropout)
    
    return emb_pos     ## this is [N, 40, 512]

  
#######################################################################################
## an intuitive version of positional encoding using for loops
## not used here but left in for reference
'''

def pos_intuitive(x, d_model=512, max_seq_len = 40):
    # create constant 'pe' matrix with values dependant on pos and i
    
    pe = np.zeros(max_seq_len, d_model)
    for pos in range(max_seq_len):
        for i in range(0, 512, 2):
            pe[pos, i]     = math.sin(pos / (10000 ** ((2 *  i)/512)))
            pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/512)))
                 
    pe = pe.unsqueeze(0)

    # make embeddings relatively larger
    x = x * tf.sqrt(d_model)
    
    seq_len = x.size(1)
    x = x + Variable(   pe[:,:seq_len]   )
    return x

'''

#######################################################################################
## input  =  [N, 40, 512]

def dec_final_linear_layer(input):
    #w_h1 = tf.get_variable(   tf.random_normal(  [512, VOCAB_SIZE_PT]  )   )
    w_h1 = tf.Variable(         xavier_init(  [batch_size, 512, VOCAB_SIZE_PT]  )   )
    b_h1 = tf.Variable(   tf.random_normal(   [batch_size, 40, VOCAB_SIZE_PT]     )    )
    
    h1_mul = tf.matmul(  input ,  w_h1   )
    h1 = tf.add( h1_mul,  b_h1  )
    
    softmax_h1 = tf.nn.softmax(  h1 , axis=-1  ) ## [N, 40, vocabulary_size]
    dec_out_one_hot = softmax_h1        ## [N, 40, vocabulary_size]
    
    #################

    dec_out_ids = tf.argmax( softmax_h1 , axis=-1)    ## is this now [N, 40] or
    dec_out_ids = tf.cast(dec_out_ids, tf.int32)      ## [N, 40 , 1 ]
    
    #################
    
    ## you could return
    ##   dec_out_ids        or          dec_out_one_hot
    ##     [N, 40]                   [N, 40, vocabulary_size]
    ## because of the loss function used (sparse_cross_entropy)
    ## dec_out_one_hot   seems to be the correct one
          
    return dec_out_one_hot         ## [N, 40, vocabulary_size]


#######################################################################################
## Feed Forward increases the non-linearity and changes representation of data
## for better generalization over RELu function and dropout

##      input   [N, 40, 512]
def fully_connected_layer(input, dropout):
    ## w_h1 = tf.get_variable(   tf.random_normal(  [512, 2048]  )    )
    w_h1 = tf.Variable(         xavier_init(  [batch_size, 512, 2048]  )    )
    b_h1 = tf.Variable(   tf.random_normal(   [batch_size, 40, 2048]     )    )
    
    h1_mul = tf.matmul(  input ,  w_h1   )
    h1 = tf.add( h1_mul,  b_h1  )
    
    h1_relu = tf.nn.relu(h1)
    h1_drop = tf.nn.dropout(h1_relu, dropout)
    
    #w_h2 = tf.get_variable(   tf.random_normal(   [2048, 512]  )    )
    w_h2 = tf.Variable(         xavier_init(   [batch_size, 2048, 512]     )    )
    b_h2 = tf.Variable(   tf.random_normal(   [batch_size, 40, 512]        )    )
    
    ## h1_drop = [N, 40, 2048]
    h2_mul = tf.matmul(  h1_drop ,  w_h2   )      ## [N, 40, 512]
    h2 = tf.add( h2_mul,  b_h2  )
    
    return h2         ## [N, 40, 512]

#######################################################################################
## not used but leaving here as reference

'''

def layer(input, weight_shape, bias_shape):
    W = tf.get_variable(  tf.random_normal(weight_shape)  )
    b = tf.get_variable(  tf.random_normal(bias_shape)    )
    mapping = tf.matmul(input, W)
    result = tf.add( mapping ,  b )
    return result
    
'''

#######################################################################################
## input   [N, 40, 512]
## encoder_output      [N, 40, 512]
## mask         [N, 40]

def encoder_decoder_attention(input, encoder_output, mask, dropout):

    ## The Query is what determines the decoder output sequence length, therefore we
    ## obtain a sequence of the correct length (i.e. target sequence length)
    ## Wq = tf.get_variable( tf.random_normal(  [512, 64]  )    )
    Wq = tf.Variable(         xavier_init(  [batch_size, 512, 64]  )    )
    bq = tf.Variable( tf.random_normal(  [batch_size, 40, 64]  )  )
    Q = tf.matmul(input, Wq)  +  bq     # Nx40x64   ## from decoder_layer below in decoder
   
    ## Wk = tf.get_variable( tf.random_normal( [512, 64]  )    )
    Wk = tf.Variable(         xavier_init( [batch_size, 512, 64]  )    )
    bk = tf.Variable( tf.random_normal(  [batch_size, 40, 64]  )  )
    K = tf.matmul(encoder_output, Wk)  +  bk    ## from encoder output  [N, 40, 64]
     
    ## Wv = tf.get_variable( tf.random_normal(  [512, 64]  )    )
    Wv = tf.Variable(         xavier_init(  [batch_size, 512, 64]  )    )
    bv = tf.Variable( tf.random_normal(  [batch_size, 40, 64]  )  )
    V = tf.matmul(encoder_output, Wv)  +  bv       ## from encoder output  [N, 40, 64]
     
    
    ## this is "the beef" of transformers
    ## calc a score of word_i importance to all other words
    ## scores_matrix = tf.matmul(a, b, transpose_b=True)    ## transpose b if true
    ## Q = [N, 40, 64],    K=[N, 40, 64],    transpose(K) = [N, 64, 40]
    ## Q  *   transpose(K) =     [N, 40, 64] * [N, 64, 40]    =   [N, 40, 40]
    
    scores_matrix = tf.matmul(Q, K, transpose_b=True)    ### [N, 40, 40]
    scores_matrix = scores_matrix/(    tf.cast(  tf.sqrt(64.0), tf.float32    )          )
    
    ###################################################################################
    ## mask look ahead attention happens before the softmax
    ## masking done to the dot_product matrix only
    ## The mask is multiplied with -1e9 (close to negative infinity). This is done because
    ## the mask is summed with the scaled matrix multiplication of Q and K and is applied
    ## immediately before a softmax. The goal is to zero out these cells, and large negative
    ## inputs to softmax are near zero in the output.
    ## mask = mask.unsqueeze(1) -- remove dimensions with value 1
                  
    ## mask     [N, 40]
    
    ##               [N, 40, 40]  +     [N, 1, 40]     ## for broadcast
    scores_matrix = scores_matrix + (mask[:, tf.newaxis, :] * -1e9)
      
    ####################################################################################

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1. ## axis -1 is for last dimension in this tensor
    a1 = tf.nn.softmax(scores_matrix, axis=-1)  # (N, seq_len_q, seq_len_k)
 
        
    a1 = tf.nn.dropout(a1, dropout)
    
    # (N, seq_len_q, depth_v)  ## scores_matrix * V
    a2 = tf.matmul(a1, V)          ##   [N, 40, 40]  *  [N, 40, 64]   = [N, 40, 64]
    
    return a2   ## [N, 40, 64]

#######################################################################################
## x   [N, 40, 512]
## look_ahead_mask     [N, 40, 40]

def Dec_MultiHeadAttention(x, look_ahead_mask, dropout):

    ## Wq = tf.get_variable( tf.random_normal(  [512, 64]  )    )
    Wq = tf.Variable(         xavier_init(  [batch_size, 512, 64]  )    )
    bq = tf.Variable( tf.random_normal(  [batch_size, 40, 64]  )  )
    Q = tf.matmul(x, Wq)  + bq     # Nx40x64
    
    ## Wk = tf.get_variable( tf.random_normal(  [512, 64]  )    )
    Wk = tf.Variable(         xavier_init(  [batch_size, 512, 64]  )    )
    bk = tf.Variable( tf.random_normal(  [batch_size, 40, 64]  )  )
    K = tf.matmul(x, Wk)  + bk      # Nx40x64
    
    ## Wv = tf.get_variable( tf.random_normal(  [512, 64]  )    )
    Wv = tf.Variable(         xavier_init(  [batch_size, 512, 64]  )    )
    bv = tf.Variable( tf.random_normal(  [batch_size, 40, 64]  )  )
    V = tf.matmul(x, Wv) + bv        # Nx40x64
    
    ## calc a score of word_i importance to all other words
    scores_matrix = tf.matmul(  Q, K, transpose_b=True)    ### (N, 40, 40)
    scores_matrix = scores_matrix/(tf.sqrt(64.0))     ## [N, 40, 40]
    
    ################################################################################
    ## mask look ahead attention happens before the softmax
    ## masking done to the dot_product matrix only
    ## The mask is multiplied with -1e9 (close to negative infinity). This is done because
    ## the mask is summed with the scaled matrix multiplication of Q and K and is applied
    ## immediately before a softmax. The goal is to zero out these cells, and large negative
    ## inputs to softmax are near zero in the output.
    ## mask = mask.unsqueeze(1) -- removes dimensions of 1
  
    ## look_ahead_mask     [N, 40, 40]
    ##              [N, 40, 40]   +    [N, 40, 40]
    
    scores_matrix = scores_matrix + (look_ahead_mask * -1e9)     ## [N, 40, 40]
    
    ################################################################################
    ## >>> a = tf.constant([0.6, 0.2, 0.3, 0.4, 0, 0, 0, 0, 0, 0])
    ## >>> tf.nn.softmax(a)
      
    ## <tf.Tensor: shape=(10,), dtype=float32, numpy=
    ## array([0.15330984, 0.10276665, 0.11357471, 0.12551947, 0.08413821,
    ##       0.08413821, 0.08413821, 0.08413821, 0.08413821, 0.08413821],
    ##      dtype=float32)>
      
    ## >>> b = tf.constant([0.6, 0.2, 0.3, 0.4, -1e9, -1e9, -1e9, -1e9, -1e9, -1e9])
    ## >>> tf.nn.softmax(b)
      
    ## <tf.Tensor: shape=(10,), dtype=float32, numpy=
    ## array([0.3096101 , 0.20753784, 0.22936477, 0.25348732, 0.        ,
    ##        0.        , 0.        , 0.        , 0.        , 0.        ],
    ##      dtype=float32)>
    
    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1. ## axis -1 is for last dimension in this tensor
    
    a1 = tf.nn.softmax(scores_matrix, axis=-1)  # (N, seq_len_q, seq_len_k)
    
    a1 = tf.nn.dropout(a1, dropout)
    
    a2 = tf.matmul(a1, V)            ##  [N, 40, 40]   *   [N, 40, 64]
    
    return a2         ## [N, 40, 64]
    
#######################################################################################

## x   ## [N, 40, 512]

def layer_norm(x):
    eps = 1e-6
    size = 512
    # create two learnable parameters to calibrate normalisation
    alpha = np.ones(size)
    bias  = np.zeros(size)
    mean, var = tf.nn.moments(x, axes=[0, 1, 2])
    standard_dev = tf.sqrt(var)
    norm = alpha * (x - mean) / (standard_dev + eps) + bias
    return norm      ##  [N, 40, 512]

#######################################################################################
## encoder_output = [N, 40, 512]
## input_dec_layer  = [N, 40, 512]
## enc_out_padding_mask       [N, 40]
## dec_look_ahead_comb_mask     [N, 40, 40]

def decoder_layer(input_dec_layer, encoder_output, enc_out_padding_mask, dec_look_ahead_comb_mask, dropout):

    ####################################################################
    ## Masked multi-head attention

    with tf.variable_scope("Dec_MultiHead_Attention_1"):
        z1 = Dec_MultiHeadAttention(input_dec_layer, dec_look_ahead_comb_mask, dropout)
    with tf.variable_scope("Dec_MultiHead_Attention_2"):
        z2 = Dec_MultiHeadAttention(input_dec_layer, dec_look_ahead_comb_mask, dropout)
    with tf.variable_scope("Dec_MultiHead_Attention_3"):
        z3 = Dec_MultiHeadAttention(input_dec_layer, dec_look_ahead_comb_mask, dropout)
    with tf.variable_scope("Dec_MultiHead_Attention_4"):
        z4 = Dec_MultiHeadAttention(input_dec_layer, dec_look_ahead_comb_mask, dropout)
    with tf.variable_scope("Dec_MultiHead_Attention_5"):
        z5 = Dec_MultiHeadAttention(input_dec_layer, dec_look_ahead_comb_mask, dropout)
    with tf.variable_scope("Dec_MultiHead_Attention_6"):
        z6 = Dec_MultiHeadAttention(input_dec_layer, dec_look_ahead_comb_mask, dropout)
    with tf.variable_scope("Dec_MultiHead_Attention_7"):
        z7 = Dec_MultiHeadAttention(input_dec_layer, dec_look_ahead_comb_mask, dropout)
    with tf.variable_scope("Dec_MultiHead_Attention_8"):
        z8 = Dec_MultiHeadAttention(input_dec_layer, dec_look_ahead_comb_mask, dropout)
         
    z_concat = tf.concat([z1, z2 ,z3, z4, z5, z6, z7, z8], -1)      ## [N, 40, 512]
    
    ## W0 = tf.get_variable( tf.random_normal(  [8*64, 512]  )    )
    W0 = tf.Variable(         xavier_init(  [batch_size, 8*64, 512]  )    )
    b0 = tf.Variable( tf.random_normal(  [batch_size, 40, 512]  )  )
    z1 = tf.matmul(z_concat, W0) + b0
    residual1 = layer_norm(input_dec_layer + z1)
    
    ####################################################################
    ## multi head attention with encoder output
    ## this is the decoder_encoder_attention segment
    
    with tf.variable_scope("En_De_Att_MultiHead_Attention_1"):
        dea1 = encoder_decoder_attention(residual1, encoder_output, enc_out_padding_mask, dropout)
    with tf.variable_scope("En_De_Att_MultiHead_Attention_2"):
        dea2 = encoder_decoder_attention(residual1, encoder_output, enc_out_padding_mask, dropout)
    with tf.variable_scope("En_De_Att_MultiHead_Attention_3"):
        dea3 = encoder_decoder_attention(residual1, encoder_output, enc_out_padding_mask, dropout)
    with tf.variable_scope("En_De_Att_MultiHead_Attention_4"):
        dea4 = encoder_decoder_attention(residual1, encoder_output, enc_out_padding_mask, dropout)
    with tf.variable_scope("En_De_Att_MultiHead_Attention_5"):
        dea5 = encoder_decoder_attention(residual1, encoder_output, enc_out_padding_mask, dropout)
    with tf.variable_scope("En_De_Att_MultiHead_Attention_6"):
        dea6 = encoder_decoder_attention(residual1, encoder_output, enc_out_padding_mask, dropout)
    with tf.variable_scope("En_De_Att_MultiHead_Attention_7"):
        dea7 = encoder_decoder_attention(residual1, encoder_output, enc_out_padding_mask, dropout)
    with tf.variable_scope("En_De_Att_MultiHead_Attention_8"):
        dea8 = encoder_decoder_attention(residual1, encoder_output, enc_out_padding_mask, dropout)
         
    dea_concat = tf.concat([dea1, dea2 ,dea3, dea4, dea5, dea6, dea7, dea8], -1)
    
    ## W0_dea = tf.get_variable( tf.random_normal(  [8*64, 512]  )    )
    W0_dea = tf.Variable(         xavier_init(  [batch_size, 8*64, 512]  )    )
    b0_dea = tf.Variable( tf.random_normal(  [batch_size, 40, 512]  )  )
    z2 = tf.matmul(dea_concat, W0_dea)  +   b0_dea
    residual2 = layer_norm(residual1 + z2)                 ## [N, 40, 512]

    ####################################################################
    ## Feed Forward segment
    
    h1 = fully_connected_layer(residual2, dropout)
    residual3 = layer_norm(residual2 + h1)
    
    return residual3
    
#######################################################################################
##   x is [N, 40, 512]
## Encoder padding mask   [N, 40]
## if [1200  45   23  1201   0    0     0]
## then [ 0   0    0    0    1    1     1]


def Enc_MultiHeadAttention(x, enc_padding_mask, dropout):         ###   [N, 40, 512]
    
    ## Wq = tf.get_variable( tf.random_normal(  [512, 64]  )    )

    Wq = tf.Variable(         xavier_init(  [batch_size, 512, 64]  )    )
    bq = tf.Variable( tf.random_normal(  [batch_size, 40, 64]  )  )
    Q = tf.matmul(x, Wq) + bq     ## Nx40x64
    print("Q shape  Nx40x64   ??? ", Q.shape)
    
    ## Wk = tf.Variable(         xavier_init(  [512, 64]  )    )
    ## bk = tf.Variable( tf.random_normal(  [64]  )  )
    ## K = tf.matmul(x, Wk) + bk      ## Nx40x64
    Wk = tf.Variable(         xavier_init(  [batch_size, 512, 64]  )    )
    bk = tf.Variable( tf.random_normal(  [batch_size, 40, 64]  )  )
    K = tf.matmul(x, Wk) + bk      ## Nx40x64
    
    Wv = tf.Variable(         xavier_init(  [batch_size, 512, 64]  )    )
    bv = tf.Variable( tf.random_normal(  [ batch_size, 40, 64]  )  )
    V = tf.matmul(x, Wv) + bv      ## Nx40x64
    
    ## calc a score of word_i importance to all other words
    scores_matrix = tf.matmul(  Q, K, transpose_b=True)    ### (N, 40, 40)
    scores_matrix = scores_matrix/(tf.sqrt(64.0))     ## depth=64 ## this is "the beef" of transformers
    
    ## scores matrix is [N, 40, 40]
    
    ########################################################################
    ## we don’t really want the network to pay attention to the padding
    ## so we’re going to mask it
    
    ## Encoder padding mask   [N, 40]
    
    ##              [N, 40, 40]   +    [N, 1, 40]              ## for broadcasting broadcast
    scores_matrix = scores_matrix + (enc_padding_mask[:, tf.newaxis, :] * -1e9)
    
    ########################################################################

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1. ## axis -1 is for last dimension in this tensor
    a1 = tf.nn.softmax(scores_matrix, axis = -1)  # (N, seq_len_q, seq_len_k)
    
    a1 = tf.nn.dropout(a1, dropout)
    
    a2 = tf.matmul(a1, V)   ##   [N, 40, 40]  *   [N, 40, 64]   ???   ## problem (2) dim ?
    
    return a2   ## [N, 40, 64]

#######################################################################################
##   x is [N, 40, 512]
## Encoder padding mask
## if [1200  45   23  1201   0    0     0]
## then [ 0   0    0    0    1    1     1]


def encoder_layer(x, enc_padding_mask, dropout):

    ##################################################################
    ## MultiHead Attention segment

    with tf.variable_scope("Enc_MultiHead_Attention_1"):
        z1 = Enc_MultiHeadAttention(x, enc_padding_mask, dropout)
    with tf.variable_scope("Enc_MultiHead_Attention_2"):
        z2 = Enc_MultiHeadAttention(x, enc_padding_mask, dropout)
    with tf.variable_scope("Enc_MultiHead_Attention_3"):
        z3 = Enc_MultiHeadAttention(x, enc_padding_mask, dropout)
    with tf.variable_scope("Enc_MultiHead_Attention_4"):
        z4 = Enc_MultiHeadAttention(x, enc_padding_mask, dropout)
    with tf.variable_scope("Enc_MultiHead_Attention_5"):
        z5 = Enc_MultiHeadAttention(x, enc_padding_mask, dropout)
    with tf.variable_scope("Enc_MultiHead_Attention_6"):
        z6 = Enc_MultiHeadAttention(x, enc_padding_mask, dropout)
    with tf.variable_scope("Enc_MultiHead_Attention_7"):
        z7 = Enc_MultiHeadAttention(x, enc_padding_mask, dropout)
    with tf.variable_scope("Enc_MultiHead_Attention_8"):
        z8 = Enc_MultiHeadAttention(x, enc_padding_mask, dropout)
        
    ## [N, 40, 64]  after concat it is [N, 40, 64*8]  = [N, 40, 512]
    z_concat = tf.concat([z1, z2 ,z3, z4, z5, z6, z7, z8], -1)  ## [N, 40, 512]
    
    ## W0 = tf.get_variable(  tf.random_normal(  [8*64, 512]  )    )

    W0 = tf.Variable(          xavier_init(  [batch_size, 8*64, 512]  )    )
    b  = tf.Variable(tf.random_normal([batch_size, 40, 512]))
    z  = tf.add(    tf.matmul(z_concat, W0) ,   b       )  ## [N, 40, 512]

    residual1 = layer_norm(x + z)  ## this is [N, 40, 512]
    
    ####################################################################################
    ## Feed Forward segment
    
    h1 = fully_connected_layer(residual1, dropout)  ## [N, 40, 512]
    residual2 = layer_norm(residual1 + h1)      ## [N, 40, 512]
    
    return residual2     #  [N, 40, 512]

#########################################################################################
## encoder_output = [N, 40, 512]
## embed_pt_pos_dec_in  = [N, 40, 512]


def decoder(encoder_output, embed_pt_pos_dec_in, enc_out_padding_mask, dec_look_ahead_comb_mask, dropout):

    with tf.variable_scope("Decoder_layer_1"):
        h1 = decoder_layer(embed_pt_pos_dec_in, encoder_output, enc_out_padding_mask, dec_look_ahead_comb_mask, dropout)
    with tf.variable_scope("Decoder_layer_2"):
        h2 = decoder_layer(h1, encoder_output, enc_out_padding_mask, dec_look_ahead_comb_mask, dropout)
    with tf.variable_scope("Decoder_layer_3"):
        h3 = decoder_layer(h2, encoder_output, enc_out_padding_mask, dec_look_ahead_comb_mask, dropout)
    with tf.variable_scope("Decoder_layer_4"):
        h4 = decoder_layer(h3, encoder_output, enc_out_padding_mask, dec_look_ahead_comb_mask, dropout)
    with tf.variable_scope("Decoder_layer_5"):
        h5 = decoder_layer(h4, encoder_output, enc_out_padding_mask, dec_look_ahead_comb_mask, dropout)
    with tf.variable_scope("Decoder_layer_6"):
        h6 = decoder_layer(h5, encoder_output, enc_out_padding_mask, dec_look_ahead_comb_mask, dropout)
        
    ## h6  =  [N, 40, 512]
    dec_out_one_hot = dec_final_linear_layer(h6)
    
    return dec_out_one_hot         ## [N, 40, vocabulary_size]
   
###########################################################################################
## embed_en_pos_enc_in is [N, 40, 512]

def encoder(embed_en_pos_enc_in, enc_padding_mask, dropout):

    with tf.variable_scope("Encoder_1"):
        h1 = encoder_layer(embed_en_pos_enc_in, enc_padding_mask, dropout)
    with tf.variable_scope("Encoder_2"):
        h2 = encoder_layer(h1, enc_padding_mask, dropout)
    with tf.variable_scope("Encoder_3"):
        h3 = encoder_layer(h2, enc_padding_mask, dropout)
    with tf.variable_scope("Encoder_4"):
        h4 = encoder_layer(h3, enc_padding_mask, dropout)
    with tf.variable_scope("Encoder_5"):
        h5 = encoder_layer(h4, enc_padding_mask, dropout)
    with tf.variable_scope("Encoder_6"):
        h6 = encoder_layer(h5, enc_padding_mask, dropout)
    return h6    ##  [N, 40, 512]

#######################################################################################
## The look-ahead mask is used to mask the future tokens in a sequence.
## In other words, the mask indicates which entries should not be used.
## This means that to predict the third word, only the first and second word
## will be used. Similarly to predict the fourth word, only the first, second
## and the third word will be used and so on

## x = tf.random.uniform(   (1, 3)   )         ##  [the cat is]
## temp = create_look_ahead_mask(x.shape[1])   ## 3 (seq_len)
## <tf.Tensor: shape=(3, 3), dtype=float32, numpy=
## array([[0., 1., 1.],
##        [0., 0., 1.],
##        [0., 0., 0.]], dtype=float32)>


def create_look_ahead_mask(seq_len):
    ones_tensor = np.ones( (seq_len, seq_len) )        ## 40x40
    mask = np.triu(ones_tensor, k=1)   ## k=1 means above diagonal
    mask = mask.astype('uint8')
    return mask       # (seq_len, seq_len)

#######################################################################################
## if [1200  45   23  1201   0    0     0]
## padding mask is [ 0   0    0    0    1    1     1]
## x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
## create_padding_mask(x)
## <tf.Tensor: shape=(3, 1, 1, 5), dtype=float32, numpy=
## array([[[[0., 0., 1., 1., 0.]]],
##       [[[0., 0., 0., 1., 1.]]],
##       [[[1., 1., 1., 0., 0.]]]], dtype=float32)>


##  source_seq       [N, 40]
   
def create_padding_mask(source_seq):

    ## put 1 where padded, elementwise
    padding_mask = tf.cast(tf.equal(source_seq, 0), dtype=tf.float32)
    
    return padding_mask          ##     [N, 40]
    

#######################################################################################
## create_masks(x_ph_enc_in, y_ph_dec_in)
## x_ph_enc_in       [N, 40]
## y_ph_dec_in       [N, 40]

def create_masks(enc_in, dec_in):
    
    #########################################################################
    ## Encoder_in padding mask - english sentence
    ## if [1200  45   23  1201   0    0     0]
    ## then [ 0   0    0    0    1    1     1]
    enc_in_padding_mask = create_padding_mask(enc_in)
       

    #########################################################################
    # Used in the decoder encoder attention block in the decoder
    # This padding mask is used to mask the encoder_outputs
    # enc_in can be used here for convenience because the enc_out has same
    # dimensions and padding as enc_in ??
    dec_enc_out_padding_mask = create_padding_mask(enc_in)
    
       
    #########################################################################
    ## decoder_in padding mask - portuguese sentence
    ## padding mask for decoder_in
    dec_in_padding_mask = create_padding_mask(dec_in)
    

    ## look ahead mask - in the decoder
    ## It is used to mask future tokens in the dec input received by
    ## the decoder. The shape of [1]  for [N, 40] is 40
    ## attention dot_product matrices are 40x40 for this case
    ## tf.shape(dec_in)[1]
    look_ahead_mask = create_look_ahead_mask(  dec_in.shape[1]  )   ## [40, 40]
       
       
    ## returns the maximum elementwise
    ##  tf.maximum    supports broadcast
    ##  dec_in_padding_mask     [N, 40]
    ##   look_ahead_mask         [40, 40]
    ##   need to braodcast so add new dimensions (see book for another example of how this works)
    ##                                                 [N, 1, 40]                        [1, 40, 40]
    dec_combined_mask = tf.maximum(dec_in_padding_mask[:, tf.newaxis, :], look_ahead_mask[tf.newaxis, ...])
   
   
    ##         [N, 40]               [N, 40, 40]            [N, 40]
    return enc_in_padding_mask, dec_combined_mask, dec_enc_out_padding_mask
       

#######################################################################################
## define the Transformer architecture here
## x_ph_enc_in       [N, 40]
## y_ph_dec_in       [N, 40]

def inference_transformer(x_ph_enc_in, y_ph_dec_in, dropout):

    ###########################################################
    ## masks
    
    enc_padding_mask, dec_look_ahead_comb_mask, enc_out_padding_mask = create_masks(x_ph_enc_in, y_ph_dec_in)
    
    ##   [N, 40]              [N, 40, 40]                [N, 40]
    
    ###########################################################
    ## embedding layer from vocab size to 512. Embeddings are initially random
    ## and are learned by backpropagation
    
    ## print("error place", VOCAB_SIZE_EN)
    
    ##embeddings_en = tf.get_variable(  tf.random_uniform(  [VOCAB_SIZE_EN, 512]           )  )
    embeddings_en   = tf.Variable(      tf.random_uniform(  [VOCAB_SIZE_EN, 512], -1.0, 1.0)  )
    embed_en_enc_in      = tf.nn.embedding_lookup(embeddings_en, x_ph_enc_in)
    ##  token embeddings are multiplied by a scaling factor which is square root of depth size
    embed_en_enc_in      = embed_en_enc_in * tf.sqrt(   tf.cast(512, tf.float32)    )
    
    
    ##embeddings_pt = tf.get_variable(  tf.random_uniform(  [VOCAB_SIZE_PT, 512]           )  )
    embeddings_pt   = tf.Variable(      tf.random_uniform(  [VOCAB_SIZE_PT, 512], -1.0, 1.0)  )
    embed_pt_dec_in      = tf.nn.embedding_lookup(embeddings_pt, y_ph_dec_in)
    ##  token embeddings are multiplied by a scaling factor which is square root of depth size
    embed_pt_dec_in      = embed_pt_dec_in * tf.sqrt(   tf.cast(512, tf.float32)    )
    
    ###########################################################
    ## positional encoding
    
    embed_en_pos_enc_in = positional_encoding(  embed_en_enc_in , dropout )    ## [N, 40, 512]
    embed_pt_pos_dec_in = positional_encoding(  embed_pt_dec_in , dropout )    ## [N, 40, 512]
    
    ###########################################################
    ## now enter the transformer network
    ##                         [N, 40, 512]          [N, 40]
    encoder_output = encoder(embed_en_pos_enc_in, enc_padding_mask, dropout)
    ##           [N, 40, 512]        [N, 40, 512]       [N, 40]             [N, 40 , 40]
    y = decoder(encoder_output, embed_pt_pos_dec_in, enc_out_padding_mask, dec_look_ahead_comb_mask, dropout)

    return y       ## [N, 40, vocabulary_size]
    
    
##########################################################################################
## with keras it is like this
## loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
## loss_ = loss_object(real, pred)
## mask = tf.cast(mask, dtype=loss_.dtype)        ## why this data type?

##########################################################################################
## We mask the loss incurred by the padded tokens to 0 so that they do not contribute
## to the mean loss. Ignore padding when calculating the loss function
## need to figure out if there is a better loss for this ??
## y_ph_dec_real is [N, 40]
## y_pred is one_hot_encoded [N, 40, vocab_size] -- which is big
## it seems that
## tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_ph_dec_real, logits=y_)
## takes y_ph_dec_real as [N, 40] and
## y_pred as [N, 40, vocab_size]


def loss(y_pred, y_ph_dec_real):
    
    y_ = label_smoothing(   y_pred   )
    
    ## y_ph_dec_real   [N, 40]
    ## tf.equal(y_ph_dec_real, 0)   -->>    ([0 0 0 1 1 1])
    ## then reverse it with tf.math.logical_not  -->>    ## [1 1 1 0 0 0]
    mask = tf.math.logical_not(   tf.equal(y_ph_dec_real, 0)    )
    mask = tf.cast(mask, tf.float32)        ## [N, 40]
    
    ## labels   [N, 40]
    ## tf.nn.sparse_softmax_cross_entropy_with_logits returns:
    ## Returns: A Tensor of the same shape as labels
    ## and of the same type as logits with the softmax cross entropy loss
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_ph_dec_real, logits=y_)
    
    ## loss_    ## [N, 40]
    ## mask     ## [N, 40]
    
    loss_ = loss_ * mask      ## element wise
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)
    
 
##########################################################################################
'''

def loss2(y, y_ph_dec_real):
    y_ = label_smoothing(   y   )
    xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_, labels=y_ph_dec_real)
    loss = tf.reduce_mean(xentropy)
    return loss

'''

##########################################################################################


def training(cost):
    ## optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-9)
    train_op = optimizer.minimize(cost)
    return train_op


###########################################################################################
## sentences + padding = 40
## tf.float32

x_ph_enc_in    = tf.placeholder(  tf.int32,  [None, 40]  )    ## english enc in
y_ph_dec_in    = tf.placeholder(  tf.int32,  [None, 40]  )    ## portuguese dec in
y_ph_dec_real  = tf.placeholder(  tf.int32,  [None, 40]  )    ## portuguese dec out

############################################################################################

dropout_ph = tf.placeholder(  tf.float32  )               ## dropout

############################################################################################


y = inference_transformer(x_ph_enc_in, y_ph_dec_in, dropout_ph)
cost = loss(y, y_ph_dec_real)
train_op = training(cost)


############################################################################################

## init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

############################################################################################
## changing to shorter more common notation
## these are now numpy arrays with padding

X_en = english_sentence_ids_list             ## [number_of_samples, 40]
X_pt = portuguese_sentence_ids_list          ## [number_of_samples, 41]

print("X_en.shape ", X_en.shape)
print("X_pt.shape ", X_pt.shape)

############################################################################################
#batch size is 64


num_samples = X_en.shape[0]
print(num_samples)
num_batches = int(num_samples/batch_size)

############################################################################################
############################################################################################
############################################################################################
##   MAIN_LOOP() - train
############################################################################################
############################################################################################
############################################################################################


for i in range(n_epochs):
    #for batch_n in range(num_batches):
    for batch_n in range(3):
    
        sta = batch_n * batch_size
        end = sta + batch_size
        
        print("current epoch is ", i)
        print("num batches ", num_batches)
        print("batch n ", batch_n)
        
        ## batches with rows like this of sequence ids for words in the sentence
        ## batch_en      ->   the cat   is  0 0 ->    [12110  12  34 ...  56  12111  0   0   0]
        ## batch_pt      ->   el  gato  es  0 0  ->   [12210  6   54 ...  23   23  12211  0   0   0]
        
        batch_en = X_en[sta:end, : ]                 ## [N, 40]
        batch_pt = X_pt[sta:end, : ]                 ## [N, 41]
      
        ###################################################################################
        ## shift - shifting happens after padding
        ## this was confusing to me at first
        ## The target (batch_pt) is divided into batch_pt_inp and batch_pt_real for the decoder
        ## batch_pt_inp is passed as an input to the decoder. batch_pt_real is that same input
        ## shifted by 1 and only used by the loss function. It is compared to y_pred
        ## At each location i in each sentence in batch_pt_inp, batch_pt_real contains the next token
        ## that should be predicted
        ## The tensorflow.org example is a bit confusing. It should be like this I think
        ## padding = 0
        ## For example, sentence = "SOS A lion in the jungle is sleeping EOS   0     0"
        ## after the shifting
        ## batch_pt      = "SOS    A    lion    in       the     jungle       is       sleeping    EOS    0     0"
        ## batch_pt_inp  = "SOS    A    lion    in       the     jungle       is       sleeping    EOS    0"
        ## batch_pt_real = " A    lion   in     the     jungle     is       sleeping     EOS        0     0"
        ## what matters is that given "jungle", the transformer should predict "is"
        ## given "is" --> "sleeping", and so on


        ## shift batch_pt_real_labes to the left
        ## decoder_in           ->   el    gato
        ## decoder_out_labels   ->  gato    es
     
                                                      ## batch_pt [N, 41]
        batch_pt_inp  = batch_pt[:, :-1]              ## [N, 40]
        batch_pt_real = batch_pt[:, 1:]               ## [N, 40]
        
      
        #######################################################################################
        ## pass the data and train
          
        train_step = sess.run( train_op , feed_dict={  x_ph_enc_in: batch_en,
                                                       y_ph_dec_in: batch_pt_inp,
                                                       y_ph_dec_real: batch_pt_real,
                                                       dropout_ph: dropout_rate
                                                     })
                                                        
       
   


###############################################################################################
###############################################################################################
###############################################################################################
## EVALUATION
## Encode the input sentence using the Portuguese tokenizer (tokenizer_pt). Add
## the start and end token so the input is equivalent to what the model is trained with.
## This is the encoder input. The decoder_input is the start token == tokenizer_en.vocab_size.
## Calculate the padding masks (3 = ?) and the look ahead mask (1 = ?).
## The decoder then outputs the predictions by looking at the encoder_output and its own
## self-attention. Select the last word and calculate the argmax of that. Concatentate the
## predicted word to the decoder input an pass it to the decoder. In this approach, the decoder
## predicts the next word based on the previous words it has predicted.


## sent_en      ->  <sos> the cat   is  <eos> ->    [12110  12  34 ...  56  12111  ]
## sent_pt      ->  <sos>                     ->    [12210 ]

def evaluate(sentence):

    en_sentence_ids = encode(sentence, en_dictionary)
    en_sentence_ids = np.array(en_sentence_ids)

    en_START_TOKEN_id = en_dictionary['<sos>']
    en_END_TOKEN_id   = en_dictionary['<eos>']
    
    pt_START_TOKEN_id = pt_dictionary['<sos>']
    pt_END_TOKEN_id   = pt_dictionary['<eos>']
    
    en_sentence_ids = np.concatenate(   [   [en_START_TOKEN_id],  en_sentence_ids,  [en_END_TOKEN_id]   ]    )
    pt_sentence_ids = np.concatenate(   [   [pt_START_TOKEN_id]  ]    )

    print("evalaution inputs")
    print(en_sentence_ids)
    print(pt_sentence_ids)
    
    ## currently looks like this
    ## [12110, 7, 15, 47, 12111]
    ## [12220]
    
    ##########################################################################
    
    index = 0    ## keep track of the predicted id we want

    for i in range(MAX_LENGTH):
    
        en_sentence_ids_list = []
        pt_sentence_ids_list = []
        
        if len(en_sentence_ids) <= MAX_LENGTH and len( pt_sentence_ids) <= MAX_LENGTH:
            ## this "for" loop is a cheat to have tensor of size batch with same input
            for nn in range(batch_size):
                en_sentence_ids_list.append(    en_sentence_ids   )
                pt_sentence_ids_list.append(    pt_sentence_ids   )
                   
        ## padding - masks will be created later to ignore the padding in the inference func
        ## [batch_size, 40]
        enc_in_pad  = tf.keras.preprocessing.sequence.pad_sequences(
                                                 en_sentence_ids_list, maxlen=MAX_LENGTH, padding='post')
                         
        ## [batch_size, 40]
        dec_in_pad  = tf.keras.preprocessing.sequence.pad_sequences(
                                                 pt_sentence_ids_list, maxlen=MAX_LENGTH, padding='post')

        print(enc_in_pad.shape)
        print(dec_in_pad.shape)
    
        ## enc_in_pad    [batch_size, 40]     ##axis=1 ->> [12110, 7, 15, 47, 12111, 0, 0, ..., 0]
        ## dec_in_pad    [batch_size, 40]     ##axis=1 ->> [12220, 0,  0,  0,     0, 0, 0, ..., 0]
        
        #######################################################################################
        
        ## shifting not needed here in test/prediction
        
        #######################################################################################
        ## pass the data and evaluate
          
        y_pred = sess.run(  y  , feed_dict={x_ph_enc_in: enc_in_pad,
                                            y_ph_dec_in: dec_in_pad,
                                            dropout_ph: dropout_rate })
        
        print(y_pred.shape)
        ## y_pred = [batch_size, 40, pt_vocab_size]
 
 
        ######################################################################################
        ## select the current word from the seq_len dimension
        ## example:
        ## iteration 0  (index=0)
        ## enc_in    [sos  the   cat   eos   0 ]
        ## dec_in    [sos    0     0     0   0 ]
        ## y_pred    [el     *     *     *   * ]
        ##            0
        ## iteration 1  (index=1)
        ## enc_in    [sos  the   cat   eos   0 ]
        ## dec_in    [sos    el     0     0   0 ]
        ## y_pred    [el    gato    *     *   * ]
        ##                   1
        ## iteration 2  (index=2)
        ## enc_in    [sos  the   cat   eos   0 ]
        ## dec_in    [sos   el   gato    0   0 ]
        ## y_pred    [el   gato   eos    *   * ]
        ##                         2
        ########################################################################################
        ## prediction = y_pred[: ,-1:, :]  # [1, 1, pt_vocab_size]
        
        ## all in batch are the same sentence so I just need the first one
        prediction = y_pred[0, index, :]  # [1, 1, pt_vocab_size]
        
        index = index + 1

        ## get id of current predicted word
        predicted_id_tf = tf.cast(tf.argmax(prediction, axis=-1), tf.int32)
        
        predicted_id = sess.run(predicted_id_tf)
        
        print("predicted_id  ", predicted_id)
        
        ## return the result if the predicted_id is equal to the end token
        if predicted_id == pt_END_TOKEN_id:
            break
            
        ## concatentate predicted_id to output then give to decoder as input
        ## dec_in = tf.concat(   [dec_in, predicted_id]   )
        
        pt_sentence_ids = np.concatenate(   [   pt_sentence_ids,  [predicted_id]   ]    )
      
        
        print("sentence being predited")
        print(pt_sentence_ids)
        

    return pt_sentence_ids

    

#####################################################################################
      
def predict(sentence):
    prediction = evaluate(sentence)
    print(prediction.shape)
    prediction_list = prediction.tolist()
    pt_sentence_words = decode(prediction_list, pt_reverse_dictionary)
    print(pt_sentence_words)

#####################################################################################

sentence = "the cat is sleeping"
predict(sentence)

#####################################################################################


print("<<<<<<<<<<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>>>>>>>>>")


