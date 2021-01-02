###################################################################################################
##!/usr/bin/env python
## A very simple Transformer based Translator using the English to spanish dataset europarl
## Runs on Tensorflow 2.0
## Version described in the paper: Attention Is All You Need by Vaswani et al.
## This code modified from original by Tensorflow.org
## Ricardo A. Calix, Ph.D.
## www.rcalix.com
## rcalix.ai@gmail.com
## Last update: January, 2021
## Algorithm described in Book:
## Deep Learning Algorithms: transformers, gans, autoencoders, cnns, rnns, and more
## By Ricardo Calix
## Transformer in TF 2.0

###################################################################################################
## Instructions 
##
## In TF 2.0 conda env, run "pip install -q tfds-nightly"

###################################################################################################

import tensorflow_datasets as tfds
import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt

import nltk
from nltk.tokenize import word_tokenize
from numpy import genfromtxt


import pickle
import collections

###################################################################################################
## some settings

import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf) #print all values in numpy array


###################################################################################################


BUFFER_SIZE = 20000    ## ??

BATCH_SIZE = 256       ## 64
MAX_LENGTH = 40

num_layers = 4         ## ??

d_model = 256          ## 512    ## 128
dff = 1024            ## 2048   ## 512
            
num_heads = 8

input_en_vocab_size  = 0
target_sp_vocab_size = 0


dropout_rate = 0.1
EPOCHS  = 20
MAX_N_WORDS = 36000       ## 12000


###################################################################################################
###################################################################################################
###################################################################################################
##
## Data Wrangling
##
###################################################################################################
###################################################################################################
###################################################################################################

def load_dictionary(file_name):
    with open(file_name, 'rb') as handle:
        dict = pickle.loads(   handle.read()  )
    return dict
    
###############################################################################################
## Includes <eos> and <sos> tokens

def build_dataset(words):
    START_TOKEN  = "<sos>"
    END_TOKEN    = "<eos>"
    UNK_TOKEN    = "<unk>"
    count = collections.Counter(words).most_common(MAX_N_WORDS)
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
    ## tokenizer = RegexpTokenizer(r'\w+')    ## remove this to keep punctuation
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
## this returns 2 lists of english and spanish sentences that are aligned by index

def get_en_and_sp_sentences(train_dict):
    en_list, sp_list = [], []
    for key, val in train_dict.items():
        print(key)
        print(val)
        en_list.append(      val['en']        )
        sp_list.append(      val['sp']        )
    return en_list, sp_list


###############################################################################################
## Read in the data of english and spanish sentences

train_dict   = load_dictionary("data/en_sp_train_dictionary.txt")
test_dict    = load_dictionary("data/en_sp_test_dictionary.txt")


###############################################################################################

## train data

english_sentence_list, spanish_sentence_list = get_en_and_sp_sentences(train_dict)

###############################################################################################

## test data

test_english_sentence_list, test_spanish_sentence_list = get_en_and_sp_sentences(test_dict)

###############################################################################################

print("")
print("")
print("creating the dictionaries takes a while ... ")

en_tokens = get_tokens(english_sentence_list)
sp_tokens = get_tokens(spanish_sentence_list)


###############################################################################################
## when 2 languages, you have 2 separate tokenizers.

en_dictionary, en_reverse_dictionary = build_dataset(en_tokens)
sp_dictionary, sp_reverse_dictionary = build_dataset(sp_tokens)


input_en_vocab_size  = len(en_dictionary)
target_sp_vocab_size = len(sp_dictionary)



print("vocab size english ", input_en_vocab_size)
print("vocab size spanish ", target_sp_vocab_size)


###############################################################################################

## test data
## test_english_sentence_list, test_spanish_sentence_list = get_en_and_sp_sentences(test_dict)

english_sentence_ids_list    = []
spanish_sentence_ids_list    = []

for i in range(   len(english_sentence_list)    ):
    en_sentence = english_sentence_list[i]
    sp_sentence = spanish_sentence_list[i]

    en_sentence_ids = encode(en_sentence, en_dictionary)
    sp_sentence_ids = encode(sp_sentence, sp_dictionary)
    
    en_sentence_ids = np.array(en_sentence_ids)
    sp_sentence_ids = np.array(sp_sentence_ids)
    
    en_START_TOKEN_id = en_dictionary['<sos>']
    en_END_TOKEN_id   = en_dictionary['<eos>']
    
    sp_START_TOKEN_id = sp_dictionary['<sos>']
    sp_END_TOKEN_id   = sp_dictionary['<eos>']
    
    en_sentence_ids = np.concatenate(   [   [en_START_TOKEN_id],  en_sentence_ids,  [en_END_TOKEN_id]   ]    )
    sp_sentence_ids = np.concatenate(   [   [sp_START_TOKEN_id],  sp_sentence_ids,  [sp_END_TOKEN_id]   ]    )

    if len(en_sentence_ids) <= MAX_LENGTH and len( sp_sentence_ids) <= MAX_LENGTH:
        english_sentence_ids_list.append( en_sentence_ids )
        spanish_sentence_ids_list.append( sp_sentence_ids )



###############################################################################################
## padding='post'    or    padding='pre'
## padding happens before shifting (i.e. [:, :-1] and [:, 1:])

## shifting will remove 1 from target so added 1 to target
## I think en and pt can have different max lengths but I just want
## everything to be sequence = 40

en_MAX_LENGTH = MAX_LENGTH
sp_MAX_LENGTH = MAX_LENGTH + 1

english_sentence_ids_list    = tf.keras.preprocessing.sequence.pad_sequences(
                                      english_sentence_ids_list, maxlen=en_MAX_LENGTH, padding='post')
                                      
spanish_sentence_ids_list = tf.keras.preprocessing.sequence.pad_sequences(
                                      spanish_sentence_ids_list, maxlen=sp_MAX_LENGTH, padding='post')
                                    


###############################################################################################
## to view data before or after padding
## to view only, comment out otherwise

'''

for i in range(   len(english_sentence_ids_list)    ):
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print(english_sentence_ids_list[i])
    print(spanish_sentence_ids_list[i])
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


###################################################################################################
###################################################################################################
###################################################################################################
##
## Begin Transformer Model
##
###################################################################################################
###################################################################################################
###################################################################################################


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates



###################################################################################################

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :],d_model)
  
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)



###################################################################################################
##  seq       [N, 40]


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)



###################################################################################################

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)



###################################################################################################

def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.
  
    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable to (..., seq_len_q, seq_len_k). Defaults to None.
    
    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)   ## 64
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits = scaled_attention_logits + (mask * -1e9)  

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights



###################################################################################################

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    
    assert d_model % self.num_heads == 0
    
    self.depth = d_model // self.num_heads
    
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)
    
    self.dense = tf.keras.layers.Dense(d_model)
        
  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]
    
    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)
    
    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    
    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
    
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
    return output, attention_weights



###################################################################################################

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)                  # (batch_size, seq_len, d_model)
  ])



###################################################################################################

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    
  def call(self, x, training, mask):

    attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
    
    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
    
    return out2



###################################################################################################

class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, dff)
 
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)
    
    
  def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_model)

    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)
    
    attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
    
    ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
    
    return out3, attn_weights_block1, attn_weights_block2



###################################################################################################

class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    
    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)
    
    
    self.enc_layers = [ EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers) ]
  
    self.dropout = tf.keras.layers.Dropout(rate)
        
  def call(self, x, training, mask):

    seq_len = tf.shape(x)[1]
    
    # adding embedding and position encoding.
    x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)
    
    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)
    
    return x  # (batch_size, input_seq_len, d_model)



###################################################################################################



class Decoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    
    self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
    
    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)
    
  def call(self, x, enc_output, training, look_ahead_mask, padding_mask):

    seq_len = tf.shape(x)[1]
    attention_weights = {}
    
    x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]
    
    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
      
      attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
      attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
    
    # x.shape == (batch_size, target_seq_len, d_model)
    return x, attention_weights



###################################################################################################

class Transformer(tf.keras.Model):

    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
                                       target_vocab_size, pe_input, pe_target, rate=0.1):

        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    
    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):

        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
    
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
    
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
    
        return final_output, attention_weights



###################################################################################################

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=4000):

        super(CustomSchedule, self).__init__()
    
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
    
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

###################################################################################################

learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)


###################################################################################################

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

###################################################################################################

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
  
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


###################################################################################################

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')


###################################################################################################

transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_en_vocab_size, target_sp_vocab_size, 
                          pe_input=input_en_vocab_size, 
                          pe_target=target_sp_vocab_size,
                          rate=dropout_rate)


###################################################################################################



def create_masks(inp, tar):
  # Encoder padding mask
  enc_padding_mask = create_padding_mask(inp)
  
  # Used in the 2nd attention block in the decoder.
  # This padding mask is used to mask the encoder outputs.
  dec_padding_mask = create_padding_mask(inp)
  
  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by 
  # the decoder.
  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
  dec_target_padding_mask = create_padding_mask(tar)
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
  
  return enc_padding_mask, combined_mask, dec_padding_mask



###################################################################################################
## checkpoints

checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')


###################################################################################################

# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp  = tar[:, :-1]
    tar_real = tar[:, 1:]
  
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
  
    with tf.GradientTape() as tape:

        predictions, _ = transformer(inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)

        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
  
    train_loss(loss)
    train_accuracy(tar_real, predictions)



############################################################################################
############################################################################################
############################################################################################
############################################################################################
##
## train
##
############################################################################################
############################################################################################
############################################################################################
############################################################################################


X_en = english_sentence_ids_list             ## [number_of_samples, 40]
X_sp = spanish_sentence_ids_list            ## [number_of_samples, 41]

print("X_en.shape ", X_en.shape)
print("X_sp.shape ", X_sp.shape)

############################################################################################


num_samples = X_en.shape[0]
print(num_samples)
num_batches = int(num_samples/BATCH_SIZE)


############################################################################################

for epoch_n in range(EPOCHS):
    for batch_n in range(num_batches):
    
        sta = batch_n * BATCH_SIZE
        end = sta + BATCH_SIZE
        
      
        
        ## batches with rows like this of sequence ids for words in the sentence
        ## batch_en      ->   the cat   is  0 0 ->    [12110  12  34 ...  56  12111  0   0   0]
        ## batch_sp      ->   el  gato  es  0 0  ->   [12210  6   54 ...  23   23  12211  0   0   0]
        
        batch_en = X_en[sta:end, : ]                 ## [N, 40]
        batch_sp = X_sp[sta:end, : ]                 ## [N, 41]
      
        
        #######################################################################################
        ## pass the data and train
          
        start = time.time()
  
        train_loss.reset_states()
        train_accuracy.reset_states()

        #######################################################################################

        inp = batch_en
        tar = batch_sp

        inp_tf = tf.convert_to_tensor(inp, dtype=tf.int64)
        tar_tf = tf.convert_to_tensor(tar, dtype=tf.int64)
  
        # inp -> english, tar -> spanish
        
  
        train_step(inp_tf, tar_tf)
    
        if batch_n % 50 == 0:
            print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                                  epoch_n + 1, batch_n, train_loss.result(), train_accuracy.result()))

            print("num batches ", num_batches)
            print("batch n ", batch_n)
            print ('Time taken for 1 batch: {} secs\n'.format(time.time() - start))
            
      
        if (epoch_n + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            #print ('Saving checkpoint for epoch {} at {}'.format(epoch_n+1, ckpt_save_path))
    
        ## print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch_n + 1, train_loss.result(), train_accuracy.result()))
        ## print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
                                                        
       


############################################################################################
############################################################################################
############################################################################################
############################################################################################
##
## evaluate and translate functions
##
############################################################################################
############################################################################################
############################################################################################
############################################################################################


def evaluate(inp_sentence):
  

  en_START_TOKEN_id = en_dictionary['<sos>']
  en_END_TOKEN_id   = en_dictionary['<eos>']
    
  sp_START_TOKEN_id = sp_dictionary['<sos>']
  sp_END_TOKEN_id   = sp_dictionary['<eos>']

  inp_start_token = [en_START_TOKEN_id]
  inp_end_token   = [en_END_TOKEN_id]

  
  # inp sentence is spanish, hence adding the start and end token
  inp_sentence = inp_start_token + encode(inp_sentence, en_dictionary) + inp_end_token
  encoder_input = tf.expand_dims(inp_sentence, 0)
  
  # as the target is spanish, the first word to the transformer should be the
  # spanish start token.
  decoder_input = [sp_START_TOKEN_id]
  output = tf.expand_dims(decoder_input, 0)
    
  for i in range(MAX_LENGTH):
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)
  
    # predictions.shape == (batch_size, seq_len, vocab_size)
    predictions, attention_weights = transformer(encoder_input, 
                                                 output,
                                                 False,
                                                 enc_padding_mask,
                                                 combined_mask,
                                                 dec_padding_mask)
    
    # select the last word from the seq_len dimension
    predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    
    # return the result if the predicted_id is equal to the end token
    if predicted_id == en_END_TOKEN_id:
      return tf.squeeze(output, axis=0), attention_weights
    
    # concatentate the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0), attention_weights



###################################################################################################

def translate(sentence, plot=''):
  result, attention_weights = evaluate(sentence)

  # print("pred sent shape ", result.shape)
  # print(result.dtype)
  # print(result)

  result_np = result.numpy()

  pred_list_ids = [i for i in result_np if i < target_sp_vocab_size]

  predicted_sentence = decode(pred_list_ids , sp_reverse_dictionary)  

  # print(  'Input sent to predict: {}'.format(sentence)  )
  # print(  'Predicted translation: {}'.format(predicted_sentence)  )
  
  if plot:
    plot_attention_weights(attention_weights, sentence, result, plot)

  return predicted_sentence

###################################################################################################

'''
translate("este é um problema que temos que resolver.")
print ("Real translation: this is a problem we have to solve .")


translate("os meus vizinhos ouviram sobre esta ideia.")
print ("Real translation: and my neighboring homes heard about this idea .")
'''


############################################################################################
############################################################################################
############################################################################################
############################################################################################
##
## test and evaluate
##
############################################################################################
############################################################################################
############################################################################################
############################################################################################


def calc_BLEU_score(real, pred):
    real_ref  = nltk.word_tokenize(real)
    candidate = nltk.word_tokenize(pred)

    smoothie = nltk.translate.bleu_score.SmoothingFunction().method4

    BLEUscore = nltk.translate.bleu_score.sentence_bleu( [real_ref], candidate, smoothing_function=smoothie  )

    return BLEUscore * 100.0        ## Google multiples the score by 100

##################################################################################################

BLEU_scores_list = []

##################################################################################################

## test data
## test_english_sentence_list, test_spanish_sentence_list = get_en_and_sp_sentences(test_dict)



f_out = open("output/pred_out.txt", "w")

for i in range(   len(test_english_sentence_list)    ):
    en_sentence = test_english_sentence_list[i]
    sp_sentence = test_spanish_sentence_list[i]

    y_pred_sent_sp = translate(en_sentence)

    BLEU_result = calc_BLEU_score(sp_sentence, ' '.join(   y_pred_sent_sp   )  )
    BLEU_scores_list.append(BLEU_result)

    if (i % 100 == 0):
        print(   str(i)    )
        print("size of predicted sent   ", len(y_pred_sent_sp)  )
        print("Real english input: ",       en_sentence)
        print("Real spanish translation: ", sp_sentence)

        print("predicted spanish sentence")
        print(y_pred_sent_sp)
    
        print("BLEU scores average")
        print(   sum(BLEU_scores_list) / len(BLEU_scores_list)     )

        print("########################################################################################")
        print("########################################################################################")
        print("########################################################################################")

    ## f_out.write("###############################################################################\n")
    ## f_out.write("###############################################################################\n")
    ## f_out.write("###############################################################################\n")

    ###################################################

    line_sep = "###############################################################################\n"

    ## f_out.write("number \n")
    ## f_out.write(  str(i)  )
    ## f_out.write("\n")

    index_head = "number \n"
    index_num = str(i)
    new_line = "\n"
    index_data = index_head + index_num + new_line

    ## f_out.write("real english\n")
    ## f_out.write(en_sentence)
    ## f_out.write("\n")

    real_eng_head = "real english\n"
    real_eng_sent_dt = en_sentence
    new_line = "\n"
    real_eng_data = real_eng_head + real_eng_sent_dt + new_line

    ## f_out.write("real spanish\n")
    ## f_out.write(sp_sentence)
    ## f_out.write("\n")

    real_spa_head = "real spanish\n"
    real_spa_sent_dt = sp_sentence
    new_line = "\n"
    real_spa_data = real_spa_head + real_spa_sent_dt + new_line

    ## f_out.write("pred spanish\n")
    ## f_out.write(     ' '.join(   y_pred_sent_sp   )       )
    ## f_out.write("\n")

    pred_spa_head = "pred spanish\n"
    pred_spa_sent_dt = ' '.join(   y_pred_sent_sp   )
    new_line = "\n"
    pred_spa_data = pred_spa_head + pred_spa_sent_dt + new_line

    ## f_out.write("BLEU score \n")
    ## f_out.write(  str(BLEU_result)  )
    ## f_out.write("\n")

    BLEU_head = "BLEU score \n"
    BLEU_num  = str(BLEU_result)  
    new_line = "\n"
    BLEU_data = BLEU_head + BLEU_num + new_line


    concat_super_string_dt = line_sep + line_sep + line_sep + index_data + real_eng_data + real_spa_data + pred_spa_data + BLEU_data

    f_out.write(         concat_super_string_dt            )


    ####################################################

    

f_out.close()

###################################################################################################


temp_result = translate("where is the cat")
print("Real translation: donde esta el gato.")
print("predicted sentence")
print(temp_result)


###################################################################################################


print("<<<<<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>>")



