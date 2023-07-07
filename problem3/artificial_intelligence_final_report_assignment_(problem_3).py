# NOTED: THIS PYTHON SCRIPT RUN ON KAGGLE'S ENVIRONMENT; SET YOUR EVIRONMENT AS KAGGLE'S ENVIRONMENT FOR NO MISTAKE#
####################################################################################################################

#Import lib
from __future__ import absolute_import, division, print_function
## Import TensorFlow >= 1.10 and enable eager execution
import tensorflow as tf
tf.enable_eager_execution()
tf.executing_eagerly()
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import unicodedata
import re
import numpy as np
import os
import time
import math
import matplotlib.pyplot as plt
from fastprogress import master_bar, progress_bar

##dowload dataset first
# NMT 
#place folder variable by the place u dowload this file
folder = 'nmt_data'

vocab_en = os.path.join(folder, 'vocab.en')
vocab_vi = os.path.join(folder, 'vocab.vi')

train_en = os.path.join(folder, 'train.en')
train_vi = os.path.join(folder, 'train.vi')

validation_en = os.path.join(folder, 'tst2012.en')
validation_vi = os.path.join(folder, 'tst2012.vi')

test_en = os.path.join(folder, 'tst2013.en')
test_vi = os.path.join(folder, 'tst2013.vi')

def preprocess_sentence(w):
    """
    Data preprocessing function

    parameter:
      w: sentence input

    Returns:
      sentence after datapreprocessing
    """
  
    w = w.lower().strip()
        
    # remove &apos; &quot; in dataset
    w = re.sub(" &apos;", "", w)
    w = re.sub(" &quot;", "", w)
    
    # remove special characters & punctuation, keep only letters, numbers from 0-9 and spaces
    w = re.sub(r"[^\w0-9 ]+", " ", w)
    
    # shorten multiple long spaces into 1 space
    w = re.sub(r"[\s]+", " ", w)
    
    # Remove the leading and trailing spaces in sentences
    w = w.strip()
    
    # add <start> and <end> at the beginning and end of the sentence so that the model knows where to start and end the prediction
    w = '<start> ' + w + ' <end>'
    
    return w

en_sentence = u"May I borrow this book?"
vi_sentence = u"Mình mượn cuốn sách này được không?"
print(preprocess_sentence(en_sentence))
print(preprocess_sentence(vi_sentence))

def create_dataset(input_path, target_path):
  """
  The function imports data from the path, cleans and returns English - Vietnamese sentence pairs
  
  Parameters:
    input_path: input data path
    target_path: output data path
    
  Returns:
    List of corresponding pairs of English - Vietnamese sentences
  """
  
  input_lines = open(input_path, encoding='UTF-8').read().strip().split('\n')
  target_lines = open(target_path, encoding='UTF-8').read().strip().split('\n')
  
  word_pairs = []
  for i in range(len(input_lines)):
    pi = preprocess_sentence(input_lines[i])
    pt = preprocess_sentence(target_lines[i])
    word_pairs.append([pi, pt])
  
  return word_pairs

train_pairs = create_dataset(train_en, train_vi)

def print_samples(pairs, number):
  """
  The function prints some English - Vietnamese sentences
  
  Parameters:
    pairs: list of English - Vietnamese sentence pairs
    number: number of pairs of sentences to print
    
  Returns:
    None
  """
  
  for i in range(number):
    print(pairs[i][0])
    print(pairs[i][1])

print_samples(train_pairs, 10)

def short_sentences(pairs):
  """
  The function filters out complete English - Vietnamese sentences with a length of 60 words or less, based on statistics
  and limit the memory of the GPU used for training
  
  Parameters:
    pairs: list of English - Vietnamese sentence pairs
    
  Returns:
    List of pairs of English - Vietnamese sentences with a length of 60 words or less
  """
  
  result = []
  
  for i in range(len(pairs)):
    if len(pairs[i][0].split()) <= 60 and len(pairs[i][1].split()) <= 60:
      result.append([pairs[i][0], pairs[i][1]])
  
  return result
short_train_pairs = short_sentences(train_pairs)
print_samples(short_train_pairs, 10)

val_pairs = create_dataset(validation_en, validation_vi)
print_samples(val_pairs, 10)

test_pairs = create_dataset(test_en, test_vi)
print_samples(test_pairs, 10)

# This class creates a word -> index mapping (e.g,. "dad" -> 5) and vice-versa 
# (e.g., 5 -> "dad") for each language,
class LanguageIndex():
  """
  Update vocab from the sentences of the language to be initialized
  Create mapping letters -> numbers ("dad" -> 5) and numbers -> letters (5 -> "dad")
  """
  
  def __init__(self, lang):
    """
    Initialization function
    
    Parameters:
      lang: language to initialize
      
    Returns:
      None
    """
    self.lang = lang
    self.word2idx = {}
    self.idx2word = {}
    self.vocab = set()
    
    self.create_index()
    
  def create_index(self):
    """
    Update vocab from the sentences of the language to be initialized
    Create mapping letters -> numbers and numbers -> letters
    """
    
    for phrase in self.lang:
      self.vocab.update(phrase.split())
    
    self.vocab = sorted(self.vocab)
    
    self.word2idx['<pad>'] = 0
    for index, word in enumerate(self.vocab):
      self.word2idx[word] = index + 1
    
    for word, index in self.word2idx.items():
      self.idx2word[index] = word

def max_length(tensor):
    """
    Returns the longest sentence in tensor
    
    Parameters:
      tensor: tensor of sentences
    """
    
    return max(len(t) for t in tensor)

def load_train_dataset(input_path, target_path):
  """
  Input data for the set Train
  Update vocab of each language English, Vietnamese
  Filter out pairs of English - Vietnamese sentences with length <= 60 words (choosing 60 words is based on statistics and GPU limitations)
  Create tensor containing the index of pairs of English - Vietnamese sentences in the set Train
  Fill short sentences with index 0
  
  Parameters:
    input_path: path of the English sentence set
    target_path: path of Vietnamese sentence set
    
  Returns:
    input_tensor: tensor index English
    output_tensor: tensor index Vietnamese
    inp_lang: English vocab
    targ_lang: Vietnamese vocab
    max_length_inp: the maximum length in the set of English sentences
    max_length_tar: the maximum length in the Vietnamese sentence set
    
  """
  
  # Input data for the Train dataset
  pairs = create_dataset(input_path, target_path)
  
  # Update vocab of each language English, Vietnamese
  inp_lang = LanguageIndex(en for en, vi in pairs)
  targ_lang = LanguageIndex(vi for en, vi in pairs)
  
  # Filter out pairs of English - Vietnamese sentences with length <= 60 words
  pairs = short_sentences(pairs)

  # Convert words in English sentences to index
  input_tensor = [[inp_lang.word2idx[word] for word in en.split()] for en, vi in pairs]

  # Convert words in Vietnamese sentences to index
  target_tensor = [[targ_lang.word2idx[word] for word in vi.split()] for en, vi in pairs]

  # Calculate the maximum length in the set of English and Vietnamese sentences
  max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)

  # Fill short sentences with index 0
  input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, maxlen=max_length_inp, padding='post')

  target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor, maxlen=max_length_tar, padding='post')

  return input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_tar
input_tensor_train, target_tensor_train, inp_lang, targ_lang, max_length_inp, max_length_targ = load_train_dataset(train_en, train_vi)

def load_val_dataset(input_path, target_path, inp_lang, targ_lang):
  """
  Enter input data for the Validation set
  Create tensor containing the index of English - Vietnamese sentence pairs in the Validation set based on the vocab of Train
  Fill short sentences with index 0
  
  Parameters:
    input_path: path of the English sentence set
    target_path: path of Vietnamese sentence set
    inp_lang: English vocab of Train
    targ_lang: Vietnamese vocab of Train
    
  Returns:
    input_tensor: English tensor index of the Validation set
    output_tensor: Vietnamese tensor index of the Validation set
    inp_lang: English vocab
    targ_lang: Vietnamese vocab
    max_length_inp: the maximum length in the set of English sentences
    max_length_tar: the maximum length in the Vietnamese sentence set
    
  """
  
  # Enter input data for the Validation dataset
  pairs = create_dataset(input_path, target_path)
  
  # Convert words in English and Vietnamese sentences to an index based on the Vocab of Train dataset
  # If the word is not in the vocab of the Train dataset, index = 0
  
  input_tensor = []
  target_tensor = []
  
  for en, vi in pairs:
    inputs = []
    for word in en.split():
      try:
        index = inp_lang.word2idx[word]
      except:
        index = 0
      inputs.append(index)
    input_tensor.append(inputs)
    
    targets = []
    for word in vi.split():
      try:
        index = targ_lang.word2idx[word]
      except:
        index = 0
      targets.append(index)
    target_tensor.append(targets)

  # Calculate the maximum length in the set of English and Vietnamese sentences
  max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)

  # Fill short sentences with index 0
  input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, maxlen=max_length_inp, padding='post')
  input_tensor = tf.convert_to_tensor(input_tensor)

  target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor, maxlen=max_length_tar, padding='post')
  target_tensor = tf.convert_to_tensor(target_tensor)

  return input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_tar
input_tensor_val, target_tensor_val, inp_lang_val, targ_lang_val, max_length_inp_val, max_length_targ_val = load_val_dataset(validation_en, validation_vi, inp_lang, targ_lang)

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 256
N_BATCH = BUFFER_SIZE//BATCH_SIZE
embedding_dim = 256
units = 512
vocab_inp_size = len(inp_lang.word2idx)
vocab_tar_size = len(targ_lang.word2idx)

train_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)

validation_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val)).shuffle(BUFFER_SIZE)
validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=True)

def gru(units):
  """
  The gru setup function is based on whether there is a GPU or not
  
  Parameters:
    units: number of hidden units
  """
  
  if tf.test.is_gpu_available():
    return tf.keras.layers.CuDNNGRU(units, 
                                    return_sequences=True, 
                                    return_state=True, 
                                    recurrent_initializer='glorot_uniform')
  else:
    return tf.keras.layers.GRU(units, 
                               return_sequences=True, 
                               return_state=True, 
                               recurrent_activation='sigmoid', 
                               recurrent_initializer='glorot_uniform')

class Encoder(tf.keras.Model):
    """
    Encoder for English
    """
  
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        """
        Parameter constructor for encoder

        Parameters:
          vocab_size: English vocab size
          embedding_dim: word vector size
          enc_units: number of hidden units
          batch_sz: size 1 batch
        """
      
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.enc_units)
        
    def call(self, x, hidden):
        """
        Run function

        Returns:
          output, state: output of Encoder
        """
      
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)        
        return output, state
    
    def initialize_hidden_state(self):
        """
        Hàm khởi tạo hidden state
        """
        return tf.zeros((self.batch_sz, self.enc_units))
    
class BahdanauAttention(tf.keras.Model):
  """
  Bahdanau Attention for seq2seq Anh - Việt
  """
  
  def __init__(self, units):
    """
    Initialize Attention
    
    Parameters:
      units: number of hidden units of decoder
    """
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # hidden shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # we are doing this to perform addition to calculate the score
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights
  
class LuongAttention(tf.keras.Model):
  """
  LuongAttention for seq2seq English - Vietnamese
  """

  def __init__(self, units):
    """
    Initialize Attention
    
    Parameters:
      units: number of hidden units of decoder
    """
    super(LuongAttention, self).__init__()
    self.W = tf.keras.layers.Dense(units)
  
  def call(self, query, values):
    # hidden shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # we are doing this to perform addition to calculate the score
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, hidden_size)
    score = tf.matmul(values, self.W(hidden_with_time_axis), transpose_b=True)

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)
    
    return context_vector, attention_weights

class Decoder(tf.keras.Model):
    """
    Decoder for Vietnamese
    """
  
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
      """
      Parameter constructor for decoder
      
      Parameters:
        vocab_size: Vietnamese vocab size
        embedding_dim: word vector size
        dec_units: number of hidden units
        batch_sz: size 1 batch
      """
      
      super(Decoder, self).__init__()
      self.batch_sz = batch_sz
      self.dec_units = dec_units
      self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
      self.gru = gru(self.dec_units)
      self.fc = tf.keras.layers.Dense(vocab_size)

      # used for attention
      self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
      # enc_output shape == (batch_size, max_length, hidden_size)
      context_vector, attention_weights = self.attention(hidden, enc_output)

      # x shape after passing through embedding == (batch_size, 1, embedding_dim)
      x = self.embedding(x)

      # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
      x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

      # passing the concatenated vector to the GRU
      output, state = self.gru(x)

      # output shape == (batch_size * 1, hidden_size)
      output = tf.reshape(output, (-1, output.shape[2]))

      # output shape == (batch_size, vocab)
      x = self.fc(output)

      return x, state, attention_weights
        
    def initialize_hidden_state(self):
        """
        Hidden state constructor
        """
        return tf.zeros((self.batch_sz, self.dec_units))

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

LEARNING_RATE = 0.001
optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
def loss_function(real, pred):
  """
  Loss calculation function between prediction and target
  """
  
  mask = 1 - np.equal(real, 0)
  loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
  
  return tf.reduce_mean(loss_)
global_step = tf.train.get_or_create_global_step()

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

def epoch_training(encoder, decoder, dataset, global_step, mb, num_step):
    """
    Training function on the set Train

    Parameters:
      encoder: English encoder
      decoder: Vietnamese decoder
      dataset: train dataset
      global_step, mb, num_step: model plotting parameters

    Returns:
      average loss of 1 epoch
    """
  
    train_losses = [] # Chứa giá trị loss của các batch
    dataset_iter = iter(dataset)
    for _ in progress_bar(range(num_step), parent=mb):
        inp, targ = next(dataset_iter)
        loss = 0
        
        with tf.GradientTape() as tape:
            # Feedforward
            enc_output, enc_hidden = encoder(inp, hidden)
            
            dec_hidden = enc_hidden
            
            dec_input = tf.expand_dims([targ_lang.word2idx['<start>']] * BATCH_SIZE, 1)
            
            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                
                loss += loss_function(targ[:, t], predictions)
                
                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)
                
            batch_loss = (loss / int(targ.shape[1]))
            
            train_losses.append(batch_loss)
        
        variables = encoder.variables + decoder.variables
        
        gradients = tape.gradient(loss, variables)
        
        optimizer.apply_gradients(zip(gradients, variables), global_step=global_step) 
        
        mb.child.comment = 'Train loss {:.4f}'.format(loss)
        
    return sum(train_losses)/ len(train_losses)

def epoch_evaluation(encoder, decoder, dataset, mb, num_step):
    """
    Evaluation function on the set of Validation

    Parameters:
      encoder: English encoder
      decoder: Vietnamese decoder
      dataset: validation dataset
      global_step, mb, num_step: model plotting parameters

    Returns:
      average loss of 1 epoch
    """
  
    val_losses = [] # Chứa giá trị loss của các batch
    dataset_iter = iter(dataset)
    for _ in progress_bar(range(num_step), parent=mb):
        inp, targ = next(dataset_iter)
        loss = 0
        
        # Feed forward
        enc_output, enc_hidden = encoder(inp, hidden)
            
        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([targ_lang.word2idx['<start>']] * BATCH_SIZE, 1)
        
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions)
            
            dec_input = tf.expand_dims(targ[:, t], 1)
            
        batch_loss = (loss / int(targ.shape[1]))
        
        val_losses.append(batch_loss)
        
        mb.child.comment = 'Validation loss {:.4f}'.format(loss)
    
    return sum(val_losses)/ len(val_losses)

############TRAIN###################
EPOCHS = 20

mb = master_bar(range(EPOCHS))
mb.names = ['Training loss', 'Validation loss']
training_losses = []
validation_losses = []
x = []

train_step = math.floor(len(input_tensor_train)*1.0/BATCH_SIZE)
val_step = math.floor(len(input_tensor_val)*1.0/BATCH_SIZE)

for epoch in mb:
    
    # initializing the hidden state at the start of every epoch
    # initally hidden is None
    hidden = encoder.initialize_hidden_state()
    x.append(epoch)
    
    ### START CODE HERE
    # Training
    training_loss = epoch_training(encoder, decoder, train_dataset, global_step, mb, train_step)
    
    # Update information after training
    training_losses.append(training_loss)
    
    # Validating
    valid_loss = epoch_evaluation(encoder, decoder, validation_dataset, mb, val_step)
    
    # Update information after validation
    validation_losses.append(valid_loss)
    
    # Graph update
    global_step.assign_add(1)
    mb.update_graph([[x, training_losses], [x, validation_losses]], [0,EPOCHS], [0,5])
    
    print('Finish train epoch {} with loss {:.4f}'.format(epoch, training_loss))
    print('Finish validate epoch {} with loss {:.4f}'.format(epoch,valid_loss))
    
    # Update score and save the model with the best score
    if (epoch + 1) % 2 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)
    ### END CODE HERE

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

###########CHECK OUTPUT###############
def evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
    """
    The function translates each sentence and creates the parameter of the correlation matrix of attention between English sentences and Vietnamese sentences

    Parameters:
      sentence: English sentence
      encoder: English encoder
      decoder: Vietnamese decoder
      inp_lang: English vocab
      targ_lang: Vietnamese vocab
      max_length_inp: longest English sentence length
      max_length_targ: maximum length of Vietnamese sentences

    Returns:
      result: translated Vietnamese sentence
      sentence: English sentence inserted
      attention_plot: statistics to plot the matrix Attention
    """
  
    attention_plot = np.zeros((max_length_targ, max_length_inp))
    
    # Clean up English sentences
    sentence = preprocess_sentence(sentence)
    
    # Convert English sentences to index
    inputs = []
    for word in sentence.split():
      try:
        index = inp_lang.word2idx[word]
      except:
        index = 0
      inputs.append(index)

    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    
    # translate
    
    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word2idx['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
        
        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang.idx2word[predicted_id]

        if targ_lang.idx2word[predicted_id] == '<end>':
            return result, sentence, attention_plot
        else:
          result += ' '
        
        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot
# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
    """
    Attention . matrix drawing function

    Parameters:
      attention: matrix parameter
      sentence: English sentence
      predicted_sentence: translated Vietnamese sentence

    Returns:
      None
    """
  
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')
    
    fontdict = {'fontsize': 14}
    
    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    plt.show()
def translate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
    """
    The translation function and drawing the attention . matrix

    Parameters:
      sentence: English sentence
      encoder: English encoder
      decoder: Vietnamese decoder
      inp_lang: English vocab
      targ_lang: Vietnamese vocab
      max_length_inp: longest English sentence length
      max_length_targ: maximum length of Vietnamese sentences

    Returns:
      None
    """
  
    result, sentence, attention_plot = evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
        
    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(result))
    
    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    plot_attention(attention_plot, sentence.split(' '), result.split(' '))

########TEST TRANSLATION###############
translate(u'i love you', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Python implementation of BLEU and smooth-BLEU.

This module provides a Python implementation of BLEU and smooth-BLEU.
Smooth BLEU is computed following the method outlined in the paper:
Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
evaluation metrics for machine translation. COLING 2004.
"""

import collections
import math


def _get_ngrams(segment, max_order):
  """Extracts all n-grams upto a given maximum order from an input segment.

  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.

  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
  ngram_counts = collections.Counter()
  for order in range(1, max_order + 1):
    for i in range(0, len(segment) - order + 1):
      ngram = tuple(segment[i:i+order])
      ngram_counts[ngram] += 1
  return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
  """Computes BLEU score of translated segments against one or more references.

  Args:
    reference_corpus: list of lists of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    smooth: Whether or not to apply Lin et al. 2004 smoothing.

  Returns:
    3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
    precisions and brevity penalty.
  """
  matches_by_order = [0] * max_order
  possible_matches_by_order = [0] * max_order
  reference_length = 0
  translation_length = 0
  for (references, translation) in zip(reference_corpus,
                                       translation_corpus):
    reference_length += min(len(r) for r in references)
    translation_length += len(translation)

    merged_ref_ngram_counts = collections.Counter()
    for reference in references:
      merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
    translation_ngram_counts = _get_ngrams(translation, max_order)
    overlap = translation_ngram_counts & merged_ref_ngram_counts
    for ngram in overlap:
      matches_by_order[len(ngram)-1] += overlap[ngram]
    for order in range(1, max_order+1):
      possible_matches = len(translation) - order + 1
      if possible_matches > 0:
        possible_matches_by_order[order-1] += possible_matches

  precisions = [0] * max_order
  for i in range(0, max_order):
    if smooth:
      precisions[i] = ((matches_by_order[i] + 1.) /
                       (possible_matches_by_order[i] + 1.))
    else:
      if possible_matches_by_order[i] > 0:
        precisions[i] = (float(matches_by_order[i]) /
                         possible_matches_by_order[i])
      else:
        precisions[i] = 0.0

  if min(precisions) > 0:
    p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
    geo_mean = math.exp(p_log_sum)
  else:
    geo_mean = 0

  ratio = float(translation_length) / reference_length

  if ratio > 1.0:
    bp = 1.
  else:
    bp = math.exp(1 - 1. / ratio)

  bleu = geo_mean * bp

  return (bleu, precisions, bp, ratio, translation_length, reference_length)
pairs_2 = create_dataset(test_en, test_vi)
translation_result = []
for i in range(len(pairs_2)):
  en_sentence = pairs_2[i][0].strip("<start>").strip("<end>").strip()
  vi_result, en_sentence, attention_plot = evaluate(en_sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
  vi_result = '<start> ' + vi_result.strip()
  vi_groundtruth = pairs_2[i][1]
  bleu_score = compute_bleu([[vi_groundtruth.split()]], [vi_result.split()], max_order=1)[0]*100
  print('\nInput: ', en_sentence)
  print('Groundtruth: ', vi_groundtruth)
  print('Translation: ', vi_result)
  print('Bleu Score: ', bleu_score)
  if len(vi_result.split(' ')) <= 10:
    attention_plot = attention_plot[:len(vi_result.split(' ')), :len(en_sentence.split(' '))]
    plot_attention(attention_plot, en_sentence.split(' '), vi_result.strip('<start> ').split(' '))
  translation_result.append([en_sentence, vi_groundtruth, vi_result, bleu_score, attention_plot])


##############CALCULATE Average Bleu Score base on test dataset#######################################

bleu_score_list = []
for i in range(len(translation_result)):
  bleu_score_list.append(translation_result[i][3])

mean_bleu_score = sum(bleu_score_list)/len(bleu_score_list)
print("Average Bleu Score: ", mean_bleu_score)