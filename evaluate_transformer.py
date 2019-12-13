#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import os.path
from transformer import Transformer, create_masks
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')

# paths
checkpoint_path = "./checkpoints/train"
output_path = "./output"
data_path = './data'

# training parameters
MAX_LENGTH = 40 # use only training examples shorter than this
DICT_SIZE = 2**13 # this is likely too small
# model hyperparameters
num_layers = 4 # base transformer uses 6
d_model = 128 # base transformer uses 512
dff = 512 # base transformer uses 2048
num_heads = 8 # base transformer uses 8
dropout_rate = 0.1

tokenizer_en = tfds.features.text.SubwordTextEncoder.load_from_file(os.path.join(output_path, "tokenizer_en_" + str(DICT_SIZE)))
tokenizer_de = tfds.features.text.SubwordTextEncoder.load_from_file(os.path.join(output_path, "tokenizer_de_" + str(DICT_SIZE)))
input_vocab_size = tokenizer_de.vocab_size + 2
target_vocab_size = tokenizer_en.vocab_size + 2

transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size, 
                          pe_input=input_vocab_size, 
                          pe_target=target_vocab_size,
                          rate=dropout_rate)
                                 

ckpt = tf.train.Checkpoint(transformer=transformer)


ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
print ('Latest checkpoint restored!!')

examples, metadata = tfds.load('wmt14_translate/de-en', data_dir=data_path, with_info=True,
                               as_supervised=True)
test_examples = examples['test']

def evaluate(inp_sentence):
  start_token = [tokenizer_de.vocab_size]
  end_token = [tokenizer_de.vocab_size + 1]
  
  # inp sentence is portuguese, hence adding the start and end token
  inp_sentence = start_token + tokenizer_de.encode(inp_sentence) + end_token
  encoder_input = tf.expand_dims(inp_sentence, 0)
  
  # as the target is english, the first word to the transformer should be the
  # english start token.
  decoder_input = [tokenizer_en.vocab_size]
  output = tf.expand_dims(decoder_input, 0)
    
  for i in range(MAX_LENGTH):
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        encoder_input, output)
  
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
    if predicted_id == tokenizer_en.vocab_size+1:
      return tf.squeeze(output, axis=0), attention_weights
    
    # concatentate the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0), attention_weights

def plot_attention_weights(attention, sentence, result, layer):
  fig = plt.figure(figsize=(16, 8))
  
  sentence = tokenizer_de.encode(sentence)
  
  attention = tf.squeeze(attention[layer], axis=0)
  
  for head in range(attention.shape[0]):
    ax = fig.add_subplot(2, 4, head+1)
    
    # plot the attention weights
    ax.matshow(attention[head][:-1, :], cmap='viridis')

    fontdict = {'fontsize': 10}
    
    ax.set_xticks(range(len(sentence)+2))
    ax.set_yticks(range(len(result)))
    
    ax.set_ylim(len(result)-1.5, -0.5)
        
    ax.set_xticklabels(
        ['<start>']+[tokenizer_de.decode([i]) for i in sentence]+['<end>'], 
        fontdict=fontdict, rotation=90)
    
    ax.set_yticklabels([tokenizer_en.decode([i]) for i in result 
                        if i < tokenizer_en.vocab_size], 
                       fontdict=fontdict)
    
    ax.set_xlabel('Head {}'.format(head+1))
  
  plt.tight_layout()
  plt.show()
  
def translate(sentence, plot=''):
  result, attention_weights = evaluate(sentence)
  
  predicted_sentence = tokenizer_en.decode([i for i in result 
                                            if i < tokenizer_en.vocab_size])  

  print('Input: {}'.format(sentence))
  print('Predicted translation: {}'.format(predicted_sentence))
  
  if plot:
    plot_attention_weights(attention_weights, sentence, result, plot)
  
  return  predicted_sentence

translations = []
inputs = []
targets = []    
BLEUs = []
for sentence in test_examples:
    inp = sentence[0].numpy().decode('utf-8')
    target = sentence[1].numpy().decode('utf-8')
    translation = translate(inp)
    BLEU = nltk.translate.bleu_score.sentence_bleu([nltk.word_tokenize(target)], nltk.word_tokenize(translation))
    translations.append(translation)
    inputs.append(inp)
    BLEUs.append(BLEU)
    print('Average BLEU score: ', 100 * np.mean(BLEUs))
    targets.append(target)

d = {'input': inputs, 'target': targets, 'translation': translations, 'BLEU': BLEUs}
df = pd.DataFrame.from_dict(d)
df.to_csv(os.path.join(output_path, 'results.csv'))
print('Average BLEU score: ', 100 * np.mean(BLEUs))