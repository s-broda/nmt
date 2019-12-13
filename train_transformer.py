#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow_datasets as tfds
import tensorflow as tf
import time
import numpy as np
import os.path
from transformer import CustomSchedule, Transformer, create_masks
# paths
checkpoint_path = "./checkpoints/garb"
tokenizer_path = "./"
data_path = './data'

# training parameters
BUFFER_SIZE = 20000
BATCH_SIZE = 64
MAX_LENGTH = 40 # use only training examples shorter than this
EPOCHS = 1
TRAIN_ON = 1 # percentage of data to train on
DICT_SIZE = 2**13 # this is likely too small
# model hyperparameters
num_layers = 4 # base transformer uses 6
d_model = 128 # base transformer uses 512
dff = 512 # base transformer uses 2048
num_heads = 8 # base transformer uses 8
dropout_rate = 0.1

split = tfds.Split.TRAIN.subsplit(tfds.percent[:TRAIN_ON])

examples, metadata = tfds.load('wmt14_translate/de-en', data_dir=data_path, with_info=True,
                               as_supervised=True, split=[split, 'validation'])
train_examples, val_examples = examples[0], examples[1]

if os.path.isfile(os.path.join(tokenizer_path, "tokenizer_en_" + str(DICT_SIZE) + "_" + str(TRAIN_ON) + ".subwords")):
    tokenizer_en = tfds.features.text.SubwordTextEncoder.load_from_file(os.path.join(tokenizer_path, "tokenizer_en_" + str(DICT_SIZE) + "_" + str(TRAIN_ON)))
else:
    tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (en.numpy() for de, en in train_examples), target_vocab_size=DICT_SIZE)
    tokenizer_en.save_to_file(os.path.join(tokenizer_path, "tokenizer_en_" + str(DICT_SIZE) + "_" + str(TRAIN_ON)))
if os.path.isfile(os.path.join(tokenizer_path, "tokenizer_de_" + str(DICT_SIZE) + "_" + str(TRAIN_ON) + ".subwords")):
    tokenizer_de = tfds.features.text.SubwordTextEncoder.load_from_file("tokenizer_de_" + str(DICT_SIZE) + "_" + str(TRAIN_ON))
else:
    tokenizer_de = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (de.numpy() for de, en in train_examples), target_vocab_size=DICT_SIZE)
    tokenizer_de.save_to_file(os.path.join(tokenizer_path, "tokenizer_de_" + str(DICT_SIZE) + "_" + str(TRAIN_ON)))
    
input_vocab_size = tokenizer_de.vocab_size + 2
target_vocab_size = tokenizer_en.vocab_size + 2

def filter_max_length(x, y, max_length=MAX_LENGTH):
  return tf.logical_and(tf.size(x) <= max_length,
                        tf.size(y) <= max_length)

def encode(lang1, lang2):
  lang1 = [tokenizer_de.vocab_size] + tokenizer_de.encode(
      lang1.numpy()) + [tokenizer_de.vocab_size+1]

  lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
      lang2.numpy()) + [tokenizer_en.vocab_size+1]
  
  return lang1, lang2


  
def tf_encode(de, en):
  return tf.py_function(encode, [de, en], [tf.int64, tf.int64])


train_dataset = train_examples.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)
# cache the dataset to memory to get a speedup while reading from it.
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(
    BATCH_SIZE, padded_shapes=([-1], [-1]))
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)


val_dataset = val_examples.map(tf_encode)
val_dataset = val_dataset.filter(filter_max_length).padded_batch(
    BATCH_SIZE, padded_shapes=([-1], [-1]))


learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  
  return tf.reduce_mean(loss_)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size, 
                          pe_input=input_vocab_size, 
                          pe_target=target_vocab_size,
                          rate=dropout_rate)



ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')

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
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]
  
  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
  with tf.GradientTape() as tape:
    predictions, _ = transformer(inp, tar_inp, 
                                 True, 
                                 enc_padding_mask, 
                                 combined_mask, 
                                 dec_padding_mask)
    loss = loss_function(tar_real, predictions)

  gradients = tape.gradient(loss, transformer.trainable_variables)    
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
  
  train_loss(loss)
  train_accuracy(tar_real, predictions)
  
for epoch in range(EPOCHS):
  start = time.time()
  
  train_loss.reset_states()
  train_accuracy.reset_states()
  
  # inp -> portuguese, tar -> english
  for (batch, (inp, tar)) in enumerate(train_dataset):
    train_step(inp, tar)
    
    if batch % 50 == 0:
      print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
          epoch + 1, batch, train_loss.result(), train_accuracy.result()))
      
  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))
    
  print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                train_loss.result(), 
                                                train_accuracy.result()))

  print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
  



