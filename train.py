#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
import numpy as np
import os
import time
import argparse
from nmt import load_wmt_dataset, max_length, Encoder, Decoder, BahdanauAttention, preprocess_sentence, loss_function

# python train.py --num_examples 20 --batch_size 16 --epochs 1 --dict_size 20 --embedding_dim 256 --units 20
# region Define parameters
# todo adjust default values to best performing model
parser = argparse.ArgumentParser()
# Data
parser.add_argument("--num_examples", type=int, default=50000, help="Number of samples - (-1) means all.")

# Learning
parser.add_argument("--batch_size", type=int, default=16, help="Batchsize used for training.")
parser.add_argument("--epochs", type=int, default=1, help="Epochs used for training.")
parser.add_argument("--dict_size", type=int, default=5000, help="Size of dictionary used for training.")
parser.add_argument("--embedding_dim", type=int, default=256, help="Dimension of embedding.")

# Architecture
parser.add_argument("--units", type=int, default=1024, help="Encoder and decoder units.")

# read variables # todo clean up - can for sure be done more elegantly
ARGS = parser.parse_args()
num_examples = ARGS.num_examples
BATCH_SIZE = ARGS.batch_size
EPOCHS = ARGS.epochs
dict_size = ARGS.dict_size
embedding_dim = ARGS.embedding_dim
units = ARGS.units

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# endregion

def train():
    # region Process Data
    path_to_files = tf.keras.utils.get_file(
        'english.txt', origin='https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en',
        extract=False)
    path_to_files = tf.keras.utils.get_file(
        'german.txt', origin='https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de',
        extract=False)
    path_to_files = os.path.dirname(path_to_files)

    input_tensor, target_tensor, inp_lang, targ_lang = load_wmt_dataset(path_to_files, num_examples, dict_size)
    max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

    BUFFER_SIZE = len(input_tensor_train)
    steps_per_epoch = len(input_tensor_train)//BATCH_SIZE

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    # endregion

    # Region: model definition
    encoder = Encoder(dict_size, embedding_dim, units, BATCH_SIZE)
    attention_layer = BahdanauAttention(10)
    decoder = Decoder(dict_size, embedding_dim, units, BATCH_SIZE)
    optimizer = tf.keras.optimizers.Adam()

    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)


    @tf.function
    def train_step(inp, targ, enc_hidden):
      loss = 0

      with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
          # passing enc_output to the decoder
          predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

          loss += loss_function(targ[:, t], predictions)

          # using teacher forcing
          dec_input = tf.expand_dims(targ[:, t], 1)

      batch_loss = (loss / int(targ.shape[1]))

      variables = encoder.trainable_variables + decoder.trainable_variables

      gradients = tape.gradient(loss, variables)

      optimizer.apply_gradients(zip(gradients, variables))

      return batch_loss


    for epoch in range(EPOCHS):
      start = time.time()

      enc_hidden = encoder.initialize_hidden_state()
      total_loss = 0

      for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
          print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                       batch,
                                                       batch_loss.numpy()))
      # saving (checkpoint) the model every 2 epochs
      if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

      print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                          total_loss / steps_per_epoch))
      print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    # region test
    # # todo move to separate script called TEST.PY
    # def translate(sentence):
    #   sentence = preprocess_sentence(sentence)
    #   inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    #   inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
    #                                                          maxlen=max_length_inp,
    #                                                          padding='post')
    #   inputs = tf.convert_to_tensor(inputs)
    #   result = ''
    #   hidden = [tf.zeros((1, units))]
    #   enc_out, enc_hidden = encoder(inputs, hidden)
    #   dec_hidden = enc_hidden
    #   dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)
    #
    #   for t in range(max_length_targ):
    #     predictions, dec_hidden, attention_weights = decoder(dec_input,
    #                                                          dec_hidden,
    #                                                          enc_out)
    #
    #     predicted_id = tf.argmax(predictions[0]).numpy()
    #     result += targ_lang.index_word[predicted_id] + ' '
    #     if targ_lang.index_word[predicted_id] == '<end>':
    #       return result, sentence
    #     # the predicted ID is fed back into the model
    #     dec_input = tf.expand_dims([predicted_id], 0)
    #
    #   return result, sentence
    # # restoring the latest checkpoint in checkpoint_dir
    # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    # print(translate(u'du da.'))
    # endregion

if __name__ == "__main__":
    train()