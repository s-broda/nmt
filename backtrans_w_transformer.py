#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# region Packages & Setup
from __future__ import absolute_import, division, print_function, unicode_literals
import json
import pandas as pd
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import os.path
import argparse
from transformer import Transformer, create_masks
from beam_search import beam_search
import nltk
nltk.download('punkt')

# validation parameters
parser = argparse.ArgumentParser()
parser.add_argument("--train_dir", type=str, help="Directory of nmt - needed for cluster")
parser.add_argument("--experiment_name", type=str, required=True, help="Model to use for backtranslation.")
parser.add_argument("--beam_width", type=int, default=10, help="Beam width for search.") # https://arxiv.org/pdf/1609.08144.pdf
parser.add_argument("--alpha", type=float, default=0.65, help="Length penalty.") # https://arxiv.org/pdf/1609.08144.pdf
parser.add_argument("--train_backtrans_on", type=int, default=2, help="Percentage of remaining train data to backtranslate.")

ARGS = parser.parse_args()
train_dir = ARGS.train_dir
experiment_name = ARGS.experiment_name
beam_width = ARGS.beam_width
alpha = ARGS.alpha
train_backtrans_on = ARGS.train_backtrans_on
# paths
checkpoint_path = os.path.join(train_dir, "checkpoints")
output_path = os.path.join(train_dir, "output")
data_path = os.path.join(train_dir, "data")

print('PATHS:   ')
print(checkpoint_path)
print(output_path)
print(data_path)

# read config of experiment_name and store in respective variables
checkpoint_path = os.path.normpath(os.path.join(checkpoint_path, experiment_name))
config = json.load(open(os.path.join(checkpoint_path, 'config.json')))
MAX_LENGTH = config['MAX_LENGTH']# use only training examples shorter than this
DICT_SIZE = config['DICT_SIZE'] # this is likely too small
TRAIN_ON = config['TRAIN_ON']

if TRAIN_ON == 100:
    raise ValueError("Only models w TRAIN_ON < 100 can be used for backtrans - as there needs to be training samples left that were not used for model training")
num_layers = config['num_layers'] # base transformer uses 6
d_model = config['d_model'] # base transformer uses 512
dff = config['dff'] # base transformer uses 2048
num_heads = config['num_heads'] # base transformer uses 8
dropout_rate = config['dropout_rate']

if TRAIN_ON < 100:
    print('Reading spec tokenizers')
    tag_new_tok = 'to' + str(TRAIN_ON)
else:
    tag_new_tok = ''
# endregion

def evaluate_transformer():
    tokenizer_en = tfds.features.text.SubwordTextEncoder.load_from_file(os.path.join(output_path, tag_new_tok + "tokenizer_en_" + str(DICT_SIZE)))
    tokenizer_de = tfds.features.text.SubwordTextEncoder.load_from_file(os.path.join(output_path, tag_new_tok + "tokenizer_de_" + str(DICT_SIZE)))
    input_vocab_size = tokenizer_en.vocab_size + 2
    target_vocab_size = tokenizer_de.vocab_size + 2

    # using transformer2 as eng-> de
    transformer2 = Transformer(num_layers, d_model, num_heads, dff,
                              input_vocab_size, target_vocab_size,
                              pe_input=input_vocab_size,
                              pe_target=target_vocab_size,
                              rate=dropout_rate)


    ckpt = tf.train.Checkpoint(transformer2=transformer2)
    ckpt.restore(tf.train.latest_checkpoint(checkpoint_path)).expect_partial()
    print('Latest checkpoint restored!!')
    # loading different part of training set for backtrans (before :TRAIN_ON)
    train_on_end = TRAIN_ON+train_backtrans_on
    split = tfds.Split.TRAIN.subsplit(tfds.percent[TRAIN_ON:train_on_end])
    print('Split is: {}'.format(split))
    examples, metadata = tfds.load('wmt14_translate/de-en', data_dir=data_path, with_info=True,
                                   as_supervised=True, split=split)
    def filter_max_length(x, y, max_length=MAX_LENGTH):
        """Function restricting used sequences x and y to <= max_lenght"""
        return tf.logical_and(tf.size(x) <= max_length,
                              tf.size(y) <= max_length)

    examples = examples.filter(filter_max_length)
    train_examples4backtrans = examples
    print('type of train_examples4backtrans: {}'.format(type(train_examples4backtrans)))
    print('shape of train_examples4backtrans: {}'.format(tf.data.experimental.cardinality(train_examples4backtrans)))
    dataset_length = [i for i, _ in enumerate(train_examples4backtrans)][-1] + 1

    def predict(inp_sentence):
      start_token = [tokenizer_en.vocab_size]
      end_token = [tokenizer_en.vocab_size + 1]

      # inp sentence is ENGLISH, hence adding the start and end token
      inp_sentence = start_token + tokenizer_en.encode(inp_sentence) + end_token
      encoder_input = tf.expand_dims(inp_sentence, 0)

      # as the target is GERMAN, the first word to the transformer should be the
      # english start token.
      decoder_input = [tokenizer_de.vocab_size]
      output = tf.expand_dims(decoder_input, 0)
      

      # predictions.shape == (batch_size, seq_len, vocab_size)
      def symbols_to_logits(output):          
          batched_input = tf.tile(encoder_input, [beam_width, 1])
          enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            batched_input, output)
          predictions, attention_weights = transformer2(batched_input,
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)
          predictions = predictions[:, -1, :]

          return  predictions
      
      finished_seq, finished_scores, states= beam_search(symbols_to_logits,
                 output,
                 beam_width,
                 MAX_LENGTH,
                 target_vocab_size,
                 alpha,
                 states=None,
                 eos_id=tokenizer_de.vocab_size+1,
                 stop_early=True,
                 use_tpu=False,
                 use_top_k_with_unique=True)
      
      return finished_seq[0, 0, :]

    def translate(sentence):
      result = predict(sentence)
      predicted_sentence = tokenizer_de.decode([i for i in result
                                                if i < tokenizer_de.vocab_size])

      print('Input: {}'.format(sentence))
      print('Predicted translation: {}'.format(predicted_sentence))
      return  predicted_sentence

    translations = []
    inputs = []
    targets = []
    BLEUs = []
    i = 0
    for sentence in train_examples4backtrans:
        # eng-> deu : hence indexes reversed
        inp = sentence[1].numpy().decode('utf-8')
        target = sentence[0].numpy().decode('utf-8')
        translation = translate(inp)
        BLEU = nltk.translate.bleu_score.sentence_bleu([nltk.word_tokenize(target)], nltk.word_tokenize(translation))
        translations.append(translation)
        inputs.append(inp)
        BLEUs.append(BLEU)
        print('Average BLEU score: ', 100 * np.mean(BLEUs))
        targets.append(target)
        # i+=1
        # store backtrans every 800 sentences
        # if i % 800 == 0:
        #     d = {'input': inputs, 'target': targets, 'translation': translations, 'BLEU': BLEUs}
        #     df = pd.DataFrame.from_dict(d)
        #     df.to_csv(os.path.join(output_path, 'results_backtrans_' + experiment_name + '_interm_'+str(i)+'.csv'))

    d = {'input': inputs, 'target': targets, 'translation': translations, 'BLEU': BLEUs}
    df = pd.DataFrame.from_dict(d)
    df.to_csv(os.path.join(output_path, 'results_backtrans_'+experiment_name+'.csv'))

    print('Average BLEU score: ', 100 * np.mean(BLEUs))

if __name__ == "__main__":
    evaluate_transformer()