#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow_datasets as tfds
import tensorflow as tf
import time
import json
import datetime
import os.path
import argparse
from transformer import CustomSchedule, Transformer, create_masks

# region Setup Experiment parameters
parser = argparse.ArgumentParser()

# paths
checkpoint_path = "./checkpoints"
output_path = "./output"
data_path = './data'
log_path = './logs'

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

parser.add_argument("--experiment_name", type=str, default=current_time, help="Insert string defining your experiment. Defaults to datetime.now()")
parser.add_argument("--pretrained_name", type=str, default='',
                    help="""Name of experiment from which to load pretrained model, if any. New checkpoints will be saved to --experiment_name. To create
                    a pretrained model, train with RTL but set LAMBDA=0""")
# training parameters
parser.add_argument("--BUFFER_SIZE", type=int, default=4000, help="Train dataset buffer size.")
parser.add_argument("--BATCH_SIZE", type=int, default=64, help="Batch size used.")
parser.add_argument("--MAX_LENGTH", type=int, default=40, help="Only using training examples shorter than this. Original transformer uses 65")
parser.add_argument("--EPOCHS", type=int, default=15, help="Epochs to train for.")
parser.add_argument("--TRAIN_ON", type=int, default=100, help="Percentage of data to train on.")
parser.add_argument("--DICT_SIZE", type=int, default=2 ** 13, help="Size of dictionary. Original transformer uses 2**15")
parser.add_argument("--WARMUP_STEPS", type=int, default=4000, help="Number of warmup steps in learning rate cycle.")
parser.add_argument("--LAMBDA", type=float, default=0.25, help="Weight of round-trip loss component")
parser.add_argument("--NO_RTL", action='store_false', help="Don't use round-trip loss.")
# model hyperparameters
parser.add_argument("--num_layers", type=int, default=4, help="Layers used - base transformer uses 6.")
parser.add_argument("--d_model", type=int, default=128, help="d_model. Base transformer uses 512. Also enters learning rate schedule.")
parser.add_argument("--dff", type=int, default=512, help="dff - base transformer uses 2048.")
parser.add_argument("--num_heads", type=int, default=8, help="number of attention heads - base transformer uses 8.")
parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate.")

# read variables # todo clean up - can for sure be done more elegantly
ARGS = parser.parse_args()
experiment_name = ARGS.experiment_name
pretrained_name = ARGS.pretrained_name
BUFFER_SIZE = ARGS.BUFFER_SIZE
BATCH_SIZE = ARGS.BATCH_SIZE
MAX_LENGTH = ARGS.MAX_LENGTH
EPOCHS = ARGS.EPOCHS
TRAIN_ON = ARGS.TRAIN_ON
DICT_SIZE = ARGS.DICT_SIZE
WARMUP_STEPS = ARGS.WARMUP_STEPS
LAMBDA = ARGS.LAMBDA
USE_RTL = ARGS.NO_RTL # NO_RTL is store_false
    
num_layers = ARGS.num_layers
d_model = ARGS.d_model
dff = ARGS.dff
num_heads = ARGS.num_heads
dropout_rate = ARGS.dropout_rate


train_log_dir = os.path.normpath(log_path + '/' + experiment_name + '/train')
val_log_dir = os.path.normpath(log_path + '/' + experiment_name + '/val')
if not os.path.exists(train_log_dir):
    os.makedirs(train_log_dir)
if not os.path.exists(val_log_dir):
    os.makedirs(val_log_dir)

# save config of experiment in directory
checkpoint_path_pretrained = os.path.normpath(os.path.join(checkpoint_path, pretrained_name))
checkpoint_path = os.path.normpath(os.path.join(checkpoint_path, experiment_name))
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
config = vars(ARGS)
json.dump(config, open(os.path.join(checkpoint_path, 'config.json'), 'w'), indent=4, sort_keys=True)

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)
# endregion

def train():
    # region Functions
    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    def filter_max_length(x, y, max_length=MAX_LENGTH):
        """Function restricting used sequences x and y to <= max_lenght"""
        return tf.logical_and(tf.size(x) <= max_length,
                              tf.size(y) <= max_length)

    def encode(lang1, lang2):
        lang1 = [tokenizer_de.vocab_size] + tokenizer_de.encode(
            lang1.numpy()) + [tokenizer_de.vocab_size + 1]

        lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
            lang2.numpy()) + [tokenizer_en.vocab_size + 1]

        return lang1, lang2

    def tf_encode(de, en):
        return tf.py_function(encode, [de, en], [tf.int64, tf.int64])

    # endregion

    # region Create tokenizers
    # read previously created tokenizers if they exist
    if (os.path.isfile(os.path.join(output_path, "tokenizer_en_" + str(DICT_SIZE) + ".subwords")) &
            os.path.isfile(os.path.join(output_path, "tokenizer_de_" + str(DICT_SIZE) + ".subwords"))):

        tokenizer_en = tfds.features.text.SubwordTextEncoder.load_from_file(
            os.path.join(output_path, "tokenizer_en_" + str(DICT_SIZE)))
        tokenizer_de = tfds.features.text.SubwordTextEncoder.load_from_file(
            os.path.join(output_path, "tokenizer_de_" + str(DICT_SIZE)))
    else:
        # create tokenizers from scratch
        examples, metadata = tfds.load('wmt14_translate/de-en', data_dir=data_path, with_info=True,
                                       as_supervised=True)
        train_examples, val_examples = examples['train'], examples['validation']

        # English tokenizer
        tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for de, en in train_examples), target_vocab_size=DICT_SIZE)
        tokenizer_en.save_to_file(os.path.join(output_path, "tokenizer_en_" + str(DICT_SIZE)))

        # German tokenizer
        tokenizer_de = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (de.numpy() for de, en in train_examples), target_vocab_size=DICT_SIZE)
        tokenizer_de.save_to_file(os.path.join(output_path, "tokenizer_de_" + str(DICT_SIZE)))

    input_vocab_size = tokenizer_de.vocab_size + 2
    target_vocab_size = tokenizer_en.vocab_size + 2
    # endregion

    # region Prepare Train dataset
    split = tfds.Split.TRAIN.subsplit(tfds.percent[:TRAIN_ON])

    examples, metadata = tfds.load('wmt14_translate/de-en', data_dir=data_path, with_info=True,
                                   as_supervised=True, split=[split, 'validation'])
    train_examples, val_examples = examples[0], examples[1]

    train_dataset = train_examples.map(tf_encode)
    train_dataset = train_dataset.filter(filter_max_length)
    # cache the dataset to memory to get a speedup while reading from it.
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([-1], [-1]))
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    # endregion

    # region Prepare Validation dataset
    val_dataset = val_examples.map(tf_encode)
    val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE, padded_shapes=([-1], [-1]))
    # endregion

    # region Define Modelling setup
    learning_rate = CustomSchedule(d_model, warmup_steps=WARMUP_STEPS)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('val_accuracy')

    transformer1 = Transformer(num_layers, d_model, num_heads, dff,
                              input_vocab_size, target_vocab_size,
                              pe_input=input_vocab_size,
                              pe_target=target_vocab_size,
                              rate=dropout_rate)
    if USE_RTL:
        transformer2 = Transformer(num_layers, d_model, num_heads, dff,
                                      target_vocab_size, input_vocab_size,
                                      pe_input=target_vocab_size,
                                      pe_target=input_vocab_size,
                                      rate=dropout_rate)
        ckpt = tf.train.Checkpoint(transformer1=transformer1,
                                       transformer2=transformer2,
                                       optimizer=optimizer)
    else:
        ckpt = tf.train.Checkpoint(transformer1=transformer1,
                                       optimizer=optimizer)    
            
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    
    # endregion

    # region Train model
    if pretrained_name:
        latest = tf.train.latest_checkpoint(checkpoint_path_pretrained)
        if latest:
            ckpt.restore(latest)
            print('Pretrained model loaded from ' + latest + '.')
        else:
            raise Exception('Pretrained model not found.')
    elif ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored.')

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
        inp_inp = inp[:, :-1]
        inp_real = inp[:, 1:]
        enc_padding_mask1, combined_mask1, dec_padding_mask1 = create_masks(inp, tar_inp)
        enc_padding_mask2, combined_mask2, dec_padding_mask2 = create_masks(tar, inp_inp)
        with tf.GradientTape() as tape:
            predictions1, _ = transformer1(inp, tar_inp, True, enc_padding_mask1, combined_mask1, dec_padding_mask1)
            loss1 = loss_function(tar_real, predictions1) # this is de->en
            if USE_RTL:
                predictions2, _ = transformer2(tar, inp_inp, True, enc_padding_mask2, combined_mask2, dec_padding_mask2)
                loss2 = loss_function(inp_real, predictions2) # this is en->de
                loss = loss1 + loss2
                if LAMBDA>0:
                    predicted_id2 = tf.argmax(predictions2, axis=-1) # find most likely token from logits
                    inp2 = tf.concat([inp[:, 0:1], predicted_id2], axis=-1) # add start token. inp2 is \hat{s} in the paper
                    predicted_id1 = tf.argmax(predictions1, axis=-1) # find most likely token from logits
                    tar2 = tf.concat([tar[:, 0:1], predicted_id1], axis=-1) # add start token. tar22 is \hat{t} in the paper
                    enc_padding_mask3, combined_mask3, dec_padding_mask3 = create_masks(inp2, tar_inp)
                    enc_padding_mask4, combined_mask4, dec_padding_mask4 = create_masks(tar2, inp_inp)
                    predictions3, _ = transformer1(inp2, tar_inp, True, enc_padding_mask3, combined_mask3, dec_padding_mask3)
                    loss3 = loss_function(tar_real, predictions3) # predictions3 is \tilde{t} in the paper
                    predictions4, _ = transformer2(tar2, inp_inp, True, enc_padding_mask4, combined_mask4, dec_padding_mask4)
                    loss4 = loss_function(inp_real, predictions4) # predictions4 is \tilde{s} in the paper
                    loss += LAMBDA * (loss3 + loss4)            
            else:
                loss = loss1
        if USE_RTL:        
            gradients = tape.gradient(loss, [transformer1.trainable_variables, transformer2.trainable_variables])
            optimizer.apply_gradients(zip(gradients[0] + gradients[1], transformer1.trainable_variables + transformer2.trainable_variables))
        else:
            gradients = tape.gradient(loss, transformer1.trainable_variables)
            optimizer.apply_gradients(zip(gradients, transformer1.trainable_variables))
        
        train_loss(loss)
        train_accuracy(tar_real, predictions1)
    def val_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        inp_inp = inp[:, :-1]
        inp_real = inp[:, 1:]
        enc_padding_mask1, combined_mask1, dec_padding_mask1 = create_masks(inp, tar_inp)
        enc_padding_mask2, combined_mask2, dec_padding_mask2 = create_masks(tar, inp_inp)
        predictions1, _ = transformer1(inp, tar_inp, True, enc_padding_mask1, combined_mask1, dec_padding_mask1)
        loss1 = loss_function(tar_real, predictions1) # this is de->en
        if USE_RTL:
            predictions2, _ = transformer2(tar, inp_inp, True, enc_padding_mask2, combined_mask2, dec_padding_mask2)
            loss2 = loss_function(inp_real, predictions2) # this is en->de
            predicted_id2 = tf.argmax(predictions2, axis=-1) # find most likely token from logits
            inp2 = tf.concat([inp[:, 0:1], predicted_id2], axis=-1) # add start token. inp2 is \hat{s} in the paper
            predicted_id1 = tf.argmax(predictions1, axis=-1) # find most likely token from logits
            tar2 = tf.concat([tar[:, 0:1], predicted_id1], axis=-1) # add start token. tar22 is \hat{t} in the paper
            enc_padding_mask3, combined_mask3, dec_padding_mask3 = create_masks(inp2, tar_inp)
            enc_padding_mask4, combined_mask4, dec_padding_mask4 = create_masks(tar2, inp_inp)
            predictions3, _ = transformer1(inp2, tar_inp, True, enc_padding_mask3, combined_mask3, dec_padding_mask3)
            loss3 = loss_function(tar_real, predictions3) # predictions3 is \tilde{t} in the paper
            predictions4, _ = transformer2(tar2, inp_inp, True, enc_padding_mask4, combined_mask4, dec_padding_mask4)
            loss4 = loss_function(inp_real, predictions4) # predictions4 is \tilde{s} in the paper
            loss = loss1 + loss2 + LAMBDA * (loss3 + loss4)            
        else:
            loss = loss1
        val_loss(loss)
        val_accuracy(tar_real, predictions1)

    
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)
    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()
        # inp -> portuguese, tar -> english
        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)
            if batch % 50 == 0:
                print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, batch, train_loss.result(), train_accuracy.result()))
                with train_summary_writer.as_default():
                    tf.summary.scalar('train_loss', train_loss.result(), step=epoch)
                    tf.summary.scalar('train_accuracy', train_accuracy.result(), step=epoch)

        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))
        for (batch, (inp, tar)) in enumerate(val_dataset):
            val_step(inp, tar)
        with val_summary_writer.as_default():
            tf.summary.scalar('val_loss', val_loss.result(), step=epoch)
            tf.summary.scalar('val_accuracy', val_accuracy.result(), step=epoch)
  
        print('Epoch {} Val Loss {:.4f} Val Accuracy {:.4f}'.format(epoch + 1,
                                                            val_loss.result(),
                                                            val_accuracy.result()))

        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
    # endregion

if __name__ == "__main__":
    train()
