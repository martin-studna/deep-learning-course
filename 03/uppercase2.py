#!/usr/bin/env python3
from tensorflow.keras.callbacks import Callback
from uppercase_data import UppercaseData
import tensorflow as tf
import numpy as np
import argparse
import datetime
import os
import re

import neptune
neptune.init(project_qualified_name='amdalifuk/c10')  # add your

# Report only TF errors by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

# 2f67b427-a885-11e7-a937-00505601122b
# c751264b-78ee-11eb-a1a9-005056ad4f31


# Fix random seeds and threads
np.random.seed(42)
tf.random.set_seed(42)
tf.config.threading.set_inter_op_parallelism_threads(16)
tf.config.threading.set_intra_op_parallelism_threads(16)

# Load data


# TODO: Implement a suitable model, optionally including regularization, select
# good hyperparameters and train the model.
#
# The inputs are _windows_ of fixed size (`args.window` characters on left,
# the character in question, and `args.window` characters on right), where
# each character is represented by a `tf.int32` index. To suitably represent
# the characters, you can:
# - Convert the character indices into _one-hot encoding_. There is no
#   explicit Keras layer, but you can
#   - use a Lambda layer which can encompass any function:
#       tf.keras.Sequential([
#         tf.keras.layers.InputLayer(input_shape=[2 * args.window + 1], dtype=tf.int32),
#         tf.keras.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet))),
#   - or use Functional API and then any TF function can be used
#     as a Keras layer:
#       inputs = tf.keras.layers.Input(shape=[2 * args.window + 1], dtype=tf.int32)
#       encoded = tf.one_hot(inputs, len(uppercase_data.train.alphabet))
#   You can then flatten the one-hot encoded windows and follow with a dense layer.
# - Alternatively, you can use `tf.keras.layers.Embedding` (which is an efficient
#   implementation of one-hot encoding followed by a Dense layer) and flatten afterwards.

# hloubka a počet parametrů, dropout
hidden_layers = [400]
dropout = 0.03  # True/False
l2 = 0.0001  # 0,0.1
bn = True  # True/False
lr = 0.0001
window = 4
alphabet_size = 70
label_smoothing = 0
loss = tf.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)  # ?
batch_size = 1000
afunkce = 'relu'
epochs = 20

uppercase_data = UppercaseData(window, alphabet_size)
labels = tf.keras.utils.to_categorical(uppercase_data.train.data["labels"])
labels_dev = tf.keras.utils.to_categorical(uppercase_data.dev.data["labels"])


def get_network():

    l1l2_regularizer = None

    if l2 != 0:
        l1l2_regularizer = tf.keras.regularizers.L1L2(l1=0, l2=l2)

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(
            input_shape=[2 * window + 1], dtype=tf.int32),
        tf.keras.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet)))])

    model.add(tf.keras.layers.Flatten())

    for hidden_layer in hidden_layers:
        model.add(tf.keras.layers.Dense(hidden_layer,
                  activation=afunkce, kernel_regularizer=l1l2_regularizer))

        if dropout != 0:
            model.add(tf.keras.layers.Dropout(dropout))

        if bn:
            model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(2,
              activation=tf.nn.softmax, kernel_regularizer=l1l2_regularizer))
    return model


model = get_network()
model.compile(
    optimizer=tf.optimizers.Adam(lr),
    loss=loss,
    metrics=[tf.metrics.CategoricalAccuracy(name="accuracy")],
)
model.summary()

PARAMS = {
    'hidden_layers': hidden_layers,
    'dropout': dropout,
    'afunkce': afunkce,
    'l2': l2,
    'bn': bn,
    'lr': lr,
    'window': window,
    'loss': loss,
    'batch_size': batch_size,
    'label_smoothing': label_smoothing,
    'alphabet_size': alphabet_size,
    'epochs': epochs,
}
neptune.create_experiment(params=PARAMS)
neptune.send_artifact('uppercase.ipynb')

# 13 epoch


class CustomCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):

        neptune.log_metric('loss', logs['loss'])
        neptune.log_metric('1-accuracy', 1-logs['accuracy'])

        if 'val_loss' in logs:
            neptune.log_metric('val_loss', logs['val_loss'])
            neptune.log_metric('1-val_accuracy', 1-logs['val_accuracy'])


model.fit(uppercase_data.train.data["windows"], labels, batch_size=batch_size, shuffle=True,
          validation_data=(uppercase_data.dev.data["windows"], labels_dev), epochs=epochs, callbacks=[CustomCallback()])


# TODO: Generate correctly capitalized test set.
# Use `uppercase_data.test.text` as input, capitalize suitable characters,
# and write the result to predictions_file (which is
# `uppercase_test.txt` in the `args.logdir` directory).

predictions = model.predict(uppercase_data.test.data["windows"])
text_result = list(uppercase_data.test.text)

with open("input.txt", "w", encoding="utf-8") as predictions_file:
    predictions_file.write(uppercase_data.test.text)

for i in range(len(predictions)):
    if predictions[i][1] >= 0.5:
        if text_result[i] == text_result[i].upper().lower():
            text_result[i] = text_result[i].upper()

text_result = "".join(text_result)
with open("uppercase_test.txt", "w", encoding="utf-8") as predictions_file:
    predictions_file.write(text_result)
