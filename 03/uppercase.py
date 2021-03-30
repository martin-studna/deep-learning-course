

from uppercase_data import UppercaseData
import tensorflow as tf
import numpy as np
import argparse
import datetime
import os
import re

# 2f67b427-a885-11e7-a937-00505601122b
# c751264b-78ee-11eb-a1a9-005056ad4f31

# Report only TF errors by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

np.random.seed(42)
tf.random.set_seed(42)
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

# hloubka a počet parametrů, dropout
hidden_layers = [1600, 1600]
dropout = 0.3  # True/False
l2 = 0.0001  # 0,0.1
bn = True  # True/False
lr = 0.0001
window = 5
alphabet_size = 100
label_smoothing = 0.0002
loss = tf.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)  # ?
batch_size = 1000
afunkce = 'relu'
epochs = 400

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

# 13 epoch
model.fit(uppercase_data.train.data["windows"], labels, batch_size=batch_size, shuffle=True,
          validation_data=(uppercase_data.dev.data["windows"], labels_dev), epochs=epochs)


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
