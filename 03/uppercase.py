#!/usr/bin/env python3
from uppercase_data import UppercaseData
import tensorflow as tf
import numpy as np
import argparse
import datetime
import os
import re

import neptune

neptune.init(project_qualified_name='amdalifuk/c10', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiZjkxYTA4NzYtMzk5OS00NDFmLTlmMWItNGNhNjU5NmQ1NDMxIn0=') # add your 

# Report only TF errors by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

np.random.seed(42)
tf.random.set_seed(42)
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

#hloubka a počet parametrů, dropout
hidden_layers = [1600,1600]
dropout = 0.2 #True/False
l2 = 0 #0,0.1
bn = True #True/False
lr = 0.001
window = 5
alphabet_size = 100
label_smoothing = 0.1
loss = tf.losses.CategoricalCrossentropy( label_smoothing=label_smoothing) #?
batch_size = 5000
afunkce= 'relu'
epochs = 40

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
            model.add(tf.keras.layers.BatchNormalization() )
            
            

    model.add(tf.keras.layers.Dense(2,
              activation=tf.nn.softmax, kernel_regularizer=l1l2_regularizer))
    return model
    
model = get_network()
model.compile(
        optimizer=tf.optimizers.Adam( lr ),
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
neptune.create_experiment(params=PARAMS,abort_callback=lambda: run_shutdown_logic_and_exit())
neptune.send_artifact('uppercase.py')

#13 epoch
from tensorflow.keras.callbacks import Callback
class CustomCallback(Callback):        
    def on_epoch_end(self, epoch, logs=None):

        neptune.log_metric('loss', logs['loss'])
        neptune.log_metric('1-accuracy', 1-logs['accuracy'])
        
        if 'val_loss' in logs:
            neptune.log_metric('val_loss', logs['val_loss'])
            neptune.log_metric('1-val_accuracy', 1-logs['val_accuracy'])
            
model.fit(uppercase_data.train.data["windows"], labels, batch_size=batch_size, shuffle=True,
          validation_data=(uppercase_data.dev.data["windows"], labels_dev), epochs=epochs, callbacks=[CustomCallback()])


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
