#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1") # Report only TF errors by default

import numpy as np
import tensorflow as tf

from modelnet import ModelNet
from callback import NeptuneCallback

# 2f67b427-a885-11e7-a937-00505601122b
# c751264b-78ee-11eb-a1a9-005056ad4f31

use_neptune = True
if use_neptune:
    import neptune
    neptune.init(project_qualified_name='amdalifuk/3d-recognition')


# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
parser.add_argument("--modelnet", default=20, type=int, help="ModelNet dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--learning_rate", default=0.005, type=int, help="Learning rate.")

parser.add_argument("--use_lrplatau", default=False, type=int, help="Use LR decay on platau")

def main(args):
    import tensorflow.keras as keras

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)


    if use_neptune:
        neptune.create_experiment(params={
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'epochs': args.epochs,
            'use_lrplatau': args.use_lrplatau
        }, abort_callback=lambda: neptune.stop())
        neptune.send_artifact('3d_recognition.py')


    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    modelnet = ModelNet(args.modelnet)

    # TODO: Create the model and train it
    classes_num = len(modelnet.LABELS)
    #modelnet.train.data['voxels']
    #modelnet.train.data['labels']
    x = modelnet.train.data['voxels']
    y = keras.utils.to_categorical( modelnet.train.data['labels'] , classes_num) 
    x_dev = modelnet.dev.data['voxels']
    y_dev = keras.utils.to_categorical( modelnet.dev.data['labels'] , classes_num) 

    input_l = keras.Input(shape=(x[0].shape[0], x[0].shape[1], x[0].shape[2], x[0].shape[3] ) )
    l = keras.layers.Conv3D( 64, 3 )(input_l)
    l = keras.layers.BatchNormalization()(l)
    l = keras.layers.ReLU()(l)    
    
    l = keras.layers.Conv3D( 64, 3 )(l)
    l = keras.layers.BatchNormalization()(l)
    l = keras.layers.ReLU()(l)    
    
    l = keras.layers.MaxPool3D()(l)

    l = keras.layers.Conv3D( 64, 3 )(l)
    l = keras.layers.BatchNormalization()(l)
    l = keras.layers.ReLU()(l)

    l = keras.layers.Conv3D( 128, 3 )(l)
    l = keras.layers.BatchNormalization()(l)
    l = keras.layers.ReLU()(l)


    l = keras.layers.Conv3D( 128, 3 )(l)
    l = keras.layers.BatchNormalization()(l)
    l = keras.layers.ReLU()(l)
    

    l = keras.layers.Flatten()(l)
    l = keras.layers.Dropout(0.2)(l)
    l = keras.layers.Dense(classes_num, activation='softmax')(l)

    model = tf.keras.Model( inputs=[input_l], outputs=[l]    )


    model.compile(optimizer=keras.optimizers.Adam(  learning_rate=args.learning_rate ),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=[keras.metrics.CategoricalAccuracy()]
    )

    callbacky = []
    if use_neptune:
        callbacky.append(NeptuneCallback())


    model.fit(x, y, validation_data=(x_dev, y_dev), epochs=args.epochs, batch_size=args.batch_size, shuffle=True  )


    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open( "3d_recognition.txt", "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the probabilities on the test set
        test_probabilities = model.predict(modelnet.test.data['voxels'])

        for probs in test_probabilities:
            print(np.argmax(probs), file=predictions_file)
    neptune.send_artifact('3d_recognition.txt')
    

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
