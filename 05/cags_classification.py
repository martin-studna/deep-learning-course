#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

# 2f67b427-a885-11e7-a937-00505601122b
# c751264b-78ee-11eb-a1a9-005056ad4f31

import numpy as np
import tensorflow as tf

from cags_dataset import CAGS
import efficient_net
from tensorflow.keras.models import Model

from callback import NeptuneCallback



use_neptune = True
if use_neptune:
    import neptune
    neptune.init(project_qualified_name='amdalifuk/cags-classification')


# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--learning_rate", default=0.05, type=int, help="Learning rate.")
parser.add_argument("--use_lrplatau", default=False, type=int, help="Use LR decay on platau")

def main(args):
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)


    
    if use_neptune:        
        neptune.create_experiment(params={
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'epochs': args.epochs,
            'use_lrplatau': args.use_lrplatau
        },abort_callback=lambda: neptune.stop() )
        neptune.send_artifact('cags_classification.py')

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    cags = CAGS()
    l = 2142
    train = cags.train.map(lambda example: (example["image"], example["label"])).batch(args.batch_size)
    '''
    train = cags.train.map(lambda example: (example["image"], example["label"])).take(-1).map(
        lambda image, label: (tf.image.resize_with_crop_or_pad(image, cags.H + 20, cags.W + 20), label), num_parallel_calls=10
        ).cache()
    train = train.shuffle(l).map(
            lambda image, label: (tf.image.random_flip_left_right(image), label)
        ).map(
            lambda image, label: (tf.image.random_crop(image, size=[cags.H, cags.W,3]) , label) , num_parallel_calls=10
        ).batch(args.batch_size)
    '''
    dev = cags.dev.map(lambda example: (example["image"], example["label"])).take(-1).cache()
    dev = dev.batch(args.batch_size)

    test = cags.test.map(lambda example: (example["image"], example["label"]))
    test = test.batch(args.batch_size)


    # Load the EfficientNet-B0 model
    efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(include_top=False)
    efficientnet_b0.trainable= False
    x = tf.keras.layers.Dropout(0.5)(efficientnet_b0.output[0])
    x = tf.keras.layers.Dense( 1000, activation='relu' )(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense( 1000, activation='relu' )(x)
    x = tf.keras.layers.Dense( len(cags.LABELS), activation='softmax' )(x)
    # TODO: Create the model and train it
    model = Model(inputs=[efficientnet_b0.input], outputs=[x] )

    '''
    a1 = list( train.take(1) )[0][0][0].numpy()
    import cv2
    cv2.namedWindow('now', cv2.WINDOW_NORMAL)
    cv2.imshow("now", a1)
    cv2.waitKey(0)
    '''

    decay_steps = args.epochs * l / args.batch_size
    lr_decayed_fn = tf.keras.experimental.CosineDecay(args.learning_rate, decay_steps)

    model.compile(optimizer=tf.keras.optimizers.SGD(lr_decayed_fn, momentum=0.9, nesterov=True), 
    loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
    metrics=['SparseCategoricalAccuracy'] 
    )


    from tensorflow.keras.callbacks import ReduceLROnPlateau

    reduce = ReduceLROnPlateau(
        monitor = 'val_loss', 
        factor = 0.5, 
        patience = 4, 
        min_lr=0.00001,
        verbose=1,
        mode='min'
    ) 

    callback = []
    if use_neptune:  
        callback.append( NeptuneCallback() )  
    if args.use_lrplatau:
        callback.append(  reduce   )



    model.fit(train, validation_data=dev, epochs=args.epochs, callbacks=callback)

    fine_tune_at = 150
    for layer in model.layers[fine_tune_at:]:
        layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.SGD(lr_decayed_fn, momentum=0.9, nesterov=True), 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
        metrics=['SparseCategoricalAccuracy'] 
        )

    model.fit(train, validation_data=dev, epochs=args.epochs, callbacks=callback)


    # Generate test set annotations, but in args.logdir to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cags_classification.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the probabilities on the test set
        test_probabilities = model.predict(test)

        for probs in test_probabilities:
            print(np.argmax(probs), file=predictions_file)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
