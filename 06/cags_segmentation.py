#!/usr/bin/env python3

import efficient_net
from cags_dataset import CAGS

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras

from tensorflow.keras.models import Model
from callback import NeptuneCallback

import numpy as np
import argparse
import datetime
import os
from os import environ
import re

environ["KERAS_BACKEND"] = "plaidml.keras.backend"


# 2f67b427-a885-11e7-a937-00505601122b
# c751264b-78ee-11eb-a1a9-005056ad4f31


# Report only TF errors by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

use_neptune = False
if use_neptune:
    import neptune
    neptune.init(project_qualified_name='amdalifuk/cags-segmentation')


# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--epochs", default=40, type=int,
                    help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int,
                    help="Maximum number of threads to use.")
parser.add_argument("--learning_rate", default=0.001,
                    type=int, help="Learning rate.")
parser.add_argument("--use_lrplatau", default=False,
                    type=int, help="Use LR decay on platau")
        


def main(args):
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub(
            "(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    cags = CAGS()
    l = 2142
    '''
    train = cags.train.map(lambda example: (example["image"], example["mask"])).batch(
        args.batch_size).take(-1).cache()
    '''
    train = cags.train.map(lambda example: (example["image"], example["mask"])).take(-1).map(
        lambda image, mask: (tf.image.resize_with_crop_or_pad(image, cags.H + 50, cags.W + 50), mask), num_parallel_calls=10
    ).cache()
    train = train.shuffle(l).map(
        lambda image, mask: (tf.image.random_flip_left_right(image), mask)
    ).map(
        lambda image, mask: (tf.image.random_crop(image, size=[cags.H, cags.W, 3]), mask), num_parallel_calls=10
    ).batch(args.batch_size)

    #train = cags.train.map(lambda example: (
    #    example["image"], example["mask"])).take(-1).cache()
    #train = train.batch(args.batch_size)

    dev = cags.dev.map(lambda example: (
        example["image"], example["mask"])).take(-1).cache()
    dev = dev.batch(args.batch_size)

    test = cags.test.map(lambda example: (example["image"], example["mask"]))
    test = test.batch(args.batch_size)

    # Load the EfficientNet-B0 model
    #0:<KerasTensor: shape=(None, 1280) dtype=float32 (created by layer 'avg_pool')>
    #1:<KerasTensor: shape=(None, 7, 7, 1280) dtype=float32 (created by layer 'top_activation')>
    #2:<KerasTensor: shape=(None, 14, 14, 112) dtype=float32 (created by layer 'block5c_add')>
    #3:<KerasTensor: shape=(None, 28, 28, 40) dtype=float32 (created by layer 'block3b_add')>
    #4:<KerasTensor: shape=(None, 56, 56, 24) dtype=float32 (created by layer 'block2b_add')>
    #5:<KerasTensor: shape=(None, 112, 112, 16) dtype=float32 (created by layer 'block1a_project_bn')>

    efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(
        include_top=False)

    efficientnet_b0.trainable = False

    #x = keras.layers.Conv2D(512, 3, padding='same', activation='relu')(efficientnet_b0.output[1])

    x = tf.keras.layers.Conv2DTranspose(256, 3, strides=2, padding='same', activation='relu')(efficientnet_b0.output[1])
    x = keras.layers.Concatenate()([x, efficientnet_b0.output[2]])
    
    x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(x)
    x = keras.layers.Concatenate()([x, efficientnet_b0.output[3]])
    

    x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(x)
    x = keras.layers.Concatenate()([x, efficientnet_b0.output[4]])
    

    x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(x)
    x = keras.layers.Concatenate()([x, efficientnet_b0.output[5]])
    

    x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(x)
    x = keras.layers.Convolution2D(1, 3, padding='same')(x)
    x = keras.activations.sigmoid(x)

    # TODO: Create the model and train it
    model = Model(inputs=[efficientnet_b0.input], outputs=[x])
    print(model.summary())
    def save():
        # Generate test set annotations, but in args.logdir to allow parallel execution.
        with open("cags_segmentation.txt", "w", encoding="utf-8") as predictions_file:
            # TODO: Predict the masks on the test set
            test_masks = model.predict(test)

            for mask in test_masks:
                zeros, ones, runs = 0, 0, []
                for pixel in np.reshape(mask >= 0.5, [-1]):
                    if pixel:
                        if zeros or (not zeros and not ones):
                            runs.append(zeros)
                            zeros = 0
                        ones += 1
                    else:
                        if ones:
                            runs.append(ones)
                            ones = 0
                        zeros += 1
                runs.append(zeros + ones)
                print(*runs, file=predictions_file)
                
    def show():
        a = list( train.take(1) )[0]
        a1 = a[0][0].numpy()
        p1 = model.predict( np.array( [a1] )   )
        import cv2
        cv2.namedWindow('input', cv2.WINDOW_NORMAL)
        cv2.imshow("input", a1)

        m1 = a[1][0].numpy()
        cv2.namedWindow('inputmask', cv2.WINDOW_NORMAL)
        cv2.imshow("inputmask", m1)

        cv2.namedWindow('output', cv2.WINDOW_NORMAL)
        cv2.imshow("output", p1[0].reshape((224,224)))
        cv2.waitKey(0)

    from tensorflow.keras import backend as K
    smooth = 0.1
    def IOU_calc(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        
        return 2*(intersection + smooth) / (K.sum(y_true_f)  + K.sum(y_pred_f) + smooth)


    def IOU_calc_loss(y_true, y_pred):
        return -IOU_calc(y_true, y_pred)

    decay_steps = args.epochs * l / args.batch_size
    lr_decayed_fn = keras.experimental.CosineDecay( args.learning_rate, decay_steps, alpha=0.00001)

    meanIoUMetric = tf.keras.metrics.MeanIoU(  num_classes=len(cags.LABELS), name="MeanIoU-metric")

    globalIoULoss = tfa.losses.SigmoidFocalCrossEntropy(      )

    model.compile(optimizer=tf.keras.optimizers.Adam(lr_decayed_fn),
                  loss=globalIoULoss,
                  metrics=[keras.metrics.BinaryAccuracy(), cags.MaskIoUMetric()]
                  )

    from tensorflow.keras.callbacks import ReduceLROnPlateau

    reduce = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=0.00001,
        verbose=1,
        mode='min'
    )

    callback = []
    if use_neptune:
        callback.append(NeptuneCallback())
    if args.use_lrplatau:
        callback.append(reduce)

    model.fit(train, validation_data=dev, epochs=args.epochs)
    '''
    fine_tune_at = 150
    for layer in model.layers[fine_tune_at:]:
        layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.SGD(lr_decayed_fn, momentum=0.9, nesterov=True),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.Accuracy(), meanIoUMetric() ]
                  )

    model.fit(train, validation_data=dev,
              epochs=args.epochs, callbacks=callback)
    '''
    save()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
