#!/usr/bin/env python3
from svhn_dataset import SVHN
import efficient_net
import bboxes_utils
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import numpy as np
import argparse
import datetime
import os
import re
from tensorflow.keras.models import Model
# Report only TF errors by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10,
                    type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int,
                    help="Maximum number of threads to use.")


class LRCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(self.model.optimizer._decayed_lr(np.float32))


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
    svhn = SVHN()

    train = svhn.train

    for item in train:
        image_grid = item["image"][0:-1:10, 0:-1:10]

    dev = svhn.dev.map(lambda example: (
        example["image"], example["classes"], example["bboxes"]))
    dev = dev.padded_batch(args.batch_size)

    test = svhn.test.map(lambda example: (
        example["image"], example["classes"], example["bboxes"]))
    test = test.padded_batch(args.batch_size)

    # Load the EfficientNet-B0 model
    efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(
        include_top=False, dynamic_input_shape=True)

    # TODO: Create the model and train it

    x = tf.keras.layers.Convolution2D(
        2, 3, padding='same', activation='sigmoid')(efficientnet_b0.output[1])

    model = Model(inputs=[efficientnet_b0.input], outputs=[x])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tfa.losses.SigmoidFocalCrossEntropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy()]
                  )

    model.fit(train, validation_data=dev,
              epochs=args.epochs, callbacks=[LRCallback()])

    predictions = model.predict()

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "svhn_competition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the digits and their bounding boxes on the test set.
        # Assume that for a single test image we get
        # - `predicted_classes`: a 1D array with the predicted digits,
        # - `predicted_bboxes`: a [len(predicted_classes), 4] array with bboxes;
        for predicted_classes, predicted_bboxes in ...:
            output = []
            for label, bbox in zip(predicted_classes, predicted_bboxes):
                output += [label] + bbox
            print(*output, file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
