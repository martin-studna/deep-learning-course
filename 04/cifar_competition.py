#!/usr/bin/env python3
from cifar10 import CIFAR10
import tensorflow as tf
import numpy as np
import argparse
import datetime
import os
import re
# Report only TF errors by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=40, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5,
                    type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int,
                    help="Maximum number of threads to use.")


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

    # Load data
    cifar = CIFAR10()
    # TODO: Create the model and train it
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(
        input_shape=[32, 32, 3], batch_size=args.batch_size))
    model.add(tf.keras.layers.Conv2D(
        20, [3, 3], strides=2, padding='valid', activation='relu'))
    model.add(tf.keras.layers.Conv2D(
        10, [3, 3], strides=2, padding='valid', activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )

    model.fit(cifar.train.data["images"], cifar.train.data["labels"], batch_size=args.batch_size,
              epochs=args.epochs, validation_data=(cifar.dev.data["images"], cifar.dev.data["labels"]), shuffle=True)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    # with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as predictions_file:
    #     for probs in model.predict(cifar.test.data["images"], batch_size=args.batch_size):
    #         print(np.argmax(probs), file=predictions_file)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open("cifar_competition_test.txt", "w", encoding="utf-8") as predictions_file:
        for probs in model.predict(cifar.test.data["images"], batch_size=args.batch_size):
            print(np.argmax(probs), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
