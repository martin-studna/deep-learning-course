#!/usr/bin/env python3
from uppercase_data import UppercaseData
import tensorflow as tf
import numpy as np
import argparse
import datetime
import os
import re
# Report only TF errors by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# 2f67b427-a885-11e7-a937-00505601122b
# c751264b-78ee-11eb-a1a9-005056ad4f31

# TODO: Set reasonable values for the hyperparameters, notably
# for `alphabet_size` and `window` and others.
parser = argparse.ArgumentParser()
parser.add_argument("--alphabet_size", default=62, type=int,
                    help="If nonzero, limit alphabet to this many most frequent chars.")
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--epochs", default=30,
                    type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int,
                    help="Maximum number of threads to use.")
parser.add_argument("--window", default=4, type=int,
                    help="Window size to use.")
parser.add_argument("--dropout", default=0, type=float,
                    help="Dropout regularization.")
parser.add_argument("--l2", default=0, type=float, help="L2 regularization.")
parser.add_argument("--label_smoothing", default=0,
                    type=float, help="Label smoothing.")
parser.add_argument(
    "--hidden_layers", default=[400], nargs="*", type=int, help="Hidden layer sizes.")


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
    uppercase_data = UppercaseData(args.window, args.alphabet_size)

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

    l1l2_regularizer = None
    if args.l2 != 0:
        l1l2_regularizer = tf.keras.regularizers.L1L2(l1=0, l2=args.l2)

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(
            input_shape=[2 * args.window + 1], dtype=tf.int32),
        tf.keras.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet)))])
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dropout(args.dropout))
    for hidden_layer in args.hidden_layers:
        model.add(tf.keras.layers.Dense(hidden_layer,
                  activation=tf.nn.relu, kernel_regularizer=l1l2_regularizer))
        model.add(tf.keras.layers.Dropout(args.dropout))

    model.add(tf.keras.layers.Dense(2,
              activation=tf.nn.softmax, kernel_regularizer=l1l2_regularizer))

    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.losses.CategoricalCrossentropy(
            label_smoothing=args.label_smoothing),
        metrics=[tf.metrics.CategoricalAccuracy(name="accuracy")],
    )

    labels = tf.keras.utils.to_categorical(uppercase_data.train.data["labels"])
    model.fit(uppercase_data.train.data["windows"],
              labels, batch_size=args.batch_size, epochs=args.epochs)

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


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
