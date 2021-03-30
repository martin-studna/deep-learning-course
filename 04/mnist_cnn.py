#!/usr/bin/env python3
from mnist import MNIST
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


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument(
    "--cnn", default='CB-8-3-5-valid,R-[CB-8-3-1-same,CB-8-3-1-same],F,H-50', type=str, help="CNN architecture.")  # default none
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False,
                    action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int,
                    help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.

# The neural network model


class Network(tf.keras.Model):
    def __init__(self, args):
        # TODO: Create the model. The template uses functional API, but
        # feel free to use subclassing if you want.
        inputs = tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C])
        x = inputs
        # TODO: Add CNN layers specified by `args.cnn`, which contains
        # comma-separated list of the following layers:
        # - `C-filters-kernel_size-stride-padding`: Add a convolutional layer with ReLU
        #   activation and specified number of filters, kernel size, stride and padding.
        # - `CB-filters-kernel_size-stride-padding`: Same as `C`, but use batch normalization.
        #   In detail, start with a convolutional layer without bias and activation,
        #   then add batch normalization layer, and finally ReLU activation.
        # - `M-pool_size-stride`: Add max pooling with specified size and stride, using
        #   the default "valid" padding.
        # - `R-[layers]`: Add a residual connection. The `layers` contain a specification
        #   of at least one convolutional layer (but not a recursive residual connection `R`).
        #   The input to the specified layers is then added to their output
        #   (after the ReLU nonlinearity of the last one).
        # - `F`: Flatten inputs. Must appear exactly once in the architecture.
        # - `H-hidden_layer_size`: Add a dense layer with ReLU activation and specified size.
        # - `D-dropout_rate`: Apply dropout with the given dropout rate.
        # You can assume the resulting network is valid; it is fine to crash if it is not.
        #
        # Produce the results in variable `hidden`.

        string_layers = list(args.cnn)

        zavorky = False
        for i in range(len(string_layers)):
            if string_layers[i] == '[':
                zavorky = True
            elif string_layers[i] == ']':
                zavorky = False

            if string_layers[i] == ',' and zavorky:
                string_layers[i] = ';'

        layers = "".join(string_layers).split(',')

        for layer in layers:
            params = layer.split('-')
            if params[0] == 'C':  # C-filters-kernel_size-stride-padding
                x = tf.keras.layers.Conv2D(int(params[1]), kernel_size=(int(params[2]), int(params[2])), strides=(
                    int(params[3]), int(params[3])), padding=params[4], activation='relu')(x)
            elif params[0] == 'CB':  # CB-filters-kernel_size-stride-padding
                x = tf.keras.layers.Conv2D(int(params[1]), kernel_size=(int(params[2]), int(
                    params[2])), strides=(int(params[3]), int(params[3])), padding=params[4], use_bias=False)(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.ReLU()(x)
            elif params[0] == 'M':  # M-pool_size-stride
                x = tf.keras.layers.MaxPool2D(pool_size=(int(params[1]), int(
                    params[1])), strides=(int(params[2]), int(params[2])))(x)
            elif params[0] == 'R':  # R-[C-16-3-1-same,C-16-3-1-same]
                residual = x

                layers_definition = layer[3:]
                layers_definition = layers_definition[:-1]

                res_params = layers_definition.split(';')

                for res_layer in res_params:
                    res_params = res_layer.split('-')
                    if res_params[0] == 'C':  # C-filters-kernel_size-stride-padding
                        x = tf.keras.layers.Conv2D(int(res_params[1]), kernel_size=(int(res_params[2]), int(res_params[2])), strides=(
                            int(res_params[3]), int(res_params[3])), padding=res_params[4], activation='relu')(x)
                    elif res_params[0] == 'CB':  # CB-filters-kernel_size-stride-padding
                        x = tf.keras.layers.Conv2D(int(res_params[1]), kernel_size=(int(res_params[2]), int(res_params[2])), strides=(
                            int(res_params[3]), int(res_params[3])), padding=res_params[4], use_bias=False)(x)
                        x = tf.keras.layers.BatchNormalization()(x)
                        x = tf.keras.layers.ReLU()(x)
                    elif res_params[0] == 'M':  # M-pool_size-stride
                        x = tf.keras.layers.MaxPool2D(pool_size=(int(res_params[1]), int(
                            res_params[1])), strides=(int(res_params[2]), int(res_params[2])))(x)
                #x = tf.keras.layers.Concatenate([x, residual])

                x = tf.keras.layers.Add()([x, residual])
            elif params[0] == 'F':  # F
                x = tf.keras.layers.Flatten()(x)
            elif params[0] == 'H':  # H-hidden_layer_size
                x = tf.keras.layers.Dense(int(params[1]), activation='relu')(x)
            elif params[0] == 'D':  # D-dropout_rate
                x = tf.keras.layers.Dropout(float(params[1]))(x)

        # Add the final output layer
        outputs = tf.keras.layers.Dense(
            MNIST.LABELS, activation=tf.nn.softmax)(x)

        super().__init__(inputs=inputs, outputs=outputs)
        self.compile(
            optimizer=tf.optimizers.Adam(),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        self.tb_callback = tf.keras.callbacks.TensorBoard(
            args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)
        # A hack allowing to keep the writers open.
        self.tb_callback._close_writers = lambda: None


def main(args):
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.recodex:
        tf.keras.utils.get_custom_objects(
        )["glorot_uniform"] = tf.initializers.GlorotUniform(seed=args.seed)
        tf.keras.utils.get_custom_objects(
        )["orthogonal"] = tf.initializers.Orthogonal(seed=args.seed)
        tf.keras.utils.get_custom_objects(
        )["uniform"] = tf.initializers.RandomUniform(seed=args.seed)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub(
            "(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    mnist = MNIST()

    # Create the network and train
    network = Network(args)
    network.fit(
        mnist.train.data["images"], mnist.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
        callbacks=[network.tb_callback],
    )

    # Compute test set accuracy and return it
    test_logs = network.evaluate(
        mnist.test.data["images"], mnist.test.data["labels"], batch_size=args.batch_size, return_dict=True,
    )
    network.tb_callback.on_epoch_end(
        args.epochs, {"val_test_" + metric: value for metric, value in test_logs.items()})

    return test_logs["accuracy"]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
