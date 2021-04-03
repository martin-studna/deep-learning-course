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
        # Create a model with two inputs, both images of size [MNIST.H, MNIST.W, MNIST.C].

        images = (
            tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C]),
            tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C]),
        )

        # TODO: The model passes each input image through the same network (with shared weights), performing
        # - convolution with 10 filters, 3x3 kernel size, stride 2, "valid" padding, ReLU activation
        # - convolution with 20 filters, 3x3 kernel size, stride 2, "valid" padding, ReLU activation
        # - flattening layer
        # - fully connected layer with 200 neurons and ReLU activation
        # obtaining a 200-dimensional feature representation of each image.

        cmodel = tf.keras.Sequential()
        cmodel.add(tf.keras.layers.Conv2D(10, kernel_size=(3, 3),
                   strides=2, padding='valid', activation='relu'))
        cmodel.add(tf.keras.layers.Conv2D(20, kernel_size=(3, 3),
                   strides=2, padding='valid', activation='relu'))
        cmodel.add(tf.keras.layers.Flatten())
        cmodel.add(tf.keras.layers.Dense(200, activation='relu'))

        img1 = cmodel(images[0])
        img2 = cmodel(images[1])

        # TODO: Then, it should produce four outputs:
        # - first, compute _direct prediction_ whether the first digit is
        #   greater than the second, by
        #   - concatenating the two 200-dimensional image representations,
        #   - processing them using another 200-neuron ReLU dense layer
        #   - computing one output with `tf.nn.sigmoid` activation
        # - then, classify the computed representation of the first image using
        #   a densely connected softmax layer into 10 classes;
        # - then, classify the computed representation of the second image using
        #   the same connected layer (with shared weights) into 10 classes;
        # - finally, compute _indirect prediction_ whether the first digit
        #   is greater than second, by comparing the predictions from the above
        #   two outputs.
        dp = tf.keras.layers.Concatenate()([img1, img2])
        dp = tf.keras.layers.Dense(200, activation='relu')(dp)
        dp = tf.keras.layers.Dense(1, activation='sigmoid')(dp)

        klas_vrstva = tf.keras.layers.Dense(10, activation='softmax')
        i1 = klas_vrstva(img1)
        i2 = klas_vrstva(img2)

        outputs = {
            "direct_prediction": dp,
            "digit_1": i1,
            "digit_2": i2,
            "indirect_prediction": tf.argmax(i1, axis=1) > tf.argmax(i2, axis=1),
        }

        # Finally, construct the model.
        super().__init__(inputs=[images], outputs=outputs)

        # Note that for historical reasons, names of a functional model outputs
        # (used for displayed losses/metric names) are derived from the name of
        # the last layer of the corresponding output. Here we instead use
        # the keys of the `outputs` dictionary.
        self.output_names = sorted(outputs.keys())

        # TODO: Train the model by computing appropriate losses of
        # direct_prediction, digit_1, digit_2. Regarding metrics, compute
        # the accuracy of both the direct and indirect predictions; name both
        # metrics "accuracy" (i.e., pass "accuracy" as the first argument of
        # the metric object).
        self.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss={
                "direct_prediction": tf.keras.losses.BinaryCrossentropy(),
                "digit_1": tf.keras.losses.SparseCategoricalCrossentropy(),
                "digit_2": tf.keras.losses.SparseCategoricalCrossentropy(),
            },
            metrics={
                "direct_prediction": tf.metrics.BinaryAccuracy("accuracy"),
                "indirect_prediction": tf.metrics.BinaryAccuracy("accuracy"),
            },
        )

        self.tb_callback = tf.keras.callbacks.TensorBoard(
            args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)
        # A hack allowing to keep the writers open.
        self.tb_callback._close_writers = lambda: None

    # Create an appropriate dataset using the MNIST data.
    def create_dataset(self, mnist_dataset, args, training=False):
        # Start by using the original MNIST data
        dataset = tf.data.Dataset.from_tensor_slices(
            (mnist_dataset.data["images"], mnist_dataset.data["labels"]))

        # TODO: If `training`, shuffle the data with `buffer_size=10000` and `seed=args.seed`
        if training:
            dataset = dataset.shuffle(10000, seed=args.seed)
        # TODO: Combine pairs of examples by creating batches of size 2
        dataset = dataset.batch(2)
        # TODO: Map pairs of images to elements suitable for our model. Notably,
        # the elements should be pairs `(input, output)`, with
        # - `input` being a pair of images,
        # - `output` being a dictionary with keys digit_1, digit_2, direct_prediction
        #   and indirect_prediction.

        def create_element(images, labels):
            return ((images[0], images[1]), {'digit_1': labels[0],
                                             'digit_2': labels[1],
                                             'direct_prediction': tf.cast(labels[0] > labels[1], dtype=tf.float32), 'indirect_prediction': labels[0] > labels[1]})
        dataset = dataset.map(create_element)

        # TODO: Create batches of size `args.batch_size`
        dataset = dataset.batch(args.batch_size)
        return dataset


def main(args):
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    # tf.config.run_functions_eagerly(True)

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

    # Create the network
    network = Network(args)

    # Construct suitable datasets from the MNIST data.
    train = network.create_dataset(mnist.train, args, training=True)
    dev = network.create_dataset(mnist.dev, args)
    test = network.create_dataset(mnist.test, args)

    # Train
    network.fit(train, epochs=args.epochs, validation_data=dev,
                callbacks=[network.tb_callback])

    # Compute test set metrics and return them
    test_logs = network.evaluate(test, return_dict=True)
    network.tb_callback.on_epoch_end(
        args.epochs, {"val_test_" + metric: value for metric, value in test_logs.items()})

    return test_logs


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
