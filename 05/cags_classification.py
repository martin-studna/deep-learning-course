#!/usr/bin/env python3
from tensorflow.keras.models import Model
import efficient_net
from cags_dataset import CAGS
import tensorflow as tf
import numpy as np
import argparse
import datetime
import os
import re
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.regularizers import l2
from callback import NeptuneCallback
from sam import sam_train_step
import neptune
from os import environ
# neptune.init(project_qualified_name='amdalifuk/cags')
# Report only TF errors by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
environ["KERAS_BACKEND"] = "plaidml.keras.backend"


# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=201, type=int, help="Batch size.")
parser.add_argument("--epochs", default=3,
                    type=int, help="Number of epochs.")
parser.add_argument("--steps_per_epoch", default=10,
                    type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int,
                    help="Maximum number of threads to use.")


class SAMModel(Model):
    def train_step(self, data):
        return sam_train_step(self, data)


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

    train = cags.train.map(lambda example: (
        example["image"], example["label"]))
    train = train.shuffle(200).batch(args.batch_size)

    dev = cags.dev.map(lambda example: (example["image"], example["label"]))
    dev = dev.shuffle(200).batch(args.batch_size)

    test = cags.test.map(lambda example: (example["image"], example["label"]))
    test = test.batch(args.batch_size)

    # Load the EfficientNet-B0 model
    efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(
        include_top=False)
    efficientnet_b0.trainable = False

    x = tf.keras.layers.Dense(len(cags.LABELS), activation='softmax')(
        efficientnet_b0.output[0])
    # TODO: Create the model and train it
    model = SAMModel(inputs=[efficientnet_b0.input], outputs=[x])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    model.fit(train, validation_data=dev, steps_per_epoch=args.steps_per_epoch,
              epochs=args.epochs, batch_size=(args.epochs * args.steps_per_epoch))
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
