#!/usr/bin/env python3
from callback import NeptuneCallback
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
from sam import sam_train_step
from os import environ
# Report only TF errors by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# 2f67b427-a885-11e7-a937-00505601122b
# c751264b-78ee-11eb-a1a9-005056ad4f31


class MyModel(tf.keras.models.Model):
    def train_step(self, data):
        return sam_train_step(self, data)


environ["KERAS_BACKEND"] = "plaidml.keras.backend"

use_neptune = True
if use_neptune:
    import neptune
    neptune.init(project_qualified_name='amdalifuk/cags-classification')


# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=3,
                    type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int,
                    help="Maximum number of threads to use.")
parser.add_argument("--learning_rate", default=0.01,
                    type=int, help="Learning rate.")
parser.add_argument("--use_lrplatau", default=True,
                    type=int, help="Use LR decay on platau")


def tfds_imgen(ds, imgen, batch_size=0, batches_per=0):
    for images, labels in ds:
        flow_ = imgen.flow(images, labels, batch_size=batch_size)
        for _ in range(batches_per):
            yield next(flow_)


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
            'threads': args.threads
        }, abort_callback=lambda: neptune.stop())

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
    model = MyModel(inputs=[efficientnet_b0.input], outputs=[x])

    model.compile(optimizer=tf.optimizers.Adam(learning_rate=args.learning_rate),
                  loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=[
        'SparseCategoricalAccuracy'])

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

    datagen = ImageDataGenerator(
        width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

    it_train = tfds_imgen(train.as_numpy_iterator(),
                          datagen, batch_size=32, batches_per=96)
    #steps = int(cags.train.example["image"].shape[0] / args.batch_size)

    model.fit(it_train, validation_data=dev, steps_per_epoch=20,
              epochs=args.epochs, callbacks=callback)
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
