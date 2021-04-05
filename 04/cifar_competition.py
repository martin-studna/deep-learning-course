#!/usr/bin/env python3
import re
from tensorflow import keras

from tensorflow.python.ops.gen_math_ops import Add
from cifar10 import CIFAR10
import tensorflow as tf
import numpy as np
import argparse
import datetime
import os
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
from tensorflow.keras.layers import Input
from tensorflow.keras.regularizers import l2
from callback import NeptuneCallback
from sam import SAM, sam_train_step

use_neptune = False
if use_neptune:
    import neptune
    neptune.init(project_qualified_name='amdalifuk/cifar')

from os import environ


class MyModel(Model):
    def train_step(self, data):
        return sam_train_step(self, data)


environ["KERAS_BACKEND"] = "plaidml.keras.backend"
# 2f67b427-a885-11e7-a937-00505601122b
# c751264b-78ee-11eb-a1a9-005056ad4f31

# Report only TF errors by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--learning_rate", default=0.02,
                    type=int, help="Batch size.")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum.")
parser.add_argument("--l2", default=0.000, type=float,
                    help="L2 regularization.")
parser.add_argument("--epochs", default=100,
                    type=int, help="Number of epochs.")
parser.add_argument("--seed", default=44, type=int, help="Random seed.")
parser.add_argument("--threads", default=32, type=int,
                    help="Maximum number of threads to use.")


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
        },abort_callback=lambda: run_shutdown_logic_and_exit())

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub(
            "(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load data
    cifar = CIFAR10()
    
    y = tf.keras.utils.to_categorical(cifar.train.data["labels"])
    y_dev = tf.keras.utils.to_categorical(cifar.dev.data["labels"])
    
    predict = False  #jestli chceme trénovat nebo predikovat
    if not predict:
        # TODO: Create the model and train it
        v = 0.5
        #model = MyModel()
        input = Input(shape=(32, 32, 3))
        x = Conv2D(32//v, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(args.l2))(input)
        x = BatchNormalization()(x)
        x = Conv2D(32//v, (3, 3), activation='relu', kernel_regularizer=l2(args.l2), kernel_initializer='he_uniform', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(32//v, (3, 3), activation='relu', kernel_regularizer=l2(args.l2), kernel_initializer='he_uniform', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        r1 = x
        x = Dropout(0.2)(x)
        x = Conv2D(64//v, (3, 3), activation='relu', kernel_regularizer=l2(args.l2), kernel_initializer='he_uniform', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64//v, (3, 3), activation='relu', kernel_regularizer=l2(args.l2), kernel_initializer='he_uniform', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64//v, (3, 3), activation='relu', kernel_regularizer=l2(args.l2), kernel_initializer='he_uniform', padding='same')(x)
        #x = keras.layers.Concatenate()([x,r1])
        x = BatchNormalization()(x)        
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.3)(x)
        r2 = x
        x = Conv2D(128//v, (3, 3), activation='relu', kernel_regularizer=l2(args.l2), kernel_initializer='he_uniform', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128//v, (3, 3), activation='relu', kernel_regularizer=l2(args.l2), kernel_initializer='he_uniform', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128//v, (3, 3), activation='relu', kernel_regularizer=l2(args.l2), kernel_initializer='he_uniform', padding='same')(x)
        x = BatchNormalization()(x)
        #x = keras.layers.Concatenate()([x,r2])

        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.4)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu', kernel_regularizer=l2(args.l2), kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(10, activation='softmax')(x)

        #model = MyModel(inputs=[input], outputs=[x])
        model = Model(inputs=[input], outputs=[x])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss=tf.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=[tf.metrics.CategoricalAccuracy(name="accuracy")]
        )


        

        from tensorflow.keras.callbacks import ReduceLROnPlateau

        #model.fit(cifar.train.data["images"], y, epochs=args.epochs, verbose=1, callbacks=[NeptuneCallback()], validation_data=(
        #    cifar.dev.data["images"], y_dev))

        
        patient = 4
        reduce = ReduceLROnPlateau(
            monitor = 'loss', 
            factor = 0.5, 
            patience = patient, 
            min_lr=0.00001,
            verbose=1,
            mode='min'
        ) 
        if use_neptune:        
            callback = [NeptuneCallback(), reduce]
        else:
            callback = [reduce]
        
        '''
        model.fit(cifar.train.data["images"], y, epochs=args.epochs, verbose=1, callbacks=callback, validation_data=(
            cifar.dev.data["images"], y_dev), batch_size=args.batch_size)

        '''
        
        train = tf.data.Dataset.from_tensor_slices(
            (cifar.train.data["images"], y))

        generator = tf.random.Generator.from_seed(args.seed)

        w,h,c = cifar.train.data["images"][0].shape
        
        train = train.map(
        lambda image, label: (tf.image.resize_with_crop_or_pad(image, CIFAR10.H + 6, CIFAR10.W + 6), label), num_parallel_calls=10
        ).cache().shuffle(len(cifar.train.data["images"]), seed=args.seed).map(
            lambda image, label: (tf.image.random_flip_left_right(image), label)
        ).map(
            lambda image, label: (tf.image.random_crop(image, size=[CIFAR10.H, CIFAR10.W,3]) , label) , num_parallel_calls=10
        ).prefetch(len(cifar.train.data["images"])//2).batch(args.batch_size)
            
        model.fit(train, verbose=1, callbacks=callback,  validation_data=(
            cifar.dev.data["images"], y_dev), epochs=args.epochs, workers=10, use_multiprocessing=True, max_queue_size=len(cifar.train.data["images"]))
        

        # Generate test set annotations, but in args.logdir to allow parallel execution.
        # with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as predictions_file:
        #     for probs in model.predict(cifar.test.data["images"], batch_size=args.batch_size):
        #         print(np.argmax(probs), file=predictions_file)

        # Generate test set annotations, but in args.logdir to allow parallel execution.
        model.save('modeel3.h5')



        with open("cifar_competition_test.txt", "w", encoding="utf-8") as predictions_file:
            for probs in model.predict(cifar.test.data["images"], batch_size=args.batch_size):
                print(np.argmax(probs), file=predictions_file)

    else: #predictujeme
        from tensorflow.keras.models import load_model
        m1 = load_model('modeel2.h5')
        #m1.evaluate(cifar.dev.data["images"], y_dev,batch_size=args.batch_size)
        p1 = m1.predict(cifar.dev.data["images"],batch_size=args.batch_size)

        a1 = tf.metrics.CategoricalAccuracy()
        a1.update_state(p1, y_dev)
        print(   a1.result().numpy()    )

        m2 = load_model('modeel3.h5', custom_objects={'MyModel': MyModel})
        p2 = m2.predict(cifar.dev.data["images"],batch_size=args.batch_size)

        a2 = tf.metrics.CategoricalAccuracy()
        a2.update_state(p2, y_dev)
        print(   a2.result().numpy()    )

        m3 = load_model('modeel3_v1.h5')
        p3 = m3.predict(cifar.dev.data["images"],batch_size=args.batch_size)

        a3 = tf.metrics.CategoricalAccuracy()
        a3.update_state(p3, y_dev)
        print(   a3.result().numpy()    )

        p0 = p1+p2+p3
        a0 = tf.metrics.CategoricalAccuracy()
        a0.update_state(p0, y_dev)
        print(   a0.result().numpy()    )

        r1 = m1.predict(cifar.test.data["images"],batch_size=args.batch_size)
        r2 = m2.predict(cifar.test.data["images"],batch_size=args.batch_size)
        r3 = m3.predict(cifar.test.data["images"],batch_size=args.batch_size)
        r0 = r1+r2+r3


        with open("cifar_competition_test.txt", "w", encoding="utf-8") as predictions_file:
            for probs in r0:
                print(np.argmax(probs), file=predictions_file)
                
        pass


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
