#!/usr/bin/env python3

import efficient_net
from cags_dataset import CAGS

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras
from tensorflow.keras.callbacks import Callback

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
parser.add_argument("--epochs", default=10, type=int,
                    help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int,
                    help="Maximum number of threads to use.")
parser.add_argument("--learning_rate", default=0.001,
                    type=int, help="Learning rate.")
parser.add_argument("--use_lrplatau", default=False,
                    type=int, help="Use LR decay on platau")
        


def main(args):
    #tf.compat.v1.disable_eager_execution()
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    '''
    os.environ['TF_DEVICE_MIN_SYS_MEMORY_IN_MB'] = '1050'
    os.environ['TF_XLA_FLAGS']='--tf_xla_enable_xla_devices --tf_xla_auto_jit=fusible --tf_xla_cpu_global_jit --tf_xla_always_defer_compilation=false --tf_xla_enable_lazy_compilation=true'

    os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION']='false'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'

    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

    print('jit:',tf.config.optimizer.get_jit())
    tf.config.optimizer.set_jit(True)
    print('jit:',tf.config.optimizer.get_jit())
    '''

    #tf.config.experimental.enable_tensor_float_32_execution(True) #16 s 15 bez???

    from tensorflow.keras import mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    #mixed_precision.set_global_policy(policy) #low acc (but quick)
    
    #tf.config.experimental.enable_mlir_graph_optimization()
    #tf.config.run_functions_eagerly
    #tf.compat.v1.enable_eager_execution()
    #tf.config.run_functions_eagerly(True)
    #tf.config.experimental_run_functions_eagerly(True)
    
    #tf.data.experimental.enable.debug_mode()
    print(f"EAGER: {tf.executing_eagerly()}")
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
    
    generator = tf.random.Generator.from_seed(args.seed)

    @tf.function
    def augment_take_mask(data):        
        return ( data['image'], data['mask'] )

    @tf.function
    def augment_onehot(image, mask):      
        oh = tf.one_hot(  tf.cast(mask*2, dtype=tf.uint8 ) ,2)
        return ( image, tf.reshape(  oh, (224,224,2)  )  )


    @tf.function
    def augment_bigger(image, mask):  
        image = tf.image.resize_with_crop_or_pad(image, cags.H + 30, cags.W + 30)
        mask = tf.image.resize_with_crop_or_pad(mask, cags.H + 30, cags.W + 30)
        return (image, mask)

    @tf.function
    def augment_flip(image, mask):  
        #random_int =  generator.uniform([], maxval=10, dtype=tf.int32)
        do_flip = generator.uniform([]) > 0.5
        #if random_int > 0.5:
        #tf.image.flip_left_right()
        #image = tf.image.random_flip_left_right(image, random_int )
        image = tf.cond(do_flip, lambda: tf.image.flip_left_right(image), lambda: image)
        mask = tf.cond(do_flip, lambda: tf.image.flip_left_right(mask), lambda: mask)


        return (image, mask)

    @tf.function
    def augment_crop(image, mask):  
        #random_int = generator.uniform([1]) *100
        #v = tf.Variable(0,dtype=tf.dtypes.uint64)

        #random_int = generator.uniform_full_int((), dtype=tf.dtypes.int32)
        #tf.image.random_crop(image, size=[cags.H, cags.W, 3 ] ,seed= random_int ) 
        #tf.raw_ops.StatefulUniformFullInt(resource=v, algorithm=[1], shape=[1], dtype=tf.dtypes.uint64, name=None)
        #v = tf.Variable(random_int, dtype=tf.int64)
        #tf.random.uniform(shape=(), minval=1, maxval=5, dtype=tf.int32)
        offset_h = generator.uniform([], maxval=tf.shape(image)[0] - cags.H + 1, dtype=tf.int32)
        offset_w = generator.uniform([], maxval=tf.shape(image)[1] - cags.W + 1, dtype=tf.int32)

        image = tf.image.crop_to_bounding_box(image, target_height=cags.H, target_width=cags.W, offset_height=offset_h, offset_width=offset_w)
        mask = tf.image.crop_to_bounding_box(mask, target_height=cags.H, target_width=cags.W, offset_height=offset_h, offset_width=offset_w)

        return (image, mask)

    def dataset_to_numpy(dset, test= False):
        a = list(dset.map(augment_take_mask).take(-1))
        xka = []
        yka = []
        for i in range(len(a)):
            xka.append(a[i][0])
            if not test:
                yka.append(a[i][1])
        nxka = np.array(xka)
        if not test:
            nyka = keras.utils.to_categorical(  np.array(yka) ,2 )
            return nxka, nyka
        else:
            return nxka

    train_x, train_y = dataset_to_numpy( cags.train )
    dev_x, dev_y = dataset_to_numpy( cags.dev )
    test_x = dataset_to_numpy( cags.test, True )

    train = tf.data.Dataset.from_tensor_slices((train_x,train_y))
    '''
    train = cags.train.map(augment_take_mask)
    train = train.map( augment_bigger , num_parallel_calls=10 ).cache()
    train = train.shuffle(100).map(augment_flip).map(augment_crop).batch(args.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    '''

    #train = cags.train.map(lambda example: (example["image"], example["mask"])).take(-1).cache()
    #train = train.batch(args.batch_size,drop_remainder=True)

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
    
    x = tf.keras.layers.Conv2DTranspose(256, 2, strides=2, padding='same', activation='relu')(efficientnet_b0.output[1])
    x = keras.layers.Concatenate()([x, efficientnet_b0.output[2]])
    x = keras.layers.Convolution2D(256, 3, padding='same')(x)

    x = tf.keras.layers.Conv2DTranspose(256, 2, strides=2, padding='same', activation='relu')(x)
    x = keras.layers.Concatenate()([x, efficientnet_b0.output[3]])
    x = keras.layers.Convolution2D(256, 3, padding='same')(x)
    

    x = tf.keras.layers.Conv2DTranspose(128, 2, strides=2, padding='same', activation='relu')(x)
    x = keras.layers.Concatenate()([x, efficientnet_b0.output[4]])
    x = keras.layers.Convolution2D(256, 3, padding='same')(x)
    

    x = tf.keras.layers.Conv2DTranspose(128, 2, strides=2, padding='same', activation='relu')(x)
    x = keras.layers.Concatenate()([x, efficientnet_b0.output[5]])
    

    x = tf.keras.layers.Conv2DTranspose(128, 2, strides=2, padding='same', activation='relu')(x)
    x = keras.layers.Convolution2D(128, 3, padding='same')(x)
    x = keras.layers.Convolution2D(128, 3, padding='same')(x)

    x = keras.layers.Convolution2D(2, 3, padding='same')(x)
    x = keras.layers.Activation('softmax')(x)

    # TODO: Create the model and train it
    model = Model(inputs=[efficientnet_b0.input], outputs=[x])


    '''
    dropout_rate = 0.5
    input_size = (512, 512, 1)

    # If you want to know more about why we are using `he_normal`:
    # https://stats.stackexchange.com/questions/319323/whats-the-difference-between-variance-scaling-initializer-and-xavier-initialize/319849#319849  
    # Or the excellent fastai course:
    # https://github.com/fastai/course-v3/blob/master/nbs/dl2/02b_initializing.ipynb
    initializer = 'he_normal'


    # -- Encoder -- #
    # Block encoder 1
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, UpSampling2D
    inputs = Input(shape=input_size)
    conv_enc_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)(inputs)
    conv_enc_1 = Conv2D(64, 3, activation = 'relu', padding='same', kernel_initializer=initializer)(conv_enc_1)

    # Block encoder 2
    max_pool_enc_2 = MaxPooling2D(pool_size=(2, 2))(conv_enc_1)
    conv_enc_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(max_pool_enc_2)
    conv_enc_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_enc_2)

    # Block  encoder 3
    max_pool_enc_3 = MaxPooling2D(pool_size=(2, 2))(conv_enc_2)
    conv_enc_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(max_pool_enc_3)
    conv_enc_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_enc_3)

    # Block  encoder 4
    max_pool_enc_4 = MaxPooling2D(pool_size=(2, 2))(conv_enc_3)
    conv_enc_4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(max_pool_enc_4)
    conv_enc_4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_enc_4)
    # -- Encoder -- #

    # ----------- #
    maxpool = MaxPooling2D(pool_size=(2, 2))(conv_enc_4)
    conv = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(maxpool)
    conv = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv)
    # ----------- #

    # -- Decoder -- #
    # Block decoder 1
    up_dec_1 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv))
    merge_dec_1 = concatenate([conv_enc_4, up_dec_1], axis = 3)
    conv_dec_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge_dec_1)
    conv_dec_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_1)

    # Block decoder 2
    up_dec_2 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv_dec_1))
    merge_dec_2 = concatenate([conv_enc_3, up_dec_2], axis = 3)
    conv_dec_2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge_dec_2)
    conv_dec_2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_2)

    # Block decoder 3
    up_dec_3 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv_dec_2))
    merge_dec_3 = concatenate([conv_enc_2, up_dec_3], axis = 3)
    conv_dec_3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge_dec_3)
    conv_dec_3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_3)

    # Block decoder 4
    up_dec_4 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv_dec_3))
    merge_dec_4 = concatenate([conv_enc_1, up_dec_4], axis = 3)
    conv_dec_4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge_dec_4)
    conv_dec_4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_4)
    conv_dec_4 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_4)
    # -- Dencoder -- #

    output = Conv2D(1, 1, activation = 'softmax')(conv_dec_4)

    '''








    print(model.summary())

    def save():
        # Generate test set annotations, but in args.logdir to allow parallel execution.
        with open("cags_segmentation.txt", "w", encoding="utf-8") as predictions_file:
            # TODO: Predict the masks on the test set
            test_masks = model.predict(test_x).argmax(axis=3)

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

    def show10():
        for i in range(10):
            shownp(i)

    def shownp(i):
        import cv2
        
        p1 = model.predict( np.array( [test_x[i]] )   )

        cv2.namedWindow('input', cv2.WINDOW_NORMAL)
        cv2.imshow("input", test_x[i])

        cv2.namedWindow('output', cv2.WINDOW_NORMAL)
        cv2.imshow("output", p1[0].argmax(axis=2).astype(np.float32).reshape((224,224)))
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

    class LRCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(self.model.optimizer._decayed_lr(np.float32))


    decay_steps = args.epochs * l / args.batch_size
    lr_decayed_fn = keras.experimental.CosineDecay( args.learning_rate, decay_steps, alpha=0.00001)

    meanIoUMetric = tf.keras.metrics.MeanIoU(  num_classes=len(cags.LABELS), name="MeanIoU-metric")

    globalIoULoss = tfa.losses.SigmoidFocalCrossEntropy(      )

    adam = tf.keras.optimizers.Adam(lr_decayed_fn)
    model.compile(optimizer=adam,
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=[keras.metrics.CategoricalAccuracy(), cags.MaskIoUMetric()]
                  )
    #shownp(0)
    
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

    model.fit(nxka, nyka, batch_size=args.batch_size, epochs=args.epochs//2, callbacks=[LRCallback()])

    #model.fit(train, validation_data=dev, epochs=args.epochs//2, callbacks=[LRCallback()])

    fine_tune_at = 0
    for layer in model.layers[fine_tune_at:]:
        layer.trainable = True

    print(K.eval(model.optimizer.lr))

    model.compile(optimizer=model.optimizer,
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=[keras.metrics.CategoricalAccuracy(), cags.MaskIoUMetric()]
                  )

    model.fit(nxka, nyka, batch_size=args.batch_size, epochs=args.epochs//2, callbacks=[LRCallback()])
    #model.fit(nxka, nyka, batch_size=args.batch_size, epochs=args.epochs//2, callbacks=[LRCallback()])

    
    save()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
