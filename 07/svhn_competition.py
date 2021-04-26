#!/usr/bin/env python3
from svhn_dataset import SVHN
import efficient_net
import bboxes_utils
import tensorflow.keras as keras
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
import argparse
import datetime
import os
import re
import cv2
# Report only TF errors by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# 2f67b427-a885-11e7-a937-00505601122b
# c751264b-78ee-11eb-a1a9-005056ad4f31

# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=None, type=int, help="Batch size.")
parser.add_argument("--epochs", default=None,
                    type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int,
                    help="Maximum number of threads to use.")

def draw(img, bboxes):
    import cv2
    cv2.namedWindow("now", cv2.WINDOW_NORMAL)
    for i in range( len(bboxes) ):
        cv2.rectangle(  img, (int(bboxes[i][1]),int(bboxes[i][0])  ), (int(bboxes[i][3]),int(bboxes[i][2])), (255, 0, 0)  ) 
    
    cv2.imshow("now", img)
    cv2.waitKey(0)


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

    @tf.function
    def augment_bigger(image, bboxes):
        image = tf.image.resize(
            image, (120, 120))
        return (image, bboxes)

    # Load the data
    svhn = SVHN()
    #train = svhn.train.map(lambda data: (data['image'], data['bboxes'])).map(augment_bigger).take(20)
    train = svhn.train.map(lambda data: (data['image'], data['bboxes'], data['classes'])).take(-1)
    t = list(train.as_numpy_iterator())    
    
    test = svhn.test.map(lambda data: (data['image']) ).take(-1)
    test_list = list(test.as_numpy_iterator())

    #draw(t[0][0], t[0][1])

    my_anchors = []

    a1 = [0.4,0.3]

    for x in np.arange(0,1, 0.1):
        for y in np.arange(0,1, 0.1):
            my_anchors.append(  (y-a1[1], x-a1[0], y+a1[1], x+a1[0])    )

    my_anchors = np.array(my_anchors)
    #co je anchor
    #kde se veme anchor
    #co je bbox
    #kdy voláme bboxes_training
    #roi pooling

    strana = 46

    x_train = []

    all_cat_classes = []
    all_bboxes = []
    all_sample_weights = []

    for i in range(len(t)):
        h, w, c = t[i][0].shape
        nasobek_h = strana / h 
        nasobek_w = strana / w 
        img= cv2.resize(t[i][0], (46,46)  )
        x_train.append( img )
            
        g_boxes = np.array( t[i][1] )
        g_boxes[:,0] *= nasobek_h / strana
        g_boxes[:,2] *= nasobek_h / strana
        g_boxes[:,1] *= nasobek_w / strana
        g_boxes[:,3] *= nasobek_w / strana

        classes, bboxesy = bboxes_utils.bboxes_training( my_anchors, t[i][2], g_boxes , 0.2 )
        #draw(img, g_boxes * strana)
        cat_classes = keras.utils.to_categorical(  classes, 11    )

        all_bboxes.append(bboxesy)
        all_cat_classes.append(cat_classes)
        all_sample_weights.append(( cat_classes.argmax(axis=1) > 0).astype(np.float32) )

    x_train = np.array(x_train)
    all_cat_classes = np.array(all_cat_classes)
    all_bboxes = np.array(all_bboxes)
    all_sample_weights = np.array(all_sample_weights)

    x_test = []
    for i in range(len(test_list)):
        x_test.append( cv2.resize(test_list[i], (46,46)  ) )
    x_test = np.array(x_test)

    

    # Load the EfficientNet-B0 model
    efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(
        include_top=False)

    # TODO: Create the model and train it
    input_l = keras.layers.Input(shape=(46,46,3))
    o0, o1, o2, o3, o4, o5*_ = efficientnet_b0(input_l)
    x = keras.layers.Conv2D(32, 7, activation='relu')(input_l)
    x = keras.layers.Conv2D(32, 5, activation='relu')(x)
    x = keras.layers.Conv2D(32, 7, strides=2, activation='relu' )(x)
    x = keras.layers.Conv2D(32, 3, activation='relu' )(x)
    x = keras.layers.Conv2D(32, 4, activation='relu' )(x)

    classes = keras.layers.Conv2D(11, 11, padding='same', activation='relu')(x)
    classes = keras.layers.Reshape((100,11))(classes)
    classes = keras.layers.Activation('softmax', name="classes_output")(classes)
    #classes = keras.layers.Flatten(name="classes_output")(classes)


    bboxes = keras.layers.Conv2D(4, 11, padding='same', activation='relu')(x)
    bboxes = keras.layers.Reshape((100,4), name="bboxes_output")(bboxes)

    model = keras.models.Model(inputs=[input_l], outputs=[classes, bboxes]   )
    model.summary()

    def my_loss_fn(y_true, y_pred):
        squared_difference = tf.square(y_true - y_pred)
        return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`
        
    losses = {
        #'classes_output':  tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE) ,
        'classes_output':   tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE) ,
        'bboxes_output':  tf.keras.losses.Huber(),
        } 

    metrics = {
        'classes_output': keras.metrics.CategoricalAccuracy(),
        } 

    model.compile(optimizer=keras.optimizers.Adam(0.0001), 
    loss=  losses, 
    metrics=metrics , 
    run_eagerly=True    )



    model.fit( x_train,  { 'classes_output': all_cat_classes , 'bboxes_output': all_bboxes    } ,batch_size=20, epochs=10, sample_weight=[all_sample_weights,all_sample_weights] )
    #model.fit( x_train,  { 'classes_output': all_cat_classes, 'bboxes_output': all_bboxes } ,batch_size=50, epochs=20 )
    #https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/bin/train.py

    #model.predict( np.array( [ t[0][0]] )  )[0].argmax(axis=2)  

    model.save('model.h5')
    
    model = keras.models.load_model('model.h5')

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open("svhn_competition.txt", "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the digits and their bounding boxes on the test set.
        # Assume that for a single test image we get
        # - `predicted_classes`: a 1D array with the predicted digits,
        # - `predicted_bboxes`: a [len(predicted_classes), 4] array with bboxes;

        tpredicted_classes, tpredicted_bboxes = model.predict(x_test, batch_size=32)
    
        #for predicted_classes, predicted_bboxes in zip(tpredicted_classes, tpredicted_bboxes):
        for i in range(len(tpredicted_classes)):
            scores = tpredicted_classes[i].max(axis=1)
            predicted_classes = tpredicted_classes[i].argmax(axis=1)
            predicted_bboxes = tpredicted_bboxes[i] 
            predicted_bboxess = bboxes_utils.bboxes_from_fast_rcnn( my_anchors ,predicted_bboxes) * 46

            selected_indices = tf.image.non_max_suppression(predicted_bboxess, scores, 5).numpy()
            #selected_boxes = tf.gather(predicted_bboxess, selected_indices)
            selected_boxes = predicted_bboxess[ selected_indices]


            if i < 10:
                draw(x_test[i], selected_boxes)

            output = ""
            for label, bbox in zip(predicted_classes, selected_boxes):
                #output += [label] + bbox
                if label != 0:
                    output += str(label-1)+ " " + str(int(bbox[0]))+ " " +  str(int(bbox[1]))+ " " +  str(int(bbox[2]))+ " " +  str(int(bbox[3])) + " " 
            print(*output, file=predictions_file, sep='')
            #print(' '.join(output), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
