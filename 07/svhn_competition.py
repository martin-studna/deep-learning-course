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
    im = np.array(img)
    for i in range( len(bboxes) ):
        cv2.rectangle(  im, (int(bboxes[i][1]),int(bboxes[i][0])  ), (int(bboxes[i][3]),int(bboxes[i][2])), (255, 0, 0), 3  ) 
    cv2.imshow("now", im)
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
    
    dev = svhn.dev.map(lambda data: (data['image']) ).take(-1)
    dev_list = list(dev.as_numpy_iterator())

    test = svhn.test.map(lambda data: (data['image']) ).take(-1)
    test_list = list(test.as_numpy_iterator())

    #draw(t[0][0], t[0][1])

    my_anchors = []
    '''
    000:array([-0.25, -0.15,  0.25,  0.15])
    001:array([-0.17307692, -0.15      ,  0.32692308,  0.15      ])
    002:array([-0.09615385, -0.15      ,  0.40384615,  0.15      ])
    003:array([-0.01923077, -0.15      ,  0.48076923,  0.15      ])
    004:array([ 0.05769231, -0.15      ,  0.55769231,  0.15      ])
    '''

    a1 = [0.15,0.2]
    a2 = [0.2,0.3]
    a3 = [0.3,0.6]
    a4 = [0.4,0.75]
    a5 = [0.5,0.85]
    anchors_count = 5
    for y in np.linspace(0,1,14):
        for x in np.linspace(0,1,14):
            my_anchors.append(  (y-a1[1]/2, x-a1[0]/2, y+a1[1]/2, x+a1[0]/2 )    )
            my_anchors.append(  (y-a2[1]/2, x-a2[0]/2, y+a2[1]/2, x+a2[0]/2 )    )
            my_anchors.append(  (y-a3[1]/2, x-a3[0]/2, y+a3[1]/2, x+a3[0]/2 )    )
            my_anchors.append(  (y-a4[1]/2, x-a4[0]/2, y+a4[1]/2, x+a4[0]/2 )    )
            my_anchors.append(  (y-a5[1]/2, x-a5[0]/2, y+a5[1]/2, x+a5[0]/2 )    )

    my_anchors = np.array(my_anchors)
    #co je anchor
    #kde se veme anchor
    #co je bbox
    #kdy voláme bboxes_training
    #roi pooling

    strana = 224

    x_train = []

    all_cat_classes = []
    all_bboxes = []
    all_sample_weights = []

    all_g_boxes = []
    x_train_nasobky = []

    for i in range(len(t)):
        h, w, c = t[i][0].shape
        nasobek_h = strana / h 
        nasobek_w = strana / w 
        x_train_nasobky.append([nasobek_h, nasobek_w])
        img= cv2.resize(t[i][0], (strana,strana)  )
        x_train.append( img )
            
        g_boxes = np.array( t[i][1] )
        g_boxes[:,0] *= nasobek_h / strana
        g_boxes[:,2] *= nasobek_h / strana
        g_boxes[:,1] *= nasobek_w / strana
        g_boxes[:,3] *= nasobek_w / strana

        '''
        for j in range(len(g_boxes)):
            all_g_boxes.append(g_boxes[j])
        continue
        '''
        classes, bboxesy = bboxes_utils.bboxes_training( my_anchors, t[i][2], g_boxes , 0.5 )
        #draw(img, g_boxes * strana)

        cat_classes = keras.utils.to_categorical(  classes, 11    )
        #if i < 10:
        #    draw(img, my_anchors[ cat_classes.argmax(axis=1) > 0] * strana)
        #draw(img, my_anchors[7+7*14:8+7*14] * strana)


        all_bboxes.append(bboxesy)
        all_cat_classes.append(cat_classes)
        all_sample_weights.append(( cat_classes.argmax(axis=1) > 0).astype(np.float32) )

    x_train = np.array(x_train)
    all_cat_classes = np.array(all_cat_classes)
    all_bboxes = np.array(all_bboxes)
    all_sample_weights = np.array(all_sample_weights)

    x_dev = []
    x_dev_nasobky = []
    for i in range(len(dev_list)):
        h, w, c = dev_list[i].shape
        nasobek_h = strana / h 
        nasobek_w = strana / w 
        x_dev.append( cv2.resize(dev_list[i], (strana,strana)  ) )
        x_dev_nasobky.append([nasobek_h, nasobek_w])
    x_dev = np.array(x_dev)

    x_test = []
    x_test_nasobky = []
    for i in range(len(test_list)):
        h, w, c = test_list[i].shape
        nasobek_h = strana / h 
        nasobek_w = strana / w         
        x_test.append( cv2.resize(test_list[i], (strana,strana)  ) )
        x_test_nasobky.append([nasobek_h, nasobek_w])
    x_test = np.array(x_test)


    # Load the EfficientNet-B0 model
    efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(
        include_top=False)
    efficientnet_b0.trainable = False    
    
    
    # TODO: Create the model and train it
    input_l = keras.layers.Input(shape=(strana,strana,3))
    o0, o1, o2, o3, o4, o5, *_ = efficientnet_b0(input_l)
    x = o2

    #classes = keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    #classes = keras.layers.Conv2D(256, 3, padding='same', activation='relu')(classes)
    #classes = keras.layers.Conv2D(256, 3, padding='same', activation='relu')(classes)
    classes = keras.layers.Conv2D(256, 3, padding='same')(x)
    classes = keras.layers.BatchNormalization()(classes)
    classes = keras.layers.Activation('relu')(classes)

    classes = keras.layers.Conv2D(256, 3, padding='same')(classes)
    classes = keras.layers.BatchNormalization()(classes)
    classes = keras.layers.Activation('relu')(classes)

    classes = keras.layers.Conv2D(256, 3, padding='same')(classes)
    classes = keras.layers.BatchNormalization()(classes)
    classes = keras.layers.Activation('relu')(classes)


    classes = keras.layers.Conv2D(10*anchors_count, 3, padding='same', activation='sigmoid')(classes)
    classes = keras.layers.Reshape((14*14*anchors_count,10), name="classes_output" )(classes)
    #classes = keras.layers.Activation('sigmoid')(classes)
    #classes = keras.layers.Flatten(name="classes_output")(classes)


    bboxes = keras.layers.Conv2D(256, 3, padding='same')(x)
    bboxes = keras.layers.BatchNormalization()(bboxes)
    bboxes = keras.layers.Activation('relu')(bboxes)

    bboxes = keras.layers.Conv2D(256, 3, padding='same')(bboxes)
    bboxes = keras.layers.BatchNormalization()(bboxes)
    bboxes = keras.layers.Activation('relu')(bboxes)

    bboxes = keras.layers.Conv2D(256, 3, padding='same')(bboxes)
    bboxes = keras.layers.BatchNormalization()(bboxes)
    bboxes = keras.layers.Activation('relu')(bboxes)


    bboxes = keras.layers.Conv2D(4*anchors_count, 3, padding='same')(bboxes)
    bboxes = keras.layers.Reshape((14*14*anchors_count,4), name="bboxes_output")(bboxes)

    model = keras.models.Model(inputs=[input_l], outputs=[classes, bboxes]   )
    model.summary()

    losses = {
        #'classes_output':  tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE) ,
        'classes_output':   tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE) , #<5% = backgroud!!!???
        'bboxes_output':  tf.keras.losses.Huber(),
        } 

    metrics = {
        'classes_output': keras.metrics.BinaryAccuracy(),
        } 

    batch_size = 32
    epochs = 20
    decay_steps = epochs * len(t) / batch_size
    lr_decayed_fn = keras.experimental.CosineDecay( 0.01, decay_steps, alpha=0.00001)
    model.compile(optimizer=keras.optimizers.Adam(lr_decayed_fn), 
    loss=  losses, 
    metrics=metrics , 
    run_eagerly=False    )

    #sample weight shape: (10000, 196)

    #pozadí 0
    #1 0 0 0 0 0 0 0 0 0 0 
    #číslo 2
    #0 0 0 1 0 0 0 0 0 0 0 




    def save_data(set= x_dev, nasobky=x_dev_nasobky, fname='dev',max_rois = 5, iou_threshold = 0.2, score_threshold = 0.2, evaluate=False):
        print(f'max_rois = {max_rois}, iou_threshold = {iou_threshold}, score_threshold = {score_threshold}')
        with open(f"svhn_competition_{fname}.txt", "w", encoding="utf-8") as predictions_file:
            # TODO: Predict the digits and their bounding boxes on the test set.
            # Assume that for a single test image we get
            # - `predicted_classes`: a 1D array with the predicted digits,
            # - `predicted_bboxes`: a [len(predicted_classes), 4] array with bboxes;

            tpredicted_classes, tpredicted_bboxes = model.predict(x_dev, batch_size=32) #DEV TEST
        
            #for predicted_classes, predicted_bboxes in zip(tpredicted_classes, tpredicted_bboxes):
            for i in range(len(tpredicted_classes)):
                scores = tpredicted_classes[i,:,1:].max(axis=1)
                predicted_classes = tpredicted_classes[i,:,1:].argmax(axis=1)
                predicted_bboxes = tpredicted_bboxes[i] 

                predicted_bboxess = bboxes_utils.bboxes_from_fast_rcnn( my_anchors ,predicted_bboxes) * strana
                
                selected_indices = tf.image.non_max_suppression(predicted_bboxess, scores, max_rois, iou_threshold=iou_threshold, score_threshold=score_threshold).numpy()
                #selected_boxes = tf.gather(predicted_bboxess, selected_indices)
                selected_boxes = predicted_bboxess[ selected_indices]
                selected_scores = scores[ selected_indices]
                selected_predicted_classes = predicted_classes[ selected_indices]

                orig_selected_boxes = np.array(selected_boxes)
                orig_selected_boxes[:, 0] /= x_dev_nasobky[i][0] #DEV TEST H
                orig_selected_boxes[:, 2] /= x_dev_nasobky[i][0] #DEV TEST H
                orig_selected_boxes[:, 1] /= x_dev_nasobky[i][1] #DEV TEST W
                orig_selected_boxes[:, 3] /= x_dev_nasobky[i][1] #DEV TEST W

                #if i < 20:
                #    print(selected_predicted_classes+1)
                #    draw(x_dev[i], selected_boxes)


                output = ""
                for label, bbox in zip(selected_predicted_classes, orig_selected_boxes):
                    #output += [label] + bbox
                    output += str(label+1)+ " " + str(int(bbox[0]))+ " " +  str(int(bbox[1]))+ " " +  str(int(bbox[2]))+ " " +  str(int(bbox[3])) + " " 
                print(*output, file=predictions_file, sep='')
        if evaluate:
            os.system(f'python .\svhn_dataset.py --evaluate svhn_competition_{fname}.txt --dataset {fname}')

    def save_train(max_rois = 5, iou_threshold = 0.2, score_threshold = 0.28):
        save_data(set=x_train, nasobky=x_train_nasobky, fname='train' ,max_rois = max_rois, iou_threshold = iou_threshold, score_threshold = score_threshold, evaluate=True)

    def save_dev(max_rois = 5, iou_threshold = 0.2, score_threshold = 0.28):
        save_data(set=x_dev, nasobky=x_dev_nasobky, fname='dev' ,max_rois = max_rois, iou_threshold = iou_threshold, score_threshold = score_threshold, evaluate=True)

    def save_test(max_rois = 5, iou_threshold = 0.2, score_threshold = 0.28):
        save_data(set=x_train, nasobky=x_test_nasobky, fname='dev' ,max_rois = max_rois, iou_threshold = iou_threshold, score_threshold = score_threshold, evaluate=False)


    from tensorflow.keras.callbacks import Callback
    class MyCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            save_dev()

    #model.fit( x_train,  { 'classes_output': all_cat_classes[:,:,1:] , 'bboxes_output': all_bboxes    } ,batch_size=16, epochs=1, sample_weight={ 'classes_output': np.ones_like(all_sample_weights) , 'bboxes_output': all_sample_weights    } )
    model.fit( x_train,  { 'classes_output': all_cat_classes[:,:,1:] , 'bboxes_output': all_bboxes    } ,batch_size=batch_size, epochs=epochs, sample_weight={ 'bboxes_output': all_sample_weights     } , callbacks=[MyCallback])
    #model.fit( x_train,  { 'classes_output': all_cat_classes[:,:,1:] , 'bboxes_output': all_bboxes    } ,batch_size=2, epochs=1, sample_weight={ 'classes_output': all_sample_weights, 'bboxes_output': all_sample_weights    } )
    #https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/bin/train.py

    #po 50 :
    # max_rois = 5, iou_threshold = 0.2, score_threshold = 0.28
    #SVHN accuracy: 62.35%

    #model.predict( np.array( [ t[0][0]] )  )[0].argmax(axis=2)  

    model.save('model.h5')
    ''' '''
    model = keras.models.load_model('model.h5')

    print('ende') 


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
