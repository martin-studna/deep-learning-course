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

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


# 2f67b427-a885-11e7-a937-00505601122b
# c751264b-78ee-11eb-a1a9-005056ad4f31

# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=None, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1,
                    type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int,
                    help="Maximum number of threads to use.")


def draw(img, bboxes):
    import cv2
    cv2.namedWindow("now", cv2.WINDOW_NORMAL)
    im = np.array(img)
    for i in range(len(bboxes)):
        cv2.rectangle(im, (int(bboxes[i][1]), int(bboxes[i][0])), (int(
            bboxes[i][3]), int(bboxes[i][2])), (255, 0, 0), 3)
    cv2.imshow("now", im)
    cv2.waitKey(0)


@tf.function
def augment_bigger(image, bboxes):
    image = tf.image.resize(
        image, (120, 120))
    return (image, bboxes)


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
    svhn = SVHN()

    train = svhn.train.map(lambda data: (
        data['image'], data['bboxes'], data['classes'])).take(-1)
    train_numpy = list(train.as_numpy_iterator())

    test = svhn.test.map(lambda data: (data['image'])).take(-1)
    test_list = list(test.as_numpy_iterator())

    my_anchors = []

    aspect_ratios = np.array([0.3, 0.5])

    for y in np.linspace(0, 1, 14):
        for x in np.linspace(0, 1, 14):
            my_anchors.append(
                (y - aspect_ratios[1] / 2, x - aspect_ratios[0] / 2, y + aspect_ratios[1] / 2, x + aspect_ratios[0] / 2))

    SIDE_SIZE = 224

    my_anchors = np.array(my_anchors)

    x_train = []

    all_cat_classes = []
    all_bboxes = []
    all_sample_weights = []

    for i in range(len(train_numpy)):
        height, width, channels = train_numpy[i][0].shape
        height_multiple = SIDE_SIZE / height
        width_multiple = SIDE_SIZE / width
        img = cv2.resize(train_numpy[i][0], (SIDE_SIZE, SIDE_SIZE))
        x_train.append(img)

        g_boxes = np.array(train_numpy[i][1])
        g_boxes[:, 0] *= height_multiple / SIDE_SIZE
        g_boxes[:, 2] *= height_multiple / SIDE_SIZE
        g_boxes[:, 1] *= width_multiple / SIDE_SIZE
        g_boxes[:, 3] *= width_multiple / SIDE_SIZE

        anchor_classes, anchor_bboxes = bboxes_utils.bboxes_training(
            my_anchors, train_numpy[i][2], g_boxes, 0.5)

        cat_classes = keras.utils.to_categorical(anchor_classes, 11)

        all_bboxes.append(anchor_bboxes)
        all_cat_classes.append(cat_classes)
        all_sample_weights.append(
            (cat_classes.argmax(axis=1) > 0).astype(np.float32))

    x_train = np.array(x_train)
    all_cat_classes = np.array(all_cat_classes)
    all_bboxes = np.array(all_bboxes)
    all_sample_weights = np.array(all_sample_weights)

    x_test = []
    for i in range(len(test_list)):
        x_test.append(cv2.resize(test_list[i], (SIDE_SIZE, SIDE_SIZE)))
    x_test = np.array(x_test)

    # Load the EfficientNet-B0 model
    efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(
        include_top=False)
    efficientnet_b0.trainable = False

    # TODO: Create the model and train it
    input_l = keras.layers.Input(shape=(SIDE_SIZE, SIDE_SIZE, 3))
    o0, o1, o2, o3, o4, o5, *_ = efficientnet_b0(input_l)
    x = o2

    classes = keras.layers.Conv2D(256, 3, padding='same')(x)
    classes = keras.layers.BatchNormalization()(classes)
    classes = keras.layers.Activation('relu')(classes)

    classes = keras.layers.Conv2D(256, 3, padding='same')(classes)
    classes = keras.layers.BatchNormalization()(classes)
    classes = keras.layers.Activation('relu')(classes)

    classes = keras.layers.Conv2D(
        10, 3, padding='same', activation='sigmoid')(classes)
    classes = keras.layers.Reshape((14*14, 10), name="classes_output")(classes)

    bboxes = keras.layers.Conv2D(256, 3, padding='same')(x)
    bboxes = keras.layers.BatchNormalization()(bboxes)
    bboxes = keras.layers.Activation('relu')(bboxes)

    bboxes = keras.layers.Conv2D(256, 3, padding='same')(bboxes)
    bboxes = keras.layers.BatchNormalization()(bboxes)
    bboxes = keras.layers.Activation('relu')(bboxes)

    bboxes = keras.layers.Conv2D(4, 3, padding='same')(bboxes)
    bboxes = keras.layers.Reshape((14*14, 4), name="bboxes_output")(bboxes)

    model = keras.models.Model(inputs=[input_l], outputs=[classes, bboxes])
    model.summary()

    losses = {
        'classes_output':   tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE),
        'bboxes_output':  tf.keras.losses.Huber(),
    }

    metrics = {
        'classes_output': keras.metrics.BinaryAccuracy(),
    }

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=losses,
                  metrics=metrics,
                  run_eagerly=False)

    model.fit(x_train,  {'classes_output': all_cat_classes[:, :, 1:], 'bboxes_output': all_bboxes},
              batch_size=16, epochs=1, sample_weight={'bboxes_output': all_sample_weights})

    model.save('model.h5')
    model = keras.models.load_model('model.h5')

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open("svhn_competition.txt", "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the digits and their bounding boxes on the test set.
        # Assume that for a single test image we get
        # - `predicted_classes`: a 1D array with the predicted digits,
        # - `predicted_bboxes`: a [len(predicted_classes), 4] array with bboxes;

        tpredicted_classes, tpredicted_bboxes = model.predict(
            x_test, batch_size=32)

        for i in range(len(tpredicted_classes)):
            scores = tpredicted_classes[i, :, 1:].max(axis=1)
            predicted_classes = tpredicted_classes[i, :, 1:].argmax(axis=1)
            predicted_bboxes = tpredicted_bboxes[i]

            predicted_bboxess = bboxes_utils.bboxes_from_fast_rcnn(
                my_anchors, predicted_bboxes) * SIDE_SIZE

            selected_indices = tf.image.non_max_suppression(
                predicted_bboxess, scores, 5, iou_threshold=0.2, score_threshold=0.2).numpy()
            selected_boxes = predicted_bboxess[selected_indices]
            selected_scores = scores[selected_indices]
            selected_predicted_classes = predicted_classes[selected_indices]

            if i < 10:
                print(selected_predicted_classes)
                draw(x_test[i], selected_boxes)

            output = ""
            for label, bbox in zip(selected_predicted_classes, selected_boxes):
                output += str(label+1) + " " + str(int(bbox[0])) + " " + str(
                    int(bbox[1])) + " " + str(int(bbox[2])) + " " + str(int(bbox[3])) + " "
            print(*output, file=predictions_file, sep='')


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
