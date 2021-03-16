#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf

from mnist import MNIST

from tqdm import tqdm

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.") #50
#parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.") #5
parser.add_argument("--hidden_layer", default=20, type=int, help="Size of the hidden layer.") #100
#parser.add_argument("--hidden_layer", default=100, type=int, help="Size of the hidden layer.")
parser.add_argument("--learning_rate", default=0.1, type=float, help="Learning rate.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.
class Model(tf.Module):
    def __init__(self, args):
        self._args = args

        self._W1 = tf.Variable(tf.random.normal(
            [MNIST.W * MNIST.H * MNIST.C, args.hidden_layer], stddev=0.1, seed=args.seed), trainable=True)
        self._b1 = tf.Variable(tf.zeros([args.hidden_layer]), trainable=True)

        # TODO(sgd_backpropagation): Create variables:
        # - _W2, which is a trainable Variable of size [args.hidden_layer, MNIST.LABELS],
        #   initialized to tf.random.normal value with stddev=0.1 and seed=args.seed,
        # - _b2, which is a trainable Variable of size [MNIST.LABELS] initialized to zeros
        self._W2 = tf.Variable(tf.random.normal(
            [args.hidden_layer, MNIST.LABELS], stddev=0.1, seed=args.seed), trainable=True)
        self._b2 = tf.Variable(tf.zeros([MNIST.LABELS]), trainable=True)

    def predict(self, inputs):
        # TODO(sgd_backpropagation): Define the computation of the network. Notably:
        # - start by reshaping the inputs to shape [inputs.shape[0], -1].
        #   The -1 is a wildcard which is computed so that the number
        #   of elements before and after the reshape fits.
        # - then multiply the inputs by self._W1 and then add self._b1
        # - apply tf.nn.tanh
        # - multiply the result by self._W2 and then add self._b2
        # - finally apply tf.nn.softmax and return the result
        inputs = tf.constant(inputs, dtype=tf.float32)
        inputs = tf.reshape(inputs, [inputs.shape[0], -1])
        h_1 = tf.nn.tanh(inputs @ self._W1 + self._b1)
        outputs = tf.nn.softmax(h_1 @ self._W2 + self._b2)

        # TODO: In order to support manual gradient computation, you should
        # return not only the output layer, but also the hidden layer after applying
        # tf.nn.tanh, and the input layer after reshaping.

        return outputs, h_1, inputs

    def train_epoch(self, dataset):
        #counter = 0
        for batch in tqdm( dataset.batches(self._args.batch_size) ):
            #counter += 1
            #if counter == 20:
            #    break
            
            # The batch contains
            # - batch["images"] with shape [?, MNIST.H, MNIST.W, MNIST.C]
            # - batch["labels"] with shape [?]
            # Size of the batch is self._args.batch_size, except for the last, which
            # might be smaller.

            # TODO: Contrary to sgd_backpropagation, the goal here is to compute
            # the gradient manually, without tf.GradientTape. ReCodEx checks
            # that tf.GradientTape is not used and if it is, your solution does
            # not pass.

            # TODO: Compute the input layer, hidden layer and output layer
            # of the batch images using self.predict.
            outputs, h_1, inputs = self.predict(batch["images"])
            mezivysledek = inputs @ self._W1 + self._b1
            # TODO: Compute the gradient of the loss with respect to all
            # variables. Note that the loss is computed as:
            # - for every batch example, it is the categorical crossentropy of the
            #   predicted probabilities and gold batch label
            # - finally, the individual batch example losses are averaged
            #
            # During the gradient computation, you will need to compute
            # a so-called outer product
            #   C[a, i, j] = A[a, i] * B[a, j]
            # which you can for example as
            #   A[:, :, tf.newaxis] * B[:, tf.newaxis, :]
            # or with
            #   tf.einsum("ai,aj->aij", A, B)

            #loss = tf.keras.losses.categorical_crossentropy(tf.keras.utils.to_categorical(
            #    batch["labels"], num_classes=outputs.shape[1]), outputs) / batch["labels"].shape[0]

            desired_outputs = tf.keras.utils.to_categorical(batch["labels"], num_classes=outputs.shape[1])

            #rozměry pro batch size 64 a hidden layer 20, 10 tříd
            #b2_derivative = tf.Variable( tf.zeros( ( len(batch), self._b1.shape[0] ) ) )    #má mít rozměr 64(batch), 10(output)
            #W2_derivative = tf.Variable(  tf.zeros( ( len(batch), self._W2.shape[0], self._W2.shape[1]    )  ) )      #má mít rozměr 64(batch), 20(hidden), 10(output) 
            #I1_derivative = tf.Variable(  tf.zeros(  (  len(batch),  self._b1.shape[0]  )    ) )  # 64, 20
            #W1_derivative = tf.Variable(  tf.zeros(    (  len(batch),  self._W1.shape[0],  self._W1.shape[1]   )    ) )    # 64, 746(input), 20(hidden)

            b2_derivative = outputs - desired_outputs #má mít rozměr 64(batch), 10(output)
            
            W2_derivative = h_1[:, :, tf.newaxis] * b2_derivative[:, tf.newaxis, :]  #má mít rozměr 64(batch), 20(hidden), 10(output) 

            h_1_derivative = b2_derivative  @ tf.transpose(self._W2) #má mít rozměr 64 , 20

            tanh_derivative =   1 - tf.pow(  tf.tanh(mezivysledek ), 2) # 64, 20

            I1_derivative_local =  h_1_derivative * tanh_derivative # 64, 20

            W1_derivative = inputs[:, :, tf.newaxis] * I1_derivative_local[:, tf.newaxis, :] #64, 746 , 20




            '''
            for i in range(len(batch["labels"])):
                outputs_derivative = tf.subtract( outputs[i], desired_outputs[i] ) #https://stats.stackexchange.com/questions/370723/how-to-calculate-the-derivative-of-crossentropy-error-function
                b2_derivative.assign_add( outputs_derivative   )     
                #outputs_derivative = tf.reshape(outputs_derivative, (1, outputs_derivative.shape[0])  )
                #W2_derivative = tf.einsum("ai,aj->aij", h_1[0], outputs_derivative)
                W2_derivative.assign_add( tf.tensordot( h_1[i] , outputs_derivative, axes=0) )

                outputs_derivative = tf.reshape(outputs_derivative, (1, outputs_derivative.shape[0])  )
                h_1_derivative = self._W2 @ tf.transpose(outputs_derivative)

                tanh_derivative =   1 - tf.pow(  tf.tanh(mezivysledek[i]), 2) #tohle by se nemělo přepočítávat stále znovu
                h_1_derivative = tf.reshape(h_1_derivative, ( h_1_derivative.shape[0]) )

                #I1_derivative = h_1_derivative @ tanh_derivative
                I1_derivative_local =  h_1_derivative * tanh_derivative 
                I1_derivative.assign_add( I1_derivative_local )

                W1_derivative.assign_add(  tf.tensordot(inputs[i], I1_derivative_local, axes=0) )
            '''
            # TODO(sgd_backpropagation): Perform the SGD update with learning rate self._args.learning_rate
            # for the variable and computed gradient. You can modify
            # variable value with variable.assign or in this case the more
            # efficient variable.assign_sub.

            #W2_derivative[0] [ 0.8569202 , -0.7486297 ,  0.23134878, -0.7014691 , -2.9862432 ,   0.17736875,  1.5916749 ,  2.887221  ,  0.31432727, -1.6225191 ]
            self._W2.assign_sub(self._args.learning_rate * tf.math.reduce_mean( W2_derivative, axis=0) )
            self._b2.assign_sub(self._args.learning_rate * tf.math.reduce_mean( b2_derivative, axis=0) )   #https://datascience.stackexchange.com/questions/20139/gradients-for-bias-terms-in-backpropagation

            self._W1.assign_sub(self._args.learning_rate * tf.math.reduce_mean( W1_derivative, axis=0) )  
            self._b1.assign_sub(self._args.learning_rate * tf.math.reduce_mean( I1_derivative_local, axis=0) )  
            #s assign_sub to funguje líp???????

    def evaluate(self, dataset):
        # Compute the accuracy of the model prediction
        correct = 0
        for batch in dataset.batches(self._args.batch_size):
            # TODO(sgd_backpropagation): Compute the probabilities of the batch images
            probabilities = tf.argmax(   self.predict(batch["images"])[0] , axis=1   )


            # TODO(sgd_backpropagation): Evaluate how many batch examples were predicted
            # correctly and increase correct variable accordingly.
            correct += tf.math.count_nonzero( probabilities == batch["labels"])


        return correct / dataset.size


def main(args):
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    tf.config.run_functions_eagerly(True)
    #tf.config.set_visible_devices([], 'GPU') #odstranit!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("_file_", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub(
            "(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load data
    mnist = MNIST()

    # Create the TensorBoard writer
    writer = tf.summary.create_file_writer(args.logdir, flush_millis=10*1000)

    # Create the model
    model = Model(args)

    for epoch in range(args.epochs):
        # TODO(sgd_backpropagation): Run the train_epoch with mnist.train dataset
        model.train_epoch(mnist.train)
        # TODO(sgd_backpropagation): Evaluate the dev data using evaluate on mnist.dev dataset
        accuracy = model.evaluate(mnist.dev)
        print("Dev accuracy after epoch {} is {:.2f}".format(
            epoch + 1, 100 * accuracy), flush=True)
        with writer.as_default():
            tf.summary.scalar("dev/accuracy", 100 * accuracy, step=epoch + 1)

    # TODO(sgd_backpropagation): Evaluate the test data using evaluate on mnist.test dataset
    accuracy = model.evaluate(mnist.test)
    print("Test accuracy after epoch {} is {:.2f}".format(
        epoch + 1, 100 * accuracy), flush=True)
    with writer.as_default():
        tf.summary.scalar("test/accuracy", 100 * accuracy, step=epoch + 1)

    # Return test accuracy for ReCodEx to validate
    return accuracy.numpy()

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
