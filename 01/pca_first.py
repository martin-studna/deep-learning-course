#!/usr/bin/env python3
from mnist import MNIST
import tensorflow as tf
import numpy as np
import argparse
import os
# Report only TF errors by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--examples", default=256, type=int,
                    help="MNIST examples to use.")
parser.add_argument("--iterations", default=100, type=int,
                    help="Iterations of the power algorithm.")
parser.add_argument("--recodex", default=False,
                    action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int,
                    help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args):
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Load data
    mnist = MNIST()

    data_indices = np.random.choice(
        mnist.train.size, size=args.examples, replace=False)
    data = tf.convert_to_tensor(mnist.train.data["images"][data_indices])

    # TODO: Data has shape [args.examples, MNIST.H, MNIST.W, MNIST.C].
    # We want to reshape it to [args.examples, MNIST.H * MNIST.W * MNIST.C].
    # We can do so using `tf.reshape(data, new_shape)` with new shape
    # `[data.shape[0], data.shape[1] * data.shape[2] * data.shape[3]]`.

    # Before reshape tensor is represented as: [E, 28, 28, 1] -> [E, H, W, C]
    # where E - is number of images, H - height, W - width, C - number of channels
    # As images are black and white, there is only one channel to represent this color: 0 - 255
    # When we are computing PCA, we need 2D matrix, so we are going to reshape the matrix:
    # First dimension stays the same. Second dimension is going to be created from the rest of the dimensions.

    new_shape = [data.shape[0], data.shape[1] * data.shape[2] * data.shape[3]]
    data = tf.reshape(data, new_shape)

    # TODO: Now compute mean of every feature. Use `tf.math.reduce_mean`,
    # and set `axis` to zero -- therefore, the mean will be computed
    # across the first dimension, so across examples.

    # Now we need to compute covariance matrix. We will take each row and we are going to compute
    # mean from every row:
    # (X - mean)^T * (X - mean) / |X|
    # Example: tf.math.reduce_mean(tf.ones([10,20]), axis=0)

    mean = tf.math.reduce_mean(data, axis=0)

    # TODO: Compute the covariance matrix. The covariance matrix is
    #   (data - mean)^T * (data - mean) / data.shape[0]
    # where transpose can be computed using `tf.transpose` and matrix
    # multiplication using either Python operator @ or `tf.linalg.matmul`.
    cov = (tf.transpose(data - mean) @ (data - mean)) / data.shape[0]

    # TODO: Compute the total variance, which is sum of the diagonal
    # of the covariance matrix. To extract the diagonal use `tf.linalg.diag_part`
    # and to sum a tensor use `tf.math.reduce_sum`.
    total_variance = tf.math.reduce_sum(tf.linalg.diag_part(cov))

    # TODO: Now run `args.iterations` of the power iteration algorithm.
    # Start with a vector of `cov.shape[0]` ones of type tf.float32 using `tf.ones`.
    v = tf.ones(cov.shape[0])
    for i in range(args.iterations):
        # TODO: In the power iteration algorithm, we compute
        # 1. v = cov * v
        #    The matrix-vector multiplication can be computed using `tf.linalg.matvec`.
        # 2. s = l2_norm(v)
        #    The l2_norm can be computed using `tf.linalg.norm`.
        # 3. v = v / s
        v = tf.linalg.matvec(cov, v)
        s = tf.linalg.norm(v)
        v = v / s

        # The `v` is now the eigenvector of the largest eigenvalue, `s`. We now
        # compute the explained variance, which is a ration of `s` and `total_variance`.
    explained_variance = s / total_variance

    # Return the total and explained variance for ReCodEx to validate
    return total_variance.numpy(), 100 * explained_variance.numpy()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    total_variance, explained_variance = main(args)
    print("Total variance: {:.2f}".format(total_variance))
    print("Explained variance: {:.2f}".format(explained_variance))
