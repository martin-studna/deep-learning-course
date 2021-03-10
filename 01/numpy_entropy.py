#!/usr/bin/env python3
import argparse

import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False,
                    action="store_true", help="Evaluation in ReCodEx.")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args):
    # TODO: Load data distribution, each line containing a datapoint -- a string.

    data_dict = {}
    with open("numpy_entropy_data.txt", "r") as data:
        for line in data:
            line = line.rstrip("\n")
            if data_dict.get(line) is None:
                data_dict[line] = 1
            else:
                data_dict[line] += 1
            # TODO: Process the line, aggregating data with built-in Python
            # data structures (not NumPy, which is not suitable for incremental
            # addition and string mapping).

    # TODO: Create a NumPy array containing the data distribution. The
    # NumPy array should contain only data, not any mapping.

    model_dict = {}
    # TODO: Load model distribution, each line `string \t probability`.
    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")
            # TODO: process the line, aggregating using Python data structures
            tokens = line.split("\t")
            model_dict[tokens[0]] = float(tokens[1])

    # TODO: Create a NumPy array containing the model distribution.

    data_labels = sorted(data_dict.keys())
    model_labels = sorted(model_dict.keys())
    data_list = sorted(data_dict.items())
    model_list = sorted(model_dict.items())
    i = 0
    while i < len(data_list) or i < len(model_list):
        if i >= len(data_list) and i < len(model_list):
            data_list.append((model_list[i][0], 0))
        elif i < len(data_list) and i >= len(model_list):
            model_list.append((data_list[i][0], 0))
        elif data_list[i][0] != model_list[i][0]:
            data_list.insert(i, (model_list[i][0], 0))
        i += 1

    model_distr = np.array([x[1] for x in model_list])
    data_distr = np.array([x[1] for x in data_list]) / \
        np.sum([x[1] for x in data_list])

    # TODO: Compute the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication).
    entropy = - np.nansum(data_distr * np.log(data_distr))

    # TODO: Compute cross-entropy H(data distribution, model distribution).
    crossentropy = - \
        np.nansum(data_distr * np.log(model_distr))

    # TODO: Compute KL-divergence D_KL(data distribution, model_distribution)
    kl_divergence = - \
        np.nansum(data_distr * (np.log(model_distr / data_distr)))

    # Return the computed values for ReCodEx to validate
    return entropy, crossentropy, kl_divergence


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    entropy, crossentropy, kl_divergence = main(args)
    print("{:.2f}".format(entropy))
    print("{:.2f}".format(crossentropy))
    print("{:.2f}".format(crossentropy - entropy))
