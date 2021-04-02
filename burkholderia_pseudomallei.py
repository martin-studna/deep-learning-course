#!/usr/bin/env python3
import argparse
import sys
import os
import csv

from six.moves import urllib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import sklearn.model_selection
import sklearn.neighbors

parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.2,
                    type=int, help="Train set size")


def main(args):
    table = pd.read_csv('burkholderia_pseudomallei.csv', header=None, sep='\t')

    print(table)

    data = table[1:].to_numpy(dtype=float)

    infected = data[data[:, -1] == 1]
    noninfected = data[data[:, -1] == 0]

    targets = data[:, -1]

    X_train_infected, X_test_infected = sklearn.model_selection.train_test_split(
        infected, test_size=args.test_size, random_state=args.seed)

    X_train_noninfected, X_test_noninfected = sklearn.model_selection.train_test_split(
        noninfected, test_size=args.test_size, random_state=args.seed)

    infected_model = sklearn.neighbors.KernelDensity(
        kernel="linear", bandwidth=1.0)
    noninfected_model = sklearn.neighbors.KernelDensity(
        kernel="linear", bandwidth=1.0)

    infected_model.fit(X_train_infected)
    noninfected_model.fit(X_train_noninfected)

    '''
    KernelDensity method `score` returns log probability density. In order to get probability,
    we must invert the logarithm by applying the exponent.
    '''
    infected_dens = infected_model.score_samples(X_test_infected)
    infected_dens = np.exp(infected_dens)

    noninfected_model.fit(X_train_noninfected)

    noninfected_dens = noninfected_model.score_samples(
        X_test_noninfected)
    noninfected_dens = np.exp(noninfected_dens)

    infected_probability = targets[targets ==
                                   1].shape[0] / X_train_infected.shape[0]
    noninfected_probability = targets[targets ==
                                      0].shape[0] / X_train_noninfected.shape[0]
    target_probabilities = np.array(
        [[infected_probability, noninfected_probability]])

    results_infected = np.argmax(
        target_probabilities.T * infected_distribution)
    results_noninfected = np.argmax(
        target_probabilities.T * noninfected_distribution)

    print(results_infected)
    print(results_noninfected)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
