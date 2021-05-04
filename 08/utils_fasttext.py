#! /usr/bin/env python

import fasttext
import fasttext.util
import numpy as np
import tensorflow as tf
import pickle

import io


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    weights = np.zeros((n, d), dtype=float)
    for line, i in zip(fin, range(int(100000))):
        tokens = line.rstrip().split(' ')
        x = np.array(tokens[1:], dtype=float)
        weights[i] = x
    return weights
