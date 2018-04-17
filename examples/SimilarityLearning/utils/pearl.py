#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: sun09.py
# Author: Akanksha Saran <asaran@cs.utexas.edu>

import os
import gzip
import numpy as np
from six.moves import range, zip, map
import cv2
import json
import random

from tensorpack.utils import logger
from tensorpack.utils.fs import download, get_dataset_path
from tensorpack.utils.timer import timed_operation
from tensorpack.dataflow.base import RNGDataFlow

__all__ = ['dataset']

class Dataset(RNGDataFlow):
    """
    Produces [image, label] in MNIST dataset,
    image is 28x28 in the range [0,1], label is an int.
    """

    def __init__(self, pathfile, train_or_test, shuffle=False):
        """
        Args:
            train_or_test (str): either 'train' or 'test'
            shuffle (bool): shuffle the dataset
        """
        assert os.path.isfile(pathfile)
        assert train_or_test in ['train', 'test']
        self.name = train_or_test
        self.shuffle = shuffle


        with open(pathfile) as datafile:
            data = json.load(datafile)
        
        self.images = []
        self.labels = []
        self.bb = []

        idxs = np.arange(len(data))
        if self.shuffle:
            #self.rng.shuffle(idxs)
            random.shuffle(idxs)

        for k in idxs:
            element = data[k]
            im = cv2.imread(element['img_path'], cv2.IMREAD_COLOR)
            assert im is not None, element['img_path']
            if im.ndim == 2:
                im = np.expand(element['img_path'],2).repeat(3,2)
            im = cv2.resize(im,(224,224))
            self.images.append(im)
            self.labels.append(int(element['label']))
            self.bb.append(np.array(element['bb']))

    def size(self):
        return len(self.labels)
    
    def get_data(self):
        idxs = np.arange(len(self.images))
        if self.shuffle:
            #self.rng.shuffle(idxs)
            random.shuffle(idxs)

        for k in idxs:
            yield [self.images[k], self.bb[k], self.labels[k]]

if __name__ == '__main__':
    ds = Dataset('/home/asaran/research/tensorpack/examples/SimilarityLearning/data/genome_train.json', 'train',
                  shuffle=True)
    ds.reset_state()
    for k in ds.get_data():
        from IPython import embed
        embed()
        break
