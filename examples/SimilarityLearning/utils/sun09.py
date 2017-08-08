#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: sun09.py
# Author: Akanksha Saran <asaran@cs.utexas.edu>

import os
import gzip
import numpy as np
from six.moves import range, zip, map
import cv2

from tensorpack.utils import logger
from tensorpack.utils.fs import download, get_dataset_path
from tensorpack.utils.timer import timed_operation
from tensorpack.dataflow.base import RNGDataFlow

__all__ = ['sun09']

class SUN09(RNGDataFlow):
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

        imgs_labels = [line.rstrip('\n') for line in open(pathfile,'r')]
        self.imglist = [img_label.split('\t') for img_label in imgs_labels]
        
        
        self.images = []
        self.labels = []
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            #print(self.imglist[k])
            fname, label = self.imglist[k]
            #if int(label)==0:
            #    print(fname,label)
            im = cv2.imread(fname, cv2.IMREAD_COLOR)
            assert im is not None, fname
            if im.ndim == 2:
                im = np.expand_dims(im, 2).repeat(3,2)
            im = cv2.resize(im,(224,224))
            self.images.append(im)
            self.labels.append(int(label))
        #print('*****')
        #print(self.images[0].shape)


    def size(self):
        return len(self.imglist)

    def get_data(self):
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            self.rng.shuffle(idxs)

        for k in idxs:
            fname, label = self.imglist[k]

            im = cv2.imread(fname, cv2.IMREAD_COLOR)
            assert im is not None, fname
            if im.ndim == 2:
                im = np.expand_dims(im, 2).repeat(3, 2)
            yield [im, int(label)]

if __name__ == '__main__':
    ds = SUN09('/home/asaran/research/tensorpack/examples/SimilarityLearning/data/train.txt', 'train',
                  shuffle=False)
    ds.reset_state()
    for k in ds.get_data():
        from IPython import embed
        embed()
        break
