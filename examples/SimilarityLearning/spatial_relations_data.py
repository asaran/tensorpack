#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: embedding_data.py
# Author: tensorpack contributors

import numpy as np
from tensorpack.dataflow import BatchData
import sys
sys.path.append('utils/')
from sun09 import SUN09

def get_test_data(pathFile,batch=64):
    ds = SUN09(pathFile, 'test')
    ds = AugmentImageComponent(ds, [imgaug.Resize((224, 224))])
    ds = BatchData(ds, batch)
    return ds


def get_digits_by_label(images, labels):
    data_dict = []
    for clazz in range(0, 10):
        #clazz_filter = np.where(labels == clazz)
        #data_dict.append(list(images[clazz_filter].reshape((-1, 225, 225))))

        clazz_filter = [i for i, j in enumerate(labels) if j == clazz]
        images_clazz = [images[i] for i in clazz_filter]
        data_dict.append(images_clazz)
    return data_dict


class Sun09Pairs(SUN09):
    """We could also write

    .. code::

        ds = dataset.Mnist('train')
        ds = JoinData([ds, ds])
        ds = MapData(ds, lambda dp: [dp[0], dp[2], dp[1] == dp[3]])
        ds = BatchData(ds, 128 // 2)

    but then the positives pairs would be really rare (p=0.1).
    """
    def __init__(self, pathFile, train_or_test):
        super(Sun09Pairs, self).__init__(pathFile, train_or_test, shuffle=False)
        # now categorize these digits
        self.data_dict = get_digits_by_label(self.images, self.labels)

    def pick(self, label):
        idx = self.rng.randint(len(self.data_dict[label]))
        return self.data_dict[label][idx].astype(np.float32)

    def get_data(self):
        while True:
            y = self.rng.randint(2)
            if y == 0:
                pick_label, pick_other = self.rng.choice(10, size=2, replace=False)
            else:
                pick_label = self.rng.randint(10)
                pick_other = pick_label

            yield [self.pick(pick_label), self.pick(pick_other), y]


class Sun09Triplets(Sun09Pairs):
    def get_data(self):
        while True:
            pick_label, pick_other = self.rng.choice(10, size=2, replace=False)
            yield [self.pick(pick_label), self.pick(pick_label), self.pick(pick_other)]
