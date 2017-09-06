#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: embedding_data.py
# Author: tensorpack contributors

from tensorpack import *

import numpy as np
from tensorpack.dataflow import BatchData
import sys
sys.path.append('utils/')
from genome import Dataset
import random

def get_test_data(pathFile,batch=64):
    ds = Dataset(pathFile, 'test', shuffle=True)
    ds = AugmentImageComponent(ds, [imgaug.Resize((224, 224))])
    ds = BatchData(ds, batch)
    return ds


def get_digits_by_label(images, labels, bb):
    img_data = []
    bb_data = []
    for clazz in range(0, 16):
        #clazz_filter = np.where(labels == clazz)
        #data_dict.append(list(images[clazz_filter].reshape((-1, 225, 225))))

        clazz_filter = [i for i, j in enumerate(labels) if j == clazz]
        images_clazz = [images[i] for i in clazz_filter]
        bb_clazz = [bb[i] for i in clazz_filter]
        img_data.append(images_clazz)
        bb_data.append(bb_clazz)
    return img_data, bb_data


class DatasetPairs(Dataset):
    """We could also write

    .. code::

        ds = dataset.Mnist('train')
        ds = JoinData([ds, ds])
        ds = MapData(ds, lambda dp: [dp[0], dp[2], dp[1] == dp[3]])
        ds = BatchData(ds, 128 // 2)

    but then the positives pairs would be really rare (p=0.1).
    """
    def __init__(self, pathFile, train_or_test):
        super(DatasetPairs, self).__init__(pathFile, train_or_test, shuffle=True)
        # now categorize these digits
        self.img_dict, self.bb_dict = get_digits_by_label(self.images, self.labels, self.bb)
        assert(len(self.img_dict)==len(self.bb_dict))
        #for i in range(len(self.img_dict)):
        #    print(str(i)+': '+str(len(self.img_dict[i])))

    def pick(self, label):
        #print(label)
        idx = self.rng.randint(len(self.img_dict[label]))
        #idx = random.randint(0,len(self.img_dict[label]))
        return self.img_dict[label][idx].astype(np.float32), self.bb_dict[label][idx]

    def get_data(self):
        while True:
            y = self.rng.randint(2)
            if y == 0:
                pick_label, pick_other = self.rng.choice(16, size=2, replace=False)
            else:
                pick_label = self.rng.randint(16)
                pick_other = pick_label
            
            a = self.pick(pick_label)
            b = self.pick(pick_other)

            yield [a[0], a[1], b[0], b[1], y]


class DatasetTriplets(DatasetPairs):
    def get_data(self):
        while True:
            pick_label, pick_other = self.rng.choice(16, size=2, replace=False)

            a = self.pick(pick_label)
            b = self.pick(pick_label)
            c = self.pick(pick_other)

            yield [a[0], a[1], b[0], b[1], c[0], c[1]]
