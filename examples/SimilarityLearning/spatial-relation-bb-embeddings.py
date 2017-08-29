#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist-embeddings.py
# Author: PatWie <mail@patwie.com>
import numpy as np
import os

from tensorpack import *
import tensorpack.tfutils.symbolic_functions as symbf
from tensorpack.tfutils.summary import add_moving_summary
import argparse
import tensorflow as tf
import tensorflow.contrib.slim as slim
import random

from spatial_relations_bb_data import get_test_data, DatasetPairs, DatasetTriplets

MATPLOTLIB_AVAIBLABLE = False
try:
    import matplotlib
    from matplotlib import offsetbox
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.patches as mpatches
    plt.switch_backend('agg')
    MATPLOTLIB_AVAIBLABLE = True
except ImportError:
    MATPLOTLIB_AVAIBLABLE = False

embed_dim=2 

class EmbeddingModel(ModelDesc):
    global embed_dim
    def embed(self, x, b, nfeatures=embed_dim):

        """Embed all given tensors into an nfeatures-dim space.  """
        list_split = 0
        if isinstance(x, list):
            list_split = len(x)
            x = tf.concat(x, 0)

        print('Printing x..')
        print(x.get_shape())
        # pre-process MNIST dataflow data
        #x = tf.expand_dims(x, 3)
        #x = x * 2 - 1

        # the embedding network
        #net = slim.layers.conv2d(x, 20, 5, scope='conv1')
        #net = slim.layers.max_pool2d(net, 2, scope='pool1')
        #net = slim.layers.conv2d(net, 50, 5, scope='conv2')
        #net = slim.layers.max_pool2d(net, 2, scope='pool2')
        #net = slim.layers.flatten(net, scope='flatten3')
        #net = slim.layers.fully_connected(net, 500, scope='fully_connected4')
        #embeddings = slim.layers.fully_connected(net, nfeatures, activation_fn=None, scope='fully_connected5')
        #i = 0
        #for img,bb in zip(x,b):
        with slim.arg_scope([slim.layers.fully_connected], weights_regularizer=slim.l2_regularizer(1e-5)):
            net = slim.layers.conv2d(x, 64, [3, 3], scope='conv1')
            net = slim.layers.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.layers.conv2d(net, 128, [3, 3], scope='conv2')
            net = slim.layers.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.layers.conv2d(net, 256, [3, 3], scope='conv3')
            net = slim.layers.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.layers.conv2d(net, 512, [3, 3], scope='conv4')
            net = slim.layers.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.layers.conv2d(net, 512, [3, 3], scope='conv5')
            net = slim.layers.max_pool2d(net, [2, 2], scope='pool5')
            net = slim.layers.flatten(net, scope='flatten5')
            net = slim.layers.fully_connected(net, 4096, scope='fc6')
            print('Printing fc shape...')
            print(net.get_shape())
            if list_split > 0:
                fc6_embed = tf.split(net, list_split, 0)
                bb_float = tf.cast(b, tf.float32)
                net = tf.concat(axis=2, values=[fc6_embed,bb_float])
                print('Printing multi bb shape..')
                print(net.get_shape())

                #net = tf.split(net,list_split,axis=0)
                #for j in range(list_split):
                #net = tf.squeeze(net)
                #net = tf.concat([tf.squeeze(net_el) for net_el in net], 0)
                net = tf.reshape(net,[tf.shape(net)[1]*list_split,4104])
                print('Printing reshaped..')
                print(net.get_shape())
            else:
                bb_float = tf.cast(b, tf.float32)
                net = tf.concat(axis=1, values=[net,bb_float])
                print('Printing single bb shape..')
                print(net.get_shape())
            
            net = slim.layers.dropout(net, 0.5, scope='dropout6')
            net = slim.layers.fully_connected(net, 4096, scope='fc7')
            net = slim.layers.dropout(net, 0.5, scope='dropout7')
            #try:
            #    _ = embeddings.shape
            #    embeddings = np.array(embeddings, slim.layers.fully_connected(net, nfeatures, activation_fn=None, scope='fc8_'+str(i)))
            #except:
            embeddings = slim.layers.fully_connected(net, nfeatures, activation_fn=None, scope='fc8')
                
            #i = i + 1

        # if "x" was a list of tensors, then split the embeddings
        if list_split > 0:
            embeddings = tf.split(embeddings, list_split, 0)

        return embeddings

    def _get_optimizer(self):
        lr = symbf.get_scalar_var('learning_rate', 1e-4, summary=True)
        return tf.train.GradientDescentOptimizer(lr)


class SiameseModel(EmbeddingModel):
    @staticmethod
    def get_data():
        ds = DatasetPairs('data/genome_train.json','train')
        ds = AugmentImageComponent(ds, [imgaug.Resize((224, 224))])
        ds = BatchData(ds, 64 // 2)
        return ds

    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, 224, 224), 'input'),
                InputDesc(tf.int32, (None, 8), 'bb'),
                InputDesc(tf.float32, (None, 224, 224), 'input_y'),
                InputDesc(tf.int32, (None, 8), 'bb_y'),
                InputDesc(tf.int32, (None,), 'label')]

    def _build_graph(self, inputs):
        # get inputs
        img_x, bb_x, img_y, bb_y, label = inputs
        # embed them
        x_embed, y_embed = self.embed([img_x, img_y], [bb_x, bb_y])

        # tag the embedding of 'input' with name 'emb', just for inference later on
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            tf.identity(self.embed(inputs[0]), name="emb")

        # compute the actual loss
        cost, pos_dist, neg_dist = symbf.contrastive_loss(x_embed, y_embed, label, 5., extra=True, scope="loss")
        self.cost = tf.identity(cost, name="cost")

        # track these values during training
        add_moving_summary(pos_dist, neg_dist, self.cost)


class CosineModel(SiameseModel):
    def _build_graph(self, inputs):
        x, y, label = inputs
        img_x. bb_x = x
        img_y, bb_y = y
        x_embed, y_embed = self.embed([img_x, img_y], [bb_x, bb_y])

        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            tf.identity(self.embed(inputs[0], inputs[1]), name="emb")

        cost = symbf.siamese_cosine_loss(x_embed, y_embed, label, scope="loss")
        self.cost = tf.identity(cost, name="cost")
        add_moving_summary(self.cost)


class TripletModel(EmbeddingModel):
    @staticmethod
    def get_data():
        ds = DatasetTriplets('data/genome_train.json','train')
        ds = AugmentImageComponent(ds, [imgaug.Resize((224, 224))])
        ds = BatchData(ds, 64 // 3)
        return ds

    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, 224, 224, 3), 'input'),
                InputDesc(tf.int32, (None, 8), 'bb'),
                InputDesc(tf.float32, (None, 224, 224, 3), 'input_p'),
                InputDesc(tf.int32, (None,8), 'bb_p'),
                InputDesc(tf.float32, (None, 224, 224, 3), 'input_n'),
                InputDesc(tf.int32, (None, 8), 'bb_n')
                ]

    def loss(self, a, p, n):
        return symbf.triplet_loss(a, p, n, 5., extra=True, scope="loss")

    def _build_graph(self, inputs):
        global embed_dim
        print(len(inputs))
        img_a, bb_a, img_p, bb_p, img_n, bb_n = inputs
        # scaling the bb coordinates wrt image
        bb_a = tf.scalar_mul(224,bb_a)
        bb_p = tf.scalar_mul(224,bb_p)
        bb_n = tf.scalar_mul(224,bb_n)
        a_embed, p_embed, n_embed = self.embed([img_a, img_p, img_n,], [bb_a, bb_p, bb_n], embed_dim)

        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            tf.identity(self.embed(inputs[0], inputs[1], embed_dim), name="emb")

        print('Printing shape of embeddings..')
        print(a_embed.get_shape())
        print(p_embed.get_shape())
        cost, pos_dist, neg_dist = self.loss(a_embed, p_embed, n_embed)

        self.cost = tf.identity(cost, name="cost")
        add_moving_summary(pos_dist, neg_dist, self.cost)


class SoftTripletModel(TripletModel):
    def loss(self, a, p, n):
        return symbf.soft_triplet_loss(a, p, n, scope="loss")


def get_config(model, algorithm_name):

    extra_display = ["cost"]
    if not algorithm_name == "cosine":
        extra_display = extra_display + ["loss/pos-dist", "loss/neg-dist"]

    return TrainConfig(
        dataflow=model.get_data(),
        model=model(),
        callbacks=[
            ModelSaver(max_to_keep=20, keep_freq=2),
            ScheduledHyperParamSetter('learning_rate', [(10, 1e-5), (20, 1e-6)])
        ],
        extra_callbacks=[
            MovingAverageSummary(),
            ProgressBar(extra_display),
            MergeAllSummaries(),
            RunUpdateOps()
        ],
        max_epoch=3000,
    )


def visualize(model_path, model, algo_name):
    if not MATPLOTLIB_AVAIBLABLE:
        logger.error("visualize requires matplotlib package ...")
        return
    pred = OfflinePredictor(PredictConfig(
        session_init=get_model_loader(model_path),
        model=model(),
        input_names=['input','bb'],
        output_names=['emb']))

    NUM_BATCHES = 6
    BATCH_SIZE = 64
    #images = np.zeros((BATCH_SIZE * NUM_BATCHES, 224, 224))  # the used digits
    embed = np.zeros((BATCH_SIZE * NUM_BATCHES, 2))  # the actual embeddings in 2-d
    labels = np.zeros((BATCH_SIZE * NUM_BATCHES)) # true labels

    # get only the embedding model data (genome test)
    ds = get_test_data('data/genome_test.json')
    ds.reset_state()

    for offset, dp in enumerate(ds.get_data()):
        img, bb, label = dp
        
        #TODO: verify input format
        prediction = pred([img, bb])[0]
        embed[offset * BATCH_SIZE:offset * BATCH_SIZE + BATCH_SIZE, ...] = prediction
        # TODO: enumerate label and color it accordingly
        #images[offset * BATCH_SIZE:offset * BATCH_SIZE + BATCH_SIZE, ...] = img
        labels[offset * BATCH_SIZE:offset * BATCH_SIZE + BATCH_SIZE, ...] = label
        offset += 1
        if offset == NUM_BATCHES: 
            break

    print('MATPLOTLIB_AVAILABLE: '+str(MATPLOTLIB_AVAIBLABLE))
    plt.ioff()
    fig = plt.figure()
    ax = plt.subplot(111)
    ax_min = np.min(embed, 0)
    ax_max = np.max(embed, 0)

    ax_dist_sq = np.sum((ax_max - ax_min)**2)
    ax.axis('off')

    # dictionary of labels
    relation_labels = {0:'at', 1:'along', 2:'across', 3:'near/beside', 4:'around', 5:'on top of', 6:'side of', 7:'in/inside', 
            8:'over', 9:'left of', 10:'under/below', 11:'by', 12:'bottom', 13:'outside', 14:'on', 15:'right of'}
    circles = []
    classes = []

    # total number of labels
    N = 16
    # define the colormap
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    x = np.arange(N)
    ys = [i+x+(i*x)**2 for i in range(N)]
    #c = cm.rainbow(np.linspace(0, 1, len(ys)))
    c = ['r','b', 'c', 'g','yellow','blueviolet','lightblue','darkgreen','orange','mediumvioletred','lightcoral',
            'olive','brown','dimgray','steelblue','k']

    for i in relation_labels:
        circles.append(mpatches.Circle((0,0),1,color=c[i]))
        classes.append(relation_labels[i])

    shown_images = np.array([[1., 1.]])
    for i in range(embed.shape[0]):
        dist = np.sum((embed[i] - shown_images)**2, 1)
        if np.min(dist) < 3e-4 * ax_dist_sq:     # don't show points that are too close
            continue
        
        shown_images = np.r_[shown_images, [embed[i]]]
        # TODO: colored circle according to label
        plt.scatter(embed[i][0], embed[i][1], color=c[int(labels[i])])
        #imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(np.reshape(images[i, ...], [224, 224]),zoom=0.6, cmap=plt.cm.gray_r), xy=embed[i], frameon=False)
        #ax.add_artist(imagebox)

    plt.axis([ax_min[0]*2, ax_max[0]*2, ax_min[1]*2, ax_max[1]*2])
    plt.xticks([]), plt.yticks([])
    plt.legend(circles, classes, loc='lower left')
    plt.title('Embedding using %s-loss' % algo_name)
    plt.savefig('%s.jpg' % algo_name)
    plt.close(fig)

def evaluate_random(model_path, model, algo_name):
    global embed_dim
    ensemble_size = 15
    correct = 0
    total = 0
    BATCH_SIZE = 64
    NUM_BATCHES = 50000

    pred = OfflinePredictor(PredictConfig(
            session_init=get_model_loader(model_path),
            model=model(),
            input_names=['input','bb'],
            output_names=['emb']))

    # get train data
    dt = get_test_data('data/genome_train.json')
    dt.reset_state()
    print('loaded training data')

    train_data = {}
    for offset,dp in enumerate(dt.get_data()):
        #print(offset)
        img, bb, label = dp
        prediction = pred([img, bb])
        embedding = prediction[0]
        for i in range(BATCH_SIZE):
            gt = label[i]
            if gt not in train_data:
                train_data[gt] = [embedding[i]]
            else:
                train_data[gt].append(embedding[i])
        offset += 1
        if offset == NUM_BATCHES:
            break

    total_tr_data = 0 
    for label in train_data:
        print(str(label) + ': '+ str(len(train_data[label])))
        total_tr_data += len(train_data[label])
    print('total training data: ' + str(total_tr_data))

    ds = get_test_data('data/genome_test.json')
    ds.reset_state()
    print('loaded test data')

    for dp in ds.get_data():
        img, bb, label = dp
        embed_test_batch = pred([img, bb])[0]
        dist = {}
        for i in range(BATCH_SIZE):
            embed_test = embed_test_batch[i]
            # choose an image randomly from every class
            for l in train_data:
                dist[l] = 0
                r = random.sample(range(0,len(train_data[l])), ensemble_size)
                for sample in r:
                    dist[l] += np.linalg.norm(embed_test-train_data[l][sample])
                dist[l] = dist[l]/ensemble_size

            min_value = min(dist.itervalues())
            min_keys = [k for k in dist if dist[k] == min_value]

            if len(min_keys)==1:
                pred_class = min_keys[0]
            else:
                pred_class = min_keys[random.randint(0,len(min_keys)-1)]

            if pred_class == label[i]:
                correct += 1
            total += 1

    return correct, total

                                                                                                                                

if __name__ == '__main__':
    global embed_dim
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('-a', '--algorithm', help='used algorithm', type=str,
                        choices=["siamese", "cosine", "triplet", "softtriplet"])
    parser.add_argument('--visualize', help='export embeddings into an image', action='store_true')
    parser.add_argument('--dim', help='dimensionality of the embedding space', type=int)
    parser.add_argument('--evaluate', help = 'compute accuracy', action='store_true')
    args = parser.parse_args()

    ALGO_CONFIGS = {"siamese": SiameseModel,
                    "cosine": CosineModel,
                    "triplet": TripletModel,
                    "softtriplet": SoftTripletModel}

    logger.auto_set_dir(name=args.algorithm)

    if args.dim:
        embed_dim = args.dim

    with change_gpu(args.gpu):
        if args.visualize:
            visualize(args.load, ALGO_CONFIGS[args.algorithm], args.algorithm)
        elif args.evaluate:
            correct, total = evaluate_random(args.load, ALGO_CONFIGS[args.algorithm], args.algorithm)
            print('accuracy: '+str(float(correct)*100/total) + '% = ' + str(correct) + '/' +str(total))
        else:
            config = get_config(ALGO_CONFIGS[args.algorithm], args.algorithm)
            if args.load:
                config.session_init = SaverRestore(args.load)
            else:
                SimpleTrainer(config).train()
