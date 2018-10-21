# Some code was borrowed from https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/models/image/mnist/convolutional.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

# Create model of CNN with slim api
def CNN(inputs, is_training=True,bottleneck_layer_size=2):
    batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        net = tf.reshape(inputs, [-1, 28, 28, 1])

        # For slim.conv2d, default argument values are like
        # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
        # padding='SAME', activation_fn=nn.relu,
        # weights_initializer = initializers.xavier_initializer(),
        # biases_initializer = init_ops.zeros_initializer,

        #Le-Net++ from Wen et al.
        #Stage-1
        net = slim.conv2d(net, 32, [5, 5], scope='conv1-1')
        net = slim.conv2d(net, 32, [5, 5], scope='conv1-2')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')

        #Stage-2
        net = slim.conv2d(net, 64, [5, 5], scope='conv2-1')
        net = slim.conv2d(net, 64, [5, 5], scope='conv2-2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')

        #Stage-3
        net = slim.conv2d(net, 128, [5, 5], scope='conv3-1')
        net = slim.conv2d(net, 128, [5, 5], scope='conv3-2')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')

        net = slim.flatten(net, scope='flatten3')

        # For slim.fully_connected, default argument values are like
        # activation_fn = nn.relu,
        # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
        # weights_initializer = initializers.xavier_initializer(),
        # biases_initializer = init_ops.zeros_initializer,

        # Bottleneck Layer -- Added by Ishaan
        net = slim.fully_connected(net,bottleneck_layer_size,activation_fn=None,scope='bottleneck')


        return net


def center_loss(features, label, alfa, nrof_classes):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    nrof_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers_batch - features)
    centers = tf.scatter_sub(centers, label, diff)
    with tf.control_dependencies([centers]):
        loss = tf.reduce_mean(tf.square(features - centers_batch))
    return loss, centers
