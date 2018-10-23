# Some code was borrowed from https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/models/image/mnist/convolutional.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

def CNN(inputs, is_training=True,bottleneck_layer_size=2):
    batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
    with slim.arg_scope([slim.conv2d, slim.fully_connected]):

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

        # Bottleneck Layer
        bottleneck_layer = slim.fully_connected(net,bottleneck_layer_size,activation_fn=None,scope='bottleneck')

        prelu_layer = PRelu(bottleneck_layer)

        logits = slim.fully_connected(prelu_layer,
                                      10,
                                      activation_fn=None,
                                      normalizer_fn=None,
                                      weights_initializer=slim.initializers.xavier_initializer(),
                                      scope='Logits',
                                      reuse=False)

        return logits,bottleneck_layer

def PRelu(x, name='PRelu'):
    """
    PReLU implementation  : https://github.com/zoli333/Center-Loss/blob/master/nn.py
    """
    with tf.variable_scope(name):
        alpha = tf.get_variable('alpha',shape=x.get_shape()[1:],dtype=tf.float32,initializer=tf.zeros_initializer(),trainable=True)
        pos = tf.nn.relu(x)
        neg = -alpha * tf.nn.relu(-x)
        return pos + neg
