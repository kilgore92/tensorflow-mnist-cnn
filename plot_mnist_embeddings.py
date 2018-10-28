#!/usr/bin/anaconda3/bin/python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os,sys
sys.path.append(os.getcwd())
import cnn_model
from tensorflow.examples.tutorials.mnist import input_data

MODEL_DIRECTORY = "./model"

def load_mnist_model(model_dir,sess,input_map=None):
    """
    Inputs : model_dir : Directory where model checkpt and meta files are stored
             sess : TF session where the model needs into which the model will be loaded

    Returns : Restored model

    """

    model_path = os.path.expanduser(model_dir)
    print('Model directory is {}'.format(model_path))
    files = os.listdir(model_path)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) > 1:
        print('More than one meta file found, exiting')
        return

    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_path)

    sys.stdout.flush()

    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        saver = tf.train.import_meta_graph(os.path.join(model_path, meta_file), input_map=input_map)
        saver.restore(sess, os.path.join(model_path, ckpt_file))
        print('Model loaded successfully')
    else:
        print('Bad checkpoint state')

    sys.stdout.flush()


def plot_embeddings():
    """
    Plot MNIST 2-D embeddings to illustrate the effect of
    center loss

    """
    with tf.Graph().as_default():
        config = tf.ConfigProto()
        with tf.Session(config=config) as sess:
            load_mnist_model(model_dir=MODEL_DIRECTORY,sess=sess)
            #DEBUG code -- REMOVE
            for op in tf.get_default_graph().get_operations():
                print(str(op.name))
            sys.stdout.flush()



if __name__ == '__main__':
    plot_embeddings()


