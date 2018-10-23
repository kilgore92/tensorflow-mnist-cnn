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

MODEL_DIRECTORY = "model/model.ckpt"
LOGS_DIRECTORY = "logs/train"

# Params for Train
TRAIN_BATCH_SIZE = 100
display_step = 100
NUM_STEPS = 20000
validation_step = 500
CENTER_LOSS_ALPHA = 0.5
CENTER_LOSS_LAMBDA = 1e-2
LEARNING_RATE = 0.0005
# Params for test
TEST_BATCH_SIZE = 5000

def train():

    mnist = input_data.read_data_sets('./mnist')

    # Boolean for MODE of train or test
    is_training = tf.placeholder(tf.bool, name='MODE')

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.int32, [None]) #answer

    # Get the bottleneck layer tensor
    logits,bottleneck_layer = cnn_model.CNN(x)

    #Tensor to store normalized bottle-neck layer values
    embeddings = tf.nn.l2_normalize(bottleneck_layer, 1, 1e-10, name='embeddings')

    with tf.name_scope("center_loss"):
        center_loss_term, _ = center_loss(features=bottleneck_layer,label=y_,alfa=CENTER_LOSS_ALPHA,nrof_classes=10)

    with tf.name_scope("classification_loss"):
        classification_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_, logits=logits, name='cross_entropy'))

    with tf.name_scope("total_loss"):
        total_loss = classification_loss + CENTER_LOSS_LAMBDA*center_loss_term

    # Create a summary to monitor loss tensor
    tf.summary.scalar('loss', total_loss)

    # Define optimizer
    with tf.name_scope("ADAM"):
        batch = tf.Variable(0)
        # Use simple momentum for the optimization.
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(total_loss,global_step=batch)

    # Get accuracy of model
    with tf.name_scope("ACC"):
        correct_prediction = tf.equal(tf.argmax(logits, 1),tf.cast(y_,tf.int64))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Create a summary to monitor accuracy tensor
    tf.summary.scalar('acc', accuracy)

    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    # Add ops to save and restore all the variables
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer(), feed_dict={is_training: True})

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(LOGS_DIRECTORY, graph=tf.get_default_graph())

    # Save the maximum accuracy value for validation data
    max_acc = 0.

    # Loop for epoch
    for steps in range(NUM_STEPS):
        # Run optimization op (backprop), loss op (to get loss value)
        # and summary nodes
        batch = mnist.train.next_batch(TRAIN_BATCH_SIZE)
        _, train_accuracy, summary = sess.run([train_step, accuracy, merged_summary_op] , feed_dict={x: batch[0], y_: batch[1], is_training: True})

        # Write logs at every iteration
        summary_writer.add_summary(summary,steps)

        # Display logs
        if steps % display_step == 0:
            print('Step : {0} Training accuracy : {1}'.format(steps,train_accuracy))

        # Get accuracy for validation data
        if steps % validation_step == 0:
            # Calculate accuracy
            test_batch = mnist.test.next_batch(TRAIN_BATCH_SIZE)
            validation_accuracy = sess.run(accuracy,
            feed_dict={x:test_batch[0], y_:test_batch[1], is_training: False})
            print('Step : {0} Test set accuracy : {1}'.format(steps,validation_accuracy))


        #Flush to std-out
        sys.stdout.flush()

        # Save the current model if the maximum accuracy is updated
        if validation_accuracy > max_acc:
            max_acc = validation_accuracy
            save_path = saver.save(sess, MODEL_DIRECTORY)
            print("Model updated and saved in file: %s" % save_path)

    print("Optimization Finished!")

    # Restore variables from disk
    saver.restore(sess, MODEL_DIRECTORY)


    acc_buffer = []

    # Avg accuracy over 20 mini-batches from the test set
    for i in range(20):
        test_batch = mnist.test.next_batch(500)
        y_final = sess.run(logits, feed_dict={x:test_batch[0], y_:test_batch[1], is_training: False})
        correct_prediction = np.equal(np.argmax(y_final, 1), test_batch[1])
        acc_buffer.append(np.sum(correct_prediction) / batch_size)

    print("test accuracy for the stored model: %g" % numpy.mean(acc_buffer))


def center_loss(features, label, alfa, nrof_classes):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    nrof_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
    label = tf.reshape(label, [-1])

    #Update centers
    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers_batch - features)
    centers = tf.scatter_sub(centers, label, diff)

    # Center-loss
    with tf.control_dependencies([centers]):
        loss = tf.reduce_mean(tf.square(features - centers_batch))
    return loss, centers

if __name__ == '__main__':
    train()
