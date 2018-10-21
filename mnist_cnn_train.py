#!/usr/bin/anaconda3/bin/python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os,sys
sys.path.append(os.getcwd())
import mnist_data
import cnn_model


MODEL_DIRECTORY = "model/model.ckpt"
LOGS_DIRECTORY = "logs/train"

# Params for Train
training_epochs = 50 # 10 for augmented training data, 20 for training data
TRAIN_BATCH_SIZE = 100
display_step = 100
validation_step = 500
CENTER_LOSS_ALPHA = 0.5
CENTER_LOSS_LAMBDA = 1e-2
LEARNING_RATE = 0.0005
# Params for test
TEST_BATCH_SIZE = 5000

def train():

    # Some parameters
    batch_size = TRAIN_BATCH_SIZE
    num_labels = mnist_data.NUM_LABELS

    # Prepare mnist data
    train_total_data, train_size, validation_data, validation_labels, test_data, test_labels = mnist_data.prepare_MNIST_data(True)

    # Boolean for MODE of train or test
    is_training = tf.placeholder(tf.bool, name='MODE')

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.int32, [None, 10]) #answer

    # Get the bottleneck layer tensor
    prelogits = cnn_model.CNN(x)

    logits = slim.fully_connected(prelogits,
                                  10,
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  weights_initializer=slim.initializers.xavier_initializer(),
                                  scope='Logits',
                                  reuse=False)

    #Tensor to store normalized bottle-neck layer values
    embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

    with tf.name_scope("center_loss"):
        center_loss_term, _ = center_loss(features=prelogits,label=tf.argmax(y_,1),alfa=CENTER_LOSS_ALPHA,nrof_classes=10)

    with tf.name_scope("classification_loss"):
        classification_loss = tf.losses.softmax_cross_entropy(onehot_labels=y_,logits=logits)

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
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Create a summary to monitor accuracy tensor
    tf.summary.scalar('acc', accuracy)

    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    # Add ops to save and restore all the variables
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer(), feed_dict={is_training: True})

    # Training cycle
    total_batch = int(train_size / batch_size)

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(LOGS_DIRECTORY, graph=tf.get_default_graph())

    # Save the maximum accuracy value for validation data
    max_acc = 0.

    # Loop for epoch
    for epoch in range(training_epochs):

        # Random shuffling
        numpy.random.shuffle(train_total_data)
        train_data_ = train_total_data[:, :-num_labels]
        train_labels_ = train_total_data[:, -num_labels:]

        # Loop over all batches
        for i in range(total_batch):

            # Compute the offset of the current minibatch in the data.
            offset = (i * batch_size) % (train_size)
            batch_xs = train_data_[offset:(offset + batch_size), :]
            batch_ys = train_labels_[offset:(offset + batch_size), :]

            # Run optimization op (backprop), loss op (to get loss value)
            # and summary nodes
            _, train_accuracy, summary = sess.run([train_step, accuracy, merged_summary_op] , feed_dict={x: batch_xs, y_: batch_ys, is_training: True})

            # Write logs at every iteration
            summary_writer.add_summary(summary, epoch * total_batch + i)

            # Display logs
            if i % display_step == 0:
                print("Epoch:", '%04d,' % (epoch + 1),
                "batch_index %4d/%4d, training accuracy %.5f" % (i, total_batch, train_accuracy))

            # Get accuracy for validation data
            if i % validation_step == 0:
                # Calculate accuracy
                validation_accuracy = sess.run(accuracy,
                feed_dict={x: validation_data, y_: validation_labels, is_training: False})

                print("Epoch:", '%04d,' % (epoch + 1),
                "batch_index %4d/%4d, validation accuracy %.5f" % (i, total_batch, validation_accuracy))

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

    # Calculate accuracy for all mnist test images
    test_size = test_labels.shape[0]
    batch_size = TEST_BATCH_SIZE
    total_batch = int(test_size / batch_size)

    acc_buffer = []

    # Loop over all batches
    for i in range(total_batch):
        # Compute the offset of the current minibatch in the data.
        offset = (i * batch_size) % (test_size)
        batch_xs = test_data[offset:(offset + batch_size), :]
        batch_ys = test_labels[offset:(offset + batch_size), :]

        y_final = sess.run(logits, feed_dict={x: batch_xs, y_: batch_ys, is_training: False})
        correct_prediction = numpy.equal(numpy.argmax(y_final, 1), numpy.argmax(batch_ys, 1))
        acc_buffer.append(numpy.sum(correct_prediction) / batch_size)

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
