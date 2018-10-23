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
TRAIN_BATCH_SIZE = 128
display_step = 100
NUM_STEPS = 20000
validation_step = 100
LEARNING_RATE = 0.001

# Center-Loss
CENTER_LOSS_ALPHA = 0.5
CENTER_LOSS_LAMBDA = 0.5

# Params for test
TEST_BATCH_SIZE = 1000

def train():

    mnist = input_data.read_data_sets('./mnist')

    image_mean = np.mean(mnist.train.images, axis=0)

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
        center_loss_term,_,centers_update_op = center_loss(features=bottleneck_layer,labels=y_,alpha=CENTER_LOSS_ALPHA,num_classes=10)

    with tf.name_scope("classification_loss"):
        classification_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=logits, name='cross_entropy'))

    with tf.name_scope("total_loss"):
        total_loss = classification_loss + CENTER_LOSS_LAMBDA*center_loss_term

    # Create a summary to monitor loss tensor
    tf.summary.scalar('loss', total_loss)

    # Define optimizer
    with tf.name_scope("ADAM"):
        global_step = tf.Variable(0,name='global_step',trainable=False)
        # Use simple momentum for the optimization.
        with tf.control_dependencies([centers_update_op]):
            train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(total_loss,global_step=global_step)

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
        batch_images = normalize_batch(batch[0],image_mean=image_mean)
        batch_labels = batch[1]
        _, train_accuracy, summary,total_loss_value,softmax_loss,center_loss_value = sess.run([train_step, accuracy, merged_summary_op,total_loss,classification_loss,center_loss_term] , feed_dict={x: batch_images, y_: batch_labels, is_training: True})

        # Write logs at every iteration
        summary_writer.add_summary(summary,steps)

        # Display logs
        if steps % display_step == 0:
            print('Step : {0} Training accuracy : {1} Total Loss : {2} Classification : {3} Center : {4}'.format(steps,train_accuracy,total_loss_value,softmax_loss,center_loss_value))

        # Get accuracy for validation data
        if steps % validation_step == 0:
            # Calculate accuracy
            test_batch = mnist.test.next_batch(TRAIN_BATCH_SIZE)
            test_batch_images = normalize_batch(test_batch[0],image_mean=image_mean)
            test_batch_labels = test_batch[1]
            validation_accuracy = sess.run(accuracy,feed_dict={x:test_batch_images, y_:test_batch_labels, is_training: False})
            print('Step : {0} Test set accuracy : {1}'.format(steps,validation_accuracy))

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
        test_batch = mnist.test.next_batch(TEST_BATCH_SIZE)
        test_batch_images = normalize_batch(test_batch[0],image_mean=image_mean)
        test_batch_labels = test_batch[1]
        y_final = sess.run(logits, feed_dict={x:test_batch_images, y_:test_batch_labels, is_training: False})
        correct_prediction = np.equal(np.argmax(y_final, 1), test_batch_labels)
        acc_buffer.append(np.sum(correct_prediction) / TEST_BATCH_SIZE)

    print("test accuracy for the stored model: %g" % numpy.mean(acc_buffer))

def normalize_batch(batch_images,image_mean):
    """
    Normalize to [0,1] range

    """
    return batch_images-image_mean


def center_loss(features, labels, alpha, num_classes):
    """获取center loss及center的更新op

    Arguments:
        features: Tensor,表征样本特征,一般使用某个fc层的输出,shape应该为[batch_size, feature_length].
        labels: Tensor,表征样本label,非one-hot编码,shape应为[batch_size].
        alpha: 0-1之间的数字,控制样本类别中心的学习率,细节参考原文.
        num_classes: 整数,表明总共有多少个类别,网络分类输出有多少个神经元这里就取多少.

    Return：
        loss: Tensor,可与softmax loss相加作为总的loss进行优化.
        centers: Tensor,存储样本中心值的Tensor，仅查看样本中心存储的具体数值时有用.
        centers_update_op: op,用于更新样本中心的op，在训练时需要同时运行该op，否则样本中心不会更新
    """
    # 获取特征的维数，例如256维
    len_features = features.get_shape()[1]
    # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
    # 设置trainable=False是因为样本中心不是由梯度进行更新的
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
    # 将label展开为一维的，输入如果已经是一维的，则该动作其实无必要
    labels = tf.reshape(labels, [-1])

    # 根据样本label,获取mini-batch中每一个样本对应的中心值
    centers_batch = tf.gather(centers, labels)
    # 计算loss
    loss = tf.nn.l2_loss(features - centers_batch)

    # 当前mini-batch的特征值与它们对应的中心值之间的差
    diff = centers_batch - features

    # 获取mini-batch中同一类别样本出现的次数,了解原理请参考原文公式(4)
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff

    centers_update_op = tf.scatter_sub(centers, labels, diff)

    return loss, centers, centers_update_op

if __name__ == '__main__':
    train()
