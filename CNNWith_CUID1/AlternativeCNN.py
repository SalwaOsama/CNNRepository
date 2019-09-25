import numpy as np
import shelve
import tensorflow as tf
import math
import os
import Utility as ut
from sklearn.metrics import classification_report


class CNN:
    def __init__(self, my_shelve, segment_size=128, n_filters=196,
                 n_channels=6, epochs=200, batch_size=200, learning_rate=5e-4,
                 dropout_rate=0.05, eval_iter=10, filters_size=16, n_classes=6,
                 IncludeFeat=0):
      # I'd like to use layers as parameters, flag to use features or not
        self.n_hidden = 1024
        self.l2_reg = 5e-4
        self.segment_size = segment_size
        self.n_channels = n_channels
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.eval_iter = eval_iter
        self.n_filters = n_filters
        self.filters_size = filters_size
        self.n_classes = n_classes  # it isn't valid to change it
        self.my_shelve = my_shelve
        # self.n_layers = n_layers
        self.IncludeFeat = IncludeFeat

    def RunAndAccuracy(self):
        # preparing data
        # 1-read datda
        my_shelve = shelve.open(self.my_shelve)
        features_test = my_shelve['features_test']
        features = my_shelve['features']
        labels_test = my_shelve['labels_test']
        labels_train = my_shelve['labels_train']
        data_train = my_shelve['data_train']
        data_test = my_shelve['data_test']
        my_shelve.close()

        # 2 Reshape data
        data_train = np.reshape(
            data_train, [-1, self.segment_size, self.n_channels])
        data_test = np.reshape(
            data_test, [-1, self.segment_size, self.n_channels])
        labels_train = np.reshape(labels_train, [-1, self.n_classes])
        labels_test = np.reshape(labels_test, [-1, self.n_classes])
        labels_test_unary = np.argmax(labels_test, axis=1)

        # 3 collect size
        train_size = data_train.shape[0]
        test_size = data_test.shape[0]
        num_features = features.shape[1]
       # prepare layers ANOTHER TECHNIQUE
        graph = tf.Graph()
        with graph.as_default():
            inputs_ = tf.placeholder(
                tf.float32, [None, self.segment_size, self.n_channels], name='inputs')
            labels_ = tf.placeholder(
                tf.float32, [None, self.n_classes], name='labels')
            keep_prob_ = tf.placeholder(tf.float32, name='keep')
            # learning_rate_ = tf.placeholder(tf.float32, name='Learning_rate')
            h_feat = tf.placeholder(
                tf.float32, [None, num_features], name='features')

        # build conv and pool layers
        with graph.as_default():
            conv1 = tf.layers.conv1d(inputs=inputs_, filters=self.n_filters,
                                     kernel_size=self.filters_size, strides=1,
                                     padding='same', activation=tf.nn.relu)
            max_pool = tf.layers.max_pooling1d(
                inputs=conv1, pool_size=2, strides=2, padding='same')

        # flat and add features
        with graph.as_default():
            # connect feature + flat as one layer
            shape = max_pool.get_shape().as_list()
            flat_size = shape[1]*shape[2]
            h_flat = tf.reshape(max_pool, (-1, flat_size))
            feat_flat = tf.concat(axis=1, values=[h_flat, h_feat])
            # flat_size += num_features
            # add dropout
            # if self.IncludeFeat == 1:
            flat = tf.nn.dropout(feat_flat, keep_prob=keep_prob_)
            # else:
            #       flat = tf.nn.dropout(h_flat, keep_prob=keep_prob_)

            logits = tf.layers.dense(flat, self.n_classes)

            # Cost function and optimizer
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=labels_))
            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)

        # Accuracy
            correct_pred = tf.equal(
                tf.argmax(logits, 1), tf.argmax(labels_, 1))
            accuracy = tf.reduce_mean(
                tf.cast(correct_pred, tf.float32), name='accuracy')

        # Train the network
        with graph.as_default():
            if (os.path.exists('checkpointswithTF-cnn') == False):
                os.mkdir('checkpointswithTF-cnn')
            saver = tf.train.Saver()
        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            train_acc = []
            train_loss = []
            max_accuracy=0.0
            for i in range(100000):
                idx_train = np.random.randint(0, train_size, self.batch_size)

                xt = np.reshape(data_train[idx_train], [
                                self.batch_size, self.segment_size , self.n_channels])
                yt = np.reshape(labels_train[idx_train], [
                                self.batch_size, self.n_classes])
                ft = np.reshape(features[idx_train], [
                                self.batch_size, num_features])
                
                sess.run(optimizer, feed_dict={
                    inputs_: xt, labels_: yt, h_feat: ft, keep_prob_: self.dropout_rate})
                if i % self.eval_iter == 0:

                    train_accuracy = sess.run(accuracy, feed_dict={inputs_: data_test, labels_: labels_test, h_feat: features_test, keep_prob_: 1})

                    print("step %d, max accuracy %g, accuracy %g" %
                        (i, max_accuracy, train_accuracy))

                    if max_accuracy < train_accuracy:
                        max_accuracy = train_accuracy
            saver.save(sess, "checkpointswithTF-cnn/har.ckpt")

    def get_batches(self, X, y, batch_size=100):
        """ Return a generator for batches """
        n_batches = len(X) // batch_size
        X, y = X[:n_batches*batch_size], y[:n_batches*batch_size]
        # Loop over batches and yield
        for b in range(0, len(X), batch_size):
            yield X[b:b+batch_size], y[b:b+batch_size]
