# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.legacy_seq2seq as seq2seq

class Net(object):
    """docstring for Net"""
    def __init__(self, data, num_units, num_layer, batch_size):
        super(Net, self).__init__()
        self.num_units = num_units
        self.num_layer = num_layer
        self.batch_size = batch_size
        self.data =data
        self.build()

    def build(self):
        self.inputs = tf.placeholder(tf.int32, [self.batch_size, None])
        self.targets = tf.placeholder(tf.int32, [self.batch_size, None])
        self.keep_prob = tf.placeholder(tf.float32)
        self.seq_len = tf.placeholder(tf.int32, [self.batch_size])
        self.learning_rate = tf.placeholder(tf.float64)

        with tf.variable_scope('rnn'):
            w = tf.get_variable("softmax_w", [self.num_units, self.data.words_size])
            b = tf.get_variable("softmax_b", [self.data.words_size])

            embedding = tf.get_variable("embedding", [self.data.words_size, self.num_units])
            inputs = tf.nn.embedding_lookup(embedding, self.inputs)

        self.cell = tf.nn.rnn_cell.MultiRNNCell([self.unit() for _ in range(self.num_layer)])
        self.init_state = self.cell.zero_state(self.batch_size, dtype=tf.float32)
        output, self.final_state = tf.nn.dynamic_rnn(self.cell,
                                                inputs=inputs,
                                                sequence_length=self.seq_len,
                                                initial_state=self.init_state,
                                                scope='rnn')

        y = tf.reshape(output, [-1, self.num_units])
        logits = tf.matmul(y, w) + b
        prob = tf.nn.softmax(logits)
        self.prob = tf.reshape(prob, [self.batch_size, -1])
        pre = tf.argmax(prob, 1)
        self.pre = tf.reshape(pre, [self.batch_size, -1])

        targets = tf.reshape(self.targets, [-1])
        loss = seq2seq.sequence_loss_by_example([logits],
                                                [targets],
                                                [tf.ones_like(targets, dtype=tf.float32)])

        self.loss = tf.reduce_mean(loss)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 5)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))


    def unit(self):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.num_units)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
        return lstm_cell