# -*- coding: utf-8 -*-

import os
import sys
import time
import tensorflow as tf
import numpy as np
from net import Net
from data import Data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def text2np(text, data):
    res = np.zeros((1, len(text)))
    res[0] = np.asarray(list(map(data.char2id, text)))
    return res


num_units = 512
num_layer = 2
batch_size = 1

data = Data(data_dir, input_file, vocab_file, tensor_file, batch_size=batch_size)
model = Net(data, num_units, num_layer, batch_size)

with tf.Session() as sess:
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, 'model/model')

    text = '两个黄鹂鸣翠柳, '
    x = text2np(text, data)

    feed = {
        model.inputs: x,
        model.keep_prob: 1.0,
        model.seq_len: [len(x)],
        model.init_state: 
    }
    sess.run([model.pre], )
