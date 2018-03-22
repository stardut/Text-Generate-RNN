# -*- coding: utf-8 -*-

import os
import sys
import time
import tensorflow as tf
import numpy as np
import random
from net import Net
from data import Data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Predictor(object):
    """docstring for predict"""
    def __init__(self):
        super(Predictor, self).__init__()
        num_units = 512
        num_layer = 2
        batch_size = 1

        self.data = Data(data_dir, input_file, vocab_file, tensor_file, batch_size=batch_size)
        self.model = Net(data, num_units, num_layer, batch_size)

        self.sess = tf.Session()
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(self.sess, 'model/model')

    def predict(self, text, chk_char):        
        x = self.text2np(text, data)
        state = self.model.cell.zero_state(1, tf.float32)
        word, state = self.run(state, x, [len(x)])
        while word != chk_char:
            text += word
            x = self.text2np(text, data)
            self.run(state, x, [len(x)])
        text += word
        return text        

    def run(self, state, inputs, seq_len):
        feed = {
            self.model.inputs: inputs,
            self.model.keep_prob: 1.0,
            self.model.seq_len: seq_len,
            self.model.init_state: state
        }
        prob, state = sess.run([model.prob, model.final_state], feed_dict=feed)
        word = self.choose_word(prob[0][-1])
        return word, state

    def text2np(self.text, data):
        res = np.zeros((1, len(text)))
        res[0] = np.asarray(list(map(data.char2id, text)))
        return res

    def choose_word(self, prob):
        flag = random.random(0, 1)
        t = 0
        for idx, i in enumerate(prob):
            t += i
            if flag < t:
                return self.data.id2char[idx]

pre = Predictor()

text1 = '两个黄鹂鸣翠柳，'
text2 = '<《春》 '
text3 = '<'

print(pre.predict(text1, '。'))
print(pre.predict(text2, '>'))
print(pre.predict(text3, '>'))