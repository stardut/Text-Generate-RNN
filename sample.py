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
        data_dir = 'data/'
        input_file = 'poetry.txt'
        vocab_file = 'vocab.pkl'
        tensor_file = 'tensor.npy'

        self.data = Data(data_dir, input_file, vocab_file, tensor_file, 
                        is_train=False, batch_size=batch_size)
        self.model = Net(self.data, num_units, num_layer, batch_size)
        self.sess = tf.Session()

        saver = tf.train.Saver(tf.global_variables())
        saver.restore(self.sess, 'model/model')
        print('Load model done.' + '\n')

    def predict(self, text, chk_char):
        x = self.text2np(text)
        state = self.model.cell.zero_state(1, tf.float32)
        state = self.sess.run(state)
        word, state = self.run(state, x, [len(x)])
        while word != chk_char:
            text += word
            x = self.text2np(word)
            word, state = self.run(state, x, [1])
        text += word
        return text

    def run(self, state, inputs, seq_len):
        feed = {
            self.model.inputs: inputs,
            self.model.keep_prob: 1.0,
            self.model.seq_len: seq_len,
            self.model.init_state: state
        }
        prob, state = self.sess.run([self.model.prob, self.model.final_state], feed_dict=feed)
        prob = np.reshape(prob, (1, -1, self.data.words_size))
        word = self.choose_word(prob[0][-1])
        return word, state

    def text2np(self, text):
        res = np.zeros((1, len(text)))
        res[0] = np.asarray(list(map(self.data.char2id, text)))
        return res

    def choose_word(self, prob):
        flag = random.random()
        t = 0
        for idx, i in enumerate(prob):
            t += i
            if flag < t:
                return self.data.id2char(idx)

pre = Predictor()

text1 = '两个黄鹂鸣翠柳，'
text2 = '<《春》 '
text3 = '<'

print(pre.predict(text1, '。') + '\n')
print(pre.predict(text2, '>')[1:-1] + '\n')
print(pre.predict(text3, '>')[1:-1] + '\n')
pre.sess.close()