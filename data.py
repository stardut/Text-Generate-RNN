import sys
import os
import time
import numpy as np
import collections
from six.moves import cPickle

BEGIN_CHAR = '<'
END_CHAR = '>'
MAX_LENGTH = 280
MIN_LENGTH = 10



class Data:
    def __init__(self, data_dir, input_file, vocab_file, 
            tensor_file, is_train=True, seq_len=300, batch_size = 64):
        global MAX_LENGTH
        MAX_LENGTH = seq_len
        self.batch_size = batch_size
        self.unknow_char = '*'
        self.point = 0
        input_file = os.path.join(data_dir, input_file)
        vocab_file = os.path.join(data_dir, vocab_file)
        tensor_file = os.path.join(data_dir, tensor_file)

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_vocab(vocab_file)
            if is_train:
                self.load_tensor(tensor_file)
                self.create_batches()
        print('load data done')
        # print(self.words_size)

    def id2char(self, idx):
        return self.vocab_id[idx]

    def char2id(self, word):
        return self.vocab[word]

    def load_tensor(self, tensor_file):
        print('reading: ' + tensor_file)
        self.texts_vector = np.load(tensor_file)
        print('poetries number: %d' % (self.texts_vector.shape[0]))

    def load_vocab(self, vocab_file):
        print('reading: ' + vocab_file)
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = {v : i for i, v in enumerate(self.chars)}
        self.vocab_id = dict(enumerate(self.chars))
        self.words_size = len(self.chars)
        print('words size: %d' % (self.words_size))
        self.words = self.chars

    def preprocess(self, input_file, vocab_file, tensor_file):
        def handle(line):
            if len(line) > MAX_LENGTH:
                index_end = line.rfind('。', 0, MAX_LENGTH)
                index_end = index_end if index_end > 0 else MAX_LENGTH
                line = line[:index_end + 1]
            return BEGIN_CHAR + line + END_CHAR

        self.texts = [line.strip().replace('\n', '') for line in
                        open(input_file, encoding='utf-8')]
        self.texts = [handle(line) for line in self.texts if len(line) > MIN_LENGTH]

        words = ['*', ' ']
        for text in self.texts:
            words += [word for word in text]
        self.words = list(set(words))
        self.words_size = len(self.words)

        self.vocab = dict(zip(self.words, range(len(self.words))))
        self.vocab_id = dict(zip(range(len(self.words)), self.words))
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.words, f)
        self.texts_vector = np.array([
            list(map(self.vocab.get, poetry)) for poetry in self.texts])
        np.save(tensor_file, self.texts_vector)

    def create_batches(self):
        self.n_size = len(self.texts_vector) // self.batch_size
        assert self.n_size > 0, 'data set is too small and need more data.'

        self.texts_vector = self.texts_vector[:self.n_size * self.batch_size]
        self.x_batches = []
        self.y_batches = []
        self.seq_lenes = []
        for i in range(self.n_size):
            batches = self.texts_vector[i * self.batch_size : (i + 1) * self.batch_size]
            length = max(map(len, batches))
            # 将长度不足的用 * 补充
            
            seq_len = []
            for row in range(self.batch_size):
                t_len = len(batches[row])
                seq_len.append(t_len)
                if t_len < length:
                    r = length - t_len
                    batches[row][t_len : length] = [self.vocab[self.unknow_char]] * r

            xdata = np.array(list(map(lambda x: np.array(x), batches)))
            ydata = np.copy(xdata)
            # 将标签整体往前移动一位， 代表当前对下一个的预测值
            ydata[:, :-1] = xdata[:, 1:]
            ydata[:, -1] = xdata[:, 0]
            self.x_batches.append(xdata)
            self.y_batches.append(ydata)
            self.seq_lenes.append(seq_len)

    def get_batches(self):
        inputs = self.x_batches[self.point]
        labels = self.y_batches[self.point]
        seq_len = self.seq_lenes[self.point]
        self.point = 0 if self.point == len(self.x_batches)-1 else self.point+1
        return inputs, labels, seq_len

