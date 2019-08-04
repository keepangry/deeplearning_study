# -*- coding: utf-8 -*-
# @Time    : 2019/8/3 上午10:23
# @Author  : yangsen
# @Email   : 0@keepangry.com
# @File    : dataset.py
# @Software: PyCharm

import numpy as np
from tasks.chinese_word_segment.bert_bilstm_crf.config import train_file, seed, tag2id, dict_path
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from keras_bert import Tokenizer
import re


def load_dict(dict_path):
    token_dict = {}
    with open(dict_path) as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict


def load_sentences(filepath):
    np.random.seed(seed)
    sentences = open(filepath).readlines()
    sentences = [re.sub(r'\s+', ' ', sent.strip()) for sent in sentences]
    np.random.shuffle(sentences)
    return sentences


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            else:
                R.append('[UNK]')
        return R


# 生产y
def tag_sentence(sentence_words):
    """
    前后给s，或者另一个label
    :param sentence_words: list
    :return:
    """
    y = ['s']
    for word in sentence_words:
        if len(word) == 0:
            pass
        elif len(word) == 1:
            y.extend(['s'])
        elif len(word) == 2:
            y.extend(['b', 'e'])
        else:
            y.extend(['b'])
            y.extend(['m']*(len(word)-2))
            y.extend(['e'])
    y.extend('s')
    return y


def process_one(origin_sentence, tokenizer):
    sentence_words = origin_sentence.split(' ')
    sentence = origin_sentence.replace(' ', '')

    # tokens = tokenizer.tokenize(sentence)
    x1, x2 = tokenizer.encode(first=sentence)

    y = [tag2id.get(tag) for tag in tag_sentence(sentence_words)]
    if len(x1) != len(y):
        print(len(x1))
        print(len(y))
        print(sentence_words)
    assert len(x1) == len(y)
    return [x1, x2], y


def gene_data():
    token_dict = load_dict(dict_path)
    tokenizer = OurTokenizer(token_dict)
    sentences = load_sentences(train_file)

    X, y = [], []
    for sentence in sentences:
        _x, _y = process_one(sentence, tokenizer)
        X.append(np.array(_x))
        y.append(_y)
    return X, y


def data_generator(X, y, batch_size=128, maxlen=24):
    len_train = len(X)
    while True:
        batch_x1, batch_x2, batch_y = [], [], []
        for i in range(len_train):
            batch_x1.append(X[i][0])
            batch_x2.append(X[i][1])
            batch_y.append(y[i])

            if len(batch_x1) == batch_size or i == len_train-1:
                result_x1 = pad_sequences(batch_x1, maxlen=maxlen, value=0)
                result_x2 = pad_sequences(batch_x2, maxlen=maxlen, value=0)
                result_y = to_categorical(pad_sequences(batch_y, maxlen=maxlen, value=4), num_classes=5)
                yield [result_x1, result_x2], result_y
                batch_x1, batch_x2, batch_y = [], [], []


if __name__ == '__main__':
    X, y = gene_data()
    train_generator = data_generator(X=X, y=y)
    x, y = train_generator.__next__()
