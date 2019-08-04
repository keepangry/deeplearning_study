# -*- coding: utf-8 -*-
# @Time    : 2019/8/3 上午10:23
# @Author  : yangsen
# @Email   : 0@keepangry.com
# @File    : dataset.py
# @Software: PyCharm

import numpy as np
from collections import Counter
from utils import v2k

from tasks.chinese_word_segment.bilstm_cnn_crf.config import train_file, seed, tag2id, char2id_json_file
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import json

np.random.seed(seed)

sentences = open(train_file).readlines()
sentences = [sent.strip() for sent in sentences]

# 统计字频
char_count = Counter()
[char_count.update(sent) for sent in sentences]
min_count = 2   # 过滤低频字
chars = {i: j for i, j in char_count.items() if j >= min_count}    # 过滤低频字
id2char = {i+2: j for i, j in enumerate(chars)}       # id到字的映射, 1留给未登录字, 0mask
char2id = v2k(id2char)        # 字到id的映射

# 保存char2id
json.dump(char2id, open(char2id_json_file, 'w'))


sentences = [sent.split('  ') for sent in sentences]
np.random.shuffle(sentences)


# 生产X, y
def tag_sentence(sentence_words):
    """

    :param sentence_words: list
    :return:
    """
    y = []
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
    return y


def process_one(sentence_words):
    x = [char2id.get(char, 1) for char in ''.join(sentence_words)]
    y = [tag2id.get(tag) for tag in tag_sentence(sentence_words)]
    if len(x) != len(y):
        print(len(x))
        print(len(y))
        print(sentence_words)
    assert len(x) == len(y)
    return x, y


X, y = [], []
for sentence in sentences:
    _x, _y = process_one(sentence)
    X.append(np.array(_x))
    y.append(_y)

X_train = X[:-10000]
y_train = y[:-10000]
X_valid = X[-10000:]
y_valid = y[-10000:]


def data_generator(X, y, batch_size=128, maxlen=24):
    len_train = len(X)
    while True:
        batch_x, batch_y = [], []
        for i in range(len_train):
            batch_x.append(X[i])
            batch_y.append(y[i])

            if len(batch_x) == batch_size or i == len_train-1:
                result_x = pad_sequences(batch_x, maxlen=maxlen, value=0)
                result_y = to_categorical(pad_sequences(batch_y, maxlen=maxlen, value=4), num_classes=5)
                yield result_x, result_y
                batch_x, batch_y = [], []


def mem_data(X, y, maxlen=24):
    result_x = pad_sequences(X, maxlen=maxlen)
    result_y = to_categorical(pad_sequences(y, maxlen=maxlen, value=4), num_classes=5)
    return result_x, result_y


if __name__ == '__main__':
    train_generator = data_generator(X=X_train, y=y_train)
    i, j = train_generator.__next__()

    mem_data(X=X_train, y=y_train)
