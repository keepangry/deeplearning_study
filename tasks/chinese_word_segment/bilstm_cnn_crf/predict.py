# -*- coding: utf-8 -*-
# @Time    : 2019/8/3 下午12:50
# @Author  : yangsen
# @Email   : 0@keepangry.com
# @File    : predict.py
# @Software: PyCharm

from tasks.chinese_word_segment.bilstm_cnn_crf.models import bilstm_cnn_crf
import numpy as np
from tasks.chinese_word_segment.bilstm_cnn_crf.config import char2id_json_file, maxlen, id2tag, model_file, num_classes
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import json

char2id = json.load(open(char2id_json_file))
char_size = len(char2id)+2


def gene_x(sentence):
    x = []
    for char in sentence:
        x.append(char2id.get(char, 1))
    return pad_sequences([x], maxlen=maxlen, value=0)


def parse_pred_y(model_pred, sentence):
    """
    todo: 预测的有可能不符合 sbme 规范。
    :param model_pred:
    :param sentence:
    :return:
    """
    length = len(sentence)
    tags = [id2tag.get(np.argmax(pos), 's') for pos in model_pred[0][-length:]]
    result = []
    for i in range(len(tags)):
        if tags[i] in 'bs':
            result.append(' ')
        result.append(sentence[i])
    return (''.join(result)).strip()


def test(model, sentence):
    test_x = gene_x(sentence)
    model_pred = model.predict(test_x)
    return parse_pred_y(model_pred, sentence)


model = bilstm_cnn_crf(maxlen=maxlen,
                       char_size=char_size,
                       num_classes=num_classes
                       )
model.load_weights(model_file)

test(model, "他来到中国，成为第一个访华的大船主。")   # real: 他  来到  中国  ，  成为  第一个  访  华  的  大  船主  。
test(model, "有些大学生眼高手低，不屑于做小事情。")   # real: 有些  大学生  眼高手低  ，  不屑于  做  小  事情  。

print(test(model, "我是中国人"))
