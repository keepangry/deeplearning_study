# -*- coding: utf-8 -*-
# @Time    : 2019/8/4 上午8:40
# @Author  : yangsen
# @Email   : 0@keepangry.com
# @File    : models.py
# @Software: PyCharm
from keras_bert import load_trained_model_from_checkpoint
from keras.layers import Input, Bidirectional, LSTM, Dropout
from keras.layers import Dense, TimeDistributed
from keras import Model

from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy
from tasks.chinese_word_segment.bert_bilstm_crf.config import config_path, checkpoint_path


# Epoch 10/10 - 180s 299ms/step - loss: 0.4085 - crf_accuracy: 0.9735 - val_loss: 0.4234 - val_crf_accuracy: 0.9657
def bert_bilstm_crf(maxlen, num_classes):
    word_input = Input(shape=(None,))
    seg_input = Input(shape=(None,))

    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
    # for l in bert_model.layers:
    #     l.trainable = True

    word_emb = bert_model([word_input, seg_input])
    bilstm = Bidirectional(LSTM(256, return_sequences=True))(word_emb)
    bilstm_d = Dropout(0.1)(bilstm)

    dense = TimeDistributed(Dense(num_classes))(bilstm_d)
    # crf
    crf = CRF(num_classes, sparse_target=False)
    crf_output = crf(dense)

    # build model
    model = Model(inputs=[word_input, seg_input], outputs=[crf_output])
    model.compile(loss=crf_loss, optimizer='adam', metrics=[crf_accuracy])

    return model


if __name__ == "__main__":
    model = bert_bilstm_crf(maxlen=40, num_classes=5)
    model.summary()
