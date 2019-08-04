# -*- coding: utf-8 -*-
# @Time    : 2019/8/3 上午11:21
# @Author  : yangsen
# @Email   : 0@keepangry.com
# @File    : models.py
# @Software: PyCharm

# keras_contrib.layers.CRF 与 tf.keras存在兼容问题。
# from tensorflow.python.keras.layers import Input, Embedding, Bidirectional, LSTM, Dropout, ZeroPadding1D, Conv1D
# from tensorflow.python.keras.layers import Dense, concatenate, TimeDistributed
# from tensorflow.python.keras import Model
from keras.layers import Input, Embedding, Bidirectional, LSTM, CuDNNLSTM, Dropout, ZeroPadding1D, Conv1D
from keras.layers import Dense, concatenate, TimeDistributed
from keras import Model

from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy


# val_crf_accuracy: 0.9556
def bilstm_cnn_crf(maxlen, char_size, num_classes):
    word_input = Input(shape=(maxlen,), dtype='int32', name='word_input')
    word_emb = Embedding(char_size, output_dim=256, input_length=maxlen, name='word_emb')(word_input)

    # bilstm
    bilstm = Bidirectional(CuDNNLSTM(128, return_sequences=True))(word_emb)
    bilstm_d = Dropout(0.2)(bilstm)

    # cnn
    conv = Conv1D(filters=128, kernel_size=5, padding='same')(bilstm_d)
    conv_d = Dropout(0.2)(conv)
    dense_conv = TimeDistributed(Dense(64))(conv_d)

    # merge
    rnn_cnn_merge = concatenate([bilstm_d, dense_conv], axis=2)
    dense = TimeDistributed(Dense(num_classes))(rnn_cnn_merge)

    # crf
    crf = CRF(num_classes, sparse_target=False)
    crf_output = crf(dense)

    # build model
    model = Model(inputs=[word_input], outputs=[crf_output])

    model.compile(loss=crf_loss, optimizer='adam', metrics=[crf_accuracy])
    return model


if __name__ == "__main__":
    model = bilstm_cnn_crf(maxlen=24, char_size=1000, num_classes=5)
    model.summary()
