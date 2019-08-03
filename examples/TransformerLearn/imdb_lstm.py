# -*- coding: utf-8 -*-
# @Time    : 2019/7/16 上午7:36
# @Author  : yangsen
# @Email   : 0@keepangry.com
# @File    : imdb_attention.py
# @Software: PyCharm
from __future__ import print_function
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import *

max_features = 10000
maxlen = 100
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


S_inputs = Input(shape=(None,), dtype='int32')
embeddings = Embedding(max_features, 128)(S_inputs)

O_seq = CuDNNLSTM(64)(embeddings)
O_seq = Dropout(0.5)(O_seq)

outputs = Dense(1, activation='sigmoid')(O_seq)

model = Model(inputs=S_inputs, outputs=outputs)
# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=5,
          validation_data=(x_test, y_test),
          verbose=2)
