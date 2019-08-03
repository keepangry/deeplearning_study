# -*- coding: utf-8 -*-
# @Time    : 2019/7/28 下午5:19
# @Author  : yangsen
# @Email   : 0@keepangry.com
# @File    : train.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense, Lambda
from tensorflow.python.keras.models import Model


inputs = Input(shape=(30, 64,))
x = Dense(32, activation='relu', use_bias=False)(inputs)
x = Dense(10, activation='relu')(x)
predictions = Dense(1, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# model.fit(data, labels)  # starts training
model.summary()