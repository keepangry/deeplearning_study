# -*- coding: utf-8 -*-
# @Time    : 2019/8/3 上午10:21
# @Author  : yangsen
# @Email   : 0@keepangry.com
# @File    : train.py
# @Software: PyCharm

from tasks.chinese_word_segment.bilstm_cnn_crf.dataset import *
from tasks.chinese_word_segment.bilstm_cnn_crf.config import *
from tasks.chinese_word_segment.bilstm_cnn_crf.models import bilstm_cnn_crf
from keras.callbacks import ModelCheckpoint


char_size = len(char2id)+2

model = bilstm_cnn_crf(maxlen=maxlen,
                       char_size=char_size,
                       num_classes=num_classes
                       )


checkpointer = ModelCheckpoint(filepath=model_file, verbose=1, save_best_only=True, save_weights_only=False, mode='min')

# in mem
X_train, y_train = mem_data(X=X_train, y=y_train, maxlen=maxlen)
X_valid, y_valid = mem_data(X=X_valid, y=y_valid, maxlen=maxlen)
model.fit(x=X_train, y=y_train, epochs=10, batch_size=batch_size, validation_data=[X_valid, y_valid],
          callbacks=[checkpointer])


# generator
# train_generator = data_generator(X=X_train, y=y_train, batch_size=batch_size, maxlen=maxlen)
# valid_generator = data_generator(X=X_valid, y=y_valid, batch_size=batch_size, maxlen=maxlen)
# model.fit_generator(train_generator,
#                     steps_per_epoch=100,
#                     validation_data=valid_generator,
#                     validation_steps=80,
#                     epochs=10,
#                     callbacks=[checkpointer])
