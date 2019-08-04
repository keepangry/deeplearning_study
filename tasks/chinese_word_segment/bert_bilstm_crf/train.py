# -*- coding: utf-8 -*-
# @Time    : 2019/8/4 下午12:43
# @Author  : yangsen
# @Email   : 0@keepangry.com
# @File    : train.py
# @Software: PyCharm
from tasks.chinese_word_segment.bert_bilstm_crf.config import maxlen, num_classes, batch_size, model_file, valid_num
from tasks.chinese_word_segment.bert_bilstm_crf.models import bert_bilstm_crf
from keras.callbacks import ModelCheckpoint

from tasks.chinese_word_segment.bert_bilstm_crf.dataset import gene_data, data_generator


X, y = gene_data()
X_train = X[:-valid_num]
y_train = y[:-valid_num]
X_valid = X[-valid_num:]
y_valid = y[-valid_num:]


checkpointer = ModelCheckpoint(filepath=model_file, verbose=1, save_best_only=True, save_weights_only=False, mode='min')

train_generator = data_generator(X=X_train, y=y_train, batch_size=batch_size, maxlen=maxlen)
valid_generator = data_generator(X=X_valid, y=y_valid, batch_size=batch_size, maxlen=maxlen)

model = bert_bilstm_crf(maxlen=maxlen, num_classes=num_classes)

model.fit_generator(train_generator,
                    steps_per_epoch=len(X_train)/batch_size,
                    validation_data=valid_generator,
                    validation_steps=len(X_valid)/batch_size,
                    epochs=10,
                    callbacks=[checkpointer])

