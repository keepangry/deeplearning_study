# -*- coding: utf-8 -*-
# @Time    : 2019/8/3 下午1:47
# @Author  : yangsen
# @Email   : 0@keepangry.com
# @File    : global_config.py
# @Software: PyCharm

from global_config import BASE_PATH
from utils import v2k
PROJECT_PATH = BASE_PATH + '/tasks/chinese_word_segment/bert_bilstm_crf'
seed = 0
train_file = BASE_PATH + '/data/chinese_word_segment/icwb2-data/training/msr_training.utf8'
batch_size = 128
maxlen = 40
num_classes = 5

valid_num = 10000

id2tag = {0: 's', 1: 'b', 2: 'm', 3: 'e'}   # 标签（sbme）与id之间的映射
tag2id = v2k(id2tag)
char2id_json_file = PROJECT_PATH + '/char2id.tmp.json'


model_file = PROJECT_PATH + '/model/v1.hdf5'

config_path = BASE_PATH + '/data/pretrain_data/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = BASE_PATH + '/data/pretrain_data/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = BASE_PATH + '/data/pretrain_data/chinese_L-12_H-768_A-12/vocab.txt'
