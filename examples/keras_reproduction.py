import os
import random as rn
import keras.backend as K
import tensorflow as tf
import numpy as np
seed = 123
rn.seed(seed)
os.environ['PYTHONHASHSEED'] = '0'
tf.set_random_seed(seed)
np.random.seed(seed)
gpu_options = tf.GPUOptions(allow_growth=True)
session_conf = tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# session_conf = tf.ConfigProto(gpu_options=gpu_options)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
