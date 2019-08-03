# -*- coding: utf-8 -*-
# @Time    : 2019/7/28 下午5:58
# @Author  : yangsen
# @Email   : 0@keepangry.com
# @File    : 1.py
# @Software: PyCharm
import tensorflow as tf
import numpy as np

tf.enable_eager_execution()


print(tf.add(1, 2))
print(tf.add([1, 2], [3, 4]))
print(tf.square(5))
print(tf.reduce_sum([1, 2, 3]))
print(tf.io.encode_base64("hello world"))
print(tf.square(2) + tf.square(3))


a = np.arange(12).reshape(3, 4)
tf.expand_dims(a, 1)
tf.greater(tf.expand_dims(a, 1), 5)
tf.cast(tf.greater(tf.expand_dims(a, 1), 5), 'float32')
tf.cast(tf.greater(tf.expand_dims(a, 1), 5), 'float32')

tf.expand_dims(a, 1)[..., 0]
