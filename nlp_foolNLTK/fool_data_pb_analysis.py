# -*- coding:utf-8 -*-
import os
import tensorflow as tf
from tensorflow.python.platform import gfile

# 基础配置：路径
basedir = os.path.dirname(os.path.abspath(__file__))
pb_data_path = os.path.join(basedir,'FoolNLTK-master/data')
ner_pb_path = os.path.join(pb_data_path, 'ner.pb')
pos_pb_path = os.path.join(pb_data_path, 'pos.pb')
seg_pb_path = os.path.join(pb_data_path, 'seg.pb')

# 加载pb文件，想要加载pb文件，需要找到保存的方法或当前的包中调用和读取的方式，否则无法或者相应的具体的参数情况下，无法直接加载读取。
# with tf.Session() as sess:
