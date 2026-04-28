# -*- coding:utf-8 -*-

import os
import tensorflow as tf

from tensorflow import keras

h5_file = "my_model_v1_final.h5" 
model.save(h5_file)

try:
    new_model = keras.models.load_model(h5_file)
    print(f"H5 加载成功: {h5_file}")
except Exception as e:
    print(f"H5 加载挂了，检查下是不是有自定义层: {e}")

export_dir = "./export/v1_production"

if not os.path.exists(export_dir):
    tf.saved_model.save(model, export_dir)

imported = tf.saved_model.load(export_dir)
