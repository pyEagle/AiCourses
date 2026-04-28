# -*- coding:utf-8 -*-

import tensorflow as tf

from tensorflow import keras


class HuberLoss(keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        error = y_true - y_pred
        
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss = self.threshold * (tf.abs(error) - self.threshold / 2)
        
        return tf.where(is_small_error, squared_loss, linear_loss)

    def get_config(self):
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config

custom_map = {"HuberLoss": HuberLoss}
try:
    model = keras.models.load_model(
        "my_model_v2_final.h5", # TODO
        custom_objects=custom_map
    )
    print("模型加载成功！")
except Exception as e:
    print(f"加载失败，检查下是不是自定义层没传对: {e}")

