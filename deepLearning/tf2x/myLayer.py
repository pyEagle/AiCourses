# -*- coding:utf-8 -*-

import tensorflow as tf

from tensorflow import keras

class MyDense(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        last_dim = input_shape[-1]
        self.kernel = self.add_weight(
            name="kernel",
            shape=[last_dim, self.units],
            initializer="glorot_normal",
            trainable=True # 显式声明可训练
        )
        self.bias = self.add_weight(
            name="bias",
            shape=[self.units],
            initializer="zeros",
            trainable=True
        )
        
        super().build(input_shape)

    def call(self, inputs):
        z = inputs @ self.kernel + self.bias
        return self.activation(z) if self.activation else z

    def get_config(self):
        # 序列化配置，model.save()能正常load出来
        cfg = super().get_config()
        cfg.update({
            "units": self.units,
            "activation": keras.activations.serialize(self.activation)
        })
        return cfg

