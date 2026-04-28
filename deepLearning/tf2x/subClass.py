# -*- coding:utf-8 -*-

import tensorflow as tf

from tensorflow import keras

class WideDeepNet(keras.Model):
    def __init__(self, units=30, activation="relu", **kwargs):
        super(WideDeepNet, self).__init__(**kwargs)
        
        self.dense_1 = keras.layers.Dense(units, activation=activation)
        self.dense_2 = keras.layers.Dense(units, activation=activation)
        
        self.head_main = keras.layers.Dense(1, name="main_out")
        self.head_aux = keras.layers.Dense(1, name="aux_out")

    def call(self, inputs):
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            in_wide, in_deep = inputs
        else:
            raise ValueError("这个模型需要两个输入：[wide_input, deep_input]")

        x = self.dense_1(in_deep)
        x = self.dense_2(x)
        
        aux_output = self.head_aux(x)
        
        merged = keras.layers.concatenate([in_wide, x])
        main_output = self.head_main(merged)
        
        return main_output, aux_output

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.dense_1.units,
            "activation": keras.activations.serialize(self.dense_1.activation),
        })
        return config

model = WideDeepNet(units=64, name="v1_resnet_style")
