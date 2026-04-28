
# -*- coding:utf-8 -*-

"""如果希望能够使用save方法保存模型并使用keras.models.load_model函数加载模型，则必须在两个ResidualBlock类和ResidualRegressor类中都实现get_config方法。另外，也可以使用save_weights和load_weights方法保存和加载权重。
注：
    Model类是Layer类的子类，因此可以像定义层一样定义和使用模型。但是模型具有一些额外的功能，包括其compile、fit、evaluate和predict方法（以及一些变体）以及get_layers方法（可以按名称或按索引返回任何模型的层）和save方法（支持keras.models.load_model和keras.models.clone_model。
    如果模型提供的功能比层更多，为什么不将每个层都定义为模型？从技术上讲可以，但是通常可以轻松地将模型的内部组件（即层或可重复使用的层块）与模型本身（即要训练的对象）区分开来。前者应继承Layer类，而后者应继承Model类。
"""

import tensorflow as tf

from tensorflow import keras


class ResBlock(keras.layers.Layer):
    def __init__(self, n_layers, n_neurons, **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.main_layers = [
            keras.layers.Dense(n_neurons, activation="elu", kernel_initializer="he_normal")
            for _ in range(n_layers)
        ]

    def call(self, inputs):
        z = inputs
        for layer in self.main_layers:
            z = layer(z)
        return inputs + z

class ResidualRegressor(keras.Model):
    def __init__(self, output_dim, **kwargs):
        super(ResidualRegressor, self).__init__(**kwargs)
        common_kwargs = {"activation": "elu", "kernel_initializer": "he_normal"}
        
        self.head = keras.layers.Dense(30, **common_kwargs)
        self.res_block_a = ResBlock(2, 30)
        self.res_block_b = ResBlock(2, 30)
        
        self.regressor_out = keras.layers.Dense(output_dim)

    def call(self, inputs):
        x = self.head(inputs)
        
        for _ in range(3): 
            x = self.res_block_a(x)
        
        x = self.res_block_b(x)

        return self.regressor_out(x)

    def get_config(self):
        conf = super().get_config()
        return conf

