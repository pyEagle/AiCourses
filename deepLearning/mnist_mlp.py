# -*- coding:utf-8 -*-

import numpy as np
import tensorflow.keras as ka
import datetime

np.random.seed(0)

(X_train, y_train), (X_test, y_test) = ka.datasets.mnist.load_data()
#(X_train, y_train), (X_test, y_test) = ka.datasets.mnist.load_data(path="./mnist.npz")

num_pixels = X_train.shape[1] * X_train.shape[2]  # 784

# 将二维的数组拉成一维的向量
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

X_train = X_train / 255
X_test = X_test / 255

y_train = ka.utils.to_categorical(y_train)  # 转化为独热编码
y_test = ka.utils.to_categorical(y_test)
num_classes = y_test.shape[1]  # 10

# 多层全连接神经网络模型
model = ka.Sequential([
    ka.layers.Dense(num_pixels, input_shape=(num_pixels,),
                    kernel_initializer='normal', activation='relu'),
    ka.layers.Dense(784, kernel_initializer='normal', activation='relu'),
    ka.layers.Dense(num_classes, kernel_initializer='normal',
                    activation='softmax')
])
model.summary()

# model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

startdate = datetime.datetime.now()  # 获取当前时间
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=200, verbose=2)
enddate = datetime.datetime.now()

print("训练用时：" + str(enddate - startdate))


