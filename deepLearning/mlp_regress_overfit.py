# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import regularizers


def myfun(x):
    '''目标函数
    input:x(float):自变量
    output:函数值'''
    return 10 + 5 * x + 4 * x ** 2 + 6 * x ** 3


# x = np.linspace(-3,3, 20)
# y = myfun(x) + np.random.random(size=len(x)) * 100 - 50

x = [-3., -2.68421053, -2.36842105, -2.05263158, -1.73684211, -1.42105263,
     -1.10526316, -0.78947368, -0.47368421, -0.15789474, 0.15789474, 0.47368421,
     0.78947368, 1.10526316, 1.42105263, 1.73684211, 2.05263158, 2.36842105,
     2.68421053, 3.]
y = [-83.60437309, -109.02680368, -99.45599857, -72.85246379, 24.27643468,
     22.32819066, 13.0134867, -37.47252415, -16.24274272, 21.5705342,
     -12.63210639, 35.16554616, 42.58380499, 21.97718399, 19.50677405,
     107.2591151, 67.41705564, 95.78691168, 130.32069909, 253.31473912]

yy = y.copy()

miny = min(y)
maxy = max(y)

def standard(y, miny, maxy):
    step = maxy - miny
    for i in range(len(y)):
        y[i] = (y[i] - miny) / step

standard(y, miny, maxy)

def invstandard(y, miny, maxy):
    step = maxy - miny
    for i in range(len(y)):
        y[i] = miny + y[i] * step

# 早停法
model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, activation='sigmoid', input_shape=(1,),
                          kernel_initializer='random_uniform', bias_initializer='zeros'),
    tf.keras.layers.Dense(5, activation='sigmoid',
                          kernel_initializer='random_uniform', bias_initializer='zeros'),
    tf.keras.layers.Dense(1, activation='sigmoid',
                          kernel_initializer='random_uniform', bias_initializer='zeros')
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.mean_squared_error)
earlyStopping=tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.000001, patience=15, verbose=1, mode='min')

model.fit(x, y, batch_size=20, epochs=1000, verbose=1, callbacks=[earlyStopping],
     validation_data=(x, y))


# # 正则化
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(5, activation='sigmoid', input_shape=(1,),
#                           kernel_initializer='random_uniform', bias_initializer='zeros'),
#     tf.keras.layers.Dense(5, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001),
#                           kernel_initializer='random_uniform', bias_initializer='zeros'),
#     tf.keras.layers.Dense(1, activation='sigmoid',
#                           kernel_initializer='random_uniform', bias_initializer='zeros')
# ])

# # Dropout
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(5, activation='sigmoid', input_shape=(1,),
#                           kernel_initializer='random_uniform', bias_initializer='zeros'),
#     tf.keras.layers.Dropout(0.1),
#     tf.keras.layers.Dense(5, activation='sigmoid',
#                           kernel_initializer='random_uniform', bias_initializer='zeros'),
#     tf.keras.layers.Dense(1, activation='sigmoid',
#                           kernel_initializer='random_uniform', bias_initializer='zeros')
# ])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.mean_squared_error)

model.fit(x, y, batch_size=20, epochs=10000, verbose=1)

model.summary()

plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', family='SimHei', size=13)
plt.scatter(x, yy, color="black", linewidth=2)
x1 = np.linspace(-3, 3, 100)
y0 = myfun(x1)
plt.plot(x1, y0, color="red", linewidth=1)
y1 = model.predict(x1)
invstandard(y1, miny, maxy)
plt.plot(x1, y1, "b--", linewidth=1)
plt.show()
