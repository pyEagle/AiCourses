# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

np.random.seed(0)

def myfun(x):
    return np.sin(x)

x = np.linspace(0,15, 150)
y = myfun(x) + 1 + np.random.random(size=len(x)) * 0.3 - 0.15

input_len = 10

train_x = []
train_y = []
for i in range(len(y)-input_len-1):
    train_data = []
    for j in range(input_len):
        train_data.append([y[i+j]])
    train_x.append(train_data)
    train_y.append((y[i+input_len+1]))


model = tf.keras.Sequential()
model.add(tf.keras.layers.SimpleRNN(100, return_sequences=False, 
                    activation='relu',
                    input_shape=(input_len, 1)))

### 深度循环神经网络序列回归问题示例模型代码，运行时注释掉上面加层语句
# model.add(tf.keras.layers.SimpleRNN(100, activation='relu',
#                                     return_sequences=True,
#                                     input_shape=(input_len, 1)))
# model.add(tf.keras.layers.SimpleRNN(100, return_sequences=False, 
#                                     activation='relu'))
###
model.add(tf.keras.layers.Dense(1))
model.add(tf.keras.layers.Activation("relu"))
model.compile(loss= 'mean_squared_error', optimizer='adam')
model.summary()
model.fit(train_x, train_y, epochs=10, batch_size=10, verbose=1)

import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus']=False
plt.rc('font', family='SimHei', size=13)
#plt.scatter(x, y, color="black", linewidth=1)
y0 = myfun(x) + 1
plt.plot(x, y0, color="red", linewidth=1)
y1 = model.predict(train_x)
plt.plot(x[input_len+1:], y1, "b--", linewidth=1)
plt.show()

