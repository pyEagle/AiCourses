# -*- coding: utf-8 -*-


import numpy as np
import tensorflow.keras as ka
import datetime

np.random.seed(0)
 
(X_train, y_train), (X_test, y_test) = ka.datasets.mnist.load_data() 
 

# 将二维的数组拉成一维的向量
X_train = X_train.reshape(X_train.shape[0],28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

X_train = X_train / 255
X_test = X_test / 255
 
y_train = ka.utils.to_categorical(y_train) # 转化为独热编码
y_test = ka.utils.to_categorical(y_test)
num_classes = y_test.shape[1] # 10

# CNN模型
model = ka.Sequential([
    ka.layers.Conv2D(filters=32, kernel_size=(5, 5), input_shape=(28, 28, 1), activation='relu'),
    ka.layers.MaxPooling2D(pool_size=(2, 2)),
    ka.layers.Dropout(0.2),
    ka.layers.Flatten(),
    ka.layers.BatchNormalization(),
    ka.layers.Dense(128, activation='relu'),
    ka.layers.Dense(num_classes, activation='softmax')
])
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

startdate = datetime.datetime.now() # 获取当前时间
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=200, verbose=2)
enddate = datetime.datetime.now()

print("训练用时：" + str(enddate - startdate))


 