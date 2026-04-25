# -*- coding: utf-8 -*-

import os
import re
import sys

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Dropout, TimeDistributed, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

# 重要参数
tags = {'S': 0, 'B': 1, 'M': 2, 'E': 3, 'X': 4} # 标签
embedding_size = 64 # 词向量大小
maxlen = 32 # 序列长度，长于32则截断，短于32则填充0
hidden_size = 64
batch_size = 64
epochs = 1
checkpointfilepath = 'weights.best.hdf5' # 中间结构保存文件
modepath = 'dz.h5' # 模型保存文件

# 0.从熟语料中提取出训练语句
txt = sys.argv[1]
file = open(txt, encoding='utf-8').read()
file = file.replace('[', '').replace(']', '')
file = file.split('\n')
new_sents = []
sents_labels = []
for para in file:
    para = para[23:]
    sents = re.split('[，。！？、]', para)
    for sent in sents:
        sent = sent.split()
        new_sent = ''
        sent_labels = ''
        for word in sent:
            word = word.strip().split('/')[0]
            if len(word) == 1:
                new_sent += word
                sent_labels += 'S'
            elif len(word) >= 2:
                new_sent += word
                sent_labels += 'B' + 'M'*(len(word)-2) + 'E'
        if new_sent != '':
            new_sents.append([new_sent])
            sents_labels.append([sent_labels])
print("训练样本准备完毕！")
print('共有数据 %d 条' % len(new_sents))
print('平均长度：', np.mean([len(d[0]) for d in new_sents]))

# 1.提取出所有用到的字，形成字典
stat = {}
for i in range(len(new_sents)):
    for v in new_sents[i][0]:
        stat[v] = stat.get(v, 0) + 1
stat = sorted(stat.items(), key=lambda x:x[1], reverse=True)
vocab = [s[0] for s in stat]
print("不同字的个数：" + str(len(vocab)))
char2id = {c : i + 1 for i, c in enumerate(vocab)} # 编号0为填充值，因此从1开始编号
id2char = {i + 1 : c for i, c in enumerate(vocab)}
print("字典创建完毕！")

# 2.将训练语句转化为训练样本
trainX = []
trainY = []
for i in range(len(new_sents)):
    x = [0] * maxlen # 默认填充值
    y = [4] * maxlen # 默认标签X
    sent = new_sents[i][0]
    labe = sents_labels[i][0]
    replace_len = len(sent)
    if len(sent) > maxlen:
        replace_len = maxlen
    for j in range(replace_len):
        x[j] = char2id[sent[j]]
        y[j] = tags[labe[j]]
    trainX.append(x)
    trainY.append(y)
trainX = np.array(trainX)
trainY = tf.keras.utils.to_categorical(trainY, 5)
print("训练样本准备完毕，训练样本共" + str(len(trainX)) + "句。")

# 3.搭建模型，并训练

X = Input(shape=(maxlen,), dtype='int32')
embedding = Embedding(input_dim=len(vocab)+1, output_dim=embedding_size, input_length=maxlen, mask_zero=True)(X)
blstm = Bidirectional(LSTM(hidden_size, return_sequences=True), merge_mode='concat')(embedding)
blstm = Dropout(0.4)(blstm)
blstm = Bidirectional(LSTM(hidden_size, return_sequences=True), merge_mode='concat')(blstm)
blstm = Dropout(0.4)(blstm)
output = TimeDistributed(Dense(5, activation='softmax'))(blstm)
model = Model(X, output)
model.summary()

if os.path.exists(checkpointfilepath): # 与下面的checkpoint起到及时保存训练结果的作用
    print("加载前次训练模型参数。。。")
    model.load_weights(checkpointfilepath)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
checkpoint = ModelCheckpoint(checkpointfilepath, monitor='acc', verbose=1, save_best_only=True,
                            mode='max')
model.fit(trainX, trainY, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint])
model.save(modepath)
print(model.evaluate(trainX, trainY, batch_size=batch_size))

# 4.利用训练好的模型进行分词
def predict(testsent):
    # 将汉字句子转换成模型需要的输入形式
    x = [0] * maxlen
    replace_len = len(testsent)
    if len(testsent) > maxlen:
        replace_len = maxlen
    for j in range(replace_len):
        x[j] = char2id[testsent[j]]
    # 调用模型进行预测
    label = model.predict([x]) 
    # 根据模型预测结果对输入句子进行切分
    label = np.array(label)[0]
    s = ''
    for i in range(len(testsent)):
        tag = np.argmax(label[i])
        if tag == 0 or tag == 3: # 单字和词结尾加空格切分
            s += testsent[i] + ' '
        elif tag ==1 or tag == 2:
            s += testsent[i]
    print(s)
testsent = "央视快评：在防控第一线考察识别评价使用干部"
print(testsent + "--->")
predict(testsent)
# output: 央视快评 ： 在 防控 第一线 考察 识别 评价 使用 干部

