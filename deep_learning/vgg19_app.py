# -*- coding: utf-8 -*-

import sys

import tensorflow.keras.applications.vgg19 as vgg19
import tensorflow.keras.preprocessing.image as imagepre

# 加载预训练模型
model_file = sys.argv[1]
model = vgg19.VGG19(weights=model_file, include_top=True)
# 加载图片并转换为合适的数据形式
img_file = sys.argv[2]
image = imagepre.load_img(img_file, target_size=(224, 224))
imagedata = imagepre.img_to_array(image)
imagedata = imagedata.reshape((1,) + imagedata.shape)

imagedata = vgg19.preprocess_input(imagedata)
prediction = model.predict(imagedata) # 分类预测
results = vgg19.decode_predictions(prediction, top=3)
print(results)
