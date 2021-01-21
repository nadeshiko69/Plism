import pandas as pd
import sys
import os
import shutil

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

# train_images.csvの上1000個をtest用にはけておく
'''
df = pd.read_csv('test.csv')
target = df.head(1000)['image_id']
for i in range(1000):
  shutil.move('train_images/'+target[i], 'test_images')
'''

# 正解ラベルでフォルダ分け 
'''
df = pd.read_csv('train.csv')
for num in range(len(df)):
  shutil.move('train_images/'+df.iloc[num]['image_id'], 'train_images/'+str(df.iloc[num]['label']))
df = pd.read_csv('test.csv')
for num in range(len(df)):
  shutil.move('test_images/'+df.iloc[num]['image_id'], 'test_images/'+str(df.iloc[num]['label']))
'''

# パラメータ設定
labels = ["0", "1", "2", "3", "4"]          # ラベル設定
NUM_LABELS = len(labels)                    # 分類数
LEARNING_RATE = 0.001                       # 学習率
EPOCHS = 20                                 # エポック数
BATCH = 8                                   # バッチサイズ
HEIGHT = 600                                # 画像の高さ
WIDTH = 800                                 # 画像の幅


# 前処理
# 画像データをTFにぶち込める形式に変換
# ラベルはone-hotに
train_data_gen = ImageDataGenerator(rescale=1./255)
val_data_gen = ImageDataGenerator(rescale=1./255)

train_data = train_data_gen.flow_from_directory('train_images/', 
                                                target_size=(WIDTH, HEIGHT),
                                                color_mode='rgb', 
                                                batch_size=BATCH,
                                                class_mode='categorical', 
                                                shuffle=True)

validation_data = val_data_gen.flow_from_directory('test_images/', 
                                                   target_size=(WIDTH, HEIGHT),
                                                   color_mode='rgb', 
                                                   batch_size=BATCH,
                                                   class_mode='categorical', 
                                                   shuffle=True)

# 画像データとラベルを連ねていく感じ
(image_data,label_data) = train_data.next()


# モデル作成
# MNISTパクった
# PCスペック足りなかったのでmodelちっちゃくした
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',input_shape=(WIDTH, HEIGHT, 3)))
model.add(Activation('relu'))
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Flatten())
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(NUM_LABELS))
model.add(Activation('softmax'))

opt = tf.keras.optimizers.Adam(lr=LEARNING_RATE)
model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

# model.summary()

history = model.fit(train_data, epochs=EPOCHS, validation_data=validation_data, verbose=1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training and validation loss')
plt.ylabel('loss')
plt.xlim([0.0, EPOCHS])
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'], loc='upper left')
plt.show()