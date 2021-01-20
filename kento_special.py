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

# パラメータ設定
labels = ["0", "1", "2", "3", "4"]          # ラベル設定
NUM_LABELS = len(labels)                    # 分類数
LEARNING_RATE = 0.001                       # 学習率
EPOCHS = 20                                 # エポック数
BATCH = 8                                   # バッチサイズ
HEIGHT = 600                                # 画像の高さ
WIDTH = 800                                 # 画像の幅


