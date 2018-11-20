# HW2-The-Simpsons-Characters-ecognition-Challenge
second HW of machine learning

# 工作環境
----------------
1.Ubuntu 16.04

2.Python 3.5.6

3.Tensorflow 1.10.0

4.Keras 2.2.3

## 作業要求
--------------------
從辛普森家族中選出20個角色，並在約1000張噴別在不同場景、不同時間中擷取出來的圖片中辨識出這些角色各是誰
注意:角色並不一定都在圖片中央

## 程式碼
-------------------
1.匯入相關所需的模組
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dense

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint

from timeit import default_timer as timer

from keras.preprocessing import image

import numpy as np

import os

from PIL import Image

from keras.models import load_model

import pandas as pd


2.將圖像大小調整為128
image_height = 128

image_width = 128

3.初始化CNN 建立一個含有6層的卷積層的網路
predator = Sequential()
 1.第一層
predator.add(Conv2D(32, (3, 3), activation="relu", input_shape=(image_height, image_width, 3)))
predator.add(MaxPooling2D(pool_size = (2, 2)))
 2.添加第二層
predator.add(Conv2D(64, (3, 3), activation="relu"))
predator.add(MaxPooling2D(pool_size = (2, 2)))
 3.添加第三層
predator.add(Conv2D(128, (3, 3), activation="relu"))
predator.add(MaxPooling2D(pool_size = (2, 2)))
 4.添加第四層
predator.add(Conv2D(256, (3, 3), activation="relu"))
predator.add(MaxPooling2D(pool_size = (2, 2)))
 5.添加第五及六層
predator.add(Conv2D(128, (3, 3), activation="relu"))
predator.add(Conv2D(64, (3, 3), activation="relu"))
predator.add(MaxPooling2D(pool_size = (2, 2)))
 6.將3D圖像展開成單行陣列
predator.add(Flatten())    
 7.Full connection
predator.add(Dense(units=32, activation="relu"))
predator.add(Dense(units=20, activation="softmax"))   #輸出層有20個神經元 每個神經元代表一個角色
 8.編譯CNN並且展示model架構
predator.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
predator.summary()

4.用ImageDataGenerator將圖片做預處理並且將圖片載入
train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory("/home/t107368084/hw2/train/characters-20",target_size = (image_height, image_width),batch_size = 50,class_mode = 'categorical')
test_set = test_datagen.flow_from_directory("/home/t107368084/hw2/test/characters-20",target_size = (image_height, image_width),batch_size = 50,class_mode = 'categorical')

5.#設定訓練相關數值
predator.fit_generator(training_set,steps_per_epoch = 16,epochs = 100,validation_data = test_set,validation_steps = 4)

6.載入test data並辨識
result = []
path = '/home/t107368084/hw2/valid/test'
files = os.listdir(path)

for file in files:
    test_image = image.load_img('/home/t107368084/hw2/valid/test/'+file, target_size=(image_height, image_width))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    pred = predator.predict_on_batch(test_image)
    result.append(pred)

result = np.asarray(result)

7.將結果表格化並輸出成CSV檔
index = files
predictions = result[:, [0]][:,0]
df = pd.DataFrame(index=index)
#以下是我們本次作業要辨認的20位人物
df['abraham_grampa_simpson'] = predictions[:,0]
df['apu_nahasapeemapetilon'] = predictions[:,1]
df['bart_simpson'] = predictions[:,2]
df['charles_montgomery_burns'] = predictions[:,3]
df['chief_wiggum'] = predictions[:,4]
df['comic_book_guy'] = predictions[:,5]
df['edna_krabappel'] = predictions[:,6]
df['homer_simpson'] = predictions[:,7]
df['kent_brockman'] = predictions[:,8]
df['krusty_the_clown'] = predictions[:,9]
df['lenny_leonard'] = predictions[:,10]
df['lisa_simpson'] = predictions[:,11]
df['marge_simpson'] = predictions[:,12]
df['mayor_quimby'] = predictions[:,13]
df['milhouse_van_houten'] = predictions[:,14]
df['moe_szyslak'] = predictions[:,15]
df['ned_flanders'] = predictions[:,16]
df['nelson_muntz'] = predictions[:,17]
df['principal_skinner'] = predictions[:,18]
df['sideshow_bob'] = predictions[:,19]

df = df.astype(int)

df.to_csv('/home/t107368084/hw2/predictions.csv')

8.儲存模型

predator.save_weights('/home/t107368084/hw2/simpsons.hdf5')

## 模型架構

## 訓練過程

## kaggle排名


