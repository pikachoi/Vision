import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

(x_train, y_train), (x_test, y_test) = ds.cifar10.load_data()
x_train = x_train.reshape(50000, 3072)  # 1024*3=3072(RGB)
x_test = x_test.reshape(10000, 3072)
x_train = x_train.astype(np.float32)/255.0
x_test = x_test.astype(np.float32)/255.0
y_train = tf.keras.utils.to_categorical(y_train,10)
y_test = tf.keras.utils.to_categorical(y_test,10)

dmlp=Sequential()
dmlp.add(Dense(units=1024, activation='relu', input_shape=(3072,)))
dmlp.add(Dense(units=512, activation='relu'))
dmlp.add(Dense(units=512, activation='relu'))
dmlp.add(Dense(units=10, activation='softmax'))
# 모델용량: (3072+1)*1024+(1024+1)*512+(512+1)*512+(512+1)*10=3,939,338개의 가중치
# 데이터셋: cifar10 훈련 데이터는 50,000
# 모델 용량이 데이터셋보다 큼
# 신경망 모델 용량은 큰데 데이터셋 크기가 작은 경우 과적합 발생

dmlp.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001),metrics=['accuracy'])
hist=dmlp.fit(x_train, y_train, batch_size=128, epochs=50, validation_data=(x_test, y_test), verbose=2)
print('정확률=', dmlp.evaluate(x_test, y_test, verbose=0)[1]*100)

import matplotlib.pyplot as plt

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy graph')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train','test'])
plt.grid()
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('loss graph')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train','test'])
plt.grid()
plt.show()

'''
CIFAR-10
부류: 10개
부류 별: 32*32 RGB 영상6,000장
6,000*10 = 총 60,000장(train 5만 test 1만)

CIFAR-100
부류: 100개
대부류: 20개
부류 별: 32*32 RGB 영상 600장
600*100 = 총 60,000장(train 5만 test 1만)
'''

# Epoch 50/50
# 391/391 - 25s - loss: 0.3032 - accuracy: 0.9012 - val_loss: 1.8564 - val_accuracy: 0.5438 - 25s/epoch - 63ms/step
# 정확률= 54.37999963760376