# SGD, Adam 비교
import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds

from tensorflow.keras.models import Sequential  # 좌에서 우로 한 줄기로 흐르는 계산은 'Sequential' 그외 'functional'
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam

(x_train, y_train), (x_test, y_test)=ds.mnist.load_data()
x_train = x_train.reshape(60000, 784)  # 2차원을 1차원으로 펼침
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype(np.float32)/255.0  # 원래 데이터형인 unint8를 실수 연산이 가능하도록 함
x_test = x_test.astype(np.float32)/255.0  # [0,255] 범위를 [0,1]로 변환
y_train = tf.keras.utils.to_categorical(y_train, 10)  # 정수로 표현된 train, test를 원핫인코딩
y_test = tf.keras.utils.to_categorical(y_test, 10)


# SGD
# 469/469 - 1s - loss: 0.0186 - accuracy: 0.8903 - val_loss: 0.0175 - val_accuracy: 0.8970 - 1s/epoch - 2ms/step
mlp_sgd = Sequential()
mlp_sgd.add(Dense(units=512, activation='tanh', input_shape=(784,)))  # 입력층에 784, 은닉층에 512노드 배치 'tach'는 은닉층 활성함수
mlp_sgd.add(Dense(units=10, activation='softmax'))  # 출력층에 10개 노드 배치

mlp_sgd.compile(loss='MSE', optimizer=SGD(learning_rate=0.01),metrics = ['accuracy'])  # MSE(평균제곱오차)
hist_sgd = mlp_sgd.fit(x_train, y_train, batch_size=128, epochs=50, validation_data=(x_test, y_test), verbose=2)  # verbose=2(세대마다 성능 측정)
print('SGD 정확률=', mlp_sgd.evaluate(x_test, y_test, verbose=0)[1]*100)  # res[1]은 정확률 출력


# Adam
# 469/469 - 3s - loss: 2.1795e-04 - accuracy: 0.9987 - val_loss: 0.0030 - val_accuracy: 0.9806 - 3s/epoch - 6ms/step
mlp_adam = Sequential()
mlp_adam.add(Dense(units=512, activation='tanh', input_shape=(784,)))
mlp_adam.add(Dense(units=10, activation='softmax'))

mlp_adam.compile(loss='MSE', optimizer=Adam(learning_rate=0.001),metrics = ['accuracy'])  
hist_adam = mlp_adam.fit(x_train, y_train, batch_size=128, epochs=50, validation_data=(x_test, y_test), verbose=2) 
print('Adam 정확률=', mlp_adam.evaluate(x_test, y_test, verbose=0)[1]*100)


import matplotlib.pyplot as plt

plt.plot(hist_sgd.history['accuracy'],'r--')  # r-- : 빨간색 점선으로 표시
plt.plot(hist_sgd.history['val_accuracy'],'r')  # r : 빨간색 실선으로 표시
plt.plot(hist_adam.history['accuracy'],'b--')
plt.plot(hist_adam.history['val_accuracy'],'b')
plt.title('Comparison of SGD and Adam optimizers')
plt.ylim((0.7,1.0))  # y축의 범위 설정
plt.xlabel('epochs')  # x축 제목
plt.ylabel('accuracy')  # y축 제목
plt.legend(['train_sgd', 'val_sgd', 'train_adam', 'val_adam'])  # 범례
plt.grid()  # 격자 넣기
plt.show()