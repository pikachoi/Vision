import tensorflow as tf
import tensorflow.keras.datasets as ds
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = ds.mnist.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

plt.figure(figsize=(24, 3))  # 그림크기
plt.suptitle('MNIST', fontsize=30)  # 제목지정
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(x_train[i], cmap='gray')  # 명암으로 출력
    plt.xticks([]); plt.yticks([])  # 이미지 x, y 축에 눈끔을 달지 않게함
    plt.title(str(y_train[i]), fontsize=30)

plt.show()

(x_train, y_train), (x_test, y_test) = ds.cifar10.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# 0_9로 표현된 부류 정보를 이름으로 변환
class_names = ['aieplane', 'car', 'bird', 'cat', 'deep', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(24, 3))
plt.suptitle('CIFAR-10', fontsize=30)
for i in range(10):
    plt.subplot(1, 10, i+1)  # 1줄로 10개 영상을 배치하고 i+1번째를 채움
    plt.imshow(x_train[i])
    plt.xticks([]); plt.yticks([])
    plt.title(class_names[y_train[i,0]], fontsize=30)  # 샘플 위에 부류 정보 표시

plt.show()