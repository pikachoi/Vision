import cv2 as cv
import numpy as np
import time

# 모든 화소에 접근해 흑백 변환
def my_cvtGray1(bgr_img):
    g = np.zeros([bgr_img.shape[0], bgr_img.shape[1]])
    for r in range(bgr_img.shape[0]):
        for c in range(bgr_img.shape[1]):
            g[r, c] = 0.114 * bgr_img[r, c, 0]\
                      + 0.587 * bgr_img[r, c, 1]\
                      + 0.299 * bgr_img[r, c, 2]

            return np.uint8(g)

# 파이썬 배열 연산으로 흑백 변환
def my_cvtGray2(bgr_img):
    g = np.zeros([bgr_img.shape[0], bgr_img.shape[1]])
    g = 0.114 * bgr_img[:, :, 0]\
        + 0.587 * bgr_img[:, :, 1]\
        + 0.299 * bgr_img[:, :, 2]

    return np.uint8(g)

img = cv.imread('OpenCV2\data\img\c2.jpg')

start = time.time()
my_cvtGray1(img)
print('My time1:', time.time()-start)

start = time.time()
my_cvtGray2(img)  # 2중 for문 보다 300배 빠름
print('My time2:', time.time()-start)

start=time.time()
cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print('OpenCV time:', time.time()-start)
'''

책에선 걸린 시간이 Gray1 > Gray2 > cv 로 cv가 가장 빠른걸로 나오나,
실제 로컬 테스트에선 반대 결과가 나옴..
'''