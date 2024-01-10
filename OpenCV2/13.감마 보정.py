import cv2 as cv
import numpy as np

img = cv.imread('OpenCV2\data\img\c2.jpg')
img = cv.resize(img, dsize=(0, 0), fx=0.25, fy=0.25) # 4/1로 축소

def gamma(f, gamma=1.0):
    f1 = f/255.0  # L=256 라고 가정
    return np.uint8(255*(f1**gamma))

gc = np.hstack((gamma(img, 0.5), gamma(img, 0.75),
                gamma(img, 1.0), gamma(img, 2.0), gamma(img, 3.0)))
cv.imshow('gamma', gc)

cv.waitKey()
cv.destroyAllWindows()

## 감마 보정
# 인간의 밝기 변화에 대한 비 선형적 시각 반응과 같은 반응을 얻기 위해 필요