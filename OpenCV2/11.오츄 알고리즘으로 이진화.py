import cv2 as cv
import sys

img = cv.imread('OpenCV2\data\img\c2.jpg')

# threshold 는 이진화 알고리즘
# cv.threshold(영상, 명암값 범위 시작, 명암값 범위 끝, 오츄 알고리즘)

t, bin_img = cv.threshold(img[:, :, 2], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
print('오츄 알고리즘이 찾은 최적 임계값=', t)

cv.imshow('R channel', img[:, :, 2])  # R 채널 영상
cv.imshow('R channel binarization', bin_img)  # R 채널 이진화 영상

cv.waitKey()
cv.destroyAllWindows()

## 오츄 알고리즘(점 연산)
# 영상의 히스토그램의 계곡이 많아 임계값 결정이 어려울 시 이진화를 최적화 문제로 취급
# 모든 명암값에 대해 목적함수 J를 계산 후 J가 최소인 명암 값을 최적값으로 결정