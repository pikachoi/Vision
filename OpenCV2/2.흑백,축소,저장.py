import cv2 as cv
import sys

img = cv.imread('OpenCV2\data\img\c2.jpg')

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray_small = cv.resize(gray, dsize=(0,0), fx=0.5, fy=0.5)

cv.imwrite('iu_gray.jpg', gray)
cv.imwrite('iu_gray_small.jpg', gray_small)

cv.imshow('iu_gray.jpg', gray)
cv.imshow('iu_gray_small.jpg', gray_small)

cv.waitKey()
cv.destroyAllWindows()

"""
cv의 흑백 변환 공식(각 픽셀마다 적용)
I=round(0.299 * R + 0.587 * G + 0.114 * B)
"""
