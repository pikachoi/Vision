import cv2 as cv
import sys

img = cv.imread('OpenCV2\data\img\c2.jpg')

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

cv.imshow('original_RGB', img)
# 왼쪽 위만 잘라냄 = 전체 이미지의 4/1
cv.imshow('Upper left half', img2 = img[0:img.shape[0]//2, 0:img.shape[1]//2, :])
# 오른 부분 잘라냄 = 전체 이미지의 4/1
cv.imshow('Center half', img[img.shape[0]//4:3*img.shape[0]//4,
                         img.shape[1]//4:3*img.shape[1]//4,:])

# 해당 컬러 밝게 표현(채
cv.imshow('R channel', img[:, :, 2])
cv.imshow('G channel', img[:, :, 1])
cv.imshow('B channel', img[:, :, 0])

cv.waitKey()
cv.destroyAllWindows()