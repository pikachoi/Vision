import cv2 as cv
import sys

img = cv.imread('OpenCV2\data\img\c2.jpg')

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

print(img.shape)
print(type(img))
print(img[0,0,0], img[0,0,1], img[0,0,2]) # y,x,bgr

cv.imshow('Image Display', img)

cv.waitKey()
cv.destroyAllWindows()