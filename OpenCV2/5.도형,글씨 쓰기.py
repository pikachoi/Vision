import cv2 as cv
import sys

img = cv.imread('OpenCV2\data\img\c2.jpg')

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

# x,y순 (영상, (왼쪽 위), (오른 아래), (색), 선 굵기)
cv.rectangle(img, (150,220), (220,255), (0,255,255), 3)

# (영상, '문자', (왼쪽 아래), 폰트 종류, 글자 크기, (색), 글자 두께)
cv.putText(img, 'I LOVE YOU', (130,280), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

cv.imshow('Draw', img)

cv.waitKey()
cv.destroyAllWindows()