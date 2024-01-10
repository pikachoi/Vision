import cv2 as cv
import numpy as np

img = cv.imread('OpenCV2\data\img\c2.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
canny = cv.Canny(gray, 50, 120)

'''
cv.findContours
인수1 : 영상
인수2 : 구멍 있을 시 구멍 경계선 찾는 방식,(여기선 맨 바깥쪽 경계선만 찾도록)
인수3 : CHAIN_APPROX_NONE(모든 점 기록)
인수3 : CHAUN_APPROX_SIMPLE(직서네 대해서는 양쪽 끝점만 기록)
인수3 : CHAUN_APPROX_TC89_L1, CHAUN_APPROX_TC89_KCOS(Teh-Chin 알고리즘 으로 굴곡 심한 점만 기록)
'''
contour, hierarchy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

lcontour=[]
for i in range(len(contour)):
    if contour[i].shape[0]>100:  # 길이가 50보다 크면 (findContours 함수는 시작->끝->시작점 으로 추적해 실제로는 50)
        lcontour.append(contour[i])

# cv.drawContours(영상, 경계선, 음수는 모든 경계선 양수는 번호에 해당하는 경계선만, 색, 두께)
cv.drawContours(img, lcontour, -1, (0, 255, 0), 3)  # 영상에 경계선을 그림

cv.imshow('Original with contours', img)
cv.imshow('Canny', canny)

cv.waitKey()
cv.destroyAllWindows()

# 길이 50 이상인 경계선 표시 시 잡다한 경계선 감소효과