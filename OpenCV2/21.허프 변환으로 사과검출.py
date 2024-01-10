# 이전까진 에지 연결해 경계선 검출 -> 자잘하게 끊기는 문제점이 있음
# 허프 변환은 끊긴 에지를 모아 선분 또는 원 등을 검출

import cv2 as cv

img = cv.imread('OpenCV2/data/img/apple.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# HoughCircles 함수는 영상에서 원을 검출해 중심, 반지름을 저장한 리스트를 반환
apples = cv.HoughCircles(gray,
                         cv.HOUGH_GRADIENT,  # 에지 방향 정보를 추가로 사용
                         1,  # 누적 배열 크기(1은 입력 영상과 같은 크기)
                         70,  # 원 사이 최소거리(작을수록 많은 원)
                         param1=500,  # 케니에서 사용한 T<high>
                         param2=15,  # 비최대 억제를 적용할 때 쓰는 임계값
                         minRadius=10,  # 원 최소 반지름
                         maxRadius=30) # 원 최대 반지름

# apple 리스트가 가진 원의 중심과 반지름 정보로 영상에 그려 넣음
for i in apples[0]:
    cv.circle(img,
              (int(i[0]), int(i[1])),
              int(i[2]),
              (255, 0, 0), 2)

cv.imshow('Apple detection', img)

cv.waitKey()
cv.destroyAllWindows()