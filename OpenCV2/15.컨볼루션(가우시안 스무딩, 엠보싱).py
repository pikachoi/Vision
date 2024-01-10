import cv2 as cv
import numpy as np

img = cv.imread('OpenCV2\data\img\c2.jpg')
img = cv.resize(img, dsize=(0, 0), fx=0.7, fy=0.7)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 스무딩 효과를 보기위해 글자 삽입
cv.putText(gray, 'soccer', (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv.imshow('Original',gray)

# cv.GaussianBlur(영상, (필터 크기), 표준편차:0.0으로 설정 시 필터크기 참고해 자동 추정)
# hastack : 이어 붙이는 함수
# 필터 크기 커질수록 흐려짐
smooth = np.hstack((cv.GaussianBlur(gray, (5, 5), 0.0),
                    cv.GaussianBlur(gray, (9, 9), 0.0),
                    cv.GaussianBlur(gray, (15, 15), 0.0)))

cv.imshow('Smooth',smooth)

femboss = np.array([[-1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0]])

gray16 = np.int16(gray) # 음수 표현을 위해 2바이트(16비트 적용)
emboss = np.uint8(np.clip(cv.filter2D(gray16, -1, femboss) + 128, 0, 255))
# clip 미 적용 시 구멍 생김(255를 넘어 오버플로우 발생한 화소 값이 엉뚱하게 바뀐 것이 원인)
emboss_bad = np.uint8(cv.filter2D(gray16, -1, femboss) + 128)
# uint16 미 적용 시 부작용 확인
emboss_worse = cv.filter2D(gray, -1, femboss)

cv.imshow('Emboss', emboss)
cv.imshow('Emboss_bad', emboss_bad)
cv.imshow('Emboss_worse', emboss_worse)

cv.waitKey()
cv.destoryAllWindos()

## 컨볼루션(영역 연산)