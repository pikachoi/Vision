import cv2 as cv

img = cv.imread('OpenCV2\data\img\c2.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 소벨 연산자 적용
# Sobel(gray, 32비트 실수 맵에 저장, x,y방향 연산자 사용, x,y방향 연산자 사용, 3*3크기 사용)
grad_x = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)  # 32비트 실수 맵에 저장
grad_y = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)

# 절대 값을 취해 양수 영상으로 변환
# convertScaleAbs 함ㅅ는 부호없는 8비트형인 CV_8U맵을 만듦(0보다 작으면 0, 255넘으면 255)
sobel_x = cv.convertScaleAbs(grad_x)
sobel_y = cv.convertScaleAbs(grad_y)

# cv.addWeighted(img1 * a + img2 * b + c)
# img1,2 는 같은 데이터 형 이어야 함
edge_strength = cv.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

cv.imshow('Original', gray)
cv.imshow('sobelx', sobel_x) # 수평 방향 에지가 선명
cv.imshow('sobely', sobel_y) # 수직 방향 에지가 선명
cv.imshow('edge strength', edge_strength)

cv.waitKey()
cv.destroyAllWindows()