import cv2 as cv

img = cv.imread('OpenCV2\data\img\c2.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 캐니는 T<high>를 T<low>의 2~3배로 권고
# 에지 추적은 T<high> 이상인 에지 화소에서 시작(에지일 가능성 높은 곳부터 시작하겠다는 의도)
canny1 = cv.Canny(gray, 50, 100)  # T<low>=50, T<high>=100 으로 설정
canny2 = cv.Canny(gray, 100, 200)  # T<low>=100, T<high>=200 으로 설정

cv.imshow('Original', gray)
cv.imshow('Canny1', canny1)
cv.imshow('Canny2', canny2)

cv.waitKey()
cv.detroyAllWindows()