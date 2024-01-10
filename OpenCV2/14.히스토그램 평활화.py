import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('OpenCV2\data\img\c2.jpg')

# 명암 영상으로 변환 후 출력
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.calcHist([영상], [채널번호], 영역 지정 마스크, [히스토그램 칸 수], [세어 볼 명암값 범위])
plt.imshow(gray, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.show()

# 히스토그램을 구해 출력
h = cv.calcHist([gray], [0], None, [256], [0,256])
plt.plot(h, color='r', linewidth=1), plt.show()


# 히스토그램을 평활화하고 출력
equal = cv.equalizeHist(gray)
plt.imshow(equal, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.show()

# 평활화 한 히스토그램을 구해 출력
h = cv.calcHist([equal], [0], None, [256], [0,256])
plt.plot(h, color='r', linewidth=1), plt.show()

## 히스토그램 평활화(점 연산)
# 명암 대비 높여 물체 식별 용이하게 함