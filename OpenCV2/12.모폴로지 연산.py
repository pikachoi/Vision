import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('OpenCV2\data\img\LT4.png', cv.IMREAD_UNCHANGED)
print(img.shape) # png는 4개의 채널을 가짐

# 3번 채널(마지막인 4번째 태널)에 이진화 적용
t, bin_img = cv.threshold(img[:, :, 3], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
plt.imshow(bin_img, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()

b = bin_img[bin_img.shape[0]//2:bin_img.shape[0], 0:bin_img.shape[0]//2+1]
plt.imshow(b, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()

# 구조 요소
se = np.uint8([[0, 0, 1, 0, 0],
               [0, 1, 1, 1, 0],
               [1, 1, 1, 1, 1],
               [0, 1, 1, 1, 0],
               [0, 0, 1, 0, 0]])

# 팽창
b_dilation = cv.dilate(b, se, iterations = 1)
plt.imshow(b_dilation, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()

# 침식
b_erosion = cv.erode(b, se, iterations = 1)
plt.imshow(b_erosion, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()

# 닫기
b_closing = cv.erode(cv.dilate(b, se, iterations = 1), se, iterations = 1)
plt.imshow(b_closing, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()


## 모폴로지 연산(영역 연산)
# 영상 변환 중 물체가 여러 영역으로 분리되거나 다른 영역으로 붙는 경우 완화
# 구조 요소를 이용해 영역 모양 조작
# 팽창 : 작음 홈 메우거나 끊어진 영역 하나로 연결(영역 확대)
# 침식 : 영역의 경계에 솟은 돌출 부분 깎음(영역 축소)
# 침식 결과에 팽창 적용 : 열림(opening)
# 팽창 결과에 침식 적용 : 닫힘(closing)