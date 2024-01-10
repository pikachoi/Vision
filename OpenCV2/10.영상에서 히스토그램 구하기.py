import cv2 as cv
import matplotlib.pyplot as plt

# 이진화 시 임계값 T보다 크면 1 작으면 0
# 보통 히스토그램의 계곡 근처를 임계값으로 결정해 쏠림 현상 완화

img = cv.imread('OpenCV2\data\img\c2.jpg')
# cv.calcHist([영상], [채널번호], 영역 지정 마스크, [히스토그램 칸 수], [세어 볼 명암값 범위])
h = cv.calcHist([img], [2], None, [256], [0,256])  # 2번 채널인 R 채널에서 히스토그램 구함
plt.plot(h, color = 'r', linewidth=1)
plt.show()