import cv2 as cv
import sys

img = cv.imread('OpenCV2\data\img\c2.jpg')

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

def draw(event,x,y,flags,param): # 콜백 함수
    if event == cv.EVENT_LBUTTONDOWN: # 마우스 좌클릭 시
        cv.rectangle(img, (x, y), (x+100, y+100), (0, 0, 255), 2)
    elif event == cv.EVENT_RBUTTONDOWN: # 마우스 우클릭 시
        cv.rectangle(img, (x, y), (x+50, y+50),(255, 0, 0), 2)

    cv.imshow('Drawing', img)

cv.namedWindow('Drawing') # 윈도우 생성
cv.imshow('Drawing', img)

cv.setMouseCallback('Drawing', draw) # 'Drawing' 윈도우에 콜백 함수 지정

# 마우스 이벤트가 언제 발생할 지 모르니 무한 반복
while(True):
    if cv.waitKey(1)==ord('q'):
        cv.destroyAllWindows()
        break

#(while문 걸지 않아도 동일 동작)

