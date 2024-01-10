import cv2 as cv
import sys

img = cv.imread('OpenCV2\data\img\c2.jpg')

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

def draw(event,x,y,flags,param): # 콜백 함수
    global ix, iy # 전역 변수로 선언 하지 않으면 지역 으로 선언돼 바로 소멸

    if event == cv.EVENT_LBUTTONDOWN:  # 좌 클릭 시 초기 위치 저장
        ix, iy = x, y
    elif event == cv.EVENT_LBUTTONUP:  # 버튼 떼었을 때
        cv.rectangle(img, (ix, iy), (x, y), (0, 0, 255), 2)

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

