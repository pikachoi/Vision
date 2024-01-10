import cv2 as cv
import sys

img = cv.imread('OpenCV2\data\img\c2.jpg')

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

BrushSiz = 5 # 붓 크기(원일 때 반지름)
LColor, RColor = (255, 0, 0), (0, 0, 255) # 좌클릭 색, 우클릭 색

def painting(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(img, (x, y), BrushSiz, LColor, -1)  # 좌클릭(-1은 내부 채움)
    elif event==cv.EVENT_RBUTTONDOWN:
        cv.circle(img, (x, y), BrushSiz, RColor, -1)  # 우클릭
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:  # 좌클릭 이동
        cv.circle(img, (x, y), BrushSiz, LColor, -1)
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_RBUTTON:  # 우클릭 이동
        cv.circle(img, (x, y), BrushSiz, RColor, -1)

    cv.imshow('Painting', img)

cv.namedWindow('Painting') # 윈도우 생성
cv.imshow('Painting', img)

cv.setMouseCallback('Painting', painting)

while(True):
    if cv.waitKey(1)==ord('q'):
        cv.destroyAllWindows()
        break