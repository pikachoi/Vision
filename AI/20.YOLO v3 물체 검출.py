import numpy as np
import cv2 as cv
import sys

def construct_yolo_v3():  # YOLO 모델 구성 함수
    # class_names 에 txt파일 부류 이름을 저장
    f = open('Ai/data/coco_names.txt', 'r')
    class_names=[line.strip() for line in f.readlines()]
    
    model = cv.dnn.readNet('Ai/data/yolov3.weights', 'Ai/data/yolov3.cfg')  # 가중치, 구조 정보를 가져옴
    layer_names = model.getLayerNames()
    out_layers = [layer_names[i-1] for i in model.getUnconnectedOutLayers()]  # 그림의 YOLO 층 정보를 객체에 저장

    return model, out_layers, class_names

def yolo_detect(img, yolo_model, out_layers):  # YOLO 모델로 img 영상에서 물체 검출하는 함수
    height, width = img.shape[0], img.shape[1]  # 원본 영상의 사이즈 저장

    '''영상을 YOLO 에 입력할 수 이는 형태로 저장
    1. [0,255] 범위 화솟값을 [0,1]로
    2. 영상 사이즈를 488 * 488로
    3. BGR순서를 RGB로 변환
    '''
    test_img = cv.dnn.blobFromImage(img, 1.0/256, (448, 448), (0, 0, 0), swapRB=True)  

    yolo_model.setInput(test_img)  # 신경망에 입력
    output3 = yolo_model.forward(out_layers)  # 신경망의 전발 계산 수행(out_layers의 텐서를 객체에 저장)

    box, conf, id = [],[],[]  # 박스, 신뢰도, 부류 정보 저장할 리스트 생성
    for output in output3:  # 세개의 텐서를 각각 반복처리
        for vec85 in output:  # 85차원 벡터를 반복처리 (x, y, h, o, p1, p1, ..., p80) 앞의 네 요소는 박스, o는 신뢰도, 뒤 80개는 부류 확률
            scores = vec85[5:]  # 뒤 80개 요소 값 최고 확률에 해당하는 부류를 알아냄
            class_id = np.argmax(scores)  # 알아낸 부류 번호를 저장
            confidence = scores[class_id]  # 부류 확률을 저장
            if confidence > 0.5:  # 신뢰도가 50% 이상인 경우
                centerx, centery = int(vec85[0] * width), int(vec85[1] * height)  # [0, 1] 범위로 표현된 박스를 원래 영상 좌표계로 변환
                w, h = int(vec85[2] * width), int(vec85[3] * height)  # 박스 너비와 높이 
                x, y = int(centerx-w/2), int(centery-h/2)  # 박스 왼쪽 위의 위치
                box.append([x, y, x+w, y+h])
                conf.append(float(confidence))
                id.append(class_id)

    ind = cv.dnn.NMSBoxes(box, conf, 0.5, 0.4)  # 박스를 대상으로 비최대 억제 적용해 중복성 제거
    object = [box[i] + [conf[i]] + [id[i]] for i in range(len(box)) if i in ind]  # 비최대 억제에서 살아남은 박스의 위치, 신뢰도, 부류를 저장
    return object

model, out_layers, class_names = construct_yolo_v3()  # YOLO 모델 생성
colors = np.random.uniform(0, 255, size=(len(class_names), 3))  # 부류마다 색깔 지정

img = cv.imread('./Ai/data/soccer.jpg')
if img is None: sys.exit('파일이 없습니다.')

res = yolo_detect(img, model, out_layers)  # YOLO 모델로 물체 검출

for i in range(len(res)):
    x1, y1, x2, y2, confidence, id=res[i]
    text = str(class_names[id]) + '% 3f' %confidence
    cv.rectangle(img, (x1, y1), (x2, y2), colors[id], 2)
    cv.putText(img, text, (x1, y1 + 30), cv.FONT_HERSHEY_PLAIN, 1.5, colors[id], 2)
    
cv.imshow("Object detection by YOLO v.3", img)

cv.waitKey()
cv.destroyAllWindows()




