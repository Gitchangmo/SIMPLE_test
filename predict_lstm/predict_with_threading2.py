import threading
import queue
import time
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from collections import deque
import mediapipe as mp
import os
import cv2

# 손 검출 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

iou_threshold = 0.3

# IoU 계산 함수
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

# LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        last_out = self.dropout(last_out)
        out = self.fc(last_out)
        return out

# 모델 로드
lstm_model = LSTMModel(input_size=20, hidden_size=64, num_layers=2, output_size=3)
lstm_model.load_state_dict(torch.load("lstm_model.pth"))
lstm_model.eval()

yolo_model = YOLO("simple.pt")

# 전역 변수
tracking = False         # 동작 감지 여부
action_result = "None"
action_buffer_count = 0  # 특징 벡터 생성을 위한 카운트
prev_frame_cx = 0

# 특징 추출 버퍼 (각 프레임에 대한 전처리 결과 저장)
feature_buffer = []

# 큐 생성 (프레임 전달용)
frame_queue = queue.Queue(maxsize=100)

# 비디오 캡처 (파일 또는 카메라)
cap = cv2.VideoCapture(".\\dataset_simple\\put_in\\carrot_put_in\\doyeon4.mp4")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 프레임 캡처 스레드: 지속적으로 프레임을 큐에 넣음
def frame_capture():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("영상을 찾을 수 없습니다.")
            break
        frame_queue.put(frame)
        time.sleep(0.005)  # 과부하 방지
    cap.release()

# per-frame 특징 추출 함수: 각 프레임에서 YOLO와 손 검출 후 특징 벡터 생성
def extract_feature(frame):
    global action_buffer_count
    yolo_results = yolo_model(frame)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hands = hands.process(frame_rgb)
    action_buffer_count += 1
    features = [action_buffer_count]
    # 객체 탐지 결과: 최대 3개 객체
    detected_objects = []
    for i, (box, conf, cls) in enumerate(zip(yolo_results[0].boxes.xyxy, yolo_results[0].boxes.conf, yolo_results[0].boxes.cls)):
        if i >= 3:
            break
        x1, y1, x2, y2 = map(int, box.tolist())
        class_id = int(cls)
        detected_objects.extend([class_id, x1, y1, x2, y2])
    while len(detected_objects) < 15:
        detected_objects.extend([-1, 0, 0, 0, 0])
    features.extend(detected_objects)
    # 손 검출 결과
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            x_min = min([lm.x for lm in hand_landmarks.landmark])
            y_min = min([lm.y for lm in hand_landmarks.landmark])
            x_max = max([lm.x for lm in hand_landmarks.landmark])
            y_max = max([lm.y for lm in hand_landmarks.landmark])
            h, w, _ = frame.shape
            x_min = int(x_min * w)
            y_min = int(y_min * h)
            x_max = int(x_max * w)
            y_max = int(y_max * h)
            features.extend([x_min, y_min, x_max, y_max])
            break  # 최대 1개 손만 사용
    else:
        features.extend([0, 0, 0, 0])
    return features

# LSTM 추론 함수: feature_buffer에 저장된 특징 벡터들을 LSTM 모델에 입력
def run_lstm_inference():
    global action_result
    if not feature_buffer:
        return
    lstm_input = feature_buffer[:]  # 복사
    lstm_input_tensor = torch.tensor(lstm_input, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = lstm_model(lstm_input_tensor)
        _, predicted_label = torch.max(output, 1)
        if predicted_label.item() == 0:
            action_result = "None"
        elif predicted_label.item() == 1:
            action_result = "Putting"
        elif predicted_label.item() == 2:
            action_result = "Taking"
    print("예측된 동작:", action_result)
    # 이후 feature_buffer를 비워줍니다.
    feature_buffer.clear()

# 스레드 시작
capture_thread = threading.Thread(target=frame_capture, daemon=True)
capture_thread.start()

# 메인 루프: 큐에서 프레임을 꺼내 처리하고, 추론 결과를 화면에 오버레이
while True:
    if not frame_queue.empty():
        frame = frame_queue.get()
        annotated_frame = frame.copy()  # 기본 프레임 복사
        # YOLO 결과 시각화 (객체 탐지)
        yolo_results = yolo_model(frame)
        annotated_frame = yolo_results[0].plot()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ROI 영역 설정 및 표시
        roi_x1, roi_y1, roi_x2, roi_y2 = int(frame_width * 0.65), 10, frame_width - 10, frame_height - 10
        cv2.rectangle(annotated_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 3)
        cv2.putText(annotated_frame, "ROI", (roi_x1 + 100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # 객체 탐지: 최대 3개 객체
        detected_boxes = []
        for i, (box, conf, cls) in enumerate(zip(yolo_results[0].boxes.xyxy, yolo_results[0].boxes.conf, yolo_results[0].boxes.cls)):
            if i >= 3:
                break
            x1, y1, x2, y2 = map(int, box.tolist())
            detected_boxes.append((x1, y1, x2, y2))
        
        # 손 검출
        hands_boxes = []
        results_hands = hands.process(frame_rgb)
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_frame, hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                x_min = min([lm.x for lm in hand_landmarks.landmark])
                y_min = min([lm.y for lm in hand_landmarks.landmark])
                x_max = max([lm.x for lm in hand_landmarks.landmark])
                y_max = max([lm.y for lm in hand_landmarks.landmark])
                h, w, _ = frame.shape
                x_min = int(x_min * w)
                y_min = int(y_min * h)
                x_max = int(x_max * w)
                y_max = int(y_max * h)
                hands_boxes.append((x_min, y_min, x_max, y_max))
                cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                break
        if not hands_boxes:
            hands_boxes.append((0, 0, 0, 0))
        
        # ROI 내 손 감지 여부
        Hands_in_roi = any(
            roi_x1 <= (hb[0] + hb[2]) / 2 <= roi_x2 and
            roi_y1 <= (hb[1] + hb[3]) / 2 <= roi_y2
            for hb in hands_boxes
        )
        
        # 객체와 손의 IoU 계산
        Food_On_Hands = False
        for detected_box in detected_boxes:
            for hand in hands_boxes:
                if calculate_iou(detected_box, hand) > iou_threshold:
                    Food_On_Hands = True
                    break
        
        if Food_On_Hands:
            cv2.putText(annotated_frame, "Food on hands", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(annotated_frame, "No Food on hands", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 동작 감지: ROI 내 손이 감지되면 tracking 시작
        if Hands_in_roi:
            cv2.putText(annotated_frame, "Hands In", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            # 중심 좌표 계산
            hand_box = hands_boxes[0]
            cur_cx = (hand_box[0] + hand_box[2]) / 2
            # 간단한 변화량 측정 (이전 프레임과 비교)
            hand_dx = cur_cx - prev_frame_cx
            prev_frame_cx = cur_cx
            if hand_dx > 0:  # 예시 조건: 오른쪽 이동 시 동작 시작
                if not tracking:
                    print("동작 감지 시작!")
                tracking = True
        
        # tracking 중이면, 현재 프레임에 대해 즉시 특징 추출 후 feature_buffer에 저장
        if tracking:
            feat = extract_feature(frame)
            feature_buffer.append(feat)
            cv2.putText(annotated_frame, "action_detecting", (50, frame_height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # 동작 종료 조건: ROI 내 손이 있지만, 이동 방향이 반대이거나 특정 조건 만족 시
        if Hands_in_roi and (hand_dx < 0) and (cur_cx < (frame_width * 0.65 + 10)):
            if tracking:
                print(f"동작 감지 종료. 저장된 프레임 수 : {len(feature_buffer)}")
                # 동작 종료 시, 별도의 스레드에서 LSTM 추론 실행
                inference_thread = threading.Thread(target=run_lstm_inference, daemon=True)
                inference_thread.start()
            tracking = False
        
        # 현재 추론 결과를 화면에 오버레이
        cv2.putText(annotated_frame, f"result: {action_result}", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv2.imshow("detection", annotated_frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cv2.destroyAllWindows()
