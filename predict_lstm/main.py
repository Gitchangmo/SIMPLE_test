import threading
import queue
import time
import torch
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import cv2

from lstm_module import lstm_model, run_lstm_inference
from utils import calculate_iou

# 손 검출 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

iou_threshold = 0.3

action_result = "None"
action_buffer_count = 0
prev_frame_cx = 0
tracking = False
feature_buffer = []

frame_queue = queue.Queue(maxsize=100)

cap = cv2.VideoCapture("./predict_lstm/doyeon4.mp4")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

yolo_model = YOLO("./predict_lstm/simple.pt")

def frame_capture():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("영상을 찾을 수 없습니다.")
            break
        frame_queue.put(frame)
        time.sleep(0.005)
    cap.release()

def extract_feature(frame):
    global action_buffer_count
    yolo_results = yolo_model(frame)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hands = hands.process(frame_rgb)
    action_buffer_count += 1
    features = [action_buffer_count]

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
            break
    else:
        features.extend([0, 0, 0, 0])
    return features

capture_thread = threading.Thread(target=frame_capture, daemon=True)
capture_thread.start()

while True:
    if not frame_queue.empty():
        frame = frame_queue.get()
        annotated_frame = frame.copy()
        yolo_results = yolo_model(frame)
        annotated_frame = yolo_results[0].plot()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        roi_x1, roi_y1, roi_x2, roi_y2 = int(frame_width * 0.65), 10, frame_width - 10, frame_height - 10
        cv2.rectangle(annotated_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 3)
        cv2.putText(annotated_frame, "ROI", (roi_x1 + 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        detected_boxes = []
        for i, (box, conf, cls) in enumerate(zip(yolo_results[0].boxes.xyxy, yolo_results[0].boxes.conf, yolo_results[0].boxes.cls)):
            if i >= 3:
                break
            x1, y1, x2, y2 = map(int, box.tolist())
            detected_boxes.append((x1, y1, x2, y2))

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

        Hands_in_roi = any(
            roi_x1 <= (hb[0] + hb[2]) / 2 <= roi_x2 and
            roi_y1 <= (hb[1] + hb[3]) / 2 <= roi_y2
            for hb in hands_boxes
        )

        Food_On_Hands = False
        for detected_box in detected_boxes:
            for hand in hands_boxes:
                if calculate_iou(detected_box, hand) > iou_threshold:
                    Food_On_Hands = True
                    break

        if Food_On_Hands:
            cv2.putText(annotated_frame, "Food on hands", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(annotated_frame, "No Food on hands", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if Hands_in_roi:
            cv2.putText(annotated_frame, "Hands In", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            hand_box = hands_boxes[0]
            cur_cx = (hand_box[0] + hand_box[2]) / 2
            hand_dx = cur_cx - prev_frame_cx
            prev_frame_cx = cur_cx
            if hand_dx > 0:
                if not tracking:
                    print("동작 감지 시작!")
                tracking = True

        if tracking:
            feat = extract_feature(frame)
            feature_buffer.append(feat)
            cv2.putText(annotated_frame, "action_detecting", (50, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        if Hands_in_roi and (hand_dx < 0) and (cur_cx < (frame_width * 0.65 + 10)):
            if tracking:
                print(f"동작 감지 종료. 저장된 프레임 수 : {len(feature_buffer)}")
                inference_thread = threading.Thread(target=run_lstm_inference, args=(feature_buffer,), daemon=True)
                inference_thread.start()
            tracking = False

        cv2.putText(annotated_frame, f"result: {action_result}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("detection", annotated_frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cv2.destroyAllWindows()