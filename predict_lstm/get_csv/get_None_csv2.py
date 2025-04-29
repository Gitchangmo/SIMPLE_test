import pandas as pd
import random
import os

# -----------------------------
# 객체 들고 왔다갔다 CSV 생성기
# -----------------------------
def generate_holding_object_sequence(save_dir, num_sequences=150, frames_per_seq=120, img_width=1280):
    os.makedirs(save_dir, exist_ok=True)

    for seq_idx in range(num_sequences):
        rows = []

        # 초기값 설정
        x1 = random.randint(50, 200)
        y1 = random.randint(300, 400)
        box_width = 200
        box_height = 200

        obj_box_offset = 10  # 객체가 손 위에 있을 때 약간 위쪽으로
        obj_box_size = int(box_width*0.8)

        moving_forward = True
        moving_backward = False
        frame_in_roi = 30
        roi_counter = 0

        for frame_idx in range(frames_per_seq):
            # 손 크기 먼저 결정
            box_w = box_width + random.randint(-20, 50)
            box_h = box_height + random.randint(-10, 10)

            if moving_forward:
                x_shift = random.randint(30, 45)
                y_shift = random.randint(-10, 10)

                x1 += x_shift
                y1 += y_shift

                if x1 > img_width - 120:
                    moving_forward = False

            elif not moving_forward and not moving_backward:
                roi_counter += 1
                if roi_counter > frame_in_roi:
                    moving_backward = True

            if moving_backward:
                x_shift = random.randint(30, 50)
                y_shift = random.randint(-10, 10)

                x1 -= x_shift
                y1 += y_shift

            x2 = x1 + box_w
            y2 = y1 + box_h

            # ROI 체류 중일 때만 손 좌표 0으로 처리
            if not moving_forward and not moving_backward:
                hand_x1 = hand_y1 = hand_x2 = hand_y2 = 0
                obj_x1 = obj_y1 = obj_x2 = obj_y2 = 0
            else:
                hand_x1 = x1
                hand_y1 = y1
                hand_x2 = x2
                hand_y2 = y2

                # 객체는 손 위에 고정 (살짝 작고 손 위쪽에 위치)
                obj_x1 = x1 + int((box_w - obj_box_size) / 2)
                obj_y1 = y1 - obj_box_offset - obj_box_size
                obj_x2 = obj_x1 + obj_box_size
                obj_y2 = obj_y1 + obj_box_size

            if x1 < 0 :
                row = {
                "frame_idx": frame_idx+1,
                "Object1_id": -1,
                "Object1_x1": 0,
                "Object1_y1": 0,
                "Object1_x2": 0,
                "Object1_y2": 0,
                "Object2_id": -1,
                "Object2_x1": 0,
                "Object2_y1": 0,
                "Object2_x2": 0,
                "Object2_y2": 0,
                "Object3_id": -1,
                "Object3_x1": 0,
                "Object3_y1": 0,
                "Object3_x2": 0,
                "Object3_y2": 0,
                "Hand_x1": 0,
                "Hand_y1": 0,
                "Hand_x2": 0,
                "Hand_y2": 0,
                "Hand2_x1": 0,
                "Hand2_y1": 0,
                "Hand2_x2": 0,
                "Hand2_y2": 0,
                "label": 0  # 무동작 (None)
                }
                rows.append(row)
                continue

            row = {
                "frame_idx": frame_idx + 1,
                "Object1_id": 3,
                "Object1_x1": obj_x1,
                "Object1_y1": obj_y1,
                "Object1_x2": obj_x2,
                "Object1_y2": obj_y2,
                "Object2_id": -1,
                "Object2_x1": 0,
                "Object2_y1": 0,
                "Object2_x2": 0,
                "Object2_y2": 0,
                "Object3_id": -1,
                "Object3_x1": 0,
                "Object3_y1": 0,
                "Object3_x2": 0,
                "Object3_y2": 0,
                "Hand_x1": 0,
                "Hand_y1": 0,
                "Hand_x2": 0,
                "Hand_y2": 0,
                "Hand2_x1": hand_x1,
                "Hand2_y1": hand_y1,
                "Hand2_x2": hand_x2,
                "Hand2_y2": hand_y2,
                "label": 0  # 무동작 (None)
            }

            rows.append(row)

        # 저장
        df = pd.DataFrame(rows)
        save_path = os.path.join(save_dir, f"None_obj_right_{seq_idx+1:03d}.csv")
        if os.path.exists(save_path):
            os.remove(save_path)
        df.to_csv(save_path, index=False)
        print(f"✅ 저장 완료: {save_path}")

# -----------------------------
# 사용법 예시
# -----------------------------
generate_holding_object_sequence(save_dir="./lstm_csv_files/None_data/onfood/", num_sequences=15)
