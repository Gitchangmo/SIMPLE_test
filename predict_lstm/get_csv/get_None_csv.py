import pandas as pd
import random
import os

# -----------------------------
# 빈손 왔다갔다 CSV 생성기
# -----------------------------
def generate_empty_hand_sequence(save_dir, num_sequences=150, frames_per_seq=120, img_width=1280):
    os.makedirs(save_dir, exist_ok=True)

    for seq_idx in range(num_sequences):
        rows = []

        # 초기값 설정
        x1 = random.randint(50, 200)
        y1 = random.randint(300, 400)
        box_width = 200
        box_height = 200

        moving_forward = True
        moving_backward = False
        frame_in_roi = 30
        roi_counter=0

        for frame_idx in range(frames_per_seq): #120번 반복
            # 손 크기 살짝 랜덤 변화
            box_w = box_width + random.randint(-20, 50)
            box_h = box_height + random.randint(-10, 10)

            # 이동 or 체류 결정
            if moving_forward == True:
                x_shift = random.randint(30, 45)
                y_shift = random.randint(-10, 10)

                x1 += x_shift
                y1 += y_shift

                if x1 > img_width - 120:  # ROI 내부 진입 완료
                    moving_forward = False

            elif not moving_forward and not moving_backward:
                roi_counter+=1
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
                if roi_counter > frame_in_roi:
                    moving_backward = True
                continue

            if moving_backward:  # 나올 때
                x_shift = random.randint(30, 50)
                y_shift = random.randint(-10, 10)

                x1 -= x_shift
                y1 += y_shift

            x2 = x1 + box_w
            y2 = y1 + box_h

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
            # 한 프레임 데이터 만들기
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
                "Hand2_x1": x1,
                "Hand2_y1": y1,
                "Hand2_x2": x2,
                "Hand2_y2": y2,
                "label": 0  # 무동작 (None)
            }

            rows.append(row)

        # 저장
        df = pd.DataFrame(rows)
        save_path = os.path.join(save_dir, f"None_Right_{seq_idx+1:03d}.csv")
        df.to_csv(save_path, index=False)
        print(f"✅ 저장 완료: {save_path}")

# -----------------------------
# 사용법 예시
# -----------------------------
generate_empty_hand_sequence(save_dir="./lstm_csv_files/None_data/nofood/", num_sequences=40)
