import pandas as pd
import random
import os

# -----------------------------
# Put 동작 CSV 생성기
# -----------------------------
def generate_empty_hand_sequence(save_dir, num_sequences=150, frames_per_seq=120, img_width=1280):
    os.makedirs(save_dir, exist_ok=True)

    for seq_idx in range(num_sequences):
        rows = []
        random.seed(seq_idx)

        obj_class = random.choice([0,1,2,3])

        # 초기값 설정
        x1 = random.randint(200, 350)
        y1 = random.randint(200, 450)
        box_width = 200
        box_height = 200
        roi = 832
        action_value = 0

        # 진입 후 멈출때 필요 변수------------
        wait_count = random.randint(25,35)    
        in_roi = False  
        #-----------------------------------

        obj_box_offset = 10  # 객체가 손 위에 있을 때 약간 위쪽으로
        obj_box_size = int(box_width*0.8)

        moving_forward = True
        moving_backward = False
        frame_in_roi = random.randint(25,35)
        roi_counter=0

        for frame_idx in range(frames_per_seq): #120번 반복
            # 손 크기 살짝 랜덤 변화
            box_w = box_width + random.randint(-20, 50)
            box_h = box_height + random.randint(-10, 10)

            if x1>roi:
                in_roi = True
                wait_count -=1
                action_value = 2 # Take 동작
            else :
                in_roi = False
                action_value = 0 # None 동작

            if wait_count >= 0 and in_roi:
                action_value = 2
                x_shift = random.randint(-10, 10)
                y_shift = random.randint(-10, 10)

                x1 += x_shift
                y1 += y_shift
                x2 = x1 + box_w
                y2 = y1 + box_h

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
                "Hand_x1": x1,
                "Hand_y1": y1,
                "Hand_x2": x2,
                "Hand_y2": y2,
                "Hand2_x1": 0,
                "Hand2_y1": 0,
                "Hand2_x2": 0,
                "Hand2_y2": 0,
                "label": action_value  # Take 동작
                }
                rows.append(row)
                
                continue

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
                "label": action_value  # 뺴는 동작 (Take)
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
                "label": action_value  # 무동작 (None)
                }
                rows.append(row)
                continue

            if moving_backward:
                row = {
                "frame_idx": frame_idx+1,
                "Object1_id": obj_class,
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
                "Hand_x1": x1,
                "Hand_y1": y1,
                "Hand_x2": x2,
                "Hand_y2": y2,
                "Hand2_x1": 0,
                "Hand2_y1": 0,
                "Hand2_x2": 0,
                "Hand2_y2": 0,
                "label": action_value  # 무동작 (None)
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
                "Hand_x1": x1,
                "Hand_y1": y1,
                "Hand_x2": x2,
                "Hand_y2": y2,
                "Hand2_x1": 0,
                "Hand2_y1": 0,
                "Hand2_x2": 0,
                "Hand2_y2": 0,
                "label": action_value  # 무동작 (None)
            }

            rows.append(row)

        # 저장
        df = pd.DataFrame(rows)
        save_path = os.path.join(save_dir, f"Take_Left_Wait_{seq_idx+1:03d}.csv")
        df.to_csv(save_path, index=False)
        print(f"✅ 저장 완료: {save_path}")

# -----------------------------
# 사용법 예시
# -----------------------------
generate_empty_hand_sequence(save_dir="./lstm_csv_files/Take_Data/", num_sequences=25)
