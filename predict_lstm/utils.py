import math

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area
    return inter_area / union if union != 0 else 0

def is_near_hand(obj_box, hand_box, dist_thresh=80):
    ox1, oy1, ox2, oy2 = obj_box
    hx1, hy1, hx2, hy2 = hand_box
    ocx, ocy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
    hcx, hcy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
    dist = math.sqrt((ocx - hcx) ** 2 + (ocy - hcy) ** 2)
    return dist < dist_thresh

def is_food_on_hand(detected_boxes, hands_boxes, iou_thresh=0.3, dist_thresh=80):
    for obj_box in detected_boxes:
        for hand_box in hands_boxes:
            if calculate_iou(obj_box, hand_box) > iou_thresh or is_near_hand(obj_box, hand_box, dist_thresh):
                return True
    return False
