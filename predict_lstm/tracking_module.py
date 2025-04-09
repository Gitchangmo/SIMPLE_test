from collections import defaultdict
from utils import calculate_iou, is_near_hand

def is_in_roi(bbox, roi):
    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h / 2
    x1, y1, x2, y2 = roi
    return x1 <= cx <= x2 and y1 <= cy <= y2

def track_and_log(frame, yolo_results, hand_boxes, roi_coords, frame_idx, tracker, track_log):
    boxes = yolo_results[0].boxes.xyxy.tolist()
    confs = yolo_results[0].boxes.conf.tolist()
    classes = yolo_results[0].boxes.cls.tolist()

    detections = [([int(x1), int(y1), int(x2)-int(x1), int(y2)-int(y1)], c, int(cl))
                  for (x1, y1, x2, y2), c, cl in zip(boxes, confs, classes)]

    tracks = tracker.update_tracks(detections, frame=frame)
    for t in tracks:
        bbox = t['bbox']
        tid = t['track_id']
        cls = t['class_id']
        in_roi_flag = is_in_roi(bbox, roi_coords)
        obj_box = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        max_iou = max([calculate_iou(obj_box, hb) for hb in hand_boxes], default=0)
        near = any([is_near_hand(obj_box, hb) for hb in hand_boxes])
        log_entry = {
            "frame": frame_idx,
            "id": tid,
            "class": cls,
            "bbox": bbox,
            "in_roi": in_roi_flag,
            "iou_with_hand": max_iou if max_iou > 0.3 else (0.4 if near else 0.0)
        }
        track_log.append(log_entry)
    return track_log

def analyze_object_events(track_log, iou_thresh=0.3):
    object_tracks = defaultdict(list)
    results = {}
    for entry in track_log:
        object_tracks[entry['id']].append(entry)
    for obj_id, logs in object_tracks.items():
        logs.sort(key=lambda x: x['frame'])
        rois = [log['in_roi'] for log in logs]
        ious = [log['iou_with_hand'] for log in logs]
        entered = any(rois[i] and not rois[i-1] for i in range(1, len(rois)))
        exited = any(not rois[i] and rois[i-1] for i in range(1, len(rois)))
        max_iou = max(ious) if ious else 0
        if max_iou > iou_thresh:
            results[obj_id] = "넣기" if entered else ("빼기" if exited else "none")
        else:
            results[obj_id] = "none"
    return results
