from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def segment_objects(frame):
    results = model(frame)
    boxes = results[0].boxes

    mask = frame.copy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        mask[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
    return mask, boxes

    