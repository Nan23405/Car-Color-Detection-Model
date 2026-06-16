import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

model = None

def load_model():
    global model
    if model is None:
        model = YOLO("yolov8n.pt")
    return model


def is_blue_car(image_bgr, box):
    """
    Determines if a car is blue using HSV color analysis.
    Returns True if the dominant color of the car is blue.
    """
    x1, y1, x2, y2 = map(int, box)
    h, w = image_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Blue color range in HSV
    lower_blue1 = np.array([100, 50, 50])
    upper_blue1 = np.array([130, 255, 255])
    lower_blue2 = np.array([85, 40, 40])
    upper_blue2 = np.array([100, 255, 255])

    mask1 = cv2.inRange(hsv, lower_blue1, upper_blue1)
    mask2 = cv2.inRange(hsv, lower_blue2, upper_blue2)
    blue_mask = cv2.bitwise_or(mask1, mask2)

    total_pixels = crop.shape[0] * crop.shape[1]
    blue_pixels = cv2.countNonZero(blue_mask)

    blue_ratio = blue_pixels / total_pixels if total_pixels > 0 else 0
    return blue_ratio > 0.12  # At least 12% blue pixels


def get_dominant_color_name(image_bgr, box):
    """Returns a human-readable dominant color label for a car bounding box."""
    x1, y1, x2, y2 = map(int, box)
    h, w = image_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    color_ranges = {
        "blue":   [(np.array([85, 40, 40]),   np.array([130, 255, 255]))],
        "red":    [(np.array([0, 70, 50]),     np.array([10, 255, 255])),
                   (np.array([170, 70, 50]),   np.array([180, 255, 255]))],
        "green":  [(np.array([35, 40, 40]),    np.array([85, 255, 255]))],
        "yellow": [(np.array([20, 100, 100]),  np.array([35, 255, 255]))],
        "white":  [(np.array([0, 0, 180]),     np.array([180, 30, 255]))],
        "black":  [(np.array([0, 0, 0]),       np.array([180, 255, 50]))],
        "silver": [(np.array([0, 0, 130]),     np.array([180, 30, 180]))],
    }

    total = crop.shape[0] * crop.shape[1]
    best_color, best_ratio = "other", 0.0

    for color_name, ranges in color_ranges.items():
        count = 0
        for lower, upper in ranges:
            mask = cv2.inRange(hsv, lower, upper)
            count += cv2.countNonZero(mask)
        ratio = count / total
        if ratio > best_ratio:
            best_ratio = ratio
            best_color = color_name

    return best_color


def detect(image_input):
    """
    Run detection on an image.
    image_input: file path (str) or PIL Image
    Returns: (annotated PIL Image, car_count, person_count, detections list)
    """
    mdl = load_model()

    if isinstance(image_input, str):
        img_bgr = cv2.imread(image_input)
    else:
        img_bgr = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)

    if img_bgr is None:
        raise ValueError("Could not read image.")

    results = mdl(img_bgr, verbose=False)[0]

    car_count = 0
    person_count = 0
    detections = []

    COCO_CAR_CLASSES = {2: "car", 5: "bus", 7: "truck"}
    COCO_PERSON_CLASS = 0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, xyxy)

        if cls_id in COCO_CAR_CLASSES:
            car_count += 1
            blue = is_blue_car(img_bgr, xyxy)
            color_name = get_dominant_color_name(img_bgr, xyxy)

            # RED box for blue cars, BLUE box for others
            rect_color = (0, 0, 255) if blue else (255, 0, 0)  # BGR
            label_bg = (0, 0, 180) if blue else (180, 0, 0)
            label = f"BLUE car" if blue else f"{color_name} car"

            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), rect_color, 3)

            # Label background
            txt_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(img_bgr, (x1, y1 - txt_size[1] - 8), (x1 + txt_size[0] + 6, y1), label_bg, -1)
            cv2.putText(img_bgr, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            detections.append({"type": "car", "color": color_name, "is_blue": blue, "conf": round(conf, 2), "box": [x1, y1, x2, y2]})

        elif cls_id == COCO_PERSON_CLASS:
            person_count += 1
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"person"
            txt_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
            cv2.rectangle(img_bgr, (x1, y1 - txt_size[1] - 8), (x1 + txt_size[0] + 6, y1), (0, 130, 0), -1)
            cv2.putText(img_bgr, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            detections.append({"type": "person", "conf": round(conf, 2), "box": [x1, y1, x2, y2]})

    # Overlay summary
    summary = f"Cars: {car_count}  |  People: {person_count}"
    cv2.rectangle(img_bgr, (0, 0), (360, 40), (20, 20, 20), -1)
    cv2.putText(img_bgr, summary, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_out = Image.fromarray(img_rgb)

    return pil_out, car_count, person_count, detections
