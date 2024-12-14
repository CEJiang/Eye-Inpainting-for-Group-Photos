import os
import cv2
import dlib
import json
import numpy as np
from scipy.spatial import distance

def calculate_confidence(ear, open_threshold=0.15, close_threshold=0.10):
    if ear >= open_threshold:
        return 1.0, 0.0
    elif ear <= close_threshold:
        return 0.0, 1.0
    else:
        opened_conf = (ear - close_threshold) / (open_threshold - close_threshold)
        closed_conf = 1.0 - opened_conf
        return opened_conf, closed_conf

def calculate_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def Align(image_dir, output_json, output_image_dir):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./config/shape_predictor_68_face_landmarks.dat")

    os.makedirs(output_image_dir, exist_ok=True)

    result = {}

    for filename in os.listdir(image_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        person_id = filename.split('-')[2].split('.')[0]

        if person_id not in result:
            result[person_id] = []

        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"無法讀取影像: {image_path}")
            continue

        h, w = image.shape[:2]
        if max(h, w) > 256:
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = detector(gray)
        if len(faces) == 0:
            print(f"沒有偵測到人臉: {image_path}")
            continue

        for face in faces:
            landmarks = predictor(gray, face)
            left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)

            left_opened_conf, left_closed_conf = calculate_confidence(left_ear)
            right_opened_conf, right_closed_conf = calculate_confidence(right_ear)

            avg_opened_conf = (left_opened_conf + right_opened_conf) / 2.0
            avg_closed_conf = (left_closed_conf + right_closed_conf) / 2.0

            result[person_id].append({
                "filename": filename,
                "eye_left": {"x": min(pt[0] for pt in left_eye), "y": min(pt[1] for pt in left_eye)},
                "box_left": {"w": max(pt[0] for pt in left_eye) - min(pt[0] for pt in left_eye),
                             "h": max(pt[1] for pt in left_eye) - min(pt[1] for pt in left_eye)},
                "eye_right": {"x": min(pt[0] for pt in right_eye), "y": min(pt[1] for pt in right_eye)},
                "box_right": {"w": max(pt[0] for pt in right_eye) - min(pt[0] for pt in right_eye),
                              "h": max(pt[1] for pt in right_eye) - min(pt[1] for pt in right_eye)},
                "opened": avg_opened_conf,
                "closed": avg_closed_conf
            })

            output_path = os.path.join(output_image_dir, filename)
            cv2.imwrite(output_path, image)
            print(f"已保存圖片: {output_path}")

    result = {person_id: photos for person_id, photos in result.items() if photos}

    with open(output_json, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"結果已儲存到 {output_json}")