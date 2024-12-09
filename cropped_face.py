import cv2
import os
import numpy as np
import json
    
def crop_faces(input_image_path, output_folder, output_json_dir, target_size, mode):
    face_model_path = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    face_config_path = "deploy.prototxt"

    face_net = cv2.dnn.readNetFromCaffe(face_config_path, face_model_path)

    image = cv2.imread(input_image_path)
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    face_positions = []

    padding = 20
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype(int)

            startX = int(max(0, startX - padding))
            startY = int(max(0, startY - padding))
            endX = int(min(w, endX + padding))
            endY = int(min(h, endY + padding))

            width = int(endX - startX)
            height = int(endY - startY)

            face_positions.append({
                "face_index": i + 1,
                "startX": startX,
                "startY": startY,
                "endX": endX,
                "endY": endY,
                "width": width,
                "height": height
            })

    row_threshold = 50 
    face_positions.sort(key=lambda pos: pos["startY"])
    rows = []
    current_row = [face_positions[0]]

    for pos in face_positions[1:]:
        if abs(pos["startY"] - current_row[-1]["startY"]) <= row_threshold:
            current_row.append(pos)
        else:
            rows.append(current_row)
            current_row = [pos]
    rows.append(current_row)

    for row in rows:
        row.sort(key=lambda pos: pos["startX"])

    sorted_faces = [pos for row in rows for pos in row]

    for idx, pos in enumerate(sorted_faces):
        pos["face_index"] = idx + 1
        startX, startY, endX, endY = pos["startX"], pos["startY"], pos["endX"], pos["endY"]
        cropped_face = image[startY:endY, startX:endX]
        resized_face = cv2.resize(cropped_face, target_size, interpolation=cv2.INTER_AREA)

        face_output_path = os.path.join(output_folder, f"cropped-face-{idx + 1}-{mode}.png")
        cv2.imwrite(face_output_path, resized_face)
        print(f"Saved resized cropped face {idx + 1} to: {face_output_path}")

    positions_json_path = os.path.join(output_json_dir, f"{mode}_face_positions.json")
    with open(positions_json_path, "w") as json_file:
        json.dump(sorted_faces, json_file, indent=4)
    print(f"Face positions saved to: {positions_json_path}")

    print("All faces have been detected, cropped, resized, and saved in the specified order.")