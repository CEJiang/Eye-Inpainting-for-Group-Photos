import cv2
import os


def crop_faces(input_image_path, output_folder, target_size):
    face_model_path = "res10_300x300_ssd_iter_140000_fp16.caffemodel"  # 人臉檢測模型
    face_config_path = "deploy.prototxt"  # 配置文件

    # 確保輸出資料夾存在
    os.makedirs(output_folder, exist_ok=True)

    # 加載人臉檢測模型
    face_net = cv2.dnn.readNetFromCaffe(face_config_path, face_model_path)

    # 讀取輸入圖片
    image = cv2.imread(input_image_path)
    (h, w) = image.shape[:2]

    # 構建 DNN 的 blob 並進行人臉檢測
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    padding = 20  # 增加邊界
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # 信心閾值
        if confidence > 0.5:
            # 計算邊界框
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")

            # 增加邊界框的邊距
            startX = max(0, startX - padding)
            startY = max(0, startY - padding)
            endX = min(w, endX + padding)
            endY = min(h, endY + padding)

            # 裁剪人臉
            cropped_face = image[startY:endY, startX:endX]

            # 調整裁剪後的圖片大小
            resized_face = cv2.resize(cropped_face, target_size, interpolation=cv2.INTER_AREA)

            # 儲存調整大小後的臉部圖片
            face_output_path = os.path.join(output_folder, f"cropped_face_{i + 1}.png")
            cv2.imwrite(face_output_path, resized_face)
            print(f"Saved resized cropped face {i + 1} to: {face_output_path}")

    print("All faces have been detected, cropped, and resized.")
