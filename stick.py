import cv2
import json
import os
import numpy as np

def match_color_statistics(src, target):
    """
    將 src 的 LAB 平均值與標準差調整為與 target 相同。
    src 與 target 均為 BGR 圖，且尺寸可相同或不同。
    """
    # 轉為 LAB 色彩空間
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)

    # 計算各通道均值與標準差
    src_mean, src_std = cv2.meanStdDev(src_lab)
    tgt_mean, tgt_std = cv2.meanStdDev(target_lab)

    # 將三個通道的值轉為 numpy array 好進行運算
    src_mean = src_mean.reshape((3,))
    src_std = src_std.reshape((3,))
    tgt_mean = tgt_mean.reshape((3,))
    tgt_std = tgt_std.reshape((3,))

    # 避免除以 0
    src_std = np.where(src_std == 0, 1, src_std)
    tgt_std = np.where(tgt_std == 0, 1, tgt_std)

    # 根據目標的平均與標準差來調整 src_lab
    # 新值 = ((原值 - src_mean) * (tgt_std / src_std)) + tgt_mean
    # 對 LAB 的每個通道進行轉換
    adjusted_lab = ((src_lab - src_mean) * (tgt_std / src_std)) + tgt_mean

    # 範圍裁切，LAB 各通道值需在 0~255 之間
    adjusted_lab = np.clip(adjusted_lab, 0, 255).astype(np.uint8)

    # 轉回 BGR
    adjusted_bgr = cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)
    return adjusted_bgr

def Stick(json_path, cropped_faces_folder, original_image_path, output_image_path):
    # 讀取 JSON 資料
    with open(json_path, "r") as f:
        face_positions = json.load(f)

    # 載入原始圖像
    original_image = cv2.imread(original_image_path)
    if original_image is None:
        raise ValueError("無法載入原始圖像，請檢查路徑")
    img_height, img_width = original_image.shape[:2]

    # 初始化畫布（與原始圖像相同大小）
    canvas = original_image.copy()

    for face in face_positions:
        face_index = face["face_index"]
        startX, startY = face["startX"], face["startY"]
        endX, endY = face["endX"], face["endY"]

        # 座標檢查與修正
        if startX < 0 or startY < 0 or endX > img_width or endY > img_height:
            print(f"處理臉部 {face_index} 時發生座標錯誤：")
            continue

        startX = max(0, min(startX, img_width - 1))
        startY = max(0, min(startY, img_height - 1))
        endX = max(0, min(endX, img_width))
        endY = max(0, min(endY, img_height))

        if endX <= startX or endY <= startY:
            print(f"臉部 {face_index} 的座標無效，跳過")
            continue

        # 載入裁切臉部圖像
        face_path = os.path.join(cropped_faces_folder, f"cropped-face-{face_index}.png")
        cropped_face = cv2.imread(face_path)
        if cropped_face is None:
            print(f"無法載入裁剪臉部圖片 {face_index}，路徑：{face_path}")
            continue

        # 尺寸調整
        face_w = endX - startX
        face_h = endY - startY
        resized_face = cv2.resize(cropped_face, (face_w, face_h), interpolation=cv2.INTER_AREA)
        if resized_face.shape[0] != face_h or resized_face.shape[1] != face_w:
            print(f"臉部 {face_index} 的裁剪大小不匹配，跳過")
            continue

        # 取出目標區域背景 (從 canvas 或 original_image 皆可)
        target_region = canvas[startY:endY, startX:endX]

        # 色彩、亮度匹配：將 resized_face 的色彩分佈匹配到 target_region
        resized_face = match_color_statistics(resized_face, target_region)

        # 建立漸層式的 mask
        face_mask = np.zeros((face_h, face_w), dtype=np.uint8)
        border = 10
        face_mask[border:face_h-border, border:face_w-border] = 255
        face_mask = cv2.GaussianBlur(face_mask, (15, 15), 0)

        # 計算中心點
        center = ((startX + endX) // 2, (startY + endY) // 2)

        # 使用 MIXED_CLONE 嘗試更佳融合
        try:
            canvas = cv2.seamlessClone(
                resized_face,  # src
                canvas,         # dst
                face_mask,      # mask
                center,         # center
                cv2.MIXED_CLONE
            )
        except cv2.error as e:
            print(f"處理臉部 {face_index} 時發生錯誤：{e}")
            continue

    # 儲存最終合成的圖像
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    cv2.imwrite(output_image_path, canvas)
    print(f"合成圖像已儲存到：{output_image_path}")

    cv2.imshow("Reconstructed Image", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()