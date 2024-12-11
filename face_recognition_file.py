import face_recognition
import os
import shutil

def compare_and_rename_faces(identity_folder, reference_folder, output_folder, tolerance=0.6):
    """
    比對 identity 資料夾與 reference 資料夾的圖片，並在匹配成功後將匹配結果複製到輸出資料夾，
    並在輸出資料夾中重命名 reference 資料夾的圖片。
    Args:
        identity_folder (str): Identity 資料夾路徑。
        reference_folder (str): Reference 資料夾路徑。
        output_folder (str): 輸出資料夾路徑。
        tolerance (float): 人臉比對的容忍度，越小越嚴格（默認為 0.6）。
    """
    # 建立輸出資料夾
    os.makedirs(output_folder, exist_ok=True)

    # 載入 identity 資料夾中的圖片及其編碼
    identity_encodings = []
    identity_files = []
    for file_name in os.listdir(identity_folder):
        if file_name.endswith(".png") or file_name.endswith(".jpg"):
            file_path = os.path.join(identity_folder, file_name)
            image = face_recognition.load_image_file(file_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                identity_encodings.append(encodings[0])  # 只取第一張臉的編碼
                identity_files.append(file_name)

    # 載入 reference 資料夾中的圖片及其編碼
    reference_encodings = []
    reference_files = []
    for file_name in os.listdir(reference_folder):
        if file_name.endswith(".png") or file_name.endswith(".jpg"):
            file_path = os.path.join(reference_folder, file_name)
            image = face_recognition.load_image_file(file_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                reference_encodings.append(encodings[0])  # 只取第一張臉的編碼
                reference_files.append(file_name)

    # 比對兩個資料夾的圖片
    for id_index, id_encoding in enumerate(identity_encodings):
        for ref_index, ref_encoding in enumerate(reference_encodings):
            match = face_recognition.compare_faces([id_encoding], ref_encoding, tolerance=tolerance)
            if match[0]:  # 如果匹配成功
                id_file_name = identity_files[id_index]
                ref_file_name = reference_files[ref_index]

                # 提取數字部分，將 ref 檔案名改成對應的 id 檔案名
                id_number = id_file_name.split('-')[2].split('.')[0]  # 提取 iden 的數字部分
                new_ref_file_name = f"cropped-face-{id_number}-ref.png"

                # 複製並重命名匹配的檔案到輸出資料夾
                shutil.copy(os.path.join(identity_folder, id_file_name), os.path.join(output_folder, id_file_name))
                shutil.copy(os.path.join(reference_folder, ref_file_name), os.path.join(output_folder, new_ref_file_name))

                print(f"匹配成功: {id_file_name} 和 {ref_file_name} -> 在輸出資料夾中重命名為 {new_ref_file_name}")