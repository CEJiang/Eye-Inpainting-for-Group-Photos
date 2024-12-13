import face_recognition
import os
import shutil

def compare_and_rename_faces(identity_folder, reference_folder, output_folder, tolerance=0.6):
    os.makedirs(output_folder, exist_ok=True)

    identity_encodings = []
    identity_files = []
    for file_name in os.listdir(identity_folder):
        if file_name.endswith(".png") or file_name.endswith(".jpg"):
            file_path = os.path.join(identity_folder, file_name)
            image = face_recognition.load_image_file(file_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                identity_encodings.append(encodings[0])
                identity_files.append(file_name)

    reference_encodings = []
    reference_files = []
    for file_name in os.listdir(reference_folder):
        if file_name.endswith(".png") or file_name.endswith(".jpg"):
            file_path = os.path.join(reference_folder, file_name)
            image = face_recognition.load_image_file(file_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                reference_encodings.append(encodings[0])
                reference_files.append(file_name)

    for id_index, id_encoding in enumerate(identity_encodings):
        for ref_index, ref_encoding in enumerate(reference_encodings):
            match = face_recognition.compare_faces([id_encoding], ref_encoding, tolerance=tolerance)
            if match[0]:
                id_file_name = identity_files[id_index]
                ref_file_name = reference_files[ref_index]

                id_number = id_file_name.split('-')[2].split('.')[0]
                new_ref_file_name = f"cropped-face-{id_number}-ref.png"

                shutil.copy(os.path.join(identity_folder, id_file_name), os.path.join(output_folder, id_file_name))
                shutil.copy(os.path.join(reference_folder, ref_file_name), os.path.join(output_folder, new_ref_file_name))

                print(f"匹配成功: {id_file_name} 和 {ref_file_name} -> 在輸出資料夾中重命名為 {new_ref_file_name}")