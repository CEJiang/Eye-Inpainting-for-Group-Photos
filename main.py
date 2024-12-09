import cropped_face
import subprocess
import os
from align import Align
from stick import Stick
import test

def main():
    output_json_dir = "./output"
    output_folder = "./data/cropped_faces"
    os.makedirs(output_json_dir, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    target_size = (256, 256)

    # # identity image path
    iden_image_path = "joel-muniz-HvZDCuRnSaY-unsplash.jpg"
    # cropped_face.crop_faces(iden_image_path, output_folder, output_json_dir, target_size, "iden")

    # # reference image path
    # ref_image_path = "joel-muniz-KodMXENNaas-unsplash.jpg"
    # cropped_face.crop_faces(ref_image_path, output_folder, output_json_dir, target_size, "ref")

    # # Super Resolution
    # gfpgan_script_path = "GFPGAN/inference_gfpgan.py"  # GFP-GAN 的腳本路徑
    # input_dir = "./data/cropped_faces"  # GFP-GAN 的輸入資料夾
    # output_dir = "./data/super_resolved_faces"  # GFP-GAN 的輸出資料夾
    # version = "1.3"  # GFP-GAN 模型版本
    # scale = 2  # 超解析放大倍率

    # # 確認 GFP-GAN 路徑是否正確
    # if not os.path.exists(gfpgan_script_path):
    #     raise FileNotFoundError(f"GFPGAN script not found at {gfpgan_script_path}")

    # # 執行 GFP-GAN
    # try:
    #     subprocess.run(
    #         [
    #             "python", gfpgan_script_path,
    #             "-i", input_dir,
    #             "-o", output_dir,
    #             "-v", version,
    #             "-s", str(scale)
    #         ],
    #         check=True
    #     )
    #     print("Super resolution completed.")
    # except subprocess.CalledProcessError as e:
    #     print(f"Error during super resolution: {e}")
    #     return
    
    # # Align
    # output_json = "./data/align/data.json"
    # super_resolution_dir = "data/super_resolved_faces/restored_imgs"
    # output_image_dir = "./data/align"
    # os.makedirs(output_image_dir, exist_ok=True)
    # Align(super_resolution_dir, output_json, output_image_dir)

    test.test()

    # Super Resolution
    gfpgan_script_path = "GFPGAN/inference_gfpgan.py"  # GFP-GAN 的腳本路徑
    input_dir = "./output/result"  # GFP-GAN 的輸入資料夾
    output_dir = "./output/result2"  # GFP-GAN 的輸出資料夾
    version = "1.3"  # GFP-GAN 模型版本
    scale = 3  # 超解析放大倍率

    # 確認 GFP-GAN 路徑是否正確
    if not os.path.exists(gfpgan_script_path):
        raise FileNotFoundError(f"GFPGAN script not found at {gfpgan_script_path}")

    # 執行 GFP-GAN
    try:
        subprocess.run(
            [
                "python", gfpgan_script_path,
                "-i", input_dir,
                "-o", output_dir,
                "-v", version,
                "-s", str(scale)
            ],
            check=True
        )
        print("Super resolution completed.")
    except subprocess.CalledProcessError as e:
        print(f"Error during super resolution: {e}")
        return
    
    # Stick
    # 路徑設定
    json_path = "output/iden_face_positions.json"  # JSON 文件，記錄人臉位置
    cropped_faces_folder = "./output/result2/restored_imgs"  # 裁剪後臉部圖像的資料夾
    original_image_path = iden_image_path
    output_image_path = "output/reconstructed_image.png"

    Stick(json_path, cropped_faces_folder, original_image_path, output_image_path)
    

if __name__ == "__main__":
    main()