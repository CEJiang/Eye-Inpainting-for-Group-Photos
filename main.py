import argparse
import cropped_face
import subprocess
import os
from align import Align
from map_face_to_image import map
import test
import face_recognition_file
import psnr

def main(args):
    output_json_dir = "./output"
    identity_folder = "./data/iden_cropped_faces"
    os.makedirs(output_json_dir, exist_ok=True)
    os.makedirs(identity_folder, exist_ok=True)
    target_size = (256, 256)

    # identity image path
    iden_image_path = args.i
    cropped_face.crop_faces(iden_image_path, identity_folder, output_json_dir, target_size, "iden")

    reference_folder = "./data/ref_cropped_faces"
    os.makedirs(reference_folder, exist_ok=True)
    # reference image path
    ref_image_path = args.r
    cropped_face.crop_faces(ref_image_path, reference_folder, output_json_dir, target_size, "ref")

    output_folder = "./data/cropped_faces"
    os.makedirs(output_folder, exist_ok=True)

    face_recognition_file.compare_and_rename_faces(identity_folder, reference_folder, output_folder, tolerance=0.5)

    # Super Resolution
    gfpgan_script_path = "GFPGAN/inference_gfpgan.py"
    input_dir = "./data/cropped_faces"
    output_dir = "./data/super_resolved_faces"
    version = "1.3"
    scale = 10

    if not os.path.exists(gfpgan_script_path):
        raise FileNotFoundError(f"GFPGAN script not found at {gfpgan_script_path}")

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
    
    # Align
    output_json = "./data/align/data.json"
    super_resolution_dir = "data/super_resolved_faces/restored_imgs"
    output_image_dir = "./data/align"
    os.makedirs(output_image_dir, exist_ok=True)
    Align(super_resolution_dir, output_json, output_image_dir)

    test.test(args)

    # Super Resolution
    gfpgan_script_path = "GFPGAN/inference_gfpgan.py"
    input_dir = "./output/result"
    output_dir = "./output/result2"
    version = "1.3"
    scale = 10

    if not os.path.exists(gfpgan_script_path):
        raise FileNotFoundError(f"GFPGAN script not found at {gfpgan_script_path}")

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
    
    json_path = "output/iden_face_positions.json"
    cropped_faces_folder = "./output/result2/restored_imgs"
    original_image_path = iden_image_path
    output_image_path = "output/reconstructed_image.png"

    map(json_path, cropped_faces_folder, original_image_path, output_image_path)
    
    print("PSNR: {:.2f} dB".format(psnr.calculate_psnr(output_image_path, original_image_path)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image inpainting pipeline.")
    
    parser.add_argument(
        "--mode",
        type=int,
        choices=[0, 1],
        required=True,
        help="Mode of operation: 0 for training, 1 for testing."
    )
    
    # test
    parser.add_argument(
        "--i",
        type=str,
        help="Path to the identity image. Required in testing mode."
    )
    parser.add_argument(
        "--r",
        type=str,
        help="Path to the reference image. Required in testing mode."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model. Required in testing mode."
    )
    
    # train
    parser.add_argument(
        "--model_folder",
        type=str,
        help="Path to the model folder. Required in training mode."
    )
    parser.add_argument(
        "--train_folder",
        type=str,
        help="Path to the training data folder. Required in training mode."
    )
    
    args = parser.parse_args()

    if args.mode == 1:
        if not all([args.i, args.r, args.model_path]):
            parser.error("In testing mode (--mode 1), --i, --r, and --model_path are required.")
    elif args.mode == 0:
        if not all([args.model_folder, args.train_folder]):
            parser.error("In training mode (--mode 0), --model_folder and --train_folder is required.")
    
    if args.mode == 1:
        print(f"Testing mode selected. Identity image: {args.i}, Reference image: {args.r}, Model path: {args.model_path}")
        main(args)
    elif args.mode == 0:
        print(f"Training mode selected. Model folder: {args.model_folder} Training floder: {args.train_folder}")
        test.test(args)