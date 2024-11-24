import cropped_face

def main():
    input_image_path = "image.png"
    output_folder = "./data/"
    target_size=(256, 256)
    cropped_face.crop_faces(input_image_path, output_folder, target_size)

if __name__ == "__main__":
    main()