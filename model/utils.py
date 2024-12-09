import os
import numpy as np
from PIL import Image
import json


def mkdir_p(path):
    """
    確保目錄存在，如果不存在則創建。
    """
    try:
        os.makedirs(path, exist_ok=True)  # Python 3.2+ 支持 exist_ok 參數
    except OSError as exc:
        if not os.path.isdir(path):
            raise RuntimeError(f"無法創建目錄 {path}: {exc}") from exc

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale=False, is_test=False):
    """
    加載並處理圖像。
    """
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w, is_test=is_test)


def transform(image, npx=64, is_crop=False, resize_w=64, is_test=False):
    """
    Transform image by optionally cropping and resizing.
    """
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w, is_test=is_test)
    else:
        cropped_image = image

    # Convert to PIL Image if it's a numpy array
    if isinstance(cropped_image, np.ndarray):
        cropped_image = Image.fromarray(cropped_image.astype(np.uint8))

    # Resize the image
    cropped_image = cropped_image.resize((resize_w, resize_w), Image.BICUBIC)

    return np.array(cropped_image) / 127.5 - 1


def center_crop(x, crop_h, crop_w=None, resize_w=64, is_test=False):
    """
    Center crop the image.
    """
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))

    if not is_test:
        rate = np.random.uniform(0, 1, size=1)
        if rate < 0.5:
            x = np.fliplr(x)  # Random horizontal flip

    # Crop the image
    cropped = x[j:j + crop_h, i:i + crop_w]

    # Convert cropped result to uint8 if it's not
    if cropped.dtype != np.uint8:
        cropped = (cropped * 255).astype(np.uint8)

    return cropped



def save_images(images, size, image_path, is_output=False):
    """
    保存圖像。
    """
    return imsave(inverse_transform(images, is_output), size, image_path)


def imread(path, is_grayscale=False):
    """
    讀取圖像。
    """
    img = Image.open(path)
    if is_grayscale:
        img = img.convert('L')
    return np.array(img)


def imsave(images, size, path):
    """
    保存合併後的圖像。
    """
    img = merge(images, size)
    img = Image.fromarray(img.astype(np.uint8))
    img.save(path)


def merge(images, size):
    """
    合并多张图像为一张大图。

    参数:
    - images: 图像数组，形状为 (B, C, H, W)。
    - size: 图像排列大小，例如 (rows, cols)。

    返回:
    - 拼接后的大图，形状为 (H * rows, W * cols, C)。
    """
    # 转换为 numpy 并确保形状为 (B, H, W, C)
    images = images.transpose(0, 2, 3, 1)  # 从 (B, C, H, W) 转为 (B, H, W, C)

    h, w, c = images.shape[1], images.shape[2], images.shape[3]  # 高度、宽度、通道数
    img = np.zeros((h * size[0], w * size[1], c))  # 初始化目标图像

    for idx, image in enumerate(images):
        i = idx % size[1]  # 列索引
        j = idx // size[1]  # 行索引
        img[j * h:j * h + h, i * w:i * w + w, :] = image  # 填充图像

    return img


def inverse_transform(images, is_output=False):
    """
    復原圖像到 [0, 255] 範圍。
    """
    result = ((images + 1) * 127.5).astype(np.uint8)
    if is_output:
        print(result)
    return result


def read_image_list_for_Eyes(category):
    json_cat = os.path.join(category, "data.json")
    with open(json_cat, 'r') as f:
        data = json.load(f)

    all_iden_info = []
    all_ref_info = []
    test_all_iden_info = []
    test_all_ref_info = []

    for c, (k, v) in enumerate(data.items()):
        identity_info = []
        is_close = False
        is_close_id = 0

        if c % 1000 == 0:  # 避免 log_interval 重複定義
            print(f'Processed {c}/{len(data)}')

        if len(v) < 2:
            continue

        for i in range(len(v)):
            # 如果 `opened` 為 None，跳過此條目
            if v[i].get('opened') is None or v[i]['opened'] < 0.60:
                is_close = True
                is_close_id = i

            str_info = str(v[i]['filename']) + "_"

            if 'eye_left' in v[i] and v[i]['eye_left'] is not None:
                str_info += f"{v[i]['eye_left']['y']}_{v[i]['eye_left']['x']}_"
            else:
                str_info += "0_0_"

            if 'box_left' in v[i] and v[i]['box_left'] is not None:
                str_info += f"{v[i]['box_left']['h']}_{v[i]['box_left']['w']}_"
            else:
                str_info += "0_0_"

            if 'eye_right' in v[i] and v[i]['eye_right'] is not None:
                str_info += f"{v[i]['eye_right']['y']}_{v[i]['eye_right']['x']}_"
            else:
                str_info += "0_0_"

            if 'box_right' in v[i] and v[i]['box_right'] is not None:
                str_info += f"{v[i]['box_right']['h']}_{v[i]['box_right']['w']}"
            else:
                str_info += "0_0"

            identity_info.append(str_info)

        if not is_close:
            for _ in range(len(v)):
                first_n = np.random.randint(0, len(v))
                all_iden_info.append(identity_info[first_n])
                middle_value = identity_info[first_n]
                identity_info.remove(middle_value)

                second_n = np.random.randint(0, len(v) - 1)
                all_ref_info.append(identity_info[second_n])

                identity_info.append(middle_value)
        else:
            middle_value = identity_info[is_close_id]
            test_all_iden_info.append(middle_value)
            identity_info.remove(middle_value)

            second_n = np.random.randint(0, len(v) - 1)
            test_all_ref_info.append(identity_info[second_n])

            test_all_iden_info.append(middle_value)
            second_n = np.random.randint(0, len(v) - 1)
            test_all_ref_info.append(identity_info[second_n])

    assert len(all_iden_info) == len(all_ref_info)
    assert len(test_all_iden_info) == len(test_all_ref_info)

    print(f"train_data: {len(all_iden_info)}")
    print(f"test_data: {len(test_all_iden_info)}")

    return all_iden_info, all_ref_info, test_all_iden_info, test_all_ref_info



class Eyes:
    """
    Eyes 數據管理類
    """
    def __init__(self, image_path):
        self.image_size = 256
        self.channel = 3
        self.image_path = image_path
        self.train_images_name, self.train_eye_pos_name, self.train_ref_images_name, self.train_ref_pos_name, \
            self.test_images_name, self.test_eye_pos_name, self.test_ref_images_name, self.test_ref_pos_name = self.load_Eyes(image_path)

    def load_Eyes(self, image_path):
        images_list, images_ref_list, test_images_list, test_images_ref_list = read_image_list_for_Eyes(image_path)

        train_images_name = []
        train_eye_pos_name = []
        train_ref_images_name = []
        train_ref_pos_name = []

        test_images_name = []
        test_eye_pos_name = []
        test_ref_images_name = []
        test_ref_pos_name = []

        # Train data
        for images_info_str in images_list:
            image_name, *eye_pos = images_info_str.split('_')
            image_name = os.path.join(self.image_path, image_name)
            train_images_name.append(image_name)
            train_eye_pos_name.append(tuple(map(int, eye_pos)))

        for images_info_str in images_ref_list:
            image_name, *eye_pos = images_info_str.split('_')
            image_name = os.path.join(self.image_path, image_name)
            train_ref_images_name.append(image_name)
            train_ref_pos_name.append(tuple(map(int, eye_pos)))

        # Test data
        for images_info_str in test_images_list:
            image_name, *eye_pos = images_info_str.split('_')
            image_name = os.path.join(self.image_path, image_name)
            test_images_name.append(image_name)
            test_eye_pos_name.append(tuple(map(int, eye_pos)))

        for images_info_str in test_images_ref_list:
            image_name, *eye_pos = images_info_str.split('_')
            image_name = os.path.join(self.image_path, image_name)
            test_ref_images_name.append(image_name)
            test_ref_pos_name.append(tuple(map(int, eye_pos)))

        assert len(train_images_name) == len(train_eye_pos_name) == len(train_ref_images_name) == len(train_ref_pos_name)
        assert len(test_images_name) == len(test_eye_pos_name) == len(test_ref_images_name) == len(test_ref_pos_name)

        return (
            train_images_name,
            train_eye_pos_name,
            train_ref_images_name,
            train_ref_pos_name,
            test_images_name,
            test_eye_pos_name,
            test_ref_images_name,
            test_ref_pos_name,
        )

    def process_images(self, images_list, ref_list):
        """
        處理圖像和參考圖像。
        """
        images, eyes, ref_images, ref_eyes = [], [], [], []
        for image_info in images_list:
            image, eye = self.parse_info(image_info)
            images.append(image)
            eyes.append(eye)
        for ref_info in ref_list:
            ref_image, ref_eye = self.parse_info(ref_info)
            ref_images.append(ref_image)
            ref_eyes.append(ref_eye)
        return images, eyes, ref_images, ref_eyes

    def parse_info(self, info_str):
        """
        解析圖像信息。
        """
        parts = info_str.split('_')
        eye_pos = tuple(map(int, parts[1:]))
        image_path = os.path.join(self.image_path, parts[0])
        return image_path, eye_pos
    