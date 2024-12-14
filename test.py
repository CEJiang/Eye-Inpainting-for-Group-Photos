import argparse
import os
import torch
import numpy as np
import json
import torch.nn.functional as F
from model.utils import mkdir_p, Eyes, save_images, get_image
from model.ExemplarGAN import ExemplarGAN
from torch.utils.data import DataLoader, Dataset

class Config:
    def __init__(self, args):
        # Mode flag: 0 for training, 1 for testing
        self.OPER_FLAG = int(args.mode)
        self.OPER_NAME = args.model_path

        self.batch_size = 1
        self.max_iters = 10000
        self.learn_rate = 0.0001
        self.use_sp = True
        self.lam_recon = 1
        self.lam_gp = 10
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.n_critic = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.OPER_FLAG == 0:
            self.path = args.train_folder
            self.is_load = False
        elif self.OPER_FLAG == 1:
            self.path = "data/align"
            self.is_load = True

class EyesDataset(Dataset):
    def __init__(self, images_name, eye_pos_name, ref_images_name, ref_pos_name, data_path, transform=None, output_size=256, device=None):
        self.images_name = images_name
        self.eye_pos_name = eye_pos_name
        self.ref_images_name = ref_images_name
        self.ref_pos_name = ref_pos_name
        self.data_path = data_path
        self.transform = transform
        self.output_size = output_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.images_name)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_path, self.images_name[idx])
        input_image = get_image(image_path, image_size=256, is_crop=False)
        input_image = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1)
        input_image = F.interpolate(input_image.unsqueeze(0), size=(256, 256)).squeeze(0)

        eye_pos = self.eye_pos_name[idx]
        input_mask = self.get_mask(eye_pos, input_img=input_image.unsqueeze(0))

        ref_image_path = os.path.join(self.data_path, self.ref_images_name[idx])
        exemplar_image = get_image(ref_image_path, image_size=256, is_crop=False)
        exemplar_image = torch.tensor(exemplar_image, dtype=torch.float32).permute(2, 0, 1)
        exemplar_image = F.interpolate(exemplar_image.unsqueeze(0), size=(256, 256)).squeeze(0)

        ref_pos = self.ref_pos_name[idx]
        exemplar_mask = self.get_mask(ref_pos, input_img=exemplar_image.unsqueeze(0))

        return input_image, exemplar_image, input_mask.squeeze(0), exemplar_mask.squeeze(0)


    def get_mask(self, eye_pos, input_img=None):
        batch_size = input_img.size(0) if input_img is not None else 1

        mask = torch.zeros((batch_size, 1, self.output_size, self.output_size), device=self.device)

        left_eye_y, left_eye_x, left_eye_h, left_eye_w = eye_pos[0], eye_pos[1], eye_pos[2], eye_pos[3]
        l1, u1 = max(0, left_eye_y - left_eye_h // 2), min(self.output_size, left_eye_y + left_eye_h // 2)
        l2, u2 = max(0, left_eye_x - left_eye_w // 2), min(self.output_size, left_eye_x + left_eye_w // 2)
        mask[:, :, l1:u1, l2:u2] = 1.0

        right_eye_y, right_eye_x, right_eye_h, right_eye_w = eye_pos[4], eye_pos[5], eye_pos[6], eye_pos[7]
        l1, u1 = max(0, right_eye_y - right_eye_h // 2), min(self.output_size, right_eye_y + right_eye_h // 2)
        l2, u2 = max(0, right_eye_x - right_eye_w // 2), min(self.output_size, right_eye_x + right_eye_w // 2)
        mask[:, :, l1:u1, l2:u2] = 1.0

        mask = mask.expand(-1, 3, -1, -1)
        return mask




def parse_test_data(test_json_path):
    with open(test_json_path, 'r') as f:
        data = json.load(f)

    test_images_name = []
    test_eye_pos_name = []
    test_ref_images_name = []
    test_ref_pos_name = []

    for group_id, images in data.items():
        if len(images) != 2:
            print(f"Group {group_id} does not have exactly 2 images. Skipping...")
            continue

        iden_image = next((img for img in images if 'iden' in img['filename']), None)
        ref_image = next((img for img in images if 'ref' in img['filename']), None)

        if not iden_image or not ref_image:
            print(f"Group {group_id} does not have both iden and ref images. Skipping...")
            continue

        test_images_name.append(iden_image['filename'])
        test_eye_pos_name.append((
            iden_image['eye_left']['y'], iden_image['eye_left']['x'],
            iden_image['box_left']['h'], iden_image['box_left']['w'],
            iden_image['eye_right']['y'], iden_image['eye_right']['x'],
            iden_image['box_right']['h'], iden_image['box_right']['w']
        ))

        test_ref_images_name.append(ref_image['filename'])
        test_ref_pos_name.append((
            ref_image['eye_left']['y'], ref_image['eye_left']['x'],
            ref_image['box_left']['h'], ref_image['box_left']['w'],
            ref_image['eye_right']['y'], ref_image['eye_right']['x'],
            ref_image['box_right']['h'], ref_image['box_right']['w']
        ))

    return test_images_name, test_eye_pos_name, test_ref_images_name, test_ref_pos_name


def test(args):
    FLAGS = Config(args)

    print("OPER_FLAG:", FLAGS.OPER_FLAG)

    root_log_dir = f"./output/log/logs{FLAGS.OPER_FLAG}"
    checkpoint_dir = f"./output/model_gan{FLAGS.OPER_NAME}/"
    model_dir = f"./output/model_gan{FLAGS.OPER_NAME}/model"
    sample_path = f"./output/result"
    mkdir_p(root_log_dir)
    mkdir_p(model_dir)
    
    mkdir_p(sample_path)

    if FLAGS.OPER_FLAG == 0:
        m_ob = Eyes(FLAGS.path)
        train_dataset = EyesDataset(
            m_ob.train_images_name,
            m_ob.train_eye_pos_name,
            m_ob.train_ref_images_name,
            m_ob.train_ref_pos_name,
        )
        train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True)

        test_dataset = EyesDataset(
            m_ob.test_images_name,
            m_ob.test_eye_pos_name,
            m_ob.test_ref_images_name,
            m_ob.test_ref_pos_name,
        )
        test_loader = DataLoader(test_dataset, batch_size=FLAGS.batch_size, shuffle=False)

        eGan = ExemplarGAN(
            batch_size=FLAGS.batch_size,
            max_iters=FLAGS.max_iters,
            model_path=checkpoint_dir,
            data_ob=m_ob,
            sample_path=sample_path,
            log_dir=root_log_dir,
            learning_rate=FLAGS.learn_rate,
            is_load=FLAGS.is_load,
            lam_recon=FLAGS.lam_recon,
            lam_gp=FLAGS.lam_gp,
            use_sp=FLAGS.use_sp,
            beta1=FLAGS.beta1,
            beta2=FLAGS.beta2,
            n_critic=FLAGS.n_critic,
            device=FLAGS.device,
        ).to(FLAGS.device)

    
        print("Start training...")
        eGan.train_model(train_loader, test_loader)
        print("Training complete!")

    elif FLAGS.OPER_FLAG == 1:
        print("Start testing...")
        m_ob = Eyes(FLAGS.path)
        data_json_dir = os.path.join(FLAGS.path, 'data.json')
        test_images_name, test_eye_pos_name, test_ref_images_name, test_ref_pos_name = parse_test_data(data_json_dir)

        test_dataset = EyesDataset(
            test_images_name,
            test_eye_pos_name,
            test_ref_images_name,
            test_ref_pos_name,
            FLAGS.path
        )
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        eGan = ExemplarGAN(
            batch_size=FLAGS.batch_size,
            max_iters=FLAGS.max_iters,
            model_path=checkpoint_dir,
            data_ob=m_ob,
            sample_path=sample_path,
            log_dir=root_log_dir,
            learning_rate=FLAGS.learn_rate,
            is_load=FLAGS.is_load,
            lam_recon=FLAGS.lam_recon,
            lam_gp=FLAGS.lam_gp,
            use_sp=FLAGS.use_sp,
            beta1=FLAGS.beta1,
            beta2=FLAGS.beta2,
            n_critic=FLAGS.n_critic,
            device=FLAGS.device,
        ).to(FLAGS.device)

        model_path = args.model_path
        if os.path.exists(model_path):
            if(torch.cuda.is_available()):
                eGan.load_model(model_path)
            else:
                state_dict = torch.load(model_path, map_location=torch.device('cpu'))
                eGan.load_state_dict(state_dict)
            eGan.eval()
            with torch.no_grad():
                for i, batch in enumerate(test_loader):
                    input_img, exemplar_images, img_mask, exemplar_mask = [x.to(FLAGS.device) for x in batch]
                    x_tilde = eGan(input_img, exemplar_images, img_mask, exemplar_mask)
                    print(f"Test batch {i} output shape: {x_tilde.shape}")
                    original_name = test_images_name[i]
                    modified_name = original_name.replace("-iden", "")
                    save_images(x_tilde.cpu().numpy(), [1, 1],
                                f"{sample_path}/{modified_name}")
            print("Testing complete!")
        else:
            print(f"Model not found at {model_path}. Ensure the correct test_step is provided.")