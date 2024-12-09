import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from model.ops import Conv2dSN, DeConv2d, lrelu, InstanceNorm2d, ResidualBlock, fully_connect
from model.utils import save_images
import numpy as np
import os


class ExemplarGAN(nn.Module):
    def __init__(self, batch_size, max_iters, model_path, data_ob, sample_path, log_dir, learning_rate, is_load, lam_recon,
                 lam_gp, use_sp, beta1, beta2, n_critic, device):
        super(ExemplarGAN, self).__init__()

        self.batch_size = batch_size
        self.max_iters = max_iters
        self.model_path = model_path
        self.data_ob = data_ob
        self.sample_path = sample_path
        self.log_dir = log_dir
        self.learning_rate = learning_rate
        self.channel = data_ob.channel
        self.output_size = data_ob.image_size
        self.lam_recon = lam_recon
        self.lam_gp = lam_gp
        self.use_sp = use_sp
        self.is_load = is_load
        self.beta1 = beta1
        self.beta2 = beta2
        self.n_critic = n_critic
        self.device = device

        # Initialize networks
        self.encoder_decoder = self.build_encoder_decoder()
        self.discriminator = self.build_discriminator()

    def build_encoder_decoder(self):
        """
        Build the encode-decode network.
        """
        layers = [
            Conv2dSN(12, 64, kernel_size=7, stride=1, padding=3, spectral_norm=self.use_sp),  # Update input channels to 12
            InstanceNorm2d(64),
            nn.ReLU(),

            Conv2dSN(64, 128, kernel_size=4, stride=2, padding=1, spectral_norm=self.use_sp),
            InstanceNorm2d(128),
            nn.ReLU(),

            Conv2dSN(128, 256, kernel_size=4, stride=2, padding=1, spectral_norm=self.use_sp),
            InstanceNorm2d(256),
            nn.ReLU(),

            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),

            DeConv2d(256, 128, kernel_size=4, stride=2, padding=1),
            InstanceNorm2d(128),
            nn.ReLU(),

            DeConv2d(128, 64, kernel_size=4, stride=2, padding=1),
            InstanceNorm2d(64),
            nn.ReLU(),

            Conv2dSN(64, self.channel, kernel_size=7, stride=1, padding=3, spectral_norm=self.use_sp),
            nn.Tanh()
        ]
        return nn.Sequential(*layers)


    def build_discriminator(self):
        """
        Build the discriminator network.
        """
        layers = []
        input_channels = self.channel * 2  # input_img + exemplar_images
        for i in range(5):
            output_channels = min(64 * (2 ** i), 512)
            layers.append(Conv2dSN(input_channels, output_channels, kernel_size=4, stride=2, padding=1, spectral_norm=self.use_sp))
            layers.append(nn.LeakyReLU(0.2))
            input_channels = output_channels

        # Add flattening and fully connected layer
        self.feature_map_size = (self.output_size // (2 ** 5))  # 5 downsampling layers, each halves the spatial dimensions
        flattened_size = self.feature_map_size ** 2 * output_channels

        layers.append(nn.Flatten())
        layers.append(nn.Linear(flattened_size, 1))
        return nn.Sequential(*layers)


    def forward(self, input_img, exemplar_images, img_mask, exemplar_mask):
        """
        Forward pass of the generator.
        """
        # Concatenate inputs
        incomplete_img = input_img * (1 - img_mask)  # Masked input image
        x_var = torch.cat([incomplete_img, img_mask, exemplar_images, exemplar_mask], dim=1)

        # Debugging print to verify input shape
        # print(f"x_var shape: {x_var.shape}")  # Should match the input channels of the first conv layer

        # Pass through encoder-decoder
        x_tilde = self.encoder_decoder(x_var)
        return x_tilde

    def discriminate(self, input_img, exemplar_images, x_tilde):
        """
        Discriminator forward pass.
        """
        # Global discriminator input: Concatenate input_img and exemplar_images
        global_input = torch.cat([input_img, exemplar_images], dim=1)
        global_logits = self.discriminator(global_input)

        # Local discriminator input: Concatenate x_tilde with exemplar_images
        local_input = torch.cat([x_tilde, exemplar_images], dim=1)
        local_logits = self.discriminator(local_input)

        return global_logits, local_logits


    def loss_dis(self, d_real_logits, d_fake_logits):
        """
        Compute the discriminator loss.
        """
        l1 = F.softplus(-d_real_logits).mean()
        l2 = F.softplus(d_fake_logits).mean()
        return l1 + l2

    def loss_gen(self, d_fake_logits):
        """
        Compute the generator loss.
        """
        return F.softplus(-d_fake_logits).mean()

    def train_model(self, train_loader, test_loader):
        """
        Train the GAN model with periodic testing.

        Parameters:
        - train_loader: DataLoader for training data.
        - test_loader: DataLoader for testing data.
        """
        os.makedirs(self.log_dir, exist_ok=True)
        writer = SummaryWriter(self.log_dir)

        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate, betas=(self.beta1, self.beta2))
        optimizer_G = torch.optim.Adam(self.encoder_decoder.parameters(), lr=self.learning_rate, betas=(self.beta1, self.beta2))

        for epoch in range(self.max_iters):
            for batch_idx, (input_img, exemplar_images, img_mask, exemplar_mask) in enumerate(train_loader):
                input_img, exemplar_images, img_mask, exemplar_mask = (
                    input_img.to(self.device),
                    exemplar_images.to(self.device),
                    img_mask.to(self.device),
                    exemplar_mask.to(self.device),
                )

                # Discriminator step
                optimizer_D.zero_grad()
                x_tilde = self(input_img, exemplar_images, img_mask, exemplar_mask)
                real_logits, fake_logits = self.discriminate(input_img, exemplar_images, x_tilde.detach())
                d_loss = self.loss_dis(real_logits, fake_logits)
                d_loss.backward()  # First backward pass for the discriminator
                optimizer_D.step()

                # Generator step
                optimizer_G.zero_grad()
                x_tilde = self(input_img, exemplar_images, img_mask, exemplar_mask)  # Recompute x_tilde
                _, fake_logits = self.discriminate(input_img, exemplar_images, x_tilde)
                g_loss = self.loss_gen(fake_logits) + self.lam_recon * F.l1_loss(x_tilde, input_img)
                g_loss.backward(retain_graph=False)  # Second backward pass for the generator
                optimizer_G.step()

                # TensorBoard logging for each batch
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar("Loss/Discriminator", d_loss.item(), global_step)
                writer.add_scalar("Loss/Generator", g_loss.item(), global_step)

                # Print training loss
                print(f"Epoch [{epoch}/{self.max_iters}] Batch [{batch_idx}/{len(train_loader)}] "
                    f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

            # Periodic testing
            if (epoch + 1) % 5 == 0:  # Test every 5 epochs
                print(f"Testing at epoch {epoch + 1}...")
                self.encoder_decoder.eval()  # Switch to evaluation mode
                self.test_model(test_loader, epoch)
                self.encoder_decoder.train()  # Switch back to training mode

            # Save model checkpoints periodically
            if (epoch + 1) % 5 == 0:  # Save every 10 epochs
                checkpoint_path = os.path.join(self.model_path, f"model_epoch_{epoch + 1}.pth")
                self.save_model(checkpoint_path)
                print(f"Model saved at {checkpoint_path}")
            
        writer.close()


    def test_model(self, test_loader, epoch):
        with torch.no_grad():
            for i, (input_img, exemplar_images, img_mask, exemplar_mask) in enumerate(test_loader):
                input_img, exemplar_images, img_mask, exemplar_mask = (
                    input_img.to(self.device),
                    exemplar_images.to(self.device),
                    img_mask.to(self.device),
                    exemplar_mask.to(self.device),
                )
                # Generate images
                generated_img = self(input_img, exemplar_images, img_mask, exemplar_mask)

                # Debug shapes
                print(f"Input image shape: {input_img.shape}")
                print(f"Generated image shape: {generated_img.shape}")

                # Concatenate images for saving
                result = torch.cat([input_img, exemplar_images, generated_img], dim=3)  # Horizontal concatenation

                # Save results
                save_images(
                    result.cpu().numpy(),
                    size=[result.size(0), 1],  # Adjust grid size based on batch size
                    image_path=f"{self.sample_path}/epoch_{(epoch + 1)}_batch_{i}.png",
                )
                if i >= 5:  # Test on a limited number of batches
                    break

        

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def get_Mask(self, eye_pos):
        """
        Generate a mask based on eye positions.
        """
        masks = []
        for pos in eye_pos:
            mask = torch.zeros((self.output_size, self.output_size, self.channel), device=self.device)
            l1, u1, l2, u2 = max(0, pos[0] - 25), min(self.output_size, pos[0] + 25), \
                             max(0, pos[1] - 35), min(self.output_size, pos[1] + 35)
            mask[l1:u1, l2:u2, :] = 1.0
            masks.append(mask)
        return torch.stack(masks)
