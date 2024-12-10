import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchvision.utils import save_image, make_grid
from torch.autograd import Variable
import itertools
import os
import time
import datetime
import numpy as np
from PIL import Image
import logging
from pathlib import Path
import argparse
import gc

torch.cuda.empty_cache()
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of available GPU memory

# Memory management settings
torch.cuda.empty_cache()
gc.collect()
torch.cuda.set_per_process_memory_fraction(0.7)  # Use only 70% of GPU memory
torch.backends.cudnn.benchmark = True

# Add this environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.8'

print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024 ** 2} MB")

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

# Configuration
CONFIG = {
    'training': {
        'n_epochs': 3,
        'batch_size': 2,
        'learning_rate': 0.0002,
        'beta1': 0.5,
        'beta2': 0.999,
        'lambda_cycle': 20.0,
        'lambda_identity': 10.0,
        'sample_interval': 100
    },
    'model': {
        'img_height': 224,
        'img_width': 224,
        'channels': 3,
        'num_residual_blocks': 9
    },
    'paths': {
        'dataset_path': "./util/datasets/real-fake_balanced/val/real",
        'checkpoint_dir': "./checkpoints",
        'sample_dir': "./samples",
        'best_models_dir': "./best_models",
        'log_dir': "./logs"
    }
}

class Generator(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(Generator, self).__init__()
        channels = input_shape[0]

        # Process each color channel separately initially
        self.channel_conv = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(1, 21, 7),  # 21 features per channel
                nn.InstanceNorm2d(21),
                nn.ReLU(inplace=True)
            ) for _ in range(3)
        ])

        # Merge channels
        model = [
            nn.Conv2d(63, 64, 1),  # Merge channel features
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        channels_list = [64, 128, 256]
        for i in range(2):
            in_features = channels_list[i]
            out_features = channels_list[i + 1]
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]

        # Residual blocks with careful attention
        self.attention = SelfAttention(256)
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(256)]

        # Upsampling with channel attention
        channels_list = [256, 128, 64]
        for i in range(2):
            in_features = channels_list[i]
            out_features = channels_list[i + 1]
            model += [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]

        # Separate channel processing for output
        self.output_conv = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(64, 1, 7),
                nn.Tanh()
            ) for _ in range(3)
        ])

        self.model = nn.Sequential(*model)

    def forward(self, x):
        # Process each channel separately
        channel_features = []
        for i, conv in enumerate(self.channel_conv):
            channel_features.append(conv(x[:, i:i + 1]))

        # Concatenate channel features
        features = torch.cat(channel_features, dim=1)

        # Main processing
        features = self.model(features)

        # Generate each channel separately
        output_channels = []
        for conv in self.output_conv:
            output_channels.append(conv(features))

        # Combine channels
        output = torch.cat(output_channels, dim=1)

        # Ensure channel balance
        channel_means = output.mean(dim=[2, 3], keepdim=True)
        channel_std = output.std(dim=[2, 3], keepdim=True)
        output = (output - channel_means) / (channel_std + 1e-8)
        output = output.tanh()

        return output

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),  # Changed from LeakyReLU
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True)  # Changed from LeakyReLU
        )

    def forward(self, x):
        return x + self.block(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()

        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, height, width = x.size()
        assert C == self.in_channels, f"Input channels {C} doesn't match initialized channels {self.in_channels}"

        # Scale factor for dot product attention
        scaling_factor = float(self.in_channels // 8) ** -0.5

        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)

        attention = torch.bmm(query, key) * scaling_factor
        attention = self.softmax(attention)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, height, width)

        return self.gamma * out + x


class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.files = [f for f in Path(root).glob('*') if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
        self.root = root

    def __getitem__(self, index):
        img_path = self.files[index]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return {'real': img}  # Remove 'imp' as it's not needed
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return next valid image
            return self.__getitem__((index + 1) % len(self))

    def __len__(self):
        return len(self.files)


class Discriminator(nn.Module):
    def __init__(self, input_shape, is_style=False):
        super(Discriminator, self).__init__()

        # Choose architecture based on discriminator type
        if is_style:
            # Style discriminator (ResNet34)
            self.base_model = models.resnet34(weights=None)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()

            # Attention for style
            self.attention = nn.Sequential(
                nn.Linear(num_features, num_features // 16),
                nn.ReLU(),
                nn.Linear(num_features // 16, num_features),
                nn.Sigmoid()
            )

            # Style classifier
            self.classifier = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1)
            )
        else:
            # Real/Fake discriminator (ResNet50)
            self.base_model = models.resnet50(weights=None)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()

            # Attention for manipulation detection
            self.attention = nn.Sequential(
                nn.Linear(num_features, num_features // 8),
                nn.ReLU(),
                nn.Linear(num_features // 8, num_features),
                nn.Sigmoid()
            )

            # Deeper classifier
            self.classifier = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1)
            )

        self.output_shape = (1, 1, 1)

    def forward(self, x):
        features = self.base_model(x)
        attention_weights = self.attention(features)
        weighted_features = features * attention_weights
        output = self.classifier(weighted_features)
        return output.view(-1, *self.output_shape)


class StyleTransfer:
    def __init__(self):
        self.config = CONFIG
        self.setup_logging()
        self.setup_directories()
        self.setup_device()
        self.setup_transforms()

        self.input_shape = (self.config['model']['channels'],
                            self.config['model']['img_height'],
                            self.config['model']['img_width'])

        self.G_real2imp = Generator(self.input_shape,
                                    self.config['model']['num_residual_blocks'])
        self.G_imp2real = Generator(self.input_shape,
                                    self.config['model']['num_residual_blocks'])

        if torch.cuda.is_available():
            self.G_real2imp = self.G_real2imp.to(self.device)
            self.G_imp2real = self.G_imp2real.to(self.device)

        # Add color balance weight to config
        self.config['training']['lambda_color'] = 0.5

    def setup_transforms(self):
        self.transforms = transforms.Compose([
            transforms.Resize((self.config['model']['img_height'],
                               self.config['model']['img_width'])),
            transforms.ToTensor()  # Just convert to [0, 1] range
        ])

    def normalize_channels(self, x):
        """Normalize each channel independently"""
        b, c, h, w = x.size()
        x_reshaped = x.view(b, c, -1)
        mean = x_reshaped.mean(dim=2).view(b, c, 1, 1)
        std = x_reshaped.std(dim=2).view(b, c, 1, 1)
        return (x - mean) / (std + 1e-8)

    def color_balance_loss(self, image):
        """Fixed color balance loss with proper patch handling"""
        # Global color balance
        channel_means = image.mean(dim=[2, 3])
        r, g, b = channel_means[:, 0], channel_means[:, 1], channel_means[:, 2]
        global_loss = torch.mean((r - g) ** 2 + (r - b) ** 2 + (g - b) ** 2)

        # Local color balance (per-patch)
        b, c, h, w = image.size()
        # Ensure patch size divides image dimensions
        patch_size = min(32, h // 4, w // 4)
        stride = patch_size

        # Reshape to handle patches properly
        patches = image.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
        patches = patches.contiguous().view(b, c, -1)  # Combine all patches

        # Calculate mean per channel per patch
        patch_means = patches.mean(dim=2)  # [batch, channels]
        patch_r, patch_g, patch_b = patch_means[:, 0], patch_means[:, 1], patch_means[:, 2]

        local_loss = torch.mean((patch_r - patch_g) ** 2 + (patch_r - patch_b) ** 2 + (patch_g - patch_b) ** 2)

        return global_loss + 0.5 * local_loss

    def save_sample_images(self, real_img, fake_imp, epoch, batch_idx):
        """Save sample images during training"""
        if not os.path.exists('samples'):
            os.makedirs('samples')

        with torch.no_grad():
            img_sample = torch.cat((real_img.data, fake_imp.data), -2)
            save_image(img_sample,
                       f"samples/sample_e{epoch}_b{batch_idx}.png",
                       normalize=True)

    def setup_logging(self):
        os.makedirs(self.config['paths']['log_dir'], exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(
                    os.path.join(self.config['paths']['log_dir'],
                                 'style_transfer.log')
                ),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_directories(self):
        for directory in [self.config['paths']['checkpoint_dir'],
                          self.config['paths']['sample_dir'],
                          self.config['paths']['best_models_dir']]:
            os.makedirs(directory, exist_ok=True)

    def setup_device(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f'Using device: {self.device}')

    def load_pretrained_discriminators(self):
        try:
            torch.cuda.empty_cache()

            # Initialize with correct architectures
            self.D_style = Discriminator(self.input_shape, is_style=True)
            self.D_real = Discriminator(self.input_shape, is_style=False)

            # Load weights
            style_checkpoint = torch.load('discriminator_style.pth',
                                          map_location=self.device)
            self.D_style.load_state_dict(style_checkpoint['model_state_dict'])

            real_checkpoint = torch.load('discriminator_real-fake.pth',
                                         map_location=self.device)
            self.D_real.load_state_dict(real_checkpoint['model_state_dict'])

            # Move to device and set to eval mode
            if self.device == 'cuda':
                self.D_style = self.D_style.cuda()
                self.D_real = self.D_real.cuda()

            self.D_style.eval()
            self.D_real.eval()

            # Freeze discriminator weights
            for param in self.D_style.parameters():
                param.requires_grad = False
            for param in self.D_real.parameters():
                param.requires_grad = False

            self.logger.info("Successfully loaded pre-trained discriminators")

        except Exception as e:
            self.logger.error(f"Failed to load discriminators: {str(e)}")
            raise

    def validate_batch(self, real_img):
        try:
            self.G_real2imp.eval()
            self.G_imp2real.eval()
            with torch.no_grad():
                fake_imp = self.G_real2imp(real_img)
                loss_style = self.D_style(fake_imp).mean()
                loss_real = self.D_real(fake_imp).mean()
            return (loss_style + loss_real) / 2
        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            return float('inf')

    def adjust_batch_size(self):
        if self.config['training']['batch_size'] > 1:
            self.config['training']['batch_size'] //= 2
            self.logger.warning(f"Reduced batch size to {self.config['training']['batch_size']}")
            return True
        return False

    # def train(self):
    #     try:
    #         # Data loading
    #         dataset = ImageDataset(self.config['paths']['dataset_path'],
    #                                self.transforms)
    #         dataloader = DataLoader(
    #             dataset,
    #             batch_size=self.config['training']['batch_size'],
    #             shuffle=True,
    #             num_workers=4
    #         )
    #
    #         # Loss functions
    #         criterion_GAN = nn.MSELoss()
    #         criterion_cycle = nn.L1Loss()
    #         criterion_identity = nn.L1Loss()
    #
    #         if self.device == 'cuda':
    #             criterion_GAN.cuda()
    #             criterion_cycle.cuda()
    #             criterion_identity.cuda()
    #
    #         # Optimizers
    #         optimizer_G = torch.optim.Adam(
    #             itertools.chain(self.G_real2imp.parameters(),
    #                             self.G_imp2real.parameters()),
    #             lr=self.config['training']['learning_rate'],
    #             betas=(self.config['training']['beta1'],
    #                    self.config['training']['beta2'])
    #         )
    #
    #         # Learning rate scheduler
    #         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #             optimizer_G, mode='min', factor=0.5, patience=2, verbose=True
    #         )
    #
    #         # Training setup
    #         best_loss = float('inf')
    #         patience = 5
    #         no_improve_count = 0
    #         self.logger.info("Starting training...")
    #
    #         for epoch in range(self.config['training']['n_epochs']):
    #             epoch_loss = 0.0
    #             self.G_real2imp.train()
    #             self.G_imp2real.train()
    #
    #             for i, batch in enumerate(dataloader):
    #                 try:
    #                     real_img = batch["real"].to(self.device)
    #                     valid = torch.ones((real_img.size(0), 1, 1, 1)).to(self.device)
    #                     fake = torch.zeros((real_img.size(0), 1, 1, 1)).to(self.device)
    #
    #                     optimizer_G.zero_grad()
    #
    #                     # Identity loss
    #                     loss_id_real = criterion_identity(self.G_imp2real(real_img), real_img)
    #                     loss_id_imp = criterion_identity(self.G_real2imp(real_img), real_img)
    #                     loss_identity = (loss_id_real + loss_id_imp) / 2
    #
    #                     # GAN loss
    #                     fake_imp = self.G_real2imp(real_img)
    #                     loss_GAN_style = criterion_GAN(self.D_style(fake_imp), valid)
    #                     loss_GAN_real = criterion_GAN(self.D_real(fake_imp), valid)
    #                     loss_GAN = (loss_GAN_style + loss_GAN_real) / 2
    #
    #                     # Cycle loss
    #                     recov_real = self.G_imp2real(fake_imp)
    #                     loss_cycle_real = criterion_cycle(recov_real, real_img)
    #                     fake_real = self.G_imp2real(real_img)
    #                     recov_imp = self.G_real2imp(fake_real)
    #                     loss_cycle_imp = criterion_cycle(recov_imp, real_img)
    #                     loss_cycle = (loss_cycle_real + loss_cycle_imp) / 2
    #
    #                     # Total loss
    #                     loss_G = (loss_GAN +
    #                               self.config['training']['lambda_cycle'] * loss_cycle +
    #                               self.config['training']['lambda_identity'] * loss_identity)
    #
    #                     loss_G.backward()
    #                     optimizer_G.step()
    #
    #                     epoch_loss += loss_G.item()
    #
    #                     if i % 100 == 0:
    #                         self.logger.info(
    #                             f"[Epoch {epoch}/{self.config['training']['n_epochs']}] "
    #                             f"[Batch {i}/{len(dataloader)}] "
    #                             f"[G loss: {loss_G.item():.4f}] "
    #                             f"[Adv loss: {loss_GAN.item():.4f}] "
    #                             f"[Cycle loss: {loss_cycle.item():.4f}] "
    #                             f"[Identity loss: {loss_identity.item():.4f}]"
    #                         )
    #
    #                     if i % self.config['training']['sample_interval'] == 0:
    #                         with torch.no_grad():
    #                             val_loss = self.validate_batch(real_img)
    #
    #                 except RuntimeError as e:
    #                     if "out of memory" in str(e) and self.adjust_batch_size():
    #                         continue
    #                     raise
    #                 finally:
    #                     # Cleanup
    #                     if 'real_img' in locals():
    #                         del real_img
    #                     if 'fake_imp' in locals():
    #                         del fake_imp
    #                     if 'fake_real' in locals():
    #                         del fake_real
    #                     torch.cuda.empty_cache()
    #
    #             # End of epoch processing
    #             avg_epoch_loss = epoch_loss / len(dataloader)
    #             scheduler.step(avg_epoch_loss)
    #
    #             if avg_epoch_loss < best_loss:
    #                 best_loss = avg_epoch_loss
    #                 no_improve_count = 0
    #                 self.save_models('best', epoch, best_loss)
    #             else:
    #                 no_improve_count += 1
    #                 if no_improve_count >= patience:
    #                     self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
    #                     break
    #
    #             if (epoch + 1) % 10 == 0:
    #                 self.save_models('checkpoint', epoch)
    #
    #         self.logger.info(f"Training completed! Best loss: {best_loss:.4f}")
    #
    #     except Exception as e:
    #         self.logger.error(f"Training failed: {str(e)}")
    #         raise

    def train(self):
        try:
            # Split dataset into train and validation
            full_dataset = ImageDataset(self.config['paths']['dataset_path'],
                                        self.transforms)
            train_size = int(0.9 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size])

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=True,
                num_workers=4
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=False,
                num_workers=4
            )

            # Loss functions
            criterion_GAN = nn.MSELoss()
            criterion_cycle = nn.L1Loss()
            criterion_identity = nn.L1Loss()

            if self.device == 'cuda':
                criterion_GAN.cuda()
                criterion_cycle.cuda()
                criterion_identity.cuda()

            # Optimizers with gradient clipping
            optimizer_G = torch.optim.Adam(
                itertools.chain(self.G_real2imp.parameters(),
                                self.G_imp2real.parameters()),
                lr=self.config['training']['learning_rate'],
                betas=(self.config['training']['beta1'],
                       self.config['training']['beta2'])
            )

            # Scheduler and tracking variables
            scheduler = ReduceLROnPlateau(
                optimizer_G, mode='min', factor=0.5, patience=2, verbose=True
            )

            best_loss = float('inf')
            patience = 5
            no_improve_count = 0
            max_grad_norm = 5.0  # For gradient clipping

            self.logger.info("Starting training...")

            for epoch in range(self.config['training']['n_epochs']):
                # Training phase
                epoch_loss = self.train_epoch(train_loader, optimizer_G,
                                              criterion_GAN, criterion_cycle,
                                              criterion_identity, epoch,
                                              max_grad_norm)

                # Validation phase
                val_loss = self.validate(val_loader)

                # Scheduler step with validation loss
                scheduler.step(val_loss)

                # Save best model
                if val_loss < best_loss:
                    best_loss = val_loss
                    no_improve_count = 0
                    self.save_models('best', epoch, val_loss)
                else:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                        break

                if (epoch + 1) % 10 == 0:
                    self.save_models('checkpoint', epoch)

                self.logger.info(
                    f"Epoch {epoch} - Train Loss: {epoch_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Best Loss: {best_loss:.4f}"
                )

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise

    # def train_epoch(self, train_loader, optimizer_G, criterion_GAN,
    #                 criterion_cycle, criterion_identity, epoch, max_grad_norm):
    #     """Single training epoch with monitoring"""
    #     epoch_loss = 0.0
    #     self.G_real2imp.train()
    #     self.G_imp2real.train()
    #
    #     for i, batch in enumerate(train_loader):
    #         try:
    #             real_img = batch["real"].to(self.device)
    #             valid = torch.ones((real_img.size(0), 1, 1, 1)).to(self.device)
    #             fake = torch.zeros((real_img.size(0), 1, 1, 1)).to(self.device)
    #
    #             optimizer_G.zero_grad()
    #
    #             # Generate fake image
    #             fake_imp = self.G_real2imp(real_img)
    #
    #             # Monitor color distribution
    #             if i % 100 == 0:
    #                 with torch.no_grad():
    #                     r_mean = fake_imp[:, 0].mean()
    #                     g_mean = fake_imp[:, 1].mean()
    #                     b_mean = fake_imp[:, 2].mean()
    #                     self.logger.info(
    #                         f"RGB means - R: {r_mean:.4f}, G: {g_mean:.4f}, B: {b_mean:.4f}"
    #                     )
    #                     self.save_sample_images(real_img, fake_imp, epoch, i)
    #
    #             # Identity loss
    #             loss_id_real = criterion_identity(self.G_imp2real(real_img), real_img)
    #             loss_id_imp = criterion_identity(self.G_real2imp(real_img), real_img)
    #             loss_identity = (loss_id_real + loss_id_imp) / 2
    #
    #             # GAN loss
    #             loss_GAN_style = criterion_GAN(self.D_style(fake_imp), valid)
    #             loss_GAN_real = criterion_GAN(self.D_real(fake_imp), valid)
    #             loss_GAN = (loss_GAN_style + loss_GAN_real) / 2
    #
    #             # Cycle loss
    #             recov_real = self.G_imp2real(fake_imp)
    #             loss_cycle_real = criterion_cycle(recov_real, real_img)
    #             fake_real = self.G_imp2real(real_img)
    #             recov_imp = self.G_real2imp(fake_real)
    #             loss_cycle_imp = criterion_cycle(recov_imp, real_img)
    #             loss_cycle = (loss_cycle_real + loss_cycle_imp) / 2
    #
    #             # Color balance loss
    #             loss_color = self.color_balance_loss(fake_imp)
    #
    #             # Total loss
    #             loss_G = (loss_GAN +
    #                       self.config['training']['lambda_cycle'] * loss_cycle +
    #                       self.config['training']['lambda_identity'] * loss_identity +
    #                       0.1 * loss_color)  # Added color balance loss with small weight
    #
    #             loss_G.backward()
    #
    #             # Gradient clipping
    #             torch.nn.utils.clip_grad_norm_(
    #                 itertools.chain(self.G_real2imp.parameters(),
    #                                 self.G_imp2real.parameters()),
    #                 max_grad_norm
    #             )
    #
    #             optimizer_G.step()
    #
    #             epoch_loss += loss_G.item()
    #
    #             if i % 100 == 0:
    #                 self.logger.info(
    #                     f"[Epoch {epoch}][Batch {i}/{len(train_loader)}] "
    #                     f"[G loss: {loss_G.item():.4f}] "
    #                     f"[Adv loss: {loss_GAN.item():.4f}] "
    #                     f"[Cycle loss: {loss_cycle.item():.4f}] "
    #                     f"[Identity loss: {loss_identity.item():.4f}] "
    #                     f"[Color loss: {loss_color.item():.4f}]"
    #                 )
    #
    #         except RuntimeError as e:
    #             if "out of memory" in str(e) and self.adjust_batch_size():
    #                 continue
    #             raise
    #         finally:
    #             # Cleanup
    #             if 'real_img' in locals():
    #                 del real_img
    #             if 'fake_imp' in locals():
    #                 del fake_imp
    #             if 'fake_real' in locals():
    #                 del fake_real
    #             torch.cuda.empty_cache()
    #
    #     return epoch_loss / len(train_loader)

    def train_epoch(self, train_loader, optimizer_G, criterion_GAN,
                    criterion_cycle, criterion_identity, epoch, max_grad_norm):
        """Updated train_epoch with safer color loss calculation"""
        epoch_loss = 0.0
        self.G_real2imp.train()
        self.G_imp2real.train()

        for i, batch in enumerate(train_loader):
            try:
                real_img = batch["real"].to(self.device)
                valid = torch.ones((real_img.size(0), 1, 1, 1)).to(self.device)
                fake = torch.zeros((real_img.size(0), 1, 1, 1)).to(self.device)

                optimizer_G.zero_grad()

                # Generate fake image
                fake_imp = self.G_real2imp(real_img)

                # Monitor color distribution
                if i % 100 == 0:
                    with torch.no_grad():
                        r_mean = fake_imp[:, 0].mean()
                        g_mean = fake_imp[:, 1].mean()
                        b_mean = fake_imp[:, 2].mean()
                        r_std = fake_imp[:, 0].std()
                        g_std = fake_imp[:, 1].std()
                        b_std = fake_imp[:, 2].std()
                        self.logger.info(
                            f"RGB means - R: {r_mean:.4f}, G: {g_mean:.4f}, B: {b_mean:.4f}"
                        )
                        self.logger.info(
                            f"RGB std - R: {r_std:.4f}, G: {g_std:.4f}, B: {b_std:.4f}"
                        )
                        self.save_sample_images(real_img, fake_imp, epoch, i)

                # Identity loss
                loss_id_real = criterion_identity(self.G_imp2real(real_img), real_img)
                loss_id_imp = criterion_identity(self.G_real2imp(real_img), real_img)
                loss_identity = (loss_id_real + loss_id_imp) / 2

                # GAN loss
                loss_GAN_style = criterion_GAN(self.D_style(fake_imp), valid)
                loss_GAN_real = criterion_GAN(self.D_real(fake_imp), valid)
                loss_GAN = (loss_GAN_style + loss_GAN_real) / 2

                # Cycle loss
                recov_real = self.G_imp2real(fake_imp)
                loss_cycle_real = criterion_cycle(recov_real, real_img)
                fake_real = self.G_imp2real(real_img)
                recov_imp = self.G_real2imp(fake_real)
                loss_cycle_imp = criterion_cycle(recov_imp, real_img)
                loss_cycle = (loss_cycle_real + loss_cycle_imp) / 2

                # Color balance loss with error handling
                try:
                    loss_color = self.color_balance_loss(fake_imp)
                except:
                    # Fallback to simple color balance if patch-based fails
                    loss_color = torch.tensor(0.0, device=self.device)
                    self.logger.warning("Fallback to simple color balance")

                # Total loss
                loss_G = (loss_GAN +
                          self.config['training']['lambda_cycle'] * loss_cycle +
                          self.config['training']['lambda_identity'] * loss_identity +
                          self.config['training']['lambda_color'] * loss_color)

                loss_G.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    itertools.chain(self.G_real2imp.parameters(),
                                    self.G_imp2real.parameters()),
                    max_grad_norm
                )

                optimizer_G.step()

                epoch_loss += loss_G.item()

                if i % 100 == 0:
                    self.logger.info(
                        f"[Epoch {epoch}][Batch {i}/{len(train_loader)}] "
                        f"[G loss: {loss_G.item():.4f}] "
                        f"[Adv loss: {loss_GAN.item():.4f}] "
                        f"[Cycle loss: {loss_cycle.item():.4f}] "
                        f"[Identity loss: {loss_identity.item():.4f}] "
                        f"[Color loss: {loss_color.item():.4f}]"
                    )

            except RuntimeError as e:
                if "out of memory" in str(e) and self.adjust_batch_size():
                    continue
                raise
            finally:
                # Cleanup
                if 'real_img' in locals():
                    del real_img
                if 'fake_imp' in locals():
                    del fake_imp
                if 'fake_real' in locals():
                    del fake_real
                torch.cuda.empty_cache()

        return epoch_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validation step"""
        self.G_real2imp.eval()
        self.G_imp2real.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                real_img = batch["real"].to(self.device)
                fake_imp = self.G_real2imp(real_img)
                loss_style = self.D_style(fake_imp).mean()
                loss_real = self.D_real(fake_imp).mean()
                val_loss += (loss_style + loss_real) / 2

        return val_loss / len(val_loader)

    def save_models(self, save_type, epoch, loss=None):
        try:
            save_dir = (self.config['paths']['best_models_dir'] if save_type == 'best'
                        else self.config['paths']['checkpoint_dir'])

            save_dict = {
                'epoch': epoch,
                'G_real2imp_state_dict': self.G_real2imp.state_dict(),
                'G_imp2real_state_dict': self.G_imp2real.state_dict(),
            }

            if loss is not None:
                save_dict['loss'] = loss

            filename = (f"best_models.pth" if save_type == 'best'
                        else f"checkpoint_epoch_{epoch}.pth")

            torch.save(save_dict, os.path.join(save_dir, filename))
            self.logger.info(f"Saved {save_type} models at epoch {epoch}")

        except Exception as e:
            self.logger.error(f"Failed to save models: {str(e)}")
            raise

    def load_models(self, checkpoint_path):
        try:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            self.G_real2imp.load_state_dict(checkpoint['G_real2imp_state_dict'])
            self.G_imp2real.load_state_dict(checkpoint['G_imp2real_state_dict'])

            self.logger.info(f"Loaded models from {checkpoint_path}")
            return checkpoint.get('epoch', 0), checkpoint.get('loss', float('inf'))

        except Exception as e:
            self.logger.error(f"Failed to load models: {str(e)}")
            raise

    def convert_to_impressionist(self, image_path, output_path=None):
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Input image not found: {image_path}")

            # Load and convert image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transforms(image)

            # Apply preprocessing
            image_tensor = self.preprocess_image(image_tensor)
            image_tensor = image_tensor.unsqueeze(0)

            # Debug print
            print(f"After preprocessing - Min: {image_tensor.min():.4f}, Max: {image_tensor.max():.4f}")

            if self.device == 'cuda':
                image_tensor = image_tensor.cuda()

            self.G_real2imp.eval()
            with torch.no_grad():
                # Generate the image
                generated_image = self.G_real2imp(image_tensor)

                # Debug print
                print(f"After generation - Min: {generated_image.min():.4f}, Max: {generated_image.max():.4f}")

                # Apply postprocessing
                generated_image = self.postprocess_image(generated_image)

                # Debug print
                print(f"After postprocessing - Min: {generated_image.min():.4f}, Max: {generated_image.max():.4f}")

            if output_path is None:
                output_path = f"impressionist_{Path(image_path).name}"

            # Save the image
            save_image(generated_image, output_path)

            return output_path

        except Exception as e:
            self.logger.error(f"Failed to convert image: {str(e)}")
            raise

    def preprocess_image(self, image_tensor):
        """
        Ensures proper preprocessing of input images
        """
        # Ensure input is in [0, 1]
        if image_tensor.min() < 0 or image_tensor.max() > 1:
            image_tensor = torch.clamp(image_tensor, 0, 1)

        # Convert to [-1, 1]
        image_tensor = 2 * image_tensor - 1
        return image_tensor

    def postprocess_image(self, image_tensor):
        # Current normalization
        image_tensor = torch.clamp(image_tensor, -1, 1)
        image_tensor = (image_tensor + 1) / 2.0

        # Stretch the histogram to use full range
        batch_min = image_tensor.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        batch_max = image_tensor.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        image_tensor = (image_tensor - batch_min) / (batch_max - batch_min)

        return image_tensor.clamp(0, 1)


def main():
    parser = argparse.ArgumentParser(description='Style Transfer Pipeline')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'convert'],
                        help='Operation mode')
    parser.add_argument('--input', type=str,
                        help='Input image path for conversion')
    parser.add_argument('--output', type=str,
                        help='Output image path for conversion')
    parser.add_argument('--checkpoint', type=str,
                        help='Checkpoint path for loading models')

    args = parser.parse_args()

    # Initialize the StyleTransfer class
    style_transfer = StyleTransfer()

    if args.mode == 'train':
        # Load discriminators and start training
        style_transfer.load_pretrained_discriminators()
        style_transfer.train()
    elif args.mode == 'convert':
        if not args.input:
            raise ValueError("Input image path required for conversion mode")
        if args.checkpoint:
            style_transfer.load_models(args.checkpoint)
        style_transfer.convert_to_impressionist(args.input, args.output)

if __name__ == "__main__":
    main()