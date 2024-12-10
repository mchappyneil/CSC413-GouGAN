import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms
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
        'n_epochs': 1,
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

        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)  # Changed from LeakyReLU
        ]

        # Downsampling with increased channels
        channels_list = [64, 128, 256]
        for i in range(2):
            in_features = channels_list[i]
            out_features = channels_list[i + 1]
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)  # Changed from LeakyReLU
            ]

        # Residual blocks with attention
        self.attention = SelfAttention(256)
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(256)]

        # Upsampling with careful normalization
        channels_list = [256, 128, 64]
        for i in range(2):
            in_features = channels_list[i]
            out_features = channels_list[i + 1]
            model += [
                nn.Upsample(scale_factor=2, mode='nearest'),  # Changed to nearest neighbor
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)  # Changed from LeakyReLU
            ]

        # Output layer with careful padding
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, channels, 7),
            nn.Tanh()  # Keeps this as it ensures output range [-1, 1]
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        features = x
        attention_applied = False

        for layer in self.model:
            if isinstance(layer, ResidualBlock) and not attention_applied:
                # Apply attention with residual connection to better preserve information
                att_features = self.attention(features)
                features = features + 0.1 * att_features  # Reduced attention influence
                attention_applied = True
            features = layer(features)

            # Add extra checks to maintain color information
            if isinstance(layer, nn.InstanceNorm2d):
                features = features.clamp(-1, 1)  # Prevent extreme values

        return features


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

    def setup_transforms(self):
        self.transforms = transforms.Compose([
            transforms.Resize((self.config['model']['img_height'],
                               self.config['model']['img_width'])),
            transforms.ToTensor()  # Just convert to [0, 1] range
        ])

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

    def train(self):
        try:
            # Data loading
            dataset = ImageDataset(self.config['paths']['dataset_path'], self.transforms)
            dataloader = DataLoader(
                dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=True,
                num_workers=4
            )

            # Loss functions
            criterion_GAN = nn.BCEWithLogitsLoss()  # Changed from MSELoss for better gradients
            criterion_cycle = nn.L1Loss()
            criterion_identity = nn.L1Loss()

            if self.device == 'cuda':
                criterion_GAN.cuda()
                criterion_cycle.cuda()
                criterion_identity.cuda()

            # Optimizers
            optimizer_G = torch.optim.Adam(
                itertools.chain(self.G_real2imp.parameters(), self.G_imp2real.parameters()),
                lr=self.config['training']['learning_rate'],
                betas=(self.config['training']['beta1'], self.config['training']['beta2'])
            )

            # Learning rate scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_G, mode='min', factor=0.5, patience=2, verbose=True
            )

            # Training setup
            best_loss = float('inf')
            patience = 5
            no_improve_count = 0
            self.logger.info("Starting training...")

            for epoch in range(self.config['training']['n_epochs']):
                epoch_loss = 0.0
                self.G_real2imp.train()
                self.G_imp2real.train()

                for i, batch in enumerate(dataloader):
                    try:
                        real_img = batch["real"].to(self.device)

                        # Generate fake impressionist image
                        fake_imp = self.G_real2imp(real_img)

                        optimizer_G.zero_grad()

                        # Identity loss
                        loss_id_real = criterion_identity(self.G_imp2real(real_img), real_img)
                        loss_id_imp = criterion_identity(self.G_real2imp(real_img), real_img)
                        loss_identity = (loss_id_real + loss_id_imp) / 2

                        # Get discriminator scores
                        style_score = self.D_style(fake_imp)
                        real_score = self.D_real(fake_imp)

                        # Calculate adversarial losses with proper scaling
                        # We want high style scores (close to 1) and moderate real scores (around 0.5)
                        style_target = torch.ones_like(style_score).to(self.device)
                        real_target = 0.5 * torch.ones_like(real_score).to(self.device)

                        loss_GAN_style = criterion_GAN(style_score, style_target)
                        loss_GAN_real = criterion_GAN(real_score, real_target)

                        # Add penalty for very low scores
                        style_penalty = torch.mean(torch.relu(0.2 - style_score)) * 2.0
                        real_penalty = torch.mean(torch.relu(0.3 - real_score)) * 2.0

                        loss_GAN = loss_GAN_style + loss_GAN_real + style_penalty + real_penalty

                        # Cycle loss
                        recov_real = self.G_imp2real(fake_imp)
                        loss_cycle_real = criterion_cycle(recov_real, real_img)
                        fake_real = self.G_imp2real(real_img)
                        recov_imp = self.G_real2imp(fake_real)
                        loss_cycle_imp = criterion_cycle(recov_imp, real_img)
                        loss_cycle = (loss_cycle_real + loss_cycle_imp) / 2

                        # Total loss with adjusted weights
                        loss_G = (
                                2.0 * loss_GAN +  # Increased weight for GAN loss
                                self.config['training']['lambda_cycle'] * loss_cycle +
                                self.config['training']['lambda_identity'] * loss_identity
                        )

                        loss_G.backward()
                        optimizer_G.step()

                        # Log discriminator scores
                        if i % 100 == 0:
                            self.logger.info(
                                f"[Epoch {epoch}/{self.config['training']['n_epochs']}] "
                                f"[Batch {i}/{len(dataloader)}] "
                                f"[G loss: {loss_G.item():.4f}] "
                                f"[Style score: {style_score.mean():.4f}] "
                                f"[Real score: {real_score.mean():.4f}] "
                                f"[Cycle loss: {loss_cycle.item():.4f}]"
                            )

                        epoch_loss += loss_G.item()

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

                # End of epoch processing
                avg_epoch_loss = epoch_loss / len(dataloader)
                scheduler.step(avg_epoch_loss)

                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    no_improve_count = 0
                    self.save_models('best', epoch, best_loss)
                else:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                        break

                if (epoch + 1) % 10 == 0:
                    self.save_models('checkpoint', epoch)

            self.logger.info(f"Training completed! Best loss: {best_loss:.4f}")

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise

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