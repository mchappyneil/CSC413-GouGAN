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

torch.cuda.empty_cache()
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of available GPU memory

print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024 ** 2} MB")

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

# Configuration
CONFIG = {
    'training': {
        'n_epochs': 5,
        'batch_size': 4,
        'learning_rate': 0.0002,
        'beta1': 0.5,
        'beta2': 0.999,
        'lambda_cycle': 10.0,
        'lambda_identity': 5.0,
        'sample_interval': 100
    },
    'model': {
        'img_height': 224,
        'img_width': 224,
        'channels': 3,
        'num_residual_blocks': 9
    },
    'paths': {
        'dataset_path': "./util/datasets/real-fake-split/real",
        'checkpoint_dir': "./checkpoints",
        'sample_dir': "./samples",
        'best_models_dir': "./best_models",
        'log_dir': "./logs"
    }
}

# Data transforms
# transforms = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                        std=[0.229, 0.224, 0.225])
# ])

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(Generator, self).__init__()
        channels = input_shape[0]

        # Initial layer
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        in_features = 64
        for _ in range(2):
            out_features = in_features * 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features = in_features // 2
            model += [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features

        # Output layer
        model += [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(64, channels, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)



class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.files = [f for f in Path(root).glob('*') if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
        self.root = root

    def __getitem__(self, index):
        img_path = self.files[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return {'real': img, 'imp': img}

    def __len__(self):
        return len(self.files)

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.base_model = models.resnet50(weights=None)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

        self.output_shape = (1, 1, 1)

    def forward(self, x):
        features = self.base_model(x)
        output = self.classifier(features)
        return output.view(-1, 1)

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
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def load_pretrained_discriminators(self):
        try:
            # Clear cache before loading models
            torch.cuda.empty_cache()

            self.D_style = Discriminator(self.input_shape)
            self.D_real = Discriminator(self.input_shape)

            # Load one model at a time
            style_checkpoint = torch.load('discriminator_style.pth', map_location=self.device, weights_only=True)
            self.D_style.load_state_dict(style_checkpoint['model_state_dict'])
            del style_checkpoint
            torch.cuda.empty_cache()

            real_checkpoint = torch.load('discriminator_real-fake.pth', map_location=self.device)
            self.D_real.load_state_dict(real_checkpoint['model_state_dict'])
            del real_checkpoint
            torch.cuda.empty_cache()

            if self.device == 'cuda':
                self.D_style = self.D_style.cuda()
                torch.cuda.empty_cache()
                self.D_real = self.D_real.cuda()
                torch.cuda.empty_cache()

            self.D_style.eval()
            self.D_real.eval()

            self.logger.info("Successfully loaded pre-trained discriminators")

        except Exception as e:
            self.logger.error(f"Failed to load discriminators: {str(e)}")
            raise

    def train(self):
        try:
            # Data loading
            dataset = ImageDataset(self.config['paths']['dataset_path'],
                                   self.transforms)
            dataloader = DataLoader(
                dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=True,
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

            # Optimizers
            optimizer_G = torch.optim.Adam(
                itertools.chain(self.G_real2imp.parameters(),
                                self.G_imp2real.parameters()),
                lr=self.config['training']['learning_rate'],
                betas=(self.config['training']['beta1'],
                       self.config['training']['beta2'])
            )

            # Training
            best_loss = float('inf')
            self.logger.info("Starting training...")

            for epoch in range(self.config['training']['n_epochs']):
                epoch_loss = 0.0
                for i, batch in enumerate(dataloader):
                    real_img = Variable(
                        batch["real"].type(torch.cuda.FloatTensor if self.device == 'cuda' else torch.FloatTensor))
                    imp_img = Variable(
                        batch["imp"].type(torch.cuda.FloatTensor if self.device == 'cuda' else torch.FloatTensor))

                    valid = Variable(torch.ones((real_img.size(0), 1)).type(
                        torch.cuda.FloatTensor if self.device == 'cuda' else torch.FloatTensor))
                    fake = Variable(torch.zeros((real_img.size(0), 1)).type(
                        torch.cuda.FloatTensor if self.device == 'cuda' else torch.FloatTensor))

                    # Train Generators
                    self.G_real2imp.train()
                    self.G_imp2real.train()
                    optimizer_G.zero_grad()

                    # Identity loss
                    loss_id_real = criterion_identity(self.G_imp2real(real_img), real_img)
                    loss_id_imp = criterion_identity(self.G_real2imp(imp_img), imp_img)
                    loss_identity = (loss_id_real + loss_id_imp) / 2

                    # GAN loss
                    fake_imp = self.G_real2imp(real_img)
                    loss_GAN_style = criterion_GAN(self.D_style(fake_imp), valid)
                    loss_GAN_real = criterion_GAN(self.D_real(fake_imp), valid)

                    fake_real = self.G_imp2real(imp_img)
                    loss_GAN = (loss_GAN_style + loss_GAN_real) / 2

                    # Cycle loss
                    recov_real = self.G_imp2real(fake_imp)
                    loss_cycle_real = criterion_cycle(recov_real, real_img)
                    recov_imp = self.G_real2imp(fake_real)
                    loss_cycle_imp = criterion_cycle(recov_imp, imp_img)
                    loss_cycle = (loss_cycle_real + loss_cycle_imp) / 2

                    # Total loss
                    loss_G = (loss_GAN +
                              self.config['training']['lambda_cycle'] * loss_cycle +
                              self.config['training']['lambda_identity'] * loss_identity)

                    loss_G.backward()
                    optimizer_G.step()

                    epoch_loss += loss_G.item()

                    if i % 100 == 0:
                        self.logger.info(
                            f"[Epoch {epoch}/{self.config['training']['n_epochs']}] "
                            f"[Batch {i}/{len(dataloader)}] "
                            f"[G loss: {loss_G.item():.4f}] "
                            f"[Adv loss: {loss_GAN.item():.4f}] "
                            f"[Cycle loss: {loss_cycle.item():.4f}] "
                            f"[Identity loss: {loss_identity.item():.4f}]"
                        )

                # Save best models
                avg_epoch_loss = epoch_loss / len(dataloader)
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    self.save_models('best', epoch, best_loss)

                # Regular checkpoint saving
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
            checkpoint = torch.load(checkpoint_path,
                                    map_location=self.device)

            self.G_real2imp.load_state_dict(checkpoint['G_real2imp_state_dict'])
            self.G_imp2real.load_state_dict(checkpoint['G_imp2real_state_dict'])

            self.logger.info(f"Loaded models from {checkpoint_path}")
            return checkpoint.get('epoch', 0), checkpoint.get('loss', float('inf'))

        except Exception as e:
            self.logger.error(f"Failed to load models: {str(e)}")
            raise

    def convert_to_impressionist(self, image_path, output_path=None):
        try:
            # Prepare image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transforms(image).unsqueeze(0)

            if self.device == 'cuda':
                image_tensor = image_tensor.cuda()

            # Generate impressionist style image
            self.G_real2imp.eval()
            with torch.no_grad():
                generated_image = self.G_real2imp(image_tensor)

            # Save the result
            if output_path is None:
                output_path = f"impressionist_{Path(image_path).name}"

            save_image(generated_image * 0.5 + 0.5, output_path)
            self.logger.info(f"Saved generated image to {output_path}")

            return output_path

        except Exception as e:
            self.logger.error(f"Failed to convert image: {str(e)}")
            raise


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
