import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torchvision


class Discriminator(nn.Module):
    def __init__(self, width=64, dropout_rate=0.5):
        super(Discriminator, self).__init__()
        self.width = width
        self.dropout_rate = dropout_rate
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        self.conv_blocks = nn.ModuleList([
            self._make_conv_block(3, self.width, normalize=False),
            self._make_conv_block(self.width, self.width * 2),
            self._make_conv_block(self.width * 2, self.width * 4),
            self._make_conv_block(self.width * 4, self.width * 8)
        ])

        self.dropout = nn.Dropout(dropout_rate)
        self.global_pool = nn.AdaptiveAvgPool2d((12, 12))
        self.fc1 = nn.Linear(self.width * 8 * 12 * 12, 100)
        self.fc2 = nn.Linear(100, 1)

    def _make_conv_block(self, in_channels, out_channels, normalize=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not normalize),
            self.activation
        ]
        if normalize:
            layers.insert(1, nn.InstanceNorm2d(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)

        x = self.global_pool(x)
        x = x.view(-1, self.width * 8 * 12 * 12)

        x = self.dropout(x)
        x = self.fc1(x)
        x = self.activation(x)

        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DiscriminatorTrainer:
    def __init__(self, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.device = device

    def train_model(self, model, train_loader, val_loader, config):
        try:
            model = model.to(self.device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(),
                                   lr=config['learning_rate'],
                                   betas=(0.5, 0.999),
                                   weight_decay=config['weight_decay'])

            history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

            for epoch in range(config['num_epochs']):
                model.train()
                total_loss = 0
                correct = 0
                total = 0

                for imgs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
                    try:
                        imgs, labels = imgs.to(self.device), labels.to(self.device)

                        optimizer.zero_grad()
                        outputs = model(imgs)
                        loss = criterion(outputs.squeeze(), labels.float())
                        loss.backward()

                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                        total_loss += loss.item()
                        predicted = (outputs > 0).float()
                        correct += (predicted.squeeze() == labels).sum().item()
                        total += labels.size(0)

                    finally:
                        # Clear intermediate variables
                        del imgs, labels, outputs, loss, predicted
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                train_loss = total_loss / len(train_loader)
                train_acc = correct / total
                val_acc = self.get_accuracy(model, val_loader)

                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_acc'].append(val_acc)

                print(f"Epoch {epoch + 1}: Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

            return model, history

        except Exception as e:
            print(f"Error during training: {e}")
            raise e

    def get_accuracy(self, model, loader):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = model(imgs)
                predicted = (outputs > 0).float()
                correct += (predicted.squeeze() == labels).sum().item()
                total += labels.size(0)

        return correct / total


def grid_search(train_loader, val_loader):
    param_grid = {
        'width': [32, 64, 128, 256],  # Base width of the network
        'batch_size': [16, 32, 64, 128],  # Common batch sizes for training
        'learning_rate': [0.0001, 0.0002, 0.0004],  # Standard GAN learning rates
        'weight_decay': [0, 1e-6, 1e-5, 1e-4],  # L2 regularization values
        'dropout_rate': [0.2, 0.3, 0.4, 0.5],  # Dropout probabilities
        'num_epochs': [32]  # Training duration options
    }

    trainer = DiscriminatorTrainer()
    results = []

    for params in tqdm(ParameterGrid(param_grid), desc="Grid Search"):
        print(f"\nTrying parameters: {params}")

        # Create model and move to GPU
        model = Discriminator(
            width=params['width'],
            dropout_rate=params['dropout_rate']
        )

        try:
            trained_model, history = trainer.train_model(
                model, train_loader, val_loader, params
            )

            final_val_acc = history['val_acc'][-1]
            results.append({
                'params': params,
                'final_val_acc': final_val_acc,
                'history': history
            })

            print(f"Final validation accuracy: {final_val_acc:.4f}")

        except Exception as e:
            print(f"Error during training: {e}")

        finally:
            # Clean up CUDA memory
            if torch.cuda.is_available():
                # Delete model and clear cache
                del model
                if 'trained_model' in locals():
                    del trained_model
                torch.cuda.empty_cache()

                # Force garbage collection
                import gc
                gc.collect()

                # Print memory stats for debugging (optional)
                print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                print(f"CUDA memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    best_result = max(results, key=lambda x: x['final_val_acc'])
    print("\nBest parameters:")
    print(best_result['params'])
    print(f"Best validation accuracy: {best_result['final_val_acc']:.4f}")

    return best_result


if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Your existing data loading code
    dataset_dir = os.path.join("../", 'util', 'old-datasets')
    print("Data Directory:", dataset_dir)

    real_vs_fake_dir = os.path.join(dataset_dir, 'real_vs_fake')

    # Add normalization to transform
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_data = torchvision.datasets.ImageFolder(os.path.join(real_vs_fake_dir, "train"), transform=transform)
    val_data = torchvision.datasets.ImageFolder(os.path.join(real_vs_fake_dir, "val"), transform=transform)
    test_data = torchvision.datasets.ImageFolder(os.path.join(real_vs_fake_dir, "test"), transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4)

    # Perform grid search
    print("Starting grid search...")
    best_result = grid_search(train_loader, val_loader)

    # Train final model with best parameters
    print("\nTraining final model with best parameters...")
    final_model = Discriminator(
        width=best_result['params']['width'],
        dropout_rate=best_result['params']['dropout_rate']
    )

    trainer = DiscriminatorTrainer()
    final_model, history = trainer.train_model(
        final_model, train_loader, val_loader, best_result['params']
    )

    # Save the model
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'best_params': best_result['params'],
        'training_history': history,
    }, 'best_discriminator.pth')

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

    # Print final results
    print("\nTraining completed!")
    print(f"Best parameters: {best_result['params']}")
    print(f"Final training accuracy: {history['train_acc'][-1]:.4f}")
    print(f"Final validation accuracy: {history['val_acc'][-1]:.4f}")
