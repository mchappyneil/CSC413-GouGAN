import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torchvision
import torchvision.models as models
import gc
import time


class PretrainedDiscriminator(nn.Module):
    def __init__(self, input_shape, pretrained=True):
        super(PretrainedDiscriminator, self).__init__()

        # Load pre-trained ResNet
        self.base_model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)

        # Remove the final fully connected layer
        num_features = self.base_model.fc.in_features

        # Replace the final layers
        self.base_model.fc = nn.Identity()  # Remove original FC layer

        # Add custom layers for the discriminator
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

        # Calculate output shape (now it's just a single value per image)
        self.output_shape = (1, 1, 1)

    def forward(self, x):
        features = self.base_model(x)
        output = self.classifier(features)
        return output.view(-1, *self.output_shape)


class PretrainedDiscriminatorTrainer:
    def __init__(self, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.device = device

    def train_model(self, model, train_loader, val_loader, config):
        model = model.to(self.device)
        criterion = nn.BCEWithLogitsLoss()

        # Use different learning rates for pre-trained layers and new layers
        base_params = list(model.base_model.parameters())
        classifier_params = list(model.classifier.parameters())

        optimizer = torch.optim.Adam([
            {'params': base_params, 'lr': config['learning_rate'] * 0.1},
            {'params': classifier_params, 'lr': config['learning_rate']}
        ], betas=(0.5, 0.999), weight_decay=config['weight_decay'])

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, verbose=True
        )

        history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

        for epoch in range(config['num_epochs']):
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            for imgs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
                try:
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    batch_size = imgs.size(0)

                    valid = torch.ones((batch_size, *model.output_shape), requires_grad=False).to(self.device)
                    fake = torch.zeros((batch_size, *model.output_shape), requires_grad=False).to(self.device)

                    optimizer.zero_grad()
                    outputs = model(imgs)

                    targets = valid * labels.view(-1, 1, 1, 1) + fake * (1 - labels.view(-1, 1, 1, 1))
                    loss = criterion(outputs, targets)
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    total_loss += loss.item()
                    predictions = (outputs > 0).float()
                    correct += ((predictions.squeeze() == labels).sum().item())
                    total += labels.size(0)

                except Exception as e:
                    print(f"Error in training batch: {e}")
                    continue

                finally:
                    del imgs, labels, outputs, loss, predictions, valid, fake
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            train_loss = total_loss / len(train_loader)
            train_acc = correct / total
            val_acc = self.get_accuracy(model, val_loader)

            scheduler.step(val_acc)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            print(f"Epoch {epoch + 1}: Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

        return model, history

    def get_accuracy(self, model, loader):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = model(imgs)
                predictions = (outputs > 0).float()
                correct += ((predictions.squeeze() == labels).sum().item())
                total += labels.size(0)

                del imgs, labels, outputs, predictions
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return correct / total


def grid_search(train_loader, val_loader, input_shape):
    param_grid = {
        'batch_size': [64],
        'learning_rate': [0.0003],
        'weight_decay': [1e-4],
        'num_epochs': [20]
    }

    trainer = PretrainedDiscriminatorTrainer()
    results = []

    for params in tqdm(ParameterGrid(param_grid), desc="Grid Search"):
        print(f"\nTrying parameters: {params}")

        model = PretrainedDiscriminator(input_shape)

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
            continue

        finally:
            if torch.cuda.is_available():
                del model
                if 'trained_model' in locals():
                    del trained_model
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(1)

    if results:
        best_result = max(results, key=lambda x: x['final_val_acc'])
        print("\nBest parameters:")
        print(best_result['params'])
        print(f"Best validation accuracy: {best_result['final_val_acc']:.4f}")
        return best_result
    else:
        raise Exception("No successful training runs completed")


if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Data loading code
    dataset_dir = os.path.join("./", 'util', 'datasets')
    print("Data Directory:", dataset_dir)

    real_vs_fake_dir = os.path.join(dataset_dir, 'style_balanced')

    # Add normalization to match ImageNet pre-training
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),  # ResNet expects 224x224 images
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet normalization
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_data = torchvision.datasets.ImageFolder(os.path.join(real_vs_fake_dir, "train"), transform=transform)
    val_data = torchvision.datasets.ImageFolder(os.path.join(real_vs_fake_dir, "val"), transform=transform)
    test_data = torchvision.datasets.ImageFolder(os.path.join(real_vs_fake_dir, "test"), transform=transform)

    # Get input shape from a sample image
    sample_imgs, _ = next(iter(DataLoader(train_data, batch_size=1)))
    input_shape = sample_imgs[0].shape

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4)

    # Perform grid search
    print("Starting grid search...")
    best_result = grid_search(train_loader, val_loader, input_shape)

    # Train final model with best parameters
    print("\nTraining final model with best parameters...")
    final_model = PretrainedDiscriminator(input_shape)
    trainer = PretrainedDiscriminatorTrainer()
    final_model, history = trainer.train_model(
        final_model, train_loader, val_loader, best_result['params']
    )

    # Save the model
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'best_params': best_result['params'],
        'training_history': history,
    }, 'discriminator_real-fake.pth')

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
    plt.savefig('training_history_pretrained.png')
    plt.show()

    # Print final results
    print("\nTraining completed!")
    print(f"Best parameters: {best_result['params']}")
    print(f"Final training accuracy: {history['train_acc'][-1]:.4f}")
    print(f"Final validation accuracy: {history['val_acc'][-1]:.4f}")