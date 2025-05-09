import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from unet import UNet
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

class SARSegmentationDataset(Dataset):
    def __init__(self, sar_dir, mask_dir, transform=None):
        self.sar_dir = sar_dir
        self.mask_dir = mask_dir
        self.file_list = sorted([f for f in os.listdir(sar_dir) if os.path.isfile(os.path.join(sar_dir, f))])
        self.transform = transform
        
    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, idx):
        sar_path = os.path.join(self.sar_dir, self.file_list[idx])
        mask_path = os.path.join(self.mask_dir, self.file_list[idx])
        
        # Load images
        sar_img = Image.open(sar_path).convert("L")
        mask_img = Image.open(mask_path).convert("L")
        
        # Convert to numpy arrays
        sar = np.array(sar_img)
        mask = np.array(mask_img) // 255  # binarize
        
        # Apply transformations if provided
        if self.transform:
            augmented = self.transform(image=sar, mask=mask)
            sar = augmented['image']
            mask = augmented['mask']
        else:
            # If no transform, manually convert to tensor and normalize
            sar = torch.tensor(sar, dtype=torch.float32).unsqueeze(0) / 255.0
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            
        return sar, mask

def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=0.5, std=0.5),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=0.5, std=0.5),
            ToTensorV2()
        ])

def dice_loss(pred, target, smooth=1e-5):
    # Check tensor dimensions and reshape if needed
    if len(pred.shape) == 2:
        pred = pred.unsqueeze(0)  # Add batch dimension
    if len(target.shape) == 2:
        target = target.unsqueeze(0)  # Add batch dimension
        
    # Make sure both tensors have the same shape
    if pred.shape != target.shape:
        # If pred has only one channel but target has multiple or vice versa
        if pred.shape[1] != target.shape[1]:
            if pred.shape[1] == 1:
                pred = pred.expand(-1, target.shape[1], -1, -1)
            elif target.shape[1] == 1:
                target = target.expand(-1, pred.shape[1], -1, -1)
    
    # Get the dimensions based on the actual tensor shape
    batch_dim = 0
    if len(pred.shape) == 4:  # B, C, H, W
        dims = (2, 3)
    elif len(pred.shape) == 3:  # B, H, W (single channel)
        dims = (1, 2)
    else:  # fallback, use the last two dimensions
        dims = tuple(range(1, len(pred.shape)))
    
    pred = pred.contiguous()
    target = target.contiguous()
    
    intersection = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims)
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return (1 - dice.mean())
    
def dice_coefficient(pred, target, threshold=0.5, smooth=1e-5):
    # Apply threshold to prediction
    pred = (pred > threshold).float()
    
    if target.dim() == 3:  # [B, H, W]
        target = target.unsqueeze(1)  # Make it [B, 1, H, W]    
    
    # Make sure shapes match
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")
    
    # Get the dimensions dynamically
    dims = tuple(range(1, pred.dim()))  # Sum over all but batch dimension
    
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims)
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return dice.mean()


def dice_coefficient_prev(pred, target, threshold=0.5, smooth=1e-5):
    # Apply threshold to prediction
    pred = (pred > threshold).float()
    
    # Check tensor dimensions and reshape if needed
    if len(pred.shape) == 2:
        pred = pred.unsqueeze(0)  # Add batch dimension
    if len(target.shape) == 2:
        target = target.unsqueeze(0)  # Add batch dimension
    
    # Get the dimensions based on the actual tensor shape
    batch_dim = 0
    if len(pred.shape) == 4:  # B, C, H, W
        dims = (2, 3)
    elif len(pred.shape) == 3:  # B, H, W (single channel)
        dims = (1, 2)
    else:  # fallback, use the last two dimensions
        dims = tuple(range(1, len(pred.shape)))
    
    pred = pred.contiguous()
    target = target.contiguous()
    
    intersection = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims)
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return dice.mean()

def combined_loss(pred, target):
    # Ensure both tensors have the same shape
    if pred.shape != target.shape:
        target = target.view(pred.shape)
    
    # Make sure pred has values in valid range for BCE
    pred = torch.clamp(pred, min=1e-7, max=1-1e-7)
    
    return dice_loss(pred, target) + F.binary_cross_entropy(pred, target.float())

def train(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    
    for sar, mask in tqdm(loader, desc="Training"):
        sar, mask = sar.to(device), mask.to(device)
        
        # Debug print for shapes
        # print(f"SAR shape: {sar.shape}, Mask shape: {mask.shape}")
        
        optimizer.zero_grad()
        pred = model(sar)
        
        # Debug print for prediction shape
        # print(f"Prediction shape: {pred.shape}")
        
        loss = criterion(pred, mask)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    dice_scores = []
    
    with torch.no_grad():
        for sar, mask in tqdm(loader, desc="Validation"):
            sar, mask = sar.to(device), mask.to(device)
            
            pred = model(sar)
            loss = criterion(pred, mask)
            val_loss += loss.item()
            
            # Calculate Dice coefficient for accuracy
            dice = dice_coefficient(pred, mask)
            dice_scores.append(dice.item())
    
    avg_dice = np.mean(dice_scores)
    return val_loss / len(loader), avg_dice

def main():
    # Print debug information
    print("Starting training script...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    base_dir = '/home/dell/Downloads/Souraja'
    sar_train_dir = os.path.join(base_dir, 'SAR', 'train')
    mask_train_dir = os.path.join(base_dir, 'Masks', 'train')
    sar_val_dir = os.path.join(base_dir, 'SAR', 'val')
    mask_val_dir = os.path.join(base_dir, 'Masks', 'val')
    
    # Create datasets
    print("Creating datasets...")
    print(f"Training data directory: {sar_train_dir}")
    print(f"Number of files in train directory: {len(os.listdir(sar_train_dir))}")
    
    train_dataset = SARSegmentationDataset(
        sar_train_dir, 
        mask_train_dir, 
        transform=get_transforms(is_train=True)
    )
    val_dataset = SARSegmentationDataset(
        sar_val_dir,
        mask_val_dir,
        transform=get_transforms(is_train=False)
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Test loading a single sample
    if len(train_dataset) > 0:
        sample_sar, sample_mask = train_dataset[0]
        print(f"Sample SAR shape: {sample_sar.shape}")
        print(f"Sample mask shape: {sample_mask.shape}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=4, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create model, optimizer and criterion
    print("Creating model...")
    model = UNet(in_channels=1, out_channels=1).to(device)
    model_path = 'unet_epoch_20.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
    criterion = combined_loss
    
    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    num_epochs = 50
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        # Train
        train_loss = train(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Dice Accuracy: {val_accuracy:.4f}")
        
        # Save model
        torch.save(model.state_dict(), f'unet_epoch_{epoch}.pth')
    
    # Plot and save training and validation curves
    print("Training complete. Creating performance plots...")
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Dice Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Coefficient')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_validation_curves.png')
    print("Performance curves saved as 'training_validation_curves.png'")

if __name__ == "__main__":
    main()
