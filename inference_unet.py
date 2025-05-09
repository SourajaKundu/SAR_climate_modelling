import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from unet import UNet
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

class SARSegmentationDataset(Dataset):
    def __init__(self, sar_dir, mask_dir, transform=None):
        self.sar_dir = sar_dir
        self.mask_dir = mask_dir
        self.file_list = sorted([f for f in os.listdir(sar_dir) if os.path.isfile(os.path.join(sar_dir, f))])
        self.transform = transform
        
    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        sar_path = os.path.join(self.sar_dir, file_name)
        mask_path = os.path.join(self.mask_dir, file_name)
        
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
            
        return {
            'image': sar,
            'mask': mask,
            'file_name': file_name
        }

def get_test_transforms():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=0.5, std=0.5),
        ToTensorV2()
    ])

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

def evaluate_metrics(pred, target):
    """Calculate various segmentation metrics"""
    if target.dim() == 3:  # [B, H, W]
        target = target.unsqueeze(1)  # Make it [B, 1, H, W]        
    dice = dice_coefficient(pred, target)
    
    # Apply threshold
    pred_binary = (pred > 0.5).float()
    
    # IOU / Jaccard Index
    intersection = (pred_binary * target).sum(dim=(1, 2, 3))
    union = pred_binary.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + 1e-5) / (union + 1e-5)
    
    # Precision & Recall
    true_positive = (pred_binary * target).sum(dim=(1, 2, 3))
    false_positive = pred_binary.sum(dim=(1, 2, 3)) - true_positive
    false_negative = target.sum(dim=(1, 2, 3)) - true_positive
    
    precision = (true_positive + 1e-5) / (true_positive + false_positive + 1e-5)
    recall = (true_positive + 1e-5) / (true_positive + false_negative + 1e-5)
    
    # F1 score is equivalent to Dice coefficient
    f1 = dice
    
    return {
        'dice': dice.item(),
        'iou': iou.mean().item(),
        'precision': precision.mean().item(),
        'recall': recall.mean().item(),
        'f1': f1.item()
    }

def visualize_sample(image, gt_mask, pred_mask, file_name, save_dir):
    """Visualize and save prediction results"""
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.title('SAR Image')
    plt.imshow(image.squeeze().cpu().numpy(), cmap='gray')
    plt.axis('off')
    
    # Ground truth mask
    plt.subplot(1, 3, 2)
    plt.title('Ground Truth')
    plt.imshow(gt_mask.squeeze().cpu().numpy(), cmap='gray')
    plt.axis('off')
    
    # Predicted mask
    plt.subplot(1, 3, 3)
    plt.title('Prediction')
    pred_np = (pred_mask > 0.5).float().squeeze().cpu().numpy()
    plt.imshow(pred_np, cmap='gray')
    plt.axis('off')
    
    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{os.path.splitext(file_name)[0]}_result.png"))
    plt.close()

def test_model(model_path, test_loader, device, visualization_dir='test_results'):
    """Test the model on the test dataset"""
    # Load model
    model = UNet(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Initialize metrics tracking
    metrics_list = []
    sample_count = 0
    dice_scores = []
    
    # Create visualization directory
    os.makedirs(visualization_dir, exist_ok=True)
    
    with torch.no_grad():
        for sample in tqdm(test_loader, desc="Testing"):
            images = sample['image'].to(device)
            masks = sample['mask'].to(device)
            file_names = sample['file_name']
            
            # Forward pass
            outputs = model(images)
            outputs = torch.sigmoid(outputs)  # Apply sigmoid if not already done in the model
            
            # Calculate metrics for batch
            batch_metrics = evaluate_metrics(outputs, masks)
            metrics_list.append(batch_metrics)
            
            # Track Dice scores
            batch_dice = dice_coefficient(outputs, masks)
            dice_scores.extend([batch_dice.item()] * images.size(0))
            
            # Visualize first few samples
            if sample_count < 10:  # Visualize first 10 images
                for i in range(min(images.size(0), 10 - sample_count)):
                    visualize_sample(
                        images[i], 
                        masks[i], 
                        outputs[i],
                        file_names[i],
                        visualization_dir
                    )
                    sample_count += 1
    
    # Calculate average metrics
    avg_metrics = {metric: np.mean([d[metric] for d in metrics_list]) for metric in metrics_list[0]}
    
    # Plot Dice score distribution
    plt.figure(figsize=(10, 6))
    plt.hist(dice_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=avg_metrics['dice'], color='red', linestyle='--', linewidth=2, 
                label=f'Mean Dice: {avg_metrics["dice"]:.4f}')
    plt.xlabel('Dice Coefficient')
    plt.ylabel('Frequency')
    plt.title('Distribution of Dice Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(visualization_dir, 'dice_distribution.png'))
    
    return avg_metrics

def main():
    # Setup paths
    base_dir = '/home/dell/Downloads/Souraja'
    sar_test_dir = os.path.join(base_dir, 'SAR', 'test')
    mask_test_dir = os.path.join(base_dir, 'Masks', 'test')
    results_dir = os.path.join(base_dir, 'UNet_test_results')
    
    # Model path - update this to your best model
    model_path = 'unet_epoch_20.pth'  # Change this to your best model
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test dataset and loader
    test_dataset = SARSegmentationDataset(
        sar_test_dir,
        mask_test_dir,
        transform=get_test_transforms()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Test model
    print(f"Testing model: {model_path}")
    test_metrics = test_model(model_path, test_loader, device, results_dir)
    
    # Print metrics
    print("\n===== Test Results =====")
    print(f"Dice Coefficient: {test_metrics['dice']:.4f}")
    print(f"IoU: {test_metrics['iou']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    
    # Create and save a summary text file
    with open(os.path.join(results_dir, 'test_results_summary.txt'), 'w') as f:
        f.write("UNet Test Results Summary\n")
        f.write("========================\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Test Dataset: {sar_test_dir}\n")
        f.write(f"Number of test samples: {len(test_dataset)}\n\n")
        f.write("Metrics:\n")
        f.write(f"- Dice Coefficient: {test_metrics['dice']:.4f}\n")
        f.write(f"- IoU: {test_metrics['iou']:.4f}\n")
        f.write(f"- Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"- Recall: {test_metrics['recall']:.4f}\n")
        f.write(f"- F1 Score: {test_metrics['f1']:.4f}\n")
    
    print(f"\nResults saved to {results_dir}")
    print(f"Summary saved to {os.path.join(results_dir, 'test_results_summary.txt')}")

if __name__ == "__main__":
    main()
