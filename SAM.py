import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor
import torchvision.transforms as transforms

class SARDataset(Dataset):
    def __init__(self, sar_dir, mask_dir, transform=None):
        self.sar_dir = sar_dir
        self.mask_dir = mask_dir
        self.file_list = sorted(os.listdir(sar_dir))
        self.transform = transform
        
    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        sar_path = os.path.join(self.sar_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        # Load as RGB for SAM (which expects 3 channels)
        # For SAR, we'll replicate the single channel to all 3 RGB channels
        sar_img = Image.open(sar_path).convert("L")
        sar_np = np.array(sar_img)
        sar_rgb = np.stack([sar_np, sar_np, sar_np], axis=2)  # Convert to RGB
        
        # Load ground truth mask
        mask = np.array(Image.open(mask_path).convert("L")) // 255  # binarize
        
        # Apply transforms if provided
        if self.transform:
            augmented = self.transform(image=sar_rgb, mask=mask)
            sar_rgb = augmented['image']
            mask = augmented['mask']
        
        return {
            'image': sar_rgb,
            'mask': mask,
            'image_path': sar_path,
            'file_name': img_name
        }

def get_transforms(is_test=True):
    if is_test:
        return A.Compose([
            A.Resize(1024, 1024),  # SAM often works better with higher resolution
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(1024, 1024),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

def calculate_metrics(pred_mask, gt_mask, threshold=0.5):
    """
    Calculate evaluation metrics for segmentation
    """
    # Threshold prediction
    pred_binary = (pred_mask > threshold).astype(np.uint8)
    gt_binary = gt_mask.astype(np.uint8)
    
    # Calculate intersection and union
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    # Calculate metrics
    dice = (2.0 * intersection) / (2.0 * intersection + (union - intersection)) if union > 0 else 1.0
    iou = intersection / union if union > 0 else 1.0
    
    # Calculate precision and recall
    true_positive = intersection
    false_positive = pred_binary.sum() - true_positive
    false_negative = gt_binary.sum() - true_positive
    
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    
    return {
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall
    }

def generate_point_prompt(mask, num_points=5):
    """
    Generate random foreground and background points from the ground truth mask
    to use as prompts for SAM
    """
    # Find foreground and background coordinates
    y_fg, x_fg = np.where(mask == 1)
    y_bg, x_bg = np.where(mask == 0)
    
    # Randomly sample points (if available)
    fg_indices = np.random.choice(len(y_fg), min(num_points, len(y_fg)), replace=False) if len(y_fg) > 0 else []
    bg_indices = np.random.choice(len(y_bg), min(num_points, len(y_bg)), replace=False) if len(y_bg) > 0 else []
    
    # Compile points and labels (1 for foreground, 0 for background)
    points = []
    labels = []
    
    for idx in fg_indices:
        points.append([x_fg[idx], y_fg[idx]])
        labels.append(1)
    
    for idx in bg_indices:
        points.append([x_bg[idx], y_bg[idx]])
        labels.append(0)
    
    if not points:
        return None, None
        
    return np.array(points), np.array(labels)

def visualize_results(image, gt_mask, pred_mask, metrics, save_dir, filename):
    """
    Visualize and save the results
    """
    plt.figure(figsize=(15, 10))
    
    # Original image
    plt.subplot(131)
    plt.title("Original SAR Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    # Ground truth mask
    plt.subplot(132)
    plt.title("Ground Truth Mask")
    plt.imshow(gt_mask, cmap='gray')
    plt.axis('off')
    
    # Predicted mask
    plt.subplot(133)
    plt.title(f"Predicted Mask\nDice: {metrics['dice']:.4f}, IoU: {metrics['iou']:.4f}")
    plt.imshow(pred_mask, cmap='gray')
    plt.axis('off')
    
    # Save the visualization
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{os.path.splitext(filename)[0]}_result.png"))
    plt.close()

def main():
    # Setup paths
    base_dir = '/home/dell/Downloads/Souraja'
    sar_test_dir = os.path.join(base_dir, 'SAR', 'test')
    mask_test_dir = os.path.join(base_dir, 'Masks', 'test')
    results_dir = os.path.join(base_dir, 'SAM_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load SAM model
    print("Loading SAM model...")
    sam_checkpoint = "sam_vit_h_4b8939.pth"  # Update this with the path to your downloaded SAM weights
    model_type = "vit_h"  # Options: vit_h, vit_l, vit_b
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    # Create dataset
    test_dataset = SARDataset(
        sar_test_dir,
        mask_test_dir,
        transform=get_transforms(is_test=True)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one image at a time for SAM
        shuffle=False
    )
    
    # Setup for metrics
    all_metrics = {
        'dice': [],
        'iou': [],
        'precision': [],
        'recall': []
    }
    
    print("Running inference with SAM...")
    for sample in tqdm(test_loader, desc="Processing"):
        image_tensor = sample['image'][0]  # Get the first (and only) image in batch
        gt_mask = sample['mask'][0].numpy()  # Get ground truth mask
        image_path = sample['image_path'][0]
        file_name = sample['file_name'][0]
        
        # Convert tensor to numpy for SAM
        image_np = image_tensor.permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
        
        # Set image in predictor
        predictor.set_image(image_np)
        
        # Generate point prompts from ground truth
        points, labels = generate_point_prompt(gt_mask, num_points=10)
        
        if points is None:
            print(f"Warning: Could not generate points for {file_name}, skipping")
            continue
            
        # Get masks from SAM with point prompts
        masks, scores, logits = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True
        )
        
        # Take the mask with the highest score
        best_mask_idx = np.argmax(scores)
        pred_mask = masks[best_mask_idx]
        
        # Calculate metrics
        metrics = calculate_metrics(pred_mask, gt_mask)
        
        # Store metrics
        for key in all_metrics:
            all_metrics[key].append(metrics[key])
        
        # Load original image for visualization
        orig_image = np.array(Image.open(image_path).convert("L"))
        
        # Visualize and save results
        visualize_results(
            orig_image, 
            gt_mask, 
            pred_mask, 
            metrics, 
            results_dir, 
            file_name
        )
    
    # Calculate and print average metrics
    print("\nAverage Metrics:")
    for key in all_metrics:
        avg_value = np.mean(all_metrics[key])
        print(f"Average {key}: {avg_value:.4f}")
    
    # Plot metrics distribution
    plt.figure(figsize=(12, 8))
    
    metrics_names = list(all_metrics.keys())
    for i, metric_name in enumerate(metrics_names):
        plt.subplot(2, 2, i+1)
        plt.hist(all_metrics[metric_name], bins=10, alpha=0.7)
        plt.title(f"{metric_name.capitalize()} Distribution")
        plt.xlabel(metric_name.capitalize())
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'metrics_distribution.png'))
    
    print(f"Results saved to {results_dir}")

if __name__ == "__main__":
    main()
