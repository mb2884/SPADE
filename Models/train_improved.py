import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import cv2
from tqdm import tqdm
import argparse
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add M2-Net directory to path
sys.path.append('../M2-Net/specular-removal')

# Import necessary modules from network.py
try:
    from network import GateGenerator, weights_init
except ImportError:
    print("Error importing GateGenerator. Make sure network.py is accessible.")
    sys.exit(1)

class NormalizeInverse(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the original domain.
    """
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

class SPADEFrameDataset(Dataset):
    def __init__(self, frames_dir, resize=None, augment=False):
        self.resize = resize
        self.augment = augment
        
        # Normalization parameters
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]
        
        # Define transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.norm_mean, std=self.norm_std)
        ])
        
        # Augmentation transformations
        if self.augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
            ])
        
        self.specular_files = []
        self.diffuse_files = []
        
        # Get specular directories
        spec_dir = os.path.join(frames_dir, "specular")
        diff_dir = os.path.join(frames_dir, "diffuse")
        
        for asset in os.listdir(spec_dir):
            asset_spec_dir = os.path.join(spec_dir, asset)
            asset_diff_dir = os.path.join(diff_dir, asset)
            
            if os.path.isdir(asset_spec_dir) and os.path.isdir(asset_diff_dir):
                # Get all frames in this asset directory
                for frame in os.listdir(asset_spec_dir):
                    if frame.endswith('.png'):
                        spec_file = os.path.join(asset_spec_dir, frame)
                        diff_file = os.path.join(asset_diff_dir, frame)
                        
                        if os.path.exists(diff_file):
                            self.specular_files.append(spec_file)
                            self.diffuse_files.append(diff_file)
        
        print(f"Loaded {len(self.specular_files)} frame pairs")
        
    def __len__(self):
        return len(self.specular_files)
    
    def __getitem__(self, idx):
        # Load images
        spec_img = cv2.imread(self.specular_files[idx])
        diff_img = cv2.imread(self.diffuse_files[idx])
        
        # Convert to RGB (from BGR)
        spec_img = cv2.cvtColor(spec_img, cv2.COLOR_BGR2RGB)
        diff_img = cv2.cvtColor(diff_img, cv2.COLOR_BGR2RGB)
        
        # Resize if specified
        if self.resize:
            spec_img = cv2.resize(spec_img, self.resize)
            diff_img = cv2.resize(diff_img, self.resize)
        
        # Convert to PIL for torchvision transforms
        spec_img = transforms.ToPILImage()(spec_img)
        diff_img = transforms.ToPILImage()(diff_img)
        
        # Apply augmentation
        if self.augment:
            # Use same seed for consistent transforms
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            spec_img = self.augment_transform(spec_img)
            torch.manual_seed(seed)
            diff_img = self.augment_transform(diff_img)
        
        # Apply normalization
        spec_tensor = self.transform(spec_img)
        diff_tensor = self.transform(diff_img)
        
        return {'input': spec_tensor, 'target': diff_tensor}

# Perceptual loss for better visual quality
class PerceptualLoss(nn.Module):
    def __init__(self, device='cuda'):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(weights='DEFAULT')
        self.vgg = vgg.features[:16].to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.mse_loss = nn.MSELoss()
        
    def forward(self, x, y):
        # Extract features
        x_features = self.vgg(x)
        y_features = self.vgg(y)
        return self.mse_loss(x_features, y_features)

def train_improved_model(args):
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Set memory usage limits if using CUDA
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    
    # Create dataset and dataloader
    dataset = SPADEFrameDataset(
        args.frames_dir, 
        resize=(args.image_size, args.image_size) if args.image_size > 0 else None,
        augment=args.augment
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Initialize mixed precision training
    scaler = GradScaler(enabled=(args.device == 'cuda'))
    
    # Create M2-Net model (GateGenerator from network.py)
    opt = argparse.Namespace(
        in_channels=6,
        out_channels=3,
        latent_channels=48,
        pad_type='zero',
        activation='relu',
        norm='none',  # Use 'none' which is supported
        n_class=3
    )

    model = GateGenerator(opt).to(device)
    model.apply(weights_init)  # Initialize weights randomly
    
    model.train()
    
    # Define loss functions
    criterion_l1 = nn.L1Loss()
    
    # Perceptual loss for better visual results
    if args.perceptual_loss:
        criterion_perceptual = PerceptualLoss(device)
    
    # Define optimizer with weight decay to prevent artifacts
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=1e-4  # Added weight decay
    )
    
    # Learning rate scheduler (cosine annealing)
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs,
        eta_min=args.learning_rate/10
    )
    
    # Denormalization transform for visualization
    denormalize = NormalizeInverse(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Create directory for sample outputs
    samples_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    # Create CSV file to track metrics
    metrics_file = os.path.join(args.output_dir, "training_metrics.csv")
    with open(metrics_file, 'w') as f:
        f.write("epoch,loss,lr\n")
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        batch_count = 0
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                try:
                    inputs = batch['input'].to(device)
                    targets = batch['target'].to(device)
                    
                    # Zero gradients
                    optimizer.zero_grad(set_to_none=True)
                    
                    # Forward pass with mixed precision if using CUDA
                    with autocast(enabled=(args.device == 'cuda')):
                        # Forward pass
                        coarse_out, refined_out, _ = model(inputs)
                        
                        # Calculate loss
                        loss = criterion_l1(refined_out, targets)
                        
                        # Add perceptual loss if enabled
                        if args.perceptual_loss:
                            p_loss = criterion_perceptual(refined_out, targets) * 0.1
                            loss += p_loss
                    
                    # Backward pass and optimize with mixed precision if using CUDA
                    if args.device == 'cuda':
                        scaler.scale(loss).backward()
                        
                        # Gradient clipping to prevent artifacts
                        if args.clip_grad:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        
                        # Gradient clipping
                        if args.clip_grad:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            
                        optimizer.step()
                    
                    # Update progress bar
                    current_loss = loss.item()
                    epoch_loss += current_loss
                    batch_count += 1
                    current_lr = optimizer.param_groups[0]['lr']
                    pbar.set_postfix(loss=f"{current_loss:.4f}", lr=f"{current_lr:.6f}")
                    
                    # Save sample images periodically
                    if batch_idx % 100 == 0:
                        with torch.no_grad():
                            # Denormalize images for visualization
                            sample_input = denormalize(inputs[0]).cpu()
                            sample_output = denormalize(refined_out[0]).cpu()
                            sample_target = denormalize(targets[0]).cpu()
                            
                            # Ensure values are in [0, 1]
                            sample_input = torch.clamp(sample_input, 0, 1)
                            sample_output = torch.clamp(sample_output, 0, 1)
                            sample_target = torch.clamp(sample_target, 0, 1)
                            
                            # Save images
                            sample_path = os.path.join(samples_dir, f"epoch_{epoch+1}_batch_{batch_idx}")
                            
                            transforms.ToPILImage()(sample_input).save(f"{sample_path}_input.png")
                            transforms.ToPILImage()(sample_output).save(f"{sample_path}_output.png")
                            transforms.ToPILImage()(sample_target).save(f"{sample_path}_target.png")
                    
                    # Free up memory
                    if args.device == 'cuda':
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        print(f"\nCUDA out of memory error: {e}")
                        print("Try reducing batch size or image size")
                        if args.device == 'cuda':
                            torch.cuda.empty_cache()
                        continue
                    else:
                        print(f"\nError during training: {e}")
                        raise e
        
        # Update learning rate
        scheduler.step()
        
        avg_loss = epoch_loss / max(batch_count, 1)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save metrics
        with open(metrics_file, 'a') as f:
            f.write(f"{epoch+1},{avg_loss:.6f},{scheduler.get_last_lr()[0]:.6f}\n")
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs:
            save_path = os.path.join(args.output_dir, f"m2net_epoch_{epoch+1}.pth")
            try:
                torch.save(model.state_dict(), save_path)
                print(f"Checkpoint saved to {save_path}")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(args.output_dir, "m2net_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"Best model saved to {best_path}")
    
    # Save final model
    final_path = os.path.join(args.output_dir, "m2net_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train improved M2-Net on SPADE dataset')
    parser.add_argument('--frames_dir', type=str, default='./frames',
                        help='Directory containing extracted frames')
    parser.add_argument('--output_dir', type=str, default='./output_improved',
                        help='Directory to save checkpoints and final model')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save checkpoint frequency (epochs)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], 
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Resize images to this size')
    parser.add_argument('--augment', action='store_true',
                        help='Use data augmentation')
    parser.add_argument('--perceptual_loss', action='store_true',
                        help='Use perceptual loss')
    parser.add_argument('--clip_grad', action='store_true',
                        help='Apply gradient clipping')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    train_improved_model(args)
