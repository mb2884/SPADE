import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import random
import tqdm
import torch.nn.functional as F


class TemporalGlareRemovalNet(nn.Module):
    def __init__(self, temporal_window=5):
        super().__init__()
        self.temporal_window = temporal_window
        
        # Temporal feature extraction
        self.temporal_conv = nn.Conv3d(
            in_channels=3,
            out_channels=64,
            kernel_size=(temporal_window, 3, 3),
            padding=(0, 1, 1)
        )
        
        # Spatial feature extraction with skip connections
        self.spatial_encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
        ])
        
        # Attention mechanism for highlight regions
        self.attention = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
        # Decoder with skip connections - Fixed channel dimensions
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 128, 3, padding=1),  # Changed input channels
                nn.BatchNorm2d(128),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1),  # Changed input channels
                nn.BatchNorm2d(64),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(64, 3, 3, padding=1),
                nn.Sigmoid()
            )
        ])
        
    def forward(self, x):
        # x shape: (batch, temporal_window, channels, height, width)
        x = x.permute(0, 2, 1, 3, 4)  # Correct permutation for 3D conv
        
        # Extract temporal features
        temp_features = self.temporal_conv(x)
        temp_features = temp_features.squeeze(2)  # Remove temporal dimension
        
        # Encoder with skip connections
        skip_connections = []
        x = temp_features
        for encoder_layer in self.spatial_encoder:
            x = encoder_layer(x)
            skip_connections.append(x)
        
        # Generate attention map
        attention_map = self.attention(x)
        x = x * attention_map
        
        # Decoder without concatenation
        for idx, decoder_layer in enumerate(self.decoder):
            x = decoder_layer(x)
        
        return x



class SHIQDataset(Dataset):
    def __init__(self, root_dir, split='train', temporal_window=5, transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.temporal_window = temporal_window
        self.transform = transform
        
        # Get all image base names (without suffixes)
        self.image_bases = []
        for filename in sorted(os.listdir(self.root_dir)):
            if filename.endswith('_A.png'):
                base_name = filename[:-6]  # Remove '_A.png'
                self.image_bases.append(base_name)
    
    def __len__(self):
        return len(self.image_bases) - self.temporal_window + 1
    
    def load_image_set(self, base_name):
        # Load all components of an image set
        input_path = os.path.join(self.root_dir, f"{base_name}_A.png")
        target_path = os.path.join(self.root_dir, f"{base_name}_D.png")
        mask_path = os.path.join(self.root_dir, f"{base_name}_S.png")
        
        # Load images
        input_img = Image.open(input_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')
        mask_img = Image.open(mask_path).convert('L')
        
        # Apply transforms if specified
        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
            # Apply same spatial transform to mask
            mask_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
            mask_img = mask_transform(mask_img)
        
        return input_img, target_img, mask_img

    def __getitem__(self, idx):
        # Create temporal window of frames
        input_window = []
        for i in range(self.temporal_window):
            input_img, _, _ = self.load_image_set(self.image_bases[idx + i])
            input_window.append(input_img)
        
        # Get target (middle frame)
        mid_idx = idx + self.temporal_window // 2
        _, target_img, mask_img = self.load_image_set(self.image_bases[mid_idx])
        
        return torch.stack(input_window), target_img, mask_img

def custom_loss(pred, target, mask):
    # Ensure mask has same spatial dimensions as pred/target
    mask = F.interpolate(mask, size=(pred.shape[2], pred.shape[3]), mode='bilinear', align_corners=False)
    
    # Calculate L1 loss
    l1_loss = nn.L1Loss(reduction='none')(pred, target)
    
    # Expand mask to match l1_loss dimensions if needed
    if l1_loss.shape != mask.shape:
        mask = mask.expand(-1, l1_loss.shape[1], -1, -1)
    
    # Apply weighted loss
    weighted_loss = (l1_loss * (1 + mask)).mean()
    return weighted_loss



def train_model(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    
    # Create epoch progress bar
    epoch_pbar = tqdm.tqdm(range(num_epochs), desc='Training Progress', position=0)
    
    for epoch in epoch_pbar:
        # Training phase
        model.train()
        train_loss = 0
        
        # Create batch progress bar for training
        train_pbar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch}', 
                              leave=False, position=1)
        
        for input_windows, targets, masks in train_pbar:
            input_windows = input_windows.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_windows)
            loss = custom_loss(outputs, targets, masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Update training progress bar
            train_pbar.set_postfix({'batch_loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        # Create batch progress bar for validation
        val_pbar = tqdm.tqdm(val_loader, desc='Validation', 
                            leave=False, position=1)
        
        with torch.no_grad():
            for input_windows, targets, masks in val_pbar:
                input_windows = input_windows.to(device)
                targets = targets.to(device)
                masks = masks.to(device)
                
                outputs = model(input_windows)
                loss = custom_loss(outputs, targets, masks)
                val_loss += loss.item()
                
                # Update validation progress bar
                val_pbar.set_postfix({'batch_loss': f'{loss.item():.4f}'})
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'train_loss': f'{avg_train_loss:.4f}',
            'val_loss': f'{avg_val_loss:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, 'best_model.pth')
            epoch_pbar.set_postfix({
                'train_loss': f'{avg_train_loss:.4f}',
                'val_loss': f'{avg_val_loss:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}',
                'saved': 'True'
            })

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Set up transforms
    # Set up transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Ensure consistent size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
    ])

    
    print("Loading datasets...")
    # Create datasets
    train_dataset = SHIQDataset(
        root_dir='SHIQ',
        split='train',
        temporal_window=5,
        transform=transform
    )
    
    val_dataset = SHIQDataset(
        root_dir='SHIQ',
        split='test',
        temporal_window=5,
        transform=transform
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print("Initializing model...")
    # Initialize model
    model = TemporalGlareRemovalNet(temporal_window=5)
    
    print("Starting training...")
    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

if __name__ == '__main__':
    main()