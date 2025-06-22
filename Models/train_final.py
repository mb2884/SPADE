import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import torchvision

# Add M2-Net directory to path
sys.path.append('../M2-Net/specular-removal')

# Fix cuDNN issues
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

class HighlightFeatureExtractor(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        pretrained = torchvision.models.resnet34(weights='DEFAULT')
        
        # Corrected: Use 3 input channels for RGB
        pretrained.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Initialize weights properly
        nn.init.kaiming_normal_(pretrained.conv1.weight, mode='fan_out', nonlinearity='relu')
        
        self.stage1 = nn.Sequential(*list(pretrained.children())[:-4])
        self.stage2 = list(pretrained.children())[-4]
        self.stage3 = list(pretrained.children())[-3]

        self.scores1 = nn.Conv2d(512, num_classes, 1)
        self.scores2 = nn.Conv2d(256, num_classes, 1)
        self.scores3 = nn.Conv2d(128, num_classes, 1)

        # Initialize decoder layers
        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        
        # Initialize weights for transpose convs
        self.upsample_8x.weight.data = self.bilinear_kernel(num_classes, num_classes, 16)
        self.upsample_4x.weight.data = self.bilinear_kernel(num_classes, num_classes, 4)
        self.upsample_2x.weight.data = self.bilinear_kernel(num_classes, num_classes, 4)

    @staticmethod
    def bilinear_kernel(in_channels, out_channels, kernel_size):
        factor = (kernel_size + 1) // 2
        center = factor - 1 if kernel_size % 2 == 1 else factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center)/factor) * (1 - abs(og[1] - center)/factor)
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
        weight[range(in_channels), range(out_channels), :, :] = filt
        return torch.from_numpy(weight)

    def forward(self, x):
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        
        s3 = self.upsample_2x(self.scores1(s3))
        s2 = self.upsample_4x(self.scores2(s2) + s3)
        s1 = self.upsample_8x(self.scores3(s1) + s2)
        return s1

class TemporalM2Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.hfe = HighlightFeatureExtractor()
        # Fixed channel count to match input (3 RGB + 1 HFE = 4 channels)
        self.refine = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )
        self.lstm = nn.LSTM(3*256*256, 128, num_layers=2, batch_first=True)

    def forward(self, x_seq):
        batch_size, seq_len = x_seq.shape[:2]
        outputs = []
        
        for t in range(seq_len):
            # Process each frame through HFE
            with torch.no_grad():  # Prevent backprop through HFE
                hfe = self.hfe(x_seq[:, t])  # [B, 1, H, W]
            
            # Concatenate RGB (3) + HFE output (1) -> 4 channels
            combined = torch.cat([x_seq[:, t], hfe], dim=1)
            
            # Process through refinement
            refined = self.refine(combined)
            outputs.append(refined)
        
        # Stack outputs instead of LSTM processing (simpler approach)
        return torch.stack(outputs, 1)

class VideoDataset(Dataset):
    def __init__(self, data_root, seq_length=8, img_size=224):  # Smaller images for stability
        self.data_root = data_root
        self.seq_length = seq_length
        self.img_size = img_size
        self.pairs = []
        
        for fname in os.listdir(data_root):
            if "_specular.mp4" in fname:
                base = fname.replace("_specular.mp4", "")
                spec_path = os.path.join(data_root, fname)
                diff_path = os.path.join(data_root, f"{base}_no_specular.mp4")
                if os.path.exists(diff_path):
                    self.pairs.append((spec_path, diff_path))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        spec_path, diff_path = self.pairs[idx]
        
        def load_video(path):
            cap = cv2.VideoCapture(path)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.img_size, self.img_size))
                frames.append(frame)
            cap.release()
            return np.stack(frames)/255.0

        spec_frames = load_video(spec_path)
        diff_frames = load_video(diff_path)
        
        # Select a subset of frames if needed
        if len(spec_frames) > self.seq_length:
            start = np.random.randint(0, len(spec_frames) - self.seq_length + 1)
            spec_frames = spec_frames[start:start+self.seq_length]
            diff_frames = diff_frames[start:start+self.seq_length]
        
        return (
            torch.from_numpy(spec_frames).float().permute(0,3,1,2),
            torch.from_numpy(diff_frames).float().permute(0,3,1,2)
        )

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalM2Net().to(device)
    
    # Dataset and loader with reduced parameters
    dataset = VideoDataset(
        "../datasetgen/output_trimmed",
        seq_length=8,
        img_size=224  # Reduced size for memory efficiency
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss()
    
    # Updated syntax for mixed precision
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(50):
        model.train()
        total_loss = 0
        
        for spec, target in tqdm(loader, desc=f"Epoch {epoch+1}"):
            spec = spec.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Updated autocast syntax
            with torch.cuda.amp.autocast():
                outputs = model(spec)
                loss = criterion(outputs, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            # Free memory
            torch.cuda.empty_cache()
        
        print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")
        torch.save(model.state_dict(), f"temporal_model_epoch{epoch+1}.pth")

if __name__ == "__main__":
    train()
