import torch
import torch.nn as nn
import sys
# Add M2-Net directory to path
sys.path.append('../M2-Net/specular-removal')
from network import GateGenerator, HighlightFeatureExtractor

class TemporalM2Net(nn.Module):
    def __init__(self, opt):
        super().__init__()
        # Initialize base M2-Net components
        self.base = GateGenerator(opt)
        self.hfe = HighlightFeatureExtractor(opt.n_class)
        
        # Temporal components
        self.lstm = nn.LSTM(
            input_size=64,  # Match encoder output channels
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        self.fusion = nn.Sequential(
            nn.Conv3d(256, 64, kernel_size=(3,3,3), padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3,3,3), padding=1)
        )

    def forward(self, x_seq):
        """Process video sequence [B, T, C, H, W]"""
        B, T = x_seq.shape[:2]
        outputs = []
        
        for t in range(T):
            # Generate highlight features
            hfe = self.hfe(x_seq[:, t])
            
            # Combine with input (now 6 channels)
            combined = torch.cat([x_seq[:, t], hfe], dim=1)
            
            # Process through base network
            _, refined, _ = self.base(combined)
            outputs.append(refined)
            
        return torch.stack(outputs, 1)
