import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet34
from model_utils import *

class TemporalHighlightFeatureExtractor(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        pretrained_net = resnet34(weights='DEFAULT')
        
        # Modified first convolution for 6 input channels
        pretrained_net.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Initialize new weights
        with torch.no_grad():
            pretrained_net.conv1.weight[:, :3] = nn.init.kaiming_normal_(pretrained_net.conv1.weight[:, :3])
            pretrained_net.conv1.weight[:, 3:] = pretrained_net.conv1.weight[:, :3].mean(dim=1, keepdim=True)
        
        self.stage1 = nn.Sequential(*list(pretrained_net.children())[:-4])
        self.stage2 = list(pretrained_net.children())[-4]
        self.stage3 = list(pretrained_net.children())[-3]

        self.scores1 = nn.Conv2d(512, num_classes, 1)
        self.scores2 = nn.Conv2d(256, num_classes, 1)
        self.scores3 = nn.Conv2d(128, num_classes, 1)

        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16)
        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_4x.weight.data = bilinear_kernel(num_classes, num_classes, 4)
        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_2x.weight.data = bilinear_kernel(num_classes, num_classes, 4)

    def forward(self, x):
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s3 = self.upsample_2x(self.scores1(s3))
        s2 = self.upsample_4x(self.scores2(s2) + s3)
        s1 = self.upsample_8x(self.scores3(s1) + s2)
        return s1

class TemporalGateGenerator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.coarse = CoarseNetwork(opt)
        self.HFE = TemporalHighlightFeatureExtractor(opt.n_class)
        
        # Original components with temporal support
        self.refine1 = nn.Sequential(
            GatedConv2d(6, opt.latent_channels, 5, 1, 2, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm),
            GatedConv2d(opt.latent_channels, opt.latent_channels, 3, 2, 1, pad_type=opt.pad_type, activation=opt.activation, norm=opt.norm)
        )
        
        # Add LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=opt.latent_channels*56*56, 
            hidden_size=512,
            num_layers=2,
            batch_first=True
        )
        
        self.final_conv = nn.Sequential(
            GatedConv2d(opt.latent_channels, 3, 3, 1, 1, pad_type=opt.pad_type, activation='tanh', norm=opt.norm)
        )

    def forward(self, x_seq):
        batch_size, seq_len = x_seq.shape[:2]
        
        # Process each frame
        features = []
        for t in range(seq_len):
            x = x_seq[:, t]
            hfe = self.HFE(x)
            x = torch.cat([x, hfe], dim=1)
            features.append(self.refine1(x).flatten(1))
        
        # Temporal processing
        features = torch.stack(features, 1)
        lstm_out, _ = self.lstm(features)
        lstm_out = lstm_out.reshape(batch_size, seq_len, -1, 56, 56)
        
        # Decode frames
        outputs = []
        for t in range(seq_len):
            outputs.append(self.final_conv(lstm_out[:, t]))
            
        return torch.stack(outputs, 1)
