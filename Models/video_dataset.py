import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class VideoPairDataset(Dataset):
    def __init__(self, data_root, seq_length=8, img_size=256):
        self.data_root = data_root
        self.seq_length = seq_length
        self.img_size = img_size
        self.pairs = []
        
        # Find all video pairs
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
            return np.stack(frames)

        # Load and process video pairs
        spec_frames = load_video(spec_path)
        diff_frames = load_video(diff_path)
        
        # Convert to tensor and normalize
        spec_tensor = torch.from_numpy(spec_frames).float().permute(0,3,1,2)/255.0
        diff_tensor = torch.from_numpy(diff_frames).float().permute(0,3,1,2)/255.0
        
        return spec_tensor, diff_tensor
