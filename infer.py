import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from model import TemporalGlareRemovalNet
import numpy as np
import cv2
from tqdm import tqdm
import shutil

class GlareRemover:
    def __init__(self, model_path, temporal_window=5, device='cuda'):
        self.temporal_window = temporal_window
        self.device = device
        
        # Initialize model
        self.model = TemporalGlareRemovalNet(temporal_window=temporal_window)
        # Use weights_only=True for safety and proper device mapping
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model.to(device)
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path):
        img = Image.open(image_path).convert('RGB')
        return self.transform(img)
    
    def denormalize(self, tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
        return tensor * std + mean
    
    def process_single_image(self, image_path):
        # Load and transform image
        img_tensor = self.preprocess_image(image_path)
        
        # Replicate image for temporal window
        batch = img_tensor.unsqueeze(0).repeat(self.temporal_window, 1, 1, 1)
        batch = batch.unsqueeze(0).to(self.device)
        
        # Process through model
        with torch.no_grad():
            output = self.model(batch)
        
        # Denormalize and convert to image
        output = self.denormalize(output[0])
        output = output.cpu().clamp(0, 1)
        output = (output * 255).numpy().transpose(1, 2, 0).astype(np.uint8)
        
        return Image.fromarray(output)

def process_directory(input_dir, output_dir, model_path):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize processor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = GlareRemover(model_path, device=device)
    
    # Process all images in directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f'processed_{filename}')
            
            # Process image
            result = processor.process_single_image(input_path)
            result.save(output_path)
            print(f"Processed {filename}")

def extract_frames(video_path, frames_dir, max_frames=None):
    # Clear and create frames directory
    shutil.rmtree(frames_dir, ignore_errors=True)
    os.makedirs(frames_dir, exist_ok=True)
    
    # Read video
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Limit frames if max_frames is specified
    if max_frames is not None:
        frame_count = min(frame_count, max_frames)
    
    # Extract frames
    for i in tqdm(range(frame_count), desc="Extracting frames"):
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(frames_dir, f'frame_{i:06d}.png'), frame)
        else:
            break
    
    cap.release()
    return frame_count, fps

def transfer_colors(source, target):
    """Transfer colors from source to target using Reinhard's method"""
    # Convert to float32
    source = source.astype(np.float32) / 255.0
    target = target.astype(np.float32) / 255.0
    
    # Convert to LAB color space
    source_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB)
    target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB)
    
    # Calculate mean and std for each channel
    source_mean = []
    source_std = []
    target_mean = []
    target_std = []
    
    for i in range(3):
        source_mean.append(np.mean(source_lab[:,:,i]))
        source_std.append(np.std(source_lab[:,:,i]))
        target_mean.append(np.mean(target_lab[:,:,i]))
        target_std.append(np.std(target_lab[:,:,i]))
    
    # Transfer colors
    result = np.copy(target_lab)
    for i in range(3):
        result[:,:,i] = ((result[:,:,i] - target_mean[i]) * 
                        (source_std[i] / target_std[i])) + source_mean[i]
    
    # Convert back to RGB
    result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
    result = np.clip(result, 0, 1)
    
    return (result * 255).astype(np.uint8)

def process_video(video_path, model_path, output_path='output.mp4', max_frames=None, use_color_transfer=False):
    frames_dir = 'frames'
    processed_frames_dir = 'processed_frames'
    
    # Extract frames with max_frames limit
    frame_count, fps = extract_frames(video_path, frames_dir, max_frames)
    
    # Get original video dimensions
    first_frame = cv2.imread(os.path.join(frames_dir, 'frame_000000.png'))
    original_height, original_width = first_frame.shape[:2]
    
    # Clear and create processed frames directory
    shutil.rmtree(processed_frames_dir, ignore_errors=True)
    os.makedirs(processed_frames_dir, exist_ok=True)
    
    # Process frames
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = GlareRemover(model_path, device=device)
    
    for i in tqdm(range(frame_count), desc="Processing frames"):
        frame_path = os.path.join(frames_dir, f'frame_{i:06d}.png')
        
        # Load original frame if using color transfer
        if use_color_transfer:
            original_frame = cv2.imread(frame_path)
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        
        # Process frame through model
        result = processor.process_single_image(frame_path)
        result = result.resize((original_width, original_height), Image.LANCZOS)
        result = np.array(result)
        
        # Apply color transfer if enabled
        if use_color_transfer:
            result = transfer_colors(original_frame, result)
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        
        # Save processed frame
        cv2.imwrite(os.path.join(processed_frames_dir, f'frame_{i:06d}.png'), result)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (original_width*2, original_height))
    
    # Add frames to video
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in tqdm(range(frame_count), desc="Creating video"):
        original = cv2.imread(os.path.join(frames_dir, f'frame_{i:06d}.png'))
        processed = cv2.imread(os.path.join(processed_frames_dir, f'frame_{i:06d}.png'))
        
        cv2.putText(original, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(processed, 'Processed', (10, 30), font, 1, (255, 255, 255), 2)
        
        combined = np.hstack((original, processed))
        out.write(combined)
    
    out.release()
    
    # Clean up
    shutil.rmtree(frames_dir)
    shutil.rmtree(processed_frames_dir)
    
    print(f"Video processing complete. Output saved to {output_path}")



if __name__ == '__main__':
    model_path = 'best_model.pth'
    input_directory = 'input'
    output_directory = 'output'
    
    # process_directory(input_directory, output_directory, model_path)
    
    video_path = 'ear.mp4'
    process_video(video_path, model_path, output_path='MLearoutput2.mp4', use_color_transfer=True, max_frames=256)         
