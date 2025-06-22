import os
import sys
import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm

# Add M2-Net directory to path
sys.path.append('../M2-Net/specular-removal')
from network import GateGenerator

# Import the TemporalM2Net class from train_final.py
from train_final import TemporalM2Net

def process_video(input_video, base_model, temporal_model, num_frames=60, output_dir="output", show_preview=False, rotate=False):
    """Process video with both models and save outputs"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video info
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height}, {fps} fps, {total_frames} frames")
    
    # Limit frames
    frames_to_process = min(num_frames, total_frames)
    print(f"Processing {frames_to_process} frames")
    
    # Create video writers with dimensions based on rotation flag
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    base_output = os.path.join(output_dir, 'base_output.mp4')
    temporal_output = os.path.join(output_dir, 'temporal_output.mp4')
    
    if rotate:
        # Swapped dimensions for rotation
        base_writer = cv2.VideoWriter(base_output, fourcc, fps, (height, width))
        temporal_writer = cv2.VideoWriter(temporal_output, fourcc, fps, (height, width))
    else:
        # Original dimensions
        base_writer = cv2.VideoWriter(base_output, fourcc, fps, (width, height))
        temporal_writer = cv2.VideoWriter(temporal_output, fourcc, fps, (width, height))
    
    # Use fixed size for model compatibility (models were trained on 512x512)
    target_size = (512, 512)
    
    # Collect frames for processing
    frames = []
    for i in tqdm(range(frames_to_process), desc="Reading frames"):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    # Process with base model (frame by frame)
    device = next(base_model.parameters()).device
    
    print("Processing with base model...")
    for i, frame in enumerate(tqdm(frames, desc="Base model")):
        # Resize for model compatibility
        resized_frame = cv2.resize(frame, target_size)
        
        # Convert frame to tensor
        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1).unsqueeze(0).to(device)
        frame_tensor = frame_tensor / 255.0  # Normalize
        
        # Process with base model - matching the processing in evaluate_ablation.py
        with torch.no_grad():
            _, refined, _ = base_model(frame_tensor)
            output_tensor = refined.cpu()
        
        # Convert output to image
        output = (output_tensor[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        
        # Resize back to original dimensions
        output = cv2.resize(output, (width, height))
        
        # Conditionally rotate the output 90 degrees clockwise
        if rotate:
            output = cv2.rotate(output, cv2.ROTATE_90_CLOCKWISE)
        
        # Write to video
        base_writer.write(output)
        
        # Show preview if requested
        if show_preview and i % 10 == 0:
            if rotate:
                preview_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            else:
                preview_frame = frame
            
            preview = np.hstack([preview_frame, output])
            preview_height = min(720, preview.shape[0])
            preview_width = int(preview.shape[1] * (preview_height / preview.shape[0]))
            cv2.imshow('Base Model Preview', cv2.resize(preview, (preview_width, preview_height)))
            cv2.waitKey(1)
    
    # Process with temporal model (entire sequence)
    print("Processing with temporal model...")
    
    # Convert all frames to tensor for temporal model
    sequence = []
    for frame in frames:
        resized_frame = cv2.resize(frame, target_size)
        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1)
        tensor = tensor / 255.0  # Normalize
        sequence.append(tensor)
    
    # Process in chunks to avoid memory issues
    chunk_size = 20
    
    for start_idx in tqdm(range(0, len(sequence), chunk_size), desc="Temporal model"):
        end_idx = min(start_idx + chunk_size, len(sequence))
        chunk_frames = sequence[start_idx:end_idx]
        
        # Stack into [T, C, H, W] then add batch dimension [1, T, C, H, W]
        chunk_tensor = torch.stack(chunk_frames, dim=0).unsqueeze(0).to(device)
        
        # Process with temporal model - matching the processing in evaluate_ablation.py
        with torch.no_grad():
            outputs = temporal_model(chunk_tensor).cpu()
        
        # Process each frame's output
        for i, output_tensor in enumerate(outputs[0]):
            output = (output_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            output = cv2.resize(output, (width, height))
            
            # Conditionally rotate the output 90 degrees clockwise
            if rotate:
                output = cv2.rotate(output, cv2.ROTATE_90_CLOCKWISE)
            
            temporal_writer.write(output)
            
            # Show preview
            if show_preview and (start_idx + i) % 10 == 0:
                frame_idx = start_idx + i
                if frame_idx < len(frames):
                    if rotate:
                        preview_frame = cv2.rotate(frames[frame_idx], cv2.ROTATE_90_CLOCKWISE)
                    else:
                        preview_frame = frames[frame_idx]
                    
                    preview = np.hstack([preview_frame, output])
                    preview_height = min(720, preview.shape[0])
                    preview_width = int(preview.shape[1] * (preview_height / preview.shape[0]))
                    cv2.imshow('Temporal Model Preview', cv2.resize(preview, (preview_width, preview_height)))
                    cv2.waitKey(1)
                
        # Free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Release resources
    cap.release()
    base_writer.release()
    temporal_writer.release()
    if show_preview:
        cv2.destroyAllWindows()
    
    print(f"Outputs saved to {base_output} and {temporal_output}")
    return base_output, temporal_output

def main():
    parser = argparse.ArgumentParser(description="Process video with base and temporal models")
    parser.add_argument("--input_video", required=True, help="Path to input video")
    parser.add_argument("--base_model", required=True, help="Path to base model checkpoint")
    parser.add_argument("--temporal_model", required=True, help="Path to temporal model checkpoint")
    parser.add_argument("--num_frames", type=int, default=60, help="Number of frames to process")
    parser.add_argument("--output_dir", default="processed_output", help="Output directory")
    parser.add_argument("--show_preview", action="store_true", help="Show processing preview")
    parser.add_argument("--rotate", action="store_true", help="Rotate output videos 90 degrees clockwise")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize models exactly as in the evaluation script
    try:
        print("Initializing base model...")
        base_model = GateGenerator(opt=argparse.Namespace(
            in_channels=6,
            out_channels=3,
            latent_channels=48,
            pad_type='zero',
            activation='relu',
            norm='none',
            n_class=3
        )).to(device)
        
        print("Initializing temporal model...")
        temporal_model = TemporalM2Net().to(device)
        
        # Load model weights
        print(f"Loading base model from {args.base_model}")
        base_model.load_state_dict(torch.load(args.base_model, map_location=device))
        
        print(f"Loading temporal model from {args.temporal_model}")
        temporal_model.load_state_dict(torch.load(args.temporal_model, map_location=device))
        
        # Set models to evaluation mode
        base_model.eval()
        temporal_model.eval()
        
        print("Models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Process video
    try:
        process_video(
            args.input_video,
            base_model,
            temporal_model,
            args.num_frames,
            args.output_dir,
            args.show_preview,
            args.rotate
        )
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
