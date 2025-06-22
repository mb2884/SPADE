import os
import sys
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import argparse

# Add M2-Net directory to path
sys.path.append('../M2-Net/specular-removal')
from network import GateGenerator

# Import the TemporalM2Net class from train_final.py
from train_final import TemporalM2Net, VideoDataset

# Set up styling for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
plt.rcParams.update({'font.size': 12})

def calculate_temporal_consistency(frames):
    """Calculate frame-to-frame consistency (lower is better)"""
    diffs = []
    for i in range(1, len(frames)):
        diff = np.mean(np.abs(frames[i] - frames[i-1]))
        diffs.append(diff)
    return np.mean(diffs) if diffs else 0, diffs

def calculate_highlight_error(pred, target, input_img, threshold=0.8):
    """Calculate error specifically in highlight regions"""
    # Estimate highlight regions (bright areas in input)
    grayscale = np.mean(input_img, axis=2)
    highlight_mask = (grayscale > threshold).astype(float)
    
    # Expand dimensions for broadcasting
    highlight_mask = highlight_mask[:, :, np.newaxis]  # Shape becomes (H,W,1)
    
    # Calculate error in highlight regions
    highlight_error = np.sum(np.abs(pred - target) * highlight_mask) / (np.sum(highlight_mask) + 1e-8)
    return highlight_error

def evaluate_model(model, dataloader, device, is_temporal=False):
    """Evaluate model performance with various metrics - memory-efficient version"""
    model.eval()
    
    # Store all metrics
    all_metrics = {
        'psnr': [],
        'ssim': [],
        'temporal_consistency': [],
        'highlight_error': [],
        'per_frame_psnr': [],
        'per_frame_ssim': [],
        'per_frame_consistency': []
    }
    
    # Store outputs for visualization (just first sequence)
    visualization_stored = False
    all_outputs = []
    all_targets = []
    all_inputs = []
    
    with torch.no_grad():
        for batch_idx, (spec_seq, target_seq) in enumerate(tqdm(dataloader, desc="Evaluating")):
            spec_seq = spec_seq.to(device)
            
            # Process the frames (different depending on model type)
            if is_temporal:
                # For temporal model, process whole sequence
                outputs = model(spec_seq).cpu()
            else:
                # For base model, process frames individually
                frame_outputs = []
                for t in range(spec_seq.shape[1]):
                    frame = spec_seq[:, t]
                    _, refined, _ = model(frame)
                    frame_outputs.append(refined.cpu())
                outputs = torch.stack(frame_outputs, dim=1)
            
            # Calculate metrics frame by frame to avoid memory issues
            psnr_values = []
            ssim_values = []
            highlight_errors = []
            frame_diffs = []
            
            # Store first sequence only for visualization
            if batch_idx == 0 and not visualization_stored:
                all_outputs.append(outputs.detach().cpu().numpy())
                all_targets.append(target_seq.detach().cpu().numpy())
                all_inputs.append(spec_seq.detach().cpu().numpy())
                visualization_stored = True
            
            # Process each frame individually
            for t in range(outputs.shape[1]):
                # Convert single frames to numpy
                pred_frame = outputs[0, t].numpy().transpose(1, 2, 0)
                target_frame = target_seq[0, t].numpy().transpose(1, 2, 0)
                input_frame = spec_seq[0, t].cpu().numpy().transpose(1, 2, 0)
                
                # Calculate metrics for this frame
                psnr_val = psnr(target_frame, pred_frame, data_range=1.0)
                ssim_val = ssim(target_frame, pred_frame, channel_axis=2, data_range=1.0)
                highlight_err = calculate_highlight_error(pred_frame, target_frame, input_frame)
                
                # Store frame metrics
                psnr_values.append(psnr_val)
                ssim_values.append(ssim_val)
                highlight_errors.append(highlight_err)
                
                # Calculate frame differences for temporal consistency
                if t > 0:
                    prev_frame = outputs[0, t-1].numpy().transpose(1, 2, 0)
                    frame_diff = np.mean(np.abs(pred_frame - prev_frame))
                    frame_diffs.append(frame_diff)
            
            # Add metrics for this sequence
            all_metrics['psnr'].extend(psnr_values)
            all_metrics['ssim'].extend(ssim_values)
            all_metrics['highlight_error'].extend(highlight_errors)
            all_metrics['temporal_consistency'].append(np.mean(frame_diffs) if frame_diffs else 0)
            
            # Store per-sequence metrics
            all_metrics['per_frame_psnr'].append(psnr_values)
            all_metrics['per_frame_ssim'].append(ssim_values)
            all_metrics['per_frame_consistency'].append(frame_diffs)
            
            # Optional: clear CUDA cache to free memory
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Calculate aggregate statistics
    results = {
        'aggregate': {k: np.mean(v) for k, v in all_metrics.items() if k not in ['per_frame_psnr', 'per_frame_ssim', 'per_frame_consistency']},
        'per_frame': {
            'psnr': all_metrics['per_frame_psnr'],
            'ssim': all_metrics['per_frame_ssim'],
            'consistency': all_metrics['per_frame_consistency']
        },
        'raw_data': {
            'outputs': all_outputs,
            'targets': all_targets,
            'inputs': all_inputs
        }
    }
    
    return results


def create_comparison_videos(results_base, results_temporal, output_dir):
    """Create multiple comparison videos highlighting different aspects"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the first sequence for visualization
    base_outputs = results_base['raw_data']['outputs'][0]
    temporal_outputs = results_temporal['raw_data']['outputs'][0]
    targets = results_base['raw_data']['targets'][0]
    inputs = results_base['raw_data']['inputs'][0]
    
    # 1. Standard side-by-side comparison
    create_standard_comparison(base_outputs, temporal_outputs, targets, inputs,
                              os.path.join(output_dir, 'standard_comparison.mp4'))
    
    # 2. Highlight-focused comparison
    create_highlight_comparison(base_outputs, temporal_outputs, targets, inputs,
                               os.path.join(output_dir, 'highlight_comparison.mp4'))
    
    # 3. Error visualization
    create_error_visualization(base_outputs, temporal_outputs, targets, inputs,
                              os.path.join(output_dir, 'error_visualization.mp4'))
    
    # 4. Temporal stability visualization
    create_stability_visualization(base_outputs, temporal_outputs, targets, inputs,
                                 os.path.join(output_dir, 'stability_visualization.mp4'))

def create_standard_comparison(base_outputs, temporal_outputs, targets, inputs, save_path):
    """Create standard side-by-side comparison video"""
    # Get video dimensions
    _, T, C, H, W = base_outputs.shape
    
    # If we have fewer than expected frames, print warning
    if T < 60:
        print(f"Warning: Only {T} frames available instead of expected 60")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_path, fourcc, 30, (W*3, H))
    
    # Consistent high-quality text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    font_color = (255, 255, 255)  # White for all text
    
    for t in range(T):
        # Get frames
        base_frame = (base_outputs[0, t].transpose(1, 2, 0) * 255).astype(np.uint8)
        temporal_frame = (temporal_outputs[0, t].transpose(1, 2, 0) * 255).astype(np.uint8)
        target_frame = (targets[0, t].transpose(1, 2, 0) * 255).astype(np.uint8)
        
        # Convert from RGB to BGR for OpenCV
        base_frame = cv2.cvtColor(base_frame, cv2.COLOR_RGB2BGR)
        temporal_frame = cv2.cvtColor(temporal_frame, cv2.COLOR_RGB2BGR)
        target_frame = cv2.cvtColor(target_frame, cv2.COLOR_RGB2BGR)
        
        # Concatenate frames
        combined = np.hstack([base_frame, temporal_frame, target_frame])
        
        # Add black background for better text visibility
        cv2.rectangle(combined, (0, 0), (W*3, 30), (0, 0, 0), -1)
        
        # Add labels with consistent formatting
        cv2.putText(combined, "Base Model", (10, 20), font, font_scale, font_color, font_thickness)
        cv2.putText(combined, "Temporal Model", (W + 10, 20), font, font_scale, font_color, font_thickness)
        cv2.putText(combined, "Ground Truth", (2*W + 10, 20), font, font_scale, font_color, font_thickness)
        
        # Add frame counter
        cv2.putText(combined, f"Frame: {t+1}/{T}", (W*3-150, H-10), font, 0.5, font_color, 1)
        
        video.write(combined)
    
    video.release()
    print(f"Standard comparison video saved to {save_path}")

def create_highlight_comparison(base_outputs, temporal_outputs, targets, inputs, save_path):
    """Create comparison video focusing on highlight regions"""
    _, T, C, H, W = base_outputs.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_path, fourcc, 30, (W*3, H))
    
    # Consistent high-quality text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    font_color = (255, 255, 255)  # White for all text
    
    for t in range(T):
        # Get frames
        base_frame = (base_outputs[0, t].transpose(1, 2, 0) * 255).astype(np.uint8)
        temporal_frame = (temporal_outputs[0, t].transpose(1, 2, 0) * 255).astype(np.uint8)
        target_frame = (targets[0, t].transpose(1, 2, 0) * 255).astype(np.uint8)
        input_frame = (inputs[0, t].transpose(1, 2, 0) * 255).astype(np.uint8)
        
        # Create highlight mask
        grayscale = cv2.cvtColor(input_frame, cv2.COLOR_RGB2GRAY)
        _, highlight_mask = cv2.threshold(grayscale, 200, 255, cv2.THRESH_BINARY)
        highlight_mask = cv2.dilate(highlight_mask, np.ones((5,5), np.uint8))
        
        # Create color overlay for highlights
        overlay_color = np.zeros_like(base_frame)
        overlay_color[:,:,2] = highlight_mask  # Red channel
        
        # Apply overlay to each frame
        alpha = 0.3
        base_highlighted = cv2.addWeighted(base_frame, 1, overlay_color, alpha, 0)
        temporal_highlighted = cv2.addWeighted(temporal_frame, 1, overlay_color, alpha, 0)
        target_highlighted = cv2.addWeighted(target_frame, 1, overlay_color, alpha, 0)
        
        # Convert to BGR for OpenCV
        base_highlighted = cv2.cvtColor(base_highlighted, cv2.COLOR_RGB2BGR)
        temporal_highlighted = cv2.cvtColor(temporal_highlighted, cv2.COLOR_RGB2BGR)
        target_highlighted = cv2.cvtColor(target_highlighted, cv2.COLOR_RGB2BGR)
        
        # Concatenate frames
        combined = np.hstack([base_highlighted, temporal_highlighted, target_highlighted])
        
        # Add black background for better text visibility
        cv2.rectangle(combined, (0, 0), (W*3, 30), (0, 0, 0), -1)
        
        # Add labels with consistent formatting
        cv2.putText(combined, "Base Model", (10, 20), font, font_scale, font_color, font_thickness)
        cv2.putText(combined, "Temporal Model", (W + 10, 20), font, font_scale, font_color, font_thickness)
        cv2.putText(combined, "Ground Truth", (2*W + 10, 20), font, font_scale, font_color, font_thickness)
        
        # Add frame counter and note about red highlights
        cv2.putText(combined, f"Frame: {t+1}/{T} - Red overlay shows specular highlight regions", 
                  (10, H-10), font, 0.5, font_color, 1)
        
        video.write(combined)
    
    video.release()
    print(f"Highlight comparison video saved to {save_path}")

def create_error_visualization(base_outputs, temporal_outputs, targets, inputs, save_path):
    """Create video visualizing error maps"""
    _, T, C, H, W = base_outputs.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_path, fourcc, 30, (W*3, H))
    
    # Consistent high-quality text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    font_color = (255, 255, 255)  # White for all text
    
    for t in range(T):
        # Get frames
        base_frame = base_outputs[0, t].transpose(1, 2, 0)
        temporal_frame = temporal_outputs[0, t].transpose(1, 2, 0)
        target_frame = targets[0, t].transpose(1, 2, 0)
        
        # Calculate error maps
        base_error = np.abs(base_frame - target_frame)
        temporal_error = np.abs(temporal_frame - target_frame)
        
        # Normalize for visualization
        base_error_viz = (base_error / (base_error.max() + 1e-8) * 255).astype(np.uint8)
        temporal_error_viz = (temporal_error / (temporal_error.max() + 1e-8) * 255).astype(np.uint8)
        
        # Create heatmaps for better visualization
        base_error_heat = cv2.applyColorMap(cv2.cvtColor(base_error_viz, cv2.COLOR_RGB2GRAY), cv2.COLORMAP_JET)
        temporal_error_heat = cv2.applyColorMap(cv2.cvtColor(temporal_error_viz, cv2.COLOR_RGB2GRAY), cv2.COLORMAP_JET)
        
        # Convert target to uint8
        target_viz = (target_frame * 255).astype(np.uint8)
        target_viz = cv2.cvtColor(target_viz, cv2.COLOR_RGB2BGR)
        
        # Concatenate frames
        combined = np.hstack([base_error_heat, temporal_error_heat, target_viz])
        
        # Add black background for better text visibility
        cv2.rectangle(combined, (0, 0), (W*3, 30), (0, 0, 0), -1)
        
        # Add labels with consistent formatting
        cv2.putText(combined, "Base Error", (10, 20), font, font_scale, font_color, font_thickness)
        cv2.putText(combined, "Temporal Error", (W + 10, 20), font, font_scale, font_color, font_thickness)
        cv2.putText(combined, "Ground Truth", (2*W + 10, 20), font, font_scale, font_color, font_thickness)
        
        # Add frame counter and legend
        cv2.putText(combined, f"Frame: {t+1}/{T} - Error: Blue (low) to Red (high)", 
                  (10, H-10), font, 0.5, font_color, 1)
        
        video.write(combined)
    
    video.release()
    print(f"Error visualization video saved to {save_path}")

def create_stability_visualization(base_outputs, temporal_outputs, targets, inputs, save_path):
    """Create video visualizing temporal stability"""
    _, T, C, H, W = base_outputs.shape
    
    if T < 2:
        print("Not enough frames for stability visualization")
        return
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_path, fourcc, 30, (W*3, H))
    
    # Consistent high-quality text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    font_color = (255, 255, 255)  # White for all text
    
    for t in range(1, T):
        # Calculate frame differences
        base_diff = np.abs(base_outputs[0, t].transpose(1, 2, 0) - base_outputs[0, t-1].transpose(1, 2, 0))
        temporal_diff = np.abs(temporal_outputs[0, t].transpose(1, 2, 0) - temporal_outputs[0, t-1].transpose(1, 2, 0))
        target_diff = np.abs(targets[0, t].transpose(1, 2, 0) - targets[0, t-1].transpose(1, 2, 0))
        
        # Enhance visibility
        base_diff = (base_diff / (max(0.01, base_diff.max())) * 255).astype(np.uint8)
        temporal_diff = (temporal_diff / (max(0.01, temporal_diff.max())) * 255).astype(np.uint8)
        target_diff = (target_diff / (max(0.01, target_diff.max())) * 255).astype(np.uint8)
        
        # Apply color map
        base_diff_heat = cv2.applyColorMap(cv2.cvtColor(base_diff, cv2.COLOR_RGB2GRAY), cv2.COLORMAP_HOT)
        temporal_diff_heat = cv2.applyColorMap(cv2.cvtColor(temporal_diff, cv2.COLOR_RGB2GRAY), cv2.COLORMAP_HOT)
        target_diff_heat = cv2.applyColorMap(cv2.cvtColor(target_diff, cv2.COLOR_RGB2GRAY), cv2.COLORMAP_HOT)
        
        # Concatenate frames
        combined = np.hstack([base_diff_heat, temporal_diff_heat, target_diff_heat])
        
        # Add black background for better text visibility
        cv2.rectangle(combined, (0, 0), (W*3, 30), (0, 0, 0), -1)
        
        # Add labels with consistent formatting
        cv2.putText(combined, "Base Stability", (10, 20), font, font_scale, font_color, font_thickness)
        cv2.putText(combined, "Temporal Stability", (W + 10, 20), font, font_scale, font_color, font_thickness)
        cv2.putText(combined, "Ground Truth", (2*W + 10, 20), font, font_scale, font_color, font_thickness)
        
        # Add frame counter and explanation
        cv2.putText(combined, f"Frame: {t+1}/{T} - Frame-to-frame changes", 
                  (10, H-10), font, 0.5, font_color, 1)
        
        video.write(combined)
    
    video.release()
    print(f"Stability visualization video saved to {save_path}")

def create_detailed_metrics_charts(results_base, results_temporal, output_dir):
    """Create detailed, publication-quality charts for metrics visualization"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Summary bar chart with error bars
    create_summary_bar_chart(results_base, results_temporal, 
                           os.path.join(output_dir, 'summary_metrics.png'))
    
    # 2. Per-frame metric progression
    create_per_frame_chart(results_base, results_temporal,
                         os.path.join(output_dir, 'per_frame_metrics.png'))
    
    # 3. Trade-off visualization
    create_tradeoff_chart(results_base, results_temporal,
                        os.path.join(output_dir, 'quality_consistency_tradeoff.png'))
    
    # 4. Highlight error analysis
    create_highlight_error_chart(results_base, results_temporal,
                               os.path.join(output_dir, 'highlight_error_analysis.png'))

def create_summary_bar_chart(results_base, results_temporal, save_path):
    """Create summary bar chart with confidence intervals"""
    # Extract metrics
    metrics = ['psnr', 'ssim', 'temporal_consistency', 'highlight_error']
    pretty_names = ['PSNR (dB)', 'SSIM', 'Temporal\nConsistency', 'Highlight\nError']
    
    base_values = [results_base['aggregate'][m] for m in metrics]
    temporal_values = [results_temporal['aggregate'][m] for m in metrics]
    
    # Calculate improvements
    improvements = []
    for i, metric in enumerate(metrics):
        if metric == 'temporal_consistency' or metric == 'highlight_error':
            # Lower is better
            imp = ((base_values[i] - temporal_values[i]) / base_values[i]) * 100
        else:
            # Higher is better
            imp = ((temporal_values[i] - base_values[i]) / base_values[i]) * 100
        improvements.append(imp)
    
    # Setup figure
    plt.figure(figsize=(12, 8))
    
    # Plot metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = ['#3498db', '#2ecc71']
    
    for i, (metric, name) in enumerate(zip(metrics, pretty_names)):
        ax = axes[i]
        
        # Decide if higher or lower is better
        better_is_lower = metric in ['temporal_consistency', 'highlight_error']
        
        # Extract values
        base_val = results_base['aggregate'][metric]
        temporal_val = results_temporal['aggregate'][metric]
        
        # Create bar chart
        bars = ax.bar([0, 1], [base_val, temporal_val], color=colors, width=0.6)
        
        # Add values on top of bars
        for j, bar in enumerate(bars):
            val = base_val if j == 0 else temporal_val
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(base_val, temporal_val),
                   f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Calculate improvement
        if better_is_lower:
            imp = ((base_val - temporal_val) / base_val) * 100
            imp_text = f"{imp:.2f}% better" if imp > 0 else f"{-imp:.2f}% worse"
        else:
            imp = ((temporal_val - base_val) / base_val) * 100
            imp_text = f"{imp:.2f}% better" if imp > 0 else f"{-imp:.2f}% worse"
        
        # Add improvement text (fixed positioning to avoid overflow)
        if imp < 0:
            # For negative improvements, place text inside the taller bar
            taller_bar_idx = 0 if base_val > temporal_val else 1
            color = 'white'  # White text inside bar
            ax.text(taller_bar_idx, max(base_val, temporal_val) * 0.5, 
                   imp_text, ha='center', va='center', color=color, fontweight='bold')
        else:
            # For positive improvements, place between the bars at the bottom
            ax.text(0.5, max(base_val, temporal_val) * 0.2, 
                   imp_text, ha='center', va='center', 
                   color='green' if imp > 0 else 'red', fontweight='bold')
        
        # Set labels and title
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Base Model', 'Temporal Model'])
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if better_is_lower:
            ax.text(0.5, -0.15, "Lower is better", transform=ax.transAxes, 
                   ha='center', fontsize=12, fontweight='bold', color='darkred')
        else:
            ax.text(0.5, -0.15, "Higher is better", transform=ax.transAxes, 
                   ha='center', fontsize=12, fontweight='bold', color='darkgreen')
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.suptitle('Ablation Study: Base vs. Temporal Model Performance', fontsize=16, fontweight='bold')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Summary metrics chart saved to {save_path}")

def create_per_frame_chart(results_base, results_temporal, save_path):
    """Create chart showing how metrics change over frames"""
    # Get first sequence for each model
    if not results_base['per_frame']['psnr'] or not results_temporal['per_frame']['psnr']:
        print("No per-frame data available")
        return
    
    base_psnr = results_base['per_frame']['psnr'][0]
    temporal_psnr = results_temporal['per_frame']['psnr'][0]
    base_ssim = results_base['per_frame']['ssim'][0]
    temporal_ssim = results_temporal['per_frame']['ssim'][0]
    
    # Ensure same length
    min_len = min(len(base_psnr), len(temporal_psnr), len(base_ssim), len(temporal_ssim))
    
    base_psnr = base_psnr[:min_len]
    temporal_psnr = temporal_psnr[:min_len]
    base_ssim = base_ssim[:min_len]
    temporal_ssim = temporal_ssim[:min_len]
    
    # Setup figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot PSNR
    frames = list(range(1, min_len + 1))
    ax1.plot(frames, base_psnr, 'o-', color='#3498db', label='Base Model', linewidth=2)
    ax1.plot(frames, temporal_psnr, 'o-', color='#2ecc71', label='Temporal Model', linewidth=2)
    ax1.set_ylabel('PSNR (dB)', fontsize=12)
    ax1.set_title('PSNR Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Plot SSIM
    ax2.plot(frames, base_ssim, 'o-', color='#3498db', label='Base Model', linewidth=2)
    ax2.plot(frames, temporal_ssim, 'o-', color='#2ecc71', label='Temporal Model', linewidth=2)
    ax2.set_xlabel('Frame Number', fontsize=12)
    ax2.set_ylabel('SSIM', fontsize=12)
    ax2.set_title('SSIM Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    
    # Add consistency info if available
    if results_base['per_frame']['consistency'] and results_temporal['per_frame']['consistency']:
        base_consistency = results_base['per_frame']['consistency'][0]
        temporal_consistency = results_temporal['per_frame']['consistency'][0]
        
        if len(base_consistency) > 0 and len(temporal_consistency) > 0:
            min_len = min(len(base_consistency), len(temporal_consistency))
            
            # Create a third subplot for consistency
            plt.figure(figsize=(12, 4))
            ax3 = plt.gca()
            
            frames = list(range(2, min_len + 2))  # Consistency starts from frame 2
            ax3.plot(frames, base_consistency[:min_len], 'o-', color='#3498db', label='Base Model', linewidth=2)
            ax3.plot(frames, temporal_consistency[:min_len], 'o-', color='#2ecc71', label='Temporal Model', linewidth=2)
            ax3.set_xlabel('Frame Number', fontsize=12)
            ax3.set_ylabel('Frame Difference', fontsize=12)
            ax3.set_title('Temporal Stability (lower is better)', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.legend(loc='best')
            
            # Save consistency chart separately
            plt.savefig(save_path.replace('.png', '_consistency.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # Save main chart
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Per-frame metrics chart saved to {save_path}")

def create_tradeoff_chart(results_base, results_temporal, save_path):
    """Create chart visualizing quality vs. consistency tradeoff"""
    # Extract relevant metrics
    base_psnr = results_base['aggregate']['psnr']
    temporal_psnr = results_temporal['aggregate']['psnr']
    base_consistency = results_base['aggregate']['temporal_consistency']
    temporal_consistency = results_temporal['aggregate']['temporal_consistency']
    
    # Setup figure
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter(base_consistency, base_psnr, s=200, color='#3498db', label='Base Model', 
               edgecolor='black', linewidth=1.5, alpha=0.8)
    plt.scatter(temporal_consistency, temporal_psnr, s=200, color='#2ecc71', label='Temporal Model', 
               edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add connecting line to show the tradeoff
    plt.plot([base_consistency, temporal_consistency], [base_psnr, temporal_psnr], 
            'k--', alpha=0.5, linewidth=2)
    
    # Add arrow and text
    dx = temporal_consistency - base_consistency
    dy = temporal_psnr - base_psnr
    
    # Note about tradeoff - this is placed safely inside the plot
    plt.annotate('Tradeoff Direction', 
                xy=(temporal_consistency, temporal_psnr),
                xytext=(base_consistency + dx/2, base_psnr + dy/2),
                arrowprops=dict(arrowstyle='->', linewidth=2, color='red'),
                fontsize=12, ha='center', fontweight='bold')
    
    # Labels and title
    plt.xlabel('Temporal Consistency (lower is better)', fontsize=12)
    plt.ylabel('PSNR (higher is better)', fontsize=12)
    plt.title('Quality vs. Consistency Tradeoff', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=12)
    
    # Add notes explaining the tradeoff - use box to constrain text
    text_box = plt.text(0.5, 0.02, 
               "This chart shows the relationship between image quality (PSNR) and temporal stability.",
               ha='center', fontsize=10, transform=plt.gcf().transFigure,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Tradeoff visualization saved to {save_path}")

def create_highlight_error_chart(results_base, results_temporal, save_path):
    """Create chart analyzing error specifically in highlight regions"""
    # Extract highlight error metrics
    base_highlight_error = results_base['aggregate']['highlight_error']
    temporal_highlight_error = results_temporal['aggregate']['highlight_error']
    
    # Also get overall error for comparison
    base_overall_error = 1.0 - results_base['aggregate']['ssim']  # Approximation
    temporal_overall_error = 1.0 - results_temporal['aggregate']['ssim']  # Approximation
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bar positions
    bar_width = 0.35
    r1 = np.arange(2)
    r2 = [x + bar_width for x in r1]
    
    # Create grouped bar chart
    ax.bar(r1, [base_highlight_error, base_overall_error], width=bar_width, 
          label='Base Model', color='#3498db', edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.bar(r2, [temporal_highlight_error, temporal_overall_error], width=bar_width,
          label='Temporal Model', color='#2ecc71', edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add improvement percentages
    highlight_improvement = ((base_highlight_error - temporal_highlight_error) / base_highlight_error) * 100
    overall_improvement = ((base_overall_error - temporal_overall_error) / base_overall_error) * 100
    
    # Add improvement text with fixed positioning to prevent overflow
    # For highlight region
    if highlight_improvement < 0:
        text = f"{abs(highlight_improvement):.1f}% worse"
        color = 'red'
        # Place text at top of chart
        ax.text(r1[0] + bar_width/2, min(base_highlight_error, temporal_highlight_error) / 2, 
               text, ha='center', color='white', fontweight='bold')
    else:
        text = f"{highlight_improvement:.1f}% better"
        color = 'green'
        # Place text at top of chart
        ax.text(r1[0] + bar_width/2, max(base_highlight_error, temporal_highlight_error) * 0.5, 
               text, ha='center', color=color, fontweight='bold')
    
    # For overall error
    if overall_improvement < 0:
        text = f"{abs(overall_improvement):.1f}% worse"
        color = 'red'
        # Place text at top of chart
        ax.text(r1[1] + bar_width/2, min(base_overall_error, temporal_overall_error) / 2, 
               text, ha='center', color='white', fontweight='bold')
    else:
        text = f"{overall_improvement:.1f}% better"
        color = 'green'
        # Place text above the bar
        ax.text(r1[1] + bar_width/2, base_overall_error * 1.1, 
               text, ha='center', color=color, fontweight='bold')
    
    # Labels and title
    ax.set_xticks([r + bar_width/2 for r in r1])
    ax.set_xticklabels(['Highlight Region Error', 'Overall Error'], fontsize=12)
    ax.set_ylabel('Error (lower is better)', fontsize=12)
    ax.set_title('Error Analysis in Highlight vs. Overall Regions', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add annotation about what this means - use text box to constrain
    plt.figtext(0.5, 0.01,
               "This chart compares error specifically in specular highlight regions vs. overall error.",
               ha='center', fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Highlight error analysis saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Enhanced ablation study for specular highlight removal")
    parser.add_argument("--base_model", required=True, help="Path to base model checkpoint")
    parser.add_argument("--temporal_model", required=True, help="Path to temporal model checkpoint")
    parser.add_argument("--data_dir", default="../datasetgen/output_enhanced", help="Directory with test videos")
    parser.add_argument("--output_dir", default="./ablation_results_enhanced", help="Directory to save results")
    parser.add_argument("--seq_length", type=int, default=60, help="Sequence length for evaluation (max 60)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize base model (M2-Net GateGenerator)
    base_model = GateGenerator(opt=argparse.Namespace(
        in_channels=6,
        out_channels=3,
        latent_channels=48,
        pad_type='zero',
        activation='relu',
        norm='none',
        n_class=3
    )).to(device)
    
    # Initialize temporal model
    temporal_model = TemporalM2Net().to(device)
    
    # Load model weights
    base_model.load_state_dict(torch.load(args.base_model, map_location=device))
    temporal_model.load_state_dict(torch.load(args.temporal_model, map_location=device))
    
    # Set models to evaluation mode
    base_model.eval()
    temporal_model.eval()
    
    # Create dataset and dataloader
    dataset = VideoDataset(args.data_dir, seq_length=args.seq_length)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Evaluate models
    print("Evaluating base model...")
    base_results = evaluate_model(base_model, dataloader, device, is_temporal=False)
    
    print("Evaluating temporal model...")
    temporal_results = evaluate_model(temporal_model, dataloader, device, is_temporal=True)
    
    # Print metrics
    base_metrics = base_results['aggregate']
    temporal_metrics = temporal_results['aggregate']
    
    print("\nAblation Study Results:")
    print("=" * 50)
    print(f"{'Metric':<20} {'Base Model':<15} {'Temporal Model':<15} {'Improvement':<15}")
    print("-" * 50)
    
    metrics_list = ['psnr', 'ssim', 'temporal_consistency', 'highlight_error']
    
    for metric in metrics_list:
        base_val = base_metrics[metric]
        temp_val = temporal_metrics[metric]
        
        if metric in ['temporal_consistency', 'highlight_error']:
            # Lower is better
            improvement = ((base_val - temp_val) / base_val) * 100
            print(f"{metric:<20} {base_val:<15.4f} {temp_val:<15.4f} {improvement:+<15.2f}%")
        else:
            # Higher is better
            improvement = ((temp_val - base_val) / base_val) * 100
            print(f"{metric:<20} {base_val:<15.4f} {temp_val:<15.4f} {improvement:+<15.2f}%")
    
    # Save metrics to file
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write("Ablation Study Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"{'Metric':<20} {'Base Model':<15} {'Temporal Model':<15} {'Improvement':<15}\n")
        f.write("-" * 50 + "\n")
        
        for metric in metrics_list:
            base_val = base_metrics[metric]
            temp_val = temporal_metrics[metric]
            
            if metric in ['temporal_consistency', 'highlight_error']:
                improvement = ((base_val - temp_val) / base_val) * 100
                f.write(f"{metric:<20} {base_val:<15.4f} {temp_val:<15.4f} {improvement:+<15.2f}%\n")
            else:
                improvement = ((temp_val - base_val) / base_val) * 100
                f.write(f"{metric:<20} {base_val:<15.4f} {temp_val:<15.4f} {improvement:+<15.2f}%\n")
    
    # Create visualizations
    create_comparison_videos(base_results, temporal_results, os.path.join(args.output_dir, 'videos'))
    create_detailed_metrics_charts(base_results, temporal_results, os.path.join(args.output_dir, 'charts'))
    
    print(f"Evaluation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
