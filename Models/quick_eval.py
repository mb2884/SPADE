#!/usr/bin/env python3
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

# Import the TemporalM2Net class and VideoDataset from your training scripts
from train_final import TemporalM2Net, VideoDataset

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
plt.rcParams.update({'font.size': 12})

def calculate_temporal_consistency(frames):
    """Calculate frame-to-frame consistency (lower is better)"""
    diffs = []
    for i in range(1, len(frames)):
        diffs.append(np.mean(np.abs(frames[i] - frames[i-1])))
    return np.mean(diffs) if diffs else 0, diffs

def calculate_highlight_error(pred, target, input_img, threshold=0.8):
    """Calculate error specifically in highlight regions"""
    grayscale = np.mean(input_img, axis=2)
    mask = (grayscale > threshold).astype(float)[:,:,None]
    return np.sum(np.abs(pred - target) * mask) / (np.sum(mask) + 1e-8)

def evaluate_model(model, dataloader, device, is_temporal=False):
    """Evaluate model performance with various metrics"""
    model.eval()
    all_metrics = {
        'psnr': [], 'ssim': [], 'highlight_error': [], 'temporal_consistency': [],
        'per_frame_psnr': [], 'per_frame_ssim': [], 'per_frame_consistency': []
    }
    visualization_stored = False
    all_outputs, all_targets, all_inputs = [], [], []

    with torch.no_grad():
        for batch_idx, (spec_seq, target_seq) in enumerate(tqdm(dataloader, desc="Evaluating")):
            spec_seq = spec_seq.to(device)

            # Forward pass
            if is_temporal:
                outputs = model(spec_seq).cpu()
            else:
                frame_outputs = []
                for t in range(spec_seq.shape[1]):
                    frame = spec_seq[:,t]
                    _, refined, _ = model(frame)
                    frame_outputs.append(refined.cpu())
                outputs = torch.stack(frame_outputs, dim=1)

            # Remap from [-1,1] to [0,1]
            outputs = (outputs + 1.0) / 2.0
            outputs = torch.clamp(outputs, 0.0, 1.0)

            # Store first sequence for visuals
            if batch_idx == 0 and not visualization_stored:
                all_outputs.append(outputs.numpy())
                all_targets.append(target_seq.numpy())
                all_inputs.append(spec_seq.cpu().numpy())
                visualization_stored = True

            # Per-frame metrics
            psnr_vals, ssim_vals, h_errs, diffs = [], [], [], []
            for t in range(outputs.shape[1]):
                pred = outputs[0,t].numpy().transpose(1,2,0)
                tgt  = target_seq[0,t].numpy().transpose(1,2,0)
                inp  = spec_seq[0,t].cpu().numpy().transpose(1,2,0)

                psnr_vals.append(psnr(tgt, pred, data_range=1.0))
                ssim_vals.append(ssim(tgt, pred, channel_axis=2, data_range=1.0))
                h_errs.append(calculate_highlight_error(pred, tgt, inp))
                if t>0:
                    prev = outputs[0,t-1].numpy().transpose(1,2,0)
                    diffs.append(np.mean(np.abs(pred - prev)))

            all_metrics['psnr'].extend(psnr_vals)
            all_metrics['ssim'].extend(ssim_vals)
            all_metrics['highlight_error'].extend(h_errs)
            all_metrics['temporal_consistency'].append(np.mean(diffs) if diffs else 0)
            all_metrics['per_frame_psnr'].append(psnr_vals)
            all_metrics['per_frame_ssim'].append(ssim_vals)
            all_metrics['per_frame_consistency'].append(diffs)

    results = {
        'aggregate': {k: np.mean(v) for k,v in all_metrics.items() if not k.startswith('per_frame')},
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

def create_standard_comparison(base_outputs, temporal_outputs, targets, inputs, save_path):
    """Standard side-by-side MP4"""
    _, T, C, H, W = base_outputs.shape
    if T < 60:
        print(f"Warning: Only {T} frames (expected 60)")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_path, fourcc, 30, (W*3, H))
    font, fs, th, col = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2, (255,255,255)
    for t in range(T):
        b = (base_outputs[0,t].transpose(1,2,0)*255).astype(np.uint8)
        m = (temporal_outputs[0,t].transpose(1,2,0)*255).astype(np.uint8)
        g = (targets[0,t].transpose(1,2,0)*255).astype(np.uint8)
        b = cv2.cvtColor(b, cv2.COLOR_RGB2BGR)
        m = cv2.cvtColor(m, cv2.COLOR_RGB2BGR)
        g = cv2.cvtColor(g, cv2.COLOR_RGB2BGR)
        combined = np.hstack([b,m,g])
        cv2.rectangle(combined,(0,0),(W*3,30),(0,0,0),-1)
        cv2.putText(combined,"Base Model",(10,20),font,fs,col,th)
        cv2.putText(combined,"Temporal Model",(W+10,20),font,fs,col,th)
        cv2.putText(combined,"Ground Truth",(2*W+10,20),font,fs,col,th)
        cv2.putText(combined,f"Frame: {t+1}/{T}",(W*3-150,H-10),font,0.5,col,1)
        video.write(combined)
    video.release()
    print(f"Saved standard comparison to {save_path}")

def create_highlight_comparison(base_outputs, temporal_outputs, targets, inputs, save_path):
    """Overlay red mask on specular regions"""
    _, T, C, H, W = base_outputs.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_path, fourcc, 30, (W*3, H))
    font, fs, th, col = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2, (255,255,255)
    for t in range(T):
        b = (base_outputs[0,t].transpose(1,2,0)*255).astype(np.uint8)
        m = (temporal_outputs[0,t].transpose(1,2,0)*255).astype(np.uint8)
        g = (targets[0,t].transpose(1,2,0)*255).astype(np.uint8)
        inp = (inputs[0,t].transpose(1,2,0)*255).astype(np.uint8)
        mask = cv2.cvtColor(inp, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(mask,200,255,cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, np.ones((5,5),np.uint8))
        overlay = np.zeros_like(b); overlay[:,:,2] = mask
        bh = cv2.addWeighted(b,1,overlay,0.3,0)
        mh = cv2.addWeighted(m,1,overlay,0.3,0)
        gh = cv2.addWeighted(g,1,overlay,0.3,0)
        bh = cv2.cvtColor(bh,cv2.COLOR_RGB2BGR)
        mh = cv2.cvtColor(mh,cv2.COLOR_RGB2BGR)
        gh = cv2.cvtColor(gh,cv2.COLOR_RGB2BGR)
        comb = np.hstack([bh,mh,gh])
        cv2.rectangle(comb,(0,0),(W*3,30),(0,0,0),-1)
        cv2.putText(comb,"Base Model",(10,20),font,fs,col,th)
        cv2.putText(comb,"Temporal Model",(W+10,20),font,fs,col,th)
        cv2.putText(comb,"Ground Truth",(2*W+10,20),font,fs,col,th)
        cv2.putText(comb,f"Frame: {t+1}/{T} - Red shows specular", (10,H-10),font,0.5,col,1)
        video.write(comb)
    video.release()
    print(f"Saved highlight comparison to {save_path}")

def create_error_visualization(base_outputs, temporal_outputs, targets, inputs, save_path):
    """Error heatmaps (JET) comparison"""
    _, T, C, H, W = base_outputs.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_path, fourcc, 30, (W*3, H))
    font, fs, th, col = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2, (255,255,255)
    for t in range(T):
        b = base_outputs[0,t].transpose(1,2,0)
        m = temporal_outputs[0,t].transpose(1,2,0)
        g = targets[0,t].transpose(1,2,0)
        be = (np.abs(b-g)/(b.max()+1e-8)*255).astype(np.uint8)
        me = (np.abs(m-g)/(m.max()+1e-8)*255).astype(np.uint8)
        be_h = cv2.applyColorMap(cv2.cvtColor(be,cv2.COLOR_RGB2GRAY),cv2.COLORMAP_JET)
        me_h = cv2.applyColorMap(cv2.cvtColor(me,cv2.COLOR_RGB2GRAY),cv2.COLORMAP_JET)
        gt = (g*255).astype(np.uint8); gt = cv2.cvtColor(gt,cv2.COLOR_RGB2BGR)
        comb = np.hstack([be_h, me_h, gt])
        cv2.rectangle(comb,(0,0),(W*3,30),(0,0,0),-1)
        cv2.putText(comb,"Base Error",(10,20),font,fs,col,th)
        cv2.putText(comb,"Temporal Error",(W+10,20),font,fs,col,th)
        cv2.putText(comb,"Ground Truth",(2*W+10,20),font,fs,col,th)
        cv2.putText(comb,f"Frame: {t+1}/{T} - Blue→Red error", (10,H-10),font,0.5,col,1)
        video.write(comb)
    video.release()
    print(f"Saved error visualization to {save_path}")

def create_stability_visualization(base_outputs, temporal_outputs, targets, inputs, save_path):
    """Frame-difference stability maps"""
    _, T, C, H, W = base_outputs.shape
    if T < 2:
        print("Not enough frames for stability viz")
        return
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_path, fourcc, 30, (W*3, H))
    font, fs, th, col = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2, (255,255,255)
    for t in range(1, T):
        bd = np.abs(base_outputs[0,t].transpose(1,2,0) - base_outputs[0,t-1].transpose(1,2,0))
        md = np.abs(temporal_outputs[0,t].transpose(1,2,0) - temporal_outputs[0,t-1].transpose(1,2,0))
        gd = np.abs(targets[0,t].transpose(1,2,0) - targets[0,t-1].transpose(1,2,0))
        bd = (bd/(max(0.01,bd.max()))*255).astype(np.uint8)
        md = (md/(max(0.01,md.max()))*255).astype(np.uint8)
        gd = (gd/(max(0.01,gd.max()))*255).astype(np.uint8)
        bd_h = cv2.applyColorMap(cv2.cvtColor(bd,cv2.COLOR_RGB2GRAY),cv2.COLORMAP_HOT)
        md_h = cv2.applyColorMap(cv2.cvtColor(md,cv2.COLOR_RGB2GRAY),cv2.COLORMAP_HOT)
        gd_h = cv2.applyColorMap(cv2.cvtColor(gd,cv2.COLOR_RGB2GRAY),cv2.COLORMAP_HOT)
        comb = np.hstack([bd_h, md_h, gd_h])
        cv2.rectangle(comb,(0,0),(W*3,30),(0,0,0),-1)
        cv2.putText(comb,"Base Stability",(10,20),font,fs,col,th)
        cv2.putText(comb,"Temporal Stability",(W+10,20),font,fs,col,th)
        cv2.putText(comb,"Ground Truth",(2*W+10,20),font,fs,col,th)
        cv2.putText(comb,f"Frame: {t+1}/{T} - Δ frame", (10,H-10),font,0.5,col,1)
        video.write(comb)
    video.release()
    print(f"Saved stability visualization to {save_path}")

def create_detailed_metrics_charts(results_base, results_temporal, output_dir):
    """Create summary bar, per-frame, tradeoff, and highlight-error charts"""
    os.makedirs(output_dir, exist_ok=True)
    # 1. Summary
    create_summary_bar_chart(results_base, results_temporal,
                             os.path.join(output_dir,'summary_metrics.png'))
    # 2. Per-frame progression
    create_per_frame_chart(results_base, results_temporal,
                           os.path.join(output_dir,'per_frame_metrics.png'))
    # 3. Tradeoff
    create_tradeoff_chart(results_base, results_temporal,
                          os.path.join(output_dir,'quality_consistency_tradeoff.png'))
    # 4. Highlight-error analysis
    create_highlight_error_chart(results_base, results_temporal,
                                 os.path.join(output_dir,'highlight_error_analysis.png'))

def create_summary_bar_chart(results_base, results_temporal, save_path):
    metrics = ['psnr','ssim','temporal_consistency','highlight_error']
    labels = ['PSNR (dB)','SSIM','Temporal\nConsistency','Highlight\nError']
    base_vals     = [results_base['aggregate'][m] for m in metrics]
    temporal_vals = [results_temporal['aggregate'][m] for m in metrics]
    colors = ['#3498db','#2ecc71']
    fig, axes = plt.subplots(2,2,figsize=(14,10)); axes=axes.flatten()
    for i,(ax,m,l) in enumerate(zip(axes,metrics,labels)):
        better_low = m in ['temporal_consistency','highlight_error']
        b,v = base_vals[i], temporal_vals[i]
        bars = ax.bar([0,1],[b,v],color=colors,width=0.6)
        for j,bar in enumerate(bars):
            val = b if j==0 else v
            ax.text(bar.get_x()+bar.get_width()/2,val+0.01*max(b,v),
                    f"{val:.3f}",ha='center',va='bottom',fontweight='bold')
        # improvement text
        if better_low:
            imp = ((b-v)/b)*100
        else:
            imp = ((v-b)/b)*100
        txt = f"{abs(imp):.1f}% {'better' if imp>0 else 'worse'}"
        ax.text(0.5,max(b,v)*0.5,txt,ha='center',
                color=('green' if imp>0 else 'red'),fontweight='bold')
        ax.set_xticks([0,1]); ax.set_xticklabels(['Base','Temporal'])
        ax.set_title(l,fontsize=14,fontweight='bold'); ax.grid(True,alpha=0.3)
        ax.text(0.5,-0.15,"Lower is better" if better_low else "Higher is better",
                transform=ax.transAxes,ha='center',
                color=('darkred' if better_low else 'darkgreen'),fontsize=10)
    fig.suptitle('Ablation Study: Base vs Temporal',fontsize=16,fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(save_path,dpi=300,bbox_inches='tight'); plt.close()
    print(f"Saved summary chart to {save_path}")

def create_per_frame_chart(results_base, results_temporal, save_path):
    if not results_base['per_frame']['psnr']: 
        print("No per-frame data."); return
    bp = results_base['per_frame']['psnr'][0]
    tp = results_temporal['per_frame']['psnr'][0]
    bs = results_base['per_frame']['ssim'][0]
    ts = results_temporal['per_frame']['ssim'][0]
    L = min(len(bp),len(tp),len(bs),len(ts))
    frames = np.arange(1,L+1)
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(12,8),sharex=True)
    ax1.plot(frames,bp[:L],'o-',label='Base',linewidth=2)
    ax1.plot(frames,tp[:L],'o-',label='Temporal',linewidth=2)
    ax1.set_ylabel('PSNR (dB)'); ax1.set_title('PSNR over Time'); ax1.legend(); ax1.grid(True,alpha=0.3)
    ax2.plot(frames,bs[:L],'o-',label='Base',linewidth=2)
    ax2.plot(frames,ts[:L],'o-',label='Temporal',linewidth=2)
    ax2.set_ylabel('SSIM'); ax2.set_xlabel('Frame'); ax2.set_title('SSIM over Time')
    ax2.legend(); ax2.grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path,dpi=300,bbox_inches='tight'); plt.close()
    print(f"Saved per-frame chart to {save_path}")

def create_tradeoff_chart(results_base, results_temporal, save_path):
    bp = results_base['aggregate']['psnr']
    tp = results_temporal['aggregate']['psnr']
    bc = results_base['aggregate']['temporal_consistency']
    tc = results_temporal['aggregate']['temporal_consistency']
    plt.figure(figsize=(10,8))
    plt.scatter(bc,bp,s=200,label='Base',edgecolor='black')
    plt.scatter(tc,tp,s=200,label='Temporal',edgecolor='black')
    plt.plot([bc,tc],[bp,tp],'k--',alpha=0.5)
    dx,dy = tc-bc, tp-bp
    plt.annotate('Tradeoff ↗',xy=(tc,tp),xytext=(bc+dx/2,bp+dy/2),
                 arrowprops=dict(arrowstyle='->',color='red'),fontsize=12)
    plt.xlabel('Temporal Consistency (↓ better)')
    plt.ylabel('PSNR (↑ better)')
    plt.title('Quality vs Consistency Tradeoff')
    plt.grid(True,alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(save_path,dpi=300,bbox_inches='tight'); plt.close()
    print(f"Saved tradeoff chart to {save_path}")

def create_highlight_error_chart(results_base, results_temporal, save_path):
    bhe = results_base['aggregate']['highlight_error']
    the = results_temporal['aggregate']['highlight_error']
    boe = 1.0 - results_base['aggregate']['ssim']
    toe = 1.0 - results_temporal['aggregate']['ssim']
    fig,ax = plt.subplots(figsize=(10,6))
    bar_w=0.35; r1=np.arange(2); r2=r1+bar_w
    ax.bar(r1, [bhe,boe], width=bar_w, label='Base')
    ax.bar(r2, [the,toe], width=bar_w, label='Temporal')
    hi = ((bhe-the)/bhe)*100
    oi = ((boe-toe)/boe)*100
    ax.text(r1[0]+bar_w/2, max(bhe,the)*0.5,
            f"{hi:.1f}% better" if hi>0 else f"{-hi:.1f}% worse",
            ha='center',color=('green' if hi>0 else 'red'),fontweight='bold')
    ax.text(r1[1]+bar_w/2, boe*1.1,
            f"{oi:.1f}% better" if oi>0 else f"{-oi:.1f}% worse",
            ha='center',color=('green' if oi>0 else 'red'),fontweight='bold')
    ax.set_xticks([r+bar_w/2 for r in r1])
    ax.set_xticklabels(['Highlight Err','Overall Err'])
    ax.set_ylabel('Error (↓ better)')
    ax.set_title('Error in Highlight vs Overall'); ax.legend(); ax.grid(axis='y',alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path,dpi=300,bbox_inches='tight'); plt.close()
    print(f"Saved highlight-error chart to {save_path}")

def create_comparison_videos(base_results, temporal_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    bo, to = base_results['raw_data']['outputs'][0], temporal_results['raw_data']['outputs'][0]
    tgt = base_results['raw_data']['targets'][0]; inp = base_results['raw_data']['inputs'][0]
    create_standard_comparison(bo,to,tgt,inp, os.path.join(output_dir,'standard.mp4'))
    create_highlight_comparison(bo,to,tgt,inp, os.path.join(output_dir,'highlight.mp4'))
    create_error_visualization(bo,to,tgt,inp, os.path.join(output_dir,'error.mp4'))
    create_stability_visualization(bo,to,tgt,inp, os.path.join(output_dir,'stability.mp4'))

def main():
    parser = argparse.ArgumentParser("Ablation evaluation for SH removal")
    parser.add_argument("--base_model",    required=True)
    parser.add_argument("--temporal_model",required=True)
    parser.add_argument("--data_dir",      default="./datasetgen/output_enhanced")
    parser.add_argument("--output_dir",    default="./ablation_results_enhanced")
    parser.add_argument("--seq_length",    type=int, default=60)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load base model
    base_opt = argparse.Namespace(in_channels=6, out_channels=3,
                                  latent_channels=48,
                                  pad_type='zero', activation='relu',
                                  norm='none', n_class=3)
    base_model = GateGenerator(opt=base_opt).to(device)
    base_model.load_state_dict(torch.load(args.base_model,map_location=device))
    base_model.eval()

    # Load temporal model
    temporal_model = TemporalM2Net().to(device)
    temporal_model.load_state_dict(torch.load(args.temporal_model,map_location=device))
    temporal_model.eval()

    # Override dataset for center-crop
    class EvalDataset(VideoDataset):
        def __getitem__(self, idx):
            spec, dif = super().__getitem__(idx)
            if spec.shape[0] > self.seq_length:
                start = (spec.shape[0] - self.seq_length)//2
                spec = spec[start:start+self.seq_length]
                dif  = dif[start:start+self.seq_length]
            return spec, dif

    dataset = EvalDataset(args.data_dir, seq_length=args.seq_length)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False)

    print("Evaluating base model...")
    base_res = evaluate_model(base_model, loader, device, is_temporal=False)
    print("Evaluating temporal model...")
    temp_res = evaluate_model(temporal_model, loader, device, is_temporal=True)

    # Write metrics.txt
    metrics = ['psnr','ssim','temporal_consistency','highlight_error']
    with open(os.path.join(args.output_dir,'metrics.txt'),'w') as f:
        f.write("Metric, Base, Temporal, Improvement (%)\n")
        for m in metrics:
            b,v = base_res['aggregate'][m], temp_res['aggregate'][m]
            imp = ((b-v)/b)*100 if m in ['temporal_consistency','highlight_error'] else ((v-b)/b)*100
            f.write(f"{m},{b:.4f},{v:.4f},{imp:+.2f}\n")

    # Generate visuals
    create_comparison_videos(base_res, temp_res, os.path.join(args.output_dir,'videos'))
    create_detailed_metrics_charts(base_res, temp_res, os.path.join(args.output_dir,'charts'))

    print(f"Done. Results in {args.output_dir}")

if __name__=="__main__":
    main()
