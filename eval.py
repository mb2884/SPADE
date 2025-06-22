import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_metrics(original_frame, processed_frame):
    """Calculate various metrics between original and processed frames."""
    
    # Convert to float32 for calculations
    original = original_frame.astype(np.float32) / 255.0
    processed = processed_frame.astype(np.float32) / 255.0
    
    # 1. PSNR
    psnr_value = psnr(original, processed)
    
    # 2. SSIM
    ssim_value = ssim(original, processed, multichannel=True)
    
    # 3. MSE
    mse_value = np.mean((original - processed) ** 2)
    
    # 4. Average Gradient
    def average_gradient(img):
        dx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        dy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        grad = np.sqrt(dx**2 + dy**2)
        return np.mean(grad)
    
    ag_value = average_gradient(processed)
    
    # 5. Angular Second Moment (Texture uniformity)
    def angular_second_moment(img):
        gray = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY)
        glcm = np.zeros((256, 256))
        for i in range(gray.shape[0]-1):
            for j in range(gray.shape[1]-1):
                i_val = int(gray[i,j] * 255)
                j_val = int(gray[i+1,j+1] * 255)
                glcm[i_val,j_val] += 1
        glcm = glcm / glcm.sum()
        asm = np.sum(glcm**2)
        return asm
    
    asm_value = angular_second_moment(processed)
    
    return {
        'PSNR': psnr_value,
        'SSIM': ssim_value,
        'MSE': mse_value,
        'AG': ag_value,
        'ASM': asm_value
    }

def evaluate_video(original_video_path, processed_video_path):
    """Calculate metrics for entire video."""
    cap_orig = cv2.VideoCapture(original_video_path)
    cap_proc = cv2.VideoCapture(processed_video_path)
    
    metrics_sum = {'PSNR': 0, 'SSIM': 0, 'MSE': 0, 'AG': 0, 'ASM': 0}
    frame_count = 0
    
    while True:
        ret_orig, frame_orig = cap_orig.read()
        ret_proc, frame_proc = cap_proc.read()
        
        if not ret_orig or not ret_proc:
            break
            
        metrics = calculate_metrics(frame_orig, frame_proc)
        for key in metrics_sum:
            metrics_sum[key] += metrics[key]
        frame_count += 1
    
    # Calculate averages
    avg_metrics = {k: v/frame_count for k, v in metrics_sum.items()}
    
    cap_orig.release()
    cap_proc.release()
    
    return avg_metrics

# Usage example:
if __name__ == "__main__":
    original_video = "input.mp4"
    processed_video = "output.mp4"
    
    metrics = evaluate_video(original_video, processed_video)
    
    print("\nVideo Quality Metrics:")
    print(f"PSNR: {metrics['PSNR']:.2f} dB")
    print(f"SSIM: {metrics['SSIM']:.4f}")
    print(f"MSE: {metrics['MSE']:.6f}")
    print(f"Average Gradient: {metrics['AG']:.4f}")
    print(f"Angular Second Moment: {metrics['ASM']:.6f}")
