# SPADE: Synthetic Paired Dataset for Specular-Diffuse Video Decomposition

![Project Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Blender](https://img.shields.io/badge/Blender-3.0+-orange)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-PyTorch-red)


[View Final Paper](mb2884_written_final_report.pdf)

## 🎯 Project Overview

SPADE addresses a critical gap in computer vision: the lack of temporally consistent training data for video-based specular highlight removal. This project introduces the **first synthetic video dataset** specifically designed for specular highlight removal, comprising perfectly aligned specular-diffuse video pairs that enable direct supervision for learning-based temporal methods.

**Key Innovation**: While existing datasets focus on still images, SPADE provides paired video sequences where highlights move naturally across surfaces, enabling development of temporally-aware algorithms that maintain consistency across frames.

## 🏆 Key Achievements

- **16.2% improvement** in PSNR (Peak Signal-to-Noise Ratio) over traditional frame-based methods
- **10.2% improvement** in SSIM (Structural Similarity Index) for perceptual quality
- **2.0% improvement** in temporal consistency, reducing flickering artifacts
- **30,000 frames** across 250+ diverse objects with perfect ground truth alignment
- **First-of-its-kind** temporal video dataset for specular highlight removal

## 🛠️ Technical Stack

### Core Technologies
- **Python** - Primary development language
- **Blender 3.0+** - 3D rendering engine with Python API integration
- **PyTorch** - Deep learning framework for neural network development
- **OpenCV** - Computer vision operations and image processing

### Rendering & Graphics
- **Physically-Based Rendering (PBR)** - Accurate material and lighting simulation
- **HDRI Environments** - 798 high-dynamic-range lighting conditions
- **Eevee Rendering Engine** - Optimized for efficient large-scale generation

### Machine Learning Architecture
- **LSTM Networks** - Bidirectional temporal processing for sequence consistency
- **ResNet Backbone** - Feature extraction with residual connections
- **Attention Mechanisms** - Highlight region detection and focused processing

## 📊 Dataset Specifications

| Specification | Value |
|---------------|-------|
| **Total Video Pairs** | 250 paired sequences |
| **Total Frames** | 30,000 frames |
| **Resolution** | 512×512 pixels |
| **Frame Rate** | 30 FPS |
| **Sequence Length** | 60 frames (2 seconds) |
| **HDRI Environments** | 798 realistic lighting conditions |
| **Object Categories** | 8 diverse categories (furniture, vehicles, etc.) |

## 🔬 Technical Approach

### Dataset Generation Pipeline
1. **Asset Preprocessing** - Automated 3D model normalization and scaling
2. **Material Configuration** - Dual rendering: specular vs. matte materials
3. **Dynamic Lighting** - HDRI environment mapping with strategic point lights
4. **Camera Animation** - Constrained random walk algorithm for natural movement
5. **Synchronized Rendering** - Perfect alignment between paired sequences

### Neural Network Architecture
- **Highlight Feature Extractor (HFE)** - ResNet-based highlight region detection
- **Temporal Processing Module** - Bidirectional LSTM for sequence consistency
- **Gate Generator** - Attention-based refinement for seamless blending
- **Multi-Loss Training** - L1, perceptual, highlight attention, and temporal consistency losses

## 🎯 Applications & Impact

### Industry Applications
- **Film Production** - Enhanced post-processing for reflective surfaces
- **Medical Imaging** - Improved diagnostic quality in endoscopic procedures
- **Cultural Heritage** - Better digitization of artifacts with reflective components
- **Augmented Reality** - More accurate environmental mapping in reflective environments

### Research Contributions
- Establishes new paradigm for temporal information in appearance decomposition
- Provides foundation for video-based computer vision algorithms
- Enables systematic evaluation of temporal consistency in highlight removal

## 🧪 Experimental Results

### Quantitative Performance
```
Metric                    | Frame-Based | Temporal Model | Improvement
--------------------------|-------------|----------------|------------
PSNR (dB)                | 23.31       | 27.08          | +16.2%
SSIM                     | 0.816       | 0.899          | +10.2%
Temporal Consistency     | 0.0272      | 0.0266         | +2.0%
```

### Qualitative Improvements
- **Reduced Flickering** - Eliminated frame-to-frame inconsistencies
- **Enhanced Color Stability** - Consistent coloration across temporal sequences
- **Improved Texture Preservation** - Better recovery of fine surface details
- **Stable Edge Handling** - Reduced geometric distortions at highlight boundaries

## 🔍 Key Research Findings

1. **Temporal Information Value** - Quantified 16.2% improvement demonstrates clear benefit of sequence-based processing
2. **Material-Specific Performance** - Metallic surfaces show largest improvements due to directional reflections
3. **Optimal Camera Movement** - Moderate movement (15-30°) provides best temporal processing benefits
4. **Trade-off Analysis** - Temporal consistency vs. local accuracy in challenging highlight regions

## 🌟 Impact & Recognition

- **Grade: A** - Princeton University Senior Thesis
- **Research Advisor** - Dr. Ruth Fong, Princeton Computer Science
- **Publication Ready** - 53-page comprehensive research document

## 📋 Future Directions

- **Real-world validation** - Testing synthetic-to-real domain transfer
- **Architecture exploration** - Attention-based temporal modeling
- **Physical constraints** - Incorporating optical principles into learning
- **Efficiency optimization** - Real-time processing capabilities

---

**Contact**: [mb2884@alumni.princeton.edu](mailto:mb2884@alumni.princeton.edu) | [LinkedIn](https://linkedin.com/in/matthew-w-barrett)
