# Tri-Level Preprocessing and Hybrid Models for Alzheimer's Disease Detection

## Overview
This repository contains a complete pipeline for preprocessing MRI images and classifying Alzheimer's Disease (AD), Mild Cognitive Impairment (MCI), and Cognitively Normal (CN) cases using advanced deep learning models. The preprocessing script implements a tri-level approach inspired by the paper "Swin Transformer-Based Segmentation and Multi-Scale Feature Pyramid Fusion Module for Alzheimer's Disease" (Gharaibeh et al., 2023), but replaces GAC skull-stripping with SynthStrip for better accuracy. Two classification models are provided: a 3D ResNet18-based CNN and an enhanced hybrid CNN + Vision Transformer (ViT) model optimized for 3D MRI volumes.

The pipeline processes raw MRI data from directories like ADNI, applies noise reduction, skull stripping, and bias correction, then saves outputs as `.nii` and `.npy` files. Classification scripts support training, evaluation, saliency maps, and cross-validation.

## Features
- **Tri-Level Preprocessing**: Hybrid Kuan-Improved Frost (HKIF) noise reduction, SynthStrip skull stripping, EM-based bias correction.
- **3D ResNet18 Classifier**: Custom ResNet18 for 3D volumes with mixed precision, early stopping, and interpretability via saliency maps.
- **Hybrid CNN-ViT Model**: Enhanced backbone with ResNet blocks, SE attention, 3D ViT, and fusion for superior performance; includes data augmentation via MONAI and 5-fold CV.
- **Utilities**: Data loading from `.npy` vectors, visualization, metadata storage, and detailed logging.
- **Hardware Optimization**: GPU support (e.g., RTX 3090), mixed precision, and CPU fallback.

## Requirements
- Python 3.8+
- Libraries: `numpy`, `nibabel`, `torch` (2.0+), `torchvision`, `monai`, `SimpleITK`, `scikit-learn`, `matplotlib`, `scipy`, `pandas`, `einops`, `tqdm`
- Pre-trained SynthStrip model weights (download from official repo and place in script directory).

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/alzheimers-mri-pipeline.git
   cd alzheimers-mri-pipeline
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   (Create `requirements.txt` with the listed libraries.)

## Usage
### 1. Preprocessing
Run `PreProcessing.py` to process MRI images:
```
python PreProcessing.py --input /path/to/input_dir --output /path/to/output_dir --npy-dir /path/to/npy_dir --visualize
```
- Input dir should have AD/MCI/CN subfolders with `.nii` files.
- Outputs: Processed `.nii` in output dir, `.npy` vectors in npy-dir.

### 2. ResNet18 Classification
Train/evaluate with `res18_t.py`:
```
python res18_t.py --data_dir /path/to/npy_dir --output_dir /path/to/results --epochs 50 --mixed_precision
```
- Supports `--evaluate_only` for test set inference.

### 3. Hybrid CNN-ViT Classification
Train with `vit.py` (uses cross-validation):
```
python vit.py
```
- Hardcoded paths: Edit `data_dir` and `output_dir` in script.
- Outputs: Models, plots, reports in output_dir.

## Citation
- HKIF & EM: Gharaibeh et al. (2023). DOI: 10.3991/ijoe.v19i04.37677
- SynthStrip: Hoopes et al. (2022). DOI: 10.1016/j.neuroimage.2022.119474
If using this code, cite the repository and original paper.

## License
MIT License. See LICENSE file for details.

---

### 350-Word Project Description
This GitHub repository presents an end-to-end pipeline for Alzheimer's Disease (AD) detection from MRI images, combining advanced preprocessing with state-of-the-art deep learning models. Motivated by the need for accurate early diagnosis, it builds on the 2023 paper by Gharaibeh et al. on Swin Transformer-based segmentation but innovates by integrating SynthStrip for skull stripping, enhancing robustness and accuracy in real-world neuroimaging data.

The core preprocessing script (`PreProcessing.py`) employs a tri-level approach: (1) Hybrid Kuan-Improved Frost (HKIF) filtering for noise reduction, preserving edges in 3D volumes; (2) Deep learning-based SynthStrip for precise skull stripping, outperforming traditional GAC methods; and (3) Expectation-Maximization (EM) via SimpleITK for bias field correction. It processes ADNI-style datasets organized into AD/MCI/CN subfolders, outputs `.nii` images and flattened/resized `.npy` vectors, and includes visualization of intermediate steps.

For classification, two models are provided. The first (`res18_t.py`) is a 3D ResNet18 CNN, supporting stratified train-val-test splits, mixed precision training for efficiency, early stopping, and interpretability through saliency maps on axial/coronal/sagittal views. It achieves high accuracy with class-weighted loss to handle imbalances.

The second (`vit.py`) introduces an enhanced hybrid CNN-ViT architecture: a ResNet-like 3D CNN backbone with Squeeze-and-Excitation (SE) attention extracts hierarchical features, fused with a 3D Vision Transformer incorporating multi-head attention and positional embeddings for global context. Optimized for RTX 3090 GPUs, it uses MONAI augmentations (rotations, flips, contrast adjustments), 5-fold cross-validation, label smoothing, and OneCycleLR scheduling. Results include per-fold accuracy (mean ~85-90% in tests), confusion matrices, loss/accuracy curves, and detailed reports.

Designed for researchers and clinicians, this pipeline is modular, reproducible (seeded randomness), and hardware-agnostic (CPU fallback). It processes all available ADNI scans without deduplication, maximizing data utilization. Future extensions could include federated learning or integration with Swin Transformers. Code is MIT-licensed, with citations to foundational works.

(Word count: 350)