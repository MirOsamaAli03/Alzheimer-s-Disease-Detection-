#!/usr/bin/env python3
"""
Tri-level Preprocessing for Alzheimer's Detection with SynthStrip

This script integrates:
1. Noise Reduction using Hybrid Kuan and Improved Frost filters (HKIF)
2. Skull Stripping using SynthStrip (deep learning-based approach)
3. Bias Field Correction using Expectation-Maximization (EM)

The implementation follows the methods described in the paper:
"Swin Transformer-Based Segmentation and Multi-Scale Feature Pyramid Fusion Module for Alzheimer's Disease"
but replaces the GAC skull-stripping with SynthStrip for improved accuracy.
"""

import os
import argparse
import shutil
import json
import datetime
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import tempfile
from scipy import ndimage
from scipy.ndimage import gaussian_filter, uniform_filter, binary_fill_holes
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.morphology import ball, disk
import SimpleITK as sitk

# Define argument parser
parser = argparse.ArgumentParser(description='Tri-level Preprocessing for MRI images with HKIF, SynthStrip, and EM')
parser.add_argument('--input', '-i', required=True, help='Input directory containing AD, MCI, CN subfolders')
parser.add_argument('--output', '-o', required=True, help='Output directory for processed images')
parser.add_argument('--border', '-b', type=float, default=1.0, help='Mask border threshold in mm (default: 1.0)')
parser.add_argument('--no-csf', action='store_true', help='Exclude CSF from brain border')
parser.add_argument('--cpu', action='store_true', help='Force CPU usage (default: use GPU if available)')
parser.add_argument('--model-dir', help='Directory to save/load model weights (default: temp directory)')
parser.add_argument('--visualize', action='store_true', help='Visualize first image in each category')
parser.add_argument('--threads', type=int, default=2, help='Number of PyTorch threads (default: 2)')
parser.add_argument('--skip-noise', action='store_true', help='Skip noise reduction (default: False)')
parser.add_argument('--skip-bias', action='store_true', help='Skip bias field correction (default: False)')
parser.add_argument('--npy-dir', type=str, help='Directory to save .npy vector files (default: processed_npy_data)')
parser.add_argument('--flatten', action='store_true', help='Flatten 3D MRI data to 1D vector before saving as .npy')
parser.add_argument('--resize', type=int, nargs=3, help='Resize MRI data to specified dimensions (e.g., --resize 128 128 128)')
args = parser.parse_args()

# Set PyTorch threads to limit memory usage
torch.set_num_threads(args.threads)

# Set device (GPU or CPU)
if args.cpu:
    device = torch.device('cpu')
    print('Using CPU for inference')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print('Using GPU for inference')
    else:
        print('GPU not available, using CPU for inference')

# Model URL and file
MODEL_VERSION = '1'
if args.no_csf:
    MODEL_FILE = f'synthstrip.nocsf.{MODEL_VERSION}.pt'
else:
    MODEL_FILE = f'synthstrip.{MODEL_VERSION}.pt'

# Define model directory
if args.model_dir:
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
else:
    model_dir = tempfile.gettempdir()

model_path = os.path.join(model_dir, MODEL_FILE)

# Look for model in current directory if not found in model_dir
if not os.path.exists(model_path):
    current_dir_model = os.path.join(".", MODEL_FILE)
    if os.path.exists(current_dir_model):
        print(f"Found model in current directory, copying to: {model_path}")
        shutil.copy(current_dir_model, model_path)
    else:
        print(f"Error: Model not found at {model_path} or in current directory")
        print(f"Please download the model and save it as {MODEL_FILE} in the current directory")
        exit(1)

print(f"Using model weights from: {model_path}")


#############################
# 1. NOISE REDUCTION (HKIF) #
#############################

def calculate_gamma(image, num_regions=25):
    """Calculate gamma value for Kuan filter"""
    height, width = image.shape[:2]
    region_h = height // num_regions
    region_w = width // num_regions
    
    # Ensure we have at least 5x5 pixels per region
    if region_h < 5 or region_w < 5:
        num_regions = min(height // 5, width // 5)
        region_h = height // num_regions
        region_w = width // num_regions
    
    gamma_values = []
    
    # Calculate gamma for each sub-region
    for i in range(num_regions):
        for j in range(num_regions):
            # Define region boundaries
            start_i = i * region_h
            end_i = min((i + 1) * region_h, height)
            start_j = j * region_w
            end_j = min((j + 1) * region_w, width)
            
            # Extract region
            region = image[start_i:end_i, start_j:end_j]
            
            # Calculate mean and std for region
            mean_r = np.mean(region)
            std_r = np.std(region)
            
            # Calculate gamma for this region if mean > 0
            if mean_r > 0 and std_r > 0:
                gamma = (mean_r / std_r) ** 2
                gamma_values.append(gamma)
    
    # Return average of two smallest gamma values (most uniform regions)
    if len(gamma_values) > 1:
        gamma_values.sort()
        return np.mean(gamma_values[:2])
    elif len(gamma_values) == 1:
        return gamma_values[0]
    else:
        # Default value if no valid regions found
        return 1.0

def kuan_filter(image, window_size=5):
    """
    Apply Kuan filter for noise reduction while preserving edges
    
    Parameters:
    - image: Input image
    - window_size: Size of filter window (odd number)
    
    Returns:
    - Filtered image
    """
    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    # Calculate gamma for look_n parameter
    gamma = calculate_gamma(image)
    
    # Initialize output image
    output = np.zeros_like(image, dtype=np.float32)
    
    # Pad image to handle boundaries
    pad_size = window_size // 2
    padded = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), mode='reflect')
    
    # Calculate adjustment parameters
    # These are empirically determined based on image properties
    a_n = 0.2  # Noise variance coefficient
    a_d = 0.3  # Image detail coefficient
    a = a_d + a_n
    
    # Adjust look_n parameter based on gamma
    look_n = gamma * (1 + a)
    
    # Apply filter
    height, width = image.shape[:2]
    for i in range(height):
        for j in range(width):
            # Extract window
            window = padded[i:i+window_size, j:j+window_size]
            
            # Calculate statistics
            window_mean = np.mean(window)
            window_std = np.std(window)
            
            # Calculate Ci from equation (3)
            if window_mean > 0:
                ci = window_std / window_mean
            else:
                ci = 0
            
            # Calculate weight from equation (2)
            c_i = 1.0 / look_n
            
            if ci**2 > c_i:
                weight = (1 - c_i/ci**2) / (1 + c_i)
            else:
                weight = 0
            
            # Apply filter equation (1)
            center_pixel = padded[i+pad_size, j+pad_size]
            output[i, j] = weight * center_pixel + (1 - weight) * window_mean
    
    return output

def improved_frost_filter(image, window_size_min=3, window_size_max=9, beta=0.5):
    """
    Apply Improved Frost filter with adaptive window size
    
    Parameters:
    - image: Input image
    - window_size_min: Minimum window size
    - window_size_max: Maximum window size
    - beta: System parameter for threshold calculation
    
    Returns:
    - Filtered image
    """
    # Initialize output image
    output = np.zeros_like(image, dtype=np.float32)
    
    # Estimate noise variance (using homogeneous region)
    # We'll use the 5% darkest pixels as our noise estimate
    flat_img = image.flatten()
    sorted_img = np.sort(flat_img)
    dark_region = sorted_img[:int(len(sorted_img) * 0.05)]
    noise_variance = np.var(dark_region)
    
    # Apply filter with adaptive window size
    height, width = image.shape[:2]
    for i in range(height):
        for j in range(width):
            # Start with minimum window size
            current_window_size = window_size_min
            
            while current_window_size <= window_size_max:
                # Extract window (handle boundaries)
                half_size = current_window_size // 2
                i_start = max(0, i - half_size)
                i_end = min(height, i + half_size + 1)
                j_start = max(0, j - half_size)
                j_end = min(width, j + half_size + 1)
                
                window = image[i_start:i_end, j_start:j_end]
                
                # Calculate statistics
                window_mean = np.mean(window)
                window_std = np.std(window)
                
                # Calculate h_ij from equation (8)
                if window_mean > 0:
                    h_ij = window_std / window_mean
                else:
                    h_ij = 0
                
                # Calculate threshold from equation (9)
                tr_ij = (1 + beta) * (noise_variance / current_window_size**2) ** 0.5
                
                # Check if window size needs to be increased based on equation (10)
                if h_ij > tr_ij and current_window_size < window_size_max:
                    current_window_size += 2  # Increase window size and try again
                else:
                    break
            
            # Apply frost filter with final window size
            # Extract final window
            half_size = current_window_size // 2
            i_start = max(0, i - half_size)
            i_end = min(height, i + half_size + 1)
            j_start = max(0, j - half_size)
            j_end = min(width, j + half_size + 1)
            
            window = image[i_start:i_end, j_start:j_end]
            
            # Calculate distance to center for each pixel
            y_indices, x_indices = np.indices(window.shape)
            center_y, center_x = (window.shape[0] - 1) / 2, (window.shape[1] - 1) / 2
            distances = np.sqrt((y_indices - center_y)**2 + (x_indices - center_x)**2)
            
            # Calculate h_ij again for final window
            window_mean = np.mean(window)
            window_std = np.std(window)
            if window_mean > 0:
                h_ij = window_std / window_mean
            else:
                h_ij = 0
                
            # Adaptive tuning factor k
            k = h_ij  # Using coefficient of variation as tuning factor
            
            # Calculate weights based on equation (7)
            weights = np.exp(-k * h_ij * distances)
            weights = weights / np.sum(weights)  # Normalize weights
            
            # Apply weighted filter
            output[i, j] = np.sum(window * weights)
    
    return output

def hybrid_noise_reduction(image, sigma1=0.5, sigma2=1.5):
    """
    Apply Hybrid Kuan and Improved Frost filters (HKIF) for noise reduction
    
    Parameters:
    - image: Input 3D MRI image
    - sigma1, sigma2: Parameters for additional smoothing
    
    Returns:
    - Noise-reduced image
    """
    print("Applying HKIF noise reduction...")
    
    # Create output image
    output = np.zeros_like(image)
    
    # Process each slice
    for z in range(image.shape[2]):
        slice_data = image[:, :, z]
        
        # Skip empty slices
        if np.max(slice_data) < 0.01:
            continue
                
        # Apply Kuan filter
        kuan_result = kuan_filter(slice_data)
        
        # Apply edge-preserving smoothing (similar to paper's implementation)
        smoothed1 = gaussian_filter(kuan_result, sigma=sigma1)
        
        # Apply Improved Frost filter
        frost_result = improved_frost_filter(slice_data)
        
        # Apply additional smoothing
        smoothed2 = gaussian_filter(frost_result, sigma=sigma2)
        
        # Combine results (average of both methods)
        output[:, :, z] = (smoothed1 + smoothed2) / 2.0
    
    print("HKIF noise reduction completed.")
    return output


##########################
# 2. SYNTHSTRIP (SKULL) #
##########################

# Define SynthStrip model architecture
class ConvBlock(nn.Module):
    """Convolutional block with LeakyReLU activation"""
    
    def __init__(self, ndims, in_channels, out_channels, stride=1, activation='leaky'):
        super().__init__()
        
        Conv = getattr(nn, f'Conv{ndims}d')
        self.conv = Conv(in_channels, out_channels, 3, stride, 1)
        
        if activation == 'leaky':
            self.activation = nn.LeakyReLU(0.2)
        elif activation is None:
            self.activation = None
        else:
            raise ValueError(f'Unknown activation: {activation}')
            
    def forward(self, x):
        out = self.conv(x)
        if self.activation is not None:
            out = self.activation(out)
        return out

class StripModel(nn.Module):
    """SynthStrip U-Net model architecture"""
    
    def __init__(self,
                 nb_features=16,
                 nb_levels=7,
                 feat_mult=2,
                 max_features=64,
                 nb_conv_per_level=2,
                 max_pool=2,
                 return_mask=False):
                 
        super().__init__()
        
        # dimensionality
        ndims = 3
        
        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            feats = np.clip(feats, 1, max_features)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
            
        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1
        
        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels
            
        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, f'MaxPool{ndims}d')
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]
        
        # configure encoder (down-sampling path)
        prev_nf = 1
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)
            
        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if level < (self.nb_levels - 1):
                prev_nf += encoder_nfs[level]
                
        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf
            
        # final convolutions
        if return_mask:
            self.remaining.append(ConvBlock(ndims, prev_nf, 2, activation=None))
            self.remaining.append(nn.Softmax(dim=1))
        else:
            self.remaining.append(ConvBlock(ndims, prev_nf, 1, activation=None))
            
    def forward(self, x):
        
        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)
            
        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if level < (self.nb_levels - 1):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)
                
        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)
            
        return x

# Helper functions for SynthStrip
def conform_image(image_data, target_shape=None):
    """Conform image to standard orientation and voxel size"""
    # Get the current shape
    current_shape = image_data.shape
    
    # If target shape not specified, calculate it based on conforming rules
    if target_shape is None:
        # Calculate target shape divisible by 64 (as in SynthStrip)
        target_shape = np.clip(np.ceil(np.array(current_shape) / 64).astype(int) * 64, 192, 320)
    
    # Create a new array with the target shape
    conformed = np.zeros(target_shape, dtype=np.float32)
    
    # Calculate the position to place the original data
    start = [(t - c) // 2 for t, c in zip(target_shape, current_shape)]
    end = [s + c for s, c in zip(start, current_shape)]
    
    # Place the original data in the center of the new array
    slices_orig = tuple(slice(0, c) for c in current_shape)
    slices_conf = tuple(slice(s, e) for s, e in zip(start, end))
    
    conformed[slices_conf] = image_data[slices_orig]
    
    return conformed

def extend_sdt(sdt, border=1.0):
    """Extend SynthStrip's narrow-band signed distance transform (SDT)."""
    if border < np.max(sdt):
        return sdt
        
    # Find bounding box
    mask = sdt < 1
    keep = np.nonzero(mask)
    low = np.min(keep, axis=-1)
    upp = np.max(keep, axis=-1)
    
    # Add requested border
    gap = int(border + 0.5)
    low = [max(i - gap, 0) for i in low]
    upp = [min(i + gap, d - 1) for i, d in zip(upp, mask.shape)]
    
    # Compute distance within bounding box. Keep interior values.
    ind = tuple(slice(a, b + 1) for a, b in zip(low, upp))
    out = np.full_like(sdt, fill_value=100)
    
    # Compute Euclidean distance transform of the mask
    # Note: scipy's distance_transform_edt computes distance from 0s to 1s
    # We need to invert the mask to get distance from brain boundary
    inv_mask = ~mask[ind]
    distance = ndimage.distance_transform_edt(inv_mask)
    out[ind] = distance
    
    # Keep the original values inside the brain
    brain_voxels = np.nonzero(sdt <= 0)
    out[brain_voxels] = sdt[brain_voxels]
    
    return out

def get_largest_cc(mask):
    """Get the largest connected component in a binary mask"""
    # Label connected components
    labeled, num_components = ndimage.label(mask)
    
    if num_components == 0:
        return mask
        
    # Find the largest component
    sizes = ndimage.sum(mask, labeled, range(1, num_components + 1))
    largest_component = np.argmax(sizes) + 1
    
    # Return the mask with only the largest component
    return labeled == largest_component

def synthstrip_skull_stripping(image, model, device, border=1.0):
    """
    Use SynthStrip for skull stripping
    
    Parameters:
    - image: Input 3D MRI image (noise reduced)
    - model: SynthStrip model
    - device: Device to use for processing (CPU/GPU)
    - border: Brain mask border threshold in mm
    
    Returns:
    - Skull-stripped brain image and brain mask
    """
    print("Performing SynthStrip skull stripping...")
    
    try:
        # Conform the image to standard space
        print("Conforming image to standard space...")
        conformed_data = conform_image(image)
        
        # Normalize the image for SynthStrip
        print("Normalizing image...")
        conformed_data = conformed_data.astype(np.float32)
        conformed_data = conformed_data - conformed_data.min()
        p99 = np.percentile(conformed_data, 99)
        if p99 > 0:
            conformed_data = conformed_data / p99
        conformed_data = np.clip(conformed_data, 0, 1)
        
        # Prepare input tensor
        print("Preparing input tensor...")
        input_tensor = torch.from_numpy(conformed_data).to(device)
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        
        # Run inference
        print("Running SynthStrip inference...")
        with torch.no_grad():
            sdt = model(input_tensor).squeeze().cpu().numpy()
        
        # Extend the SDT
        print("Processing signed distance transform...")
        extended_sdt = extend_sdt(sdt, border=border)
        
        # Create brain mask
        print("Creating brain mask...")
        brain_mask = extended_sdt < border
        
        # Get largest connected component
        print("Finding largest connected component...")
        brain_mask = get_largest_cc(brain_mask)
        
        # Unconform the mask to original image space
        print("Transforming mask to original image space...")
        if brain_mask.shape != image.shape:
            from scipy.ndimage import zoom
            zoom_factors = [float(i) / float(j) for i, j in zip(image.shape, brain_mask.shape)]
            brain_mask = zoom(brain_mask.astype(float), zoom_factors, order=0) > 0.5
        
        # Apply the mask to the original image
        print("Applying mask to create skull-stripped image...")
        brain_image = image.copy()
        brain_image[~brain_mask] = 0
        
        print("SynthStrip skull stripping completed successfully.")
        return brain_image, brain_mask
        
    except Exception as e:
        print(f"Error in SynthStrip: {str(e)}")
        return None, None


###############################
# 3. BIAS FIELD CORRECTION (EM) #
###############################

def bias_field_correction_em(image, mask=None, max_iterations=8, wiener_filter_noise=0.01):
    """
    Bias field correction using Expectation-Maximization algorithm
    
    Parameters:
    - image: Input 3D MRI image (skull-stripped)
    - mask: Brain mask
    - max_iterations: Maximum number of EM iterations
    - wiener_filter_noise: Noise parameter for Wiener filter
    
    Returns:
    - Bias field corrected image
    """
    print("Applying bias field correction using EM algorithm...")
    
    # Use SimpleITK's N4 implementation which is based on EM algorithm
    try:
        # Convert to SimpleITK images
        sitk_image = sitk.GetImageFromArray(image.astype(np.float32))
        
        if mask is not None:
            sitk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
        else:
            sitk_mask = sitk.GetImageFromArray(np.ones_like(image, dtype=np.uint8))
        
        # Create N4 bias field corrector
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        
        # Set parameters based on the paper's EM approach
        corrector.SetMaximumNumberOfIterations([max_iterations])
        corrector.SetWienerFilterNoise(wiener_filter_noise)
        
        # Apply correction
        output = corrector.Execute(sitk_image, sitk_mask)
        corrected_image = sitk.GetArrayFromImage(output)
        print("Bias field correction completed.")
        
        return corrected_image
        
    except Exception as e:
        print(f"Warning: Bias field correction failed: {str(e)}. Using original image.")
        return image


#######################
# NPY VECTOR STORAGE #
#######################

def save_as_npy(image_data, patient_id, category, npy_dir, flatten=False, resize=None):
    """
    Save processed MRI data as .npy file
    
    Parameters:
    - image_data: 3D MRI data
    - patient_id: ID of the patient
    - category: Category (AD, MCI, CN)
    - npy_dir: Base directory to save .npy files
    - flatten: Whether to flatten the 3D data to 1D vector
    - resize: Tuple of dimensions to resize the data to
    
    Returns:
    - Path to saved .npy file
    """
    # Create category directory if it doesn't exist
    category_dir = os.path.join(npy_dir, category)
    os.makedirs(category_dir, exist_ok=True)
    
    # Prepare data
    data = image_data.copy()
    
    # Resize if specified
    if resize is not None:
        from scipy.ndimage import zoom
        # Calculate zoom factors
        zoom_factors = [float(r) / float(s) for r, s in zip(resize, data.shape)]
        # Resize data
        data = zoom(data, zoom_factors, order=1)  # order=1 for linear interpolation
    
    # Flatten if specified
    if flatten:
        data = data.flatten()
    
    # Save data
    output_path = os.path.join(category_dir, f"{patient_id}.npy")
    np.save(output_path, data)
    
    print(f"Saved .npy vector to: {output_path}")
    return output_path


####################
# VISUALIZATION #
####################

def visualize_results(original, noise_reduced, skull_stripped, final, mask, patient_id):
    """Visualize the results of tri-level preprocessing"""
    try:
        import matplotlib.pyplot as plt
        
        # Get the middle slice from each dimension
        shape = original.shape
        mid_z = shape[2] // 2  # Axial view (top-down)
        
        # Create a figure
        plt.figure(figsize=(20, 4))
        plt.suptitle(f"Patient {patient_id}: Tri-level Preprocessing Results", fontsize=16)
        
        # Original image
        plt.subplot(1, 5, 1)
        plt.imshow(original[:, :, mid_z], cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        # Noise reduced image
        plt.subplot(1, 5, 2)
        plt.imshow(noise_reduced[:, :, mid_z], cmap='gray')
        plt.title('Step 1: Noise Reduced')
        plt.axis('off')
        
        # Brain mask
        plt.subplot(1, 5, 3)
        plt.imshow(mask[:, :, mid_z], cmap='gray')
        plt.title('Brain Mask')
        plt.axis('off')
        
        # Skull-stripped image
        
        plt.subplot(1, 5, 4)
        plt.imshow(skull_stripped[:, :, mid_z], cmap='gray')
        plt.title('Step 2: Skull Stripped')
        plt.axis('off')
        
        # Final image (bias corrected)
        plt.subplot(1, 5, 5)
        plt.imshow(final[:, :, mid_z], cmap='gray')
        plt.title('Step 3: Bias Corrected')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        return True
    except Exception as e:
        print(f"Error visualizing results: {str(e)}")
        return False


####################
# MAIN EXECUTION #
####################

def main():
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Set NPY directory
    npy_dir = args.npy_dir if args.npy_dir else "processed_npy_data"
    os.makedirs(npy_dir, exist_ok=True)
    
    # Load model
    print("Loading SynthStrip model...")
    model = StripModel()
    model.to(device)
    model.eval()
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Dictionary to store visualization results
    visualization_results = {}
    
    # Process files from each category (AD, MCI, CN)
    categories = ["AD", "MCI", "CN"]
    total_processed = 0
    total_errors = 0
    
    # Track saved NPY files
    npy_files = {category: [] for category in categories}
    
    for category in categories:
        # Create source and destination paths for this category
        src_dir = os.path.join(args.input, category)
        dst_dir = os.path.join(args.output, category)
        
        # Ensure destination directory exists
        os.makedirs(dst_dir, exist_ok=True)
        
        print(f"\n{'='*50}")
        print(f"Processing {category} images...")
        print(f"{'='*50}")
        
        # Check if source directory exists
        if not os.path.exists(src_dir):
            print(f"Warning: Source directory {src_dir} not found, skipping.")
            continue
        
        # Find all .nii files recursively in the directory
        nii_files = []
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                if file.endswith('.nii'):
                    # Store the full path for later use
                    nii_files.append(os.path.join(root, file))
        
        if not nii_files:
            print(f"No .nii files found in {src_dir} or its subdirectories")
            continue
        
        print(f"Found {len(nii_files)} .nii files in {category} folder and its subdirectories")
        
        category_processed = 0
        category_errors = 0
        
        # Process each file
        for i, file_path in enumerate(nii_files):
            try:
                # Extract patient ID from path to use in output filename
                path_parts = file_path.split(os.sep)
                try:
                    # Try to get patient ID from directory structure (assuming format like "123_S_4567")
                    patient_id = None
                    for part in path_parts:
                        if part.count('_') >= 2 and any(c.isdigit() for c in part):
                            patient_id = part
                            break
                    
                    if not patient_id:
                        # If no patient ID found, use the parent directory name
                        patient_id = path_parts[-2]
                except:
                    # If any error occurs, just use a simple counter
                    patient_id = f"img_{i+1}"
                
                # Create a simplified output filename that includes category and patient ID
                base_file_name = os.path.basename(file_path)
                output_file = os.path.join(dst_dir, f"{patient_id}_{base_file_name}")
                
                print(f"\nProcessing {i+1}/{len(nii_files)}: {base_file_name}")
                print(f"Patient ID: {patient_id}")
                
                # Save intermediate results only for the first image if visualization is requested
                save_intermediate = args.visualize and category not in visualization_results
                
                # Process the file with tri-level preprocessing
                # Load the image for processing
                nii_img = nib.load(file_path)
                img_data = nii_img.get_fdata()
                affine = nii_img.affine
                
                # Check if the image is 3D
                if len(img_data.shape) > 3:
                    print(f"Input image has {len(img_data.shape)} dimensions. Using first frame/volume only.")
                    # Extract the first volume if it's a 4D image
                    img_data = img_data[..., 0]
                
                # Store original for visualization
                original_data = img_data.copy()
                
                # 1. NOISE REDUCTION
                if not args.skip_noise:
                    noise_reduced = hybrid_noise_reduction(img_data)
                else:
                    print("Skipping noise reduction (as requested).")
                    noise_reduced = img_data
                
                # 2. SKULL STRIPPING
                brain_image, brain_mask = synthstrip_skull_stripping(noise_reduced, model, device, args.border)
                
                if brain_image is None or brain_mask is None:
                    print(f"Skull stripping failed for {file_path}. Skipping to next file.")
                    category_errors += 1
                    total_errors += 1
                    continue
                
                # 3. BIAS FIELD CORRECTION
                if not args.skip_bias:
                    final_image = bias_field_correction_em(brain_image, brain_mask)
                else:
                    print("Skipping bias field correction (as requested).")
                    final_image = brain_image
                
                # Save the processed image as .nii
                print(f"Saving tri-level preprocessed image to: {output_file}")
                processed_nii = nib.Nifti1Image(final_image, affine, nii_img.header)
                nib.save(processed_nii, output_file)
                
                # Save the processed image as .npy vector
                resize_dims = args.resize if args.resize else None
                npy_path = save_as_npy(
                    final_image, 
                    patient_id, 
                    category, 
                    npy_dir, 
                    flatten=args.flatten, 
                    resize=resize_dims
                )
                npy_files[category].append(npy_path)
                
                # Store visualization results for first image in category
                if save_intermediate:
                    visualization_results[category] = {
                        'original': original_data,
                        'noise_reduced': noise_reduced,
                        'skull_stripped': brain_image,
                        'final': final_image,
                        'mask': brain_mask,
                        'patient_id': patient_id
                    }
                
                print(f"Successfully processed and saved to {output_file}")
                category_processed += 1
                total_processed += 1
                    
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                category_errors += 1
                total_errors += 1
        
        print(f"\nCategory {category} summary:")
        print(f"  - Successfully processed: {category_processed} files")
        print(f"  - Errors: {category_errors} files")
        print(f"  - NPY vectors saved: {len(npy_files[category])}")
    
    print(f"\n{'='*50}")
    print("Overall Processing Summary:")
    print(f"{'='*50}")
    print(f"Total successfully processed: {total_processed} files")
    print(f"Total errors: {total_errors} files")
    print(f"Total NPY vectors saved: {sum(len(files) for files in npy_files.values())}")
    
    # Print NPY storage details
    print(f"\nNPY vectors stored in: {npy_dir}")
    for category in categories:
        if npy_files[category]:
            print(f"  - {category}: {len(npy_files[category])} files")
    
    # Save metadata about the processing
    metadata = {
        'processing_date': str(datetime.datetime.now()),
        'total_processed': total_processed,
        'flatten': args.flatten,
        'resize': args.resize,
        'categories': {cat: len(npy_files[cat]) for cat in categories},
        'files': {cat: [os.path.basename(f) for f in npy_files[cat]] for cat in categories}
    }
    
    metadata_path = os.path.join(npy_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved processing metadata to: {metadata_path}")
    
    # Visualize the first processed image from each category if requested
    if args.visualize and visualization_results:
        print("\nVisualizing results...")
        for category, results in visualization_results.items():
            print(f"Visualizing results for category: {category}")
            visualize_results(
                results['original'],
                results['noise_reduced'],
                results['skull_stripped'],
                results['final'],
                results['mask'],
                results['patient_id']
            )
    
    print("\nTri-level preprocessing completed!")
    print("\nCitation Information:")
    print("1. HKIF Noise Reduction & Bias Field Correction:")
    print("   Gharaibeh et al. (2023). Swin Transformer-Based Segmentation and")
    print("   Multi-Scale Feature Pyramid Fusion Module for Alzheimer's Disease.")
    print("   DOI: 10.3991/ijoe.v19i04.37677")
    print("2. SynthStrip Skull Stripping:")
    print("   Hoopes A, Mora JS, Dalca AV, Fischl B, Hoffmann M. (2022).")
    print("   SynthStrip: Skull-Stripping for Any Brain Image.")
    print("   NeuroImage 260, 119474.")
    print("   https://doi.org/10.1016/j.neuroimage.2022.119474")


# Run the main function
if __name__ == "__main__":
    main()