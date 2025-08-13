"""
Enhanced CNN + ViT Hybrid Model for Alzheimer's Disease Prediction (Fixed for PyTorch 2.6.0)

This script implements an advanced hybrid CNN + Vision Transformer (ViT) model to classify 3D MRI images
into Alzheimer's Disease (AD), Cognitively Normal (CN), and Mild Cognitive Impairment (MCI) classes.
It uses all available scans from the ADNI dataset and is optimized for RTX 3090 GPUs.
"""

# Fix matplotlib backend to avoid Tkinter errors
import matplotlib
matplotlib.use('Agg')  # Use the non-interactive Agg backend

import os
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from monai.transforms import (
    RandRotate,
    RandFlip,
    RandAdjustContrast,
    RandZoom,
    RandAffine,
    Compose,
    ToTensor,
)
import torch.nn.functional as F
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define standalone functions to replace lambdas (for multiprocessing compatibility)
def convert_to_float32(x):
    return x.to(dtype=torch.float32)

# Custom Dataset for .npy files with ADNI filename parsing and subfolder label inference
class ADNIDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Verify directory exists
        if not self.data_dir.exists():
            logger.error(f"Data directory {self.data_dir} does not exist!")
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist!")
        
        # List all .npy files
        self.files = list(self.data_dir.rglob("*.npy"))
        if not self.files:
            logger.error(f"No .npy files found in {self.data_dir}!")
            raise FileNotFoundError(f"No .npy files found in {self.data_dir}!")
        logger.info(f"Found {len(self.files)} .npy files in {self.data_dir}")
        
        # Skip deduplication to use all scans
        # Infer labels from subfolders (case-insensitive)
        self.labels = []
        self.subject_ids = []  # Store subject IDs for potential group-wise splitting
        for npy_file in self.files:
            file_name = npy_file.name
            parts = file_name.split('_')
            if len(parts) < 5:
                logger.warning(f"Skipping file {file_name}: not enough parts in filename")
                continue
            subject_id = f"{parts[1]}_{parts[2]}_{parts[3]}"  # e.g., 002_S_0619
            self.subject_ids.append(subject_id)
            
            path_str = str(npy_file).replace('\\', '/')
            path_str_lower = path_str.lower()
            
            if '/ad/' in path_str_lower:
                self.labels.append(0)  # AD
            elif '/cn/' in path_str_lower:
                self.labels.append(1)  # CN
            elif '/mci/' in path_str_lower:
                self.labels.append(2)  # MCI
            else:
                logger.warning(f"Unknown label for file {npy_file}")
                self.labels.append(-1)  # Unknown
        
        self.label_map = {0: 'AD', 1: 'CN', 2: 'MCI'}
        logger.info(f"Dataset size: {len(self.files)} samples")
        logger.info(f"Label distribution: AD={self.labels.count(0)}, CN={self.labels.count(1)}, MCI={self.labels.count(2)}, Unknown={self.labels.count(-1)}")

        # Check if all labels are unknown (likely no subfolders)
        if self.labels.count(-1) == len(self.files):
            logger.error("All labels are unknown! Subfolders (AD/, CN/, MCI/) may not exist. Please provide a label CSV file or organize files into subfolders.")
            raise ValueError("All labels are unknown. Ensure files are in AD/, CN/, MCI/ subfolders or provide a label CSV file.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        npy_file = self.files[idx]
        label = self.labels[idx]
        
        # Load preprocessed .npy file with error handling
        try:
            image = np.load(npy_file)
            image = torch.tensor(image, dtype=torch.float32)
        except Exception as e:
            logger.error(f"Failed to load {npy_file}: {e}")
            raise
        
        # Ensure channel dimension (if not present)
        if image.ndim == 3:  # Shape: (H, W, D)
            image = image.unsqueeze(0)  # Shape: (C, H, W, D)
        elif image.ndim != 4:
            logger.error(f"Unexpected shape for {npy_file}: {image.shape}")
            raise ValueError(f"Expected 3D or 4D tensor, got shape {image.shape}")
        
        # Validate expected spatial dimensions
        expected_shape = (1, 128, 128, 128)
        if image.shape != expected_shape:
            logger.error(f"Shape mismatch for {npy_file}: expected {expected_shape}, got {image.shape}")
            raise ValueError(f"Shape mismatch: expected {expected_shape}, got {image.shape}")
        
        # Apply transforms (augmentation) if specified
        if self.transform:
            image = self.transform(image)
        
        return image, label

#-------------------------#
# NEW ENHANCED ARCHITECTURE
#-------------------------#

# 3D ResNet Block with residual connection
class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out

# Squeeze-and-Excitation (SE) attention for 3D data
class SELayer3D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

# Enhanced 3D CNN Backbone with ResNet blocks and SE attention
class Enhanced3DCNN(nn.Module):
    def __init__(self, in_channels=1, base_filters=32):
        super(Enhanced3DCNN, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv3d(in_channels, base_filters, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(base_filters, base_filters, blocks=2)
        self.layer2 = self._make_layer(base_filters, base_filters*2, blocks=2, stride=2)
        self.layer3 = self._make_layer(base_filters*2, base_filters*4, blocks=2, stride=2)
        
        # SE Attention modules after each residual block
        self.se1 = SELayer3D(base_filters)
        self.se2 = SELayer3D(base_filters*2)
        self.se3 = SELayer3D(base_filters*4)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResBlock3D(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResBlock3D(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Input shape: [B, 1, 128, 128, 128]
        x = self.conv1(x)       # [B, 32, 64, 64, 64]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # [B, 32, 32, 32, 32]
        
        x = self.layer1(x)      # [B, 32, 32, 32, 32]
        x = self.se1(x)
        
        x = self.layer2(x)      # [B, 64, 16, 16, 16]
        x = self.se2(x)
        
        x = self.layer3(x)      # [B, 128, 8, 8, 8]
        x = self.se3(x)
        
        return x

# 3D Attention mechanism for Vision Transformer
class Attention3D(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# MLP block for Transformer
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Transformer Encoder Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention3D(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

# 3D Vision Transformer for MRI volume patches
class VisionTransformer3D(nn.Module):
    def __init__(self, volume_size=(8, 8, 8), patch_size=2, in_channels=128, dim=512, 
                 depth=6, heads=8, mlp_dim=1024, pool='cls', dropout=0.1, emb_dropout=0.1):
        super().__init__()
        
        # Calculate number of patches
        patch_dim = in_channels * (patch_size ** 3)
        num_patches = (volume_size[0] // patch_size) * (volume_size[1] // patch_size) * (volume_size[2] // patch_size)
        
        # Patch embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)', 
                       p1=patch_size, p2=patch_size, p3=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        
        # CLS token and position embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        # Transformer blocks
        self.transformer = nn.Sequential(*[
            TransformerBlock(dim, heads, dim // heads, mlp_dim, dropout) 
            for _ in range(depth)
        ])
        
        # Output head
        self.pool = pool
        self.to_latent = nn.Identity()
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, dim // 4)
        )
        
    def forward(self, x):
        # x shape: [batch_size, channels, depth, height, width]
        
        # Convert volume to patches
        x = self.to_patch_embedding(x)  # [batch_size, num_patches, dim]
        b, n, _ = x.shape
        
        # Add CLS token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        
        # Apply transformer blocks
        x = self.transformer(x)
        
        # Pool according to strategy
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        
        # Get latent vector
        x = self.to_latent(x)
        
        # Apply MLP head
        return self.mlp_head(x)

# Enhanced Hybrid CNN-ViT Model for Alzheimer's Disease Classification
class EnhancedCNNViTHybrid(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.3):
        super(EnhancedCNNViTHybrid, self).__init__()
        
        # CNN feature extractor
        self.cnn = Enhanced3DCNN(in_channels=1, base_filters=32)
        
        # Vision Transformer for global feature extraction
        self.vit = VisionTransformer3D(
            volume_size=(8, 8, 8),  # Size after CNN
            patch_size=2,           # Each patch is 2x2x2
            in_channels=128,        # Output channels from CNN
            dim=512,                # Transformer embedding dimension
            depth=6,                # Number of transformer blocks
            heads=8,                # Number of attention heads
            mlp_dim=1024,           # MLP hidden dimension
            pool='cls',             # Use CLS token for classification
            dropout=0.1             # Dropout rate
        )
        
        # Feature fusion module
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128, 256),  # Combine CNN global features with ViT features
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2)
        )
        
        # CNN global pooling for feature fusion
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # Extract CNN features
        cnn_features = self.cnn(x)  # [B, 128, 8, 8, 8]
        
        # Global CNN features for fusion
        cnn_global = self.global_pool(cnn_features).view(x.size(0), -1)  # [B, 128]
        
        # ViT features
        vit_features = self.vit(cnn_features)  # [B, 128]
        
        # Feature fusion
        fused_features = torch.cat([cnn_global, vit_features], dim=1)  # [B, 256]
        fused_features = self.fusion(fused_features)  # [B, 128]
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits

#-------------------------#
# TRAINING FUNCTIONS
#-------------------------#

def train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device):
    model.train()
    train_loss = 0.0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Check if using GPU with mixed precision
        if device.type == 'cuda':
            # For PyTorch 2.6.0: Use torch.cuda.amp.autocast() without device_type
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            # CPU training without mixed precision
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
        # Step the scheduler
        scheduler.step()
        
        train_loss += loss.item()
        
        # Print batch progress
        if (batch_idx + 1) % 5 == 0:
            logger.info(f"Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    return train_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_preds, val_true = [], []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_true.extend(labels.cpu().numpy())
    
    val_accuracy = accuracy_score(val_true, val_preds)
    return val_loss / len(val_loader), val_accuracy, val_preds, val_true

#-------------------------#
# MAIN EXECUTION
#-------------------------#

def main():
    # Paths
    data_dir = "H:\\k214827\\vector_data"  # Confirmed path
    output_dir = "H:\\k214827\\classification_results_enhanced_hybrid"
    os.makedirs(output_dir, exist_ok=True)
    
    # Force CUDA device if available, otherwise use CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA is not available, using CPU instead. This will be much slower!")
    
    logger.info(f"Using device: {device}")
    
    # Define transforms with proper functions (not lambdas, for multiprocessing)
    train_transform = Compose([
        RandRotate(range_x=0.2, range_y=0.2, range_z=0.2, prob=0.5),
        RandFlip(spatial_axis=0, prob=0.5),
        RandFlip(spatial_axis=1, prob=0.5),
        RandFlip(spatial_axis=2, prob=0.5),
        RandAdjustContrast(prob=0.5, gamma=(0.7, 1.3)),
        RandZoom(prob=0.3, min_zoom=0.8, max_zoom=1.2),
        RandAffine(prob=0.3, translate_range=0.1, scale_range=(0.9, 1.1)),
        ToTensor(dtype=torch.float32)  # Use ToTensor instead of lambda
    ])
    
    val_transform = Compose([
        ToTensor(dtype=torch.float32)  # Use ToTensor instead of lambda
    ])
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = ADNIDataset(data_dir, transform=None)
    
    # Training parameters
    num_epochs = 100
    batch_size = 8  # For RTX 3090
    
    # Use 0 workers with newer Python versions to avoid multiprocessing issues
    # If you're using Python 3.13, it's safer to use 0 workers
    num_workers = 0  
    
    logger.info(f"Batch size: {batch_size}, Number of workers: {num_workers}")
    
    # 5-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracies = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dataset)), dataset.labels)):
        logger.info(f"Training fold {fold + 1}/5")
        
        # Split dataset
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        
        # Apply transforms
        train_subset.dataset.transform = train_transform
        val_subset.dataset.transform = val_transform
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        # Initialize model
        model = EnhancedCNNViTHybrid(num_classes=3, dropout_rate=0.3).to(device)
        
        # Compute class weights for imbalance
        labels = np.array(dataset.labels)
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        
        # Loss function with label smoothing
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        
        # Optimizer with weight decay
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=2e-4, 
            epochs=num_epochs, 
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            div_factor=10.0,
            final_div_factor=100.0
        )
        
        # Gradient scaler for mixed precision training (only if using CUDA)
        if device.type == 'cuda':
            # For PyTorch 2.6.0: Use torch.cuda.amp.GradScaler() without device_type
            scaler = torch.cuda.amp.GradScaler()
        else:
            # Dummy scaler for CPU
            class DummyScaler:
                def scale(self, loss):
                    return loss
                def step(self, optimizer):
                    pass
                def update(self):
                    pass
            scaler = DummyScaler()
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(num_epochs):
            # Train one epoch
            train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device)
            train_losses.append(train_loss)
            
            # Validate
            val_loss, val_accuracy, val_preds, val_true = validate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            logger.info(f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}, "
                       f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), os.path.join(output_dir, f"best_model_fold_{fold + 1}.pth"))
                logger.info(f"Model saved at epoch {epoch + 1}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
        
        # Plot training and validation loss
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Fold {fold + 1} Training and Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'loss_curve_fold_{fold + 1}.png'))
        plt.close()
        
        # Plot confusion matrix
        cm = confusion_matrix(val_true, val_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['AD', 'CN', 'MCI'], yticklabels=['AD', 'CN', 'MCI'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Fold {fold + 1} Confusion Matrix')
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_fold_{fold + 1}.png'))
        plt.close()
        
        # Plot accuracy curve
        plt.figure(figsize=(10, 5))
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Fold {fold + 1} Validation Accuracy')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'accuracy_curve_fold_{fold + 1}.png'))
        plt.close()
        
        # Load best model and evaluate
        model.load_state_dict(torch.load(os.path.join(output_dir, f"best_model_fold_{fold + 1}.pth")))
        _, fold_accuracy, val_preds, val_true = validate(model, val_loader, criterion, device)
        
        report = classification_report(val_true, val_preds, target_names=['AD', 'CN', 'MCI'], zero_division=1)
        logger.info(f"Fold {fold + 1} Final Accuracy: {fold_accuracy:.4f}")
        logger.info(f"Fold {fold + 1} Classification Report:\n{report}")
        cv_accuracies.append(fold_accuracy)
        
        # Save classification report
        with open(os.path.join(output_dir, f"classification_report_fold_{fold + 1}.txt"), "w") as f:
            f.write(f"Fold {fold + 1} Final Accuracy: {fold_accuracy:.4f}\n")
            f.write(f"Classification Report:\n{report}")
    
    # Summarize cross-validation results
    mean_accuracy = np.mean(cv_accuracies)
    std_accuracy = np.std(cv_accuracies)
    logger.info(f"Mean CV Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy * 2:.4f})")
    
    # Save overall results
    with open(os.path.join(output_dir, "classification_results_enhanced_hybrid.txt"), "w") as f:
        f.write(f"Mean CV Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy * 2:.4f})\n")
        f.write("Per-fold accuracies:\n")
        for fold, acc in enumerate(cv_accuracies, 1):
            f.write(f"Fold {fold}: {acc:.4f}\n")
    
    logger.info(f"Classification results saved to {os.path.join(output_dir, 'classification_results_enhanced_hybrid.txt')}")
    logger.info("Training completed!")

if __name__ == "__main__":
    main()