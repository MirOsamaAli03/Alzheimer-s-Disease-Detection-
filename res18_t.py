import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# Fix for matplotlib Tcl/Tk error - add this at the top
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend that doesn't require Tcl/Tk
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import argparse
from datetime import datetime

# Define ArgumentParser for command-line arguments
parser = argparse.ArgumentParser(description='Train 3D CNN for Alzheimer\'s disease detection')
parser.add_argument('--data_dir', type=str, default='./vector_data', help='Directory containing the MRI data')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=70, help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save results')
parser.add_argument('--model_path', type=str, default=None, help='Path to saved model to continue training')
parser.add_argument('--class_names', type=str, default='CN,MCI,AD', help='Class names separated by commas')
parser.add_argument('--val_size', type=float, default=0.15, help='Validation set size (0-1)')
parser.add_argument('--test_size', type=float, default=0.15, help='Test set size (0-1)')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--freeze_layers', type=int, default=0, help='Number of initial layers to freeze')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training (faster)')
parser.add_argument('--evaluate_only', action='store_true', help='Only evaluate on test set, no training')

# Parse arguments
args = parser.parse_args()

# Set random seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Configure GPU settings for optimal performance
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes (faster)
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere GPUs (faster with minimal precision loss)
    torch.backends.cudnn.allow_tf32 = True

# Set up mixed precision training if requested - UPDATED for different PyTorch versions
if args.mixed_precision:
    try:
        scaler = torch.amp.GradScaler('cuda')  # New syntax
    except:
        try:
            scaler = torch.amp.GradScaler()  # Middle version syntax
        except:
            scaler = torch.cuda.amp.GradScaler()  # Old version fallback
else:
    scaler = None

# Print GPU information
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("No GPU available, using CPU")

# Define custom Dataset for loading .npy MRI volumes
class MRIDataset(Dataset):
    def __init__(self, root_dir, classes=None, transform=None):
        if classes is None:
            classes = args.class_names.split(',')
            
        self.paths = []
        self.labels = []
        self.class_names = classes
        
        # Print data directory for debugging
        print(f"Loading data from: {root_dir}")

        # Collect file paths and labels
        for i, cls in enumerate(classes):
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                print(f"Warning: Directory not found: {cls_dir}")
                continue

            files = sorted(os.listdir(cls_dir))
            npy_files = [f for f in files if f.endswith('.npy')]
            print(f"Found {len(npy_files)} .npy files in {cls_dir}")

            for fname in npy_files:
                self.paths.append(os.path.join(cls_dir, fname))
                self.labels.append(i)

        print(f"Dataset initialized with {len(self.paths)} total samples")
        
        # Check class distribution
        unique_classes, counts = np.unique(self.labels, return_counts=True)
        for cls_idx, count in zip(unique_classes, counts):
            print(f"Class {classes[cls_idx]}: {count} samples")
            
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Load volume and normalize
        try:
            vol = np.load(self.paths[idx]).astype(np.float32)
            
            # Normalize each volume to zero mean and unit variance
            vol = (vol - np.mean(vol)) / (np.std(vol) + 1e-7)
            
            if vol.ndim != 3:
                raise ValueError(f"Expected 3D volume, got shape {vol.shape}")
                
            # Add channel dimension and replicate to 3 channels (for pretrained models)
            vol = np.expand_dims(vol, axis=0)         # shape (1, D, H, W)
            vol = np.repeat(vol, 3, axis=0)           # shape (3, D, H, W)
            
            if self.transform:
                vol = self.transform(vol)
                
            label = self.labels[idx]
            return torch.from_numpy(vol), label
            
        except Exception as e:
            print(f"Error loading file {self.paths[idx]}: {str(e)}")
            # Return a zero tensor as fallback to prevent training crash
            return torch.zeros((3, 128, 128, 128), dtype=torch.float32), self.labels[idx]


# 3D CNN model definition - Using ResNet3D architecture
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet3D(nn.Module):
    def __init__(self, block, num_blocks, num_classes=3):
        super(ResNet3D, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet18_3D(num_classes=3):
    return ResNet3D(ResidualBlock, [2, 2, 2, 2], num_classes)

# Utility function to get the current time as a string
def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# Function to save training plots
def save_training_plots(train_losses, val_accs, current_epoch, output_dir, timestamp):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses)+1), train_losses, marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(val_accs)+1), val_accs, marker='o', color='orange')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(output_dir, f"training_curves_{timestamp}.png")
    plt.savefig(save_path)
    plt.close()
    
    print(f"Training plots saved to {save_path}")

# Function to generate saliency maps for interpretability
def generate_saliency_maps(model, data_loader, device, class_names, output_dir, timestamp, dataset_type="validation"):
    print(f"Generating saliency maps for {dataset_type} samples...")
    
    # Ensure model is in eval mode
    model.eval()
    
    # Process multiple samples
    num_samples = min(5, len(data_loader))
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    # Make sure axes is always a 2D array, even with single sample
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    samples_processed = 0
    data_iter = iter(data_loader)
    
    for sample_idx in range(num_samples):
        try:
            # Get next batch
            sample_imgs, sample_labels = next(data_iter)
            
            # Process first image in batch
            sample = sample_imgs[0].unsqueeze(0).to(device)  # shape (1,3,D,H,W)
            sample.requires_grad = True
            
            # Forward pass
            outputs = model(sample)
            pred_class = outputs.argmax(1).item()
            pred_class_name = class_names[pred_class]
            true_class_name = class_names[sample_labels[0].item()]
            
            # Backpropagate for saliency
            model.zero_grad()
            score = outputs[0, pred_class]
            score.backward()
            
            # Get saliency map
            saliency = sample.grad.detach().abs()  # (1,3,D,H,W)
            saliency = saliency.mean(dim=1)[0].cpu().numpy()  # shape (D,H,W)
            
            # Get middle slices
            D, H, W = saliency.shape
            mid_d, mid_h, mid_w = D//2, H//2, W//2
            
            # Original MRI - axial view
            axes[sample_idx, 0].imshow(sample.detach().cpu().numpy()[0, 0, mid_d, :, :], cmap='gray')
            axes[sample_idx, 0].set_title(f'Original MRI (Axial)')
            axes[sample_idx, 0].axis('off')
            
            # Saliency maps - three views
            axes[sample_idx, 1].imshow(saliency[mid_d, :, :], cmap='hot')
            axes[sample_idx, 1].set_title(f'Axial Saliency')
            axes[sample_idx, 1].axis('off')
            
            axes[sample_idx, 2].imshow(saliency[:, mid_h, :], cmap='hot')
            axes[sample_idx, 2].set_title(f'Coronal Saliency')
            axes[sample_idx, 2].axis('off')
            
            axes[sample_idx, 3].imshow(saliency[:, :, mid_w], cmap='hot')
            axes[sample_idx, 3].set_title(f'Sagittal Saliency')
            axes[sample_idx, 3].axis('off')
            
            # Add text annotation for prediction
            axes[sample_idx, 0].text(5, 5, f'True: {true_class_name}, Pred: {pred_class_name}', 
                            bbox=dict(facecolor='white', alpha=0.8),
                            horizontalalignment='left', verticalalignment='top')
            
            samples_processed += 1
            
        except StopIteration:
            # No more samples in validation loader
            for j in range(4):
                axes[sample_idx, j].axis('off')
                if j == 0:
                    axes[sample_idx, j].text(0.5, 0.5, 'No more samples', 
                                horizontalalignment='center', verticalalignment='center')
            break
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{dataset_type}_saliency_maps_{timestamp}.png")
    plt.savefig(save_path)
    plt.close()
    
    print(f"Saliency maps saved to {save_path}")

# Function to evaluate model on test set
def evaluate_test_set(model, test_loader, device, class_names, output_dir, timestamp):
    print("\n" + "="*50)
    print("EVALUATING ON TEST SET")
    print("="*50)
    
    model.eval()
    test_preds = []
    test_labels = []
    test_probs = []  # Store probabilities for ROC curve
    
    # Process test set
    test_pbar = tqdm(test_loader, desc="Testing")
    with torch.no_grad():
        for imgs, labels in test_pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(imgs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            # Collect predictions and labels
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    test_acc = 100 * accuracy_score(test_labels, test_preds)
    cm = confusion_matrix(test_labels, test_preds)
    report = classification_report(test_labels, test_preds, target_names=class_names, zero_division=0)
    
    # Display results
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    
    # Create a more detailed report with per-class metrics
    class_correct = {}
    class_total = {}
    
    for i, label in enumerate(test_labels):
        pred = test_preds[i]
        if label not in class_correct:
            class_correct[label] = 0
            class_total[label] = 0
        
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1
    
    print("\nDetailed Class Accuracy:")
    for label in sorted(class_total.keys()):
        accuracy = 100 * class_correct[label] / class_total[label]
        print(f"  - {class_names[label]}: {accuracy:.2f}% ({class_correct[label]}/{class_total[label]})")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save the confusion matrix plot
    cm_path = os.path.join(output_dir, f"test_confusion_matrix_{timestamp}.png")
    plt.savefig(cm_path)
    plt.close()
    
    print(f"Test confusion matrix saved to {cm_path}")
    
    # Save test results to a text file
    results_path = os.path.join(output_dir, f"test_results_{timestamp}.txt")
    with open(results_path, 'w') as f:
        f.write(f"Test Accuracy: {test_acc:.2f}%\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification Report:\n")
        f.write(report)
        f.write("\n\nDetailed Class Accuracy:\n")
        for label in sorted(class_total.keys()):
            accuracy = 100 * class_correct[label] / class_total[label]
            f.write(f"  - {class_names[label]}: {accuracy:.2f}% ({class_correct[label]}/{class_total[label]})\n")
    
    print(f"Test results saved to {results_path}")
    
    # Generate saliency maps for test samples
    generate_saliency_maps(model, test_loader, device, class_names, output_dir, timestamp, "test")
    
    return test_acc, cm

# Main training function
def train_model():
    # Start timing
    start_time = time.time()
    
    # Create dataset
    print("Creating dataset...")
    full_dataset = MRIDataset(args.data_dir)
    
    # Check if dataset is empty
    if len(full_dataset) == 0:
        raise ValueError("Dataset is empty! Check the data path and class folders.")
    
    # Create a three-way split: train, validation, and test
    # First, split into temp and test sets
    temp_idx, test_idx = train_test_split(
        np.arange(len(full_dataset)),
        test_size=args.test_size,
        stratify=full_dataset.labels,
        random_state=args.seed
    )
    
    # Then split temp into train and validation
    val_ratio = args.val_size / (1 - args.test_size)
    train_idx, val_idx = train_test_split(
        temp_idx,
        test_size=val_ratio,
        stratify=[full_dataset.labels[i] for i in temp_idx],
        random_state=args.seed
    )
    
    # Create dataset subsets
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)
    
    print(f"Train set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")
    
    # Use class weights to handle class imbalance
    class_counts = np.bincount(full_dataset.labels)
    class_weights = torch.tensor(sum(class_counts) / class_counts, dtype=torch.float32)
    print(f"Class weights: {class_weights}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True  # Faster data transfer to GPU
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Initialize model
    print("Initializing model...")
    try:
        # Try using torchvision's r3d_18 first
        from torchvision.models.video import r3d_18
        model = r3d_18(pretrained=False)  # Don't download weights again
        # Replace final layer for our number of classes
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, len(full_dataset.class_names))
        print("Using torchvision's r3d_18 architecture")
    except:
        # Fall back to our custom ResNet3D
        model = ResNet18_3D(num_classes=len(full_dataset.class_names))
        print("Using custom ResNet18_3D implementation")
    
    # Load saved model if provided
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading weights from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path))
        print("Model weights loaded successfully!")
    else:
        print("No model path provided or model not found. Starting with pre-trained weights.")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # If evaluate only mode, just run test set evaluation and exit
    if args.evaluate_only:
        timestamp = get_timestamp()
        evaluate_test_set(model, test_loader, device, full_dataset.class_names, args.output_dir, timestamp)
        return
    
    # Freeze early layers if specified
    if args.freeze_layers > 0:
        print(f"Freezing first {args.freeze_layers} layers...")
        layers_to_freeze = [
            'conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4'
        ][:args.freeze_layers]
        
        for name, param in model.named_parameters():
            for layer in layers_to_freeze:
                if layer in name:
                    param.requires_grad = False
                    print(f"Frozen: {name}")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.1%} of total)")
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Store initial learning rate for comparison
    old_lr = optimizer.param_groups[0]['lr']
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    train_losses = []
    val_accs = []
    epoch_times = []
    
    timestamp = get_timestamp()
    model_save_path = os.path.join(args.output_dir, f"best_model_{timestamp}.pth")
    
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        train_preds = []
        train_labels = []
        
        # Track per-class metrics
        train_class_correct = [0] * len(full_dataset.class_names)
        train_class_total = [0] * len(full_dataset.class_names)
        
        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        
        for imgs, labels in train_pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass - with mixed precision if enabled
            if scaler is not None:
                try:
                    with torch.amp.autocast('cuda'):  # New recommended syntax
                        outputs = model(imgs)
                        loss = criterion(outputs, labels)
                    
                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                except:
                    with torch.cuda.amp.autocast():  # Fallback for older PyTorch
                        outputs = model(imgs)
                        loss = criterion(outputs, labels)
                    
                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                # Standard forward/backward pass
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Stats tracking
            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            
            # Store predictions and labels for metrics
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            # Track per-class accuracy
            for i in range(len(full_dataset.class_names)):
                mask = (labels == i)
                train_class_correct[i] += (preds[mask] == i).sum().item()
                train_class_total[i] += mask.sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100 * accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
            })
        
        # Calculate training metrics
        train_loss = running_loss / len(train_dataset)
        train_acc = 100 * accuracy_score(train_labels, train_preds)
        train_losses.append(train_loss)
        
        # Per-class training accuracy
        train_class_acc = [100 * c / max(t, 1) for c, t in zip(train_class_correct, train_class_total)]
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []
        
        # Progress bar for validation
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]")
        
        with torch.no_grad():
            for imgs, labels in val_pbar:
                imgs = imgs.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(imgs)
                _, preds = outputs.max(1)
                
                # Update metrics
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                val_pbar.set_postfix({
                    'acc': 100 * accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
                })
        
        # Calculate validation accuracy
        val_acc = 100 * accuracy_score(val_labels, val_preds)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Check if learning rate changed and manually print it
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != old_lr:
            print(f"Learning rate changed from {old_lr:.2e} to {current_lr:.2e}")
            old_lr = current_lr
        
        # Confusion matrix and classification report
        cm = confusion_matrix(val_labels, val_preds)
        report = classification_report(
            val_labels, 
            val_preds,
            target_names=full_dataset.class_names,
            zero_division=0
        )
        
        # Calculate epoch time
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        epoch_times.append(epoch_time)
        
        # Print epoch results
        print(f"\nEpoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.1f}% | Val Acc={val_acc:.1f}%")
        print(f"Per-class train acc: " + ", ".join([f"{c}={a:.1f}%" for c, a in zip(full_dataset.class_names, train_class_acc)]))
        print(f"Current learning rate: {current_lr:.2e}")
        print(f"Epoch completed in {epoch_time:.1f} seconds")
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(report)
        
        # Check for improvement and save model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved to {model_save_path} with validation accuracy: {val_acc:.1f}%")
            
            # Also save as best_model.pth for easier reference
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
            
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
        
        # Early stopping check
        if patience_counter >= args.patience:
            print(f"Early stopping after {epoch} epochs due to no improvement.")
            break
        
        # Save training progress plot after each epoch
        save_training_plots(train_losses, val_accs, epoch, args.output_dir, timestamp)
        
        # Save intermediate model every 5 epochs
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"model_epoch_{epoch}_{timestamp}.pth"))
            print(f"Saved checkpoint at epoch {epoch}")
    
    # Final training summary
    total_time = time.time() - start_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    
    print("\n" + "="*50)
    print("Training Summary:")
    print(f"Total training time: {total_time/60:.1f} minutes")
    print(f"Average epoch time: {avg_epoch_time:.1f} seconds")
    print(f"Best validation accuracy: {best_val_acc:.1f}%")
    print(f"Best model saved to: {model_save_path}")
    print("="*50)
    
    # Generate final interpretability visualizations
    generate_saliency_maps(model, val_loader, device, full_dataset.class_names, args.output_dir, timestamp, "validation")
    
    # Load the best model for test evaluation
    print("Loading best model for test evaluation...")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pth")))
    
    # Evaluate on test set
    evaluate_test_set(model, test_loader, device, full_dataset.class_names, args.output_dir, timestamp)

# Entry point
if __name__ == "__main__":
    # Print arguments
    print("Running with arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    
    # Run training
    train_model()