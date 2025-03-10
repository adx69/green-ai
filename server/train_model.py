import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from torchvision import transforms
from unet_model import UNet
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import sys
import time
from tqdm import tqdm

# Add flush=True to all print statements for real-time output
def progress_print(message):
    print(message, flush=True)

# Use this simpler progress bar function if you don't want to add tqdm
def progress_bar(iterable, desc="Processing", total=None):
    """Create a text-based progress bar"""
    if total is None and hasattr(iterable, "__len__"):
        total = len(iterable)
    
    for i, item in enumerate(iterable):
        if total:
            percent = (i + 1) / total * 100
            bar_length = 30
            filled_length = int(bar_length * (i + 1) / total)
            bar = '#' * filled_length + '-' * (bar_length - filled_length)
            
            # Print progress bar with carriage return to update in-place
            print(f"\r{desc}: |{bar}| {percent:.1f}% ({i+1}/{total})", end="", flush=True)
            
            # Print newline on completion
            if i+1 == total:
                print()
                
        yield item

class SaplingDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        # Ensure images are in the correct format (N, C, H, W)
        if images.shape[1] != 3:  # If channels are not in the correct position
            images = np.transpose(images, (0, 3, 1, 2))

        self.images = torch.FloatTensor(images)
        self.labels = torch.FloatTensor(labels)
        self.transform = transform

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images)

def load_data(data_path):
    progress_print(f"Searching for data in {data_path}")
    data_files = [f for f in os.listdir(data_path) if f.endswith('.npy')]
    progress_print(f"Found {len(data_files)} data files")
    
    all_data = []
    for i, file in enumerate(progress_bar(data_files, desc="Loading files")):
        try:
            data = np.load(os.path.join(data_path, file))
            if len(data.shape) == 4:
                all_data.append(data)
                progress_print(f"  - Loaded shape: {data.shape}")
        except Exception as e:
            progress_print(f"  - Error loading {file}: {e}")

    if not all_data:
        progress_print("No valid data found!")
        return None

    data = np.concatenate(all_data, axis=0)
    progress_print(f"Combined data shape: {data.shape}")
    return data

def create_dummy_labels(data):
    num_samples = data.shape[0]
    height = data.shape[2]  # Height is the third dimension after transpose
    width = data.shape[3]  # Width is the fourth dimension after transpose

    progress_print(f"Creating dummy labels for {num_samples} samples...")
    
    # Create binary masks for each image
    labels = np.zeros((num_samples, 1, height, width), dtype=np.float32)

    # Add some random sapling locations
    for i in progress_bar(range(num_samples), desc="Generating labels"):
        # Create 5-10 random sapling locations per image
        num_saplings = np.random.randint(5, 11)
        for _ in range(num_saplings):
            # Random position
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            # Create a small circular mask for each sapling
            for dy in range(-5, 6):
                for dx in range(-5, 6):
                    if dx*dx + dy*dy <= 25:  # Circle with radius 5
                        nx, ny = x+dx, y+dy
                        if 0 <= nx < width and 0 <= ny < height:
                            labels[i, 0, ny, nx] = 1.0
    
    progress_print("Dummy labels created successfully")
    return labels

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

def train_model(data_path, model_path):
    progress_print("Starting model training...")
    
    # Get the current time for tracking duration
    start_time = time.time()
    
    # Check for CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    progress_print(f"Using device: {device}")
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    progress_print("Loading data...")
    # For a quick test, we'll create some dummy data if no data path is provided
    if data_path and os.path.exists(data_path):
        images = load_data(data_path)
        if images is None:
            progress_print("No valid data found, creating dummy data")
            # Create dummy data (10 samples of 256x256x3 images)
            images = np.random.rand(10, 3, 256, 256).astype(np.float32)
    else:
        progress_print("Data path not provided or doesn't exist, creating dummy data")
        # Create dummy data (10 samples of 256x256x3 images)
        images = np.random.rand(10, 3, 256, 256).astype(np.float32)
    
    progress_print("Creating labels...")
    labels = create_dummy_labels(images)
    
    # Split data into train and validation sets
    progress_print("Splitting data into train/validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )
    
    progress_print(f"Train set: {X_train.shape[0]} samples")
    progress_print(f"Validation set: {X_val.shape[0]} samples")
    
    # Create datasets and dataloaders
    train_dataset = SaplingDataset(X_train, y_train)
    val_dataset = SaplingDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    # Create UNet model
    progress_print("Creating UNet model...")
    model = UNet(in_channels=3, out_channels=1)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
    
    # Training loop
    num_epochs = 5  # Use a small number for testing
    progress_print(f"Starting training for {num_epochs} epochs...")
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        progress_print(f"\nEpoch {epoch+1}/{num_epochs}")
        progress_print("=" * 40)
        progress_print("Training:")
        
        # Use progress bar for training
        batch_bar = progress_bar(enumerate(train_loader), 
                                desc=f"Train Epoch {epoch+1}", 
                                total=len(train_loader))
        
        for i, (inputs, targets) in batch_bar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        progress_print(f"\nTraining Loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        progress_print("Validation:")
        with torch.no_grad():
            # Use progress bar for validation
            val_bar = progress_bar(enumerate(val_loader), 
                                desc=f"Val Epoch {epoch+1}", 
                                total=len(val_loader))
                                
            for i, (inputs, targets) in val_bar:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        progress_print(f"\nValidation Loss: {avg_val_loss:.4f}")
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            progress_print(f"Saving improved model (loss: {best_val_loss:.4f})")
            torch.save(model.state_dict(), model_path)
    
    # Calculate training duration
    duration = time.time() - start_time
    progress_print(f"\nTraining completed in {duration:.2f} seconds")
    progress_print(f"Best validation loss: {best_val_loss:.4f}")
    progress_print(f"Model saved to: {model_path}")
    
    return True

# Replace the emoji characters with ASCII alternatives
if __name__ == "__main__":
    progress_print("\n" + "="*60)
    progress_print("STARTING GREEN-AI MODEL TRAINING")
    progress_print("="*60 + "\n")
    
    # Set default paths
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_data")
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    model_path = os.path.join(model_dir, "unet_model.pth")
    
    progress_print(f"Data directory: {data_dir}")
    progress_print(f"Model will be saved to: {model_path}")
    
    success = train_model(data_dir, model_path)
    
    if success:
        progress_print("\n[SUCCESS] Training completed successfully!")
        sys.exit(0)
    else:
        progress_print("\n[FAILED] Training failed!")
        sys.exit(1)