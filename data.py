import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from datasets import load_dataset
import numpy as np
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm

class LocalDigitDataset(Dataset):
    """
    Dataset class for local digit images.
    Can convert images to either 7x12 bitmap format (for RBF) or 32x32 (for LeNet5).
    """
    def __init__(self, root_dir, transform=None, rbf_mode=False):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.rbf_mode = rbf_mode
        self.images = []
        self.labels = []
        
        print("Loading local digit dataset...")
        # Load images from each digit directory
        for digit in range(10):
            digit_dir = self.root_dir / str(digit)
            if not digit_dir.exists():
                print(f"Warning: Directory for digit {digit} not found")
                continue
                
            # Look for both PNG and JPEG files
            image_files = list(digit_dir.glob('*.png')) + list(digit_dir.glob('*.jpeg'))
            if not image_files:
                print(f"Warning: No images found in directory {digit_dir}")
                continue
                
            for img_path in image_files:
                self.images.append(img_path)
                self.labels.append(digit)
            
            print(f"Loaded {len(image_files)} images for digit {digit}")
        
        if not self.images:
            raise ValueError(f"No images found in {self.root_dir}. Please check the directory structure.")
        
        print(f"\nTotal images loaded: {len(self.images)}")
        print("Class distribution:")
        for digit in range(10):
            count = sum(1 for label in self.labels if label == digit)
            print(f"Digit {digit}: {count} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('L')  # Convert to grayscale
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise
        
        # Resize based on mode
        if self.rbf_mode:
            # For RBF: resize to 7x12
            image = transforms.Resize((7, 12))(image)
        else:
            # For LeNet5: resize to 32x32
            image = transforms.Resize((32, 32))(image)
        
        # Apply transform if specified
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform: convert to tensor and normalize
            image = transforms.ToTensor()(image)
        
        # Get label
        label = self.labels[idx]
        
        return image, label

    def get_rbf_parameters(self):
        """
        Generate RBF parameters from the dataset.
        Returns:
            centers: The centers for each digit class
            variances: The variances for each digit class
        """
        print("\nGenerating RBF parameters...")
        centers = []
        variances = []
        
        # Create a temporary dataset in RBF mode
        rbf_dataset = LocalDigitDataset(self.root_dir, rbf_mode=True)
        
        # Group images by digit
        digit_images = {i: [] for i in range(10)}
        for img, label in rbf_dataset:
            digit_images[label].append(img)
        
        # Calculate centers and variances for each digit
        for digit in tqdm(range(10), desc="Processing digits"):
            if not digit_images[digit]:
                print(f"Warning: No images found for digit {digit}")
                continue
                
            # Stack all images for this digit
            digit_tensors = torch.stack(digit_images[digit])
            
            # Calculate center (mean)
            center = digit_tensors.mean(dim=0)
            centers.append(center)
            
            # Calculate variance
            variance = ((digit_tensors - center) ** 2).mean(dim=0)
            variances.append(variance)
            
            print(f"Digit {digit}: Processed {len(digit_images[digit])} images")
        
        return torch.stack(centers), torch.stack(variances)

class MNISTDataset(Dataset):
    def __init__(self, train=True, transform=None):
        self.train = train
        self.transform = transform
        
        # Load dataset from HuggingFace
        dataset = load_dataset('mnist', split='train' if train else 'test')
        
        # Store images and labels
        self.images = dataset['image']
        self.labels = dataset['label']
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get PIL Image and label
        image = self.images[idx]
        label = self.labels[idx]
        
        # Apply transform if specified
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform: pad to 32x32, convert to tensor and normalize
            image = transforms.Compose([
                transforms.Pad(2),  # Pad 2 pixels on each side to get 32x32
                transforms.ToTensor(),
            ])(image)
        
        return image, label

def get_data_loaders(batch_size=1, use_local=False):
    """
    Get data loaders for training and testing.
    If use_local is True, uses the local digit dataset instead of MNIST.
    """
    if use_local:
        # Transform for local digits: convert to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # Create datasets (not in RBF mode for training)
        train_dataset = LocalDigitDataset('digits', transform=transform, rbf_mode=False)
        test_dataset = LocalDigitDataset('digits', transform=transform, rbf_mode=False)
    else:
        # Transform: pad to 32x32 and convert to tensor
        transform = transforms.Compose([
            transforms.Pad(2),  # Pad 2 pixels on each side to get 32x32
            transforms.ToTensor(),
        ])
        
        # Create datasets
        train_dataset = MNISTDataset(train=True, transform=transform)
        test_dataset = MNISTDataset(train=False, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
