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
    def __init__(self, root_dir, transform=None, rbf_mode=False):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.rbf_mode = rbf_mode
        self.images = []
        self.labels = []
        
        for digit in range(10):
            digit_dir = self.root_dir / str(digit)
            if not digit_dir.exists():
                print(f"Warning: Directory for digit {digit} not found")
                continue
                
            image_files = list(digit_dir.glob('*.png')) + list(digit_dir.glob('*.jpeg'))
            if not image_files:
                print(f"Warning: No images found in directory {digit_dir}")
                continue
                
            for img_path in image_files:
                self.images.append(img_path)
                self.labels.append(digit)
        
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
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('L') # grayscaling
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise
        
        if self.rbf_mode:
            image = transforms.Resize((7, 12))(image)
        else:
            image = transforms.Resize((32, 32))(image)
        
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        label = self.labels[idx]
        
        return image, label

    def get_rbf_parameters(self):
        print("\nGenerating RBF parameters...")
        centers = []
        variances = []
        
        rbf_dataset = LocalDigitDataset(self.root_dir, rbf_mode=True)
        
        digit_images = {i: [] for i in range(10)}
        for img, label in rbf_dataset:
            digit_images[label].append(img)
        
        for digit in tqdm(range(10), desc="Processing digits"):
            if not digit_images[digit]:
                print(f"Warning: No images found for digit {digit}")
                continue
                
            digit_tensors = torch.stack(digit_images[digit])
            center = digit_tensors.mean(dim=0)
            centers.append(center)
            variance = ((digit_tensors - center) ** 2).mean(dim=0)
            variances.append(variance)
        
        return torch.stack(centers), torch.stack(variances)

class MNISTDataset(Dataset):
    def __init__(self, train=True, transform=None):
        self.train = train
        self.transform = transform
        
        dataset = load_dataset('mnist', split='train' if train else 'test')
        
        self.images = dataset['image']
        self.labels = dataset['label']
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.Compose([
                transforms.Pad(2),
                transforms.ToTensor(),
            ])(image)
        
        return image, label

def get_data_loaders(batch_size=1, train_transform=None, test_transform=None):
    
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    if test_transform is None:
        test_transform = transforms.Compose([
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    train_dataset = MNISTDataset(train=True, transform=train_transform)
    test_dataset = MNISTDataset(train=False, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
