import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LeNet5(nn.Module):
    """
    Original LeNet5 implementation as described in the paper.
    Uses average pooling and tanh activation.
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        
        # Layer C1: 6 feature maps, 5x5 kernel
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        
        # Layer S2: 6 feature maps, 2x2 kernel, stride 2
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Layer C3: 16 feature maps, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        
        # Layer S4: 16 feature maps, 2x2 kernel, stride 2
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Layer C5: 120 feature maps, 5x5 kernel
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        
        # Layer F6: 84 units
        self.fc1 = nn.Linear(120, 84)
        
        # Output layer: 10 units (digits 0-9)
        self.fc2 = nn.Linear(84, 10)
        
        # Initialize weights as described in the paper
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Initialize with uniform distribution between -2.4/F and 2.4/F
                # where F is the fan-in
                fan_in = m.weight.size(1) * m.weight.size(2) * m.weight.size(3)
                bound = 2.4 / np.sqrt(fan_in)
                nn.init.uniform_(m.weight, -bound, bound)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # Initialize with uniform distribution between -2.4/F and 2.4/F
                fan_in = m.weight.size(1)
                bound = 2.4 / np.sqrt(fan_in)
                nn.init.uniform_(m.weight, -bound, bound)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # C1: Convolution + tanh
        x = torch.tanh(self.conv1(x))
        
        # S2: Average pooling
        x = self.pool1(x)
        
        # C3: Convolution + tanh
        x = torch.tanh(self.conv2(x))
        
        # S4: Average pooling
        x = self.pool2(x)
        
        # C5: Convolution + tanh
        x = torch.tanh(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # F6: Fully connected + tanh
        x = torch.tanh(self.fc1(x))
        
        # Output layer
        x = self.fc2(x)
        
        return x

class LeNet5Modified(nn.Module):
    """
    Modified LeNet5 with modern improvements:
    - Max pooling instead of average pooling
    - ReLU activation instead of tanh
    - Batch normalization
    - Dropout
    """
    def __init__(self):
        super(LeNet5Modified, self).__init__()
        
        # Layer C1: 6 feature maps, 5x5 kernel
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)  # Add padding to maintain size
        self.bn1 = nn.BatchNorm2d(6)
        
        # Layer S2: 6 feature maps, 2x2 kernel, stride 2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer C3: 16 feature maps, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=2)  # Add padding to maintain size
        self.bn2 = nn.BatchNorm2d(16)
        
        # Layer S4: 16 feature maps, 2x2 kernel, stride 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer C5: 120 feature maps, 3x3 kernel (reduced from 5x5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=3, padding=1)  # Use 3x3 kernel with padding
        self.bn3 = nn.BatchNorm2d(120)
        
        # Layer F6: 84 units
        self.fc1 = nn.Linear(120 * 8 * 8, 84)  # Adjusted for 8x8 output from conv3
        self.dropout = nn.Dropout(0.5)
        
        # Output layer: 10 units (digits 0-9)
        self.fc2 = nn.Linear(84, 10)
        
        # Initialize weights using Kaiming initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # C1: Convolution + BN + ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        
        # S2: Max pooling
        x = self.pool1(x)
        
        # C3: Convolution + BN + ReLU
        x = F.relu(self.bn2(self.conv2(x)))
        
        # S4: Max pooling
        x = self.pool2(x)
        
        # C5: Convolution + BN + ReLU
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # F6: Fully connected + ReLU + Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Output layer
        x = self.fc2(x)
        
        return x 