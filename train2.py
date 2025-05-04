import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from LeNet import LeNet5Modified
from data import get_data_loaders
import os
import time
from torchvision import transforms

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def train_model(model, train_loader, test_loader, epochs=1000, lr=0.001, early_stopping_patience=5):
    device = get_device()
    print(f"Using device: {device}")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    train_errors = []
    test_errors = []
    
    # For early stopping
    best_test_error = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    # Create results directory
    os.makedirs('results/2', exist_ok=True)

    start_time = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        train_error = 1.0 - train_correct / train_total
        train_errors.append(train_error)
        
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        test_error = 1.0 - test_correct / test_total
        test_errors.append(test_error)
        
        # Plot error rates after each epoch
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epoch+2), train_errors, 'b-', label='Training Error')
        plt.plot(range(1, epoch+2), test_errors, 'r-', label='Test Error')
        plt.title('Error Rates vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Error Rate')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/2/error_rates.png')
        plt.close()
        
        # Early stopping
        if test_error < best_test_error:
            best_test_error = test_error
            best_model_state = model.state_dict()  # Save the best model
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}. Best test error: {best_test_error:.4f}")
                break
        
        scheduler.step(test_loss) # change learning rate if necessary
        
        epoch_time = time.time() - epoch_start
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Error: {train_error:.4f}')
        print(f'  Test Error: {test_error:.4f}')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'  Epoch Time: {epoch_time:.2f}s')
    
    total_time = time.time() - start_time
    print(f'\nTotal Training Time: {total_time:.2f}s')
    
    # Get best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': 'LeNet5Modified'
    }, 'LeNet2.pth')
    print("Saved model to LeNet2.pth")
    
    return model, train_errors, test_errors

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Define transforms with augmentation for training
    train_transform = transforms.Compose([
        transforms.Pad(2),
        transforms.RandomRotation(20),  # Add rotation
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Add shift
        transforms.RandomAffine(0, scale=(0.4, 1.4)),  # Add scale
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    # Define transforms for testing (no augmentation)
    test_transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    train_loader, test_loader = get_data_loaders(
        batch_size=64,
        train_transform=train_transform,
        test_transform=test_transform
    )
    
    model = LeNet5Modified()
    model, train_errors, test_errors = train_model(model, train_loader, test_loader, epochs=100, early_stopping_patience=5)
    
    print(f"\nFinal Training Error: {train_errors[-1]:.4f}")
    print(f"Final Test Error: {test_errors[-1]:.4f}")
    print(f"Total epochs run: {len(train_errors)}")

if __name__ == "__main__":
    main() 