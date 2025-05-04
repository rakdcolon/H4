import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from LeNet import LeNet5
from data import get_data_loaders, LocalDigitDataset
import os
import time
import seaborn as sns
from tqdm import tqdm
from torchvision import transforms

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def train_model(model, train_loader, test_loader, epochs=20, lr=0.001):
    device = get_device()
    print(f"Using device: {device}")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    train_errors = []
    test_errors = []
    
    most_confusing = {i: {'confidence': -1, 'image': None, 'true_label': None, 'pred_label': None} for i in range(10)}
    
    # Create results directory
    os.makedirs('results/1', exist_ok=True)
    
    start_time = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # forward
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # backward
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*train_correct/train_total:.2f}%'}) # progress bar
        
        train_error = 1.0 - train_correct / train_total
        train_errors.append(train_error)
        
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        confusion_matrix = np.zeros((10, 10), dtype=int)
        
        test_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{epochs} [Test]')
        with torch.no_grad():
            for inputs, targets in test_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
                
                for t, p in zip(targets.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                
                for i in range(len(targets)):
                    true_label = targets[i].item()
                    pred_label = predicted[i].item()
                    confidence = outputs[i][pred_label].item()
                    
                    if true_label != pred_label and confidence > most_confusing[true_label]['confidence']:
                        most_confusing[true_label]['confidence'] = confidence
                        most_confusing[true_label]['image'] = inputs[i].cpu()
                        most_confusing[true_label]['true_label'] = true_label
                        most_confusing[true_label]['pred_label'] = pred_label
                
                test_pbar.set_postfix({'acc': f'{100.*test_correct/test_total:.2f}%'}) # progress bar
        
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
        plt.savefig('results/1/error_rates.png')
        plt.close()
        
        scheduler.step(test_loss)
        
        epoch_time = time.time() - epoch_start
        print(f'\nEpoch {epoch+1}/{epochs} Summary:')
        print(f'  Train Error: {train_error:.4f}')
        print(f'  Test Error: {test_error:.4f}')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'  Epoch Time: {epoch_time:.2f}s')
    
    total_time = time.time() - start_time
    print(f'\nTotal Training Time: {total_time:.2f}s')
    
    # Plot most confusing examples in a single grid
    plt.figure(figsize=(15, 10))
    for i in range(10):
        if most_confusing[i]['image'] is not None:
            plt.subplot(2, 5, i+1)
            plt.imshow(most_confusing[i]['image'].squeeze(), cmap='gray')
            plt.title(f'True: {most_confusing[i]["true_label"]}\nPred: {most_confusing[i]["pred_label"]}')
            plt.axis('off')
    
    plt.suptitle('Most Confusing Examples')
    plt.tight_layout()
    plt.savefig('results/1/most_confusing.png')
    plt.close()
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': 'LeNet5'
    }, 'LeNet1.pth')
    print("Saved model to LeNet1.pth")
    
    return model, train_errors, test_errors, confusion_matrix, most_confusing

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    test_transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    train_loader, test_loader = get_data_loaders(
        batch_size=32,
        train_transform=train_transform,
        test_transform=test_transform
    )
    
    dataset = LocalDigitDataset('digits')
    centers, variances = dataset.get_rbf_parameters()
    print("\nRBF Parameters:")
    print(f"Centers shape: {centers.shape}")
    print(f"Variances shape: {variances.shape}")
    
    model = LeNet5()
    model, train_errors, test_errors, confusion_matrix, most_confusing = train_model(model, train_loader, test_loader)
    
    print(f"\nFinal Training Error: {train_errors[-1]:.4f}")
    print(f"Final Test Error: {test_errors[-1]:.4f}")

if __name__ == "__main__":
    main() 