from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import torchvision.transforms as transforms
import seaborn as sns
from LeNet import LeNet5Modified
from data import LocalDigitDataset

def get_transformed_test_loader(transform):
    """
    Get a test loader with the specified transformation.
    """
    # Create dataset with the transform
    test_dataset = LocalDigitDataset('digits', transform=transform, rbf_mode=False)
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )
    
    return test_loader

def test_model(model, test_loader, transform_name='Standard'):
    """
    Test the modified LeNet5 model with transformed data.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Initialize confusion matrix
    confusion_matrix = np.zeros((10, 10), dtype=int)
    
    # Track most confusing examples
    most_confusing = {i: {'confidence': -1, 'image': None, 'true_label': None, 'pred_label': None} 
                     for i in range(10)}
    
    # Track statistics
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Get predictions
            _, predicted = outputs.max(1)
            
            # Update confusion matrix
            for t, p in zip(targets.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            
            # Update most confusing examples
            for i in range(len(targets)):
                true_label = targets[i].item()
                pred_label = predicted[i].item()
                confidence = outputs[i][pred_label].item()
                
                if true_label != pred_label and confidence > most_confusing[true_label]['confidence']:
                    most_confusing[true_label]['confidence'] = confidence
                    most_confusing[true_label]['image'] = inputs[i].cpu()
                    most_confusing[true_label]['true_label'] = true_label
                    most_confusing[true_label]['pred_label'] = pred_label
            
            # Update accuracy
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    # Calculate error rate
    error_rate = 1.0 - correct / total
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {transform_name}')
    
    # Save confusion matrix
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/confusion_matrix_{transform_name}.png')
    plt.close()
    
    # Plot most confusing examples
    plt.figure(figsize=(15, 10))
    for i in range(10):
        if most_confusing[i]['image'] is not None:
            plt.subplot(2, 5, i+1)
            plt.imshow(most_confusing[i]['image'].squeeze(), cmap='gray')
            plt.title(f'True: {most_confusing[i]["true_label"]}\nPred: {most_confusing[i]["pred_label"]}')
            plt.axis('off')
    
    plt.suptitle(f'Most Confusing Examples - {transform_name}')
    plt.savefig(f'results/most_confusing_{transform_name}.png')
    plt.close()
    
    return error_rate, confusion_matrix, most_confusing

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load model
    model_path = 'LeNet2.pth'
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train the model first.")
        return
    
    checkpoint = torch.load(model_path)
    model = LeNet5Modified()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Define different transformations
    transformations = {
        'Standard': transforms.Compose([
            transforms.ToTensor(),
        ]),
        'Rotation10': transforms.Compose([
            transforms.RandomRotation(30, fill=255),
            transforms.ToTensor(),
        ]),
        'Rotation30': transforms.Compose([
            transforms.RandomRotation(60, fill=255),
            transforms.ToTensor(),
        ]),
        'Shift': transforms.Compose([
            transforms.RandomAffine(0, translate=(0.1, 0.1), fill=255),
            transforms.ToTensor(),
        ]),
        'Scale': transforms.Compose([
            transforms.RandomAffine(0, scale=(0.7, 1.3), fill=255),
            transforms.ToTensor(),
        ])
    }
    
    # Test model with each transformation
    results = {}
    for name, transform in transformations.items():
        print(f"\nTesting with {name} transformation...")
        test_loader = get_transformed_test_loader(transform)
        error_rate, confusion_matrix, most_confusing = test_model(model, test_loader, name)
        results[name] = {
            'error_rate': error_rate,
            'confusion_matrix': confusion_matrix,
            'most_confusing': most_confusing
        }
        
        # Print results
        print(f"Error Rate: {error_rate:.4f}")
        print("\nMost Confusing Examples:")
        for i in range(10):
            if most_confusing[i]['image'] is not None:
                print(f"Digit {i}:")
                print(f"  True Label: {most_confusing[i]['true_label']}")
                print(f"  Predicted Label: {most_confusing[i]['pred_label']}")
                print(f"  Confidence: {most_confusing[i]['confidence']:.4f}")
    
    # Print summary
    print("\nSummary of Results:")
    for name, result in results.items():
        print(f"{name}: Error Rate = {result['error_rate']:.4f}")

if __name__ == "__main__":
    main()
