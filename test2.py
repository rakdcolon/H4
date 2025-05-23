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
from data import get_data_loaders, MNISTDataset

def get_transformed_test_loader(transform):
    test_dataset = MNISTDataset(train=False, transform=transform)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )
    
    return test_loader

def test_model(model, test_loader, transform_name='Standard'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    confusion_matrix = np.zeros((10, 10), dtype=int)
    most_confusing = {i: {'confidence': -1, 'image': None, 'true_label': None, 'pred_label': None} for i in range(10)}
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            _, predicted = outputs.max(1)
            
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
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    error_rate = 1.0 - correct / total
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {transform_name}')

    os.makedirs('results', exist_ok=True)
    os.makedirs('results/2', exist_ok=True)
    plt.savefig(f'results/2/confusion_matrix_{transform_name}.png')
    plt.close()
    
    plt.figure(figsize=(15, 10))
    for i in range(10):
        if most_confusing[i]['image'] is not None:
            plt.subplot(2, 5, i+1)
            plt.imshow(most_confusing[i]['image'].squeeze(), cmap='gray')
            plt.title(f'True: {most_confusing[i]["true_label"]}\nPred: {most_confusing[i]["pred_label"]}')
            plt.axis('off')
    
    plt.suptitle(f'Most Confusing Examples - {transform_name}')
    plt.savefig(f'results/2/most_confusing_{transform_name}.png')
    plt.close()
    
    return error_rate, confusion_matrix, most_confusing

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    model_path = 'LeNet2.pth'
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train the model first.")
        return
    
    checkpoint = torch.load(model_path)
    model = LeNet5Modified()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    transformations = {
        'Standard': transforms.Compose([
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]),
        'Rotation20': transforms.Compose([
            transforms.Pad(2),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]),
        'Rotation40': transforms.Compose([
            transforms.Pad(2),
            transforms.RandomRotation(40),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]),
        'Shift': transforms.Compose([
            transforms.Pad(2),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]),
        'Scale': transforms.Compose([
            transforms.Pad(2),
            transforms.RandomAffine(0, scale=(0.4, 1.4)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    }
    
    results = {}
    for name, transform in transformations.items():
        print(f"\nTesting with {name} transformation...")
        _, test_loader = get_data_loaders(
            batch_size=64,
            test_transform=transform
        )
        error_rate, confusion_matrix, most_confusing = test_model(model, test_loader, name)
        results[name] = {
            'error_rate': error_rate,
            'confusion_matrix': confusion_matrix,
            'most_confusing': most_confusing
        }
        
        print(f"Error Rate: {error_rate:.4f}")
        print("\nMost Confusing Examples:")
        for i in range(10):
            if most_confusing[i]['image'] is not None:
                print(f"Digit {i}:")
                print(f"  True Label: {most_confusing[i]['true_label']}")
                print(f"  Predicted Label: {most_confusing[i]['pred_label']}")
                print(f"  Confidence: {most_confusing[i]['confidence']:.4f}")
    
    print("\nSummary of Results:")
    for name, result in results.items():
        print(f"{name}: Error Rate = {result['error_rate']:.4f}")

if __name__ == "__main__":
    main()