import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from LeNet import LeNet5
from data import get_data_loaders
import os

def test_model(model, test_loader):
    """
    Test the original LeNet5 model and generate performance metrics.
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
    plt.title('Confusion Matrix')
    
    # Save confusion matrix
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/confusion_matrix.png')
    plt.close()
    
    # Plot most confusing examples
    plt.figure(figsize=(15, 10))
    for i in range(10):
        if most_confusing[i]['image'] is not None:
            plt.subplot(2, 5, i+1)
            plt.imshow(most_confusing[i]['image'].squeeze(), cmap='gray')
            plt.title(f'True: {most_confusing[i]["true_label"]}\nPred: {most_confusing[i]["pred_label"]}')
            plt.axis('off')
    
    plt.suptitle('Most Confusing Examples')
    plt.savefig('results/most_confusing.png')
    plt.close()
    
    return error_rate, confusion_matrix, most_confusing

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load model
    model_path = 'LeNet1.pth'
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train the model first.")
        return
    
    checkpoint = torch.load(model_path)
    model = LeNet5()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get test loader
    _, test_loader = get_data_loaders(batch_size=1, use_local=True)
    
    # Test model
    error_rate, confusion_matrix, most_confusing = test_model(model, test_loader)
    
    # Print results
    print(f"\nTest Error Rate: {error_rate:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix)
    print("\nMost Confusing Examples:")
    for i in range(10):
        if most_confusing[i]['image'] is not None:
            print(f"Digit {i}:")
            print(f"  True Label: {most_confusing[i]['true_label']}")
            print(f"  Predicted Label: {most_confusing[i]['pred_label']}")
            print(f"  Confidence: {most_confusing[i]['confidence']:.4f}")

if __name__ == "__main__":
    main()
