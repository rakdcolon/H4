# MNIST Digit Recognition Models

This repository contains two trained models for MNIST digit recognition:
1. Model 1: Standard LeNet
2. Model 2: Modified LeNet with data augmentation

## Prerequisites

- Python 3.x
- Required Python packages:
  ```bash
  pip install torch torchvision numpy matplotlib seaborn pillow
  ```

## Required Files

- `LeNet1.pth` or `LeNet2.pth` (trained model files)
- `test1.py` or `test2.py` (test scripts)
- `data.py` (data loader module)
- `LeNet.py` (model definition)

## Running Inference

### Model 1 (Standard LeNet-5)

To run inference on Model 1:
```bash
python test1.py
```

This will:
1. Load the trained model from `LeNet1.pth`
2. Run inference on the test set
3. Generate results in `results/1/`:
   - `confusion_matrix.png`: Shows the confusion matrix
   - `most_confusing.png`: Shows the most confusing examples in a grid

### Model 2 (Modified LeNet-5)

To run inference on Model 2:
```bash
python test2.py
```

This will:
1. Load the trained model from `LeNet2.pth`
2. Run inference on the test set with different transformations
3. Generate results in `results/2/`:
   - `confusion_matrix_Standard.png`: Confusion matrix for standard test
   - `confusion_matrix_Rotation20.png`: Confusion matrix for 20° rotation
   - `confusion_matrix_Rotation40.png`: Confusion matrix for 40° rotation
   - `confusion_matrix_Shift.png`: Confusion matrix for shifted images
   - `confusion_matrix_Scale.png`: Confusion matrix for scaled images
   - Similar `most_confusing_*.png` files for each transformation

## Notes

- The test scripts assume the test data is available in the expected format (DIGIT data set is in digit/* where each subdirectory is the corresponding digit)