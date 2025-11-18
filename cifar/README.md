# CIFAR-10 CNN - Task 2

## What I Implemented

- **Architecture**: Convolutional neural network with configurable layers
- **Framework**: PyTorch Lightning
- **Hyperparameter Search**: Optuna with TPE algorithm
- **Data Split**: 45k train / 5k validation / 10k test
- **Optimization**: Adam, SGD, RMSprop

## Architecture Details

CNN structure: Conv2d -> ReLU -> MaxPool2d -> Dropout -> Flatten -> FC layers

Each convolutional layer:

- Configurable number of filters (32, 64, 128, 256)
- Kernel size (3x3 or 5x5)
- MaxPooling reduces spatial dimensions by 2
- Dropout for regularization

### Hyperparameter Search

Optuna tries different configurations:

- Number of convolutional layers (2-4)
- Number of filters per layer (32, 64, 128, 256)
- Kernel size (3 or 5)
- Dropout rate (0.0-0.5)
- Learning rate (0.0001-0.01)
- Batch size (32, 64, 128)
- Optimizer (Adam, SGD, RMSprop)

It runs 5 trials with 5 epochs each, then picks the best configuration.

### Training Flow

1. Run hyperparameter search (5 trials x 5 epochs)
2. Find best configuration based on validation accuracy
3. Train final model with best params for 20 epochs
4. Use early stopping if validation accuracy stops improving
5. Save best model checkpoint
6. Test on test set and report final accuracy

### Results

- Best validation accuracy: 76.3%
- Test accuracy: 75.36%
- Model saved to: weights_cifar/best_model.ckpt
