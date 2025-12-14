# MNIST MLP - Task 1

## What I Implemented

- **Architecture**: Multi-layer perceptron with configurable hidden layers
- **Framework**: PyTorch Lightning
- **Hyperparameter Search**: Optuna with TPE algorithm
- **Data Split**: 54k train / 6k validation / 10k test
- **Optimization**: Adam, SGD, RMSprop

## Hyperparameter Search

Optuna tries different configurations:

- Number of hidden layers (1-3)
- Layer sizes (64, 128, 256, 512 neurons)
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

- Best validation accuracy: 97.9%
- Test accuracy: 97.76%
- Model saved to: weights_mnist/best_model.ckpt
