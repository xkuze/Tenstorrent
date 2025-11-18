## Structure
- `mnist/` - Task 1: MNIST digit classification with MLP
- `cifar/` - Task 2: CIFAR-10 image classification with CNN

The code between these two folders is very similar. I basically copied and adapted it for each task. Ideally I would have created a common module with shared functions for both tasks but I didn't want to spend time on that. I could also increase the number of TRIALS for the models but I settled on five for faster training completion and result validation.

I should have also written better docstrings but I decided not to spend time on that. If shared modules or more detailed docstrings are important for you, I can add them to the code.

I used uv as the package manager and ruff for linting.

I split the data into train/val/test with 90/10 split (train from validation) plus a separate test set. More detailed data split information is in each task's README.

I didn't implement explicit layer analysis or dimension tracking code. PyTorch Lightning automatically handles model summaries and shows layer dimensions, parameter counts and architecture details during training. For fast prototyping I relied on Lightning's built-in analysis rather than building custom layer inspection tools. If there was a separate research phase requirement with deeper model analysis needs, I would add more detailed layer-by-layer analysis, feature map visualization and dimension tracking.

## Weights

Model weights are saved in:
- `weights_mnist/` - for MNIST task
- `weights_cifar/` - for CIFAR-10 task

## Task Details

Each folder has its own README describing what was implemented from the Excel requirements.
