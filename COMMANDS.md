# Commands for working with Tenstorrent

## Connecting to the server

```bash
# SSH connection
ssh ekaterina_kuzmina1@10.30.147.170

# Or via VS Code:
# Cmd+Shift+P -> "Remote-SSH: Connect to Host..."
# Enter: ekaterina_kuzmina1@10.30.147.170
```

## Activating the environment

```bash
cd ~/tenstorrent
source .venv/bin/activate
```

## Selecting a device (IMPORTANT!)

**Available devices:** 0, 1, 2, 3 (local), 4, 5, 6, 7 (remote)

1. **First, write in Teams chat "TT Hardware Access":**
   ```
   I'll use device 2
   ```

2. **Check available devices:**
   ```bash
   python -c "import ttnn; print(ttnn.get_device_ids())"
   ```

3. **Check that the device works:**
   ```bash
   python -c "import ttnn; d = ttnn.open_device(2); print('OK'); ttnn.close_device(d)"
   ```

## Running inference on Tenstorrent

### MNIST (MLP model)
```bash
python -m mnist.inference_ttnn --device_id 2
```

### CIFAR-10 (CNN model)
```bash
python -m cifar.inference_ttnn --device_id 2
```

### With other parameters
```bash
# Different checkpoint
python -m mnist.inference_ttnn --device_id 2 --checkpoint weights_mnist/best_model.ckpt

# More samples
python -m mnist.inference_ttnn --device_id 2 --num_samples 100

# Different batch size
python -m mnist.inference_ttnn --device_id 2 --batch_size 64
```

## Training models (PyTorch)

```bash
# MNIST
python -m mnist.train

# CIFAR-10
python -m cifar.train
```

## Git commands

```bash
# Status
git status

# Add all changes
git add .

# Commit
git commit -m "description of changes"

# Push
git push
```

## Disconnecting

```bash
# 1. Exit venv
deactivate

# 2. Disconnect from SSH
exit
# or Ctrl+D

# 3. In VS Code:
# Cmd+Shift+P -> "Remote: Close Remote Connection"
```

## Useful commands

```bash
# View GPU/devices
tt-smi

# List installed packages
pip list | grep -i tt

# Check PyTorch version
python -c "import torch; print(torch.__version__)"
```

## Project structure

```
~/tenstorrent/
├── mnist/
│   ├── model.py           # MLP model
│   ├── train.py           # Training
│   ├── utils.py           # DataModule
│   └── inference_ttnn.py  # Inference on TT
├── cifar/
│   ├── model.py           # CNN model
│   ├── train.py           # Training
│   ├── utils.py           # DataModule
│   └── inference_ttnn.py  # Inference on TT
├── weights_mnist/         # Saved MNIST weights
├── weights_cifar/         # Saved CIFAR weights
├── info/                  # PDF with assignments
└── COMMANDS.md            # This file
```
