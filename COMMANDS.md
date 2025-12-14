# Команды для работы с Tenstorrent

## Подключение к серверу

```bash
# SSH подключение
ssh ekaterina_kuzmina1@10.30.0.207

# Или через VS Code:
# Cmd+Shift+P -> "Remote-SSH: Connect to Host..."
# Ввести: ekaterina_kuzmina1@10.30.0.207
```

## Активация окружения

```bash
cd ~/tenstorrent
source .venv/bin/activate
```

## Выбор устройства (ВАЖНО!)

**Доступные устройства:** 0, 1, 2, 3 (локальные), 4, 5, 6, 7 (remote)

1. **Сначала напиши в Teams чат "TT Hardware Access":**
   ```
   I'll use device 2
   ```

2. **Проверить доступные устройства:**
   ```bash
   python -c "import ttnn; print(ttnn.get_device_ids())"
   ```

3. **Проверить что устройство работает:**
   ```bash
   python -c "import ttnn; d = ttnn.open_device(2); print('OK'); ttnn.close_device(d)"
   ```

## Запуск inference на Tenstorrent

### MNIST (MLP модель)
```bash
python -m mnist.inference_ttnn --device_id 2
```

### CIFAR-10 (CNN модель)
```bash
python -m cifar.inference_ttnn --device_id 2
```

### С другими параметрами
```bash
# Другой checkpoint
python -m mnist.inference_ttnn --device_id 2 --checkpoint weights_mnist/best_model.ckpt

# Больше samples
python -m mnist.inference_ttnn --device_id 2 --num_samples 100

# Другой batch size
python -m mnist.inference_ttnn --device_id 2 --batch_size 64
```

## Обучение моделей (PyTorch)

```bash
# MNIST
python -m mnist.train

# CIFAR-10
python -m cifar.train
```

## Git команды

```bash
# Статус
git status

# Добавить все изменения
git add .

# Коммит
git commit -m "описание изменений"

# Push
git push
```

## Отключение

```bash
# 1. Выйти из venv
deactivate

# 2. Отключиться от SSH
exit
# или Ctrl+D

# 3. В VS Code:
# Cmd+Shift+P -> "Remote: Close Remote Connection"
```

## Полезные команды

```bash
# Посмотреть GPU/устройства
tt-smi

# Список установленных пакетов
pip list | grep -i tt

# Проверить версию PyTorch
python -c "import torch; print(torch.__version__)"
```

## Структура проекта

```
~/tenstorrent/
├── mnist/
│   ├── model.py           # MLP модель
│   ├── train.py           # Обучение
│   ├── utils.py           # DataModule
│   └── inference_ttnn.py  # Inference на TT
├── cifar/
│   ├── model.py           # CNN модель
│   ├── train.py           # Обучение
│   ├── utils.py           # DataModule
│   └── inference_ttnn.py  # Inference на TT
├── weights_mnist/         # Сохранённые веса MNIST
├── weights_cifar/         # Сохранённые веса CIFAR
├── info/                  # PDF с заданиями
└── COMMANDS.md            # Этот файл
```
