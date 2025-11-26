# Simple Training Setup - Quick Start Guide

## Overview

Created a simple, non-distributed training setup using raw Tiny-ImageNet data (no WebDataset) to validate the model and training process.

## What Was Done

### 1. Created Training Script
[`train_simple.py`](file:///Users/danieltorosoto/universidad/arq-cliente-servidor/asynchronous_distributed_training_imagenet1k/train_simple.py) - Simple PyTorch training using `ImageFolder`

**Features**:
- Uses raw Tiny-ImageNet data directly
- Standard PyTorch DataLoader (no WebDataset complexity)
- Proper train/validation split
- Checkpoint saving (best and last)
- Learning rate scheduling

### 2. Reorganized Validation Data
[`reorganize_val_data.py`](file:///Users/danieltorosoto/universidad/arq-cliente-servidor/asynchronous_distributed_training_imagenet1k/reorganize_val_data.py) - Converts validation structure

**Before**:
```
val/
  images/
    val_0.JPEG
    val_1.JPEG
    ...
  val_annotations.txt
```

**After**:
```
val/
  n03444034/
    val_0.JPEG
    ...
  n04067472/
    val_1.JPEG
    ...
```

## Running Training

### Basic Usage
```bash
python train_simple.py --epochs 10
```

### Full Options
```bash
python train_simple.py \
  --data_dir data/tiny-imagenet-200 \
  --batch_size 64 \
  --epochs 10 \
  --lr 0.001 \
  --num_workers 4 \
  --checkpoint_dir checkpoints_simple
```

### Current Test Run
```bash
python train_simple.py --epochs 2 --batch_size 64 --num_workers 2
```

**Status**: Running (Epoch 1, ~150/1563 batches)
- Training samples: 100,000
- Validation samples: 10,000
- Classes: 200
- Initial loss: ~5.4 (normal)
- Current accuracy: ~1% (expected for early training)

## Expected Results

### Training Metrics
- **Epoch 1**: Loss ~4.5-5.0, Acc ~2-5%
- **Epoch 5**: Loss ~3.5-4.0, Acc ~10-20%
- **Epoch 10**: Loss ~3.0-3.5, Acc ~25-40%

### Validation Metrics
- **Should be similar to training** (within 0.5-1.0 for loss)
- **NOT millions** like before
- Accuracy should gradually improve

## Key Differences from Distributed Training

| Aspect | Simple Training | Distributed Training |
|--------|----------------|---------------------|
| Data Format | ImageFolder (raw) | WebDataset (tar files) |
| Data Loading | Standard DataLoader | WebDataset pipeline |
| Training | Single process | Multiple workers + PS |
| Complexity | Low | High |
| Purpose | Validation/Testing | Production |

## Next Steps

1. ✅ Wait for training to complete (2 epochs)
2. ⏭️ Verify validation loss is reasonable (~3-6 range)
3. ⏭️ Confirm model checkpoints are saved correctly
4. ⏭️ If successful, can extend to more epochs or return to distributed training

## Files Created

1. [`train_simple.py`](file:///Users/danieltorosoto/universidad/arq-cliente-servidor/asynchronous_distributed_training_imagenet1k/train_simple.py) - Main training script
2. [`reorganize_val_data.py`](file:///Users/danieltorosoto/universidad/arq-cliente-servidor/asynchronous_distributed_training_imagenet1k/reorganize_val_data.py) - Data reorganization utility

## Checkpoints

Saved to `checkpoints_simple/`:
- `last.pth` - Most recent epoch
- `best.pth` - Best validation accuracy

Each checkpoint contains:
- `model_state_dict`
- `optimizer_state_dict`
- `epoch`
- `train_loss`, `train_acc`
- `val_loss`, `val_acc`
