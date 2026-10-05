# CIFAR-10 CNN (PyTorch)

A small convolutional neural network trained from scratch on CIFAR-10 with PyTorch: two conv blocks, one hidden fully connected layer, dropout, and basic data augmentation. No pretrained weights and no transfer learning.

## Results

**Test accuracy: 74.46%** on the 10,000-image CIFAR-10 test set (seeded run on a GPU, `seed=42`, final average training loss 0.86).

Treat this as indicative rather than exact. Results vary by a few tenths of a point between runs even with a fixed seed, because GPU operations are not fully deterministic. An earlier development run was recorded at 76.4%, higher than the clean seeded rerun reported here (see Known limitations).

## Architecture

| Layer | Details |
|---|---|
| Conv1 | 3 → 32 channels, 3×3 kernel, ReLU, 2×2 max-pool |
| Conv2 | 32 → 64 channels, 3×3 kernel, ReLU, 2×2 max-pool |
| Flatten | 64 × 6 × 6 = 2,304 features |
| FC1 | 2,304 → 128, dropout (p = 0.5), ReLU |
| FC2 | 128 → 10 class scores |

About 316K trainable parameters in total.

## Training setup

- Optimizer: Adam, learning rate 0.000825
- Loss: cross-entropy
- Batch size: 32
- Epochs: 80
- Augmentation (training set only): random crop (32×32 with 4-pixel padding) and random horizontal flip
- Test set: `ToTensor()` only, no augmentation
- No input normalization

## How the hyperparameters were chosen

By hand. Learning rates above roughly 0.001 made the training loss unstable (it jumped around instead of decreasing), so the rate was kept below that and settled at 0.000825. The epoch count was raised from 50 to 80 on the expectation that longer training would help. No separate validation set was used.

## How to run

```bash
pip install torch torchvision
python cifar10_cnn.py
```

The script downloads CIFAR-10 (about 170 MB) into `./data`, trains for 80 epochs, prints the average training loss per epoch, prints the test accuracy, and saves the trained weights to `cifar10_cnn.pth`. A GPU is used automatically if one is available. Training for 80 epochs on a CPU is slow.

## Known limitations

- **No validation split.** The learning rate and epoch count were set by hand without a separate validation set, so the test accuracy may be mildly optimistic. An earlier development run recorded a higher figure (76.4%) than the clean seeded rerun above, which suggests run-to-run variance and/or some tuning against the test set. Expect a few points of uncertainty either way.
- **The model may be capacity-limited.** Training loss was still about 0.86 at epoch 80, measured with augmentation and dropout switched on. That suggests the network is small and heavily regularised for this task, but accuracy on un-augmented training images was not measured, so this has not been confirmed.
- **Minimal architecture.** There is no BatchNorm, no input normalization, and only two conv layers.
- **Fixed learning rate.** No scheduler is used.
- **Single run, single seed.** No error bars.


