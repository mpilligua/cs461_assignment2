# Part 1: Test-Time Adaptation

## Overview

My approach combines three complementary techniques for robust test-time adaptation:

1. **Entropy Minimization (Tent)** [1]: Adapts batch normalization parameters using prediction entropy as an unsupervised signal.
2. **Discriminator-Guided Adaptation**: Trains an MLP discriminator to distinguish "clean" from "corrupted" activations, providing an additional optimization signal during TTA.
3. **Consistency Finetuning**: Pre-trains the ResNet-50 backbone to produce corruption-invariant early-layer representations.

---

## Method Details

### 1. Tent: Entropy Minimization

Tent [1] leverages a key insight: model predictions tend to have higher entropy (uncertainty) when inputs come from a shifted distribution. By minimizing the entropy of predictions at test time, we can adapt batch normalization parameters to better handle the new domain.

In my implementation, I restrict adaptation to **only the first layer's batch normalization** parameters. This choice is motivated by two factors:
- Early layers capture low-level features most affected by corruption
- Limiting adaptation prevents overfitting to individual test batches

### 2. Discriminator-Guided Adaptation

I hypothesized that a more direct approach to domain adaptation would be to explicitly guide early-layer activations toward a "clean" distribution. To achieve this, I trained a discriminator (a simple MLP) that:
- **Input**: First convolutional layer activations
- **Output**: Probability that the activation comes from a "clean" (uncorrupted) image

During TTA, the combined objective becomes:
$$\mathcal{L} = \mathcal{L}_{\text{entropy}} + \lambda \cdot \mathcal{L}_{\text{discriminator}}$$

where $\mathcal{L}_{\text{discriminator}} = -\log(D(\text{activations}))$ encourages the model to produce activations that the discriminator considers "clean-like".

**Why target only the first layer?** CNNs exhibit a well-documented hierarchy: early layers capture low-frequency, general features (edges, textures), while later layers capture high-frequency, semantic features. When the input domain shifts due to corruption, the "visual language" of the first layer changes, causing downstream layers to misinterpret the signal. By "translating" corrupted inputs back into the clean feature space, we restore the model's ability to classify correctly.

### 3. Consistency Finetuning of ResNet-50

Prior to TTA, I finetuned the ResNet-50 backbone to be inherently more robust to corruptions. The finetuning objective combines three losses:

1. **Consistency Loss**: Forces the first layer to produce identical representations for the same image under different corruptions (cosine similarity).
2. **Classification Loss**: Maintains accuracy on clean CIFAR-10 images to prevent representation collapse.
3. **Weight Regularization**: L2 penalty (thresholded) to prevent excessive drift from the pretrained weights.

Importantly, I only use corruptions from the **exploratory set** (not the test corruptions) to avoid data leakage.

---

## Results

| Method | contrast | fog | frost | gaussian_blur | pixelate | shot_noise | **Mean** |
|--------|----------|-----|-------|---------------|----------|------------|----------|
| Unadapted | 66.82% | 84.06% | 80.20% | 73.55% | 83.24% | 56.82% | 74.12% |
| Norm | 84.73% | 87.61% | 82.61% | 88.41% | 86.21% | 77.12% | 84.45% |
| **Ours** | **89.25%** | **90.15%** | **86.90%** | **89.53%** | **89.44%** | **83.03%** | **88.05%** |

Our method achieves **88.05% mean accuracy** on the public test bench, improving by **+3.6%** over the TestTimeNorm baseline. The largest gains are observed on challenging corruptions:
- **shot_noise**: +5.9% improvement
- **contrast**: +4.5% improvement

These results demonstrate that consistency finetuning combined with entropy minimization is particularly effective for severe distribution shifts.

---

## Reproduction of Results

To train the discriminator:
```bash
cd part1/
python train_discriminator.py  # saves to ckpts/activation_discriminator.pt
```

To retrain the finetuned model:
```bash
python finetune_consistency_v3.py  # saves to ckpts/finetuned_resnet50.pt
```

To run the evaluation pipeline:
```bash
python eval.py submission
```

---

## References

[1] Wang, D., Shelhamer, E., Liu, S., Olshausen, B., & Darrell, T. (2021). *Tent: Fully Test-time Adaptation by Entropy Minimization*. ICLR 2021.