# Part 2: Multiple Instance Learning for Histopathology Classification

## Method Details

Given a patient with $N$ patches $\{x_1, x_2, \ldots, x_N\}$, where each $x_i \in \mathbb{R}^{3072}$ is a DINOv2 embedding, we want to predict the tissue type among 7 classes: ADI, BACK, DEB, LYM, MUC, MUS, and NORM.

### Architecture

The model processes each patient's patches through four stages:

**1. Attention Network**

The attention network learns to score each patch by how informative it is for classification. For each patch embedding $x_i \in \mathbb{R}^{3072}$, we compute:

$$s_i = \mathbf{w}^T \tanh(\mathbf{W} x_i)$$

where $\mathbf{W} \in \mathbb{R}^{256 \times 3072}$ projects the embedding to a hidden space and $\mathbf{w} \in \mathbb{R}^{256}$ computes a scalar score. The scores are then normalized across all patches using softmax.

**2. Top-K Selection**

Rather than using all $N$ patches, we select only the top $K$ patches with the highest attention weights:

$$K = \max(5, \lfloor 0.5 \cdot N \rfloor)$$

This keeps at least 5 patches, or 50% of the total, whichever is larger. 

<!-- The selection serves two purposes:
- **Noise reduction**: Discards uninformative patches (e.g., background, out-of-focus regions)
- **Focus on discriminative regions**: Forces the model to identify the most relevant tissue features -->

**3. Attention-Weighted Aggregation**

The selected patches are aggregated into a single patient-level representation. First, the attention weights of the selected patches are renormalized:

$$\tilde{a}_i = \frac{a_i}{\sum_{j \in \text{TopK}} a_j}$$

Then, the aggregated representation is computed as a weighted sum:

$$z = \sum_{i \in \text{TopK}} \tilde{a}_i \cdot x_i$$

This produces a single vector $z \in \mathbb{R}^{3072}$ that summarizes the patient's most informative patches.

**4. MLP Classifier**

The aggregated representation passes through a deep MLP:
- Hidden layers: [1024, 512, 256]
- Layer normalization after each linear layer for training stability
- ReLU activations
- Final linear layer to 7 classes with softmax for probabilities

### Training Strategy

**Data Split**

The dataset is split at the patient level using stratified sampling to ensure balanced class representation:
- **Train set**: 85%
- **Test set**: 15% (~500 patients)

As loss we use Cross-entropy with inverse-frequency class weights to handle imbalance
<!-- - **Optimizer**: AdamW with learning rate 1e-4
- **Scheduler**: Cosine annealing over 100 epochs
- **Batch size**: 16 patients per batch
- **Regularization Techniques**: Dropout, gradient clipping, early stopping and L2 regularization -->

---

## Results


### Comparison with Baseline (evaluated on my test set)

| Method | Accuracy | Macro F1 |
|--------|----------|----------|
| Linear Probe (mean pooling) | ~75% | ~72% |
| **TopK Attention MLP (Ours)** | **84.23%** | **81.14%** |


### Per-Class F1 Scores

| Class | Description | F1 Score |
|-------|-------------|----------|
| ADI | Adipose tissue | 66.07% |
| BACK | Background | 92.37% |
| DEB | Debris | 85.85% |
| LYM | Lymphocytes | 98.82% |
| MUC | Mucus | 75.95% |
| MUS | Smooth muscle | 68.47% |
| NORM | Normal colon mucosa | 66.07% |

### Confusion Matrix

|          | ADI | BACK | DEB | LYM | MUC | MUS | NORM |
|----------|-----|------|-----|-----|-----|-----|------|
| **ADI**  |  37 |    0 |   2 |   0 |   3 |   4 |   11 |
| **BACK** |   0 |   54 |   5 |   0 |   0 |   0 |    0 |
| **DEB**  |   2 |    3 |  45 |   1 |   1 |   1 |    0 |
| **LYM**  |   0 |    0 |   1 |  84 |   0 |   0 |    0 |
| **MUC**  |   4 |    0 |   0 |   0 |  60 |   3 |   10 |
| **MUS**  |   1 |    0 |   0 |   0 |   1 |  37 |   11 |
| **NORM** |   6 |    1 |   0 |   0 |  14 |   5 |   37 |

---

## Analysis

We can see that the model finds some classes much easier than others: 
- **LYM class**: Achieves near-perfect classification (98.82% F1), likely due to distinctive morphological features
- **NORM class**: Most challenging (66.07% F1), frequently confused with ADI and MUC
- **ADI vs NORM confusion**: These tissue types share similar visual characteristics, leading to misclassifications

### Why TopK Attention Works

The TopK selection mechanism addresses a key challenge in histopathology: not all patches are equally informative. By focusing on the most discriminative regions, the model:
1. Reduces noise from uninformative patches
2. Handles the high within-class variance typical of histopathology images
3. Learns to identify the most relevant tissue features automatically

---

## Reproduction

To train the model from scratch:
```bash
cd part2/models/
python submission.py
```

To evaluate:
```bash
cd part2/
python eval.py submission
```

Checkpoints are saved to `part2/ckpts/best_submission.pt`.
