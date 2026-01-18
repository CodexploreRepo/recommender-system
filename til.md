# Today I Learn

## Day 5

### Torch Operations

- `dim=-1` means concatenate along the last dimension of the tensors.

```Python
import torch

# Batch of 3 users/items, each with 4-dimensional embeddings
user_embed = torch.tensor([
    [0.1, 0.2, 0.3, 0.4],  # user 0
    [0.5, 0.6, 0.7, 0.8],  # user 1
    [0.9, 1.0, 1.1, 1.2],  # user 2
])  # Shape: (3, 4)

item_embed = torch.tensor([
    [1.0, 1.1, 1.2, 1.3],  # item 0
    [1.4, 1.5, 1.6, 1.7],  # item 1
    [1.8, 1.9, 2.0, 2.1],  # item 2
])  # Shape: (3, 4)

concat = torch.cat([user_embed, item_embed], dim=-1)
# Shape: (3, 4+4) - concatenated along the embedding dimension
```

- `dim=0`

```Python
torch.cat([user_embed, item_embed], dim=0)
# Shape: (6, 4)
```

## Day 4

### PCA (Principal Component Analysis)

#### What is PCA?

- **PCA** reduces high-dimensional data to lower dimensions while preserving as much variance (information) as possible
- Finds new axes (principal components) that capture the most variation in the data
- Each PC is a weighted combination of original dimensions

#### How PCA Works

**Step 1: Center the Data**

```python
X_centered = X - X.mean(axis=0)  # Subtract mean from each dimension
```

**Step 2: Compute Covariance Matrix**

```python
C = (1/n) × X_centered^T × X_centered  # Shows how dimensions vary together
```

**Step 3: Find Eigenvectors and Eigenvalues**

```python
eigenvalues, eigenvectors = np.linalg.eig(C)
# eigenvectors = directions in original space
# eigenvalues = variance along each direction
```

**Step 4: Sort by Eigenvalue**

- Eigenvectors sorted by eigenvalue (descending)
- PC1 = direction with MOST variance
- PC2 = direction with 2nd most variance (perpendicular to PC1)

**Step 5: Project Data**

```python
# Select top k eigenvectors
PC_matrix = eigenvectors[:, :k]
# Project data onto principal components
X_pca = X_centered @ PC_matrix  # (n_samples, k)
```

#### Principal Components (PCs)

- **PC** = a new axis through your data
- **PC1** = direction of maximum variance
- **PC2** = perpendicular direction, next most variance
- All PCs are orthogonal (perpendicular) to each other
- Ordered by importance: PC1 > PC2 > PC3 > ...

```shell
PC1 = w₁·factor₀ + w₂·factor₁ + ... + wₙ·factorₙ
      ↑
  Weighted combination of original dimensions
```

#### Variance Explained

**What is variance?**

- Measures how spread out data is along a direction
- Higher variance = more information captured

**Calculation:**

```python
variance_pc1 = np.var(X_pca[:, 0])
total_variance = np.sum(np.var(X, axis=0))
pc1_percentage = variance_pc1 / total_variance
```

**Example:** PC1 = 11.7%, PC2 = 2.5%

- Total preserved in 2D = 14.2%
- Remaining 85.8% lost (in other dimensions)

**Typical PC1+PC2 variance by data type:**

| Data Type               | PC1+PC2 | Reason                      |
| ----------------------- | ------- | --------------------------- |
| Face images             | 60-80%  | Dominated by lighting, pose |
| Word2Vec/ALS embeddings | 10-25%  | Distributed representations |
| Random vectors          | ~3%     | No structure                |

#### Why Learned Embeddings Have Low PC Variance

1. **Distributed representation**: Information spread across ALL dimensions
2. **Regularization**: Prevents any dimension from dominating
3. **Complex patterns**: Many factors influence preferences
4. **By design**: Makes embeddings more expressive

```shell
High regularization → more uniform → lower PC variance
Low regularization  → some dims dominate → higher PC variance
```

#### Linear vs Non-Linear Patterns

**Linear (PCA can capture):**

```shell
y = w₁·x₁ + w₂·x₂ + ... + wₙ·xₙ
    ↑
  Straight lines/planes
```

**Non-linear (PCA cannot capture):**

```shell
y = x₁² + x₂²              # Squared terms
y = x₁ · x₂                # Interaction
y = exp(x₁)                # Exponential
    ↑
  Curves, circles, complex patterns
```

**Visual example:**

```shell
Linear pattern:            Non-linear pattern:
  x₂                         x₂
   ↑                          ↑
   │    *                     │  * *
   │  *   *                   │ *   *
   │*   *                     │*     *
  ─┴──────→ x₁                │ *   *
                              │  * *
  PCA works ✓               ─┴──────→ x₁
                              PCA fails ✗
```

#### When to Use PCA vs Other Methods

| Method    | Type       | Use Case                                      |
| --------- | ---------- | --------------------------------------------- |
| **PCA**   | Linear     | Quick, interpretable dimensionality reduction |
| **t-SNE** | Non-linear | Visualization, finds clusters                 |
| **UMAP**  | Non-linear | Faster than t-SNE, preserves structure        |

#### Interpreting PCA Results in ALS

**Low variance (10-15%) is normal:**

- ALS learns 64-dimensional embeddings
- Information distributed across many dimensions
- 2D projection loses most information

**What PCs often capture:**

- PC1: Popularity (mainstream vs niche)
- PC2: Genre or content patterns
- Remaining: Fine-grained collaborative patterns

**Validation checks:**

```python
# Check what PC1 represents
correlation = np.corrcoef(X_pca[:, 0], item_popularity)[0,1]
# High correlation (>0.6) → PC1 captures popularity

# Check full variance distribution
pca_full = PCA(n_components=20)
pca_full.fit(item_factors)
print(np.cumsum(pca_full.explained_variance_ratio_))
# See how many PCs needed for 50%+ variance
```

#### Summary

| Concept          | Meaning                                        |
| ---------------- | ---------------------------------------------- |
| **PCA**          | Dimensionality reduction via linear projection |
| **PC**           | New axis capturing maximum variance            |
| **Variance %**   | Information preserved in that dimension        |
| **Low variance** | Original data uses many dimensions equally     |
| **Linear only**  | Cannot capture curved/complex patterns         |

### t-SNE (t-Distributed Stochastic Neighbor Embedding)

#### What is t-SNE?

- **t-SNE** is a non-linear dimensionality reduction technique primarily used for **visualization**
- Preserves local structure (nearby points stay nearby) while revealing global patterns like clusters
- Unlike PCA (linear), t-SNE can capture complex, non-linear relationships

#### How t-SNE Works

**Step 1: Compute Pairwise Similarities in High Dimensions**

```python
# For each pair of points i, j, compute probability that i picks j as neighbor
# Uses Gaussian kernel centered at each point
p_ij = exp(-||x_i - x_j||² / 2σ²) / Σ_k exp(-||x_i - x_k||² / 2σ²)
```

**Step 2: Initialize Low-Dimensional Embeddings**

```python
# Randomly initialize 2D (or 3D) points
Y = np.random.randn(n_samples, 2) * 0.01
```

**Step 3: Compute Similarities in Low Dimensions**

```python
# Use Student's t-distribution (heavy tails) instead of Gaussian
q_ij = (1 + ||y_i - y_j||²)⁻¹ / Σ_k,l (1 + ||y_k - y_l||²)⁻¹
```

**Step 4: Minimize KL Divergence**

```python
# Make low-D similarities match high-D similarities
KL(P||Q) = Σ_i,j p_ij log(p_ij / q_ij)
# Use gradient descent to move points
```

**Step 5: Iterate**

- Repeat for typically 1000-5000 iterations
- Points move to minimize cost function

#### Key Parameters

**perplexity** (most important)

```python
# Controls neighborhood size (typical: 5-50)
tsne = TSNE(perplexity=30)  # balanced
tsne = TSNE(perplexity=5)   # focuses on very local structure
tsne = TSNE(perplexity=50)  # focuses on broader structure
```

**n_iter** (number of iterations)

```python
tsne = TSNE(n_iter=1000)   # default, often good enough
tsne = TSNE(n_iter=5000)   # more stable, better convergence
```

**learning_rate** and **random_state**

```python
tsne = TSNE(
    learning_rate=200,     # default, can tune 10-1000
    random_state=42        # for reproducibility
)
```

#### PCA vs t-SNE Comparison

```shell
PCA (Linear):                t-SNE (Non-linear):
  x₂                           x₂
   ↑                            ↑
   │   cluster 1                │  ●●●     cluster 1
   │   ● ● ●                    │  ●●●
   │   ● ● ●                    │
   │                            │
   │         cluster 2          │         ●●●  cluster 2
   │         ● ● ●              │         ●●●
  ─┴──────────────→ x₁         ─┴──────────────→ x₁

Preserves: global variance    Preserves: local neighborhoods
Best for:  linear patterns    Best for:  finding clusters
```

#### Important Properties of t-SNE

**1. Distances are NOT meaningful**

```shell
# Only cluster structure matters
Distance between points A-B vs C-D cannot be compared
Only use t-SNE to see: "Are these points clustered?"
```

**2. Cluster sizes don't mean anything**

```shell
Large cluster ≠ more points
Small cluster ≠ fewer points
t-SNE can expand/contract clusters arbitrarily
```

**3. Non-deterministic (unless random_state set)**

```python
# Same data, different runs → different plots
tsne1 = TSNE(random_state=42).fit_transform(X)  # reproducible ✓
tsne2 = TSNE().fit_transform(X)                 # different each run
```

**4. Computationally expensive**

```python
# O(n²) complexity
# 1000 samples: ~seconds
# 10000 samples: ~minutes
# 100000+ samples: use UMAP instead
```

#### When to Use t-SNE vs Alternatives

| Method    | Speed     | Use Case                                          |
| --------- | --------- | ------------------------------------------------- |
| **PCA**   | Very fast | Linear structure, interpretability, preprocessing |
| **t-SNE** | Slow      | Visualization, finding clusters in complex data   |
| **UMAP**  | Fast      | Like t-SNE but faster, better global structure    |

#### Practical Usage

**Basic usage:**

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Fit and transform
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(item_factors)  # (n_items, 2)

# Plot
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.5)
plt.title('t-SNE visualization of item embeddings')
plt.show()
```

**Tuning perplexity:**

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, perp in enumerate([5, 30, 50]):
    tsne = TSNE(perplexity=perp, random_state=42)
    X_tsne = tsne.fit_transform(item_factors)
    axes[i].scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.5)
    axes[i].set_title(f'perplexity={perp}')
```

**Coloring by metadata:**

```python
# Color by popularity
popularity = np.array(R_train.sum(axis=0)).flatten()
plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
            c=popularity, cmap='viridis', alpha=0.6)
plt.colorbar(label='Popularity')
```

#### Interpreting t-SNE for ALS Embeddings

**Why ALS embeddings show "uniform blob":**

- Collaborative filtering learns gradual patterns (not discrete clusters)
- Users have overlapping preferences (no clear item categories)
- Implicit feedback creates smooth transitions

**What to look for:**

```python
# Good signs:
✓ Similar items (same genre) cluster together
✓ Popular items in one region, niche in another
✓ Smooth gradients (not random scatter)

# Bad signs:
✗ Completely random scatter (no structure)
✗ All points in tiny corner (embedding collapsed)
✗ Perfect grid pattern (optimization failed)
```

**Validation:**

```python
# Check if similar items are close in t-SNE space
item1_idx = 42  # e.g., "Toy Story"
distances = np.sum((X_tsne - X_tsne[item1_idx])**2, axis=1)
nearest = np.argsort(distances)[1:6]  # 5 nearest
print(f"Items nearest to item {item1_idx}: {nearest}")
# Manually check if these items are actually similar
```

#### Common Pitfalls

**1. Over-interpreting distances**

```shell
❌ "These clusters are twice as far apart"
✓  "These items form distinct groups"
```

**2. Comparing different runs**

```shell
❌ Using different random_state and comparing
✓  Always set random_state=42 for fair comparison
```

**3. Using t-SNE for anything except visualization**

```shell
❌ Using t-SNE embeddings as features for ML
❌ Computing distances in t-SNE space
✓  Only use for human visual interpretation
```

**4. Wrong perplexity**

```shell
Too low (perp=5):   Many tiny clusters (over-fragmented)
Too high (perp=100): Everything merged (under-fragmented)
Good (perp=30):      Balanced, reveals true structure
```

#### Summary

| Concept               | Meaning                                                    |
| --------------------- | ---------------------------------------------------------- |
| **t-SNE**             | Non-linear visualization via neighbor preservation         |
| **Perplexity**        | Neighborhood size (tune 5-50)                              |
| **Non-deterministic** | Different runs → different plots (unless random_state set) |
| **Local structure**   | Preserves which points are neighbors                       |
| **Distances**         | NOT meaningful, only cluster membership matters            |
| **Speed**             | O(n²), slow for large datasets                             |

## Day 3

### Debug Result

- Same model, recall and hit rate but ndcg & map are lower => something wrong with the order of the prediction
- Precision is good Recall is worst => something wrong with the ground truth
- NDCG should be higher when K is larger

### Reproducibility

```Python
# for reproducibility
import random
import numpy as np
import torch
from torch.backends import cudnn

def init_seeds(seed=0, cuda_deterministic=True):
    """
    Initialize the random number seed
    :param seed: random number seed
    :param cuda_deterministic: Whether to fix the random number seed of cuda
    Setting this flag to True allows us to pre-optimize the convolutional layers of the model in PyTorch
    We can't set cudnn.benchmark=True if our network model keeps changing. Because it takes time to find the optimal convolution algorithm.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True
```

### Metrics

#### AUC

- **AUC (Area Under the ROC Curve)** measures how well a model ranks positive items higher than negative items for a user.
- AUC answers: "If I randomly pick one item the user liked and one they didn't, what's the probability the model ranks the liked item higher?"
- In binary classification:
  - **Positive class**: Label = 1
  - **Negative class**: Label = 0
  - AUC = Probability that a random positive is scored higher than a random negative
- In recommender systems:
  - **Positive class**: Items the user interacted with
  - **Negative class**: Items the user didn't interact with
  - AUC = Probability that a random liked item is ranked higher than a random non-liked item

#### ROC Curve

- The ROC curve plots True Positive Rate vs False Positive Rate at different thresholds. Where:
  - **TPR (True Positive Rate)** = `TP / (TP + FN)` — How many positives were correctly ranked high?
  - **FPR (False Positive Rate)** = `FP / (FP + TN)` — How many negatives were incorrectly ranked high?

```shell
TPR (Recall)
    │
1.0 │        ┌───────── Perfect (AUC=1.0)
    │       /
    │      /  ┌──────── Good model (AUC=0.85)
    │     /  /
0.5 │    /  /
    │   /  /  ────────── Random (AUC=0.5)
    │  /  /
    │ /  /
0.0 └────────────────── FPR
    0.0      0.5     1.0
```

#### AUC Calculation for Recommender Systems

- AUC formula: `AUC = P(score(positive) > score(negative))`

```shell
Global AUC (classification):
  AUC = compare all positives vs all negatives

Per-user AUC (recommenders):
  AUC = (1/|Users|) × Σᵤ AUC(user u)

  where AUC(user u) compares user u's
  positive items vs negative items
```

- For a single user with positive items `P` and negative items `N`:
  - `AUC = (1 / |P||N|) × Σᵢ∈P Σⱼ∈N (scoreᵢ > scoreⱼ)`
- Example:

```shell
User's items:
  Positive (liked): A, B
  Negative (not interacted): X, Y, Z

Model scores:
  A: 0.9,  B: 0.7,  X: 0.6,  Y: 0.4,  Z: 0.8
                                        ↑ problem!

Ranking: A(0.9) > Z(0.8) > B(0.7) > X(0.6) > Y(0.4)

Pairwise comparisons (positive vs negative):
  A > X? ✓    A > Y? ✓    A > Z? ✓
  B > X? ✓    B > Y? ✓    B > Z? ✗

Correct pairs: 5 out of 6
AUC = 5/6 = 0.833
```

#### Limitation of AUC

- AUC treats all positions equally, but in recommendations **top positions matter more** (users only see top-k). Metrics like **NDCG@k** or **Precision@k** focus on top-ranked items:

```shell
Ranking 1: [✓, ✓, ✗, ✗, ✗]  — Good! Relevant items at top
Ranking 2: [✗, ✗, ✗, ✓, ✓]  — Bad! Relevant items buried

Both might have similar AUC, but Ranking 1 is better for users.
```

#### train_auc in BPR

**What is train_auc?**

- In BPR (Bayesian Personalized Ranking), `train_auc` measures how well the model ranks positive items above negative items during training
- Specifically: `train_auc` = % of training triples where `score(user, positive_item) > score(user, negative_item)`

**Example output:**

```shell
100%|██████████| 50/50 [train_auc=96.96%, skipped=22.80%]
                            ↑                ↑
                      96.96% correct      22.8% samples skipped
                      pairwise rankings   (false negatives avoided)
```

**Interpretation Guidelines:**

| train_auc  | Meaning              | Action                                    |
| ---------- | -------------------- | ----------------------------------------- |
| **95-99%** | ✓ Excellent learning | Model trained well                        |
| **90-95%** | ~ Good learning      | Acceptable, validate on test              |
| **80-90%** | ⚠ Weak learning      | Increase iterations or tune learning_rate |
| **<80%**   | ✗ Poor learning      | Check data quality or hyperparameters     |

**Why >90% is a common threshold:**

- **Empirical observation**: Original BPR paper (Rendle et al., 2009) reported ~93-95% train_auc on MovieLens
- **Community convention**: Papers and practitioners found train_auc >90% correlates with good test performance
- **NOT a hard rule**: The threshold depends on dataset density:
  ```
  Dense datasets (Netflix, Spotify):  Expected 92-98%
  Medium datasets (MovieLens):        Expected 90-97%
  Sparse datasets (e-commerce, news): Expected 80-92%
  ```

**Better approach: Monitor convergence**

```shell
iterations=10  → train_auc=85.2%
iterations=20  → train_auc=92.1%
iterations=30  → train_auc=95.3%
iterations=40  → train_auc=96.8%
iterations=50  → train_auc=96.96% ← Converged (gain <0.5%)
iterations=100 → train_auc=97.1%  ← Marginal improvement

Stop when gains are <0.5% between iterations
```

**What affects train_auc:**

1. **Iterations (epochs)**

   ```python
   iterations=10  → train_auc=75%  ❌ Under-trained
   iterations=50  → train_auc=97%  ✓ Good
   iterations=100 → train_auc=97.5% (marginal gains)
   ```

2. **Learning rate**

   ```python
   learning_rate=0.001 → train_auc=85%  ❌ Too slow
   learning_rate=0.05  → train_auc=96%  ✓ Good
   learning_rate=0.5   → train_auc=70%  ❌ Too high, unstable
   ```

3. **Regularization**
   ```python
   regularization=0.1  → train_auc=93%  ← More constraint
   regularization=0.01 → train_auc=97%  ← Less constraint
   ```

**Important: train_auc ≠ final performance**

```shell
# Bad model (overfitting):
train_auc = 99.5%  ← Very high! ✓
test_ndcg = 0.045  ← Low! ✗
→ Model memorized training data

# Good model (generalization):
train_auc = 94.2%  ← Lower
test_ndcg = 0.069  ← Higher! ✓
→ Better on unseen data
```

**Key takeaway:**

- train_auc is a **training signal**, not the goal
- Use it as a **sanity check**: <80% means training failed
- **Always validate on test metrics** (NDCG@K, MAP@K) to judge real performance
- Sweet spot: **95-98%** balances learning without overfitting

### ALS (Point-wise) vs BPR (Pair-wise)

- **ALS** (Point-wise) &#8594; recall (retrieval: narrow down search space)
  - Is called a pointwise method because it optimizes the loss function by treating each user-item interaction **independently** as a single data point.
  - Learn representation of the item &#8594; the item embedding can be clustered
- **BPR** (Pair-wise ranking loss): which items user prefer should be ranked higher than the user does not like ) &#8594; ndcg (ranking)

| Approach  | Unit of Optimization             | Example Methods                     |
| --------- | -------------------------------- | ----------------------------------- |
| Pointwise | Single (user, item) pair         | ALS, Matrix Factorization, SVD      |
| Pairwise  | Pair of items (preferred vs not) | BPR (Bayesian Personalized Ranking) |
| Listwise  | Entire ranked list               | LambdaRank, ListNet                 |

### Terminology

#### Stochastic

- Stochastic: sampling-based optimization

#### Convex vs Non-Convex

- A non-convex problem in optimization means the objective function has multiple local minima, making it difficult to find the global optimum.
  - Gradient-based methods can get "stuck" in suboptimal solutions

```shell
Convex:                    Non-Convex:
    \     /                  \  /\  /
     \   /                    \/  \/
      \_/                    local minima
   global min
```

## Day 2

### LayerNorm vs BatchNorm

- The Key Difference: Which Axis to Normalize

```shell
Input tensor shape: (batch_size=3, seq_length=4, hidden_size=5)

┌─────────────────────────────────────────────────────────┐
│                    hidden_size (5)                      │
│              ┌───┬───┬───┬───┬───┐                      │
│  seq_len(4)  │   │   │   │   │   │  ← Sample 1, pos 0   │
│              ├───┼───┼───┼───┼───┤                      │
│              │   │   │   │   │   │  ← Sample 1, pos 1   │
│              ├───┼───┼───┼───┼───┤                      │
│              │   │   │   │   │   │                      │
│              ├───┼───┼───┼───┼───┤                      │
│              │   │   │   │   │   │                      │
│              └───┴───┴───┴───┴───┘                      │
│                                                         │
│              ┌───┬───┬───┬───┬───┐                      │
│              │   │   │   │   │   │  ← Sample 2          │
│              ├───┼───┼───┼───┼───┤                      │
│              │   │   │   │   │   │                      │
│              ├───┼───┼───┼───┼───┤                      │
│              │   │   │   │   │   │                      │
│              ├───┼───┼───┼───┼───┤                      │
│              │   │   │   │   │   │                      │
│              └───┴───┴───┴───┴───┘                      │
│                                                         │
│              ┌───┬───┬───┬───┬───┐                      │
│              │   │   │   │   │   │  ← Sample 3          │
│              ...                                        │
└─────────────────────────────────────────────────────────┘
```

- `LayerNorm`: Normalize Across Features (per sample, per position)

```shell
LayerNorm normalizes EACH ROW independently:

Sample 1, Position 0: [2.0, -1.0, 5.0, 0.0, 3.0] → normalize these 5 values
Sample 1, Position 1: [1.0,  2.0, 0.5, 1.5, 2.5] → normalize these 5 values
...
Sample 2, Position 0: [3.0,  1.0, 2.0, 4.0, 0.0] → normalize these 5 values
...

Each row gets its own mean & std:
┌─────────────────────────────────────────┐
│  [■ ■ ■ ■ ■] ← normalize this row       │
│  [■ ■ ■ ■ ■] ← normalize this row       │
│  [■ ■ ■ ■ ■] ← normalize this row       │
│  [■ ■ ■ ■ ■] ← normalize this row       │
└─────────────────────────────────────────┘
```

- `BatchNorm`: Normalize Across Batch (per feature)

```shell
BatchNorm normalizes EACH COLUMN across ALL samples:

Feature 0 across all samples & positions:
  Sample1[0,0], Sample1[1,0], Sample1[2,0], Sample1[3,0],
  Sample2[0,0], Sample2[1,0], Sample2[2,0], Sample2[3,0],
  Sample3[0,0], ...
  → normalize ALL these values together

Each column (feature) gets one mean & std from the entire batch:
┌───┬───┬───┬───┬───┐
│ ▼ │ ▼ │ ▼ │ ▼ │ ▼ │  ← Sample 1
│ │ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │ │
├───┼───┼───┼───┼───┤
│ │ │ │ │ │ │ │ │ │ │  ← Sample 2
│ │ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │ │
├───┼───┼───┼───┼───┤
│ │ │ │ │ │ │ │ │ │ │  ← Sample 3
│ ▼ │ ▼ │ ▼ │ ▼ │ ▼ │
└───┴───┴───┴───┴───┘
  ↑   ↑   ↑   ↑   ↑
  normalize each column across entire batch
```

- Example:

```shell
# Batch of 3 samples, each with 4 features
batch = [
    [2.0, -1.0, 5.0, 0.0],   # Sample 1
    [4.0,  1.0, 3.0, 2.0],   # Sample 2
    [0.0,  3.0, 1.0, 4.0],   # Sample 3
]

# ═══════════════════════════════════════════════════════
# LayerNorm: normalize each SAMPLE (row) independently
# ═══════════════════════════════════════════════════════

Sample 1: [2.0, -1.0, 5.0, 0.0]
  mean = (2 + -1 + 5 + 0) / 4 = 1.5
  std  = 2.29
  normalized = [(2-1.5)/2.29, (-1-1.5)/2.29, (5-1.5)/2.29, (0-1.5)/2.29]
             = [0.22, -1.09, 1.53, -0.65]

Sample 2: [4.0, 1.0, 3.0, 2.0]
  mean = 2.5, std = 1.12
  normalized = [1.34, -1.34, 0.45, -0.45]

Sample 3: [0.0, 3.0, 1.0, 4.0]
  mean = 2.0, std = 1.58
  normalized = [-1.26, 0.63, -0.63, 1.26]

# ═══════════════════════════════════════════════════════
# BatchNorm: normalize each FEATURE (column) across batch
# ═══════════════════════════════════════════════════════

Feature 0 (column): [2.0, 4.0, 0.0]
  mean = (2 + 4 + 0) / 3 = 2.0
  std  = 1.63
  normalized = [(2-2)/1.63, (4-2)/1.63, (0-2)/1.63]
             = [0.0, 1.22, -1.22]

Feature 1 (column): [-1.0, 1.0, 3.0]
  mean = 1.0, std = 1.63
  normalized = [-1.22, 0.0, 1.22]

Feature 2 (column): [5.0, 3.0, 1.0]
  mean = 3.0, std = 1.63
  normalized = [1.22, 0.0, -1.22]

Feature 3 (column): [0.0, 2.0, 4.0]
  mean = 2.0, std = 1.63
  normalized = [-1.22, 0.0, 1.22]

```

| Aspect                    | LayerNorm                | BatchNorm                   |
| ------------------------- | ------------------------ | --------------------------- |
| Normalizes across         | Features (hidden_size)   | Batch samples               |
| Each sample               | Normalized independently | Depends on other samples    |
| Batch size dependency     | No                       | Yes                         |
| Works with `batch_size=1` | Yes                      | Fails (can't compute stats) |

LayerNorm is preferred in Transformers because:

1. Each user sequence is normalized independently as we don't want User A's embedding statistics to affect User B's normalization.
2. Works regardless of batch size
3. Better for variable-length sequences

### LayerNorm

- LayerNorm (Layer Normalization) normalizes the values across the feature dimension for each individual sample.
- For a single vector `x = [x₁, x₂, ..., xₙ]` of dimension `hidden_size`:

```shell
1. Compute mean:     μ = (x₁ + x₂ + ... + xₙ) / n
2. Compute variance: σ² = Σ(xᵢ - μ)² / n
3. Normalize:        x̂ᵢ = (xᵢ - μ) / √(σ² + ε)
4. Scale & shift:    yᵢ = γ · x̂ᵢ + β    (learnable parameters)

Where ε is a small constant (1e-12) to prevent division by zero.
```

- Visualization Example

```shell
Input embedding (hidden_size=4):
x = [2.0, -1.0, 5.0, 0.0]

Step 1: Mean
μ = (2.0 + -1.0 + 5.0 + 0.0) / 4 = 1.5

Step 2: Variance
σ² = [(2-1.5)² + (-1-1.5)² + (5-1.5)² + (0-1.5)²] / 4
   = [0.25 + 6.25 + 12.25 + 2.25] / 4 = 5.25
σ  = √5.25 ≈ 2.29

Step 3: Normalize (subtract mean, divide by std)
x̂ = [(2-1.5)/2.29, (-1-1.5)/2.29, (5-1.5)/2.29, (0-1.5)/2.29]
   = [0.22, -1.09, 1.53, -0.65]

Step 4: Scale & Shift (γ and β are learned)
y = γ · x̂ + β
```

- What It Achieves

```shell
Before LayerNorm:
┌────────────────────────────────────────┐
│ Seq 1: [12.3, -45.2,  0.8,  7.1, ...]  │  ← wildly different scales
│ Seq 2: [ 0.1,   0.3, -0.2,  0.5, ...]  │
│ Seq 3: [99.0,  88.0, 77.0, 66.0, ...]  │
└────────────────────────────────────────┘

After LayerNorm:
┌────────────────────────────────────────┐
│ Seq 1: [ 0.5,  -1.8,  0.1,  0.3, ...]  │  ← all ~mean=0, ~std=1
│ Seq 2: [-0.2,   0.8, -1.1,  1.2, ...]  │
│ Seq 3: [ 1.1,   0.4, -0.3, -0.9, ...]  │
└────────────────────────────────────────┘
```

#### The Learnable Parameters (γ, β)

- After normalizing to mean=0, std=1, the network might need different statistics. The learnable γ (scale) and β (shift) let the model recover any distribution it needs

```shell
# In _modules.py LayerNorm
self.weight = nn.Parameter(torch.ones(hidden_size))   # γ, initialized to 1
self.bias = nn.Parameter(torch.zeros(hidden_size))    # β, initialized to 0
```

- Initially γ=1, β=0, so output = normalized input. During training, the model learns optimal γ and β values.

### Embeddings

- Item IDs start from 1, not 0 (index 0 is reserved for padding)
- **Index 0** is the padding token (used to pad sequences to fixed length)
- Item IDs in the dataset are 1-indexed (start from 1)
- If `max_item = 100`, then valid items are 1, 2, 3, ..., 100
  The embedding table needs 101 slots (indices 0-100): `max_item + 1`

```Python
args.item_size = max_item + 1
self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
```

#### `padding_idx=0`

- When you set `padding_idx=0`, PyTorch ensures that:
  - The embedding vector at index 0 is always zeros, so that Padding tokens map to **zero vectors**
  - The gradient for index 0 is always zero, so the model focuses only on real items

### Sequential Model

#### Padding

- With left padding: Position -1 (last position) always contains the most recent item

```Python
pad_len = self.max_len - len(input_ids)
input_ids = [0] * pad_len + input_ids
input_ids = input_ids[-self.max_len :]
assert len(input_ids) == self.max_len

# max_seq_length = 5
# Original sequence: [3, 7, 2]

pad_len = 5 - 3  # = 2
input_ids = [0] * 2 + [3, 7, 2]  # = [0, 0, 3, 7, 2]
```

#### K-score

- Sequential recommendation models need sufficient interaction history to learn patterns. Users with only 1-2 interactions don't provide enough signal, and items with very few interactions are hard to recommend meaningfully
- K-core filtering is a graph-based data preprocessing technique used to remove sparse users and items from the dataset.
- A K-core means:
  - Every user must have interacted with at least K items
  - Every item must have been interacted with by at least K users
  - Usually, choose K=5

## Day 1

### Linear Algebra

#### Element-wise

- `user_embed * item_embed` (Element-wise Product)

```Python
# If user_embed and item_embed are both shape (batch_size, embed_dim)
# Result: (batch_size, embed_dim)

user_embed = [0.1, 0.2, 0.3, 0.4]
item_embed = [0.5, 0.6, 0.7, 0.8]
result     = [0.1*0.05, 0.12, 0.21, 0.32]  # Same dimension, multiplied position-wise
```

- Preserves dimensionality - output has same shape as inputs

#### Dot Product & Cosine Similarity

- **Dot product** is the raw weighted sum $(a \cdot b)$; it depends on both direction and magnitude.
- **Cosine similarity** is the dot product after normalizing both vectors: $\frac{a \cdot b}{|a||b|})$, so it measures only direction (scale‑invariant), ranging from (-1) to (1)

#### Dot Product Numpy Implementation

- Geometric Method (Magnitude & Angle): $a \cdot b = |a|\times |b|\times \cos (\theta )$.
- In recommenders, it’s used to score how well a user profile matches an item (higher dot product ⇒ more aligned preferences)

```Python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(a @ b)  # dot product
# Because (1*4 + 2*5 + 3*6 = 32).

A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

print(A @ B)
# Each output cell is the dot product of a row of A with a column of B.
# [[19 22]
#  [43 50]]
```

- For 1D or 2D arrays, `@` and `np.dot` give the same result.
- For 3D+: prefer` @/np.matmul` for matrix multiplication semantics.

### SciPy's Sparse Matrix

- `csr_matrix` stores only non-zero values with their coordinates and reconstructs full matrix using `.toarray()`.

```Python
import numpy as np
from scipy.sparse import csr_matrix

d = np.array([3, 4, 5, 7, 2, 6])     # interaction
r = np.array([0, 0, 1, 1, 3, 3])     # user
c = np.array([2, 4, 2, 3, 1, 2])     # item

csr = csr_matrix((d, (r, c)), shape=(4, 5))
print(csr.toarray())
```

#### Spare Matrix Operations

```Python
# Create sparse matrix for training
R_train = csr_matrix(
    (
        df_train["implicit"].values,
        (df_train["uid"].values, df_train["iid"].values)
    ),
    shape=(n_users, n_items)
)
```

- `csr[uid].indices` returns the column indices of the non‑zero entries in that row [uid].
- `csr.sum(axis=0)`: sums each column of the user‑item matrix -> total interactions per item in the training set
  - `axis=0` means “sum down the rows,” i.e., column‑wise.

### Unix Timestamp

- A Unix timestamp is the number of seconds (or sometimes milliseconds) since 1970‑01‑01 00:00:00 UTC. It’s a compact, timezone‑agnostic way to store date/time values that you can later convert to human‑readable datetimes.
- UTC vs GMT: When UTC was established in 1972, it was designed to be almost exactly the same as GMT for practical purposes, so people still often say GMT when they mean UTC

```Python
# convert Unix timestamp into the human‑readable datetimes
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
```

|     | user | item | rating |   timestamp |
| --: | ---: | ---: | -----: | ----------: |
|   0 |  196 |  242 |      3 | 8.81251e+08 |
|   1 |  186 |  302 |      3 | 8.91718e+08 |
|   2 |   22 |  377 |      1 | 8.78887e+08 |
|   3 |  244 |   51 |      2 | 8.80607e+08 |
|   4 |  166 |  346 |      1 | 8.86398e+08 |
