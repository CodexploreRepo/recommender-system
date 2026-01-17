# ALTERNATING LEAST SQUARES (ALS)

# Table of Contents

- [Table of Contents](#table-of-contents)
- [1. What is ALS?](#1-what-is-als)
  - [1.1. Why ALS?](#11-why-als)
- [2. ALS Algorithm](#2-als-algorithm)
  - [2.1. Objective](#21-objective)
  - [2.2. Closed-Form Solution](#22-closed-form-solution)
- [3. Implicit Feedback](#3-implicit-feedback)
  - [3.1. Confidence Weighting](#31-confidence-weighting)
- [4. Implementation](#4-implementation)
  - [4.1. Hyperparameters](#41-hyperparameters)
  - [4.2. Code Example](#42-code-example)
- [5. Key Insights](#5-key-insights)

# 1. What is ALS?

**Alternating Least Squares (ALS)** decomposes the user-item interaction matrix **R** into two lower-rank matrices:

```
R ≈ U × Vᵀ

where:
  U (n_users × k)   = user latent factors
  V (n_items × k)   = item latent factors
  k                 = number of latent dimensions
```

## 1.1. Why ALS?

**Problem:** Optimizing both U and V simultaneously is **non-convex** (multiple local minima)

**Solution:** Alternate between fixing one and solving for the other

```shell
Fix V, solve for U  →  convex (ridge regression)
Fix U, solve for V  →  convex (ridge regression)
```

**Benefits:**

- Closed-form solution (no learning rate tuning)
- Highly parallelizable (each user/item solved independently)
- Native support for implicit feedback

[(Back to top)](#table-of-contents)

# 2. ALS Algorithm

## 2.1. Objective

Minimize the reconstruction error with regularization:

```
L = Σ(i,j)∈Ω (rᵢⱼ - uᵢᵀvⱼ)² + λ(||U||²F + ||V||²F)

where:
  Ω    = set of observed ratings
  λ    = regularization parameter
  ||·||F = Frobenius norm
```

**Algorithm:**

```python
Initialize U, V randomly
for iteration in range(iterations):
    # Fix V, update all user factors
    for each user i:
        uᵢ = (VᵢᵀVᵢ + λI)⁻¹ Vᵢᵀrᵢ

    # Fix U, update all item factors
    for each item j:
        vⱼ = (UⱼᵀUⱼ + λI)⁻¹ Uⱼᵀrⱼ
```

## 2.2. Closed-Form Solution

For user **i** (with V fixed):

```
uᵢ = (VᵢᵀVᵢ + λI)⁻¹ Vᵢᵀrᵢ

where:
  Vᵢ = rows of V corresponding to items rated by user i
  rᵢ = ratings given by user i
  I  = identity matrix (k × k)
```

Similarly for item **j** (with U fixed):

```
vⱼ = (UⱼᵀUⱼ + λI)⁻¹ Uⱼᵀrⱼ
```

[(Back to top)](#table-of-contents)

# 3. Implicit Feedback

## 3.1. Confidence Weighting

**Challenge:** Implicit feedback (clicks, views, purchases) only shows positive interactions. Absence could mean dislike OR unawareness.

**Solution:** Transform interactions into:

1. **Preference** pᵢⱼ: Binary indicator (1 if interaction, 0 otherwise)
2. **Confidence** cᵢⱼ: How confident we are in this preference

**Formulation:**

```python
# Preference (binary)
pᵢⱼ = 1  if rᵢⱼ > 0 else 0

# Confidence (weighted)
cᵢⱼ = 1 + α × rᵢⱼ

where:
  α     = confidence scaling (typical: 1-40)
  rᵢⱼ   = interaction count
```

**Intuition:**

```shell
User watched movie 10 times  →  high confidence they like it
User never watched movie     →  low confidence (maybe unaware)
```

**Objective with confidence weighting:**

```
L = Σᵢ Σⱼ cᵢⱼ(pᵢⱼ - uᵢᵀvⱼ)² + λ(||U||²F + ||V||²F)
```

**Update rules:**

```
uᵢ = (VᵀCⁱV + λI)⁻¹ VᵀCⁱpᵢ
vⱼ = (UᵀCʲU + λI)⁻¹ UᵀCʲpⱼ

where Cⁱ, Cʲ are diagonal confidence matrices
```

[(Back to top)](#table-of-contents)

# 4. Implementation

## 4.1. Hyperparameters

| Parameter          | Range     | Description        | Impact                           |
| ------------------ | --------- | ------------------ | -------------------------------- |
| **factors**        | 32-128    | Latent dimensions  | Higher = more expressive, slower |
| **regularization** | 0.001-0.1 | L2 penalty         | Higher = less overfitting        |
| **iterations**     | 15-30     | ALS cycles         | More = better convergence        |
| **alpha**          | 1-40      | Confidence scaling | Higher = trust positives more    |

**Tuning strategy:**

```python
# Start with reasonable defaults
factors = 64
regularization = 0.001
iterations = 20
alpha = 10

# Grid search on validation set
# Optimize for Recall@K or NDCG@K
```

## 4.2. Code Example

**Training:**

```python
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

# Prepare implicit feedback (binary)
df["implicit"] = (df["rating"] >= 4).astype(int)

# Create sparse matrix
R_train = csr_matrix(
    (df_train["implicit"], (df_train["uid"], df_train["iid"])),
    shape=(n_users, n_items)
)

# Train ALS
als = AlternatingLeastSquares(
    factors=64,
    regularization=0.001,
    iterations=20,
    random_state=42
)

# Apply confidence weighting
alpha = 10
als.fit(R_train * alpha)
```

**Recommendation:**

```python
def recommend(als_model, R_train, user_id, K=10):
    """Get top-K recommendations for a user."""
    items, scores = als_model.recommend(
        userid=user_id,
        user_items=R_train[user_id],  # Filter seen items
        N=K,
        filter_already_liked_items=True
    )
    return items, scores
```

**Evaluation:**

```python
def recall_at_k(rec_items, true_items, K):
    """Recall@K metric."""
    n_hit = len(set(rec_items[:K]) & true_items)
    n_true = len(true_items)
    return n_hit / n_true if n_true > 0 else 0.0

# Evaluate on test set
recalls = []
for uid, true_items in test_items.items():
    rec_items, _ = recommend(als, R_train, uid, K=10)
    recalls.append(recall_at_k(rec_items, true_items, K=10))

print(f"Recall@10: {np.mean(recalls):.4f}")
```

[(Back to top)](#table-of-contents)

# 5. Key Insights

**Advantages:**

- ✓ **Scalable**: Parallelizable across users/items
- ✓ **No learning rate**: Closed-form solution
- ✓ **Implicit feedback**: Native confidence weighting
- ✓ **Convergence**: Guaranteed to local minimum

**Limitations:**

- ✗ **Cold start**: Cannot recommend for new users/items
- ✗ **Static**: Requires retraining for new data
- ✗ **Linear**: Cannot capture non-linear patterns

**When to use ALS:**

| Use ALS             | Use Alternatives     |
| ------------------- | -------------------- |
| Batch processing    | Real-time updates    |
| Implicit feedback   | Complex features     |
| Large-scale data    | Cold-start critical  |
| Distributed systems | Neural models needed |

**Typical Results (MovieLens 100K):**

```shell
Model          Recall@10  NDCG@10
Random         0.0096     0.0036
Popularity     0.0817     0.0399
ALS            0.1316     0.0675  ✓ Best
```

**Embedding Visualization:**

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Extract item factors
item_factors = als.item_factors  # (n_items, k)

# Project to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(item_factors)

# Color by popularity
item_popularity = np.array(R_train.sum(axis=0)).flatten()

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=item_popularity,
            cmap='viridis', alpha=0.6, s=10)
plt.colorbar(label='Popularity')
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
plt.title("ALS Item Embeddings (PCA)")
plt.show()
```

**Interpretation:**

- PC1 often captures **popularity** (mainstream vs niche)
- PC2 captures **genre** or content patterns
- Low variance (10-25%) is normal for distributed embeddings
- Check personalization: different users should get different recs

[(Back to top)](#table-of-contents)
