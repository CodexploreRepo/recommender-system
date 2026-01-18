# MATRIX FACTORIZATION: ALS & BPR

# Table of Contents

- [Table of Contents](#table-of-contents)
- [Part 1: Alternating Least Squares (ALS)](#part-1-alternating-least-squares-als)
  - [1.1. What is ALS?](#11-what-is-als)
    - [1.1.1. Why ALS?](#111-why-als)
  - [1.2. ALS Algorithm](#12-als-algorithm)
    - [1.2.1. Objective](#121-objective)
    - [1.2.2. Closed-Form Solution](#122-closed-form-solution)
  - [1.3. Implicit Feedback](#13-implicit-feedback)
    - [1.3.1. Confidence Weighting](#131-confidence-weighting)
  - [1.4. Implementation](#14-implementation)
    - [1.4.1. Hyperparameters](#141-hyperparameters)
    - [1.4.2. Code Example](#142-code-example)
  - [1.5. Key Insights](#15-key-insights)
- [Part 2: Bayesian Personalized Ranking (BPR)](#part-2-bayesian-personalized-ranking-bpr)
  - [2.1. What is BPR?](#21-what-is-bpr)
    - [2.1.1. Why BPR?](#211-why-bpr)
    - [2.1.2. Key Insight](#212-key-insight)
  - [2.2. BPR Algorithm](#22-bpr-algorithm)
    - [2.2.1. Pairwise Ranking Objective](#221-pairwise-ranking-objective)
    - [2.2.2. BPR Optimization Criterion](#222-bpr-optimization-criterion)
    - [2.2.3. Learning Algorithm](#223-learning-algorithm)
  - [2.3. BPR vs ALS](#23-bpr-vs-als)
    - [2.3.1. Fundamental Differences](#231-fundamental-differences)
    - [2.3.2. When to Use Which](#232-when-to-use-which)
  - [2.4. Implementation](#24-implementation)
    - [2.4.1. Hyperparameters](#241-hyperparameters)
    - [2.4.2. Code Example](#242-code-example)
- [Part 3: Two-Stage Recommender System](#part-3-two-stage-recommender-system)
  - [3.1. Architecture](#31-architecture)
  - [3.2. When Two-Stage Actually Helps](#32-when-two-stage-actually-helps)
  - [3.3. When NOT to Use Two-Stage](#33-when-not-to-use-two-stage)
  - [3.4. Common Mistakes](#34-common-mistakes)
  - [3.5. Implementation](#35-implementation)
  - [3.6. Case Studies](#36-case-studies)
    - [3.6.1. MovieLens 100K (When Single-Stage Wins)](#361-movielens-100k-when-single-stage-wins)
    - [3.6.2. Production E-Commerce (When Two-Stage Wins)](#362-production-e-commerce-when-two-stage-wins)
  - [3.7. Best Practices](#37-best-practices)
- [Part 4: Evaluation Metrics](#part-4-evaluation-metrics)
  - [4.1. Hit Rate@K](#41-hit-ratek)
  - [4.2. NDCG@K](#42-ndcgk)
  - [4.3. MAP@K](#43-mapk)
  - [4.4. Recall@K](#44-recallk)
- [Part 5: Summary](#part-5-summary)

---

# Part 1: Alternating Least Squares (ALS)

## 1.1. What is ALS?

**Alternating Least Squares (ALS)** decomposes the user-item interaction matrix **R** into two lower-rank matrices:

```
R ≈ U × Vᵀ

where:
  U (n_users × k)   = user latent factors
  V (n_items × k)   = item latent factors
  k                 = number of latent dimensions
```

### 1.1.1. Why ALS?

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

## 1.2. ALS Algorithm

### 1.2.1. Objective

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

### 1.2.2. Closed-Form Solution

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

## 1.3. Implicit Feedback

### 1.3.1. Confidence Weighting

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

## 1.4. Implementation

### 1.4.1. Hyperparameters

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

### 1.4.2. Code Example

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

## 1.5. Key Insights

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

---

# Part 2: Bayesian Personalized Ranking (BPR)

## 2.1. What is BPR?

**Bayesian Personalized Ranking (BPR)** is a state-of-the-art learning algorithm specifically designed for **implicit feedback** recommendation systems. Unlike traditional methods that predict absolute ratings, BPR learns to **rank items** relative to each other.

### 2.1.1. Why BPR?

**Problem with pointwise approaches (like ALS):**
- Treat recommendation as regression/classification
- Predict individual item scores independently
- Don't directly optimize for ranking quality

**BPR's solution:**
- **Pairwise ranking**: Learn relative preferences between items
- Directly optimize the ordering of recommendations
- Better suited for top-K recommendation tasks

### 2.1.2. Key Insight

**Core Assumption:**
```
If user u interacted with item i but not with item j
  ⟹ User u prefers item i over item j
  ⟹ Model should rank i higher than j for user u
```

**Example:**
```shell
User watched Movie A ✓
User didn't watch Movie B ✗
  → BPR learns: score(u, A) > score(u, B)
```

This pairwise comparison is more informative than treating items independently.

[(Back to top)](#table-of-contents)

## 2.2. BPR Algorithm

### 2.2.1. Pairwise Ranking Objective

**Training Data:**
- Observed interactions: **D_s** = {(u, i) | user u interacted with item i}
- Training triples: **(u, i, j)** where:
  - u = user
  - i = positive item (observed)
  - j = negative item (not observed)

**Preference Assumption:**
```
(u, i, j) ∈ D_s  ⟹  user u prefers i over j
```

### 2.2.2. BPR Optimization Criterion

**Objective Function:**
```
BPR-OPT = Σ_(u,i,j) ln σ(x̂_uij) - λ_Θ ||Θ||²

where:
  x̂_uij = x̂_ui - x̂_uj          # Score difference
  σ(x)  = 1 / (1 + e^(-x))      # Sigmoid function
  Θ     = model parameters       # (user/item factors)
  λ_Θ   = regularization weight
```

**Interpretation:**
- **x̂_ui**: Predicted score for user u and item i
- **x̂_uj**: Predicted score for user u and item j
- **x̂_uij > 0**: Model correctly ranks i above j
- **σ(x̂_uij)**: Probability that i is ranked higher than j

**For Matrix Factorization:**
```
x̂_ui = u_u^T · v_i

where:
  u_u ∈ ℝ^k  = user u's latent factor
  v_i ∈ ℝ^k  = item i's latent factor
```

### 2.2.3. Learning Algorithm

**Model Parameters:**
```
Θ = {U, V}    # All trainable parameters

where:
  U ∈ ℝⁿˣᵏ   # User latent factor matrix
  V ∈ ℝᵐˣᵏ   # Item latent factor matrix
  n         # Number of users
  m         # Number of items
  k         # Number of latent dimensions (factors)
```

**LearnBPR (Stochastic Gradient Descent):**

```python
# Initialize parameters randomly
U ~ N(0, 0.01)  # User factors: (n_users × k)
V ~ N(0, 0.01)  # Item factors: (n_items × k)

for iteration in iterations:
    # Bootstrap sampling
    for (u, i, j) in bootstrap_sample(D_s):
        # Compute scores using current factors
        x̂_ui = u_u^T · v_i
        x̂_uj = u_u^T · v_j
        x_uij = x̂_ui - x̂_uj

        # Compute gradients
        σ_uij = σ(-x_uij)

        ∂L/∂u_u = -σ_uij · (v_i - v_j) + λ · u_u
        ∂L/∂v_i = -σ_uij · u_u + λ · v_i
        ∂L/∂v_j =  σ_uij · u_u + λ · v_j

        # Update parameters (gradient ascent)
        u_u ← u_u - α · ∂L/∂u_u
        v_i ← v_i - α · ∂L/∂v_i
        v_j ← v_j - α · ∂L/∂v_j

where:
  α = learning rate
  λ = regularization parameter
```

**Key Components:**
1. **Bootstrap Sampling**: Randomly sample triples (u, i, j) with replacement
2. **Negative Sampling**: For each positive item i, randomly sample negative j
3. **SGD Updates**: Update parameters based on pairwise ranking loss

[(Back to top)](#table-of-contents)

## 2.3. BPR vs ALS

### 2.3.1. Fundamental Differences

| Aspect | ALS | BPR |
|--------|-----|-----|
| **Objective** | Pointwise (predict scores) | Pairwise (rank items) |
| **Loss Function** | Squared error | Pairwise ranking loss |
| **Optimization** | Alternating closed-form | Stochastic gradient descent |
| **Training** | Faster (closed-form) | Slower (iterative SGD) |
| **Ranking Quality** | Indirect | Direct optimization |
| **Use Case** | Candidate retrieval | Precision ranking |

### 2.3.2. When to Use Which

**Use ALS for:**
- ✓ Fast candidate generation (retrieval stage)
- ✓ Large-scale batch processing
- ✓ When speed is critical
- ✓ Initial embeddings

**Use BPR for:**
- ✓ Precision ranking (ranking stage)
- ✓ Top-K recommendation quality
- ✓ When ranking order matters most
- ✓ Fine-tuning recommendations

**Best Practice: Combine Both (Two-Stage)**
```
Stage 1: ALS retrieval (100-1000 candidates)
Stage 2: BPR ranking (top-10 results)
```

[(Back to top)](#table-of-contents)

## 2.4. Implementation

### 2.4.1. Hyperparameters

| Parameter | Range | Description | Impact |
|-----------|-------|-------------|--------|
| **factors** | 32-128 | Latent dimensions | Higher = more expressive |
| **learning_rate** | 0.01-0.1 | Step size for SGD | Too high = unstable, too low = slow |
| **regularization** | 0.001-0.1 | L2 penalty | Higher = less overfitting |
| **iterations** | 30-100 | Training epochs | More = better convergence |
| **verify_negative_samples** | True/False | Check negatives not in training | Prevents false negatives |

**Tuning Strategy:**
```python
# Conservative defaults
factors = 64
learning_rate = 0.05
regularization = 0.01
iterations = 50
verify_negative_samples = True

# Monitor: train_auc should increase (>90% is good)
# Adjust learning_rate if training is unstable
```

### 2.4.2. Code Example

**Training:**

```python
from implicit.bpr import BayesianPersonalizedRanking
from scipy.sparse import csr_matrix

# Prepare implicit feedback (binary)
df["implicit"] = (df["rating"] >= 4).astype(int)

# Create sparse matrix
R_train = csr_matrix(
    (df_train["implicit"], (df_train["user"], df_train["item"])),
    shape=(n_users, n_items)
)

# Train BPR
bpr = BayesianPersonalizedRanking(
    factors=64,
    learning_rate=0.05,
    regularization=0.01,
    iterations=50,
    verify_negative_samples=True,
    random_state=42
)

bpr.fit(R_train)

# Monitor output: train_auc should be high (>90%)
# Example: 100%|██████| 50/50 [train_auc=96.96%, skipped=22.80%]
```

**Direct Ranking (BPR Only):**

```python
def rank_with_bpr(bpr_model, user_id, candidate_items, K=10):
    """Rank candidate items using BPR model."""
    user_vec = bpr_model.user_factors[user_id]
    item_vecs = bpr_model.item_factors[candidate_items]

    # Compute scores
    scores = item_vecs @ user_vec

    # Get top-K
    top_k_idx = np.argsort(scores)[::-1][:K]

    return candidate_items[top_k_idx]
```

**Embedding Comparison:**

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Compare ALS vs BPR embeddings
pca = PCA(n_components=2)

X_als = pca.fit_transform(als.item_factors)
X_bpr = pca.fit_transform(bpr.item_factors)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X_als[:, 0], X_als[:, 1], s=3, alpha=0.6)
ax1.set_title("ALS Item Embeddings")

ax2.scatter(X_bpr[:, 0], X_bpr[:, 1], s=3, alpha=0.6)
ax2.set_title("BPR Item Embeddings")

plt.tight_layout()
plt.show()
```

**Observations:**
- BPR embeddings often show clearer clustering
- Both capture latent item similarities
- BPR optimized for ranking → better separation of liked/disliked items

[(Back to top)](#table-of-contents)

---

# Part 3: Two-Stage Recommender System

## 3.1. Architecture

**Motivation:**
- ALS is fast but less precise for ranking
- BPR is precise but too slow to score millions of items
- **Solution**: Combine both in a pipeline

**Pipeline:**
```
All Items (1M+)
      ↓
[ Stage 1: Retrieval (Fast) ]
  e.g., ALS, HNSW, LSH
      ↓
~100-1000 Candidates
      ↓
[ Stage 2: Ranking (Precise) ]
  e.g., BPR, Neural Network, LambdaRank
      ↓
Top-10 Results
```

[(Back to top)](#table-of-contents)

## 3.2. When Two-Stage Actually Helps

**✓ Use two-stage when:**

1. **Large item catalog (millions of items)**

   ```shell
   E-commerce: 10M products
   → BPR too slow to score all items (10M dot products per user)
   → Use fast retrieval to narrow down to 100 candidates
   → BPR re-ranks only 100 items (100x faster!)
   ```

2. **Retrieval model has high recall at top-C**

   ```shell
   Good scenario:
     ALS Recall@10:  0.15  ← What we get with top-10
     ALS Recall@100: 0.45  ← What we get with top-100

     Gap = 0.30 (30% more relevant items in top-100)
     → BPR has room to improve by re-ranking these 100 items

   Bad scenario (MovieLens 100K):
     ALS Recall@10:  0.1316
     ALS Recall@100: 0.15-0.20  ← Only slightly higher!

     Gap = 0.02-0.07 (small improvement)
     → BPR can't help much (retrieval bottleneck)
   ```

3. **Different optimization objectives**

   ```shell
   Stage 1: Optimize for RECALL (find all relevant items)
            → Use collaborative filtering (ALS, HNSW)

   Stage 2: Optimize for PRECISION (rank relevant items correctly)
            → Use learning-to-rank (BPR, LambdaRank)

   Benefit: Each model does what it's best at
   ```

4. **Real-time inference constraints**

   ```shell
   Retrieval: Pre-computed offline
              → ALS embeddings cached
              → Fast nearest-neighbor search

   Ranking:   Real-time scoring
              → Neural network with user features
              → Only scores 100 candidates (fast enough)
   ```

[(Back to top)](#table-of-contents)

## 3.3. When NOT to Use Two-Stage

**✗ Don't use two-stage when:**

1. **Small item catalog**

   ```shell
   MovieLens 100K: 1,674 items
   → BPR can score all items directly in milliseconds
   → Two-stage adds complexity without speed benefit
   ```

2. **Retrieval bottleneck exists**

   ```shell
   If retrieval misses relevant items:
     ALS ranks true item at position 150
     → Not in top-100 candidates
     → BPR can NEVER recommend it
     → Two-stage Recall ≤ Retrieval Recall@C
   ```

3. **Single model performs well**

   ```shell
   If ALS-only achieves:
     Recall@10 = 0.13
     NDCG@10 = 0.067

   And two-stage achieves:
     Recall@10 = 0.10  ← Worse!
     NDCG@10 = 0.058   ← Worse!

   → Stick with simple ALS-only
   ```

[(Back to top)](#table-of-contents)

## 3.4. Common Mistakes

**Mistake 1: Using same model for both stages**

```shell
❌ ALS (retrieval) → ALS (ranking)
   → No benefit, just adds latency

✓ ALS (retrieval) → BPR (ranking)
   → Different optimization objectives
```

**Mistake 2: Too few candidates (C)**

```shell
❌ C = 50 candidates
   → If true item at position 51, lost forever

✓ C = 200-500 candidates
   → Higher recall ceiling for ranking stage
```

**Mistake 3: Not tuning retrieval model**

```shell
❌ Focus only on tuning ranking model (BPR)
   → But retrieval is the bottleneck!

✓ Tune retrieval first to maximize Recall@C
   → Then tune ranking to optimize NDCG@K
```

[(Back to top)](#table-of-contents)

## 3.5. Implementation

**Full Two-Stage Pipeline:**

```python
# Stage 1: Candidate retrieval with ALS
def retrieve_candidates(als_model, R_train, user_id, C=100):
    """Retrieve top-C candidates using ALS."""
    items, _ = als_model.recommend(
        userid=user_id,
        user_items=R_train[user_id],
        N=C,
        filter_already_liked_items=True
    )
    return np.array(items)

# Stage 2: Ranking with BPR
def rank_with_bpr(bpr_model, user_id, candidate_items, K=10):
    """Rank candidates using BPR."""
    user_vec = bpr_model.user_factors[user_id]
    item_vecs = bpr_model.item_factors[candidate_items]

    scores = item_vecs @ user_vec
    top_k_idx = np.argsort(scores)[::-1][:K]

    return candidate_items[top_k_idx]

# Combined recommender
def recommend_2stage(als, bpr, R_train, user_id, C=100, K=10):
    """Two-stage recommendation: ALS → BPR."""
    # Stage 1: Fast retrieval
    candidates = retrieve_candidates(als, R_train, user_id, C)

    # Stage 2: Precision ranking
    top_k = rank_with_bpr(bpr, user_id, candidates, K)

    return top_k
```

**Training Both Models:**

```python
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking

# Train ALS for retrieval
als = AlternatingLeastSquares(
    factors=64,
    regularization=0.1,
    iterations=20
)
als.fit(R_train)

# Train BPR for ranking
bpr = BayesianPersonalizedRanking(
    factors=64,
    learning_rate=0.05,
    regularization=0.01,
    iterations=50,
    verify_negative_samples=True
)
bpr.fit(R_train)

# Get recommendations
user_id = 42
recommendations = recommend_2stage(als, bpr, R_train, user_id, C=100, K=10)
```

[(Back to top)](#table-of-contents)

## 3.6. Case Studies

### 3.6.1. MovieLens 100K (When Single-Stage Wins)

```shell
Dataset: MovieLens 100K (1,674 items, 942 users)

Model Comparison (K=10):
                    Recall@10  NDCG@10  When to Use
──────────────────────────────────────────────────────
ALS-only               0.1316   0.0675  ✓ Best overall
BPR-only               0.1062   0.0577  × Overfits
ALS→BPR (Two-Stage)    0.1072   0.0585  × Worse than ALS

Conclusion: ALS-only wins (simpler + better performance)
```

**Why two-stage failed here:**

1. Small catalog → BPR can score all items directly
2. Retrieval bottleneck → ALS Recall@100 ≈ Recall@10
3. BPR overfitting → Poor ranking even with good candidates
4. Compounding errors → Bad retrieval + bad ranking = worst results

### 3.6.2. Production E-Commerce (When Two-Stage Wins)

```shell
Dataset: 10M products, 100M users

Model Comparison (K=10):
                    Recall@10  NDCG@10  Latency
────────────────────────────────────────────────
BPR-only               0.25     0.15     5000ms  ← Too slow!
ALS-only               0.18     0.10      100ms  ← Fast but worse
ALS→BPR (C=500)        0.24     0.14      150ms  ✓ Best balance

Why two-stage wins:
  - ALS Recall@500 = 0.40 (high ceiling for BPR)
  - BPR re-ranks 500 candidates (500x faster than scoring all 10M)
  - Achieves near-BPR-only quality at 33x lower latency
```

[(Back to top)](#table-of-contents)

## 3.7. Best Practices

1. **Measure retrieval ceiling**

   ```python
   # Check if two-stage is worth it
   als_recall_10 = check_recall(als, test_items, K=10)
   als_recall_100 = check_recall(als, test_items, K=100)
   als_recall_500 = check_recall(als, test_items, K=500)

   print(f"Recall@10:  {als_recall_10:.3f}")
   print(f"Recall@100: {als_recall_100:.3f}")
   print(f"Recall@500: {als_recall_500:.3f}")

   if als_recall_100 - als_recall_10 > 0.1:
       print("✓ Two-stage worth trying (large gap)")
   else:
       print("✗ Stick with single-stage (small gap)")
   ```

2. **Tune retrieval first, ranking second**

   ```shell
   Step 1: Optimize retrieval for Recall@C
           → Tune ALS hyperparameters
           → Goal: Maximize coverage

   Step 2: Optimize ranking for NDCG@K
           → Tune BPR hyperparameters
           → Goal: Improve ordering
   ```

3. **Monitor both stages in production**
   ```python
   # Track metrics for each stage
   metrics = {
       "retrieval_recall@100": 0.45,  # Stage 1 quality
       "ranking_ndcg@10": 0.12,        # Stage 2 quality
       "end_to_end_recall@10": 0.25,   # Final performance
       "retrieval_latency_ms": 50,     # Stage 1 speed
       "ranking_latency_ms": 100,      # Stage 2 speed
   }
   ```

[(Back to top)](#table-of-contents)

---

# Part 4: Evaluation Metrics

## 4.1. Hit Rate@K

**Definition:** Percentage of users for whom at least one relevant item appears in top-K

```python
def hit_rate_at_k(rec_items, test_items, K=10):
    """Hit Rate@K metric."""
    hits = []

    for u, true_items in test_items.items():
        # Did we hit at least one relevant item?
        hit = int(len(set(rec_items[u][:K]) & set(true_items)) > 0)
        hits.append(hit)

    return np.mean(hits)
```

**Interpretation:**
```shell
Hit Rate@10 = 0.126
  → 12.6% of users got at least 1 relevant item in top-10
```

[(Back to top)](#table-of-contents)

## 4.2. NDCG@K

**Normalized Discounted Cumulative Gain** - Rewards placing relevant items higher in the list

```python
def ndcg_at_k(rec_items, test_items, K=10):
    """NDCG@K metric."""
    ndcgs = []

    for u, true_items in test_items.items():
        recs = rec_items[u][:K]

        # Compute DCG
        dcg = 0.0
        for idx, item in enumerate(recs):
            if item in true_items:
                dcg += 1 / np.log2(idx + 2)  # Position discount

        # Compute ideal DCG
        ideal_hits = min(len(true_items), K)
        idcg = sum(1 / np.log2(i + 2) for i in range(ideal_hits))

        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return np.mean(ndcgs)
```

**Interpretation:**
```shell
NDCG@10 = 0.069
  → Accounts for position: item at rank 1 > item at rank 10
```

[(Back to top)](#table-of-contents)

## 4.3. MAP@K

**Mean Average Precision@K** - Average precision across all relevant items

```python
def map_at_k(rec_items, test_items, K=10):
    """MAP@K metric."""
    aps = []

    for u, true_items in test_items.items():
        recs = rec_items[u][:K]

        hit_count = 0
        score = 0.0

        for idx, item in enumerate(recs):
            if item in true_items:
                hit_count += 1
                # Precision at this position
                score += hit_count / (idx + 1)

        # Average over relevant items
        ap = score / min(len(true_items), K) if true_items else 0.0
        aps.append(ap)

    return np.mean(aps)
```

**Interpretation:**
```shell
MAP@10 = 0.052
  → Considers both relevance and position of all hits
```

[(Back to top)](#table-of-contents)

## 4.4. Recall@K

**Definition:** Proportion of relevant items that appear in top-K

```python
def recall_at_k(rec_items, test_items, K=10):
    """Recall@K metric."""
    recalls = []

    for u, true_items in test_items.items():
        recs = rec_items[u][:K]
        n_hit = len(set(recs) & set(true_items))
        n_true = len(true_items)
        recalls.append(n_hit / n_true if n_true > 0 else 0.0)

    return np.mean(recalls)
```

**Interpretation:**
```shell
Recall@10 = 0.1316
  → On average, 13.16% of relevant items appear in top-10
```

[(Back to top)](#table-of-contents)

---

# Part 5: Summary

## Algorithm Comparison

| Aspect                 | ALS                         | BPR                         | Two-Stage (ALS→BPR)         |
| ---------------------- | --------------------------- | --------------------------- | --------------------------- |
| **Objective**          | Pointwise (score)           | Pairwise (rank)             | Retrieval + Ranking         |
| **Optimization**       | Closed-form (fast)          | SGD (slower)                | Both                        |
| **Use Case**           | Retrieval, batch processing | Precision ranking           | Large-scale production      |
| **Item Catalog**       | Any size                    | Small (<100K)               | Large (1M+)                 |
| **Latency**            | Fast (ms)                   | Moderate (10-100ms)         | Moderate (50-200ms)         |
| **Ranking Quality**    | Indirect                    | Direct                      | Best (when applicable)      |
| **Cold Start**         | Poor                        | Poor                        | Poor                        |
| **Parallelizable**     | Highly                      | Partially                   | Stage 1 (yes), Stage 2 (no) |

## When to Use What

**Use ALS-only when:**
- ✓ Small to medium datasets (<1M items)
- ✓ Speed is critical
- ✓ Batch processing workflow
- ✓ Good baseline needed quickly

**Use BPR-only when:**
- ✓ Small item catalog (<100K items)
- ✓ Ranking quality critical
- ✓ Can afford longer training
- ✓ Have quality negative samples

**Use Two-Stage when:**
- ✓ Large item catalog (1M+ items)
- ✓ Retrieval Recall@C >> Recall@K
- ✓ Production system with latency constraints
- ✓ Different models excel at different objectives

## Key Takeaways

1. **ALS** is your **fast, reliable baseline** for collaborative filtering with implicit feedback
2. **BPR** directly optimizes **ranking quality** but requires careful tuning
3. **Two-stage** is an **engineering optimization** for scale, not a guaranteed improvement
4. **Always validate** with experiments on your specific dataset
5. **Simple beats complex** when performance is similar (Occam's Razor)

## Typical Results

**MovieLens 100K (1,674 items):**
```shell
Model              Recall@10  NDCG@10  Training Time
─────────────────────────────────────────────────────
Random             0.0096     0.0036   —
Popularity         0.0817     0.0399   —
ALS-only           0.1316     0.0675   0.07s  ✓ Best
BPR-only           0.1062     0.0577   0.10s
ALS→BPR            0.1072     0.0585   0.17s
```

**Production E-Commerce (10M items):**
```shell
Model              Recall@10  NDCG@10  Latency
──────────────────────────────────────────────
ALS-only           0.18       0.10     100ms
BPR-only           0.25       0.15     5000ms
ALS→BPR (C=500)    0.24       0.14     150ms  ✓ Best
```

**Key Insight:** The best model depends on your **dataset size**, **latency requirements**, and **quality targets**. Start simple (ALS), measure carefully, and only add complexity (BPR, two-stage) when experiments prove it helps.

[(Back to top)](#table-of-contents)