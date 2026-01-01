# Today I Learn

## Day 1

### Linear Algebra

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
