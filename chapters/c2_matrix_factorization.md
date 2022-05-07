# COLLABORATIVE FILTERING AS MATRIX COMPLETION


# Table of contents
- [Table of contents](#table-of-contents)
- [1. Matrix Representation of Rating Data](#1-matrix-representation-of-rating-data)
  - [1.1. Matrix Completion](#11-matrix-completion)
- [2. Singular Value Decomposition (SVD)](#2-singular-value-decomposition) 
  - [2.1. How to perform SVD](#21-how-to-perform-svd) 
  - [2.2. Truncated SVD](#22-truncated-svd)
  - [2.3. Problem with SVD](#23-problem-with-svd)
- [3. Matrix Factorization](#3-matrix-factorization)
  - [3.1. Estimating Latent Factors](#31-estimating-latent-factors) 
    - [3.1.1. Interpreting Latent Factors](#311-interpreting-latent-factors) 
  - [3.2. Enhancement with Regularization](#32-enhancement-with-regularization)
  - [3.3. User and Item Biases](#33-user-and-item-biases)
- [4. Variants](#4-variants)
  - [4.1. Orthnonormality SVD ](#41-orthnonormality-svd) 
  - [4.2. Non-negative Matrix Factorization (NMF)](#42-non-negative-matrix-factorization)

# 1. Matrix Representation of Rating Data
- Sparse Matrix: there are a lot of missing values
<p align="center">
  <img src="https://user-images.githubusercontent.com/64508435/164448804-3430ba28-72ec-4f22-92cd-33e48ea1dbf6.jpeg" width="800" />
</p>

## 1.1. Matrix Completion
- **Goal**: Find the ratings that currently not being observed
- Commonly formulated as finding a low-rank decomposition of the matrix
<img width="600" alt="Screenshot 2022-04-21 at 19 30 41" src="https://user-images.githubusercontent.com/64508435/164448994-46dc8c7e-02e6-4ac5-b0ea-807bae5a0ba1.png">

[(Back to top)](#table-of-contents)

# 2. Singular Value Decomposition
- Classical way to decompose (factorize) a matrix
- **Rank of the Matrix**: dimensionality
  -  Say, for Matrix X, every users has a vector representation of M (items) dimensions, so the User Matrix X has `Rank = M`
  -  Also, for Matrix X, every items has a vector representation of N (users) dimensions, so the Item Matrix X has `Rank = N`
- Assume: N (users) > M (items)
<p align="center">
  <img src="https://user-images.githubusercontent.com/64508435/167238911-105debd9-c636-42db-8b7c-2df68086ea35.jpeg" width="800" />
</p>

- Note:
  - For Users, we did not reduce the dimension, still M
  - For Items, we used the dimension to M, instead of N.

## 2.1. How to perform SVD
<img src="https://user-images.githubusercontent.com/64508435/164450077-cdc5d277-cab9-4da2-a5b0-1e0cd876c7bd.jpeg" width="600" />

## 2.2. Truncated SVD
- The eigenvalues of the covariance matrix (by extension also the singular values of SVD) would decay fast
- We can thus “truncate” the singular vectors and the singular values to approximate 𝑿 with fewer parameters
- Truncated SVD: lower the rank (dimensionality) from M to K, where K << M. 
<img src="https://user-images.githubusercontent.com/64508435/167239197-2f2f9f33-1cef-4c7c-b901-ec0516f1e404.jpeg" width="800" />
<p align="center">
  <img src="https://user-images.githubusercontent.com/64508435/164450889-20e0aa72-b806-4e2c-b966-abda5f8c1073.jpeg" width="800" />
  <br> Lower Rank Approximation
</p>

## 2.3. Problem with SVD
- Problem: SVD only can work on fully-specified matrix (not the spare matrix like our problem)
- Solution: Matrix Completion

#### Matrix Completion for SVD
- Zero-filling: Put zeros wherever missing observation
- Mean imputation: Estimate missing information with the mean of each row (or column)
- **Pipeline**: `In-complete Matrix` &#8594; `Imputation` (filling Missing Values) &#8594; `Truncated SVD` (Match the values partially only) &#8594; `Complete Matrix` (Those missing value will be filled based on Trucated SVD)
- **Problem**: Severe overfitting if the original matrix is sparse, not perfectly matched with original nature of the dataset.

[(Back to top)](#table-of-contents)

# 3. Matrix Factorization
- **Case Study: Netflix Dataset** 
  - Training set:
    - ~100 million ratings
    - ~480,000 anonymous customers
    - 17,770 movies
    - Each movie being rated on a scale of 1 to 5 stars
  - Qualifying set: 1.4 million ratings for quiz set and 1.4 million ratings for test set
  - Quote from winning Team: Our winning entries consist of more than 100 different predictor sets, the majority of which are **factorization** models 
<p align="center"><img width="600" alt="Screenshot 2022-04-21 at 19 54 44" src="https://user-images.githubusercontent.com/64508435/164452794-7ac44f05-9941-4761-b9c1-5f408922c3a6.png"></p>

- **Matrix Factorization**: another method to decomposing a sparse matrix or filling missing values within the matrix
- Find a vector U (item)  & V (item). 
  - Each row of U represents a user vector with k dimension
  - Each column of V represents a item vector with k dimension  
  - Example: Rank-2 Matrix Factorization
<p align="center"><img width="800" alt="Screenshot 2022-04-21 at 19 54 44" src="https://user-images.githubusercontent.com/64508435/164461463-659d428b-8ed8-4ac6-ba23-7685e876ac6c.jpeg"></p>

## 3.1. Estimating Latent Factors
- **Latent Factor**: a factor that is not observed, but extracted from the observed matrix.
- **Minimize loss function**:
  - Key Ideas: only calculate the loss based on the ratings seen (Not the entire matrix like SVD)
  - Intuiation: if user  i and item j are similar (or having the same k-value), we will achieve a higher dot product (14 vs 10) in below example.
<p align="center"><img width="800" alt="Screenshot 2022-04-21 at 19 54 44" src="https://user-images.githubusercontent.com/64508435/164455436-677c32f3-f333-4167-9e83-06b8484a0720.jpeg"></p>

### 3.1.1. Interpreting Latent Factors
- To gain a sense of the inner workings of matrix factorization techniques, let us see how the latent factors could give us some information about how items are associated with one another, as well as which items tend to be associated with a particular factor.
- Note: **Dimension has the most information is the dimension that has the highest variance**
  -  If a factor or dimension has higher variance, it is better at discriminating among different items. 
  -  Conversely, if the variance of a factor is too small, it implies that most items would have a similar value for that factor.
- Example 1: Latent dimensions across different users.
<p align="center"><img width="800" alt="Screenshot 2022-04-21 at 19 54 44" src="https://user-images.githubusercontent.com/64508435/167255191-d7628b49-5c23-42a2-a4a8-25f8954ef747.jpeg"></p>

- Example 2: the variance of each latent dimension across different items.
  - In this case, the top dimension is dim 0 and dim 2. 
<p align="center"><img width="500" alt="Screenshot 2022-04-21 at 19 54 44" src="https://user-images.githubusercontent.com/64508435/167255229-4453478d-206e-4ce3-8449-33fec6cce806.png"></p>


## 3.2. Enhancement with Regularization
- Improving the generalization performance of the model
- Constraining the complexity of models to avoid overfitting
- Allow the regularization term to be raised to different powers q:
  - The larger the q (say q=4 vs q=0.5), the more uniform (high values in all dimension) the element of vector is 
<p align="center"><img width="800" alt="Screenshot 2022-04-21 at 19 54 44" src="https://user-images.githubusercontent.com/64508435/164456339-c458878a-cd3b-46fa-80af-a9b566335f5f.jpeg"></p>

- Use regularization to prevent overfitting, but sometimes too much regularization will end up under fitting. Need to fine tune.
- Due to sparsity, some users or items may have very few ratings
- To prevent overfitting, we can introduce regularization:
<p align="center"><img width="800" alt="Screenshot 2022-04-21 at 19 54 44" src="https://user-images.githubusercontent.com/64508435/167247785-440ac021-0e28-4850-9a76-3e4fcc60e45e.jpeg"></p>

## 3.3. User and Item Biases
- Users have different ranges of ratings
  - some are generous with their ratings mainly in the upper range, others are strict
- Items differ in popularity or likeability
  - some have mainly high ratings, others mainly low
- Introduce bias parameters:
<p align="center"><img width="600" alt="Screenshot 2022-04-21 at 19 54 44" src="https://user-images.githubusercontent.com/64508435/164457193-a7bfe723-a7b7-4058-b868-18c3cd1a079f.jpeg">
<img width="800" alt="Screenshot 2022-04-21 at 19 54 44" src="https://user-images.githubusercontent.com/64508435/164457212-bcd59e63-31fc-4c48-bf17-56429d127775.jpeg">
</p>

- Absorbing Bias Terms into the Latent Factors
<img width="600" alt="Screenshot 2022-05-07 at 17 39 08" src="https://user-images.githubusercontent.com/64508435/167248677-5f1ce20f-f236-4efd-bbf6-cb5fa39636e0.png">

[(Back to top)](#table-of-contents)

# 4. Variants
- Constraining the latent factors of matrix factorization models
## 4.1. Orthnonormality SVD  
- SVD can be seen as matrix factorization (Loss Function concept) with orthonormality constraints
- Orthonormality constraints: each element vector in U must be orthogonal to each other, same for V.
<img width="600" alt="Screenshot 2022-05-07 at 17 39 08" src="https://user-images.githubusercontent.com/64508435/164462212-39131d0f-f365-4b85-8df2-6ea0e338a1c9.jpeg">

## 4.2. Non-negative Matrix Factorization
- Similar to MF, but add the constraint that the value must be non-negative as the training data (observation) are non-negative
<img width="700" alt="Screenshot 2022-05-07 at 17 39 08" src="https://user-images.githubusercontent.com/64508435/167249255-eb3b3cd0-6cde-4051-b0fc-e6f6ec6621aa.jpeg">

### 4.2.1. Sum-of-Parts Interpretation
<img width="700" alt="Screenshot 2022-05-07 at 17 39 08" src="https://user-images.githubusercontent.com/64508435/164463050-a94c2b7e-51b6-4573-ac5c-0b5cab0e70dc.jpeg">

- What are singular values? 
  - Ans: lambda values in the sigma matrix, ordered by the magnitude 
- How many of them are there?
- What is the effect of the number of latent factors 𝑘 on the RMSE of matrix factorization (MF)?
- Is there such a thing as too much regularization? Why?
- Is there an equivalent concept to MF bias terms in neighborhood-based collaborative filtering? What are the similarities or differences?
- Do we expect non-negative MF to achieve better RMSE than MF? Why?
- How does non-negativity affect interpretability in terms of: 
  - visualizing the factors
  - identifying clusters

[(Back to top)](#table-of-contents)
