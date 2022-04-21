# COLLABORATIVE FILTERING AS MATRIX COMPLETION

# 2. Matrix Representation of Rating Data

![IMG_284DAEA71AB1-1](https://user-images.githubusercontent.com/64508435/164448804-3430ba28-72ec-4f22-92cd-33e48ea1dbf6.jpeg)

## 2.1. Matrix Completion
<img width="1216" alt="Screenshot 2022-04-21 at 19 30 41" src="https://user-images.githubusercontent.com/64508435/164448994-46dc8c7e-02e6-4ac5-b0ea-807bae5a0ba1.png">

- Commonly formulated as finding a low-rank decomposition of the matrix

## 2.2. Singular Value Decomposition
- N (users) > M 
![IMG_BCF099E76677-1](https://user-images.githubusercontent.com/64508435/164450477-2872eabd-c12e-4d02-8101-158b0698e292.jpeg)


![IMG_2D5A64A846A6-1](https://user-images.githubusercontent.com/64508435/164450077-cdc5d277-cab9-4da2-a5b0-1e0cd876c7bd.jpeg)


### 2.2.1. Truncated SVD
- Truncated SVD: lower the rank from M to K 
<p align="center">
  <img src="https://user-images.githubusercontent.com/64508435/164450889-20e0aa72-b806-4e2c-b966-abda5f8c1073.jpeg" width="800" />
  <br> Lower Rank Approximation
</p>

- Problem with SVD: SVD only can work on fully-specified matrix (not the spare matrix like our problem)

### 2.2.2. SVD for Incomplete Matrix
- In-complete Matrix &#8594; Imputation &#8594; Truncated SVD &#8594; Complete Matrix
  - Zero-filling: Put zeros wherever missing observation
  - Mean imputation: Estimate missing information with the mean of each row (or column)
- Problem: Severe overfitting if the original matrix is sparse

# 3. Matrix Factorization
- **Matrix Factorization**: Decomposing a sparse matrix
- Netflix Dataset: 
- Training set:
  - ~100 million ratings
  - ~480,000 anonymous customers
  - 17,770 movies
  - each movie being rated on a scale of 1 to 5 stars
- Qualifying set: 1.4 million ratings for quiz set and 1.4 million ratings for test set

<img width="1216" alt="Screenshot 2022-04-21 at 19 54 44" src="https://user-images.githubusercontent.com/64508435/164452794-7ac44f05-9941-4761-b9c1-5f408922c3a6.png">

- Example: Rank-2 Matrix Factorization

![IMG_8A95FD52123F-1](https://user-images.githubusercontent.com/64508435/164453629-6b0a2909-69ac-4c1a-a567-8e09b85129bc.jpeg)

## 3.1. Estimating Latent Factors
- Minimize loss function:
- Key Ideas: only calculate the loss based on the ratings seen (Not the entire matrix like SVD)
- Intuiation: if user  i and item j are similar (or having the same k-value), we will achieve a higher dot product (14 vs 10) in below example.

![IMG_5EE3A720973B-1](https://user-images.githubusercontent.com/64508435/164455436-677c32f3-f333-4167-9e83-06b8484a0720.jpeg)

## 3.2. ENHANCEMENTS
- Improving the generalization performance of the model
- Constraining the complexity of models to avoid overfitting
- Allow the regularization term to be raised to different powers q:
![IMG_DF70DE636022-1](https://user-images.githubusercontent.com/64508435/164456339-c458878a-cd3b-46fa-80af-a9b566335f5f.jpeg)

- Due to sparsity, some users or items may have very few ratings
- To prevent overfitting, we can introduce regularization:
- ![IMG_AEF925E5F7D5-1](https://user-images.githubusercontent.com/64508435/164456693-87ed6f1b-264b-41ac-9015-8c572c25efd5.jpeg)

## 3.3. User and Item Biases
- Users have different ranges of ratings
  - some are generous with their ratings mainly in the upper range, others are strict
- Items differ in popularity or likeability
  - some have mainly high ratings, others mainly low
- Introduce bias parameters:
![IMG_477F178B2855-1](https://user-images.githubusercontent.com/64508435/164457193-a7bfe723-a7b7-4058-b868-18c3cd1a079f.jpeg)
![IMG_8E352F42E671-1](https://user-images.githubusercontent.com/64508435/164457212-bcd59e63-31fc-4c48-bf17-56429d127775.jpeg)

