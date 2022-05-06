# Introduction to Recommender Systems
- Basics of Recommender Systems = Interactions between users and items

# Table of contents
- [Table of contents](#table-of-contents)
- [1. Concepts](#1-concepts)
  - [1.1. Explicit vs Implicit Feedback](#11-explicit-vs-implicit-feedback)
  - [1.2. Problem Formulation](#12-problem-formulation)
  - [1.3. How to solve the problem: Collaborative & Content-Based Filtering](#13-how-to-solve-the-problem)
  - [1.4. Multimodality](#14-multimodality)
  - [1.5. Methodology: Algorithms to solve the problem](#15-methodology)
    - [1.5.1. Memory-Based: Neighborhood Collaborative Filtering](#151-memory-based) 
    - [1.5.2. Model-Based: Matrix Factorization](#152-model-based)
  - [1.6. Contextual Awareness](#16-contextual-awareness)
  - [1.7. Explainability](#17-explainability)
- [2. Common Issues of recommendation datasets](#2-common-issues-of-recommendation-datasets) 
  - [2.1. Sparsity](#21-sparsity)
  - [2.2. Power-Law distribution](#22-power-law-distribution) 
- [3. Cornac: Python-based recommender library](#3-cornac)
  - [3.1. Cornac Introduction](#31-cornac-introduction) 

# 1. Concepts
## 1.1. Explicit vs Implicit Feedback
- **Feedback**: interaction between users and items 
### 1.1.1. Explicit Feedback
- Stated clearly and readily observable
- Ratings
- Thumbs up/down
### 1.1.2. Implicit Feedback
- Implicit = feedback embedded into user's behaviour like (watch time, number of clicks, or browsing)
- Suggested, though not directly expressed

## 1.2. Problem Formulation
- There are two types of problem formulation: Rating Prediction and Ranking
<p align="center">
  <img src="https://user-images.githubusercontent.com/64508435/167171900-4a732866-fd9c-44b0-9cc0-6e4919bd9c1e.png" width="600" />
</p>

### 1.2.1. Rating Prediction 
- Look at the rating that does not exist in the data & try to predict them.
- “Is the user going to like this item?”
- Usually involves a threshold applied on the predicted real value

### 1.2.2. Ranking
- To rank the options/items
- ”Which items are the user most likely to have an interest?”
- Known as **top-k** list of recommendation

## 1.3. How to solve the problem
- There are two way to solve the problem:
  - Collaborative Filtering: Interaction between users and items
  - Content-Based Filtering: Contents associated with the items
<p align="center">
  <img src="https://user-images.githubusercontent.com/64508435/167173215-75c3565f-fb12-496a-a434-c67213dae170.png" width="600" />
</p>

- Readings: 
  - [Tutorial: Content-based Recommender Using Natural Language Processing (NLP)](https://www.kdnuggets.com/2019/11/content-based-recommender-using-natural-language-processing-nlp.html)

### 1.3.1. Collaborative Filtering
- Basic idea: A user tends to have similar consumption behavior to other ‘like-minded’ users
- Based on user rating and consumption to group similar users together, then to recommend products/services to users
- Tools: **Mix between Matrix Factorization + Deep Learning** (Convolutional Matrix Factorization) due to the Sparsity of the matrix 
<p align="center">
  <img src="https://user-images.githubusercontent.com/64508435/167178893-de39db77-d890-4cd1-8acf-d5e799f03ad5.png" width="600" />
  <br>Convolutional Matrix Factorization
</p>


### 1.3.2. Content-Based Filtering
- A user tends to like items with similar contents to those previously consumed &#8594; focus on the content of the items
- to make recommendations based on similar products/services according to their attributes.
- Tools: Deep Learning Model
## 1.4. Multimodality
- Content can be intepreted in different ways such as:
  - Rating
  - Product images
  - Product description
  - Related products 
- Modalities:
  - Images
  - Text
  - Network

## 1.5. Methodology
- Method 1: **Memory-Based** (similar to a search problem)
  - Memorizes the data itself (data in this case is the interaction between users and items)
  - Employs algorithms to search for recommended items
  - *Pros*: Minimal preprocessing or learning is involved
  - *Cons*: Significant memory storage requirement
- Method 2: **Model-Based** ⭐ (most of the time we will focus to build this model)
  - Instead of memorizing the entire data, we will build a model, so that we can forget about the data
  - Employs algorithms to learn a model from the dataset
  - *Pros*: Significant preprocessing or learning involved
  - *Cons*: Much smaller memory requirement

### 1.5.1. Memory-Based
### 1.5.2. Model-Based
- The original dataset is modelled as user-item matrix &#8594; perform the low-rank matrix-factorization such that
  - Each user will have a representative vector
  - Each item will also have a representative vector
<p align="center">
  <img src="https://user-images.githubusercontent.com/64508435/167178092-2fb90a1f-6dc2-4b1f-a52b-1b6562c836a4.jpeg" width="600" />
</p>

## 1.6. Contextual Awareness
- Recommendation varies depending on **context** (i.e: we can recommend different things to a person depending on the context)
  - `User context`
    - Demographics e.g., age, gender
    - Social network, e.g., like you go with this group of friends, you will watch this movie, but other groups will watch other movies
  - `Item context`
    - Categories, e.g., genres – location
  - `Rating context`
    - Time of day 
    - Tromotions
<p align="center">
  <img src="https://user-images.githubusercontent.com/64508435/167179410-7c7fa8c0-379a-4d38-b41f-6fbbca8c2c41.png" width="600" />
</p>

## 1.7. Explainability
<p align="center">
  <img src="https://user-images.githubusercontent.com/64508435/167179649-85542dc9-df59-4e7c-ae48-486d28de0104.png" width="600" />
</p>

[(Back to top)](#table-of-contents)

# 2. Common Issues of recommendation datasets
## 2.1. Sparsity
<p align="center">
<img width="800" alt="Screenshot 2022-05-07 at 01 16 44" src="https://user-images.githubusercontent.com/64508435/167180945-3e9cd473-d9e2-4941-a0c8-d3c78decaf27.png">
</p>

## 2.2. Power-Law distribution
[(Back to top)](#table-of-contents)

# 3. Cornac
## 3.1. Cornac Introduction
- Website: https://cornac.preferred.ai
<p align="center">
  <img src="https://user-images.githubusercontent.com/64508435/167180117-7b2d4f1c-53ab-47e3-903b-b19a732c3e2d.png" width="800" />
  <img width="800" alt="Screenshot 2022-05-07 at 01 12 47" src="https://user-images.githubusercontent.com/64508435/167180327-8e07f75c-0ca0-4945-a922-b6090ba4c075.png">
</p>


[(Back to top)](#table-of-contents)
