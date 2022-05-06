# Introduction to Recommender Systems
- Basics of Recommender Systems = Interactions between users and items

# Table of contents
- [Table of contents](#table-of-contents)
- [1. Concepts](#1-concepts)
  - [1.1. Explicit vs Implicit Feedback](#11-explicit-vs-implicit-feedback)
  - [1.2. Problem Formulation](#12-problem-formulation)
  - [1.3. How to solve the problem ?](#13-how-to-solve-the-problem)
  - [1.4. Multimodality](#14-multimodality)
  - [1.5. Methodology: algorithm to solve the problem](#15-methodology)
    - [1.5.1. Memory-Based](#151-memory-based) 

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
### 1.3.2. Content-Based Filtering
- A user tends to like items with similar contents to those previously consumed &#8594; focus on the content of the items
- to make recommendations based on similar products/services according to their attributes.

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

[(Back to top)](#table-of-contents)
