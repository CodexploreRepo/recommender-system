# Introduction to Recommender Systems
- Basics of Recommender Systems = Interactions between users and items

# Table of contents
- [Table of contents](#table-of-contents)
- [1. Concepts](#1-concepts)
  - [1.1. Explicit vs Implicit Feedback](#11-explicit-vs-implicit-feedback)
  - [1.2. Formulation](#12-formulation)

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

## 1.2. Formulation
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

[(Back to top)](#table-of-contents)
