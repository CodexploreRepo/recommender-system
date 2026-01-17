# Recommender Systems (RecSys) Learning Roadmap

This document provides a comprehensive overview of the core concepts, algorithms, and evaluation techniques used in modern recommendation systems, as detailed in the sources.

### 1. Understanding User Feedback

Recommender systems primarily rely on two types of data to model user preferences:

- **Explicit Feedback:** Direct input from users, such as star ratings or thumbs-up/down buttons.
- **Implicit Feedback:** Indirect signals tracked through user behaviour, such as **purchase history, browsing activity, or click-throughs**.
  - **Asymmetry:** Unlike explicit feedback, implicit data lacks clear negative signals; a non-interaction could mean a user dislikes an item or simply does not know it exists.
  - **Confidence:** In implicit systems, numerical values (like watch time) often represent **confidence** in a preference rather than the magnitude of the preference itself.
- **Data Sparsity:** Real-world datasets often suffer from sparsity and **popularity bias**, where a few items receive most of the interactions.

### 2. The RecSys Pipeline

A production-grade recommendation system typically follows a multi-stage architecture:

1.  **Retrieval:** Filtering millions of items down to a few hundred candidates.
2.  **Ranking:** Using complex models to predict the exact preference score for these candidates.
3.  **Reranking:** Adjusting the list for business logic, **diversity, novelty, and coverage**.
4.  **Feedback Loop:** Continuously updating the model based on new user actions.

### 3. Core Algorithms

#### Matrix Factorization (MF)

MF maps users and items into a shared latent factor space where their similarity can be compared via inner products.

- **Implicit ALS (Alternating Least Squares):** Tailored for implicit feedback by introducing a **confidence matrix** ($C_{ui}$) and preference variables ($p_{ui}$). It alternates between fixing user factors to solve for item factors and vice versa to minimize a quadratic cost function.
- **BPR (Bayesian Personalized Ranking):** A state-of-the-art **pairwise ranking** approach. Instead of predicting a single score for one item, BPR optimizes the model to rank an observed item higher than a non-observed item for a specific user.

#### Deep Learning & Sequence Models

- **Item2vec:** Interprets user interaction sequences as "sentences" and items as "words," applying skip-gram techniques to learn item embeddings.
- **NeuMF (Neural Matrix Factorization):** Combines a Generalized Matrix Factorization (GMF) layer with a Multi-Layer Perceptron (MLP) to model complex user-item interactions.
- **Two-Tower Models:** Use separate neural networks (towers) for users and items, often trained with **in-batch negatives** to scale retrieval.
- **Sequential Models (BERT4Rec):** Utilise bidirectional attention and a **Cloze objective** (predicting masked items in a sequence) to capture evolving user behavior.

#### Graph-Based Models

These treat the dataset as a **bipartite graph** of users and items. Models like **GraphSAGE** use neighborhood sampling and aggregation to generate embeddings based on the graph structure.

### 4. Exploration and Multi-Armed Bandits (MAB)

To solve the **cold-start problem** and balance discovery with accuracy, MAB algorithms are used:

- **$\epsilon$-greedy:** Randomly explores new items a small percentage of the time.
- **UCB1 (Upper Confidence Bound):** Selects items based on their potential reward and uncertainty.
- **Thompson Sampling:** Uses a probabilistic approach to exploration.
- **Contextual Bandits (Linear UCB):** Incorporates user features to make exploration decisions more personalized.

### 5. Evaluation Metrics

Models are evaluated using both offline metrics and online A/B testing:

- **Recall@K & Precision@K:** Measuring how many relevant items appear in the top $K$ results.
- **NDCG (Normalized Discounted Cumulative Gain):** Rewards models for placing relevant items higher in the list.
- **AUC (Area Under the ROC Curve):** Measures the probability that a randomly chosen positive item is ranked higher than a randomly chosen negative one.
- **Expected Percentile Ranking:** A measure where lower values indicate that actually watched items were ranked closer to the top of the list.

### 6. Implementation Notes

- **Negative Sampling:** Essential for training implicit models; techniques include random sampling, popularity-weighted sampling, and **hard negative mining**.
- **Time Decay:** Weighting recent interactions more heavily to reflect current interests.
- **Scalability:** Efficient implementations use **Coordinate Descent** or **Stochastic Gradient Descent (SGD)** with bootstrap sampling (LearnBPR) to handle billions of user-item pairs.

---

**Analogy for Item2vec:** Think of a user's shopping history like a **story**. Each item they buy is a **word** in that story. By looking at thousands of these stories, the model learns that "bread" often appears near "butter," just like a language model learns which words belong together in a sentence.
