# Multimodality

- Items have content that describe them
  - Images,
  - Text
  - Related Productds
<img width="1189" alt="Screenshot 2022-05-12 at 19 28 07" src="https://user-images.githubusercontent.com/64508435/168064631-484e9ee0-558c-46eb-a0be-94d89a39df43.png">

-  Users have “content” too
 - Photos
 - Friends  

- Automatic Feature Extraction:
  - Text: Document Vector (like traditional way: TF-IDF, Word2Vec, BERT)
  - Image: Vector of Features (CNN)
  - Graph: Vector of Edges (Graph Convolution Network)
# Model with Graph Modality
- SoRec: Social Recommendation Extends PMF with Graph Modality
  - Social Network Graph: Graph between Users
  - User-Item Matrix: which we can factorize using MF 
<img width="1189" alt="Screenshot 2022-05-12 at 19 39 16" src="https://user-images.githubusercontent.com/64508435/168066568-21b59112-070b-4706-9165-61c8c15bab8f.png">

- **Factorizing Rating Matrix**:
  - <img width="700" alt="Screenshot 2022-05-12 at 19 41 00" src="https://user-images.githubusercontent.com/64508435/168066843-9a1e30f1-aff2-43b2-bd1e-834cf51abeb3.png">
- **Factorizing Graph Adjacency Matrix**:
  - Graph can be modelled as Adjency matrix between Users 
  - Recommend someone for you to follow
<img width="669" alt="Screenshot 2022-05-12 at 19 41 48" src="https://user-images.githubusercontent.com/64508435/168066955-7ca07fbf-eb94-440f-b913-874438029c5e.png">
<img width="713" alt="Screenshot 2022-05-12 at 19 42 46" src="https://user-images.githubusercontent.com/64508435/168067141-827a6a1c-aa61-4cab-bff4-9d47bc5626f1.png">

- Combine: tie the users’ latent factors in both factorizations by using 𝑼 in both
  -  𝑼i tie with both rating and friends
<img width="1031" alt="Screenshot 2022-05-12 at 19 47 19" src="https://user-images.githubusercontent.com/64508435/168068030-ce47249e-4d48-4797-8f42-6c974e863a08.png">
<img width="1128" alt="Screenshot 2022-05-12 at 19 50 07" src="https://user-images.githubusercontent.com/64508435/168068459-2da2e93d-c131-4f08-b812-0937d606feef.png">

# Cornac-Supported Graph-Based Models
- User network:
  - Social Recommendation using PMF, SoRec
    - PMF is only for User-Item MF, not include User Graph, but it is a based for SoRec (SoRec in-cooperate with User Graph) 
  - Social Bayesian Personalized Ranking (SBPR) 
- Item network:
  - Collaborative Context Poisson Factorization (C2PF)
  - Probabilistic Collaborative Representation Learning (PCRL)
  - Matrix Co-Factorization (MCF)
