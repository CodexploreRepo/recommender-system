# Sequential Model

```shell
SequentialRecModel (nn.Module)
├── Embeddings
│ ├── item_embeddings # Maps item IDs → vectors
│ └── position_embeddings # Maps positions → vectors
├── Methods
│ ├── add_position_embedding() # Combines item + position embeddings
│ ├── init_weights() # Weight initialization
│ ├── get_attention_mask() # Unidirectional (causal) mask
│ ├── get_bi_attention_mask() # Bidirectional mask
│ ├── forward() # Abstract (override in subclass)
│ └── calculate_loss() # Abstract (override in subclass)
```

## Embeddings

- This pattern (Add → LayerNorm → Dropout) is standard in Transformer architectures (from BERT, GPT, etc.):
- Applying them before the encoder ensures:
  - The encoder receives well-normalized input from the start
  - Regularization begins at the earliest point in the network

```shell
┌─────────────────────────────────────┐
│ Item Embedding + Position Embedding │
└─────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│ LayerNorm (stabilize scale)         │  ← before entering encoder
└─────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│ Dropout (prevent overfitting)       │
└─────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│ Transformer Encoder layers          │
└─────────────────────────────────────┘
```

- **Item embeddings**: Learns a vector for each item. `padding_idx=0` means item ID 0 (padding) always returns zeros.
- **Position embeddings**: Learns a vector for each position (0 to `max_seq_length`-1).

```Python
self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
```

### `add_position_embedding()`

- Combines item and position information:

```shell
Input:  [item_3, item_7, item_2]  (sequence of item IDs)
         ↓
Item Embed:    [v3, v7, v2]       (lookup item vectors)
Position Embed: [p0, p1, p2]      (lookup position vectors)
         ↓
Output: [v3+p0, v7+p1, v2+p2]     (element-wise sum)
         ↓
LayerNorm + Dropout
```

- Python code implementation

```Python
def add_position_embedding(self, sequence):
    seq_length = sequence.size(1)
    position_ids = torch.arange(
        seq_length, dtype=torch.long, device=sequence.device
    )
    position_ids = position_ids.unsqueeze(0).expand_as(sequence)
    item_embeddings = self.item_embeddings(sequence)
    position_embeddings = self.position_embeddings(position_ids)
    sequence_emb = item_embeddings + position_embeddings
    sequence_emb = self.LayerNorm(sequence_emb)
    sequence_emb = self.dropout(sequence_emb)

    return sequence_emb
```

- `position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)` creates a 1D tensor of consecutive integers from 0 to seq_length-1

```shell
seq_length = 5

position_ids = [0, 1, 2, 3, 4]
              shape: (5,)
```

- `position_ids = position_ids.unsqueeze(0).expand_as(sequence)`
  - `unsqueeze(0)` - Add a batch dimension at position 0
  - `expand_as(sequence)` - Broadcast to match the input batch size

```shell
# Step 2a: unsqueeze(0)
Before: [0, 1, 2, 3, 4]        shape: (5,)
After:  [[0, 1, 2, 3, 4]]      shape: (1, 5)
# Step 2b: expand_as(sequence) - Broadcast to match the input batch size
sequence shape: (batch_size, seq_length) = (3, 5)

position_ids after expand:
[[0, 1, 2, 3, 4],
 [0, 1, 2, 3, 4],
 [0, 1, 2, 3, 4]]     shape: (3, 5)

Input sequences (batch_size=3, seq_length=5):
┌─────────────────────────────────────────────┐
│ User A: [pad, pad, item3, item7, item2]     │  ← items
│ User B: [item1, item5, item9, item4, item8] │
│ User C: [pad, item6, item1, item3, item5]   │
└─────────────────────────────────────────────┘

Position IDs (same for all):
┌──────────────────────────────────────┐
│ User A: [  0,   1,    2,    3,    4] │  ← positions
│ User B: [  0,   1,    2,    3,    4] │
│ User C: [  0,   1,    2,    3,    4] │
└──────────────────────────────────────┘

Final embedding = item_embedding + position_embedding
```

#### Layer Norm

- When you add two embedding vectors, the resulting values can have inconsistent scale/magnitude across different sequences and training steps.
- What LayerNorm does:
  - Normalizes each sequence to have mean ≈ 0 and variance ≈ 1
  - Makes the input to subsequent layers (attention, feed-forward) more stable
  - Helps gradients flow better during backpropagation

```shell
Before LayerNorm:  [2.3, -5.1, 0.8, 12.4, ...]  ← varying scale
After LayerNorm:   [0.1, -0.9, 0.2,  1.5, ...]  ← normalized
```

#### Dropout - Regularization

- `sequence_emb = self.dropout(sequence_emb)  # ← Regularize`
- Overfitting - the model memorizes training data instead of learning generalizable patterns.
- What Dropout does:
  - Randomly zeros out some embedding dimensions during training
  - Forces the model to not rely on any single feature too heavily
  - Acts like training an ensemble of smaller networks

```shell
During training (dropout=0.1):
[0.1, -0.9, 0.2, 1.5, 0.3, -0.4, ...]
       ↓ randomly zero 10%
[0.1,  0.0, 0.2, 1.5, 0.0, -0.4, ...]

During inference: no dropout applied
```

## Attention Mask

- Unidirectional (Causal) Mask - `get_attention_mask()`
- Used by: SASRec, BSARec, DuoRec, FEARec Each position can only attend to itself and earlier positions (no peeking at future).
