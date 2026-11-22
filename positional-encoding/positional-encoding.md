## My understanding

### 1. Why attention needs position

A token embedding tells the model **what the token is**. A positional encoding tells it **where the token is**.

Take `dog sees dog`. Both occurrences of `dog` start with the same token embedding. In plain self-attention, `sees` would give them the same score if all it had were those identical embeddings. The original Transformer adds the position vector for `0` to the first `dog` and the vector for `2` to the second. Now the two inputs differ, so attention can learn to treat them differently.

### 2. One position, two clock hands

I picture sinusoidal encoding as **clock hands turning at different speeds**. Each hand gives me two numbers, the sine and cosine of its angle. With `d_model = 4`, I need two hands to fill four columns.

- The **fast hand** uses the angle `p` at position `p`.
- The **slow hand** uses the angle `p / 100` when the base is `10000`.

So at position `2`, the angles are `2` and `0.02`. Taking sine and cosine of each gives the position vector `[0.91, -0.42, 0.02, 1.00]`, rounded to two decimals.

| Token position ↓ | Dim 0 · Fast sine | Dim 1 · Fast cosine | Dim 2 · Slow sine | Dim 3 · Slow cosine |
|---|---:|---:|---:|---:|
| 0 | 0.00 | 1.00 | 0.00 | 1.00 |
| 1 | 0.84 | 0.54 | 0.01 | 1.00 |
| 2 | 0.91 | −0.42 | 0.02 | 1.00 |

This is how I read the heatmap. **Across a row** is one token’s whole position vector. **Down a column** shows how one number changes as I advance through the sequence. The nearly solid stripes on the right come from the slow hands barely moving over the positions shown. A larger base makes the later hands turn even more slowly. The first hand always turns by one radian per token.

### 3. What the different speeds buy me

**Nearby positions** can look similar in the slow columns even when they look quite different in the fast ones. The fast hand also comes back around quickly, so the slow hand gives the model another reading over a longer stretch.

**Relative position** is a separate idea. Going from position `1` to `3` turns each hand by the same amount as going from `6` to `8`. Both moves are `+2` tokens. That’s what the paper means by a *linear function*. For each complete pair, I can get the new sine and cosine by mixing the old two values with weights that depend only on the gap. The vector describes an absolute position, and the way it changes gives attention a pattern for learning about offsets.

### 4. How the NumPy solution matches the picture

I’m building two grids. The first holds **one angle per position and hand**. The second holds **one sine or cosine value per output column**.

1. `positions` is a column of shape `(seq_len, 1)`. `divisors` is a 1D array of shape `(num_pairs,)`, with one divisor per hand.
2. `positions / divisors` gives a grid of shape `(seq_len, num_pairs)`. Broadcasting treats the divisors like a row, so **every position gets divided by every divisor**.
3. In the final grid, `pe[:, 0::2]` fills columns `0, 2, 4, ...` with sine and `pe[:, 1::2]` fills columns `1, 3, 5, ...` with cosine. With `d_model = 5`, there are three sine slots and two cosine slots, so I take cosine of only the first two angle columns.

### 5. How the alternatives fit

- **Fixed sinusoidal** calculates each position’s vector from a formula. I can calculate one for a position beyond the training length. How well the model uses that position is a separate question.
- **Learned absolute** looks up a vector for each position in a table filled in during training. A position beyond the table has no entry until the table is extended.
- **Relative** gives attention information about the gap directly, like “two tokens to my left.”
- **For images**, the same idea works with patches. A 2D sinusoidal encoding can use half the vector for the patch’s row and half for its column. Moving down changes the row half. Moving right changes the column half.

---

## The Position Problem in Transformers

Self-attention, the core mechanism in transformers, treats the input as a set. If you shuffle the tokens, the attention outputs just shuffle correspondingly. The model has no inherent sense of word order.

But word order matters! "The cat sat on the mat" means something different from "The mat sat on the cat."

Positional encodings inject position information into the model by adding position-dependent vectors to the token embeddings.

---

## The Sinusoidal Encoding Scheme

The original "Attention Is All You Need" paper introduced sinusoidal positional encodings:

$$
PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right)
$$

$$
PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

Where:
- $pos$ is the position in the sequence (0, 1, 2, ...)
- $i$ is the dimension index (0, 1, 2, ..., d/2)
- $d$ is the model dimension (embedding size)

Even dimensions use sine, odd dimensions use cosine.

---

## Understanding the Frequencies

Each dimension pair $(2i, 2i+1)$ oscillates at a different frequency:

$$
\omega_i = \frac{1}{10000^{2i/d}}
$$

**Low dimensions (small $i$):** High frequency, completes many cycles over the sequence.

**High dimensions (large $i$):** Low frequency, changes slowly across positions.

This creates a spectrum of frequencies, like different notes in music. Each position has a unique "chord" of sine and cosine values.

---

## Why This Scheme Works

**Unique encodings:**
No two positions have the same encoding vector. The combination of frequencies ensures uniqueness.

**Smooth interpolation:**
Nearby positions have similar encodings (their sine/cosine values are close). This provides a notion of locality.

**Relative positions:**
The encoding allows the model to learn attention patterns based on relative position. The paper shows that $PE(pos + k)$ can be written as a linear function of $PE(pos)$.

**Extrapolation:**
Sinusoidal encodings extend naturally beyond the training sequence length. (Though in practice, performance degrades on much longer sequences.)

---

## A Concrete Example

**Settings:** d_model = 4, position = 3

Frequencies:
- i=0: $\omega_0 = 1/10000^{0/4} = 1$
- i=1: $\omega_1 = 1/10000^{2/4} = 1/100 = 0.01$

Encodings:
- PE(3, 0) = sin(3 * 1) = sin(3) = 0.141
- PE(3, 1) = cos(3 * 1) = cos(3) = -0.990
- PE(3, 2) = sin(3 * 0.01) = sin(0.03) = 0.030
- PE(3, 3) = cos(3 * 0.01) = cos(0.03) = 0.9996

Position 3 encoding: [0.141, -0.990, 0.030, 1.000]

Compare to position 4:
- PE(4, 0) = sin(4) = -0.757
- PE(4, 1) = cos(4) = -0.654
- PE(4, 2) = sin(0.04) = 0.040
- PE(4, 3) = cos(0.04) = 0.9992

Position 4 encoding: [-0.757, -0.654, 0.040, 0.999]

The high-frequency components (indices 0, 1) change significantly between positions 3 and 4. The low-frequency components (indices 2, 3) change only slightly.

---

## Adding Encodings to Embeddings

The positional encoding is simply added to the token embedding:

$$
\text{input}_i = \text{embedding}(\text{token}_i) + PE(i)
$$

This assumes embeddings and positional encodings have the same dimension.

For a batch of sequences with shape (batch_size, seq_len, d_model):
1. Create PE matrix of shape (seq_len, d_model)
2. Add it to each sequence in the batch (broadcasting)

---

## Learned vs. Sinusoidal Encodings

**Sinusoidal (fixed):**
- No parameters to learn
- Extrapolates to longer sequences
- Works well in practice
- Used in original transformer

**Learned positional embeddings:**
- Each position has a learnable vector
- More flexible
- Does not extrapolate (fixed maximum length)
- Used in BERT, GPT

**Relative positional encodings:**
- Encode position differences, not absolute positions
- Used in Transformer-XL, some newer architectures
- Better for long sequences

---

## The Role of 10000

The base 10000 controls the range of frequencies:

**Smaller base (e.g., 100):**
- Frequencies decay faster
- Less distinction between distant positions at high dimensions

**Larger base (e.g., 100000):**
- Frequencies decay slower
- More variation at high dimensions

10000 was chosen empirically and works well for typical sequence lengths (up to a few thousand tokens).

---

## Handling Odd Dimensions

If $d_{model}$ is odd, there is one extra dimension without a pair. The standard approach:
- Use sine for the last dimension
- Or simply ensure $d_{model}$ is always even (common practice)

---

## Positional Encodings in Vision

Transformers for images (like ViT) also need positional information since patches are treated as tokens. Options:

**2D sinusoidal:** Separate encodings for row and column, concatenated.

**Learned 2D:** A learnable embedding for each (row, column) position.

The same principle applies: inject position information so the model knows where each patch came from.

---

## Why Not Concatenate?

An alternative to adding is concatenating position encodings to embeddings:

$$
\text{input}_i = [\text{embedding}(\text{token}_i); PE(i)]
$$

This doubles the input dimension. Adding is preferred because:
- Keeps dimension constant
- Works well empirically
- More efficient

The model learns to use the added positional signal to modulate attention.