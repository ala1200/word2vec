# word2vec: skip-gram with negative sampling

Implementation of the word2vec skip-gram model using negative sampling. This project calculates vector representations (embeddings) for words based on text in t.txt file.

---

## Features

- **Pure NumPy Architecture**  
  Built entirely from scratch without high-level deep learning frameworks (like PyTorch or TensorFlow).

- **Optimized Mini-Batching**  
  Utilizes 3D tensor math and vectorized gradient updates for rapid training over large datasets.

- **Smoothed Unigram Distribution**  
  Implements Mikolov's 3/4 power law for negative sampling to prevent high-frequency stop words from dominating the noise distribution.

- **Interactive Evaluation**  
  Includes a built-in CLI tool test program effectiveness using cosine similarity.

---

## Usage

### 1. Configure Parameters

Open `ns-skip-gram.py` and adjust the hyperparameters in the **PARAMS INITIALIZATION** section:

- `N` (embedding dimensions)
- `K` (number of negative samples)
- `lr` (learning rate)
- `window`
- `batch_size`
- `epochs`
---

### 2. Train the Model

Run the main script from your terminal:

```bash
python3 ns-skip-gram.py
```

### 3. Test the Embeddings

Once training completes, the script enters an interactive testing loop.

- Type a target word in lowercase at the `test:` prompt  
- Press Enter  
- The system will output the **top 10 most similar words** based on cosine similarity  

---

### 4. Exit

To terminate the interactive testing loop, press (KeyboardInterrupt):

```bash
Ctrl + C
```
