import numpy as np
from collections import Counter
import re

def sig(x):
    return 1/(1+np.exp(-np.clip(x, -15, 15)))

def generate_pairs(token_ids, window):
    l = len(token_ids)
    pairs = []
    for i, word in enumerate(token_ids):
        
        start = max(0, i-window)
        end = min(l, i+window+1)
        for j in range(start, end):
            if i == j:
                continue
            context = token_ids[j]
            pairs.append((word, context))
    return pairs


# --- TRAINING DATA PREPARATION --- #

with open("t.txt", "r", encoding="utf-8") as file:
    clean_text = re.sub(r'[^\w\s]', ' ', file.read()).lower()
    
tokens = clean_text.split()

vocab = list(set(tokens))
word_to_id = {w: i for i, w in enumerate(vocab)}
id_to_word = {i: w for w, i in word_to_id.items()}

token_ids = [word_to_id[w] for w in tokens]
V = len(vocab)


# --- NEGATIVE SAMPLES TABLE --- #

word_counter = Counter(token_ids)
counts = np.array([word_counter[i] for i in range(V)])
weights = np.power(counts, 0.75)
probabilities = weights/np.sum(weights)
noise_size = 10**6
noise = np.random.choice(V, size=noise_size, p=probabilities)


# --- PARAMS INITIALIZATION --- #

N = 100                             #
K = 10                              #
W1 = np.random.randn(V, N)*0.1      # 
W2 = np.random.randn(V, N)*0.1      # Adjust these parameters
lr = 0.025                          #
window = 4                          #
batch_size = 256                    #
epochs = 15                         #

prev = float('inf')                 #
pointer = 0                         # Do not change these
eps = 1e-12                         #

pairs = generate_pairs(token_ids=token_ids, window=window)


# --- CORE TRAINING LOOP --- #

pairs_np = np.array(pairs)

for epoch in range(epochs):
    loss = 0
    np.random.shuffle(pairs_np)
    
    for i in range(0, len(pairs_np), batch_size):
        
        batch = pairs_np[i : i + batch_size]
        centers = batch[:, 0]
        contexts = batch[:, 1]
        B = len(batch)

        if pointer + B * K > noise_size:
            pointer = 0
            
        # Shape: (B, K)
        negatives = noise[pointer : pointer + B * K].reshape(B, K)
        pointer += B * K

        c = W1[centers]       # Shape: (B, N)
        w = W2[contexts]      # Shape: (B, N)
        NEG = W2[negatives]   # Shape: (B, K, N)

        # Forward Pass
        sig_pos = sig(np.sum(c * w, axis=1))
        sig_neg = sig((NEG @ c.reshape(B, N, 1)).reshape(B, K))

        # Gradients

        # Shape: (B, 1) * (B, N) -> (B, N)
        dw = (sig_pos - 1.0).reshape(B, 1) * c 
        
        # Shape: (B, K, 1) * (B, 1, N) -> (B, K, N)
        dNEG = sig_neg.reshape(B, K, 1) * c.reshape(B, 1, N) 
        
        # Shape: (B, N) + (B, 1, K) @ (B, K, N) -> (B, N)
        dc = (sig_pos - 1.0).reshape(B, 1) * w + (sig_neg.reshape(B, 1, K) @ NEG).reshape(B, N)

        # Updates
        np.subtract.at(W1, centers, lr * dc)
        np.subtract.at(W2, contexts, lr * dw)
        np.subtract.at(W2, negatives, lr * dNEG)

        # Loss
        batch_loss = np.sum(-np.log(sig_pos + eps)) + np.sum(-np.log(1.0 - sig_neg + eps))
        loss += batch_loss

    loss /= len(pairs_np)
    
    if loss > prev:
        print("↑")
        ups += 1
    else:
        print("↓")
        ups = 0
        
    prev = loss
    print("epoch", epoch, ":", loss)

# --- END OF TRAINING LOOP --- #



# --- TESTS --- #

W_norm = np.linalg.norm(W1, axis=1, keepdims=True)
W_normalized = W1 / W_norm

while True:
    try:
        inp = input("test: ")
        if(inp[0] < 'a'):
            word_id = int(inp)
            if word_id >= len(vocab):
                continue
        else:
            if inp not in word_to_id:
                continue
            word_id = word_to_id[inp]

        query_vec = W_normalized[word_id]
        similarities = np.dot(W_normalized, query_vec)
        top_indices = np.argsort(similarities)[::-1][1:11] 
        
        for idx in top_indices:
            print(id_to_word[idx], similarities[idx])
    except EOFError:
        break

