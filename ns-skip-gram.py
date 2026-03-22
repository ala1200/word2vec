import numpy as np # type: ignore
import re

def sig(x):
    return 1/(1+np.exp(-x))

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
    clean_text = re.sub(r'[^\w\s]', '', file.read()).lower()
    
tokens = clean_text.split()

vocab = list(set(tokens))
word_to_id = {w: i for i, w in enumerate(vocab)}
id_to_word = {i: w for w, i in word_to_id.items()}

token_ids = [word_to_id[w] for w in tokens]


pairs = generate_pairs(token_ids=token_ids, window=3)

# --- PARAMS INITIALIZATION --- #

V = len(vocab)
N = 50
K = 5
W1 = np.random.randn(V, N)*0.1
W2 = np.random.randn(V, N)*0.1
lr = 0.05
prev = float('inf')
ups = 0

# --- CORE TRAINING LOOP --- #

for epoch in range(10):
    loss = 0
    
    for center, context in pairs:
        c = W1[center]
        w = W2[context]
        dc = np.dot(sig(np.dot(c, w))-1, w)
        dw = np.dot(sig(np.dot(c, w))-1, c)
        W2[context] -= lr*dw

        x = np.dot(c, w)
        loss -= np.log(sig(x))
        negatives = np.random.randint(0, V, size=K)
        
        for neg in negatives:
            w = W2[neg]
            dc += np.dot(sig(np.dot(c, w)), w)
            dw = np.dot(sig(np.dot(c, w)), c)
            W2[neg] -= lr*dw

            x = np.dot(c, w)
            loss -= np.log(sig(-x))

        W1[center] -= lr*dc
    if loss > prev:
        print("↑")
        end = 1
        ups += 1
    else:
        print("↓")
    prev = loss
    print("epoch", epoch, ":", loss)
    if ups == 2:
        break

# --- END OF TRAINING LOOP --- #

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

